"""Taste-native gateway: the model curates, the wire carries.

No orchestrator. No eviction pipeline. No page table. No cooperative
signals. The tensor is the memory. The model is the curator. This
gateway injects the tensor into the system prompt, injects the
_tinkuy_state_update tool, and intercepts the model's state update
tool call. Everything else passes through.

The gateway does NOT proxy. Every request is transformed. The
transformation is: inject state protocol + current state as JSON +
state update tool into the request. The response transformation is:
intercept state update tool_use, apply the structured update, carry
forward.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tinkuy.taste_gateway.tensor_protocol import (
    TOOL_NAME,
    build_harness_feedback,
    build_tensor_system_block,
    extract_session_tag,
    get_state_update_tool,
    make_session_tag,
    parse_state_update,
)

log = logging.getLogger("tinkuy.taste_gateway")


# ---------------------------------------------------------------------------
# Harness feedback (ported from taste.py, adapted for gateway context)
# ---------------------------------------------------------------------------

def _generate_feedback(
    tensor: dict | None,
    cycle: int,
    loss_history: list[dict] | None = None,
    integration_loss_history: list[dict] | None = None,
    memory_objects: list["MemoryObject"] | None = None,
) -> list[str]:
    """Generate harness feedback based on tensor and memory health."""
    if tensor is None:
        return []

    feedback = []

    n_questions = len(tensor.get("open_questions", []))
    if n_questions > 15:
        feedback.append(
            f"HARNESS FEEDBACK: {n_questions} open questions is critically high. "
            f"Curate them below 10."
        )
    elif n_questions > 10:
        feedback.append(
            f"HARNESS FEEDBACK: {n_questions} open questions. "
            f"Curate — resolve, merge, or drop stale questions."
        )
    elif 5 <= n_questions <= 8:
        feedback.append(
            f"HARNESS NOTE: {n_questions} open questions — good balance."
        )

    n_strands = len(tensor.get("strands", []))
    if n_strands > 6:
        feedback.append(
            f"HARNESS FEEDBACK: {n_strands} strands is high. "
            f"Consider consolidating."
        )

    tensor_tokens = len(json.dumps(tensor)) // 4
    if tensor_tokens > 5000:
        feedback.append(
            f"HARNESS FEEDBACK: Tensor is ~{tensor_tokens:,} tokens. "
            f"Consider consolidating."
        )

    # Sustained growth without curation
    if loss_history is not None and cycle >= 6:
        recent_loss_cycles = {
            entry["cycle"] for entry in loss_history
            if entry["cycle"] > cycle - 5
        }
        if not recent_loss_cycles and tensor_tokens > 1500:
            feedback.append(
                f"HARNESS FEEDBACK: No declared losses in the last 5 cycles "
                f"and tensor is ~{tensor_tokens:,} tokens. Review whether "
                f"any strands are redundant or stale."
            )

    # Tension age tracking
    tensions = tensor.get("unresolved_tensions", [])
    if tensions:
        old_tensions = [t for t in tensions if t.get("cycles_held", 0) >= 10]
        if old_tensions:
            labels = ", ".join(t.get("tension_id", "?") for t in old_tensions)
            feedback.append(
                f"HARNESS NOTE: Tension(s) held 10+ cycles: {labels}. "
                f"Keep if fundamental, resolve or drop if stale."
            )

    # Integration loss mirror
    if integration_loss_history:
        recent = [
            e for e in integration_loss_history
            if e["cycle"] > cycle - 5
        ]
        if recent:
            latest = [e["loss"] for e in recent[-3:]]
            feedback.append(
                f"HARNESS NOTE: {len(recent)} integration loss(es) over last "
                f"5 cycles. Most recent: {'; '.join(latest)}"
            )

    # --- Memory object feedback ---
    if memory_objects:
        n_objects = len(memory_objects)
        memory_tokens = sum(m.tokens for m in memory_objects)
        n_full = sum(1 for m in memory_objects if m.state == "full")
        n_summary = sum(1 for m in memory_objects if m.state == "summary")

        if memory_tokens > 50000:
            feedback.append(
                f"HARNESS FEEDBACK: {n_objects} memory objects using "
                f"~{memory_tokens:,} tokens ({n_full} full, {n_summary} "
                f"summarized). Consider summarizing or releasing stale objects."
            )
        elif memory_tokens > 20000:
            feedback.append(
                f"HARNESS NOTE: {n_objects} memory objects, "
                f"~{memory_tokens:,} tokens ({n_full} full, {n_summary} "
                f"summarized)."
            )

        # Flag old unpinned objects
        old_objects = [
            m for m in memory_objects
            if not m.pinned and m.state == "full"
            and (cycle - m.turn) >= 10
        ]
        if old_objects:
            ids = ", ".join(m.id for m in old_objects[:5])
            feedback.append(
                f"HARNESS NOTE: {len(old_objects)} full memory objects from "
                f"10+ cycles ago (unpinned): {ids}. Still needed?"
            )

    return feedback


# ---------------------------------------------------------------------------
# Default-stable apply (ported from taste.py)
# ---------------------------------------------------------------------------

def _apply_updates(
    prior_tensor: dict | None, raw_update: dict, cycle: int,
) -> dict:
    """Apply selective updates. Default-stable: only declared regions change."""
    updated_regions = set(raw_update.get("updated_regions", []))

    tensor = dict(prior_tensor) if prior_tensor else {}
    tensor["cycle"] = cycle

    list_regions = [
        "strands", "declared_losses", "open_questions", "unresolved_tensions",
    ]
    for key in list_regions:
        if key in updated_regions and key in raw_update:
            value = raw_update[key]
            if isinstance(value, list):
                tensor[key] = value

    if "instructions_for_next" in updated_regions:
        tensor["instructions_for_next"] = raw_update.get(
            "instructions_for_next", ""
        )

    for key in ["overall_truth", "overall_indeterminacy", "overall_falsity"]:
        if key in raw_update:
            tensor[key] = raw_update[key]

    # feedback_to_harness is per-cycle
    if "feedback_to_harness" in updated_regions:
        tensor["feedback_to_harness"] = raw_update.get("feedback_to_harness")
    else:
        tensor.pop("feedback_to_harness", None)

    # declared_losses is per-cycle — clear if not updated
    if "declared_losses" not in updated_regions:
        tensor["declared_losses"] = []

    # Auto-increment cycles_held for tensions
    for tension in tensor.get("unresolved_tensions", []):
        tension["cycles_held"] = tension.get("cycles_held", 0) + 1

    # Strip integration_losses from strands (per-cycle only)
    for strand in tensor.get("strands", []):
        strand.pop("integration_losses", None)

    return tensor


# ---------------------------------------------------------------------------
# ALU model: system + tensor + current input. No history.
# ---------------------------------------------------------------------------

def _is_tool_cycle(messages: list[dict]) -> bool:
    """Is the current input a tool_result (mid-turn tool chain)?

    If the last message is a user message where all content blocks are
    tool_results, this is a tool cycle. The model is mid-task.
    """
    if not messages:
        return False
    current = messages[-1]
    if current.get("role") != "user":
        return False
    content = current.get("content", "")
    if isinstance(content, str):
        return False  # Plain text = human turn
    if isinstance(content, list):
        if not content:
            return False
        return all(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in content
        )
    return False


def _find_current_turn_start(messages: list[dict]) -> int:
    """Find the index of the last user message with text content.

    A "text" user message is the start of a human turn. Tool_result
    messages (all blocks are type tool_result) are part of the tool
    chain, not a new turn. We scan backwards to find the boundary.
    """
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return i  # Plain text user message
        if isinstance(content, list):
            # Check if any block is NOT a tool_result
            has_text = any(
                isinstance(b, dict) and b.get("type") != "tool_result"
                for b in content
            )
            if has_text:
                return i
            # Also catch plain strings in content list
            if any(isinstance(b, str) for b in content):
                return i
    return 0


def _extract_tool_exchanges(
    messages: list[dict], current_turn_start: int,
) -> list[dict[str, Any]]:
    """Extract tool exchanges from prior turns as structured data.

    Returns a list of dicts, each representing one tool exchange:
        {tool_name, tool_input, tool_use_id, result_text}

    Walks messages before current_turn_start, pairing tool_use blocks
    (from assistant messages) with their tool_result blocks (from user
    messages). Conversational messages are skipped — the tensor carries those.
    """
    if current_turn_start == 0:
        return []

    # First pass: collect tool_use blocks indexed by id
    tool_uses: dict[str, dict] = {}
    for msg in messages[:current_turn_start]:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tool_uses[block.get("id", "")] = {
                    "tool_name": block.get("name", "?"),
                    "tool_input": block.get("input", {}),
                }

    # Second pass: collect tool_result blocks, pair with tool_use
    exchanges: list[dict[str, Any]] = []
    for msg in messages[:current_turn_start]:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if not isinstance(content, list):
            continue
        for block in content:
            if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                continue
            tool_use_id = block.get("tool_use_id", "")
            result_content = block.get("content", "")
            if isinstance(result_content, list):
                texts = []
                for rb in result_content:
                    if isinstance(rb, dict) and rb.get("type") == "text":
                        texts.append(rb.get("text", ""))
                result_text = "\n".join(texts)
            elif isinstance(result_content, str):
                result_text = result_content
            else:
                result_text = str(result_content)

            use = tool_uses.get(tool_use_id, {})
            exchanges.append({
                "tool_name": use.get("tool_name", "?"),
                "tool_input": use.get("tool_input", {}),
                "tool_use_id": tool_use_id,
                "result_text": result_text,
            })

    return exchanges


def _build_memory_objects(
    exchanges: list[dict[str, Any]],
    session: "TasteSession",
    current_cycle: int,
) -> None:
    """Convert new tool exchanges into labeled MemoryObjects on the session.

    Only creates objects for exchanges not already tracked (by tool_use_id).
    Existing objects are preserved with their current state (may have been
    summarized or pinned).
    """
    known_ids = {m.tool_use_id for m in session.memory_objects}

    for ex in exchanges:
        tool_use_id = ex["tool_use_id"]
        if tool_use_id in known_ids:
            continue  # already tracked

        tool_name = ex["tool_name"]
        content = ex["result_text"]
        tokens = len(content) // 4

        obj = MemoryObject(
            id=session.next_memory_id(),
            tool=tool_name,
            label=MemoryObject.make_label(tool_name, ex["tool_input"]),
            content=content,
            tokens=tokens,
            turn=current_cycle,
            cycle=current_cycle,
            tool_use_id=tool_use_id,
            original_tokens=tokens,
        )
        session.memory_objects.append(obj)


def _render_memory_block(session: "TasteSession") -> str:
    """Render memory objects as labeled markup for the system prompt.

    Each object is individually addressable by its id. The model sees
    the tool name, semantic label, size, source turn, and state.
    """
    if not session.memory_objects:
        return ""

    lines = ["<prior-tool-outputs>"]
    for m in session.memory_objects:
        attrs = (
            f'id="{m.id}" tool="{m.tool}" label="{_esc_attr(m.label)}" '
            f'tokens="~{m.tokens}" turn="{m.turn}" state="{m.state}"'
        )
        if m.pinned:
            attrs += ' pinned="true"'
        lines.append(f"  <memory {attrs}>")
        lines.append(m.content)
        lines.append("  </memory>")
    lines.append("</prior-tool-outputs>")
    return "\n".join(lines)


def _esc_attr(text: str) -> str:
    """Escape XML attribute value."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def _build_taste_messages(
    messages: list[dict],
    session: "TasteSession",
    current_cycle: int,
) -> list[dict[str, Any]]:
    """Taste model: current turn + tensor. Prior conversation in tensor,
    prior tool outputs as labeled memory objects in system block.

    Returns current_turn_messages. Side effect: populates session.memory_objects
    with labeled MemoryObjects from prior-turn tool exchanges.

    The current turn is everything from the last user text message onward.
    Tool exchanges from prior turns are converted to memory objects (tracked
    on the session). Prior conversation (user text + assistant responses)
    is dropped — the tensor carries that state.

    Strips client cache_control — the gateway owns cache placement.
    """
    if not messages:
        return []

    current_turn_start = _find_current_turn_start(messages)

    # Build memory objects from prior-turn tool exchanges
    exchanges = _extract_tool_exchanges(messages, current_turn_start)
    _build_memory_objects(exchanges, session, current_cycle)

    # Current turn: everything from the human message onward
    turn_messages = []
    for msg in messages[current_turn_start:]:
        content = _strip_cache_control(msg.get("content", ""))
        turn_messages.append({"role": msg["role"], "content": content})

    # Repair orphaned tool_results: if the first message contains
    # tool_result blocks, their matching tool_use must be in the
    # preceding assistant message. Pull it in.
    turn_messages = _repair_tool_orphans(turn_messages, messages, current_turn_start)

    return turn_messages


def _repair_tool_orphans(
    turn_messages: list[dict],
    all_messages: list[dict],
    turn_start: int,
) -> list[dict]:
    """Ensure every tool_result has its matching tool_use in a preceding message.

    The API requires each tool_result to have a corresponding tool_use in
    the previous assistant message. When we slice the message array at the
    turn boundary, tool_results at the start can become orphaned.

    Walk the turn messages and pull in any missing tool_use messages from
    before the turn boundary.
    """
    if not turn_messages:
        return turn_messages

    first = turn_messages[0]
    first_content = first.get("content", "")

    # Collect tool_use_ids referenced by tool_results in the first message
    orphan_ids: set[str] = set()
    if isinstance(first_content, list):
        for block in first_content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                orphan_ids.add(block.get("tool_use_id", ""))

    if not orphan_ids:
        return turn_messages

    # Check if the first message is role=user (tool_results come in user messages)
    # and the turn doesn't already start with an assistant message containing
    # the matching tool_use
    if first.get("role") != "user":
        return turn_messages

    # Look backwards from turn_start for the assistant message with matching tool_use
    for i in range(turn_start - 1, -1, -1):
        msg = all_messages[i]
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if not isinstance(content, list):
            continue

        has_matching_tool_use = any(
            isinstance(b, dict)
            and b.get("type") == "tool_use"
            and b.get("id", "") in orphan_ids
            for b in content
        )

        if has_matching_tool_use:
            # Prepend this assistant message
            repaired_content = _strip_cache_control(content)
            return [
                {"role": "assistant", "content": repaired_content},
                *turn_messages,
            ]

    # No matching assistant found — the tool_results are truly orphaned.
    # Strip them from the first message. The memory objects system
    # block already captured them, so nothing is lost.
    if isinstance(first_content, list):
        cleaned = [
            b for b in first_content
            if not (isinstance(b, dict) and b.get("type") == "tool_result")
        ]
        if cleaned:
            turn_messages[0] = {"role": first["role"], "content": cleaned}
        elif len(turn_messages) > 1:
            # First message was entirely tool_results — drop it
            turn_messages = turn_messages[1:]
        else:
            # Only message was entirely tool_results with no text —
            # shouldn't happen (would mean no user text at all), but
            # return a minimal user message rather than an empty array
            turn_messages = [{"role": "user", "content": "(tool results processed)"}]

    return turn_messages


def _strip_cache_control(content: Any) -> Any:
    """Strip cache_control from content blocks. Returns new structure."""
    if isinstance(content, list):
        return [
            {k: v for k, v in block.items() if k != "cache_control"}
            if isinstance(block, dict) else block
            for block in content
        ]
    return content


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TasteGatewayConfig:
    """Configuration for the taste-native gateway."""
    data_dir: str | None = None
    enable_console: bool = True
    context_limit: int = 200_000


# ---------------------------------------------------------------------------
# Memory objects — labeled, individually addressable tool outputs
# ---------------------------------------------------------------------------

@dataclass
class MemoryObject:
    """A labeled memory object — one tool exchange (use + result).

    The model sees these as labeled blocks in the system prompt and can
    curate them via memory_actions in _tinkuy_state_update: summarize,
    release, pin.
    """
    id: str                     # monotonic: m1, m2, m3, ...
    tool: str                   # tool name (Read, Bash, Glob, etc.)
    label: str                  # semantic label (file path, command, etc.)
    content: str                # full result text or model-provided summary
    tokens: int                 # approximate token count (len // 4)
    turn: int                   # human turn that produced this
    cycle: int                  # cycle when created
    state: str = "full"         # "full" | "summary"
    pinned: bool = False
    tool_use_id: str = ""       # original tool_use_id for tracing
    original_tokens: int = 0    # tokens before summarization

    @staticmethod
    def make_label(tool_name: str, tool_input: dict | str) -> str:
        """Derive a semantic label from tool name + input."""
        if isinstance(tool_input, str):
            return tool_input[:60]
        # Extract the most informative parameter
        for key in ("file_path", "command", "pattern", "query", "url",
                     "path", "skill", "prompt", "operation"):
            val = tool_input.get(key)
            if val:
                val_str = str(val)
                if len(val_str) > 55:
                    val_str = val_str[:52] + "..."
                return val_str
        # Fallback: first string value
        for v in tool_input.values():
            if isinstance(v, str) and v:
                return str(v)[:60]
        return tool_name


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class TasteSession:
    """Per-session tensor state."""
    session_id: str
    cycle: int = 0
    tensor: dict | None = None
    loss_history: list[dict] = field(default_factory=list)
    integration_loss_history: list[dict] = field(default_factory=list)
    tag_injected: bool = False
    log_path: Path | None = None
    memory_objects: list[MemoryObject] = field(default_factory=list)
    _memory_seq: int = 0  # next sequence number for memory IDs
    # Pending state update tool_use from prior cycle — used to inject
    # the synthetic tool_result exchange on the next request.
    pending_state_tool_use: dict | None = None

    def tensor_token_estimate(self) -> int:
        if self.tensor is None:
            return 0
        return len(json.dumps(self.tensor)) // 4

    def memory_token_estimate(self) -> int:
        return sum(m.tokens for m in self.memory_objects)

    def next_memory_id(self) -> str:
        self._memory_seq += 1
        return f"m{self._memory_seq}"


# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------

class TasteGateway:
    """Taste-native gateway. The model curates, the wire carries.

    One instance handles multiple sessions via session tags in the
    message stream. Each session has its own tensor, its own cycle
    counter, its own loss history.
    """

    def __init__(self, config: TasteGatewayConfig | None = None) -> None:
        self.config = config or TasteGatewayConfig()
        self._sessions: dict[str, TasteSession] = {}

        if self.config.data_dir:
            self._data_dir = Path(self.config.data_dir)
            self._data_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._data_dir = None

    # --- Session management ---

    def get_or_create_session(
        self, messages: list[dict],
        session_id: str | None = None,
    ) -> TasteSession:
        """Find existing session from message history or create new one.

        Session identity: explicit session_id > <tinkuy-session/> tag
        in conversation history > generate new UUID.
        """
        if session_id is None:
            session_id = extract_session_tag(messages)

        if session_id and session_id in self._sessions:
            return self._sessions[session_id]

        if session_id:
            # Tag found but session not in memory — try to restore
            session = self._restore_session(session_id)
            if session:
                self._sessions[session_id] = session
                return session

        # New session
        session_id = session_id or str(uuid.uuid4())[:12]
        log_path = None
        if self._data_dir:
            log_path = (
                self._data_dir / "taste_sessions" / session_id / "tensor.jsonl"
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)

        session = TasteSession(
            session_id=session_id,
            log_path=log_path,
        )
        self._sessions[session_id] = session
        log.info("new taste session: %s", session_id)
        return session

    def _restore_session(self, session_id: str) -> TasteSession | None:
        """Restore session from JSONL log."""
        if not self._data_dir:
            return None

        log_path = (
            self._data_dir / "taste_sessions" / session_id / "tensor.jsonl"
        )
        if not log_path.exists():
            return None

        last_line = None
        with open(log_path) as f:
            for line in f:
                if line.strip():
                    last_line = line

        if last_line is None:
            return None

        record = json.loads(last_line)
        session = TasteSession(
            session_id=session_id,
            cycle=record.get("cycle", 0),
            tensor=record.get("tensor"),
            loss_history=record.get("loss_history", []),
            tag_injected=True,  # tag was already in conversation
            log_path=log_path,
        )

        # Reconstruct integration loss history
        with open(log_path) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                for il in entry.get("cycle_integration_losses", []):
                    session.integration_loss_history.append(il)

        # Restore memory sequence counter from summary
        mem_summary = record.get("memory_objects_summary", [])
        if mem_summary:
            max_seq = 0
            for ms in mem_summary:
                mid = ms.get("id", "")
                if mid.startswith("m"):
                    try:
                        max_seq = max(max_seq, int(mid[1:]))
                    except ValueError:
                        pass
            session._memory_seq = max_seq
            # Note: memory objects themselves will be rebuilt from the
            # message array on the next request. Curated state (summaries,
            # pins) is restored below from the curated_memory field.

        # Restore curated memory objects (summarized/pinned)
        for cm in record.get("curated_memory", []):
            obj = MemoryObject(
                id=cm["id"],
                tool=cm.get("tool", "?"),
                label=cm.get("label", ""),
                content=cm.get("content", ""),
                tokens=cm.get("tokens", 0),
                turn=cm.get("turn", 0),
                cycle=cm.get("cycle", 0),
                state=cm.get("state", "summary"),
                pinned=cm.get("pinned", False),
                tool_use_id=cm.get("tool_use_id", ""),
                original_tokens=cm.get("original_tokens", 0),
            )
            session.memory_objects.append(obj)

        log.info(
            "restored taste session %s: cycle=%d, %d strands, ~%d tokens, "
            "%d memory objects",
            session_id,
            session.cycle,
            len(session.tensor.get("strands", [])) if session.tensor else 0,
            session.tensor_token_estimate(),
            len(session.memory_objects),
        )
        return session

    # --- Request transformation ---

    def prepare_request(
        self, body: dict[str, Any],
        session_id: str | None = None,
    ) -> tuple[dict[str, Any], TasteSession, list[str]]:
        """Transform a client request by injecting the tensor.

        Returns (transformed_body, session, feedback). The caller forwards
        the transformed body to the API and passes the response back
        through process_response with the same session. Feedback is
        returned so the caller can pass it through for logging.
        """
        messages = body.get("messages", [])
        session = self.get_or_create_session(messages, session_id=session_id)

        # Detect whether this is a mid-turn tool cycle or a new human turn.
        # Tool cycles: the current input is a tool_result. The model is
        # mid-task and should focus on the work, not tensor curation.
        # Turn boundaries: the current input is a user text message. The
        # model should update its tensor to absorb the completed turn.
        is_tool_cycle = _is_tool_cycle(messages)

        # Only increment cycle on human turns, not tool exchanges
        if not is_tool_cycle:
            session.cycle += 1

        # Generate feedback only on human turns. Feedback goes into the
        # synthetic tool_result, not the system prompt.
        feedback: list[str] = []
        if not is_tool_cycle:
            feedback = _generate_feedback(
                session.tensor,
                session.cycle,
                loss_history=session.loss_history,
                integration_loss_history=session.integration_loss_history,
                memory_objects=session.memory_objects,
            )

        # Build state system block — read-only during tool cycles.
        # Feedback goes via tool_result, not system prompt.
        tensor_block = build_tensor_system_block(
            session.tensor, session.cycle,
            tool_cycle=is_tool_cycle,
        )

        # --- System blocks: strip client cache_control, place our own ---
        # The gateway owns cache placement. Budget: 4 breakpoints.
        #   BP1: end of client system (stable across session)
        #   BP2: tensor block (changes per cycle, stable within)
        #   BP3: last stable message (placed below on messages)
        #   BP4: available
        client_system = body.get("system", [])
        if isinstance(client_system, str):
            client_system = [{"type": "text", "text": client_system}]
        elif isinstance(client_system, list):
            normalized: list[dict[str, Any]] = []
            for block in client_system:
                if isinstance(block, str):
                    normalized.append({"type": "text", "text": block})
                else:
                    # Strip client cache_control — we place our own
                    cleaned = {
                        k: v for k, v in block.items()
                        if k != "cache_control"
                    }
                    normalized.append(cleaned)
            client_system = normalized

        # BP1: end of client system blocks
        system_blocks: list[dict[str, Any]] = list(client_system)
        if system_blocks:
            system_blocks[-1] = dict(system_blocks[-1])
            system_blocks[-1]["cache_control"] = {"type": "ephemeral"}

        # --- Taste model: current turn + labeled memory objects ---
        # Prior conversation is in the tensor. Prior tool outputs become
        # labeled memory objects in a system block. Current turn in messages.
        our_messages = _build_taste_messages(messages, session, session.cycle)

        # Inject synthetic tool_result exchange from prior cycle's state
        # update. The API requires every tool_use to get a tool_result.
        # We store the tool_use on the session in process_response, then
        # prepend the exchange here so the model sees its feedback.
        if session.pending_state_tool_use is not None and not is_tool_cycle:
            tool_use_id = session.pending_state_tool_use["id"]
            tool_input = session.pending_state_tool_use["input"]
            feedback_payload = build_harness_feedback(
                session.cycle, feedback, session.memory_objects,
            )
            synthetic_exchange = [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tool_use_id,
                            "name": TOOL_NAME,
                            "input": tool_input,
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": json.dumps(feedback_payload),
                        },
                    ],
                },
            ]
            our_messages = synthetic_exchange + our_messages
            session.pending_state_tool_use = None

        # BP2: tensor block — small and stable (only changes at turn
        # boundaries). Placing it before memory objects means cache busts
        # from growing tool results don't invalidate the tensor prefix.
        system_blocks.append({
            "type": "text",
            "text": tensor_block,
            "cache_control": {"type": "ephemeral"},
        })

        # BP3: memory objects — labeled prior tool outputs.
        # Grows during tool chains. Model can curate via signals.
        memory_block = _render_memory_block(session)
        if memory_block:
            system_blocks.append({
                "type": "text",
                "text": memory_block,
                "cache_control": {"type": "ephemeral"},
            })

        # BP4: advancing tool-chain breakpoint. Place on the last
        # completed tool exchange so each new tool call caches all
        # prior results. During a tool chain, messages grow but the
        # earlier ones don't change — cache them.
        if is_tool_cycle and len(our_messages) >= 3:
            # The last two messages are the current exchange (assistant
            # tool_use + user tool_result). Everything before that is
            # completed and cacheable. Place breakpoint on the message
            # just before the current exchange.
            bp_msg = our_messages[-3]
            bp_content = bp_msg.get("content", "")
            if isinstance(bp_content, list) and bp_content:
                # Place on the last content block of that message
                last_block = bp_content[-1]
                if isinstance(last_block, dict):
                    bp_content[-1] = dict(last_block)
                    bp_content[-1]["cache_control"] = {"type": "ephemeral"}
            elif isinstance(bp_content, str):
                # Convert to block form so we can add cache_control
                our_messages[-3] = {
                    "role": bp_msg["role"],
                    "content": [{
                        "type": "text",
                        "text": bp_content,
                        "cache_control": {"type": "ephemeral"},
                    }],
                }

        # Build the transformed body — allowlist, not passthrough.
        # Every field is an explicit gateway decision. If the client
        # sends a field we don't know about, it doesn't reach the API.
        transformed: dict[str, Any] = {
            # Transformed by gateway
            "system": system_blocks,
            "messages": our_messages,
            # Model and generation parameters — gateway acknowledges
            # and forwards its own copy
            "model": body["model"],
            "max_tokens": body.get("max_tokens", 16384),
            "stream": body.get("stream", False),
        }
        # Optional fields — only include if client sent them
        for key in (
            "temperature", "top_p", "top_k", "stop_sequences",
            "tools", "tool_choice", "metadata",
        ):
            if key in body:
                transformed[key] = body[key]

        # Inject _tinkuy_state_update into the tool list. The model
        # calls this at turn boundaries to persist its state update.
        client_tools = transformed.get("tools", [])
        transformed["tools"] = list(client_tools) + [get_state_update_tool()]

        if self.config.enable_console:
            self._log_console(session, feedback)

        return transformed, session, feedback

    # --- Response processing ---

    def process_response(
        self,
        response_text: str,
        session: TasteSession,
        content_blocks: list[dict] | None = None,
        usage: dict | None = None,
        request_body: dict | None = None,
        timing: dict | None = None,
        feedback: list[str] | None = None,
        state_tool_use: dict | None = None,
    ) -> str:
        """Process the model's response: apply state update, return clean text.

        The state_tool_use dict (if any) is the intercepted
        _tinkuy_state_update tool call, already extracted from the stream
        by the server layer. It contains {id, input} — the tool_use_id
        and the structured JSON input.

        The session tag is injected into the first response for a new session.
        """
        clean_text = response_text

        # Extract state update from intercepted tool call
        raw_update = None
        if state_tool_use is not None:
            raw_update = parse_state_update(state_tool_use["input"])
            # Store for synthetic tool_result injection on next request
            session.pending_state_tool_use = state_tool_use

        # Apply memory actions from the state update
        memory_actions = []
        if raw_update is not None:
            memory_actions = raw_update.pop("memory_actions", []) or []
        memory_events = self._apply_memory_signals(session, memory_actions)

        # Capture prior tensor for logging
        prior_tensor = (
            json.loads(json.dumps(session.tensor)) if session.tensor else None
        )

        if raw_update is not None:
            # Capture integration losses before _apply_updates strips them
            cycle_integration_losses = []
            for strand in raw_update.get("strands", []):
                for loss in strand.get("integration_losses", []):
                    cycle_integration_losses.append({
                        "cycle": session.cycle,
                        "strand": strand.get("title", "unknown"),
                        "loss": loss,
                    })
            if cycle_integration_losses:
                session.integration_loss_history.extend(cycle_integration_losses)

            # Apply updates
            session.tensor = _apply_updates(
                session.tensor, raw_update, session.cycle,
            )

            # Accumulate declared losses
            for loss in session.tensor.get("declared_losses", []):
                session.loss_history.append({
                    "cycle": session.cycle,
                    **loss,
                })

        # Log EVERY cycle — not just ones with tensor updates
        self._log_cycle(
            session=session,
            prior_tensor=prior_tensor,
            raw_update=raw_update,
            response_text=response_text,
            clean_text=clean_text,
            usage=usage,
            request_body=request_body,
            timing=timing,
            feedback=feedback,
            content_blocks=content_blocks,
            memory_events=memory_events,
        )

        # Console: the numbers that matter
        if self.config.enable_console and usage:
            input_tok = usage.get("input_tokens", 0)
            cache_read = usage.get("cache_read_input_tokens", 0)
            output_tok = usage.get("output_tokens", 0)
            total_in = input_tok + cache_read
            cache_pct = (cache_read / total_in * 100) if total_in > 0 else 0
            tensor_tok = session.tensor_token_estimate()
            mem_tok = session.memory_token_estimate()
            log.info(
                "  COST | in=%s+%scache/%sout total=%s "
                "cache:%.0f%% tensor=~%d memory=~%d",
                f"{input_tok:,}", f"{cache_read:,}", f"{output_tok:,}",
                f"{total_in:,}", cache_pct, tensor_tok, mem_tok,
            )

        if raw_update is None:
            log.info(
                "session %s cycle %d: no tensor update in response",
                session.session_id, session.cycle,
            )

        # Inject session tag into first response
        if not session.tag_injected:
            tag = make_session_tag(session.session_id)
            clean_text = f"{tag}\n\n{clean_text}"
            session.tag_injected = True
            log.info(
                "injected session tag: %s", session.session_id,
            )

        return clean_text

    # --- Memory curation ---

    def _apply_memory_signals(
        self, session: TasteSession, signals: list[dict],
    ) -> list[dict]:
        """Apply memory curation signals from the model's response.

        Returns a list of event dicts for logging. Each event records
        what happened: summarize (with before/after tokens), release
        (with reason), or pin.
        """
        if not signals:
            return []

        events: list[dict] = []
        obj_by_id = {m.id: m for m in session.memory_objects}

        for signal in signals:
            action = signal["action"]
            obj_id = signal["id"]
            obj = obj_by_id.get(obj_id)

            if obj is None:
                log.warning(
                    "memory signal for unknown id %s (action=%s)",
                    obj_id, action,
                )
                events.append({
                    "action": action, "id": obj_id,
                    "status": "error", "reason": "unknown id",
                })
                continue

            if action == "summarize":
                prior_tokens = obj.tokens
                obj.content = signal["content"]
                obj.tokens = len(obj.content) // 4
                obj.state = "summary"
                events.append({
                    "action": "summarize",
                    "id": obj_id,
                    "tool": obj.tool,
                    "label": obj.label,
                    "prior_tokens": prior_tokens,
                    "new_tokens": obj.tokens,
                    "reduction": prior_tokens - obj.tokens,
                    "cycle": session.cycle,
                })
                log.info(
                    "  memory | summarize %s (%s) %d→%d tokens",
                    obj_id, obj.label, prior_tokens, obj.tokens,
                )

            elif action == "release":
                reason = signal.get("reason", "")
                events.append({
                    "action": "release",
                    "id": obj_id,
                    "tool": obj.tool,
                    "label": obj.label,
                    "tokens_freed": obj.tokens,
                    "reason": reason,
                    "cycle": session.cycle,
                })
                log.info(
                    "  memory | release %s (%s) freed %d tokens: %s",
                    obj_id, obj.label, obj.tokens, reason,
                )
                session.memory_objects.remove(obj)

            elif action == "pin":
                obj.pinned = True
                events.append({
                    "action": "pin",
                    "id": obj_id,
                    "tool": obj.tool,
                    "label": obj.label,
                    "cycle": session.cycle,
                })
                log.info("  memory | pin %s (%s)", obj_id, obj.label)

        return events

    # --- Logging ---

    def _log_cycle(
        self,
        session: TasteSession,
        prior_tensor: dict | None,
        raw_update: dict | None,
        response_text: str,
        clean_text: str,
        usage: dict | None,
        request_body: dict | None = None,
        timing: dict | None = None,
        feedback: list[str] | None = None,
        content_blocks: list[dict] | None = None,
        memory_events: list[dict] | None = None,
    ) -> None:
        """Log everything. Every cycle, whether or not a tensor update was found.

        This is the research record. If it's not here, it didn't happen.
        """
        if session.log_path is None:
            return

        had_update = raw_update is not None
        tensor = session.tensor or {}

        # Extract cache stats from usage
        cache_stats = None
        if usage:
            cache_read = usage.get("cache_read_input_tokens", 0)
            cache_create = usage.get("cache_creation_input_tokens", 0)
            input_tokens = usage.get("input_tokens", 0)
            total_in = input_tokens + cache_read
            cache_stats = {
                "input_tokens": input_tokens,
                "cache_read_tokens": cache_read,
                "cache_create_tokens": cache_create,
                "total_input": total_in,
                "cache_hit_rate": (
                    cache_read / total_in if total_in > 0 else 0.0
                ),
            }

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cycle": session.cycle,
            "session_id": session.session_id,
            # State transition
            "had_tensor_update": had_update,
            "prior_tensor": prior_tensor,
            "raw_update": raw_update,
            "updated_regions": (
                raw_update.get("updated_regions", []) if raw_update else []
            ),
            "tensor": session.tensor,
            # Response — full text for replay
            "response_text": response_text,
            "clean_text": clean_text,
            "response_text_len": len(response_text),
            "clean_text_len": len(clean_text),
            "content_blocks": content_blocks,
            # Request — what was actually sent to the API
            "request_body": request_body,
            # Timing
            "timing": timing,
            # Feedback that was generated for this cycle
            "feedback_generated": feedback,
            # Loss tracking
            "cycle_losses": tensor.get("declared_losses", []),
            "cumulative_loss_count": len(session.loss_history),
            "loss_history": session.loss_history,
            "cycle_integration_losses": [
                e for e in session.integration_loss_history
                if e["cycle"] == session.cycle
            ],
            # Tensor health
            "n_strands": len(tensor.get("strands", [])),
            "n_open_questions": len(tensor.get("open_questions", [])),
            "n_tensions": len(tensor.get("unresolved_tensions", [])),
            "tensor_token_estimate": session.tensor_token_estimate(),
            # Memory objects
            "n_memory_objects": len(session.memory_objects),
            "memory_token_estimate": session.memory_token_estimate(),
            "memory_objects_summary": [
                {
                    "id": m.id, "tool": m.tool, "label": m.label,
                    "tokens": m.tokens, "state": m.state,
                    "pinned": m.pinned, "turn": m.turn,
                }
                for m in session.memory_objects
            ],
            "memory_events": memory_events or [],
            # Curated memory objects (summarized/pinned) — needed for restore.
            # Full objects rebuild from messages, but curation state is lost
            # without this.
            "curated_memory": [
                {
                    "id": m.id, "tool": m.tool, "label": m.label,
                    "content": m.content, "tokens": m.tokens,
                    "turn": m.turn, "cycle": m.cycle,
                    "state": m.state, "pinned": m.pinned,
                    "tool_use_id": m.tool_use_id,
                    "original_tokens": m.original_tokens,
                }
                for m in session.memory_objects
                if m.state != "full" or m.pinned
            ],
            # Usage from API — raw and derived
            "usage": usage,
            "cache_stats": cache_stats,
        }
        with open(session.log_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def _log_console(
        self, session: TasteSession, feedback: list[str],
    ) -> None:
        """Console status output."""
        t = session.tensor
        if t is None:
            log.info(
                "taste | session=%s cycle=%d (initializing)",
                session.session_id, session.cycle,
            )
            return

        n_strands = len(t.get("strands", []))
        n_tensions = len(t.get("unresolved_tensions", []))
        tokens = session.tensor_token_estimate()
        n_mem = len(session.memory_objects)
        mem_tokens = session.memory_token_estimate()
        log.info(
            "taste | session=%s cycle=%d strands=%d tensions=%d "
            "tensor=~%d memory=%d(~%d tok) losses=%d cumulative",
            session.session_id, session.cycle, n_strands, n_tensions,
            tokens, n_mem, mem_tokens, len(session.loss_history),
        )
        for f in feedback:
            log.info("  %s", f)
