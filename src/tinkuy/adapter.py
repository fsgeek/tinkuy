"""Adapters for ingesting conversations and synthesizing API payloads.

Two adapters, one orchestrator:

1. IngestAdapter — batch-replays a conversation log into a projection.
   Supports multiple input formats (Anthropic messages, Claude Code
   JSONL, raw message lists). This is how you rehydrate tinkuy from
   an existing conversation.

2. LiveAdapter — sits in the message stream of a live session, feeding
   each turn into the orchestrator and synthesizing the projection back
   into valid Anthropic API message payloads.

The critical discipline: API payloads are SYNTHESIZED FROM the
projection, never passed through from the client. The projection
is the source of truth. This is the anti-proxy-gravity boundary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Iterator

from tinkuy.orchestrator import (
    EventType,
    InboundEvent,
    Orchestrator,
    ResponseSignal,
    ResponseSignalType,
)
from tinkuy.regions import (
    ContentBlock,
    ContentKind,
    ContentStatus,
    Projection,
    RegionID,
)


# --- Conversation formats ---


class ConversationFormat(Enum):
    """Supported conversation log formats."""
    ANTHROPIC_MESSAGES = auto()   # Anthropic API messages format
    JSONL = auto()                # One JSON message per line
    RAW_MESSAGES = auto()         # Simple list of {role, content} dicts


@dataclass
class ConversationMessage:
    """Normalized representation of a single conversation message."""
    role: str              # "user", "assistant", "system", "tool"
    content: str
    turn: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_user(self) -> bool:
        return self.role == "user"

    @property
    def is_assistant(self) -> bool:
        return self.role == "assistant"

    @property
    def is_system(self) -> bool:
        return self.role == "system"

    @property
    def is_tool(self) -> bool:
        return self.role in ("tool", "tool_result")


# --- Format parsers ---


def parse_anthropic_messages(
    data: dict[str, Any],
) -> list[ConversationMessage]:
    """Parse Anthropic API messages format.

    Handles both simple string content and content block arrays.
    Extracts system prompt if present.
    """
    messages: list[ConversationMessage] = []

    # System prompt (top-level in Anthropic format)
    system = data.get("system")
    if system:
        if isinstance(system, str):
            messages.append(ConversationMessage(role="system", content=system))
        elif isinstance(system, list):
            # Content block array
            text = _extract_text_from_blocks(system)
            if text:
                messages.append(
                    ConversationMessage(role="system", content=text)
                )

    # Messages
    for i, msg in enumerate(data.get("messages", [])):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            messages.append(
                ConversationMessage(
                    role=role, content=content, turn=i // 2
                )
            )
        elif isinstance(content, list):
            # Content block array — may contain text, tool_use, tool_result
            for block in content:
                block_type = block.get("type", "text")
                if block_type == "text":
                    messages.append(
                        ConversationMessage(
                            role=role,
                            content=block.get("text", ""),
                            turn=i // 2,
                        )
                    )
                elif block_type == "tool_use":
                    messages.append(
                        ConversationMessage(
                            role="tool",
                            content=json.dumps(block.get("input", {})),
                            turn=i // 2,
                            metadata={
                                "tool_name": block.get("name", ""),
                                "tool_use_id": block.get("id", ""),
                            },
                        )
                    )
                elif block_type == "tool_result":
                    result_content = block.get("content", "")
                    if isinstance(result_content, list):
                        result_content = _extract_text_from_blocks(
                            result_content
                        )
                    messages.append(
                        ConversationMessage(
                            role="tool_result",
                            content=str(result_content),
                            turn=i // 2,
                            metadata={
                                "tool_use_id": block.get(
                                    "tool_use_id", ""
                                ),
                            },
                        )
                    )

    return messages


def parse_jsonl(lines: list[str]) -> list[ConversationMessage]:
    """Parse JSONL format — one JSON message per line."""
    messages: list[ConversationMessage] = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        messages.append(
            ConversationMessage(
                role=data.get("role", "user"),
                content=data.get("content", ""),
                turn=data.get("turn", i // 2),
                metadata=data.get("metadata", {}),
            )
        )
    return messages


def parse_raw_messages(
    messages: list[dict[str, Any]],
) -> list[ConversationMessage]:
    """Parse simple {role, content} message list."""
    return [
        ConversationMessage(
            role=msg.get("role", "user"),
            content=msg.get("content", ""),
            turn=i // 2,
            metadata=msg.get("metadata", {}),
        )
        for i, msg in enumerate(messages)
    ]


# Allowed keys per content block type for the Anthropic API.
# Everything else (citations, parsed_output, etc.) must be stripped.
_ALLOWED_BLOCK_KEYS: dict[str, set[str]] = {
    "text": {"type", "text", "cache_control"},
    "tool_use": {"type", "id", "name", "input", "cache_control"},
    "tool_result": {"type", "tool_use_id", "content", "is_error", "cache_control"},
    "thinking": {"type", "thinking", "signature", "cache_control"},
    "image": {"type", "source", "cache_control"},
}


def _sanitize_content_block(block: dict[str, Any]) -> dict[str, Any]:
    """Strip fields the Anthropic API won't accept on re-submission."""
    if not isinstance(block, dict):
        return block
    block_type = block.get("type", "")
    allowed = _ALLOWED_BLOCK_KEYS.get(block_type)
    if allowed is None:
        # Unknown block type — pass through as-is
        return block
    return {k: v for k, v in block.items() if k in allowed}


def _has_content(content: str | list[dict[str, Any]]) -> bool:
    """Check if a message content field is non-empty."""
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        return len(content) > 0
    return bool(content)


def _extract_text_from_blocks(blocks: list[dict[str, Any]]) -> str:
    """Extract text content from an Anthropic content block array."""
    parts = []
    for block in blocks:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)


# --- Ingest adapter ---


class IngestAdapter:
    """Batch-replays a conversation log into an orchestrator.

    Walks through normalized messages, grouping by turn, and feeds
    them to the orchestrator. The result is a fully populated
    projection that can then be continued live.
    """

    def __init__(self, orchestrator: Orchestrator) -> None:
        self.orchestrator = orchestrator

    def ingest_messages(
        self, messages: list[ConversationMessage]
    ) -> None:
        """Replay a list of messages into the orchestrator."""
        # Group messages into turns
        for turn_messages in self._group_by_turn(messages):
            user_events: list[InboundEvent] = []
            assistant_content: list[str] = []

            for msg in turn_messages:
                if msg.is_system:
                    user_events.append(
                        InboundEvent(
                            type=EventType.SYSTEM_UPDATE,
                            content=msg.content,
                            label="system",
                            metadata=msg.metadata,
                        )
                    )
                elif msg.is_user:
                    user_events.append(
                        InboundEvent(
                            type=EventType.USER_MESSAGE,
                            content=msg.content,
                            label="user",
                            metadata=msg.metadata,
                        )
                    )
                elif msg.is_tool:
                    user_events.append(
                        InboundEvent(
                            type=EventType.TOOL_RESULT,
                            content=msg.content,
                            label=msg.metadata.get(
                                "tool_name", "tool_result"
                            ),
                            metadata=msg.metadata,
                        )
                    )
                elif msg.is_assistant:
                    assistant_content.append(msg.content)

            # Begin turn with user-side events
            if user_events:
                self.orchestrator.begin_turn(user_events)

            # Ingest assistant response if present
            if assistant_content:
                self.orchestrator.ingest_response(
                    content="\n".join(assistant_content),
                    label="assistant",
                )

    def ingest_anthropic(self, data: dict[str, Any]) -> None:
        """Ingest from Anthropic API messages format."""
        messages = parse_anthropic_messages(data)
        self.ingest_messages(messages)

    def ingest_jsonl(self, text: str) -> None:
        """Ingest from JSONL format."""
        messages = parse_jsonl(text.strip().split("\n"))
        self.ingest_messages(messages)

    def ingest_file(self, path: str | Path) -> None:
        """Ingest from a file, auto-detecting format."""
        path = Path(path)
        content = path.read_text(encoding="utf-8")

        if path.suffix == ".jsonl":
            self.ingest_jsonl(content)
        elif path.suffix == ".json":
            data = json.loads(content)
            if isinstance(data, dict) and "messages" in data:
                self.ingest_anthropic(data)
            elif isinstance(data, list):
                messages = parse_raw_messages(data)
                self.ingest_messages(messages)
            else:
                raise ValueError(
                    f"Unrecognized JSON format in {path}"
                )
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def _group_by_turn(
        self, messages: list[ConversationMessage]
    ) -> Iterator[list[ConversationMessage]]:
        """Group messages into turns.

        A turn is a sequence of messages that belong together —
        typically a user message (possibly with tool results)
        followed by an assistant response. System messages are
        placed in the first turn.
        """
        if not messages:
            return

        current_turn: list[ConversationMessage] = []
        last_role: str | None = None

        for msg in messages:
            # System messages always go with the current group
            if msg.is_system:
                current_turn.append(msg)
                continue

            # Start a new turn when we transition from assistant to user
            if last_role == "assistant" and not msg.is_assistant:
                if current_turn:
                    yield current_turn
                    current_turn = []

            current_turn.append(msg)
            last_role = "assistant" if msg.is_assistant else msg.role

        if current_turn:
            yield current_turn


# --- Live adapter ---


class LiveAdapter:
    """Synthesizes API payloads from the projection.

    THIS IS THE ANTI-PROXY-GRAVITY BOUNDARY.

    The live adapter reads the projection and produces valid Anthropic
    API message payloads. It does not preserve or pass through the
    original client messages. The projection is the source of truth.
    """

    def __init__(self, orchestrator: Orchestrator) -> None:
        self.orchestrator = orchestrator

    def synthesize_messages(self) -> dict[str, Any]:
        """Synthesize a complete Anthropic API messages payload.

        Reads the projection regions and constructs a valid messages
        array with proper user/assistant alternation.
        """
        projection = self.orchestrator.projection
        payload: dict[str, Any] = {}

        # R0 + R1 → system prompt
        system_parts = self._collect_system(projection)
        if system_parts:
            payload["system"] = system_parts

        # R2 + R3 + R4 → messages with proper alternation
        payload["messages"] = self._collect_messages(projection)

        return payload

    def _collect_system(self, projection: Projection) -> list[dict[str, Any]]:
        """Collect system content from R0 (tools) and R1 (system)."""
        parts: list[dict[str, Any]] = []

        for rid in (RegionID.TOOLS, RegionID.SYSTEM):
            region = projection.region(rid)
            for block in region.blocks:
                if block.status != ContentStatus.PRESENT:
                    continue
                part: dict[str, Any] = {
                    "type": "text",
                    "text": block.content,
                }
                parts.append(part)

        # Place one cache breakpoint on the last system block.
        # The API allows at most 4 breakpoints — spending one here
        # caches the entire system prompt prefix, leaving budget
        # for the durable message boundary in _finalize_messages.
        if parts:
            parts[-1]["cache_control"] = {"type": "ephemeral"}

        return parts

    def _collect_messages(
        self, projection: Projection
    ) -> list[dict[str, Any]]:
        """Collect conversation messages from R2, R3, R4.

        Produces valid Anthropic alternation: user/assistant pairs.
        Evicted content is replaced with tensor markers. Available
        (evicted) blocks emit their tensor content if it exists.
        """
        raw_messages: list[dict[str, Any]] = []

        for rid in (RegionID.DURABLE, RegionID.EPHEMERAL, RegionID.CURRENT):
            region = projection.region(rid)
            for block in region.blocks:
                msg = self._block_to_message(block, rid)
                if msg is not None:
                    raw_messages.append(msg)

        # Filter out empty content before enforcing alternation
        raw_messages = [
            m for m in raw_messages
            if _has_content(m.get("content", ""))
        ]

        # Enforce alternation
        messages = self._enforce_alternation(raw_messages)

        # Finalize: place cache breakpoint and strip internal annotations
        return self._finalize_messages(messages)

    def _block_to_message(
        self,
        block: ContentBlock,
        region: RegionID,
    ) -> dict[str, Any] | None:
        """Convert a content block to a message dict."""
        if block.kind == ContentKind.SYSTEM:
            return None  # System content handled separately

        # Determine role
        if block.kind == ContentKind.TENSOR:
            # Tensors are summaries — present as assistant content
            return {
                "role": "assistant",
                "content": block.content,
                "_region": region.name,
                "_handle": block.handle,
            }

        if block.kind == ContentKind.TOOL_RESULT:
            role = "user"  # Tool results are in user turns
        elif block.kind == ContentKind.CONVERSATION:
            # Infer role from label or default based on position
            role = "assistant" if "assistant" in block.label else "user"
        elif block.kind == ContentKind.FILE:
            role = "user"  # File content is user-side
        else:
            role = "user"

        # Handle evicted content
        if block.status == ContentStatus.AVAILABLE:
            # Content was evicted — emit tensor marker
            content: str | list[dict[str, Any]] = (
                f"[tensor:{block.handle[:8]} — "
                f"{block.label} "
                f"({block.size_tokens} tokens)]"
            )
        else:
            # Use full content blocks (text + tool_use) if available,
            # otherwise fall back to plain text string
            stored_blocks = block.metadata.get("content_blocks")
            if stored_blocks:
                content = stored_blocks
            else:
                content = block.content

        msg: dict[str, Any] = {
            "role": role,
            "content": content,
            "_region": region.name,
            "_handle": block.handle,
        }

        # Cache control hints based on region
        if region == RegionID.DURABLE:
            msg["_cache_control"] = {"type": "ephemeral"}

        return msg

    def _enforce_alternation(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Enforce strict user/assistant alternation.

        The Anthropic API requires messages to alternate between
        user and assistant roles. This merges consecutive same-role
        messages and inserts minimal padding where needed.
        """
        if not messages:
            return []

        result: list[dict[str, Any]] = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if result and result[-1]["role"] == role:
                # Merge consecutive same-role messages
                prev = result[-1]
                if isinstance(prev["content"], str):
                    prev["content"] = prev["content"] + "\n" + content
                else:
                    prev["content"] = str(prev["content"]) + "\n" + content
                # Preserve cache hint: if either message is durable, keep it
                if "_cache_control" in msg and "_cache_control" not in prev:
                    prev["_cache_control"] = msg["_cache_control"]
            else:
                # Need alternation — insert padding if needed
                if result and role == "assistant" and result[-1]["role"] == "assistant":
                    result.append({"role": "user", "content": "[continued]"})
                elif result and role == "user" and result[-1]["role"] == "user":
                    result.append({"role": "assistant", "content": "[continued]"})
                # Preserve metadata through alternation
                clean = {"role": role, "content": content}
                for k in msg:
                    if k.startswith("_"):
                        clean[k] = msg[k]
                result.append(clean)

        # Must start with user
        if result and result[0]["role"] != "user":
            result.insert(0, {"role": "user", "content": "[conversation start]"})

        return result

    def _finalize_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Place cache breakpoint on last durable message, strip annotations.

        Anthropic caches from the start of the request up to each
        cache_control breakpoint. We place one on the last message
        that carries _cache_control — the boundary between stable
        (durable) and volatile (ephemeral/current) conversation.

        cache_control goes on a content block, not the message dict,
        so we convert the message's string content to a content array.
        """
        # Find the last message with a cache hint
        last_cached_idx = None
        for i, msg in enumerate(messages):
            if "_cache_control" in msg:
                last_cached_idx = i

        result: list[dict[str, Any]] = []
        for i, msg in enumerate(messages):
            clean = {k: v for k, v in msg.items() if not k.startswith("_")}

            # Sanitize content blocks — strip fields the API won't accept
            if isinstance(clean.get("content"), list):
                clean["content"] = [
                    _sanitize_content_block(b) for b in clean["content"]
                ]

            if i == last_cached_idx:
                # Promote to content array with cache_control on last block
                content = clean["content"]
                if isinstance(content, str):
                    clean["content"] = [{
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"},
                    }]
                elif isinstance(content, list) and content:
                    content[-1]["cache_control"] = {"type": "ephemeral"}

            result.append(clean)

        return result

    def synthesize_page_table(self) -> str:
        """Synthesize the page table as text for system prompt injection.

        Uses episodic coalescing: consecutive present blocks are grouped
        into episode summaries. Individual entries are preserved only for
        blocks that are evicted/available (faultable) or have non-zero
        fault counts (hot). Recent blocks (age <= 2 turns) also get
        individual entries since the model may need fine-grained access.

        This is the "indirect mapping table" — the model sees episodes,
        not individual pages, unless a page needs individual attention.
        """
        entries = self.orchestrator.page_table()
        if not entries:
            return ""

        current_turn = self.orchestrator.turn

        # Partition: entries that need individual listing vs coalescing
        individual: list[dict[str, Any]] = []
        coalescable: list[dict[str, Any]] = []

        for e in entries:
            if (e["status"] != "present"
                    or e["fault_count"] > 0
                    or e["age_turns"] <= 2):
                individual.append(e)
            else:
                coalescable.append(e)

        # Group coalescable entries into episodes by turn proximity
        episodes = _coalesce_episodes(coalescable, current_turn)

        lines = ["<yuyay-page-table>"]

        # Episode summaries (older, stable content)
        for ep in episodes:
            lines.append(
                f'  <episode turns="{ep["turn_range"]}" '
                f'blocks="{ep["block_count"]}" '
                f'tokens="{ep["total_tokens"]}" '
                f'kinds="{ep["kinds"]}"/>'
            )

        # Individual entries (recent, evicted, or hot)
        for e in individual:
            lines.append(
                f'  <entry handle="{e["handle"]}" '
                f'kind="{e["kind"]}" '
                f'status="{e["status"]}" '
                f'size_tokens="{e["size_tokens"]}" '
                f'faults="{e["fault_count"]}" '
                f'age_turns="{e["age_turns"]}" '
                f'label="{e["label"]}"/>'
            )

        lines.append("</yuyay-page-table>")
        return "\n".join(lines)


def _coalesce_episodes(
    entries: list[dict[str, Any]],
    current_turn: int,
) -> list[dict[str, Any]]:
    """Group entries into temporal episodes.

    Consecutive entries (by created turn) within 3 turns of each other
    are merged into one episode. This is the simplest coalescing
    strategy — temporal proximity only, no topic analysis.
    """
    if not entries:
        return []

    # Sort by age descending (oldest first)
    sorted_entries = sorted(entries, key=lambda e: -e["age_turns"])

    episodes: list[dict[str, Any]] = []
    current_ep: list[dict[str, Any]] = [sorted_entries[0]]

    for entry in sorted_entries[1:]:
        prev = current_ep[-1]
        # Merge if within 3 turns of the previous entry
        if abs(entry["age_turns"] - prev["age_turns"]) <= 3:
            current_ep.append(entry)
        else:
            episodes.append(_summarize_episode(current_ep, current_turn))
            current_ep = [entry]

    if current_ep:
        episodes.append(_summarize_episode(current_ep, current_turn))

    return episodes


def _summarize_episode(
    entries: list[dict[str, Any]],
    current_turn: int,
) -> dict[str, Any]:
    """Produce a summary for a group of entries."""
    ages = [e["age_turns"] for e in entries]
    oldest = max(ages)
    newest = min(ages)

    if oldest == newest:
        turn_range = f"{current_turn - oldest}"
    else:
        turn_range = f"{current_turn - oldest}-{current_turn - newest}"

    kinds = set(e["kind"] for e in entries)

    return {
        "turn_range": turn_range,
        "block_count": len(entries),
        "total_tokens": sum(e["size_tokens"] for e in entries),
        "kinds": "+".join(sorted(kinds)),
    }
