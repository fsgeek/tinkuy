"""Anthropic API payload synthesis from projection.

LiveAdapter reads the projection and produces valid Anthropic API
message payloads with proper alternation, cache breakpoints, and
content block sanitization.

THIS IS THE ANTI-PROXY-GRAVITY BOUNDARY.

The live adapter does not preserve or pass through the original
client messages. The projection is the source of truth.
"""

from __future__ import annotations

from typing import Any

from tinkuy.core.adapter import _has_content, coalesce_episodes
from tinkuy.core.orchestrator import Orchestrator
from tinkuy.core.regions import (
    ContentBlock,
    ContentKind,
    ContentStatus,
    Projection,
    RegionID,
)


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
        array with proper user/assistant alternation. The page table
        is injected here (not by the gateway) because placement must
        respect tool_result ordering and other layout constraints.
        """
        projection = self.orchestrator.projection
        payload: dict[str, Any] = {}

        # R0 + R1 → system prompt
        system_parts = self._collect_system(projection)
        if system_parts:
            payload["system"] = system_parts

        # R2 + R3 + R4 → messages with proper alternation
        messages = self._collect_messages(projection)

        # Page table → last user message, after any tool_result blocks.
        # The page table is per-turn volatile and must NOT go in the
        # system block (cache-busting) or before tool_results (API error).
        page_table = self.synthesize_page_table()
        if page_table:
            self._inject_page_table(messages, page_table)

        payload["messages"] = messages
        return payload

    def _inject_page_table(
        self, messages: list[dict[str, Any]], page_table: str
    ) -> None:
        """Place page table in the last user message, after tool_results.

        The Anthropic API requires tool_result blocks to appear before
        other content in a user message when the prior assistant message
        had tool_use blocks. The page table goes after all tool_results.
        """
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") != "user":
                continue
            msg = messages[i]
            content = msg["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]

            # Find insertion point: after the last tool_result
            insert_idx = 0
            for j, block in enumerate(content):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    insert_idx = j + 1

            content.insert(insert_idx, {
                "type": "text",
                "text": page_table,
            })
            msg["content"] = content
            return

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

        # Repair tool_use/tool_result pairing — strip orphaned tool_use
        # blocks that don't have matching tool_results in the next message
        messages = self._repair_tool_pairing(messages)

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

        # Determine role and content based on block kind
        if block.kind == ContentKind.TENSOR:
            role = "assistant"
            content: str | list[dict[str, Any]] = block.content
        elif block.kind == ContentKind.TOOL_RESULT:
            role = "user"
            tool_use_id = block.metadata.get("tool_use_id", block.label)
            result_content = block.content
            if block.status == ContentStatus.AVAILABLE:
                result_content = (
                    f"[tensor:{block.handle[:8]} — evicted tool result]"
                )
            tool_result_block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": result_content,
            }
            if block.metadata.get("is_error"):
                tool_result_block["is_error"] = True
            content = [tool_result_block]
        elif block.kind == ContentKind.CONVERSATION:
            role = "assistant" if "assistant" in block.label else "user"
            if block.status == ContentStatus.AVAILABLE:
                content = (
                    f"[tensor:{block.handle[:8]} — "
                    f"{block.label} "
                    f"({block.size_tokens} tokens)]"
                )
            else:
                stored_blocks = block.metadata.get("content_blocks")
                content = stored_blocks if stored_blocks else block.content
        else:
            role = "user"
            if block.status == ContentStatus.AVAILABLE:
                content = (
                    f"[tensor:{block.handle[:8]} — "
                    f"{block.label} "
                    f"({block.size_tokens} tokens)]"
                )
            else:
                stored_blocks = block.metadata.get("content_blocks")
                content = stored_blocks if stored_blocks else block.content

        msg: dict[str, Any] = {
            "role": role,
            "content": content,
            "_region": region.name,
            "_handle": block.handle,
        }

        # Cache control hints based on region.
        # R2 (durable) and R3 (ephemeral) both get hints — content in
        # both regions is stable between turns. Only R4 (current) changes.
        if region in (RegionID.DURABLE, RegionID.EPHEMERAL):
            msg["_cache_control"] = {"type": "ephemeral"}
            msg["_cache_tier"] = region.name

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
                # Merge consecutive same-role messages.
                # Content may be str or list (content blocks).  When
                # either side is a list (e.g. tool_result blocks),
                # merge into a unified content-block list so the API
                # sees valid structured content.
                prev = result[-1]
                prev_content = prev["content"]
                new_content = content

                # Normalize both sides to lists
                if isinstance(prev_content, str):
                    prev_content = [{"type": "text", "text": prev_content}]
                if isinstance(new_content, str):
                    new_content = [{"type": "text", "text": new_content}]

                if isinstance(prev_content, list) and isinstance(new_content, list):
                    prev["content"] = prev_content + new_content
                else:
                    # Fallback — should not happen
                    prev["content"] = str(prev_content) + "\n" + str(new_content)

                # Preserve cache hints through merge. When merging messages
                # from different tiers, the later message's tier wins (it
                # determines the boundary position in _finalize_messages).
                if "_cache_control" in msg:
                    prev["_cache_control"] = msg["_cache_control"]
                if "_cache_tier" in msg:
                    prev["_cache_tier"] = msg["_cache_tier"]
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

    def _repair_tool_pairing(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Strip tool_use blocks that lack matching tool_results.

        The API requires every tool_use in an assistant message to have
        a matching tool_result in the immediately following user message.
        When the projection has orphaned tool_use blocks (e.g., from
        checkpoint restore or bootstrap), strip them to prevent 400s.
        """
        import logging
        log = logging.getLogger(__name__)

        for i, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue

            # Collect tool_use IDs in this assistant message
            tool_use_ids = {
                b.get("id")
                for b in content
                if isinstance(b, dict) and b.get("type") == "tool_use"
            } - {None}

            if not tool_use_ids:
                continue

            # Check next message for matching tool_results
            if i + 1 < len(messages) and messages[i + 1].get("role") == "user":
                next_content = messages[i + 1].get("content", "")
                if isinstance(next_content, list):
                    result_ids = {
                        b.get("tool_use_id")
                        for b in next_content
                        if isinstance(b, dict) and b.get("type") == "tool_result"
                    } - {None}
                else:
                    result_ids = set()
            else:
                result_ids = set()

            # Strip orphaned tool_use blocks
            orphaned = tool_use_ids - result_ids
            if orphaned:
                log.warning(
                    "stripping %d orphaned tool_use blocks at messages[%d]: %s",
                    len(orphaned), i, orphaned,
                )
                msg["content"] = [
                    b for b in content
                    if not (
                        isinstance(b, dict)
                        and b.get("type") == "tool_use"
                        and b.get("id") in orphaned
                    )
                ]
                # If content is now empty or only has empty text, use placeholder
                if not msg["content"] or all(
                    isinstance(b, dict) and b.get("type") == "text" and not b.get("text", "").strip()
                    for b in msg["content"]
                    if isinstance(b, dict)
                ):
                    msg["content"] = "[tool calls omitted]"

        # Ensure tool_result blocks come first in user messages that
        # follow tool_use. Content from different regions (R3/R4) can
        # get merged by alternation enforcement with wrong ordering.
        for i, msg in enumerate(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue

            # Check if previous message has tool_use
            if i > 0 and messages[i - 1].get("role") == "assistant":
                prev_content = messages[i - 1].get("content", "")
                has_tool_use = (
                    isinstance(prev_content, list)
                    and any(
                        isinstance(b, dict) and b.get("type") == "tool_use"
                        for b in prev_content
                    )
                )
                if has_tool_use:
                    # Partition: tool_results first, everything else after
                    tool_results = [
                        b for b in content
                        if isinstance(b, dict) and b.get("type") == "tool_result"
                    ]
                    rest = [
                        b for b in content
                        if not (isinstance(b, dict) and b.get("type") == "tool_result")
                    ]
                    if tool_results and rest:
                        msg["content"] = tool_results + rest

        return messages

    def _finalize_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Place cache breakpoints at region boundaries, strip annotations.

        Anthropic caches from the start of the request up to each
        cache_control breakpoint (max 4). We use up to 2 in messages:

        1. Last R2 (durable) message — the stable curated content
        2. Last R3 (ephemeral) message — stable between turns, only
           changes when eviction runs

        R4 (current turn) is never cached — it changes every turn.
        This gives us: system(1) + R2(1) + R3(1) = 3 of 4 budget.

        cache_control goes on a content block, not the message dict,
        so we convert the message's string content to a content array.
        """
        # Find the last message in each cache tier
        last_durable_idx = None
        last_ephemeral_idx = None
        for i, msg in enumerate(messages):
            tier = msg.get("_cache_tier")
            if tier == "DURABLE":
                last_durable_idx = i
            elif tier == "EPHEMERAL":
                last_ephemeral_idx = i

        # Collect the indices that get breakpoints
        breakpoint_indices = set()
        if last_durable_idx is not None:
            breakpoint_indices.add(last_durable_idx)
        if last_ephemeral_idx is not None:
            breakpoint_indices.add(last_ephemeral_idx)

        result: list[dict[str, Any]] = []
        for i, msg in enumerate(messages):
            clean = {k: v for k, v in msg.items() if not k.startswith("_")}

            # Sanitize content blocks — strip fields the API won't accept
            if isinstance(clean.get("content"), list):
                clean["content"] = [
                    _sanitize_content_block(b) for b in clean["content"]
                ]

            if i in breakpoint_indices:
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
        episodes = coalesce_episodes(coalescable, current_turn)

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
            attrs = (
                f'handle="{e["handle"]}" '
                f'kind="{e["kind"]}" '
                f'status="{e["status"]}" '
                f'size_tokens="{e["size_tokens"]}" '
                f'faults="{e["fault_count"]}" '
                f'age_turns="{e["age_turns"]}" '
                f'label="{e["label"]}"'
            )
            if e.get("depends_on"):
                attrs += f' depends_on="{",".join(e["depends_on"])}"'
            lines.append(f'  <entry {attrs}/>')

        lines.append("</yuyay-page-table>")
        return "\n".join(lines)
