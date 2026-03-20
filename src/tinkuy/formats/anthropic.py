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
