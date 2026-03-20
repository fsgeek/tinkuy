"""LiteLLM/OpenAI-format payload synthesis from projection.

Produces OpenAI-shaped message arrays that litellm can route to any
provider (Anthropic, OpenAI, Gemini, etc.). System content goes as
role="system" messages. cache_control is preserved through litellm's
translation layer.

This adapter does NOT enforce alternation — litellm handles that
per-provider. It does NOT sanitize content blocks to Anthropic's
whitelist — litellm's provider translation handles format differences.

The projection is the source of truth. Same anti-proxy-gravity
discipline as the Anthropic adapter, different output format.
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


class LiteLLMAdapter:
    """Synthesizes OpenAI-format messages from the projection.

    Output is suitable for litellm.completion(messages=...).
    System content is inline as role="system" messages.
    """

    def __init__(self, orchestrator: Orchestrator) -> None:
        self.orchestrator = orchestrator

    def synthesize_messages(self) -> dict[str, Any]:
        """Synthesize a complete message payload.

        Returns {"messages": [...]} where messages are OpenAI-format
        dicts with role, content, and optional cache_control.
        """
        projection = self.orchestrator.projection

        messages: list[dict[str, Any]] = []

        # System content (R0 + R1) → role="system" messages
        system_text = self._collect_system_text(projection)
        if system_text:
            msg: dict[str, Any] = {
                "role": "system",
                "content": system_text,
            }
            # cache_control on system message — litellm preserves this
            # and places it correctly per provider
            msg["cache_control"] = {"type": "ephemeral"}
            messages.append(msg)

        # Conversation content (R2 + R3 + R4)
        messages.extend(self._collect_messages(projection))

        return {"messages": messages}

    def _collect_system_text(self, projection: Projection) -> str:
        """Collect system content as a single text string."""
        parts: list[str] = []
        for rid in (RegionID.TOOLS, RegionID.SYSTEM):
            region = projection.region(rid)
            for block in region.blocks:
                if block.status == ContentStatus.PRESENT and block.content:
                    parts.append(block.content)
        return "\n\n".join(parts) if parts else ""

    def _collect_messages(
        self, projection: Projection
    ) -> list[dict[str, Any]]:
        """Collect conversation messages from R2, R3, R4."""
        raw_messages: list[dict[str, Any]] = []

        for rid in (RegionID.DURABLE, RegionID.EPHEMERAL, RegionID.CURRENT):
            region = projection.region(rid)
            for block in region.blocks:
                msg = self._block_to_message(block, rid)
                if msg is not None:
                    raw_messages.append(msg)

        # Filter empties
        raw_messages = [
            m for m in raw_messages
            if _has_content(m.get("content", ""))
        ]

        # Strip internal annotations
        return [
            {k: v for k, v in m.items() if not k.startswith("_")}
            for m in raw_messages
        ]

    def _block_to_message(
        self,
        block: ContentBlock,
        region: RegionID,
    ) -> dict[str, Any] | None:
        """Convert a content block to an OpenAI-format message dict."""
        if block.kind == ContentKind.SYSTEM:
            return None  # Handled in _collect_system_text

        # Determine role
        if block.kind == ContentKind.TENSOR:
            return {
                "role": "assistant",
                "content": block.content,
            }

        if block.kind == ContentKind.TOOL_RESULT:
            role = "user"
        elif block.kind == ContentKind.CONVERSATION:
            role = "assistant" if "assistant" in block.label else "user"
        elif block.kind == ContentKind.FILE:
            role = "user"
        else:
            role = "user"

        # Handle evicted content
        if block.status == ContentStatus.AVAILABLE:
            content = (
                f"[tensor:{block.handle[:8]} — "
                f"{block.label} "
                f"({block.size_tokens} tokens)]"
            )
        else:
            # For litellm, always use string content — content blocks
            # are provider-specific and litellm expects strings
            content = block.content

        return {
            "role": role,
            "content": content,
        }

    def synthesize_page_table(self) -> str:
        """Synthesize the page table as text for system prompt injection.

        Identical to the Anthropic adapter — the page table format is
        model-facing, not API-facing, so it's provider-independent.
        """
        entries = self.orchestrator.page_table()
        if not entries:
            return ""

        current_turn = self.orchestrator.turn

        individual: list[dict[str, Any]] = []
        coalescable: list[dict[str, Any]] = []

        for e in entries:
            if (e["status"] != "present"
                    or e["fault_count"] > 0
                    or e["age_turns"] <= 2):
                individual.append(e)
            else:
                coalescable.append(e)

        episodes = coalesce_episodes(coalescable, current_turn)

        lines = ["<yuyay-page-table>"]

        for ep in episodes:
            lines.append(
                f'  <episode turns="{ep["turn_range"]}" '
                f'blocks="{ep["block_count"]}" '
                f'tokens="{ep["total_tokens"]}" '
                f'kinds="{ep["kinds"]}"/>'
            )

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
