"""System-block synthesizer: stability-ordered context stack.

Serializes the entire projection as system blocks with cache_control
breakpoints between stability tiers:

  R1 (tools + system, stable forever)
    → cache_control breakpoint
  R2 (durable, stable per-session)
    → cache_control breakpoint
  R3 (ephemeral, stable across tool calls within a human turn)
    → cache_control breakpoint
  R4 (current turn, uncached)
  [user message — messages array]

No user/assistant alternation. No message framing. No orphan repair.
No tool pairing. The entire pathology family is eliminated.

The only message is the current user turn. Everything else is
system context, ordered by how often it changes.

THIS IS THE ANTI-PROXY-GRAVITY BOUNDARY.
"""

from __future__ import annotations

import logging
from typing import Any

from tinkuy.core.adapter import coalesce_episodes
from tinkuy.core.orchestrator import Orchestrator
from tinkuy.core.pressure import PressureZone
from tinkuy.core.regions import (
    ContentBlock,
    ContentKind,
    ContentStatus,
    Projection,
    RegionID,
)

log = logging.getLogger(__name__)

_CACHE_BREAKPOINT = {"type": "ephemeral"}


class SystemBlockSynthesizer:
    """Serialize projection regions as system blocks.

    Each stability tier becomes one or more text blocks in the system
    array. Cache breakpoints go between tiers. The messages array
    contains only the current user turn.

    This replaces the LiveAdapter's message-based synthesis for the
    Anthropic path. The projection is the source of truth — we never
    pass through client messages.
    """

    def __init__(self, orchestrator: Orchestrator) -> None:
        self.orchestrator = orchestrator
        self.last_page_table_tokens: int = 0

    def synthesize(
        self,
        *,
        skip_page_table: bool = False,
    ) -> dict[str, Any]:
        """Produce a complete API payload.

        Returns:
            {"system": [...blocks...], "messages": [...]}
        """
        projection = self.orchestrator.projection
        system: list[dict[str, Any]] = []

        # --- R1: tools + system instructions (stable forever) ---
        r1_text = self._serialize_region(projection, RegionID.TOOLS, RegionID.SYSTEM)
        if r1_text:
            system.append({"type": "text", "text": r1_text})

        # Breakpoint after R1
        if system:
            system[-1]["cache_control"] = _CACHE_BREAKPOINT.copy()

        # --- R2: durable content (stable per-session) ---
        r2_text = self._serialize_region(projection, RegionID.DURABLE)
        if r2_text:
            system.append({"type": "text", "text": r2_text})
            system[-1]["cache_control"] = _CACHE_BREAKPOINT.copy()

        # --- R3: ephemeral content (stable across tool calls) ---
        r3_text = self._serialize_region(projection, RegionID.EPHEMERAL)
        if r3_text:
            system.append({"type": "text", "text": r3_text})
            system[-1]["cache_control"] = _CACHE_BREAKPOINT.copy()

        # --- R4: current turn content (uncached) ---
        r4_parts: list[str] = []
        r4_text = self._serialize_region(projection, RegionID.CURRENT)
        if r4_text:
            r4_parts.append(r4_text)

        # Page table goes in R4 (per-turn volatile)
        if not skip_page_table:
            page_table = self.synthesize_page_table()
            if page_table:
                r4_parts.append(page_table)
                self.last_page_table_tokens = len(page_table) // 4
        else:
            self.last_page_table_tokens = 0

        if r4_parts:
            system.append({"type": "text", "text": "\n\n".join(r4_parts)})
            # No cache_control on R4 — changes every API call

        payload: dict[str, Any] = {"system": system, "messages": []}
        return payload

    # ------------------------------------------------------------------
    # Region serialization
    # ------------------------------------------------------------------

    def _serialize_region(
        self,
        projection: Projection,
        *region_ids: RegionID,
    ) -> str:
        """Serialize all PRESENT blocks from the given regions as text.

        Each block gets a role-tagged header so the model can follow
        the conversational structure. Evicted blocks get tensor markers.

        User messages in CURRENT (R4) are skipped — they belong in
        the messages array, not in system blocks. Including them in
        both would double-count tokens.
        """
        parts: list[str] = []
        for rid in region_ids:
            region = projection.region(rid)
            for block in region.blocks:
                # Skip user messages in CURRENT — they go in messages
                if (rid == RegionID.CURRENT
                        and block.kind == ContentKind.CONVERSATION
                        and "user" in block.label):
                    continue
                rendered = self._render_block(block)
                if rendered:
                    parts.append(rendered)
        return "\n\n".join(parts)

    def _render_block(self, block: ContentBlock) -> str:
        """Render a single content block as text for a system block.

        PRESENT blocks get their full content with role markers.
        AVAILABLE (evicted) blocks get tensor markers showing what
        was there and how to recall it.
        """
        if block.status == ContentStatus.AVAILABLE:
            return self._render_evicted(block)
        if block.status != ContentStatus.PRESENT:
            return ""

        # System and tensor content render directly — no role marker
        if block.kind in (ContentKind.SYSTEM, ContentKind.TENSOR):
            return block.content

        # Conversation blocks get role markers
        if block.kind == ContentKind.CONVERSATION:
            role = "assistant" if "assistant" in block.label else "user"
            # If we have structured content_blocks, extract text
            stored = block.metadata.get("content_blocks")
            if stored and isinstance(stored, list):
                text = self._extract_text_from_blocks(stored)
            else:
                text = block.content
            return f"[{role}] {text}" if text else ""

        # Tool results get a minimal representation
        if block.kind == ContentKind.TOOL_RESULT:
            tool_id = block.metadata.get("tool_use_id", block.label)
            return f"[tool_result:{tool_id}] {block.content}"

        # File content
        if block.kind == ContentKind.FILE:
            return f"[file:{block.label}] {block.content}"

        return block.content

    def _render_evicted(self, block: ContentBlock) -> str:
        """Render an evicted block as a tensor marker."""
        return (
            f"[evicted:{block.handle[:8]} — "
            f"{block.label} ({block.size_tokens} tokens) — "
            f"use <recall handle=\"{block.handle}\"/> to restore]"
        )

    def _extract_text_from_blocks(
        self, blocks: list[dict[str, Any]]
    ) -> str:
        """Extract text content from Anthropic content blocks.

        Preserves tool_use summaries so the model knows what tools
        were called, but strips the bulky input payloads.
        """
        parts: list[str] = []
        for b in blocks:
            if not isinstance(b, dict):
                continue
            btype = b.get("type", "")
            if btype == "text":
                text = b.get("text", "")
                if text.strip():
                    parts.append(text)
            elif btype == "tool_use":
                name = b.get("name", "?")
                parts.append(f"[called {name}]")
            # tool_result, thinking, etc. — skip (they're separate blocks)
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Page table — same format as LiveAdapter, reusable
    # ------------------------------------------------------------------

    def synthesize_page_table(self) -> str:
        """Synthesize the page table as text.

        Uses episodic coalescing: consecutive present blocks are grouped
        into episode summaries. Individual entries are preserved for
        evicted/faultable blocks, hot blocks, and recent blocks.
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

        outcomes = self.orchestrator.signal_outcomes
        if outcomes:
            lines.append("  <signal-feedback>")
            for o in outcomes:
                attrs = f'signal="{o["signal"]}" handle="{o["handle"]}" outcome="{o["outcome"]}"'
                if o.get("reason"):
                    attrs += f' reason="{o["reason"]}"'
                if o.get("edges"):
                    attrs += f' edges="{o["edges"]}"'
                lines.append(f'    <outcome {attrs}/>')
            lines.append("  </signal-feedback>")

        # Pressure advisory — tell the model about pressure state so it
        # can cooperate. At MODERATE, blocks are being nominated but not
        # yet force-evicted — this is the model's window to produce tensors.
        # At ELEVATED/CRITICAL, force-eviction is imminent.
        pressure = self.orchestrator.scheduler.read_pressure(
            self.orchestrator.projection
        )
        pending = [
            e for e in entries
            if e["status"] == "pending_removal"
        ]
        if pressure.zone in (
            PressureZone.MODERATE,
            PressureZone.ELEVATED,
            PressureZone.CRITICAL,
        ) or pending:
            urgency = {
                PressureZone.MODERATE: "approaching limits",
                PressureZone.ELEVATED: "high — force-eviction imminent",
                PressureZone.CRITICAL: "critical — force-evicting now",
            }.get(pressure.zone, "")

            lines.append(
                f'  <pressure zone="{pressure.zone.name}" '
                f'usage="{pressure.usage:.0%}" '
                f'pending="{len(pending)}">'
            )
            if pending:
                if pressure.zone in (
                    PressureZone.ELEVATED, PressureZone.CRITICAL,
                ):
                    lines.append(
                        f"    URGENT ({urgency}): The following blocks will be "
                        "force-evicted WITHOUT summary if you do not act now. "
                        "For each one, emit <release/> with a <tensor/> or "
                        "<retain/> if still needed."
                    )
                else:
                    lines.append(
                        f"    Context pressure is {urgency}. The following "
                        "blocks are scheduled for eviction. Emit <release/> "
                        "with a <tensor/> for blocks you can summarize, or "
                        "<retain/> for blocks you still need."
                    )
                for p in pending:
                    lines.append(
                        f'    <pending handle="{p["handle"]}" '
                        f'label="{p["label"]}" '
                        f'size_tokens="{p["size_tokens"]}" '
                        f'age_turns="{p["age_turns"]}"/>'
                    )
            elif urgency:
                lines.append(
                    f"    Context pressure is {urgency}. Consider releasing "
                    "blocks you no longer need to free space."
                )
            lines.append("  </pressure>")

        lines.append("</yuyay-page-table>")
        return "\n".join(lines)
