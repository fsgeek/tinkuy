"""Pressure-gated eviction policy.

Eviction is pressure-gated, not age-based. The context usage determines
the pressure zone, which determines eviction behavior. Age and other
signals inform *candidate selection*, not eviction triggers.

Pressure zones:
  Low      (< 50%)   — no eviction, hold everything
  Moderate (50-70%)  — schedule candidates, execute only at idle
  Elevated (70-85%)  — request model cooperation (release signals)
  Critical (> 85%)   — gateway-initiated pending_removal, aggressive

The scheduler is the single authority on what gets evicted and when.
All removal nominations are advisory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from tinkuy.core.regions import (
    ContentBlock,
    ContentKind,
    ContentStatus,
    Projection,
    RegionID,
    RemovalNomination,
)


class PressureZone(Enum):
    """Context pressure zones, ordered by severity."""
    LOW = auto()        # < 50% — hold everything
    MODERATE = auto()   # 50-70% — schedule candidates, idle-only execution
    ELEVATED = auto()   # 70-85% — request model cooperation
    CRITICAL = auto()   # > 85% — aggressive gateway-initiated removal


@dataclass
class PressureState:
    """Current pressure readings."""
    total_tokens: int = 0
    context_limit: int = 200_000
    waste_tokens: int = 0

    @property
    def usage(self) -> float:
        """Context usage as a fraction (0.0 to 1.0)."""
        if self.context_limit == 0:
            return 1.0
        return self.total_tokens / self.context_limit

    @property
    def zone(self) -> PressureZone:
        u = self.usage
        if u < 0.50:
            return PressureZone.LOW
        elif u < 0.70:
            return PressureZone.MODERATE
        elif u < 0.85:
            return PressureZone.ELEVATED
        else:
            return PressureZone.CRITICAL

    @property
    def headroom_tokens(self) -> int:
        """Tokens available before hitting the context limit."""
        return max(0, self.context_limit - self.total_tokens)


@dataclass
class EvictionCandidate:
    """A scored candidate for eviction."""
    block: ContentBlock
    score: float           # higher = more evictable
    reasons: list[str] = field(default_factory=list)


class PressureScheduler:
    """The single authority on eviction decisions.

    The scheduler reads the projection state, computes pressure,
    selects eviction candidates, and returns decisions. It does
    not mutate the projection — the orchestrator applies decisions.
    """

    def __init__(self, context_limit: int = 200_000) -> None:
        self.context_limit = context_limit
        self.overhead_tokens: int = 0  # passthrough cost (tools, system, etc.)
        # Note: max_tokens is an output cap, not an input reservation.
        # The API does not reject requests where input + max_tokens
        # exceeds the context limit — it just truncates generation.
        # Weights for candidate scoring
        self._kind_weights: dict[ContentKind, float] = {
            ContentKind.TOOL_RESULT: 1.5,    # most evictable
            ContentKind.FILE: 1.2,           # faultable, so safe to evict
            ContentKind.CONVERSATION: 0.8,   # harder to reconstruct
            ContentKind.TENSOR: 0.1,         # almost never evict tensors
            ContentKind.SYSTEM: 0.0,         # never evict system content
        }

    def read_pressure(self, projection: Projection) -> PressureState:
        """Read current pressure from the projection.

        The effective context limit is reduced by the passthrough
        overhead (tools, system prompt, etc.) so that pressure
        reflects the actual budget available for projection content.
        """
        effective_limit = max(
            0, self.context_limit - self.overhead_tokens
        )
        return PressureState(
            total_tokens=projection.total_tokens,
            context_limit=effective_limit,
            waste_tokens=projection.waste_tokens,
        )

    def score_candidate(
        self, block: ContentBlock, current_turn: int,
        dependents: int = 0,
    ) -> EvictionCandidate:
        """Score a single block for eviction candidacy.

        Higher score = more evictable. Scoring factors:
        - Content kind (tool results > files > conversation >> tensors)
        - Age since last access (older = more evictable)
        - Size (larger blocks free more space)
        - Fault history (frequently recalled = less evictable)
        - Dependency edges (blocks depended on by others = less evictable)
        """
        reasons: list[str] = []
        score = 0.0

        # Kind weight
        kind_w = self._kind_weights.get(block.kind, 0.5)
        score += kind_w * 10.0
        if kind_w > 1.0:
            reasons.append(f"kind={block.kind.name} (high evictability)")

        # Age: turns since last access
        age = current_turn - block.access.last_access_turn
        if age > 0:
            # Logarithmic age scaling — diminishing returns past ~20 turns
            import math
            age_score = math.log2(1 + age) * 3.0
            score += age_score
            if age > 5:
                reasons.append(f"age={age} turns since last access")

        # Size bonus: larger blocks free more space (mild preference)
        if block.size_tokens > 0:
            import math
            size_score = math.log2(1 + block.size_tokens / 100) * 1.0
            score += size_score
            if block.size_tokens > 2000:
                reasons.append(f"size={block.size_tokens} tokens")

        # Fault penalty: frequently recalled content is valuable
        if block.access.fault_count > 0:
            fault_penalty = block.access.fault_count * 5.0
            score -= fault_penalty
            reasons.append(
                f"fault_count={block.access.fault_count} (penalty)"
            )

        # Edge penalty: blocks that other blocks depend on are load-bearing.
        # Evicting them breaks the coherence graph — downstream blocks lose
        # their provenance chain. Each dependent adds a penalty.
        if dependents > 0:
            edge_penalty = dependents * 8.0
            score -= edge_penalty
            reasons.append(f"dependents={dependents} (edge penalty)")

        # Already nominated? Slight bonus.
        if block.status == ContentStatus.PENDING_REMOVAL:
            score += 5.0
            reasons.append("already pending removal")

        return EvictionCandidate(block=block, score=score, reasons=reasons)

    def _build_dependent_counts(self, projection: Projection) -> dict[str, int]:
        """Build a reverse index: handle → number of blocks that depend on it.

        Walks all blocks in all regions, collecting depends_on edges.
        Returns a count of how many live blocks reference each handle.
        """
        counts: dict[str, int] = {}
        for region in projection.regions.values():
            for block in region.blocks:
                if block.status == ContentStatus.AVAILABLE:
                    continue  # evicted blocks don't count as live dependents
                for dep in block.metadata.get("depends_on", []):
                    counts[dep] = counts.get(dep, 0) + 1
        return counts

    def select_candidates(
        self, projection: Projection, *, limit: int = 10
    ) -> list[EvictionCandidate]:
        """Select and rank eviction candidates from the projection.

        Only considers blocks in R3 (ephemeral) and R2 (durable, excluding
        tensors). R0, R1, R4 are not candidates.
        """
        candidates: list[EvictionCandidate] = []
        dep_counts = self._build_dependent_counts(projection)

        for rid in (RegionID.EPHEMERAL, RegionID.DURABLE):
            region = projection.region(rid)
            for block in region.blocks:
                # Skip already-evicted blocks
                if block.status == ContentStatus.AVAILABLE:
                    continue
                # Skip tensors in R2 — they're the compressed survivors
                if rid == RegionID.DURABLE and block.kind == ContentKind.TENSOR:
                    continue
                candidate = self.score_candidate(
                    block, projection.turn,
                    dependents=dep_counts.get(block.handle, 0),
                )
                candidates.append(candidate)

        # Sort by score descending (most evictable first)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:limit]

    def decide(
        self,
        projection: Projection,
        *,
        is_idle: bool = False,
    ) -> list[EvictionDecision]:
        """Make eviction decisions based on current pressure.

        Returns a list of decisions the orchestrator should execute.
        The scheduler never mutates the projection directly.
        """
        pressure = self.read_pressure(projection)
        zone = pressure.zone
        decisions: list[EvictionDecision] = []

        if zone == PressureZone.LOW:
            # No eviction. But if idle, we can still clean up waste.
            if is_idle and pressure.waste_tokens > 0:
                decisions.append(
                    EvictionDecision(
                        action=EvictionAction.RESTRUCTURE,
                        reason="idle cleanup of waste",
                    )
                )
            return decisions

        candidates = self.select_candidates(projection)
        if not candidates:
            return decisions

        if zone == PressureZone.MODERATE:
            if is_idle:
                # At idle boundaries, execute scheduled evictions
                for c in candidates[:3]:
                    decisions.append(
                        EvictionDecision(
                            action=EvictionAction.REQUEST_TENSOR,
                            handle=c.block.handle,
                            reason=f"moderate pressure, idle boundary; "
                                   f"score={c.score:.1f}",
                        )
                    )
            # Not idle? Just note the candidates, don't act yet.
            return decisions

        if zone == PressureZone.ELEVATED:
            # Request model cooperation for top candidates
            for c in candidates[:5]:
                decisions.append(
                    EvictionDecision(
                        action=EvictionAction.REQUEST_TENSOR,
                        handle=c.block.handle,
                        reason=f"elevated pressure; score={c.score:.1f}",
                    )
                )
            return decisions

        # CRITICAL — aggressive
        for c in candidates[:8]:
            if c.block.status == ContentStatus.PENDING_REMOVAL:
                # Already pending and has a tensor? Force evict.
                if c.block.tensor_handle is not None:
                    decisions.append(
                        EvictionDecision(
                            action=EvictionAction.EVICT,
                            handle=c.block.handle,
                            reason=f"critical pressure, tensor available; "
                                   f"score={c.score:.1f}",
                        )
                    )
                else:
                    # No tensor yet — mark pending and demand one
                    decisions.append(
                        EvictionDecision(
                            action=EvictionAction.DEMAND_TENSOR,
                            handle=c.block.handle,
                            reason=f"critical pressure, no tensor yet; "
                                   f"score={c.score:.1f}",
                        )
                    )
            else:
                decisions.append(
                    EvictionDecision(
                        action=EvictionAction.REQUEST_TENSOR,
                        handle=c.block.handle,
                        reason=f"critical pressure; score={c.score:.1f}",
                    )
                )

        return decisions


class EvictionAction(Enum):
    """Actions the scheduler can recommend."""
    REQUEST_TENSOR = auto()   # ask the model to produce a tensor
    DEMAND_TENSOR = auto()    # urgently ask — critical pressure
    EVICT = auto()            # execute eviction (tensor already exists)
    RESTRUCTURE = auto()      # restructure at idle boundary


@dataclass
class EvictionDecision:
    """A decision from the scheduler for the orchestrator to execute."""
    action: EvictionAction
    handle: str | None = None  # content block handle (None for RESTRUCTURE)
    reason: str = ""
