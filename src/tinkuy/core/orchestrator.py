"""Orchestrator — the event loop that sequences projection mutations.

The orchestrator receives events, classifies them into regions,
manages turn boundaries, triggers pressure checks, and coordinates
eviction. It does NOT talk to the API or client directly — adapters
do that. The orchestrator speaks only in projection mutations and
eviction decisions.

When a Projector sidecar is configured, the orchestrator dispatches
eviction candidates to it rather than waiting for the primary model
to cooperate. The Projector (from Hamutay) is the cognitive processor
unit — it produces structured tensors with declared losses, epistemic
state, and thematic strands. Without it, eviction relies on whatever
the model volunteers inline via cooperative signals.

Event flow:
  1. Receive inbound event (from client adapter)
  2. Advance turn: age R4 → R3
  3. Classify and place new content
  4. Pressure check → eviction decisions
  5. If projector available: dispatch eviction candidates to sidecar
  6. Generate API payload from projection (via API adapter)
  7. Receive outbound event (API response)
  8. Ingest response into R3
  9. Process cooperative memory signals
  10. Execute queued removals
  11. Pressure check again
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from tinkuy.core.events import Event, EventBus, EventKind

log = logging.getLogger("tinkuy.orchestrator")
from tinkuy.core.pressure import (
    EvictionAction,
    EvictionDecision,
    PressureScheduler,
    PressureZone,
)
from tinkuy.core.regions import (
    ContentBlock,
    ContentKind,
    ContentStatus,
    Projection,
    RegionID,
)
from tinkuy.core.store import CheckpointStore, PageStore


class EventType(Enum):
    """Inbound event types from the client adapter."""
    USER_MESSAGE = auto()     # user text → R4
    TOOL_RESULT = auto()      # tool output → R3 or R4
    SYSTEM_UPDATE = auto()    # system context change → R1 (rare)
    TOOL_DEFINITION = auto()  # tool schema → R0


class ResponseSignalType(Enum):
    """Cooperative memory signals from the model."""
    RELEASE = auto()    # model wants to release content (with tensor)
    RETAIN = auto()     # model wants to keep content (cancel pending removal)
    RECALL = auto()     # model wants to recall evicted content
    DECLARE = auto()    # model declares dependency edges (immutable)
    TRACE = auto()      # model wants provenance chain for a handle


@dataclass
class InboundEvent:
    """An event from the client adapter."""
    type: EventType
    content: str
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseSignal:
    """A cooperative memory signal extracted from a model response."""
    type: ResponseSignalType
    handle: str
    tensor_content: str | None = None    # for RELEASE
    declared_losses: str | None = None   # for RELEASE
    depends_on: list[str] | None = None  # for DECLARE — parent handles


@dataclass
class TurnRecord:
    """Record of what happened during a turn, for observability."""
    turn: int
    inbound_handles: list[str] = field(default_factory=list)
    response_handle: str | None = None
    eviction_decisions: list[EvictionDecision] = field(default_factory=list)
    signals_processed: list[ResponseSignal] = field(default_factory=list)
    evictions_executed: int = 0
    pressure_zone_before: PressureZone | None = None
    pressure_zone_after: PressureZone | None = None
    timestamp: float = field(default_factory=time.time)


class Orchestrator:
    """Sequences projection mutations through the event loop.

    The orchestrator is the only entity that mutates the projection.
    It coordinates between the pressure scheduler, region lifecycle,
    the Projector sidecar (if available), and the adapters.
    """

    def __init__(
        self,
        projection: Projection | None = None,
        context_limit: int = 200_000,
        event_bus: EventBus | None = None,
        page_store: PageStore | None = None,
        checkpoint_store: CheckpointStore | None = None,
        tensor_store: Any | None = None,
        projector: Any | None = None,
    ) -> None:
        self.projection = projection or Projection()
        self.scheduler = PressureScheduler(context_limit=context_limit)
        self.history: list[TurnRecord] = []
        self.bus = event_bus or EventBus()
        self._page_store = page_store
        self._checkpoint_store = checkpoint_store
        self._tensor_store = tensor_store
        self._idle = False

        # Signal feedback: outcomes of cooperative signals from the last turn.
        # Cleared at the start of each turn, populated during signal processing.
        # The page table synthesizer reads this to tell the model what happened.
        self.signal_outcomes: list[dict[str, Any]] = []

        # Hamutay Projector sidecar — the cognitive processor unit.
        # When present, eviction candidates are dispatched to the
        # projector rather than waiting for inline model cooperation.
        self._projector = projector
        self._projection_executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._pending_projections: dict[str, concurrent.futures.Future] = {}
        if projector is not None:
            self._projection_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="tinkuy-projector"
            )

    def _emit(self, kind: EventKind, **data: Any) -> None:
        """Emit an event through the bus."""
        self.bus.emit(Event(kind=kind, turn=self.turn, data=data))

    @property
    def turn(self) -> int:
        return self.projection.turn

    @property
    def is_idle(self) -> bool:
        return self._idle

    def mark_idle(self) -> list[EvictionDecision]:
        """Mark the system as idle (cache boundary).

        At idle, restructuring is free. Drain any in-flight projections,
        execute pending removals, run pressure check with is_idle=True,
        then checkpoint.
        """
        self._idle = True
        self._emit(EventKind.IDLE_ENTERED)
        # Drain projector sidecar — block until all in-flight complete
        self._drain_projections_blocking()
        # Execute all pending removals that have tensors
        executed = self._execute_pending_removals()
        # Pressure check at idle
        decisions = self.scheduler.decide(self.projection, is_idle=True)
        self._emit_pressure_read()
        # Checkpoint at idle boundary — projection is consistent
        self._checkpoint()
        return decisions

    def mark_active(self) -> None:
        """Mark the system as active (no longer idle)."""
        self._idle = False
        self._emit(EventKind.IDLE_EXITED)

    # --- Inbound event handling ---

    def begin_turn(self, events: list[InboundEvent]) -> TurnRecord:
        """Begin a new turn: advance, age R4→R3, place new content.

        Returns a TurnRecord for observability.
        """
        # Clear signal outcomes from prior turn
        self.signal_outcomes = []

        # Collect any completed projections from the sidecar
        self._drain_projections()

        record = TurnRecord(turn=self.projection.turn)
        record.pressure_zone_before = self.scheduler.read_pressure(
            self.projection
        ).zone

        # Age: move current R4 content into R3
        self._age_current_to_ephemeral()

        # Advance turn counter
        self.projection.advance_turn()
        record.turn = self.projection.turn
        self._idle = False

        self._emit(EventKind.TURN_BEGAN, turn=self.turn)

        # Place new content
        for event in events:
            block = self._place_event(event)
            record.inbound_handles.append(block.handle)
            self._emit(
                EventKind.BLOCK_CREATED,
                handle=block.handle,
                content_kind=block.kind.name,
                region=block.region.name,
                label=block.label,
                size_tokens=block.size_tokens,
            )

        # Pressure check
        self._emit_pressure_read()
        decisions = self.scheduler.decide(
            self.projection, is_idle=self._idle
        )
        for d in decisions:
            self._emit(
                EventKind.EVICTION_DECIDED,
                handle=d.handle,
                action=d.action.name,
                reason=d.reason,
            )
        record.eviction_decisions = decisions

        record.pressure_zone_after = self.scheduler.read_pressure(
            self.projection
        ).zone

        # Checkpoint at turn boundary
        self._checkpoint()

        self.history.append(record)
        return record

    def ingest_response(
        self,
        content: str,
        label: str = "assistant response",
        signals: list[ResponseSignal] | None = None,
        content_blocks: list[dict] | None = None,
    ) -> TurnRecord:
        """Ingest an API response into the projection.

        The response goes into R4 (current) alongside the user content
        that prompted it.  Both are from the same turn.  When the next
        turn begins, R4 ages to R3 as a unit — preserving chronological
        order (user before assistant).

        Cooperative memory signals are processed.  Queued removals
        are executed.
        """
        record = self.history[-1] if self.history else TurnRecord(
            turn=self.turn
        )

        # Store assistant response in R4 — same turn as the user content
        block = self.projection.add_content(
            content=content,
            kind=ContentKind.CONVERSATION,
            label=label,
            region=RegionID.CURRENT,
            content_blocks=content_blocks,
        )
        record.response_handle = block.handle
        self._emit(
            EventKind.TURN_RESPONSE_INGESTED,
            handle=block.handle,
            size_tokens=block.size_tokens,
        )

        # Process cooperative memory signals
        if signals:
            for signal in signals:
                self._process_signal(signal)
                record.signals_processed.append(signal)

        # Execute queued removals
        record.evictions_executed = self._execute_pending_removals()

        # Post-response pressure check
        self._emit_pressure_read()
        record.pressure_zone_after = self.scheduler.read_pressure(
            self.projection
        ).zone

        return record

    # --- Projector sidecar ---

    def _dispatch_to_projector(self, handle: str, content: str) -> None:
        """Dispatch content to the Hamutay Projector sidecar.

        Runs asynchronously in a thread pool. The resulting tensor is
        ingested into R2 when the projection completes (either at the
        next turn boundary or when explicitly drained).
        """
        if self._projector is None or self._projection_executor is None:
            return

        def _project() -> tuple[str, Any]:
            """Run projection in background thread. Returns (handle, tensor)."""
            tensor = self._projector.project(content)
            return handle, tensor

        future = self._projection_executor.submit(_project)
        self._pending_projections[handle] = future
        self._emit(
            EventKind.PROJECTION_DISPATCHED,
            handle=handle,
            label=f"projector:{handle[:8]}",
            size_tokens=len(content) // 4,
        )
        log.info("dispatched to projector: %s (%d chars)", handle[:8], len(content))

    def _drain_projections(self) -> int:
        """Collect completed projections and ingest tensors into R2.

        Non-blocking: only collects futures that are already done.
        Returns the number of tensors ingested.
        """
        if not self._pending_projections:
            return 0

        ingested = 0
        completed: list[str] = []

        for handle, future in self._pending_projections.items():
            if not future.done():
                continue
            completed.append(handle)

            try:
                _, tensor = future.result()
            except Exception as e:
                log.error("projector failed for %s: %s", handle[:8], e)
                self._emit(
                    EventKind.EVICTION_DECIDED,
                    handle=handle,
                    action="PROJECTION_FAILED",
                    reason=str(e),
                )
                continue

            # Serialize the tensor
            tensor_data = tensor.model_dump()
            tensor_text = tensor.model_dump_json(indent=2)
            declared_losses = "; ".join(
                f"{loss.what_was_lost} ({loss.category.value})"
                for loss in tensor.declared_losses
            )

            # Persist the verbatim original before evicting
            for region in self.projection.regions.values():
                block = region.find(handle)
                if block and block.content:
                    self._persist_page(handle, block.content)
                    break

            # Write tensor to the tensor store (immutable create)
            if self._tensor_store is not None:
                self._tensor_store.create(handle, tensor_data)

            # Create the tensor block in R2. When a tensor store is
            # configured, R2 holds a lightweight reference — the full
            # tensor lives in the store. Without a store, R2 holds the
            # serialized tensor directly (local convenience mode).
            if self._tensor_store is not None:
                # Reference mode: R2 block is a stub with metadata
                tensor_block = ContentBlock.create(
                    content=f"[tensor:{handle[:8]} — {len(tensor.strands)} strands, "
                            f"{len(tensor.declared_losses)} losses]",
                    kind=ContentKind.TENSOR,
                    label=f"tensor:{handle[:8]}",
                    region=RegionID.DURABLE,
                    turn=self.turn,
                )
            else:
                # Inline mode: R2 block holds the full serialized tensor
                tensor_block = ContentBlock.create(
                    content=tensor_text,
                    kind=ContentKind.TENSOR,
                    label=f"tensor:{handle[:8]}",
                    region=RegionID.DURABLE,
                    turn=self.turn,
                )
            tensor_block.metadata["declared_losses"] = declared_losses
            tensor_block.metadata["tensor_cycle"] = tensor.cycle
            tensor_block.metadata["n_strands"] = len(tensor.strands)
            tensor_block.metadata["epistemic"] = {
                "truth": tensor.epistemic.truth,
                "indeterminacy": tensor.epistemic.indeterminacy,
                "falsity": tensor.epistemic.falsity,
            }
            tensor_block.metadata["source_handle"] = handle

            # Execute the eviction
            self.projection.evict(handle, tensor_block)
            self._emit(
                EventKind.BLOCK_EVICTED,
                handle=handle,
                tensor_handle=tensor_block.handle,
                declared_losses=declared_losses,
                source="projector",
            )
            ingested += 1
            log.info(
                "projector tensor ingested: %s → %s (%d strands, %d losses, store=%s)",
                handle[:8],
                tensor_block.handle[:8],
                len(tensor.strands),
                len(tensor.declared_losses),
                "yes" if self._tensor_store is not None else "inline",
            )

        for handle in completed:
            del self._pending_projections[handle]

        return ingested

    def _drain_projections_blocking(self, timeout: float = 30.0) -> int:
        """Block until all pending projections complete.

        Used at idle boundaries and shutdown. Returns number ingested.
        """
        if not self._pending_projections:
            return 0

        futures = list(self._pending_projections.values())
        concurrent.futures.wait(futures, timeout=timeout)
        return self._drain_projections()

    # --- Internal machinery ---

    def _checkpoint(self) -> None:
        """Write a projection checkpoint if a store is configured."""
        if self._checkpoint_store is not None:
            self._checkpoint_store.save(self.projection.snapshot())
            log.info(
                "checkpoint written | turn=%d tokens=%d",
                self.projection.turn,
                self.projection.total_tokens,
            )

    def _persist_page(self, handle: str, content: str) -> None:
        """Eagerly persist a verbatim original to the page store."""
        if self._page_store is not None:
            self._page_store.put(handle, content)

    def _recall_page(self, handle: str) -> str | None:
        """Retrieve a verbatim original from the page store."""
        if self._page_store is not None:
            return self._page_store.get(handle)
        return None

    def _evict_with_stored_tensor(
        self,
        block: ContentBlock,
        handle: str,
        tensor_data: dict[str, Any],
    ) -> None:
        """Evict a block using a tensor already in the store.

        This is the fast path: the tensor was produced in a prior session
        (or earlier in this one) and persisted. We skip the projector
        entirely and go straight to eviction.
        """
        # Persist the verbatim original before evicting
        if block.content:
            self._persist_page(handle, block.content)

        # Build the R2 stub — same shape as _drain_projections produces
        n_strands = len(tensor_data.get("strands", []))
        n_losses = len(tensor_data.get("declared_losses", []))
        tensor_block = ContentBlock.create(
            content=f"[tensor:{handle[:8]} — {n_strands} strands, "
                    f"{n_losses} losses]",
            kind=ContentKind.TENSOR,
            label=f"tensor:{handle[:8]}",
            region=RegionID.DURABLE,
            turn=self.turn,
        )
        tensor_block.metadata["source_handle"] = handle
        tensor_block.metadata["n_strands"] = n_strands
        tensor_block.metadata["source"] = "store"

        self.projection.evict(handle, tensor_block)
        self._emit(
            EventKind.BLOCK_EVICTED,
            handle=handle,
            tensor_handle=tensor_block.handle,
            source="store",
        )
        log.info(
            "evicted %s using stored tensor (%d strands, %d losses)",
            handle[:8],
            n_strands,
            n_losses,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_store: CheckpointStore,
        page_store: PageStore | None = None,
        tensor_store: Any | None = None,
        context_limit: int = 200_000,
        event_bus: EventBus | None = None,
        projector: Any | None = None,
    ) -> Orchestrator | None:
        """Restore an orchestrator from a checkpoint.

        Returns None if no checkpoint exists.
        """
        data = checkpoint_store.load()
        if data is None:
            return None
        projection = Projection.from_snapshot(data)
        return cls(
            projection=projection,
            context_limit=context_limit,
            event_bus=event_bus,
            page_store=page_store,
            checkpoint_store=checkpoint_store,
            tensor_store=tensor_store,
            projector=projector,
        )

    def _emit_pressure_read(self) -> None:
        """Emit a pressure read event with current state."""
        pressure = self.scheduler.read_pressure(self.projection)
        self._emit(
            EventKind.PRESSURE_READ,
            total_tokens=pressure.total_tokens,
            context_limit=pressure.context_limit,
            usage=pressure.usage,
            zone=pressure.zone.name,
            headroom=pressure.headroom_tokens,
        )

    def _age_current_to_ephemeral(self) -> None:
        """Move all R4 (current) content into R3 (ephemeral)."""
        r4 = self.projection.region(RegionID.CURRENT)
        r3 = self.projection.region(RegionID.EPHEMERAL)

        # Move blocks, preserving order
        while r4.blocks:
            block = r4.blocks.pop(0)
            block.region = RegionID.EPHEMERAL
            r3.blocks.append(block)
            self._emit(
                EventKind.BLOCK_AGED,
                handle=block.handle,
                from_region="CURRENT",
                to_region="EPHEMERAL",
            )

        # Promote stable R3 content to R2 (durable).
        # A block that has survived in R3 for PROMOTION_AGE turns without
        # mutation is likely reference material (file reads, large tool
        # outputs) that benefits from the R2 cache breakpoint.
        self._promote_stable_to_durable()

    # Minimum age (in turns) before an R3 block is promoted to R2.
    PROMOTION_AGE: int = 3
    # Minimum size (tokens) — don't promote tiny blocks, the cache
    # benefit is negligible and they clutter R2.
    PROMOTION_MIN_TOKENS: int = 200

    def _promote_stable_to_durable(self) -> None:
        """Promote stable R3 blocks to R2.

        Candidates: blocks that have been in R3 for at least PROMOTION_AGE
        turns, are large enough to matter for cache, and are PRESENT
        (not already evicted). Conversation blocks are excluded — they're
        ephemeral by nature and will be stripped by Taste later.
        """
        r3 = self.projection.region(RegionID.EPHEMERAL)
        r2 = self.projection.region(RegionID.DURABLE)
        current_turn = self.projection.turn

        promoted: list[ContentBlock] = []
        remaining: list[ContentBlock] = []

        for block in r3.blocks:
            age = current_turn - block.access.created_turn
            if (
                block.status == ContentStatus.PRESENT
                and block.kind in (ContentKind.TOOL_RESULT, ContentKind.FILE)
                and block.size_tokens >= self.PROMOTION_MIN_TOKENS
                and age >= self.PROMOTION_AGE
            ):
                promoted.append(block)
            else:
                remaining.append(block)

        if not promoted:
            return

        r3.blocks = remaining
        for block in promoted:
            block.region = RegionID.DURABLE
            r2.blocks.append(block)
            self._emit(
                EventKind.BLOCK_AGED,
                handle=block.handle,
                from_region="EPHEMERAL",
                to_region="DURABLE",
            )
            log.info(
                "promoted %s to R2: %s (%d tokens, age %d turns)",
                block.handle[:8], block.label,
                block.size_tokens, current_turn - block.access.created_turn,
            )

    def _place_event(self, event: InboundEvent) -> ContentBlock:
        """Classify an event and place it in the right region."""
        region, kind = self._classify_event(event)
        return self.projection.add_content(
            content=event.content,
            kind=kind,
            label=event.label,
            region=region,
            **event.metadata,
        )

    def _classify_event(
        self, event: InboundEvent
    ) -> tuple[RegionID, ContentKind]:
        """Map event type to region and content kind."""
        match event.type:
            case EventType.USER_MESSAGE:
                return RegionID.CURRENT, ContentKind.CONVERSATION
            case EventType.TOOL_RESULT:
                return RegionID.EPHEMERAL, ContentKind.TOOL_RESULT
            case EventType.SYSTEM_UPDATE:
                return RegionID.SYSTEM, ContentKind.SYSTEM
            case EventType.TOOL_DEFINITION:
                return RegionID.TOOLS, ContentKind.SYSTEM

    def _process_signal(self, signal: ResponseSignal) -> None:
        """Process a cooperative memory signal from the model."""
        match signal.type:
            case ResponseSignalType.RELEASE:
                self._handle_release(signal)
            case ResponseSignalType.RETAIN:
                self._handle_retain(signal)
            case ResponseSignalType.RECALL:
                self._handle_recall(signal)
            case ResponseSignalType.DECLARE:
                self._handle_declare(signal)
            case ResponseSignalType.TRACE:
                self._handle_trace(signal)

    def _handle_release(self, signal: ResponseSignal) -> None:
        """Model wants to release content with a tensor replacement."""
        if signal.tensor_content is None:
            return  # Release without tensor is ignored

        # Persist the verbatim original BEFORE evicting.
        # If we crash after evict but before persist, the content is gone.
        for region in self.projection.regions.values():
            block = region.find(signal.handle)
            if block and block.content:
                self._persist_page(signal.handle, block.content)
                break

        # Create the tensor block
        tensor = ContentBlock.create(
            content=signal.tensor_content,
            kind=ContentKind.TENSOR,
            label=f"tensor:{signal.handle[:8]}",
            region=RegionID.DURABLE,
            turn=self.turn,
        )
        if signal.declared_losses:
            tensor.metadata["declared_losses"] = signal.declared_losses
        tensor.metadata["source"] = "cooperative"

        # Persist to tensor store — same contract as the projector path.
        # Cooperative tensors are plain text (not structured Tensor objects),
        # so we wrap them in a minimal dict that the store can serialize.
        if self._tensor_store is not None:
            self._tensor_store.create(signal.handle, {
                "content": signal.tensor_content,
                "declared_losses": signal.declared_losses,
                "source": "cooperative",
                "turn": self.turn,
            })

        # Execute the eviction
        self.projection.evict(signal.handle, tensor)
        self._emit(
            EventKind.SIGNAL_RELEASE,
            handle=signal.handle,
            tensor_handle=tensor.handle,
            declared_losses=signal.declared_losses,
        )
        self._emit(
            EventKind.BLOCK_EVICTED,
            handle=signal.handle,
            tensor_handle=tensor.handle,
        )
        self.signal_outcomes.append({
            "signal": "release",
            "handle": signal.handle,
            "outcome": "accepted",
            "tensor_persisted": self._tensor_store is not None,
            "tensor_handle": tensor.handle,
        })

    def _handle_retain(self, signal: ResponseSignal) -> None:
        """Model wants to cancel a pending removal."""
        for region in self.projection.regions.values():
            block = region.find(signal.handle)
            if block and block.status == ContentStatus.PENDING_REMOVAL:
                block.status = ContentStatus.PRESENT
                # Remove any nominations for this block
                region.nominations = [
                    n for n in region.nominations
                    if n.handle != signal.handle
                ]
                self._emit(
                    EventKind.SIGNAL_RETAIN,
                    handle=signal.handle,
                )
                self.signal_outcomes.append({
                    "signal": "retain",
                    "handle": signal.handle,
                    "outcome": "accepted",
                })
                return
        # Block wasn't pending — retain was a no-op
        self.signal_outcomes.append({
            "signal": "retain",
            "handle": signal.handle,
            "outcome": "no_effect",
            "reason": "block not pending removal",
        })

    def _handle_recall(self, signal: ResponseSignal) -> None:
        """Model wants to recall evicted content.

        Fallback chain:
          1. Projection in-memory page store (fastest)
          2. Persistent page store (verbatim original)
          3. Tensor store (compressed summary — degraded but present)

        A tensor recall is better than silence. The model gets its own
        compressed notes back, which is enough to reconstruct context
        even if the verbatim original is gone.
        """
        result = self.projection.recall(signal.handle)
        if result is None and self._page_store is not None:
            # Try the persistent store
            content = self._page_store.get(signal.handle)
            if content is not None:
                # Inject into the projection's page store, then recall
                self.projection.page_store[signal.handle] = content
                result = self.projection.recall(signal.handle)
        if result is None and self._tensor_store is not None:
            # Degraded recall: use the tensor (compressed summary)
            tensor_data = self._tensor_store.get(signal.handle)
            if tensor_data is not None:
                tensor_content = tensor_data.get("content", "")
                if tensor_content:
                    self.projection.page_store[signal.handle] = tensor_content
                    result = self.projection.recall(signal.handle)
                    log.info(
                        "degraded recall: %s restored from tensor (original lost)",
                        signal.handle[:8],
                    )
        if result is not None:
            # Determine recall source for feedback
            # Find the block to get fault count
            for region in self.projection.regions.values():
                block = region.find(signal.handle)
                if block:
                    self._emit(
                        EventKind.SIGNAL_RECALL,
                        handle=signal.handle,
                        fault_count=block.access.fault_count,
                    )
                    self._emit(
                        EventKind.BLOCK_RECALLED,
                        handle=signal.handle,
                        fault_count=block.access.fault_count,
                        evicted_at=block.access.evicted_at,
                    )
                    break
            self.signal_outcomes.append({
                "signal": "recall",
                "handle": signal.handle,
                "outcome": "restored",
            })
        else:
            self.signal_outcomes.append({
                "signal": "recall",
                "handle": signal.handle,
                "outcome": "failed",
                "reason": "content not found in any store",
            })

    def _handle_declare(self, signal: ResponseSignal) -> None:
        """Model declares dependency edges for a content block.

        Edges are immutable — once declared, they are never modified.
        Like Paxos consensus: a decision once made doesn't change.
        It can be superseded by a new decision, but the original
        edges remain as historical record.

        Edges are stored in metadata["depends_on"] as a list of
        handles. These survive checkpointing and eviction.
        """
        if not signal.depends_on:
            return

        for region in self.projection.regions.values():
            block = region.find(signal.handle)
            if block is None:
                continue
            # Immutable: only write if not already declared
            if "depends_on" not in block.metadata:
                block.metadata["depends_on"] = list(signal.depends_on)
                self._emit(
                    EventKind.SIGNAL_DECLARE,
                    handle=signal.handle,
                    depends_on=signal.depends_on,
                )
                log.info(
                    "declared edges: %s → %s",
                    signal.handle[:8],
                    [h[:8] for h in signal.depends_on],
                )
                self.signal_outcomes.append({
                    "signal": "declare",
                    "handle": signal.handle,
                    "outcome": "accepted",
                    "edges": len(signal.depends_on),
                })
            else:
                self.signal_outcomes.append({
                    "signal": "declare",
                    "handle": signal.handle,
                    "outcome": "no_effect",
                    "reason": "edges already declared",
                })
            return

    def _handle_trace(self, signal: ResponseSignal) -> None:
        """Model wants the provenance chain for a handle.

        Walks the dependency graph breadth-first, recalling each
        block in the chain. The model gets the full reasoning chain
        paged back into context, not just the immediate block.

        Depth is capped to prevent runaway traversal.
        """
        max_depth = 10
        visited: set[str] = set()
        frontier: list[str] = [signal.handle]
        recalled: list[str] = []

        while frontier and len(visited) < max_depth:
            handle = frontier.pop(0)
            if handle in visited:
                continue
            visited.add(handle)

            # Find the block anywhere in the projection
            block = None
            for region in self.projection.regions.values():
                block = region.find(handle)
                if block is not None:
                    break

            if block is None:
                continue

            # If evicted, recall it
            if block.status == ContentStatus.AVAILABLE:
                result = self._recall_page(handle)
                if result is not None:
                    recalled.append(handle)
                    self._emit(
                        EventKind.BLOCK_RECALLED,
                        handle=handle,
                        fault_count=block.access.fault_count,
                        evicted_at=block.access.evicted_at,
                    )

            # Follow edges
            deps = block.metadata.get("depends_on", [])
            for dep_handle in deps:
                if dep_handle not in visited:
                    frontier.append(dep_handle)

        if recalled:
            log.info(
                "trace from %s recalled %d blocks: %s",
                signal.handle[:8],
                len(recalled),
                [h[:8] for h in recalled],
            )
        self._emit(
            EventKind.SIGNAL_TRACE,
            handle=signal.handle,
            visited=[h[:8] for h in visited],
            recalled=[h[:8] for h in recalled],
        )

    def _execute_pending_removals(self) -> int:
        """Execute all pending removals.

        Blocks with tensor replacements are evicted normally.
        Blocks stuck pending without tensors for more than 3 turns
        are force-evicted — content is persisted to the page store
        and the block becomes faultable with a stub marker. Better
        to lose the tensor summary than deadlock the eviction pipeline.

        Returns the number of evictions executed.
        """
        count = 0
        pressure = self.scheduler.read_pressure(self.projection)
        for region in self.projection.regions.values():
            for block in list(region.blocks):
                if block.status != ContentStatus.PENDING_REMOVAL:
                    continue

                if block.tensor_handle is not None:
                    # Normal path — tensor available
                    block.status = ContentStatus.AVAILABLE
                    block.content = ""
                    self._emit(
                        EventKind.EVICTION_EXECUTED,
                        handle=block.handle,
                        tensor_handle=block.tensor_handle,
                    )
                    count += 1
                elif (
                    self.turn - block.access.last_access_turn >= 3
                    and pressure.zone in (
                        PressureZone.ELEVATED, PressureZone.CRITICAL,
                    )
                ):
                    # Force-evict: stuck pending with no tensor and
                    # pressure is high. Persist content, evict with stub.
                    if block.content:
                        self._persist_page(block.handle, block.content)
                    log.warning(
                        "force-evicting %s (%d tok) — pending %d turns, "
                        "no tensor, pressure=%s",
                        block.handle[:8],
                        block.size_tokens,
                        self.turn - block.access.last_access_turn,
                        pressure.zone.name,
                    )
                    block.status = ContentStatus.AVAILABLE
                    block.content = ""
                    self._emit(
                        EventKind.EVICTION_EXECUTED,
                        handle=block.handle,
                        tensor_handle=None,
                        force=True,
                    )
                    count += 1
        return count

    def apply_decisions(
        self, decisions: list[EvictionDecision]
    ) -> list[str]:
        """Apply eviction decisions from the scheduler.

        Returns handles of blocks that were marked pending_removal.
        This is separate from begin_turn so the caller can inspect
        decisions before applying them.

        When a Projector sidecar is available, REQUEST_TENSOR and
        DEMAND_TENSOR decisions dispatch content to the projector
        immediately. The projector produces a structured tensor
        asynchronously; the result is ingested at the next turn
        boundary or idle drain.

        Without a projector, blocks are marked pending_removal and
        the cooperative signal protocol remains the only path to
        tensor production.
        """
        marked: list[str] = []
        for decision in decisions:
            if decision.handle is None:
                continue

            # Check if the tensor store already has a tensor for this handle.
            # If so, upgrade REQUEST/DEMAND to EVICT — no projector call needed.
            action = decision.action
            if (
                action in (EvictionAction.REQUEST_TENSOR, EvictionAction.DEMAND_TENSOR)
                and self._tensor_store is not None
                and self._tensor_store.has(decision.handle)
            ):
                action = EvictionAction.EVICT
                log.info(
                    "tensor store hit for %s — upgrading %s to EVICT",
                    decision.handle[:8],
                    decision.action.name,
                )

            match action:
                case EvictionAction.REQUEST_TENSOR | EvictionAction.DEMAND_TENSOR:
                    # Mark block as pending removal
                    for region in self.projection.regions.values():
                        block = region.find(decision.handle)
                        if block and block.status == ContentStatus.PRESENT:
                            block.status = ContentStatus.PENDING_REMOVAL
                            region.nominate_removal(
                                decision.handle,
                                source="scheduler",
                                reason=decision.reason,
                            )
                            marked.append(decision.handle)
                            # Dispatch to projector sidecar if available
                            if self._projector is not None and block.content:
                                self._dispatch_to_projector(
                                    decision.handle, block.content
                                )
                            break

                case EvictionAction.EVICT:
                    # Direct eviction — tensor available (in block or store)
                    for region in self.projection.regions.values():
                        block = region.find(decision.handle)
                        if block is None:
                            continue
                        # If the block doesn't have a tensor_handle yet but
                        # the store does, ingest the stored tensor now.
                        if block.tensor_handle is None and self._tensor_store is not None:
                            tensor_data = self._tensor_store.get(decision.handle)
                            if tensor_data is not None:
                                self._evict_with_stored_tensor(
                                    block, decision.handle, tensor_data
                                )
                                break
                        if block.tensor_handle is not None:
                            block.status = ContentStatus.AVAILABLE
                            block.content = ""
                            break

        return marked

    # --- Page table generation ---

    def page_table(self) -> list[dict[str, Any]]:
        """Generate the page table for inclusion in the projection.

        The page table is the model's map of available memory. It
        lists all content in R2 and R3 with their status, size, and
        access metadata.
        """
        entries: list[dict[str, Any]] = []
        for rid in (RegionID.DURABLE, RegionID.EPHEMERAL):
            region = self.projection.region(rid)
            for block in region.blocks:
                entry = {
                    "handle": block.handle,
                    "kind": block.kind.name.lower(),
                    "label": block.label,
                    "status": block.status.name.lower(),
                    "region": rid.value,
                    "size_tokens": block.size_tokens,
                    "fault_count": block.access.fault_count,
                    "age_turns": max(
                        0, self.turn - block.access.last_access_turn
                    ),
                }
                if block.metadata.get("depends_on"):
                    entry["depends_on"] = block.metadata["depends_on"]
                entries.append(entry)
        return entries
