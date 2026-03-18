"""Orchestrator — the event loop that sequences projection mutations.

The orchestrator receives events, classifies them into regions,
manages turn boundaries, triggers pressure checks, and coordinates
eviction. It does NOT talk to the API or client directly — adapters
do that. The orchestrator speaks only in projection mutations and
eviction decisions.

Event flow:
  1. Receive inbound event (from client adapter)
  2. Advance turn: age R4 → R3
  3. Classify and place new content
  4. Pressure check → eviction decisions
  5. Generate API payload from projection (via API adapter)
  6. Receive outbound event (API response)
  7. Ingest response into R3
  8. Process cooperative memory signals
  9. Execute queued removals
  10. Pressure check again
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from tinkuy.events import Event, EventBus, EventKind
from tinkuy.pressure import (
    EvictionAction,
    EvictionDecision,
    PressureScheduler,
    PressureZone,
)
from tinkuy.regions import (
    ContentBlock,
    ContentKind,
    ContentStatus,
    Projection,
    RegionID,
)
from tinkuy.store import CheckpointStore, PageStore


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
    and (eventually) the adapters.
    """

    def __init__(
        self,
        projection: Projection | None = None,
        context_limit: int = 200_000,
        event_bus: EventBus | None = None,
        page_store: PageStore | None = None,
        checkpoint_store: CheckpointStore | None = None,
    ) -> None:
        self.projection = projection or Projection()
        self.scheduler = PressureScheduler(context_limit=context_limit)
        self.history: list[TurnRecord] = []
        self.bus = event_bus or EventBus()
        self._page_store = page_store
        self._checkpoint_store = checkpoint_store
        self._idle = False

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

        At idle, restructuring is free. Execute pending removals,
        run pressure check with is_idle=True, then checkpoint.
        """
        self._idle = True
        self._emit(EventKind.IDLE_ENTERED)
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
    ) -> TurnRecord:
        """Ingest an API response into the projection.

        The response goes into R3 (ephemeral). Cooperative memory
        signals are processed. Queued removals are executed.
        """
        record = self.history[-1] if self.history else TurnRecord(
            turn=self.turn
        )

        # Store assistant response in R3
        block = self.projection.add_content(
            content=content,
            kind=ContentKind.CONVERSATION,
            label=label,
            region=RegionID.EPHEMERAL,
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

    # --- Internal machinery ---

    def _checkpoint(self) -> None:
        """Write a projection checkpoint if a store is configured."""
        if self._checkpoint_store is not None:
            self._checkpoint_store.save(self.projection.snapshot())

    def _persist_page(self, handle: str, content: str) -> None:
        """Eagerly persist a verbatim original to the page store."""
        if self._page_store is not None:
            self._page_store.put(handle, content)

    def _recall_page(self, handle: str) -> str | None:
        """Retrieve a verbatim original from the page store."""
        if self._page_store is not None:
            return self._page_store.get(handle)
        return None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_store: CheckpointStore,
        page_store: PageStore | None = None,
        context_limit: int = 200_000,
        event_bus: EventBus | None = None,
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
                return

    def _handle_recall(self, signal: ResponseSignal) -> None:
        """Model wants to recall evicted content.

        Falls back to the persistent page store if the projection's
        in-memory store doesn't have the content.
        """
        result = self.projection.recall(signal.handle)
        if result is None and self._page_store is not None:
            # Try the persistent store
            content = self._page_store.get(signal.handle)
            if content is not None:
                # Inject into the projection's page store, then recall
                self.projection.page_store[signal.handle] = content
                result = self.projection.recall(signal.handle)
        if result is not None:
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
                    )
                    break

    def _execute_pending_removals(self) -> int:
        """Execute all pending removals that have tensor replacements.

        Returns the number of evictions executed.
        """
        count = 0
        for region in self.projection.regions.values():
            for block in list(region.blocks):
                if (
                    block.status == ContentStatus.PENDING_REMOVAL
                    and block.tensor_handle is not None
                ):
                    # The tensor should already be in R2 (placed by evict)
                    # Just update the block status
                    block.status = ContentStatus.AVAILABLE
                    block.content = ""  # Free the content, page store has it
                    self._emit(
                        EventKind.EVICTION_EXECUTED,
                        handle=block.handle,
                        tensor_handle=block.tensor_handle,
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
        """
        marked: list[str] = []
        for decision in decisions:
            if decision.handle is None:
                continue

            match decision.action:
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
                            break

                case EvictionAction.EVICT:
                    # Direct eviction — block already has a tensor
                    for region in self.projection.regions.values():
                        block = region.find(decision.handle)
                        if block and block.tensor_handle is not None:
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
                entries.append({
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
                })
        return entries
