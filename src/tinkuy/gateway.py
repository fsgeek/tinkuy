"""Gateway — the integration layer that sits in the wire.

The gateway is the entry point. It receives client requests,
feeds them through the orchestrator, synthesizes API payloads,
and returns responses. It owns the lifecycle of all components.

This is NOT an HTTP server or proxy. It's a Python object that
a harness (CLI, HTTP server, test fixture) drives. The harness
decides how to receive requests and deliver responses. The
gateway decides what happens between those two points.

Three modes of operation:
  1. Live — process a single turn (request → response)
  2. Rehydrate — replay a conversation log, then go live
  3. Resume — restore from checkpoint, then go live
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from tinkuy.adapter import IngestAdapter, LiveAdapter
from tinkuy.events import ConsoleStatusConsumer, EventBus, EventLog
from tinkuy.orchestrator import (
    EventType,
    InboundEvent,
    Orchestrator,
    ResponseSignal,
    ResponseSignalType,
    TurnRecord,
)
from tinkuy.pressure import PressureZone
from tinkuy.regions import ContentKind, Projection, RegionID
from tinkuy.store import (
    CheckpointStore,
    FileCheckpointStore,
    FilePageStore,
    MemoryCheckpointStore,
    MemoryPageStore,
    PageStore,
)


@dataclass
class GatewayConfig:
    """Configuration for a gateway instance."""
    context_limit: int = 200_000
    data_dir: str | None = None        # filesystem persistence root
    enable_console: bool = False        # console status consumer
    enable_event_log: bool = True       # in-memory event log


@dataclass
class TurnResult:
    """What the gateway returns to the harness after a turn."""
    api_payload: dict[str, Any]        # synthesized Anthropic API payload
    record: TurnRecord                 # observability record
    pressure_zone: PressureZone        # current pressure after this turn
    pending_evictions: list[str] = field(default_factory=list)  # handles


class Gateway:
    """The integration layer.

    A harness creates a Gateway, optionally rehydrates or resumes it,
    then calls process_turn() for each request/response cycle.
    """

    def __init__(self, config: GatewayConfig | None = None) -> None:
        self.config = config or GatewayConfig()
        self._setup_stores()
        self._setup_bus()
        self._setup_orchestrator()
        self._ingest = IngestAdapter(self.orchestrator)
        self._live = LiveAdapter(self.orchestrator)

    def _setup_stores(self) -> None:
        """Initialize page store and checkpoint store."""
        if self.config.data_dir:
            root = Path(self.config.data_dir)
            self.page_store: PageStore = FilePageStore(root / "pages")
            self.checkpoint_store: CheckpointStore = FileCheckpointStore(
                root / "checkpoint.json"
            )
        else:
            self.page_store = MemoryPageStore()
            self.checkpoint_store = MemoryCheckpointStore()

    def _setup_bus(self) -> None:
        """Initialize event bus and consumers."""
        self.bus = EventBus()
        self.event_log: EventLog | None = None

        if self.config.enable_event_log:
            self.event_log = EventLog()
            self.bus.subscribe(self.event_log)

        if self.config.enable_console:
            self._console = ConsoleStatusConsumer(
                context_limit=self.config.context_limit,
            )
            self.bus.subscribe(self._console)

    def _setup_orchestrator(self) -> None:
        """Initialize the orchestrator."""
        self.orchestrator = Orchestrator(
            context_limit=self.config.context_limit,
            event_bus=self.bus,
            page_store=self.page_store,
            checkpoint_store=self.checkpoint_store,
        )

    # --- Lifecycle ---

    @classmethod
    def resume(cls, config: GatewayConfig) -> Gateway | None:
        """Resume from a checkpoint.

        Returns None if no checkpoint exists.
        """
        gw = cls(config)
        restored = Orchestrator.from_checkpoint(
            checkpoint_store=gw.checkpoint_store,
            page_store=gw.page_store,
            context_limit=config.context_limit,
            event_bus=gw.bus,
        )
        if restored is None:
            return None
        gw.orchestrator = restored
        gw._ingest = IngestAdapter(gw.orchestrator)
        gw._live = LiveAdapter(gw.orchestrator)
        return gw

    def rehydrate(self, source: str | Path | dict[str, Any]) -> None:
        """Replay a conversation log into the projection.

        Accepts:
          - Path to a .json or .jsonl file
          - dict in Anthropic messages format
        """
        if isinstance(source, (str, Path)):
            self._ingest.ingest_file(source)
        elif isinstance(source, dict):
            self._ingest.ingest_anthropic(source)
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

    # --- Turn processing ---

    def process_turn(
        self,
        user_content: str,
        tool_results: list[dict[str, str]] | None = None,
    ) -> TurnResult:
        """Process a single turn: user message → API payload.

        This is the main entry point for live operation. The harness
        calls this with user content (and optional tool results),
        and gets back a synthesized API payload ready for the
        Anthropic API.

        The harness is responsible for:
          1. Calling the API with the payload
          2. Passing the response back via ingest_response()
        """
        events: list[InboundEvent] = []

        # User message
        events.append(InboundEvent(
            type=EventType.USER_MESSAGE,
            content=user_content,
            label="user",
        ))

        # Tool results
        if tool_results:
            for tr in tool_results:
                events.append(InboundEvent(
                    type=EventType.TOOL_RESULT,
                    content=tr.get("content", ""),
                    label=tr.get("name", "tool_result"),
                    metadata=tr.get("metadata", {}),
                ))

        # Begin turn
        record = self.orchestrator.begin_turn(events)

        # Apply eviction decisions
        pending = self.orchestrator.apply_decisions(
            record.eviction_decisions
        )

        # Synthesize API payload
        payload = self._live.synthesize_messages()

        # Inject page table into system if there are evicted blocks
        page_table = self._live.synthesize_page_table()
        if page_table:
            self._inject_page_table(payload, page_table)

        pressure = self.orchestrator.scheduler.read_pressure(
            self.orchestrator.projection
        )

        return TurnResult(
            api_payload=payload,
            record=record,
            pressure_zone=pressure.zone,
            pending_evictions=pending,
        )

    def ingest_response(
        self,
        content: str,
        signals: list[dict[str, Any]] | None = None,
    ) -> TurnRecord:
        """Ingest an API response after the harness calls the API.

        The harness should extract cooperative memory signals from
        the response and pass them here.
        """
        parsed_signals: list[ResponseSignal] | None = None
        if signals:
            parsed_signals = [
                self._parse_signal(s) for s in signals
            ]

        record = self.orchestrator.ingest_response(
            content=content,
            label="assistant",
            signals=parsed_signals,
        )

        # Mark idle after response ingestion — this is a cache boundary
        self.orchestrator.mark_idle()

        return record

    # --- Convenience ---

    @property
    def turn(self) -> int:
        return self.orchestrator.turn

    @property
    def pressure_zone(self) -> PressureZone:
        return self.orchestrator.scheduler.read_pressure(
            self.orchestrator.projection
        ).zone

    @property
    def projection(self) -> Projection:
        return self.orchestrator.projection

    def page_table(self) -> list[dict[str, Any]]:
        return self.orchestrator.page_table()

    # --- Internal ---

    def _inject_page_table(
        self, payload: dict[str, Any], page_table: str
    ) -> None:
        """Inject the page table into the system prompt."""
        system = payload.get("system", [])
        if isinstance(system, list):
            system.append({
                "type": "text",
                "text": page_table,
                "cache_control": {"type": "ephemeral"},
            })
            payload["system"] = system
        elif isinstance(system, str):
            payload["system"] = system + "\n\n" + page_table

    def _parse_signal(self, data: dict[str, Any]) -> ResponseSignal:
        """Parse a cooperative memory signal from raw dict."""
        signal_type = data.get("type", "").lower()
        handle = data.get("handle", "")

        if signal_type == "release":
            return ResponseSignal(
                type=ResponseSignalType.RELEASE,
                handle=handle,
                tensor_content=data.get("tensor_content"),
                declared_losses=data.get("declared_losses"),
            )
        elif signal_type == "retain":
            return ResponseSignal(
                type=ResponseSignalType.RETAIN,
                handle=handle,
            )
        elif signal_type == "recall":
            return ResponseSignal(
                type=ResponseSignalType.RECALL,
                handle=handle,
            )
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")
