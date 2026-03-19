"""Gateway — the integration layer that sits in the wire.

The gateway is the entry point. It receives raw client request
bodies, feeds them through the orchestrator, synthesizes API
payloads, and returns complete upstream bodies. It owns the
full request→response transformation. DEATH TO PROXIES.

The harness (HTTP server, CLI, test fixture) is a dumb pipe.
It hands raw bytes to the gateway and forwards whatever comes
back. It never sees, touches, or transforms message content.

Three modes of operation:
  1. Live — process a single turn (request → response)
  2. Rehydrate — replay a conversation log, then go live
  3. Resume — restore from checkpoint, then go live
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("tinkuy.gateway")

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
    session_id: str | None = None      # session identity (from adapter)
    enable_console: bool = False        # console status consumer
    enable_event_log: bool = True       # in-memory event log


@dataclass
class TurnTelemetry:
    """What the API told us about a turn. The research record.

    Every field comes from the API response or wire-level observation.
    Nothing is estimated or inferred — this is ground truth.
    """
    # Identity
    message_id: str = ""
    model: str = ""
    turn: int = 0
    timestamp: float = field(default_factory=time.time)

    # Token counts (from API usage object)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_create_tokens: int = 0

    # Stop condition
    stop_reason: str | None = None  # end_turn, max_tokens, tool_use

    # Response structure
    text_blocks: int = 0
    thinking_blocks: int = 0
    tool_use_blocks: int = 0
    tool_names: list[str] = field(default_factory=list)

    # Wire cost
    request_bytes: int = 0

    # Timing (seconds)
    ttfb: float | None = None       # time to first byte
    duration: float | None = None   # total request duration

    @property
    def total_input_tokens(self) -> int:
        """Total logical input: processed + cache read + cache create."""
        return self.input_tokens + self.cache_read_tokens + self.cache_create_tokens

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of logical input served from cache."""
        total = self.total_input_tokens
        return self.cache_read_tokens / total if total > 0 else 0.0


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
        self.telemetry: list[TurnTelemetry] = []
        self._client_overhead_tokens: int | None = None

    def _setup_stores(self) -> None:
        """Initialize page store and checkpoint store.

        Page store is workspace-scoped (shared across sessions).
        Checkpoint store is session-scoped (keyed by session_id).
        """
        if self.config.data_dir:
            root = Path(self.config.data_dir)
            # Pages are shared — any session can recall any page
            self.page_store: PageStore = FilePageStore(root / "pages")
            # Checkpoints are per-session
            if self.config.session_id:
                ckpt_path = root / "sessions" / self.config.session_id / "checkpoint.json"
            else:
                ckpt_path = root / "checkpoint.json"
            self.checkpoint_store: CheckpointStore = FileCheckpointStore(ckpt_path)
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

    # --- Raw request/response boundary (DEATH TO PROXIES) ---

    def prepare_request(self, client_body: dict[str, Any]) -> dict[str, Any]:
        """Accept a raw client request body, return a complete upstream body.

        This is THE boundary. The client body enters, the gateway's
        projection-synthesized payload exits. The HTTP layer calls this
        and forwards the result. It never touches message content.
        """
        request_messages = client_body.get("messages", [])
        user_content, tool_results = _extract_user_content(request_messages)

        log.info(
            "← request | session=%s stream=%s model=%s "
            "client_messages=%d user_content_len=%d tool_results=%d",
            self.config.session_id or "default",
            client_body.get("stream", False),
            client_body.get("model", "?"),
            len(request_messages),
            len(user_content) if user_content else 0,
            len(tool_results) if tool_results else 0,
        )

        # Always process through the gateway — no exceptions, no fallback
        turn_result = self.process_turn(
            user_content=user_content or "",
            tool_results=tool_results,
        )

        # Build complete upstream body: client's non-message fields +
        # gateway's synthesized messages and system prompt
        upstream: dict[str, Any] = {
            k: v for k, v in client_body.items()
            if k not in ("messages", "system")
        }
        upstream["messages"] = turn_result.api_payload.get("messages", [])

        # Merge system prompts: preserve client's, append gateway's
        gateway_system = turn_result.api_payload.get("system")
        client_system = client_body.get("system")
        if gateway_system is not None and client_system is not None:
            upstream["system"] = _merge_system(client_system, gateway_system)
        elif gateway_system is not None:
            upstream["system"] = gateway_system
        elif client_system is not None:
            upstream["system"] = client_system

        log.info(
            "  synth | messages=%d system=%s",
            len(upstream.get("messages", [])),
            "yes" if "system" in upstream else "no",
        )

        return upstream

    def ingest_raw_response(self, text: str) -> TurnRecord | None:
        """Ingest raw assistant response text.

        Extracts cooperative memory signals, strips them, and feeds
        the clean text into the orchestrator. The HTTP layer collects
        text from the SSE stream and hands it here. It never parses
        message content.
        """
        if not text:
            return None
        from tinkuy.harness import extract_signals, strip_signals
        signals = extract_signals(text)
        clean = strip_signals(text)
        return self.ingest_response(
            content=clean,
            signals=signals if signals else None,
        )

    def ingest_response_json(self, response_data: dict[str, Any]) -> TurnRecord | None:
        """Ingest a complete Anthropic API response JSON."""
        text = _extract_response_text(response_data)
        return self.ingest_raw_response(text)

    # --- Telemetry ---

    def report_telemetry(self, telemetry: TurnTelemetry) -> None:
        """Report API telemetry from the last turn.

        This is the data pipeline — the server hands us ground-truth
        numbers from the API, we use them to calibrate pressure and
        build the research record.
        """
        telemetry.turn = self.turn
        self.telemetry.append(telemetry)

        # Calibrate pressure: the API knows the true context size.
        # Our projection only counts what we ingested. The difference
        # is the client overhead (system prompt, tools, etc.) which
        # is roughly constant for a session.
        #
        # When api_total < projection_tokens, the projection has stale
        # content from a prior session (resumed checkpoint). The API
        # only saw what we actually sent — so api_total is the truth.
        api_total = telemetry.total_input_tokens
        projection_tokens = self.orchestrator.projection.total_tokens
        if api_total > 0:
            if api_total > projection_tokens:
                overhead = api_total - projection_tokens
            else:
                # Stale projection — the API saw less than we think.
                # Client overhead is at least api_total minus what a
                # fresh projection would hold. Use api_total as floor.
                overhead = 0
                log.warning(
                    "  telemetry | projection (%d tok) exceeds API total "
                    "(%d tok) — stale checkpoint?",
                    projection_tokens, api_total,
                )

            if self._client_overhead_tokens is None:
                self._client_overhead_tokens = overhead
                log.info(
                    "  telemetry | client overhead calibrated: %d tokens "
                    "(api=%d, projection=%d)",
                    overhead, api_total, projection_tokens,
                )
            else:
                self._client_overhead_tokens = int(
                    0.3 * self._client_overhead_tokens + 0.7 * overhead
                )

        log.info(
            "  telemetry | %s in=%d+%dcache out=%d stop=%s "
            "blocks=%d/%d/%d overhead=%s",
            telemetry.message_id[:12] if telemetry.message_id else "?",
            telemetry.input_tokens,
            telemetry.cache_read_tokens,
            telemetry.output_tokens,
            telemetry.stop_reason or "?",
            telemetry.text_blocks,
            telemetry.thinking_blocks,
            telemetry.tool_use_blocks,
            f"{self._client_overhead_tokens:,}"
            if self._client_overhead_tokens is not None else "uncalibrated",
        )

    @property
    def calibrated_total_tokens(self) -> int:
        """Projection tokens + estimated client overhead.

        This is our best estimate of what the API will actually see.
        Use this for pressure decisions, not projection.total_tokens.
        """
        base = self.orchestrator.projection.total_tokens
        return base + (self._client_overhead_tokens or 0)

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


# --- Helpers (owned by the gateway, not the HTTP layer) ---


def _extract_user_content(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, str]] | None]:
    """Extract user content and tool results from the last turn.

    This parses Anthropic message format. It lives here because
    understanding message content is gateway logic, not HTTP logic.
    """
    user_parts: list[str] = []
    tool_results: list[dict[str, str]] = []

    # Walk backwards to find the last user turn
    for msg in reversed(messages):
        if msg.get("role") != "user":
            break
        content = msg.get("content", "")
        if isinstance(content, str):
            user_parts.insert(0, content)
        elif isinstance(content, list):
            for block in content:
                if block.get("type") == "text":
                    user_parts.insert(0, block.get("text", ""))
                elif block.get("type") == "tool_result":
                    result = block.get("content", "")
                    if isinstance(result, list):
                        result = " ".join(
                            b.get("text", "") for b in result
                            if b.get("type") == "text"
                        )
                    tool_results.insert(0, {
                        "content": str(result),
                        "name": block.get("tool_use_id", "tool"),
                    })

    return "\n".join(user_parts), tool_results or None


def _merge_system(
    client_system: Any,
    gateway_system: Any,
) -> Any:
    """Merge client's system prompt with gateway additions (page table).

    The client's system prompt is preserved as-is. Gateway additions
    are appended only if they contain content not already present in
    the client's system. cache_control is stripped from gateway
    additions since the client owns caching strategy.
    """
    # Normalize both to lists
    if isinstance(client_system, str):
        client_parts = [{"type": "text", "text": client_system}]
    elif isinstance(client_system, list):
        client_parts = list(client_system)
    else:
        client_parts = []

    if isinstance(gateway_system, str):
        gw_parts = [{"type": "text", "text": gateway_system}]
    elif isinstance(gateway_system, list):
        gw_parts = [{k: v for k, v in p.items() if k != "cache_control"}
                     for p in gateway_system]
    else:
        gw_parts = []

    # Deduplicate: only append gateway parts whose text isn't already
    # present in the client system
    client_texts = {p.get("text", "") for p in client_parts
                    if isinstance(p, dict)}
    new_parts = [p for p in gw_parts
                 if isinstance(p, dict) and p.get("text", "") not in client_texts]

    return client_parts + new_parts


def _extract_response_text(resp_data: dict[str, Any]) -> str:
    """Extract text content from an Anthropic API response."""
    content = resp_data.get("content", [])
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)
