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

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

log = logging.getLogger("tinkuy.gateway")

# ---------------------------------------------------------------------------
# Cooperative memory protocol — injected alongside the page table
# ---------------------------------------------------------------------------

MEMORY_PROTOCOL = """\
<yuyay-memory-protocol>
This conversation is managed by Tinkuy, a virtual memory system. Each
turn includes a <yuyay-page-table> showing what content is in memory,
including evicted blocks that can be recalled.

You can emit cooperative memory signals inside <yuyay-response> blocks.
These are metadata — they will be stripped from your visible response.

SIGNALS:

  <release handle="H" losses="what was lost">
    Offer to release content block H. Provide a tensor (compressed
    summary) that can replace it. Declare what information is lost.
  </release>
  <tensor handle="H">compressed summary here</tensor>

  <retain handle="H" />
    Cancel a pending eviction for block H. Use when you still need it.

  <recall handle="H" />
    Page fault: request that evicted block H be restored to context.

  <declare handle="H">
    <depends-on handle="P1" />
    <depends-on handle="P2" />
  </declare>
    Declare that block H depends on blocks P1, P2. Emit this when you
    make a decision that builds on prior content. Edges are immutable —
    they record the reasoning chain at the moment it was live.

  <trace handle="H" />
    Request the full provenance chain for block H. The system walks
    the dependency graph and recalls all ancestors. Use when you need
    to reconstruct *why* a decision was made, not just *what* it was.

Wrap all signals in <yuyay-response>...</yuyay-response>.

WHEN TO DECLARE EDGES: When your response makes a decision, conclusion,
or recommendation that depends on earlier content, declare the edges
immediately. You know the dependencies now — they will be harder to
reconstruct later.

WHEN TO TRACE: When you need to explain *why* something was decided
and the reasoning chain is not in your current context, trace the
handle. The system will page in the chain.
</yuyay-memory-protocol>"""

from tinkuy.core.adapter import IngestAdapter
from tinkuy.core.events import ConsoleStatusConsumer, EventBus, EventLog
from tinkuy.core.orchestrator import (
    EventType,
    InboundEvent,
    Orchestrator,
    ResponseSignal,
    ResponseSignalType,
    TurnRecord,
)
from tinkuy.core.events import EventKind
from tinkuy.core.pressure import PressureZone
from tinkuy.core.regions import ContentStatus, Projection
from tinkuy.core.store import (
    CheckpointStore,
    FileCheckpointStore,
    FilePageStore,
    FileTensorStore,
    MemoryCheckpointStore,
    MemoryPageStore,
    MemoryTensorStore,
    PageStore,
)
from tinkuy.formats.anthropic import LiveAdapter as AnthropicLiveAdapter
from tinkuy.formats.gemini import (
    GeminiLiveAdapter,
    GeminiInboundAdapter,
    GeminiResponseIngester,
)

class APIFormat(Enum):
    """Supported API formats.

    This is a per-request property, not a gateway configuration.
    The gateway is format-agnostic — the projection is the source
    of truth.  Format describes how a specific request arrived and
    how its response should be synthesized.
    """
    ANTHROPIC = auto()
    GEMINI = auto()

@dataclass
class GatewayConfig:
    """Configuration for a gateway instance."""
    context_limit: int = 200_000
    data_dir: str | None = None        # filesystem persistence root
    session_id: str | None = None      # session identity (from adapter)
    enable_console: bool = False        # console status consumer
    enable_event_log: bool = True       # in-memory event log
    projector: Any = None              # Hamutay Projector sidecar (optional)
    tensor_store: Any = None           # TensorStore backend (optional)


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
    api_payload: dict[str, Any]        # synthesized API payload (Anthropic or Gemini)
    record: TurnRecord                 # observability record
    pressure_zone: PressureZone        # current pressure after this turn
    pending_evictions: list[str] = field(default_factory=list)  # handles


class Gateway:
    """The integration layer.

    A harness creates a Gateway, optionally rehydrate or resumes it,
    then calls process_turn() for each request/response cycle.
    """

    def __init__(self, config: GatewayConfig | None = None) -> None:
        self.config = config or GatewayConfig()
        self._setup_stores()
        self._setup_bus()
        self._setup_orchestrator()
        
        # Adapters
        self._ingest = IngestAdapter(self.orchestrator)
        self._anthropic_live = AnthropicLiveAdapter(self.orchestrator)
        
        self._gemini_live = GeminiLiveAdapter(self.orchestrator)
        self._gemini_inbound = GeminiInboundAdapter()
        self._gemini_response = GeminiResponseIngester(self.orchestrator)
        
        self.telemetry: list[TurnTelemetry] = []
        self._client_overhead_tokens: int | None = None
        self._pending_turn_context: dict[str, Any] | None = None
        self._telemetry_path: Path | None = None
        if self.config.data_dir and self.config.session_id:
            self._telemetry_path = (
                Path(self.config.data_dir) / "sessions"
                / self.config.session_id / "telemetry.jsonl"
            )

    def _setup_stores(self) -> None:
        """Initialize page store, checkpoint store, and tensor store.

        Page store is workspace-scoped (shared across sessions).
        Checkpoint store is session-scoped (keyed by session_id).
        Tensor store is workspace-scoped (tensors are immutable and shared).

        If config.tensor_store is provided (e.g. a Yanantin backend),
        it takes precedence over the default filesystem store.
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
            # Tensors are shared and immutable
            self.tensor_store = self.config.tensor_store or FileTensorStore(root / "tensors")
        else:
            self.page_store = MemoryPageStore()
            self.checkpoint_store = MemoryCheckpointStore()
            self.tensor_store = self.config.tensor_store or MemoryTensorStore()

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
            tensor_store=self.tensor_store,
            projector=self.config.projector,
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
            log.info("no checkpoint found — fresh start")
            return None
        log.info(
            "resumed from checkpoint: turn=%d, tokens=%d",
            restored.projection.turn,
            restored.projection.total_tokens,
        )
        gw.orchestrator = restored
        # Rebuild all adapters — they hold a reference to the orchestrator
        gw._ingest = IngestAdapter(gw.orchestrator)
        gw._anthropic_live = AnthropicLiveAdapter(gw.orchestrator)
        gw._gemini_live = GeminiLiveAdapter(gw.orchestrator)
        gw._gemini_response = GeminiResponseIngester(gw.orchestrator)
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
        user_content: str | None = None,
        tool_results: list[dict[str, Any]] | None = None,
        events: list[InboundEvent] | None = None,
        format: APIFormat = APIFormat.ANTHROPIC,
    ) -> TurnResult:
        """Process a single turn.

        Accepts either user_content/tool_results (shorthand) or
        a pre-parsed list of InboundEvents.

        format is a per-request property: it determines which adapter
        synthesizes the outbound payload.  The projection itself is
        format-agnostic.
        """
        inbound = events or []

        # User message shorthand
        if user_content:
            inbound.append(InboundEvent(
                type=EventType.USER_MESSAGE,
                content=user_content,
                label="user",
            ))

        # Tool results shorthand
        if tool_results:
            for tr in tool_results:
                inbound.append(InboundEvent(
                    type=EventType.TOOL_RESULT,
                    content=tr.get("content", ""),
                    label=tr.get("name", "tool_result"),
                    metadata=tr.get("metadata", {}),
                ))

        # Begin turn
        record = self.orchestrator.begin_turn(inbound)

        # Apply eviction decisions
        pending = self.orchestrator.apply_decisions(
            record.eviction_decisions
        )

        # Synthesize API payload — adapter chosen per-request
        payload = self._synthesize(format)

        pressure = self.orchestrator.scheduler.read_pressure(
            self.orchestrator.projection
        )

        return TurnResult(
            api_payload=payload,
            record=record,
            pressure_zone=pressure.zone,
            pending_evictions=pending,
        )

    def _synthesize(self, format: APIFormat) -> dict[str, Any]:
        """Synthesize the outbound API payload for the given format."""
        if format == APIFormat.GEMINI:
            payload = self._gemini_live.synthesize_request()
            # TODO: Inject page table for Gemini format
            return payload

        # Anthropic (default) — synthesize_messages handles page table
        # placement internally, respecting tool_result ordering.
        payload = self._anthropic_live.synthesize_messages()
        self._inject_memory_protocol(payload)

        # Pre-flight validation — catch layout violations before the wire
        from tinkuy.formats.validate import validate_anthropic_payload
        validation = validate_anthropic_payload(payload)
        if not validation.valid:
            for err in validation.errors:
                loc = f" at {err.location}" if err.location else ""
                log.error("PREFLIGHT [%s]%s: %s", err.rule, loc, err.message)
            # Log but don't block — we want to see what the API says too
            # TODO: consider raising after confidence in the validator grows

        return payload

    def ingest_response(
        self,
        content: str,
        signals: list[dict[str, Any]] | None = None,
        content_blocks: list[dict[str, Any]] | None = None,
    ) -> TurnRecord:
        """Ingest an API response after the harness calls the API."""
        parsed_signals: list[ResponseSignal] | None = None
        if signals:
            parsed_signals = [
                self._parse_signal(s) for s in signals
            ]

        record = self.orchestrator.ingest_response(
            content=content,
            label="assistant",
            signals=parsed_signals,
            content_blocks=content_blocks,
        )

        # Mark idle after response ingestion — this is a cache boundary
        self.orchestrator.mark_idle()

        return record

    # --- Raw request/response boundary (DEATH TO PROXIES) ---

    def prepare_request(self, client_body: dict[str, Any]) -> dict[str, Any]:
        """Anthropic-specific request preparation."""
        request_messages = client_body.get("messages", [])

        # Cold start: if the projection is empty but the client has
        # conversation history, bootstrap the projection from the
        # client's messages. This is NOT the proxy pattern — we ingest
        # the history into the projection so the gateway has state.
        if self.orchestrator.turn == 0 and len(request_messages) > 1:
            log.info(
                "cold start: bootstrapping projection from %d client messages",
                len(request_messages),
            )
            self._bootstrap_from_client(request_messages, client_body)

        user_content, tool_results = _extract_user_content(request_messages)

        log.info(
            "← request (anthropic) | session=%s stream=%s "
            "client_messages=%d user_content_len=%d tool_results=%d",
            self.config.session_id or "default",
            client_body.get("stream", False),
            len(request_messages),
            len(user_content) if user_content else 0,
            len(tool_results) if tool_results else 0,
        )

        # Feed max_tokens into pressure scheduler so it reserves
        # output budget when computing effective context limit
        max_tokens = client_body.get("max_tokens", 0)
        if isinstance(max_tokens, int) and max_tokens > 0:
            self.orchestrator.scheduler.output_budget = max_tokens

        # Snapshot client request metadata before processing
        self._pending_turn_context = _capture_client_context(
            client_body, request_messages,
        )

        turn_result = self.process_turn(
            user_content=user_content or "",
            tool_results=tool_results,
            format=APIFormat.ANTHROPIC,
        )

        # Snapshot gateway state after processing (projection is updated)
        self._pending_turn_context["projection"] = (
            self._snapshot_gateway_state()
        )
        # Capture repair counts from the synthesizer
        self._pending_turn_context["repairs"] = (
            self._anthropic_live.last_repair_counts
        )

        # Build complete upstream body
        upstream: dict[str, Any] = {
            k: v for k, v in client_body.items()
            if k not in ("messages", "system")
        }
        upstream["messages"] = turn_result.api_payload.get("messages", [])

        # Merge system prompts
        gateway_system = turn_result.api_payload.get("system")
        client_system = client_body.get("system")
        if gateway_system is not None and client_system is not None:
            upstream["system"] = _merge_system(client_system, gateway_system)
        elif gateway_system is not None:
            upstream["system"] = gateway_system
        elif client_system is not None:
            upstream["system"] = client_system

        # Capture wire metadata
        self._pending_turn_context["wire"] = {
            "message_count": len(upstream.get("messages", [])),
        }

        return upstream

    def prepare_gemini_request(self, client_body: dict[str, Any]) -> dict[str, Any]:
        """Gemini-specific request preparation."""
        events = self._gemini_inbound.parse_request(client_body)

        log.info(
            "← request (gemini) | session=%s model=%s events=%d",
            self.config.session_id or "default",
            client_body.get("model", "?"),
            len(events),
        )

        turn_result = self.process_turn(events=events, format=APIFormat.GEMINI)

        # Build complete upstream Gemini request
        upstream = {
            k: v for k, v in client_body.items()
            if k not in ("contents", "system_instruction", "tools")
        }
        synth = turn_result.api_payload
        upstream["contents"] = synth.get("contents", [])
        if "system_instruction" in synth:
            upstream["system_instruction"] = synth["system_instruction"]
        if "tools" in synth:
            upstream["tools"] = synth["tools"]
            
        return upstream

    def ingest_raw_response(
        self, text: str, content_blocks: list[dict[str, Any]] | None = None,
    ) -> TurnRecord | None:
        """Ingest raw assistant response text and structured content."""
        if not text and not content_blocks:
            return None
        from tinkuy.gateway.harness import extract_signals, strip_signals
        signals = extract_signals(text) if text else []
        clean = strip_signals(text) if text else ""
        return self.ingest_response(
            content=clean,
            signals=signals if signals else None,
            content_blocks=content_blocks,
        )

    def ingest_response_json(self, response_data: dict[str, Any]) -> TurnRecord | None:
        """Ingest a complete Anthropic API response JSON."""
        text, content_blocks = _extract_response_content_from_json(response_data)
        return self.ingest_raw_response(text, content_blocks=content_blocks)

    def ingest_gemini_response(self, response_data: dict[str, Any]) -> TurnRecord | None:
        """Ingest a complete Gemini API response JSON."""
        return self._gemini_response.ingest_response(response_data)

    # --- Telemetry ---

    def report_telemetry(self, telemetry: TurnTelemetry) -> None:
        """Report API telemetry from the last turn."""
        telemetry.turn = self.turn
        self.telemetry.append(telemetry)

        api_total = telemetry.total_input_tokens
        projection_tokens = self.orchestrator.projection.total_tokens
        if api_total > 0:
            if api_total > projection_tokens:
                overhead = api_total - projection_tokens
            else:
                overhead = 0

            if self._client_overhead_tokens is None:
                self._client_overhead_tokens = overhead
            else:
                self._client_overhead_tokens = int(
                    0.3 * self._client_overhead_tokens + 0.7 * overhead
                )

            # Feed overhead into pressure scheduler so it knows the
            # real budget available for projection content
            self.orchestrator.scheduler.overhead_tokens = (
                self._client_overhead_tokens
            )

        log.info(
            "  telemetry | %s in=%d+%dcache out=%d overhead=%s",
            telemetry.message_id[:12] if telemetry.message_id else "?",
            telemetry.input_tokens,
            telemetry.cache_read_tokens,
            telemetry.output_tokens,
            f"{self._client_overhead_tokens:,}"
            if self._client_overhead_tokens is not None else "uncalibrated",
        )

        # Persist the complete turn record
        self._write_turn_record(telemetry)

    def _snapshot_gateway_state(self) -> dict[str, Any]:
        """Snapshot the gateway's view of the world for telemetry."""
        projection = self.orchestrator.projection
        pressure = self.orchestrator.scheduler.read_pressure(projection)
        regions: dict[str, Any] = {}
        for rid, region in projection.regions.items():
            regions[rid.name] = {
                "tokens": region.size_tokens,
                "blocks": len(region.blocks),
                "waste_tokens": region.waste_tokens,
                "present": sum(
                    1 for b in region.blocks
                    if b.status == ContentStatus.PRESENT
                ),
                "evicted": sum(
                    1 for b in region.blocks
                    if b.status == ContentStatus.AVAILABLE
                ),
            }
        return {
            "total_tokens": projection.total_tokens,
            "waste_tokens": projection.waste_tokens,
            "regions": regions,
            "pressure_zone": pressure.zone.name,
            "usage_ratio": round(pressure.usage, 4),
            "headroom_tokens": pressure.headroom_tokens,
            "overhead_estimate": self._client_overhead_tokens,
            "projected_message_count": None,  # filled by caller after synthesis
        }

    def _write_turn_record(self, telemetry: TurnTelemetry) -> None:
        """Append a complete turn record to the session JSONL file."""
        if self._telemetry_path is None:
            return

        ctx = self._pending_turn_context or {}
        self._pending_turn_context = None

        # Collect eviction events from the event log for this turn
        eviction_data = self._collect_eviction_data()

        record: dict[str, Any] = {
            "session_id": self.config.session_id,
            "turn": telemetry.turn,
            "timestamp": telemetry.timestamp,
            # Client request metadata
            "request": ctx.get("request", {}),
            # Gateway state at request time
            "projection": ctx.get("projection", {}),
            # Synthesizer repairs
            "repairs": ctx.get("repairs", {}),
            # Wire metadata
            "wire": {
                **(ctx.get("wire", {})),
                "request_bytes": telemetry.request_bytes,
            },
            # API response — ground truth
            "response": {
                "message_id": telemetry.message_id,
                "model": telemetry.model,
                "input_tokens": telemetry.input_tokens,
                "cache_read_tokens": telemetry.cache_read_tokens,
                "cache_create_tokens": telemetry.cache_create_tokens,
                "output_tokens": telemetry.output_tokens,
                "stop_reason": telemetry.stop_reason,
                "text_blocks": telemetry.text_blocks,
                "thinking_blocks": telemetry.thinking_blocks,
                "tool_use_blocks": telemetry.tool_use_blocks,
                "tool_names": telemetry.tool_names,
                "ttfb": telemetry.ttfb,
                "duration": telemetry.duration,
            },
            # Eviction activity
            "eviction": eviction_data,
            # Derived — overhead calibration after this turn
            "overhead_calibrated": self._client_overhead_tokens,
        }

        # Update projected_message_count from wire data
        proj = record.get("projection", {})
        if proj and record["wire"].get("message_count"):
            proj["projected_message_count"] = record["wire"]["message_count"]

        try:
            self._telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._telemetry_path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError:
            log.warning("failed to write telemetry record", exc_info=True)

    def _collect_eviction_data(self) -> dict[str, Any]:
        """Collect eviction/fault counts from the event log."""
        data: dict[str, Any] = {
            "evictions_this_turn": 0,
            "faults_this_turn": 0,
            "evicted_handles": [],
            "faulted_handles": [],
        }
        if self.event_log is None:
            return data

        for event in self.event_log.events:
            if event.turn != self.turn:
                continue
            if event.kind in (EventKind.EVICTION_EXECUTED, EventKind.BLOCK_EVICTED):
                data["evictions_this_turn"] += 1
                handle = event.data.get("handle")
                if handle:
                    data["evicted_handles"].append(handle)
            elif event.kind == EventKind.BLOCK_RECALLED:
                data["faults_this_turn"] += 1
                handle = event.data.get("handle")
                if handle:
                    data["faulted_handles"].append(handle)
        return data

    @property
    def calibrated_total_tokens(self) -> int:
        """Projection tokens + estimated client overhead."""
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

    def _bootstrap_from_client(
        self,
        messages: list[dict[str, Any]],
        client_body: dict[str, Any],
    ) -> None:
        """Bootstrap the projection from the client's conversation history.

        Called on cold start when the projection is empty but the client
        has existing conversation state. Ingests all messages except the
        last user turn (which will be processed normally).

        This is NOT the proxy pattern. The client messages are ingested
        into the projection — the gateway owns the resulting state.
        """
        from tinkuy.core.orchestrator import EventType, InboundEvent

        # Ingest system prompt if present
        client_system = client_body.get("system")
        if client_system:
            system_text = client_system
            if isinstance(client_system, list):
                system_text = "\n".join(
                    p.get("text", "") for p in client_system
                    if isinstance(p, dict)
                )
            self.orchestrator.begin_turn([
                InboundEvent(
                    type=EventType.SYSTEM_UPDATE,
                    content=system_text,
                    label="client system prompt",
                )
            ])

        # Find the boundary: everything before the last user turn is history
        last_user_start = len(messages)
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_start = i
            else:
                break

        # Ingest history messages into the projection as turns
        history = messages[:last_user_start]
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, list):
                # Extract text and tool_use/tool_result blocks
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                content_str = "\n".join(text_parts)
            else:
                content_str = str(content)

            if not content_str.strip():
                continue

            if role == "user":
                self.orchestrator.begin_turn([
                    InboundEvent(
                        type=EventType.USER_MESSAGE,
                        content=content_str,
                        label="user",
                    )
                ])
            elif role == "assistant":
                self.orchestrator.ingest_response(
                    content=content_str,
                    label="assistant",
                )

        log.info(
            "bootstrap complete: ingested %d history messages, "
            "projection turn=%d tokens=%d",
            len(history),
            self.orchestrator.turn,
            self.orchestrator.projection.total_tokens,
        )

    def _inject_memory_protocol(self, payload: dict[str, Any]) -> None:
        """Inject the cooperative memory protocol instructions."""
        system = payload.get("system", [])
        if isinstance(system, list):
            system.append({
                "type": "text",
                "text": MEMORY_PROTOCOL,
            })
            payload["system"] = system
        elif isinstance(system, str):
            payload["system"] = system + "\n\n" + MEMORY_PROTOCOL

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
        elif signal_type == "declare":
            return ResponseSignal(
                type=ResponseSignalType.DECLARE,
                handle=handle,
                depends_on=data.get("depends_on"),
            )
        elif signal_type == "trace":
            return ResponseSignal(
                type=ResponseSignalType.TRACE,
                handle=handle,
            )
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")


# --- Helpers ---


def _extract_user_content(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]] | None]:
    """Extract user content and tool results from the last turn."""
    user_parts: list[str] = []
    tool_results: list[dict[str, Any]] = []

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
                    tool_use_id = block.get("tool_use_id", "tool")
                    tool_results.insert(0, {
                        "content": str(result),
                        "name": tool_use_id,
                        "metadata": {
                            "tool_use_id": tool_use_id,
                            "is_error": block.get("is_error", False),
                        },
                    })

    return "\n".join(user_parts), tool_results or None


def _merge_system(
    client_system: Any,
    gateway_system: Any,
) -> Any:
    """Merge client's system prompt with gateway additions.

    The gateway is the cache authority. It places cache_control
    breakpoints on stable content boundaries. The merge preserves
    these breakpoints — stripping them would destroy cache hit rate.

    Order: client parts first (stable across turns in practice),
    then gateway parts (R0/R1 stable, page table volatile).
    """
    if isinstance(client_system, str):
        client_parts = [{"type": "text", "text": client_system}]
    elif isinstance(client_system, list):
        client_parts = list(client_system)
    else:
        client_parts = []

    if isinstance(gateway_system, str):
        gw_parts = [{"type": "text", "text": gateway_system}]
    elif isinstance(gateway_system, list):
        gw_parts = list(gateway_system)
    else:
        gw_parts = []

    # Deduplicate — don't repeat content the client already sent
    client_texts = {p.get("text", "") for p in client_parts
                    if isinstance(p, dict)}
    new_parts = [p for p in gw_parts
                 if isinstance(p, dict) and p.get("text", "") not in client_texts]

    return client_parts + new_parts


def _extract_response_content_from_json(
    resp_data: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Extract text and full content blocks from an Anthropic API response."""
    content = resp_data.get("content", [])
    text_parts: list[str] = []
    content_blocks: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type", "")
        if block_type == "text":
            text_parts.append(block.get("text", ""))
            content_blocks.append({"type": "text", "text": block.get("text", "")})
        elif block_type == "tool_use":
            content_blocks.append({
                "type": "tool_use",
                "id": block.get("id", ""),
                "name": block.get("name", ""),
                "input": block.get("input", {}),
            })
    return "\n".join(text_parts), content_blocks


def _capture_client_context(
    client_body: dict[str, Any],
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Capture client request metadata for telemetry.

    Hashes large structures (tools, system) instead of storing them.
    This is the 'what came in' side of the research record.
    """
    # Hash tools array
    tools = client_body.get("tools")
    tools_hash = None
    tools_count = 0
    if tools and isinstance(tools, list):
        tools_count = len(tools)
        tools_hash = hashlib.sha256(
            json.dumps(tools, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

    # Hash system prompt
    system = client_body.get("system")
    system_hash = None
    system_token_estimate = 0
    if system:
        system_str = system
        if isinstance(system, list):
            system_str = json.dumps(system, default=str)
        elif not isinstance(system, str):
            system_str = str(system)
        system_hash = hashlib.sha256(
            system_str.encode() if isinstance(system_str, str)
            else system_str
        ).hexdigest()[:16]
        # Rough token estimate: ~4 chars per token
        system_token_estimate = len(system_str) // 4

    return {
        "request": {
            "model": client_body.get("model"),
            "max_tokens": client_body.get("max_tokens"),
            "stream": client_body.get("stream"),
            "effort": (
                client_body.get("output_config", {}).get("effort")
                if isinstance(client_body.get("output_config"), dict)
                else None
            ),
            "thinking": client_body.get("thinking"),
            "context_management": client_body.get("context_management"),
            "client_message_count": len(messages),
            "tools_hash": tools_hash,
            "tools_count": tools_count,
            "system_hash": system_hash,
            "system_token_estimate": system_token_estimate,
            # Non-standard fields the client sends
            "extra_fields": [
                k for k in client_body
                if k not in {
                    "model", "messages", "system", "tools", "max_tokens",
                    "stream", "temperature", "top_p", "top_k",
                    "stop_sequences", "metadata", "tool_choice",
                    "thinking", "context_management", "output_config",
                }
            ],
        },
    }
