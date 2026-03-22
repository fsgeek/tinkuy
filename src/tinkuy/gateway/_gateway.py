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
from tinkuy.core.pressure import PressureZone
from tinkuy.core.regions import Projection
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
        gw._ingest = IngestAdapter(gw.orchestrator)
        # Note: adapters use self.orchestrator which is now 'restored'
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

        turn_result = self.process_turn(
            user_content=user_content or "",
            tool_results=tool_results,
            format=APIFormat.ANTHROPIC,
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

        log.info(
            "  telemetry | %s in=%d+%dcache out=%d overhead=%s",
            telemetry.message_id[:12] if telemetry.message_id else "?",
            telemetry.input_tokens,
            telemetry.cache_read_tokens,
            telemetry.output_tokens,
            f"{self._client_overhead_tokens:,}"
            if self._client_overhead_tokens is not None else "uncalibrated",
        )

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
