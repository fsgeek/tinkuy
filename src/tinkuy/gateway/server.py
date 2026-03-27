"""HTTP server — the thin network layer for the tinkuy gateway.

Receives HTTP requests, hands them to the gateway, forwards the
gateway's synthesized payload upstream, streams the response back.

The server is a dumb pipe between the network and the gateway.
It never sees, parses, or transforms message content. The gateway
owns that. DEATH TO PROXIES.

Usage:
    tinkuy serve --port 8340 --upstream https://api.anthropic.com

Then point your client at:
    export ANTHROPIC_BASE_URL=http://127.0.0.1:8340
"""

import json
import logging
import os
import socket
import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, Optional

from tinkuy.gateway._gateway import Gateway, GatewayConfig, TurnTelemetry
from tinkuy.gateway.stream import BlockType, StreamBuffer
from tinkuy.taste_gateway import TasteGateway, TasteGatewayConfig

log = logging.getLogger("tinkuy.server")


def find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _resolve_upstream(provider: str, explicit: str | None = None) -> str:
    """Resolve upstream URL for a provider.

    Priority: explicit arg > environment variable > hardcoded default.
    """
    if explicit:
        return explicit
    defaults = {
        "anthropic": (
            ["ANTHROPIC_BASE_URL_UPSTREAM"],
            "https://api.anthropic.com",
        ),
        "gemini": (
            ["GOOGLE_GEMINI_BASE_URL", "GOOGLE_VERTEX_BASE_URL"],
            "https://generativelanguage.googleapis.com",
        ),
    }
    env_keys, fallback = defaults.get(provider, ([], ""))
    for key in env_keys:
        val = os.environ.get(key)
        if val:
            return val
    return fallback


def create_app(
    gateway_config: GatewayConfig | None = None,
    upstream: str | None = None,
    gemini_upstream: str | None = None,
    taste: bool = False,
) -> Any:
    """Create the FastAPI application.

    Import is deferred so the rest of tinkuy works without
    fastapi/uvicorn installed.

    Each provider gets its own upstream URL, resolved from explicit
    args, environment variables, or hardcoded defaults.
    """
    try:
        from fastapi import FastAPI, Request, Response
        import httpx
    except ImportError:
        raise ImportError(
            "FastAPI server requires extra dependencies. "
            "Install with: pip install tinkuy[serve]"
        )

    anthropic_upstream = _resolve_upstream("anthropic", upstream)
    gemini_upstream_url = _resolve_upstream("gemini", gemini_upstream)

    config = gateway_config or GatewayConfig(enable_console=True)

    # Taste mode: one TasteGateway handles all sessions internally
    taste_gw: TasteGateway | None = None
    if taste:
        taste_gw = TasteGateway(TasteGatewayConfig(
            data_dir=config.data_dir,
            enable_console=True,
            context_limit=config.context_limit,
        ))

    # Per-session gateways, keyed by session ID or a default
    gateways: dict[str, Gateway] = {}


    def _extract_session_id(
        header_value: Optional[str],
        body: dict[str, Any] | None = None,
    ) -> str | None:
        """Extract session ID from header or body metadata.

        Priority: explicit header > Claude Code metadata > None.
        Claude Code buries session_id inside a JSON-encoded string
        in metadata.user_id.
        """
        if header_value:
            return header_value
        if body:
            metadata = body.get("metadata", {})
            user_id_raw = metadata.get("user_id", "")
            if isinstance(user_id_raw, str) and user_id_raw.startswith("{"):
                try:
                    parsed = json.loads(user_id_raw)
                    sid = parsed.get("session_id")
                    if sid:
                        return sid
                except (json.JSONDecodeError, TypeError):
                    pass
        return None

    def get_gateway(
        session_id: Optional[str],
        lightweight: bool = False,
    ) -> Gateway:
        key = session_id or "__default__"
        if lightweight:
            key = f"{key}:lightweight"
        if key not in gateways:
            cfg = GatewayConfig(
                context_limit=config.context_limit,
                data_dir=config.data_dir,
                session_id=session_id,
                enable_console=config.enable_console,
                enable_event_log=config.enable_event_log,
                tensor_store=config.tensor_store,
                lightweight=lightweight,
            )
            if not lightweight:
                gw = Gateway.resume(cfg)
                if gw is None:
                    gw = Gateway(cfg)
            else:
                gw = Gateway(cfg)
            gateways[key] = gw
        return gateways[key]

    # Wire log: full request bodies for every outbound API call.
    # Written to {data_dir}/wire.jsonl — one JSON line per request.
    _wire_log_path: Path | None = None
    if config.data_dir:
        _wire_log_path = Path(config.data_dir) / "wire.jsonl"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async def log_request(request: httpx.Request):
            content_len = len(request.content) if request.content else 0
            log.info(
                "  WIRE -> %s %s (%d bytes)",
                request.method, request.url, content_len,
            )
            # Write full request body to wire log.
            if _wire_log_path and request.content:
                import time as _time
                record = {
                    "ts": _time.time(),
                    "method": str(request.method),
                    "url": str(request.url),
                    "body_bytes": content_len,
                    "body": json.loads(request.content)
                        if request.headers.get("content-type", "").startswith("application/json")
                           or request.content[:1] == b"{"
                        else request.content.decode("utf-8", errors="replace"),
                }
                with open(_wire_log_path, "a") as f:
                    f.write(json.dumps(record, default=str) + "\n")

        event_hooks = {"request": [log_request]}
        timeout = httpx.Timeout(300.0, connect=10.0)

        app.state.anthropic_client = httpx.AsyncClient(
            base_url=anthropic_upstream,
            timeout=timeout,
            event_hooks=event_hooks,
        )
        app.state.gemini_client = httpx.AsyncClient(
            timeout=timeout,
            event_hooks=event_hooks,
        )
        yield
        await app.state.anthropic_client.aclose()
        await app.state.gemini_client.aclose()

    app = FastAPI(
        title="tinkuy",
        description="Projective gateway for transformer memory hierarchy",
        lifespan=lifespan,
    )

    @app.head("/")
    @app.get("/")
    async def root(request: Request) -> Response:
        """Health check — Claude Code sends HEAD / before each session."""
        return Response(
            content=json.dumps({"status": "ok", "service": "tinkuy"}),
            status_code=200,
            media_type="application/json",
        )

    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc: Any):
        log.warning(
            "!! 404 | %s %s (unhandled route)",
            request.method, request.url.path,
        )
        return Response(
            content=json.dumps({"error": "not found", "path": request.url.path}),
            status_code=404,
            media_type="application/json",
        )

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request) -> Response:
        """Count tokens against the gateway's view, not the client's.

        The client's message array may be much larger (no eviction)
        or much smaller (no tensor injection) than what actually goes
        to the API. Counting against the raw client body lies about
        context usage. Transform first, then count.
        """
        body = await request.json()
        headers = _forward_headers(request)

        if taste_gw is not None:
            # Taste mode: inject tensor into system blocks, same as
            # a real request. Don't increment cycle — this is a probe.
            # We use prepare_request but roll back the cycle after.
            session_id = _extract_session_id(
                request.headers.get("x-tinkuy-session"), body
            )
            transformed, session, _ = taste_gw.prepare_request(
                body, session_id=session_id,
            )
            session.cycle -= 1  # undo the cycle increment
            count_body = {
                k: v for k, v in transformed.items()
                if k != "stream"
            }
        else:
            # Standard mode: transform through the gateway
            session_id = _extract_session_id(
                request.headers.get("x-tinkuy-session"), body
            )
            gw = get_gateway(session_id, lightweight=True)
            count_body = gw.prepare_request(body)

        upstream_resp = await request.app.state.anthropic_client.post(
            "/v1/messages/count_tokens",
            json=count_body,
            headers=headers,
        )
        log.info("count_tokens | status=%d (gateway-transformed)", upstream_resp.status_code)
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            media_type="application/json",
        )

    @app.post("/v1/messages")
    async def messages(request: Request) -> Response:
        """The gateway path. No exceptions. No fallback. No bypass."""
        from starlette.responses import StreamingResponse

        body = await request.json()
        is_streaming = body.get("stream", False)

        # --- Inbound request logging (research data) ---
        # Full body to wire log; summary to gateway log.
        metadata = body.get("metadata", {})
        system_blocks = body.get("system", [])
        system_summary = []
        if isinstance(system_blocks, list):
            for sb in system_blocks:
                if isinstance(sb, dict):
                    system_summary.append({
                        "type": sb.get("type", "?"),
                        "cache_control": sb.get("cache_control"),
                        "text_len": len(sb.get("text", "")),
                    })
        log.info(
            "  inbound | metadata=%s system_blocks=%d system_summary=%s",
            json.dumps(metadata, default=str),
            len(system_blocks) if isinstance(system_blocks, list) else 1,
            json.dumps(system_summary),
        )
        if _wire_log_path:
            import time as _time
            record = {
                "ts": _time.time(),
                "direction": "inbound",
                "body": body,
            }
            with open(_wire_log_path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")

        session_id = _extract_session_id(
            request.headers.get("x-tinkuy-session"), body
        )

        # Forward auth headers — we never touch credentials
        headers = _forward_headers(request)
        client: httpx.AsyncClient = request.app.state.anthropic_client

        # --- Taste mode: tensor in the wire ---
        if taste_gw is not None:
            return await _handle_taste_message(
                taste_gw, body, session_id, headers, client,
                is_streaming, _wire_log_path,
            )

        # --- Standard gateway mode ---
        # Every request goes through the gateway. No proxy path.
        # No exceptions. No "just this one case." No passthrough.
        gw = get_gateway(session_id)

        # Gateway owns the transformation — one call, no branching
        upstream_body = gw.prepare_request(body)

        # Stash request headers in pending telemetry context
        if gw._pending_turn_context is not None:
            gw._pending_turn_context["request"]["beta_headers"] = (
                headers.get("anthropic-beta", "")
            )
            gw._pending_turn_context["request"]["anthropic_version"] = (
                headers.get("anthropic-version", "")
            )

        _log_message_structure(upstream_body)
        log.info(
            "  headers | version=%s beta=%s",
            headers.get("anthropic-version", "MISSING"),
            headers.get("anthropic-beta", "MISSING"),
        )

        if is_streaming:
            upstream_bytes = json.dumps(upstream_body).encode("utf-8")
            headers["content-type"] = "application/json"
            upstream_req = client.build_request(
                "POST", "/v1/messages",
                content=upstream_bytes, headers=headers,
            )
            t_start = time.monotonic()
            upstream_resp = await client.send(upstream_req, stream=True)

            log.info("-> upstream | status=%d (stream)", upstream_resp.status_code)

            if upstream_resp.status_code >= 400:
                return await _handle_error(
                    upstream_resp, upstream_body, upstream_bytes
                )

            buffer = StreamBuffer()
            t_first_byte: float | None = None

            async def stream_and_collect():
                nonlocal t_first_byte
                async for chunk in upstream_resp.aiter_bytes():
                    if t_first_byte is None:
                        t_first_byte = time.monotonic()
                    # StreamBuffer parses, reconstructs, re-serializes
                    for out_chunk in buffer.feed(chunk):
                        yield out_chunk
                await upstream_resp.aclose()

                t_end = time.monotonic()

                # Build telemetry from the reconstructed message
                if buffer.complete:
                    message = buffer.finish()
                    telemetry = _build_telemetry(
                        message,
                        request_bytes=len(upstream_bytes),
                        ttfb=(t_first_byte - t_start) if t_first_byte else None,
                        duration=t_end - t_start,
                    )
                    # Gateway ingests the response content
                    text, content_blocks = _extract_response_content(message)
                    gw.ingest_raw_response(text, content_blocks=content_blocks)
                    # Gateway receives the telemetry
                    gw.report_telemetry(telemetry)
                    _log_request_summary_from_telemetry(telemetry)

            resp_headers = _strip_encoding_headers(upstream_resp)
            return StreamingResponse(
                stream_and_collect(),
                status_code=upstream_resp.status_code,
                headers=resp_headers,
                media_type="text/event-stream",
            )
        else:
            upstream_resp = await client.post(
                "/v1/messages",
                json=upstream_body,
                headers=headers,
            )

            log.info("-> upstream | status=%d", upstream_resp.status_code)
            resp_headers = _strip_encoding_headers(upstream_resp)

            if upstream_resp.status_code >= 400:
                log.error(
                    "x upstream error | status=%d body=%s",
                    upstream_resp.status_code,
                    upstream_resp.text,
                )
                _dump_rejected_payload(upstream_body, upstream_resp.status_code)
                return Response(
                    content=upstream_resp.content,
                    status_code=upstream_resp.status_code,
                    headers=resp_headers,
                )

            resp_data = upstream_resp.json()
            # Persist the full response JSON for reproducibility.
            # The ingest path extracts text but discards structure.
            if gw._pending_turn_context is not None:
                gw._pending_turn_context["response_json"] = resp_data
            gw.ingest_response_json(resp_data)

            # Build telemetry from non-streaming response
            telemetry = _build_telemetry_from_json(resp_data)
            gw.report_telemetry(telemetry)
            _log_request_summary_from_telemetry(telemetry)

            return Response(
                content=upstream_resp.content,
                status_code=upstream_resp.status_code,
                headers=resp_headers,
                media_type="application/json",
            )

    @app.get("/v1/tinkuy/taste/status")
    async def taste_status(request: Request) -> dict[str, Any]:
        """Taste gateway status — tensor state for all sessions."""
        if taste_gw is None:
            return {"error": "taste mode not enabled"}
        sessions = {}
        for sid, s in taste_gw._sessions.items():
            sessions[sid] = {
                "cycle": s.cycle,
                "n_strands": len(s.tensor.get("strands", [])) if s.tensor else 0,
                "n_tensions": len(s.tensor.get("unresolved_tensions", [])) if s.tensor else 0,
                "tensor_tokens": s.tensor_token_estimate(),
                "cumulative_losses": len(s.loss_history),
            }
        return {"mode": "taste", "sessions": sessions}

    @app.post("/v1beta/models/{model_id}:streamGenerateContent")
    async def gemini_stream(model_id: str, request: Request) -> Response:
        """The Gemini gateway path."""
        from starlette.responses import StreamingResponse

        body = await request.json()
        session_id = _extract_session_id(
            request.headers.get("x-tinkuy-session"), body
        )
        gw = get_gateway(session_id)

        # 1. Transform through Gateway
        upstream_body = gw.prepare_gemini_request(body)
        
        # 2. Forward to actual Google API
        headers = _forward_headers(request)
        # Fix host header for Google if necessary
        if "host" in headers:
            del headers["host"]

        client: httpx.AsyncClient = request.app.state.gemini_client

        target_url = f"{gemini_upstream_url.rstrip('/')}/v1beta/models/{model_id}:streamGenerateContent"

        # Forward query params (like API key)
        if request.url.query:
            target_url += f"?{request.url.query}"

        upstream_bytes = json.dumps(upstream_body).encode("utf-8")
        headers["content-type"] = "application/json"
        
        upstream_req = client.build_request(
            "POST", target_url,
            content=upstream_bytes, headers=headers,
        )
        
        t_start = time.monotonic()
        upstream_resp = await client.send(upstream_req, stream=True)

        if upstream_resp.status_code >= 400:
            return await _handle_error(
                upstream_resp, upstream_body, upstream_bytes
            )

        # Gemini streams a JSON array of response objects, usually chunked.
        # We will pass the chunks directly to the client, but accumulate
        # the JSON strings to re-parse the full response at the end.
        
        accumulated_chunks = []
        
        async def stream_and_collect():
            async for chunk in upstream_resp.aiter_bytes():
                accumulated_chunks.append(chunk)
                yield chunk
            await upstream_resp.aclose()
            
            # Reconstruct the full JSON array once the stream is done
            full_bytes = b"".join(accumulated_chunks)
            try:
                # Gemini streaming returns a JSON array of responses: [ {...}, {...} ]
                responses = json.loads(full_bytes)
                if responses and isinstance(responses, list):
                    # We can synthesize a single combined GenerateContentResponse
                    # by merging the text from the candidates.
                    combined_text = ""
                    for r in responses:
                        candidates = r.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            for p in parts:
                                if "text" in p:
                                    combined_text += p["text"]
                    
                    # Create a synthetic final response to ingest
                    synthetic_response = {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [{"text": combined_text}]
                                }
                            }
                        ]
                    }
                    # Ingest synchronously
                    gw.ingest_gemini_response(synthetic_response)
            except json.JSONDecodeError as e:
                log.error("Failed to decode accumulated Gemini stream: %s", e)

        resp_headers = _strip_encoding_headers(upstream_resp)
        return StreamingResponse(
            stream_and_collect(),
            status_code=upstream_resp.status_code,
            headers=resp_headers,
            media_type="application/json",
        )

    @app.get("/v1/tinkuy/status")
    async def status(request: Request) -> dict[str, Any]:
        """Gateway status endpoint."""
        session_id = request.headers.get("x-tinkuy-session")
        gw = get_gateway(session_id)
        pressure = gw.orchestrator.scheduler.read_pressure(
            gw.orchestrator.projection
        )
        return {
            "turn": gw.turn,
            "pressure_zone": pressure.zone.name,
            "usage": pressure.usage,
            "total_tokens": pressure.total_tokens,
            "context_limit": pressure.context_limit,
            "headroom_tokens": pressure.headroom_tokens,
            "session_id": session_id,
            "page_table": gw.page_table(),
        }

    @app.get("/v1/tinkuy/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "tinkuy"}

    return app


# --- Taste gateway handler ---


class _TasteStreamHandler:
    """StreamHandler for taste mode: injects session tag, strips tensor.

    Two responsibilities:
    1. Prepend <tinkuy-session id="..."/> to the first text delta
       (for session identity propagation through conversation echo)
    2. Strip <yuyay-tensor> blocks from text deltas
       (so tensor XML doesn't echo back through conversation history)

    The reconstructor sees the full unmodified text (it runs before
    handlers). The client sees: session tag + clean response text.
    """

    def __init__(self, session_tag: str | None = None) -> None:
        self._session_tag = session_tag  # None = don't inject
        self._tag_injected = False
        self._suppressing = False
        self._accumulated = ""

    def on_event(self, event: Any) -> Any:
        from tinkuy.gateway.stream import SSEEvent, SSEEventType

        if event.type == SSEEventType.CONTENT_BLOCK_START:
            self._accumulated = ""
            self._suppressing = False
            return event

        if event.type != SSEEventType.CONTENT_BLOCK_DELTA:
            return event

        delta = event.data.get("delta", {})
        delta_type = delta.get("type", "")

        if delta_type != "text_delta":
            return event

        if self._suppressing:
            return None

        text = delta.get("text", "")

        # Inject session tag into the first text delta
        if self._session_tag and not self._tag_injected:
            text = f"{self._session_tag}\n\n{text}"
            self._tag_injected = True
            new_data = dict(event.data)
            new_delta = dict(delta)
            new_delta["text"] = text
            new_data["delta"] = new_delta
            event = SSEEvent(
                type=event.type,
                data=new_data,
                raw_line=event.raw_line,
            )
            # Fall through to tensor detection with modified event

        self._accumulated += text

        # Check for tensor start tag
        tag_start = self._accumulated.find("<yuyay-tensor")
        if tag_start >= 0:
            self._suppressing = True
            prior_len = len(self._accumulated) - len(text)
            keep_chars = tag_start - prior_len
            if keep_chars > 0:
                new_data = dict(event.data)
                new_delta = dict(event.data.get("delta", {}))
                new_delta["text"] = text[:keep_chars]
                new_data["delta"] = new_delta
                return SSEEvent(
                    type=event.type,
                    data=new_data,
                    raw_line=event.raw_line,
                )
            else:
                return None

        return event

    def on_complete(self, message: Any) -> None:
        pass


async def _handle_taste_message(
    taste_gw: TasteGateway,
    body: dict[str, Any],
    session_id: str | None,
    headers: dict[str, str],
    client: Any,  # httpx.AsyncClient
    is_streaming: bool,
    wire_log_path: Path | None,
) -> Any:
    """Handle a /v1/messages request through the taste gateway.

    Streaming: stream chunks to client AND accumulate via StreamBuffer.
    After stream completes, reconstruct full text, process through
    TasteGateway for state tracking + logging. The client sees the
    tensor XML at the end of the response — that's research data.

    Non-streaming: full round-trip, process response, return.
    """
    from starlette.responses import Response, StreamingResponse

    # 1. Transform request — inject tensor into system prompt
    # Do NOT pass metadata session_id for session lookup. Session identity
    # comes from the <tinkuy-session/> tag in conversation history, which
    # is per-conversation-thread. The metadata session_id is shared across
    # mainline and agents — using it would cause session collisions.
    upstream_body, session, feedback = taste_gw.prepare_request(body)

    _log_message_structure(upstream_body)
    log.info(
        "  taste | session=%s cycle=%d metadata_sid=%s | "
        "headers version=%s beta=%s",
        session.session_id, session.cycle,
        session_id or "none",
        headers.get("anthropic-version", "MISSING"),
        headers.get("anthropic-beta", "MISSING"),
    )

    # Wire log — full transformed request for replay
    if wire_log_path:
        record = {
            "ts": time.time(),
            "direction": "taste_outbound",
            "session_id": session.session_id,
            "cycle": session.cycle,
            "body": upstream_body,
        }
        with open(wire_log_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    if is_streaming:
        upstream_bytes = json.dumps(upstream_body).encode("utf-8")
        headers["content-type"] = "application/json"
        upstream_req = client.build_request(
            "POST", "/v1/messages",
            content=upstream_bytes, headers=headers,
        )
        t_start = time.monotonic()
        upstream_resp = await client.send(upstream_req, stream=True)

        log.info(
            "  taste -> upstream | status=%d (stream)",
            upstream_resp.status_code,
        )

        if upstream_resp.status_code >= 400:
            return await _handle_error(
                upstream_resp, upstream_body, upstream_bytes,
            )

        # Inject session tag if this is a new session
        from tinkuy.taste_gateway.tensor_protocol import make_session_tag
        tag = (
            make_session_tag(session.session_id)
            if not session.tag_injected else None
        )
        handler = _TasteStreamHandler(session_tag=tag)
        buffer = StreamBuffer(handlers=[handler])
        t_first_byte: float | None = None

        async def taste_stream_and_collect():
            nonlocal t_first_byte
            async for chunk in upstream_resp.aiter_bytes():
                if t_first_byte is None:
                    t_first_byte = time.monotonic()
                # Stream through to client AND accumulate
                for out_chunk in buffer.feed(chunk):
                    yield out_chunk
            await upstream_resp.aclose()

            t_end = time.monotonic()

            # Reconstruct full message and process through taste gateway
            if buffer.complete:
                message = buffer.finish()
                text, content_blocks = _extract_response_content(message)
                usage = message.usage

                timing = {
                    "ttfb": (
                        (t_first_byte - t_start) if t_first_byte else None
                    ),
                    "duration": t_end - t_start,
                    "request_bytes": len(upstream_bytes),
                }

                # Mark session tag as injected if handler did it
                if handler._tag_injected:
                    session.tag_injected = True

                # Process through taste gateway — state tracking + logging
                taste_gw.process_response(
                    response_text=text,
                    session=session,
                    content_blocks=content_blocks,
                    usage=usage,
                    request_body=upstream_body,
                    timing=timing,
                    feedback=feedback,
                )

                # Also log telemetry summary to console
                telemetry = _build_telemetry(
                    message,
                    request_bytes=len(upstream_bytes),
                    ttfb=timing["ttfb"],
                    duration=timing["duration"],
                )
                _log_request_summary_from_telemetry(telemetry)

        resp_headers = _strip_encoding_headers(upstream_resp)
        return StreamingResponse(
            taste_stream_and_collect(),
            status_code=upstream_resp.status_code,
            headers=resp_headers,
            media_type="text/event-stream",
        )
    else:
        # Non-streaming — full round-trip
        t_start = time.monotonic()
        upstream_resp = await client.post(
            "/v1/messages",
            json=upstream_body,
            headers=headers,
        )
        t_end = time.monotonic()

        log.info(
            "  taste -> upstream | status=%d",
            upstream_resp.status_code,
        )

        if upstream_resp.status_code >= 400:
            log.error(
                "x upstream error | status=%d body=%s",
                upstream_resp.status_code,
                upstream_resp.text,
            )
            _dump_rejected_payload(upstream_body, upstream_resp.status_code)
            resp_headers = _strip_encoding_headers(upstream_resp)
            return Response(
                content=upstream_resp.content,
                status_code=upstream_resp.status_code,
                headers=resp_headers,
            )

        try:
            resp_data = upstream_resp.json()
        except Exception:
            log.error(
                "x non-streaming response not valid JSON | "
                "status=%d content-type=%s body=%r",
                upstream_resp.status_code,
                upstream_resp.headers.get("content-type", "?"),
                upstream_resp.text[:500] if upstream_resp.text else "(empty)",
            )
            resp_headers = _strip_encoding_headers(upstream_resp)
            return Response(
                content=upstream_resp.content or b'{"error": "empty upstream response"}',
                status_code=502,
                headers=resp_headers,
            )
        usage = resp_data.get("usage")

        # Extract text from response
        text_parts = []
        content_blocks = []
        for block in resp_data.get("content", []):
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                content_blocks.append(block)
        response_text = "\n".join(text_parts)

        timing = {
            "duration": t_end - t_start,
        }

        # Process through taste gateway — strips tensor, tracks state
        clean_text = taste_gw.process_response(
            response_text=response_text,
            session=session,
            content_blocks=content_blocks,
            usage=usage,
            request_body=upstream_body,
            timing=timing,
            feedback=feedback,
        )

        # Replace text in response with clean version (tensor stripped)
        cleaned_content = []
        clean_idx = 0
        for block in resp_data.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                if clean_idx == 0:
                    cleaned_content.append({
                        **block,
                        "text": clean_text,
                    })
                    clean_idx += 1
                # Skip additional text blocks — clean_text has everything
            else:
                cleaned_content.append(block)
        resp_data["content"] = cleaned_content

        telemetry = _build_telemetry_from_json(resp_data)
        _log_request_summary_from_telemetry(telemetry)

        resp_headers = _strip_encoding_headers(upstream_resp)
        return Response(
            content=json.dumps(resp_data),
            status_code=upstream_resp.status_code,
            headers=resp_headers,
            media_type="application/json",
        )


# --- HTTP helpers (wire-level only, no message content) ---


def _forward_headers(request: Any) -> dict[str, str]:
    """Extract auth and protocol headers from the client request."""
    headers: dict[str, str] = {}
    for key in ("authorization", "x-api-key", "x-goog-api-key", "anthropic-version",
                 "anthropic-beta", "content-type"):
        val = request.headers.get(key)
        if val:
            headers[key] = val
    return headers


def _strip_encoding_headers(resp: Any) -> dict[str, str]:
    """Strip content-encoding/length — httpx already decompressed."""
    headers = dict(resp.headers)
    headers.pop("content-encoding", None)
    headers.pop("content-length", None)
    return headers


async def _handle_error(
    upstream_resp: Any,
    upstream_body: dict[str, Any],
    upstream_bytes: bytes,
) -> Any:
    """Handle upstream error responses with logging and diagnostics."""
    from starlette.responses import Response

    error_body = b""
    async for chunk in upstream_resp.aiter_bytes():
        error_body += chunk
    await upstream_resp.aclose()

    log.error(
        "x upstream error | status=%d body=%s",
        upstream_resp.status_code,
        error_body.decode("utf-8", errors="replace"),
    )
    log.error("x response headers: %s", dict(upstream_resp.headers))

    # Best-effort diagnostics — never let a write failure swallow
    # the upstream error that the client needs to see
    try:
        _dump_rejected_payload(upstream_body, upstream_resp.status_code)
        wire_dir = Path(".tinkuy-data/rejected")
        wire_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        wire_path = wire_dir / f"wire-{upstream_resp.status_code}-{ts}.json"
        wire_path.write_bytes(upstream_bytes)
        log.error("x wire bytes written to %s (%d bytes)", wire_path, len(upstream_bytes))
    except OSError:
        log.warning("x could not write diagnostic files")

    resp_headers = _strip_encoding_headers(upstream_resp)
    return Response(
        content=error_body,
        status_code=upstream_resp.status_code,
        headers=resp_headers,
    )


def _extract_response_content(
    message: Any,
) -> tuple[str, list[dict[str, Any]]]:
    """Extract text and full content blocks from a ReconstructedMessage.

    Returns (text, content_blocks) where text is for scoring/display
    and content_blocks is the full Anthropic content array for storage.
    """
    text_parts: list[str] = []
    content_blocks: list[dict[str, Any]] = []
    for block in message.blocks:
        if block.block_type == BlockType.TEXT and block.text:
            text_parts.append(block.text)
            content_blocks.append({"type": "text", "text": block.text})
        elif block.block_type == BlockType.TOOL_USE:
            content_blocks.append({
                "type": "tool_use",
                "id": block.tool_id,
                "name": block.tool_name,
                "input": block.input_parsed or {},
            })
    return "\n".join(text_parts), content_blocks


def _build_telemetry(
    message: Any,
    request_bytes: int = 0,
    ttfb: float | None = None,
    duration: float | None = None,
) -> TurnTelemetry:
    """Build a TurnTelemetry from a ReconstructedMessage."""
    usage = message.usage
    tool_names = [
        b.tool_name for b in message.blocks
        if b.block_type == BlockType.TOOL_USE and b.tool_name
    ]
    return TurnTelemetry(
        message_id=message.id,
        model=message.model,
        stop_reason=message.stop_reason,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cache_read_tokens=usage.get("cache_read_input_tokens", 0),
        cache_create_tokens=usage.get("cache_creation_input_tokens", 0),
        text_blocks=sum(1 for b in message.blocks if b.block_type == BlockType.TEXT),
        thinking_blocks=sum(
            1 for b in message.blocks
            if b.block_type in (BlockType.THINKING, BlockType.REDACTED_THINKING)
        ),
        tool_use_blocks=sum(1 for b in message.blocks if b.block_type == BlockType.TOOL_USE),
        tool_names=tool_names,
        request_bytes=request_bytes,
        ttfb=ttfb,
        duration=duration,
    )


def _build_telemetry_from_json(resp_data: dict[str, Any]) -> TurnTelemetry:
    """Build a TurnTelemetry from a non-streaming JSON response."""
    usage = resp_data.get("usage", {})
    content = resp_data.get("content", [])
    tool_names = [
        b.get("name", "") for b in content
        if isinstance(b, dict) and b.get("type") == "tool_use"
    ]
    return TurnTelemetry(
        message_id=resp_data.get("id", ""),
        model=resp_data.get("model", ""),
        stop_reason=resp_data.get("stop_reason"),
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cache_read_tokens=usage.get("cache_read_input_tokens", 0),
        cache_create_tokens=usage.get("cache_creation_input_tokens", 0),
        text_blocks=sum(1 for b in content if isinstance(b, dict) and b.get("type") == "text"),
        thinking_blocks=sum(1 for b in content if isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking")),
        tool_use_blocks=sum(1 for b in content if isinstance(b, dict) and b.get("type") == "tool_use"),
        tool_names=tool_names,
    )


def _dump_rejected_payload(body: dict[str, Any], status: int) -> None:
    """Write the full rejected payload to a file for post-mortem."""
    dump_dir = Path(".tinkuy-data/rejected")
    dump_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    dump_path = dump_dir / f"rejected-{status}-{ts}.json"
    dump_path.write_text(json.dumps(body, indent=2, default=str))
    log.error("x rejected payload written to %s", dump_path)
    standard = {"model", "messages", "system", "tools", "max_tokens",
                "stream", "temperature", "top_p", "top_k",
                "stop_sequences", "metadata", "tool_choice"}
    extra = {k: v for k, v in body.items() if k not in standard}
    if extra:
        log.error(
            "x non-standard body fields: %s",
            json.dumps(extra, indent=2, default=str),
        )


def _log_message_structure(body: dict[str, Any]) -> None:
    """Log the shape of the upstream payload for debugging."""
    messages = body.get("messages", [])
    parts = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, str):
            length = len(content)
            kind = "text"
        elif isinstance(content, list):
            types = [b.get("type", "?") for b in content]
            length = sum(
                len(b.get("text", "")) for b in content
                if isinstance(b, dict)
            )
            kind = ",".join(types)
        else:
            length = 0
            kind = type(content).__name__
        parts.append(f"  [{i}] {role}: {kind} ({length} chars)")

    log.debug(
        "  payload structure | model=%s messages=%d system=%s\n%s",
        body.get("model", "?"),
        len(messages),
        "system" in body,
        "\n".join(parts),
    )


def _log_request_summary_from_telemetry(t: TurnTelemetry) -> None:
    """Log a one-line summary from telemetry. All numbers are tokens."""
    total_in = t.total_input_tokens
    cache_pct = t.cache_hit_rate * 100

    parts = [
        f"ok {t.input_tokens:,}+{t.cache_read_tokens:,}cache/{t.output_tokens:,}out",
        f"({total_in:,} total in)",
        f"cache:{cache_pct:.0f}%",
    ]
    if t.cache_create_tokens > 0:
        parts.append(f"write:{t.cache_create_tokens:,}")
    if t.stop_reason:
        parts.append(f"stop:{t.stop_reason}")
    if t.tool_names:
        parts.append(f"tools:[{','.join(t.tool_names[:3])}]")
    if t.ttfb is not None:
        parts.append(f"ttfb:{t.ttfb:.1f}s")
    if t.duration is not None:
        parts.append(f"total:{t.duration:.1f}s")
    if t.request_bytes > 0:
        parts.append(f"wire:{t.request_bytes:,}B")

    log.info(" | ".join(parts))


def serve(
    port: int | None = None,
    upstream: str | None = None,
    gemini_upstream: str | None = None,
    data_dir: str | None = None,
    context_limit: int = 200_000,
    taste: bool = False,
) -> None:
    """Start the tinkuy gateway server.

    Both Anthropic and Gemini routes are always active.  Upstream
    URLs resolve from explicit args > env vars > defaults.
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "Serving requires extra dependencies. "
            "Install with: pip install tinkuy[serve]"
        )

    log_level = os.environ.get("TINKUY_LOG_LEVEL", "INFO").upper()

    # Log to both stderr (console) and a persistent file.
    # Console is ephemeral — the file is the research record.
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(logging.Formatter("[tinkuy] %(message)s"))
    root_logger.addHandler(console_handler)

    if data_dir:
        log_path = Path(data_dir) / "gateway.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(message)s"
        ))
        root_logger.addHandler(file_handler)
        print(f"\n  logging to {log_path}\n", file=sys.stderr)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    if port is None:
        port = find_free_port()

    config = GatewayConfig(
        context_limit=context_limit,
        data_dir=data_dir,
        enable_console=True,
        enable_event_log=True,
    )

    app = create_app(
        gateway_config=config,
        upstream=upstream,
        gemini_upstream=gemini_upstream,
        taste=taste,
    )

    anthropic_url = _resolve_upstream("anthropic", upstream)
    gemini_url = _resolve_upstream("gemini", gemini_upstream)
    base_url = f"http://127.0.0.1:{port}"

    mode_label = "TASTE" if taste else "standard"
    print(f"\n  tinkuy gateway [{mode_label}] listening on {base_url}", file=sys.stderr)
    print(f"  anthropic upstream: {anthropic_url}", file=sys.stderr)
    print(f"  gemini upstream:    {gemini_url}", file=sys.stderr)
    print(f"\n  export ANTHROPIC_BASE_URL={base_url}", file=sys.stderr)
    print(f"  export GOOGLE_GEMINI_BASE_URL={base_url}\n", file=sys.stderr)

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
