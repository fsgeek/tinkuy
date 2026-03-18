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

from tinkuy.gateway import Gateway, GatewayConfig

log = logging.getLogger("tinkuy.server")


def find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def create_app(
    gateway_config: GatewayConfig | None = None,
    upstream: str = "https://api.anthropic.com",
) -> Any:
    """Create the FastAPI application.

    Import is deferred so the rest of tinkuy works without
    fastapi/uvicorn installed.
    """
    try:
        from fastapi import FastAPI, Request, Response
        import httpx
    except ImportError:
        raise ImportError(
            "FastAPI server requires extra dependencies. "
            "Install with: pip install tinkuy[serve]"
        )

    config = gateway_config or GatewayConfig(enable_console=True)

    # Per-session gateways, keyed by session ID or a default
    gateways: dict[str, Gateway] = {}

    def get_gateway(session_id: Optional[str]) -> Gateway:
        key = session_id or "__default__"
        if key not in gateways:
            cfg = GatewayConfig(
                context_limit=config.context_limit,
                data_dir=config.data_dir,
                session_id=session_id,
                enable_console=config.enable_console,
                enable_event_log=config.enable_event_log,
            )
            gw = Gateway.resume(cfg)
            if gw is None:
                gw = Gateway(cfg)
            gateways[key] = gw
        return gateways[key]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        transport = httpx.AsyncHTTPTransport()

        async def log_request(request: httpx.Request):
            log.info(
                "  WIRE -> %s %s (%d bytes)",
                request.method,
                request.url,
                len(request.content) if request.content else 0,
            )

        app.state.client = httpx.AsyncClient(
            base_url=upstream,
            timeout=httpx.Timeout(300.0, connect=10.0),
            transport=transport,
            event_hooks={"request": [log_request]},
        )
        yield
        await app.state.client.aclose()

    app = FastAPI(
        title="tinkuy",
        description="Projective gateway for transformer memory hierarchy",
        lifespan=lifespan,
    )

    @app.post("/v1/messages")
    async def messages(request: Request) -> Response:
        """The gateway path. No exceptions. No fallback. No bypass."""
        from starlette.responses import StreamingResponse

        body = await request.json()
        is_streaming = body.get("stream", False)
        session_id = request.headers.get("x-tinkuy-session")
        gw = get_gateway(session_id)

        # Gateway owns the transformation — one call, no branching
        upstream_body = gw.prepare_request(body)

        # Forward auth headers — we never touch credentials
        headers = _forward_headers(request)

        _log_message_structure(upstream_body)
        log.info(
            "  headers | version=%s beta=%s",
            headers.get("anthropic-version", "MISSING"),
            headers.get("anthropic-beta", "MISSING"),
        )

        client: httpx.AsyncClient = request.app.state.client

        if is_streaming:
            upstream_bytes = json.dumps(upstream_body).encode("utf-8")
            headers["content-type"] = "application/json"
            upstream_req = client.build_request(
                "POST", "/v1/messages",
                content=upstream_bytes, headers=headers,
            )
            upstream_resp = await client.send(upstream_req, stream=True)

            log.info("-> upstream | status=%d (stream)", upstream_resp.status_code)

            if upstream_resp.status_code >= 400:
                return await _handle_error(
                    upstream_resp, upstream_body, upstream_bytes
                )

            collected_text: list[str] = []
            collected_usage: dict[str, int] = {}

            async def stream_and_collect():
                async for chunk in upstream_resp.aiter_bytes():
                    _collect_sse_text(chunk, collected_text, collected_usage)
                    yield chunk
                await upstream_resp.aclose()

                # Post-stream: gateway ingests the response
                full_text = "".join(collected_text)
                if full_text:
                    gw.ingest_raw_response(full_text)
                _log_request_summary(collected_usage, len(full_text))

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
                    upstream_resp.text[:2000],
                )
                _dump_rejected_payload(upstream_body, upstream_resp.status_code)
                return Response(
                    content=upstream_resp.content,
                    status_code=upstream_resp.status_code,
                    headers=resp_headers,
                )

            resp_data = upstream_resp.json()
            gw.ingest_response_json(resp_data)
            _log_request_summary(
                resp_data.get("usage", {}),
                sum(
                    len(b.get("text", ""))
                    for b in resp_data.get("content", [])
                    if isinstance(b, dict) and b.get("type") == "text"
                ),
            )

            return Response(
                content=upstream_resp.content,
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


# --- HTTP helpers (wire-level only, no message content) ---


def _forward_headers(request: Any) -> dict[str, str]:
    """Extract auth and protocol headers from the client request."""
    headers: dict[str, str] = {}
    for key in ("authorization", "x-api-key", "anthropic-version",
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
        error_body.decode("utf-8", errors="replace")[:2000],
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


def _collect_sse_text(
    chunk: bytes,
    collected: list[str],
    usage: dict[str, int] | None = None,
) -> None:
    """Extract text content and usage from SSE chunks."""
    for line in chunk.decode("utf-8", errors="replace").split("\n"):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            continue
        try:
            event = json.loads(data)
            event_type = event.get("type", "")
            if event_type == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    collected.append(delta.get("text", ""))
            elif usage is not None:
                msg_usage = None
                if event_type == "message_start":
                    msg_usage = event.get("message", {}).get("usage")
                elif event_type == "message_delta":
                    msg_usage = event.get("usage")
                if msg_usage:
                    for k, v in msg_usage.items():
                        if isinstance(v, int):
                            usage[k] = usage.get(k, 0) + v
        except json.JSONDecodeError:
            pass


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
            json.dumps(extra, indent=2, default=str)[:1000],
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


def _log_request_summary(usage: dict[str, int], response_len: int) -> None:
    """Log a one-line summary with cache utilization after each request."""
    input_tok = usage.get("input_tokens", 0)
    output_tok = usage.get("output_tokens", 0)
    cache_read = usage.get("cache_read_input_tokens", 0)
    cache_create = usage.get("cache_creation_input_tokens", 0)

    cache_eligible = input_tok + cache_read + cache_create
    cache_pct = (cache_read / cache_eligible * 100) if cache_eligible > 0 else 0.0

    parts = [
        f"ok {input_tok:,}in/{output_tok:,}out",
        f"cache:{cache_pct:.0f}%",
    ]
    if cache_create > 0:
        parts.append(f"write:{cache_create:,}")
    if cache_read > 0:
        parts.append(f"read:{cache_read:,}")
    if response_len > 0:
        parts.append(f"resp:{response_len:,}ch")

    log.info(" | ".join(parts))


def serve(
    port: int | None = None,
    upstream: str = "https://api.anthropic.com",
    data_dir: str | None = None,
    context_limit: int = 200_000,
) -> None:
    """Start the tinkuy gateway server.

    Prints the ANTHROPIC_BASE_URL for copy-paste convenience.
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "Serving requires extra dependencies. "
            "Install with: pip install tinkuy[serve]"
        )

    log_level = os.environ.get("TINKUY_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        format="[tinkuy] %(message)s",
        level=getattr(logging, log_level, logging.INFO),
        stream=sys.stderr,
    )
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

    app = create_app(gateway_config=config, upstream=upstream)

    base_url = f"http://127.0.0.1:{port}"
    print(f"\n  tinkuy gateway listening on {base_url}", file=sys.stderr)
    print(f"\n  export ANTHROPIC_BASE_URL={base_url}\n", file=sys.stderr)

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
