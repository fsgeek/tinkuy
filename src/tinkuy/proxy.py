"""FastAPI proxy — the HTTP layer that makes tinkuy a running gateway.

Intercepts /v1/messages requests, routes them through the gateway,
forwards to the upstream Anthropic API, and returns the response.

The proxy does NOT hold credentials. It forwards the client's
Authorization header to the upstream API. The gateway prepares
the payload; the proxy delivers it.

Usage:
    tinkuy serve --port 8340 --upstream https://api.anthropic.com

Then point your client at:
    export ANTHROPIC_BASE_URL=http://127.0.0.1:8340
"""

from __future__ import annotations

import json
import os
import socket
import sys
from contextlib import asynccontextmanager
from typing import Any

from tinkuy.gateway import Gateway, GatewayConfig
from tinkuy.harness import extract_signals, strip_signals


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
            "FastAPI proxy requires extra dependencies. "
            "Install with: pip install tinkuy[serve]"
        )

    config = gateway_config or GatewayConfig(enable_console=True)

    # Per-session gateways, keyed by session ID or a default
    gateways: dict[str, Gateway] = {}

    def get_gateway(session_id: str | None) -> Gateway:
        key = session_id or "__default__"
        if key not in gateways:
            cfg = GatewayConfig(
                context_limit=config.context_limit,
                data_dir=config.data_dir,
                session_id=session_id,
                enable_console=config.enable_console,
                enable_event_log=config.enable_event_log,
            )
            # Try to resume from checkpoint
            gw = Gateway.resume(cfg)
            if gw is None:
                gw = Gateway(cfg)
            gateways[key] = gw
        return gateways[key]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(
            base_url=upstream,
            timeout=httpx.Timeout(300.0, connect=10.0),
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
        """Intercept /v1/messages — the main gateway path."""
        body = await request.json()

        # Extract session ID from header if provided
        session_id = request.headers.get("x-tinkuy-session")

        gw = get_gateway(session_id)

        # Extract user content from the request
        request_messages = body.get("messages", [])
        user_content, tool_results = _extract_user_content(
            request_messages
        )

        if user_content:
            # Process through the gateway
            turn_result = gw.process_turn(
                user_content=user_content,
                tool_results=tool_results,
            )

            # Apply eviction decisions
            gw.orchestrator.apply_decisions(
                turn_result.record.eviction_decisions
            )

            # Use the gateway's synthesized payload
            synth = turn_result.api_payload
        else:
            # No user content to process — pass through
            # (shouldn't happen in normal flow, but be safe)
            synth = {"messages": request_messages}
            if "system" in body:
                synth["system"] = body["system"]

        # Build the upstream request, preserving non-message fields
        upstream_body = {**body}
        upstream_body["messages"] = synth.get("messages", [])
        if "system" in synth:
            upstream_body["system"] = synth["system"]

        # Forward auth header from client — we don't touch credentials
        headers = {}
        for key in ("authorization", "x-api-key", "anthropic-version",
                     "anthropic-beta", "content-type"):
            val = request.headers.get(key)
            if val:
                headers[key] = val

        # Forward to upstream
        client: httpx.AsyncClient = request.app.state.client
        upstream_resp = await client.post(
            "/v1/messages",
            json=upstream_body,
            headers=headers,
        )

        # Parse upstream response
        resp_data = upstream_resp.json()

        # Extract response text
        response_text = _extract_response_text(resp_data)

        if response_text:
            # Extract cooperative memory signals
            signals = extract_signals(response_text)
            clean_response = strip_signals(response_text)

            # Ingest response into the gateway
            gw.ingest_response(
                content=clean_response,
                signals=signals if signals else None,
            )

        # Return the upstream response unmodified to the client
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers=dict(upstream_resp.headers),
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


def _extract_user_content(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, str]] | None]:
    """Extract user content and tool results from the last turn."""
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


def _extract_response_text(resp_data: dict[str, Any]) -> str:
    """Extract text content from an Anthropic API response."""
    content = resp_data.get("content", [])
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)


def serve(
    port: int | None = None,
    upstream: str = "https://api.anthropic.com",
    data_dir: str | None = None,
    context_limit: int = 200_000,
) -> None:
    """Start the tinkuy proxy server.

    Prints the ANTHROPIC_BASE_URL for copy-paste convenience.
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "Serving requires extra dependencies. "
            "Install with: pip install tinkuy[serve]"
        )

    if port is None:
        port = find_free_port()

    config = GatewayConfig(
        context_limit=context_limit,
        data_dir=data_dir,
        enable_console=True,
        enable_event_log=True,
    )

    app = create_app(gateway_config=config, upstream=upstream)

    # Print the convenience line — the one that got yanked from Pichay
    base_url = f"http://127.0.0.1:{port}"
    print(f"\n  tinkuy gateway listening on {base_url}", file=sys.stderr)
    print(f"\n  export ANTHROPIC_BASE_URL={base_url}\n", file=sys.stderr)

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
