"""Harness — drives the gateway in a live Claude Code session.

The harness is the outermost layer. It connects tinkuy to an actual
message stream — reading input, calling the gateway, forwarding to
the API, and delivering responses.

This module provides:

1. SessionHarness — drives a gateway through a turn-based session,
   extracting cooperative memory signals from model responses and
   feeding them back through the gateway.

2. Signal extraction — parses yuyay-response blocks from model
   output to identify release/retain/recall signals.

3. Message stream protocol — the interface a frontend must implement
   to plug into the harness.

The harness owns the API call. Auth credentials come from the
frontend (environment, config, etc.) and are passed to the API
client opaquely. The gateway never touches credentials.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger("tinkuy.harness")

from tinkuy.gateway._gateway import Gateway, GatewayConfig, TurnResult
from tinkuy.core.orchestrator import TurnRecord


# --- Signal extraction ---


_RELEASE_PATTERN = re.compile(
    r'<release\s+handle="([^"]+)"'
    r'(?:\s+losses="([^"]*)")?'
    r'\s*/?>',
    re.DOTALL,
)

_RETAIN_PATTERN = re.compile(
    r'<retain\s+handle="([^"]+)"'
    r'(?:\s+reason="([^"]*)")?'
    r'\s*/?>',
    re.DOTALL,
)

_RECALL_PATTERN = re.compile(
    r'<recall\s+handle="([^"]+)"\s*/?>',
    re.DOTALL,
)

_YUYAY_RESPONSE_PATTERN = re.compile(
    r'<yuyay-response>(.*?)</yuyay-response>',
    re.DOTALL,
)

_TENSOR_CONTENT_PATTERN = re.compile(
    r'<tensor\s+handle="([^"]+)"[^>]*>(.*?)</tensor>',
    re.DOTALL,
)

_DECLARE_PATTERN = re.compile(
    r'<declare\s+handle="([^"]+)"[^>]*>(.*?)</declare>',
    re.DOTALL,
)

_DEPENDS_ON_PATTERN = re.compile(
    r'<depends-on\s+handle="([^"]+)"\s*/?>',
)


def extract_signals(response_text: str) -> list[dict[str, Any]]:
    """Extract cooperative memory signals from a model response.

    Parses yuyay-response blocks to find release/retain/recall
    directives. Returns a list of signal dicts suitable for
    Gateway.ingest_response().
    """
    signals: list[dict[str, Any]] = []

    # Find yuyay-response blocks
    for block_match in _YUYAY_RESPONSE_PATTERN.finditer(response_text):
        block = block_match.group(1)

        # Release signals
        for m in _RELEASE_PATTERN.finditer(block):
            handle = m.group(1)
            losses = m.group(2) or ""

            # Look for associated tensor content
            tensor_content = None
            for tm in _TENSOR_CONTENT_PATTERN.finditer(response_text):
                if tm.group(1) == handle:
                    tensor_content = tm.group(2).strip()
                    break

            signals.append({
                "type": "release",
                "handle": handle,
                "tensor_content": tensor_content,
                "declared_losses": losses if losses else None,
            })

        # Retain signals
        for m in _RETAIN_PATTERN.finditer(block):
            signals.append({
                "type": "retain",
                "handle": m.group(1),
            })

        # Recall signals
        for m in _RECALL_PATTERN.finditer(block):
            signals.append({
                "type": "recall",
                "handle": m.group(1),
            })

        # Declare signals — dependency edges (immutable once emitted)
        for m in _DECLARE_PATTERN.finditer(block):
            handle = m.group(1)
            body = m.group(2)
            depends_on = _DEPENDS_ON_PATTERN.findall(body)
            if depends_on:
                signals.append({
                    "type": "declare",
                    "handle": handle,
                    "depends_on": depends_on,
                })

    return signals


def strip_signals(response_text: str) -> str:
    """Remove yuyay-response blocks from response text.

    The model's cooperative memory signals are metadata, not
    conversation content. Strip them before storing the response
    as conversation content in the projection.
    """
    text = _YUYAY_RESPONSE_PATTERN.sub("", response_text)
    text = _TENSOR_CONTENT_PATTERN.sub("", text)
    return text.strip()


# --- Frontend protocol ---


class MessageStream(Protocol):
    """Protocol for a frontend message stream.

    A frontend (Claude Code, HTTP server, CLI) implements this to
    plug into the harness. The harness calls these methods to
    receive input and deliver output.
    """

    @property
    def session_id(self) -> str | None:
        """Return the session ID, or None if unknown.

        The adapter is responsible for obtaining this — from an API
        header, the Claude Code session index, or wherever. The
        gateway uses it to key per-session checkpoints.
        """
        ...

    def receive(self) -> str | None:
        """Receive the next user message, or None to end the session."""
        ...

    def deliver(self, content: str, metadata: dict[str, Any]) -> None:
        """Deliver assistant response content to the frontend."""
        ...

    def deliver_status(self, status: str) -> None:
        """Deliver a status update (pressure, evictions, etc.)."""
        ...


# --- API client protocol ---


class APIClient(Protocol):
    """Protocol for an API client.

    The harness calls this to send the synthesized payload to the
    model API. The implementation owns authentication — the harness
    and gateway never touch credentials.
    """

    def send(self, payload: dict[str, Any]) -> str:
        """Send a payload and return the response text.

        The implementation is responsible for:
          - Attaching credentials (from its own config/env)
          - Making the HTTP call
          - Extracting the text response
          - Handling errors/retries
        """
        ...


# --- Session harness ---


@dataclass
class SessionConfig:
    """Configuration for a session."""
    gateway_config: GatewayConfig = field(default_factory=GatewayConfig)
    rehydrate_source: str | None = None  # path to conversation log


class SessionHarness:
    """Drives a gateway through a turn-based session.

    The harness is the outer loop:
      1. Receive user message from frontend
      2. Call gateway.process_turn() to get API payload
      3. Call API client to get response
      4. Extract cooperative memory signals
      5. Strip signals and pass clean response to gateway
      6. Deliver response to frontend
      7. Repeat

    The harness owns the API call. The gateway never talks to the API.
    """

    def __init__(
        self,
        frontend: MessageStream,
        api_client: APIClient,
        config: SessionConfig | None = None,
    ) -> None:
        self.frontend = frontend
        self.api_client = api_client
        self.config = config or SessionConfig()
        self._gateway: Gateway | None = None

    @property
    def gateway(self) -> Gateway:
        if self._gateway is None:
            raise RuntimeError("Session not started. Call start() first.")
        return self._gateway

    def start(self) -> None:
        """Initialize the gateway, optionally resuming or rehydrating."""
        # Plumb session ID from frontend into gateway config
        session_id = self.frontend.session_id
        if session_id and not self.config.gateway_config.session_id:
            self.config.gateway_config.session_id = session_id

        # Try to resume from checkpoint
        gw = Gateway.resume(self.config.gateway_config)

        if gw is not None:
            log.info("session resumed from checkpoint (turn %d)",
                     gw.orchestrator.projection.turn)
        else:
            # Fresh start
            gw = Gateway(self.config.gateway_config)
            log.info("session started fresh")

            # Rehydrate if source provided
            if self.config.rehydrate_source:
                gw.rehydrate(self.config.rehydrate_source)
                log.info("rehydrated from source")

        self._gateway = gw

    def run(self) -> None:
        """Run the session loop until the frontend ends it."""
        self.start()

        while True:
            # Receive user input
            user_message = self.frontend.receive()
            if user_message is None:
                break

            # Process turn
            turn_result = self.gateway.process_turn(user_message)

            # Report status
            self.frontend.deliver_status(
                f"Turn {self.gateway.turn} | "
                f"Pressure: {turn_result.pressure_zone.name} | "
                f"Pending evictions: {len(turn_result.pending_evictions)}"
            )

            # Call API
            response_text = self.api_client.send(turn_result.api_payload)

            # Extract signals before stripping
            signals = extract_signals(response_text)

            # Strip signals from response content
            clean_response = strip_signals(response_text)

            # Ingest response with signals
            self.gateway.ingest_response(
                content=clean_response,
                signals=signals if signals else None,
            )

            # Deliver clean response to frontend
            self.frontend.deliver(
                content=clean_response,
                metadata={
                    "turn": self.gateway.turn,
                    "pressure_zone": self.gateway.pressure_zone.name,
                },
            )

    def step(self, user_message: str) -> tuple[str, TurnResult]:
        """Process a single turn without the loop.

        Useful for testing and for frontends that manage their own
        event loop. Returns (clean_response, turn_result).
        """
        if self._gateway is None:
            self.start()

        turn_result = self.gateway.process_turn(user_message)
        response_text = self.api_client.send(turn_result.api_payload)

        signals = extract_signals(response_text)
        clean_response = strip_signals(response_text)

        self.gateway.ingest_response(
            content=clean_response,
            signals=signals if signals else None,
        )

        return clean_response, turn_result
