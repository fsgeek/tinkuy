"""Streaming buffer library for SSE packet parsing and reconstruction.

Handles the full vocabulary of Anthropic SSE events with fail-stop
on unrecognized types. Data flows THROUGH the buffer — bytes in,
bytes out — with structured reconstruction as a side effect.

This module has zero dependencies on other tinkuy modules. It is
testable in complete isolation.

Components (bottom-up):
  SSELineBuffer    — bytes → complete data lines
  SSEParser        — data lines → typed SSEEvent objects
  MessageReconstructor — SSEEvent sequence → ReconstructedMessage
  StreamHandler    — pluggable observer/transformer protocol
  StreamBuffer     — the composition point (feed/finish)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol


# --- Exceptions ---


class SSEParseError(Exception):
    """Malformed SSE data."""


class UnrecognizedEventError(SSEParseError):
    """Event type not in KNOWN_TYPES. Crash so we can update the handler."""

    def __init__(self, event_type: str, raw_data: str) -> None:
        self.event_type = event_type
        self.raw_data = raw_data
        super().__init__(f"Unrecognized SSE event type: {event_type!r}")


class UnrecognizedDeltaError(SSEParseError):
    """Delta type not handled. Crash so we can update the handler."""

    def __init__(self, delta_type: str, block_type: str) -> None:
        self.delta_type = delta_type
        self.block_type = block_type
        super().__init__(
            f"Unrecognized delta type {delta_type!r} "
            f"on block type {block_type!r}"
        )


class IncompleteMessageError(SSEParseError):
    """Stream ended before message_stop."""


# --- Enums ---


class SSEEventType(Enum):
    """All SSE event types from the Anthropic streaming API."""
    MESSAGE_START = "message_start"
    CONTENT_BLOCK_START = "content_block_start"
    CONTENT_BLOCK_DELTA = "content_block_delta"
    CONTENT_BLOCK_STOP = "content_block_stop"
    MESSAGE_DELTA = "message_delta"
    MESSAGE_STOP = "message_stop"
    PING = "ping"
    ERROR = "error"


class BlockType(Enum):
    """Content block types in assistant responses."""
    TEXT = "text"
    THINKING = "thinking"
    TOOL_USE = "tool_use"
    REDACTED_THINKING = "redacted_thinking"


class DeltaType(Enum):
    """Delta types within content_block_delta events."""
    TEXT_DELTA = "text_delta"
    THINKING_DELTA = "thinking_delta"
    INPUT_JSON_DELTA = "input_json_delta"
    SIGNATURE_DELTA = "signature_delta"


# Known type strings for fail-stop validation
_KNOWN_EVENT_TYPES = frozenset(e.value for e in SSEEventType)
_KNOWN_DELTA_TYPES = frozenset(d.value for d in DeltaType)
_KNOWN_BLOCK_TYPES = frozenset(b.value for b in BlockType)


# --- Data structures ---


@dataclass(frozen=True)
class SSEEvent:
    """A parsed SSE event."""
    type: SSEEventType
    data: dict[str, Any]
    raw_line: str  # original "data: ..." content for re-emission


@dataclass
class ReconstructedBlock:
    """A content block reconstructed from SSE deltas."""
    index: int
    block_type: BlockType
    # Text blocks
    text: str = ""
    # Thinking blocks
    thinking: str = ""
    signature: str = ""
    # Redacted thinking blocks (no content, just marker)
    # Tool use blocks
    tool_id: str = ""
    tool_name: str = ""
    input_json: str = ""
    input_parsed: dict[str, Any] | None = None


@dataclass
class ReconstructedMessage:
    """A complete message reconstructed from an SSE stream."""
    id: str = ""
    model: str = ""
    role: str = ""
    stop_reason: str | None = None
    blocks: list[ReconstructedBlock] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


# --- SSELineBuffer ---


class SSELineBuffer:
    """Buffers raw bytes and yields complete SSE data lines.

    Handles TCP chunk boundaries — a single SSE event may span
    multiple chunks, or a chunk may contain multiple events.
    """

    def __init__(self) -> None:
        self._partial: bytes = b""

    def feed(self, chunk: bytes) -> list[str]:
        """Feed raw bytes, return complete data lines.

        Returns only the payload strings from "data: ..." lines.
        Discards event:/id:/comment lines (the type is redundantly
        available inside the JSON payload).
        """
        self._partial += chunk
        lines: list[str] = []

        while b"\n" in self._partial:
            line_bytes, self._partial = self._partial.split(b"\n", 1)
            line = line_bytes.decode("utf-8", errors="replace").rstrip("\r")

            if line.startswith("data: "):
                lines.append(line[6:])
            # Silently consume event:, id:, comments, blank lines

        return lines


# --- SSEParser ---


class SSEParser:
    """Parses data line payloads into typed SSEEvent objects.

    Fail-stops on unrecognized event types.
    """

    def parse(self, data_content: str) -> SSEEvent | None:
        """Parse a data line payload into an SSEEvent.

        Returns None for the [DONE] sentinel.
        Raises UnrecognizedEventError for unknown event types.
        Raises SSEParseError for malformed JSON.
        """
        if data_content == "[DONE]":
            return None

        try:
            data = json.loads(data_content)
        except json.JSONDecodeError as e:
            raise SSEParseError(
                f"Malformed JSON in SSE data: {e}"
            ) from e

        event_type_str = data.get("type", "")
        if event_type_str not in _KNOWN_EVENT_TYPES:
            raise UnrecognizedEventError(event_type_str, data_content)

        return SSEEvent(
            type=SSEEventType(event_type_str),
            data=data,
            raw_line=data_content,
        )


# --- MessageReconstructor ---


class _ReconstructorState(Enum):
    IDLE = auto()
    STREAMING = auto()
    IN_BLOCK = auto()
    COMPLETE = auto()


class MessageReconstructor:
    """State machine that accumulates SSE events into a ReconstructedMessage.

    States: IDLE → message_start → STREAMING
            STREAMING → content_block_start → IN_BLOCK
            IN_BLOCK → content_block_delta → IN_BLOCK (accumulate)
            IN_BLOCK → content_block_stop → STREAMING
            STREAMING → message_delta → STREAMING
            STREAMING → message_stop → COMPLETE
    """

    def __init__(self) -> None:
        self._state = _ReconstructorState.IDLE
        self._message = ReconstructedMessage()
        self._current_block: ReconstructedBlock | None = None

    def on_event(self, event: SSEEvent) -> None:
        """Feed a parsed SSE event. Updates internal state."""
        handler = _EVENT_HANDLERS.get(event.type)
        if handler is not None:
            handler(self, event)

    @property
    def message(self) -> ReconstructedMessage:
        return self._message

    @property
    def complete(self) -> bool:
        return self._state == _ReconstructorState.COMPLETE

    def _on_message_start(self, event: SSEEvent) -> None:
        msg_data = event.data.get("message", {})
        self._message.id = msg_data.get("id", "")
        self._message.model = msg_data.get("model", "")
        self._message.role = msg_data.get("role", "")
        usage = msg_data.get("usage")
        if usage:
            for k, v in usage.items():
                if isinstance(v, int):
                    self._message.usage[k] = (
                        self._message.usage.get(k, 0) + v
                    )
        self._state = _ReconstructorState.STREAMING

    def _on_content_block_start(self, event: SSEEvent) -> None:
        cb = event.data.get("content_block", {})
        block_type_str = cb.get("type", "")
        if block_type_str not in _KNOWN_BLOCK_TYPES:
            raise UnrecognizedDeltaError(block_type_str, "content_block_start")

        block_type = BlockType(block_type_str)
        block = ReconstructedBlock(
            index=event.data.get("index", 0),
            block_type=block_type,
        )

        # Initialize from content_block_start data
        if block_type == BlockType.TOOL_USE:
            block.tool_id = cb.get("id", "")
            block.tool_name = cb.get("name", "")
        elif block_type == BlockType.TEXT:
            block.text = cb.get("text", "")
        elif block_type == BlockType.THINKING:
            block.thinking = cb.get("thinking", "")

        self._current_block = block
        self._state = _ReconstructorState.IN_BLOCK

    def _on_content_block_delta(self, event: SSEEvent) -> None:
        if self._current_block is None:
            return

        delta = event.data.get("delta", {})
        delta_type_str = delta.get("type", "")
        if delta_type_str not in _KNOWN_DELTA_TYPES:
            raise UnrecognizedDeltaError(
                delta_type_str, self._current_block.block_type.value
            )

        delta_type = DeltaType(delta_type_str)
        block = self._current_block

        if delta_type == DeltaType.TEXT_DELTA:
            block.text += delta.get("text", "")
        elif delta_type == DeltaType.THINKING_DELTA:
            block.thinking += delta.get("thinking", "")
        elif delta_type == DeltaType.INPUT_JSON_DELTA:
            block.input_json += delta.get("partial_json", "")
        elif delta_type == DeltaType.SIGNATURE_DELTA:
            block.signature += delta.get("signature", "")

    def _on_content_block_stop(self, event: SSEEvent) -> None:
        if self._current_block is not None:
            # Parse accumulated JSON for tool_use blocks
            if (
                self._current_block.block_type == BlockType.TOOL_USE
                and self._current_block.input_json
            ):
                try:
                    self._current_block.input_parsed = json.loads(
                        self._current_block.input_json
                    )
                except json.JSONDecodeError:
                    self._current_block.input_parsed = None

            self._message.blocks.append(self._current_block)
            self._current_block = None
        self._state = _ReconstructorState.STREAMING

    def _on_message_delta(self, event: SSEEvent) -> None:
        delta = event.data.get("delta", {})
        if "stop_reason" in delta:
            self._message.stop_reason = delta["stop_reason"]
        usage = event.data.get("usage")
        if usage:
            for k, v in usage.items():
                if isinstance(v, int):
                    self._message.usage[k] = (
                        self._message.usage.get(k, 0) + v
                    )

    def _on_message_stop(self, event: SSEEvent) -> None:
        self._state = _ReconstructorState.COMPLETE


# Dispatch table for event handling
_EVENT_HANDLERS: dict[SSEEventType, Any] = {
    SSEEventType.MESSAGE_START: MessageReconstructor._on_message_start,
    SSEEventType.CONTENT_BLOCK_START: MessageReconstructor._on_content_block_start,
    SSEEventType.CONTENT_BLOCK_DELTA: MessageReconstructor._on_content_block_delta,
    SSEEventType.CONTENT_BLOCK_STOP: MessageReconstructor._on_content_block_stop,
    SSEEventType.MESSAGE_DELTA: MessageReconstructor._on_message_delta,
    SSEEventType.MESSAGE_STOP: MessageReconstructor._on_message_stop,
    # ping and error are intentionally not handled — they don't affect
    # message reconstruction. They pass through the buffer unchanged.
}


# --- StreamHandler protocol ---


class StreamHandler(Protocol):
    """Pluggable observer/transformer for SSE events.

    Handlers are chained: each sees the event (possibly modified by
    a prior handler). Return None to suppress emission downstream.
    """

    def on_event(self, event: SSEEvent) -> SSEEvent | None:
        """Observe or transform an SSE event.

        Return the event to continue emission.
        Return None to suppress (absorb) the event.
        """
        ...

    def on_complete(self, message: ReconstructedMessage) -> None:
        """Called when the message is fully reconstructed."""
        ...


# --- StreamBuffer ---


class StreamBuffer:
    """The composition point: bytes in, bytes out.

    Orchestrates line buffering, parsing, reconstruction, and handler
    chaining. The main entry point for integrating streaming into the
    server layer.

    Usage:
        buffer = StreamBuffer(handlers=[my_handler])
        for upstream_chunk in stream:
            for downstream_chunk in buffer.feed(upstream_chunk):
                yield downstream_chunk
        message = buffer.finish()
    """

    def __init__(
        self, handlers: list[StreamHandler] | None = None
    ) -> None:
        self._line_buffer = SSELineBuffer()
        self._parser = SSEParser()
        self._reconstructor = MessageReconstructor()
        self._handlers: list[StreamHandler] = list(handlers or [])

    def feed(self, chunk: bytes) -> list[bytes]:
        """Feed raw bytes from upstream. Return bytes for downstream.

        This is the main entry point. It:
        1. Buffers bytes, extracts complete data lines
        2. Parses lines into SSEEvents (fail-stop on unknown)
        3. Feeds events to the reconstructor
        4. Passes events through handlers
        5. Re-serializes surviving events to bytes
        """
        output: list[bytes] = []

        for data_content in self._line_buffer.feed(chunk):
            event = self._parser.parse(data_content)
            if event is None:
                # [DONE] sentinel — pass through
                output.append(b"data: [DONE]\n\n")
                continue

            # Feed to reconstructor (always, before handlers)
            self._reconstructor.on_event(event)

            # Pass through handler chain
            current: SSEEvent | None = event
            for handler in self._handlers:
                if current is None:
                    break
                current = handler.on_event(current)

            # Re-emit if not suppressed
            if current is not None:
                output.append(self._serialize_event(current))

        return output

    def finish(self) -> ReconstructedMessage:
        """Signal end of stream. Returns the reconstructed message.

        Calls on_complete() on all handlers.
        Raises IncompleteMessageError if message_stop was not received.
        """
        if not self._reconstructor.complete:
            raise IncompleteMessageError(
                "Stream ended before message_stop"
            )

        message = self._reconstructor.message
        for handler in self._handlers:
            handler.on_complete(message)

        return message

    @property
    def message(self) -> ReconstructedMessage:
        """Access the in-progress message (may be incomplete)."""
        return self._reconstructor.message

    @property
    def complete(self) -> bool:
        return self._reconstructor.complete

    @staticmethod
    def _serialize_event(event: SSEEvent) -> bytes:
        """Re-serialize an SSEEvent to SSE wire format."""
        # Re-encode from the event's data dict so handler modifications
        # are reflected in the output bytes
        json_str = json.dumps(event.data, separators=(",", ":"))
        return f"event: {event.type.value}\ndata: {json_str}\n\n".encode(
            "utf-8"
        )


# --- Fixture builders (for testing) ---


def build_sse_stream(events: list[dict[str, Any]]) -> bytes:
    """Build a complete SSE byte stream from a list of event dicts.

    Each dict must have a "type" key. The dict is serialized as the
    data payload. A [DONE] sentinel is appended.

    This is a test utility — it produces the exact byte format that
    the Anthropic API sends.
    """
    parts: list[bytes] = []
    for event_data in events:
        event_type = event_data.get("type", "unknown")
        json_str = json.dumps(event_data, separators=(",", ":"))
        parts.append(
            f"event: {event_type}\ndata: {json_str}\n\n".encode("utf-8")
        )
    parts.append(b"data: [DONE]\n\n")
    return b"".join(parts)
