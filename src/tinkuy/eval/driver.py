"""Conversation driver — runs a task through Tinkuy and the Anthropic API.

The driver owns the conversation loop. It sends user messages, receives
assistant responses, and feeds them through the gateway's projection
engine. No HTTP server, no Claude Code, no client protocol noise.

Usage:
    driver = ConversationDriver(
        gateway=Gateway(GatewayConfig(context_limit=200_000)),
        model="claude-haiku-4-5-20251001",
    )
    transcript = await driver.run(task)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from tinkuy.gateway import Gateway, GatewayConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task protocol — what we ask the model to do
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """A single evaluation task.

    A task is a sequence of user messages that drive a conversation.
    The simplest task is one message; a multi-turn task provides
    follow-ups that depend on the model's prior responses.
    """

    name: str
    system: str = ""
    messages: list[str] = field(default_factory=list)
    max_turns: int = 1

    # If set, messages[i] can be a format string using {response} for
    # the prior assistant response. This enables dependent follow-ups.
    dependent: bool = False


# ---------------------------------------------------------------------------
# Transcript — what happened
# ---------------------------------------------------------------------------

@dataclass
class TurnRecord:
    """One turn of conversation as observed by the harness."""

    turn: int
    user_content: str
    assistant_content: str
    api_payload: dict[str, Any]  # exact payload sent to API
    api_response: dict[str, Any]  # exact response from API
    pressure_zone: str
    total_tokens_projection: int
    total_tokens_api: int  # input_tokens from API usage
    latency_s: float
    signals: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Transcript:
    """Complete record of a task execution."""

    task_name: str
    model: str
    mode: str  # "baseline" | "projection" | "projection+eviction" | etc.
    turns: list[TurnRecord] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

class ConversationDriver:
    """Drives a conversation through the gateway and the Anthropic API.

    The driver is the evaluation harness's core loop:
        user_text → gateway.process_turn() → API call → gateway.ingest_response()

    It captures everything needed for post-hoc analysis: the exact
    payloads sent, the exact responses received, pressure readings,
    token counts, and timing.
    """

    def __init__(
        self,
        gateway: Gateway,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 64000,
    ) -> None:
        self.gateway = gateway
        self.model = model
        self.max_tokens = max_tokens
        self._client: Any = None  # lazy init

    def _get_client(self) -> Any:
        """Lazy-init the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "Evaluation requires the anthropic package. "
                    "Install with: uv pip install anthropic"
                )
            self._client = anthropic.Anthropic()
        return self._client

    async def run(self, task: Task, mode: str = "full") -> Transcript:
        """Execute a task and return the complete transcript.

        Modes control which gateway features are active:
            "baseline"       — bypass gateway, send raw messages to API
            "full"           — full pipeline including page table
            "no_page_table"  — projection but page table stripped from system
            "no_padding"     — projection but [conversation start] padding stripped
            "no_meta"        — projection with neither page table nor padding
        """
        transcript = Transcript(
            task_name=task.name,
            model=self.model,
            mode=mode,
            config={
                "context_limit": self.gateway.config.context_limit
                if hasattr(self.gateway, "config") else 200_000,
                "max_tokens": self.max_tokens,
            },
        )

        last_response = ""

        # Inject task system prompt into gateway's system region
        if task.system and mode != "baseline":
            from tinkuy.core.orchestrator import EventType, InboundEvent
            self.gateway.orchestrator.begin_turn([
                InboundEvent(
                    type=EventType.SYSTEM_UPDATE,
                    content=task.system,
                    label="task system prompt",
                )
            ])

        for i, user_text in enumerate(task.messages[: task.max_turns]):
            # Allow dependent follow-ups
            if task.dependent and i > 0 and "{response}" in user_text:
                user_text = user_text.format(response=last_response)

            if mode == "baseline":
                record = await self._turn_baseline(
                    i + 1, user_text, task.system, transcript.turns
                )
            else:
                record = await self._turn_gateway(i + 1, user_text, mode=mode)

            transcript.turns.append(record)
            last_response = record.assistant_content

            log.info(
                "turn %d | pressure=%s proj_tok=%d api_tok=%d latency=%.1fs",
                record.turn,
                record.pressure_zone,
                record.total_tokens_projection,
                record.total_tokens_api,
                record.latency_s,
            )

        return transcript

    async def _turn_gateway(
        self, turn: int, user_text: str, mode: str = "full"
    ) -> TurnRecord:
        """Execute one turn through the gateway pipeline."""
        # Gateway synthesizes the payload
        turn_result = self.gateway.process_turn(
            user_content=user_text,
            tool_results=[],
        )

        api_payload = turn_result.api_payload

        # Ablations — strip components to isolate their effect
        strip_page_table = mode in ("no_page_table", "no_meta")
        strip_padding = mode in ("no_padding", "no_meta")

        if strip_page_table and "system" in api_payload:
            api_payload["system"] = [
                part for part in api_payload["system"]
                if not isinstance(part, dict)
                or "<yuyay-page-table>" not in part.get("text", "")
            ]
            if not api_payload["system"]:
                del api_payload["system"]

        if strip_padding and api_payload.get("messages"):
            msgs = api_payload["messages"]
            if (msgs and isinstance(msgs[0].get("content"), str)
                    and msgs[0]["content"] == "[conversation start]"):
                api_payload["messages"] = msgs[1:]
        api_payload["model"] = self.model
        api_payload["max_tokens"] = self.max_tokens

        # Call the API
        t0 = time.monotonic()
        response_data = self._call_api(api_payload)
        latency = time.monotonic() - t0

        # Extract content
        text, content_blocks = _extract_content(response_data)
        usage = response_data.get("usage", {})

        # Feed response back to gateway
        from tinkuy.harness import extract_signals, strip_signals
        signals = extract_signals(text)
        clean_text = strip_signals(text)
        self.gateway.ingest_response(
            content=clean_text,
            signals=signals,
            content_blocks=content_blocks,
        )

        return TurnRecord(
            turn=turn,
            user_content=user_text,
            assistant_content=clean_text,
            api_payload=api_payload,
            api_response=response_data,
            pressure_zone=turn_result.pressure_zone.name
            if hasattr(turn_result.pressure_zone, "name")
            else str(turn_result.pressure_zone),
            total_tokens_projection=self.gateway.projection.total_tokens,
            total_tokens_api=usage.get("input_tokens", 0),
            latency_s=latency,
            signals=signals,
        )

    async def _turn_baseline(
        self,
        turn: int,
        user_text: str,
        system: str,
        prior_turns: list[TurnRecord],
    ) -> TurnRecord:
        """Execute one turn as a raw API call — no gateway."""
        messages = []
        for t in prior_turns:
            messages.append({"role": "user", "content": t.user_content})
            messages.append({"role": "assistant", "content": t.assistant_content})
        messages.append({"role": "user", "content": user_text})

        api_payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if system:
            api_payload["system"] = system

        t0 = time.monotonic()
        response_data = self._call_api(api_payload)
        latency = time.monotonic() - t0

        text, _ = _extract_content(response_data)
        usage = response_data.get("usage", {})

        return TurnRecord(
            turn=turn,
            user_content=user_text,
            assistant_content=text,
            api_payload=api_payload,
            api_response=response_data,
            pressure_zone="N/A",
            total_tokens_projection=0,
            total_tokens_api=usage.get("input_tokens", 0),
            latency_s=latency,
        )

    def _call_api(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Synchronous API call with streaming. Returns the raw response dict."""
        client = self._get_client()
        with client.messages.stream(**payload) as stream:
            response = stream.get_final_message()
        return response.model_dump()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_content(
    response_data: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Extract text and content blocks from an API response."""
    content = response_data.get("content", [])
    text_parts: list[str] = []
    blocks: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        blocks.append(block)
    return "\n".join(text_parts), blocks
