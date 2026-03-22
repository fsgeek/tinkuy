"""Tests for Gemini format adapters and gateway integration."""

from __future__ import annotations

import json

from tinkuy.formats.gemini import (
    GeminiInboundAdapter,
    GeminiLiveAdapter,
    GeminiResponseIngester,
)
from tinkuy.gateway import APIFormat, Gateway
from tinkuy.orchestrator import (
    EventType,
    InboundEvent,
    Orchestrator,
    ResponseSignalType,
)
from tinkuy.regions import ContentKind, ContentStatus, Projection, RegionID


def test_gemini_live_adapter_synthesizes_request_from_projection_r0_to_r4():
    projection = Projection()

    # R0 (tools): currently ignored by GeminiLiveAdapter._collect_tools.
    projection.add_content(
        content="tool schema",
        kind=ContentKind.SYSTEM,
        label="tool def",
        region=RegionID.TOOLS,
    )

    # R1 (system): only PRESENT blocks should be emitted.
    projection.add_content(
        content="System policy",
        kind=ContentKind.SYSTEM,
        label="system",
        region=RegionID.SYSTEM,
    )
    stale_system = projection.add_content(
        content="stale policy",
        kind=ContentKind.SYSTEM,
        label="stale",
        region=RegionID.SYSTEM,
    )
    stale_system.status = ContentStatus.AVAILABLE

    # R2 (durable): tensor should become marker text, plus function_call block.
    durable_tensor = projection.add_content(
        content="compressed summary",
        kind=ContentKind.TENSOR,
        label="durable tensor",
        region=RegionID.DURABLE,
    )
    durable_tensor.status = ContentStatus.AVAILABLE
    projection.add_content(
        content="",
        kind=ContentKind.CONVERSATION,
        label="assistant tool-call",
        region=RegionID.DURABLE,
        function_call={"name": "search", "args": {"q": "tinkuy"}},
    )

    # R3 (ephemeral): tool result + user text should coalesce as role=user.
    projection.add_content(
        content="tool output",
        kind=ContentKind.TOOL_RESULT,
        label="tool_result",
        region=RegionID.EPHEMERAL,
        tool_name="shell",
    )
    projection.add_content(
        content="user follow-up",
        kind=ContentKind.CONVERSATION,
        label="user",
        region=RegionID.EPHEMERAL,
    )

    # R4 (current): assistant content should become role=model.
    projection.add_content(
        content="assistant final",
        kind=ContentKind.CONVERSATION,
        label="assistant",
        region=RegionID.CURRENT,
    )

    payload = GeminiLiveAdapter(Orchestrator(projection=projection)).synthesize_request()

    assert payload["system_instruction"] == {"parts": [{"text": "System policy"}]}
    assert "tools" not in payload
    assert payload["contents"] == [
        {
            "role": "model",
            "parts": [
                {"text": f"[tensor:{durable_tensor.handle[:8]} — durable tensor]"},
                {"function_call": {"name": "search", "args": {"q": "tinkuy"}}},
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "shell",
                        "response": {"result": "tool output"},
                    }
                },
                {"text": "user follow-up"},
            ],
        },
        {"role": "model", "parts": [{"text": "assistant final"}]},
    ]


def test_gemini_live_adapter_omits_optional_fields_when_no_system_or_tools():
    payload = GeminiLiveAdapter(Orchestrator(projection=Projection())).synthesize_request()

    assert payload == {"contents": []}


def test_gemini_inbound_adapter_parses_last_user_turn_into_events():
    body = {
        "contents": [
            {"role": "user", "parts": [{"text": "older turn"}]},
            {"role": "model", "parts": [{"text": "assistant turn"}]},
            {
                "role": "user",
                "parts": [
                    {"text": "latest user text"},
                    {"function_response": {"name": "calc", "response": {"value": 3}}},
                    {"function_response": {"response": {"ok": True}}},
                ],
            },
        ]
    }

    events = GeminiInboundAdapter().parse_request(body)

    assert len(events) == 3
    assert events[0].type == EventType.USER_MESSAGE
    assert events[0].content == "latest user text"
    assert events[0].label == "user"

    assert events[1].type == EventType.TOOL_RESULT
    assert events[1].label == "calc"
    assert events[1].metadata == {"tool_name": "calc"}
    assert json.loads(events[1].content) == {"value": 3}

    assert events[2].type == EventType.TOOL_RESULT
    assert events[2].label == "tool"
    assert events[2].metadata == {"tool_name": "tool"}
    assert json.loads(events[2].content) == {"ok": True}


def test_gemini_inbound_adapter_ignores_when_last_turn_is_not_user():
    body = {
        "contents": [
            {"role": "user", "parts": [{"text": "hello"}]},
            {"role": "model", "parts": [{"text": "response"}]},
        ]
    }

    assert GeminiInboundAdapter().parse_request(body) == []


def test_gemini_response_ingester_ingests_content_and_extracts_signals():
    orch = Orchestrator()
    turn = orch.begin_turn(
        [InboundEvent(type=EventType.USER_MESSAGE, content="remember this", label="user")]
    )
    handle = turn.inbound_handles[0]

    # Put block into pending state so RETAIN has a meaningful target before release/recall flow.
    user_block = orch.projection.region(RegionID.CURRENT).find(handle)
    assert user_block is not None
    user_block.status = ContentStatus.PENDING_REMOVAL
    orch.projection.region(RegionID.CURRENT).nominate_removal(
        handle=handle,
        source="test",
        reason="exercise retain",
    )

    response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": (
                                f'Visible answer\n'
                                f'<yuyay-response>'
                                f'<retain handle="{handle}" />'
                                f'<release handle="{handle}" losses="trimmed" />'
                                f'<recall handle="{handle}" />'
                                f'</yuyay-response>\n'
                                f'<tensor handle="{handle}">short tensor</tensor>'
                            )
                        },
                        {"function_call": {"name": "search", "args": {"q": "tinkuy"}}},
                    ]
                }
            }
        ]
    }

    record = GeminiResponseIngester(orch).ingest_response(response)
    assert record is not None
    assert record.response_handle is not None

    signal_types = {s.type for s in record.signals_processed}
    assert ResponseSignalType.RETAIN in signal_types
    assert ResponseSignalType.RELEASE in signal_types
    assert ResponseSignalType.RECALL in signal_types

    response_block = orch.projection.region(RegionID.EPHEMERAL).find(record.response_handle)
    assert response_block is not None
    assert response_block.content == "Visible answer"

    content_blocks = response_block.metadata["content_blocks"]
    assert content_blocks[0]["type"] == "text"
    assert content_blocks[0]["text"].startswith("Visible answer")
    assert content_blocks[1] == {
        "type": "tool_use",
        "name": "search",
        "input": {"q": "tinkuy"},
        "metadata": {"function_call": {"name": "search", "args": {"q": "tinkuy"}}},
    }

    updated_user_block = orch.projection.region(RegionID.CURRENT).find(handle)
    assert updated_user_block is not None
    assert updated_user_block.status == ContentStatus.PRESENT
    assert updated_user_block.tensor_handle is not None


def test_gemini_response_ingester_returns_none_without_candidates():
    assert GeminiResponseIngester(Orchestrator()).ingest_response({}) is None


def test_gateway_prepare_gemini_request_sets_format_and_synthesizes_payload():
    gateway = Gateway()

    # Verify format switch behavior.
    gateway.prepare_request({"messages": [{"role": "user", "content": "hi"}]})
    assert gateway.config.format == APIFormat.ANTHROPIC

    upstream = gateway.prepare_gemini_request(
        {
            "model": "gemini-2.5-pro",
            "temperature": 0.2,
            "contents": [{"role": "user", "parts": [{"text": "hello gemini"}]}],
            "system_instruction": {"parts": [{"text": "client system"}]},
            "tools": [{"function_declarations": [{"name": "search"}]}],
        }
    )

    assert gateway.config.format == APIFormat.GEMINI
    assert upstream["model"] == "gemini-2.5-pro"
    assert upstream["temperature"] == 0.2
    assert "contents" in upstream
    assert upstream["contents"]
    assert any(
        part.get("text") == "hello gemini"
        for msg in upstream["contents"]
        for part in msg.get("parts", [])
        if isinstance(part, dict)
    )
    assert "system_instruction" not in upstream
    assert "tools" not in upstream


def test_gateway_ingest_gemini_response_ingests_assistant_turn():
    gateway = Gateway()
    gateway.prepare_gemini_request(
        {
            "model": "gemini-2.5-pro",
            "contents": [{"role": "user", "parts": [{"text": "question"}]}],
        }
    )

    record = gateway.ingest_gemini_response(
        {
            "candidates": [
                {"content": {"parts": [{"text": "assistant answer"}]}}
            ]
        }
    )

    assert record is not None
    assert record.response_handle is not None
    response_block = gateway.projection.region(RegionID.EPHEMERAL).find(record.response_handle)
    assert response_block is not None
    assert response_block.content == "assistant answer"
