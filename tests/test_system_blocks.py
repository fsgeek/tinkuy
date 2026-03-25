"""Tests for system-block synthesis."""

from __future__ import annotations

from tinkuy.core.orchestrator import Orchestrator
from tinkuy.formats.system_blocks import SystemBlockSynthesizer
from tinkuy.regions import ContentKind, ContentStatus, RegionID


def test_synthesize_empty_projection_returns_empty_system_and_messages():
    payload = SystemBlockSynthesizer(Orchestrator()).synthesize()

    assert payload == {"system": [], "messages": []}


def test_synthesize_single_region_produces_one_system_block_with_cache_control():
    orch = Orchestrator(context_limit=200_000)
    orch.projection.add_content(
        "durable here",
        kind=ContentKind.TENSOR,
        label="durable",
        region=RegionID.DURABLE,
    )

    payload = SystemBlockSynthesizer(orch).synthesize(skip_page_table=True)
    system = payload["system"]

    assert len(system) == 1
    assert system[0]["text"] == "durable here"
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert payload["messages"] == []


def test_synthesize_all_regions_preserves_r1_to_r4_order_and_cache_placement():
    orch = Orchestrator(context_limit=200_000)
    orch.projection.add_content(
        "tools here",
        kind=ContentKind.SYSTEM,
        label="tools",
        region=RegionID.TOOLS,
    )
    orch.projection.add_content(
        "system here",
        kind=ContentKind.SYSTEM,
        label="system",
        region=RegionID.SYSTEM,
    )
    orch.projection.add_content(
        "durable here",
        kind=ContentKind.TENSOR,
        label="durable",
        region=RegionID.DURABLE,
    )
    orch.projection.add_content(
        "user said hello",
        kind=ContentKind.CONVERSATION,
        label="user",
        region=RegionID.EPHEMERAL,
    )
    orch.projection.add_content(
        "assistant said hi",
        kind=ContentKind.CONVERSATION,
        label="assistant response",
        region=RegionID.EPHEMERAL,
    )
    # User message in CURRENT is skipped (goes in messages, not system blocks).
    # Add a non-user block in CURRENT to test R4 serialization.
    orch.projection.add_content(
        "current msg",
        kind=ContentKind.CONVERSATION,
        label="user",
        region=RegionID.CURRENT,
    )
    orch.projection.add_content(
        "tool output",
        kind=ContentKind.TOOL_RESULT,
        label="tool_result",
        region=RegionID.CURRENT,
        tool_use_id="toolu_123",
    )

    payload = SystemBlockSynthesizer(orch).synthesize(skip_page_table=True)
    system = payload["system"]

    # R1 (tools+system), R2 (durable), R3 (ephemeral), R4 (current non-user)
    assert len(system) == 4
    assert system[0]["text"] == "tools here\n\nsystem here"
    assert system[1]["text"] == "durable here"
    assert system[2]["text"] == "[user] user said hello\n\n[assistant] assistant said hi"
    assert "[tool_result:" in system[3]["text"]

    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert system[1]["cache_control"] == {"type": "ephemeral"}
    assert system[2]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in system[3]


def test_synthesize_available_blocks_render_evicted_marker_with_recall_hint():
    orch = Orchestrator(context_limit=200_000)
    block = orch.projection.add_content(
        "verbatim content that was evicted",
        kind=ContentKind.FILE,
        label="file read",
        region=RegionID.EPHEMERAL,
    )
    block.status = ContentStatus.AVAILABLE

    payload = SystemBlockSynthesizer(orch).synthesize(skip_page_table=True)
    text = payload["system"][0]["text"]

    assert text.startswith(f"[evicted:{block.handle[:8]} ")
    assert "file read" in text
    assert f'use <recall handle="{block.handle}"/>' in text


def test_synthesize_conversation_blocks_include_role_markers():
    orch = Orchestrator(context_limit=200_000)
    orch.projection.add_content(
        "hello there",
        kind=ContentKind.CONVERSATION,
        label="user",
        region=RegionID.EPHEMERAL,
    )
    orch.projection.add_content(
        "hi back",
        kind=ContentKind.CONVERSATION,
        label="assistant response",
        region=RegionID.EPHEMERAL,
    )

    payload = SystemBlockSynthesizer(orch).synthesize(skip_page_table=True)
    text = payload["system"][0]["text"]

    assert "[user] hello there" in text
    assert "[assistant] hi back" in text


def test_synthesize_tool_result_blocks_include_tool_use_id_prefix():
    orch = Orchestrator(context_limit=200_000)
    orch.projection.add_content(
        "tool output",
        kind=ContentKind.TOOL_RESULT,
        label="tool result",
        region=RegionID.EPHEMERAL,
        tool_use_id="toolu_123",
    )

    payload = SystemBlockSynthesizer(orch).synthesize(skip_page_table=True)

    assert payload["system"][0]["text"] == "[tool_result:toolu_123] tool output"


def test_synthesize_system_and_tensor_blocks_render_without_role_markers():
    orch = Orchestrator(context_limit=200_000)
    orch.projection.add_content(
        "system policy",
        kind=ContentKind.SYSTEM,
        label="system",
        region=RegionID.SYSTEM,
    )
    orch.projection.add_content(
        "compressed durable summary",
        kind=ContentKind.TENSOR,
        label="durable",
        region=RegionID.DURABLE,
    )

    payload = SystemBlockSynthesizer(orch).synthesize(skip_page_table=True)

    assert payload["system"][0]["text"] == "system policy"
    assert payload["system"][1]["text"] == "compressed durable summary"


def test_synthesize_includes_page_table_in_r4_when_not_skipped():
    orch = Orchestrator(context_limit=200_000)
    orch.projection.add_content(
        "ephemeral context",
        kind=ContentKind.FILE,
        label="file",
        region=RegionID.EPHEMERAL,
    )
    # Use a tool result in CURRENT (user messages are skipped — they go in messages)
    orch.projection.add_content(
        "tool output",
        kind=ContentKind.TOOL_RESULT,
        label="tool_result",
        region=RegionID.CURRENT,
        tool_use_id="toolu_123",
    )

    synth = SystemBlockSynthesizer(orch)
    payload = synth.synthesize()
    r4_text = payload["system"][-1]["text"]

    assert "[tool_result:" in r4_text
    assert "<yuyay-page-table>" in r4_text
    assert "</yuyay-page-table>" in r4_text
    assert synth.last_page_table_tokens > 0


def test_synthesize_skip_page_table_omits_page_table_text():
    orch = Orchestrator(context_limit=200_000)
    orch.projection.add_content(
        "ephemeral context",
        kind=ContentKind.FILE,
        label="file",
        region=RegionID.EPHEMERAL,
    )
    # Use a tool result in CURRENT (user messages go to messages array)
    orch.projection.add_content(
        "tool output",
        kind=ContentKind.TOOL_RESULT,
        label="tool_result",
        region=RegionID.CURRENT,
        tool_use_id="toolu_123",
    )

    synth = SystemBlockSynthesizer(orch)
    payload = synth.synthesize(skip_page_table=True)
    # R4 block exists (tool result) but has no page table
    r4_text = payload["system"][-1]["text"]

    assert "<yuyay-page-table>" not in r4_text
    assert synth.last_page_table_tokens == 0


def test_synthesize_extracts_text_and_tool_use_summary_from_content_blocks():
    orch = Orchestrator(context_limit=200_000)
    orch.projection.add_content(
        "fallback content should be ignored",
        kind=ContentKind.CONVERSATION,
        label="assistant",
        region=RegionID.EPHEMERAL,
        content_blocks=[
            {"type": "text", "text": "first text"},
            {"type": "tool_use", "name": "search", "input": {"q": "tinkuy"}},
            {"type": "text", "text": "second text"},
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "skip"},
        ],
    )

    payload = SystemBlockSynthesizer(orch).synthesize(skip_page_table=True)
    text = payload["system"][0]["text"]

    assert text == "[assistant] first text\n[called search]\nsecond text"


def test_user_message_in_current_excluded_from_system_blocks():
    """User messages in CURRENT go in messages, not system blocks.

    This prevents double-counting: the user's current message must
    appear only in the messages array, not in both system and messages.
    """
    orch = Orchestrator(context_limit=200_000)
    orch.projection.add_content(
        "earlier context",
        kind=ContentKind.CONVERSATION,
        label="assistant response",
        region=RegionID.EPHEMERAL,
    )
    orch.projection.add_content(
        "current user message",
        kind=ContentKind.CONVERSATION,
        label="user",
        region=RegionID.CURRENT,
    )

    payload = SystemBlockSynthesizer(orch).synthesize(skip_page_table=True)

    # The user message should NOT appear in any system block
    all_system_text = " ".join(b["text"] for b in payload["system"])
    assert "current user message" not in all_system_text

    # Assistant response from R3 should still be present
    assert "earlier context" in all_system_text
