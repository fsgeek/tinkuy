"""Tests for _repair_tool_pairing on the LiveAdapter."""

from __future__ import annotations

import copy

from tinkuy.adapter import LiveAdapter
from tinkuy.orchestrator import Orchestrator


def _make_adapter() -> LiveAdapter:
    """Minimal LiveAdapter for calling _repair_tool_pairing directly."""
    return LiveAdapter(Orchestrator())


# ── Helpers to build message fixtures ───────────────────────────────


def _assistant_with_tool_use(text: str, tool_id: str, tool_name: str = "grep") -> dict:
    return {
        "role": "assistant",
        "content": [
            {"type": "text", "text": text},
            {"type": "tool_use", "id": tool_id, "name": tool_name, "input": {}},
        ],
    }


def _user_with_tool_result(tool_id: str, result_text: str, extra_text: str | None = None) -> dict:
    blocks: list[dict] = [
        {"type": "tool_result", "tool_use_id": tool_id, "content": result_text},
    ]
    if extra_text is not None:
        blocks.append({"type": "text", "text": extra_text})
    return {"role": "user", "content": blocks}


def _assistant_text(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _user_text(text: str) -> dict:
    return {"role": "user", "content": text}


# ── Tests ───────────────────────────────────────────────────────────


def test_repair_strips_forward_orphan_tool_use():
    """Assistant has tool_use but next user message has no matching tool_result."""
    adapter = _make_adapter()
    messages = [
        _assistant_with_tool_use("thinking", "toolu_1"),
        _user_text("thanks"),
    ]
    result, counts = adapter._repair_tool_pairing(copy.deepcopy(messages))

    # The tool_use block should be stripped; text should remain.
    assistant_content = result[0]["content"]
    assert isinstance(assistant_content, list)
    assert all(b.get("type") != "tool_use" for b in assistant_content)
    assert any(b.get("text") == "thinking" for b in assistant_content)
    assert counts["forward_orphans_stripped"] == 1


def test_repair_strips_backward_orphan_tool_result():
    """User has tool_result referencing a tool_use_id absent from preceding assistant.

    This is the exact production bug: messages[i]=assistant with only text,
    messages[i+1]=user with tool_result + text.
    """
    adapter = _make_adapter()
    messages = [
        _assistant_text("just text, no tool_use"),
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_ghost", "content": "result"},
                {"type": "text", "text": "follow-up question"},
            ],
        },
    ]
    result, counts = adapter._repair_tool_pairing(copy.deepcopy(messages))

    user_content = result[1]["content"]
    assert isinstance(user_content, list)
    # tool_result stripped, text preserved
    assert all(b.get("type") != "tool_result" for b in user_content)
    assert any(b.get("text") == "follow-up question" for b in user_content)
    assert counts["backward_orphans_stripped"] == 1


def test_repair_preserves_valid_tool_pairing():
    """A properly paired tool_use / tool_result survives repair unchanged."""
    adapter = _make_adapter()
    messages = [
        _assistant_with_tool_use("let me search", "toolu_ok"),
        _user_with_tool_result("toolu_ok", "3 results found"),
    ]
    original = copy.deepcopy(messages)
    result, counts = adapter._repair_tool_pairing(copy.deepcopy(messages))

    # Both messages should be structurally identical to the originals.
    assert result[0]["content"] == original[0]["content"]
    assert result[1]["content"] == original[1]["content"]
    assert counts["forward_orphans_stripped"] == 0
    assert counts["backward_orphans_stripped"] == 0


def test_repair_placeholder_when_all_content_stripped():
    """Stripping all blocks from a message produces the placeholder string."""
    adapter = _make_adapter()

    # Forward orphan: assistant with only tool_use, no text -> placeholder
    messages_fwd = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_lone", "name": "rg", "input": {}},
            ],
        },
        _user_text("ok"),
    ]
    result_fwd, counts_fwd = adapter._repair_tool_pairing(copy.deepcopy(messages_fwd))
    assert result_fwd[0]["content"] == "[tool calls omitted]"
    assert counts_fwd["forward_orphans_stripped"] == 1

    # Backward orphan: user with only tool_result, no text -> placeholder
    messages_bwd = [
        _assistant_text("no tools here"),
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_gone", "content": "data"},
            ],
        },
    ]
    result_bwd, counts_bwd = adapter._repair_tool_pairing(copy.deepcopy(messages_bwd))
    assert result_bwd[1]["content"] == "[tool results omitted]"
    assert counts_bwd["backward_orphans_stripped"] == 1


def test_repair_mixed_valid_and_orphan():
    """User message has both a valid tool_result and an orphan — only orphan stripped."""
    adapter = _make_adapter()
    messages = [
        _assistant_with_tool_use("searching", "toolu_valid"),
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_valid", "content": "found it"},
                {"type": "tool_result", "tool_use_id": "toolu_stale", "content": "stale data"},
                {"type": "text", "text": "next question"},
            ],
        },
    ]
    result, counts = adapter._repair_tool_pairing(copy.deepcopy(messages))

    user_content = result[1]["content"]
    assert isinstance(user_content, list)

    result_ids = {
        b["tool_use_id"]
        for b in user_content
        if isinstance(b, dict) and b.get("type") == "tool_result"
    }
    assert "toolu_valid" in result_ids
    assert "toolu_stale" not in result_ids
    # Text block preserved
    assert any(b.get("text") == "next question" for b in user_content)
    assert counts["backward_orphans_stripped"] == 1
    assert counts["forward_orphans_stripped"] == 0


def test_repair_backward_orphan_at_conversation_start():
    """tool_result in messages[0] (user, no preceding assistant) gets stripped."""
    adapter = _make_adapter()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_nowhere", "content": "orphan"},
                {"type": "text", "text": "hello"},
            ],
        },
        _assistant_text("hi"),
    ]
    result, counts = adapter._repair_tool_pairing(copy.deepcopy(messages))

    user_content = result[0]["content"]
    assert isinstance(user_content, list)
    assert all(b.get("type") != "tool_result" for b in user_content)
    assert any(b.get("text") == "hello" for b in user_content)
    assert counts["backward_orphans_stripped"] == 1
