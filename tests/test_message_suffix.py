"""Tests for compute_message_suffix — the minimal message suffix invariant.

The system-block architecture puts the conversation projection in
system blocks. The messages array contains only the structural minimum
required by the Anthropic API: 1 message (user turn) or 3 messages
(placeholder + stripped assistant + user with tool_results).
"""

from tinkuy.gateway._gateway import compute_message_suffix


# --- Case 1: No tool_results → 1 message ---


def test_simple_user_text():
    msgs = [{"role": "user", "content": "hello"}]
    result = compute_message_suffix(msgs)
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "hello"


def test_user_text_with_history():
    """Only the last user message matters; history is in system blocks."""
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "response"},
        {"role": "user", "content": "second"},
    ]
    result = compute_message_suffix(msgs)
    assert len(result) == 1
    assert result[0]["content"] == "second"


def test_user_with_content_blocks_no_tool_results():
    """User message with text blocks (no tool_results) → Case 1."""
    msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]},
    ]
    result = compute_message_suffix(msgs)
    assert len(result) == 1


# --- Case 2: Tool_results → 3 messages ---


def test_tool_result_produces_three_messages():
    msgs = [
        {"role": "user", "content": "do something"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "let me think..."},
            {"type": "text", "text": "I'll read that file."},
            {"type": "tool_use", "id": "toolu_123", "name": "Read",
             "input": {"path": "/tmp/test"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_123",
             "content": "file contents here"},
        ]},
    ]
    result = compute_message_suffix(msgs)
    assert len(result) == 3
    assert result[0] == {"role": "user", "content": "[continued]"}
    assert result[2]["role"] == "user"
    assert result[2]["content"][0]["type"] == "tool_result"


def test_assistant_stripped_to_tool_use_only():
    """Thinking and text blocks are stripped from the assistant message."""
    msgs = [
        {"role": "user", "content": "do something"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "x" * 5000},
            {"type": "text", "text": "I'll read that file."},
            {"type": "tool_use", "id": "toolu_456", "name": "Read",
             "input": {"path": "/tmp/test"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_456",
             "content": "file contents"},
        ]},
    ]
    result = compute_message_suffix(msgs)
    assistant = result[1]
    assert assistant["role"] == "assistant"
    assert len(assistant["content"]) == 1
    assert assistant["content"][0]["type"] == "tool_use"
    assert assistant["content"][0]["id"] == "toolu_456"


def test_multiple_tool_use_all_preserved():
    """Multiple tool_use blocks in one assistant message are all kept."""
    msgs = [
        {"role": "user", "content": "read three files"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "reading..."},
            {"type": "tool_use", "id": "t1", "name": "Read", "input": {}},
            {"type": "tool_use", "id": "t2", "name": "Read", "input": {}},
            {"type": "tool_use", "id": "t3", "name": "Read", "input": {}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "a"},
            {"type": "tool_result", "tool_use_id": "t2", "content": "b"},
            {"type": "tool_result", "tool_use_id": "t3", "content": "c"},
        ]},
    ]
    result = compute_message_suffix(msgs)
    assistant = result[1]
    tool_use_ids = {b["id"] for b in assistant["content"]}
    assert tool_use_ids == {"t1", "t2", "t3"}


def test_deep_tool_chain_still_three_messages():
    """Even in a deep tool chain, suffix is always 3 messages."""
    msgs = [
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "Read", "input": {}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "a"},
        ]},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t2", "name": "Read", "input": {}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t2", "content": "b"},
        ]},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t3", "name": "Read", "input": {}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t3", "content": "c"},
        ]},
    ]
    result = compute_message_suffix(msgs)
    assert len(result) == 3
    # Only the LAST tool_use/tool_result pair, not the whole chain
    assert result[1]["content"][0]["id"] == "t3"
    assert result[2]["content"][0]["tool_use_id"] == "t3"


# --- Suffix validates cleanly ---


def test_suffix_passes_validator():
    """The suffix must pass the Anthropic payload validator."""
    from tinkuy.formats.validate import validate_anthropic_payload

    msgs = [
        {"role": "user", "content": "do something"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "thinking hard"},
            {"type": "tool_use", "id": "toolu_789", "name": "Bash",
             "input": {"command": "ls"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_789",
             "content": "file1.txt\nfile2.txt"},
        ]},
    ]
    suffix = compute_message_suffix(msgs)
    # Wrap in a minimal payload for the validator
    payload = {"system": [{"type": "text", "text": "test"}], "messages": suffix}
    validation = validate_anthropic_payload(payload)
    assert validation.valid, f"Validation errors: {[e.message for e in validation.errors]}"


def test_simple_user_passes_validator():
    from tinkuy.formats.validate import validate_anthropic_payload

    suffix = compute_message_suffix([{"role": "user", "content": "hello"}])
    payload = {"system": [{"type": "text", "text": "test"}], "messages": suffix}
    validation = validate_anthropic_payload(payload)
    assert validation.valid


# --- Edge cases ---


def test_empty_messages():
    result = compute_message_suffix([])
    assert len(result) == 1
    assert result[0]["role"] == "user"


def test_last_message_not_user():
    """Degenerate case — pass through and let validator catch it."""
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "bye"},
    ]
    result = compute_message_suffix(msgs)
    assert len(result) == 1
    assert result[0]["role"] == "assistant"


def test_assistant_no_tool_use_degrades_to_case_1():
    """If assistant has only thinking/text, degrade to Case 1."""
    msgs = [
        {"role": "user", "content": "think about this"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "deep thoughts"},
            {"type": "text", "text": "here's my answer"},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_orphan",
             "content": "this shouldn't happen"},
        ]},
    ]
    result = compute_message_suffix(msgs)
    # Degrades to Case 1 — just the user message
    assert len(result) == 1
    assert result[0]["role"] == "user"
