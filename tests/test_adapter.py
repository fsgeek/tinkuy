"""Tests for adapter parsing, ingestion, and synthesis behavior."""

from __future__ import annotations

import json

from tinkuy.adapter import (
    ConversationMessage,
    IngestAdapter,
    LiveAdapter,
    parse_anthropic_messages,
    parse_jsonl,
    parse_raw_messages,
)
from tinkuy.orchestrator import Orchestrator
from tinkuy.regions import ContentKind, ContentStatus, Projection, RegionID


def _assert_strict_alternation(messages: list[dict[str, str]]) -> None:
    assert messages
    assert messages[0]["role"] == "user"
    for i in range(1, len(messages)):
        assert messages[i]["role"] != messages[i - 1]["role"]


def test_parse_anthropic_messages_simple_strings_and_system_prompt():
    data = {
        "system": "You are terse.",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    }

    parsed = parse_anthropic_messages(data)

    assert [(m.role, m.content, m.turn) for m in parsed] == [
        ("system", "You are terse.", None),
        ("user", "hi", 0),
        ("assistant", "hello", 0),
    ]


def test_parse_anthropic_messages_content_blocks_tool_use_and_tool_result():
    data = {
        "system": [
            {"type": "text", "text": "line 1"},
            {"type": "text", "text": "line 2"},
        ],
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "thinking"},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "search",
                        "input": {"q": "tinkuy"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [
                            {"type": "text", "text": "result a"},
                            {"type": "text", "text": "result b"},
                        ],
                    }
                ],
            },
        ],
    }

    parsed = parse_anthropic_messages(data)

    assert parsed[0].role == "system"
    assert parsed[0].content == "line 1\nline 2"

    text_block = parsed[1]
    assert text_block.role == "assistant"
    assert text_block.content == "thinking"
    assert text_block.turn == 0

    tool_use = parsed[2]
    assert tool_use.role == "tool"
    assert json.loads(tool_use.content) == {"q": "tinkuy"}
    assert tool_use.turn == 0
    assert tool_use.metadata == {"tool_name": "search", "tool_use_id": "toolu_1"}

    tool_result = parsed[3]
    assert tool_result.role == "tool_result"
    assert tool_result.content == "result a\nresult b"
    assert tool_result.turn == 0
    assert tool_result.metadata == {"tool_use_id": "toolu_1"}


def test_parse_jsonl_supports_defaults_and_metadata():
    lines = [
        '{"role":"user","content":"u1","metadata":{"k":"v"}}',
        "",
        '{"role":"assistant","content":"a1","turn":7}',
    ]

    parsed = parse_jsonl(lines)

    assert len(parsed) == 2
    assert parsed[0] == ConversationMessage(
        role="user",
        content="u1",
        turn=0,
        metadata={"k": "v"},
    )
    assert parsed[1].role == "assistant"
    assert parsed[1].content == "a1"
    assert parsed[1].turn == 7
    assert parsed[1].metadata == {}


def test_parse_raw_messages_assigns_turns_and_preserves_metadata():
    raw = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1", "metadata": {"m": 1}},
        {"role": "user", "content": "u2"},
    ]

    parsed = parse_raw_messages(raw)

    assert [(m.role, m.content, m.turn) for m in parsed] == [
        ("user", "u1", 0),
        ("assistant", "a1", 0),
        ("user", "u2", 1),
    ]
    assert parsed[1].metadata == {"m": 1}


def test_ingest_messages_groups_by_turn_and_places_blocks():
    orch = Orchestrator()
    ingest = IngestAdapter(orch)
    messages = [
        ConversationMessage(role="system", content="sys"),
        ConversationMessage(role="user", content="u1"),
        ConversationMessage(role="assistant", content="a1"),
        ConversationMessage(role="user", content="u2"),
        ConversationMessage(role="assistant", content="a2"),
    ]

    ingest.ingest_messages(messages)

    assert orch.turn == 2
    assert len(orch.history) == 2

    system_blocks = orch.projection.region(RegionID.SYSTEM).present_blocks()
    assert [b.content for b in system_blocks] == ["sys"]

    current_blocks = orch.projection.region(RegionID.CURRENT).present_blocks()
    assert [b.content for b in current_blocks] == ["u2", "a2"]

    ephemeral_blocks = orch.projection.region(RegionID.EPHEMERAL).blocks
    assert [b.content for b in ephemeral_blocks] == ["u1", "a1"]


def test_ingest_anthropic_and_ingest_jsonl():
    orch = Orchestrator()
    ingest = IngestAdapter(orch)

    ingest.ingest_anthropic(
        {
            "system": "sys",
            "messages": [
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
            ],
        }
    )

    ingest.ingest_jsonl(
        "\n".join(
            [
                '{"role":"user","content":"u2"}',
                '{"role":"assistant","content":"a2"}',
            ]
        )
    )

    assert orch.turn == 2
    assert [b.content for b in orch.projection.region(RegionID.SYSTEM).blocks] == ["sys"]
    assert [b.content for b in orch.projection.region(RegionID.CURRENT).blocks] == ["u2", "a2"]
    assert [b.content for b in orch.projection.region(RegionID.EPHEMERAL).blocks] == ["u1", "a1"]


def test_ingest_file_json_and_jsonl(tmp_path):
    # .json (raw messages)
    raw_path = tmp_path / "conversation.json"
    raw_path.write_text(
        json.dumps(
            [
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
            ]
        ),
        encoding="utf-8",
    )

    orch_json = Orchestrator()
    ingest_json = IngestAdapter(orch_json)
    ingest_json.ingest_file(raw_path)

    assert orch_json.turn == 1
    assert [b.content for b in orch_json.projection.region(RegionID.CURRENT).blocks] == ["u1", "a1"]
    assert [b.content for b in orch_json.projection.region(RegionID.EPHEMERAL).blocks] == []

    # .jsonl
    jsonl_path = tmp_path / "conversation.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                '{"role":"user","content":"u2"}',
                '{"role":"assistant","content":"a2"}',
            ]
        ),
        encoding="utf-8",
    )

    orch_jsonl = Orchestrator()
    ingest_jsonl_adapter = IngestAdapter(orch_jsonl)
    ingest_jsonl_adapter.ingest_file(jsonl_path)

    assert orch_jsonl.turn == 1
    assert [b.content for b in orch_jsonl.projection.region(RegionID.CURRENT).blocks] == ["u2", "a2"]
    assert [b.content for b in orch_jsonl.projection.region(RegionID.EPHEMERAL).blocks] == []


def test_live_synthesize_messages_uses_system_from_r0_r1_and_valid_alternation():
    projection = Projection()
    projection.add_content(
        content="tool schema",
        kind=ContentKind.SYSTEM,
        label="tool def",
        region=RegionID.TOOLS,
    )
    projection.add_content(
        content="system policy",
        kind=ContentKind.SYSTEM,
        label="system",
        region=RegionID.SYSTEM,
    )
    projection.add_content(
        content="u1",
        kind=ContentKind.CONVERSATION,
        label="user",
        region=RegionID.CURRENT,
    )
    projection.add_content(
        content="a1",
        kind=ContentKind.CONVERSATION,
        label="assistant",
        region=RegionID.EPHEMERAL,
    )

    live = LiveAdapter(Orchestrator(projection=projection))
    payload = live.synthesize_messages()

    assert "system" in payload
    assert [part["text"] for part in payload["system"]] == ["tool schema", "system policy"]
    # Cache breakpoint goes on the last system part only (budget conservation)
    assert payload["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in payload["system"][0]

    _assert_strict_alternation(payload["messages"])


def test_live_synthesize_messages_emits_tensor_marker_for_available_evicted_block():
    projection = Projection()
    block = projection.add_content(
        content="verbatim user content",
        kind=ContentKind.FILE,
        label="file read",
        region=RegionID.EPHEMERAL,
    )
    block.status = ContentStatus.AVAILABLE

    live = LiveAdapter(Orchestrator(projection=projection))
    payload = live.synthesize_messages()

    assert payload["messages"] == [
        {
            "role": "user",
            "content": f"[tensor:{block.handle[:8]} — file read ({block.size_tokens} tokens)]",
        }
    ]


def test_live_synthesize_messages_merges_consecutive_same_role_messages():
    projection = Projection()
    projection.add_content(
        content="u1",
        kind=ContentKind.CONVERSATION,
        label="user",
        region=RegionID.EPHEMERAL,
    )
    projection.add_content(
        content="u2",
        kind=ContentKind.FILE,
        label="file",
        region=RegionID.CURRENT,
    )

    live = LiveAdapter(Orchestrator(projection=projection))
    payload = live.synthesize_messages()

    # Consecutive same-role messages merge into a content block list
    assert payload["messages"] == [{"role": "user", "content": [
        {"type": "text", "text": "u1"},
        {"type": "text", "text": "u2"},
    ]}]


def test_live_synthesize_messages_starts_with_user_when_first_message_is_assistant():
    projection = Projection()
    projection.add_content(
        content="assistant first",
        kind=ContentKind.CONVERSATION,
        label="assistant",
        region=RegionID.EPHEMERAL,
    )

    live = LiveAdapter(Orchestrator(projection=projection))
    payload = live.synthesize_messages()

    assert payload["messages"][0] == {"role": "user", "content": "[conversation start]"}
    assert payload["messages"][1] == {"role": "assistant", "content": "assistant first"}


def test_live_synthesize_page_table_output_format():
    projection = Projection(turn=5)
    durable = projection.add_content(
        content="tensor",
        kind=ContentKind.TENSOR,
        label="tensor:abc",
        region=RegionID.DURABLE,
    )
    ephemeral = projection.add_content(
        content="tool out",
        kind=ContentKind.TOOL_RESULT,
        label="rg",
        region=RegionID.EPHEMERAL,
    )
    durable.access.fault_count = 2
    durable.access.last_access_turn = 4
    ephemeral.access.last_access_turn = 1

    live = LiveAdapter(Orchestrator(projection=projection))
    table = live.synthesize_page_table()

    assert table.startswith("<yuyay-page-table>\n")
    assert table.endswith("\n</yuyay-page-table>")
    # Durable tensor has faults > 0, so it gets an individual entry
    assert f'handle="{durable.handle}"' in table
    assert 'kind="tensor"' in table
    assert 'status="present"' in table
    # Ephemeral block is coalescable (present, no faults, age > 2)
    # so it appears in an episode summary, not individually
    assert "episode" in table
    assert 'kinds="tool_result"' in table


def test_full_cycle_ingest_then_synthesize_messages():
    orch = Orchestrator()
    ingest = IngestAdapter(orch)

    ingest.ingest_messages(
        [
            ConversationMessage(role="system", content="global policy"),
            ConversationMessage(role="user", content="u1"),
            ConversationMessage(role="assistant", content="a1"),
            ConversationMessage(role="user", content="u2"),
            ConversationMessage(role="assistant", content="a2"),
        ]
    )

    payload = LiveAdapter(orch).synthesize_messages()

    assert [part["text"] for part in payload["system"]] == ["global policy"]
    _assert_strict_alternation(payload["messages"])

    joined = "\n".join(m["content"] for m in payload["messages"])
    assert "u1" in joined
    assert "u2" in joined
    assert "a1" in joined
    assert "a2" in joined
