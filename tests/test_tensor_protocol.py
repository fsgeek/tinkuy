"""Tests for tensor protocol state-update parsing and deserialization."""

from __future__ import annotations

import json
import logging

from tinkuy.taste_gateway.tensor_protocol import parse_state_update


def test_parse_state_update_happy_path_with_well_typed_input():
    tool_input = {
        "updated_regions": [
            "strands",
            "declared_losses",
            "open_questions",
            "unresolved_tensions",
            "instructions_for_next",
            "overall_truth",
            "overall_indeterminacy",
            "overall_falsity",
            "feedback_to_harness",
            "memory_actions",
        ],
        "strands": [
            {
                "title": "A",
                "content": "alpha",
                "depends_on": ["B"],
                "key_claims": [{"claim": "x"}],
                "integration_losses": ["old detail"],
            }
        ],
        "declared_losses": [
            {
                "what_was_lost": "detail",
                "why": "compression",
                "shed_from": "A",
                "category": "context_pressure",
            }
        ],
        "open_questions": ["what next?"],
        "unresolved_tensions": [
            {
                "tension_id": "t1",
                "framings": [{"label": "f1", "weight": 0.5}],
                "cycles_held": 2,
                "touches_strands": ["A"],
                "what_would_collapse_it": "new evidence",
            }
        ],
        "instructions_for_next": "continue",
        "overall_truth": 0.8,
        "overall_indeterminacy": 0.1,
        "overall_falsity": 0.1,
        "feedback_to_harness": {"memory_pressure": "low"},
        "memory_actions": [
            {"action": "pin", "id": "m1", "content": "", "reason": "important"}
        ],
    }

    parsed = parse_state_update(tool_input)

    assert parsed["updated_regions"] == tool_input["updated_regions"]
    assert parsed["strands"][0]["title"] == "A"
    assert parsed["declared_losses"][0]["shed_from"] == "A"
    assert parsed["open_questions"] == ["what next?"]
    assert parsed["unresolved_tensions"][0]["tension_id"] == "t1"
    assert parsed["instructions_for_next"] == "continue"
    assert parsed["feedback_to_harness"] == {"memory_pressure": "low"}
    assert parsed["memory_actions"][0]["action"] == "pin"


def test_parse_state_update_deserializes_string_encoded_strands():
    tool_input = {
        "updated_regions": ["strands"],
        "strands": json.dumps(
            [
                {
                    "title": "S1",
                    "content": "serialized",
                    "depends_on": [],
                    "key_claims": [],
                    "integration_losses": [],
                }
            ]
        ),
    }

    parsed = parse_state_update(tool_input)

    assert isinstance(parsed["strands"], list)
    assert parsed["strands"][0]["title"] == "S1"
    assert parsed["strands"][0]["content"] == "serialized"


def test_parse_state_update_deserializes_other_string_encoded_known_fields():
    tool_input = {
        "updated_regions": [
            "declared_losses",
            "open_questions",
            "unresolved_tensions",
            "feedback_to_harness",
            "memory_actions",
        ],
        "declared_losses": json.dumps(
            [{"what_was_lost": "x", "why": "y", "shed_from": "S1"}]
        ),
        "open_questions": json.dumps(["q1", "q2"]),
        "unresolved_tensions": json.dumps(
            [{"tension_id": "t1", "framings": [{"f": "a"}]}]
        ),
        "feedback_to_harness": json.dumps({"curation": "aggressive"}),
        "memory_actions": json.dumps(
            [{"action": "summarize", "id": "m2", "content": "short"}]
        ),
    }

    parsed = parse_state_update(tool_input)

    assert parsed["declared_losses"][0]["what_was_lost"] == "x"
    assert parsed["open_questions"] == ["q1", "q2"]
    assert parsed["unresolved_tensions"][0]["tension_id"] == "t1"
    assert parsed["feedback_to_harness"] == {"curation": "aggressive"}
    assert parsed["memory_actions"][0]["id"] == "m2"


def test_parse_state_update_preserves_extra_unknown_fields():
    tool_input = {
        "updated_regions": ["strands"],
        "strands": [{"title": "S", "content": "c"}],
        "model_hunch": "keep me",
        "future_regions": ["r1", "r2"],
        "extra_blob": {"k": "v"},
    }

    parsed = parse_state_update(tool_input)

    assert parsed["model_hunch"] == "keep me"
    assert parsed["future_regions"] == ["r1", "r2"]
    assert parsed["extra_blob"] == {"k": "v"}


def test_parse_state_update_deserializes_string_encoded_extra_fields():
    tool_input = {
        "updated_regions": ["strands"],
        "strands": [{"title": "S", "content": "c"}],
        "extra_json_object": '{"x": 1, "y": true}',
        "extra_json_list": '[{"a": 1}, {"a": 2}]',
    }

    parsed = parse_state_update(tool_input)

    assert parsed["extra_json_object"] == {"x": 1, "y": True}
    assert parsed["extra_json_list"] == [{"a": 1}, {"a": 2}]


def test_parse_state_update_malformed_json_string_warns_and_does_not_crash(caplog):
    tool_input = {
        "updated_regions": ["strands"],
        "strands": [{"title": "S", "content": "c"}],
        "broken_extra": '{"x": 1,,}',
    }

    with caplog.at_level(logging.WARNING, logger="tinkuy.taste_gateway.tensor_protocol"):
        parsed = parse_state_update(tool_input)

    assert parsed["broken_extra"] == '{"x": 1,,}'
    assert any(
        "looks like JSON but failed to parse" in record.message
        for record in caplog.records
    )


def test_parse_state_update_handles_mixed_typed_string_encoded_and_unknown_fields():
    tool_input = {
        "updated_regions": ["strands", "declared_losses", "feedback_to_harness"],
        "strands": [{"title": "typed", "content": "already structured"}],
        "declared_losses": json.dumps(
            [{"what_was_lost": "detail", "why": "merge", "shed_from": "typed"}]
        ),
        "feedback_to_harness": '{"pressure":"moderate"}',
        "new_region_candidate": '[{"name":"rX"}]',
        "raw_note": "plain string",
    }

    parsed = parse_state_update(tool_input)

    assert parsed["strands"][0]["title"] == "typed"
    assert parsed["declared_losses"][0]["shed_from"] == "typed"
    assert parsed["feedback_to_harness"] == {"pressure": "moderate"}
    assert parsed["new_region_candidate"] == [{"name": "rX"}]
    assert parsed["raw_note"] == "plain string"


def test_parse_state_update_edge_cases_empty_lists_nulls_and_missing_optional_fields():
    tool_input = {
        "updated_regions": ["strands", "open_questions", "instructions_for_next"],
        "strands": [],
        "open_questions": None,
        "instructions_for_next": None,
    }

    parsed = parse_state_update(tool_input)

    assert parsed["updated_regions"] == ["strands", "open_questions", "instructions_for_next"]
    assert parsed["strands"] == []
    assert "open_questions" not in parsed
    assert "instructions_for_next" not in parsed
    assert "declared_losses" not in parsed
