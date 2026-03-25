"""Independent tests for projection-first gateway behavior."""

from __future__ import annotations

import inspect

import pytest

from tinkuy.core.orchestrator import EventType, InboundEvent, Orchestrator
from tinkuy.core.regions import ContentKind, ContentStatus, Projection, RegionID
from tinkuy.gateway import Gateway, GatewayConfig
from tinkuy.gateway._gateway import APIFormat, MEMORY_PROTOCOL, _parse_client_system


def _r1_labels(gateway: Gateway) -> list[str]:
    return [b.label for b in gateway.projection.region(RegionID.SYSTEM).blocks]


def _r1_contents(gateway: Gateway) -> list[str]:
    return [b.content for b in gateway.projection.region(RegionID.SYSTEM).blocks]


def test_parse_client_system_empty_inputs_produce_no_events():
    assert _parse_client_system([]) == []
    assert _parse_client_system(["", {"type": "text", "text": ""}, 123]) == []


def test_parse_client_system_mixed_types_keeps_only_textual_content():
    blocks: list[dict[str, object] | str] = [
        "alpha",
        {"type": "text", "text": "beta", "cache_control": {"type": "ephemeral"}},
        {"type": "image", "source": {"type": "base64", "data": "..."}},
        {"type": "tool_use", "name": "read", "input": {"path": "README.md"}},
        {"text": "gamma"},
    ]

    events = _parse_client_system(blocks)

    assert [e.type for e in events] == [EventType.SYSTEM_UPDATE] * 3
    assert [e.label for e in events] == ["client-system-0", "client-system-1", "client-system-4"]
    assert [e.content for e in events] == ["alpha", "beta", "gamma"]


def test_parse_client_system_preserves_large_blocks():
    big = "X" * 120_000
    events = _parse_client_system([{"type": "text", "text": big}])

    assert len(events) == 1
    assert events[0].type == EventType.SYSTEM_UPDATE
    assert events[0].label == "client-system-0"
    assert len(events[0].content) == 120_000


def test_ingest_client_system_skips_when_fingerprint_equivalent_despite_structure_changes():
    gw = Gateway(GatewayConfig(lightweight=False))

    gw._ingest_client_system(["Alpha", {"type": "text", "text": "Beta"}])
    first_labels = _r1_labels(gw)
    first_contents = _r1_contents(gw)

    # Same concatenated content bytes but different block structure.
    gw._ingest_client_system([
        {"type": "text", "text": "Al"},
        {"type": "text", "text": "phaB"},
        {"type": "text", "text": "eta"},
    ])

    assert _r1_labels(gw) == first_labels
    assert _r1_contents(gw) == first_contents


def test_ingest_client_system_reingests_when_content_changes_at_same_length():
    gw = Gateway(GatewayConfig(lightweight=False))

    gw._ingest_client_system([{"type": "text", "text": "abcd"}])
    before = _r1_contents(gw)

    # Same length as "abcd", different content.
    gw._ingest_client_system([{"type": "text", "text": "wxyz"}])
    after = _r1_contents(gw)

    assert before != after
    assert "wxyz" in after
    assert "abcd" not in after


def test_ingest_client_system_clears_stale_client_blocks_and_preserves_memory_protocol():
    gw = Gateway(GatewayConfig(lightweight=False))

    gw._ingest_client_system([
        {"type": "text", "text": "sys-v1-a"},
        {"type": "text", "text": "sys-v1-b"},
    ])
    gw._ingest_client_system([{"type": "text", "text": "sys-v2-only"}])

    r1 = gw.projection.region(RegionID.SYSTEM).blocks
    labels = [b.label for b in r1]
    contents = [b.content for b in r1]

    assert labels.count("memory-protocol") == 1
    assert "sys-v2-only" in contents
    assert "sys-v1-a" not in contents
    assert "sys-v1-b" not in contents
    assert labels.count("client-system-0") == 1


def test_structural_impediment_client_system_removed_from_public_and_internal_signatures():
    process_params = inspect.signature(Gateway.process_turn).parameters
    synth_params = inspect.signature(Gateway._synthesize).parameters

    assert "client_system" not in process_params
    assert "client_system" not in synth_params

    gw = Gateway()
    with pytest.raises(TypeError):
        gw.process_turn(user_content="hello", client_system=[{"type": "text", "text": "x"}])
    with pytest.raises(TypeError):
        gw._synthesize(APIFormat.ANTHROPIC, client_system=[{"type": "text", "text": "x"}])



def test_structural_impediment_inject_memory_method_removed():
    assert not hasattr(Gateway, "_inject_memory_protocol_r1")


def test_prepare_request_populates_r1_and_system_is_projection_driven():
    gw = Gateway(GatewayConfig(lightweight=False))

    upstream = gw.prepare_request(
        {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "stream": True,
            "system": [
                {
                    "type": "text",
                    "text": "Independent policy A",
                    "cache_control": {"type": "persistent"},
                    "x-raw": "must-not-leak",
                },
                {"type": "text", "text": "Independent policy B"},
            ],
            "messages": [{"role": "user", "content": "hello"}],
        }
    )

    r1_contents = _r1_contents(gw)
    assert any(c == MEMORY_PROTOCOL for c in r1_contents)
    assert "Independent policy A" in r1_contents
    assert "Independent policy B" in r1_contents

    system = upstream["system"]
    assert isinstance(system, list)
    assert system

    all_system_text = "\n".join(block.get("text", "") for block in system)
    assert "Independent policy A" in all_system_text
    assert "Independent policy B" in all_system_text
    assert "must-not-leak" not in all_system_text

    # R1 tier gets the cache breakpoint; client-provided cache_control never survives.
    assert system[0].get("cache_control") == {"type": "ephemeral"}
    assert all(block.get("cache_control", {}).get("type") != "persistent" for block in system)


def test_prepare_request_proxy_escape_hatch_is_dead_no_raw_client_system_blocks():
    gw = Gateway(GatewayConfig(lightweight=False))

    client_system = [
        {
            "type": "text",
            "text": "A system directive",
            "cache_control": {"type": "persistent"},
            "internal": "secret",
        },
        {"type": "image", "source": {"type": "base64", "data": "ignored"}},
    ]

    upstream = gw.prepare_request(
        {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 512,
            "stream": False,
            "system": client_system,
            "messages": [{"role": "user", "content": "ping"}],
        }
    )

    # Output system blocks are synthesized text blocks, not pass-through dicts from client input.
    assert upstream["system"] != client_system
    assert all(set(block.keys()) <= {"type", "text", "cache_control"} for block in upstream["system"])

    joined = "\n".join(block.get("text", "") for block in upstream["system"])
    assert "A system directive" in joined
    assert "secret" not in joined
    assert "ignored" not in joined


def test_resume_keeps_single_memory_protocol_from_fresh_checkpoint(tmp_path):
    config = GatewayConfig(data_dir=str(tmp_path), session_id="fresh", lightweight=False)
    gw = Gateway(config)
    gw.process_turn("first")

    resumed = Gateway.resume(config)

    assert resumed is not None
    labels = _r1_labels(resumed)
    assert labels.count("memory-protocol") == 1


def test_resume_injects_memory_protocol_into_legacy_checkpoint(tmp_path):
    legacy = GatewayConfig(data_dir=str(tmp_path), session_id="legacy", lightweight=True)
    old_gw = Gateway(legacy)
    old_gw.process_turn("legacy turn")

    resumed = Gateway.resume(
        GatewayConfig(data_dir=str(tmp_path), session_id="legacy", lightweight=False)
    )

    assert resumed is not None
    labels = _r1_labels(resumed)
    contents = _r1_contents(resumed)
    assert labels.count("memory-protocol") == 1
    assert MEMORY_PROTOCOL in contents


def test_parse_client_system_events_route_to_r1_via_orchestrator_place_event():
    projection = Projection()
    orch = Orchestrator(projection=projection)

    events = _parse_client_system(["route-me", {"type": "text", "text": "and-me"}])
    placed = [orch._place_event(event) for event in events]

    assert all(isinstance(event, InboundEvent) for event in events)
    assert all(block.region == RegionID.SYSTEM for block in placed)
    assert [b.content for b in projection.region(RegionID.SYSTEM).blocks] == ["route-me", "and-me"]
    assert all(b.kind == ContentKind.SYSTEM for b in projection.region(RegionID.SYSTEM).blocks)


def test_prepare_request_cache_control_boundary_follows_r1_region():
    gw = Gateway(GatewayConfig(lightweight=False))

    # First request creates an assistant response in CURRENT, so second request
    # has non-empty R3 and we can verify breakpoint placement across tiers.
    gw.prepare_request(
        {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 256,
            "stream": True,
            "system": [{"type": "text", "text": "R1 seed"}],
            "messages": [{"role": "user", "content": "turn-1"}],
        }
    )
    gw.ingest_response("assistant-1")

    upstream = gw.prepare_request(
        {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 256,
            "stream": True,
            "system": [{"type": "text", "text": "R1 seed"}],
            "messages": [{"role": "user", "content": "turn-2"}],
        }
    )

    system = upstream["system"]
    assert len(system) >= 3

    # R1, R2 (if any), and R3 blocks are cache-marked; R4 is not.
    assert system[0].get("cache_control") == {"type": "ephemeral"}
    assert "cache_control" not in system[-1]

    # Projection-driven output still carries the original R1 content.
    assert "R1 seed" in system[0]["text"]
