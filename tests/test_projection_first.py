"""Tests for projection-first gateway: no proxy escape hatch."""

from __future__ import annotations

import inspect

from tinkuy.core.orchestrator import EventType, InboundEvent, Orchestrator
from tinkuy.core.regions import ContentKind, ContentStatus, RegionID
from tinkuy.gateway import Gateway, GatewayConfig


def test_parse_client_system_converts_text_blocks_to_inbound_events():
    """Client system text blocks become SYSTEM_UPDATE InboundEvents."""
    from tinkuy.gateway._gateway import _parse_client_system

    client_system = [
        {"type": "text", "text": "You are Claude, made by Anthropic."},
        {"type": "text", "text": "# CLAUDE.md\nBe concise."},
    ]

    events = _parse_client_system(client_system)

    assert len(events) == 2
    assert all(e.type == EventType.SYSTEM_UPDATE for e in events)
    assert events[0].content == "You are Claude, made by Anthropic."
    assert events[1].content == "# CLAUDE.md\nBe concise."


def test_parse_client_system_handles_bare_strings():
    """Bare string system blocks get normalized."""
    from tinkuy.gateway._gateway import _parse_client_system

    client_system = [
        "You are Claude.",
        {"type": "text", "text": "More instructions."},
    ]

    events = _parse_client_system(client_system)

    assert len(events) == 2
    assert events[0].content == "You are Claude."


def test_parse_client_system_strips_cache_control():
    """cache_control is stripped — the gateway owns cache placement."""
    from tinkuy.gateway._gateway import _parse_client_system

    client_system = [
        {"type": "text", "text": "instructions",
         "cache_control": {"type": "ephemeral"}},
    ]

    events = _parse_client_system(client_system)

    assert len(events) == 1
    assert events[0].content == "instructions"
    # No cache_control in metadata
    assert "cache_control" not in events[0].metadata


def test_client_system_ingested_on_first_turn_populates_r1():
    """First turn: client system blocks land in R1 via the projection."""
    gw = Gateway(GatewayConfig(lightweight=True))

    client_body = {
        "system": [
            {"type": "text", "text": "You are Claude."},
            {"type": "text", "text": "# CLAUDE.md\nBe concise."},
        ],
        "messages": [{"role": "user", "content": "hello"}],
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 8096,
    }

    payload = gw.prepare_request(client_body)

    # R1 should have content from client system blocks
    r1 = gw.projection.region(RegionID.SYSTEM)
    assert r1.block_count >= 2, f"R1 should have client system blocks, got {r1.block_count}"

    # The synthesized system array should contain R1 content
    system_texts = [b.get("text", "") for b in payload.get("system", [])]
    full_text = "\n".join(system_texts)
    assert "You are Claude." in full_text
    assert "CLAUDE.md" in full_text


def test_client_system_not_reingested_when_fingerprint_unchanged():
    """Subsequent turns with same system blocks don't duplicate R1."""
    gw = Gateway(GatewayConfig(lightweight=True))

    client_body = {
        "system": [{"type": "text", "text": "You are Claude."}],
        "messages": [{"role": "user", "content": "hello"}],
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 8096,
    }

    gw.prepare_request(client_body)
    r1_count_after_first = gw.projection.region(RegionID.SYSTEM).block_count

    # Simulate response ingestion so we can do another turn
    gw.ingest_response("hi there")

    gw.prepare_request(client_body)
    r1_count_after_second = gw.projection.region(RegionID.SYSTEM).block_count

    assert r1_count_after_second == r1_count_after_first, (
        f"R1 grew from {r1_count_after_first} to {r1_count_after_second} — "
        "client system blocks were re-ingested"
    )


def test_client_system_reingested_when_fingerprint_changes():
    """Changed system blocks trigger re-ingestion into R1."""
    gw = Gateway(GatewayConfig(lightweight=True))

    body_v1 = {
        "system": [{"type": "text", "text": "Version 1"}],
        "messages": [{"role": "user", "content": "hello"}],
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 8096,
    }
    gw.prepare_request(body_v1)
    gw.ingest_response("ack")

    body_v2 = {
        "system": [{"type": "text", "text": "Version 2 with more content"}],
        "messages": [{"role": "user", "content": "hello again"}],
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 8096,
    }
    gw.prepare_request(body_v2)

    # R1 should now contain the v2 content
    r1 = gw.projection.region(RegionID.SYSTEM)
    r1_text = " ".join(b.content for b in r1.present_blocks())
    assert "Version 2" in r1_text


def test_synthesize_has_no_client_system_parameter():
    """Structural impediment: _synthesize cannot accept raw client data."""
    from tinkuy.gateway._gateway import Gateway

    sig = inspect.signature(Gateway._synthesize)
    param_names = list(sig.parameters.keys())

    assert "client_system" not in param_names, (
        "_synthesize still accepts client_system — the proxy escape hatch is open"
    )


def test_process_turn_has_no_client_system_parameter():
    """Structural impediment: process_turn cannot accept raw client data."""
    from tinkuy.gateway._gateway import Gateway

    sig = inspect.signature(Gateway.process_turn)
    param_names = list(sig.parameters.keys())

    assert "client_system" not in param_names, (
        "process_turn still accepts client_system — the proxy escape hatch is open"
    )


def test_payload_system_blocks_come_only_from_projection():
    """The system array contains only projection-sourced blocks."""
    gw = Gateway(GatewayConfig(lightweight=True))

    client_body = {
        "system": [
            {"type": "text", "text": "Client instruction alpha."},
        ],
        "messages": [{"role": "user", "content": "hello"}],
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 8096,
    }

    payload = gw.prepare_request(client_body)

    # Content should be present (via R1 projection)
    system_texts = [b.get("text", "") for b in payload.get("system", [])]
    full_text = "\n".join(system_texts)
    assert "Client instruction alpha." in full_text

    # R1 in the projection should have the content
    r1 = gw.projection.region(RegionID.SYSTEM)
    assert r1.block_count >= 1


def test_memory_protocol_is_in_r1_projection_not_injected():
    """Memory protocol lives in the projection, not injected post-synthesis."""
    gw = Gateway(GatewayConfig(lightweight=False))

    # Memory protocol should already be in R1
    r1 = gw.projection.region(RegionID.SYSTEM)
    r1_text = " ".join(b.content for b in r1.present_blocks())

    assert "yuyay-memory-protocol" in r1_text, (
        "Memory protocol not found in R1 projection"
    )


def test_inject_memory_protocol_r1_method_does_not_exist():
    """The post-synthesis injection method should be deleted."""
    assert not hasattr(Gateway, "_inject_memory_protocol_r1"), (
        "_inject_memory_protocol_r1 still exists — payload mutation backdoor"
    )
