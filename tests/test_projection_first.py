"""Tests for projection-first gateway: no proxy escape hatch."""

from __future__ import annotations

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
