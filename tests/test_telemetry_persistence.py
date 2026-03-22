"""Tests for telemetry JSONL persistence in the gateway."""

from __future__ import annotations

import json

import pytest

from tinkuy.gateway import Gateway, GatewayConfig, TurnTelemetry
from tinkuy.gateway._gateway import _capture_client_context


def _minimal_client_body(**overrides) -> dict:
    body = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 8096,
        "stream": True,
        "messages": [{"role": "user", "content": "hello"}],
    }
    body.update(overrides)
    return body


def _make_gateway(tmp_path, session_id="test-session-001") -> Gateway:
    return Gateway(GatewayConfig(
        data_dir=str(tmp_path / ".tinkuy-data"),
        session_id=session_id,
    ))


def _telemetry_path(tmp_path, session_id="test-session-001"):
    return tmp_path / ".tinkuy-data" / "sessions" / session_id / "telemetry.jsonl"


def _do_turn(gw, client_body=None, **telemetry_kwargs):
    """Run prepare_request + report_telemetry and return the telemetry object."""
    body = client_body or _minimal_client_body()
    gw.prepare_request(body)
    t = TurnTelemetry(
        input_tokens=1000,
        cache_read_tokens=500,
        output_tokens=200,
        stop_reason="end_turn",
        **telemetry_kwargs,
    )
    gw.report_telemetry(t)
    return t


# -----------------------------------------------------------------------
# 1. Basic write
# -----------------------------------------------------------------------

def test_telemetry_jsonl_written_on_report(tmp_path):
    gw = _make_gateway(tmp_path)
    _do_turn(gw)

    path = _telemetry_path(tmp_path)
    assert path.exists(), "telemetry.jsonl should be created after report_telemetry"

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    for key in (
        "session_id", "turn", "timestamp",
        "request", "projection", "repairs",
        "wire", "response", "eviction",
        "overhead_calibrated",
    ):
        assert key in record, f"missing top-level key: {key}"

    assert record["session_id"] == "test-session-001"
    assert record["turn"] >= 1


# -----------------------------------------------------------------------
# 2. Append across turns
# -----------------------------------------------------------------------

def test_telemetry_appends_multiple_turns(tmp_path):
    gw = _make_gateway(tmp_path)
    _do_turn(gw)
    _do_turn(gw, client_body=_minimal_client_body(
        messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "again"},
        ],
    ))

    path = _telemetry_path(tmp_path)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2

    r1 = json.loads(lines[0])
    r2 = json.loads(lines[1])
    assert r2["turn"] > r1["turn"]


# -----------------------------------------------------------------------
# 3. Client metadata captured
# -----------------------------------------------------------------------

def test_telemetry_captures_client_metadata(tmp_path):
    gw = _make_gateway(tmp_path)
    body = _minimal_client_body(
        tools=[{"name": "read_file", "description": "read", "input_schema": {}}],
        system="You are helpful.",
        thinking={"type": "enabled", "budget_tokens": 10000},
    )
    _do_turn(gw, client_body=body)

    path = _telemetry_path(tmp_path)
    record = json.loads(path.read_text().strip())
    req = record["request"]

    assert req["model"] == "claude-sonnet-4-20250514"
    assert req["max_tokens"] == 8096
    assert req["stream"] is True
    assert req["tools_count"] == 1
    assert req["tools_hash"] is not None
    assert req["system_hash"] is not None
    assert req["thinking"] == {"type": "enabled", "budget_tokens": 10000}


# -----------------------------------------------------------------------
# 4. Gateway state captured
# -----------------------------------------------------------------------

def test_telemetry_captures_gateway_state(tmp_path):
    gw = _make_gateway(tmp_path)
    _do_turn(gw)

    path = _telemetry_path(tmp_path)
    record = json.loads(path.read_text().strip())
    proj = record["projection"]

    assert "total_tokens" in proj
    assert "regions" in proj
    assert isinstance(proj["regions"], dict)
    assert "pressure_zone" in proj
    assert "usage_ratio" in proj


# -----------------------------------------------------------------------
# 5. No crash without data_dir
# -----------------------------------------------------------------------

def test_telemetry_no_write_without_data_dir(tmp_path):
    gw = Gateway()  # no data_dir, no session_id
    body = _minimal_client_body()
    gw.prepare_request(body)
    t = TurnTelemetry(input_tokens=100, output_tokens=50)
    gw.report_telemetry(t)
    # Should not raise, and no files created
    assert gw._telemetry_path is None


# -----------------------------------------------------------------------
# 6. _capture_client_context hash stability
# -----------------------------------------------------------------------

def test_capture_client_context_hashes():
    body_a = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 8096,
        "tools": [{"name": "tool_a", "description": "a", "input_schema": {}}],
        "system": "Be concise.",
        "messages": [{"role": "user", "content": "hi"}],
    }
    messages_a = body_a["messages"]

    ctx1 = _capture_client_context(body_a, messages_a)
    ctx2 = _capture_client_context(body_a, messages_a)

    # Same input -> same hashes
    assert ctx1["request"]["tools_hash"] == ctx2["request"]["tools_hash"]
    assert ctx1["request"]["system_hash"] == ctx2["request"]["system_hash"]
    assert ctx1["request"]["tools_hash"] is not None
    assert ctx1["request"]["system_hash"] is not None

    # Different tools -> different tools_hash
    body_b = {**body_a, "tools": [{"name": "tool_b", "description": "b", "input_schema": {}}]}
    ctx3 = _capture_client_context(body_b, messages_a)
    assert ctx3["request"]["tools_hash"] != ctx1["request"]["tools_hash"]

    # Different system -> different system_hash
    body_c = {**body_a, "system": "Be verbose."}
    ctx4 = _capture_client_context(body_c, messages_a)
    assert ctx4["request"]["system_hash"] != ctx1["request"]["system_hash"]

    # No tools -> None hash
    body_d = {k: v for k, v in body_a.items() if k != "tools"}
    ctx5 = _capture_client_context(body_d, messages_a)
    assert ctx5["request"]["tools_hash"] is None
    assert ctx5["request"]["tools_count"] == 0
