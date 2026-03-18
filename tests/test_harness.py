"""Tests for harness signal extraction and session loop behavior."""

from __future__ import annotations

import json

from tinkuy.gateway import Gateway, GatewayConfig, TurnResult
from tinkuy.harness import (
    SessionConfig,
    SessionHarness,
    extract_signals,
    strip_signals,
)
from tinkuy.regions import RegionID


class MockMessageStream:
    def __init__(self, incoming: list[str | None], session_id: str | None = None) -> None:
        self._incoming = iter(incoming)
        self.delivered: list[tuple[str, dict[str, object]]] = []
        self.statuses: list[str] = []
        self._session_id = session_id

    @property
    def session_id(self) -> str | None:
        return self._session_id

    def receive(self) -> str | None:
        return next(self._incoming, None)

    def deliver(self, content: str, metadata: dict[str, object]) -> None:
        self.delivered.append((content, metadata))

    def deliver_status(self, status: str) -> None:
        self.statuses.append(status)


class MockAPIClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.payloads: list[dict[str, object]] = []

    def send(self, payload: dict[str, object]) -> str:
        self.payloads.append(payload)
        assert self.responses, "No mock response left for send()"
        return self.responses.pop(0)


def test_extract_signals_release_with_tensor_content():
    text = """
Assistant content.
<yuyay-response>
  <release handle="h_user_1" losses="exact phrase" />
</yuyay-response>
<tensor handle="h_user_1">Compressed summary</tensor>
"""

    signals = extract_signals(text)

    assert signals == [
        {
            "type": "release",
            "handle": "h_user_1",
            "tensor_content": "Compressed summary",
            "declared_losses": "exact phrase",
        }
    ]


def test_extract_signals_release_without_tensor_content():
    text = """
<yuyay-response>
  <release handle="h_missing_tensor" losses="minor detail" />
</yuyay-response>
"""

    signals = extract_signals(text)

    assert signals == [
        {
            "type": "release",
            "handle": "h_missing_tensor",
            "tensor_content": None,
            "declared_losses": "minor detail",
        }
    ]


def test_extract_signals_retain_with_reason():
    text = """
<yuyay-response>
  <retain handle="h_keep" reason="still needed for tool call" />
</yuyay-response>
"""

    signals = extract_signals(text)

    assert signals == [{"type": "retain", "handle": "h_keep"}]


def test_extract_signals_recall():
    text = """
<yuyay-response>
  <recall handle="h_recall" />
</yuyay-response>
"""

    signals = extract_signals(text)

    assert signals == [{"type": "recall", "handle": "h_recall"}]


def test_extract_signals_multiple_signals_in_one_block():
    text = """
<yuyay-response>
  <release handle="h1" losses="l1" />
  <retain handle="h2" reason="needed" />
  <recall handle="h3" />
</yuyay-response>
<tensor handle="h1">tensor for h1</tensor>
"""

    signals = extract_signals(text)

    assert signals == [
        {
            "type": "release",
            "handle": "h1",
            "tensor_content": "tensor for h1",
            "declared_losses": "l1",
        },
        {"type": "retain", "handle": "h2"},
        {"type": "recall", "handle": "h3"},
    ]


def test_extract_signals_returns_empty_when_no_yuyay_response_block():
    text = "Assistant says hello. <release handle=\"h1\" />"

    assert extract_signals(text) == []


def test_strip_signals_removes_yuyay_and_tensor_blocks_but_preserves_other_content():
    text = """
Prelude text.
<yuyay-response>
  <release handle="h1" losses="x" />
</yuyay-response>
Mid text stays.
<tensor handle="h1">Tensor content</tensor>
Trailing text.
<custom>leave me</custom>
"""

    stripped = strip_signals(text)

    assert "<yuyay-response>" not in stripped
    assert "<tensor handle=\"h1\">" not in stripped
    assert "Prelude text." in stripped
    assert "Mid text stays." in stripped
    assert "Trailing text." in stripped
    assert "<custom>leave me</custom>" in stripped


def test_session_harness_start_creates_gateway_instance():
    harness = SessionHarness(
        frontend=MockMessageStream([None]),
        api_client=MockAPIClient([]),
    )

    harness.start()

    assert isinstance(harness.gateway, Gateway)


def test_session_harness_step_processes_turn_end_to_end_with_mocks():
    response = """
Answer body for user.
<yuyay-response>
  <release handle="h_user_1" losses="verbatim details" />
</yuyay-response>
<tensor handle="h_user_1">compact summary</tensor>
"""
    frontend = MockMessageStream([])
    api_client = MockAPIClient([response])
    harness = SessionHarness(frontend=frontend, api_client=api_client)

    clean_response, turn_result = harness.step("Please remember this")

    assert isinstance(turn_result, TurnResult)
    assert "Answer body for user." in clean_response
    assert "<yuyay-response>" not in clean_response
    assert "<tensor" not in clean_response
    assert len(api_client.payloads) == 1

    assert harness.gateway.turn == 1
    ephemeral_blocks = harness.gateway.projection.region(RegionID.EPHEMERAL).blocks
    assert ephemeral_blocks
    assert ephemeral_blocks[-1].content == clean_response


def test_session_harness_run_loops_until_frontend_returns_none():
    frontend = MockMessageStream(["u1", "u2", None])
    api_client = MockAPIClient(
        [
            "first answer <yuyay-response><retain handle=\"h\"/></yuyay-response>",
            "second answer",
        ]
    )
    harness = SessionHarness(frontend=frontend, api_client=api_client)

    harness.run()

    assert len(api_client.payloads) == 2
    assert len(frontend.statuses) == 2
    assert len(frontend.delivered) == 2
    assert frontend.delivered[0][0] == "first answer"
    assert frontend.delivered[1][0] == "second answer"
    assert frontend.delivered[0][1]["turn"] == 1
    assert frontend.delivered[1][1]["turn"] == 2


def test_session_harness_start_rehydrates_from_source(tmp_path):
    source = tmp_path / "conversation.json"
    source.write_text(
        json.dumps(
            {
                "system": "sys",
                "messages": [
                    {"role": "user", "content": "rehydrated user"},
                    {"role": "assistant", "content": "rehydrated assistant"},
                ],
            }
        ),
        encoding="utf-8",
    )

    harness = SessionHarness(
        frontend=MockMessageStream([None]),
        api_client=MockAPIClient([]),
        config=SessionConfig(rehydrate_source=str(source)),
    )

    harness.start()

    assert harness.gateway.turn == 1
    assert harness.gateway.projection.region(RegionID.SYSTEM).present_blocks()


def test_session_harness_start_resumes_from_checkpoint_instead_of_rehydrating(tmp_path):
    config = GatewayConfig(data_dir=str(tmp_path))

    seeded_gateway = Gateway(config)
    seeded_gateway.process_turn("persisted user turn")
    persisted_turn = seeded_gateway.turn

    source = tmp_path / "conversation.json"
    source.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "rehydrate this"},
                    {"role": "assistant", "content": "answer"},
                ]
            }
        ),
        encoding="utf-8",
    )

    harness = SessionHarness(
        frontend=MockMessageStream([None]),
        api_client=MockAPIClient([]),
        config=SessionConfig(
            gateway_config=config,
            rehydrate_source=str(source),
        ),
    )

    harness.start()

    assert harness.gateway.turn == persisted_turn
