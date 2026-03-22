"""Tests for gateway integration behavior."""

from __future__ import annotations

import json

import pytest

from tinkuy.gateway import Gateway, GatewayConfig, TurnResult
from tinkuy.orchestrator import ResponseSignalType
from tinkuy.regions import ContentKind, ContentStatus, RegionID
from tinkuy.store import (
    FileCheckpointStore,
    FilePageStore,
    MemoryCheckpointStore,
    MemoryPageStore,
)


def test_gateway_construction_defaults_to_memory_stores():
    gateway = Gateway()

    assert isinstance(gateway.page_store, MemoryPageStore)
    assert isinstance(gateway.checkpoint_store, MemoryCheckpointStore)
    assert gateway.event_log is not None
    assert gateway.turn == 0


def test_gateway_construction_with_data_dir_uses_file_stores(tmp_path):
    gateway = Gateway(GatewayConfig(data_dir=str(tmp_path)))

    assert isinstance(gateway.page_store, FilePageStore)
    assert isinstance(gateway.checkpoint_store, FileCheckpointStore)
    assert (tmp_path / "pages").exists()


def test_process_turn_returns_turn_result_and_payload_contains_messages_and_system():
    gateway = Gateway()
    gateway.rehydrate(
        {
            "system": "Policy: be concise.",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        }
    )

    result = gateway.process_turn("new user turn")

    assert isinstance(result, TurnResult)
    assert isinstance(result.api_payload, dict)
    assert "messages" in result.api_payload
    assert "system" in result.api_payload
    assert result.api_payload["messages"]
    assert result.api_payload["system"]
    assert result.record.turn == gateway.turn


def test_ingest_response_processes_release_retain_and_recall_signals():
    gateway = Gateway()
    gateway.process_turn("remember this")

    user_block = gateway.projection.region(RegionID.CURRENT).present_blocks()[0]
    handle = user_block.handle

    # Prepare pending-removal state so RETAIN has work to do.
    user_block.status = ContentStatus.PENDING_REMOVAL
    gateway.projection.region(RegionID.CURRENT).nominate_removal(
        handle,
        source="test",
        reason="test retain",
    )

    retain_record = gateway.ingest_response(
        content="assistant ack",
        signals=[{"type": "retain", "handle": handle}],
    )
    assert any(s.type == ResponseSignalType.RETAIN for s in retain_record.signals_processed)
    assert user_block.status == ContentStatus.PRESENT

    release_record = gateway.ingest_response(
        content="assistant summary",
        signals=[
            {
                "type": "release",
                "handle": handle,
                "tensor_content": "short tensor",
                "declared_losses": "exact wording",
            }
        ],
    )
    assert any(s.type == ResponseSignalType.RELEASE for s in release_record.signals_processed)
    assert user_block.status == ContentStatus.AVAILABLE
    assert user_block.tensor_handle is not None

    durable = gateway.projection.region(RegionID.DURABLE).find(user_block.tensor_handle)
    assert durable is not None
    assert durable.kind == ContentKind.TENSOR

    recall_record = gateway.ingest_response(
        content="assistant recall",
        signals=[{"type": "recall", "handle": handle}],
    )
    assert any(s.type == ResponseSignalType.RECALL for s in recall_record.signals_processed)
    assert user_block.status == ContentStatus.PRESENT
    assert user_block.content == "remember this"
    assert user_block.access.fault_count >= 1


def test_full_cycle_process_turn_then_ingest_response():
    gateway = Gateway()

    turn_result = gateway.process_turn("first question")
    record = gateway.ingest_response("first answer")

    assert turn_result.record.turn == 1
    assert turn_result.api_payload["messages"]
    assert record.response_handle is not None
    assert gateway.turn == 1


def test_rehydrate_from_dict_and_from_file(tmp_path):
    dict_gateway = Gateway()
    data = {
        "system": "System prompt",
        "messages": [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ],
    }
    dict_gateway.rehydrate(data)

    assert dict_gateway.turn == 1
    assert dict_gateway.projection.region(RegionID.SYSTEM).blocks

    file_gateway = Gateway()
    path = tmp_path / "conversation.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    file_gateway.rehydrate(path)

    assert file_gateway.turn == 1
    assert file_gateway.projection.region(RegionID.SYSTEM).blocks


def test_resume_from_checkpoint_in_same_data_dir(tmp_path):
    config = GatewayConfig(data_dir=str(tmp_path))

    gateway = Gateway(config)
    gateway.process_turn("persist me")

    resumed = Gateway.resume(config)

    assert resumed is not None
    assert resumed.turn == gateway.turn
    assert (tmp_path / "checkpoint.json").exists() or (tmp_path / "checkpoint.bak").exists()

    # Ensure resumed gateway is live and can continue turns.
    resumed.process_turn("next turn")
    assert resumed.turn == gateway.turn + 1


def test_resume_returns_none_when_no_checkpoint(tmp_path):
    resumed = Gateway.resume(GatewayConfig(data_dir=str(tmp_path)))
    assert resumed is None


def test_pressure_zone_property_and_page_table_method():
    gateway = Gateway()
    gateway.process_turn("add current")
    gateway.ingest_response("assistant response")
    # Second turn ages first turn's content from R4 to R3, making it
    # visible in the page table (which only covers R2 and R3).
    gateway.process_turn("second turn")

    zone = gateway.pressure_zone
    table = gateway.page_table()

    assert zone.name in {"LOW", "MODERATE", "ELEVATED", "CRITICAL"}
    assert isinstance(table, list)
    assert table
    assert {"handle", "kind", "status", "region", "size_tokens", "fault_count", "age_turns", "label"} <= set(
        table[0].keys()
    )


def test_page_table_is_injected_into_system_prompt_when_evicted_blocks_exist():
    gateway = Gateway()
    gateway.process_turn("content to evict")
    user_block = gateway.projection.region(RegionID.CURRENT).present_blocks()[0]

    gateway.ingest_response(
        content="assistant release",
        signals=[
            {
                "type": "release",
                "handle": user_block.handle,
                "tensor_content": "tensorized",
                "declared_losses": "details omitted",
            }
        ],
    )

    result = gateway.process_turn("next request")

    assert "system" in result.api_payload
    system = result.api_payload["system"]
    assert isinstance(system, list)
    assert any("<yuyay-page-table>" in part.get("text", "") for part in system)


def test_ingest_response_raises_on_unknown_signal_type():
    gateway = Gateway()
    gateway.process_turn("hello")

    with pytest.raises(ValueError, match="Unknown signal type"):
        gateway.ingest_response(
            content="assistant",
            signals=[{"type": "unsupported", "handle": "abcd1234"}],
        )
