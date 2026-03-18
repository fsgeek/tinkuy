"""Tests for store backends and store/orchestrator integration."""

from __future__ import annotations

import json
import os

from tinkuy.orchestrator import (
    EventType,
    InboundEvent,
    Orchestrator,
    ResponseSignal,
    ResponseSignalType,
)
from tinkuy.regions import ContentKind, ContentStatus, Projection, RegionID
from tinkuy.store import (
    FileCheckpointStore,
    FilePageStore,
    MemoryCheckpointStore,
    MemoryPageStore,
)


class RecordingCheckpointStore:
    """Checkpoint store test double that records snapshots passed to save()."""

    def __init__(self) -> None:
        self.saved: list[dict] = []

    def save(self, data: dict) -> None:
        self.saved.append(data)

    def load(self) -> dict | None:
        return self.saved[-1] if self.saved else None

    def exists(self) -> bool:
        return bool(self.saved)


class RecordingPageStore:
    """Page store test double that records writes and serves reads."""

    def __init__(self) -> None:
        self.pages: dict[str, str] = {}
        self.put_calls: list[tuple[str, str]] = []

    def put(self, handle: str, content: str) -> None:
        self.put_calls.append((handle, content))
        self.pages[handle] = content

    def get(self, handle: str) -> str | None:
        return self.pages.get(handle)

    def has(self, handle: str) -> bool:
        return handle in self.pages

    def delete(self, handle: str) -> None:
        self.pages.pop(handle, None)

    def handles(self) -> list[str]:
        return list(self.pages.keys())


def test_memory_page_store_put_get_has_delete_handles():
    store = MemoryPageStore()

    assert store.get("a") is None
    assert not store.has("a")

    store.put("a", "alpha")
    store.put("b", "beta")

    assert store.has("a")
    assert store.get("a") == "alpha"
    assert set(store.handles()) == {"a", "b"}

    store.delete("a")

    assert not store.has("a")
    assert store.get("a") is None
    assert store.handles() == ["b"]


def test_memory_checkpoint_store_save_load_exists():
    store = MemoryCheckpointStore()

    assert not store.exists()
    assert store.load() is None

    data = {"turn": 7, "regions": {"EPHEMERAL": {"blocks": []}}}
    store.save(data)

    assert store.exists()
    assert store.load() == data


def test_file_page_store_put_get_has_delete_handles_and_missing_returns_none(tmp_path):
    store = FilePageStore(tmp_path / "pages")

    assert store.get("missing") is None
    assert not store.has("missing")

    store.put("h1", "content one")
    store.put("h2", "content two")

    assert store.get("h1") == "content one"
    assert store.has("h1")
    assert set(store.handles()) == {"h1", "h2"}

    store.delete("h1")

    assert not store.has("h1")
    assert store.get("h1") is None
    assert store.handles() == ["h2"]


def test_file_page_store_put_uses_atomic_replace(tmp_path, monkeypatch):
    store = FilePageStore(tmp_path / "pages")
    calls: list[tuple[str, str]] = []
    original_replace = os.replace

    def recording_replace(src: str, dst: str) -> None:
        calls.append((src, dst))
        original_replace(src, dst)

    monkeypatch.setattr("tinkuy.store.os.replace", recording_replace)

    store.put("abc", "payload")

    assert len(calls) == 1
    src, dst = calls[0]
    assert src.endswith("abc.tmp")
    assert dst.endswith("abc.page")
    assert (tmp_path / "pages" / "abc.page").read_text(encoding="utf-8") == "payload"


def test_file_checkpoint_store_save_load_exists_and_backup_rotation(tmp_path):
    path = tmp_path / "state" / "checkpoint.json"
    store = FileCheckpointStore(path)

    assert not store.exists()
    assert store.load() is None

    first = {"turn": 1, "regions": {"CURRENT": {"blocks": []}}}
    second = {"turn": 2, "regions": {"CURRENT": {"blocks": ["x"]}}}

    store.save(first)
    assert store.exists()
    assert store.load() == first
    assert not path.with_suffix(".bak").exists()

    store.save(second)

    assert store.load() == second
    assert json.loads(path.with_suffix(".bak").read_text(encoding="utf-8")) == first


def test_file_checkpoint_store_recovers_from_backup_when_main_missing(tmp_path):
    path = tmp_path / "checkpoint.json"
    store = FileCheckpointStore(path)

    old = {"turn": 3}
    new = {"turn": 4}
    store.save(old)
    store.save(new)

    path.unlink()

    assert store.exists()
    assert store.load() == old


def test_orchestrator_from_checkpoint_restores_projection_state():
    projection = Projection(turn=5)
    block = projection.add_content(
        content="restored message",
        kind=ContentKind.CONVERSATION,
        label="user",
        region=RegionID.CURRENT,
    )
    checkpoint_store = MemoryCheckpointStore()
    checkpoint_store.save(projection.snapshot())

    restored = Orchestrator.from_checkpoint(checkpoint_store)

    assert restored is not None
    assert restored.turn == 5
    restored_block = restored.projection.region(RegionID.CURRENT).find(block.handle)
    assert restored_block is not None
    assert restored_block.label == "user"
    assert restored_block.status == ContentStatus.PRESENT


def test_release_signal_persists_to_page_store_before_evict(monkeypatch):
    page_store = RecordingPageStore()
    orch = Orchestrator(page_store=page_store)
    block = orch.projection.add_content(
        content="verbatim payload",
        kind=ContentKind.FILE,
        label="file",
        region=RegionID.EPHEMERAL,
    )

    original_evict = orch.projection.evict

    def asserting_evict(handle, tensor):
        assert page_store.get(handle) == "verbatim payload"
        return original_evict(handle, tensor)

    monkeypatch.setattr(orch.projection, "evict", asserting_evict)

    orch.ingest_response(
        "assistant",
        signals=[
            ResponseSignal(
                type=ResponseSignalType.RELEASE,
                handle=block.handle,
                tensor_content="tensorized",
            )
        ],
    )

    assert page_store.put_calls == [(block.handle, "verbatim payload")]
    assert block.status == ContentStatus.AVAILABLE


def test_recall_signal_falls_back_to_persistent_page_store():
    page_store = MemoryPageStore()
    orch = Orchestrator(page_store=page_store)
    block = orch.projection.add_content(
        content="original text",
        kind=ContentKind.FILE,
        label="doc",
        region=RegionID.EPHEMERAL,
    )
    block.status = ContentStatus.AVAILABLE
    block.content = ""

    orch.projection.page_store.pop(block.handle)
    page_store.put(block.handle, "original text")

    orch.ingest_response(
        "assistant",
        signals=[ResponseSignal(type=ResponseSignalType.RECALL, handle=block.handle)],
    )

    assert block.status == ContentStatus.PRESENT
    assert block.content == "original text"
    assert block.access.fault_count == 1


def test_checkpoint_written_on_turn_boundary_and_idle_boundary():
    checkpoint_store = RecordingCheckpointStore()
    orch = Orchestrator(checkpoint_store=checkpoint_store)

    orch.begin_turn([InboundEvent(EventType.USER_MESSAGE, "hello", "user")])
    orch.mark_idle()

    assert len(checkpoint_store.saved) == 2
    assert checkpoint_store.saved[0]["turn"] == 1
    assert checkpoint_store.saved[1]["turn"] == 1
