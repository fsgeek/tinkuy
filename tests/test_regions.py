"""Tests for projection region data structures."""

from tinkuy.regions import (
    ContentBlock,
    ContentKind,
    ContentStatus,
    Projection,
    RegionID,
)


class TestContentBlock:
    def test_create_computes_handle(self):
        block = ContentBlock.create(
            content="hello world",
            kind=ContentKind.CONVERSATION,
            label="test message",
        )
        assert len(block.handle) == 8
        assert all(c in "0123456789abcdef" for c in block.handle)

    def test_same_content_same_handle(self):
        a = ContentBlock.create("hello", ContentKind.FILE, "a")
        b = ContentBlock.create("hello", ContentKind.FILE, "b")
        assert a.handle == b.handle

    def test_different_content_different_handle(self):
        a = ContentBlock.create("hello", ContentKind.FILE, "a")
        b = ContentBlock.create("world", ContentKind.FILE, "b")
        assert a.handle != b.handle

    def test_token_estimate(self):
        content = "a" * 400
        block = ContentBlock.create(content, ContentKind.FILE, "test")
        assert block.size_tokens == 100  # 400 chars / 4


class TestAccessRecord:
    def test_touch_increments(self):
        block = ContentBlock.create("x", ContentKind.FILE, "test", turn=0)
        assert block.access.access_count == 0
        block.access.touch(1)
        assert block.access.access_count == 1
        assert block.access.last_access_turn == 1

    def test_fault_records(self):
        block = ContentBlock.create("x", ContentKind.FILE, "test", turn=0)
        block.access.record_fault(5)
        assert block.access.fault_count == 1
        assert block.access.access_count == 1
        assert block.access.last_access_turn == 5


class TestRegion:
    def test_add_and_find(self):
        proj = Projection()
        block = proj.add_content("test", ContentKind.FILE, "test file")
        found = proj.region(RegionID.EPHEMERAL).find(block.handle)
        assert found is not None
        assert found.content == "test"

    def test_size_tokens(self):
        proj = Projection()
        proj.add_content("a" * 400, ContentKind.FILE, "a")
        proj.add_content("b" * 800, ContentKind.FILE, "b")
        r = proj.region(RegionID.EPHEMERAL)
        assert r.size_tokens == 300  # 100 + 200

    def test_waste_tracking(self):
        proj = Projection()
        block = proj.add_content("test content", ContentKind.FILE, "test")
        block.status = ContentStatus.PENDING_REMOVAL
        r = proj.region(RegionID.EPHEMERAL)
        assert r.waste_tokens == block.size_tokens
        assert r.size_tokens == 0  # present only

    def test_nominate_removal(self):
        proj = Projection()
        block = proj.add_content("test", ContentKind.FILE, "test")
        r = proj.region(RegionID.EPHEMERAL)
        r.nominate_removal(block.handle, "model", "no longer needed")
        assert len(r.nominations) == 1
        assert r.nominations[0].source == "model"


class TestProjection:
    def test_total_tokens(self):
        proj = Projection()
        proj.add_content("a" * 400, ContentKind.SYSTEM, "sys",
                         region=RegionID.SYSTEM)
        proj.add_content("b" * 800, ContentKind.CONVERSATION, "msg")
        assert proj.total_tokens == 300

    def test_touch(self):
        proj = Projection()
        block = proj.add_content("x", ContentKind.FILE, "test")
        proj.advance_turn()
        assert proj.touch(block.handle)
        assert block.access.last_access_turn == 1

    def test_touch_missing(self):
        proj = Projection()
        assert not proj.touch("nonexistent")

    def test_evict_and_recall(self):
        proj = Projection()
        block = proj.add_content("original content", ContentKind.FILE, "test")
        handle = block.handle

        # Create a tensor to replace it
        tensor = ContentBlock.create(
            content="compressed summary",
            kind=ContentKind.TENSOR,
            label="tensor for test",
            region=RegionID.DURABLE,
            turn=0,
        )

        # Evict
        assert proj.evict(handle, tensor)
        assert block.status == ContentStatus.AVAILABLE
        assert block.tensor_handle == tensor.handle

        # Tensor is in R2
        r2 = proj.region(RegionID.DURABLE)
        assert r2.find(tensor.handle) is not None

        # Recall from page store
        proj.advance_turn()
        content = proj.recall(handle)
        assert content == "original content"
        assert block.status == ContentStatus.PRESENT
        assert block.access.fault_count == 1

    def test_page_store_preserves_original(self):
        proj = Projection()
        block = proj.add_content("verbatim original", ContentKind.FILE, "test")
        assert proj.page_store[block.handle] == "verbatim original"

    def test_advance_turn(self):
        proj = Projection()
        assert proj.turn == 0
        proj.advance_turn()
        assert proj.turn == 1

    def test_snapshot_roundtrip(self):
        proj = Projection()
        proj.add_content("system prompt", ContentKind.SYSTEM, "sys",
                         region=RegionID.SYSTEM)
        proj.add_content("user msg", ContentKind.CONVERSATION, "msg")
        proj.advance_turn()

        snap = proj.snapshot()
        assert snap["turn"] == 1
        assert "SYSTEM" in snap["regions"]
        assert "EPHEMERAL" in snap["regions"]
        assert len(snap["regions"]["SYSTEM"]["blocks"]) == 1
        assert len(snap["regions"]["EPHEMERAL"]["blocks"]) == 1

    def test_checkpoint_only_recovery(self):
        """Acceptance test: projection reconstructable from checkpoint
        alone, without the client's message history."""
        proj = Projection()
        proj.add_content("system", ContentKind.SYSTEM, "sys",
                         region=RegionID.SYSTEM)
        b = proj.add_content("content", ContentKind.FILE, "file")
        b.access.touch(1)
        proj.advance_turn()

        snap = proj.snapshot()

        # Verify snapshot contains all state needed for recovery
        assert snap["turn"] == 1
        blocks = snap["regions"]["EPHEMERAL"]["blocks"]
        assert len(blocks) == 1
        assert blocks[0]["access"]["access_count"] == 1
        assert blocks[0]["access"]["last_access_turn"] == 1

    def test_multiple_regions(self):
        proj = Projection()
        proj.add_content("tools", ContentKind.SYSTEM, "tools",
                         region=RegionID.TOOLS)
        proj.add_content("system", ContentKind.SYSTEM, "sys",
                         region=RegionID.SYSTEM)
        proj.add_content("tensor", ContentKind.TENSOR, "t",
                         region=RegionID.DURABLE)
        proj.add_content("recent", ContentKind.CONVERSATION, "msg",
                         region=RegionID.EPHEMERAL)
        proj.add_content("now", ContentKind.CONVERSATION, "current",
                         region=RegionID.CURRENT)

        assert proj.region(RegionID.TOOLS).block_count == 1
        assert proj.region(RegionID.SYSTEM).block_count == 1
        assert proj.region(RegionID.DURABLE).block_count == 1
        assert proj.region(RegionID.EPHEMERAL).block_count == 1
        assert proj.region(RegionID.CURRENT).block_count == 1
