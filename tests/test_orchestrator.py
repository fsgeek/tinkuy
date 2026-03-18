"""Tests for orchestrator event-loop behavior."""

from tinkuy.orchestrator import (
    EventType,
    InboundEvent,
    Orchestrator,
    ResponseSignal,
    ResponseSignalType,
)
from tinkuy.pressure import EvictionAction, EvictionDecision
from tinkuy.regions import ContentKind, ContentStatus, Projection, RegionID


def test_begin_turn_advances_turn_and_ages_r4_to_r3():
    projection = Projection(turn=3)
    old_current = projection.add_content(
        content="old current",
        kind=ContentKind.CONVERSATION,
        label="old",
        region=RegionID.CURRENT,
    )
    orch = Orchestrator(projection=projection)

    record = orch.begin_turn(
        [
            InboundEvent(
                type=EventType.USER_MESSAGE,
                content="new user",
                label="user",
            )
        ]
    )

    assert orch.turn == 4
    assert record.turn == 4
    assert orch.projection.region(RegionID.CURRENT).find(old_current.handle) is None

    aged = orch.projection.region(RegionID.EPHEMERAL).find(old_current.handle)
    assert aged is not None
    assert aged.region == RegionID.EPHEMERAL

    new_current = orch.projection.region(RegionID.CURRENT).present_blocks()
    assert len(new_current) == 1
    assert new_current[0].content == "new user"
    assert new_current[0].kind == ContentKind.CONVERSATION


def test_begin_turn_classifies_and_places_all_event_types_with_metadata():
    orch = Orchestrator()
    events = [
        InboundEvent(EventType.USER_MESSAGE, "u", "user", {"channel": "chat"}),
        InboundEvent(EventType.TOOL_RESULT, "tool out", "tool", {"tool": "rg"}),
        InboundEvent(EventType.SYSTEM_UPDATE, "sys", "system update"),
        InboundEvent(EventType.TOOL_DEFINITION, "schema", "tool def"),
    ]

    record = orch.begin_turn(events)

    handles = record.inbound_handles
    user = orch.projection.region(RegionID.CURRENT).find(handles[0])
    tool = orch.projection.region(RegionID.EPHEMERAL).find(handles[1])
    system = orch.projection.region(RegionID.SYSTEM).find(handles[2])
    tool_def = orch.projection.region(RegionID.TOOLS).find(handles[3])

    assert user is not None and user.kind == ContentKind.CONVERSATION
    assert user.metadata["channel"] == "chat"
    assert tool is not None and tool.kind == ContentKind.TOOL_RESULT
    assert tool.metadata["tool"] == "rg"
    assert system is not None and system.kind == ContentKind.SYSTEM
    assert tool_def is not None and tool_def.kind == ContentKind.SYSTEM


def test_ingest_response_stores_assistant_content_in_r3_and_updates_record():
    orch = Orchestrator()
    turn_record = orch.begin_turn(
        [InboundEvent(EventType.USER_MESSAGE, "hello", "user")]
    )

    updated = orch.ingest_response("assistant text", label="assistant")

    assert updated is turn_record
    assert updated.response_handle is not None
    response_block = orch.projection.region(RegionID.EPHEMERAL).find(updated.response_handle)
    assert response_block is not None
    assert response_block.content == "assistant text"
    assert response_block.kind == ContentKind.CONVERSATION
    assert response_block.label == "assistant"


def test_release_signal_evicts_block_and_creates_durable_tensor_with_losses():
    orch = Orchestrator()
    block = orch.projection.add_content(
        content="evict me",
        kind=ContentKind.CONVERSATION,
        label="target",
        region=RegionID.EPHEMERAL,
    )

    orch.ingest_response(
        "assistant",
        signals=[
            ResponseSignal(
                type=ResponseSignalType.RELEASE,
                handle=block.handle,
                tensor_content="compressed tensor",
                declared_losses="lost raw details",
            )
        ],
    )

    target = orch.projection.region(RegionID.EPHEMERAL).find(block.handle)
    assert target is not None
    assert target.status == ContentStatus.AVAILABLE
    assert target.tensor_handle is not None

    tensor = orch.projection.region(RegionID.DURABLE).find(target.tensor_handle)
    assert tensor is not None
    assert tensor.kind == ContentKind.TENSOR
    assert tensor.content == "compressed tensor"
    assert tensor.metadata["declared_losses"] == "lost raw details"


def test_retain_signal_cancels_pending_removal_and_clears_nominations():
    orch = Orchestrator()
    block = orch.projection.add_content(
        content="keep me",
        kind=ContentKind.FILE,
        label="candidate",
        region=RegionID.EPHEMERAL,
    )
    block.status = ContentStatus.PENDING_REMOVAL
    region = orch.projection.region(RegionID.EPHEMERAL)
    region.nominate_removal(block.handle, source="scheduler", reason="pressure")

    orch.ingest_response(
        "assistant",
        signals=[ResponseSignal(type=ResponseSignalType.RETAIN, handle=block.handle)],
    )

    assert block.status == ContentStatus.PRESENT
    assert region.nominations == []


def test_recall_signal_faults_available_block_back_to_present():
    orch = Orchestrator()
    block = orch.projection.add_content(
        content="verbatim",
        kind=ContentKind.FILE,
        label="file",
        region=RegionID.EPHEMERAL,
    )
    block.status = ContentStatus.AVAILABLE
    block.content = ""

    orch.ingest_response(
        "assistant",
        signals=[ResponseSignal(type=ResponseSignalType.RECALL, handle=block.handle)],
    )

    assert block.status == ContentStatus.PRESENT
    assert block.content == "verbatim"
    assert block.access.fault_count == 1
    assert block.access.access_count == 1


def test_pending_removal_execution_runs_for_blocks_with_tensor_handle():
    orch = Orchestrator()
    block = orch.projection.add_content(
        content="pending",
        kind=ContentKind.TOOL_RESULT,
        label="pending",
        region=RegionID.EPHEMERAL,
    )
    block.status = ContentStatus.PENDING_REMOVAL
    block.tensor_handle = "deadbeef"

    record = orch.ingest_response("assistant")

    assert record.evictions_executed == 1
    assert block.status == ContentStatus.AVAILABLE
    assert block.content == ""


def test_apply_decisions_marks_pending_and_executes_direct_evict():
    orch = Orchestrator()
    req = orch.projection.add_content("req", ContentKind.FILE, "req", region=RegionID.EPHEMERAL)
    dem = orch.projection.add_content("dem", ContentKind.FILE, "dem", region=RegionID.EPHEMERAL)
    ev = orch.projection.add_content("ev", ContentKind.FILE, "ev", region=RegionID.EPHEMERAL)
    ev.tensor_handle = "abc12345"

    decisions = [
        EvictionDecision(EvictionAction.REQUEST_TENSOR, handle=req.handle, reason="moderate"),
        EvictionDecision(EvictionAction.DEMAND_TENSOR, handle=dem.handle, reason="critical"),
        EvictionDecision(EvictionAction.EVICT, handle=ev.handle, reason="tensor exists"),
        EvictionDecision(EvictionAction.RESTRUCTURE),
    ]

    marked = orch.apply_decisions(decisions)

    assert set(marked) == {req.handle, dem.handle}
    assert req.status == ContentStatus.PENDING_REMOVAL
    assert dem.status == ContentStatus.PENDING_REMOVAL
    assert ev.status == ContentStatus.AVAILABLE
    assert ev.content == ""

    nominations = orch.projection.region(RegionID.EPHEMERAL).nominations
    by_handle = {n.handle: n for n in nominations}
    assert by_handle[req.handle].source == "scheduler"
    assert by_handle[dem.handle].reason == "critical"


def test_page_table_contains_r2_r3_entries_with_age_and_faults():
    projection = Projection(turn=9)
    orch = Orchestrator(projection=projection)

    r2 = projection.add_content(
        content="tensor body",
        kind=ContentKind.TENSOR,
        label="tensor",
        region=RegionID.DURABLE,
    )
    r3 = projection.add_content(
        content="ephemeral body",
        kind=ContentKind.FILE,
        label="file",
        region=RegionID.EPHEMERAL,
    )
    r2.access.last_access_turn = 2
    r2.access.fault_count = 1
    r3.access.last_access_turn = 8

    table = orch.page_table()

    assert len(table) == 2
    entries = {entry["handle"]: entry for entry in table}
    assert entries[r2.handle]["kind"] == "tensor"
    assert entries[r2.handle]["status"] == "present"
    assert entries[r2.handle]["region"] == RegionID.DURABLE.value
    assert entries[r2.handle]["age_turns"] == 7
    assert entries[r2.handle]["fault_count"] == 1
    assert entries[r3.handle]["kind"] == "file"
    assert entries[r3.handle]["region"] == RegionID.EPHEMERAL.value
    assert entries[r3.handle]["age_turns"] == 1


def test_mark_idle_executes_pending_removals_and_requests_restructure_for_waste():
    orch = Orchestrator(context_limit=10_000)
    block = orch.projection.add_content(
        content="wasteful content" * 10,
        kind=ContentKind.CONVERSATION,
        label="waste",
        region=RegionID.EPHEMERAL,
    )
    block.status = ContentStatus.PENDING_REMOVAL

    decisions = orch.mark_idle()

    assert orch.is_idle
    assert block.status == ContentStatus.PENDING_REMOVAL
    assert any(d.action == EvictionAction.RESTRUCTURE for d in decisions)


def test_full_begin_turn_to_ingest_response_cycle_with_decision_application():
    orch = Orchestrator(context_limit=80)

    begin = orch.begin_turn(
        [
            InboundEvent(EventType.USER_MESSAGE, "u" * 80, "user"),
            InboundEvent(EventType.TOOL_RESULT, "t" * 80, "tool"),
        ]
    )
    marked_handles = orch.apply_decisions(begin.eviction_decisions)

    # Provide release tensors for whatever the scheduler marked pending.
    signals = [
        ResponseSignal(
            type=ResponseSignalType.RELEASE,
            handle=h,
            tensor_content=f"tensor:{h}",
            declared_losses="compressed",
        )
        for h in marked_handles
    ]
    after = orch.ingest_response("assistant follow-up", signals=signals)

    assert after is begin
    assert after.response_handle is not None
    assert len(after.signals_processed) == len(marked_handles)
    assert orch.history[-1] is begin

    for handle in marked_handles:
        block = orch.projection.region(RegionID.EPHEMERAL).find(handle)
        assert block is not None
        assert block.status == ContentStatus.AVAILABLE
        assert block.tensor_handle is not None
        assert orch.projection.region(RegionID.DURABLE).find(block.tensor_handle) is not None

    response = orch.projection.region(RegionID.EPHEMERAL).find(after.response_handle)
    assert response is not None
    assert response.content == "assistant follow-up"
