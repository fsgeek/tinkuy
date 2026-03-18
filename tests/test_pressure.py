"""Tests for pressure-gated eviction policy."""

import math

import pytest

from tinkuy.pressure import (
    EvictionAction,
    PressureScheduler,
    PressureState,
    PressureZone,
)
from tinkuy.regions import ContentKind, ContentStatus, Projection, RegionID


_counter = 0


def add_block(
    projection: Projection,
    *,
    kind: ContentKind,
    region: RegionID,
    size_tokens: int,
    last_access_turn: int,
    fault_count: int = 0,
    status: ContentStatus = ContentStatus.PRESENT,
    tensor_handle: str | None = None,
):
    """Create a block and override fields relevant to scheduler tests."""
    global _counter
    _counter += 1
    block = projection.add_content(
        content=f"block-{_counter}",
        kind=kind,
        label=f"block-{_counter}",
        region=region,
    )
    block.size_tokens = size_tokens
    block.access.last_access_turn = last_access_turn
    block.access.fault_count = fault_count
    block.status = status
    block.tensor_handle = tensor_handle
    return block


class TestPressureState:
    def test_zone_thresholds(self):
        assert PressureState(total_tokens=49, context_limit=100).zone == PressureZone.LOW
        assert (
            PressureState(total_tokens=50, context_limit=100).zone
            == PressureZone.MODERATE
        )
        assert (
            PressureState(total_tokens=69, context_limit=100).zone
            == PressureZone.MODERATE
        )
        assert (
            PressureState(total_tokens=70, context_limit=100).zone
            == PressureZone.ELEVATED
        )
        assert (
            PressureState(total_tokens=84, context_limit=100).zone
            == PressureZone.ELEVATED
        )
        assert (
            PressureState(total_tokens=85, context_limit=100).zone
            == PressureZone.CRITICAL
        )

    def test_usage_with_zero_context_limit_defaults_to_full_pressure(self):
        state = PressureState(total_tokens=1, context_limit=0)
        assert state.usage == 1.0
        assert state.zone == PressureZone.CRITICAL

    def test_headroom_tokens_never_negative(self):
        assert PressureState(total_tokens=70, context_limit=100).headroom_tokens == 30
        assert PressureState(total_tokens=120, context_limit=100).headroom_tokens == 0


class TestCandidateScoring:
    def test_kind_weights_order_evictability(self):
        scheduler = PressureScheduler()
        projection = Projection(turn=10)

        tool = add_block(
            projection,
            kind=ContentKind.TOOL_RESULT,
            region=RegionID.EPHEMERAL,
            size_tokens=0,
            last_access_turn=10,
        )
        file_block = add_block(
            projection,
            kind=ContentKind.FILE,
            region=RegionID.EPHEMERAL,
            size_tokens=0,
            last_access_turn=10,
        )
        convo = add_block(
            projection,
            kind=ContentKind.CONVERSATION,
            region=RegionID.EPHEMERAL,
            size_tokens=0,
            last_access_turn=10,
        )
        tensor = add_block(
            projection,
            kind=ContentKind.TENSOR,
            region=RegionID.EPHEMERAL,
            size_tokens=0,
            last_access_turn=10,
        )
        system = add_block(
            projection,
            kind=ContentKind.SYSTEM,
            region=RegionID.EPHEMERAL,
            size_tokens=0,
            last_access_turn=10,
        )

        s_tool = scheduler.score_candidate(tool, current_turn=10)
        s_file = scheduler.score_candidate(file_block, current_turn=10)
        s_convo = scheduler.score_candidate(convo, current_turn=10)
        s_tensor = scheduler.score_candidate(tensor, current_turn=10)
        s_system = scheduler.score_candidate(system, current_turn=10)

        assert s_tool.score > s_file.score > s_convo.score > s_tensor.score > s_system.score
        assert any("kind=TOOL_RESULT" in r for r in s_tool.reasons)
        assert any("kind=FILE" in r for r in s_file.reasons)

    def test_scoring_includes_age_size_fault_penalty_and_pending_bonus(self):
        scheduler = PressureScheduler()
        projection = Projection(turn=30)
        block = add_block(
            projection,
            kind=ContentKind.FILE,
            region=RegionID.EPHEMERAL,
            size_tokens=2500,
            last_access_turn=20,
            fault_count=2,
            status=ContentStatus.PENDING_REMOVAL,
        )

        candidate = scheduler.score_candidate(block, current_turn=30)

        expected = (
            12.0
            + math.log2(1 + 10) * 3.0
            + math.log2(1 + 2500 / 100)
            - (2 * 5.0)
            + 5.0
        )
        assert candidate.score == pytest.approx(expected)
        assert any("age=10 turns" in r for r in candidate.reasons)
        assert any("size=2500 tokens" in r for r in candidate.reasons)
        assert any("fault_count=2" in r for r in candidate.reasons)
        assert "already pending removal" in candidate.reasons

    def test_fault_penalty_reduces_candidate_score(self):
        scheduler = PressureScheduler()
        projection = Projection(turn=20)

        base = add_block(
            projection,
            kind=ContentKind.FILE,
            region=RegionID.EPHEMERAL,
            size_tokens=500,
            last_access_turn=10,
            fault_count=0,
        )
        high_fault = add_block(
            projection,
            kind=ContentKind.FILE,
            region=RegionID.EPHEMERAL,
            size_tokens=500,
            last_access_turn=10,
            fault_count=3,
        )

        s_base = scheduler.score_candidate(base, current_turn=20)
        s_fault = scheduler.score_candidate(high_fault, current_turn=20)

        assert s_fault.score == pytest.approx(s_base.score - 15.0)


class TestCandidateSelection:
    def test_select_candidates_only_r2_r3_excludes_tensors_and_available(self):
        scheduler = PressureScheduler()
        projection = Projection(turn=20)

        # Non-candidate regions
        add_block(
            projection,
            kind=ContentKind.TOOL_RESULT,
            region=RegionID.TOOLS,
            size_tokens=300,
            last_access_turn=0,
        )
        add_block(
            projection,
            kind=ContentKind.FILE,
            region=RegionID.SYSTEM,
            size_tokens=300,
            last_access_turn=0,
        )
        add_block(
            projection,
            kind=ContentKind.FILE,
            region=RegionID.CURRENT,
            size_tokens=300,
            last_access_turn=0,
        )

        # R2 tensor excluded
        add_block(
            projection,
            kind=ContentKind.TENSOR,
            region=RegionID.DURABLE,
            size_tokens=300,
            last_access_turn=0,
        )

        # Candidate blocks
        r2_file = add_block(
            projection,
            kind=ContentKind.FILE,
            region=RegionID.DURABLE,
            size_tokens=300,
            last_access_turn=0,
        )
        r3_convo = add_block(
            projection,
            kind=ContentKind.CONVERSATION,
            region=RegionID.EPHEMERAL,
            size_tokens=300,
            last_access_turn=0,
        )
        r3_tool = add_block(
            projection,
            kind=ContentKind.TOOL_RESULT,
            region=RegionID.EPHEMERAL,
            size_tokens=300,
            last_access_turn=0,
        )

        # AVAILABLE excluded
        add_block(
            projection,
            kind=ContentKind.TOOL_RESULT,
            region=RegionID.EPHEMERAL,
            size_tokens=300,
            last_access_turn=0,
            status=ContentStatus.AVAILABLE,
        )

        selected = scheduler.select_candidates(projection, limit=2)

        assert [c.block.handle for c in selected] == [r3_tool.handle, r2_file.handle]
        all_handles = {c.block.handle for c in scheduler.select_candidates(projection, limit=10)}
        assert r3_convo.handle in all_handles
        assert len(all_handles) == 3


class TestEvictionDecisions:
    def test_low_busy_makes_no_decisions(self):
        scheduler = PressureScheduler(context_limit=100)
        projection = Projection(turn=10)
        add_block(
            projection,
            kind=ContentKind.FILE,
            region=RegionID.SYSTEM,
            size_tokens=40,
            last_access_turn=0,
        )

        assert scheduler.decide(projection, is_idle=False) == []

    def test_low_idle_with_waste_requests_restructure(self):
        scheduler = PressureScheduler(context_limit=100)
        projection = Projection(turn=10)
        add_block(
            projection,
            kind=ContentKind.CONVERSATION,
            region=RegionID.EPHEMERAL,
            size_tokens=45,
            last_access_turn=0,
            status=ContentStatus.PENDING_REMOVAL,
        )

        decisions = scheduler.decide(projection, is_idle=True)

        assert len(decisions) == 1
        assert decisions[0].action == EvictionAction.RESTRUCTURE
        assert decisions[0].handle is None

    def test_moderate_busy_defers_eviction(self):
        scheduler = PressureScheduler(context_limit=100)
        projection = Projection(turn=10)
        add_block(
            projection,
            kind=ContentKind.TOOL_RESULT,
            region=RegionID.EPHEMERAL,
            size_tokens=60,
            last_access_turn=0,
        )

        assert scheduler.decide(projection, is_idle=False) == []

    def test_moderate_idle_requests_top_three(self):
        scheduler = PressureScheduler(context_limit=100)
        projection = Projection(turn=10)

        for idx in range(5):
            add_block(
                projection,
                kind=ContentKind.TOOL_RESULT,
                region=RegionID.EPHEMERAL,
                size_tokens=12,
                last_access_turn=9 - idx,
            )

        expected = [
            c.block.handle for c in scheduler.select_candidates(projection)[:3]
        ]

        decisions = scheduler.decide(projection, is_idle=True)

        assert len(decisions) == 3
        assert all(d.action == EvictionAction.REQUEST_TENSOR for d in decisions)
        assert [d.handle for d in decisions] == expected
        assert all("moderate pressure, idle boundary" in d.reason for d in decisions)

    @pytest.mark.parametrize("is_idle", [False, True])
    def test_elevated_requests_top_five_in_idle_or_busy(self, is_idle):
        scheduler = PressureScheduler(context_limit=100)
        projection = Projection(turn=10)

        for idx in range(7):
            add_block(
                projection,
                kind=ContentKind.FILE,
                region=RegionID.EPHEMERAL,
                size_tokens=11,
                last_access_turn=9 - idx,
            )

        expected = [
            c.block.handle for c in scheduler.select_candidates(projection)[:5]
        ]

        decisions = scheduler.decide(projection, is_idle=is_idle)

        assert len(decisions) == 5
        assert [d.handle for d in decisions] == expected
        assert all(d.action == EvictionAction.REQUEST_TENSOR for d in decisions)
        assert all("elevated pressure" in d.reason for d in decisions)

    def test_critical_mixes_request_demand_and_evict_actions(self):
        scheduler = PressureScheduler(context_limit=100)
        projection = Projection(turn=30)

        # Ensure critical pressure from PRESENT content.
        add_block(
            projection,
            kind=ContentKind.SYSTEM,
            region=RegionID.SYSTEM,
            size_tokens=60,
            last_access_turn=0,
            status=ContentStatus.PRESENT,
        )

        pending_with_tensor = add_block(
            projection,
            kind=ContentKind.FILE,
            region=RegionID.EPHEMERAL,
            size_tokens=30,
            last_access_turn=0,
            status=ContentStatus.PENDING_REMOVAL,
            tensor_handle="tensor-abc",
        )
        pending_without_tensor = add_block(
            projection,
            kind=ContentKind.FILE,
            region=RegionID.EPHEMERAL,
            size_tokens=30,
            last_access_turn=0,
            status=ContentStatus.PENDING_REMOVAL,
            tensor_handle=None,
        )
        present_block = add_block(
            projection,
            kind=ContentKind.TOOL_RESULT,
            region=RegionID.EPHEMERAL,
            size_tokens=30,
            last_access_turn=0,
            status=ContentStatus.PRESENT,
        )

        decisions = scheduler.decide(projection, is_idle=False)

        by_handle = {d.handle: d for d in decisions}
        assert by_handle[pending_with_tensor.handle].action == EvictionAction.EVICT
        assert by_handle[pending_without_tensor.handle].action == EvictionAction.DEMAND_TENSOR
        assert by_handle[present_block.handle].action == EvictionAction.REQUEST_TENSOR

    def test_critical_limits_to_top_eight_candidates(self):
        scheduler = PressureScheduler(context_limit=100)
        projection = Projection(turn=30)

        for idx in range(10):
            add_block(
                projection,
                kind=ContentKind.TOOL_RESULT,
                region=RegionID.EPHEMERAL,
                size_tokens=10,
                last_access_turn=29 - idx,
            )

        expected = [
            c.block.handle for c in scheduler.select_candidates(projection)[:8]
        ]

        decisions = scheduler.decide(projection, is_idle=False)

        assert len(decisions) == 8
        assert [d.handle for d in decisions] == expected

    def test_no_actions_when_zone_high_but_no_candidates(self):
        scheduler = PressureScheduler(context_limit=100)
        projection = Projection(turn=10)

        # High pressure from non-candidate regions only.
        add_block(
            projection,
            kind=ContentKind.SYSTEM,
            region=RegionID.SYSTEM,
            size_tokens=50,
            last_access_turn=0,
        )
        add_block(
            projection,
            kind=ContentKind.SYSTEM,
            region=RegionID.CURRENT,
            size_tokens=40,
            last_access_turn=0,
        )

        assert projection.total_tokens == 90
        assert scheduler.read_pressure(projection).zone == PressureZone.CRITICAL
        assert scheduler.decide(projection, is_idle=False) == []
