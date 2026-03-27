# Cache Optimization Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build three complementary cache optimization tools — an algebraic cost model, a trace replayer, and a live shadow synthesizer — that together provide structural understanding, validation, and ongoing data collection for Anthropic prefix cache optimization.

**Architecture:** The algebraic model is a pure-function library (`src/tinkuy/cache/algebra.py`) that computes steady-state costs from region sizes and mutation rates. The trace replayer (`src/tinkuy/cache/replay.py`) reads telemetry JSONL and simulates prefix cache behavior under different policies. The shadow synthesizer (`src/tinkuy/cache/shadow.py`) runs a second SystemBlockSynthesizer with an experimental policy alongside the live one, logging what it would have produced. All three are independent modules that share a common cost model.

**Tech Stack:** Python 3.14, dataclasses, existing telemetry JSONL format, existing SystemBlockSynthesizer/Orchestrator, pytest.

---

## File Structure

```
src/tinkuy/cache/
    __init__.py              — Package init, exports public API
    algebra.py               — Steady-state cost equations
    replay.py                — Trace replayer with prefix cache model
    shadow.py                — Live shadow synthesizer for A/B comparison
    pricing.py               — Anthropic pricing constants (shared)

tests/
    test_cache_algebra.py    — Tests for algebraic model
    test_cache_replay.py     — Tests for trace replayer
    test_cache_shadow.py     — Tests for shadow synthesizer
```

Existing files modified:
- `src/tinkuy/gateway/_gateway.py` — wire in shadow synthesizer (Task 5)

---

### Task 1: Pricing constants module

**Files:**
- Create: `src/tinkuy/cache/__init__.py`
- Create: `src/tinkuy/cache/pricing.py`
- Test: `tests/test_cache_algebra.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for cache pricing and algebraic cost model."""

from tinkuy.cache.pricing import sonnet_pricing, cost_per_turn


def test_sonnet_pricing_has_required_fields():
    p = sonnet_pricing()
    assert p.input_per_mtok > 0
    assert p.output_per_mtok > 0
    assert p.cache_read_per_mtok > 0
    assert p.cache_write_per_mtok > 0
    # Cache reads are cheaper than input
    assert p.cache_read_per_mtok < p.input_per_mtok
    # Cache writes are more expensive than input
    assert p.cache_write_per_mtok > p.input_per_mtok


def test_cost_per_turn_no_cache():
    """With zero cache, all input tokens are uncached."""
    p = sonnet_pricing()
    cost = cost_per_turn(
        input_tokens=10000,
        cache_read_tokens=0,
        cache_write_tokens=0,
        output_tokens=100,
        pricing=p,
    )
    expected = 10000 * p.input_per_mtok / 1e6 + 100 * p.output_per_mtok / 1e6
    assert abs(cost - expected) < 0.0001


def test_cost_per_turn_full_cache():
    """With full cache read, cost is dramatically lower."""
    p = sonnet_pricing()
    no_cache = cost_per_turn(10000, 0, 0, 100, p)
    full_cache = cost_per_turn(0, 10000, 0, 100, p)
    assert full_cache < no_cache * 0.2  # cache reads are 10% of input price
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cache_algebra.py -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: Write minimal implementation**

```python
# src/tinkuy/cache/__init__.py
"""Cache optimization tools for Tinkuy gateway."""

# src/tinkuy/cache/pricing.py
"""Anthropic API pricing constants and cost functions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Pricing:
    """Token pricing for a model. All prices per million tokens."""
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float
    cache_write_per_mtok: float


def sonnet_pricing() -> Pricing:
    """Claude Sonnet pricing as of 2026-03."""
    return Pricing(
        input_per_mtok=3.00,
        output_per_mtok=15.00,
        cache_read_per_mtok=0.30,
        cache_write_per_mtok=3.75,
    )


def cost_per_turn(
    input_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
    output_tokens: int,
    pricing: Pricing | None = None,
) -> float:
    """Compute dollar cost for a single API turn."""
    p = pricing or sonnet_pricing()
    return (
        input_tokens * p.input_per_mtok
        + cache_read_tokens * p.cache_read_per_mtok
        + cache_write_tokens * p.cache_write_per_mtok
        + output_tokens * p.output_per_mtok
    ) / 1_000_000
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cache_algebra.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/tinkuy/cache/__init__.py src/tinkuy/cache/pricing.py tests/test_cache_algebra.py
git commit -m "feat(cache): pricing constants and cost_per_turn function"
```

---

### Task 2: Algebraic steady-state cost model

**Files:**
- Create: `src/tinkuy/cache/algebra.py`
- Modify: `tests/test_cache_algebra.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cache_algebra.py`:

```python
from tinkuy.cache.algebra import (
    steady_state_cost,
    promotion_break_even_turns,
    RegionProfile,
)


def test_steady_state_cost_all_cached():
    """When all regions are stable, cost is dominated by cache reads."""
    profile = RegionProfile(
        r1_tokens=7000,
        r2_tokens=500,
        r3_tokens=0,
        r4_tokens=500,
        output_tokens=100,
    )
    cost = steady_state_cost(profile)
    # R1+R2 cached (read), R4 uncached (write first turn, read thereafter)
    # But R4 changes every turn so it's always write
    assert cost > 0
    # With no R3, most input is cache-read — should be cheap
    assert cost < 0.01  # less than 1 cent per turn


def test_steady_state_cost_large_r3_is_expensive():
    """Large R3 means large cache writes every turn."""
    small_r3 = RegionProfile(7000, 500, 1000, 500, 100)
    large_r3 = RegionProfile(7000, 500, 20000, 500, 100)
    assert steady_state_cost(large_r3) > steady_state_cost(small_r3)


def test_promotion_break_even_is_about_2_turns():
    """Promoting a block saves cache-write cost per turn.
    Break-even is when cumulative savings exceed the one-time cache miss."""
    turns = promotion_break_even_turns()
    # With Sonnet pricing: write=3.75, read=0.30, input=3.00
    # Promoting saves (write-read) per turn but costs one input miss
    # Break-even: ceil(input / (write - read)) = ceil(3.00 / 3.45) = 1
    # But the promoted content also shifts the cache boundary, causing
    # a full re-read of everything after it, so it's slightly more.
    assert 1 <= turns <= 4


def test_steady_state_cost_matches_manual_calculation():
    """Verify against hand-computed values."""
    from tinkuy.cache.pricing import sonnet_pricing
    p = sonnet_pricing()
    profile = RegionProfile(
        r1_tokens=29000,
        r2_tokens=500,
        r3_tokens=8000,
        r4_tokens=1000,
        output_tokens=100,
    )
    cost = steady_state_cost(profile, pricing=p)
    # R1+R2 = 29500 tokens cached read: 29500 * 0.30 / 1e6
    # R3 = 8000 tokens cache write: 8000 * 3.75 / 1e6
    # R4 = 1000 tokens uncached input: 1000 * 3.00 / 1e6
    # Output = 100 tokens: 100 * 15.00 / 1e6
    expected = (29500 * 0.30 + 8000 * 3.75 + 1000 * 3.00 + 100 * 15.00) / 1e6
    assert abs(cost - expected) < 0.0001
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cache_algebra.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Write implementation**

```python
# src/tinkuy/cache/algebra.py
"""Algebraic steady-state cost model for Anthropic prefix caching.

Models the per-turn cost as a function of region sizes, assuming:
- R1+R2 are fully cached (read) every turn after the first
- R3 is cache-written every turn (content changes per turn)
- R4 is uncached input every turn (changes every API call)

This gives structural insight into cache geometry: which region
sizes dominate cost, when promotion is worthwhile, what the
theoretical cost floor is for a given conversation shape.

The model assumes steady state — it does not account for cold
start, promotion oscillation, or non-stationary growth. Use the
trace replayer for those dynamics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from tinkuy.cache.pricing import Pricing, sonnet_pricing


@dataclass
class RegionProfile:
    """Token sizes for each cache stability region.

    Represents a snapshot of the projection layout at a point
    in the conversation.
    """
    r1_tokens: int    # System (stable forever)
    r2_tokens: int    # Durable (stable per-session, cached)
    r3_tokens: int    # Ephemeral (changes per turn, cache-written)
    r4_tokens: int    # Current (changes every call, uncached)
    output_tokens: int  # Average output per turn


def steady_state_cost(
    profile: RegionProfile,
    pricing: Pricing | None = None,
) -> float:
    """Compute the per-turn dollar cost at steady state.

    Assumes:
      - R1+R2 tokens are cache-read (stable prefix, already cached)
      - R3 tokens are cache-written (content changes, written to cache
        for potential intra-burst reuse, but counted as write cost)
      - R4 tokens are uncached input (new every call)

    Returns cost in dollars.
    """
    p = pricing or sonnet_pricing()
    cached_read = profile.r1_tokens + profile.r2_tokens
    cache_write = profile.r3_tokens
    uncached = profile.r4_tokens

    return (
        cached_read * p.cache_read_per_mtok
        + cache_write * p.cache_write_per_mtok
        + uncached * p.input_per_mtok
        + profile.output_tokens * p.output_per_mtok
    ) / 1_000_000


def promotion_break_even_turns(
    pricing: Pricing | None = None,
) -> int:
    """Minimum turns a promoted block must survive to pay for itself.

    When a block promotes from R3 to R2:
      - It stops costing cache_write per turn
      - It starts costing cache_read per turn (cheaper)
      - But the promotion shifts the R2 cache boundary, causing
        a one-time re-write of content after the boundary

    Simplified model: ignoring the boundary shift (which is amortized
    across all turns), break-even is when cumulative savings from
    (write - read) exceed zero — which is immediate since write > read.

    The real cost is the one-turn cache miss when the boundary moves.
    That costs input_price for the promoted block's tokens (it was
    in cache as R3 write, now it's re-read as uncached input for one
    turn). Break-even: ceil(input_price / (write_price - read_price)).
    """
    p = pricing or sonnet_pricing()
    savings_per_turn = p.cache_write_per_mtok - p.cache_read_per_mtok
    one_time_cost = p.input_per_mtok  # one turn of uncached input
    return math.ceil(one_time_cost / savings_per_turn)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cache_algebra.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/tinkuy/cache/algebra.py tests/test_cache_algebra.py
git commit -m "feat(cache): algebraic steady-state cost model and promotion break-even"
```

---

### Task 3: Telemetry trace loader

**Files:**
- Create: `src/tinkuy/cache/replay.py`
- Test: `tests/test_cache_replay.py`

This task builds the trace loading infrastructure. The actual cache simulation is Task 4.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for cache trace replayer."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from tinkuy.cache.replay import TurnSnapshot, load_trace


def _write_trace(tmp_path: Path, turns: list[dict]) -> Path:
    """Write a minimal telemetry JSONL file."""
    path = tmp_path / "telemetry.jsonl"
    with open(path, "w") as f:
        for t in turns:
            f.write(json.dumps(t) + "\n")
    return path


def _minimal_turn(
    turn: int,
    input_tokens: int = 500,
    cache_read: int = 0,
    cache_write: int = 10000,
    output_tokens: int = 100,
    r1: int = 7000,
    r2: int = 500,
    r3: int = 0,
    r4: int = 500,
) -> dict:
    return {
        "turn": turn,
        "session_id": "test",
        "timestamp": 1000.0 + turn,
        "request": {"model": "claude-sonnet-4-20250514"},
        "projection": {
            "total_tokens": r1 + r2 + r3 + r4,
            "regions": {
                "TOOLS": {"tokens": 0},
                "SYSTEM": {"tokens": r1},
                "DURABLE": {"tokens": r2},
                "EPHEMERAL": {"tokens": r3},
                "CURRENT": {"tokens": r4},
            },
        },
        "response": {
            "input_tokens": input_tokens,
            "cache_read_tokens": cache_read,
            "cache_create_tokens": cache_write,
            "output_tokens": output_tokens,
        },
    }


def test_load_trace_reads_jsonl(tmp_path):
    path = _write_trace(tmp_path, [
        _minimal_turn(1, input_tokens=600, cache_write=10000),
        _minimal_turn(2, input_tokens=400, cache_read=10000, cache_write=1000),
    ])
    trace = load_trace(path)
    assert len(trace) == 2
    assert isinstance(trace[0], TurnSnapshot)
    assert trace[0].turn == 1
    assert trace[1].cache_read_tokens == 10000


def test_load_trace_extracts_region_sizes(tmp_path):
    path = _write_trace(tmp_path, [
        _minimal_turn(1, r1=7000, r2=500, r3=2000, r4=800),
    ])
    trace = load_trace(path)
    assert trace[0].r1_tokens == 7000
    assert trace[0].r2_tokens == 500
    assert trace[0].r3_tokens == 2000
    assert trace[0].r4_tokens == 800


def test_load_trace_from_real_experiment():
    """Load a real telemetry file if available (skip if not)."""
    import pytest
    path = Path(".tinkuy-data/experiments/prd-promotion/sessions")
    sessions = list(path.glob("*/telemetry.jsonl")) if path.exists() else []
    if not sessions:
        pytest.skip("No experiment data available")
    trace = load_trace(sessions[0])
    assert len(trace) > 0
    assert all(t.turn > 0 for t in trace)
    assert all(t.output_tokens >= 0 for t in trace)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cache_replay.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Write implementation**

```python
# src/tinkuy/cache/replay.py
"""Trace replayer for cache cost analysis.

Loads telemetry JSONL traces and replays them through different
cache policies to compute costs. Each turn becomes a TurnSnapshot
with the region sizes and actual API token counts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TurnSnapshot:
    """One turn's cache-relevant data, extracted from telemetry."""
    turn: int
    timestamp: float

    # Region sizes at request time
    r1_tokens: int
    r2_tokens: int
    r3_tokens: int
    r4_tokens: int

    # Actual API response token counts
    input_tokens: int        # uncached input
    cache_read_tokens: int   # prefix cache hits
    cache_write_tokens: int  # new content written to cache
    output_tokens: int

    @property
    def total_input(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    @property
    def cache_hit_rate(self) -> float:
        total = self.total_input
        return self.cache_read_tokens / total if total > 0 else 0.0


def load_trace(path: str | Path) -> list[TurnSnapshot]:
    """Load a telemetry JSONL file into a list of TurnSnapshots."""
    path = Path(path)
    snapshots: list[TurnSnapshot] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            response = record.get("response", {})
            projection = record.get("projection", {})
            regions = projection.get("regions", {})

            snapshots.append(TurnSnapshot(
                turn=record.get("turn", 0),
                timestamp=record.get("timestamp", 0.0),
                r1_tokens=regions.get("SYSTEM", {}).get("tokens", 0),
                r2_tokens=regions.get("DURABLE", {}).get("tokens", 0),
                r3_tokens=regions.get("EPHEMERAL", {}).get("tokens", 0),
                r4_tokens=regions.get("CURRENT", {}).get("tokens", 0),
                input_tokens=response.get("input_tokens", 0),
                cache_read_tokens=response.get("cache_read_tokens", 0),
                cache_write_tokens=response.get("cache_create_tokens", 0),
                output_tokens=response.get("output_tokens", 0),
            ))

    return snapshots
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cache_replay.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/tinkuy/cache/replay.py tests/test_cache_replay.py
git commit -m "feat(cache): trace loader — TurnSnapshot from telemetry JSONL"
```

---

### Task 4: Trace cost analyzer and what-if simulator

**Files:**
- Modify: `src/tinkuy/cache/replay.py`
- Modify: `tests/test_cache_replay.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cache_replay.py`:

```python
from tinkuy.cache.replay import trace_cost, what_if_cost
from tinkuy.cache.algebra import RegionProfile


def test_trace_cost_sums_actual_costs(tmp_path):
    """trace_cost computes total dollar cost from actual API token counts."""
    path = _write_trace(tmp_path, [
        _minimal_turn(1, input_tokens=10000, cache_read=0, cache_write=30000, output_tokens=200),
        _minimal_turn(2, input_tokens=500, cache_read=30000, cache_write=5000, output_tokens=100),
    ])
    trace = load_trace(path)
    result = trace_cost(trace)
    assert result.total_cost > 0
    assert result.turns == 2
    assert result.total_input_tokens == 10500
    assert result.total_cache_read_tokens == 30000


def test_what_if_cost_uses_algebra_per_turn(tmp_path):
    """what_if_cost replaces actual token counts with algebraic model."""
    path = _write_trace(tmp_path, [
        _minimal_turn(1, r1=7000, r2=500, r3=5000, r4=500, output_tokens=100),
        _minimal_turn(2, r1=7000, r2=500, r3=6000, r4=600, output_tokens=100),
        _minimal_turn(3, r1=7000, r2=500, r3=7000, r4=700, output_tokens=100),
    ])
    trace = load_trace(path)
    result = what_if_cost(trace)
    assert result.turns == 3
    assert result.total_cost > 0
    # First turn has no cache reads (cold start)
    assert result.per_turn_costs[0] > result.per_turn_costs[2]


def test_what_if_with_promotion_is_cheaper(tmp_path):
    """Simulating promotion (moving tokens from R3 to R2) reduces cost."""
    # Without promotion: 5000 tokens in R3 every turn
    base_turns = [
        _minimal_turn(i, r1=7000, r2=500, r3=5000, r4=500, output_tokens=100)
        for i in range(1, 11)
    ]
    # With promotion: after turn 3, 3000 tokens move from R3 to R2
    promoted_turns = []
    for i in range(1, 11):
        if i <= 3:
            promoted_turns.append(
                _minimal_turn(i, r1=7000, r2=500, r3=5000, r4=500, output_tokens=100)
            )
        else:
            promoted_turns.append(
                _minimal_turn(i, r1=7000, r2=3500, r3=2000, r4=500, output_tokens=100)
            )

    base_path = _write_trace(tmp_path / "base", base_turns)
    promo_path = _write_trace(tmp_path / "promo", promoted_turns)

    base_result = what_if_cost(load_trace(base_path))
    promo_result = what_if_cost(load_trace(promo_path))

    assert promo_result.total_cost < base_result.total_cost
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cache_replay.py::test_trace_cost_sums_actual_costs -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Write implementation**

Append to `src/tinkuy/cache/replay.py`:

```python
from tinkuy.cache.algebra import RegionProfile, steady_state_cost
from tinkuy.cache.pricing import Pricing, sonnet_pricing, cost_per_turn


@dataclass
class TraceCostResult:
    """Summary of cost analysis over a trace."""
    turns: int
    total_cost: float
    total_input_tokens: int
    total_cache_read_tokens: int
    total_cache_write_tokens: int
    total_output_tokens: int
    per_turn_costs: list[float]

    @property
    def avg_cost_per_turn(self) -> float:
        return self.total_cost / self.turns if self.turns else 0.0

    @property
    def avg_cache_hit_rate(self) -> float:
        total = self.total_input_tokens + self.total_cache_read_tokens + self.total_cache_write_tokens
        return self.total_cache_read_tokens / total if total else 0.0


def trace_cost(
    trace: list[TurnSnapshot],
    pricing: Pricing | None = None,
) -> TraceCostResult:
    """Compute actual dollar costs from recorded API token counts."""
    p = pricing or sonnet_pricing()
    per_turn: list[float] = []
    total_input = 0
    total_read = 0
    total_write = 0
    total_output = 0

    for snap in trace:
        c = cost_per_turn(
            snap.input_tokens,
            snap.cache_read_tokens,
            snap.cache_write_tokens,
            snap.output_tokens,
            p,
        )
        per_turn.append(c)
        total_input += snap.input_tokens
        total_read += snap.cache_read_tokens
        total_write += snap.cache_write_tokens
        total_output += snap.output_tokens

    return TraceCostResult(
        turns=len(trace),
        total_cost=sum(per_turn),
        total_input_tokens=total_input,
        total_cache_read_tokens=total_read,
        total_cache_write_tokens=total_write,
        total_output_tokens=total_output,
        per_turn_costs=per_turn,
    )


def what_if_cost(
    trace: list[TurnSnapshot],
    pricing: Pricing | None = None,
) -> TraceCostResult:
    """Compute costs using the algebraic model on per-turn region sizes.

    Instead of using the actual API token counts, this computes what
    each turn *should* cost based on the region layout at that point.
    Turn 1 is treated as cold start (all tokens are cache-write).
    Subsequent turns use the steady-state model.
    """
    p = pricing or sonnet_pricing()
    per_turn: list[float] = []
    total_input = 0
    total_read = 0
    total_write = 0
    total_output = 0

    for i, snap in enumerate(trace):
        profile = RegionProfile(
            r1_tokens=snap.r1_tokens,
            r2_tokens=snap.r2_tokens,
            r3_tokens=snap.r3_tokens,
            r4_tokens=snap.r4_tokens,
            output_tokens=snap.output_tokens,
        )

        if i == 0:
            # Cold start: everything is cache-write + uncached
            all_input = profile.r1_tokens + profile.r2_tokens + profile.r3_tokens + profile.r4_tokens
            c = cost_per_turn(0, 0, all_input, snap.output_tokens, p)
            total_write += all_input
        else:
            c = steady_state_cost(profile, p)
            cached = profile.r1_tokens + profile.r2_tokens
            total_read += cached
            total_write += profile.r3_tokens
            total_input += profile.r4_tokens

        total_output += snap.output_tokens
        per_turn.append(c)

    return TraceCostResult(
        turns=len(trace),
        total_cost=sum(per_turn),
        total_input_tokens=total_input,
        total_cache_read_tokens=total_read,
        total_cache_write_tokens=total_write,
        total_output_tokens=total_output,
        per_turn_costs=per_turn,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cache_replay.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/tinkuy/cache/replay.py tests/test_cache_replay.py
git commit -m "feat(cache): trace cost analyzer and what-if simulator"
```

---

### Task 5: Live shadow synthesizer

**Files:**
- Create: `src/tinkuy/cache/shadow.py`
- Test: `tests/test_cache_shadow.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for live shadow synthesizer."""

from __future__ import annotations

from tinkuy.cache.shadow import ShadowSynthesizer, ShadowResult
from tinkuy.core.orchestrator import Orchestrator
from tinkuy.core.regions import ContentKind, Projection, RegionID
from tinkuy.formats.system_blocks import SystemBlockSynthesizer


def _setup_projection() -> tuple[Orchestrator, SystemBlockSynthesizer]:
    """Create an orchestrator with some content in each region."""
    proj = Projection()
    proj.add_content("system prompt", ContentKind.SYSTEM, "sys", RegionID.SYSTEM)
    proj.add_content("durable data", ContentKind.SYSTEM, "dur", RegionID.DURABLE)
    proj.add_content("ephemeral stuff " * 50, ContentKind.TOOL_RESULT, "eph", RegionID.EPHEMERAL)
    proj.add_content("current turn", ContentKind.CONVERSATION, "cur", RegionID.CURRENT)
    orch = Orchestrator(projection=proj)
    synth = SystemBlockSynthesizer(orch)
    return orch, synth


def test_shadow_compares_two_layouts():
    """Shadow synthesizer produces a comparison between baseline and experimental."""
    orch, baseline_synth = _setup_projection()
    shadow = ShadowSynthesizer(orch)
    result = shadow.compare(baseline_synth)
    assert isinstance(result, ShadowResult)
    assert result.baseline_cached_tokens >= 0
    assert result.experimental_cached_tokens >= 0
    assert result.baseline_system_blocks > 0


def test_shadow_logs_difference():
    """Shadow result can report the token difference."""
    orch, baseline_synth = _setup_projection()
    shadow = ShadowSynthesizer(orch)
    result = shadow.compare(baseline_synth)
    diff = result.cached_token_difference
    # Difference is experimental - baseline; sign depends on policy
    assert isinstance(diff, int)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cache_shadow.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Write implementation**

```python
# src/tinkuy/cache/shadow.py
"""Live shadow synthesizer for A/B cache policy comparison.

Runs a second synthesis pass with an experimental region layout
alongside the real synthesizer. Logs what the experimental policy
would have produced without sending it to the API. Zero API cost.

Usage in gateway:
    shadow = ShadowSynthesizer(orchestrator)
    result = shadow.compare(real_synthesizer)
    log.info("shadow: baseline=%d experimental=%d diff=%+d cached tokens",
             result.baseline_cached_tokens,
             result.experimental_cached_tokens,
             result.cached_token_difference)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from tinkuy.core.orchestrator import Orchestrator
from tinkuy.formats.system_blocks import SystemBlockSynthesizer

log = logging.getLogger(__name__)


@dataclass
class ShadowResult:
    """Comparison between baseline and experimental synthesis."""
    baseline_system_blocks: int
    baseline_cached_tokens: int
    baseline_uncached_tokens: int
    experimental_system_blocks: int
    experimental_cached_tokens: int
    experimental_uncached_tokens: int

    @property
    def cached_token_difference(self) -> int:
        """Positive = experimental caches more."""
        return self.experimental_cached_tokens - self.baseline_cached_tokens


def _count_cached_tokens(system_blocks: list[dict[str, Any]]) -> tuple[int, int]:
    """Count cached and uncached tokens in a system block array.

    Blocks with cache_control are assumed to be cache-eligible
    (prefix up to and including that block). Blocks without
    cache_control contribute to uncached tokens.

    Returns (cached_tokens, uncached_tokens).
    """
    cached = 0
    uncached = 0
    for block in system_blocks:
        text = block.get("text", "")
        tokens = len(text) // 4  # rough estimate
        if "cache_control" in block:
            cached += tokens
        else:
            uncached += tokens
    return cached, uncached


class ShadowSynthesizer:
    """Shadow synthesizer that compares cache layouts.

    Takes the same orchestrator as the real synthesizer but can
    apply different policies to see what they would produce.
    """

    def __init__(self, orchestrator: Orchestrator) -> None:
        self.orchestrator = orchestrator

    def compare(
        self,
        baseline: SystemBlockSynthesizer,
    ) -> ShadowResult:
        """Compare baseline synthesis against experimental.

        For now, the experimental policy is the same as baseline.
        This establishes the plumbing — the experimental policy
        will be swapped in as we test alternatives.
        """
        baseline_payload = baseline.synthesize(skip_page_table=True)
        baseline_system = baseline_payload.get("system", [])
        b_cached, b_uncached = _count_cached_tokens(baseline_system)

        # Experimental: same synthesizer for now (placeholder).
        # Future: different promotion thresholds, region assignments, etc.
        experimental_payload = baseline.synthesize(skip_page_table=True)
        experimental_system = experimental_payload.get("system", [])
        e_cached, e_uncached = _count_cached_tokens(experimental_system)

        return ShadowResult(
            baseline_system_blocks=len(baseline_system),
            baseline_cached_tokens=b_cached,
            baseline_uncached_tokens=b_uncached,
            experimental_system_blocks=len(experimental_system),
            experimental_cached_tokens=e_cached,
            experimental_uncached_tokens=e_uncached,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cache_shadow.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/tinkuy/cache/shadow.py tests/test_cache_shadow.py
git commit -m "feat(cache): live shadow synthesizer for A/B cache comparison"
```

---

### Task 6: Wire shadow into the gateway

**Files:**
- Modify: `src/tinkuy/gateway/_gateway.py`

- [ ] **Step 1: Add shadow synthesizer to gateway init**

In `_gateway.py`, after the existing synthesizer setup (~line 210):

```python
from tinkuy.cache.shadow import ShadowSynthesizer

# In __init__, after self._system_block = SystemBlockSynthesizer(...)
self._shadow = ShadowSynthesizer(self.orchestrator)
```

- [ ] **Step 2: Add shadow comparison to prepare_request**

In `prepare_request()`, after the upstream body is built (~line 670):

```python
# Shadow comparison: log what experimental policy would have produced.
shadow_result = self._shadow.compare(self._system_block)
if shadow_result.cached_token_difference != 0:
    log.info(
        "shadow: baseline=%d experimental=%d diff=%+d cached tokens",
        shadow_result.baseline_cached_tokens,
        shadow_result.experimental_cached_tokens,
        shadow_result.cached_token_difference,
    )
```

- [ ] **Step 3: Update resume() to rebuild shadow**

In `resume()`, after the adapters are rebuilt (~line 367):

```python
gw._shadow = ShadowSynthesizer(gw.orchestrator)
```

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All pass (189+)

- [ ] **Step 5: Commit**

```bash
git add src/tinkuy/gateway/_gateway.py
git commit -m "feat(cache): wire shadow synthesizer into gateway for live comparison"
```

---

### Task 7: CLI tool for trace analysis

**Files:**
- Create: `src/tinkuy/cache/__main__.py`

- [ ] **Step 1: Write the CLI entry point**

```python
# src/tinkuy/cache/__main__.py
"""CLI for cache cost analysis.

Usage:
    uv run python -m tinkuy.cache analyze <telemetry.jsonl>
    uv run python -m tinkuy.cache compare <trace1.jsonl> <trace2.jsonl>
    uv run python -m tinkuy.cache algebra --r1=29000 --r2=500 --r3=8000 --r4=1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tinkuy.cache.algebra import RegionProfile, steady_state_cost, promotion_break_even_turns
from tinkuy.cache.pricing import sonnet_pricing, cost_per_turn
from tinkuy.cache.replay import load_trace, trace_cost, what_if_cost


def cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze a single trace file."""
    trace = load_trace(args.trace)
    actual = trace_cost(trace)
    modeled = what_if_cost(trace)

    print(f"Trace: {args.trace}")
    print(f"Turns: {actual.turns}")
    print()
    print(f"{'':20s} {'Actual':>12s} {'Modeled':>12s}")
    print(f"{'Total cost':20s} ${actual.total_cost:>11.2f} ${modeled.total_cost:>11.2f}")
    print(f"{'Avg cost/turn':20s} ${actual.avg_cost_per_turn:>11.4f} ${modeled.avg_cost_per_turn:>11.4f}")
    print(f"{'Avg cache hit':20s} {actual.avg_cache_hit_rate:>11.0%} {modeled.avg_cache_hit_rate:>11.0%}")
    print(f"{'Input tokens':20s} {actual.total_input_tokens:>12,} {modeled.total_input_tokens:>12,}")
    print(f"{'Cache read tokens':20s} {actual.total_cache_read_tokens:>12,} {modeled.total_cache_read_tokens:>12,}")
    print(f"{'Cache write tokens':20s} {actual.total_cache_write_tokens:>12,} {modeled.total_cache_write_tokens:>12,}")
    print(f"{'Output tokens':20s} {actual.total_output_tokens:>12,} {modeled.total_output_tokens:>12,}")


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare two traces."""
    trace_a = load_trace(args.trace_a)
    trace_b = load_trace(args.trace_b)
    cost_a = trace_cost(trace_a)
    cost_b = trace_cost(trace_b)

    print(f"{'':20s} {'Trace A':>12s} {'Trace B':>12s} {'Diff':>12s}")
    print(f"{'Turns':20s} {cost_a.turns:>12d} {cost_b.turns:>12d} {cost_b.turns - cost_a.turns:>+12d}")
    print(f"{'Total cost':20s} ${cost_a.total_cost:>11.2f} ${cost_b.total_cost:>11.2f} ${cost_b.total_cost - cost_a.total_cost:>+11.2f}")
    print(f"{'Avg cost/turn':20s} ${cost_a.avg_cost_per_turn:>11.4f} ${cost_b.avg_cost_per_turn:>11.4f} ${cost_b.avg_cost_per_turn - cost_a.avg_cost_per_turn:>+11.4f}")
    print(f"{'Cache hit rate':20s} {cost_a.avg_cache_hit_rate:>11.0%} {cost_b.avg_cache_hit_rate:>11.0%}")


def cmd_algebra(args: argparse.Namespace) -> None:
    """Compute steady-state cost from region sizes."""
    profile = RegionProfile(
        r1_tokens=args.r1,
        r2_tokens=args.r2,
        r3_tokens=args.r3,
        r4_tokens=args.r4,
        output_tokens=args.output,
    )
    cost = steady_state_cost(profile)
    break_even = promotion_break_even_turns()

    print(f"Region layout: R1={args.r1:,} R2={args.r2:,} R3={args.r3:,} R4={args.r4:,}")
    print(f"Output: {args.output:,} tokens/turn")
    print(f"Steady-state cost: ${cost:.6f}/turn (${cost * 1000:.4f}/1000 turns)")
    print(f"Promotion break-even: {break_even} turn(s)")
    print()
    print(f"Cache read (R1+R2):  {args.r1 + args.r2:>10,} tokens @ $0.30/M = ${(args.r1 + args.r2) * 0.30 / 1e6:.6f}")
    print(f"Cache write (R3):    {args.r3:>10,} tokens @ $3.75/M = ${args.r3 * 3.75 / 1e6:.6f}")
    print(f"Uncached (R4):       {args.r4:>10,} tokens @ $3.00/M = ${args.r4 * 3.00 / 1e6:.6f}")
    print(f"Output:              {args.output:>10,} tokens @ $15.0/M = ${args.output * 15.0 / 1e6:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="tinkuy.cache", description="Cache cost analysis tools")
    sub = parser.add_subparsers(dest="command")

    p_analyze = sub.add_parser("analyze", help="Analyze a telemetry trace")
    p_analyze.add_argument("trace", type=Path)

    p_compare = sub.add_parser("compare", help="Compare two traces")
    p_compare.add_argument("trace_a", type=Path)
    p_compare.add_argument("trace_b", type=Path)

    p_algebra = sub.add_parser("algebra", help="Steady-state cost from region sizes")
    p_algebra.add_argument("--r1", type=int, default=7000, help="R1 (system) tokens")
    p_algebra.add_argument("--r2", type=int, default=500, help="R2 (durable) tokens")
    p_algebra.add_argument("--r3", type=int, default=8000, help="R3 (ephemeral) tokens")
    p_algebra.add_argument("--r4", type=int, default=1000, help="R4 (current) tokens")
    p_algebra.add_argument("--output", type=int, default=100, help="Output tokens/turn")

    args = parser.parse_args()
    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "algebra":
        cmd_algebra(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test with real data**

Run: `uv run python -m tinkuy.cache algebra --r1=29000 --r2=500 --r3=8000 --r4=1000`
Expected: Formatted output showing per-region costs and break-even.

Run: `uv run python -m tinkuy.cache analyze .tinkuy-data/experiments/prd-promotion/sessions/*/telemetry.jsonl`
Expected: Side-by-side actual vs modeled costs.

- [ ] **Step 3: Commit**

```bash
git add src/tinkuy/cache/__main__.py
git commit -m "feat(cache): CLI tool for trace analysis and algebraic cost modeling"
```

---

### Task 8: Validate against benchmark data

**Files:**
- No new files — this is a validation step

- [ ] **Step 1: Run algebra on the PRD benchmark's actual region sizes**

```bash
uv run python -m tinkuy.cache algebra --r1=6856 --r2=11648 --r3=8608 --r4=1000 --output=100
```

Compare the steady-state cost against the actual per-turn costs from the promotion benchmark.

- [ ] **Step 2: Run analyze on both benchmark traces**

```bash
uv run python -m tinkuy.cache analyze .tinkuy-data/experiments/prd-cache-fix/sessions/*/telemetry.jsonl
uv run python -m tinkuy.cache analyze .tinkuy-data/experiments/prd-promotion/sessions/*/telemetry.jsonl
```

- [ ] **Step 3: Run compare between them**

```bash
uv run python -m tinkuy.cache compare \
    .tinkuy-data/experiments/prd-cache-fix/sessions/*/telemetry.jsonl \
    .tinkuy-data/experiments/prd-promotion/sessions/*/telemetry.jsonl
```

- [ ] **Step 4: Record findings**

If the algebraic model diverges significantly from actual costs, document why (cold start, overhead tokens, promotion oscillation) and file as a known limitation.

- [ ] **Step 5: Commit any adjustments**

```bash
git add -A && git commit -m "fix(cache): calibrate cost model against benchmark data"
```
