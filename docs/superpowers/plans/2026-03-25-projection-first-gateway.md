# Projection-First Gateway: Eliminate the Proxy Escape Hatch

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make it structurally impossible for the gateway to proxy client system blocks — all content flows through the projection, and `_synthesize()` never touches raw client data.

**Architecture:** Client system blocks are parsed into `InboundEvent`s (`SYSTEM_UPDATE`, `TOOL_DEFINITION`) and ingested into the projection's R1 (SYSTEM, TOOLS) regions by the orchestrator. The `client_system` parameter is deleted from `process_turn()` and `_synthesize()`. Memory protocol becomes a `ContentBlock` in R1 at gateway init. The synthesizer reads only from the projection — which is already how it works, except now R1 actually has content.

**Tech Stack:** Python, existing tinkuy projection/orchestrator/synthesizer infrastructure.

**Key invariant:** After this change, `_synthesize()` accepts no raw client data. Any future attempt to pass client blocks through requires adding a parameter — visible in review, grep-able, structurally obvious.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/tinkuy/gateway/_gateway.py` | Modify | Remove `client_system` plumbing, add ingestion of client system blocks into projection, move memory protocol to init |
| `src/tinkuy/core/orchestrator.py` | No change | Already routes `SYSTEM_UPDATE` → R1, `TOOL_DEFINITION` → R0. Just unused until now. |
| `src/tinkuy/core/regions.py` | No change | `Projection.add_content()` already handles R1 placement. |
| `src/tinkuy/formats/system_blocks.py` | No change | `_serialize_region(projection, RegionID.TOOLS, RegionID.SYSTEM)` already reads R1. Currently returns empty string because R1 is empty. After this change, it returns content. |
| `tests/test_projection_first.py` | Create | Tests that verify the structural impediment: no proxy path, R1 populated, client blocks absent from payload. |
| `tests/test_gateway.py` | Modify | Existing tests may need `client_system=` removal from `process_turn()` calls. |
| `tests/test_system_blocks.py` | No change | Already tests synthesis from projection content in R1. |

---

### Task 1: Parse Client System Blocks into InboundEvents

**Files:**
- Create: `tests/test_projection_first.py`
- Modify: `src/tinkuy/gateway/_gateway.py` (new function `_parse_client_system`)

The gateway currently extracts `client_system` as raw dicts and passes them through. We need a function that parses them into typed `InboundEvent`s that the orchestrator can place in R1.

Client system blocks from Claude Code are text blocks containing:
- Anthropic-injected instructions (system reminders, tool definitions preamble)
- CLAUDE.md content
- Skill content
- Session context

These are all stable across turns (~38k tokens, same content every turn). We don't need to parse them into subcategories yet — they all go into `RegionID.SYSTEM` as `ContentKind.SYSTEM` blocks.

**Fingerprint deduplication:** On the first turn, all client system blocks are ingested. On subsequent turns, we fingerprint (hash of text lengths, same as subagent detection) and skip ingestion if unchanged. This avoids re-adding ~38k tokens every turn.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_projection_first.py
"""Tests for projection-first gateway: no proxy escape hatch."""

from __future__ import annotations

from tinkuy.core.orchestrator import EventType, InboundEvent, Orchestrator
from tinkuy.core.regions import ContentKind, RegionID
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/tony/projects/tinkuy && uv run pytest tests/test_projection_first.py::test_parse_client_system_converts_text_blocks_to_inbound_events -v`
Expected: FAIL — `_parse_client_system` does not exist.

- [ ] **Step 3: Implement `_parse_client_system`**

Add to `src/tinkuy/gateway/_gateway.py` near the other `_extract_*` helper functions (around line 1220):

```python
def _parse_client_system(
    client_system: list[dict[str, Any] | str],
) -> list[InboundEvent]:
    """Parse client system blocks into InboundEvents for projection ingestion.

    All client system blocks become SYSTEM_UPDATE events placed in R1.
    cache_control is stripped — the gateway owns cache placement.
    This is the ingestion side of the anti-proxy boundary.
    """
    events: list[InboundEvent] = []
    for i, block in enumerate(client_system):
        if isinstance(block, str):
            text = block
        elif isinstance(block, dict):
            text = block.get("text", "")
        else:
            continue
        if not text:
            continue
        events.append(InboundEvent(
            type=EventType.SYSTEM_UPDATE,
            content=text,
            label=f"client-system-{i}",
        ))
    return events
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/tony/projects/tinkuy && uv run pytest tests/test_projection_first.py -v`
Expected: All 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_projection_first.py src/tinkuy/gateway/_gateway.py
git commit -m "feat: parse client system blocks into InboundEvents for R1 ingestion"
```

---

### Task 2: Fingerprint-Based Deduplication for Client System Blocks

**Files:**
- Modify: `tests/test_projection_first.py` (add tests)
- Modify: `src/tinkuy/gateway/_gateway.py` (add fingerprint tracking to Gateway)

Client system blocks are ~38k tokens and identical across turns. We must not re-ingest them every turn. The gateway tracks a fingerprint (hash of block text lengths — same scheme as subagent detection) and only ingests on first turn or when the fingerprint changes.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_projection_first.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/tony/projects/tinkuy && uv run pytest tests/test_projection_first.py::test_client_system_ingested_on_first_turn_populates_r1 -v`
Expected: FAIL — `prepare_request` still passes client system through as raw blocks, R1 is empty.

- [ ] **Step 3: Implement fingerprint tracking and conditional ingestion**

In `Gateway.__init__` (around line 200 of `_gateway.py`), add:

```python
self._client_system_fingerprint: str | None = None
```

In `prepare_request()`, replace lines 577-583 (the `client_system` extraction and passthrough) with:

```python
# Ingest client system blocks into the projection (R1).
# Fingerprint-gated: skip if unchanged since last turn.
raw_system = client_body.get("system")
if isinstance(raw_system, str):
    raw_system = [{"type": "text", "text": raw_system}]
elif not isinstance(raw_system, list):
    raw_system = None

if raw_system:
    self._ingest_client_system(raw_system)
```

Add this method to the `Gateway` class:

```python
def _ingest_client_system(
    self, client_system: list[dict[str, Any] | str],
) -> None:
    """Ingest client system blocks into the projection's R1.

    Fingerprint-gated: only ingests on first call or when the
    system blocks change. This is the anti-proxy boundary — client
    system content enters the projection, never the payload.
    """
    # Fingerprint: hash of full content (not just lengths — content
    # can change at the same length, e.g. updated CLAUDE.md)
    hasher = hashlib.sha256()
    for block in client_system:
        if isinstance(block, str):
            hasher.update(block.encode())
        elif isinstance(block, dict):
            hasher.update(block.get("text", "").encode())
    fingerprint = hasher.hexdigest()[:16]

    if fingerprint == self._client_system_fingerprint:
        return  # Unchanged — skip re-ingestion

    self._client_system_fingerprint = fingerprint

    # Clear existing client system blocks from R1 (they're stale).
    # Preserve non-client blocks (e.g. memory-protocol).
    r1 = self.orchestrator.projection.region(RegionID.SYSTEM)
    r1.blocks = [b for b in r1.blocks
                 if not b.label.startswith("client-system-")]

    # Parse into events and route through the orchestrator's
    # existing SYSTEM_UPDATE → R1 placement path.
    events = _parse_client_system(client_system)
    for event in events:
        self.orchestrator._place_event(event)

    log.info(
        "ingested %d client system blocks into R1 (fingerprint=%s)",
        len(events), fingerprint,
    )
```

You'll need to add the `RegionID` and `ContentKind` imports at the top of `_gateway.py`:

```python
from tinkuy.core.regions import ContentKind, ContentStatus, Projection, RegionID
```

(Check if `ContentKind` and `RegionID` are already imported — `ContentStatus` and `Projection` are imported at line 95.)

**Note:** We call `orchestrator._place_event()` directly rather than going through `begin_turn()` because client system ingestion happens *before* the turn begins. The orchestrator's `_place_event` uses `_classify_event` which already maps `SYSTEM_UPDATE` → `RegionID.SYSTEM`. This is the existing routing that was never used until now.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/tony/projects/tinkuy && uv run pytest tests/test_projection_first.py -v`
Expected: All 6 PASS (3 from Task 1, 3 new).

- [ ] **Step 5: Commit**

```bash
git add tests/test_projection_first.py src/tinkuy/gateway/_gateway.py
git commit -m "feat: ingest client system blocks into R1 with fingerprint dedup"
```

---

### Task 3: Remove the Proxy Escape Hatch

**Files:**
- Modify: `tests/test_projection_first.py` (add structural impediment test)
- Modify: `src/tinkuy/gateway/_gateway.py` (delete `client_system` parameter)

This is the structural impediment. We delete the `client_system` parameter from `process_turn()` and `_synthesize()`, and remove the raw-block prepend logic from `_synthesize()`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_projection_first.py`:

```python
import inspect

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/tony/projects/tinkuy && uv run pytest tests/test_projection_first.py::test_synthesize_has_no_client_system_parameter -v`
Expected: FAIL — `client_system` parameter still exists.

- [ ] **Step 3: Remove the proxy path**

In `_synthesize()` (around line 407):

**Delete** the `client_system` parameter from the signature. The new signature:

```python
def _synthesize(
    self,
    format: APIFormat,
    message_suffix: list[dict[str, Any]] | None = None,
    user_content: str | None = None,
) -> dict[str, Any]:
```

**Delete** lines 440-455 — the entire `if client_system:` block. Leave the `_inject_memory_protocol_r1` call at line 462 for now — Task 4 handles that. Update the docstring to remove references to `client_system`.

In `process_turn()` (around line 334):

**Delete** the `client_system` parameter from the signature. **Delete** `client_system=client_system` from the `_synthesize()` call at line 391.

In `prepare_request()` (around line 590):

**Delete** `client_system=client_system` from the `process_turn()` call at line 594.

- [ ] **Step 4: Run all tests**

Run: `cd /home/tony/projects/tinkuy && uv run pytest tests/test_projection_first.py tests/test_gateway.py tests/test_system_blocks.py -v`
Expected: All PASS. Existing gateway tests that used `client_system` in `process_turn()` should still work because they either:
- Don't pass `client_system` (most tests), or
- Will need the parameter removed from their calls (fix any that break)

- [ ] **Step 5: Commit**

```bash
git add src/tinkuy/gateway/_gateway.py tests/test_projection_first.py tests/test_gateway.py
git commit -m "feat: delete client_system parameter — structural anti-proxy impediment"
```

---

### Task 4: Move Memory Protocol into the Projection

**Files:**
- Modify: `tests/test_projection_first.py` (add test)
- Modify: `src/tinkuy/gateway/_gateway.py` (init-time injection, delete `_inject_memory_protocol_r1`)

The memory protocol is stable forever. It belongs in R1 as a `ContentBlock`, added once at gateway init. No more post-synthesis payload mutation.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_projection_first.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/tony/projects/tinkuy && uv run pytest tests/test_projection_first.py::test_memory_protocol_is_in_r1_projection_not_injected -v`
Expected: FAIL — memory protocol is not in R1.

- [ ] **Step 3: Add memory protocol to R1 at init, delete the injection method**

In `Gateway.__init__`, after the orchestrator and adapters are set up, add:

```python
# Memory protocol is stable forever — it belongs in R1.
if not self.config.lightweight:
    self.orchestrator.projection.add_content(
        content=MEMORY_PROTOCOL,
        kind=ContentKind.SYSTEM,
        label="memory-protocol",
        region=RegionID.SYSTEM,
    )
```

**Delete** the `_inject_memory_protocol_r1` method entirely (lines 987-1014).

**Delete** the call to it in `_synthesize()` (already removed in Task 3, but verify).

Also handle the `resume()` classmethod — when resuming from checkpoint, the memory protocol is already in the snapshot's R1, so no re-injection needed. Verify this is the case; if the checkpoint was from a pre-projection-first session, the protocol won't be there. Add a check:

```python
# In resume(), after rebuilding adapters:
if not gw.config.lightweight:
    r1 = gw.orchestrator.projection.region(RegionID.SYSTEM)
    has_protocol = any(
        b.label == "memory-protocol" for b in r1.blocks
    )
    if not has_protocol:
        gw.orchestrator.projection.add_content(
            content=MEMORY_PROTOCOL,
            kind=ContentKind.SYSTEM,
            label="memory-protocol",
            region=RegionID.SYSTEM,
        )
```

- [ ] **Step 4: Run all tests**

Run: `cd /home/tony/projects/tinkuy && uv run pytest tests/ -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tinkuy/gateway/_gateway.py tests/test_projection_first.py
git commit -m "feat: memory protocol lives in R1 projection, delete injection method"
```

---

### Task 5: Update Fingerprint Dedup to Preserve Memory Protocol on R1 Clear

**Files:**
- Modify: `tests/test_projection_first.py` (add test)
- Modify: `src/tinkuy/gateway/_gateway.py` (verify `_ingest_client_system` preserves protocol)

When client system blocks change (rare), `_ingest_client_system` clears R1 and re-ingests. It must not delete the memory protocol block.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_projection_first.py`:

```python
def test_client_system_change_preserves_memory_protocol_in_r1():
    """When client system blocks change, memory protocol survives."""
    gw = Gateway(GatewayConfig(lightweight=False))

    body_v1 = {
        "system": [{"type": "text", "text": "Version 1"}],
        "messages": [{"role": "user", "content": "hello"}],
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 8096,
    }
    gw.prepare_request(body_v1)
    gw.ingest_response("ack")

    body_v2 = {
        "system": [{"type": "text", "text": "Version 2"}],
        "messages": [{"role": "user", "content": "hi"}],
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 8096,
    }
    gw.prepare_request(body_v2)

    r1 = gw.projection.region(RegionID.SYSTEM)
    r1_labels = [b.label for b in r1.present_blocks()]

    assert "memory-protocol" in r1_labels, "Memory protocol was deleted during R1 clear"
    r1_text = " ".join(b.content for b in r1.present_blocks())
    assert "Version 2" in r1_text, "New client system content missing"
    assert "Version 1" not in r1_text, "Old client system content not cleared"
```

- [ ] **Step 2: Run test to verify it passes (already handled in Task 2's implementation)**

Run: `cd /home/tony/projects/tinkuy && uv run pytest tests/test_projection_first.py::test_client_system_change_preserves_memory_protocol_in_r1 -v`
Expected: PASS — the filter in `_ingest_client_system` already preserves `memory-protocol` labels.

If it fails, the fix is in the R1 clear logic in `_ingest_client_system` — ensure the filter preserves non-client blocks:

```python
r1.blocks = [b for b in r1.blocks
             if not b.label.startswith("client-system-")]
```

(This is already in the Task 2 implementation.)

- [ ] **Step 3: Commit (if any changes needed)**

```bash
git add tests/test_projection_first.py
git commit -m "test: verify memory protocol survives R1 re-ingestion"
```

---

### Task 6: Verify End-to-End with Existing Test Suite

**Files:**
- Modify: `tests/test_gateway.py` (fix any broken calls)

Run the full test suite and fix any breakage from the removed `client_system` parameter.

- [ ] **Step 1: Run full test suite**

Run: `cd /home/tony/projects/tinkuy && uv run pytest tests/ -v`

- [ ] **Step 2: Fix any broken tests**

Likely issues:
- Tests calling `process_turn(client_system=...)` — remove that parameter
- Tests expecting empty R1 — they'll now have memory protocol content if `lightweight=False`
- Tests checking specific system block count or order — may need adjustment

For each broken test, the fix should be removing the proxy path, not re-adding it. If a test fundamentally depends on the proxy pattern, it should be rewritten to use projection-first.

- [ ] **Step 3: Run suite again to confirm green**

Run: `cd /home/tony/projects/tinkuy && uv run pytest tests/ -v`
Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "fix: update existing tests for projection-first gateway"
```

---

### Task 7: Live Validation with PRD Benchmark

**Files:** None (validation only)

Run the PRD benchmark to verify the gateway works end-to-end with real Claude Code traffic patterns.

- [ ] **Step 1: Run PRD benchmark if available**

Run: `cd /home/tony/projects/tinkuy && bash scripts/run-prd-benchmark.sh` (or equivalent exerciser)

If no benchmark script exists, construct a minimal exerciser run:

```bash
cd /home/tony/projects/tinkuy && uv run python -m tinkuy.exerciser --turns 5 --lightweight
```

- [ ] **Step 2: Verify R1 is populated in telemetry**

Check logs for:
- `"ingested N client system blocks into R1"` on first turn
- No re-ingestion on subsequent turns (fingerprint match)
- System blocks in the payload contain R1 content

- [ ] **Step 3: Commit any fixes**

```bash
git commit -m "fix: address issues found in live validation"
```

---

## Summary of Structural Changes

| Before | After |
|--------|-------|
| `_synthesize(client_system=...)` accepts raw blocks | `_synthesize()` reads only from projection |
| `process_turn(client_system=...)` passes through | `process_turn()` has no client data parameter |
| `prepare_request()` extracts and forwards raw system blocks | `prepare_request()` ingests into R1, forwards nothing |
| `_inject_memory_protocol_r1()` mutates payload post-synthesis | Memory protocol is a ContentBlock in R1 at init |
| R1 empty, client blocks prepended as raw dicts | R1 populated, synthesizer serializes from projection |

**The impediment:** After this change, there is no parameter, method, or code path through which raw client blocks can reach the API payload. Re-introducing one requires adding a parameter to `_synthesize()` — a reviewable, grep-able structural change, not an accidental drift.
