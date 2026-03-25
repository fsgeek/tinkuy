# System-Block Architecture: Stability-Ordered Context Stack

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the user/assistant message-based projection with a stability-ordered stack of system blocks, eliminating role alternation overhead and integrating the self-curating tensor from taste.py.

**Architecture:** The entire projection becomes system blocks ordered by stability (R1→R4), with cache_control breakpoints between stability tiers. The current user/assistant synthesizer is replaced with a system-block synthesizer that serializes each region as text blocks. The self-curating tensor (taste.py) replaces block accumulation in R3 — the model integrates new content into the tensor each turn instead of appending blocks forever.

**Tech Stack:** Python, Anthropic API (system block arrays with cache_control), existing tinkuy core (orchestrator, regions, pressure), hamutay taste.py schema.

---

## Context

### Why This Change

The current synthesizer packs projection state into user/assistant messages. This costs:
- ~160 tokens per block in message framing (role tags, alternation enforcement)
- Engineering complexity: orphan repair, tool_use pairing, alternation enforcement
- Cache inefficiency: message content shifts position as blocks are added/evicted

The Anthropic API accepts an array of system blocks, each with individual cache_control. Prefix caching means everything before a cache_control breakpoint stays warm. By moving the entire projection into system blocks ordered by stability, we:
- Eliminate all message framing overhead
- Eliminate orphan repair and alternation enforcement
- Get optimal cache hit rates (stable content is always a prefix of volatile content)
- Simplify the synthesizer to "serialize regions, place breakpoints"

### The R1–R4 Stability Stack

```
R1: Stable forever (client instructions, tools)
    ↓ cache_control breakpoint
R2: Stable per-session (CLAUDE.md, memory protocol)
    ↓ cache_control breakpoint
R3: Curated per-human-turn (projected conversation strands from tensor)
    ↓ cache_control breakpoint
R4: Uncached, regenerated each call (state tensor metadata, page table)
<user> Current turn content
```

Cache behavior:
- R1 content: always cached (never changes)
- R2 content: cached until CLAUDE.md edit or session restart
- R3 content: cached across tool calls within a human turn (strands change only when human speaks)
- R4 content: never cached (changes every API call — tensor metadata, page table)
- User message: never cached (current turn)

### Self-Curating Tensor Integration

Currently: the projection accumulates conversation blocks in R3/R4 that grow forever. Eviction is pressure-gated but has no quality signal.

New: taste.py's self-curating tensor replaces block accumulation. Each human turn, the gateway feeds the new conversation content to the tensor update path. The tensor integrates, consolidates, declares losses. R3 contains the tensor's strands (stable across tool calls). R4 contains per-call metadata. The conversation doesn't grow — the tensor curates it.

The tensor schema (from taste.py) provides:
- **strands**: thematic threads with dependency edges and key claims
- **declared_losses**: what was dropped and why (visible compression)
- **open_questions**: curated, not accumulated
- **instructions_for_next**: branch prediction
- **epistemic values**: confidence tracking

### Key Design Decisions

1. **System blocks, not messages**: All projected content goes in system blocks. Only the current user turn is a message.
2. **Tensor replaces block accumulation**: No more individual conversation blocks aging through R3→eviction. The tensor is the curated summary.
3. **Cache breakpoints at stability boundaries**: Exactly 3 breakpoints (after R1, R2, R3). R4 and user message are never cached.
4. **Cooperative signals remain**: The model still emits release/retain/recall/declare/trace signals. These operate on tensor strands instead of conversation blocks.
5. **Page table simplifies**: Instead of listing individual blocks, the page table describes tensor strands and their metadata.
6. **Bootstrap changes**: Cold start ingests client history into the tensor (one-shot integration) instead of creating individual blocks.

### What's NOT Changing

- The orchestrator event loop (begin_turn, apply_decisions, ingest_response)
- The pressure scheduler (reads projection state, makes decisions)
- The region data model (ContentBlock, AccessRecord, RegionID)
- The server routing (session detection, passthrough for subagents)
- The store interfaces (checkpoint, page, tensor)
- The Gemini adapter (separate path)
- The cooperative memory protocol signals (just retargeted)

---

## File Structure

### Files to Create

| File | Responsibility |
|------|---------------|
| `src/tinkuy/formats/system_blocks.py` | New synthesizer: serialize projection regions as system blocks with cache_control breakpoints |
| `src/tinkuy/core/tensor_curator.py` | Adapter to run taste-style tensor curation within the gateway turn loop |
| `tests/test_system_blocks.py` | Tests for system-block synthesizer |
| `tests/test_tensor_curator.py` | Tests for tensor curation adapter |

### Files to Modify

| File | Changes |
|------|---------|
| `src/tinkuy/formats/anthropic.py` | Keep for reference/fallback; new synthesizer replaces it for system-block mode |
| `src/tinkuy/gateway/_gateway.py` | Wire new synthesizer and tensor curator into turn loop; simplify `_merge_system`; update bootstrap |
| `src/tinkuy/core/orchestrator.py` | Add tensor curation dispatch after response ingestion; adjust R3/R4 semantics |
| `src/tinkuy/core/regions.py` | Update region descriptions; R3 becomes "curated strands", R4 becomes "per-call metadata" |

---

## Task Breakdown

### Task 1: System-Block Synthesizer — Core Serialization

**Files:**
- Create: `src/tinkuy/formats/system_blocks.py`
- Create: `tests/test_system_blocks.py`

The synthesizer reads the projection and produces a list of system blocks (dicts with `type`, `text`, and optional `cache_control`). No messages, no alternation, no tool pairing.

- [ ] **Step 1: Write failing test for empty projection**

```python
# tests/test_system_blocks.py
from tinkuy.core.regions import Projection
from tinkuy.formats.system_blocks import SystemBlockSynthesizer

def test_empty_projection_produces_no_blocks():
    proj = Projection(turn=0)
    synth = SystemBlockSynthesizer(proj)
    result = synth.synthesize()
    assert result["system"] == []
    assert result["messages"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_system_blocks.py::test_empty_projection_produces_no_blocks -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write minimal SystemBlockSynthesizer**

```python
# src/tinkuy/formats/system_blocks.py
"""System-block synthesizer: stability-ordered context stack.

Serializes the projection as an array of system blocks with
cache_control breakpoints between stability tiers:

  R1 (stable forever) → cache_control → R2 (stable per-session)
  → cache_control → R3 (curated per-turn) → cache_control
  → R4 (uncached) + user message

No user/assistant alternation. No message framing. No orphan repair.
"""

from __future__ import annotations

import json
from typing import Any

from tinkuy.core.regions import (
    ContentStatus,
    Projection,
    RegionID,
)

_CACHE_BREAKPOINT = {"type": "ephemeral"}


class SystemBlockSynthesizer:
    """Serialize projection regions as system blocks."""

    def __init__(self, projection: Projection) -> None:
        self._proj = projection

    def synthesize(
        self,
        user_content: str | None = None,
        page_table: str | None = None,
    ) -> dict[str, Any]:
        """Produce a complete API payload.

        Returns:
            {"system": [...blocks...], "messages": [...]}
        """
        system: list[dict[str, Any]] = []

        # R1: tools + system (stable forever)
        r1_text = self._serialize_region(RegionID.TOOLS, RegionID.SYSTEM)
        if r1_text:
            system.append({"type": "text", "text": r1_text})

        # Breakpoint after R1
        if system:
            system[-1]["cache_control"] = _CACHE_BREAKPOINT

        # R2: durable (stable per-session)
        r2_text = self._serialize_region(RegionID.DURABLE)
        if r2_text:
            system.append({"type": "text", "text": r2_text})
            system[-1]["cache_control"] = _CACHE_BREAKPOINT

        # R3: ephemeral (curated per-human-turn — tensor strands)
        r3_text = self._serialize_region(RegionID.EPHEMERAL)
        if r3_text:
            system.append({"type": "text", "text": r3_text})
            system[-1]["cache_control"] = _CACHE_BREAKPOINT

        # R4: current (uncached — tensor metadata, page table)
        r4_parts = []
        r4_text = self._serialize_region(RegionID.CURRENT)
        if r4_text:
            r4_parts.append(r4_text)
        if page_table:
            r4_parts.append(page_table)
        if r4_parts:
            system.append({"type": "text", "text": "\n\n".join(r4_parts)})
            # No cache_control on R4 — never worth caching

        # Messages: just the current user turn
        messages: list[dict[str, Any]] = []
        if user_content:
            messages.append({"role": "user", "content": user_content})

        return {"system": system, "messages": messages}

    def _serialize_region(self, *region_ids: RegionID) -> str:
        """Serialize all PRESENT blocks from the given regions as text."""
        parts: list[str] = []
        for rid in region_ids:
            region = self._proj.region(rid)
            for block in region.blocks:
                if block.status == ContentStatus.PRESENT:
                    parts.append(block.content)
        return "\n\n".join(parts) if parts else ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_system_blocks.py::test_empty_projection_produces_no_blocks -v`
Expected: PASS

- [ ] **Step 5: Write test for cache_control placement**

```python
def test_cache_breakpoints_between_regions():
    proj = Projection(turn=1)
    proj.add_content("tools here", kind=ContentKind.SYSTEM, label="tools", region=RegionID.TOOLS)
    proj.add_content("durable here", kind=ContentKind.TENSOR, label="durable", region=RegionID.DURABLE)
    proj.add_content("ephemeral here", kind=ContentKind.CONVERSATION, label="eph", region=RegionID.EPHEMERAL)
    proj.add_content("current here", kind=ContentKind.CONVERSATION, label="cur", region=RegionID.CURRENT)

    synth = SystemBlockSynthesizer(proj)
    result = synth.synthesize(user_content="hello")

    system = result["system"]
    # R1 block has cache_control
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    # R2 block has cache_control
    assert system[1]["cache_control"] == {"type": "ephemeral"}
    # R3 block has cache_control
    assert system[2]["cache_control"] == {"type": "ephemeral"}
    # R4 block does NOT have cache_control
    assert "cache_control" not in system[3]
    # One user message
    assert len(result["messages"]) == 1
    assert result["messages"][0]["content"] == "hello"
```

- [ ] **Step 6: Run test, verify pass**

Run: `uv run pytest tests/test_system_blocks.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/tinkuy/formats/system_blocks.py tests/test_system_blocks.py
git commit -m "feat: system-block synthesizer — stability-ordered context stack"
```

---

### Task 2: System-Block Synthesizer — Client System Prompt Merging

**Files:**
- Modify: `src/tinkuy/formats/system_blocks.py`
- Modify: `tests/test_system_blocks.py`

The client (Claude Code) sends its own system prompt blocks (instructions, tools, CLAUDE.md, skills). These must be placed in R1/R2 of the system block array, preserving their cache_control. The gateway's additions (memory protocol, page table, tensor) go after them.

- [ ] **Step 1: Write failing test for client system merging**

```python
def test_client_system_blocks_precede_gateway_blocks():
    proj = Projection(turn=1)
    proj.add_content("memory protocol", kind=ContentKind.SYSTEM, label="protocol", region=RegionID.SYSTEM)

    synth = SystemBlockSynthesizer(proj)
    client_system = [
        {"type": "text", "text": "client instructions", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "client CLAUDE.md"},
    ]
    result = synth.synthesize(client_system=client_system, user_content="hi")

    system = result["system"]
    # Client blocks come first
    assert "client instructions" in system[0]["text"]
    # Gateway additions come after
    texts = [b["text"] for b in system]
    assert any("memory protocol" in t for t in texts)
    # Client blocks appear before gateway blocks
    client_idx = next(i for i, b in enumerate(system) if "client instructions" in b["text"])
    gateway_idx = next(i for i, b in enumerate(system) if "memory protocol" in b["text"])
    assert client_idx < gateway_idx
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_system_blocks.py::test_client_system_blocks_precede_gateway_blocks -v`
Expected: FAIL — synthesize() doesn't accept client_system

- [ ] **Step 3: Add client_system parameter to synthesize()**

Update `SystemBlockSynthesizer.synthesize()` to accept `client_system: list[dict] | None = None` and place client blocks first in the R1 tier, stripping their cache_control (gateway is cache authority), then add gateway R1 content after.

- [ ] **Step 4: Run test to verify it passes**

- [ ] **Step 5: Write test for cache_control stripping on client blocks**

```python
def test_client_cache_control_stripped():
    """Gateway is cache authority — client cache_control is stripped."""
    proj = Projection(turn=1)
    synth = SystemBlockSynthesizer(proj)
    client_system = [
        {"type": "text", "text": "instructions", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
    ]
    result = synth.synthesize(client_system=client_system, user_content="hi")
    # Client's cache_control is stripped; gateway places its own breakpoints
    # The last R1 block gets a breakpoint, but not the client's original one
    for block in result["system"]:
        cc = block.get("cache_control", {})
        assert cc.get("ttl") is None  # no client TTLs survive
```

- [ ] **Step 6: Run test, verify pass**

- [ ] **Step 7: Commit**

```bash
git add src/tinkuy/formats/system_blocks.py tests/test_system_blocks.py
git commit -m "feat: merge client system blocks into stability stack"
```

---

### Task 3: Tensor Curator Adapter

**Files:**
- Create: `src/tinkuy/core/tensor_curator.py`
- Create: `tests/test_tensor_curator.py`

This adapter bridges taste.py's self-curating tensor with the gateway's turn loop. It maintains the tensor state and produces updated strands after each human turn. During tool-calling bursts (no new human content), the tensor is not updated — strands remain stable, enabling cache hits on R3.

- [ ] **Step 1: Write failing test for tensor initialization**

```python
# tests/test_tensor_curator.py
from tinkuy.core.tensor_curator import TensorCurator

def test_new_curator_has_no_tensor():
    curator = TensorCurator()
    assert curator.tensor is None
    assert curator.strands_text() == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tensor_curator.py::test_new_curator_has_no_tensor -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write minimal TensorCurator**

```python
# src/tinkuy/core/tensor_curator.py
"""Tensor curator: self-curating memory for the gateway.

Bridges taste.py's self-curating tensor with the gateway turn loop.
The curator maintains a tensor that integrates conversation content
and produces strands for R3 of the system-block stack.

The tensor is updated once per human turn, not per API call.
During tool-calling bursts, strands remain stable → cache hits on R3.
"""

from __future__ import annotations

import json
from typing import Any


class TensorCurator:
    """Maintains a self-curating tensor within the gateway."""

    def __init__(self, prior_tensor: dict | None = None) -> None:
        self._tensor: dict | None = prior_tensor
        self._cycle: int = prior_tensor.get("cycle", 0) if prior_tensor else 0

    @property
    def tensor(self) -> dict | None:
        return self._tensor

    @property
    def cycle(self) -> int:
        return self._cycle

    def strands_text(self) -> str:
        """Serialize current strands for R3 system block."""
        if self._tensor is None:
            return ""
        strands = self._tensor.get("strands", [])
        if not strands:
            return ""
        parts = []
        for s in strands:
            title = s.get("title", "untitled")
            content = s.get("content", "")
            deps = s.get("depends_on", [])
            dep_str = f" [depends on: {', '.join(deps)}]" if deps else ""
            parts.append(f"## {title}{dep_str}\n{content}")
        return "\n\n".join(parts)

    def tensor_metadata_text(self) -> str:
        """Serialize tensor metadata for R4 system block."""
        if self._tensor is None:
            return ""
        meta = {
            "cycle": self._tensor.get("cycle"),
            "n_strands": len(self._tensor.get("strands", [])),
            "open_questions": self._tensor.get("open_questions", []),
            "instructions_for_next": self._tensor.get("instructions_for_next", ""),
            "overall_truth": self._tensor.get("overall_truth"),
            "overall_indeterminacy": self._tensor.get("overall_indeterminacy"),
        }
        return f"<tensor-state>\n{json.dumps(meta, indent=2)}\n</tensor-state>"

    def update(self, raw_output: dict) -> str:
        """Apply model's tensor update and return the response text.

        Uses taste.py's default-stable semantics: only regions listed
        in updated_regions are replaced. Everything else carries forward.

        Args:
            raw_output: The model's structured output (think_and_respond schema)

        Returns:
            The response text from the model.
        """
        self._cycle += 1
        updated_regions = set(raw_output.get("updated_regions", []))

        if self._tensor is not None:
            tensor = dict(self._tensor)
        else:
            tensor = {}

        tensor["cycle"] = self._cycle

        # Apply declared updates
        for key in ["strands", "declared_losses", "open_questions"]:
            if key in updated_regions and key in raw_output:
                value = raw_output[key]
                if isinstance(value, list):
                    tensor[key] = value

        if "instructions_for_next" in updated_regions:
            tensor["instructions_for_next"] = raw_output.get(
                "instructions_for_next", ""
            )

        for key in ["overall_truth", "overall_indeterminacy", "overall_falsity"]:
            if key in raw_output:
                tensor[key] = raw_output[key]

        # Per-cycle fields
        if "feedback_to_harness" in updated_regions:
            tensor["feedback_to_harness"] = raw_output.get("feedback_to_harness")
        else:
            tensor.pop("feedback_to_harness", None)

        if "declared_losses" not in updated_regions:
            tensor["declared_losses"] = []

        # Strip per-cycle integration losses from strands
        for strand in tensor.get("strands", []):
            strand.pop("integration_losses", None)

        tensor.pop("response", None)
        tensor.pop("updated_regions", None)

        self._tensor = tensor
        return raw_output.get("response", "")

    def snapshot(self) -> dict | None:
        """Return tensor for checkpoint persistence."""
        return self._tensor
```

- [ ] **Step 4: Run test to verify it passes**

- [ ] **Step 5: Write test for tensor update cycle**

```python
def test_update_applies_strands():
    curator = TensorCurator()
    raw = {
        "response": "hello",
        "updated_regions": ["strands", "open_questions"],
        "strands": [
            {"title": "Architecture", "content": "System blocks replace messages", "key_claims": []},
        ],
        "open_questions": ["Does cache hit rate improve?"],
    }
    response = curator.update(raw)
    assert response == "hello"
    assert curator.cycle == 1
    assert len(curator.tensor["strands"]) == 1
    assert "Architecture" in curator.strands_text()
```

- [ ] **Step 6: Run test, verify pass**

- [ ] **Step 7: Write test for default-stable semantics**

```python
def test_default_stable_preserves_unmentioned_regions():
    curator = TensorCurator()
    # Cycle 1: initialize everything
    curator.update({
        "response": "init",
        "updated_regions": ["strands", "open_questions"],
        "strands": [{"title": "A", "content": "first", "key_claims": []}],
        "open_questions": ["Q1"],
    })
    # Cycle 2: only update strands, not open_questions
    curator.update({
        "response": "update",
        "updated_regions": ["strands"],
        "strands": [{"title": "A", "content": "revised", "key_claims": []}],
    })
    # open_questions carried forward unchanged
    assert curator.tensor["open_questions"] == ["Q1"]
    # strands updated
    assert curator.tensor["strands"][0]["content"] == "revised"
```

- [ ] **Step 8: Run test, verify pass**

- [ ] **Step 9: Commit**

```bash
git add src/tinkuy/core/tensor_curator.py tests/test_tensor_curator.py
git commit -m "feat: tensor curator adapter for gateway self-curation"
```

---

### Task 4: Wire System-Block Synthesizer into Gateway

**Files:**
- Modify: `src/tinkuy/gateway/_gateway.py`
- Modify: `tests/test_system_blocks.py`

Replace the current `_synthesize()` → `AnthropicLiveAdapter` path with the new `SystemBlockSynthesizer` for Anthropic requests. The old synthesizer remains available as fallback.

- [ ] **Step 1: Write integration test**

```python
# tests/test_system_blocks.py (append)
def test_gateway_uses_system_blocks():
    """Gateway produces system-block payload, not message-based."""
    from tinkuy.gateway._gateway import Gateway, GatewayConfig

    gw = Gateway(GatewayConfig())
    # Simulate a turn
    result = gw.process_turn(user_content="test message")
    payload = result.api_payload

    # System is an array of blocks, not a string
    assert isinstance(payload["system"], list)
    # Messages should be minimal (just user turn)
    user_msgs = [m for m in payload["messages"] if m["role"] == "user"]
    assert len(user_msgs) == 1
    assert "test message" in str(user_msgs[0]["content"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_system_blocks.py::test_gateway_uses_system_blocks -v`
Expected: FAIL — gateway still uses old synthesizer

- [ ] **Step 3: Update Gateway._synthesize() to use SystemBlockSynthesizer**

In `_gateway.py`, add a code path in `_synthesize()` that uses `SystemBlockSynthesizer` when format is ANTHROPIC. The synthesizer receives the projection, user_content, and page_table text. The gateway handles `_merge_system` by passing client system blocks to the synthesizer's `client_system` parameter instead of merging after synthesis.

Key changes:
- Import `SystemBlockSynthesizer`
- In `_synthesize()`: construct synthesizer, call `synthesize(client_system=..., user_content=..., page_table=...)`
- In `prepare_request()`: pass client system blocks to the turn instead of merging after
- Remove `_merge_system` call (no longer needed — synthesizer handles placement)

- [ ] **Step 4: Run test to verify it passes**

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: all pass (existing tests may need adjustment for new payload format)

- [ ] **Step 6: Fix any broken tests**

Existing tests that assert on message-based payload format will need updating. The key change: `payload["system"]` is now a list of blocks, and `payload["messages"]` is minimal.

- [ ] **Step 7: Commit**

```bash
git add src/tinkuy/gateway/_gateway.py tests/
git commit -m "feat: wire system-block synthesizer into gateway turn loop"
```

---

### Task 5: Wire Tensor Curator into Gateway

**Files:**
- Modify: `src/tinkuy/gateway/_gateway.py`
- Modify: `tests/test_tensor_curator.py`

The tensor curator is initialized on gateway creation and updated after each response ingestion (when the response contains a tensor update). The curator's strands go into R3, metadata into R4.

- [ ] **Step 1: Write integration test**

```python
# tests/test_tensor_curator.py (append)
def test_gateway_initializes_curator():
    from tinkuy.gateway._gateway import Gateway, GatewayConfig
    gw = Gateway(GatewayConfig())
    assert gw.tensor_curator is not None
    assert gw.tensor_curator.tensor is None  # no tensor yet
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Add TensorCurator to Gateway.__init__**

```python
# In Gateway.__init__:
from tinkuy.core.tensor_curator import TensorCurator
self.tensor_curator = TensorCurator()
```

- [ ] **Step 4: Run test to verify it passes**

- [ ] **Step 5: Write test for curator checkpoint persistence**

```python
def test_curator_survives_checkpoint():
    from tinkuy.gateway._gateway import Gateway, GatewayConfig
    gw = Gateway(GatewayConfig())
    # Simulate tensor update
    gw.tensor_curator.update({
        "response": "test",
        "updated_regions": ["strands"],
        "strands": [{"title": "Test", "content": "data", "key_claims": []}],
    })
    # Checkpoint should include tensor
    snapshot = gw.orchestrator.projection.snapshot()
    # Verify tensor can be restored
    # (Implementation detail: tensor stored in checkpoint alongside projection)
```

- [ ] **Step 6: Implement checkpoint integration**

Update `Gateway.checkpoint()` to include `tensor_curator.snapshot()` and `Gateway.resume()` to restore it.

- [ ] **Step 7: Run full test suite**

Run: `uv run pytest tests/ -x -q`

- [ ] **Step 8: Commit**

```bash
git add src/tinkuy/gateway/_gateway.py tests/test_tensor_curator.py
git commit -m "feat: wire tensor curator into gateway with checkpoint persistence"
```

---

### Task 6: Bootstrap Redesign — Cold Start with Tensor Integration

**Files:**
- Modify: `src/tinkuy/gateway/_gateway.py`
- Create: `tests/test_bootstrap.py`

On cold start, the client sends its full conversation history. Currently bootstrap ingests each message as a separate block. New behavior: ingest the history into the tensor in one shot, producing strands that go into R3.

- [ ] **Step 1: Write failing test**

```python
# tests/test_bootstrap.py
def test_bootstrap_produces_tensor_not_blocks():
    from tinkuy.gateway._gateway import Gateway, GatewayConfig

    gw = Gateway(GatewayConfig())
    # Simulate cold start with client history
    client_body = {
        "messages": [
            {"role": "user", "content": "What is the gateway?"},
            {"role": "assistant", "content": "It manages context."},
            {"role": "user", "content": "How does eviction work?"},
            {"role": "assistant", "content": "Pressure-gated."},
            {"role": "user", "content": "Tell me more."},
        ],
        "system": [{"type": "text", "text": "You are helpful."}],
    }
    gw._bootstrap_from_client(client_body["messages"], client_body)

    # Should have tensor strands, not individual message blocks
    assert gw.tensor_curator.tensor is not None
    r3_blocks = gw.orchestrator.projection.region(RegionID.EPHEMERAL).blocks
    # R3 should NOT have one block per message
    assert len(r3_blocks) < len(client_body["messages"])
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Implement tensor-based bootstrap**

Replace the current per-message ingestion in `_bootstrap_from_client` with:
1. Concatenate the conversation history into a summary prompt
2. Either: run one tensor curation call to produce initial strands (requires API call)
3. Or: create a simple initial tensor from the message content (no API call, just structure)

For the initial implementation, option 3 is simpler — create a single strand from the conversation history. The tensor will self-curate on subsequent turns.

- [ ] **Step 4: Run test to verify it passes**

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -x -q`

- [ ] **Step 6: Commit**

```bash
git add src/tinkuy/gateway/_gateway.py tests/test_bootstrap.py
git commit -m "feat: bootstrap produces tensor instead of individual blocks"
```

---

### Task 7: Update Page Table for Tensor-Based Projection

**Files:**
- Modify: `src/tinkuy/formats/system_blocks.py`
- Modify: `tests/test_system_blocks.py`

The page table currently lists individual content blocks. With the tensor architecture, it should describe tensor strands and their metadata instead.

- [ ] **Step 1: Write failing test**

```python
def test_page_table_describes_strands():
    curator = TensorCurator()
    curator.update({
        "response": "test",
        "updated_regions": ["strands"],
        "strands": [
            {"title": "Architecture", "content": "system blocks", "key_claims": [], "depends_on": ["Design"]},
            {"title": "Design", "content": "stability ordering", "key_claims": []},
        ],
    })
    page_table = SystemBlockSynthesizer.synthesize_page_table(curator)
    assert "Architecture" in page_table
    assert "Design" in page_table
    assert "depends_on" in page_table or "depends on" in page_table
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Implement tensor-aware page table**

Add a `synthesize_page_table(curator)` static method that produces XML describing the tensor's strands, their sizes, dependencies, and epistemic values.

- [ ] **Step 4: Run test to verify it passes**

- [ ] **Step 5: Commit**

```bash
git add src/tinkuy/formats/system_blocks.py tests/test_system_blocks.py
git commit -m "feat: tensor-aware page table for system-block synthesizer"
```

---

### Task 8: End-to-End Integration Test

**Files:**
- Create: `tests/test_system_block_integration.py`

Full round-trip: client request → gateway → system-block payload → verify structure.

- [ ] **Step 1: Write integration test**

```python
# tests/test_system_block_integration.py
def test_full_round_trip():
    """Client request produces system-block payload with correct structure."""
    from tinkuy.gateway._gateway import Gateway, GatewayConfig

    gw = Gateway(GatewayConfig())

    client_body = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 16000,
        "stream": True,
        "system": [
            {"type": "text", "text": "You are helpful.", "cache_control": {"type": "ephemeral"}},
        ],
        "messages": [
            {"role": "user", "content": "Hello"},
        ],
    }

    upstream = gw.prepare_request(client_body)

    # System is array of blocks
    assert isinstance(upstream["system"], list)
    assert len(upstream["system"]) >= 1

    # At least one block has cache_control
    has_cc = any("cache_control" in b for b in upstream["system"])
    assert has_cc

    # Messages minimal — just user turn
    assert len(upstream["messages"]) >= 1

    # Client fields preserved
    assert upstream["model"] == "claude-sonnet-4-20250514"
    assert upstream["max_tokens"] == 16000
    assert upstream["stream"] is True
```

- [ ] **Step 2: Run test**

Run: `uv run pytest tests/test_system_block_integration.py -v`

- [ ] **Step 3: Fix any issues, iterate**

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add tests/test_system_block_integration.py
git commit -m "test: end-to-end integration for system-block architecture"
```

---

## Deferred Work (Not In This Plan)

These items are intentionally excluded to keep scope manageable:

1. **Live tensor curation calls**: Currently the tensor curator updates from model output. Full taste-style curation (where the gateway makes an API call to update the tensor) is deferred. This requires a separate API call per human turn, which has cost/latency implications.

2. **Cooperative signal retargeting**: The release/retain/recall/declare/trace signals currently target content blocks. Retargeting them to tensor strands requires updating the signal parsing and handling. The signals continue to work on the existing block model.

3. **Gemini adapter**: The system-block layout is Anthropic-specific. The Gemini adapter has its own path and is not affected by this plan.

4. **Pressure scheduler updates**: The scheduler scores individual blocks. With the tensor architecture, scoring changes (strands vs blocks). Deferred until the basic architecture is validated.

5. **Hamutay projector integration**: The sidecar projector currently produces tensors from individual blocks. Integration with the self-curating tensor is a separate concern.

6. **Local model demonstration**: Running against a local torch model to demonstrate the API framing overhead. Noted for the paper but not implementation work.
