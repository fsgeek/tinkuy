# Unified State Update via Tool Callback

**Date:** 2026-03-28
**Status:** Design approved, ready for implementation

## Problem

The taste gateway uses two in-band XML protocols embedded in the model's response text:

1. `<yuyay-tensor>` — cognitive state updates (strands, losses, questions, tensions)
2. `<yuyay-memory>` — memory curation signals (summarize, release, pin)

This conflates control and data. The model discussing the protocol emits tag names that the stream handler and response parser match as control signals, causing response truncation and parse failures. The XML parsing infrastructure is ~300 lines of fragile regex and ElementTree code with heuristics to distinguish protocol output from conversational mentions.

Additionally, the model sometimes hits max_tokens mid-tensor-update, producing truncated XML that the parser rejects. The tensor update — the model's most important output — is lost.

## Solution

Replace both XML protocols with a single tool callback: `_tinkuy_state_update`. The gateway injects this tool into the client's tool list. The model calls it at turn boundaries. The gateway intercepts the tool_use, strips it from the client-visible response, and processes the structured JSON state update.

The tool_result return channel carries harness feedback (memory pressure, curation advisories) back to the model.

## Design Principles

- **Control/data separation.** State updates travel as structured tool calls, not as text in the response stream.
- **Late binding.** The Pydantic schema uses `extra='allow'`. The model can add fields we haven't defined. We carry them faithfully and log them. We don't interpret them yet.
- **Composition.** The tensor and memory curation are one state update, not two protocols. Composition of state regions is commutative so long as fields don't overlap.
- **Default-stable.** The `updated_regions` field declares which regions the model changed. Everything else is carried forward unchanged by the gateway.
- **Clean cut.** No XML fallback. No migration period. Remove all XML parsing infrastructure.

## Architecture

### The Tool

The gateway injects one tool into the client's tool list:

```json
{
  "name": "_tinkuy_state_update",
  "description": "Update your cognitive state. Call at the end of each conversational turn (not during tool chains). Fields you don't include in updated_regions are preserved unchanged.",
  "input_schema": { ... derived from Pydantic model ... }
}
```

### Pydantic Schema

```python
from pydantic import BaseModel, ConfigDict

class Strand(BaseModel):
    title: str
    content: str = ""
    depends_on: list[str] = []
    key_claims: list[dict] = []
    integration_losses: list[str] = []

class Loss(BaseModel):
    what_was_lost: str = ""
    why: str = ""
    shed_from: str = ""
    category: str = "context_pressure"

class Tension(BaseModel):
    tension_id: str
    framings: list[dict] = []
    cycles_held: int = 0
    touches_strands: list[str] = []
    what_would_collapse_it: str = ""

class MemoryAction(BaseModel):
    action: str  # "summarize" | "release" | "pin"
    id: str      # memory object id (e.g. "m3")
    content: str = ""   # summary text (for summarize)
    reason: str = ""    # why (for release)

class StateUpdate(BaseModel):
    model_config = ConfigDict(extra='allow')

    updated_regions: list[str]

    # Known cognitive regions — typed, optional
    strands: list[Strand] | None = None
    declared_losses: list[Loss] | None = None
    open_questions: list[str] | None = None
    unresolved_tensions: list[Tension] | None = None
    instructions_for_next: str | None = None
    overall_truth: float | None = None
    overall_indeterminacy: float | None = None
    overall_falsity: float | None = None
    feedback_to_harness: dict | None = None

    # Memory curation — unified into the same update
    memory_actions: list[MemoryAction] | None = None
```

The `extra='allow'` means the model can add arbitrary fields. They are carried forward, logged, and not interpreted by the gateway.

### Request Flow

On each request, `prepare_request` does:

1. **Inject the tool.** Append `_tinkuy_state_update` to the client's tool list with the JSON schema derived from `StateUpdate.model_json_schema()`. The tool definition is stable across calls.

2. **Present current state.** The system prompt contains:
   - Client system instructions (BP1 — cache breakpoint)
   - Current state as JSON (BP2)
   - Labeled memory objects (BP3)

3. **Inject synthetic tool_result.** If the prior cycle ended with a state update tool call, prepend a tool_result exchange to the messages array:
   ```
   messages:
     assistant: [tool_use: _tinkuy_state_update {...}]  ← stored on session from prior cycle
     user: [tool_result: {harness feedback JSON}]       ← synthetic, carries feedback
     ...current turn messages...
   ```
   This satisfies the API's requirement that every tool_use gets a tool_result, maintains message alternation, and gives the model its harness feedback.

   **Lifecycle:** When `process_response` intercepts a state update tool_use, it stores the tool_use_id and input on the session (`session.pending_state_tool_use`). On the next call to `prepare_request`, if `pending_state_tool_use` is set, the gateway constructs the assistant tool_use message and synthetic user tool_result, prepends them to the messages, and clears the pending flag.

4. **Build current turn messages.** Same as current: `_find_current_turn_start`, keep everything from there onward. `_repair_tool_orphans` for API constraints.

### Cache Breakpoint Layout

```
BP1: client system instructions          (stable forever)
BP2: state JSON                          (stable within turn — small, ~900 tokens)
BP3: labeled memory objects              (grows during tool chains)
BP4: advancing breakpoint in messages    (last completed tool exchange)
messages tail: current exchange          (uncached)
```

### Response Flow

1. **Stream handler.** Scans content blocks by type, not text. When it sees `content_block_start` with `type: "tool_use"` and `name: "_tinkuy_state_update"`:
   - Suppress that content block and all its deltas
   - Accumulate the tool input JSON internally
   - All other content blocks (text, other tool_use) pass through normally

2. **Stop reason.** If the state update was the model's only tool call, the API returns `stop_reason: "tool_use"`. The gateway rewrites this to `stop_reason: "end_turn"` before the client sees it.

3. **Post-stream processing.** After the stream completes, `process_response`:
   - Extracts the state update from the intercepted tool_use input (already valid JSON, validated by the API)
   - Parses it as a `StateUpdate` Pydantic model
   - Applies `updated_regions` logic (default-stable)
   - Applies memory actions (summarize/release/pin on session memory objects)
   - Stores updated state on the session
   - Generates harness feedback for the next cycle's tool_result
   - Logs everything to JSONL

### Tool Cycle Behavior

The existing `_is_tool_cycle` detection is unchanged. During tool chains:

- State is presented read-only in the system prompt
- `_tinkuy_state_update` is in the tool list but the protocol says "not during tool chains"
- If the model calls it anyway, the gateway processes it (no harm)
- Cycle counter only increments on human turns
- Harness feedback is suppressed

### Protocol Instructions

Replaces the current `TENSOR_PROTOCOL` string in `tensor_protocol.py`:

```
You are operating as a stateful processor. Your state is shown below
as JSON — it contains your accumulated cognitive understanding from
all prior turns.

You will see:
- Your current state (strands, questions, tensions, etc.)
- Prior tool outputs as labeled memory objects
- The current turn (user message + any tool chain in progress)

You will NOT see prior user messages or prior assistant responses.

AT THE END OF EACH CONVERSATIONAL TURN (not during tool chains), call
_tinkuy_state_update to persist what you've learned. Declare which
regions you're updating via updated_regions. Regions you don't list
are carried forward unchanged.

You can also curate memory objects via the memory_actions field:
- summarize: replace full content with your summary
- release: drop entirely (gone, no recall)
- pin: mark as important

Your state is working memory, not a monument. Update when the
conversation warrants it. Leave it alone when it doesn't.

Distinguish between empirical findings and your own speculation. When
consolidating strands, empirical findings are load-bearing — keep the
specific numbers and results even if you reorganize the framing.

You may add fields beyond the known regions. They will be carried
faithfully. We don't interpret them yet.
```

## What Gets Deleted

Clean cut. Remove entirely:

**From `tensor_protocol.py`:**
- `TENSOR_PROTOCOL` string (replaced by shorter protocol above)
- `_tensor_to_xml()` and `_esc()` — XML serialization
- `_TENSOR_PATTERN` regex and `parse_tensor_update()` — XML response parsing
- `_xml_to_tensor_update()` — XML→dict conversion
- `_MEMORY_PATTERN` regex and `parse_memory_signals()` — memory XML parsing

**From `gateway/server.py`:**
- The text-sniffing heuristics in `_TasteStreamHandler` (replaced by content block type check)

**From `gateway.py`:**
- Any XML rendering of the tensor state (replaced by `json.dumps`)

**Net effect:** ~300 lines of XML infrastructure replaced by ~50 lines of JSON handling and tool interception.

## What Is NOT Changed

- `_find_current_turn_start`, `_build_taste_messages`, `_is_tool_cycle` — message selection logic
- `_repair_tool_orphans` — API constraint handling
- `MemoryObject`, `MemoryStore`, `_render_memory_block` — labeled memory objects
- `_apply_updates` — default-stable region application (same logic, JSON input instead of XML)
- `_generate_feedback` — harness feedback generation (output moves to tool_result)
- `_log_cycle` — JSONL logging (same data, JSON source instead of XML)
- Session management, session tags, session restore
- Cache breakpoint strategy

## Harness Feedback via Tool Result

The synthetic tool_result carries harness feedback as JSON:

```json
{
  "status": "ok",
  "cycle": 35,
  "feedback": [
    "12 open questions is high. Curate below 10.",
    "Tension(s) held 10+ cycles: identity_vs_performance"
  ],
  "memory_summary": {
    "objects": 24,
    "tokens": 45000,
    "stale_count": 8
  }
}
```

This replaces the harness feedback section that currently lives in the system prompt. The system prompt gets lighter; the feedback arrives through the natural tool_result channel.

## Testing

The implementation should be validated by running the taste gateway (`tinkuy serve --taste`) and pointing Claude Code at it. The model should:

1. Receive the `_tinkuy_state_update` tool in its tool list
2. Call it at turn boundaries with a JSON state update
3. NOT call it during tool chains
4. Receive harness feedback in the tool_result
5. Not have its response truncated when discussing the protocol

All 187 existing tests should continue to pass (they don't test the taste gateway directly).

## Future Work

- **Model-designed regions.** The `extra='allow'` schema permits the model to add fields. A future experiment: tell the model it can extend its own state structure and observe what it creates.
- **Recall mechanism.** Released memory objects are currently gone. A future recall tool could page them back from the JSONL log.
- **Eval harness.** Wire the eval runner to the taste gateway for systematic testing (needle-in-haystack adapted for tensor-only model, coherence retention, etc.).
