# Minimal Message Suffix Invariant

**Goal:** Define and implement `compute_message_suffix` — a function that extracts the minimal valid message array from client messages, satisfying all Anthropic API constraints while keeping projected content exclusively in system blocks.

**Context:** The system-block architecture (R1-R4) moves the entire conversation projection into stability-ordered system blocks. The messages array should contain only the structural minimum required by the API. Currently, the gateway either sends too little (empty messages → 400) or loses structure (tool_result without preceding tool_use → 400).

---

## The Invariant

The messages array is always **1 or 3 messages**:

### Case 1: No tool_results (human turn or text-only response)

```
[{role: user, content: <last user message content>}]
```

One message. The user's content goes verbatim.

### Case 2: Tool_results present (tool call turn)

```
[{role: user, content: "[continued]"},
 {role: assistant, content: [<tool_use blocks only from messages[-2]>]},
 {role: user, content: <last user message verbatim>}]
```

Three messages. The placeholder satisfies user-first and alternation. The assistant message is stripped to only `tool_use` blocks (no `thinking`, no `text`). The user message with `tool_result` blocks goes verbatim.

### Why this works

| API Constraint | How satisfied |
|---|---|
| First message must be `user` | Placeholder is `user` role (case 2), or real message (case 1) |
| Role alternation | 1 message trivially alternates; 3 messages alternate user/assistant/user |
| `tool_result` needs preceding `tool_use` | Assistant message contains matching `tool_use` blocks |
| `tool_result` ordering | User message passed verbatim, preserving client's ordering |
| Non-empty messages | Placeholder has content; real messages have content |

### Why O(1)

The key insight: we don't need to walk backward through the tool chain. Earlier tool_use/tool_result pairs are historical — they're projected into system blocks. We only need the *most recent* pair for API constraint satisfaction. The placeholder user message absorbs the "first message must be user" constraint without containing tool_results that would trigger further pairing validation.

### What gets stripped from the assistant message

| Block type | Keep? | Why |
|---|---|---|
| `tool_use` | Yes | Required for tool_result ID matching |
| `thinking` | No | Internal reasoning, not useful in history, often 1000+ tokens |
| `text` | No | Model's textual response — already ingested into projection |

---

## Integration Points

### In `_gateway.py`

The new function replaces both `_extract_raw_user_message` (the broken fix attempt) and the inline message construction in `_synthesize`.

**Current flow:**
1. `prepare_request` extracts `user_content` (text only, loses structure)
2. `_synthesize` wraps `user_content` in a single user message
3. Tool_results are lost → 400

**New flow:**
1. `prepare_request` calls `compute_message_suffix(client_messages)`
2. `_synthesize` uses the returned messages directly
3. No wrapping, no structure loss

### In `_synthesize`

Currently `_synthesize` constructs messages inline. With this change, the messages come pre-built from `compute_message_suffix`. The synthesizer's job for messages becomes: accept them and attach them to the payload.

### In `process_turn`

The `user_content` parameter is still needed for projection ingestion (the orchestrator needs the text content to build the projection). But the `raw_user_message` parameter (added by the previous fix attempt) is removed — `compute_message_suffix` handles the messages array independently.

### In `validate.py`

The existing validator already checks all the constraints this invariant satisfies. No changes needed — the validator becomes a safety net confirming the invariant holds.

---

## Function Specification

```python
def compute_message_suffix(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compute the minimal valid message suffix for the API.

    The system-block architecture puts the conversation projection in
    system blocks. The messages array contains only the structural
    minimum required by the Anthropic API.

    Returns 1 message (user turn) or 3 messages (tool_result turn).
    Never returns 0 or 2 messages.

    Invariants:
      - First message is always role=user
      - If tool_results present, preceding assistant has matching tool_use
      - Assistant messages stripped to tool_use blocks only
      - Result passes validate_anthropic_payload (messages portion)
    """
```

---

## Edge Cases

| Scenario | Handling |
|---|---|
| Empty messages list | Return `[{role: user, content: "[continued]"}]` — degenerate, shouldn't happen |
| Last message not user role | Return `[messages[-1]]` — pass through, let validator catch issues |
| messages[-2] not assistant role | Log warning, return `[messages[-1]]` — degrade to case 1 |
| Assistant has no tool_use blocks | Strip to empty content → log warning, use `[{type: text, text: "[no content]"}]` |
| Multiple tool_use in one assistant | Keep all — user message will have matching tool_results |
| tool_result with text content mixed | Preserve verbatim — client ordering is correct |

---

## Files to Change

| File | Change |
|---|---|
| `src/tinkuy/gateway/_gateway.py` | Add `compute_message_suffix`, wire into `prepare_request` and `_synthesize`, remove `_extract_raw_user_message` and `raw_user_message` parameter |
| `tests/test_message_suffix.py` | New: unit tests for the invariant |
| `src/tinkuy/formats/validate.py` | No changes — existing validator serves as safety net |

---

## What's NOT Changing

- System-block synthesizer (`system_blocks.py`) — it already produces the system array correctly
- Projection ingestion — `_extract_user_content` still extracts text for the orchestrator
- Bootstrap path — cold start ingestion is a separate flow
- Gemini adapter — separate message format entirely
- Validator — already encodes the constraints we're satisfying
