# Implementation Plan: Unified State Update via Tool Callback

**Date:** 2026-03-28
**Status:** Analysis of current implementation and next steps

## Current State Analysis

Based on the recent commit (32e280b) and specification documents, the project has begun implementing the "Unified State Update via Tool Callback" design described in `docs/superpowers/specs/2026-03-28-tensor-tool-callback-design.md`.

### What Has Been Implemented

1. **Core Pydantic Schema** (`src/tinkuy/taste_gateway/tensor_protocol.py`)
   - ✅ Complete StateUpdate model with all cognitive regions
   - ✅ Tool definition generator for `_tinkuy_state_update`
   - ✅ State protocol instructions (replacing XML-based approach)
   - ✅ Session tag handling for conversation continuity
   - ✅ JSON serialization for tensor state

2. **Gateway Foundation** (`src/tinkuy/taste_gateway/gateway.py`)
   - ✅ Harness feedback generation (ported from taste.py)
   - ✅ Default-stable update application logic
   - ✅ Partial implementation foundation

3. **Server Integration** (`src/tinkuy/gateway/server.py`)
   - ✅ TasteGateway import and configuration setup

### What Remains To Be Implemented

## Phase 1: Complete the Gateway Implementation (High Priority)

### 1.1 Memory Object Management
**Files:** `src/tinkuy/taste_gateway/gateway.py`

The gateway currently has feedback generation but lacks the MemoryObject and MemoryStore implementation that the specs reference.

**Required Components:**
- `MemoryObject` class with fields: id, content, tokens, state, pinned, turn
- `MemoryStore` class for managing memory object lifecycle
- Memory action processing (summarize/release/pin) in the state update handler

### 1.2 Request Preparation Pipeline
**Files:** `src/tinkuy/taste_gateway/gateway.py`

**Components needed:**
- `prepare_request()` - main entry point
- Tool injection logic - append `_tinkuy_state_update` to client tools
- System prompt synthesis with cache breakpoints (R1/R2 layout)
- Current state presentation as JSON in system prompt
- Synthetic tool_result injection for harness feedback

### 1.3 Response Processing Pipeline
**Files:** `src/tinkuy/taste_gateway/gateway.py`, likely new `stream.py`

**Components needed:**
- Stream handler to intercept `_tinkuy_state_update` tool calls
- Tool call extraction from content blocks (not text parsing)
- Stop reason rewriting (tool_use → end_turn when only state update)
- State update application and session persistence
- Harness feedback generation for next cycle

### 1.4 Session Management
**Files:** `src/tinkuy/taste_gateway/gateway.py`

**Components needed:**
- Session state persistence (tensor + memory objects)
- Session restoration from prior state
- Session tag propagation through responses
- Cycle counter management

## Phase 2: Integration and Configuration (Medium Priority)

### 2.1 Cache Breakpoint Strategy
**Reference:** Based on existing gateway patterns

Implement the R1/R2/R3/R4 cache layout:
- R1: Client system instructions (stable forever)
- R2: State JSON + protocol (stable within session)
- R3: Recent message history (if needed for cache continuity)
- R4: Current turn (uncached)

### 2.2 TasteGateway Class Interface
**Files:** `src/tinkuy/taste_gateway/__init__.py`, `gateway.py`

Create the main `TasteGateway` class that the server can instantiate:
- Constructor taking `TasteGatewayConfig`
- `process_request()` method matching existing gateway signature
- Configuration class for data directory, logging, etc.

### 2.3 CLI Integration
**Files:** `src/tinkuy/gateway/server.py`, CLI modules

Add `--taste` flag to `tinkuy serve` command:
- Route requests through TasteGateway instead of regular Gateway
- Maintain existing proxy/upstream behavior
- Preserve all existing CLI options

## Phase 3: Robustness and Validation (Medium Priority)

### 3.1 Error Handling
**Files:** `src/tinkuy/taste_gateway/gateway.py`

**Critical error scenarios:**
- Malformed tool input JSON (invalid schema)
- Missing or corrupted session state
- API errors during state persistence
- Tool call parsing failures

**Strategy:**
- Robust fallback: carry forward prior tensor unchanged on errors
- Comprehensive logging of all failure modes
- Graceful degradation rather than request failures

### 3.2 Tool Cycle Behavior
**Files:** `src/tinkuy/taste_gateway/gateway.py`

**Requirements:**
- Detect tool chains vs conversational turns
- Present state read-only during tool chains
- Suppress harness feedback during tool execution
- Handle mixed tool calls (state update + user tools)

### 3.3 Message Suffix Invariant
**Reference:** `docs/superpowers/specs/2026-03-25-message-suffix-invariant-design.md`

The gateway may need to implement proper message array handling to avoid API constraint violations:
- Support both single message (conversational) and three message (tool) patterns
- Preserve tool_use/tool_result relationships in message history

## Phase 4: Testing and Validation (High Priority)

### 4.1 Unit Tests
**Files:** `tests/taste_gateway/`

**Test coverage needed:**
- StateUpdate Pydantic model validation
- Tool definition schema generation
- State update parsing and application
- Memory action processing
- Session tag handling

### 4.2 Integration Tests
**Files:** `tests/integration/`

**Test scenarios:**
- Full request/response cycle through taste gateway
- State persistence across multiple turns
- Memory object lifecycle (create/summarize/release)
- Tool chain vs conversational turn detection
- Error recovery and fallback behavior

### 4.3 Compatibility Testing
**Files:** `tests/compatibility/`

**Requirements:**
- Existing tests should continue to pass (they don't test taste gateway directly)
- Claude Code client compatibility
- Response format validation
- API constraint compliance

## Phase 5: Documentation and Cleanup (Low Priority)

### 5.1 Implementation Documentation
**Files:** `docs/taste_gateway/`

- Architecture overview
- API reference for TasteGateway class
- Configuration guide
- Troubleshooting guide

### 5.2 Code Cleanup
**Target:** Remove legacy XML infrastructure

Once tool callback approach is validated, clean removal of:
- XML parsing in `tensor_protocol.py` (already done)
- Text-sniffing heuristics in stream handlers
- Legacy protocol strings and patterns

## Dependencies and Risks

### External Dependencies
- **Pydantic**: Core to the tool schema approach - version compatibility critical
- **Anthropic API**: Tool schema compliance and content block streaming behavior
- **Claude Code client**: Response format expectations and tool call handling

### Risk Factors

1. **Tool Call Interception Complexity**
   - Risk: Content block streaming and tool call extraction is untested
   - Mitigation: Start with simple test cases, robust error handling

2. **Session State Growth**
   - Risk: Memory objects and tensor state growing unbounded
   - Mitigation: Implement aggressive feedback and curation policies

3. **Cache Strategy Effectiveness**
   - Risk: Poor cache hit rates negating performance benefits
   - Mitigation: Instrument cache behavior, tune R3 window size

4. **Model Compliance**
   - Risk: Model not calling state update tool consistently
   - Mitigation: Clear protocol instructions, fallback behavior for missing updates

## Success Metrics

- [ ] All existing tests pass
- [ ] Claude Code integration works end-to-end
- [ ] State persistence survives session boundaries
- [ ] Memory curation reduces token usage over time
- [ ] Response quality maintained under dual task (responding + curating)
- [ ] Cache hit rates comparable to existing gateway
- [ ] JSONL telemetry captures complete request/response cycle

## Implementation Priority

**Phase 1** (Complete gateway implementation) is blocking - the gateway currently exists but is incomplete. **Phase 4** (testing) should run parallel to Phase 1 development to catch integration issues early. Phases 2-3 can be implemented incrementally once the core functionality is working.

The key insight from the specification: this is a clean break from XML protocols to tool-based structured updates. The implementation should focus on tool interception and state management rather than trying to preserve XML compatibility.