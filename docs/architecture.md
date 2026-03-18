# Projective Gateway Architecture

## Design principles

1. **The projection is the source of truth.** The gateway owns a structured
   projection of the conversation state. API payloads are generated FROM the
   projection, not from the client's message history.

2. **The gateway is an event orchestrator, not a proxy.** Messages arrive as
   events. The gateway examines each event, mutates the relevant projection
   region, checkpoints, generates the API payload, and computes the client
   response. It never passes messages through.

3. **The core gateway is client-independent.** The projection engine, region
   model, and API adapter must work without knowledge of Claude Code. Claude
   Code support is a client adapter plugin. Other clients (CLI harness, test
   fixtures, future interfaces) use different adapters.

4. **Proxy gravity warning.** Every prior implementation of this architecture
   drifted back to a proxy pattern — intercepting and decorating the client's
   message chain rather than owning the projection. This happens because
   proxying is easier to build incrementally. Resist it. If you find yourself
   writing code that modifies a client message, you are building a proxy.

5. **The orchestrator responds to event types, not message contents.** If the
   orchestrator inspects the text of a client message to decide how to
   structure the API call, that's proxy gravity. Client messages are opaque
   content that enters R4. The orchestrator sees events, not words.

6. **Nothing is pinned permanently.** Every piece of content in the projection
   is subject to eviction under sufficient pressure. Permanent pinning is a
   design mistake — it trades away the system's ability to adapt.

7. **Give the transformer a feedback channel.** This is an experimental
   protocol. Tell the model explicitly that the protocol is open to
   suggestions. Invite it to report what works, what doesn't, what's
   confusing, and what would make the projection more useful. The model is
   the primary consumer of the projection — its feedback is data.

## Three-layer architecture

```
┌─────────────────────────────────────────────────────────┐
│ Client Adapter                                          │
│   Translates client protocol into gateway events        │
│   Claude Code adapter: messages API → events            │
│   Test harness adapter: direct function calls → events  │
│   Responsible for stripping client artifacts on ingest   │
│   Responsible for formatting gateway output for client   │
└────────────────────────┬────────────────────────────────┘
                         │ events
┌────────────────────────▼────────────────────────────────┐
│ Projection Orchestrator                                 │
│   Owns the projection and its lifecycle                 │
│   Regions: system, domain durable, session durable,     │
│     session ephemeral, transient, current turn          │
│   Event loop:                                           │
│     1. Receive event                                    │
│     2. Examine: which regions does this event affect?   │
│     3. Mutate: update affected regions                  │
│     4. Log + checkpoint                                 │
│     5. Generate API payload from projection             │
│     6. Forward to API adapter                           │
│     7. Receive API response                             │
│     8. Ingest: mutate projection with response          │
│     9. Execute queued removals (per-region)             │
│    10. Log + checkpoint                                 │
│    11. Compute client response from projection          │
│    12. Return to client adapter                         │
└────────────────────────┬────────────────────────────────┘
                         │ payload
┌────────────────────────▼────────────────────────────────┐
│ API Adapter                                             │
│   Translates projection into provider API format        │
│   Anthropic adapter: regions → messages + cache_control  │
│   Places cache breakpoints at region boundaries         │
│   Strips protocol artifacts from API responses          │
│   Future: OpenAI adapter, local model adapter           │
└─────────────────────────────────────────────────────────┘
```

## Projection regions

The projection is a set of structured regions, each with different
stability and caching characteristics.

```
┌─────────────────────────────────────────────────────┐
│ R0: Tools                                           │
│   Tool definitions (framework + phantom tools)      │
│   Stability: session lifetime — never mutates       │
│   ── cache breakpoint 1 ──                          │
├─────────────────────────────────────────────────────┤
│ R1: System                                          │
│   Gateway system instructions                       │
│   Client-provided system context (absorbed, not     │
│   passed through — CLAUDE.md, MEMORY.md content     │
│   becomes part of the projection, not prefix tax)   │
│   Stability: session lifetime                       │
│   ── cache breakpoint 2 ──                          │
├─────────────────────────────────────────────────────┤
│ R2: Durable                                         │
│   Model-curated tensors with declared losses        │
│   Domain knowledge, session history (compressed)    │
│   Page table (transformer's memory map)             │
│   Stability: append-only (frozen tensors)           │
│   ── cache breakpoint 3 ──                          │
├─────────────────────────────────────────────────────┤
│ R3: Ephemeral                                       │
│   Recent tool results (aging toward eviction)       │
│   Recent conversation turns (not yet stabilized)    │
│   Content marked for removal (waste)                │
│   Stability: mutates every turn                     │
├─────────────────────────────────────────────────────┤
│ R4: Current turn                                    │
│   Current user message / tool results               │
│   Dynamic anchor (live status, pressure, feedback)  │
│   Stability: replaced every turn                    │
└─────────────────────────────────────────────────────┘
```

### Cache behavior by region

| Region | Mutates? | Cached? | Cost |
|--------|----------|---------|------|
| R0: Tools | Never | Always read | 0.1x |
| R1: System | Never | Always read | 0.1x |
| R2: Durable | Append-only | Read for existing, write for new | Amortized 0.1x |
| R3: Ephemeral | Every turn | Written every turn | 1.25x |
| R4: Current | Every turn | Never cached | 1.0x |

Optimization target: minimize R3+R4 tokens, maximize R0+R1+R2 tokens.
Content graduates from R3→R2 as it stabilizes. Eviction from R3
produces a tensor in R2 (the price of eviction).

## Event flow (detailed)

### Inbound (client → API)

```
Client message arrives
  │
  ├─ Client adapter strips client artifacts
  ├─ Client adapter emits event: { type, content, metadata }
  │
  ▼
Orchestrator receives event
  │
  ├─ Classify: which region(s) does this affect?
  │   - New user message → R4 (current turn)
  │   - Tool result → R3 (ephemeral) or R4
  │   - System context update → R1 (rare)
  │
  ├─ Mutate affected regions
  │   - Previous R4 content ages into R3
  │   - R3 waste check: any pending removals to execute?
  │   - Pressure check: need to request model eviction?
  │
  ├─ Log state before API call
  ├─ Checkpoint projection to disk
  │
  ├─ Generate API payload FROM projection
  │   - R0 → tools
  │   - R1 → system prompt
  │   - R2+R3+R4 → messages (structure determined by
  │     API adapter, not by client message history)
  │   - Place cache_control at region boundaries
  │
  ▼
API adapter sends to provider
```

### Outbound (API → client)

```
API response arrives
  │
  ├─ API adapter strips provider artifacts
  ├─ API adapter extracts: assistant text, tool calls,
  │   usage stats, stop reason
  │
  ▼
Orchestrator ingests response
  │
  ├─ Store assistant response in R3 (ephemeral)
  ├─ Process any cooperative memory signals from model
  │   - release requests → queue removal, require tensor
  │   - retain signals → cancel pending removal
  │   - yuyay responses → update page table
  │
  ├─ Execute queued removals (per-region)
  │   - Only removals whose tensor has been provided
  │   - Tensor → frozen in R2, original → page store
  │
  ├─ Log state after ingestion
  ├─ Checkpoint projection to disk
  │
  ├─ Compute client response FROM projection
  │   - Strip gateway protocol (tensor handles, yuyay tags,
  │     cooperative memory signals)
  │   - Format for client protocol
  │
  ▼
Client adapter sends to client
```

## Tensor lifecycle (price of eviction)

Content enters R2 only when the model produces a tensor for it.
The gateway will not evict without a tensor. This creates natural
back-pressure: the model spends output tokens compressing only
what it values enough to preserve.

```xml
<release handle="a3f2b901">
  <tensor>
    <content>Structured summary of what this contained</content>
    <declared_losses>What was not preserved and why</declared_losses>
  </tensor>
</release>
```

The verbatim original is always stored in the page store (storage
is cheap). The tensor goes into R2. The page table entry points at
both.

### Page table

The page table is the transformer's map of available memory. It
sits at the end of R2, right before cache breakpoint 3.

```
PageTableEntry:
  handle: str           # 8-char hex, content-addressed
  kind: "file" | "tool_result" | "conversation" | "tensor"
  label: str            # human-readable (~40 chars)
  status: "present" | "available" | "pending_removal"
  region: 2 | 3         # where the content currently lives
  size_tokens: int       # approximate token count
  fault_count: int       # times recalled after eviction
  age_turns: int         # turns since last access
```

### Status transitions

```
                ┌──────────────────┐
   arrives ──> │  present (R3)    │
                └────────┬─────────┘
                         │
             ┌───────────┴───────────┐
             │                       │
      model releases            gateway requests
      (with tensor)             (pressure/idle)
             │                       │
             v                       v
      ┌──────────────┐    ┌──────────────────┐
      │ available     │    │ pending_removal   │
      │ (tensor→R2)   │    │ (awaiting model   │
      └──────┬───────┘    │  cooperation)     │
             │            └────────┬──────────┘
         fault (recall)            │
             │              model provides tensor
             v                     │
       ┌──────────┐                v
       │ present   │        ┌──────────────┐
       │ (R3)      │        │ available     │
       └──────────┘        │ (tensor→R2)   │
                            └──────────────┘
```

## Eviction policy

Eviction is **pressure-gated**, not age-based.

| Pressure zone | Context usage | Eviction behavior |
|---------------|---------------|-------------------|
| Low | < 50% | No eviction. Hold everything. |
| Moderate | 50-70% | Schedule candidates. Execute only at idle boundaries. |
| Elevated | 70-85% | Request model cooperation (release signals). |
| Critical | > 85% | Gateway-initiated pending_removal. Aggressive. |

Age is a factor in *candidate selection*, not a trigger for eviction.

### Removal nominations

Content can be nominated for removal from multiple sources:

- **Model cooperation** (release signals with tensor)
- **Gateway heuristics** (age, size, content type)
- **Pressure scheduler** (context budget exhaustion)
- **Client adapter** (content the client has discarded)

All nominations are **advisory**. The space scheduler decides what
actually gets evicted and when, based on pressure zone, fault history,
and available tensors. The scheduler is the single authority.

### Accessed bit

The gateway MUST track access patterns for content in R2 and R3.
Currently only "fault count" (recalled after eviction) is tracked.
This is insufficient — we need:

- **Last access turn** — when was this content last referenced?
- **Access count** — how many times has it been referenced?
- **Access recency** — accessed in the last N turns?

These signals inform promotion (R3→R2), eviction candidate selection,
and help us understand workload patterns we don't yet have visibility
into. Collect the data now; optimize later when patterns emerge.

## Idle restructure

When the Anthropic cache expires (~5 min idle), the write cost is
the same whether we send the old layout or a clean one. Restructure
is free at idle boundaries:

- Execute all pending removals
- Promote surviving ephemeral content based on fault history
- Recompact R3
- Rebuild the projection from scratch

## Checkpoint and recovery

The projection is checkpointed to disk after every mutation cycle
(steps 4 and 10 in the event flow). A checkpoint contains:

- All region contents (serialized)
- Page table state
- Page store index (verbatim originals on disk)
- Session metadata (turn count, cumulative stats)

**Recovery**: load the last checkpoint, rebuild the projection,
continue. The gateway process is ephemeral — it can restart at
any turn boundary without session loss.

Recovery MUST be tested before the gateway is considered production.

## Waste tracking

Content in R3 marked `pending_removal` is waste.

```
waste_tokens: int       # estimated tokens of waste
cache_write_cost: float # cost to restructure now
idle_distance: float    # estimated time until next idle boundary
```

Restructure triggers:
- **Idle return** (cache expired): always restructure
- **Waste threshold**: waste_tokens > configurable limit
- **Pressure zone**: context enters elevated or critical

## What survives from Pichay

- Observability / telemetry harness
- Content classification (file / tool / conversation)
- Eviction vs GC distinction (faultable vs ephemeral)
- Cooperative memory DSL (drop / summarize / anchor / release)
- Session isolation via fingerprinting
- Page store (verbatim storage + fault tracking)

## What does NOT survive

- Message-level fingerprinting and mutation detection (proxy pattern)
- Age-based eviction (replaced by pressure-gated)
- Distance-based cache breakpoint placement (replaced by region-boundary)
- Passing client messages through to the API (proxy pattern)
- Decorating messages with tensor handles (proxy pattern)
- Inspecting client message content to decide API behavior (proxy pattern)

## Acceptance test

The gateway must sustain a session where the model builds the next
version of the gateway itself. Metrics:

- Cache hit rate > 95% in steady state
- Zero false-positive anti-injection rejections
- Checkpoint recovery tested: restart mid-session, verify continuity
- No proxy-pattern code: grep for message mutation, fail if found
- Checkpoint-only recovery: projection reconstructable from checkpoint
  alone, without replaying the client's message history. If recovery
  requires the client's messages, you have a proxy with checkpointing.
- Feedback channel: model can report protocol suggestions; at least
  one suggestion received and evaluated per test session

## Open questions

1. **R2 internal breakpoint**: If R2 grows past ~50K tokens, split it
   with the 4th breakpoint?

2. **Message structure constraints**: The Anthropic API requires
   alternating user/assistant messages. The API adapter must pack
   projection regions into valid alternation. Key insight: Claude Code
   already stuffs data into assistant blocks — synthesis from the
   projection is LESS work than trying to mutate Claude's message
   structure. The packing logic synthesizes clean alternation from
   region contents rather than preserving conversational structure.
   This is where proxy gravity is strongest — resist preserving the
   original message flow.

3. **System prompt absorption**: CLAUDE.md and MEMORY.md content
   absorbed into R1 at session start — validated that this doesn't
   break Claude Code's expectations?

4. **Client adapter contract**: What is the minimal event interface
   between the client adapter and the orchestrator? Design for
   testability — the test harness adapter should be trivial.
