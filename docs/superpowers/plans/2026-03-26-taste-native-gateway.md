# Taste-Native Projective Gateway

**Date:** 2026-03-26
**Status:** Design (pre-implementation)
**Author:** Claude (Opus 4.6), PI: Tony

## Premise

Three gholas have attempted to integrate Hamutay's self-curating tensor
(taste.py) into Tinkuy's existing projective gateway. All three stalled.
The existing gateway carries archaeological layers from proxy, message-based,
cooperative-signal, and pressure-scheduling approaches. Each integration
attempt spends most of its context reading the existing code and gets
captured by its architecture.

Tony's question: what if we don't integrate, but instead build a new
gateway that starts from the taste loop and adds only what's needed
to sit in the wire?

## What we know (empirically, not from ghola reports)

1. **Taste works.** 60-cycle conversation, 2,232-token tensor, 6 strands,
   24 cumulative losses. Subjectively coherent from the user's side.
   Total API payload ~5K tokens at cycle 60.

2. **The gateway reduces context.** Claude Code sessions show ~135%
   apparent context usage (from client's perspective) while the API
   sees ~75%. The gateway strips detritus the client accumulates.
   Pre-tensor baseline: 64% reduction (Arbiter), 48% (Aider).

3. **The auto/bio divergence is real.** Controlled experiment (n=1,
   Sonnet, 15 cycles) shows autobiographical curation achieves a
   phase transition (global compression) that biographical curation
   does not. Both conditions curate locally. Only AUTO integrates
   globally. The divergence may require confound-free replication.

4. **Haiku results are contaminated.** Prior claims that Haiku cannot
   produce valid tensors are based on confounded experiments. Haiku
   was successful in the original cooperative experiment. Clean
   replication needed.

## The taste loop

taste.py's core loop (from TasteSession.exchange):

```
1. Generate harness feedback based on tensor health
2. Build system prompt: protocol + prior tensor + feedback
3. Build messages: [user message]
4. API call (tool_use: think_and_respond)
5. Extract response + tensor update
6. Apply default-stable carry-forward (_apply_updates)
7. Accumulate declared losses
8. Log everything
```

The harness is ~200 lines of code around this loop. It handles:
- Default-stable semantics (unchanged regions carry forward)
- Loss history accumulation
- Integration loss tracking (per-strand micro-losses)
- Health feedback (strand count, question count, tensor size)
- Resume from log
- Full JSONL telemetry

It does NOT handle:
- Cache placement (no cache_control breakpoints)
- Multiple clients or formats
- Session restore from checkpoint
- Wire interception (it owns the API call directly)
- Tool schema passthrough (only tool is think_and_respond)

## Design: taste-native gateway

### Core idea

The gateway intercepts traffic between client (Claude Code, Aider, etc.)
and the Anthropic API. It injects the tensor protocol into the system
prompt and strips tensor updates from the response. The client never
sees the tensor machinery. The model does two jobs: respond to the user
AND curate its cognitive state.

### Architecture

```
Client (Claude Code)
  │
  │  raw request (system + messages + tools)
  ▼
┌──────────────────────────┐
│  Taste-Native Gateway    │
│                          │
│  1. Extract client system│ ─── R1 (stable, cached)
│  2. Inject tensor        │ ─── R2 (tensor, cached per-session)
│  3. Inject protocol      │ ─── R2 (protocol instructions)
│  4. Format messages      │ ─── R3/R4 (recent turns + current)
│  5. Place cache controls │ ─── breakpoints at tier boundaries
│  6. Forward to API       │
│                          │
│  On response:            │
│  7. Parse tensor update  │ ─── XML signals in response
│  8. Apply default-stable │ ─── carry forward unchanged regions
│  9. Strip signals        │ ─── client sees clean response
│  10. Log telemetry       │ ─── full request/response bodies
│  11. Return to client    │
└──────────────────────────┘
  │
  │  clean response (no tensor signals)
  ▼
Client
```

### What changes from taste.py

**taste.py owns the API call.** The gateway intercepts someone else's.
This means:

- The gateway doesn't control the tool schema. The client's tools
  must pass through. The tensor update can't use tool_use — it must
  use inline XML signals (which the existing gateway already parses).

- The gateway doesn't control the user message. It arrives from the
  client. The gateway adds tensor context to the system prompt, not
  to the messages.

- The gateway doesn't control max_tokens or model selection. These
  pass through from the client.

**taste.py uses tool_use for structured output.** The gateway must use
XML signals instead. This changes the tensor format from JSON schema
to XML, but the semantics are identical:

```xml
<yuyay-tensor>
  <updated-regions>strands, declared_losses</updated-regions>
  <strands>
    <strand title="..." depends-on="...">
      <content>...</content>
      <claim truth="0.9" indeterminacy="0.1" falsity="0.0">...</claim>
      <integration-loss>...</integration-loss>
    </strand>
  </strands>
  <declared-losses>
    <loss shed-from="..." category="context_pressure">
      <what>...</what>
      <why>...</why>
    </loss>
  </declared-losses>
</yuyay-tensor>
```

### What carries over from taste.py

- Default-stable semantics (_apply_updates logic)
- Health feedback (strand count, tensor size, stale questions)
- Loss history accumulation
- Integration loss tracking
- The system prompt (with adaptation for XML output format)
- JSONL telemetry (log everything)

### What carries over from the existing gateway

- Cache placement strategy (R1/R2/R3/R4 breakpoints)
- System block synthesis (stability-ordered)
- Client system block ingestion (fingerprint-gated dedup)
- Billing header passthrough
- Response streaming
- Thrashing detection
- Format detection (Anthropic vs Gemini)
- Wire-level telemetry

### What gets dropped

- Pressure scheduler (the tensor self-regulates via model curation)
- Eviction nomination/pending-removal pipeline
- Force-eviction with stub markers
- Cooperative signal protocol (replaced by tensor protocol)
- Page table with individual block entries (replaced by tensor)
- Sidecar/projector dispatch

### What might be needed but isn't clear yet

- **R3 management.** Taste has no R3 — the tensor IS the memory.
  The gateway might keep recent turns in R3 for cache prefix
  continuity (so the API doesn't re-process them). But the model's
  cognitive access to prior conversation goes through the tensor,
  not through R3. How many turns to keep in R3? This is an
  empirical question — measure cache hit rates.

- **Session restore.** When a new ghola (fresh model instance) takes
  over, it receives the tensor but not the KV cache. This is the
  biographical condition. The auto/bio experiment suggests it will
  curate differently. Is that acceptable? Is there a way to minimize
  the impact? This might be where R3 helps — a few recent turns
  provide narrative continuity even if the tensor provides cognitive
  continuity.

- **Multi-model.** The tensor protocol assumes the same model across
  turns. If the client switches models mid-session (Claude Code does
  this), the tensor needs to survive the transition. Probably fine —
  the tensor is text, any model can read it — but untested.

- **Tensor corruption.** What if the model produces malformed XML?
  taste.py has tool_use schema validation. XML parsing is more
  fragile. Need robust fallback: if parsing fails, carry forward
  the prior tensor unchanged and log the failure.

- **Response quality under dual task.** Taste proves the model can
  respond AND curate simultaneously with no quality degradation
  (judge scores 4.7-4.9). But Taste's responses are conversational.
  Claude Code responses include code blocks, tool calls, structured
  output. Does the dual task interfere with complex tool use? This
  needs testing.

## Relationship to existing gateway

The existing gateway (`src/tinkuy/gateway/`) is not replaced. It
continues to work for clients that don't need the tensor. The
taste-native gateway is a new module — possibly `src/tinkuy/taste_gateway/`
or a separate package — that shares infrastructure (format adapters,
cache placement, streaming) but has its own request loop.

Over time, if the taste-native approach proves superior, the existing
gateway's eviction machinery becomes dead code. But that's a finding,
not an assumption.

## Experimental validation needed

Before building this, two experiments would increase confidence:

### Experiment 1: Belief vs Substrate (2x2)

Does the auto/bio divergence come from the KV cache trace or from
the model's belief about whose content it's curating?

| | Told "yours" | Told "someone else's" |
|--|--|--|
| Actually generated it | AUTO baseline | Belief test A |
| Injected history | Belief test B | BIO baseline |

If the diagonal dominates (substrate matters more than framing),
the gateway's biographical condition at session restore is a real
limitation. If framing dominates, we can mitigate by telling the
model "this is your tensor from your prior session."

### Experiment 2: Dual-task quality under Claude Code workload

Run taste.py with Claude Code-style prompts (code generation, tool
use, multi-step reasoning) instead of conversational prompts. Measure
whether tensor maintenance degrades code quality or vice versa.

### Experiment 3: Multi-provider tensor replication

Run taste.py against non-Anthropic models (Gemini, GLM, Kimi, MiniMax)
to determine whether self-curation is architecture-specific or general
to autoregressive transformers. This strengthens the paper regardless
of gateway integration.

## Open questions

1. Should the tensor use XML (wire-compatible) or JSON (schema-validatable)?
   Could we use a hybrid: XML on the wire, JSON internally?

2. How does the tensor interact with extended thinking? The model's
   thinking tokens aren't visible to the gateway. Does the tensor
   protocol need to be in the thinking block?

3. What's the right R3 window? Zero turns (pure tensor)? 3 turns
   (cache optimization)? Adaptive based on cache hit rate?

4. Should the tensor protocol be in R1 (stable, cached) or R2
   (durable, per-session)? If R1, it becomes part of the cache
   prefix and never needs re-processing. If R2, it can evolve
   per-session.

5. How does this interact with the Gemini format path? Gemini's
   cache semantics differ from Anthropic's. The tensor is
   format-agnostic but the placement strategy isn't.

## For the next ghola

Read taste.py first. Then this document. Then the gateway code.
The integration has stalled three times because each ghola read
the gateway first and tried to fit the tensor into its architecture.
The architecture should serve the tensor, not the other way around.

The key insight from the auto/bio experiment: the model is the only
entity that can do global integration. The gateway's job is to
present the tensor, accept updates, and handle the wire. Everything
else is the model's job. Trust it.
