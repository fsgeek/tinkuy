# Tinkuy Paper Outline (v2)

**Working title:** The Missing Memory Hierarchy: From Demand Paging to
Cooperative Virtual Memory for Transformer Context Windows

**Combines Pichay + Hamut'ay + Tinkuy into one narrative.**

**Venue target:** Systems conference (SOSP/OSDI) or ML-systems (MLSys)

---

## 1. Introduction

The Atlas computer (1961) introduced demand paging. Before that,
programmers managed overlays manually. LLM context windows are in the
overlay era — every tool, every message, manually assembled, no eviction,
no hierarchy.

This paper makes four contributions:
1. Empirical evidence that context management is VM (Pichay data)
2. The inverted cost model (keeping costs, faulting is cheap)
3. Cooperative memory management — the model participates in eviction
4. Episodic page tables — the indirect mapping table for transformers

## 2. The Problem: Context Windows as Mismanaged Memory

### 2.1 Empirical Waste Characterization
857 sessions, 4.45B tokens. 21.8% structural waste. Taxonomy:
tool schemas (11%), duplication (2.2%), stale results (8.7%).
(Pichay data, honestly reported)

### 2.2 The Inverted Cost Model
Keeping is expensive ($O(n^2)$ attention per turn), faulting is cheap
(one extra round trip). Changes the optimization objective from
"minimize faults" to "minimize total cost." FIFO works well because
aggressive eviction is correct by default.

### 2.3 Why Bigger Windows Don't Solve It
1M token windows = more RAM, not virtual memory. Waste scales linearly
with window size. The problem is architectural, not capacity.

## 3. System Design

### 3.1 Three Generations of Architecture

**Pichay (proxy):** Transparent HTTP proxy. Intercepts and modifies the
client's message array. Works but fragile — proxy gravity caused 4
failed iterations. The client owns the conversation; the proxy borrows it.

**Hamut'ay (cooperative proxy):** Introduced cooperative eviction. The
model produces tensors (compressed summaries with declared losses).
97.3% compression ratio across 159KB of content from 4 sessions.
Still a proxy — but now the model participates.

**Tinkuy (gateway):** The projection is the source of truth. API payloads
are synthesized from five stability regions, not modified from the client.
Anti-proxy-gravity as a structural discipline. The gateway owns the
conversation.

### 3.2 Five-Region Projection
R0 (Tools) → R1 (System) → R2 (Durable) → R3 (Ephemeral) → R4 (Current)
Stability gradient. Cache breakpoints at region boundaries.

### 3.3 Pressure-Gated Eviction
Four zones: LOW, MODERATE, ELEVATED, CRITICAL.
Triggered by context pressure, not age. Age informs candidate selection.

### 3.4 Cooperative Tensors
The model produces compressed summaries with declared losses. Eviction
costs output tokens (back-pressure). The gateway requires a tensor
before evicting — the model must cooperate.

Measured compression: 97.3% across Hamut'ay sessions.
This is unavailable in hardware VM — applications are non-cooperative.

### 3.5 Episodic Page Tables

**The flat page table problem:** One entry per content block. Grows
linearly. At 22 turns: 5,747 chars. Caused 26x token amplification
by changing model behavior (increased verbosity → larger context →
larger page table → feedback loop).

**Episodic coalescing:** Group entries by temporal proximity. Recent
and faulted entries get individual listings; older stable content
summarized as episodes. At 22 turns: 728 chars (87% reduction).
Eliminated amplification entirely.

**Analogy:** Multi-level page tables in hardware map sparse address
spaces. Episodic page tables map sparse *temporal* spaces. The levels
are temporal, not spatial. Older episodes consolidate; recent episodes
stay granular. Same principle, different dimension.

## 4. Evaluation

### 4.1 Controlled Evaluation Harness
Conversation driver → gateway → Anthropic API → gateway.
No HTTP server, no client protocol noise. Clean payloads.
Ablation methodology: same task, same model — vary one component.

### 4.2 Page Table Overhead (measured)

| Condition | API tokens (turn 22) | vs. Baseline |
|-----------|---------------------|--------------|
| Baseline (no gateway) | 4,638 | 1.0x |
| Flat page table | 176,944 | 38x |
| Coalesced page table | 4,365 | 0.94x |
| No page table | 3,809 | 0.82x |

Flat page table: catastrophic amplification.
Coalesced: cheaper than baseline (attention concentration effect).

### 4.3 Needle Retrieval Fidelity (measured, needs N>1)

Needle-in-haystack, 20 diverse padding turns, 64k max_tokens:
- Baseline: found needle, 76k API tokens consumed
- Full (coalesced): found needle, 25k API tokens consumed (68% savings)

**TODO:** Run N=10+ per condition for confidence intervals.

### 4.4 Tensor Compression (measured from Hamut'ay logs)

97.3% compression across 159KB of tensored content in 4 sessions.
Summaries preserve key findings, numbers, and decisions.

**TODO:** Quality evaluation — can the model answer questions about
tensored content as well as original? A/B comparison.

### 4.5 Coherence Retention: Provenance Under Pressure (measured)

Three drift dimensions, each producing a profile not a scalar:
- **FR (Factual Retention):** counterfactual rejection over distance
- **CR (Coherence Retention):** reasoning chain traversal over distance
- **WS (Working Set):** page reference patterns over time

CR evaluation methodology: seed a DAG of architectural decisions with
explicit dependencies (D0→D1→D2→D3, with branching). Each decision
has arbitrary rationale details not reconstructible from domain knowledge.
Pad with filler for distance. Probe with two question types:
direct-parent (one hop) and root-cause (full chain traversal).

**Passive CR results (measured, 2026-03-21):**

| Detail type | Baseline | Projection | Notes |
|------------|----------|------------|-------|
| Technical facts (90% reads, 10M vectors) | 3/3 | 3/3 | Reconstructible |
| Organizational constraints (3 engineers) | 2/3 | 1/3 | Fragile |
| Arbitrary rationale (4 months skill gap) | 2/3 | **0/8** | Dead |

Key finding: arbitrary rationale details — the kind only recoverable
by tracing the actual chain — decay to zero by turn 29 even with
full context preservation and no eviction. The model confabulates
technically-plausible rationales while losing the real constraints.
This is the FR-high-CR-low signature of compaction.

The projection does not make this worse through eviction (nothing was
evicted at these token volumes). The degradation is pure attention
decay. This means passive CR has a ceiling set by the model's natural
attention. Active CR — provenance page faults via dependency edges —
is the only mechanism that can exceed that ceiling.

**Active CR results (measured, 2026-03-21):**

Same task with cooperative memory protocol enabled. The model
spontaneously used the signal protocol:
- 10 declare signals (dependency edges emitted at decision time)
- 5 trace signals (provenance chain traversal requests)
- Multiple retain signals (actively protecting needed blocks)

| Turn | 90% | 3eng | 4mo | 10M | unix | 2ms | Signals |
|------|-----|------|-----|-----|------|-----|---------|
| 29 | - | - | - | + | + | - | 5 |
| 30 | - | - | - | + | + | - | 5 |
| 31 | + | - | - | + | + | + | 6 |
| 32 | + | + | **+** | + | + | + | 2 (traces) |
| 33 | + | + | **+** | + | + | + | 7 |

The model started probes without using trace (turns 29-31), got
partial results matching passive CR. On turn 32 it fired two trace
signals and **recovered all chain elements including "4 months"**.
The model learned to use the protocol mid-conversation.

**Passive CR: 0/8 on arbitrary rationale details.**
**Active CR: 2/2 on root-cause probes where trace was used.**

This is the paper's core result: cooperative provenance recovery
exceeds the ceiling set by the model's natural attention.

**Aggregation:** Per-axis floor constraints, not scalar collapse.
FR and CR are not fungible — a system with high FR and collapsed CR
is compaction by another name. Report min of components with bottleneck
identity for automated comparison; full profiles for paper presentation.

### 4.6 Trace-Driven Policy Simulation (planned)

Replay Hamut'ay conversation logs through Tinkuy. At each turn,
evict everything outside a small working set. Record what the model
faults back in. Use the reference string to evaluate policies:
FIFO, LRU, frequency-based, working-set, pressure-gated.

Key insight: the reference string depends on the page table interface.
Different interfaces produce different traces. This is why we must
evaluate interfaces, not just policies.

### 4.6 Address-Space Eval (planned)

50 facts planted across 200 turns. Context can hold ~40 turns.
Score = facts recovered when asked. Tests:
- No page table (model has no memory map)
- Flat page table (every entry visible)
- Episodic page table (coalesced)
- Hierarchical page table (multi-level)

This is the VM promise: unlimited virtual memory backed by limited
physical memory. The score measures interface quality.

### 4.7 Cross-Document Reasoning (planned)

Large codebase (>500k tokens). Task requires reasoning across multiple
files. Model must page in what it needs, work, release, page in next.
The scenario where VM is necessary, not just helpful.

## 5. Discussion

### 5.1 The Cooperative Advantage
Hardware applications can't help with eviction. LLMs can. The tensor
protocol is a new point in the design space.

### 5.2 Episodic Memory as Temporal Indirection
Hardware page tables: spatial. LLM page tables: temporal/episodic.
Same mechanism (indirection to handle sparse spaces), different dimension.

### 5.3 The Attention-Concentration Effect
Preliminary evidence: the managed projection uses fewer tokens and
preserves quality. Less context → better attention → better quality.
Consistent with Pichay's observation that judges sometimes preferred
the evicted output.

### 5.4 The Interface Changes the Trace
In traditional OS, the reference string is program-determined. In our
system, the model's access patterns depend on what it sees in the page
table. The interface is not just a presentation — it changes behavior.
This argues for evaluating interfaces as first-class research objects.

### 5.5 The Provenance Advantage
Dependency edges are a second cooperative capability. The model declares
reasoning chains at decision time (when edges are fresh). Edges are
immutable (Paxos ideal: decisions don't change, they get superseded).
The page table surfaces edges, enabling graph traversal for provenance
reconstruction. Combined with vector similarity for cluster finding,
this gives: vectors narrow the search space, graphs navigate within it.

### 5.6 Limitations
- CR eval needs N>1 runs for statistical validity
- Needle test needs N>1 for confidence intervals
- Fault rate from Pichay was offline simulation, not live measurement
- Tensor quality not yet evaluated with downstream tasks
- Cooperative protocol depends on model capability (may not work with
  weaker models)
- Gemini integration built but not yet tested — cross-model claims
  are preliminary

## 6. Related Work

- Pichay (our prior work, if published separately; otherwise folded in)
- MemGPT / virtual context management
- RAG and retrieval-augmented approaches
- Context compression (LLMLingua, AutoCompressors, etc.)
- Classical VM literature (Denning, Belady, working set theory)
- Attention mechanisms and sparse attention

## 7. Future Work

- Multi-model evaluation (open-source via Codex/Gemini CLI forks)
- Cross-session memory (L4 — persistent across conversations)
- Self-hosting test (Tinkuy managing its own context window)
- Dirty-page detection (same file read multiple times as it evolves)
- Production deployment at scale

## 8. Conclusion

Context windows are L1 cache. The field builds bigger L1. We build
the hierarchy. Cooperative eviction with episodic page tables reduces
token consumption by 68% while preserving retrieval quality. The
missing memory hierarchy is buildable, and the model is a better
curator of its own memory than any external heuristic.

---

## Data We Have

- Pichay: 857 sessions, waste taxonomy, cost model
- Hamut'ay: 4 sessions with tensors, 97.3% compression;
  40+ cycle chat session at 10K avg tokens with stable identity
- Tinkuy: page table ablation (flat vs coalesced vs none),
  needle test (baseline vs full), token growth curves
- CR eval: passive CR baseline (4 conditions), active CR (in progress)
- Five integration surfaces: tinkuy-chat, eval harness, Claude Code,
  Gemini CLI (built), Aider (status TBD)

## Data We Need

1. Needle test N=10+ per condition (statistical validity)
2. CR eval N=3+ per condition (statistical validity)
3. Active CR results: does the model use declare/trace signals?
4. Tensor quality A/B (can model answer from tensor vs original?)
5. Trace-driven policy simulation from Hamut'ay logs
6. Gemini CLI validation (cross-model proof)
7. Pressure zone transitions in a long session (tinkuy-chat data)
