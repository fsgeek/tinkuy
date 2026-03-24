# SOSP Abstract Draft — Epstein: Why Is Forgetting Important?

## Draft 3 (2026-03-22)

### Title

Epstein: Why Is Forgetting Important?

Subtitle: Virtual Memory for Transformer Context Windows

### Abstract

Transformer context windows are managed as fixed-size append-only
buffers. When the buffer fills, systems either discard the
conversation, summarize it lossily, or demand a larger window —
the equivalent of buying more RAM rather than implementing virtual
memory. We find this analogy is not merely illustrative but
architectural: context windows are L1 caches that need a memory
hierarchy, not bigger caches.

We present Tinkuy, a virtual memory system for transformer context
windows. The system introduces cooperative paging, in which the
model participates in its own memory management by producing
structured tensor summaries with declared losses before eviction.
Tensor compression is stable over 60+ turns at 97.3% compression
ratio — the model converges on appropriate summaries without
divergence. An episodic page table tracks present and evicted
content, enabling demand paging at retrieval time.

Our key finding is that managed forgetting requires preserving
provenance, not just conclusions. We define three drift dimensions —
Factual Retention (does the model remember what was decided?),
Coherence Retention (does the model remember *why*?), and Working
Set trajectory (does the model reference the right content?) — and
show that existing approaches collapse to compaction: high FR, zero
CR. Arbitrary rationale details vanish within 30 turns even when
physically present in context. We introduce a dependency edge protocol
that captures reasoning chains at decision time and a provenance
page fault mechanism that reconstructs chains on demand — cooperative
capabilities unavailable in hardware VM. Active provenance recovery
achieves 2/2 on root-cause probes where passive CR scores 0/8.

We validate the system on AI coding agent workloads — tool-heavy,
long-running sessions that produce natural working-set locality.
A controlled evaluation harness provides ablation measurements.
A live coding agent session achieves 81% cache hit rate with the
projective gateway managing a 52K token projection from 101K tokens
of raw conversation. The provider-independent adapter architecture
supports multiple API formats, though empirical validation is on
the Anthropic API.

A surprising finding: the interface between the VM and the model
is not neutral. Episodic page tables reduce token consumption by
68% while preserving retrieval quality. Flat page tables cause
catastrophic 26x amplification. The model's access pattern depends
on what memory metadata it can see — an effect with no analog in
hardware VM. Similarly, the KV prefix cache creates a paging
constraint absent in hardware: eviction saves attention cost but
can invalidate the cached prefix, making remaining content more
expensive to process.

---

## Notes

- Word count: ~310. Needs trimming for some registration forms.
- "Epstein" title: dual meaning. Jeffrey Epstein (a fact that refuses
  to be forgotten under any amount of pressure) and Robert Epstein
  (memory researcher — forgetting as essential for generalization).
  The provocation is intentional.
- v3 changes: dropped "five independent integrations" claim. Now
  honest about validation scope: Anthropic API via Claude Code,
  eval harness, and tinkuy-chat. Gemini CLI broken, Aider pivoted
  to integration pattern. Cross-provider claims are architectural.
- CR data: passive CR shows arbitrary rationale decay to 0/8.
  Active CR with trace signals recovers all chain elements.
- "101K → 52K" from live session: the 2:1 compression ratio
  improves as sessions get longer (sublinear episode growth vs
  linear raw history growth).
- Cache economics: 31% → 81% hit rate with R3 breakpoint. This
  is the measurement that makes long sessions economically viable.
- The "no analog in hardware VM" insights are now three:
  1. Page tables that change program behavior
  2. Cooperative eviction and provenance-preserving tensors
  3. Cache prefix as a paging constraint (eviction can increase cost)
- 60+ turn tensor stability is new data since draft 2.
- The engineering difficulty (API layout constraints, tool pairing,
  checkpoint restart) is reported honestly as cost-of-abstraction,
  not swept under the rug.
