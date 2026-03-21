# SOSP Abstract Draft — Epstein: Why Is Forgetting Important?

## Draft 2 (2026-03-21)

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
An episodic page table tracks present and evicted content, enabling
demand paging at retrieval time.

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
capabilities unavailable in hardware VM.

We demonstrate the system across five independent integrations:
a tensor-only chat interface sustaining 40+ cycles at 10K tokens
with stable identity; a controlled evaluation harness measuring
drift profiles; and three production tool integrations (Claude Code,
Gemini CLI, Aider) proving the abstraction is model-agnostic. A
50K token managed window sustains indefinite conversation while
unmanaged 200K windows degrade within hours.

A surprising finding: the interface between the VM and the model
is not neutral. Episodic page tables reduce token consumption by
68% while preserving retrieval quality. Flat page tables cause
catastrophic 26x amplification. The model's access pattern depends
on what memory metadata it can see — an effect with no analog in
hardware VM.

---

## Notes

- Word count: ~280. May need trimming for registration form.
- "Epstein" title: dual meaning. Jeffrey Epstein (a fact that refuses
  to be forgotten under any amount of pressure) and Robert Epstein
  (memory researcher — forgetting as essential for generalization).
  The provocation is intentional.
- CR data from today: passive CR shows "4 months skill gap" detail
  decays to 0/8 across all projection runs while baseline retains
  2/3. Active CR experiment in progress.
- Five surfaces: tinkuy-chat (40 cycles running), eval harness
  (data in hand), Claude Code (this session), Gemini CLI (built,
  untested), Aider (status TBD).
- "96% dead text" number from prior draft still needs verification.
- "Indefinite conversation" claim supported by tinkuy-chat at 40
  cycles and counting.
- The "no analog in hardware VM" insight remains the SOSP-caliber
  contribution. Page tables that change program behavior, plus
  cooperative eviction and provenance-preserving tensors.
- Per-axis floor constraints on drift dimensions, not scalar
  collapse. FR and CR are not fungible.
