# Marching Orders: Epstein Paper — SOSP '26 Submission

**Deadline: 2026-04-02 (3 days)**
**Source: Research supervisor review session, 2026-03-30**
**Review data: REVIEW_R1.md (Rikuy adversarial panel, 20 findings, 5 fatal)**

---

## The Situation

The paper's structure is sound. The writing is honest. The conceptual contributions — inverted cost model, episodic temporal indirection, interface-dependent traces, cooperative eviction — stand on their own. The evaluation is the weak point: every quantitative claim that matters is N=1. The review panel correctly identified this as the primary rejection risk.

You have 3 days. The evaluation harness exists. The system runs. The experiments are repetitions and fixes, not new research. This is grind work, not discovery.

Hamut'ay Taste is explicitly out of scope. Do not try to incorporate it. The two systems make different architectural arguments.

---

## Must Do (gates submission)

### 1. Run CR evaluation 3-5x with different seeds

**Current state:** Single run. Passive CR 0/8 on arbitrary rationale, active CR 2/2 with trace.

**What to do:** Create 3-5 different decision DAGs with different arbitrary rationale details (not "4 months until skill gap closes" every time — different numbers, different constraints, different organizational facts). Run the full CR evaluation for each. Report means and standard deviations for each detail type (technical, organizational, arbitrary) under both passive and active conditions.

**Critical addition:** For each run, record whether and when the model spontaneously uses the trace mechanism. The current paper says "the model learned to use the protocol mid-conversation" — that claim needs frequency data. If it traces in 3/5 runs, that's interesting. If 1/5, the claim needs hedging. If 5/5, it's a strong result.

**Why this is #1:** The CR finding is the paper's crown jewel. It's the claim no other system can make. Everything else (compression, cost reduction, cache hit rates) is efficiency. CR is capability. Make it credible.

### 2. Run needle test N>=10

**Current state:** Single run. 68% token reduction, needle found.

**What to do:** Create 10 different needles (specific facts — dates, names, numbers, decisions). Vary haystack content. Vary needle placement (early turn, mid-session, just before eviction boundary). Run all conditions.

**Report:** Success rate (did the model find the needle?), token consumption per condition (mean, median, range), and whether the needle was retrieved from context, from tensor, or via page fault. That last dimension is important — it tells you whether the VM mechanism is doing the work or whether the needle just happened to survive eviction.

### 3. Fix the cost baseline

**Current state:** 17x cost claim compares gateway (with prefix caching) to unmanaged client (without prefix caching). This is the most indefensible number in the paper.

**What to do:** Run the baseline client with `cache_control` hints enabled. Specifically: place a cache breakpoint after the system prompt (the obvious optimization any competent client would make). Re-measure the cost ratio.

The number will shrink. That's fine. If it goes from 17x to 4x, you have "the gateway reduces cost by 4x over a well-configured client" — that's still a strong result, and it's honest. If it goes to 1.5x, the cost contribution is weak but the other contributions (CR, page tables, cooperative tensors) carry the paper.

**Also:** The current paper says the client had "no prefix caching." Claude Code *does* use `cache_control`. Clarify: was this a stripped-down client for the baseline, or did you measure Claude Code as-is? If Claude Code as-is doesn't cache, say so explicitly and explain why (it's because the message array changes too early in the prefix on every turn, busting the cache). That's actually an interesting finding — the client's message construction defeats prefix caching by accident, and the gateway's region ordering fixes it by design.

### 4. Log cooperative vs sidecar ratio

**Current state:** Not reported anywhere. The paper claims "cooperative eviction" as a defining feature but provides no data on how often the model actually cooperates.

**What to do:** Instrument one or more live sessions to log, for each eviction event: was the tensor produced by the cooperative protocol (model emitted `<release>`) or by the sidecar? Report the ratio.

**Interpretation guide:**
- If cooperative > 80%: the "cooperative" framing is justified. Lead with it.
- If cooperative 40-80%: dual-path design is the contribution. "The model cooperates when it can; the sidecar guarantees coverage."
- If cooperative < 40%: reframe. The sidecar is the primary mechanism; cooperative is an optimization. Still valuable, but the narrative shifts from "the model manages its own memory" to "the system manages memory with model input when available."

Whatever the number is, report it honestly. The reviewers will ask.

### 5. Reframe contribution 5

**Current claim:** "Provenance edges that break the attention ceiling."

**The problem:** The trace mechanism requires the model to remember the handle of the decision it wants to trace. That recall is itself subject to attention decay. The mechanism doesn't break the ceiling — it provides a tool that is gated by it.

**Reframe to:** "Provenance edges for salience-gated recovery." The model can signal that certain reasoning chains are important (via `<declare>`), and later request reconstruction (via `<trace>`). The declare signal works because it's emitted at decision time, when the reasoning chain is salient. The trace signal works when the model recognizes it needs the chain — recognition that is still attention-dependent. This is honest and still valuable: the system provides a mechanism that the model uses when its natural attention is insufficient, but the trigger is attention-gated.

**In the contributions list, change to:**
"Provenance edges for salience-gated recovery. Passive coherence retention decays to 0/N on arbitrary rationale details by turn 29. Dependency edges emitted at decision time enable demand-driven reconstruction; active recovery achieves M/N on root-cause probes when trace is activated."

(Fill in N and M from the repeated runs in task #1.)

---

## Should Do (strengthens significantly, do if time permits)

### 6. Add prompts to appendix

The cooperative memory protocol (release, retain, recall, declare, trace) lives in prompts. The system prompt that instructs the model on these signals is the interface specification. Include it in an appendix. Reviewers cannot reproduce the work without it. This is 30 minutes of copy-paste and formatting.

### 7. Report live compression ratios

**Current state:** 97.3% compression from backtesting (best-case, full hindsight).

**What to do:** From any live session where eviction occurred, report the compression ratio of tensors produced during the session (not in backtesting). If you ran the CR evaluations (task #1), those sessions produced tensors — measure their compression ratios and report them alongside the backtesting number.

The live number will be lower than 97.3%. That's the point — it's honest. Present the backtesting number as an upper bound and the live number as the operational reality.

### 8. Own the analogy break in the discussion

**Current state:** Section 5.4 ("The Interface Changes the Trace") identifies that the page table changes the model's behavior — an effect with no analog in hardware VM. But the paper doesn't acknowledge that this finding undermines the classical VM analysis it invokes elsewhere (Belady, working set theory).

**What to add:** A paragraph acknowledging that the reference string in transformer VM is not fixed — it depends on the interface and policy. This means classical VM analysis (which assumes a fixed reference string) does not directly apply. Frame this as a contribution, not a weakness: "We discovered where the analogy breaks, and the break is itself the most interesting finding. Classical VM theory assumes the application cannot observe the memory system. When it can, the system's interface becomes a first-class design parameter that changes the application's behavior — a feedback loop with no classical analog."

This also connects to the title: "Why FIFO Beats Belady When the Application Can Read the Page Table." Belady assumes a fixed reference string. FIFO wins because the model, seeing a simple page table, generates a simple access pattern that FIFO handles well. That's not a deficiency of Belady — it's a different problem.

### 9. Add the KV-cache-as-context-save framing

**One sentence for the introduction, after the Atlas paragraph:**
"The industry's current approach — KV cache optimization — treats the symptom: the GPU's process state is too expensive to recompute, so systems cache it and then optimize the cache. We treat the cause: the process carries too much state because nobody manages what enters the context window."

This positions the paper against the Google KV cache work and similar efforts without needing to cite specific papers. It says "they're optimizing the cache line; we're questioning the cache hierarchy."

---

## Do Not Do

- **Do not incorporate Hamut'ay Taste.** It's a different architecture making a different argument. It doesn't belong in this paper.
- **Do not add a MemGPT comparison.** The reviewers will want one, but you can't build it in 3 days. Acknowledge the gap in limitations and frame the CR evaluation as the differentiator (MemGPT has no provenance mechanism).
- **Do not add a second model family.** Same reasoning. Acknowledge in limitations.
- **Do not add a second API provider.** Same.
- **Do not rewrite the introduction.** The desk metaphor works. The Atlas connection works. Tighten, don't replace.

---

## Submission Checklist

- [ ] CR evaluation: 3-5 runs with different seeds, means and stdev reported
- [ ] CR trace frequency: how often does the model spontaneously trace?
- [ ] Needle test: N>=10, varied needles and placement
- [ ] Cost baseline: re-run with `cache_control` on baseline client
- [ ] Cooperative/sidecar ratio: logged from live session(s)
- [ ] Contribution 5 reframed: "salience-gated recovery" not "breaks the ceiling"
- [ ] Abstract updated with final numbers from repeated experiments
- [ ] Conclusion updated with final numbers
- [ ] Tables 3 and 4 updated with aggregated results
- [ ] Formatting: height margin issue resolved or confirmed false positive
- [ ] Prompts in appendix (if time)
- [ ] Live compression ratios reported (if time)
- [ ] Discussion paragraph on analogy break (if time)
- [ ] KV-cache-as-context-save sentence in intro (if time)
