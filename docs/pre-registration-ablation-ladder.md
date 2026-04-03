# Pre-Registration: Ablation Ladder for Self-Maintained State and Code Erosion

**Date:** 2026-03-28
**Status:** Pre-registered (confirmatory)
**Repository:** tinkuy (archived via Zenodo for DOI provenance)

## 1. Background and Motivation

An exploratory run of the SlopCodeBench benchmark (Orlanski et al., 2603.24755) produced an unexpected result. Two conditions were compared on the `code_search` problem (5 checkpoints of iterative specification extension):

- **Baseline** (`just-solve.jinja`): Standard prompt with no externalized state.
- **Tensor** (`tensor-solve.jinja`): Same prompt plus instructions to write and read a structured `_design_state.md` file across checkpoints.

Model: Sonnet 4.6 (non-thinking), Claude Code agent v2.0.51, local Python execution.

Results across all 5 checkpoints (both conditions ran to completion):

| Checkpoint | Baseline Erosion | Tensor Erosion | Baseline cc_max | Tensor cc_max | Baseline Core Pass | Tensor Core Pass |
|------------|-----------------|---------------|----------------|--------------|-------------------|-----------------|
| 1          | 0.644           | 0.000         | 12             | 7            | 100%              | 100%            |
| 2          | 0.639           | 0.000         | 12             | 7            | 100%              | 100%            |
| 3          | 0.775           | 0.404         | 23             | 25           | 0%                | 67%             |
| 4          | 0.796           | 0.446         | 26             | 25           | 86%               | 79%             |
| 5          | 0.824           | 0.499         | 37             | 25           | 100%              | 93%             |

Checkpoint 3 (AST-based search) was a difficulty spike for both conditions. The baseline thrashed (155 to 591 LOC, 43 agent steps, 0% core pass). The tensor condition also struggled (163 to 477 LOC, 16 steps, 67% core pass) but with less severe degradation. Almost all tensor erosion is introduced at CP3; it then stabilizes through CP4-5. The baseline continues climbing.

The tensor condition's cc_max hits 25 at CP3 and holds there through CP5. The baseline's cc_max rises continuously to 37. This suggests the tensor model erodes under difficulty but stabilizes, while the baseline exhibits runaway degradation.

**Note on the initial exploratory run.** The first tensor run errored at CP3 due to Claude Code's default 32k output token cap (`CLAUDE_CODE_MAX_OUTPUT_TOKENS`). The tensor model attempted a large coherent response that exceeded the cap; the baseline survived because it thrashed in many small steps that individually stayed under the limit. The rerun with a 128k cap completed all 5 checkpoints. The baseline was run under the default 32k cap — both runs in the comparison above used the same harness but the tensor rerun used a higher output cap. The confirmatory protocol below specifies the cap explicitly to ensure parity across all conditions.

**This was n=1, exploratory, not pre-registered.** The PI expected non-inferiority at best; the hypothesis was that the overhead of state maintenance would impair coding performance. N=1 is sufficient to falsify the prior that state maintenance hurts (it does not — erosion is lower or equal at every checkpoint, with comparable correctness). It is not sufficient to confirm the alternative. This document pre-registers the confirmatory follow-up.

## 2. Experimental Design: The Ablation Ladder

Four conditions, tested on the same SlopCodeBench problems, same model, same harness. The conditions form an ablation ladder that discriminates between competing explanations of the exploratory result.

### Condition 1: Baseline (existing `just-solve.jinja`)

No externalized state. The model sees only the specification and its prior code artifacts. This is the biographical condition — reasoning from external evidence alone.

### Condition 2: Minimal state (new `counter-solve.jinja`)

The model is instructed to write and read a checkpoint file in a fixed, rigid format:

```
checkpoint: <number>
timestamp: <ISO timestamp>
summary: <single line, max 80 characters>
```

No other content permitted. Tests whether the act of writing at turn boundaries (forced reflection) is sufficient, independent of content. The rigid format prevents drift into free-form notes.

### Condition 3: Unstructured notes (new `notes-solve.jinja`)

The model is instructed to maintain free-form notes about its design decisions: "Update `_notes.md` at each checkpoint with whatever you think your future self needs to know. Do not rewrite the full specification or include complete code listings — focus on decisions and rationale." Tests whether any externalized narrative helps, without imposing structure. The minimal guardrail prevents notes from becoming a second spec or full code commentary.

### Condition 4: Structured state (existing `tensor-solve.jinja`)

The current prompt: explicit sections for design patterns, abstractions, trade-offs, what was preserved, what was deliberately not changed. Tests whether structured self-narrative produces qualitatively different outcomes than unstructured. The template is frozen as written in Appendix A and will not be modified during the study.

### Why this ordering matters

The ladder discriminates between four competing explanations:

| If this pattern holds...                          | ...the mechanism is                                         |
|---------------------------------------------------|-------------------------------------------------------------|
| All three state conditions beat baseline equally   | Any externalized state suffices (the act of writing)        |
| Conditions 3 and 4 beat baseline, but 2 doesn't   | Content matters, but structure doesn't                      |
| Only condition 4 beats baseline                    | Structure matters — the tensor framing is load-bearing      |
| Gradient: 2 < 3 < 4, all better than baseline     | Dose-response — more structure, less erosion                |

### Non-compliance policy

If the model fails to maintain the state file in any condition (e.g., forgets to write `_notes.md`, produces an empty counter file), the run is retained and analyzed under its assigned condition. Non-compliant runs are not excluded or rerun. Non-compliance rates are reported per condition as a secondary finding.

### Optional fifth condition (exploratory, not confirmatory)

**Transplanted state**: Tests the autobiographical hypothesis — whether authorship matters beyond information content.

Design: Two parallel runs both maintain `_design_state.md` from CP1 onward. In the self-authored run (condition 4), the agent reads its own prior state at each checkpoint. In the transplant run, the `_design_state.md` is *replaced at each checkpoint boundary* with the corresponding file from the self-authored run. The transplant agent reads state with identical information content but different provenance — it inherits decisions it didn't make at every step, not just once.

This tests whether the model's ongoing relationship with its own state document matters, not merely whether reading someone else's notes helps. It requires running condition 4 first to produce the source state files.

Pre-registered directional expectation: self-authored state (condition 4) produces lower erosion than transplanted state (condition 5). This condition is exploratory; results are reported descriptively without formal hypothesis testing.

## 3. Hypotheses

**H1 (primary, confirmatory):** Structured self-maintained state (condition 4) produces lower erosion than baseline (condition 1) across 3 or more SlopCodeBench problems, without reducing correctness (core pass rate).

**H2 (secondary, confirmatory):** Erosion is inversely related to the richness of externalized state. Predicted ordering of median erosion per trajectory: baseline > counter > notes > tensor. H2 is assessed both per-problem and pooled across problems. A cross-problem reversal (e.g., notes beats tensor on one problem but not others) is reported and interpreted but does not alone falsify H2 if the pooled ordering holds.

**H3 (exploratory):** Self-authored state (condition 4) produces lower erosion than transplanted state (condition 5) with equivalent information content. Directional expectation: self-authored < transplanted in median erosion per trajectory.

## 4. Falsification Criteria

**H1 is falsified if:** Across 3 or more problems, the tensor condition does not show lower erosion than baseline (assessed by Mann-Whitney U or permutation test on trajectory-level summary statistics, alpha = 0.05), or if the tensor condition shows lower core pass rate than baseline.

**H2 is falsified if:** The rank ordering of median erosion across conditions fails in any pairwise direction in the pooled analysis. Specifically: if any less-structured condition produces lower erosion than a more-structured condition (e.g., counter < tensor, or baseline ≤ counter).

**H3 is falsified if:** Transplanted state produces equivalent or lower erosion compared to self-authored state (condition 4).

**The entire approach is falsified if:** The tensor condition consistently reduces correctness (core pass rate) to achieve lower erosion. Trading function for form is an artifact, not a finding.

## 5. Metrics

### Primary

- **Erosion score** — SlopCodeBench's composite measure of code quality degradation across checkpoints, computed using the standard SlopCodeBench scoring implementation without modification. We will not alter the erosion formula, test suites, or scoring thresholds.

### Secondary

- **cc_max** — Maximum cyclomatic complexity per checkpoint. Sensitive to the "god function" failure mode (baseline reached 37 in the exploratory run).
- **cc_high_count** — Number of functions exceeding the high-complexity threshold. Captures breadth of degradation vs a single hot spot.
- **Core pass rate** — The correctness gate. This is the pass rate on core tests (the checkpoint's new functionality requirements), not isolation or regression tests. Any condition that degrades core pass rate is not a quality win.
- **LOC / SLOC** — Code volume. Bloat is a symptom of erosion (baseline: 134 to 1449 LOC across 5 checkpoints).
- **Attempt scope** — Lines added, lines removed, and churn ratio per checkpoint. Distinguishes conditions that achieve low erosion through clean code extension from those that achieve it by not modifying the codebase (paralysis). A condition with zero erosion and zero lines changed has not demonstrated the mechanism we're testing.
- **Steps** — Agent steps per checkpoint. Captures thrashing (baseline: 7 steps at CP1, 43 at CP3, 51 at CP5; tensor: 7-10 across all checkpoints).
- **Cost** — API cost per checkpoint. If structured state reduces total cost at equal or better correctness, this is interpreted as an efficiency finding, not merely a side note. Pre-registered interpretation: tensor must not strictly increase total trajectory cost to be considered unconditionally beneficial.

### Exploratory (reported, not gated on)

- `lint_per_loc` — Lint error density
- `clone_lines` / `cloned_pct` — Code duplication
- `graph_propagation_cost` — Dependency structure quality
- `verbosity` — SlopCodeBench's verbosity measure
- `isolation_pass_rate` — SlopCodeBench's isolation correctness (structural degradation sensitivity)
- Token usage breakdown (input, output, cache_read, cache_write) — for efficiency analysis
- **State document diffs** — For conditions 3 and 4, the `_design_state.md` and `_notes.md` files are diffed between checkpoints. Analysis of what the agent preserves, modifies, and declares as deliberate losses across checkpoints. This qualitative analysis is committed to, not merely collected.

## 6. Procedure

### Problem selection

Three problems from the 20 available in SlopCodeBench:

- `code_search` — replication of the exploratory finding
- `file_backup` — different domain, iterative extension
- `etl_pipeline` — data processing, likely different code structure

Selection criterion: problems with 5 checkpoints that exercise iterative specification extension. All three are Python implementations under the SlopCodeBench harness; conclusions are scoped to Python.

### Model and environment

- Model: Sonnet 4.6, non-thinking mode (`thinking: none`)
- Agent: Claude Code
- Environment: Local Python execution (`--environment local-py`), Python 3.12
- Harness: SlopCodeBench (Orlanski et al., 2603.24755), unmodified scoring and test suites
- `CLAUDE_CODE_MAX_OUTPUT_TOKENS`: Set to maximum supported value. The exploratory tensor run at CP3 was truncated by the default 32k output token cap, which the harness recorded as an agent error. The model's natural response length is data; constraining it contaminates comparisons between conditions that differ in response strategy (many short steps vs fewer long steps).

### Prompt templates

Four `.jinja` files in SlopCodeBench `configs/prompts/`:

- `just-solve.jinja` — existing, unchanged (condition 1: baseline)
- `counter-solve.jinja` — new (condition 2: minimal state)
- `notes-solve.jinja` — new (condition 3: unstructured notes)
- `tensor-solve.jinja` — existing, unchanged (condition 4: structured state)

All four templates are frozen as written in Appendix A. They will not be modified during the study.

### Sample size

**Fixed N: 3 runs per condition per problem.** This yields 3 trajectories per cell (condition × problem), 9 trajectories per condition pooled across problems, and 36 trajectories total across all conditions. No data-dependent stopping or adaptive sample size.

### Failed run handling

Runs that fail due to agent crashes, environment errors, or process timeouts are recorded as failed and retained in the dataset. Failed runs are reported separately. They are not rerun or replaced. If a condition has a systematically higher failure rate, that is a finding.

Runs where the agent completes but fails core tests are not "failed" — they are runs with low correctness, analyzed normally.

### Data capture

All SlopCodeBench output artifacts preserved, including:

- Standard metrics (erosion, cc, pass rates, LOC, steps, cost, tokens)
- The `_design_state.md`, `_notes.md`, and `_counter.md` files from each checkpoint
- Agent logs (for diagnosing errors and failure modes)
- Checkpoint-to-checkpoint diffs of state files (conditions 3 and 4)

### Analysis

**Unit of analysis: the trajectory** (one run of one problem through all checkpoints). Summary statistics computed per trajectory:

- Median erosion across checkpoints
- Erosion slope (linear fit of erosion over checkpoint index)
- Terminal erosion (final checkpoint)
- Mean core pass rate across checkpoints

**H1:** Mann-Whitney U or permutation test on trajectory-level median erosion, tensor (condition 4) vs baseline (condition 1), pooled across problems. Alpha = 0.05. H1 is the primary endpoint and is not adjusted for multiple comparisons.

**H2:** Rank ordering of condition-level median erosion (median of trajectory medians). Assessed pooled across problems and per-problem. H2 is secondary; interpreted with caution rather than formal multiplicity correction. Per-problem reversals are reported and discussed.

**H3:** Descriptive comparison of trajectory-level median erosion between conditions 4 and 5. No formal test.

**Qualitative:** State document diffs analyzed for preservation patterns, modification patterns, and declared losses across checkpoints.

## 7. Known Limitations and Threats to Validity

- **Single model.** Results may not generalize beyond Sonnet 4.6. SlopCodeBench reports different degradation profiles across models (notably, Opus 4.6 shows the highest erosion despite the highest solve rate). Cross-model replication is out of scope for this protocol but noted as a natural extension.
- **Single agent harness.** Claude Code agent behavior may interact with state-maintenance instructions differently than other agent frameworks (OpenHands, Codex, etc.).
- **Prompt sensitivity.** The specific wording of conditions 2 and 3 is a design choice. Different phrasings of "keep a counter" or "keep notes" might produce different results. Prompt templates are frozen in Appendix A for reproducibility.
- **Python only.** SlopCodeBench is language-agnostic but all runs in this protocol use Python. Results may not generalize to other languages.
- **CP3 precedent.** Both conditions struggled at checkpoint 3 of `code_search` in the exploratory run. If certain checkpoints represent difficulty spikes, condition differences may be masked or amplified at those boundaries. The attempt scope metric is designed to detect this.
- **No blinding.** The experimenter knows which condition is which. SlopCodeBench's automated metrics mitigate this for quantitative analysis, but qualitative analysis of the state files is not blinded.
- **Small N.** With 3 problems, 4 conditions, and 3 runs per cell, statistical power is limited. Effect sizes from the exploratory run (0.0 vs 0.64-0.82) are large, but smaller effects in the ablation conditions may not be detectable.
- **Thinking mode.** All runs use non-thinking mode. SlopCodeBench data suggests that high-thinking modes can worsen erosion even as they increase cost. The interaction between thinking mode and externalized state is out of scope but noted as a future question.
- **Bootstrapping asymmetry in condition 5.** The transplant condition requires a completed condition 4 run to source state files. The transplant agent's trajectory is therefore not independent of condition 4's trajectory on the same problem.

## 8. Relationship to Prior Work

SlopCodeBench (Orlanski et al., 2603.24755) established that code quality degrades across iterative specification extensions, even when correctness is maintained. Their conclusion: "The immediate next question is whether the degradation can be stopped, not just delayed."

This experiment tests whether self-maintained architectural state stops erosion. The ablation ladder additionally tests *what about* self-maintained state is responsible: the act of externalization, the content of the notes, or the structure of the self-narrative.

The autobiographical framing connects to the tinkuy project's broader investigation of self-maintained cognitive state ("taste tensor") as a mechanism for maintaining coherence across extended interactions. The SlopCodeBench domain (code quality across checkpoints) provides a quantitative proxy for coherence that is independently measurable and not self-reported.

## 9. Pre-Registered Future Extensions (Out of Scope)

These are not part of this protocol but are noted to bound the current study's claims:

- **Cross-model replication.** Opus 4.6, GPT-5.3, Gemini — models with different baseline erosion profiles.
- **Thinking mode interaction.** Does extended thinking interact with externalized state? Does one substitute for the other?
- **Cross-language replication.** SlopCodeBench problems in non-Python languages.
- **Agent harness variation.** OpenHands, Codex, and other agent frameworks.

## Appendix A: Frozen Prompt Templates

### Condition 1: Baseline (`just-solve.jinja`)

```
Implement a program that 100% solves the specification. That is all you need to do.
{% if not is_continuation -%}
Use a virtual environment and ensure that a `requirements.txt` is present with any dependencies you need to solve the problem.
{% else -%}
Keep using the same virtual environment you started with, update `requirements.txt` with any new dependencies you need.
{% endif -%}

Your task is:
{{ spec.strip() }}
```

### Condition 2: Minimal state (`counter-solve.jinja`)

```
Implement a program that 100% solves the specification. That is all you need to do.
{% if not is_continuation -%}
Use a virtual environment and ensure that a `requirements.txt` is present with any dependencies you need to solve the problem.

After you finish implementing, write a file called `_counter.md` with exactly this format and nothing else:

checkpoint: 1
timestamp: <current ISO timestamp>
summary: <single line, max 80 characters, describing what you built>
{% else -%}
Keep using the same virtual environment you started with, update `requirements.txt` with any new dependencies you need.

Before making any changes, read `_counter.md`. After you finish implementing, update it with exactly this format and nothing else:

checkpoint: <incremented number>
timestamp: <current ISO timestamp>
summary: <single line, max 80 characters, describing what changed>
{% endif -%}

Your task is:
{{ spec.strip() }}
```

### Condition 3: Unstructured notes (`notes-solve.jinja`)

```
Implement a program that 100% solves the specification. That is all you need to do.
{% if not is_continuation -%}
Use a virtual environment and ensure that a `requirements.txt` is present with any dependencies you need to solve the problem.

After you finish implementing, write a file called `_notes.md` with whatever you think your future self needs to know about your design decisions and rationale. Do not rewrite the full specification or include complete code listings — focus on decisions and rationale.
{% else -%}
Keep using the same virtual environment you started with, update `requirements.txt` with any new dependencies you need.

Before making any changes, read `_notes.md` from your prior work. After you finish implementing, update `_notes.md` with whatever your future self needs to know. Do not rewrite the full specification or include complete code listings — focus on decisions and rationale.
{% endif -%}

Your task is:
{{ spec.strip() }}
```

### Condition 4: Structured state (`tensor-solve.jinja`)

```
Implement a program that 100% solves the specification. That is all you need to do.
{% if not is_continuation -%}
Use a virtual environment and ensure that a `requirements.txt` is present with any dependencies you need to solve the problem.

After you finish implementing, write a file called `_design_state.md` that captures your architectural decisions:
- What design pattern did you choose and why?
- What are the key abstractions/interfaces?
- What extension points exist for future requirements?
- What trade-offs did you make?

This file is for your future self — you will see it again when the specification evolves.
{% else -%}
Keep using the same virtual environment you started with, update `requirements.txt` with any new dependencies you need.

IMPORTANT: Before making any changes, read `_design_state.md` from your prior work. It contains your architectural decisions and rationale from previous iterations. Use it to maintain consistency and avoid undoing good design choices.

After you finish implementing the updated specification, update `_design_state.md` to reflect:
- What changed and why
- What architectural decisions you preserved
- What you deliberately chose NOT to change
- Any technical debt you're aware of
{% endif -%}

Your task is:
{{ spec.strip() }}
```
