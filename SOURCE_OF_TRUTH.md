# SOURCE_OF_TRUTH.md

## 1. Project Goal

SDI-Substitutive tests whether expensive resident tensors can be replaced by a lower-cost packed tensor plus a compressed residual correction, while keeping W_ref absent from the substitutive runtime path.

This project is currently tensor/runtime research only.

**Forbidden:**
- no model quality claim
- no behavior recovery claim
- no speedup claim
- no full-model memory claim
- no llama.cpp integration claim
- no production claim

## 2. Current Canonical Architecture

Canonical formula:
```
W_low_runtime = decode(.sdiw packed W_low)
R_runtime = W_ref - W_low_runtime
Y_sub = X @ W_low_runtime + X @ R_runtime
```

Residuals generated from raw/ideal W_low are invalid for current claims.

## 3. Accepted Known-Good Facts

Current accepted facts:

- Additive sidecar path is archived as a negative residency result.
- Substitutive path is the active research direction.
- Runtime-consistent residual source is mandatory: `R = W_ref - decode(packed W_low)`
- k=9% is the current selected policy by gate-based selection:
  - margin > 0
  - margin >= 256KB preferred
  - best approximation among preferred-margin passers
- k=7% is conservative fallback.
- k=12% is accuracy-experimental only because margin is too thin.
- k=15% fails memory.
- No future phase may use a weighted score as final policy selector unless Matt approves the scoring rule.
- Phase 31AJ-STABLE created the source-of-truth regression harness:
  `python -m tests.run_source_of_truth_regression`
  - On this host, `/usr/bin/python3` exists but `python` is not installed; use
    `python3 -m tests.run_source_of_truth_regression` unless a `python` alias is added.
- Phase 31AJ-STABLE verified canonical `.sdir` dense-vs-stream residual apply for:
  - tiny controlled residual matrix
  - ffn_up layer0-shaped fixture `(4864, 896)`
  - ffn_down layer0-shaped fixture `(896, 4864)`
- Phase 31AJ-STABLE verified manifest loader resolves ffn_up and ffn_down artifact paths separately.
- Phase 31AJ-STABLE implemented real canonical `.sdiw` parsing for header, rows, cols, scale bytes, and packed nibble bytes.
- Phase 31AH-RERUN passed combined strict validation against the 31AJ-clean manifest loader/runtime:
  - classification: `PASS_31AH_RERUN_COMBINED_STRICT`
  - scope: ffn_up layers 0-5 and ffn_down layers 0-5
  - policy: k=9%, alpha=1.0, runtime-consistent residuals only
  - all 12 layer/family rows had positive `delta_cos`, positive `MAE_delta`, and positive memory margin
  - strict counters were clean: no W_ref/W_low/R generation or fallback in substitutive mode; `.sdiw_loaded=12`, `.sdir_loaded=12`
  - source equivalence gates passed for `.sdiw`, `.sdir`, and combined dense-vs-stream output
- Phase 31AH-FREEZE checkpoint tag `phase31ah-combined-ffn-runtime-checkpoint` created at HEAD `5b2c1e3` (later updated by freeze commit)
  - Combined ffn_up + ffn_down standalone strict runtime is now checkpointed
  - Checkpoint tag points to the freeze commit
  - Checkpoint/tag restriction is lifted by explicit authorization in this phase
- Phase 31AI completed MLP semantics analysis and ffn_gate standalone feasibility:
  - Exact MLP formula verified from GGUF: `Y = SiLU(X @ W_gate.T) * (X @ W_up.T) @ W_down.T`
  - SiLU (Swish) = x × sigmoid(x)
  - ffn_gate canonical shape: (d_out=4864, d_in=896) — same orientation as ffn_up
  - ffn_gate k-sweep layers 0–5: k=12% selected by gate-based policy (margin > 0, margin ≥ 256KB preferred, best delta_cos among preferred)
  - ffn_gate avg delta_cos at k=12%: +0.002585, avg margin: 315,280 bytes/layer
  - ffn_gate source equivalence gates passed for .sdiw, .sdir, and combined (all 6 layers)
  - Full MLP design documented: Reference / Low-only / Partial substitutive / Strict substitutive
  - Note: k=9% policy applies to ffn_up/ffn_down; ffn_gate selects its own k independently

## 4. Invalidated / Superseded Claims

List claims that must not be reused:

- Any result using residuals generated as `R = W_ref - W_low_raw` is historical only.
- Earlier low-MAE reports from raw-W_low residuals are invalid for current claims.
- Any "strict substitutive" claim is invalid if substitutive mode generates W_ref internally.
- Any combined ffn_up/ffn_down result is invalid if family-specific orientation or artifact loading is not verified.
- Any manifest runtime result is invalid if it silently falls back to ffn_up artifact paths for ffn_down.
- 31X/31Y manifest runtime results that depended on `execute_substitutive_path()` synthesizing W_ref/W_low/R internally are superseded by 31AJ source-of-truth cleanup.
- Pre-31AJ Phase 31AH combined strict validation is superseded by Phase 31AH-RERUN against the 31AJ-clean source-of-truth runtime.

## 5. Suspected / Unproven

Rules:
- Do not put suspected claims in "Accepted Known-Good Facts."
- Do not use suspected claims as the basis for public claims.
- If a later phase verifies a suspected claim, move it to accepted facts.
- If a later phase disproves it, move it to invalidated/superseded claims.

Current suspected/unproven items:


- ffn_gate approximation quality at selected k is measured on activation probe; full MLP behavior in actual inference is unknown

## 6. Current Open Blockers

Current blockers:

- Historical scripts may still contain older orientation assumptions; do not use them for current claims unless they pass the source-of-truth regression contract.
- OpenClaw/prt-lab routing remains a process issue, not a repo blocker.
- Full MLP toy probe requires combined artifact generation for ffn_up + ffn_gate + ffn_down.

## 7. Canonical Orientation Convention

Select one and enforce everywhere.

**Recommended:**
- Artifact tensor shape = (d_out, d_in)
- Then:
  - W_ref shape = (d_out, d_in)
  - W_low_runtime shape = (d_out, d_in)
  - R_runtime shape = (d_out, d_in)
  - X shape = (d_in,)
  - Y shape = (d_out,)
  - Y = X @ W.T
  - residual bitmap row-major index = row * d_in + col
  - row = output dimension
  - col = input dimension
- ffn_up:
  - d_out = 4864
  - d_in = 896
- ffn_gate:
  - d_out = 4864
  - d_in = 896
  - same canonical orientation as ffn_up
- ffn_down:
  - d_out = 896
  - d_in = 4864

If this convention is changed, update this file before any code changes.

## 8. Required Regression Before Any New Phase

No new phase may proceed unless this command passes:
```
python -m tests.run_source_of_truth_regression
```

If `/usr/bin/python` is unavailable on the host, use:
```
python3 -m tests.run_source_of_truth_regression
```
until a `python` alias is deliberately installed.

The regression must test:
- W_low pack/decode roundtrip
- .sdiw parse/apply vs dense decoded W_low
- .sdir encode/decode roundtrip
- .sdir streaming apply vs dense sparse residual apply
- combined stream output vs dense source-of-truth output
- manifest resolves ffn_up and ffn_down separately
- wrong orientation fails fast
- stale/missing paths fail fast
- substitutive mode reports:
  - W_ref_loaded = 0
  - W_ref_generated = 0
  - dense_W_low_materialized = 0
  - dense_R_materialized = 0
  - fallback_count = 0
  - error_count = 0

## 9. Current Allowed Next Phase

Current allowed next phase:
**Phase 31AJ — full MLP toy probe (ffn_up + ffn_gate + ffn_down combined), only if requested explicitly**

Goal:
- Compose ffn_up, ffn_gate, and ffn_down into full MLP substitutive path
- Test strict substitutive MLP against reference MLP using activation probe
- Maintain all forbidden-claim boundaries
- Do not checkpoint/tag unless explicitly authorized

Do not continue 31AJ unless Matt explicitly requests it.

## 10. Update Rules

Update this file after every phase.

Rules:
- Add only verified facts.
- Do not add guesses as accepted facts.
- Do not add "probably fixed" as accepted.
- If something is suspected, put it under "Suspected / Unproven."
- Move invalidated claims into "Invalidated / Superseded Claims."
- Do not delete bad history. Mark it superseded.
- If a phase discovers a contradiction, update this file before continuing.
- If code and this file disagree, stop and resolve the disagreement.
- If a requested phase conflicts with this file, stop and report the conflict.
- Do not continue new phase work until contradictions are resolved.

Commit message for future updates:
`Update SOURCE_OF_TRUTH after Phase <phase>`

## 11. Agent Start Instruction

Every agent must start new work with:
> "I have read SOURCE_OF_TRUTH.md. The current allowed next phase is _____. I will not proceed if the requested task conflicts with the source of truth."

If the agent cannot say that honestly, it must stop.

## 12. Phase Completion Requirement

From now on, final phase reports must include a SOURCE_OF_TRUTH.md section with:
- changed: yes/no
- sections updated
- new accepted facts
- new invalidated/superseded claims
- new suspected/unproven claims
- current blockers
- current allowed next phase

If this section is missing, the phase is not complete.
