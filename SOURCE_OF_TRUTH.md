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
- Phase 31AJ (partial): full MLP composition probe completed as partial result:
  - Full MLP toy composition (ffn_up + ffn_gate + ffn_down) improves approximation over low-only on all 6 layers
  - Source equivalence clean: sdiw/sdir/combined pass for ffn_up, ffn_gate, ffn_down (layer 0)
  - Strict counters clean: sdiw_loaded=18, sdir_loaded=18, fallback_count=0, error_count=0
  - MLP formula/residual staging verified correct:
    `Y = (SiLU(X @ W_gate.T) * (X @ W_up.T)) @ W_down.T`
    up residual: before SiLU/gating ✓
    gate residual: before SiLU ✓
    down residual: after hidden is formed ✓
  - Individual row margins positive: all 18 layer/family rows under Q4 budget individually
  - Combined MLP memory fails: aggregate margin_bytes = −30,414,836 (69.6 MB artifacts vs 39.2 MB Q4 budget)
  - Classification: `PARTIAL_MLP_APPROX_PASS_MEMORY_FAIL`
  - NOT memory-positive for combined MLP composition
  - NOT a full pass — artifact budget/economics must be fixed before continuation
- Phase 31AK: full MLP artifact budget/economics fix completed — classification `PARTIAL_MLP_BUDGET_FAILS_CURRENT_ENCODING`:
  - All 7 family subsets (single, double, triple) fail memory check at current encoding
  - No combination of family residual ON/OFF can produce memory-positive aggregate
  - Even with ALL residuals OFF, each family still has margin = −1.56 MB because W_low packed+scale (14.03 MB) exceeds per-family Q4 budget (12.47 MB)
  - Root cause: W_low_scale bytes (272,384 per layer-family) are stored separately and are not accounted for in the Q4 budget definition
  - W_low packed alone (42.08 MB) exceeds total Q4 budget (37.41 MB) by 4.68 MB
  - Residuals add 24.33 MB additional overrun
  - Classification: `PARTIAL_MLP_BUDGET_FAILS_CURRENT_ENCODING`
  - NOT a runtime policy problem — no k-selection or family-skipping can fix this under current encoding
  - Artifact encoding redesign required (Phase 31AL)
- Phase 31AL: artifact encoding redesign — classification `PASS_WLOW_ENCODING_CANDIDATE_FOUND` — BUT SEE 31AL-R CORRECTION BELOW:
  - Q4_budget defined as nibble storage only (n_elements × 4/8 = 2,179,072 bytes/family-layer)
  - Current sdiw overhead: scale bytes (+272,384/family-layer = +12.5%) cause W_low to exceed Q4_budget
  - SDIR residual is inefficient: full bitmap (1 bit/elem = 544,768 bytes) regardless of k%, plus fp16 values
  - Viable W_low candidate: Q2_K (GGUF format, 2.625 bits/elem, 1,430,028 bytes/family-layer)
  - Q2_K W_low total: 24.55 MB (fits under Q4_budget of 37.41 MB)
  - SDIR residual is NOT viable with any W_low format at current k% and encoding
  - W_low Q2_K decode quality not measured (requires GGUF dequantize, unavailable locally)
  - Residual Q2 encoding quality not verified
  - **NOTE**: 31AL had labels swapped and a fabricated byte count — see 31AL-R correction below
- Phase 31AL-R: quant byte accounting audit completed — classification `PARTIAL_31AL_VIABILITY_REVISED`:
  - 31AL had label swap: claimed Q4_K_M = 1,430,028 bytes but correct format is Q2_K (2.625 bits/elem)
  - 31AL had fabricated value: 1,818,060 bytes = 3.337 bits/elem — NOT a standard GGUF quant type
  - Corrected Q4_K = 2,451,468 bytes = 4.5 bits/elem (same as sdiw, over Q4_budget)
  - Corrected combined viability at k=9-12%: NO combined W_low + residual policy is memory-positive
  - Corrected viable: Q2_K W_low + int8 sparse residual at k ≤ 3% is viable (margin 1-3 MB)
  - Qualitative direction unchanged: Q2_K is still the right W_low format; SDIR is still the budget blocker
- Phase 31AM: low-k Q2_K + sparse residual viability probe — classification `PARTIAL_Q2_SIM_ONLY_LOWK_RESIDUAL_HURTS`:
  - Actual GGUF Q2_K decode: **NOT available** — W_low numerical results are from **simulated Q2-like** (block_size=16 aggressive 2-bit quantization)
  - Q2_K byte accounting used for budget modeling (per GGUF constants, block_size=256, 2.625 bits/elem)
  - Memory-positive low-k policies exist: 27/29 tested policies are memory-positive at k≤2% (SDIR) or k≤3% (int8)
  - **Residual-on hurts approximation:** ALL tested residual policies (k=0.5% to 2%, all family-targeting and asymmetric variants) worsen both cosine and MAE versus residual-off (uniform_0)
  - Best policy: `uniform_0` (residual-off, k=0%) — cos=0.9260, MAE=0.00904, margin=+3,591 KB
  - Root cause: quantization error is magnitude-correlated (r=0.9999) — residual captures error on already-well-approximated large-magnitude elements, yielding no net gain
  - **No Pareto frontier exists:** no memory-positive policy improves both cosine and MAE simultaneously
  - Q2_K decode quality (block_size=256) remains unverified; actual behavior may differ from block_size=16 simulation
  - Classification: `PARTIAL_Q2_SIM_ONLY` — numerical results are simulated, not actual Q2_K decode
- Phase 31AN: actual Q2_K decode probe — classification `PARTIAL_ACTUAL_DECODED_RESIDUAL_IMPROVES_MEMORY_FAILS`:
  - Actual GGUF decode available: YES (IQ4_NL/Q3_K dequantized via GGUFReader), but NOT actual Q2_K (type=10)
  - The file `qwen2.5-0.5b-instruct-q2_k.gguf` uses IQ4_NL (4.5 bpe) for ffn_up/ffn_gate and Q3_K (3.44 bpe) for ffn_down — NO type=10 (Q2_K) tensors present
  - Actual decoded cos ≈ 0.996, MAE ≈ 0.001 — significantly BETTER than 31AM simulated (cos≈0.926, MAE≈0.009)
  - **Residual-on IMPROVES approximation** at every k% for every family (opposite of 31AM)
  - Layers 0-5 avg: k=3% → delta_cos=+0.0015, residual-on improves cosine and MAE simultaneously
  - **Memory FAILS:** IQ4_NL exceeds Q4_budget by 272 KB per ffn_up/ffn_gate tensor; net −1,396 KB across 18 tensors
  - True Q2_K (2.625 bpe) would be memory-positive (+731 KB margin each), but encode is NotImplemented in gguf-py
  - Q2_K.quantize_blocks = NotImplemented — cannot encode W_ref to Q2_K ourselves
  - **31AM's simulation conclusions do NOT transfer to actual decode paths** — 31AM used block_size=16 aggressive quantization, not actual GGUF types
  - Classification: `PARTIAL_ACTUAL_DECODED_RESIDUAL_IMPROVES_MEMORY_FAILS` — three parts: actual decoded, residual improves, memory fails
- Phase 31AO: true Q2_K encoder/decoder prototype — classification `BLOCKED_Q2K_ENCODER`:
  - Q2_K format confirmed from ggml-common.h: block_q2_K = d(2)+dmin(2)+scales(16)+qs(64) = 84 bytes, block_size=256, bpe=2.625
  - Expected Q2_K bytes for ffn_up: 1,430,016 (1,396.5 KB), Q4 budget: 2,179,072, margin: +731.5 KB — would be memory-positive
  - Q2_K GGUF contains **0 type-10 (Q2_K) tensors** — only IQ4_NL/Q3_K present
  - gguf-py Q2_K.quantize_blocks = NotImplemented — cannot encode W_ref to Q2_K
  - Custom Q2_K encoder attempted: blocked by divide-by-zero and overflow issues
  - **No validation reference exists** — no type-10 Q2_K tensors in any available GGUF to validate encoder against
  - Classification: `BLOCKED_Q2K_ENCODER` — encoder not implementable without Q2_K reference tensors or quantize_blocks support
- Phase 31AP: Q2_K encode via llama.cpp quantize_row_q2_K_ref + dequantize_row_q2_K — classification `PASS_Q2K_LLAMA_QUANT_LOWK_IMPROVES`:
  - llama.cpp `libggml-base.so` exposes `quantize_row_q2_K_ref` and `dequantize_row_q2_K` as exported symbols — callable via ctypes
  - Q2_K encode for layer0 ffn_up (4,358,144 elements, 17,024 blocks): byte match = True (1,430,016 bytes exactly), bpe=2.625
  - Q2_K memory: 1,430,016 bytes vs Q4_budget 2,179,072 bytes → margin = +749,056 bytes (+731.5 KB) — **memory-positive**
  - True Q2_K numerical quality: cos=0.9567, MAE=0.00360 — significantly better than 31AM simulated Q2-like (cos≈0.926, MAE≈0.009)
  - True Q2_K numerical quality: slightly below 31AN actual IQ4_NL (cos≈0.996, MAE≈0.001), but IQ4_NL was memory-negative
  - **Low-k residual improves at every tested k**: k=0.5% → delta_cos=+0.0019, k=1% → +0.0033, k=2% → +0.0055, k=3% → +0.0074
  - **k=3% fails memory** (margin = -57,200 bytes); **k≤2% is memory-positive** with k=2% → margin=+29,964 bytes
  - Best memory-positive policy: k=2% (delta_cos=+0.0055, margin=+29,964 bytes) or k=1% (delta_cos=+0.0033, margin=+117,126 bytes)
  - Classification: `PASS_Q2K_LLAMA_QUANT_LOWK_IMPROVES` — Q2_K encode/decode works, residual improves, memory-positive at k≤2%
- Phase 31AQ: layer0 full MLP Q2_K + low-k residual — classification `PASS_LAYER0_MLP_Q2K_LOWK_POLICY_FOUND`:
  - All three layer0 families (ffn_up, ffn_gate, ffn_down) encode to Q2_K with byte-exact match: 1,430,016 bytes each, margin=+749,056 per family
  - All three families have **residual-on improves** at every tested k (0.5% to 3%) — consistent with 31AP ffn_up finding
  - **Layer0 full MLP composition** (SiLU gating): Q2-only cos=0.8862, MAE=0.038; residual improves at ALL tested k
  - **k=0.5%**: margin=+482,124, delta_cos=+0.0083, MAE improves (0.0364 vs 0.038) — **memory-positive**
  - **k=1%**: margin=+351,378, delta_cos=+0.0112, MAE improves — **memory-positive**
  - **k=2%**: margin=+89,892, delta_cos=+0.0165, MAE improves — **memory-positive**
  - **k=3%**: margin=−171,600 — **fails memory** but improves cosine
  - **Best passing policy: k=2%** (margin=+89,892, delta_cos=+0.0165, MAE=0.035)
  - **Family ablation at k=1%**: all subsets improve cosine; gate+down gives highest delta_cos (+0.0090) but all are memory-positive (+117,126 each)
  - **First passing full layer0 MLP result**: aggregate margin=+89,892 at k=2% with delta_cos=+0.0165
  - Classification: `PASS_LAYER0_MLP_Q2K_LOWK_POLICY_FOUND` — layer0 MLP Q2_K + low-k residual is memory-positive and improves cosine+MAE
- Phase 31AR (layers 0-5 full MLP Q2_K + low-k residual sweep) — classification `PASS_LAYERS0_5_MLP_Q2K_LOWK_POLICY_FOUND`:
  - **All 18 Q2_K tensor checks PASS** (byte_match=True for all layers 0-5, families up/gate/down)
  - Q2_K via llama.cpp `quantize_row_q2_K_ref` + `dequantize_row_q2_K`: 1,430,016 bytes/family, margin=+749,056/family
  - **Per-layer MLP policy sweep (encode_sdir residual format):**
    - k=0.5%: all 6 layers memory-positive (worst_margin=+481,950), avg_delta_cos=+0.00583
    - k=1%: all 6 layers memory-positive (worst_margin=+351,076), avg_delta_cos=+0.00806
    - k=2%: all 6 layers memory-positive (worst_margin=+89,524), avg_delta_cos=+0.01184, MAE improves on all layers
    - k=3%: **fails memory** (aggregate_margin=−1,031,746, worst_margin=−172,098, n_mem_pos=0/6)
  - **Aggregate at k=2%**: margin=+537,844, 6/6 layers memory-positive, all layers improve cosine and MAE
  - **Selected policy: k=2%** — margin=+537,844 aggregate, worst layer=+89,524, delta_cos=+0.01184 avg, MAE improves on all layers
  - Residual encoding: source-of-truth `encode_sdir` (bitmap + fp16 values), not raw float storage
  - Layer shapes consistent across all layers (ffn_up/gate: 4864×896, ffn_down: 896×4864)
  - 31AS remains next allowed phase only if explicitly requested

## 4. Invalidated / Superseded Claims

List claims that must not be reused:

- Any result using residuals generated as `R = W_ref - W_low_raw` is historical only.
- Earlier low-MAE reports from raw-W_low residuals are invalid for current claims.
- Any "strict substitutive" claim is invalid if substitutive mode generates W_ref internally.
- Any combined ffn_up/ffn_down result is invalid if family-specific orientation or artifact loading is not verified.
- Any manifest runtime result is invalid if it silently falls back to ffn_up artifact paths for ffn_down.
- 31X/31Y manifest runtime results that depended on `execute_substitutive_path()` synthesizing W_ref/W_low/R internally are superseded by 31AJ source-of-truth cleanup.
- Pre-31AJ Phase 31AH combined strict validation is superseded by Phase 31AH-RERUN against the 31AJ-clean source-of-truth runtime.
- Any claim that Phase 31AJ is `PASS_FULL_MLP_TOY_PROBE` is invalid.
- Any claim that Phase 31AL found a `PASS_WLOW_ENCODING_CANDIDATE_FOUND` viable policy at k=9-12% is INVALID — labels were swapped and the combined viability at current k was miscomputed. The corrected finding is that no combined policy is viable at current k=9-12%.
- Any claim that runtime k-selection or family-skipping policy can make the full MLP substitutive path memory-positive is invalid under current encoding.

## 5. Suspected / Unproven

Rules:
- Do not put suspected claims in "Accepted Known-Good Facts."
- Do not use suspected claims as the basis for public claims.
- If a later phase verifies a suspected claim, move it to accepted facts.
- If a later phase disproves it, move it to invalidated/superseded claims.

Current suspected/unproven items:

- ffn_gate approximation quality at selected k is measured on activation probe; full MLP behavior in actual inference is unknown
- Whether a memory-positive compact W_low format (with embedded scales) is achievable without degrading approximation below the low-only baseline is unknown
- Q2_K_M W_low decode quality vs reference is unknown (GGUF dequantize not available locally)
- Q2 dense residual encoding quality vs fp16 SDIR residual is unknown
- **Actual Q2_K decode approximation quality:** block_size=16 simulation (cos≈0.926) may differ materially from actual GGUF Q2_K decode (block_size=256). Actual Q2_K numerical behavior requires a true decoder/probe before implementation claims are trustworthy.
- **Actual IQ4_NL residual behavior:** 31AN found residuals IMPROVE under IQ4_NL decode (opposite of 31AM's simulated finding). Whether this holds for true Q2_K (type=10) is unknown.
- **Memory feasibility with true Q2_K encoding:** IQ4_NL exceeds Q4_budget; true Q2_K (2.625 bpe) would be memory-positive. If Q2_K encode becomes available, residual-on at k≤3% with true Q2_K W_low may be both memory-positive AND approximation-improving.

## 6. Current Open Blockers

Current blockers:

- Historical scripts may still contain older orientation assumptions; do not use them for current claims unless they pass the source-of-truth regression contract.
- OpenClaw/prt-lab routing remains a process issue, not a repo blocker.
- Full MLP toy probe artifact budget fails at current encoding. 31AL identified Q2_K as viable W_low format but had label/byte errors. 31AL-R corrected: Q2_K + int8 sparse residual at k≤3% is the only viable path found at current k. 31AM found that residual-on (magnitude-sparse) policies ALL hurt approximation under simulated Q2-like W_low. 31AN found residuals IMPROVE under actual IQ4_NL decode — but IQ4_NL exceeds Q4 budget and true Q2_K encode is NotImplemented. Memory-positive Q2_K path requires true Q2_K encode implementation. Actual Q2_K numerical behavior remains unverified.

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
**Phase 31AS — Full Model (all 32 layers) Q2_K + Low-k Residual Sweep, only if explicitly requested.**

Findings from 31AR:
- Layers 0-5 all pass with k=2% (margin=+537,844 aggregate, worst_layer=+89,524)
- All 6 layers memory-positive at k=2%, cosine and MAE improve on all layers
- k=3% fails memory on all layers
- Q2_K encode via llama.cpp verified byte-exact for all 18 tensor checks
- encode_sdir residual format confirmed operational for full MLP composition
- Layer0 full MLP passes at k=2% (margin=+89,892, delta_cos=+0.0165)
- k=0.5% is most conservative (margin=+482,124), k=2% is best quality
- All three families (up/gate/down) are memory-positive individually
- Residual improves cosine AND MAE at every tested k
- encode_sdir residual format confirmed operational for full MLP composition

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
