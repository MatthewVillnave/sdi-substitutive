# SOURCE_OF_TRUTH.md

## 0. Current Working State

- **SOT version / state label:** v2 (post-31BT, SOT-01 hardened)
- **Last verified commit:** `17c2281c` (Phase 31BT: verify 1.5B MLP orientation parity)
- **Current selected policy:** `corrected_q2k_policy_v1` (corrected_ceil_per_row Q2_K, ffn_up+ffn_gate SDIR k=0.5%, alpha=1.0, no ffn_down residual)
- **Frozen checkpoint:** `phase31bn-corrected-q2k-full-aggregate-checkpoint` → `0304590c92d43fdf48d3d28998255d39c9a20c07`
- **Current model under test:** Qwen2.5-0.5B-Instruct (validated for the frozen checkpoint); Qwen2.5-1.5B-Instruct Q4_K_M (downloaded, orientation parity confirmed via 31BT, no tensor validation yet)
- **Current allowed scientific next phase:** **Phase 31BU — Qwen2.5-1.5B Corrected Q2_K Anchor Probe, only if explicitly requested.** Must use 1.5B Q4_K_M as W_ref; corrected Q2_K policy; layer 0 only or small fixed set; no aggregate without explicit approval.
- **Current blockers:** none.
- **Active forbidden claims (summary; full canonical master list in Section 0.A):** no model quality/behavior recovery claim, no speedup, no full-model runtime memory savings, no llama.cpp integration, no production readiness, no inference/generation, no larger-model validation claim unless explicitly proven, no runtime-ready output-residual claim, no claim beyond standalone tensor harness unless proven, no orientation claim for a larger model unless parity-tested, no commit/push/tag/download without explicit Matt approval where applicable.
- **Current validation scope:** standalone tensor harness. Qwen2.5-0.5B for accepted numeric metrics (31BN freeze). Qwen2.5-1.5B orientation-only (31BT) — no tensor validation, anchor, or aggregate.
- **Current artifact/package status:** `corrected_q2k_policy_v1` package (frozen via 31BO); 0.5B 31BN aggregate freeze; 1.5B 31BS download+metadata; 1.5B 31BT orientation parity result.

**How agents should use this file:**
1. Read **Section 0** first to confirm current state.
2. Use **Section 3** as audit trail. Do **not** infer current state from old phase entries — old claims are superseded by newer accepted facts (Section 3) / invalidated facts (Section 4).
3. **Section 9** states the *currently allowed next phase*. If Section 0 and Section 9 disagree, **Section 9 wins** until reconciled.
4. **Section 0.A** holds the canonical master forbidden-claims list. Per-phase forbidden-claim notes elsewhere are audit context, not the canonical list.
5. **Section 13** documents the commit/approval workflow and approval keywords.
6. **Section 14** documents the full phase workflow protocol (request → regression → report → approval → next phase).

**Future SOT v3 split plan (not done in SOT-01; for reference only):**
- `SOURCE_OF_TRUTH.md` — current working state (this section)
- `PHASE_LOG.md` — historical audit trail (Section 3 entries, all phase history)
- `SOURCE_OF_TRUTH.json` — machine-readable current state
- `FORBIDDEN_CLAIMS.md` — canonical forbidden claims (Section 0.A)
- `WORKFLOW_PROTOCOL.md` — agent workflow rules (Sections 13, 14)

## 0.A. Master Forbidden Claims (Canonical)

This is the **canonical master list** of forbidden claims. Per-phase forbidden-claim notes elsewhere in this file (e.g. inside individual Section 3 phase entries) are audit context for that specific phase, not the canonical list. If a per-phase note conflicts with this master list, the master list wins.

1. no model quality recovery claim
2. no behavior recovery claim
3. no speedup claim
4. no full-model runtime memory savings claim
5. no llama.cpp integration claim
6. no production readiness claim
7. no inference/generation claim
8. no larger-model validation/result claim unless explicitly proven by a parity-tested anchor+aggregate pass on that model
9. no runtime-ready output-residual claim
10. no claim beyond standalone tensor harness unless proven
11. no claim that 31AY / 31BA exact anchors are current canonical metrics (they are historical only)
12. no orientation claim for a larger model unless parity-tested (e.g. via a 31BT-style micro-probe)
13. no commit, push, tag, push-tag, or download (any new model) without explicit Matt approval where applicable (per-phase scope governs which approvals are required)

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
- Phase 31AR-FREEZE checkpoint created for layers 0–5 full MLP Q2_K + k=2% residual:
  - Tag: `phase31ar-layers0-5-mlp-q2k-lowk-checkpoint` at HEAD `cc9fa391bc8f92525053c9a6f4298861d8434873`
  - Classification: `PASS_LAYERS0_5_MLP_Q2K_LOWK_POLICY_FOUND` — checkpointed as historically verified
  - Accepted claim: layers 0–5 full MLP substitutive prototype passes; k=2%; all 18 Q2_K checks byte-exact; all 6 layers memory-positive; all 6 layers improve cosine and MAE
  - Checkpoint artifacts: `docs/PHASE31AR_FREEZE_LAYERS0_5_MLP_Q2K_LOWK_CHECKPOINT.md`, `results/PHASE31AR_FREEZE_LAYERS0_5_MLP_Q2K_LOWK_CHECKPOINT.json`
  - Forbidden claims unchanged from 31AR specification
  - Next phase remains 31AS (full 32-layer sweep), only if explicitly requested
- Phase 31AS (all available layers full MLP Q2_K + low-k residual sweep) — classification `PARTIAL_LAYER_VARIANCE` (corrected from initial `PASS_ALL_LAYERS_MLP_Q2K_LOWK_POLICY_FOUND`):
  - **24 layers discovered** (indices 0–23; Qwen2.5-0.5B has 24 FFN layers, not 32)
  - **All 72 Q2_K tensor checks PASS** (byte_match=True for all layers/families)
  - Q2_K via llama.cpp `quantize_row_q2_K_ref` + `dequantize_row_q2_K`: 1,430,016 bytes/family, margin=+749,056/family
  - **Aggregate per policy:**
    - k=0.5%: agg_margin=+11,567,890, worst=+481,950, n_mem_pos=24/24, n_cos_pos=22/24
    - k=1%:   agg_margin=+8,428,606, worst=+351,076, n_mem_pos=24/24, n_cos_pos=23/24
    - k=2%:   agg_margin=+2,152,334, worst=+89,524, n_mem_pos=24/24, n_cos_pos=23/24
    - k=3%:   agg_margin=−4,126,454, worst=−172,222, n_mem_pos=0/24 — **fails memory**
  - **Layer 21 issue:** cosine regresses at all k > 0 (structural property, not k-dependent). delta_cos: k=0.5%→−0.00785, k=1%→−0.02577, k=2%→−0.02349. MAE improves at all k > 0.
  - **No policy achieves 24/24 cosine-positive.** Best is 23/24 (k=1% and k=2%).
  - **All policies achieve 24/24 memory-positive and 24/24 MAE-improving.**
  - **Selected policy: k=1%** — most conservative among memory-positive policies (agg_margin=+8,428,606, worst=+351,076)
  - Residual encoding: `encode_sdir` (bitmap + fp16 values) — same as 31AR
  - Model: Qwen2.5-0.5B (24 layers, ffn_up/gate: 4864×896, ffn_down: 896×4864)
  - No full-model claim beyond 24 available layers
  - 31AT remains next allowed phase only if explicitly requested
  - **31AS-R audit note:** Original `PASS_ALL_LAYERS_MLP_Q2K_LOWK_POLICY_FOUND` claim was incorrect — layer 21 cosine regression at all k > 0 was missed in initial report. Corrected to `PARTIAL_LAYER_VARIANCE`. k=1% selected as most conservative memory-positive policy.
  - **31AT diagnosis (partial):** Classification `PARTIAL_LAYER21_ACTIVATION_SENSITIVE`.
    - Layer 21 is **activation-sensitive, not structurally failing**. With seed=42: delta_cos=+0.02151 (improves). With seed=63 (31AS): delta_cos=−0.02577 (regresses).
    - 31AS used per-layer seed (seed=42+layer_idx), which produced an outlier result for layer 21.
    - 9/10 seeds: cosine improves; MAE improves on all 10/10 seeds.
    - MAE improves at all k for all seeds — residual works correctly.
    - Full 24-layer consistent-seed resweep timed out. 31AS corrected classification would be `PASS_ALL_LAYERS_MLP_Q2K_LOWK_POLICY_FOUND` with consistent seed=42.
    - Layer 21 activation sensitivity is a methodological artifact, not a policy failure.
    - Next phase: 31AU (full 24-layer resweep with consistent seed=42) or 31AT-FREEZE checkpoint, only if explicitly requested.
  - **31AU PASS:** Classification `PASS_ALL_LAYERS_CONSISTENT_SEED_POLICY_FOUND`:
    - Consistent seed=42 across all 24 layers: **all 24/24 pass at k=1%**
    - n_mem_pos=24/24, n_cosine_positive=24/24, n_MAE_improving=24/24
    - agg_margin=+8,428,606, worst_margin=layer3(+351,076)
    - worst_cosine=layer4(delta_cos=+0.00500)
    - avg_delta_cos=+0.01133
    - k=2% also passes: n_mem_pos=24/24, n_cosine_positive=24/24, n_MAE_improving=24/24
    - k=0: baseline (no residual) — n_mem_pos=24/24, but cosine and MAE unchanged (as expected)
    - **31AS-R `PARTIAL_LAYER_VARIANCE` was a per-layer seed artifact** — with consistent seed=42, all 24 layers pass
    - Residual encoding: `encode_sdir`; model: Qwen2.5-0.5B (24 FFN layers)
    - **31AU-R label-only fix:** MD table column `MAE_delta` renamed to `MAE_improvement`; actual numeric values unchanged; MAE_delta formula confirmed: `MAE_sub - MAE_low`; `n_MAE_improving=24/24` unchanged; pass/fail unchanged.
    - **31AT-FREEZE checkpoint:** Classification `PASS_31AT_FREEZE_ALL24_MLP_Q2K_LOWK_CHECKPOINT`.
      - Tag: `phase31at-all24-mlp-q2k-lowk-checkpoint` → `899022d4730edf9f1ea56c599e49561a5081d333`
      - Accepted claim: all-24-layer full MLP Q2_K + k=1% residual, consistent seed=42, 24/24 memory/cosine/MAE-positive
      - Known limitations: standalone harness only; no llama.cpp integration; no generation claim; no speed claim; no larger-model claim
    - **31AV robustness characterization:** Classification `PARTIAL_MULTI_SEED_COSINE_SENSITIVE`.
      - Seeds tested: 0, 5, 9 at k=1%.
      - 2/3 seeds (0, 5): all 24 layers pass all gates — memory, cosine, MAE all positive.
      - 1/3 seeds (9): layer 21 cosine regresses severely (−0.14606) but MAE still improves and memory stays positive.
      - MAE fully robust: 24/24 layers x 3/3 seeds all improve.
      - Memory fully robust: 24/24 layers x 3/3 seeds all positive.
      - Layer 21 is activation-sensitive at specific seeds (seed=9); seed=0 and seed=5 pass.
      - Consistent with 31AT finding (layer 21 sensitive at 1/10 seeds).
      - Only 3 seeds characterized; broader seed space not fully tested.
    - **31AW layer 21 seed=9 diagnosis:** Classification `PARTIAL_LAYER21_SEED9_METRIC_CONFLICT`.
      - No alpha (0.0-1.5), k (0-2%), or family-subset policy fixes cosine regression for layer 21/seed=9.
      - All 7 family subsets (up, gate, down, up+gate, up+down, gate+down, all) regress cosine — not family-specific.
      - MAE improves at all memory-positive policies; cosine regresses — metric conflict.
      - Error geometry: residual increases output norm +7.4% (6.1259 to 6.5807) and rotates direction significantly.
      - Layer 21 is activation-sensitive not structurally broken: seeds 0 and 5 pass cleanly.
      - This is a metric-conflict case; no policy achieves both cosine and MAE improvement for this activation.
    - **31AX activation-space probe:** Classification `PARTIAL_NO_FIX_METRIC_CONFLICT_CONFIRMED`.
      - Cosine is scale-invariant: scaling Y_sub to match ||Y_ref|| or ||Y_low|| changes MAE but NOT cosine.
      - Norm matching improves MAE significantly but cosine is completely unchanged.
      - Output interpolation: no beta simultaneously improves cosine and MAE.
      - Oracle projection (proj_T(D)) fails for seed=9: cos still −0.097, MAE worsens.
      - Safe seeds (0, 5) WORK CORRECTLY with oracle — oracle projection actually improves cosine for them.
      - Root cause: residual direction D is fundamentally misaligned with reference correction T for seed=9/layer=21 specifically.
      - The problem is the residual direction itself, not scale or parameter choice.
    - **31AY layer 21 sensitivity map:** Classification `PARTIAL_LAYER21_ACTIVATION_SENSITIVE`.
      - Layer 21 (seeds 0-63): cosine failure rate=12.5% (8/64), MAE failure rate=7.8% (5/64), memory 100%.
      - Severe regressions: 2/64 seeds (seed=9: delta_cos=-0.146; worst overall).
      - Mild regressions: 6/64 seeds.
      - Mean delta_cos=+0.02968, median=+0.01419 — overall positive.
      - Layers 20 and 22 (seeds 0-31 each): cosine failure rate=0.0% (0/32 each).
      - cos_low is dominant predictor of cosine regression: r=-0.82 with delta_cos.
      - High baseline cosine (Q2_K already close to FP16) predicts cosine regression — regression-to-mean effect.
      - Layer 21 is activation-sensitive but not systematic; 87.5% of seeds pass.
      - Not multi-layer sensitivity; neighboring layers fully robust.
    - **31AZ gating policy evaluation:** Classification `PARTIAL_SKIP_POLICY_TRADEOFF`.
      - Oracle cos_low gate: best at thr=0.90 gives cos_pos=45/64 severe=1 — WORSE than baseline 56/64 severe=2.
      - Oracle gating trades false skips for true skips without improving pass rate.
      - Best runtime proxy gate (norm_ratio_sub_low<1.15): cos_pos=52/64 severe=2 — slightly worse than baseline.
      - No runtime proxy reduces severe regressions.
      - Always-skip L21: eliminates regressions (64/64) but loses all MAE improvement.
      - Oracle gate destroys L20/L22 performance (0/32 cos_pos) — not layer-transferable.
      - Key: corr(cos_low, residual_update_norm)=+0.018 (negligible) — residual direction not predictable from runtime features.
      - No gating policy meaningfully improves over always-on for layer 21.
      - Accept the 12.5% cosine failure rate as the cost of the residual approach.
    - **31BA full 24-layer multi-seed aggregate:** Classification `PARTIAL_LAYER21_DOMINANT_SENSITIVITY`.
      - 24 layers x 16 seeds = 384 seed-layer pairs tested.
      - Cosine improved: 382/384 (99.48%). MAE improved: 382/384 (99.48%). Memory positive: 384/384 (100%).
      - Severe regressions: 1/384 (0.26%) — layer 21 seed 9, δ_cos=-0.14606.
      - Cosine failures: 2/384 (0.52%) — layer 2 mild + layer 21 severe.
      - All 22 other layers: robust (0 cosine failures across 16 seeds each).
      - Layer 21: 15/16 seeds pass (93.75%); layer 2: 15/16 seeds pass.
      - Layer 21 is sole source of severe regressions; accounts for 100% of severe cases.
      - Seed 9 is precise layer21-specific outlier — passes all other layers cleanly.
      - Network-wide corr(delta_cos, cos_low) = -0.55 — weaker than layer21-only (r=-0.82).
      - Residual approach is broadly robust network-wide.
    - **31BB k-parameter sensitivity:** Classification `PARTIAL_K_TRADEOFF` (full 384-pair aggregate).
      - Aggregate (384 pairs, all 24 layers × 16 seeds):
        - k=0.5: cos_fail=5, severe=0, mae_fail=5, mean_delta_cos=+0.00881
        - k=1.0: cos_fail=2, severe=1, mae_fail=2, mean_delta_cos=+0.01265
        - k=1.5: cos_fail=1, severe=1, mae_fail=0, mean_delta_cos=+0.01637
      - Layer 21 seed 9 severe: only k=0.0 fixes it (eliminates severe, cosine positive). No k>0 fixes it.
      - k=0.50 is non-severe (0 severe) but 5 cosine failures — too many.
      - Layer 2 seed 7 mild failure: fixed by k>=1.5 (both cosine and MAE improve).
      - k>=1.5 would worsen layer 21 seed 9 (larger residual = worse misalignment).
      - k=1.0 remains best aggregate default: lowest cos_fail (2) among candidates with severe=1.
      - k=1.5 is an alternative: 1 cos_fail (best) but same severe count as k=1.0.
      - k-sweep confirms: problem is residual direction, not parameterization.
    - **31BC output residual probe:** Classification `PARTIAL_OUTPUT_RESIDUAL_FIXES_BUT_NOT_STATIC`.
      - Output residual fp16 at k=1% (54 bytes) completely fixes L21-seed9 severe regression: delta_cos from −0.146 to +0.105.
      - Output residual at k=5% (270 bytes) achieves 0/32 aggregate failures (layers 2+21, seeds 0–15).
      - Output residual at k=1% (54 bytes) reduces aggregate cosine failures from 10/32 to 3/32 vs WR k=1.
      - Dense oracle confirms: output residual direction is correct — the problem is weight residual formulation, not residual amount.
      - Int8 output residual is broken (zero improvement) due to dynamic range clipping.
      - Output residual is activation-specific: cannot be precomputed statically, requires Y_ref at runtime — not static-deployable.
      - L21_seed14 (near-1.0 baseline) shows minor regression at k=1% output residual (−0.0004) but recovers at k=5%+.
      - Classification: fix exists but is not static/runtime-deployable with current architecture.
    - **31BC-R report hygiene:** Classification `PASS_31BC_R_REPORT_RECONCILED`.
      - Patched placeholders in 31BC doc: replaced `[pending commit]` → actual SHA, `[pending]` → confirmed push.
      - Fixed JSON classification: `PENDING` → `PARTIAL_OUTPUT_RESIDUAL_FIXES_BUT_NOT_STATIC`.
      - Added bug-narrative section to 31BC doc: np.array.flatten() copy bug, fix with Y_out.flat[indices], stale output superseded.
      - No numeric or scientific result changed; all values verified consistent with corrected run.
    - **31BD residual formulation decision:** Classification `PASS_31BD_ACCEPT_STATIC_DEFAULT_WITH_KNOWN_OUTLIER`.
      - Decision: accept static Q2_K + k=1% weight residual as current default; defer output residual to future dynamic-architecture research.
      - k=1.0 confirmed as pragmatic default: 2 cosine failures, 1 severe, 2 MAE failures on 384-pair aggregate.
      - k=1.5 noted as legitimate alternative: 1 cosine failure, 1 severe, 0 MAE failures.
      - Rare outlier (L21-seed9, 0.26%) accepted as documented limitation.
      - Output residual confirmed as fix-exists-but-not-static; deferred to future dynamic architecture work.
      - Future work: dynamic/output residual architecture, learned correction, larger model testing.
      - llama.cpp integration: only after more proof, not current path.
    - **31BE post-checkpoint roadmap:** Classification `PASS_31BE_ROADMAP_STATIC_ARTIFACT_HARDENING_SELECTED`.
      - Selected lane: static artifact spec + regression hardening.
      - Rejected for now: larger-model validation (needs hardened schema first), llama.cpp integration (needs stable artifact format), robustness extension (diminishing returns; mini-task in 31BF), technical writeup (sub-artifact in 31BF).
      - Rationale: current artifact schema is implicit and script-dependent; phase scripts contain hardcoded private paths; regression suite is minimal; must harden foundation before 1.5B/3B validation or integration.
      - Proposed next phase: Phase 31BF — Static Artifact Spec + Regression Hardening.
    - **Phase 31BF — Static Artifact Spec + Regression Hardening:** Classification `PASS_31BF_STATIC_ARTIFACT_SCHEMA_HARDENED`.
      - Schema v1.0 documented in `docs/STATIC_ARTIFACT_SCHEMA.md` with full field specifications, metric conventions, byte accounting, path rules, orientation contract, and bundle types.
      - `bundle_manifest.py` updated: added schema constants (`SCHEMA_VERSION_ACCEPTED`, `ALLOWED_FAMILIES`, `CANONICAL_ORIENTATION`, `MIN_DELTA_COS_ACCEPT`, `MAX_SEVERE_DELTA_COS`, `MAE_IMPROVED_MAX_DELTA`, `Q4_BUDGET_FAMILY`, `Q4_BUDGET_LAYER`), updated family validation to include `ffn_gate`, added orientation check.
      - Schema version validation updated: accepts both `"0.2.0"` and `"1.0"`.
      - Regression suite updated: imports schema constants, uses v1.0 manifest for fixtures, added `test_metric_convention_sanity()` and `test_schema_validation_smoke()` (5 negative test cases: wrong schema version, missing family, wrong orientation, bad family, negative margin — all pass).
      - Private path audit completed: 50+ phase scripts contain hardcoded private paths; core infrastructure (`bundle_manifest.py`, `phase31x_manifest_runtime.py`, regression suite) is clean. Phase scripts remain to be hardened in future work.
      - Reproducibility notes created at `docs/REPRODUCIBILITY_NOTES.md`.
      - No accepted numeric result changed.
    - **Phase 31BG — Clean Reproduction Run:** Classification `PARTIAL_31BG_SCHEMA_VALIDATES_QUANT_MISMATCH`.
      - Schema v1.0 validation: PASS (both L21-S9 and L21-S0 bundles validated correctly).
      - Anchor reproduction: PARTIAL — quantization format mismatch blocks exact reproduction.
      - Root cause: `pack_wlow` (reference custom quantization) ≠ llama.cpp Q2_K; accepted anchor values used Q2_K via `lib.quantize_row_q2_K_ref`; manifest runtime uses `pack_wlow` which produces near-lossless quantization (cos_low≈0.995) vs actual Q2_K degradation (cos_low≈0.795).
      - Aggregate reproduction: SKIPPED (expensive, anchor-level diagnosis sufficient).
      - New portable runner created at `src/phase31bg_clean_reproduction.py` using env vars only, no private paths.
      - Core infrastructure confirmed clean of private paths.
      - No committed numeric result changed.
    - **Phase 31BH — Q2_K Runtime Quantization Path (partial):** Classification `PARTIAL_31BH_Q2K_BACKEND_ADDED_ANCHOR_MISMATCH`.
      - Q2_K backend created (`src/q2k_backend.py`) with correct llama.cpp ctypes wrappers.
      - Schema v1.0 support added to `bundle_manifest.py` (Q2_K format constant and load method).
      - Portable runner created (`src/phase31bh_q2k_clean_reproduction.py`) using env vars only.
      - cos_low ≈ 0.897 observed vs 0.795 expected — anchors did NOT reproduce.
      - At time of initial 31BH classification, "prompt-derived activation" was hypothesized but NOT proven from scripts.
    - **Phase 31BH-R — Q2_K Anchor Provenance Reconciliation:** Classification `PARTIAL_31BH_R_X_PROVENANCE_MISMATCH`.
      - Full fingerprint audit of OLD (31AY/31BA) vs NEW (31BH) code paths completed.
      - **Root cause 1 (PRIMARY): X vector RNG mismatch** — `np.random.RandomState(seed)` (31BH) ≠ `np.random.default_rng(seed)` (31AY). These produce completely different sequences. For seed=9: OLD=[-0.803,...], NEW=[0.001,...]. Separation experiment: X swap changes cos_low by ~0.75. This is the dominant cause.
      - **Root cause 2 (SECONDARY): Q2_K buffer under-allocation** — OLD uses floor(d_out*d_in/QK_K)*84=1,430,016 bytes (truncates 128 elements/row). NEW uses ceil-based 1,634,304 bytes. Effect ~0.19 on cos_low.
      - **Residual construction: VERIFIED MATCH** — both use `encode_sdir(W_ref-W_low, k_pct=1.0)`, same nnz=43,616, same SDIR bytes.
      - **W_ref extraction: VERIFIED MATCH** — same GGUF model, same tensor, same dequantization.
      - **"Prompt-derived activation": REJECTED** — 31AY uses pure Gaussian random X, no prompts/tokenization/inference.
      - Q2_K backend (`q2k_backend.py`) is correctly implemented; W_low difference is a floor-vs-ceil accounting choice, not a correctness error.
      - Separation-of-effects (L21-S9): cos_low = 0.79207 (OLD_X+OLD_Wlow) vs -0.46109 (NEW_X+NEW_Wlow) vs 0.98211 (NEW_X+OLD_Wlow) vs 0.04225 (OLD_X+NEW_Wlow).
      - Recommended fix: use `np.random.default_rng(seed)` + decide floor-vs-ceil for W_low byte target.
      - Artifacts: `docs/PHASE31BH_R_Q2K_ANCHOR_PROVENANCE.md`, `results/PHASE31BH_R_Q2K_ANCHOR_PROVENANCE.json`.
      - Phase 31BH files (q2k_backend.py, bundle_manifest.py, schema docs): kept, not overstating anchor reproduction.
      - Next phase: 31BH-R2 to fix X vector and W_low buffer, then re-run anchor reproduction.
    - **Phase 31BH-R2 — Q2_K Anchor Reproduction Fix / Floor-vs-Ceil Decision:** Classification `PASS_31BH_R2_FIX_RUNTIME_DISPATCH_REPAIRED`.
      - Root causes of 31BH-R failure (3 bugs found and fixed):
        1. **X RNG:** Changed `np.random.RandomState(seed)` → `np.random.default_rng(seed)` (matches OLD 31AY/31BA path)
        2. **W_ref provenance:** OLD 31AY uses raw GGUF weights as W_ref; Q2_K roundtrip applied separately for all 3 MLP families
        3. **MLP residual:** OLD 31AY applies SDIR residual to ALL 3 families (ffn_up, ffn_gate, ffn_down), not just ffn_up
      - X fingerprints confirmed (np.random.default_rng):
        - seed=0: shape=(1,896), norm≈29.18, SHA256=550ad1f0...
        - seed=9: shape=(1,896), norm≈29.41, SHA256=b4c9c8e3...
      - Runtime dispatch repaired:
        - Added `"packed_nibble_v0.1"` to accepted SDIW format aliases in bundle_manifest.py validation
        - Fixed `execute_substitutive_path()` to dispatch by `W_low_format`: SDIW formats → `load_sdiw()`, Q2_K → `load_q2k()` + dequantize
        - Q2_K runtime raises clear error if backend unavailable — no silent fallback to SDIW
        - Added `q2k_loaded` counter to runtime counters
        - Added `w_low_format` field to runtime info output
      - Regression result: `PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN` — error_count=0, fallback_count=0
      - Phase 31BH files (q2k_backend.py, bundle_manifest.py, phase31bh_q2k_clean_reproduction.py): untracked, pending full anchor re-verification
      - Classification: `PASS_31BH_R2_FIX_RUNTIME_DISPATCH_REPAIRED` — regression clean, runtime dispatch fixed
    - **Phase 31BJ — Corrected Q2_K Mode Rebaseline:** Classification `PARTIAL_31BJ_CORRECTED_Q2K_MEMORY_FAIL`.
      - Anchor results (k=1%, alpha=1.0, all 3 MLP families, np.random.default_rng):
        - `historical_floor_flat` mode: margin ≈ +351,242 bytes/layer (memory-positive), cosine improves on all tested anchors
        - `corrected_ceil_per_row` mode: margin ≈ −261,654 bytes/layer (memory-negative), cannot be used at k=1%
      - Memory accounting:
        - historical_floor: Q2_K 1,430,016 × 3 + SDIR ≈ 4.50 MB vs 3×Q4_budget=6.54 MB → margin ≈ +2.04 MB per layer
        - corrected: Q2_K 1,634,304 × 3 + SDIR ≈ 5.11 MB vs 6.54 MB → margin ≈ −1.43 MB per layer
      - **Corrected Q2_K mode is not memory-positive at k=1%** with all 3 families.
      - Historical Phase 31AY/31BA anchor values (cos_low=0.794913, delta_cos=-0.146059 for L21-S9) are **legacy provenance only** — do NOT reproduce with current code.
      - Full 32-pair or 384-pair aggregate not completed (ctypes overhead).
      - Runner: `src/phase31bj_corrected_q2k_rebaseline.py` (untracked).
      - Results: `src/results/PHASE31BJ_CORRECTED_Q2K_REBASELINE.json`.
      - Corrected mode cosine behavior at tested anchors is similar or better than historical floor, but memory fails — moot.
    - **Phase 31BK — Corrected Q2_K Mode Memory Policy Retune:** Classification `PASS_31BK_CORRECTED_Q2K_MEMORY_POLICY_FOUND`.
      - k sweep confirmed: corrected mode all-3-families is memory-positive up to k≈0.75%, fails at k=1%.
      - Family subset sweep: up+gate is the memory/quality sweet spot.
      - **Selected policy: corrected_ceil_per_row, up+gate families only, k=0.5%.**
      - Memory accounting (selected policy per layer):
        - Q2_K up: 1,634,304 | Q2_K gate: 1,634,304 | Q2_K down: 1,430,016
        - SDIR up (k=0.5%): 588,376 | SDIR gate (k=0.5%): 588,376 | SDIR down: 0
        - Total: 5,874,376 | Margin: +662,840 | memory-positive: true
      - 3-anchor comparison (selected vs historical floor):
        - L21-S9 delta_cos: +0.1575 vs +0.0172 (9x better)
        - L21-S0 delta_cos: +0.3152 vs +0.0880 (3.6x better)
        - L2-S7 delta_cos: +0.0203 vs +0.0206 (comparable)
        - MAE improves on all anchors under both policies
        - No severe regressions under either policy
      - Key insight: down-family excluded from residual — its transposed shape (896×896) means corrected ceil matches historical floor, and it provides Q4 budget slack.
      - Runner: `src/phase31bk_corrected_q2k_memory_policy_retune.py` (not written; direct computation was sufficient).
      - Results: `src/results/PHASE31BK_CORRECTED_Q2K_MEMORY_POLICY_RETUNE.json`.
    - **Phase 31BL — Corrected Q2_K Small Aggregate Validation:** Classification `PARTIAL_31BL_CORRECTED_Q2K_MINOR_FAILURES`.
      - 32-pair aggregate (L2+L21, seeds 0–15): 32/32 memory-positive, 31/32 cosine-improved, 31/32 MAE-improved, 0 severe regressions.
      - 2 isolated minor failures: L21-S10 (cosine failure dc=−0.0294, not severe, MAE improves), L2-S13 (MAE regression +0.00783, cosine improves).
      - Anchor consistency: L21-S9 dc=+0.1575, L21-S0 dc=+0.3152, L2-S7 dc=+0.0203 — all match 31BK expectations exactly.
      - Selected policy validated on small aggregate: memory-positive on all 32 pairs, mean delta_cos=+0.0545, median delta_cos=+0.0302.
      - Layer-specific: L2 mean dc=+0.0293 (16/16 cos-improved), L21 mean dc=+0.0606 (15/16 cos-improved).
      - Results: `src/results/PHASE31BL_CORRECTED_Q2K_SMALL_AGGREGATE.json`.
    - **Phase 31BM — Corrected Q2_K Broader Aggregate Validation / Minor Failure Tracking:** Classification `PASS_31BM_CORRECTED_Q2K_BROADER_AGGREGATE_VALIDATED`.
      - Route A: 24 layers × 16 seeds = 384 pairs.
      - 384/384 memory-positive (min margin 661,766 bytes/layer), 383/384 cosine-improved (99.74%), 383/384 MAE-improved (99.74%), 0 severe regressions.
      - Policy status: STRONG VALIDATION (all 4 strong criteria met).
      - Known 31BL failures both reproduce exactly and remain minor: L21-S10 dc=−0.0294, L2-S13 md=+0.0078.
      - No new failures found across 22 additional layers.
      - Aggregate margin: ~254 MB (254,130,400 bytes) for full 24-layer model.
      - Runner: `src/phase31bm_corrected_q2k_broader_aggregate.py`.
      - Results: `src/results/PHASE31BM_CORRECTED_Q2K_BROADER_AGGREGATE.json`.
    - **Phase 31BN — Corrected Q2_K Full Aggregate Checkpoint / Freeze:** Classification `PASS_31BN_FREEZE_PREPARED_CORRECTED_Q2K_FULL_AGGREGATE`.
      - This is a freeze/checkpoint phase — no new science.
      - Canonical selected policy frozen: corrected_ceil_per_row Q2_K, ffn_up+ffn_gate residual, k=0.5%, alpha=1.0, no ffn_down residual.
      - Validated scope: Qwen2.5-0.5B, all 24 FFN layers, seeds 0–15, 384 pairs, Route A.
      - Full aggregate: 384/384 memory-positive, 383/384 cosine-improved (99.74%), 383/384 MAE-improved (99.74%), 0 severe regressions.
      - Mean delta_cos=+0.0383, median delta_cos=+0.0351, aggregate margin ~254 MB.
      - Known minor failures (accepted non-severe): L21-S10 (cosine failure dc=−0.0294), L2-S13 (MAE regression md=+0.0078).
      - Checkpoint target commit: `0304590c92d43fdf48d3d28998255d39c9a20c07`.
      - Proposed tag: `phase31bn-corrected-q2k-full-aggregate-checkpoint` (not created until approved).
      - Freeze doc: `docs/PHASE31BN_CORRECTED_Q2K_FULL_AGGREGATE_CHECKPOINT.md`.
      - Freeze results: `src/results/PHASE31BN_CORRECTED_Q2K_FULL_AGGREGATE_CHECKPOINT.json`.
      - Prior accepted numeric results were not changed.
      - **Accepted claim:** 31BN proves the corrected Q2_K policy (`corrected_q2k_policy_v1`, corrected_ceil_per_row, ffn_up+ffn_gate SDIR k=0.5%, alpha=1.0, no ffn_down residual) is memory-positive and strongly validated for Qwen2.5-0.5B in a standalone full-MLP tensor harness (384/384 memory-positive, 383/384 cosine-improved, 383/384 MAE-improved, mean delta_cos=+0.0383, median delta_cos=+0.0351, aggregate margin ~254 MB).
      - **Valid as long as:**
        - model is Qwen2.5-0.5B-Instruct
        - scope is standalone full-MLP tensor harness, not runtime inference
        - policy remains `corrected_ceil_per_row` Q2_K + ffn_up/ffn_gate SDIR k=0.5%, alpha=1.0, no ffn_down residual
        - Q4 budget accounting (per the policy package) remains unchanged
        - canonical orientation convention (Section 7) remains unchanged
        - no later phase invalidates or supersedes this result
    - **Phase 31BO — Corrected Q2_K Artifact/Policy Package Hardening:** Classification `PASS_31BO_CORRECTED_Q2K_POLICY_PACKAGE_HARDENED`.
      - Created canonical policy package: `docs/CORRECTED_Q2K_POLICY_PACKAGE.md` and `src/results/CORRECTED_Q2K_POLICY_PACKAGE.json`.
      - Package version: `corrected_q2k_policy_v1` — records selected policy, artifact formats, env vars, memory accounting, frozen validation.
      - Created constants helper `src/corrected_q2k_policy.py` (stdlib-only, no model loading): `describe_policy()`, `validate_policy_dict()`.
      - Created smoke test `tests/test_corrected_q2k_policy.py` — no model files or llama.cpp required.
      - Added policy constants smoke test to `tests/run_source_of_truth_regression.py` — runs without model files.
      - Updated `docs/REPRODUCIBILITY_NOTES.md` with corrected Q2_K policy package section.
      - No numeric/scientific result changed in this phase.
      - No private paths added to new package files.
      - Private path debt exists in older phase scripts (31AG–31AW range) — documented, not remediated in this phase.
      - Next allowed phase: Phase 31BP — Corrected Q2_K Larger-Model Feasibility Planning (only if explicitly requested).
      - **Accepted claim:** `corrected_q2k_policy_v1` is the canonical policy package: corrected_ceil_per_row Q2_K, ffn_up+ffn_gate SDIR k=0.5%, alpha=1.0, no ffn_down residual; documented in `docs/CORRECTED_Q2K_POLICY_PACKAGE.md` and `src/results/CORRECTED_Q2K_POLICY_PACKAGE.json`; smoke-tested via `tests/test_corrected_q2k_policy.py` and the regression suite's `metric_convention_sanity` / `schema_validation_smoke` checks.
      - **Valid as long as:**
        - package version remains `corrected_q2k_policy_v1` (any change must bump the version and freeze a successor)
        - the five policy parameters (Q2_K mode, residual families, k, alpha, ffn_down residual) match the values above
        - the package doc and JSON are kept in sync
        - the smoke test continues to pass in `tests/run_source_of_truth_regression.py`
        - no later phase supersedes the package (e.g. `corrected_q2k_policy_v2` with different parameters)
        - the underlying 31BN frozen numeric result remains valid (the policy parameters must not be silently loosened)
    - **Phase 31BP — Corrected Q2_K Larger-Model Feasibility Planning:** Classification `PASS_31BP_LARGER_MODEL_FEASIBILITY_PLAN_READY`.
      - Planning only — no larger-model validation was executed, no models were downloaded.
      - Model availability: Qwen2.5-0.5B locally available; 1.5B/3B/7B not locally available.
      - Verified 0.5B metadata (from GGUFReader): n_layers=24, hidden_size=896, intermediate_size=4864, ffn_up shape=[896,4864] in GGUF (storage/representation format in GGUFReader — NOT the canonical artifact orientation). Canonical MLP orientation: ffn_up and ffn_gate = (d_out=4864, d_in=896), ffn_down = (d_out=896, d_in=4864), using Y=X@W.T. Any GGUFReader raw shape display must be treated as reader/storage-specific and not used to redefine canonical orientation.
      - Selected policy 0.5B memory: ~5.21 MB/layer total, ~0.662 MB/layer margin.
      - Estimated larger-model metadata: 1.5B (n_layers=28, hidden=1536, int=13104), 3B (n_layers=36, hidden=2048, int=**CONFLICTING** — public spec 8192 vs prior evidence 11008; **needs GGUFReader verification**), 7B (n_layers=32, hidden=4096, int=18432) — from public spec where not marked unknown.
      - Runtime projections: 1.5B ~770s, 3B ~640s, 7B ~2710s for 384-pair aggregate (linear scaling estimate).
      - Margin risk: row-ceil overhead increases with hidden_size; margin at 0.5B may not hold at larger models without measurement.
      - Disk estimate: ~6.1 GB temporary artifacts for 384-pair larger-model run — must use temp directory, do not commit.
      - Recommended next: Phase 31BQ — Larger-Model Local Availability / Metadata Probe, only if explicitly requested — no validation execution until metadata is confirmed.
      - No numeric/scientific result changed; no larger-model execution started.
      - Next allowed phase: Phase 31BQ — Larger-Model Local Availability / Metadata Probe (only if explicitly requested) — no validation execution until metadata is confirmed.
    - **Phase 31BQ — Larger-Model Local Availability / Metadata Probe:** Classification `PARTIAL_31BQ_NO_LARGER_QWEN_LOCAL_USABLE`.
      - Metadata-only probe — no larger-model validation executed, no models downloaded, no tensor artifacts generated.
      - Local availability scan (bounded, no downloads): Qwen2.5-0.5B fully available (already validated in 31BN); Qwen2.5-32B Q4_K_M **partial** (1 of 5 shards, ~3.69 GB of ~20 GB total, metadata-extractable from shard 1 only); 1.5B/3B/7B/14B **not locally available**.
      - Qwen2.5-32B verified metadata (from GGUFReader on shard 1): n_layers=64, hidden=5120, intermediate=27648, context=131072, attn_heads=40/8 (Q/KV).
      - **Orientation caveat (31BQ):** FFN raw tensor shape display from GGUFReader is model/file/reader-specific and must not be used directly as canonical orientation. Canonical MLP orientation must be derived from tensor role and dimensions: ffn_up/ffn_gate map hidden → intermediate, so canonical artifact orientation is (d_out=intermediate, d_in=hidden); ffn_down maps intermediate → hidden, so canonical artifact orientation is (d_out=hidden, d_in=intermediate). For 32B metadata, hidden=5120 and intermediate=27648, so canonical ffn_up/ffn_gate would be (27648,5120) and ffn_down would be (5120,27648). The GGUFReader raw display [5120,27648] for ffn_up/gate should be treated as storage/reader representation until validated by a model-specific orientation parity test. Orientation must be re-derived and parity-tested per model; raw GGUFReader display alone is not canonical.
      - 32B tensor shapes (shard 1, layer 0): ffn_up=[5120,27648], ffn_gate=[5120,27648], ffn_down=[27648,5120], elements_per_family=141,557,760.
      - 3B intermediate_size conflict **remains UNRESOLVED** — no local 3B GGUF to verify against.
      - 32B memory estimate (linear scaling, unverified at d_out=5120): total selected policy ~11.05 GB across 64 layers, margin ~1.4 GB, runtime working set ~30-40 GB.
      - Recommended next: Phase 31BR — Larger-Model Acquisition Plan / Download Approval (only if explicitly requested) — no validation possible without complete local model and explicit download approval.
    - **Phase 31BR — Larger-Model Acquisition Plan / Download Approval:** Classification `PASS_31BR_ACQUISITION_PLAN_READY`.
      - Planning/approval phase only — no model downloaded, no validation executed, no tensor artifacts generated.
      - Disk: 96 GB free on root NVMe, 42 GB free on VL_usb — sufficient for 1.5B target.
      - Selected acquisition target: Qwen2.5-1.5B-Instruct, Q4_K_M quantization.
      - Recommended destination: `$SDI_MODEL_DIR/qwen2.5-1.5b-official/` (local operator sets `SDI_MODEL_DIR` to a real path, e.g. `/media/matthew-villnave/VL_usb/models`).
      - Estimated disk budget: ~7 GB (1.0-1.4 GB model + 3-5 GB temp validation artifacts).
      - Why 1.5B: cleanest step up from 0.5B; lowest resource risk; 3B has unresolved intermediate_size conflict; 7B/32B too large.
      - Why Q4_K_M: 0.5B baseline used Q2_K + Q4_K_M comparison; Q4_K_M is the comparator reference; single file, no sharding.
      - Staged plan: Stage 0 (approval) → Stage 1 (download, 31BS) → Stage 2 (metadata probe) → Stage 3 (orientation parity micro-probe) → Stage 4 (anchor probe) → Stage 5 (aggregate validation, only if warranted).
      - Stop conditions: disk <5GB free, sharded unexpectedly, checksum mismatch, GGUFReader fails, metadata missing, orientation ambiguous, architecture mismatch, file too large, regression fails.
      - Matt approval phrase template: "I approve Phase 31BS to download Qwen2.5-1.5B-Instruct Q4_K_M into $SDI_MODEL_DIR/qwen2.5-1.5b-official/, max 7 GB budget, huggingface-cli download, metadata-probe only, no tensor validation."
      - Next allowed phase: Phase 31BS — Approved Larger-Model Download / Metadata Verification, only if explicitly requested.
    - **Phase 31BS — Approved Larger-Model Download / Metadata Verification:** Classification `PASS_31BS_1_5B_DOWNLOAD_METADATA_VERIFIED` (on rerun after the earlier in-session `BLOCKED_31BS_SDI_MODEL_DIR_UNSET` blocker; the blocker was a pure environment-preflight miss and was **not** frozen into this file — the 31BS blocker artifacts were overwritten with the successful download/metadata report, per Matt's explicit instruction "Do not keep the blocker docs as the final 31BS artifacts if the rerun succeeds").
      - Approval: explicit from Matt for Qwen2.5-1.5B-Instruct Q4_K_M only, into `$SDI_MODEL_DIR/qwen2.5-1.5b-official/`, max 7 GB budget, metadata-probe only, no tensor validation.
      - Rerun preflight: local operator exported `SDI_MODEL_DIR=/media/matthew-villnave/VL_usb/models` (operator-specific, not committed) in the same process; `df -h` showed 42 GB free at that mount — well over 7 GB budget.
      - Active venv env-only installs (no project source files modified): `numpy==2.4.6`, `huggingface-hub==1.17.0`, `gguf==0.19.0`. The installed `huggingface-cli` printed a deprecation warning and recommended `hf`; the agent used the new `hf` CLI with the same arguments (identical upstream toolchain).
      - Download: `hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir "$SDI_MODEL_DIR/qwen2.5-1.5b-official/"` — exact filename identified via `huggingface_hub.list_repo_files` filtered for `Q4_K_M`/`q4_k_m` (single match, no sharding). Downloaded file: `qwen2.5-1.5b-instruct-q4_k_m.gguf`, **1,117,320,736 bytes (1.04 GiB)**, well under 7 GB budget. GGUF magic `0x47475546` and version 3 verified.
      - Post-download disk: VL_usb 41 GB free (down from 42 GB pre-download), consistent with the 1.04 GiB model file plus HF cache.
      - GGUFReader probe: opened successfully, GGUF v3, 26 KV fields, 339 tensors.
      - Metadata: `general.architecture=qwen2`, `general.name=qwen2.5-1.5b-instruct`, `general.file_type=15` (LlamaFileType.MOSTLY_Q4_K_M), `general.quantization_version=2`, `general.size_label=1.8B`. `qwen2.block_count=28`, `qwen2.embedding_length=1536`, `qwen2.feed_forward_length=8960`, `qwen2.context_length=32768`, `qwen2.attention.head_count=12`, `qwen2.attention.head_count_kv=2` (GQA 6:1), `qwen2.attention.layer_norm_rms_epsilon≈1e-6`, `qwen2.rope.freq_base=1.0e6`. Tokenizer: BPE (`gpt2`), 151,937 tokens, 151,388 merges, 75,969 token types, chat_template present (2509 chars), BOS id 151643, EOS id 151645, padding id 151643.
      - Layer-0 raw GGUFReader shapes (storage-specific, not canonical): `blk.0.ffn_up.weight=[1536, 8960]` Q4_K (13,762,560 elems); `blk.0.ffn_gate.weight=[1536, 8960]` Q4_K (13,762,560 elems); `blk.0.ffn_down.weight=[8960, 1536]` Q6_K (13,762,560 elems).
      - **Orientation caveat (unresolved by 31BS):** raw GGUFReader shapes suggest storage ordering `[d_in, d_out]` for ffn_up/ffn_gate/ffn_down, but this is a hypothesis. Canonical artifact orientation per this file is `(d_out=intermediate=8960, d_in=hidden=1536)` for ffn_up/ffn_gate and `(d_out=hidden=1536, d_in=intermediate=8960)` for ffn_down. **31BS makes no orientation claim**; orientation must be parity-tested in a future phase before any tensor validation.
      - Model file location: `$SDI_MODEL_DIR/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf` (operator-specific value, not committed). The file is **outside the repo** (`git ls-files` errors with "is outside repository"). The model is untracked and outside the working tree.
      - Upheld: no tensor harness validation, no full tensor dequantization, no Q2_K artifacts, no SDIR artifacts, no orientation parity probe, no anchor probe, no aggregate validation, no model files committed, no commit/push/tag without explicit Matt approval.
      - Prior accepted numeric results (0.5B Q2_K and Q4_K_M reference metrics in 31AY / 31BA / 31BM / 31BN / 31BO): **unchanged**. 31BS did not validate any tensor.
      - Artifacts (untracked, prepared for this phase — not committed yet):
        - `docs/PHASE31BS_1_5B_DOWNLOAD_METADATA_VERIFICATION.md` (this document, after overwriting the earlier blocker version)
        - `src/results/PHASE31BS_1_5B_DOWNLOAD_METADATA_VERIFICATION.json`
      - Next allowed phase: **Phase 31BT — Qwen2.5-1.5B Orientation Parity Micro-Probe** (layer 0 only, tiny seed, orientation formula only, no aggregate validation), only if explicitly requested.
      - **Accepted claim:** 31BS successfully downloaded Qwen2.5-1.5B-Instruct Q4_K_M (1,117,320,736 bytes, 1.04 GiB, GGUF v3, single file, outside the repo at `$SDI_MODEL_DIR/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf`, untracked) and verified its metadata: `general.architecture=qwen2`, `general.file_type=15` (MOSTLY_Q4_K_M), `qwen2.block_count=28`, `qwen2.embedding_length=1536`, `qwen2.feed_forward_length=8960`, `qwen2.context_length=32768`, `head_count=12`, `head_count_kv=2` (GQA 6:1), 339 tensors. 31BS made no orientation claim and did not validate any tensor.
      - **Valid as long as:**
        - the downloaded file at `$SDI_MODEL_DIR/...` remains unmodified (any re-download or different quantization invalidates the 31BS metadata snapshot)
        - the file is treated as a Q4_K_M, single-file, 1.5B Qwen2 architecture model
        - orientation remains unresolved by 31BS itself (must be settled by a separate parity probe before any tensor validation, e.g. 31BT)
        - no tensor validation, anchor probe, or aggregate validation is implied or claimed by 31BS
        - no later phase invalidates or supersedes this metadata snapshot
    - **Phase 31BT — Qwen2.5-1.5B Orientation Parity Micro-Probe:** Classification `PASS_31BT_1_5B_ORIENTATION_PARITY_CONFIRMED`.
      - Scope: orientation parity micro-probe only. Layer-0 MLP tensors only (ffn_up, ffn_gate, ffn_down). Tiny deterministic input (seed 42, batch 1). No anchor probe, no aggregate validation, no multi-layer sweep, no Q2_K encoding, no SDIR residual, no generation/inference, no model files committed.
      - Probe config: `np.random.default_rng(42)`, batch=1, hidden=1536, intermediate=8960. Tensors read: `blk.0.ffn_up.weight`, `blk.0.ffn_gate.weight`, `blk.0.ffn_down.weight`. Model file at `$SDI_MODEL_DIR/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf` (operator-specific, not committed).
      - Raw GGUFReader shapes (storage-order-specific, not canonical): ffn_up/ffn_gate `[1536, 8960]` Q4_K; ffn_down `[8960, 1536]` Q6_K. n_elements=13,762,560 each.
      - **Important empirical finding discovered by the probe:** `gguf.dequantize()` returns the tensor in **canonical (d_out, d_in) layout** — ffn_up/ffn_gate as `(intermediate, hidden) = (8960, 1536)` and ffn_down as `(hidden, intermediate) = (1536, 8960)`. The raw GGUFReader shape display is storage-order-specific (likely `[d_in, d_out]`) and the dequantize step performs the orientation-correcting reshape. This is consistent with `CANONICAL_ORIENTATION = "canonical_d_out_d_in"` (Section 13 / `metric_convention_sanity`).
      - Per-tensor candidate results (X shape `[1, 1536]`, H shape `[1, 8960]`):
        - ffn_up canonical A (`Y = X @ W.T`): shape `[1, 8960]`, finite, norm 64.12 ✓
        - ffn_up raw B (`Y = X @ W`): shape-fail (size 8960 ≠ 1536) — expected and recorded ✓
        - ffn_gate canonical A: shape `[1, 8960]`, finite, norm 64.12 ✓
        - ffn_gate raw B: shape-fail — expected and recorded ✓
        - ffn_down canonical D (`Y = H @ W.T`): shape `[1, 1536]`, finite, norm 51.20 ✓
        - ffn_down raw E (`Y = H @ W`): shape-fail (size 1536 ≠ 8960) — expected and recorded ✓
      - Full MLP parity (canonical `X @ W.T` formulation, both Formulation 1 and Formulation 2 are identical by construction): `up=[1,8960]`, `gate=[1,8960]`, `act=[1,8960]`, `out=[1,1536]` ✓ (expected `[batch, hidden]`). `out` finite, norm 21.21. `max_abs_diff = 0.0`, `cosine = 1.0` between the two formulations.
      - Interpretation: the canonical formulation (`Y = X @ W.T` after dequantize) is the correct orientation interpretation for 1.5B layer-0 MLP. The raw GGUFReader shape display ordering is storage-specific; downstream tensor validation must use the canonical (d_out, d_in) layout returned by `dequantize`. 31BT does NOT validate model quality; it only verifies orientation equivalence.
      - Upheld: no Q2_K encoding, no SDIR residual, no anchor probe, no aggregate validation, no multi-layer sweep, no generation/inference, no model files committed, no quality/performance claim.
      - Prior accepted numeric results (0.5B Q2_K and Q4_K_M reference metrics in 31AY / 31BA / 31BM / 31BN / 31BO): **unchanged**. 31BT did not run any tensor validation, anchor probe, or aggregate validation. 31BS metadata is unchanged.
      - Artifacts (untracked, prepared for this phase — not committed yet):
        - `docs/PHASE31BT_1_5B_ORIENTATION_PARITY_MICRO_PROBE.md`
        - `src/phase31bt_1_5b_orientation_parity_micro_probe.py`
        - `src/results/PHASE31BT_1_5B_ORIENTATION_PARITY_MICRO_PROBE.json`
      - Next allowed phase: **Phase 31BU — Qwen2.5-1.5B Corrected Q2_K Anchor Probe**, only if explicitly requested. Must use the downloaded 1.5B Q4_K_M as the W_ref. Must use the corrected Q2_K policy (`corrected_ceil_per_row`, ffn_up + ffn_gate, k=0.5%, alpha=1.0, no ffn_down residual). No aggregate validation, no multi-layer sweep without explicit approval. Orientation is now settled (canonical `X @ W.T` after dequantize) and should not need to be re-derived.
      - **Accepted claim:** 31BT confirms that for Qwen2.5-1.5B layer-0 MLP, the canonical orientation interpretation is `Y = X @ W.T` after dequantize (where W_up/W_gate are `(intermediate, hidden) = (8960, 1536)` and W_down is `(hidden, intermediate) = (1536, 8960)`), and the raw GGUFReader shape display is storage-order-specific (likely `[d_in, d_out]`). Probe results: per-tensor canonical candidates produce shape `[1, 8960]` (ffn_up/ffn_gate) and `[1, 1536]` (ffn_down) with finite values; raw-formulation candidates shape-fail as expected; full MLP output has shape `[1, 1536]`, finite, norm 21.21, with `max_abs_diff = 0.0` and `cosine = 1.0` between the two equivalent formulations. 31BT is an orientation equivalence check, NOT a model quality validation.
      - **Valid as long as:**
        - the downloaded file at `$SDI_MODEL_DIR/...` remains the same 1.5B Q4_K_M file (any re-download or different quantization invalidates this result)
        - the probe inputs remain tiny deterministic (`np.random.default_rng(42)`, batch 1) — not real activations
        - `gguf.dequantize()` continues to return tensors in canonical (d_out, d_in) layout (this is a library-level observation; if the upstream `gguf` package changes, re-verify)
        - the canonical orientation convention (Section 7) remains unchanged
        - the result is interpreted as orientation equivalence only, not model quality
        - no tensor validation, anchor probe, aggregate validation, or generation/inference is implied or claimed by 31BT
        - no later phase invalidates or supersedes this orientation parity result
    - **Phase SOT-01 — SOURCE_OF_TRUTH v2 Current-State Header / Anti-Drift Hardening:** Classification `PASS_SOT01_SOT_V2_CURRENT_STATE_HARDENED`.
      - This is a **documentation/protocol maintenance interlude**, NOT a scientific phase. It does not consume, replace, or invalidate the current scientific next phase.
      - Current scientific next phase (preserved as-is): **Phase 31BU — Qwen2.5-1.5B Corrected Q2_K Anchor Probe, only if explicitly requested.**
      - SOT version / state label bumped to `v2` (post-31BT, SOT-01 hardened). Last verified commit at the time of this phase: `17c2281c` (Phase 31BT).
      - Added **Section 0. Current Working State** — concise (≤40 lines) header for fast agent lookup: SOT version, last verified commit, current selected policy, frozen checkpoint, current model under test, current allowed scientific next phase, current blockers, active forbidden claims summary, current validation scope, current artifact/package status, and "How agents should use this file" guide.
      - Added **Section 0.A. Master Forbidden Claims (Canonical)** — top-level master list of 13 forbidden claims, declared canonical. Per-phase forbidden-claim notes elsewhere in this file are audit context, not the canonical list.
      - Added **"Valid as long as"** clauses to the major current accepted claims: 31BN (corrected Q2_K memory-positive + strongly validated), 31BO (`corrected_q2k_policy_v1` package), 31BS (1.5B download + metadata), 31BT (1.5B orientation parity). No numeric claims were altered.
      - Added **Section 14. Full Phase Workflow Protocol** — the full request → regression → report → approval → next-phase cycle plus additional rules: download approval requires explicit wording, model/artifact files must never be committed, blocked phases should not be committed if trivially fixable in-session unless Matt approves freezing the blocker, stale files must be classified before commit, documentation/protocol maintenance interludes must not silently replace the scientific next phase, numeric claims must not be changed without an explicit numeric phase, per-phase hygiene defaults.
      - Added a "Future SOT v3 split plan" note in Section 0 (for reference only; not done in SOT-01): split into `SOURCE_OF_TRUTH.md`, `PHASE_LOG.md`, `SOURCE_OF_TRUTH.json`, `FORBIDDEN_CLAIMS.md`, `WORKFLOW_PROTOCOL.md`.
      - **No scientific/numeric results changed.** No phase classifications changed. No model files touched. No Q2_K/SDIR artifacts generated. No model validation, anchor probe, aggregate validation, or generation/inference executed.
      - Prior accepted numeric results (0.5B Q2_K and Q4_K_M reference metrics in 31AY / 31BA / 31BM / 31BN / 31BO, 1.5B orientation parity in 31BT): **unchanged**.
      - Artifacts (untracked, prepared for this phase — not committed yet):
        - This file (`SOURCE_OF_TRUTH.md` modified)
      - Next allowed phase (preserved from prior state, unchanged by SOT-01): **Phase 31BU — Qwen2.5-1.5B Corrected Q2_K Anchor Probe**, only if explicitly requested. Documentation/protocol maintenance interludes (future SOT-XX) may be requested at any time without disturbing the current scientific next phase.

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
**Phase 31BU — Qwen2.5-1.5B Corrected Q2_K Anchor Probe, only if explicitly requested.**

Rationale:
- Phase 31BT re-ran successfully. Classification: `PASS_31BT_1_5B_ORIENTATION_PARITY_CONFIRMED`.
- The probe read only layer-0 MLP tensors (`blk.0.ffn_up.weight`, `blk.0.ffn_gate.weight`, `blk.0.ffn_down.weight`) from the downloaded 1.5B Q4_K_M GGUF at `$SDI_MODEL_DIR/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf` (operator-specific, not committed).
- Empirical finding: `gguf.dequantize()` returns the tensor in canonical (d_out, d_in) layout — ffn_up/ffn_gate as `(intermediate, hidden) = (8960, 1536)`, ffn_down as `(hidden, intermediate) = (1536, 8960)`. The raw GGUFReader shape display is storage-order-specific and the dequantize step performs the orientation-correcting reshape. This is consistent with `CANONICAL_ORIENTATION = "canonical_d_out_d_in"` in this file.
- Per-tensor candidate results: ffn_up/ffn_gate canonical `Y = X @ W.T` produces shape `[1, 8960]` (finite, norm 64.12); ffn_down canonical `Y = H @ W.T` produces shape `[1, 1536]` (finite, norm 51.20). The raw formulations `Y = X @ W` and `Y = H @ W` shape-fail (size mismatch), as expected and recorded.
- Full MLP parity: `up=[1,8960]`, `gate=[1,8960]`, `act=[1,8960]`, `out=[1,1536]`. `max_abs_diff = 0.0`, `cosine = 1.0` between the two equivalent formulations. `out` finite, norm 21.21.
- 31BT did not run any Q2_K encoding, SDIR residual, anchor probe, aggregate validation, multi-layer sweep, generation, or inference. No model files were committed. No quality/performance claim was made. Prior accepted 0.5B Q2_K and Q4_K_M reference metrics (31AY / 31BA / 31BM / 31BN / 31BO) are unchanged. 31BS metadata is unchanged.
- Staged plan from 31BR: Stage 4 (anchor probe, 31BU) → Stage 5 (aggregate validation, only if warranted).
- 31BU scope: use the downloaded 1.5B Q4_K_M as the W_ref; use the corrected Q2_K policy (`corrected_ceil_per_row`, ffn_up + ffn_gate, k=0.5%, alpha=1.0, no ffn_down residual); orientation is settled (canonical `X @ W.T` after dequantize); layer 0 only or a small fixed set of layers; no aggregate validation, no multi-layer sweep without explicit approval.
- Stop conditions: orientation regression, regression failure, model file tracking, scope creep, or any Q2_K/SDIR/anchor/aggregate validation triggered without explicit approval.
- The 31BT artifacts (`docs/PHASE31BT_1_5B_ORIENTATION_PARITY_MICRO_PROBE.md`, `src/phase31bt_1_5b_orientation_parity_micro_probe.py`, `src/results/PHASE31BT_1_5B_ORIENTATION_PARITY_MICRO_PROBE.json`) are untracked. They will be committed only if Matt explicitly approves.

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

## 13. Commit Approval Process

All future SDI phases must follow this required workflow:

### Per-Phase Workflow

1. Run the requested phase work.
2. Update docs/results/SOURCE_OF_TRUTH as needed.
3. Run required regression (`python3 -m tests.run_source_of_truth_regression`).
4. Produce a **PRE-COMMIT REPORT**.
5. **Stop and wait for approval.**

Do NOT commit, push, or proceed to the next phase without explicit approval.

### Pre-Commit Report Template

When a phase is complete, report:

- **current branch** — e.g., `master`
- **current HEAD** — full SHA
- **`git status --short`** — working tree state
- **`git diff --stat`** — staged + unstaged changes
- **files changed** — list of modified and new files
- **regression result** — pass/fail, error_count, fallback_count
- **classification** — one of the standard phase classification strings
- **whether numeric results changed** — yes/no, and what changed if yes
- **SOURCE_OF_TRUTH sections changed** — which sections were updated
- **proposed commit message** — exact proposed message
- **large/private/generated files** — list any files that are .gguf, .bin, .npy, .npz, .venv, large logs, or contain private paths
- **untracked/stale/rescued files** — classification of all untracked files

### Approval Rules

- **No normal phase commit without approval** from Matt.
- **No push without approval.**
- **No force-push unless explicitly approved.**
- **Safety snapshots:** only on a safety branch (never directly on master).
- **Dirty files:** classify before touching. Do not lose work.
- **Regression fails:** do not commit — except a safety snapshot on a safety branch if requested.
- **SOURCE_OF_TRUTH and code disagree:** stop and report the conflict before continuing.

### Commit Approval Keywords

- `"commit"` — stage and commit to local branch only (no push)
- `"commit and push"` — stage, commit, and push to origin
- `"safety snapshot"` — create an unsigned commit on a safety branch (never master)
- `"abort"` — discard all changes, return to clean state at current HEAD

## 14. Full Phase Workflow Protocol

This section is the **full canonical workflow** for any phase (scientific, maintenance, or interlude). It expands Section 13 with the full request → report → approval → next-phase cycle plus additional rules.

### 14.1 Full Phase Workflow (mandatory)

For every phase, in order:

1. **Request phase** — Matt explicitly requests a specific phase by name and scope.
2. **Read SOURCE_OF_TRUTH** — agents read this file, beginning with **Section 0** (current working state).
3. **Confirm allowed next phase** — agents confirm the requested phase matches the current allowed next phase. If not, **stop and report the conflict**. If the request is a documentation/protocol maintenance interlude (e.g. SOT-01), the *current scientific next phase* must be preserved as-is and not silently replaced.
4. **Preflight regression** — run `python3 -m tests.run_source_of_truth_regression`. Required: `PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN`, `error_count=0`, `fallback_count=0`. If this fails, stop and report.
5. **Execute only approved scope** — execute *exactly* the scope Matt approved. Do not run additional validations, do not download additional models, do not generate additional artifacts. Scope creep is a stop condition.
6. **Update artifacts / SOURCE_OF_TRUTH** — write the phase's docs/JSON/MD artifacts. Update Sections 3 (audit trail) and 9 (current allowed next phase) as needed. Do not modify prior accepted numeric claims.
7. **Run regression after edits** — re-run `python3 -m tests.run_source_of_truth_regression`. Required: same as step 4.
8. **PRE-COMMIT REPORT** — produce the report. Stop. Do not commit.
9. **Wait for Matt approval** — Matt must explicitly approve the commit. Approval keywords: `"commit"`, `"commit and push"`, `"safety snapshot"`, `"abort"` (see Section 13).
10. **Commit / push only approved files** — stage and commit only the files Matt explicitly approved. Do not stage stale, rescued, or generated side-effect files.
11. **Post-push report** — report `old HEAD`, `new HEAD`, push status, working tree status, files committed, stale/model-file non-commit confirmations, final classification, and current allowed next phase.
12. **Do not start next phase** — without explicit Matt request. Even if the new "current allowed next phase" is set in Section 9, the agent does NOT proceed to it without a new request.

### 14.2 Additional Rules

- **Download approval requires explicit wording.** Downloading any new model file requires Matt's explicit approval using the documented approval phrase template (e.g. 31BR/31BS template). The agent must not download anything, even an apparently-related model, on its own initiative.
- **Model and artifact files must never be committed.** `.gguf`, `.bin`, `.npy`, `.npz`, large logs, HF cache, and any model-derived raw tensors must not enter the repo. Even if a phase generated them, they stay outside the repo and are referenced via env-var placeholders.
- **Blocked phases should not be committed if trivially fixable in-session.** If a phase is blocked by an environment-only issue (e.g. unset env var, missing dependency) that the agent can fix and re-run in the same session, the agent should re-run the phase after the fix and not freeze a `BLOCKED_*` report as a final deliverable, unless Matt explicitly approves freezing the blocker. Blockers that are *content* blockers (e.g. cannot run a calculation because of missing data) may be frozen only with explicit Matt approval.
- **Stale files must be classified before commit.** Before staging, every untracked file must be classified as: `needed for this phase`, `stale/rescue` (prior phase, not needed now), `generated/do-not-commit` (side-effect of this phase, do not commit), or `safe-to-commit later after approval`. Do not delete silently. Do not commit without explicit approval of the staged set.
- **Documentation/protocol maintenance interludes must not silently replace the scientific next phase.** A maintenance interlude (e.g. SOT-01) updates this file, the workflow, or the master forbidden-claims list. It must preserve the current scientific next phase (Section 9) as-is. If Section 0 and Section 9 disagree, Section 9 wins until reconciled. An interlude must not advance the scientific program; it only hardens documentation/protocol.
- **Numeric claims must not be changed without an explicit numeric phase.** A maintenance interlude cannot alter accepted numeric results. A new scientific phase (with Matt approval) is required to update or invalidate a numeric claim.

### 14.3 Per-Phase Hygiene Defaults (recommended)

- Record the model path under the env-var form `$SDI_MODEL_DIR/...` (or equivalent). If a runner needs to record the operator-specific path it actually opened, use a clearly non-canonical field name such as `model_path_observed_redacted` and store the env-var form there.
- Use a deterministic seed for any probe (e.g. `np.random.default_rng(42)`).
- Keep probe inputs tiny (e.g. batch 1, single seed) for orientation/probe phases; do not use real activations without explicit approval.
- Prefer reading only the tensors the phase needs (e.g. layer 0 only for orientation probes), not full-model sweeps, unless explicitly approved.
