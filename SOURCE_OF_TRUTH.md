# SOURCE_OF_TRUTH.md

## 0. Current Working State

- **SOT version / state label:** v2 (post-31BT, SOT-01 hardened)
- **Last verified commit:** `c93bcdf96c8422f91f01fbc8c4e522710922fe61` (Phase 31CB: clean stale phase provenance artifacts)
- **31CC planning result and classification:** **Phase 31CC passed with `PASS_31CC_REAL_ACTIVATION_CAPTURE_PLAN_SELECTED`** (revised after a pre-commit wording/contradiction review). 31CC produced a 5-strategy capture table documented in `docs/PHASE31CC_REAL_ACTIVATION_CAPTURE_PLANNING.md` (45.6 KB) and `src/results/PHASE31CC_REAL_ACTIVATION_CAPTURE_PLANNING.json` (38.9 KB). The 5 strategies: **Option A** (HuggingFace `transformers` forward-hook capture, HF-derived real-activation **proxy**), **Option B** (llama.cpp instrumentation capture, **the only path that can claim exact Q4_K_M GGUF runtime activation capture**), **Option C** (stop and re-plan via 31CC-R), **Option D** (minimal local forward-proxy, rejected), **Option E** (runtime / loader format planning, out of scope). 31CC does NOT auto-select between A and B; the 31CD-entry decision point is a hard gate that requires Matt to explicitly choose A, B, or C before 31CD can begin. The default 31CD plan (applies to both A and B) defines: default prompt `"The capital of France is"` (6 BPE tokens, PII-free, public-domain, sourced from 31I prompt index 1); default layers `{0, 14, 27}` (reducible to `{0}` for first iteration); default token position last prefill token; batch=1, shape `[1, 1536]`; metrics `cos_low, cos_sub, delta_cos, MAE_low, MAE_sub, MAE_delta, finite, severe, memory margin`; activation artifact policy (raw X in `tempfile.mkdtemp`, not committed by default; result JSON contains summary metrics + SHA256 + metadata only); prompt redaction (>64 chars); 17 stop conditions; 17 classifications; **4 allowed claims** (3 universal + 1 Option-B-only; the previously-listed "real-activation replay reproduces synthetic-Gaussian micro-probe behavior" allowed claim is REMOVED in the latest revision to close a contradiction with the universal "no real activations behave like synthetic Gaussian" forbidden claim); **20 forbidden claims** (12 universal + 4 Option-A-only + 2 Option-B-only + 2 explicit-claim-language reinforcements: (1) Do not claim real activations behave like synthetic Gaussian. (2) Do not claim activation-distribution equivalence. (3) Do not claim transfer beyond the selected prompt/layer/token scope.). **NO W_ref source pivot in Option A** (an earlier 31CC draft proposed pivoting W_ref to FP16/BF16 safetensors for Option A; this is **RETRACTED** — the W_ref for replay stays as the local 1.5B Q4_K_M GGUF dequantized, matching 31BU/31BV/31BX/31BZ exactly, for BOTH Option A and Option B). Option A's only difference from 31BU/31BV/31BX/31BZ is the X distribution: real prompt-derived instead of synthetic Gaussian. Option A is labeled "HF-derived real-activation proxy" because the X is captured from the HF safetensors (a different model binary from the local Q4_K_M GGUF). Option B is the only path that can claim exact Q4_K_M GGUF runtime activation capture because it captures X from the local Q4_K_M GGUF via llama.cpp instrumentation. 31CC did not run activation capture, did not run generation, did not run inference beyond metadata inspection, did not modify llama.cpp, did not create hooks, did not download model files, did not generate Q2_K/SDIR artifacts, did not make quality/performance/runtime/behavior claims. 31CC did not change the 0.5B or 1.5B frozen evidence tiers; did not change the `corrected_q2k_policy_v1` package; did not create a new tag; did not introduce any W_ref source pivot.
- **Current selected policy:** `corrected_q2k_policy_v1` (corrected_ceil_per_row Q2_K, ffn_up+ffn_gate SDIR k=0.5%, alpha=1.0, no ffn_down residual)
- **Frozen checkpoint:** `phase31bn-corrected-q2k-full-aggregate-checkpoint` → `0304590c92d43fdf48d3d28998255d39c9a20c07`
- **Current model under test:** Qwen2.5-0.5B-Instruct (validated for the frozen checkpoint); Qwen2.5-1.5B-Instruct Q4_K_M (downloaded, orientation parity confirmed via 31BT, layer-0 anchor probe PASSED via 31BU; small multi-layer probe (L0/L14/L27) PASSED via 31BV; broader layer probe planning completed via 31BW — Option A (7-layer stratified: 0, 4, 8, 12, 16, 20, 27 × seeds 0, 9 = 14 pairs) selected; stratified probe PASSED via 31BX: 14/14 pairs memory-positive, cosine-improved, MAE-improved, 0 severe, all finite, per-layer margin +3,380,342 to +3,380,374 bytes; L0 result exactly reproduces 31BU and 31BV; aggregate planning / stop-go decision COMPLETED via 31BY — Option A (28-layer × 2-seed aggregate, 56 pairs) selected as next executable phase; full-layer two-seed aggregate PASSED via 31BZ: 56/56 pairs memory-positive, cosine-improved, MAE-improved, 0 severe, all finite, mean delta_cos=+0.004758, mean MAE improvement=−0.012865, min per-layer margin +3,380,312 bytes, max +3,380,376 bytes, worst pair L26-S9 (Δcos=+0.000390, MAE Δ=−0.029222), total wall clock 481.5 sec (8.0 min); L0 result bit-identical to 31BU/31BV/31BX; L4 result bit-identical to 31BX; L14 result bit-identical to 31BV; 28 of 28 layers PASS independently; aggregate freeze + package update COMPLETED via 31CA — `corrected_q2k_policy_v1` package updated with 1.5B (31BZ/31CA) as a second evidence tier; policy parameters UNCHANGED; policy version NOT bumped; README patched conservatively; proposed tag `phase31ca-1_5b-corrected-q2k-aggregate-checkpoint` (NOT created in 31CA; the tag `phase31ca-1_5b-corrected-q2k-aggregate-checkpoint` was created after the 31CA package commit and before 31CB; it points to `a433875aa20431da8749e42c3449494434fa9f23`); **stale file provenance cleanup COMPLETED via 31CB — 5 stale 31BH-R2 / 31BJ files moved to `rescue/stale_phase31bh_31bj/` with provenance README; no file committed, no file deleted, no file promoted to canonical evidence, no scientific/numeric claim changed**)
- **Current allowed scientific next phase:** **Phase 31CD — Qwen2.5-1.5B Real-Activation Capture Micro-Probe, only if explicitly requested.** 31CD does NOT begin until Matt explicitly chooses between **Option A** (HF-derived real-activation proxy, HF safetensors forward-hook capture, replay through the local Q4_K_M GGUF W_ref, requires HF safetensors download approval), **Option B** (llama.cpp/GGUF activation capture instrumentation, the only path that can claim exact Q4_K_M GGUF runtime activation capture, requires llama.cpp modification approval), or **Option C** (stop and re-plan via 31CC-R) at 31CD entry. **NO W_ref source pivot in Option A** — both Option A and Option B use the same W_ref (local 1.5B Q4_K_M GGUF dequantized), the same W_low (corrected_ceil_per_row Q2_K from the Q4_K_M GGUF), and the same SDIR (ffn_up+ffn_gate SDIR k=0.5%, alpha=1.0, no ffn_down residual) for replay — all matching 31BU/31BV/31BX/31BZ exactly. The only thing that differs across the two options is the **X source**: Option A captures X from the HF safetensors (an HF-derived proxy, NOT exact Q4_K_M GGUF runtime activation); Option B captures X from the local Q4_K_M GGUF via llama.cpp instrumentation (EXACT Q4_K_M GGUF runtime activation). The W_ref source pivot idea from an earlier 31CC draft is **RETRACTED**. 31CC did NOT run activation capture, did NOT run generation/inference/sampling, did NOT modify llama.cpp, did NOT create hooks, did NOT download model files, did NOT generate Q2_K/SDIR artifacts, did NOT make quality/performance/runtime/behavior claims, did NOT commit/push/tag without explicit Matt approval. 31CC is a planning-only phase; the design for 31CD is in `docs/PHASE31CC_REAL_ACTIVATION_CAPTURE_PLANNING.md` and `src/results/PHASE31CC_REAL_ACTIVATION_CAPTURE_PLANNING.json`. If Matt chooses Option C at 31CD entry, the next permissible phase is **Phase 31CC-R — Real-Activation Capture Planning Repair, only if explicitly requested** (re-plan).
- **Current blockers:** none (provenance debt resolved in 31CB: 5 stale 31BH-R2 / 31BJ files moved to `rescue/stale_phase31bh_31bj/` with provenance README; not committed, not deleted, not promoted to canonical evidence).
- **Active forbidden claims (summary; full canonical master list in Section 0.A):** no model quality/behavior recovery claim, no speedup, no full-model runtime memory savings, no llama.cpp integration, no production readiness, no inference/generation, no larger-model validation claim unless explicitly proven, no runtime-ready output-residual claim, no claim beyond standalone tensor harness unless proven, no orientation claim for a larger model unless parity-tested, no commit/push/tag/download without explicit Matt approval where applicable.
- **Current validation scope:** standalone tensor harness. Qwen2.5-0.5B for accepted numeric metrics (31BN freeze). Qwen2.5-1.5B layer-0 anchor probe passed (31BU); small multi-layer probe (L0/L14/L27) passed (31BV); broader layer probe planning completed (31BW, Option A selected); stratified probe (7 layers × 2 seeds = 14 pairs) passed (31BX); aggregate planning / stop-go decision completed (31BY, Option A: 28-layer × 2-seed aggregate selected as next executable phase); full-layer two-seed aggregate (28 layers × 2 seeds = 56 pairs) passed (31BZ); aggregate freeze + package update completed (31CA: `corrected_q2k_policy_v1` package now has two frozen evidence tiers — 0.5B 31BN and 1.5B 31BZ/31CA — policy parameters UNCHANGED, policy version NOT bumped); stale file provenance cleanup completed (31CB: 5 stale 31BH-R2 / 31BJ files moved to `rescue/stale_phase31bh_31bj/`, no claim changes, no model files touched); real-activation capture planning completed (31CC, revised: 5-strategy table; Option A = HF-derived real-activation proxy with HF safetensors forward-hook capture, NO W_ref source pivot (W_ref for replay stays as Q4_K_M GGUF dequantized); Option B = llama.cpp instrumentation, the only path that can claim exact Q4_K_M GGUF runtime activation capture; Option C = stop and re-plan; Options D and E rejected/out-of-scope; 31CD-entry decision point is a hard gate requiring Matt's explicit choice of A, B, or C before 31CD can begin; exact 31CD scope defined; 17 stop conditions, 17 classifications, **4 allowed claims, 20 forbidden claims** (4 allowed = 3 universal + 1 Option-B-only, the previously-listed "real-activation replay reproduces synthetic-Gaussian micro-probe behavior" allowed claim is REMOVED to close a contradiction with the universal "no real activations behave like synthetic Gaussian" forbidden claim; 20 forbidden = 12 universal + 4 Option-A-only + 2 Option-B-only + 2 explicit-claim-language reinforcements: (1) Do not claim real activations behave like synthetic Gaussian. (2) Do not claim activation-distribution equivalence. (3) Do not claim transfer beyond the selected prompt/layer/token scope.); the earlier 31CC draft's W_ref source pivot idea is RETRACTED; no activation capture executed, no model inference beyond metadata inspection, no llama.cpp modification, no hook creation, no model file download, no Q2_K/SDIR artifacts, no quality/performance/runtime/behavior claims); no generation, no inference, no llama.cpp runtime integration, no model files committed, no Q2_K/SDIR blobs committed.
- **Current artifact/package status:** `corrected_q2k_policy_v1` package (frozen via 31BO, updated in 31CA with 1.5B evidence tier); 0.5B 31BN aggregate freeze; 1.5B 31BZ/31CA aggregate freeze. The tag `phase31ca-1_5b-corrected-q2k-aggregate-checkpoint` was created after the 31CA package commit and before 31CB; it points to `a433875aa20431da8749e42c3449494434fa9f23`. 1.5B 31BS download+metadata; 1.5B 31BT orientation parity result; 5 stale 31BH-R2 / 31BJ files in `rescue/stale_phase31bh_31bj/` (rescue, not canonical); 31CC planning artifacts (`docs/PHASE31CC_REAL_ACTIVATION_CAPTURE_PLANNING.md`, `src/results/PHASE31CC_REAL_ACTIVATION_CAPTURE_PLANNING.json`, README patch) — planning only, no Q2_K/SDIR artifacts, no model files, no raw activation arrays, no W_ref source pivot.

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
    - **Phase README-01 — Align public README with SOURCE_OF_TRUTH:** Classification `PASS_README01_README_ALIGNED_WITH_SOT`.
      - This is a **README documentation maintenance interlude**, NOT a scientific phase. It does not consume, replace, or invalidate the current scientific next phase.
      - Current scientific next phase (preserved as-is): **Phase 31BU — Qwen2.5-1.5B Corrected Q2_K Anchor Probe, only if explicitly requested.**
      - Drift found and corrected: the previous `README.md` claimed the repo was in "Phase 31A complete / Phase 31B next" state, called itself an "early research scaffold", "design-only repo", and said "no code has been written yet". All of this contradicted the accepted state in `SOURCE_OF_TRUTH.md` (which records many later phases, real code, real results, the `corrected_q2k_policy_v1` package, regression tests, 1.5B download + metadata, and 1.5B orientation parity). The README has been rewritten as a public orientation page that defers current truth to `SOURCE_OF_TRUTH.md` Section 0 and Section 9.
      - New README structure: high-level orientation page, "Status / Drift Guard" section declaring `SOURCE_OF_TRUTH.md` wins on any conflict, "Current State" section summarising the post-31BT / post-SOT-01 state (with explicit pointers into `SOURCE_OF_TRUTH.md` for full details), "What this repo contains" section (phase docs, result JSON/MD reports, source scripts/harness code, policy package, regression tests, claim-boundary docs), "What this repo does not contain / does not claim" section (no model weights, no production runtime, no speedup, no quality/behavior recovery, no inference/generation validation, no full llama.cpp integration, no broad larger-model claim, no runtime-ready output-residual claim, no claim that 31AY/31BA are current canonical), background memory-math table, high-level repository layout, and a "How to verify the current state" section.
      - Added a lightweight **README drift guard** to `tests/run_source_of_truth_regression.py` (`_run_readme_drift_guard`) — a stale-phrase guard that fails the regression if `README.md` contains any of: `Phase 31A` (with negative lookahead so 31AT/31AY/31AG/31AJ/... don't false-positive), `Phase 31B` (same caveat), `design-only repo`, `no code has been written yet`, `plans and claim boundaries only`. The guard is wired into `all_passed` and surfaced in the regression output as `readme_drift_guard.passed`. This is a narrow, high-confidence drift guard — not a general "is the README good?" check.
      - **No scientific/numeric results changed.** No phase classifications changed. No model files touched. No Q2_K/SDIR artifacts generated. No model validation, anchor probe, aggregate validation, or generation/inference executed.
      - Prior accepted numeric results (0.5B Q2_K and Q4_K_M reference metrics in 31AY / 31BA / 31BM / 31BN / 31BO, 1.5B orientation parity in 31BT): **unchanged**.
      - Artifacts (untracked, prepared for this phase — not committed yet):
        - `README.md` (rewritten as a public orientation page)
        - `tests/run_source_of_truth_regression.py` (added `_run_readme_drift_guard` and wired it into `all_passed`)
        - This file (`SOURCE_OF_TRUTH.md` modified)
      - Next allowed phase (preserved from prior state, unchanged by README-01): **Phase 31BU — Qwen2.5-1.5B Corrected Q2_K Anchor Probe**, only if explicitly requested. Documentation/protocol maintenance interludes (future README-XX / SOT-XX) may be requested at any time without disturbing the current scientific next phase.
    - **Phase 31BU — Qwen2.5-1.5B Corrected Q2_K Anchor Probe:** Classification `PASS_31BU_1_5B_Q2K_ANCHOR_PROBE_CLEAN`.
      - Scope: anchor probe only — layer 0, 3 seeds (0, 9, 13) per Route A.
      - Model / W_ref: Qwen2.5-1.5B-Instruct Q4_K_M, dequantized (NOT FP16). W_ref source: downloaded 1.5B Q4_K_M GGUF (`$SDI_MODEL_DIR/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf`, 1,117,320,736 bytes / 1.04 GiB).
      - Tensors read (layer 0): `blk.0.ffn_up.weight` (Q4_K, raw [1536,8960], dequant [8960,1536]), `blk.0.ffn_gate.weight` (Q4_K, same), `blk.0.ffn_down.weight` (Q6_K, raw [8960,1536], dequant [1536,8960]). 31BT-confirmed canonical orientation `Y = X @ W.T` after dequantize.
      - Policy: `corrected_q2k_policy_v1` (corrected_ceil_per_row Q2_K, ffn_up+ffn_gate SDIR k=0.5%, alpha=1.0, ffn_down W_low only — no ffn_down residual).
      - Anchor metrics: L0-S0 dc=+0.007008, L0-S9 dc=+0.001090, L0-S13 dc=+0.004319; MAE improves on all 3 (Δ = −0.0046, −0.0037, −0.0058); 0 severe regressions; all finite.
      - Memory: 3/3 memory-positive; per-layer margin +3,380,374 bytes (~3.22 MB / ~16.4% of 3 × Q4_budget_family=20,643,840); Q2_K encode 4,515,840 bytes/family (corrected_ceil_per_row: 8960 × ⌈1536/256⌉ × 84); SDIR ~1,127,786 bytes/family @ k=0.5% (computed at runtime, deterministic per seed).
      - Per-pair Q2_K byte sanity: ffn_up / ffn_gate = 4,515,840 (corrected ceil, same as 0.5B after scaling); ffn_down = 4,515,840 (same, shape (1536,8960)).
      - Memory accounting observation (NOT a claim): the 1.5B layer margin (+3.22 MB) is larger than the 0.5B layer margin (+0.66 MB at the same k=0.5%, per 31BK) because Q4_budget scales with `d_out × d_in` faster than Q2_K W_low bytes scale with `d_in × ceil(d_in/QK_K)`. This is a byte-accounting observation, not a quality / performance / generalization claim.
      - Forbidden claims preserved: no quality / behavior / speedup / runtime / llama.cpp / inference / production / larger-model-validation / FP16-recovery claim; no aggregate validation; no multi-layer sweep beyond layer 0; no commit/push/tag without explicit Matt approval.
      - Runner: `src/phase31bu_1_5b_corrected_q2k_anchor_probe.py` (env-vars-only: `SDI_MODEL_DIR`, `SDI_LLAMA_CPP_LIB` / `SDI_LLAMA_CPP_BUILD` / `SDI_LLAMA_CPP_ROOT`; no private paths; lint-clean).
      - Result JSON: `src/results/PHASE31BU_1_5B_CORRECTED_Q2K_ANCHOR_PROBE.json` (small, ~7.4 KB; model path redacted to `$SDI_MODEL_DIR/...`).
      - Doc: `docs/PHASE31BU_1_5B_CORRECTED_Q2K_ANCHOR_PROBE.md`.
      - Pre-flight regression: `python3 -m tests.run_source_of_truth_regression` → `PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN`, `error_count=0`, `fallback_count=0`, `readme_drift_guard.passed=true` (run before AND after SOT/doc/result edits; must remain clean for commit approval).
      - Limitations: anchor probe only (3 pairs, layer 0); no FP16 W_ref; no 28-layer or 0.5B comparison in this phase; no llama.cpp runtime integration; no generation/inference; no claim that 1.5B behaves like 0.5B.
      - Pre-existing untracked 31BH-R2 / 31BJ files (`docs/PHASE31BH_R2_Q2K_ANCHOR_REPRO_FIX.md`, `src/phase31bh_q2k_clean_reproduction.py`, `src/results/PHASE31BH_Q2K_RUNTIME_QUANT_PATH.json`, `src/results/PHASE31BH_R2_Q2K_ANCHOR_REPRO_FIX.json`, `src/results/PHASE31BJ_ANCHOR_ONLY.json`): untouched per explicit instruction; not in this commit; will be handled in a dedicated stale-file cleanup / provenance phase.
      - Next allowed phase: **Phase 31BV — Qwen2.5-1.5B Corrected Q2_K Small Multi-Layer Probe**, only if explicitly requested. (Not entered — explicit request required.)
      - **Accepted claim:** 31BU proves the corrected Q2_K policy (`corrected_q2k_policy_v1`, corrected_ceil_per_row, ffn_up + ffn_gate SDIR k=0.5%, alpha=1.0, no ffn_down residual) is **memory-positive and quality-improving on Qwen2.5-1.5B layer 0** in a standalone tensor harness (3/3 pairs memory-positive, cosine-improved, MAE-improved, 0 severe, all finite, mean delta_cos=+0.00414, mean MAE improvement=−0.00471, per-layer margin +3,380,374 bytes). Phrased relative to Q4_K_M W_ref (NOT FP16).
      - **Valid as long as:**
        - the downloaded 1.5B Q4_K_M file at `$SDI_MODEL_DIR/...` remains unmodified
        - the probe inputs remain tiny deterministic (`np.random.default_rng(seed)`, batch 1) — not real activations
        - the policy parameters match `corrected_q2k_policy_v1` (Q2_K mode, residual families, k, alpha, ffn_down residual)
        - the result is interpreted as anchor probe (3 pairs, layer 0), NOT aggregate validation, NOT a larger-model generalization, NOT a speed / runtime claim
        - the canonical orientation convention (Section 7) remains unchanged
        - no later phase invalidates or supersedes this anchor result
        - `gguf.dequantize()` continues to return tensors in canonical (d_out, d_in) layout (re-verify if upstream `gguf` package changes)

    - **Phase 31BV — Qwen2.5-1.5B Corrected Q2_K Small Multi-Layer Probe:** Classification `PASS_31BV_1_5B_Q2K_SMALL_MULTILAYER_CLEAN`.
      - Scope: small fixed multi-layer probe only — Route A: layers [0, 14, 27] × seeds [0, 9] = 6 anchor pairs.
      - Model / W_ref: Qwen2.5-1.5B-Instruct Q4_K_M, dequantized (NOT FP16). W_ref source: downloaded 1.5B Q4_K_M GGUF (`$SDI_MODEL_DIR/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf`, 1,117,320,736 bytes / 1.04 GiB).
      - Tensors read (per layer) — **selected source-GGUF tensor types (per family per layer):**
        - ffn_up: **Q4_K** for layers 0, 14, 27 (raw [1536, 8960], dequant [8960, 1536])
        - ffn_gate: **Q4_K** for layers 0, 14, 27 (raw [1536, 8960], dequant [8960, 1536])
        - ffn_down: **Q6_K** for layers 0 and 27; **Q4_K** for layer 14 (raw [8960, 1536], dequant [1536, 8960])
        - Shapes consistent across L0, L14, L27 (all 13,762,560 elements per family).
      - 31BV observed **mixed source-GGUF quant types for ffn_down** in the selected layers: L0/L27 use Q6_K, while L14 uses Q4_K. This is a source-GGUF characteristic. The corrected Q2_K memory accounting is unaffected because W_ref is dequantized to float32 before corrected Q2_K encoding, and the selected corrected_ceil_per_row Q2_K byte count is shape-dependent (constant across quant types for the same shape).
      - Policy: `corrected_q2k_policy_v1` (corrected_ceil_per_row Q2_K, ffn_up+ffn_gate SDIR k=0.5%, alpha=1.0, ffn_down W_low only — no ffn_down residual).
      - Per-pair metrics:
        - L0-S0: dc=+0.007008, MAE_delta=−0.004610
        - L0-S9: dc=+0.001090, MAE_delta=−0.003686 (worst pair; exactly reproduces 31BU)
        - L14-S0: dc=+0.003992, MAE_delta=−0.006208
        - L14-S9: dc=+0.001325, MAE_delta=−0.002346
        - L27-S0: dc=+0.002961, MAE_delta=−0.005428
        - L27-S9: dc=+0.006858, MAE_delta=−0.013261 (best MAE improvement)
      - Per-layer summary:
        - L0:  mean_dc=+0.004049, mean_MAE_imp=−0.004148, min_margin=+3,380,374
        - L14: mean_dc=+0.002659, mean_MAE_imp=−0.004277, min_margin=+3,380,376
        - L27: mean_dc=+0.004909, mean_MAE_imp=−0.009345, min_margin=+3,380,350
      - Aggregate (within 6-pair scope): n=6, n_mem_pos=6, n_cos_pos=6, n_mae_imp=6, n_severe=0, n_finite=6. mean_dc=+0.003872, median_dc=+0.003476, mean_MAE_imp=−0.005923, min_per_layer_margin=+3,380,350.
      - Memory: 6/6 memory-positive; per-layer margin consistent across the 3 selected layers (variance 26 bytes ≈ 0.0008% of margin). Q2_K encode 4,515,840 bytes/family (corrected_ceil_per_row); SDIR ~1,127,786 bytes/family @ k=0.5% (computed at runtime, deterministic per seed).
      - L0 result **exactly reproduces** 31BU's L0-S0 (dc=+0.007008) and L0-S9 (dc=+0.001090) — confirms cross-runner reproducibility.
      - Forbidden claims preserved: no quality / behavior / speedup / runtime / llama.cpp / inference / production / aggregate / full-28-layer / FP16-recovery / "1.5B behaves like 0.5B" claim; no commit/push/tag without explicit Matt approval.
      - Runner: `src/phase31bv_1_5b_corrected_q2k_small_multilayer_probe.py` (env-vars-only, no private paths, lint-clean).
      - Result JSON: `src/results/PHASE31BV_1_5B_CORRECTED_Q2K_SMALL_MULTILAYER_PROBE.json` (~13.8 KB; model path redacted to `$SDI_MODEL_DIR/...`).
      - Doc: `docs/PHASE31BV_1_5B_CORRECTED_Q2K_SMALL_MULTILAYER_PROBE.md`.
      - Pre-flight regression: `python3 -m tests.run_source_of_truth_regression` → `PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN`, `error_count=0`, `fallback_count=0`, `readme_drift_guard.passed=true` (run before AND after SOT/doc/result edits; must remain clean for commit approval).
      - Limitations: small multi-layer probe only (6 pairs, 3 layers: 0, 14, 27); no FP16 W_ref; no 0.5B comparison in this phase; no llama.cpp runtime integration; no generation/inference; no claim that 1.5B behaves like 0.5B or that the result generalizes to the 25 untested layers.
      - Pre-existing untracked 31BH-R2 / 31BJ files: untouched per explicit instruction; not in this commit; will be handled in a dedicated stale-file cleanup / provenance phase.
      - Hygiene patch applied before commit (per Matt's pre-commit review): `per_layer_summary[layer]['tensors']` is now fully populated with real tensor metadata (name, raw_gguf_shape, dequant_shape, tensor_type, n_elements) for ffn_up / ffn_gate / ffn_down on all 3 selected layers — no null fields in the JSON. The original draft had a cosmetic issue where the per-layer summary referenced a not-yet-populated dict; the runner was patched to load the per-layer tensor diag BEFORE building the per-layer summary, and the result JSON was regenerated. All metrics are unchanged (classification `PASS_31BV_1_5B_Q2K_SMALL_MULTILAYER_CLEAN` preserved).
      - Next allowed phase: **Phase 31BW — Qwen2.5-1.5B Corrected Q2_K Broader Layer Probe Planning**, only if explicitly requested. (Not entered — explicit request required.)
      - **Accepted claim:** 31BV proves the corrected Q2_K policy (`corrected_q2k_policy_v1`, corrected_ceil_per_row, ffn_up + ffn_gate SDIR k=0.5%, alpha=1.0, no ffn_down residual) is **memory-positive and quality-improving on Qwen2.5-1.5B across the small fixed multi-layer set [0, 14, 27] × seeds [0, 9]** in a standalone tensor harness (6/6 pairs memory-positive, cosine-improved, MAE-improved, 0 severe, all finite, mean delta_cos=+0.003872, mean MAE improvement=−0.005923, min per-layer margin +3,380,350 bytes). Per-layer margins are consistent across the 3 selected layers (variance 26 bytes). L0 result exactly reproduces 31BU. Phrased relative to Q4_K_M W_ref (NOT FP16).
      - **Valid as long as:**
        - the downloaded 1.5B Q4_K_M file at `$SDI_MODEL_DIR/...` remains unmodified
        - the probe inputs remain tiny deterministic (`np.random.default_rng(seed)`, batch 1) — not real activations
        - the policy parameters match `corrected_q2k_policy_v1` (Q2_K mode, residual families, k, alpha, ffn_down residual)
        - the result is interpreted as small multi-layer probe (6 pairs, 3 layers), NOT aggregate validation, NOT a full 28-layer generalization, NOT a 0.5B comparison, NOT a larger-model claim
        - the canonical orientation convention (Section 7) remains unchanged
        - no later phase invalidates or supersedes this anchor result
        - `gguf.dequantize()` continues to return tensors in canonical (d_out, d_in) layout (re-verify if upstream `gguf` package changes)

    - **Phase 31BW — Qwen2.5-1.5B Corrected Q2_K Broader Layer Probe Planning:** Classification `PASS_31BW_BROADER_LAYER_PROBE_PLAN_SELECTED`.
      - Scope: planning-only. No validation execution in 31BW. No model load. No Q2_K / SDIR artifacts. No generation / inference / runtime integration. No commit/push/tag without explicit Matt approval.
      - Proof trail reviewed (read-only): 31BU (3 pairs, layer 0, PASSED), 31BV (6 pairs, layers 0/14/27, PASSED). 31BV observed per-pair time ~40 sec, per-layer margin variance 26 bytes, worst pair L0-S9 (dc=+0.001090, MAE_delta=−0.003686).
      - Four candidate next-executable scopes considered:
        - Option A — 7-layer stratified probe (layers 0, 4, 8, 12, 16, 20, 27; seeds 0, 9; 14 pairs; ~9 min)
        - Option B — 14-layer half-model probe (every other layer; 28 pairs; ~19 min)
        - Option C — full 28-layer aggregate (56 pairs; ~37 min; crosses aggregate territory — forbidden by 31BW spec)
        - Option D — seed sensitivity probe (3 layers × 8 seeds = 24 pairs; ~16 min; no new layer coverage)
      - Cost / risk table grounded in 31BV observations (per-pair time, per-layer margin, peak resident memory).
      - **Selected: Option A (7-layer stratified).** Rationale: 2x scope step from 31BV (3 → 7 layers), preserves top/mid/end sampling, claim boundary stays 'stratified probe' (7 of 28 layers = 25%), no aggregate territory, runtime well within tolerance.
      - Selected next phase: **Phase 31BX — Qwen2.5-1.5B Corrected Q2_K Stratified Layer Probe**, only if explicitly requested. Layers [0, 4, 8, 12, 16, 20, 27] × seeds [0, 9] = 14 pairs. Same `corrected_q2k_policy_v1`. W_ref = 1.5B Q4_K_M dequantized. No generation / inference / runtime integration.
      - Success criteria for 31BX: regression passes before/after; all 14 pairs finite; all 14 pairs memory-positive; 0 severe regressions; >= 12/14 cosine-improved; >= 12/14 MAE-improved; no model files committed; no blobs committed; no scope creep.
      - Classification rules for 31BX: PASS / PARTIAL (minor failures / tradeoff / memory fail) / BLOCKED (env / model / Q2_K backend / SDIR / regression).
      - Stop conditions for 31BX: regression fail, model tracking issue, non-finite outputs, severe regression, memory-negative, scope creep, claim trigger, blob commit.
      - Allowed claims if 31BX passes (conservative): "Corrected Q2_K policy remains memory-positive and improves cosine/MAE across a stratified 1.5B standalone tensor-harness probe (7 layers × 2 seeds = 14 pairs)". Plus L0-cross-runner-reproducibility + mixed source-GGUF quant types observation. Forbidden: full larger-model validation, inference/generation, speedup, quality/behavior, runtime, production, "1.5B behaves like 0.5B".
      - Runner: `src/phase31bw_broader_layer_probe_planning.py` (no model load; planning-only; reads 31BU/31BV result JSONs; writes planning JSON; lint-clean).
      - Result JSON: `src/results/PHASE31BW_BROADER_LAYER_PROBE_PLANNING.json` (~17 KB; no blobs).
      - Doc: `docs/PHASE31BW_BROADER_LAYER_PROBE_PLANNING.md`.
      - Pre-flight regression: `python3 -m tests.run_source_of_truth_regression` → `PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN`, `error_count=0`, `fallback_count=0`, `readme_drift_guard.passed=true` (run before AND after SOT/doc/result edits; must remain clean for commit approval).
      - Pre-existing untracked 31BH-R2 / 31BJ files: untouched per explicit instruction; not in this commit; will be handled in a dedicated stale-file cleanup / provenance phase.
      - Next allowed phase: **Phase 31BX — Qwen2.5-1.5B Corrected Q2_K Stratified Layer Probe**, only if explicitly requested. (Not entered — explicit request required.)
      - **Accepted claim:** 31BW successfully selected Option A (7-layer stratified) as the safest next executable phase after 31BU + 31BV. Selected scope: layers [0, 4, 8, 12, 16, 20, 27] × seeds [0, 9] = 14 pairs, under `corrected_q2k_policy_v1`. Estimated runtime ~9 min (14 pairs × ~40 sec/pair from 31BV observation). Claim boundary stays 'stratified probe' (7 of 28 layers = 25%), does not approach aggregate. Success criteria, classification rules, stop conditions, and allowed claims defined for 31BX.
      - **Valid as long as:**
        - 31BU and 31BV accepted facts remain unchanged (the planning inputs)
        - the 1.5B Q4_K_M file at `$SDI_MODEL_DIR/...` remains unmodified
        - the next phase (31BX) is entered only on explicit Matt request
        - the planning options (A/B/C/D) and their cost/risk estimates are recorded and unchanged in the planning JSON
        - the canonical orientation convention (Section 7) remains unchanged
        - no later phase invalidates or supersedes this plan
    - **Phase 31BX — Qwen2.5-1.5B Corrected Q2_K Stratified Layer Probe:** Classification `PASS_31BX_1_5B_Q2K_STRATIFIED_LAYER_CLEAN`.
      - Scope: stratified standalone tensor-harness probe only — 31BW Option A: layers [0, 4, 8, 12, 16, 20, 27] × seeds [0, 9] = 14 anchor pairs (7 of 28 layers = 25% layer coverage).
      - Model / W_ref: Qwen2.5-1.5B-Instruct Q4_K_M, dequantized (NOT FP16). W_ref source: downloaded 1.5B Q4_K_M GGUF (`$SDI_MODEL_DIR/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf`, 1,117,320,736 bytes / 1.04 GiB).
      - Tensors read (per layer) — **selected source-GGUF tensor types (per family per layer):**
        - ffn_up: **Q4_K** for layers 0, 4, 8, 12, 16, 20, 27 (raw [1536, 8960], dequant [8960, 1536])
        - ffn_gate: **Q4_K** for layers 0, 4, 8, 12, 16, 20, 27 (raw [1536, 8960], dequant [8960, 1536])
        - ffn_down: **Q6_K** for layers 0, 8, 16, 27; **Q4_K** for layers 4, 12, 20 (raw [8960, 1536], dequant [1536, 8960])
      - 31BX observed **mixed source-GGUF quant types for ffn_down** in the 7 selected layers: L0/L8/L16/L27 use Q6_K (4 of 7), while L4/L12/L20 use Q4_K (3 of 7). This is a source-GGUF characteristic. The corrected Q2_K memory accounting is unaffected because W_ref is dequantized to float32 before corrected Q2_K encoding, and the selected corrected_ceil_per_row Q2_K byte count is shape-dependent (constant across quant types for the same shape).
      - Policy: `corrected_q2k_policy_v1` (corrected_ceil_per_row Q2_K, ffn_up+ffn_gate SDIR k=0.5%, alpha=1.0, ffn_down W_low only — no ffn_down residual).
      - Per-pair metrics:
        - L0-S0:  dc=+0.007008, MAE_delta=−0.004610 (reproduces 31BU/31BV)
        - L0-S9:  dc=+0.001090, MAE_delta=−0.003686 (reproduces 31BU/31BV)
        - L4-S0:  dc=+0.004882, MAE_delta=−0.012024
        - L4-S9:  dc=+0.005096, MAE_delta=−0.013454
        - L8-S0:  dc=+0.005260, MAE_delta=−0.017519
        - L8-S9:  dc=+0.004245, MAE_delta=−0.008187
        - L12-S0: dc=+0.002792, MAE_delta=−0.017636
        - L12-S9: dc=+0.003998, MAE_delta=−0.016719
        - L16-S0: dc=+0.002604, MAE_delta=−0.005480
        - L16-S9: dc=+0.000652, MAE_delta=−0.004184 (worst pair)
        - L20-S0: dc=+0.008336, MAE_delta=−0.019044
        - L20-S9: dc=+0.009200, MAE_delta=−0.012580
        - L27-S0: dc=+0.002961, MAE_delta=−0.005428
        - L27-S9: dc=+0.006858, MAE_delta=−0.013261
      - Per-layer summary:
        - L0:  mean_dc=+0.004049, mean_MAE_imp=−0.004148, min_margin=+3,380,374 (reproduces 31BU/31BV)
        - L4:  mean_dc=+0.004989, mean_MAE_imp=−0.012739, min_margin=+3,380,352
        - L8:  mean_dc=+0.004752, mean_MAE_imp=−0.012853, min_margin=+3,380,342
        - L12: mean_dc=+0.003395, mean_MAE_imp=−0.017177, min_margin=+3,380,372
        - L16: mean_dc=+0.001628, mean_MAE_imp=−0.004832, min_margin=+3,380,364
        - L20: mean_dc=+0.008768, mean_MAE_imp=−0.015812, min_margin=+3,380,356
        - L27: mean_dc=+0.004909, mean_MAE_imp=−0.009345, min_margin=+3,380,350
      - Aggregate (within 14-pair stratified scope): n=14, n_mem_pos=14, n_cos_pos=14, n_mae_imp=14, n_severe=0, n_finite=14. mean_dc=+0.004641, median_dc=+0.004563, mean_MAE_imp=−0.010987, min_per_layer_margin=+3,380,342, max_per_layer_margin=+3,380,374. Per-layer margin variance: 32 bytes (~0.0009% of margin).
      - Memory: 14/14 memory-positive; per-layer margin consistent across the 7 selected layers (variance 32 bytes). Q2_K encode 4,515,840 bytes/family (corrected_ceil_per_row); SDIR ~1,127,786 bytes/family @ k=0.5% (computed at runtime, deterministic per seed).
      - L0 result **exactly reproduces** 31BU's L0-S0 and L0-S9, and 31BV's L0-S0 and L0-S9 — confirms cross-runner reproducibility across 3 separate runners.
      - Forbidden claims preserved: no quality / behavior / speedup / runtime / llama.cpp / inference / production / aggregate / full-28-layer / FP16-recovery / "1.5B behaves like 0.5B" claim; no commit/push/tag without explicit Matt approval.
      - Runner: `src/phase31bx_1_5b_corrected_q2k_stratified_layer_probe.py` (env-vars-only, no private paths, lint-clean).
      - Result JSON: `src/results/PHASE31BX_1_5B_CORRECTED_Q2K_STRATIFIED_LAYER_PROBE.json` (~20 KB; model path redacted to `$SDI_MODEL_DIR/...`).
      - Doc: `docs/PHASE31BX_1_5B_CORRECTED_Q2K_STRATIFIED_LAYER_PROBE.md`.
      - Pre-flight regression: `python3 -m tests.run_source_of_truth_regression` → `PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN`, `error_count=0`, `fallback_count=0`, `readme_drift_guard.passed=true` (run before AND after SOT/doc/result edits; must remain clean for commit approval).
      - Limitations: stratified standalone tensor-harness probe only (14 pairs, 7 layers: 0, 4, 8, 12, 16, 20, 27); no FP16 W_ref; no 0.5B comparison in this phase; no llama.cpp runtime integration; no generation/inference; no claim that 1.5B behaves like 0.5B or that the result generalizes to the 21 untested layers.
      - Pre-existing untracked 31BH-R2 / 31BJ files: untouched per explicit instruction; not in this commit; will be handled in a dedicated stale-file cleanup / provenance phase.
      - Next allowed phase: **Phase 31BY — Qwen2.5-1.5B Corrected Q2_K Aggregate Planning / Stop-Go Decision**, only if explicitly requested. (Not entered — explicit request required.)
      - **Accepted claim:** 31BX proves the corrected Q2_K policy (`corrected_q2k_policy_v1`, corrected_ceil_per_row, ffn_up + ffn_gate SDIR k=0.5%, alpha=1.0, no ffn_down residual) is **memory-positive and quality-improving on Qwen2.5-1.5B across the stratified 7-layer set [0, 4, 8, 12, 16, 20, 27] × seeds [0, 9]** in a standalone tensor harness (14/14 pairs memory-positive, cosine-improved, MAE-improved, 0 severe, all finite, mean delta_cos=+0.004641, mean MAE improvement=−0.010987, min per-layer margin +3,380,342 bytes). Per-layer margins are consistent across the 7 selected layers (variance 32 bytes). L0 result exactly reproduces 31BU and 31BV. Phrased relative to Q4_K_M W_ref (NOT FP16).
      - **Valid as long as:**
        - the downloaded 1.5B Q4_K_M file at `$SDI_MODEL_DIR/...` remains unmodified
        - the probe inputs remain tiny deterministic (`np.random.default_rng(seed)`, batch 1) — not real activations
        - the policy parameters match `corrected_q2k_policy_v1` (Q2_K mode, residual families, k, alpha, ffn_down residual)
        - the result is interpreted as stratified probe (14 pairs, 7 layers), NOT aggregate validation, NOT a full 28-layer generalization, NOT a 0.5B comparison, NOT a larger-model claim
        - the canonical orientation convention (Section 7) remains unchanged
        - no later phase invalidates or supersedes this probe result
        - `gguf.dequantize()` continues to return tensors in canonical (d_out, d_in) layout (re-verify if upstream `gguf` package changes)
    - **Phase 31BY — Qwen2.5-1.5B Corrected Q2_K Aggregate Planning / Stop-Go Decision:** Classification `PASS_31BY_AGGREGATE_STOP_GO_PLAN_SELECTED`.
      - Scope: PLANNING-ONLY phase. No aggregate validation execution. No full 28-layer sweep. No generation / inference / sampling. No llama.cpp runtime integration. No Q2_K / SDIR artifact generation. No model files committed. No performance / quality / runtime / behavior claim.
      - Reviewed prior evidence: 31BU (L0 only, 3 seeds) PASSED; 31BV (L0/L14/L27, 2 seeds) PASSED; 31BW (planning, Option A selected) PASSED; 31BX (L0/L4/L8/L12/L16/L20/L27, 2 seeds) PASSED. 23 cumulative pairs across 3 independent runners, 0 severe, 0 memory-fail, 0 finite-fail. L0 result is bit-identical across 31BU, 31BV, and 31BX (reproducibility verified, not assumed). 31BX worst pair (L16-S9) still at Δcos=+0.000652, MAE Δ=−0.004184 — strongly positive.
      - Selected policy unchanged from 31BX: `corrected_q2k_policy_v1` (corrected_ceil_per_row Q2_K, ffn_up + ffn_gate SDIR k=0.5%, alpha=1.0, no ffn_down residual). W_ref = Qwen2.5-1.5B-Instruct Q4_K_M dequantized (NOT FP16). 31BT canonical MLP formula unchanged: `Y = (silu(X @ W_gate.T) * (X @ W_up.T)) @ W_down.T`.
      - Options considered:
        - **Option A (SELECTED)** — GO full 28-layer × 2-seed aggregate: layers 0–27, seeds {0, 9}, 56 pairs. Crosses into aggregate territory. Estimated runtime ~25 min working / ~35 min upper bound (per 31BX 9-min wall-clock observation scaled sub-linearly for single-process 4× pair count, minus one-time model dequantize amortization).
        - Option B — GO 28-layer × 1-seed aggregate first: layers 0–27, seed 9 only, 28 pairs. ~16 min. Strictly weaker than A at comparable cost; budget-cut version of A.
        - Option C — MORE SEED SENSITIVITY before full aggregate: layers {0, 8, 16, 27}, seeds {0,1,2,3,4,5,9,13}, 32 pairs. ~14 min. Does not increase layer coverage (still 4 of 28 = 14% — a regression from 31BX's 25%). Does not advance the project from stratified to full-layer evidence.
        - Option D — STOP / FREEZE current 1.5B probe evidence: no further 1.5B validation. Unjustified: worst observed pair is still strongly positive; there is no current signal of a layer where the policy misbehaves; freezing at 7-of-28 coverage leaves the natural next-step gap unfilled.
      - Cost / risk summary: Option A peak temp disk 17,263,466 B per layer (freed after the layer's pair(s) complete); peak resident memory ~120 MB; artifact size ~60 KB. No 0.5B comparison triggered. No inference / generation / runtime. Step ladder is controlled: anchor (1 layer, 3 seeds) → small multi-layer (3 layers, 2 seeds) → stratified (7 layers, 2 seeds) → full-layer aggregate (28 layers, 2 seeds); each rung ran under the same policy, same runner family, same harness, same W_ref source.
      - Success criteria for 31BZ (unambiguous CLEAN definition): regression passes before AND after (PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN, error_count=0, fallback_count=0, readme_drift_guard.passed=true); all 56 pairs finite (n_finite == 56); all 56 pairs memory-positive (n_memory_positive == 56); 0 severe regressions (n_severe_regressions == 0); >= 54/56 cosine-improved; >= 54/56 MAE-improved; no model files committed; no Q2_K/SDIR blobs committed (all temp under tempfile.mkdtemp); no scope creep beyond layers 0–27, seeds {0, 9}, no ffn_down residual, no FP16 W_ref. **The 2-pair allowance applies ONLY to the cos+ and MAE+ soft thresholds (the two allowances are independent). The hard criteria (finite, memory-positive, non-severe, no files/blobs, no scope creep) carry NO allowance — 1 violation of any hard criterion disqualifies CLEAN regardless of cos+/MAE+ counts.** 54/56 threshold (96.4%) reflects proportional increase in sample size (4× over 31BX), not a relaxation of per-pair pass criterion; per-pair pass criterion remains identical.
      - Stop conditions for 31BZ (severity routing): non-finite pairs are HARD (1–2 non-finite → PARTIAL_TRADEOFF; >= 3 non-finite → BLOCKED structural harness failure); memory-negative is HARD (ANY single memory-negative pair/layer → PARTIAL_MEMORY_FAIL, no allowance, no count threshold); severe regression (Δcos < -0.05) is HARD (ANY single severe pair → PARTIAL_TRADEOFF, unless also memory-negative in which case PARTIAL_MEMORY_FAIL takes precedence); cos+/MAE+ soft-fails are SOFT (1–2 pairs not cos+/not MAE+ under the 2-pair allowance → CLEAN-eligible; 3+ pairs not cos+ OR 3+ pairs not MAE+ → PARTIAL_MINOR_FAILURES). Block: regression fail, env unset, model file missing, Q2_K backend fail, SDIR fail (each → its BLOCKED_* classification).
      - Classifications for 31BZ: PASS_31BZ_1_5B_Q2K_FULL_LAYER_TWO_SEED_AGGREGATE_CLEAN, PARTIAL_31BZ_1_5B_Q2K_AGGREGATE_MINOR_FAILURES, PARTIAL_31BZ_1_5B_Q2K_AGGREGATE_TRADEOFF, PARTIAL_31BZ_1_5B_Q2K_AGGREGATE_MEMORY_FAIL, BLOCKED_31BZ_SDI_MODEL_DIR_UNSET, BLOCKED_31BZ_MODEL_FILE_MISSING, BLOCKED_31BZ_Q2K_BACKEND_FAIL, BLOCKED_31BZ_SDIR_FAIL, BLOCKED_SOURCE_OF_TRUTH_REGRESSION.
      - Claim boundaries for 31BZ: allowed if PASSES — "Corrected Q2_K policy remained memory-positive and improved cosine/MAE across a full-layer, two-seed Qwen2.5-1.5B standalone tensor-harness aggregate (28 layers × 2 seeds = 56 pairs)." Still forbidden if PASSES — no inference/generation claim, no runtime/speedup/latency claim, no model quality claim, no behavior recovery claim, no production readiness claim, no llama.cpp integration claim, no FP16 recovery claim, no 0.5B-vs-1.5B generalization claim, no broader model-family claim, no claim that the result transfers to real activations or to a serving stack, no claim that the result transfers to 7B/14B/32B/72B/110B+ Qwen2.5 or any other family.
      - Pre-flight regression: `python3 -m tests.run_source_of_truth_regression` → `PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN`, `error_count=0`, `fallback_count=0`, `readme_drift_guard.passed=true` (run before AND after SOT/doc/result edits; must remain clean for commit approval).
      - Limitations: planning-only phase; no validation executed; no Q2_K/SDIR artifacts; no model load; no generation/inference; no claim that 1.5B behaves like 0.5B; no 0.5B comparison in 31BY. Cost estimates are derived from 31BX 9-min wall-clock observation scaled sub-linearly for single-process pair count increase; actual 31BZ runtime will be reported in the 31BZ PRE-COMMIT REPORT.
      - Pre-existing untracked 31BH-R2 / 31BJ files: untouched per explicit instruction; not in this commit; will be handled in a dedicated stale-file cleanup / provenance phase.
      - Next allowed phase: **Phase 31BZ — Qwen2.5-1.5B Corrected Q2_K Full-Layer Two-Seed Aggregate**, only if explicitly requested. (Not entered — explicit request required.)
      - **Accepted claim:** 31BY selected Option A (full 28-layer × 2-seed aggregate) as the next executable phase based on three prior accepted phases (31BU/31BV/31BX, 23 cumulative pairs all clean, L0 bit-identical across three runners, worst Δcos still strongly positive at +0.000652). Step ladder: anchor → small multi-layer → stratified → full-layer aggregate. Strict but proportional success criteria (≥54/56 cos+ and MAE+, 0 severe, all finite, all memory-positive). Claim boundary explicitly excludes inference/generation/runtime/speedup/quality/behavior/production/llama.cpp integration/FP16 recovery/0.5B-vs-1.5B generalization/broader model family. 31BY is planning-only; 31BZ is the executable phase that would run the selected 56-pair aggregate.
      - **Valid as long as:**
        - the downloaded 1.5B Q4_K_M file at `$SDI_MODEL_DIR/...` remains unmodified
        - the policy parameters match `corrected_q2k_policy_v1` (Q2_K mode, residual families, k, alpha, ffn_down residual)
        - 31BU, 31BV, 31BW, 31BX accepted facts remain valid (no later phase invalidates or supersedes them)
        - the canonical orientation convention (Section 7) remains unchanged
        - no later phase invalidates or supersedes this stop-go decision
        - the cost estimate (~25 min working / ~35 min upper) is treated as a planning estimate, not a runtime guarantee; 31BZ PRE-COMMIT REPORT will report actual runtime
    - **Phase 31BZ — Qwen2.5-1.5B Corrected Q2_K Full-Layer Two-Seed Aggregate:** Classification `PASS_31BZ_1_5B_Q2K_FULL_LAYER_TWO_SEED_AGGREGATE_CLEAN`.
      - Scope: standalone tensor-harness aggregate only (NOT generation, NOT inference, NOT llama.cpp runtime integration). 31BY Option A executed: 28 layers [0..27] × seeds [0, 9] = 56 anchor pairs (28 of 28 layers = 100% layer coverage). corrected_ceil_per_row Q2_K for all 3 families; SDIR residual on ffn_up + ffn_gate only, k=0.5%, alpha=1.0; ffn_down W_low only (no residual). W_ref = Qwen2.5-1.5B-Instruct Q4_K_M dequantized (NOT FP16). 31BT canonical MLP formula.
      - Result: 56/56 pairs memory-positive, 56/56 cosine-improved, 56/56 MAE-improved, 0 severe regressions, all finite. mean_delta_cos = +0.0047576, median_delta_cos = +0.0045015, mean_MAE_improvement = −0.0128652, min_per_layer_margin = +3,380,312 bytes, max_per_layer_margin = +3,380,376 bytes (variance 64 bytes across 28 layers). All 28 layers PASS independently when bucketed (each layer has 2 pairs, all mem+/cos+/MAE+/non-severe/finite).
      - Worst pair: L26-S9, delta_cos = +0.000390, MAE_delta = −0.029222, per_layer_margin = +3,380,356. Still strongly positive. No severe regression, no memory-negative, no non-finite.
      - Reproducibility vs prior phases (cross-runner / cross-phase): L0 result (L0-S0, L0-S9) bit-identical to 31BU, 31BV, 31BX (3 prior runners, all matching to 6+ decimal places). L4 result bit-identical to 31BX (L4-S0: dc=+0.004882, L4-S9: dc=+0.005096). L14 result bit-identical to 31BV (L14-S0: dc=+0.003992, L14-S9: dc=+0.001325). L27 result bit-identical to 31BV / 31BX. 4 prior runners (31BU L0 only, 31BV L0/L14/L27, 31BX L0/L4/L8/L12/L16/L20/L27, 31BZ all 28) all agree to the bit on the overlapping layer/seed pairs. The corrected Q2_K + SDIR + corrected_ceil_per_row pipeline is reproducible across 4 independent runners.
      - Tensor-type observations across all 28 layers: ffn_up and ffn_gate are uniformly Q4_K across all 28 layers. ffn_down is mixed: 14 of 28 layers (L0, L1, L5, L6, L7, L8, L9, L10, L13, L16, L19, L21, L24, L27) use Q6_K; 14 of 28 layers (L2, L3, L4, L11, L12, L14, L15, L17, L18, L20, L22, L23, L25, L26) use Q4_K. The alternating pattern is a source-GGUF characteristic of the official Qwen2.5-1.5B-Instruct Q4_K_M build, not a defect. The corrected Q2_K memory accounting is unaffected because W_ref is dequantized to float32 before corrected Q2_K encoding, and the corrected_ceil_per_row Q2_K byte count is shape-dependent (constant across quant types for the same shape).
      - Wall-clock observation: 481.5 sec total = 8.0 min for 56 pairs (~8.6 sec/pair average; first-pair model load + 56 pair runs). Informational only. NOT used for any claim. Far faster than the 31BY planning estimate of 25 min working / 35 min upper bound (actual was ~3× faster than my lower bound), because the dominant cost is per-pair dequantize + matmul and the model is hot in page cache across all pairs.
      - Pre-flight regression: `python3 -m tests.run_source_of_truth_regression` → `PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN`, `error_count=0`, `fallback_count=0`, `readme_drift_guard.passed=true` (run before AND after SOT/doc/result edits; must remain clean for commit approval).
      - Limitations: standalone tensor-harness aggregate only (56 pairs, all 28 layers, 2 seeds); no FP16 W_ref; no 0.5B comparison in this phase; no llama.cpp runtime integration; no generation/inference; no claim that 1.5B behaves like 0.5B or that the result generalizes to 3B / 7B / 14B / 32B / 72B / 110B+ Qwen2.5 or any other family. Wall-clock is informational only — does not establish any latency, throughput, or production-runtime claim.
      - Pre-existing untracked 31BH-R2 / 31BJ files: untouched per explicit instruction; not in this commit; will be handled in a dedicated stale-file cleanup / provenance phase.
      - Next allowed phase: **Phase 31CA — Qwen2.5-1.5B Corrected Q2_K Aggregate Freeze / Package Update**, only if explicitly requested. (Not entered — explicit request required.)
      - **Accepted claim:** 31BZ proves the corrected Q2_K policy (`corrected_q2k_policy_v1`, corrected_ceil_per_row, ffn_up + ffn_gate SDIR k=0.5%, alpha=1.0, no ffn_down residual) is **memory-positive and quality-improving on Qwen2.5-1.5B across the FULL 28-layer set [0..27] × seeds [0, 9]** in a standalone tensor harness (56/56 pairs memory-positive, cosine-improved, MAE-improved, 0 severe, all finite, mean delta_cos=+0.004758, mean MAE improvement=−0.012865, min per-layer margin +3,380,312 bytes). L0 result bit-identical to 31BU / 31BV / 31BX across 3 prior runners. L4 bit-identical to 31BX. L14 / L27 bit-identical to 31BV. Per-layer margins consistent across all 28 layers (variance 64 bytes). The 28-layer aggregate cleanly extends the prior 7-layer stratified probe (31BX) with no new failure mode introduced. Phrased relative to Q4_K_M W_ref (NOT FP16). NOT a larger-model claim (only Qwen2.5-1.5B tested). NOT a 0.5B-vs-1.5B comparison. NOT a generation/inference/runtime integration claim.
      - **Valid as long as:**
        - the downloaded 1.5B Q4_K_M file at `$SDI_MODEL_DIR/...` remains unmodified
        - the probe inputs remain tiny deterministic (`np.random.default_rng(seed)`, batch 1) — not real activations
        - the policy parameters match `corrected_q2k_policy_v1` (Q2_K mode, residual families, k, alpha, ffn_down residual)
        - the result is interpreted as standalone tensor-harness aggregate (56 pairs, 28 layers × 2 seeds), NOT generation / inference / llama.cpp runtime integration, NOT a full larger-model claim, NOT a 0.5B comparison, NOT a broader-family claim
        - the canonical orientation convention (Section 7) remains unchanged
        - no later phase invalidates or supersedes this aggregate result
        - `gguf.dequantize()` continues to return tensors in canonical (d_out, d_in) layout (re-verify if upstream `gguf` package changes)
        - the wall-clock observation (481.5 sec / 8.0 min) is treated as informational only, not as a speedup, latency, throughput, or production-runtime claim
    - **Phase 31CA — Qwen2.5-1.5B Corrected Q2_K Aggregate Freeze / Package Update:** Classification `PASS_31CA_1_5B_AGGREGATE_FREEZE_PACKAGE_READY`.
      - Scope: **freeze + package + provenance + documentation phase only.** NO new validation, NO generation / inference / sampling, NO llama.cpp runtime integration, NO new scientific work. NO tag was created in 31CA. NO commit / push / tag occurred in 31CA. The 31BZ accepted fact (56/56 pairs PASS on Qwen2.5-1.5B full-layer × 2-seed aggregate) is preserved as a frozen checkpoint and the policy package is updated to record the 1.5B as a second evidence tier.
      - Checkpoint target commit: `f7f2a91d1b904f8f156d6c89584ec0d32229c23e` (the 31BZ commit).
      - Proposed tag (NOT created in 31CA): `phase31ca-1_5b-corrected-q2k-aggregate-checkpoint`. Tag message: "Phase 31CA: freeze 1.5B corrected Q2_K aggregate checkpoint (56/56 pairs PASS)". Operator must approve tag creation explicitly.
      - Policy package update: `corrected_q2k_policy_v1` package (`docs/CORRECTED_Q2K_POLICY_PACKAGE.md` + `src/results/CORRECTED_Q2K_POLICY_PACKAGE.json`) updated to add 1.5B (31BZ/31CA) as a second evidence tier alongside 0.5B (31BN/31BM). **Policy parameters UNCHANGED. Policy version NOT bumped.** The 1.5B evidence tier is documentation, not code. `src/corrected_q2k_policy.py`, `src/q2k_backend.py`, `src/phase31x_manifest_runtime.py` all UNCHANGED.
      - README update: conservative in-place patch to lines 22, 29, 30, 50. The README was out of date on the 1.5B status (it still said "No 1.5B tensor validation yet" and "next phase is 31BU"); these are now corrected. README structure and tone preserved. README continues to defer to SOURCE_OF_TRUTH.md for the full audit trail.
      - Frozen allowed claim: "Corrected Q2_K policy (`corrected_q2k_policy_v1`, corrected_ceil_per_row Q2_K, ffn_up + ffn_gate SDIR k=0.5%, alpha=1.0, no ffn_down residual) is memory-positive and quality-improving on Qwen2.5-1.5B across the full 28-layer set [0..27] × seeds [0, 9] in a standalone tensor harness (56/56 pairs memory-positive, cosine-improved, MAE-improved, 0 severe, all finite). Phrased relative to Q4_K_M W_ref (NOT FP16)."
      - Frozen allowed claim: "The policy is reproducible across 4 independent runners (31BU, 31BV, 31BX, 31BZ) at the bit level (6+ decimal places) on all 8 overlapping layer/seed pairs."
      - Frozen forbidden claims (preserved from 31BZ, no additions, no removals): no model quality recovery claim, no behavior recovery claim, no speedup claim, no full-model runtime memory savings claim, no llama.cpp integration claim, no production readiness claim, no inference / generation / sampling claim, no runtime-ready output-residual claim, no FP16 recovery claim, no 0.5B-vs-1.5B generalization claim, no "1.5B behaves like 0.5B" claim, no broader model-family claim, no real-activation-transfer claim, no larger-model claim, no claim that 31AY / 31BA exact anchors are current canonical metrics, no claim beyond standalone tensor harness.
      - Valid-as-long-as clauses (preserved from 31BZ, with the 31CA-specific addition): all 31BZ valid-as-long-as clauses hold; AND the `corrected_q2k_policy_v1` package remains at version `v1` (if a future phase bumps the policy version, the 31CA freeze does not implicitly carry forward).
      - Pre-existing untracked 31BH-R2 / 31BJ files: untouched per explicit instruction; not in this commit; will be handled in a dedicated stale-file cleanup / provenance phase (31CB, recommended default per 31CA's reasoning).
      - Next allowed phase: **Phase 31CB — Stale File Provenance Cleanup, only if explicitly requested** (recommended default per 31CA's reasoning). Alternative 31CB options (operator may choose instead): 31CB-A Real-Activation Capture Planning, 31CB-B Runtime Artifact Format / Loader Planning. (Not entered — explicit request required.)
      - **Accepted claim:** 31CA successfully preserves the 31BZ 1.5B aggregate as a frozen checkpoint and updates the `corrected_q2k_policy_v1` package to record the 1.5B as a second evidence tier alongside the 0.5B. Policy parameters are unchanged. Policy version is unchanged. The 1.5B aggregate is reproduced bit-identically across 4 independent runners (31BU, 31BV, 31BX, 31BZ) on all 8 overlapping layer/seed pairs. The package is now backed by two frozen evidence tiers (0.5B and 1.5B), both under the same `corrected_q2k_policy_v1`. No new validation, generation, inference, or runtime integration ran in 31CA. 31CA is a documentation + provenance + handoff phase.
      - **Valid as long as:**
        - the 31BZ valid-as-long-as clauses all hold (model file unmodified; inputs remain tiny deterministic; policy parameters match `corrected_q2k_policy_v1`; canonical orientation convention unchanged; no later phase invalidates 31BZ; `gguf.dequantize()` returns canonical layout; wall-clock observation treated as informational only)
        - the `corrected_q2k_policy_v1` package remains at version `v1` (if a future phase bumps the policy version, the 31CA freeze does not implicitly carry forward)
        - the policy package docs (`docs/CORRECTED_Q2K_POLICY_PACKAGE.md` + `src/results/CORRECTED_Q2K_POLICY_PACKAGE.json`) and the 31CA freeze doc (`docs/PHASE31CA_1_5B_AGGREGATE_FREEZE_PACKAGE.md` + `src/results/PHASE31CA_1_5B_AGGREGATE_FREEZE_PACKAGE.json`) remain accessible to future agents as the package manifest
        - no later phase invalidates or supersedes this freeze
        - the proposed tag `phase31ca-1_5b-corrected-q2k-aggregate-checkpoint` is treated as a recommendation, not a requirement; if the operator chooses to tag at a different commit (e.g. a future commit that further refines the package), the 31CA freeze remains valid as long as the policy parameters and 1.5B evidence tier remain unchanged
    - **Phase 31CB — Stale File Provenance Cleanup:** Classification `PASS_31CB_STALE_FILE_PROVENANCE_CLEANED`.
      - Scope: **provenance / working-tree hygiene phase only.** NO new validation, NO generation / inference / sampling, NO llama.cpp runtime integration, NO modification of scientific results, NO tags. The 5 stale untracked 31BH-R2 / 31BJ files are moved to `rescue/stale_phase31bh_31bj/` with a provenance README. No commit, push, tag, delete, or move occurs without explicit Matt approval.
      - Stale file inventory (as found, before 31CB): 5 files at original paths `docs/PHASE31BH_R2_Q2K_ANCHOR_REPRO_FIX.md` (5.7K, 154 lines, classification `PASS_31BH_R2_HISTORICAL_ANCHORS_REPRODUCED` per MD header), `src/phase31bh_q2k_clean_reproduction.py` (24K, 606 lines, runner code), `src/results/PHASE31BH_Q2K_RUNTIME_QUANT_PATH.json` (2.1K, 74 lines, classification `PARTIAL_31BH_Q2K_BACKEND_ADDED_ANCHOR_MISMATCH`), `src/results/PHASE31BH_R2_Q2K_ANCHOR_REPRO_FIX.json` (11K, 379 lines, classification `BLOCKED_31BH_R2_ANCHORS_STILL_MISMATCH` with `historical_reproduction: {L21_S9: false, L21_S0: false}`), `src/results/PHASE31BJ_ANCHOR_ONLY.json` (2.1K, 73 lines, raw single-pair L21-S9 / L21-S0 numbers with one entry showing `margin: -261638`).
      - Per-file disposition: 5 of 5 → **MOVE_TO_RESCUE**. 0 → COMMIT_AS_HISTORICAL_RECORD. 0 → DELETE_AFTER_CONFIRMATION. 0 → LEAVE_UNTRACKED_WITH_REASON. Rationale for each file documented in `docs/PHASE31CB_STALE_FILE_PROVENANCE_CLEANUP.md` §4 and in `rescue/stale_phase31bh_31bj/README.md`.
      - Specific contradictions found: (a) 31BH-R2 MD header claims PASS for historical anchor reproduction but corresponding JSON says BLOCKED with `historical_reproduction: {false, false}` — MD-vs-JSON classification drift is real; (b) 31BJ ANCHOR_ONLY contains a single `corrected_ceil_per_row` row with `margin: -261638` (memory-NEGATIVE for that single Q2_K W_low variant) which could be misread as contradicting the 0.5B/1.5B memory-positive aggregate without context; (c) 31BH-R2 SOT entry uses a different classification name (`PASS_31BH_R2_FIX_RUNTIME_DISPATCH_REPAIRED`) referring to a runtime dispatch fix rather than anchor reproduction — SOT entry is the committed audit-trail fact and cannot be re-edited.
      - Rescue folder contents: `rescue/stale_phase31bh_31bj/` contains 6 files (1 provenance README + 5 moved stale files). The original relative path is encoded in the moved filename (`docs/` → `docs_`, `src/results/` → `src_results_`) so future agents can recover the original location.
      - No scientific / numeric claim changes: 0.5B memory+ / cos+ / MAE+ (31BN/31BM), 1.5B memory+ / cos+ / MAE+ (31BZ/31CA), 31AY/31BA historical anchors (historical-only, not canonical), 31BJ single-pair raw numbers (rescue-only, not canonical), 31BH-R2 PASS/BLOCKED classification drift (rescue-only, not canonical) — all **unchanged**. The 31CB cleanup is provenance and working-tree hygiene only.
      - Re-validation rule (for any future phase): if a future phase needs to re-validate the historical anchors (e.g. to refresh the 31AY/31BA reference metrics or to re-test the `historical_floor_flat` Q2_K mode for any reason), the future phase MUST (1) read the rescue README first, (2) re-derive the metrics from scratch using the current `src/corrected_q2k_policy.py` constants and the current `src/q2k_backend.py` + `src/phase31x_manifest_runtime.py` implementations — do NOT import the stale runner code from the rescue folder, (3) record the re-validation under a new phase name with a fresh result JSON, not by editing the stale files in the rescue folder, (4) not cite the stale files in the rescue folder as current evidence, even after re-validation.
      - Pre-flight regression (post-cleanup): `python3 -m tests.run_source_of_truth_regression` → `PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN`, `error_count=0`, `fallback_count=0`, `readme_drift_guard.passed=true` (run before AND after SOT/doc/result edits; must remain clean for commit approval).
      - Limitations: provenance / working-tree hygiene only. No validation, no generation, no inference, no llama.cpp runtime integration, no tag creation, no modification of scientific results, no model files touched, no policy package modification. 5 stale files were moved (NOT deleted, NOT committed). The working tree's noise from 5 stale untracked files at original paths is resolved; the files are now at rescue paths.
      - Pre-existing committed artifacts: unchanged. 31CA tag `phase31ca-1_5b-corrected-q2k-aggregate-checkpoint` (created after the 31CA package commit and before 31CB; points to commit `a433875a`, the 31CA package commit) was NOT touched in 31CB. 0.5B tag `phase31bn-corrected-q2k-full-aggregate-checkpoint` (at commit `0304590c`) was NOT touched in 31CB. All other tags unchanged.
      - Next allowed phase: **Phase 31CC — Qwen2.5-1.5B Real-Activation Capture Planning, only if explicitly requested** (recommended default per 31CB's reasoning). Alternative: 31CC Runtime Artifact Format / Loader Planning. (Not entered — explicit request required.)
      - **Accepted claim:** 31CB successfully inspected 5 stale 31BH-R2 / 31BJ files, classified each, and moved all 5 to `rescue/stale_phase31bh_31bj/` with a 12.9 KB provenance README. The disposition rationale is documented per-file in `docs/PHASE31CB_STALE_FILE_PROVENANCE_CLEANUP.md` and in the rescue README. No file was committed, no file was deleted, no file was promoted to canonical evidence, no scientific/numeric claim was changed. The 0.5B and 1.5B frozen evidence tiers remain exactly as 31CA froze them. The working tree is now free of stale untracked files at the original paths. 31CB is a provenance + working-tree hygiene phase.
      - **Valid as long as:**
        - the 5 moved files remain in `rescue/stale_phase31bh_31bj/` and are not promoted to canonical evidence without an explicit future phase
        - the `rescue/stale_phase31bh_31bj/README.md` remains accessible to future agents as the canonical warning
        - the 0.5B and 1.5B frozen evidence tiers (31BN/31BM, 31BZ/31CA) remain unchanged
        - the 31CB accepted fact and rescue-folder documentation are not silently re-edited
        - any future re-validation of historical anchors follows the re-validation rule in `rescue/stale_phase31bh_31bj/README.md` §5
        - the canonical orientation convention (SOT Section 7) remains unchanged
        - no later phase invalidates or supersedes this cleanup
        - the 31CB disposition (5 files → MOVE_TO_RESCUE) remains the audit-trail fact; future phases may add new evidence but should not retroactively reclassify the 5 moved files as canonical

    - **Phase 31CC — Qwen2.5-1.5B Real-Activation Capture Planning:** Classification `PASS_31CC_REAL_ACTIVATION_CAPTURE_PLAN_SELECTED` (revised after a pre-commit wording/contradiction review; classification unchanged — only wording/planning details were patched; 31CC is a planning-only phase, no scientific/numeric results changed).
      - Scope: **planning-only phase.** NO activation capture execution, NO model inference beyond metadata inspection, NO generation / inference / sampling, NO llama.cpp modification, NO hook creation, NO model file download, NO Q2_K / SDIR artifact generation, NO quality / performance / runtime / behavior / production claim, NO commit / push / tag without explicit Matt approval. 31CC is the design phase for 31CD, not an executable validation phase.
      - Real-activation definition (canonical for 31CC and 31CD): the actual hidden-state vector X entering a selected FFN/MLP block during a real model forward pass on tokenized prompt text, captured **before** `ffn_up` and `ffn_gate` are applied. X has shape `[tokens_or_batch, hidden]` with `hidden=1536` for 1.5B, `896` for 0.5B. Capture point is the pre-FFN residual stream input to the FFN block of layer L (tensor flowing into `blk.L.ffn_up` and `blk.L.ffn_gate`). Replay uses the canonical 31BT formula `Y = (silu(X @ W_gate.T) * (X @ W_up.T)) @ W_down.T`. Policy is `corrected_q2k_policy_v1`. W_ref for replay is **always** the local 1.5B Q4_K_M GGUF dequantized, matching 31BU/31BV/31BX/31BZ exactly — UNCHANGED for both Option A and Option B.
      - 5-strategy table evaluated in `docs/PHASE31CC_REAL_ACTIVATION_CAPTURE_PLANNING.md` and `src/results/PHASE31CC_REAL_ACTIVATION_CAPTURE_PLANNING.json`:
        - **Option A — HF-derived real-activation proxy (one of two valid 31CD mechanisms):** HuggingFace `transformers` `forward_hook` capture on the FFN block. **Proven methodology on this host** — Phase 31I used exactly this mechanism for Qwen2.5-0.5B (12 hooks registered, 15 prompts processed, output saved to `data/PHASE31I_activations.npz`). Requires a new HF safetensors download (~3 GB BF16/FP16) into `$SDI_MODEL_DIR/qwen2.5-1.5b-hf/` with explicit Matt approval per the 31BS approval phrase template. **NO W_ref source pivot** — the W_ref for replay stays as the local 1.5B Q4_K_M GGUF dequantized, matching 31BU/31BV/31BX/31BZ exactly. The X captured under Option A is **labeled an HF-derived real-activation proxy** because it comes from the HF safetensors model binary (a different binary from the local Q4_K_M GGUF), not from the local Q4_K_M GGUF itself. Option A cannot claim exact Q4_K_M GGUF runtime activation behavior.
        - **Option B — llama.cpp/GGUF activation capture instrumentation (the other valid 31CD mechanism; the only path that can claim exact Q4_K_M GGUF runtime activation capture):** minimal diagnostic hook inside llama.cpp to dump pre-FFN activation for a specified layer and token position. Uses the **existing local 1.5B Q4_K_M GGUF**. Preserves the W_ref source convention (Q4_K_M dequantized). 31CD's cos_low is directly comparable to 31BZ's cos_low. Requires operator permission to modify and rebuild `~/llama.cpp` on this host (approval required at 31CD entry, not 31CC). **Preferred if the goal is exact Q4_K_M GGUF runtime activation capture.** No existing precedent in this repo for llama.cpp Python integration beyond `libggml-base.so` ctypes for Q2_K encode/decode.
        - **Option C — Stop and re-plan via 31CC-R (valid 31CD-entry choice):** Matt chooses neither A nor B at 31CD entry; 31CD classifies as `BLOCKED_31CD_CAPTURE_PATH_UNAVAILABLE`; next permissible phase is 31CC-R.
        - **Option D — minimal local forward-proxy (rejected):** reconstruct one layer's upstream in numpy. Almost certainly incorrect, high implementation risk, no methodological value, would cross into territory 31CC is forbidden to enter.
        - **Option E — runtime / loader format planning (out of scope):** legitimate alternative already permitted in SOT Section 9 (31CC Runtime Artifact Format / Loader Planning); not selected because 31CB reasoning identified real-activation capture as the largest remaining scientific gap.
      - 31CD-entry decision point (HARD GATE): **31CD does not begin until Matt explicitly chooses A, B, or C at 31CD entry.** 31CC does NOT auto-select between A and B. If no option is selected at entry, 31CD classifies as `BLOCKED_31CD_NO_OPTION_SELECTED_AT_ENTRY`. The two 31CD mechanisms (A and B) are equivalent for everything except the X source: W_ref, W_low, SDIR, prompt, layers, token position, metrics, and artifact policy are identical.
      - 31CD scope (applies to both A and B; designed in 31CC, executed only on explicit request with operator approval at entry): default prompt `"The capital of France is"` (6 BPE tokens, PII-free, public-domain, sourced from 31I prompt index 1; operator may select from the 31I prompt set or supply a new short prompt at 31CD entry); default layers `{0, 14, 27}` (mirrors 31BV's first 1.5B layer choice; reducible to `{0}` for the first iteration); default token position **last prefill token** (the standard prefill final-position hidden state); batch=1, shape `[1, 1536]`; replay through W_ref MLP / Q2_K-only MLP / Q2_K+SDIR MLP per `corrected_q2k_policy_v1`; metrics: `cos_low, cos_sub, delta_cos, MAE_low, MAE_sub, MAE_delta, finite, severe, memory margin`. W_ref for replay = local 1.5B Q4_K_M GGUF dequantized (unchanged from 31BU/31BV/31BX/31BZ). W_low = corrected_ceil_per_row Q2_K from Q4_K_M GGUF W_ref (unchanged). SDIR = ffn_up+ffn_gate SDIR k=0.5%, alpha=1.0, no ffn_down residual (unchanged).
      - 31CD activation artifact policy (conservative default): raw activation arrays written to `tempfile.mkdtemp(prefix="phase31cd_")`, NOT committed. 31CD result JSON contains only summary metrics, `sha256_of_raw_X_per_layer`, `captured_shape_per_layer`, `captured_dtype`, redacted prompt metadata (prompts > 64 chars redacted to `sha256[:16] + "..." + len_chars + "(redacted)"`), and token-position metadata. A commit-exception (small activation fixture ≤ 1 MB under `data/phase31cd_*.npz`) requires explicit Matt approval at 31CD entry. No PII in default 31CD prompt set.
      - 31CD result JSON mandatory fields: `option_selected` ("A" or "B"); `w_ref_source` ("local 1.5B Q4_K_M GGUF dequantized, matches 31BU/31BV/31BX/31BZ convention" — identical for A and B); `w_low_source`; `sdir_source`; `x_source` (Option A: "HF-derived real-activation proxy; NOT exact Q4_K_M GGUF runtime activation"; Option B: "exact Q4_K_M GGUF runtime activation"); `model_path_observed_redacted`; `sha256_of_raw_X_per_layer`; `captured_shape_per_layer`; `captured_dtype`.
      - 31CD stop conditions: 17 documented (regression fail; no option selected at 31CD entry; `SDI_MODEL_DIR` unset; model file missing; capture path unavailable; unapproved dependency; invasive runtime patch beyond approved scope; X shape != `[1, 1536]`; X contains NaN/Inf; replay formula mismatch; Q2_K backend fail; SDIR fail; raw activation artifact > 1 MB committed accidentally; model file / HF cache / raw activation blob committed accidentally; any inference / generation / quality / behavior / speed / runtime claim added; Option-A-only: any claim of exact Q4_K_M GGUF runtime activation behavior; both options: any claim of real llama.cpp runtime integration).
      - 31CD classification list: 17 documented (`PASS_31CD_REAL_ACTIVATION_MICRO_REPLAY_CLEAN`, `PARTIAL_31CD_REAL_ACTIVATION_MICRO_REPLAY_MINOR_FAILURES`, `PARTIAL_31CD_REAL_ACTIVATION_CAPTURE_ONLY`, `PARTIAL_31CD_REPLAY_FORMULA_MISMATCH`, `BLOCKED_31CD_NO_OPTION_SELECTED_AT_ENTRY`, `BLOCKED_31CD_CAPTURE_PATH_UNAVAILABLE`, `BLOCKED_31CD_HF_DOWNLOAD_NOT_APPROVED`, `BLOCKED_31CD_LLAMA_CPP_MODIFICATION_NOT_APPROVED`, `BLOCKED_31CD_SDI_MODEL_DIR_UNSET`, `BLOCKED_31CD_MODEL_FILE_MISSING`, `BLOCKED_31CD_Q2K_BACKEND_FAIL`, `BLOCKED_31CD_SDIR_FAIL`, `BLOCKED_31CD_TOKENIZER_OR_MODEL_LOAD_FAIL`, `BLOCKED_31CD_FORWARD_HOOK_FAIL`, `BLOCKED_31CD_LLAMA_CPP_BUILD_FAIL`, `BLOCKED_31CD_REGRESSION_FAIL`, `BLOCKED_31CD_ALLOWED_PHASE_CONFLICT`).
      - 31CD allowed claims: 4 documented (corrected Q2_K + SDIR non-degrading on real-activation micro-probe; corrected Q2_K policy memory-positive on real-activation micro-probe; real-activation capture methodology viable via HF/llama.cpp; Option-B-only: exact Q4_K_M GGUF runtime activation capture viable via llama.cpp instrumentation). The 31CD allowed claims list does **NOT** include any "reproduces synthetic-Gaussian micro-probe behavior" / "matches / behaves like / is equivalent to synthetic Gaussian" / "consistent with synthetic-Gaussian micro-probe" claim. The 31CD result MAY be described as **directionally consistent** with prior synthetic-Gaussian tensor-harness results only within the selected 31CD prompt/layer/token scope, and only if the real-activation metrics show the same sign pattern: memory-positive, finite, delta_cos non-negative or positive, and MAE non-worsening or improving — and even this is **not** an "allowed claim" of 31CD, it is a **permitted description language** in a 31CD result text, with no scientific force.
      - 31CD forbidden claims: 20 documented (12 universal + 4 Option-A-only + 2 Option-B-only + **2 explicit-claim-language reinforcements**). Universal: no generation quality, no model behavior, no inference latency / throughput / wall-clock beyond informational note, no runtime integration, no production readiness, no broader activation-distribution claim, no all-token / all-layer / all-prompt claim, no claim that real generation will work, no transfer-to-other-prompts/layers/models claim, no 0.5B-vs-1.5B real-activation comparison, **no claim that real activations behave like synthetic Gaussian**, no model files / HF cache / raw activation arrays committed without explicit operator approval. Option-A-only: no claim of exact Q4_K_M GGUF runtime activation behavior; no claim that HF safetensors forward pass is equivalent to Q4_K_M GGUF forward pass; no claim of real llama.cpp runtime integration (Option A does not touch llama.cpp); no claim of generation / inference quality (restated). Option-B-only: no claim of real llama.cpp runtime integration (the Option B hook is diagnostic, not an integration); no claim of generation / inference quality (restated). **Explicit-claim-language reinforcements:** **(1) Do not claim real activations behave like synthetic Gaussian. (2) Do not claim activation-distribution equivalence. (3) Do not claim transfer beyond the selected prompt/layer/token scope.** These three reinforcements close the loophole where weaker language like "consistent with", "broadly consistent with", "matches", "reproduces", "comparable to", or "similar to" might be misread as a behavioral-equivalence claim — they are not allowed as 31CD claims, only as **permitted description language** in 31CD result text (see the "31CD allowed claims" line above), and only with the sign-pattern caveat.
      - 31CC entry-dependency approvals (deferred to 31CD entry, not required for 31CC): if Option A is chosen, HF safetensors download approval per the 31BS approval phrase template; if Option B is chosen, llama.cpp modification and rebuild approval; if Option C is chosen, the next permissible phase is **Phase 31CC-R — Real-Activation Capture Planning Repair, only if explicitly requested** (re-plan). Other 31CD-entry decisions: final prompt selection, final layer selection, activation artifact disposition.
      - 31CC upstream review (read-only): SOT Section 0, Section 0.A, Section 3 (31BT / 31BU / 31BV / 31BW / 31BX / 31BY / 31BZ / 31CA / 31CB entries), Section 7, Section 8, Section 9, Section 13, Section 14; `docs/CORRECTED_Q2K_POLICY_PACKAGE.md`; `src/results/CORRECTED_Q2K_POLICY_PACKAGE.json`; `src/corrected_q2k_policy.py`; `src/phase31x_manifest_runtime.py`; `src/phase31bt_1_5b_orientation_parity_micro_probe.py`; `src/phase31bx_1_5b_corrected_q2k_stratified_layer_probe.py`; `src/phase31bz_1_5b_corrected_q2k_full_layer_two_seed_aggregate.py`; `data/phase31i_capture.log`; `data/phase31i_sweep.log`; `data/phase31i_capture.sh`; `data/PHASE31I_activations.npz` (size + keys metadata only; no payload extraction); `tests/run_source_of_truth_regression.py`; `README.md` (also patched in 31CC). 31CC acknowledged Phase 31I (historical, untracked, not committed) as the only prior real-activation capture methodology in this repo; 31I's runner (`phase31i_capture_activations.py`) is not on disk — only the log, the shell wrapper, and the `.npz` artifact survive. 31I's residual policy was the pre-31AL `k=7.5%`, not the current `corrected_q2k_policy_v1` k=0.5%; 31I's activations are historical methodology evidence only and do not represent any current accepted numeric result.
      - 31CC deliverables (untracked, awaiting Matt approval): `docs/PHASE31CC_REAL_ACTIVATION_CAPTURE_PLANNING.md` (45.6 KB, revised), `src/results/PHASE31CC_REAL_ACTIVATION_CAPTURE_PLANNING.json` (38.9 KB, revised), `README.md` (conservative patch: advance "current scientific next phase" from 31CC to 31CD; still defers to SOT), and the SOT edit (Section 0 last-verified-commit + revised 31CC planning-result line + revised current-allowed-scientific-next-phase line + revised validation-scope line + revised artifact-status line; Section 3 new 31CC entry above; Section 9 advance to 31CD with revised rationale). **No new file was committed in 31CC. No new tag was created in 31CC. No model file was committed. No Q2_K / SDIR artifact was generated. No raw activation array was generated. No W_ref source pivot was introduced.**
      - Pre-flight regression (before SOT/doc/result edits): `python3 -m tests.run_source_of_truth_regression` → `PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN`, `error_count=0`, `fallback_count=0`, `readme_drift_guard.passed=True` (confirmed).
      - Post-edit regression (after SOT/doc/result edits): same regression is re-run and must report the same clean state (will be re-run before the PRE-COMMIT REPORT).
      - Limitations: planning-only phase; no validation, no generation, no inference, no llama.cpp runtime integration, no tag creation, no modification of scientific results, no model files touched, no policy package modification, no Q2_K/SDIR artifact generation, no raw activation arrays generated. 31CC does NOT prove that real-activation capture will succeed; 31CC is the design phase for the experiment. 31CC does NOT prove that the corrected Q2_K policy holds on real activations; that is 31CD's job (and 31CD is conditional on Matt's explicit choice of A, B, or C at entry). 31CC does NOT auto-select between Option A and Option B; the choice is Matt's at 31CD entry; 31CC documents both. **31CC does NOT introduce a W_ref source pivot** — the W_ref for replay in 31CD is the local Q4_K_M GGUF dequantized, matching 31BU/31BV/31BX/31BZ exactly, for BOTH Option A and Option B. The only thing that differs across the two options is the X source. The earlier 31CC draft's W_ref source pivot idea is RETRACTED.
      - Pre-existing committed artifacts: unchanged. 0.5B tag `phase31bn-corrected-q2k-full-aggregate-checkpoint` (at `0304590c`), 1.5B tag `phase31ca-1_5b-corrected-q2k-aggregate-checkpoint` (at `a433875a`), and all other tags unchanged. The `corrected_q2k_policy_v1` package, the 0.5B and 1.5B frozen evidence tiers, and the 5 files in `rescue/stale_phase31bh_31bj/` are unchanged.
      - Next allowed phase (after 31CC): **Phase 31CD — Qwen2.5-1.5B Real-Activation Capture Micro-Probe, only if explicitly requested**. 31CD does NOT begin until Matt explicitly chooses Option A, Option B, or Option C at 31CD entry. If Option C is chosen (or both A and B are declined), the next permissible phase is **Phase 31CC-R — Real-Activation Capture Planning Repair, only if explicitly requested**.
      - **Accepted claim:** 31CC successfully produced a complete real-activation capture plan for Qwen2.5-1.5B. The plan defines (a) a precise definition of "real activation" for this project, (b) a 5-strategy capture table with Option A (HuggingFace `transformers` forward-hook capture, HF-derived real-activation **proxy**) and Option B (llama.cpp instrumentation capture, the only path that can claim exact Q4_K_M GGUF runtime activation capture) as the two valid 31CD mechanisms, Option C (stop and re-plan via 31CC-R) as the explicit re-plan branch, Options D (local forward-proxy) rejected and E (runtime/loader format planning) recorded as out of scope, (c) a hard 31CD-entry decision point requiring Matt's explicit choice of A, B, or C before 31CD can begin, (d) the exact 31CD micro-probe scope including default prompt (the 31I prompt index 1: "The capital of France is", 6 BPE tokens, PII-free, public-domain), default layers ({0, 14, 27}, reducible to {0}), default token position (last prefill token), batch=1, shape [1, 1536], and replay through W_ref / Q2_K-only / Q2_K+SDIR MLPs per `corrected_q2k_policy_v1`, (e) W_ref / W_low / SDIR are UNCHANGED from the frozen 31BU/31BV/31BX/31BZ convention (local 1.5B Q4_K_M GGUF dequantized W_ref) for both Option A and Option B — **no W_ref source pivot**; the only thing that differs is the X source (Option A: HF safetensors; Option B: local Q4_K_M GGUF via llama.cpp), (f) a conservative activation artifact policy (raw X in tempfile, not committed by default; commit-exception requires explicit operator approval at 31CD entry; hard cap 1 MB), (g) 17 stop conditions, (h) 17 classifications, (i) **4 allowed claims** (3 universal + 1 Option-B-only; the previously-listed "real-activation replay reproduces synthetic-Gaussian micro-probe behavior" allowed claim is REMOVED to close a contradiction with the universal "no real activations behave like synthetic Gaussian" forbidden claim), (j) **20 forbidden claims** (12 universal + 4 Option-A-only + 2 Option-B-only + 2 explicit-claim-language reinforcements: (1) Do not claim real activations behave like synthetic Gaussian. (2) Do not claim activation-distribution equivalence. (3) Do not claim transfer beyond the selected prompt/layer/token scope.), and (k) 3 entry-dependency approvals deferred to 31CD entry (Option A choice + HF safetensors download; OR Option B choice + llama.cpp modification; OR Option C choice). 31CC is planning-only; no activation capture, no generation, no inference, no llama.cpp modification, no hook creation, no Q2_K/SDIR artifact generation, no model file download, no commit/push/tag without explicit Matt approval. The corrected_q2k_policy_v1 package, the 0.5B and 1.5B frozen evidence tiers, the 0.5B and 1.5B checkpoint tags, and the 5 rescue files in `rescue/stale_phase31bh_31bj/` are unchanged. The earlier 31CC draft's W_ref source pivot idea (FP16/BF16 safetensors for Option A) is RETRACTED and is NOT claimed by 31CC or any future 31CD result.
      - **Valid as long as:**
        - the 0.5B and 1.5B frozen evidence tiers (31BN/31BM, 31BZ/31CA) remain unchanged
        - the `corrected_q2k_policy_v1` package remains at version v1 (parameters UNCHANGED)
        - the canonical orientation convention (SOT Section 7) remains unchanged
        - the existing local 1.5B Q4_K_M GGUF model file at `$SDI_MODEL_DIR/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf` remains unmodified
        - `gguf.dequantize()` continues to return tensors in canonical (d_out, d_in) layout
        - the 31CC planning doc, planning JSON, README patch, and SOT edits are not silently re-edited after commit
        - no later phase invalidates or supersedes this planning result without an explicit SOT update
        - the 5 moved 31BH-R2 / 31BJ files in `rescue/stale_phase31bh_31bj/` remain rescue-only and are not promoted to canonical evidence
        - the 31CC W_ref convention (Q4_K_M GGUF dequantized for replay, for both Option A and Option B) is preserved; the earlier 31CC draft's W_ref source pivot idea remains RETRACTED
        - the 31CD-entry decision point (Matt chooses A, B, or C) is honored; 31CD does not begin until Matt's choice is recorded

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
**Phase 31CD — Qwen2.5-1.5B Real-Activation Capture Micro-Probe, only if explicitly requested** (selected as 31CC's next executable phase; **31CD does NOT begin until Matt explicitly chooses between Option A, Option B, or Option C at 31CD entry**; the choice is the 31CD-entry hard gate).

Rationale:
- Phase 31CC PASSED with classification `PASS_31CC_REAL_ACTIVATION_CAPTURE_PLAN_SELECTED` (revised after a pre-commit wording/contradiction review; classification unchanged — only wording/planning details were patched).
- 31CC is a **planning-only phase**. NO activation capture execution, NO model inference beyond metadata inspection, NO generation / inference / sampling, NO llama.cpp modification, NO hook creation, NO model file download, NO Q2_K / SDIR artifact generation, NO quality / performance / runtime / behavior / production claim, NO commit / push / tag without explicit Matt approval.
- 31CC produced `docs/PHASE31CC_REAL_ACTIVATION_CAPTURE_PLANNING.md` (45.6 KB, revised) and `src/results/PHASE31CC_REAL_ACTIVATION_CAPTURE_PLANNING.json` (38.9 KB, revised). The planning doc + JSON define a 5-strategy capture table (Option A = HF-derived real-activation proxy; Option B = the only path that can claim exact Q4_K_M GGUF runtime activation capture; Option C = stop and re-plan via 31CC-R; Option D = local forward-proxy, rejected; Option E = runtime/loader format planning, out of scope), the exact 31CD micro-probe scope (default prompt "The capital of France is" 6 BPE tokens; default layers {0, 14, 27} reducible to {0}; default token position last prefill token; batch=1, shape [1, 1536]; replay through W_ref / Q2_K-only / Q2_K+SDIR MLPs per `corrected_q2k_policy_v1`), the conservative activation artifact policy (raw X in tempfile, not committed by default; commit-exception requires explicit operator approval at 31CD entry; hard cap 1 MB), the 17 stop conditions, the 17 classifications, the 5 allowed claims, and the 18 forbidden claims for 31CD.
- 5-strategy table from 31CC: **Option A** (HuggingFace `transformers` forward-hook capture, **HF-derived real-activation proxy**, requires new HF safetensors download ~3 GB BF16/FP16 with explicit Matt approval per the 31BS approval phrase template; W_ref for replay stays as Q4_K_M GGUF dequantized — **no W_ref source pivot**; Option A cannot claim exact Q4_K_M GGUF runtime activation behavior because the X is captured from the HF safetensors, a different model binary from the local Q4_K_M GGUF); **Option B** (llama.cpp instrumentation capture, **the only path that can claim exact Q4_K_M GGUF runtime activation capture**, uses existing local 1.5B Q4_K_M GGUF, preserves W_ref source convention, requires operator permission to modify and rebuild ~/llama.cpp on this host, no existing precedent in this repo for llama.cpp Python integration beyond `libggml-base.so` ctypes for Q2_K encode/decode, **preferred if the goal is exact Q4_K_M GGUF runtime activation capture**); **Option C** (stop and re-plan via 31CC-R, valid 31CD-entry choice); **Option D** (minimal local forward-proxy, rejected for methodology / implementation / forbidden-territory reasons); **Option E** (runtime / loader format planning, out of scope for 31CC; legitimate alternative already permitted in SOT as 31CC Runtime Artifact Format / Loader Planning but not selected because 31CB reasoning identified real-activation capture as the largest remaining scientific gap).
- **NO W_ref source pivot in Option A** — the earlier 31CC draft proposed pivoting W_ref to FP16/BF16 safetensors for Option A; this is **RETRACTED**. In the revised 31CD plan, BOTH Option A and Option B use the same W_ref (local 1.5B Q4_K_M GGUF dequantized, matching 31BU/31BV/31BX/31BZ exactly), the same W_low (corrected_ceil_per_row Q2_K from the Q4_K_M GGUF), and the same SDIR (ffn_up+ffn_gate SDIR k=0.5%, alpha=1.0, no ffn_down residual) for replay. The only thing that differs across the two options is the **X source**: Option A captures X from the HF safetensors (an HF-derived proxy, NOT exact Q4_K_M GGUF runtime activation); Option B captures X from the local Q4_K_M GGUF via llama.cpp instrumentation (EXACT Q4_K_M GGUF runtime activation). The deployed artifacts are identical across A and B and across 31BU/31BV/31BX/31BZ. 31CD's cos_low is directly comparable to 31BU/31BV/31BX/31BZ's cos_low in both options.
- 31CD entry-dependency approvals (NOT required at 31CC time, required at 31CD entry): if Matt chooses **Option A**, explicit HF safetensors download approval per the 31BS approval phrase template; if Matt chooses **Option B**, explicit permission to modify and rebuild ~/llama.cpp on this host; if Matt chooses **Option C**, no further approvals needed and the next permissible phase is **Phase 31CC-R — Real-Activation Capture Planning Repair, only if explicitly requested** (re-plan). Until Matt's choice is recorded, 31CD classifies as `BLOCKED_31CD_NO_OPTION_SELECTED_AT_ENTRY`.
- Prior accepted numeric results (0.5B 31AY / 31BA / 31BM / 31BN / 31BO; 1.5B 31BS / 31BT / 31BU / 31BV / 31BW / 31BX / 31BY / 31BZ; 31CA freeze; 31CB cleanup) are unchanged. The 31CC planning phase does not modify any prior accepted fact. The `corrected_q2k_policy_v1` package (corrected_ceil_per_row Q2_K, ffn_up+ffn_gate SDIR k=0.5%, alpha=1.0, no ffn_down residual) is unchanged. The 0.5B and 1.5B checkpoint tags (phase31bn-corrected-q2k-full-aggregate-checkpoint at 0304590c, phase31ca-1_5b-corrected-q2k-aggregate-checkpoint at a433875a) are unchanged. The 5 rescue files in `rescue/stale_phase31bh_31bj/` are unchanged.
- Stop conditions for 31CC (enforced in 31CC; not future conditions): regression failure, scope creep, any quality / behavior / performance / runtime / inference / generation / production claim, any FP16 recovery claim, any model file committed, any Q2_K/SDIR artifact generated, any raw activation array generated, any llama.cpp modification, any hook creation, any model file download, any commit/push/tag without explicit Matt approval, any modification of the corrected_q2k_policy_v1 package, any modification of prior accepted numeric results, any W_ref source pivot (the W_ref for 31CD stays as Q4_K_M GGUF dequantized for both A and B).
- The 31CC artifacts (`docs/PHASE31CC_REAL_ACTIVATION_CAPTURE_PLANNING.md`, `src/results/PHASE31CC_REAL_ACTIVATION_CAPTURE_PLANNING.json`, the README patch, and the SOT edit) are untracked. They will be committed only if Matt explicitly approves.

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
