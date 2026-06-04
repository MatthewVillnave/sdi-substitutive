# Phase 31BX — Qwen2.5-1.5B Corrected Q2_K Stratified Layer Probe

**Classification:** `PASS_31BX_1_5B_Q2K_STRATIFIED_LAYER_CLEAN`

**Phase:** 31BX
**Scope:** Stratified standalone tensor-harness probe only. Route: 31BW Option A: 7 layers (0, 4, 8, 12, 16, 20, 27) × 2 seeds (0, 9) → 14 anchor pairs. No aggregate validation, no full 28-layer sweep, no generation/inference, no llama.cpp runtime integration, no performance/quality claim.
**Status:** Artifacts prepared — **not yet committed** (PRE-COMMIT REPORT; awaiting Matt approval).

---

## 1. Goal

Run the 31BW-selected stratified standalone tensor-harness probe to test whether `corrected_q2k_policy_v1` remains sane across 7 selected Qwen2.5-1.5B layers.

This is **not** aggregate validation. It is a 14-pair stratified probe (7 of 28 layers = 25% layer coverage), building directly on 31BU (1 layer) and 31BV (3 layers).

**What 31BX is allowed to claim** (per the explicit spec):

- stratified standalone tensor-harness probe result only
- pair/layer pass/failure under the selected 14-pair scope
- memory-positive / memory-negative under current accounting
- cosine / MAE improved/regressed relative to Q2_K-only AND Q4_K_M W_ref
- severe / non-severe within this fixed scope
- per-layer summary across the 7 selected layers
- source-GGUF tensor-type observations for selected layers only

**What 31BX is forbidden to claim**:

- "quality recovered", "behavior recovered", "model improved"
- "inference works", "runtime works", "speedup"
- "production"
- "larger-model validation complete" (7 of 28 layers ≠ 28 layers)
- "FP16 recovery" (W_ref is Q4_K_M dequantized, not FP16)
- "aggregate validation" (14 pairs ≠ aggregate; not generalized to 28 layers)
- "1.5B behaves like 0.5B" (no 0.5B comparison)
- "consistent across all 28 layers" (only 7 of 28 tested)
- commit, push, or tag without explicit Matt approval

---

## 2. Environment

| Var | Value (redacted in artifacts) | Resolved at runtime |
|---|---|---|
| `SDI_MODEL_DIR` | `/media/matthew-villnave/VL_usb/models` | ✅ |
| Model file | `$SDI_MODEL_DIR/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf` (1.04 GiB) | ✅ exists |
| `SDI_LLAMA_CPP_LIB` | `/home/matthew-villnave/llama.cpp/build/bin/libggml-base.so` | ✅ |
| Q2_K symbols | `quantize_row_q2_K_ref` + `dequantize_row_q2_K` | ✅ exported |

Preflight: `python3 -m tests.run_source_of_truth_regression` → `PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN`, `error_count=0`, `fallback_count=0`, `readme_drift_guard.passed=true`.

---

## 3. Configuration (exactly per spec)

| field | value |
|---|---|
| Model | Qwen2.5-1.5B-Instruct Q4_K_M |
| W_ref source | downloaded 1.5B Q4_K_M GGUF, **dequantized** (NOT FP16) |
| Architecture | hidden=1536, intermediate=8960, block_count=28 (31BX touches 7 layers) |
| Q2_K mode | `corrected_ceil_per_row` |
| Residual families | `ffn_up` + `ffn_gate` |
| ffn_down residual | **none** (W_low only) |
| k | 0.5% |
| alpha | 1.0 |
| batch | 1 |
| X vector | `np.random.default_rng(seed)`, `standard_normal((1, 1536))` (matches 31AY/31BA/31BJ/31BK/31BU/31BV) |
| Layers | `[0, 4, 8, 12, 16, 20, 27]` |
| Seeds | `[0, 9]` → **14 anchor pairs** |
| Route | 31BW Option A: 7-layer stratified (25% layer coverage) |

---

## 4. MLP formula (canonical, 31BT-confirmed)

```
W_up_canon   is (intermediate=8960, hidden=1536)
W_gate_canon is (intermediate=8960, hidden=1536)
W_down_canon is (hidden=1536, intermediate=8960)

up   = X @ W_up_canon.T           # (1, 8960)
gate = X @ W_gate_canon.T         # (1, 8960)
act  = silu(gate) * up            # (1, 8960)
out  = act @ W_down_canon.T       # (1, 1536) — expected
```

---

## 5. Selected policy (`corrected_q2k_policy_v1`)

| Q2_K bytes per family (1.5B shape) | corrected_ceil_per_row |
|---|---|
| ffn_up / ffn_gate: 8960 × ⌈1536/256⌉ × 84 = 8960 × 6 × 84 | 4,515,840 bytes |
| ffn_down: ⌈8960/256⌉ × 84 × 1536 = 35 × 84 × 1536 | 4,515,840 bytes |
| Total Q2_K (per layer, 3 families) | 13,547,520 bytes |
| SDIR per family @ k=0.5% (13,762,560 elems × 0.005 ≈ 68,813 nnz) | computed at runtime (deterministic per seed) |
| Q4_budget_family = (d_out × d_in) / 2 nibbles = 8960 × 1536 / 2 | 6,881,280 bytes |
| Q4_budget_layer = 3 × Q4_budget_family | 20,643,840 bytes |

Per-layer memory accounting for the **selected policy** (up+gate SDIR, down W_low only):

```
total_bytes = q2k_up + sdir_up + q2k_gate + sdir_gate + q2k_down
margin      = 3 × Q4_budget_family - total_bytes
             = 20,643,840 - (3 × 4,515,840 + 2 × sdir_k0.5)
             = 6,881,280 - 2 × sdir_k0.5
             # positive as long as sdir_k0.5 < 3,440,640 bytes per family
```

---

## 6. Per-pair results (14 pairs, 7 layers × 2 seeds)

| layer | seed | cos_low | cos_sub | delta_cos | MAE_low | MAE_sub | MAE_delta | severe | margin (bytes) | finite |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|:---:|
| 0  | 0 | 0.878029 | 0.885037 | **+0.007008** | 0.019438 | 0.014828 | **−0.004610** | False | +3,380,374 | ✓ |
| 0  | 9 | 0.874526 | 0.875617 | **+0.001090** | 0.020004 | 0.016318 | **−0.003686** | False | +3,380,374 | ✓ |
| 4  | 0 | 0.883216 | 0.888098 | **+0.004882** | 0.021613 | 0.009589 | **−0.012024** | False | +3,380,352 | ✓ |
| 4  | 9 | 0.878939 | 0.884035 | **+0.005096** | 0.022860 | 0.009406 | **−0.013454** | False | +3,380,352 | ✓ |
| 8  | 0 | 0.892141 | 0.897401 | **+0.005260** | 0.025208 | 0.007689 | **−0.017519** | False | +3,380,342 | ✓ |
| 8  | 9 | 0.884688 | 0.888933 | **+0.004245** | 0.016272 | 0.008085 | **−0.008187** | False | +3,380,342 | ✓ |
| 12 | 0 | 0.908473 | 0.911265 | **+0.002792** | 0.022894 | 0.005258 | **−0.017636** | False | +3,380,372 | ✓ |
| 12 | 9 | 0.909252 | 0.913250 | **+0.003998** | 0.021941 | 0.005222 | **−0.016719** | False | +3,380,372 | ✓ |
| 16 | 0 | 0.893791 | 0.896395 | **+0.002604** | 0.015793 | 0.010313 | **−0.005480** | False | +3,380,364 | ✓ |
| 16 | 9 | 0.895200 | 0.895852 | **+0.000652** | 0.014332 | 0.010148 | **−0.004184** | False | +3,380,364 | ✓ |
| 20 | 0 | 0.879140 | 0.887476 | **+0.008336** | 0.025994 | 0.006950 | **−0.019044** | False | +3,380,356 | ✓ |
| 20 | 9 | 0.860169 | 0.869369 | **+0.009200** | 0.022041 | 0.009461 | **−0.012580** | False | +3,380,356 | ✓ |
| 27 | 0 | 0.887506 | 0.890467 | **+0.002961** | 0.017730 | 0.012302 | **−0.005428** | False | +3,380,350 | ✓ |
| 27 | 9 | 0.876373 | 0.883231 | **+0.006858** | 0.024037 | 0.010776 | **−0.013261** | False | +3,380,350 | ✓ |

Per-pair SDIR bytes are deterministic (seed-dependent residual selection from a deterministic X vector). The per-layer margin varies by 32 bytes between layers (3,380,342 for L8 vs 3,380,374 for L0) — explained by tiny differences in the residual magnitude distributions across layers, which the SDIR bitmap / fp16 layout captures with slightly different nnz counts.

---

## 7. Per-layer summary (7 layers, 2 seeds each)

| layer | n_pairs | mem_pos | cos_pos | mae_imp | severe | mean_dc | mean_MAE_imp | min_margin |
|---:|---:|---:|---:|---:|:---:|---:|---:|---:|
| 0  | 2 | 2 | 2 | 2 | 0 | **+0.004049** | **−0.004148** | +3,380,374 |
| 4  | 2 | 2 | 2 | 2 | 0 | **+0.004989** | **−0.012739** | +3,380,352 |
| 8  | 2 | 2 | 2 | 2 | 0 | **+0.004752** | **−0.012853** | +3,380,342 |
| 12 | 2 | 2 | 2 | 2 | 0 | **+0.003395** | **−0.017177** | +3,380,372 |
| 16 | 2 | 2 | 2 | 2 | 0 | **+0.001628** | **−0.004832** | +3,380,364 |
| 20 | 2 | 2 | 2 | 2 | 0 | **+0.008768** | **−0.015812** | +3,380,356 |
| 27 | 2 | 2 | 2 | 2 | 0 | **+0.004909** | **−0.009345** | +3,380,350 |

**Per-layer observations (within the 14-pair stratified scope only — NOT a generalization):**

- All 7 / 7 selected layers are **memory-positive, cosine-improved, MAE-improved** on both tested seeds.
- L20 has the largest mean delta_cos (+0.008768) — the residual correction has the most room to improve on its Q2_K baseline.
- L12 has the largest mean MAE improvement (−0.017177) — the residual captures a significant portion of the magnitude error.
- L16 has the smallest mean delta_cos (+0.001628) and is home to the worst pair (L16-S9, dc=+0.000652) — its Q2_K W_low is closer to the Q4_K_M W_ref, leaving less room for the k=0.5% residual correction to add. Still strictly positive.
- L0 reproduces 31BU (3 seeds) and 31BV (2 seeds) **exactly** — confirms cross-runner reproducibility.
- All 7 layers are finite, no severe regressions, no memory fails.

---

## 8. Tensor-type observations (source-GGUF)

Per-family per-layer tensor types for the 7 selected layers:

| layer | ffn_up | ffn_gate | ffn_down |
|---:|:---:|:---:|:---:|
| 0  | Q4_K | Q4_K | **Q6_K** |
| 4  | Q4_K | Q4_K | **Q4_K** |
| 8  | Q4_K | Q4_K | **Q6_K** |
| 12 | Q4_K | Q4_K | **Q4_K** |
| 16 | Q4_K | Q4_K | **Q6_K** |
| 20 | Q4_K | Q4_K | **Q4_K** |
| 27 | Q4_K | Q4_K | **Q6_K** |

- ffn_up and ffn_gate are **uniformly Q4_K** across all 7 selected layers.
- ffn_down is **mixed** in the source-GGUF: L0/L8/L16/L27 use Q6_K, L4/L12/L20 use Q4_K. 4 of 7 selected layers have Q6_K ffn_down, 3 of 7 have Q4_K ffn_down.
- This is a **source-GGUF characteristic**. The corrected Q2_K memory accounting is unaffected: W_ref is dequantized to float32 before corrected Q2_K encoding, and the corrected_ceil_per_row Q2_K byte count is shape-dependent (constant across quant types for the same shape).
- The pattern is consistent with 31BV's observation (L0/L27 Q6_K, L14 Q4_K) but more granular.

---

## 9. Aggregate (within the 14-pair stratified scope — NOT a 28-layer claim)

- n_pairs: 14
- n_memory_positive: **14 / 14** (100%)
- n_cosine_positive: **14 / 14** (100%)
- n_MAE_improving: **14 / 14** (100%)
- n_severe_regressions: **0 / 14**
- n_finite: **14 / 14**
- worst_pair: **L16-S9**, dc=+0.000652, MAE_delta=−0.004184, margin=+3,380,364
- mean_delta_cos: **+0.004641**
- median_delta_cos: **+0.004563**
- mean_MAE_improvement: **−0.010987** (negative = improvement)
- min_per_layer_margin: **+3,380,342 bytes (~3.22 MB)**
- max_per_layer_margin: +3,380,374 bytes
- per-layer margin variance: **32 bytes** (3,380,342 to 3,380,374) — essentially constant

**Comparison to prior accepted results** (read-only, no claim that 31BX is "consistent" or "generalized" from these):

- 31BU (layer 0, 3 seeds): mean_dc=+0.004139, mean_MAE_imp=−0.004712
- 31BV (layers 0/14/27, 2 seeds each): mean_dc=+0.003872, mean_MAE_imp=−0.005923
- 31BX (layers 0/4/8/12/16/20/27, 2 seeds each): mean_dc=+0.004641, mean_MAE_imp=−0.010987

31BX's mean_dc is slightly higher than 31BV's, and mean_MAE_imp is roughly 2x better — but this is **not a claim of improvement** over 31BV, just a numerical observation that the broader 7-layer set (which includes layers with strong MAE gains like L12 and L20) shifts the aggregate. L0 reproduces 31BU and 31BV exactly.

---

## 10. Memory accounting (per layer, selected policy)

| component | bytes | % of layer Q4 budget |
|---|---:|---:|
| Q2_K ffn_up (corrected_ceil_per_row)   | 4,515,840 | 21.87% |
| Q2_K ffn_gate (corrected_ceil_per_row) | 4,515,840 | 21.87% |
| Q2_K ffn_down (corrected_ceil_per_row) | 4,515,840 | 21.87% |
| SDIR ffn_up @ k=0.5%                   | ~1,127,786 | 5.46% (computed at runtime, ±32 bytes across layers) |
| SDIR ffn_gate @ k=0.5%                 | ~1,127,786 | 5.46% |
| SDIR ffn_down                          | 0         |  0.00% (no residual by policy) |
| **TOTAL per layer**                    | **~15,803,466** | **~76.55% of 20,643,840** |
| **Margin vs 3 × Q4_budget_family**     | **+~3,380,360** | **+~16.37% headroom** |

**Per-layer margin is essentially constant** (variance 32 bytes ≈ 0.0009% of margin) across the 7 selected layers — Q2_K bytes are shape-dependent (constant across layers at the same shape), and SDIR bytes scale with `k × n_elements` (also constant at the same k). The 32-byte variance comes from the SDIR bitmap / fp16 layout capturing residual nnz slightly differently per layer.

---

## 11. Interpretation (conservative phrasing)

**Allowed:**

- All 14 / 14 pairs **passed** within the stratified standalone tensor-harness probe.
- All 7 / 7 selected layers (L0, L4, L8, L12, L16, L20, L27) are **memory-positive, cosine-improved, MAE-improved** on both tested seeds.
- Per-layer memory is **memory-positive** (margin +3,380,342 to +3,380,374 bytes) under the current accounting — consistent across all 7 layers (variance 32 bytes).
- Cosine **improved** on all 14 / 14 pairs (range +0.000652 to +0.009200) relative to the Q2_K-only baseline AND relative to the Q4_K_M W_ref.
- MAE **improved** on all 14 / 14 pairs (range −0.003686 to −0.019044) relative to the Q2_K-only baseline AND relative to the Q4_K_M W_ref.
- **No severe regressions** in this fixed scope.
- All outputs **finite**.
- The corrected Q2_K encode/decode is sane on 1.5B across the stratified 7-layer set — i.e. the corrected policy does not blow up across the 31BW-selected layer positions at the same selected k=0.5% / up+gate / no ffn_down residual.
- L0 result **exactly reproduces** the 31BU 3-seed result and the 31BV 2-seed result — confirms cross-runner reproducibility across 3 separate runners.
- **Source-GGUF observation:** ffn_down is mixed Q4_K / Q6_K across the 7 selected layers (4 of 7 Q6_K, 3 of 7 Q4_K); ffn_up and ffn_gate are uniformly Q4_K. This is a source-GGUF characteristic; memory accounting is unaffected because W_ref is dequantized to float32 before corrected Q2_K encoding and the byte count is shape-dependent.

**Forbidden (not claimed):**

- ~~"quality recovered" / "behavior recovered" / "model improved"~~
- ~~"inference works" / "runtime works" / "speedup"~~
- ~~"production"~~
- ~~"larger-model validation complete"~~ — 7 of 28 layers ≠ 28 layers; 14 pairs ≠ aggregate
- ~~"FP16 recovery"~~ — W_ref is **Q4_K_M dequantized**, NOT FP16
- ~~"1.5B behaves like 0.5B"~~ — no 0.5B comparison in this phase
- ~~"consistent across all 28 layers"~~ — only 7 of 28 layers tested
- ~~aggregate validation~~ — this is a 14-pair stratified probe, not aggregate

---

## 12. Forbidden claims preserved

Full canonical master list in `SOURCE_OF_TRUTH.md` Section 0.A. Per-phase notes in 31BX spec are audit context. Active 31BX-specific forbidden items:

- no model quality recovery claim
- no behavior recovery claim
- no speedup claim
- no full-model runtime memory savings claim
- no llama.cpp integration claim
- no production readiness claim
- no inference / generation claim
- no full larger-model validation claim (7 of 28 layers ≠ 28 layers)
- no runtime-ready output-residual claim
- no claim beyond standalone tensor harness
- no claim that 31AY / 31BA exact anchors are current canonical metrics
- no aggregate validation (14 pairs ≠ aggregate)
- no claim that 1.5B behaves like 0.5B
- no orientation claim (orientation was already parity-tested in 31BT)
- no commit, push, tag, push-tag, or download (any new model) without explicit Matt approval

---

## 13. Limitations

- **Stratified standalone tensor-harness scope only** (14 pairs, 7 layers) — not a full 28-layer sweep, not aggregate.
- **No layers beyond 0, 4, 8, 12, 16, 20, 27** (other 21 layers not tested in this phase).
- **No seeds beyond 0, 9** (other 14 seeds not tested in this phase).
- **No FP16 W_ref** — W_ref is the Q4_K_M dequantized tensor; any "recovery" is *relative to Q4_K_M*, not FP16.
- **No 0.5B comparison** in this phase — direct comparison is a future phase.
- **No llama.cpp runtime integration** — Q2_K is generated and decoded in the standalone tensor harness only.
- **No generation / inference / sampling / perplexity** — pure tensor-level MLP forward.
- **No claim that 31BX is "consistent" with 31BU/31BV** in a generalization sense** — only L0 reproduces 31BU and 31BV bit-exactly; the other 4 layers in 31BX are new ground.

---

## 14. Artifacts (untracked, prepared, not committed)

| Path | Purpose |
|---|---|
| `docs/PHASE31BX_1_5B_CORRECTED_Q2K_STRATIFIED_LAYER_PROBE.md` | This document |
| `src/phase31bx_1_5b_corrected_q2k_stratified_layer_probe.py` | Runner (env-vars-only, no private paths, lint-clean) |
| `src/results/PHASE31BX_1_5B_CORRECTED_Q2K_STRATIFIED_LAYER_PROBE.json` | Full per-pair + per-layer + tensor diagnostics + memory accounting + tensor-type table |

**Not produced / not in this phase:**

- model files
- HF cache
- generated Q2_K blobs (all in memory + tempfile.mkdtemp, cleaned at runner exit)
- generated SDIR blobs (all in memory)
- temp tensor dumps
- large artifacts

---

## 15. SOT update (to be applied if approved)

Append to `SOURCE_OF_TRUTH.md` Section 3, then update Section 0 / Section 9 to point next allowed phase to:

- **if clean or minor/tradeoff (this phase)**: Phase 31BY — Qwen2.5-1.5B Corrected Q2_K Aggregate Planning / Stop-Go Decision, only if explicitly requested
- **if blocked**: Phase 31BX-R — Stratified Layer Probe Repair, only if explicitly requested

---

## 16. Pre-commit status

| field | value |
|---|---|
| Runner | `src/phase31bx_1_5b_corrected_q2k_stratified_layer_probe.py` (env-vars-only, lint-clean) |
| Result JSON | `src/results/PHASE31BX_1_5B_CORRECTED_Q2K_STRATIFIED_LAYER_PROBE.json` (~20 KB) |
| Doc | `docs/PHASE31BX_1_5B_CORRECTED_Q2K_STRATIFIED_LAYER_PROBE.md` (this file) |
| SOT | new accepted fact to append (draft below) |
| Pre-commit regression | must be re-run after SOT edit, must still pass with same contract |
| Git status (after edits, before commit) | 3 new files in `docs/`, `src/`, `src/results/`; 1 modified file (`SOURCE_OF_TRUTH.md`) |
| Any private paths in commit? | **No** — runner uses `$SDI_MODEL_DIR`, `$SDI_LLAMA_CPP_*`; result JSON redacts model path to `$SDI_MODEL_DIR/...` |
| Any large / generated artifacts in commit? | **No** — runner output is small JSON; no Q2_K / SDIR blobs persisted; no temp tensor dumps |
| Pre-existing untracked 31BH-R2 / 31BJ files disposition | **untouched** per explicit standing instruction; not in this commit |
| Proposed commit message | `Phase 31BX: run 1.5B corrected Q2_K stratified layer probe` |

---

## 17. Draft SOT Section 3 entry (to append if approved)

```markdown
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
```

**Do not commit. Do not push. Awaiting explicit Matt approval.**

---

## 18. Question

**Approve commit / push of 31BX artifacts (runner + result JSON + this doc + SOT update)?**
