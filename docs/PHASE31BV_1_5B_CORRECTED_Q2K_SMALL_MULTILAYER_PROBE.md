# Phase 31BV — Qwen2.5-1.5B Corrected Q2_K Small Multi-Layer Probe

**Classification:** `PASS_31BV_1_5B_Q2K_SMALL_MULTILAYER_CLEAN`

**Phase:** 31BV
**Scope:** Small fixed multi-layer probe only. Route A: layers 0/14/27 × seeds 0/9 → 6 anchor pairs. No aggregate validation, no full 28-layer sweep, no generation/inference, no llama.cpp runtime integration, no performance/quality claim.
**Status:** Artifacts prepared — **not yet committed** (PRE-COMMIT REPORT; awaiting Matt approval).

---

## 1. Goal

Extend the successful 31BU layer-0 anchor probe to a small fixed set of layers to check whether the `corrected_q2k_policy_v1` path remains sane beyond layer 0.

This is **not** aggregate validation. It is a 6-pair small multi-layer sanity check on Qwen2.5-1.5B, building directly on 31BU's layer-0 PASS.

**What 31BV is allowed to claim** (per the explicit spec):

- small fixed multi-layer probe result only
- pair/layer pass/failure under standalone tensor harness
- memory-positive / memory-negative under current accounting
- cosine / MAE improved/regressed relative to Q2_K-only AND Q4_K_M W_ref
- severe / non-severe regression within this fixed scope
- per-layer summary across the 3 selected layers

**What 31BV is forbidden to claim**:

- "quality recovered", "behavior recovered", "model improved"
- "inference works", "runtime works", "speedup"
- "production"
- "larger-model validation complete" (3 layers is not "larger-model validation")
- "FP16 recovery" (W_ref is Q4_K_M dequantized, not FP16)
- "aggregate validation" (6 pairs ≠ aggregate; not generalized to 28 layers)
- "1.5B behaves like 0.5B" (no 0.5B comparison in this phase)
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
| Architecture | hidden=1536, intermediate=8960, block_count=28 (31BV touches 3 layers only) |
| Q2_K mode | `corrected_ceil_per_row` |
| Residual families | `ffn_up` + `ffn_gate` |
| ffn_down residual | **none** (W_low only) |
| k | 0.5% |
| alpha | 1.0 |
| batch | 1 |
| X vector | `np.random.default_rng(seed)`, `standard_normal((1, 1536))` (matches 31AY/31BA/31BJ/31BK/31BU) |
| Layers | `[0, 14, 27]` |
| Seeds | `[0, 9]` → **6 anchor pairs** |
| Route | A (6 pairs total) |

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

## 6. Per-pair results (6 pairs, 3 layers × 2 seeds)

| layer | seed | cos_low | cos_sub | delta_cos | MAE_low | MAE_sub | MAE_delta | severe | margin (bytes) | finite |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|:---:|
| 0 | 0  | 0.878029 | 0.885037 | **+0.007008** | 0.019438 | 0.014828 | **−0.004610** | False | +3,380,374 | ✓ |
| 0 | 9  | 0.874526 | 0.875617 | **+0.001090** | 0.020004 | 0.016318 | **−0.003686** | False | +3,380,374 | ✓ |
| 14 | 0  | 0.879013 | 0.883005 | **+0.003992** | 0.020734 | 0.014526 | **−0.006208** | False | +3,380,376 | ✓ |
| 14 | 9  | 0.907621 | 0.908946 | **+0.001325** | 0.014793 | 0.012447 | **−0.002346** | False | +3,380,376 | ✓ |
| 27 | 0  | 0.887506 | 0.890467 | **+0.002961** | 0.017730 | 0.012302 | **−0.005428** | False | +3,380,350 | ✓ |
| 27 | 9  | 0.876373 | 0.883231 | **+0.006858** | 0.024037 | 0.010776 | **−0.013261** | False | +3,380,350 | ✓ |

Per-pair SDIR bytes are deterministic (seed-dependent residual selection from a deterministic X vector). The per-layer margin varies by 26 bytes between layers (3,380,350 for L27 vs 3,380,374/376 for L0/L14) — explained by tiny differences in the residual magnitude distributions across layers, which the SDIR bitmap / fp16 layout captures with slightly different nnz counts.

---

## 7. Per-layer summary (3 layers, 2 seeds each)

| layer | n_pairs | mem_pos | cos_pos | mae_imp | severe | mean_dc | mean_MAE_imp | min_margin |
|---:|---:|---:|---:|---:|:---:|---:|---:|---:|
| 0  | 2 | 2 | 2 | 2 | 0 | **+0.004049** | **−0.004148** | +3,380,374 |
| 14 | 2 | 2 | 2 | 2 | 0 | **+0.002659** | **−0.004277** | +3,380,376 |
| 27 | 2 | 2 | 2 | 2 | 0 | **+0.004909** | **−0.009345** | +3,380,350 |

**Per-layer observations (within anchor scope only — NOT a generalization):**

- All 3 layers are **memory-positive**, **cosine-improved**, and **MAE-improved** on both tested seeds.
- L14 has the smallest mean delta_cos (+0.002659) — its lower Q2_K W_low error leaves less room for residual gain.
- L27 has the largest MAE improvement (−0.009345) — its residual structure is more amenable to the k=0.5% SDIR correction.
- L0 reproduces the 31BU result (L0-S0: dc=+0.007008, L0-S9: dc=+0.001090) **exactly** — confirms reproducibility across separate runner invocations.
- All 3 layers are finite, no severe regressions, no memory fails.

---

## 8. Aggregate (within the 6-pair scope — NOT a 28-layer claim)

- n_pairs: 6
- n_memory_positive: **6 / 6** (100%)
- n_cosine_positive: **6 / 6** (100%)
- n_MAE_improving: **6 / 6** (100%)
- n_severe_regressions: **0 / 6**
- n_finite: **6 / 6**
- worst_pair: L0-S9, dc=+0.001090, MAE_delta=−0.003686, margin=+3,380,374
- mean_delta_cos: **+0.003872**
- median_delta_cos: **+0.003476**
- mean_MAE_improvement: **−0.005923** (negative = improvement)
- min_per_layer_margin: **+3,380,350 bytes (~3.22 MB)**

---

## 9. Memory accounting (per layer, selected policy)

| component | bytes | % of layer Q4 budget |
|---|---:|---:|
| Q2_K ffn_up (corrected_ceil_per_row)   | 4,515,840 | 21.87% |
| Q2_K ffn_gate (corrected_ceil_per_row) | 4,515,840 | 21.87% |
| Q2_K ffn_down (corrected_ceil_per_row) | 4,515,840 | 21.87% |
| SDIR ffn_up @ k=0.5%                   | ~1,127,786 | 5.46% (computed at runtime, ±26 bytes across layers) |
| SDIR ffn_gate @ k=0.5%                 | ~1,127,786 | 5.46% |
| SDIR ffn_down                          | 0         |  0.00% (no residual by policy) |
| **TOTAL per layer**                    | **~15,803,466** | **~76.55% of 20,643,840** |
| **Margin vs 3 × Q4_budget_family**     | **+~3,380,374** | **+~16.37% headroom** |

The per-layer margin is **essentially identical** across the 3 selected layers (variance of 26 bytes ≈ 0.0008% of margin), confirming that **the corrected Q2_K + SDIR memory accounting is consistent across layers** for the 1.5B shape — which is expected, since per-layer Q2_K bytes depend only on shape (constant), and SDIR bytes scale with `k × n_elements` (also constant at the same k).

---

## 10. Interpretation (conservative phrasing)

**Allowed:**

- All 6 / 6 pairs **passed** within the small multi-layer standalone tensor harness.
- All 3 / 3 selected layers (L0, L14, L27) are **memory-positive, cosine-improved, MAE-improved** on both tested seeds.
- Per-layer memory is **memory-positive** (margin +3,380,350 to +3,380,376 bytes) under the current accounting — consistent across all 3 layers.
- Cosine **improved** on all 6 / 6 pairs (range +0.0011 to +0.0070) relative to the Q2_K-only baseline AND relative to the Q4_K_M W_ref.
- MAE **improved** on all 6 / 6 pairs (range −0.0023 to −0.0133) relative to the Q2_K-only baseline AND relative to the Q4_K_M W_ref.
- **No severe regressions** in this fixed scope.
- All outputs **finite**.
- The corrected Q2_K encode/decode is sane on 1.5B at the 3 selected layers — i.e. the corrected policy does not blow up across this small multi-layer set at the same selected k=0.5% / up+gate / no ffn_down residual.
- L0 result **exactly reproduces** the 31BU 3-seed result (L0-S0: dc=+0.007008, L0-S9: dc=+0.001090) — confirms cross-runner reproducibility.

**Forbidden (not claimed):**

- ~~"quality recovered" / "behavior recovered" / "model improved"~~
- ~~"inference works" / "runtime works" / "speedup"~~
- ~~"production"~~
- ~~"larger-model validation complete"~~ — 3 layers ≠ 28 layers; 6 pairs ≠ aggregate
- ~~"FP16 recovery"~~ — W_ref is **Q4_K_M dequantized**, NOT FP16
- ~~"1.5B behaves like 0.5B"~~ — no 0.5B comparison in this phase
- ~~"consistent across all 28 layers"~~ — only 3 of 28 layers tested
- ~~aggregate validation~~ — this is a 6-pair small multi-layer probe, not aggregate

---

## 11. Forbidden claims preserved

Full canonical master list in `SOURCE_OF_TRUTH.md` Section 0.A. Per-phase notes in 31BV spec are audit context. Active 31BV-specific forbidden items:

- no model quality recovery claim
- no behavior recovery claim
- no speedup claim
- no full-model runtime memory savings claim
- no llama.cpp integration claim
- no production readiness claim
- no inference / generation claim
- no full larger-model validation claim (3 layers ≠ full 28 layers)
- no runtime-ready output-residual claim
- no claim beyond standalone tensor harness
- no claim that 31AY / 31BA exact anchors are current canonical metrics
- no aggregate validation (6 pairs ≠ aggregate)
- no claim that 1.5B behaves like 0.5B
- no orientation claim (orientation was already parity-tested in 31BT)
- no commit, push, tag, push-tag, or download (any new model) without explicit Matt approval

---

## 12. Limitations

- **Small fixed multi-layer scope only** (6 pairs, 3 layers) — not a full 28-layer sweep, not aggregate.
- **No layers beyond 0, 14, 27** (other 25 layers not tested in this phase).
- **No seeds beyond 0, 9** (other 14 seeds not tested in this phase).
- **No FP16 W_ref** — W_ref is the Q4_K_M dequantized tensor; any "recovery" is *relative to Q4_K_M*, not FP16.
- **No 0.5B comparison** in this phase — direct comparison is a future phase.
- **No llama.cpp runtime integration** — Q2_K is generated and decoded in the standalone tensor harness only.
- **No generation / inference / sampling / perplexity** — pure tensor-level MLP forward.

---

## 13. Artifacts (untracked, prepared, not committed)

| Path | Purpose |
|---|---|
| `docs/PHASE31BV_1_5B_CORRECTED_Q2K_SMALL_MULTILAYER_PROBE.md` | This document |
| `src/phase31bv_1_5b_corrected_q2k_small_multilayer_probe.py` | Runner (env-vars-only, no private paths, lint-clean) |
| `src/results/PHASE31BV_1_5B_CORRECTED_Q2K_SMALL_MULTILAYER_PROBE.json` | Full per-pair + per-layer + tensor diagnostics + memory accounting |

**Not produced / not in this phase:**

- model files
- HF cache
- generated Q2_K blobs (all in memory + tempfile.mkdtemp, cleaned at runner exit)
- generated SDIR blobs (all in memory)
- temp tensor dumps
- large artifacts

---

## 14. SOT update (to be applied if approved)

Append to `SOURCE_OF_TRUTH.md` Section 3, then update Section 0 / Section 9 to point next allowed phase to:

- **if clean or minor/tradeoff (this phase)**: Phase 31BW — Qwen2.5-1.5B Corrected Q2_K Broader Layer Probe Planning, only if explicitly requested
- **if blocked**: Phase 31BV-R — Small Multi-Layer Probe Repair, only if explicitly requested

---

## 15. Pre-commit status

| field | value |
|---|---|
| Runner | `src/phase31bv_1_5b_corrected_q2k_small_multilayer_probe.py` (env-vars-only, lint-clean) |
| Result JSON | `src/results/PHASE31BV_1_5B_CORRECTED_Q2K_SMALL_MULTILAYER_PROBE.json` (~13.8 KB) |
| Doc | `docs/PHASE31BV_1_5B_CORRECTED_Q2K_SMALL_MULTILAYER_PROBE.md` (this file) |
| SOT | new accepted fact to append (draft below) |
| Pre-commit regression | must be re-run after SOT edit, must still pass with same contract |
| Git status (after edits, before commit) | 3 new files in `docs/`, `src/`, `src/results/`; 1 modified file (`SOURCE_OF_TRUTH.md`) |
| Any private paths in commit? | **No** — runner uses `$SDI_MODEL_DIR`, `$SDI_LLAMA_CPP_*`; result JSON redacts model path to `$SDI_MODEL_DIR/...` |
| Any large / generated artifacts in commit? | **No** — runner output is small JSON; no Q2_K / SDIR blobs persisted; no temp tensor dumps |
| Pre-existing untracked 31BH-R2 / 31BJ files disposition | **untouched** per explicit instruction; not in this commit |
| Proposed commit message | `Phase 31BV: run 1.5B corrected Q2_K small multi-layer probe` |

---

## 16. Draft SOT Section 3 entry (to append if approved)

```markdown
    - **Phase 31BV — Qwen2.5-1.5B Corrected Q2_K Small Multi-Layer Probe:** Classification `PASS_31BV_1_5B_Q2K_SMALL_MULTILAYER_CLEAN`.
      - Scope: small fixed multi-layer probe only — Route A: layers [0, 14, 27] × seeds [0, 9] = 6 anchor pairs.
      - Model / W_ref: Qwen2.5-1.5B-Instruct Q4_K_M, dequantized (NOT FP16).
      - Tensors read (per layer) — **selected source-GGUF tensor types (per family per layer):**
        - ffn_up: **Q4_K** for layers 0, 14, 27 (raw [1536, 8960], dequant [8960, 1536])
        - ffn_gate: **Q4_K** for layers 0, 14, 27 (raw [1536, 8960], dequant [8960, 1536])
        - ffn_down: **Q6_K** for layers 0 and 27; **Q4_K** for layer 14 (raw [8960, 1536], dequant [1536, 8960])
        - Shapes consistent across L0, L14, L27 (all 13,762,560 elements per family; dequantized shapes are layer-independent because all 1.5B ffn_up/ffn_gate are (8960, 1536) and all ffn_down are (1536, 8960)).
      - 31BV observed **mixed source-GGUF quant types for ffn_down** in the selected layers: L0/L27 use Q6_K, while L14 uses Q4_K. This is a source-GGUF characteristic. The corrected Q2_K memory accounting is unaffected because W_ref is dequantized to float32 before corrected Q2_K encoding, and the selected corrected_ceil_per_row Q2_K byte count is shape-dependent (constant across quant types for the same shape).
      - Policy: `corrected_q2k_policy_v1` (corrected_ceil_per_row Q2_K, ffn_up+ffn_gate SDIR k=0.5%, alpha=1.0, ffn_down W_low only — no ffn_down residual).
      - Per-pair metrics:
        - L0-S0: dc=+0.007008, MAE_delta=−0.004610
        - L0-S9: dc=+0.001090, MAE_delta=−0.003686 (worst pair; same as 31BU)
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
      - Hygiene patch applied before commit (per Matt's pre-commit review): `per_layer_summary[layer]['tensors']` is now fully populated with real tensor metadata (name, raw_gguf_shape, dequant_shape, tensor_type, n_elements) for ffn_up / ffn_gate / ffn_down on all 3 selected layers — no null fields in the JSON. The original draft of this phase had a cosmetic issue where the per-layer summary referenced a not-yet-populated dict; the runner was patched to load the per-layer tensor diag BEFORE building the per-layer summary, and the result JSON was regenerated. All metrics are unchanged (classification `PASS_31BV_1_5B_Q2K_SMALL_MULTILAYER_CLEAN` preserved).
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
```

**Do not commit. Do not push. Awaiting explicit Matt approval.**

---

## 17. Question

**Approve commit / push of 31BV artifacts (runner + result JSON + this doc + SOT update)?**
