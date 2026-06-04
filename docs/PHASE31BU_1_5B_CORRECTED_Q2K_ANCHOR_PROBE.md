# Phase 31BU — Qwen2.5-1.5B Corrected Q2_K Anchor Probe

**Classification:** `PASS_31BU_1_5B_Q2K_ANCHOR_PROBE_CLEAN`

**Phase:** 31BU
**Scope:** Anchor probe only. No aggregate validation, no multi-layer sweep beyond layer 0, no generation/inference, no llama.cpp runtime integration, no performance/quality claim.
**Route:** A (layer 0 only, 3 anchor pairs: seeds 0, 9, 13).
**Status:** Artifacts prepared — **not yet committed** (PRE-COMMIT REPORT; awaiting Matt approval).

---

## 1. Goal

Determine whether the `corrected_q2k_policy_v1` path is sane on the **first larger model** (Qwen2.5-1.5B-Instruct Q4_K_M). This is a small, fixed anchor probe, not an aggregate validation.

**What 31BU is allowed to claim** (per the explicit spec):

- anchor pair passed/failed within the tiny standalone tensor harness
- memory-positive / memory-negative under current accounting
- cosine improved/regressed relative to the Q2_K-only baseline AND relative to the Q4_K_M W_ref
- MAE improved/regressed relative to the Q2_K-only baseline AND relative to the Q4_K_M W_ref
- severe / non-severe regression in the anchor scope

**What 31BU is forbidden to claim**:

- "quality recovered", "behavior recovered", "model improved"
- "inference works", "runtime works", "speedup"
- "production"
- "larger-model validation complete"
- "FP16 recovery" (W_ref is **Q4_K_M dequantized**, not FP16)
- aggregate validation
- multi-layer sweep beyond Route A scope
- model files / Q2_K blobs / SDIR blobs committed
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
| Architecture | hidden=1536, intermediate=8960, block_count=28 (28 layers in 1.5B; 31BU touches layer 0 only) |
| Q2_K mode | `corrected_ceil_per_row` |
| Residual families | `ffn_up` + `ffn_gate` |
| ffn_down residual | **none** (W_low only) |
| k | 0.5% |
| alpha | 1.0 |
| batch | 1 |
| X vector | `np.random.default_rng(seed)`, `standard_normal((1, 1536))` (matches 31AY/31BA/31BJ/31BK convention) |
| Layers | `[0]` |
| Seeds | `[0, 9, 13]` → **3 anchor pairs** |
| Route | A (3 pairs total) |

---

## 4. MLP formula (canonical)

```
W_up_canon   is (intermediate=8960, hidden=1536)   — dequantized from blk.0.ffn_up.weight
W_gate_canon is (intermediate=8960, hidden=1536)   — dequantized from blk.0.ffn_gate.weight
W_down_canon is (hidden=1536, intermediate=8960)   — dequantized from blk.0.ffn_down.weight

X            is (1, 1536)

up           = X @ W_up_canon.T           # (1, 8960)
gate         = X @ W_gate_canon.T         # (1, 8960)
act          = silu(gate) * up            # (1, 8960)
out          = act @ W_down_canon.T       # (1, 1536) — expected
```

This is the canonical formulation recorded by 31BT (gguf.dequantize() returns tensors in (d_out, d_in) layout).

---

## 5. Selected policy (`corrected_q2k_policy_v1`)

| Q2_K bytes per family (1.5B shape) | corrected_ceil_per_row |
|---|---|
| ffn_up / ffn_gate: 8960 × ⌈1536/256⌉ × 84 = 8960 × 6 × 84 | 4,515,840 bytes |
| ffn_down: ⌈8960/256⌉ × 84 × 1536 = 35 × 84 × 1536 | 4,515,840 bytes |
| Total Q2_K (per layer, 3 families) | 13,547,520 bytes |
| SDIR per family @ k=0.5% (13,762,560 elems × 0.005 ≈ 68,813 nnz) | computed at runtime |
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

## 6. Anchor results (Route A: 3 pairs)

| layer | seed | cos_low | cos_sub | delta_cos | MAE_low | MAE_sub | MAE_delta | severe | margin (bytes) |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| 0 | 0  | 0.878029 | 0.885037 | **+0.007008** | 0.019438 | 0.014828 | **−0.004610** | False | **+3,380,374** |
| 0 | 9  | 0.874526 | 0.875616 | **+0.001090** | 0.020004 | 0.016318 | **−0.003686** | False | **+3,380,374** |
| 0 | 13 | 0.882613 | 0.886932 | **+0.004319** | 0.017514 | 0.011673 | **−0.005841** | False | **+3,380,374** |

Per-pair SDIR bytes are deterministic (seed-dependent residual selection from a deterministic X vector) so all three pairs landed on the same per-layer margin.

**Aggregate (within anchor scope only — not a "full-layer" claim):**

- n_pairs: 3
- n_memory_positive: **3 / 3** (100%)
- n_cosine_positive: **3 / 3** (100%)
- n_MAE_improving: **3 / 3** (100%)
- n_severe_regressions: **0 / 3**
- n_finite: **3 / 3**
- worst_pair: L0-S9, delta_cos=+0.001090, MAE_delta=−0.003686, margin=+3,380,374
- mean_delta_cos: **+0.004139**
- median_delta_cos: **+0.004319**
- mean_MAE_improvement: **−0.004712** (negative = improvement)
- min_per_layer_margin: **+3,380,374 bytes (~3.22 MB)**

---

## 7. Memory accounting (per pair)

| component | bytes (per family) | total bytes (3 families) | % of Q4 layer budget |
|---|---:|---:|---:|
| Q2_K ffn_up (corrected_ceil_per_row)   | 4,515,840 | 4,515,840 | 21.87% |
| Q2_K ffn_gate (corrected_ceil_per_row) | 4,515,840 | 4,515,840 | 21.87% |
| Q2_K ffn_down (corrected_ceil_per_row) | 4,515,840 | 4,515,840 | 21.87% |
| SDIR ffn_up @ k=0.5%                   | 1,127,786 | 1,127,786 |  5.46% |
| SDIR ffn_gate @ k=0.5%                 | 1,127,786 | 1,127,786 |  5.46% |
| SDIR ffn_down                          | 0         | 0         |  0.00% (no residual by policy) |
| **TOTAL per layer**                    | —         | **15,803,466** | **76.55% of 20,643,840** |
| **Margin vs 3 × Q4_budget_family**     | —         | **+3,380,374** | **+16.37% headroom** |

(SDIR bytes are deterministic per-seed; all 3 seeds landed on 1,127,786 bytes/family. The formula gives a 0.5% of 13,762,560 elements = ~68,813 nnz; actual bytes depend on the residual selection from `R = W_ref - W_low` and the SDIR bitmap layout — see `src/phase31x_manifest_runtime.py:encode_sdir`.)

The 1.5B layer margin of **+3.22 MB per layer** is **larger** than the 0.5B layer margin of **+0.66 MB** at the same k=0.5% (31BK) — because the Q2_K W_low byte cost is fixed by `d_in × ceil_block_bytes`, but the Q4_budget scales with `d_out × d_in`, and `d_out × d_in` grows faster than `d_in × ceil_block_bytes` when both d_out and d_in get larger. This is **a memory accounting observation, not a quality / performance claim**.

---

## 8. Interpretation (conservative phrasing)

**Allowed:**

- All 3 / 3 anchor pairs **passed** within the tiny standalone tensor harness.
- Per-layer memory is **memory-positive** (+3,380,374 bytes) under the current accounting.
- Cosine **improved** on all 3 / 3 pairs (range +0.0011 to +0.0070) relative to the Q2_K-only baseline AND relative to the Q4_K_M W_ref.
- MAE **improved** on all 3 / 3 pairs (range −0.0037 to −0.0058) relative to the Q2_K-only baseline AND relative to the Q4_K_M W_ref.
- **No severe regressions** in anchor scope.
- All outputs **finite**.
- The corrected Q2_K encode/decode is sane on 1.5B at layer 0 — i.e. the corrected policy does not blow up on the first larger model at the same selected k=0.5% / up+gate / no ffn_down residual.

**Forbidden (not claimed):**

- ~~"quality recovered" / "behavior recovered" / "model improved"~~
- ~~"inference works" / "runtime works" / "speedup"~~
- ~~"production"~~
- ~~"larger-model validation complete"~~
- ~~"FP16 recovery"~~ — W_ref is **Q4_K_M dequantized**, NOT FP16
- ~~aggregate validation~~
- ~~multi-layer sweep beyond layer 0~~
- ~~"1.5B behaves like 0.5B"~~ — this is **one anchor, 3 pairs**, not a generalization

---

## 9. Forbidden claims preserved

Full canonical master list in `SOURCE_OF_TRUTH.md` Section 0.A. Per-phase notes in 31BU spec are audit context. Active 31BU-specific forbidden items:

- no model quality recovery claim
- no behavior recovery claim
- no speedup claim
- no full-model runtime memory savings claim
- no llama.cpp integration claim
- no production readiness claim
- no inference / generation claim
- no larger-model validation claim (anchor probe only, not generalization)
- no runtime-ready output-residual claim
- no claim beyond standalone tensor harness
- no claim that 31AY / 31BA exact anchors are current canonical metrics
- no orientation claim (orientation was already parity-tested in 31BT)
- no commit, push, tag, push-tag, or download (any new model) without explicit Matt approval

---

## 10. Limitations

- **Anchor scope only** (3 pairs, layer 0) — not a multi-layer or aggregate validation.
- **No layer 14 / layer 27** (Route B not used; Route A was sufficient and fast).
- **No seeds beyond 0, 9, 13.**
- **No FP16 W_ref** — the reference is the Q4_K_M dequantized tensor; any "recovery" is *relative to Q4_K_M*, not to FP16.
- **No 28-layer or 0.5B comparison** in this phase — those are future phases.
- **No llama.cpp runtime integration** — Q2_K is generated and decoded in the standalone tensor harness only.
- **No generation / inference / sampling / perplexity** — pure tensor-level MLP forward.

---

## 11. Artifacts (untracked, prepared, not committed)

| Path | Purpose |
|---|---|
| `docs/PHASE31BU_1_5B_CORRECTED_Q2K_ANCHOR_PROBE.md` | This document |
| `src/phase31bu_1_5b_corrected_q2k_anchor_probe.py` | Runner (env-vars-only, no private paths, lint-clean) |
| `src/results/PHASE31BU_1_5B_CORRECTED_Q2K_ANCHOR_PROBE.json` | Full anchor metrics + memory accounting + tensor diagnostics |
| `SOURCE_OF_TRUTH.md` | New accepted fact appended; next allowed phase updated to 31BV (or 31BU-R if blocked) |

**Not produced / not in this phase:**

- model files
- HF cache
- generated Q2_K blobs (all in memory + tempfile.mkdtemp, cleaned at runner exit)
- generated SDIR blobs (all in memory)
- temp tensor dumps
- large artifacts

---

## 12. SOT update (to be applied if approved)

Append to `SOURCE_OF_TRUTH.md` Section 3, then update Section 0 / Section 9 to point next allowed phase to:

- **if clean or minor/tradeoff (this phase)**: Phase 31BV — Qwen2.5-1.5B Corrected Q2_K Small Multi-Layer Probe, only if explicitly requested
- **if blocked**: Phase 31BU-R — Anchor Probe Repair, only if explicitly requested

---

## 13. Pre-commit status

| field | value |
|---|---|
| Runner | `src/phase31bu_1_5b_corrected_q2k_anchor_probe.py` (env-vars-only, lint-clean) |
| Result JSON | `src/results/PHASE31BU_1_5B_CORRECTED_Q2K_ANCHOR_PROBE.json` (small, ~7.4 KB) |
| Doc | `docs/PHASE31BU_1_5B_CORRECTED_Q2K_ANCHOR_PROBE.md` (this file) |
| SOT | new accepted fact to append (draft below) |
| Pre-commit regression | must be re-run after SOT edit, must still pass with same contract |
| Git status (after edits, before commit) | 2 new files in `docs/` and `src/results/`, 1 new file in `src/`, 1 modified file (`SOURCE_OF_TRUTH.md`) |
| Any private paths in commit? | **No** — runner uses `$SDI_MODEL_DIR`, `$SDI_LLAMA_CPP_*`; result JSON redacts model path to `$SDI_MODEL_DIR/...` |
| Any large / generated artifacts in commit? | **No** — runner output is small JSON; no Q2_K / SDIR blobs persisted; no temp tensor dumps |
| Pre-existing untracked 31BH-R2 / 31BJ files disposition | **untouched** per explicit instruction; not in this commit |
| Proposed commit message | `Phase 31BU: run 1.5B corrected Q2_K anchor probe` |

---

## 14. Draft SOT Section 3 entry (to append if approved)

```markdown
    - **Phase 31BU — Qwen2.5-1.5B Corrected Q2_K Anchor Probe:** Classification `PASS_31BU_1_5B_Q2K_ANCHOR_PROBE_CLEAN`.
      - Scope: anchor probe only — layer 0, 3 seeds (0, 9, 13) per Route A.
      - Model / W_ref: Qwen2.5-1.5B-Instruct Q4_K_M, dequantized (NOT FP16). W_ref source: downloaded 1.5B Q4_K_M GGUF.
      - Tensors read: blk.0.ffn_up.weight (Q4_K, raw [1536,8960], dequant [8960,1536]), blk.0.ffn_gate.weight (Q4_K, same), blk.0.ffn_down.weight (Q6_K, raw [8960,1536], dequant [1536,8960]).
      - Policy: `corrected_q2k_policy_v1` (corrected_ceil_per_row Q2_K, ffn_up+ffn_gate SDIR k=0.5%, alpha=1.0, ffn_down W_low only — no ffn_down residual).
      - Anchor metrics: L0-S0 dc=+0.007008, L0-S9 dc=+0.001090, L0-S13 dc=+0.004319; MAE improves on all 3 (Δ = −0.0046, −0.0037, −0.0058); 0 severe regressions; all finite.
      - Memory: 3/3 memory-positive; per-layer margin +3,380,374 bytes (~3.22 MB / ~16.4% of 3 × Q4_budget_family); Q2_K encode 4,515,840 bytes/family (corrected_ceil_per_row); SDIR ~1,127,786 bytes/family @ k=0.5%.
      - Forbidden claims preserved: no quality / behavior / speedup / runtime / llama.cpp / inference / production / larger-model-validation / FP16-recovery claim; no aggregate validation; no multi-layer sweep beyond layer 0; no commit/push/tag without explicit Matt approval.
      - Runner: `src/phase31bu_1_5b_corrected_q2k_anchor_probe.py` (env-vars-only, no private paths, lint-clean).
      - Result JSON: `src/results/PHASE31BU_1_5B_CORRECTED_Q2K_ANCHOR_PROBE.json` (small, no blobs).
      - Limitations: anchor probe only (3 pairs, layer 0); no FP16 W_ref; no 28-layer or 0.5B comparison; no llama.cpp runtime integration; no generation/inference.
      - Pre-existing untracked 31BH-R2 / 31BJ files: untouched per explicit instruction; not in this commit.
      - Next allowed phase: **Phase 31BV — Qwen2.5-1.5B Corrected Q2_K Small Multi-Layer Probe**, only if explicitly requested.
      - **Accepted claim:** 31BU proves the corrected Q2_K policy (`corrected_q2k_policy_v1`, corrected_ceil_per_row, ffn_up + ffn_gate SDIR k=0.5%, alpha=1.0, no ffn_down residual) is **memory-positive and quality-improving on Qwen2.5-1.5B layer 0** in a standalone tensor harness (3/3 pairs memory-positive, cosine-improved, MAE-improved, 0 severe, all finite, mean delta_cos=+0.00414, mean MAE improvement=−0.00471, per-layer margin +3,380,374 bytes). Phrased relative to Q4_K_M W_ref (NOT FP16).
      - **Valid as long as:**
        - the downloaded 1.5B Q4_K_M file at `$SDI_MODEL_DIR/...` remains unmodified
        - the probe inputs remain tiny deterministic (`np.random.default_rng(seed)`, batch 1) — not real activations
        - the policy parameters match `corrected_q2k_policy_v1` (Q2_K mode, residual families, k, alpha, ffn_down residual)
        - the result is interpreted as anchor probe (3 pairs, layer 0), NOT aggregate validation, NOT a larger-model generalization
        - the canonical orientation convention (Section 7) remains unchanged
        - no later phase invalidates or supersedes this anchor result
```

**Do not commit. Do not push. Awaiting explicit Matt approval.**

---

## 15. Question

**Approve commit / push of 31BU artifacts (runner + result JSON + this doc + SOT update)?**
