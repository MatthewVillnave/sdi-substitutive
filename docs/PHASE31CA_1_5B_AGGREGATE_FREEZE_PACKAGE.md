# Phase 31CA — Qwen2.5-1.5B Corrected Q2_K Aggregate Freeze / Package Update

> **Freeze + package + provenance + documentation phase.** No new validation. No generation. No inference. No llama.cpp runtime integration. No tag was created. No commit or push occurred.

---

## 1. Purpose

Freeze the 31BZ 1.5B full-layer aggregate result as a clean checkpoint, and update the policy/provenance documentation so future agents know exactly what is proven and what is still forbidden. This is the final rung of the 31BU → 31BZ step ladder: a documentation + handoff phase, not a scientific phase.

**Checkpoint target commit:** `f7f2a91d1b904f8f156d6c89584ec0d32229c23e` (the 31BZ commit).

**Proposed tag (NOT created in 31CA):** `phase31ca-1_5b-corrected-q2k-aggregate-checkpoint`

**Classification:** `PASS_31CA_1_5B_AGGREGATE_FREEZE_PACKAGE_READY`

---

## 2. Frozen 31BZ result

| field | value |
|---|---|
| phase | 31BZ |
| title | Qwen2.5-1.5B Corrected Q2_K Full-Layer Two-Seed Aggregate |
| classification | `PASS_31BZ_1_5B_Q2K_FULL_LAYER_TWO_SEED_AGGREGATE_CLEAN` |
| scope | standalone tensor-harness aggregate only |
| model | Qwen2.5-1.5B-Instruct Q4_K_M |
| W_ref | Q4_K_M dequantized (NOT FP16) |
| layers | 0–27 (all 28 = 100% layer coverage) |
| seeds | {0, 9} |
| total pairs | **56** |
| n_memory_positive | **56 / 56 (100%)** |
| n_cosine_positive | **56 / 56 (100%)** |
| n_MAE_improving | **56 / 56 (100%)** |
| n_severe_regressions | **0** |
| n_finite | **56 / 56 (100%)** |
| mean_delta_cos | **+0.004758** |
| median_delta_cos | +0.004501 |
| mean_MAE_improvement | **−0.012865** |
| min_per_layer_margin | +3,380,312 bytes |
| max_per_layer_margin | +3,380,376 bytes |
| per-layer margin variance | 64 bytes across all 56 pairs |
| worst pair | **L26-S9**, Δcos=+0.000390, MAE Δ=−0.029222 |
| n_layers PASS independently | **28 / 28** |
| wall clock (informational) | 481.5 sec / 8.0 min |

---

## 3. Cross-runner / cross-phase reproducibility (frozen as a known fact)

| layer | seed | 31BU | 31BV | 31BX | 31BZ | bit-identical? |
|---:|---:|---:|---:|---:|---:|:---:|
| 0  | 0 | dc=+0.007008 | dc=+0.007008 | dc=+0.007008 | dc=+0.007008 | ✓ |
| 0  | 9 | dc=+0.001090 | dc=+0.001090 | dc=+0.001090 | dc=+0.001090 | ✓ |
| 4  | 0 | — | — | dc=+0.004882 | dc=+0.004882 | ✓ |
| 4  | 9 | — | — | dc=+0.005096 | dc=+0.005096 | ✓ |
| 14 | 0 | — | dc=+0.003992 | — | dc=+0.003992 | ✓ |
| 14 | 9 | — | dc=+0.001325 | — | dc=+0.001325 | ✓ |
| 27 | 0 | — | dc=+0.002961 | dc=+0.002961 | dc=+0.002961 | ✓ |
| 27 | 9 | — | dc=+0.006858 | dc=+0.006858 | dc=+0.006858 | ✓ |

**8 overlapping layer/seed pairs are bit-identical to 6+ decimal places across 4 independent runners** (31BU, 31BV, 31BX, 31BZ). The corrected Q2_K + SDIR + corrected_ceil_per_row pipeline is **reproducible, not just approximately so** — the harness is deterministic and the policy is stable.

---

## 4. Selected policy (UNCHANGED — frozen at `corrected_q2k_policy_v1`)

| field | value |
|---|---|
| policy package | `corrected_q2k_policy_v1` |
| Q2_K mode | `corrected_ceil_per_row` |
| residual families | `ffn_up` + `ffn_gate` |
| residual k | 0.5% |
| alpha | 1.0 |
| ffn_down residual | **none** (W_low only) |
| W_ref | Qwen2.5-1.5B-Instruct Q4_K_M, dequantized (NOT FP16) |
| hidden | 1536 |
| intermediate | 8960 |
| batch | 1 |
| RNG | `np.random.default_rng(seed).standard_normal((1, 1536))` |
| MLP formula (31BT canonical) | `Y = (silu(X @ W_gate.T) * (X @ W_up.T)) @ W_down.T` |

**No policy parameters were changed in 31CA.** The 1.5B aggregate is added as a second evidence tier under the same policy, not as a new policy. Policy version is NOT bumped.

---

## 5. Model metadata summary

| field | value |
|---|---|
| model | Qwen2.5-1.5B-Instruct |
| source quantization | Q4_K_M (downloaded from official Qwen2.5 1.5B GGUF build) |
| model path | `$SDI_MODEL_DIR/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf` (outside repo, not committed) |
| model size | 1.1 GB |
| n FFN layers | 28 |
| hidden | 1536 |
| intermediate | 8960 |
| FFN shape per family | up: (8960, 1536) · gate: (8960, 1536) · down: (1536, 8960) |
| FFN per-layer floats | 41,287,680 |
| Q4 budget per family | 6,881,280 bytes |
| Q4 budget per layer | 20,643,840 bytes |

---

## 6. W_ref source and limitations (frozen caveat)

W_ref is the **dequantized Q4_K_M** tensor, not the original FP32 reference. The harness cannot claim parity with an FP32 reference because the FP32 reference is not available in the source GGUF. The `cos_sub vs cos_low` and `MAE_sub vs MAE_low` metrics compare against `W_ref = Q4_K_M` — the closest available approximation of the model's actual weights. The harness measures **how much the corrected Q2_K + SDIR improves over naive Q2_K**, with W_ref = Q4_K_M as the reference.

This is a **calibrated comparison**, not a ground-truth comparison. It is sufficient to claim that the policy is memory-positive and quality-improving (in the tensor-harness sense) relative to the Q4_K_M baseline. It is **not** sufficient to claim parity with the original FP32 model, because FP32 is not available.

---

## 7. Exact validation scope

| field | value |
|---|---|
| scope | standalone tensor harness only |
| harness | pure NumPy + ctypes against llama.cpp `libggml-base.so` (Q2_K encode/dequantize) + `phase31x_manifest_runtime` (SDIR encode/decode) + `np.dot` (matmul) |
| n_pairs | 56 |
| layer coverage | 100% (28 of 28 layers) |
| seed coverage | {0, 9} |
| activation distribution | `np.random.default_rng(seed).standard_normal((1, 1536))` — zero-mean unit-variance Gaussian on a single token |
| is NOT | generation · inference · sampling · llama.cpp runtime integration · production runtime · real-activation transfer |

---

## 8. Metrics summary (frozen)

| metric | value |
|---|---:|
| n_memory_positive | 56/56 (100%) |
| n_cosine_positive | 56/56 (100%) |
| n_MAE_improving | 56/56 (100%) |
| n_severe_regressions | 0 |
| n_finite | 56/56 (100%) |
| mean_delta_cos | +0.004758 |
| median_delta_cos | +0.004501 |
| mean_MAE_improvement | −0.012865 |
| min_per_layer_margin_bytes | +3,380,312 |
| max_per_layer_margin_bytes | +3,380,376 |
| worst pair | L26-S9, Δcos=+0.000390 (still strongly positive) |
| best per-layer mean Δcos | L20, +0.008768 |
| best per-layer mean MAE improvement | L3, −0.050887 |

---

## 9. Memory summary (frozen)

| field | value |
|---|---:|
| per-layer total bytes (typical) | 17,263,466 |
| per-layer total % of Q4 budget (typical) | 83.6% |
| per-layer margin bytes (typical) | +3,380,360 |
| per-layer margin % of Q4 budget (typical) | +16.4% headroom |
| per-layer margin variance (across 56 pairs) | 64 bytes |

**Structural note on the 64-byte variance:** the variance comes from the SDIR residual size scaling (the number of stored correction pairs depends on the row distribution of the residual, which depends on the per-row scale of the weight). The variance is structural and expected; it does not threaten the per-layer memory-positive claim.

---

## 10. Tensor-type observations (frozen)

| family | observation across 28 layers |
|---|---|
| ffn_up | uniformly **Q4_K** (28/28) |
| ffn_gate | uniformly **Q4_K** (28/28) |
| ffn_down | mixed: **14/28 Q6_K** (L0, L1, L5, L6, L7, L8, L9, L10, L13, L16, L19, L21, L24, L27) + **14/28 Q4_K** (L2, L3, L4, L11, L12, L14, L15, L17, L18, L20, L22, L23, L25, L26) — 50/50 split |

**Interpretation:** the Q6_K / Q4_K pattern for ffn_down is a source-GGUF characteristic of the official Qwen2.5-1.5B-Instruct Q4_K_M build (the model was originally quantized with a per-layer sensitivity-aware scheme). The corrected Q2_K memory accounting is **unaffected** because:

1. W_ref is dequantized to float32 before corrected Q2_K encoding (so the source quant type is irrelevant once dequantized).
2. The corrected_ceil_per_row Q2_K byte count is shape-dependent and constant for a given (d_out, d_in) regardless of the source quant type.

---

## 11. Wall-clock observation (frozen as informational only)

| field | value |
|---|---:|
| total wall clock (56 pairs) | **481.5 sec = 8.0 min** |
| mean per-pair wall clock | **~8.6 sec/pair** |
| 31BY planning estimate (working / upper) | 25 min / 35 min |
| actual vs estimate | **~3× faster than the planning lower bound** |

**This is informational only.** It is NOT a speedup, latency, throughput, or production-runtime claim. It is recorded as a planning-estimate-vs-actual check, useful for sizing future phases (3B, 7B, etc.) if those are ever explicitly approved.

---

## 12. Allowed claims (frozen)

The following claims are supported by the 31BZ result and may be used in future documentation:

> **A1.** Corrected Q2_K policy (`corrected_q2k_policy_v1`, corrected_ceil_per_row Q2_K, ffn_up + ffn_gate SDIR k=0.5%, alpha=1.0, no ffn_down residual) is memory-positive and quality-improving on Qwen2.5-1.5B across the **full 28-layer set [0..27] × seeds [0, 9]** in a standalone tensor harness (56/56 pairs memory-positive, cosine-improved, MAE-improved, 0 severe, all finite).

> **A2.** The policy is **reproducible across 4 independent runners** (31BU, 31BV, 31BX, 31BZ) at the bit level (6+ decimal places) on all 8 overlapping layer/seed pairs.

> **A3.** Phrased relative to Q4_K_M W_ref (NOT FP16).

---

## 13. Forbidden claims (frozen)

The following claims are **not supported** by the 31BZ result and remain forbidden:

- no model quality recovery claim
- no behavior recovery claim
- no speedup claim
- no full-model runtime memory savings claim
- no llama.cpp integration claim
- no production readiness claim
- no inference / generation / sampling claim
- no runtime-ready output-residual claim
- no FP16 recovery claim
- no 0.5B-vs-1.5B generalization claim
- no "1.5B behaves like 0.5B" claim
- no broader model-family claim (3B / 7B / 14B / 32B / 72B / 110B+ Qwen2.5, Llama, Mistral, etc. are NOT tested)
- no real-activation-transfer claim (the harness uses `np.random.default_rng` `standard_normal`; real activations are not tested)
- no larger-model claim
- no claim that 31AY / 31BA exact anchors are current canonical metrics
- no claim beyond standalone tensor harness

---

## 14. Valid-as-long-as clauses (frozen)

The 31BZ result remains valid as long as **all** of the following hold:

1. the downloaded 1.5B Q4_K_M file at `$SDI_MODEL_DIR/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf` remains unmodified
2. the probe inputs remain tiny deterministic (`np.random.default_rng(seed)`, batch 1) — not real activations
3. the policy parameters match `corrected_q2k_policy_v1` (Q2_K mode, residual families, k, alpha, ffn_down residual)
4. the result is interpreted as standalone tensor-harness aggregate (56 pairs, 28 layers × 2 seeds), NOT generation / inference / llama.cpp runtime integration, NOT a full larger-model claim, NOT a 0.5B comparison, NOT a broader-family claim
5. the canonical orientation convention (SOT Section 7) remains unchanged
6. no later phase invalidates or supersedes this aggregate result
7. `gguf.dequantize()` continues to return tensors in canonical (d_out, d_in) layout (re-verify if upstream `gguf` package changes)
8. the wall-clock observation (481.5 sec / 8.0 min) is treated as informational only, not as a speedup, latency, throughput, or production-runtime claim

---

## 15. Policy package updates

The 1.5B aggregate is added as a **second evidence tier** under the same policy. **Policy parameters are UNCHANGED. Policy version is NOT bumped.**

| file | change |
|---|---|
| `docs/CORRECTED_Q2K_POLICY_PACKAGE.md` | PATCHED — adds 1.5B (31BZ) as a second evidence tier alongside 0.5B (31BN); updates "Frozen Validation Summary" to show both tiers; updates "Supported Families" to clarify cross-tier consistency; updates "Next Recommended Phase" to 31CB. **No parameter changes.** |
| `src/results/CORRECTED_Q2K_POLICY_PACKAGE.json` | PATCHED — adds `validation_evidence_tiers` field with [0.5B_31BN, 1.5B_31BZ/31CA]; adds 1.5B `aggregate_metrics` as a sibling of 0.5B `aggregate_metrics`; updates `next_recommended_phase` to 31CB. **No parameter changes.** |
| `src/corrected_q2k_policy.py` | UNCHANGED — the constants helper already reflects `corrected_q2k_policy_v1`; the 1.5B evidence tier is documentation, not code. |
| `src/q2k_backend.py` | UNCHANGED — the Q2_K encode/dequantize backend was already general over the dequantized W_ref. |
| `src/phase31x_manifest_runtime.py` | UNCHANGED — the SDIR encode/decode was already general over the family choice. |

**Policy version remains `corrected_q2k_policy_v1`.**

---

## 16. README updates (conservative, deferred style)

The README is a high-level public orientation page that defers to `SOURCE_OF_TRUTH.md`. The README was last updated in the **README-01** documentation maintenance interlude (committed at `49318089`).

As of 31CA, the README is **out of date** in two specific places:

| line | stale claim | 31CA patch |
|---|---|---|
| 22 | "next phase is **31BU**" | "next phase is **31CB**" (31CB recommended in §17 of this doc) |
| 29 | "**No 1.5B tensor validation yet.**" | "1.5B full-layer aggregate **passed** via 31BZ (56/56); see SOT §0 and §3." |
| 30 | "Current scientific next phase: **31BU**" | "Current scientific next phase: **31CB**" |
| 50 | "No broad larger-model claim — the 1.5B model has only been downloaded and metadata/orientation-checked." | "1.5B is a frozen evidence tier under `corrected_q2k_policy_v1` via 31BZ; broader model-family claims remain forbidden." |

These patches are **conservative, in-place, and additive**. The README's structure and tone are preserved. No new sections are added. The README continues to defer to `SOURCE_OF_TRUTH.md` for the full audit trail.

---

## 17. Next recommended phase

The 31CA spec lists three options for the next allowed phase. **31CA recommends Option C: 31CB Stale File Provenance Cleanup.**

| option | name | rationale |
|---|---|---|
| A | 31CB Real-Activation Capture Planning | would plan how to capture real activations from a 1.5B model run; prerequisite for any real-activation claim. |
| B | 31CB Runtime Artifact Format / Loader Planning | would plan the on-disk artifact format and a Python loader; prerequisite for any runtime-integration claim. |
| **C** | **31CB Stale File Provenance Cleanup** | **the repo still has 5 stale untracked 31BH-R2 / 31BJ files carried through 31BU → 31BZ without resolution. With 1.5B aggregate now frozen, these are the only outstanding provenance debt. Cleanup is the lowest-risk next step and removes noise before a new scientific lane is opened.** |

**31CA's recommendation: Option C (Stale File Provenance Cleanup).** Set as the next allowed phase in `SOURCE_OF_TRUTH.md` Section 9.

---

## 18. Proposed tag (NOT created in 31CA)

**Proposed tag name:** `phase31ca-1_5b-corrected-q2k-aggregate-checkpoint`

**Proposed target commit:** `f7f2a91d1b904f8f156d6c89584ec0d32229c23e`

**Proposed tag message:**

> Phase 31CA: freeze 1.5B corrected Q2_K aggregate checkpoint (56/56 pairs PASS)

**Proposed tag command (NOT executed in 31CA):**

```
git tag -a phase31ca-1_5b-corrected-q2k-aggregate-checkpoint f7f2a91d1b904f8f156d6c89584ec0d32229c23e -m 'Phase 31CA: freeze 1.5B corrected Q2_K aggregate checkpoint (56/56 pairs PASS)'
```

**Tag is NOT created in 31CA.** Tag creation is a separate step from commit. Operator must approve tag creation explicitly.

---

## 19. What 31CA did NOT do

- ❌ Did not run new validation
- ❌ Did not run generation / inference / sampling
- ❌ Did not run llama.cpp runtime integration
- ❌ Did not run any model
- ❌ Did not bump the policy version
- ❌ Did not change the policy parameters
- ❌ Did not create a tag
- ❌ Did not push a tag
- ❌ Did not commit (waiting for explicit approval)
- ❌ Did not push (waiting for explicit approval)
- ❌ Did not move, delete, stage, or commit the stale 31BH-R2 / 31BJ files (5 files remain untouched per the 31CA spec)

---

## 20. Classification

**`PASS_31CA_1_5B_AGGREGATE_FREEZE_PACKAGE_READY`** — 31BZ accepted fact preserved as a frozen checkpoint; 31CA freeze docs and JSON created; policy package updated with 1.5B evidence tier (without bumping policy version or changing parameters); README patched conservatively to reflect 31BZ PASS; SOT updated; regression passes; no commit/push/tag without explicit Matt approval.
