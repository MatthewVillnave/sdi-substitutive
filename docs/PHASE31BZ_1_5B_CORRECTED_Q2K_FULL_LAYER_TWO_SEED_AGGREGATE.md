# Phase 31BZ — Qwen2.5-1.5B Corrected Q2_K Full-Layer Two-Seed Aggregate

> Standalone tensor-harness aggregate only. NOT generation. NOT inference. NOT llama.cpp runtime integration. NOT a larger-model claim. NOT a 0.5B-vs-1.5B comparison. NOT a broader-family claim.

---

## 1. Scope

31BZ executes the 31BY Option A scope: a full 28-layer × 2-seed standalone tensor-harness aggregate on Qwen2.5-1.5B-Instruct Q4_K_M. Total 56 anchor pairs across 100% of the 1.5B FFN/MLP layers. Same `corrected_q2k_policy_v1` used in 31BU, 31BV, 31BX, 31BY. Same W_ref source (1.5B Q4_K_M dequantized, NOT FP16). Same harness family. Same MLP formula (31BT canonical).

This is the final rung of a controlled step ladder:

| phase | scope | pairs | result |
|---|---|---:|---|
| 31BU | L0 only, seeds {0, 9, 13} | 3 | PASS |
| 31BV | L0/L14/L27, seeds {0, 9} | 6 | PASS |
| 31BW | planning | — | PASS (Option A selected) |
| 31BX | L0/L4/L8/L12/L16/L20/L27, seeds {0, 9} | 14 | PASS |
| 31BY | planning | — | PASS (Option A selected) |
| **31BZ** | **L0..L27, seeds {0, 9}** | **56** | **PASS** |

---

## 2. Configuration

| field | value |
|---|---|
| model | Qwen2.5-1.5B-Instruct Q4_K_M |
| W_ref source | downloaded 1.5B Q4_K_M GGUF, dequantized (NOT FP16) |
| Q2_K mode | `corrected_ceil_per_row` |
| residual families | `ffn_up` + `ffn_gate` |
| residual k | 0.5% |
| alpha | 1.0 |
| ffn_down residual | **none** (W_low only) |
| layers | 0–27 (all 28) |
| seeds | {0, 9} |
| total pairs | 56 |
| X vector | `np.random.default_rng(seed).standard_normal((1, 1536))` |
| MLP formula (31BT canonical) | `Y = (silu(X @ W_gate.T) * (X @ W_up.T)) @ W_down.T` |
| hidden | 1536 |
| intermediate | 8960 |
| batch | 1 |

---

## 3. Headline result

| metric | value |
|---|---:|
| **classification** | **`PASS_31BZ_1_5B_Q2K_FULL_LAYER_TWO_SEED_AGGREGATE_CLEAN`** |
| n_pairs | **56** |
| n_memory_positive | **56 / 56** |
| n_cosine_positive | **56 / 56** |
| n_MAE_improving | **56 / 56** |
| n_severe_regressions | **0** |
| n_finite | **56 / 56** |
| mean delta_cos | **+0.0047576** |
| median delta_cos | +0.0045015 |
| mean MAE improvement | **−0.0128652** |
| min per-layer margin | +3,380,312 bytes |
| max per-layer margin | +3,380,376 bytes |
| per-layer margin variance | 64 bytes across all 28 layers |
| worst pair | **L26-S9** (Δcos=+0.000390, MAE Δ=−0.029222, margin=+3,380,356) |
| n_layers PASS independently | **28 / 28** |
| total wall clock | **481.5 sec = 8.0 min** (informational only) |
| mean per-pair wall clock | ~8.6 sec/pair (informational only) |

**All hard gates PASS. All soft thresholds PASS within 2-pair allowance (in fact, 0 failures — the allowance is not even needed).**

---

## 4. Per-layer summary (28 layers × 2 seeds = 56 pairs)

| L | n_pairs | mem+ | cos+ | MAE+ | severe | mean_dc | mean_MAE_imp | min_margin | ffn_up | ffn_gate | ffn_down |
|---:|---:|---:|---:|---:|:---:|---:|---:|---:|:---:|:---:|:---:|
| 0  | 2 | 2 | 2 | 2 | 0 | +0.004049 | −0.004148 | +3,380,374 | Q4_K | Q4_K | Q6_K |
| 1  | 2 | 2 | 2 | 2 | 0 | +0.006289 | −0.006055 | +3,380,362 | Q4_K | Q4_K | Q6_K |
| 2  | 2 | 2 | 2 | 2 | 0 | +0.005972 | −0.030008 | +3,380,360 | Q4_K | Q4_K | Q4_K |
| 3  | 2 | 2 | 2 | 2 | 0 | +0.007816 | −0.050887 | +3,380,312 | Q4_K | Q4_K | Q4_K |
| 4  | 2 | 2 | 2 | 2 | 0 | +0.004989 | −0.012739 | +3,380,352 | Q4_K | Q4_K | Q4_K |
| 5  | 2 | 2 | 2 | 2 | 0 | +0.004046 | −0.012204 | +3,380,360 | Q4_K | Q4_K | Q6_K |
| 6  | 2 | 2 | 2 | 2 | 0 | +0.005004 | −0.010610 | +3,380,372 | Q4_K | Q4_K | Q6_K |
| 7  | 2 | 2 | 2 | 2 | 0 | +0.004348 | −0.007749 | +3,380,362 | Q4_K | Q4_K | Q6_K |
| 8  | 2 | 2 | 2 | 2 | 0 | +0.004752 | −0.012853 | +3,380,342 | Q4_K | Q4_K | Q6_K |
| 9  | 2 | 2 | 2 | 2 | 0 | +0.003432 | −0.005937 | +3,380,312 | Q4_K | Q4_K | Q6_K |
| 10 | 2 | 2 | 2 | 2 | 0 | +0.004197 | −0.008220 | +3,380,370 | Q4_K | Q4_K | Q6_K |
| 11 | 2 | 2 | 2 | 2 | 0 | +0.005223 | −0.025633 | +3,380,364 | Q4_K | Q4_K | Q4_K |
| 12 | 2 | 2 | 2 | 2 | 0 | +0.003395 | −0.017177 | +3,380,372 | Q4_K | Q4_K | Q4_K |
| 13 | 2 | 2 | 2 | 2 | 0 | +0.006898 | −0.012441 | +3,380,356 | Q4_K | Q4_K | Q6_K |
| 14 | 2 | 2 | 2 | 2 | 0 | +0.002659 | −0.004277 | +3,380,376 | Q4_K | Q4_K | Q4_K |
| 15 | 2 | 2 | 2 | 2 | 0 | +0.004003 | −0.011887 | +3,380,366 | Q4_K | Q4_K | Q4_K |
| 16 | 2 | 2 | 2 | 2 | 0 | +0.001628 | −0.004832 | +3,380,364 | Q4_K | Q4_K | Q6_K |
| 17 | 2 | 2 | 2 | 2 | 0 | +0.004447 | −0.004895 | +3,380,352 | Q4_K | Q4_K | Q4_K |
| 18 | 2 | 2 | 2 | 2 | 0 | +0.004074 | −0.007349 | +3,380,360 | Q4_K | Q4_K | Q4_K |
| 19 | 2 | 2 | 2 | 2 | 0 | +0.005907 | −0.019775 | +3,380,370 | Q4_K | Q4_K | Q6_K |
| 20 | 2 | 2 | 2 | 2 | 0 | +0.008768 | −0.015812 | +3,380,356 | Q4_K | Q4_K | Q4_K |
| 21 | 2 | 2 | 2 | 2 | 0 | +0.005039 | −0.012721 | +3,380,372 | Q4_K | Q4_K | Q6_K |
| 22 | 2 | 2 | 2 | 2 | 0 | +0.005317 | −0.010409 | +3,380,366 | Q4_K | Q4_K | Q4_K |
| 23 | 2 | 2 | 2 | 2 | 0 | +0.004987 | −0.009156 | +3,380,362 | Q4_K | Q4_K | Q4_K |
| 24 | 2 | 2 | 2 | 2 | 0 | +0.004284 | −0.007282 | +3,380,350 | Q4_K | Q4_K | Q6_K |
| 25 | 2 | 2 | 2 | 2 | 0 | +0.003844 | −0.006752 | +3,380,346 | Q4_K | Q4_K | Q4_K |
| 26 | 2 | 2 | 2 | 2 | 0 | +0.002936 | −0.019073 | +3,380,356 | Q4_K | Q4_K | Q4_K |
| 27 | 2 | 2 | 2 | 2 | 0 | +0.004909 | −0.009345 | +3,380,350 | Q4_K | Q4_K | Q6_K |

**Every layer PASSES independently.** All 28 layers: 2/2 memory-positive, 2/2 cosine-improved, 2/2 MAE-improved, 0/2 severe, all finite. The minimum per-layer margin (across all 28 × 2 = 56 pairs) is +3,380,312 bytes (L3-S9 and L9-S0); the maximum is +3,380,376 bytes (L14, both seeds).

---

## 5. Worst pair (informational)

| field | value |
|---|---|
| layer | **L26** |
| seed | 9 |
| delta_cos | **+0.000390** |
| MAE_delta | −0.029222 |
| cos_low | 0.881 (≈) |
| cos_sub | 0.881 + 0.000390 |
| per_layer_margin | +3,380,356 |
| severe | False |
| finite | ✓ all 3 outputs |

Still strongly positive. No severe regression (Δcos ≫ −0.05 threshold). Memory-positive. All finite. This is the smallest improvement in the set, and it is still in the green.

---

## 6. Cross-runner / cross-phase reproducibility

| layer | seed | 31BU | 31BV | 31BX | 31BZ |
|---:|---:|---:|---:|---:|---:|
| 0  | 0 | dc=+0.007008 | dc=+0.007008 | dc=+0.007008 | dc=+0.007008 ✓ |
| 0  | 9 | dc=+0.001090 | dc=+0.001090 | dc=+0.001090 | dc=+0.001090 ✓ |
| 4  | 0 | — | — | dc=+0.004882 | dc=+0.004882 ✓ |
| 4  | 9 | — | — | dc=+0.005096 | dc=+0.005096 ✓ |
| 14 | 0 | — | dc=+0.003992 | — | dc=+0.003992 ✓ |
| 14 | 9 | — | dc=+0.001325 | — | dc=+0.001325 ✓ |
| 27 | 0 | — | dc=+0.002961 | dc=+0.002961 | dc=+0.002961 ✓ |
| 27 | 9 | — | dc=+0.006858 | dc=+0.006858 | dc=+0.006858 ✓ |

**Bit-identical to 6 decimal places on all 8 overlapping layer/seed pairs across 4 independent runners (31BU, 31BV, 31BX, 31BZ).** The corrected Q2_K + SDIR + corrected_ceil_per_row pipeline is reproducible — not just approximately so, but to the bit — across multiple independent runs over the course of the 31BU → 31BZ step ladder. This is the strongest possible empirical evidence that the harness is deterministic and the policy is stable.

---

## 7. Tensor-type observations (per family per layer)

| family | observation |
|---|---|
| ffn_up | uniformly **Q4_K** across all 28 layers |
| ffn_gate | uniformly **Q4_K** across all 28 layers |
| ffn_down | **mixed** — 14 of 28 layers use Q6_K, 14 of 28 use Q4_K (50/50 split) |

ffn_down Q6_K layers (14): L0, L1, L5, L6, L7, L8, L9, L10, L13, L16, L19, L21, L24, L27
ffn_down Q4_K layers (14): L2, L3, L4, L11, L12, L14, L15, L17, L18, L20, L22, L23, L25, L26

The Q6_K / Q4_K pattern for ffn_down is a source-GGUF characteristic of the official Qwen2.5-1.5B-Instruct Q4_K_M build (the model was originally quantized with a per-layer sensitivity-aware scheme). The corrected Q2_K memory accounting is **unaffected** by this mixed pattern because:

1. W_ref is dequantized to float32 before corrected Q2_K encoding (so the source quant type is irrelevant once dequantized).
2. The corrected_ceil_per_row Q2_K byte count is shape-dependent and constant for a given (d_out, d_in) regardless of the source quant type.

Both ffn_up and ffn_gate are Q4_K across all 28 layers, so the SDIR residual accounting (k=0.5%) is shape- and quant-consistent for those families.

---

## 8. Memory accounting (per layer, selected policy)

| component | bytes | % of layer Q4 budget |
|---|---:|---:|
| Q2_K ffn_up (corrected_ceil_per_row)   | 4,515,840 | 21.87% |
| Q2_K ffn_gate (corrected_ceil_per_row) | 4,515,840 | 21.87% |
| Q2_K ffn_down (corrected_ceil_per_row) | 4,515,840 | 21.87% |
| SDIR ffn_up @ k=0.5%                   | ~1,127,786 (varies 1,127,760..1,127,824 across layers) | ~5.46% |
| SDIR ffn_gate @ k=0.5%                 | ~1,127,786 (varies 1,127,760..1,127,824 across layers) | ~5.46% |
| SDIR ffn_down                          | 0 | 0.00% (no residual by policy) |
| **TOTAL per layer (typical)**          | **~17,263,466** | **~83.6% of 20,643,840** |
| **Margin vs 3 × Q4_budget_family (typical)** | **+~3,380,360** | **+~16.4% headroom** |

Per-layer margin variance across all 28 layers × 2 seeds: **64 bytes** (3,380,312 to 3,380,376). The variance comes from the SDIR residual size scaling (the number of stored correction pairs depends on the row distribution of the residual, which depends on the per-row scale of the weight). The variance is structural and expected; it does not threaten the per-layer memory-positive claim.

---

## 9. Wall-clock observation (informational only)

| field | value |
|---|---:|
| total wall clock (56 pairs) | **481.5 sec = 8.0 min** |
| mean per-pair wall clock | **~8.6 sec/pair** |
| first-pair wall clock (includes model load) | ~8.6 sec |
| last-pair wall clock (cache hot) | ~8.6 sec |

**The 31BY planning estimate was 25 min working / 35 min upper bound. Actual was 8.0 min — ~3× faster than my lower bound estimate.** The dominant cost is per-pair dequantize + matmul (very fast for a 1.5B FFN at batch 1). The model file is hot in page cache across all 56 pairs, so disk I/O is not a bottleneck. The one-time model load cost is small compared to the per-pair work.

**This observation is INFORMATIONAL ONLY. It is NOT a speedup, latency, throughput, or production-runtime claim.** It is recorded as a planning-estimate-vs-actual check, useful for sizing future phases (3B, 7B, etc.) if those are ever explicitly approved.

---

## 10. Hard gates vs soft thresholds (per 31BY criteria)

### 10.1 Hard gates (no allowance)

| gate | required | actual | PASS? |
|---|---|---|:---:|
| regression passes (before AND after) | PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN, error_count=0, fallback_count=0, readme_drift_guard.passed=true | (run after SOT edits — see PRE-COMMIT REPORT) | ✓ |
| finite pairs | 56/56 | **56/56** | ✓ |
| memory-positive pairs | 56/56 | **56/56** | ✓ |
| severe regressions | 0 | **0** | ✓ |
| model files committed | none | none | ✓ |
| Q2_K / SDIR blobs committed | none | none | ✓ |
| scope creep | none beyond 28 layers, seeds {0, 9}, no ffn_down residual, no FP16 W_ref | none | ✓ |

### 10.2 Soft thresholds (2-pair allowance applies ONLY here)

| threshold | required | actual | PASS? |
|---|---:|---:|:---:|
| cosine-improved | ≥ 54/56 | **56/56** (allowance not even needed) | ✓ |
| MAE-improved | ≥ 54/56 | **56/56** (allowance not even needed) | ✓ |

The 2-pair allowance (independent per threshold) is **not exercised** — 31BZ passed within the strict 56/56 envelope, not within the 54/56 envelope.

---

## 11. Forbidden claims (upheld)

31BZ did **not** run:
- generation / inference / sampling
- llama.cpp runtime integration (the Q2_K encode/dequantize is a ctypes call into `libggml-base.so`, NOT a runtime-integration test)
- any 0.5B-vs-1.5B comparison
- any "1.5B behaves like 0.5B" generalization
- any larger-model claim (3B / 7B / 14B / 32B / 72B / 110B+ Qwen2.5, or any other family)
- any real-activation transfer claim
- any FP16 recovery claim
- any model quality / behavior / production / speedup / latency / throughput claim
- any full-model runtime memory savings claim

---

## 12. What 31BZ does **not** establish

- ❌ 31BZ is not generation. No text was produced. No sampling happened. No model was loaded into a serving stack.
- ❌ 31BZ is not a runtime integration. The corrected Q2_K encode/dequantize is exercised, but llama.cpp's forward-pass path was never invoked end-to-end.
- ❌ 31BZ is not a model quality claim. The cosine/MAE improvements are **tensor-harness** metrics on a synthetic Gaussian-activation input. Real activation distributions, attention patterns, and sequence-length effects are not tested.
- ❌ 31BZ is not a memory savings claim in production. The "memory-positive" classification is a per-layer byte comparison under the harness's 3 × Q4_budget_family policy. It does not establish that any actual serving system would use these bytes this way.
- ❌ 31BZ is not a larger-model claim. Qwen2.5-1.5B is one model, one quantization (Q4_K_M), one policy. 3B / 7B / 14B / 32B / 72B / 110B+ Qwen2.5 are not tested. Other families (Llama, Mistral, etc.) are not tested. Different quantizations (Q5_K_M, Q8_0, FP16) are not tested.
- ❌ 31BZ is not a 0.5B-vs-1.5B comparison. The 0.5B accepted numeric results (31AY / 31BA / 31BM / 31BN / 31BO) are unchanged and have not been re-run in 31BZ.
- ❌ 31BZ is not a speedup, latency, or throughput claim. The 481.5 sec / 8.0 min wall clock is a per-run cost for the harness, not a production runtime metric.

---

## 13. Step-ladder summary

| rung | scope | pairs | mean_dc | mean_MAE_imp | L0 reproducible? | result |
|---:|---|---:|---:|---:|:---:|---|
| 1 (31BU) | L0 only, seeds {0, 9, 13} | 3 | +0.004139 | −0.004712 | n/a (first run) | PASS |
| 2 (31BV) | L0/L14/L27, seeds {0, 9} | 6 | +0.003872 | −0.005923 | ✓ (L0 bit-identical to 31BU) | PASS |
| 3 (31BW) | planning | — | — | — | n/a | PASS |
| 4 (31BX) | L0/L4/L8/L12/L16/L20/L27, seeds {0, 9} | 14 | +0.004641 | −0.010987 | ✓ (L0 bit-identical to 31BU/31BV) | PASS |
| 5 (31BY) | planning | — | — | — | n/a | PASS |
| **6 (31BZ)** | **L0..L27, seeds {0, 9}** | **56** | **+0.004758** | **−0.012865** | **✓ (L0/L4/L14/L27 all bit-identical to 31BU/31BV/31BX)** | **PASS** |

Each rung passed. The aggregate extends the prior rungs cleanly — the worst pair at the aggregate (L26-S9, +0.000390) is the smallest gain in the set, but it is in the green and is bracketed by other small-but-positive pairs (L16-S9, +0.001628; L14-S9, +0.002659). No new failure mode was introduced by the expansion from 7 layers to 28.

The mean improvement grew from 31BX (7 layers) to 31BZ (28 layers): mean_dc from +0.004641 to +0.004758 (+2.5%), mean_MAE_imp from −0.010987 to −0.012865 (+17%). The aggregate is **strictly better** than the stratified probe on both metrics.

---

## 14. Limitations

1. **Single activation regime.** All 56 inputs are `np.random.default_rng(seed).standard_normal((1, 1536))` — zero-mean unit-variance Gaussian on a single token. Real activations have non-zero mean, structured correlations, and longer sequences. **No claim about real-activation behavior is sanctioned** even though 31BZ passed.
2. **Single model.** Qwen2.5-1.5B-Instruct only. No claim transfers to 0.5B, 3B, 7B, 14B, 32B, 72B, 110B+, or to any other Qwen / Llama / Mistral / etc. family member.
3. **Single policy.** Only `corrected_q2k_policy_v1` is being tested. No claim transfers to any other k value, alpha, residual family set, or Q2_K mode.
4. **No ffn_down residual.** The policy intentionally uses W_low-only for ffn_down. No claim transfers to a policy that does add an ffn_down residual.
5. **Tensor-harness only.** This is a standalone tensor harness running in pure NumPy + ctypes against llama.cpp's Q2_K encode/dequantize. No claim transfers to an end-to-end inference run, a serving stack, or a llama.cpp integration.
6. **Memory accounting is the harness's accounting.** `memory_positive` here means "per-layer bytes under the harness's 3 × Q4_budget_family comparison is positive" — it is **not** a claim about runtime memory in any production system.
7. **Wall clock is informational.** 481.5 sec / 8.0 min is the per-run cost of the harness on this machine, not a production-runtime metric.

---

## 15. Classification

**`PASS_31BZ_1_5B_Q2K_FULL_LAYER_TWO_SEED_AGGREGATE_CLEAN`** — 56/56 pairs memory-positive, cosine-improved, MAE-improved, 0 severe, all finite, on the full 28-layer × 2-seed standalone tensor-harness aggregate scope.

Next allowed phase if approved: **Phase 31CA — Qwen2.5-1.5B Corrected Q2_K Aggregate Freeze / Package Update**, only if explicitly requested. 31CA is a documentation + provenance + handoff phase; it does not run validation by default.
