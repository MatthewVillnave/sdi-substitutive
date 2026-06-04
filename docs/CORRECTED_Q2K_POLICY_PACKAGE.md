# Corrected Q2_K Policy Package

**Package version:** `corrected_q2k_policy_v1` (UNCHANGED in 31CA; policy parameters are not bumped)
**Package created:** Phase 31BO
**Package last updated:** Phase 31CA (added 1.5B evidence tier; policy parameters UNCHANGED; policy version NOT bumped)
**Frozen 0.5B checkpoint:** `phase31bn-corrected-q2k-full-aggregate-checkpoint` → `0304590c92d43fdf48d3d28998255d39c9a20c07`
**Frozen 1.5B checkpoint (proposed tag, NOT created in 31CA):** `phase31ca-1_5b-corrected-q2k-aggregate-checkpoint` → `f7f2a91d1b904f8f156d6c89584ec0d32229c23e`

> **31CA update:** the 1.5B 31BZ aggregate is added as a **second evidence tier** under the same `corrected_q2k_policy_v1`. Policy parameters are unchanged. The package is now backed by two frozen evidence tiers: 0.5B (31BN/31BM) and 1.5B (31BZ/31CA). For the detailed 31CA freeze doc see `docs/PHASE31CA_1_5B_AGGREGATE_FREEZE_PACKAGE.md`.

---

## Selected Policy

| Parameter | Value |
|-----------|-------|
| Q2_K mode | `corrected_ceil_per_row` |
| Residual families | `ffn_up` + `ffn_gate` |
| Residual k | `0.5%` (fraction of elements per family channel) |
| Alpha | `1.0` |
| ffn_down residual | **disabled** — left as Q4 budget slack |

**What this means:** The selected policy encodes `ffn_up` and `ffn_gate` weights as `corrected_ceil_per_row` Q2_K and applies a small SDIR residual only to those two families. The `ffn_down` family stays as raw Q2_K without any residual — its budget slack acts as the memory margin buffer.

---

## Tensor Orientation

- Artifact tensor shape: `(d_out, d_in)`
- MLP computation: `Y = X @ W.T` (standard right-transpose convention)
- Shape is identical between Qwen2.5-0.5B official GGUF and dequantized FP32 reference

---

## Supported Families

| Family | Role | Residual? | Notes |
|--------|------|-----------|-------|
| `ffn_up` | MLP up-projection | ✓ SDIR residual | Q2_K + residual = up+gate policy |
| `ffn_gate` | MLP gate projection | ✓ SDIR residual | Q2_K + residual = up+gate policy |
| `ffn_down` | MLP down-projection | ✗ no residual | Q2_K only; no residual budget allocated |

**Cross-tier consistency:** the residual families and ffn_down treatment are identical across both frozen evidence tiers (0.5B 31BN and 1.5B 31BZ/31CA). The same corrected_ceil_per_row Q2_K + ffn_up/ffn_gate SDIR k=0.5% + no ffn_down residual policy is applied to both models. The corrected_ceil_per_row Q2_K byte count is shape-dependent and constant across quant types for the same shape, so the policy scales from 0.5B (hidden=896, intermediate=4864) to 1.5B (hidden=1536, intermediate=8960) without parameter changes.

---

## Artifact Formats

### Q2_K Weights (`corrected_ceil_per_row`)

Produced by `q2k_backend.py: quantize_q2k_f32_to_bytes()` with `mode='corrected_ceil_per_row'`.
Dequantized by `q2k_backend.py: dequantize_q2k_bytes_to_f32()`.

### SDIR Residual

Produced by `phase31x_manifest_runtime: encode_sdir()`.
Decoded by `phase31x_manifest_runtime: decode_sdir()`.

SDIR is applied only to `ffn_up` and `ffn_gate` at `k=0.5%` per family.
SDIR is **not** applied to `ffn_down`.

---

## Backend Requirements

- **q2k_backend.py** — `quantize_q2k_f32_to_bytes`, `dequantize_q2k_bytes_to_f32`
- **phase31x_manifest_runtime.py** — `encode_sdir`, `decode_sdir`, `cosine`
- **libggml-base.so** — loaded via `SDI_LLAMA_CPP_LIB` env var
- **numpy** — all tensor operations
- **Python stdlib** — `json`, `re`, `os`, `sys`

---

## Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `SDI_GGUF_MODEL_PATH` | Path to Qwen2.5-0.5B Q2_K GGUF | `/media/.../qwen2.5-0.5b-instruct-q2_k.gguf` |
| `SDI_LLAMA_CPP_ROOT` | Path to llama.cpp root | `/home/.../llama.cpp` |
| `SDI_LLAMA_CPP_LIB` | Path to `libggml-base.so` | `/home/.../llama.cpp/build/bin/libggml-base.so` |

> **Note:** No private paths are hardcoded in `corrected_q2k_policy.py` — runners must inject these via environment.

---

## Byte / Memory Accounting

| Component | Per-Layer Bytes | Notes |
|-----------|----------------|-------|
| Q2_K `ffn_gate` | ~1.17 MB | corrected_ceil_per_row |
| Q2_K `ffn_up` | ~1.17 MB | corrected_ceil_per_row |
| Q2_K `ffn_down` | ~1.17 MB | corrected_ceil_per_row |
| SDIR `ffn_gate` @ k=0.5% | ~588 KB | encode_sdir at k=0.5% |
| SDIR `ffn_up` @ k=0.5% | ~588 KB | encode_sdir at k=0.5% |
| **Total per layer** | **~5.21 MB** | |
| **Q4 baseline per layer** | **~5.88 MB** | 2179072 × 3 bytes |
| **Margin per layer** | **~661,766 bytes** | always positive |

**Aggregate margin (24 layers):** ~254,130,400 bytes (~254 MB)

---

## Frozen Validation Summary — Two Evidence Tiers

This package is backed by **two frozen evidence tiers** under the same `corrected_q2k_policy_v1`:

### Tier 1 — Qwen2.5-0.5B (Phase 31BN/31BM)

**Source:** Phase 31BM Route A — 384 pairs (all 24 layers × 16 seeds)
**Frozen at commit:** `0304590c92d43fdf48d3d28998255d39c9a20c07` (tag `phase31bn-corrected-q2k-full-aggregate-checkpoint`)

| Metric | Value |
|--------|-------|
| n_memory_positive | **384 / 384 (100%)** |
| n_cosine_improved | **383 / 384 (99.74%)** |
| n_MAE_improved | **383 / 384 (99.74%)** |
| n_severe_regressions | **0** |
| mean_delta_cos | +0.0383 |
| median_delta_cos | +0.0351 |
| min_margin | 661,766 bytes/layer |

**Policy status (0.5B):** STRONG VALIDATION

### Tier 2 — Qwen2.5-1.5B (Phase 31BZ/31CA)

**Source:** Phase 31BZ — 56 pairs (all 28 layers × 2 seeds, 100% layer coverage)
**Frozen at commit:** `f7f2a91d1b904f8f156d6c89584ec0d32229c23e` (proposed tag `phase31ca-1_5b-corrected-q2k-aggregate-checkpoint`, NOT created in 31CA)

| Metric | Value |
|--------|-------|
| n_memory_positive | **56 / 56 (100%)** |
| n_cosine_positive | **56 / 56 (100%)** |
| n_MAE_improving | **56 / 56 (100%)** |
| n_severe_regressions | **0** |
| n_finite | **56 / 56 (100%)** |
| mean_delta_cos | +0.004758 |
| median_delta_cos | +0.004501 |
| mean_MAE_improvement | −0.012865 |
| min_per_layer_margin | +3,380,312 bytes |
| max_per_layer_margin | +3,380,376 bytes |
| worst pair | L26-S9, Δcos=+0.000390 (still strongly positive) |
| cross-runner reproducibility | 4 runners, 8 overlapping layer/seed pairs bit-identical |

**Policy status (1.5B):** CLEAN

### Known Minor Failures (0.5B tier only)

| Pair | Type | delta_cos | MAE_delta | Severity |
|------|------|-----------|-----------|----------|
| L21-S10 | cosine failure | −0.0294 | −0.0284 | **Non-severe** |
| L2-S13 | MAE regression | +0.0077 | +0.0078 | **Non-severe** |

Both failures are stable across 31BL and 31BM runs, isolated, and non-severe. The 1.5B tier (31BZ) has **no known minor failures** — 56/56 PASS at all gates.

---

## Forbidden Claims

The following claims are **not supported** by this package:

- no model quality recovery claim
- no behavior recovery claim
- no speedup claim
- no full-model runtime memory savings claim
- no llama.cpp integration claim
- no production readiness claim
- no inference/generation claim
- no larger-model claim (3B / 7B / 14B / 32B / 72B / 110B+ Qwen2.5 are NOT tested; other families Llama/Mistral/etc. are NOT tested)
- no runtime-ready output-residual claim
- no claim beyond standalone tensor harness
- no claim that 31AY/31BA exact anchors are current canonical metrics
- no FP16 recovery claim (W_ref is dequantized Q4_K_M, NOT FP16)
- no 0.5B-vs-1.5B generalization claim
- no "1.5B behaves like 0.5B" claim
- no broader model-family claim
- no real-activation-transfer claim (the harness uses `np.random.default_rng` `standard_normal`; real activations are not tested)

## What This Package Does NOT Prove

- No runtime integration with llama.cpp or any inference server
- No generation or instruction-following capability
- No behavior recovery in downstream tasks
- No results beyond the two frozen evidence tiers (Qwen2.5-0.5B and Qwen2.5-1.5B). 3B / 7B / 14B / 32B / 72B / 110B+ Qwen2.5 NOT tested. Other families (Llama / Mistral / etc.) NOT tested.
- No full-model memory savings when `ffn_down` residual is disabled at scale (memory is computed per-layer under the harness's 3× Q4 budget family comparison; not a claim about any production serving system)
- No real-activation parity (the harness uses `np.random.default_rng` `standard_normal` on a single token; real activation distributions, attention patterns, and sequence lengths are not tested)

## Next Recommended Phase

**Phase 31CB — Stale File Provenance Cleanup, only if explicitly requested** (recommended default per 31CA's reasoning)

The repo still has 5 stale untracked 31BH-R2 / 31BJ files carried through 31BU → 31BZ without resolution. With the 1.5B aggregate now frozen as a second evidence tier, these are the only outstanding provenance debt in the working tree. Cleanup is the lowest-risk next step and removes noise before a new scientific lane is opened.

Alternative 31CB options (operator may choose instead):
- 31CB-A — Real-Activation Capture Planning (prerequisite for any real-activation claim)
- 31CB-B — Runtime Artifact Format / Loader Planning (prerequisite for any runtime-integration claim)

---

## Policy Package Files

| File | Purpose |
|------|---------|
| `docs/CORRECTED_Q2K_POLICY_PACKAGE.md` | This document |
| `src/results/CORRECTED_Q2K_POLICY_PACKAGE.json` | Machine-readable policy manifest |
| `src/corrected_q2k_policy.py` | Optional constants helper (stdlib only) |
| `src/q2k_backend.py` | Q2_K quantization backend |
| `src/phase31x_manifest_runtime.py` | SDIR runtime utilities |
| `src/phase31bm_corrected_q2k_broader_aggregate.py` | Full 384-pair validation runner (0.5B tier) |
| `docs/PHASE31BN_CORRECTED_Q2K_FULL_AGGREGATE_CHECKPOINT.md` | 0.5B freeze checkpoint doc |
| `src/results/PHASE31BN_CORRECTED_Q2K_FULL_AGGREGATE_CHECKPOINT.json` | 0.5B freeze checkpoint JSON |
| `docs/PHASE31BM_CORRECTED_Q2K_BROADER_AGGREGATE.md` | 0.5B full aggregate validation doc |
| `src/results/PHASE31BM_CORRECTED_Q2K_BROADER_AGGREGATE.json` | 0.5B full aggregate results |
| `src/phase31bz_1_5b_corrected_q2k_full_layer_two_seed_aggregate.py` | Full 56-pair validation runner (1.5B tier) |
| `docs/PHASE31BZ_1_5B_CORRECTED_Q2K_FULL_LAYER_TWO_SEED_AGGREGATE.md` | 1.5B full aggregate validation doc |
| `src/results/PHASE31BZ_1_5B_CORRECTED_Q2K_FULL_LAYER_TWO_SEED_AGGREGATE.json` | 1.5B full aggregate results |
| `docs/PHASE31CA_1_5B_AGGREGATE_FREEZE_PACKAGE.md` | 1.5B freeze package doc (this phase) |
| `src/results/PHASE31CA_1_5B_AGGREGATE_FREEZE_PACKAGE.json` | 1.5B freeze package JSON (this phase) |
| `SOURCE_OF_TRUTH.md` | Current canonical source of truth |