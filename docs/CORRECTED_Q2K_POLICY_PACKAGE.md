# Corrected Q2_K Policy Package

**Package version:** `corrected_q2k_policy_v1`
**Frozen checkpoint:** `phase31bn-corrected-q2k-full-aggregate-checkpoint` → `0304590c92d43fdf48d3d28998255d39c9a20c07`
**Date frozen:** Phase 31BN
**Package created:** Phase 31BO

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

## Frozen Validation Summary

**Source:** Phase 31BM Route A — 384 pairs (all 24 layers × 16 seeds)

| Metric | Value |
|--------|-------|
| n_memory_positive | **384 / 384 (100%)** |
| n_cosine_improved | **383 / 384 (99.74%)** |
| n_MAE_improved | **383 / 384 (99.74%)** |
| n_severe_regressions | **0** |
| mean_delta_cos | +0.0383 |
| median_delta_cos | +0.0351 |
| min_margin | 661,766 bytes/layer |

**Policy status:** STRONG VALIDATION

---

## Known Minor Failures

| Pair | Type | delta_cos | MAE_delta | Severity |
|------|------|-----------|-----------|----------|
| L21-S10 | cosine failure | −0.0294 | −0.0284 | **Non-severe** |
| L2-S13 | MAE regression | +0.0077 | +0.0078 | **Non-severe** |

Both failures are stable across 31BL and 31BM runs, isolated, and non-severe.

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
- no larger-model claim
- no runtime-ready output-residual claim
- no claim beyond standalone tensor harness
- no claim that 31AY/31BA exact anchors are current canonical metrics

---

## What This Package Does NOT Prove

- No runtime integration with llama.cpp or any inference server
- No generation or instruction-following capability
- No behavior recovery in downstream tasks
- No results beyond Qwen2.5-0.5B
- No full-model memory savings when `ffn_down` residual is disabled at scale

---

## Next Recommended Phase

**Phase 31BP — Corrected Q2_K Larger-Model Feasibility Planning** (only if explicitly requested)

Before attempting larger-model validation (e.g., Qwen2.5-1.5B or Qwen2.5-3B), Phase 31BP should produce a feasibility analysis: estimate memory budget at larger layer sizes, determine whether the `up+gate` policy scales, and plan a staged validation route.

---

## Policy Package Files

| File | Purpose |
|------|---------|
| `docs/CORRECTED_Q2K_POLICY_PACKAGE.md` | This document |
| `src/results/CORRECTED_Q2K_POLICY_PACKAGE.json` | Machine-readable policy manifest |
| `src/corrected_q2k_policy.py` | Optional constants helper (stdlib only) |
| `src/q2k_backend.py` | Q2_K quantization backend |
| `src/phase31x_manifest_runtime.py` | SDIR runtime utilities |
| `src/phase31bm_corrected_q2k_broader_aggregate.py` | Full 384-pair validation runner |
| `docs/PHASE31BN_CORRECTED_Q2K_FULL_AGGREGATE_CHECKPOINT.md` | Freeze checkpoint doc |
| `src/results/PHASE31BN_CORRECTED_Q2K_FULL_AGGREGATE_CHECKPOINT.json` | Freeze checkpoint JSON |
| `docs/PHASE31BM_CORRECTED_Q2K_BROADER_AGGREGATE.md` | Full aggregate validation doc |
| `src/results/PHASE31BM_CORRECTED_Q2K_BROADER_AGGREGATE.json` | Full aggregate results |
| `SOURCE_OF_TRUTH.md` | Current canonical source of truth |