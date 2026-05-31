# Static Artifact Schema v1.0

> Schema version: **"1.0"**
> Status: canonical for Phase 31BF onward

---

## Scope

This schema defines the artifact format for the SDI-Substitutive static tensor harness: packed low-rank weights (`.sdiw`) and compressed sparse residuals (`.sdir`), bundled with a `manifest.json`. All artifacts for the static substitutive path must conform to this schema.

---

## Artifacts

### `.sdiw` — Packed W_low Artifact

Binary format for the low-rank packed weight tensor.

**Layout:**

```
[header: 16 bytes][fp16 scale bytes][packed nibble bytes]
```

**Header fields (little-endian, `<`):**

| offset | type    | name            | value                          |
|--------|---------|-----------------|--------------------------------|
| 0      | 4s      | magic           | `b"SDIW"`                      |
| 4      | H       | version         | 1                              |
| 6      | H       | flags           | 0                              |
| 8      | I       | rows            | d_out                          |
| 12     | I       | cols            | d_in                           |
| 16     | I       | scale_nbytes    | (rows×cols//32)×2              |
| 20     | I       | packed_nbytes   | (rows×cols+1)//2               |

**Notes:**
- `BLOCK_SIZE = 32` for scale block granularity
- Packed nibble encoding: `q = clamp(round(w/scale), -8, 7)`, stored as `(q+8)`
- Scale = `max(|block|) / 7.5`, minimum `1e-9`

**Byte accounting:**

```
W_low_bytes = packed_nbytes + scale_nbytes
```

---

### `.sdir` — Sparse Residual Artifact

Binary format for the top-k sparse residual correction.

**Encoding:** `encode_sdir` stream — bitmap + fp16 header values per block.

**Layout:**

```
[block 0: bitmap(4 bytes) + values(n_nnz_0 × 2 bytes)][block 1: ...]...
```

Each block: 4-byte little-endian u32 bitmap indicating which of the 32 elements are non-zero, followed by fp16 values for the set bits in row-major order.

**Byte accounting:**

```
residual_bytes = total blocks × 4 + sum(n_nnz_per_block × 2)
```

---

### `manifest.json` — Bundle Manifest

JSON metadata describing the artifact bundle.

---

## Manifest Schema v1.0

### Top-Level Fields

| field               | type     | required | description |
|---------------------|----------|----------|-------------|
| `schema_version`    | string   | **yes**  | Must be `"1.0"` |
| `package_id`        | string   | yes      | Unique identifier for this artifact bundle |
| `bundle_type`       | string   | yes      | One of the allowed bundle types (see below) |
| `source_model`      | object   | yes      | Source model metadata |
| `substitution_policy` | object | yes      | Residual policy parameters |
| `layers`            | array    | **yes**  | Array of tensor entries (see below) |
| `global_memory`     | object   | no       | Aggregate memory accounting |
| `runtime_requirements` | object | no | Runtime constraints (informational) |

### `source_model` Fields

| field          | type   | required | description |
|----------------|--------|----------|-------------|
| `model_name`   | string | yes      | Model identifier, e.g. `qwen2.5-0.5b` |
| `architecture`  | string | no       | e.g. `ffn`, `mlp` |
| `quantization` | string | yes      | Quantization format of W_low, e.g. `Q2_K`, `Q4_K_M` |

### `substitution_policy` Fields

| field             | type    | required | description |
|-------------------|---------|----------|-------------|
| `k_percent`       | float   | yes      | Residual sparsity percentage (0.0–100.0) |
| `alpha`           | float   | yes      | Residual scale multiplier |
| `W_low_format`    | string  | yes      | e.g. `sdiw_v1`, `packed_nibble_v1.0` |
| `residual_encoding` | string | yes      | e.g. `sdir_v1`, `bitmap+fp16_v1.0` |
| `residual_policy` | string  | yes      | `always_on`, `gated`, or `skip` |
| `scale_policy`    | string  | yes      | e.g. `block32_fp16` |
| `q2_k_backend_mode` | string | no       | Q2_K quantization mode: `historical_floor_flat` (legacy, truncates final partial block) or `corrected_ceil_per_row` (canonical, pads to full blocks). Required when `W_low_format` includes Q2_K. |

### `layers[]` — Tensor Entry Fields

| field                      | type             | required | description |
|----------------------------|------------------|----------|-------------|
| `tensor_name`              | string           | **yes**  | e.g. `blk.21.ffn_up.weight` |
| `layer`                    | integer          | **yes**  | Layer index (non-negative) |
| `family`                   | string           | **yes**  | One of: `ffn_up`, `ffn_gate`, `ffn_down` |
| `shape`                    | array[2]         | **yes**  | `[d_out, d_in]` — canonical orientation |
| `orientation`               | string           | yes      | Must be `canonical_d_out_d_in` |
| `formats`                  | object           | yes      | Encoding details |
| `checksums`                | object           | no       | SHA256 of artifacts if available |
| `paths`                    | object           | no       | Explicit artifact paths (relative preferred) |
| `W_ref_bytes`              | integer          | no       | Reference tensor bytes (informational) |
| `W_ref_Q4_budget_bytes`    | integer          | **yes**  | Q4 budget for this tensor |
| `W_low_packed_bytes`        | integer          | **yes**  | `.sdiw` packed bytes |
| `W_low_scale_bytes`         | integer          | **yes**  | `.sdiw` scale bytes |
| `residual_bytes`            | integer          | **yes**  | `.sdir` bytes |
| `total_substitutive_bytes`  | integer          | **yes**  | `W_low_packed_bytes + residual_bytes` |
| `memory_margin_bytes`       | integer          | **yes**  | `W_ref_Q4_budget - total_substitutive_bytes` |

**Allowed values for `family`:** `ffn_up`, `ffn_gate`, `ffn_down`

**Allowed values for `orientation`:** `canonical_d_out_d_in`

### `formats` Sub-Object Fields

| field              | type    | required | description |
|--------------------|---------|----------|-------------|
| `W_low_format`     | string  | yes      | e.g. `sdiw_v1`, `packed_nibble_v1.0` |
| `residual_format`  | string  | yes      | e.g. `sdir_v1`, `bitmap+fp16_v1.0` |
| `k_percent`        | float   | yes      | Must match top-level `substitution_policy.k_percent` |
| `value_dtype`      | string  | yes      | `fp16` for residuals |
| `mask_encoding`    | string  | yes      | `dense_bitmap` for current format |
| `scale_policy`     | string  | yes      | e.g. `block32_fp16` |

### `checksums` Sub-Object Fields

| field       | type   | required | description |
|-------------|--------|----------|-------------|
| `wlow`      | string | no       | SHA256 of `.sdiw` bytes |
| `residual`  | string | no       | SHA256 of `.sdir` bytes |

### `paths` Sub-Object Fields

| field       | type   | required | description |
|-------------|--------|----------|-------------|
| `sdiw_path` | string | no       | Path to `.sdiw` file, relative to bundle dir preferred |
| `sdir_path` | string | no       | Path to `.sdir` file, relative to bundle dir preferred |

---

## Metric Conventions

These conventions are fixed for all Phase 31 artifacts and regression tests.

| name                    | formula                    | description |
|-------------------------|----------------------------|-------------|
| `delta_cos`             | `cos_sub − cos_low`        | Cosine improvement vs Q2-only |
| `cosine_improved`       | `delta_cos > 0`            | Residual improved cosine |
| `cosine_nonnegative`    | `delta_cos ≥ 0`             | Residual did not hurt cosine |
| `severe_regression`     | `delta_cos < −0.05`        | Severe cosine regression threshold |
| `MAE_delta`             | `MAE_sub − MAE_low`        | MAE change vs Q2-only |
| `MAE_improved`          | `MAE_delta < 0`            | Residual improved MAE |
| `memory_positive`       | `margin ≥ 0`              | Within Q4 budget |

**Threshold constants:**

```
MIN_DELTA_COS_ACCEPT = 0.0          # cosine_improved threshold
MAX_SEVERE_DELTA_COS = -0.05         # severe_regression threshold
MAE_IMPROVED_MAX_DELTA = 0.0         # MAE_improved threshold
```

---

## Memory Budget Rules

For Qwen2.5-0.5B FFN layers (d_out=4864, d_in=896):

```
Q4_BUDGET_FAMILY = 2,179,072 bytes   # per-family Q4 budget
Q4_BUDGET_LAYER  = 6,537,216 bytes   # per-layer Q4 budget (3 families)
```

**Memory-positive condition:**

```
W_low_packed_bytes + residual_bytes <= W_ref_Q4_budget_bytes
memory_margin_bytes = W_ref_Q4_budget_bytes - (W_low_packed_bytes + residual_bytes)
memory_margin_bytes must be >= 0
```

---

## Path Rules

1. All artifact paths in `manifest.json` must be **relative to the bundle directory**, not absolute.
2. Bundle directory is the directory containing `manifest.json`.
3. The recommended artifact subdirectory is `tensors/`.
4. Standard naming: `tensors/blk.{layer}.{family}.wlow.sdiw`, `tensors/blk.{layer}.{family}.residual.sdir`
5. No hardcoded private paths (e.g., `/home/matthew-villnave/...`, `/media/matthew-villnave/...`) in committed artifact metadata.
6. The regression suite uses temporary fixture bundles and does not require model files.

---

## Orientation Contract

**Canonical orientation:** `(d_out, d_in)`

All `.sdiw` and `.sdir` artifacts encode tensors in canonical orientation. The substitutive runtime applies them as:

```
Y = X @ W.T   (X shape: (d_in,), W shape: (d_out, d_in), Y shape: (d_out,))
```

**Wrong orientation (e.g., transposed) fails fast in the manifest loader.**

---

## Bundle Types

| `bundle_type` | description |
|---------------|-------------|
| `source_of_truth_regression` | Regression test fixtures only |
| `ffn_up_substitutive` | FFN up-projection substitutive bundle |
| `ffn_down_substitutive` | FFN down-projection substitutive bundle |
| `full_mlp_substitutive` | Full MLP (up+gate+down) substitutive bundle |
| `layer_substitutive` | Full layer substitutive bundle |

---

## Compatibility Notes

- Schema v1.0 is forward-compatible only within the v1.x series.
- The `.sdiw` binary format (version 1) is immutable once written.
- The `.sdir` bitmap+fp16 encoding is the current canonical format.
- This schema does NOT cover llama.cpp runtime integration — that is a future phase.
- This schema does NOT cover output-residual / activation-specific artifacts — those are future work.

---

## Forbidden Claims (Schema Context)

This schema supports the static substitutive tensor harness only. It does not enable:

- no model quality recovery claim
- no behavior recovery claim
- no llama.cpp integration claim
- no production readiness claim
- no inference/generation claim
- no runtime-ready output-residual claim
- no claim beyond standalone tensor harness

---

## Schema Version History

| version | phase    | description |
|---------|----------|-------------|
| `0.2.0` | 31AJ     | Initial source-of-truth schema with ffn_up/down |
| `1.0`   | 31BF     | Added ffn_gate, formal spec, metric conventions, path rules |
