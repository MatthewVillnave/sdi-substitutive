# Phase 31X: Manifest-Driven Bundle Runtime

**Classification:** `PASS_MANIFEST_BUNDLE_RUNTIME`

---

## What Was Implemented

### bundle_manifest.py — Manifest v0.2 Loader

`ManifestLoader` class that:
- Parses `manifest.json` (schema v0.2) with `schema_version` validation
- `load()` — parses JSON, validates version field
- `validate_bundle(bundle_dir)` — validates all tensor entries; returns counters
- `_validate_tensor_entry(entry, bundle_dir)` — per-entry validation:
  - Required fields check (tensor_name, layer, family, shape, bytes, checksums, etc.)
  - `memory_margin_bytes > 0` guard
  - `total_substitutive_bytes < W_ref_Q4_budget_bytes` guard
  - `decode_temp_bound_bytes >= 128` guard
  - File presence via `os.path.isfile` (not `os.path.exists`) — fixes dir false-positive bug
  - Checksum validation (sha256) with placeholder_sha256 skip
- `select_tensor(family, layer)` — returns tensor entry or None
- `load_sdiw(entry, bundle_dir)` / `load_sdir(entry, bundle_dir)` helpers
- `sha256_file()` and `sha256_bytes()` utilities

**Critical bug fixed:** `os.path.exists("")` returns True (the directory itself), causing missing-file false negatives. Changed to `os.path.isfile()`.

### phase31x_manifest_runtime.py — ManifestRuntime + Tests

**`ManifestRuntime` class:**
- `load_and_validate_manifest(bundle_dir)` — loads manifest, runs validation, returns counters
- `execute_substitutive_path(entry, X)` — combined .sdiw+.sdir compute, W_ref never loaded
- No-additive-trap counters: W_ref_loaded=0, dense_W_low_materialized=0, dense_R_materialized=0, sdiw_loaded, sdir_loaded, manifest_loaded, checksum_validated, memory_budget_validated, fallback_count, error_count, path_label="[SDI-SUB-RUNTIME]"

**Negative tests (8):**
| Test | Passed | Error Count | W_ref Loaded |
|------|--------|-------------|--------------|
| missing_manifest | ✅ | 1 | 0 |
| malformed_manifest | ✅ | 1 | 0 |
| missing_sdiw | ✅ | 1 | 0 |
| missing_sdir | ✅ | 1 | 0 |
| checksum_mismatch | ✅ | 1 | 0 |
| shape_mismatch | ✅ | 1 | 0 |
| memory_budget_violation | ✅ | 1 | 0 |
| requested_tensor_not_in_manifest | ✅ | 1 | 0 |

**Compute tests:**
- **Test A — Tiny synthetic (4×8, 2 layers):** PASS
  - manifest_loaded=1, checksum_validated=2, memory_budget_validated=2, error_count=0
  - Layer 0: cos_sub=-0.556377, Layer 1: cos_sub=0.475341, no NaN/Inf
- **Test B — Realistic ffn_up layer0 (896×4864):** PASS
  - manifest_loaded=1, checksum_validated=1, memory_budget_validated=1, error_count=0
  - cos_sub=0.79939562, delta_cos=+0.00004095, no NaN/Inf
  - W_ref bytes: 0, margin: 708,198 bytes

## Memory Reporting — Test B

| Metric | Value |
|--------|-------|
| W_ref bytes | 0 |
| W_low packed bytes | 2,179,072 |
| W_low scale bytes | 272,384 |
| Residual bytes | 1,198,490 |
| Total substitutive bytes | 3,649,946 |
| Q4 budget | 4,358,144 |
| Margin bytes | 708,198 |
| Decode scratch peak | ≤128 |
| Dense W_low avoided | 1 |
| Dense R avoided | 1 |
| W_ref avoided | 1 |

## Negative Test Design

All 8 fail-fast tests enforce the same contract:
- **Error raised** for invalid condition
- **error_count incremented** in all failure paths
- **W_ref_loaded stays 0** — no silent fallback to W_ref
- **fallback_count stays 0** — no silent resolution of missing files

### Key design decisions

1. **Placeholder checksums** (`placeholder_sha256`) are skipped but NOT treated as error
2. **`os.path.isfile`** used instead of `os.path.exists` to avoid false positives from empty string paths resolving to the bundle directory itself
3. **Checksum test budget fix:** test_checksum_mismatch sets W_ref_Q4_budget_bytes=256 (not 128) to ensure the memory budget check passes before the checksum check can run; the actual checksum mismatch is still correctly detected as ValueError
4. **select_tensor returns None** for missing layers (not an exception) — callers must handle None

## Results JSON

Full results at `results/PHASE31X_MANIFEST_BUNDLE_RUNTIME.json`:
- `classification`: PASS_MANIFEST_BUNDLE_RUNTIME
- `phase`: 31X
- `neg_results`: 8 test entries with pass/error_count/W_ref_loaded/fallback_count
- `tiny_result`: counters + result0 + result1
- `realistic_result`: counters + result + memory breakdown

## Phase 31Y Unlocked

**Yes.** Manifest-driven bundle loading is operational. Phase 31Y can now implement:
- Real artifact integration (actual model weights)
- Multilayer batch processing
- End-to-end substitutive decode pipeline

---
*Phase 31X — ELVIS — SDI Substitutive*