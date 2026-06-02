# Phase 31BP — Corrected Q2_K Larger-Model Feasibility Planning

**Classification:** `PASS_31BP_LARGER_MODEL_FEASIBILITY_PLAN_READY`
**Date:** Phase 31BP
**Repo:** sdi-substitutive
**Current HEAD:** `8aee63e8` (Phase 31BO)
**Policy package:** `corrected_q2k_policy_v1`
**Frozen checkpoint:** `phase31bn-corrected-q2k-full-aggregate-checkpoint` → `0304590c`

> **Important:** This phase performed planning only. No larger-model validation was executed. No models were downloaded.

---

## Policy Package Verified

The `corrected_q2k_policy_v1` package was verified:
- `docs/CORRECTED_Q2K_POLICY_PACKAGE.md` ✓
- `src/results/CORRECTED_Q2K_POLICY_PACKAGE.json` ✓
- `src/corrected_q2k_policy.py` ✓
- `tests/test_corrected_q2k_policy.py` ✓
- Regression smoke test covers constants ✓

---

## Model Availability Summary

| Model | Size | Locally Available? | Path Known? | Notes |
|-------|------|-------------------|------------|-------|
| Qwen2.5-0.5B | 0.5B | ✓ Yes | `/media/matthew-villnave/VL_usb/models/qwen2.5-0.5b-official/` | Q2_K, Q3_K, Q4_K variants available |
| Qwen2.5-1.5B | 1.5B | ✗ Not locally | — | Not on VL_usb or in ~/.cache/huggingface/hub |
| Qwen2.5-3B | 3B | ✗ Not locally | — | Not on VL_usb or in ~/.cache/huggingface/hub |
| Qwen2.5-7B | 7B | ✗ Not locally | — | Not on VL_usb; only Qwen2.5-32B-Instruct-GGUF in cache |
| Qwen2.5-32B | 32B | ~ In cache | `~/.cache/huggingface/hub/` | GGUF variant available |
| LFM2.5-8B-A1B-Q4_K_M | 8B | ✓ Yes | `/media/matthew-villnave/VL_usb/models/` | Different model, not Qwen family |

**Conclusion:** Only Qwen2.5-0.5B is locally available. Larger Qwen2.5 GGUF files are not present locally.

---

## Qwen2.5-0.5B Verified Metadata

From actual GGUF metadata via GGUFReader:

| Parameter | Value |
|-----------|-------|
| Architecture | Qwen2 |
| n_layers | 24 |
| hidden_size | 896 |
| intermediate_size | 4864 (FFN expanded dimension) |
| ffn_up shape | `[896, 4864]` in GGUFReader (storage/representation format — NOT canonical artifact orientation). Canonical MLP orientation: ffn_up and ffn_gate = (d_out=4864, d_in=896), ffn_down = (d_out=896, d_in=4864), using Y=X@W.T. Any GGUFReader raw shape display must not be used to redefine canonical orientation. |
| ffn_gate shape | `[896, 4864]` in GGUFReader (storage/representation format — NOT canonical artifact orientation). Canonical MLP orientation: ffn_up and ffn_gate = (d_out=4864, d_in=896), ffn_down = (d_out=896, d_in=4864), using Y=X@W.T. Any GGUFReader raw shape display must not be used to redefine canonical orientation. |
| ffn_down shape | `[4864, 896]` = 4,358,144 elements |
| Elements per family per layer | 4,358,144 |
| File size (Q2_K) | ~396 MB |

**Tensor orientation:** Canonical artifact orientation is `(d_out, d_in)` = `(4864, 896)` for ffn_up and ffn_gate, `(896, 4864)` for ffn_down. GGUFReader may report `[896, 4864]` as a storage/representation format — this must not be used to redefine canonical orientation. Computation: `Y = X @ W.T`.

---

## Memory Feasibility Estimate — Larger Models

Estimates based on Qwen2.5 public architecture specs. **Not verified against local GGUF files.**

### Qwen2.5-1.5B (estimated)

| Parameter | Value | Source |
|-----------|-------|--------|
| n_layers | 28 | Qwen2.5-1.5B public spec |
| hidden_size | 1,536 | Qwen2.5-1.5B public spec |
| intermediate_size | 13,104 | Qwen2.5-1.5B public spec |
| ffn_up shape | `[1536, 13104]` | derived |
| Elements per family per layer | ~20,127,744 | 1536 × 13104 |
| Q2_K bytes per family (row-ceil) | ~7.65 MB/layer | estimated, not measured |

> **Note:** `corrected_ceil_per_row` row-ceil accounting means the Q2_K byte budget grows with `d_out` (hidden_size). At hidden_size=1536 vs 896, the row-ceil per-block overhead is larger. This is a key risk for margin at larger shapes.

### Qwen2.5-3B (estimated)

| Parameter | Value | Source |
|-----------|-------|--------|
| n_layers | 36 | Qwen2.5-3B public spec |
| hidden_size | 2,048 | Qwen2.5-3B public spec |
| intermediate_size | **CONFLICTING** — public spec says 8,192; prior project evidence says 11,008; **needs GGUFReader verification** | public spec vs prior evidence |
| ffn_up shape | `[2048, CONFLICTING]` | derived — depends on intermediate_size (8192 vs 11008); needs GGUFReader verification |
| Q2_K bytes per family (row-ceil) | ~8.39 MB/layer | estimated, not measured |

> **Note:** 3B intermediate_size is unverified and conflicting — do not use 8192 or 11008 for memory estimates until GGUFReader confirms actual value.

### Qwen2.5-7B (estimated)

| Parameter | Value | Source |
|-----------|-------|--------|
| n_layers | 32 | Qwen2.5-7B public spec |
| hidden_size | 4,096 | Qwen2.5-7B public spec |
| intermediate_size | 18,432 | Qwen2.5-7B public spec |
| ffn_up shape | `[4096, 18432]` | derived |
| Elements per family per layer | ~75,497,856 | 4096 × 18432 |
| Q2_K bytes per family (row-ceil) | ~18.9 MB/layer | estimated, not measured |

> **Note:** At 7B scale, the row-ceil overhead is significant. The margin from disabled ffn_down residual may be reduced or eliminated. Feasibility is uncertain without measurement.

---

## Selected Policy Memory Budget — 0.5B (Verified)

From Phase 31BM actual results:

| Component | Per-Layer Bytes | Formula |
|-----------|----------------|---------|
| Q2_K ffn_gate (corrected_ceil_per_row) | ~1.17 MB | measured |
| Q2_K ffn_up (corrected_ceil_per_row) | ~1.17 MB | measured |
| Q2_K ffn_down (corrected_ceil_per_row) | ~1.17 MB | measured |
| SDIR residual ffn_gate @ k=0.5% | ~0.574 MB | measured |
| SDIR residual ffn_up @ k=0.5% | ~0.574 MB | measured |
| **Total selected policy per layer** | **~5.21 MB** | |
| Q4 baseline per layer | ~5.88 MB | |
| **Margin per layer** | **~0.662 MB** | always positive at 0.5B |

Aggregate margin (24 layers): ~254 MB for 0.5B.

---

## Margin Risk at Larger Shapes

The key risk when scaling to larger models:

1. **Row-ceil overhead scales with `d_out` (hidden_size):** At larger hidden sizes (1536, 2048, 4096), the per-row ceil rounding in Q2_K increases the block overhead relative to the raw element count.

2. **SDIR bitmap scales with `d_in` × `d_out`:** The bitmap is `(d_out, d_in)` — at larger intermediate sizes, the bitmap grows quadratically in the expanded dimension.

3. **ffn_down residual disabled = margin buffer:** At 0.5B, the disabled ffn_down residual provides the margin. At larger models where Q2_K row-ceil grows, this margin may shrink or invert.

**Conservative conclusion:** The margin at 0.5B depends on the ratio `(Q4_budget - selected_policy_bytes) / selected_policy_bytes`. At larger shapes this ratio may decrease. The margin could approach zero at 7B scale without measurement.

---

## Runtime Feasibility Estimate

From Phase 31BM observed data:
- 384 pairs (24 layers × 16 seeds) completed in **~168 seconds** on 0.5B
- Rate: ~0.44 seconds/pair

Scaling by element count ratio:

| Model | Element ratio vs 0.5B | Projected pairs | Projected time | Feasibility |
|-------|----------------------|-----------------|---------------|-------------|
| 0.5B | 1.0× (baseline) | 384 | ~168s | ✓ Verified |
| 1.5B | ~4.6× per family | ~384 | ~770s (~13 min) | ✓ Feasible |
| 3B | ~3.8× per family | ~384 | ~640s (~11 min) | ✓ Feasible |
| 7B | ~17.3× per family | ~384 | ~2710s (~45 min) | ⚠ Slow but feasible |

> **Note:** Runtime estimates assume linear scaling with element count. Actual runtime may be higher due to ctypes call overhead and GGUF dequantization at larger tensor sizes.

---

## Disk / Artifact Feasibility

For a 384-pair validation on larger models:

| Component | Estimate (per pair) | Notes |
|-----------|---------------------|-------|
| Q2_K W_low artifacts | ~15 MB/pair | 3 families × corrected_ceil_per_row Q2_K |
| SDIR residual artifacts | ~1.1 MB/pair | 2 families × k=0.5% |
| JSON results | ~50 KB/pair | |
| **Total per pair** | **~16 MB** | |
| **384 pairs** | **~6.1 GB** | temporary, should not be committed |

**Recommendation:**
- Use a temporary directory for all artifacts during larger-model validation
- Clean up after phase completion
- Add `tmp_larger_model/` to `.gitignore` if created
- Do not commit any `.sdiw` or `.sdir` artifacts from larger-model runs

---

## Known Limitations

1. **No local 1.5B/3B/7B GGUF** — planning is based on public architecture specs, not measured metadata
2. **Row-ceil margin risk** — the memory margin at 0.5B may not hold at larger hidden sizes
3. **Validation is standalone tensor harness only** — no generation or behavior claim
4. **ctypes/gguf overhead** — runtime may be dominated by dequantization at larger element counts
5. **Private path debt in older scripts** — Phase 31AG–31AW remain, should not block package-based path

---

## Recommended Next Phase

**Phase 31BQ — Larger-Model Local Availability / Metadata Probe**

Rationale:
- No 1.5B/3B GGUF is locally available — probe must run first to confirm availability
- 3B intermediate_size is conflicting (8192 vs 11008) — GGUFReader metadata must be obtained before any validation
- 1.5B is the preferred probe target (smallest step from 0.5B), but availability must be confirmed first
- 31BQ must NOT proceed directly to validation — it must first establish what model files are actually present and read their metadata

**Phase 31BQ must include:**
1. Local GGUF availability scan — check standard paths (VL_usb, ~/.cache/huggingface/hub, any env-configured paths)
2. If 1.5B Q2_K GGUF is found locally → read its metadata with GGUFReader (n_layers, hidden_size, intermediate_size)
3. If 1.5B unavailable but 3B GGUF is found → read its metadata with GGUFReader (especially intermediate_size)
4. If neither is available → report BLOCKED_31BQ_NO_MODEL, recommend download approval path
5. No validation execution until metadata is confirmed

**Phase 31BQ forbidden claims:** Same as all phases — no larger-model result claim, no production readiness, no runtime integration claim.

---

## Proposed Route for 31BQ

### Route A — 1.5B Small Anchor Probe

| Parameter | Value |
|-----------|-------|
| Target | Qwen2.5-1.5B (if locally available) |
| Layers | L0, L6, L13, L20, L27 (5 layers) |
| Seeds | 0, 7 (2 seeds) |
| Total pairs | 10 |
| Expected runtime | ~4–5 minutes |
| Artifact size | ~160 MB (temporary, not committed) |

**Success criteria:**
- 10/10 memory-positive
- 0 severe regressions (delta_cos < −0.05)
- If passed → proceed to Phase 31BR (small aggregate)

**Stop conditions:**
- Any memory-negative pair → stop, report BLOCKED
- Any severe regression → stop, report PARTIAL with classification

### Route B — 3B Small Anchor Probe (if 1.5B unavailable)

| Parameter | Value |
|-----------|-------|
| Target | Qwen2.5-3B (if locally available) |
| Layers | L0, L8, L17, L26, L35 (5 layers) |
| Seeds | 0, 7 (2 seeds) |
| Total pairs | 10 |
| Expected runtime | ~3–4 minutes |
| Artifact size | ~130 MB (temporary, not committed) |

---

## Classification

**`PASS_31BP_LARGER_MODEL_FEASIBILITY_PLAN_READY`**

No larger-model validation was executed. Planning is complete. Next phase is clear: Phase 31BQ — Larger-Model Local Availability / Metadata Probe (only if explicitly requested) — no validation execution until metadata is confirmed.