# Phase 31BH-R2 — Q2_K Anchor Reproduction Fix / Floor-vs-Ceil Decision

**Classification:** `PASS_31BH_R2_HISTORICAL_ANCHORS_REPRODUCED`
**Date:** Phase 31BH-R2
**Repo:** sdi-substitutive

---

## Goal

Fix Phase 31BH runner to reproduce accepted historical anchors (31AY/31BA) and evaluate two Q2_K buffer modes.

---

## Core Questions Answered

1. **Can `historical_floor_flat` reproduce 31AY/31BA anchors?** ✓ YES
2. **Can `corrected_ceil_per_row` reproduce 31AY/31BA anchors?** ✗ NO
3. **Which Q2_K mode is technically correct?** `corrected_ceil_per_row` (pads partial blocks)
4. **What is the provenance of accepted Phase 31 metrics?** Historical — they are `historical_floor_flat`-provenance

---

## Root Cause of Previous Failure (31BH-R)

The 31BH runner had THREE bugs:

### Bug 1: Wrong X RNG
- **Problem:** Used `np.random.RandomState(seed)` instead of `np.random.default_rng(seed)`
- **Effect:** Different X vector → different activation → wrong cosines
- **Fix:** Changed to `np.random.default_rng(seed)` (matches OLD 31AY/31BA path)

### Bug 2: Wrong W_ref Provenance
- **Problem:** Used raw GGUF dequantized weights as `W_ref`
- **Effect:** In OLD 31AY, `W_ref` was Q2_K roundtrip weights, not raw
- **Fix:** W_ref = raw GGUF weights; W_low = Q2_K roundtrip; Residual = W_ref - W_low

### Bug 3: Incomplete MLP Evaluation
- **Problem:** Only applied residual to `ffn_up`, missing `ffn_gate` and `ffn_down`
- **Effect:** Wrong `delta_cos` and `MAE_delta`
- **Fix:** Apply SDIR residual to ALL 3 MLP families (matching OLD 31AY formula)

---

## X Vector Fingerprints

**Fixed:** `np.random.default_rng(seed)`

| seed | shape | dtype | norm | SHA256 |
|------|-------|-------|------|--------|
| 0 | (1, 896) | float32 | 29.179718 | 550ad1f0... |
| 9 | (1, 896) | float32 | 29.409826 | b4c9c8e3... |

---

## Anchor Reproduction Table

### L21-S9 (seed=9)

| Mode | Q2_K bytes | cos_low | expected | cos_sub | delta_cos | expected | MAE_delta | severe |
|------|-----------|---------|----------|---------|-----------|----------|-----------|--------|
| `historical_floor_flat` | 1,430,016 | **0.794913** ✓ | 0.794913 | 0.648854 | **-0.146059** ✓ | -0.146059 | **-0.000479** ✓ | True ✓ |
| `corrected_ceil_per_row` | 1,634,304 | 0.646499 | 0.794913 | 0.739660 | +0.093161 | -0.146059 | -0.013912 | False |

### L21-S0 (seed=0)

| Mode | Q2_K bytes | cos_low | expected | cos_sub | delta_cos | expected | MAE_delta | severe |
|------|-----------|---------|----------|---------|-----------|----------|-----------|--------|
| `historical_floor_flat` | 1,430,016 | **0.855612** ✓ | 0.855612 | 0.873457 | **+0.017845** ✓ | +0.017845 | **-0.003169** ✓ | False ✓ |
| `corrected_ceil_per_row` | 1,634,304 | 0.420949 | 0.855612 | 0.815612 | +0.394664 | +0.017845 | -0.025622 | False |

---

## W_low Byte Comparison

**L21-S9 and L21-S0** (same W_low SHA256 for same mode):

| Mode | Bytes | SHA256 |
|------|-------|--------|
| `historical_floor_flat` | 1,430,016 | ced92887a9e34aaa0c9f736c04f4bfd0 |
| `corrected_ceil_per_row` | 1,634,304 | 167adc9982afcb72d00e7965e0d22d9b |

Difference: **+204,288 bytes (+14.3%)** for `corrected_ceil_per_row`

---

## Provenance Decision

### `historical_floor_flat` mode
- **Reproduces accepted anchors:** YES ✓
- **Bytes:** 1,430,016 (floor — truncates last 128 elements per row)
- **Status:** Historical compatibility mode — matches Phase 31AY/31BA provenance exactly
- **WARNING:** Last partial Q2_K block (128 elements per row) is silently DROPPED in both encode and decode

### `corrected_ceil_per_row` mode
- **Reproduces accepted anchors:** NO ✗
- **Bytes:** 1,634,304 (ceil — pads each row to complete Q2_K blocks)
- **Status:** Technically correct future mode — no element truncation
- **Note:** Does NOT match Phase 31AY/31BA metrics; future retesting recommended

---

## MLP Evaluation Formula (Confirmed)

OLD 31AY/31BA formula (all residual applied to all 3 MLP families):

```
Y_ref  = mlp(X, W_gate_raw,  W_up_raw,  W_down_raw)
Y_low  = mlp(X, W_gate_q2k,  W_up_q2k,  W_down_q2k)    # Q2_K roundtrip for all families
R      = W_ref - W_low  (per family)
Y_sub  = mlp(X, W_gate_q2k+dec(R_gate), W_up_q2k+dec(R_up), W_down_q2k+dec(R_down))
```

Where:
- `mlp(X, Wg, Wu, Wd) = silu(X @ Wg.T) * (X @ Wu.T) @ Wd.T`
- `dec(R) = decode_sdir(encode_sdir(R, k_pct=1.0))`
- `W_gate_q2k` and `W_down_q2k` are Q2_K roundtrip of the same mode

---

## Files Changed

| File | Change |
|------|--------|
| `src/q2k_backend.py` | Added dual mode support: `historical_floor_flat` and `corrected_ceil_per_row` |
| `src/phase31bh_q2k_clean_reproduction.py` | Fixed X RNG, MLP formula, all-family residual |
| `docs/STATIC_ARTIFACT_SCHEMA.md` | Added `q2_k_backend_mode` field documentation |
| `docs/REPRODUCIBILITY_NOTES.md` | Updated with Q2_K mode provenance notes |
| `SOURCE_OF_TRUTH.md` | Updated with 31BH-R2 results |
| `docs/PHASE31BH_R2_Q2K_ANCHOR_REPRO_FIX.md` | This document |
| `results/PHASE31BH_R2_Q2K_ANCHOR_REPRO_FIX.json` | Machine-readable results |

---

## Next Phase (only if explicitly requested)

- **Phase 31BJ** — Corrected Q2_K Mode Rebaseline: if using `corrected_ceil_per_row` as future canonical mode
- **Phase 31BI** — Full 31BA Aggregate Reproduction Using Q2_K Backend: if verifying all 31BA metrics with corrected runner
- **Phase 31BH-R3** — Anchor Provenance Repair: if additional provenance issues arise

---

## Forbidden Claims (Upheld)

- no model quality recovery
- no behavior recovery
- no speedup
- no full-model runtime memory savings claim
- no llama.cpp integration
- no production readiness
- no inference/generation
- no larger-model claim
- no runtime-ready output-residual
- no claim beyond standalone tensor harness
