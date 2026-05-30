# Phase 31AD: Policy Decision + Strict Runtime Validation

## Header
- **Phase:** 31AD
- **Date:** 2026-05-30
- **Classification:** `PASS_POLICY_SELECTED_K9`
- **OLD_HEAD:** `5f820dd`
- **Repo:** `sdi-substitutive`

## Decision: k=9% selected as official residual policy

## Margin Thresholds
- **Hard minimum:** margin > 0 bytes
- **Preferred minimum:** margin >= 256KB per layer
- **k=9% margin:** 305,042 bytes per layer — clears both thresholds

## Policy Candidates Compared

| Policy | avg_cos | avg_MAE | delta_cos | MAE_delta | min_margin | cos_pos | mae_ok | pref_ok |
|--------|---------|---------|-----------|-----------|------------|---------|--------|---------|
| k=7% | 0.995591 | 0.0704 | +0.0002 | +0.0042 | 479,366 | ✓ | ✗ | ✓ |
| **k=8%** | **0.995708** | **0.0697** | **+0.0004** | **+0.0035** | **392,202** | ✓ | ✗ | ✓ |
| **k=9%** | **0.995833** | **0.0688** | **+0.0005** | **+0.0026** | **305,042** | ✓ | ✗ | ✓ |

## Layer-by-Layer Results

### k=7% (reference)

| L | cos_low | cos_sub | delta | MAE_low | MAE_sub | MAE_d | margin |
|---|---------|---------|-------|---------|---------|-------|--------|
| 0 | 0.995337 | 0.995586 | +0.0002 | 0.0657 | 0.0697 | +0.0040 | 479,368 |
| 1 | 0.995335 | 0.995589 | +0.0003 | 0.0672 | 0.0714 | +0.0042 | 479,368 |
| 2 | 0.995354 | 0.995605 | +0.0003 | 0.0655 | 0.0700 | +0.0045 | 479,368 |
| 3 | 0.995515 | 0.995813 | +0.0003 | 0.0654 | 0.0697 | +0.0043 | 479,368 |
| 4 | 0.995254 | 0.995613 | +0.0004 | 0.0666 | 0.0698 | +0.0032 | 479,366 |
| 5 | 0.995263 | 0.995344 | +0.0001 | 0.0668 | 0.0717 | +0.0049 | 479,368 |

**avg cos=0.995591 | avg MAE=0.0704 | margin=479,366 per layer**

### k=9% (selected)

| L | cos_low | cos_sub | delta | MAE_low | MAE_sub | MAE_d | margin |
|---|---------|---------|-------|---------|---------|-------|--------|
| 0 | 0.995337 | 0.995804 | +0.0005 | 0.0657 | 0.0685 | +0.0028 | 305,044 |
| 1 | 0.995335 | 0.995875 | +0.0005 | 0.0672 | 0.0695 | +0.0023 | 305,044 |
| 2 | 0.995354 | 0.995815 | +0.0005 | 0.0655 | 0.0684 | +0.0029 | 305,044 |
| 3 | 0.995515 | 0.996033 | +0.0005 | 0.0654 | 0.0685 | +0.0031 | 305,044 |
| 4 | 0.995254 | 0.995807 | +0.0006 | 0.0666 | 0.0686 | +0.0020 | 305,042 |
| 5 | 0.995263 | 0.995664 | +0.0004 | 0.0668 | 0.0695 | +0.0027 | 305,044 |

**avg cos=0.995833 | avg MAE=0.0688 | margin=305,042 per layer**

## Why k=9% over k=7%

1. **Cosine improvement:** 0.995833 vs 0.995591 — +0.000242 better
2. **MAE improvement:** 0.0688 vs 0.0704 — -0.0016 better (less regression)
3. **Margin still positive:** 305K > 256KB preferred threshold
4. **All 6 layers cos-positive:** cosine improves on every layer at k=9%

## Tradeoffs Documented
- k=9% consumes ~36% less margin than k=7% (305K vs 479K)
- k=7% has wider safety margin but weaker approximation
- MAE regression persists at k=9% but is materially reduced (+0.0026 vs +0.0042)
- Alpha scaling confirmed non-helpful; not used

## Claim Boundaries
**Allowed:** "k=9% selected as official policy; clears cosine research gate (0.9958 > 0.995); margin positive at 305K per layer."
**Forbidden:** no model quality, no behavior recovery, no speedup.

## Next Phase
Phase 31AE: ffn_down strict substitutive feasibility
