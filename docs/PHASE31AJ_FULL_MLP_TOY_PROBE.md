# Phase 31AJ — Full MLP Toy Probe Results

## Classification
**`PARTIAL_MLP_APPROX_PASS_MEMORY_FAIL`**

Full MLP toy composition improves approximation over low-only. Source equivalence is clean. Strict counters are clean. MLP math and residual staging are correct. Individual tensor/family rows are under budget.

Combined up+gate+down MLP artifact total exceeds the Q4 combined budget. Therefore this is not memory-positive and not a full pass.

---

## Scope
- **Layers**: 0–5 (6 layers)
- **Families**: ffn_up + ffn_gate + ffn_down
- **Policies**: ffn_up k=9%, ffn_down k=9%, ffn_gate k=12%

---

## MLP Formula

```
Y = (SiLU(X @ W_gate.T) * (X @ W_up.T)) @ W_down.T
```

### Residual Staging (correct)
| Stage | Residual Applied | When |
|--------|-----------------|------|
| up residual | before SiLU/gating | `up_s = sdiw_streaming_apply(...) + sdir_streaming_apply(...)` |
| gate residual | before SiLU | `gate_s = sdiw_streaming_apply(...) + sdir_streaming_apply(...)` |
| down residual | after hidden is formed | `down_s = sdiw_streaming_apply(hidden,...) + sdir_streaming_apply(hidden,...)` |

---

## Approximation Quality (all 6 layers)

All layers show improvement over low-only (delta positive on cosine, delta positive on MAE reduction).

| Layer | cos_low | cos_sub | delta_cos | mae_low | mae_sub | delta_mae |
|-------|---------|---------|-----------|---------|---------|-----------|
| 0 | 0.995945 | 0.998256 | +0.002312 | 0.007523 | 0.005308 | +0.002215 |
| 1 | 0.992987 | 0.996140 | +0.003153 | 0.008697 | 0.006605 | +0.002092 |
| 2 | 0.999780 | 0.999988 | +0.000208 | 0.140498 | 0.068465 | +0.072033 |
| 3 | 0.999913 | 0.999994 | +0.000081 | 0.022402 | 0.011742 | +0.010660 |
| 4 | 0.988271 | 0.994706 | +0.006436 | 0.025722 | 0.018141 | +0.007581 |
| 5 | 0.999271 | 0.999690 | +0.000419 | 0.046936 | 0.032897 | +0.014039 |

---

## Source Equivalence

Verified on layer 0 for each family. All three tests (sdiw, sdir, combined) pass within threshold (cosine ≥ 0.999999, MAE < 1e-4).

| Family | sdiw cosine | sdir cosine | combined cosine | Status |
|--------|-------------|-------------|-----------------|--------|
| ffn_up | 0.999999881 | 0.999999940 | 1.000000000 | ✓ pass |
| ffn_gate | 1.000000000 | 1.000000000 | 1.000000000 | ✓ pass |
| ffn_down | 1.000000000 | 0.999999940 | 1.000000000 | ✓ pass |

---

## Strict Counters

```
W_ref_loaded:             0
W_ref_generated:          0
dense_W_low_materialized:  0
dense_R_materialized:     0
sdiw_loaded:             18
sdir_loaded:             18
fallback_count:           0
error_count:              0
```

---

## Memory Analysis

### Per-Row Margins (individual — all positive, under budget)
| Layer | Family | Total Sub (bytes) | Q4 Budget (bytes) | Margin (bytes) |
|-------|--------|-------------------|------------------|---------------|
| 0 | ffn_up | 3,780,824 | 2,179,072 | +1,601,752 |
| 0 | ffn_gate | 4,042,262 | 2,179,072 | +1,863,190 |
| 0 | ffn_down | 3,780,784 | 2,179,072 | +1,601,712 |
| 1 | ffn_up | 3,782,528 | 2,179,072 | +1,603,456 |
| 1 | ffn_gate | 4,044,556 | 2,179,072 | +1,865,484 |
| 1 | ffn_down | 3,783,818 | 2,179,072 | +1,603,818 |
| 2 | ffn_up | 3,783,148 | 2,179,072 | +1,603,148 |
| 2 | ffn_gate | 4,044,068 | 2,179,072 | +1,864,996 |
| 2 | ffn_down | 3,781,950 | 2,179,072 | +1,602,878 |
| 3 | ffn_up | 3,783,366 | 2,179,072 | +1,604,294 |
| 3 | ffn_gate | 4,043,442 | 2,179,072 | +1,864,370 |
| 3 | ffn_down | 3,781,798 | 2,179,072 | +1,602,726 |
| 4 | ffn_up | 3,782,464 | 2,179,072 | +1,603,392 |
| 4 | ffn_gate | 4,045,450 | 2,179,072 | +1,866,378 |
| 4 | ffn_down | 3,782,082 | 2,179,072 | +1,603,010 |
| 5 | ffn_up | 3,781,698 | 2,179,072 | +1,602,626 |
| 5 | ffn_gate | 4,043,968 | 2,179,072 | +1,864,896 |
| 5 | ffn_down | 3,781,990 | 2,179,072 | +1,602,918 |

All 18 rows: individual margins positive ✓

### Aggregate (combined MLP — FAILS)
| Metric | Value |
|--------|-------|
| Total substitutive bytes | 69,638,132 |
| Q4 budget bytes | 39,223,296 |
| **Aggregate margin bytes** | **−30,414,836** |
| combined_mlp_memory_positive | **false** |

**Memory failure reason**: Combined up+gate+down substitutive artifacts (69.6 MB) exceed the combined Q4 budget (39.2 MB) by ~30.4 MB. Individual row margins are positive but the sum of all 18 artifacts does not fit within the Q4 memory envelope.

---

## What This Result Does NOT Claim

The following claims are **not supported** by this phase:
- NOT `PASS_FULL_MLP_TOY_PROBE`
- NOT "memory-positive" for combined MLP composition
- NOT "quality recovery" or "model behavior recovery"
- NOT "speedup" or "full-model memory savings"
- NOT "integration-ready"

---

## Script Portability Notes
- Hardcoded model path: `/media/matthew-villnave/VL_usb/models/...` — configurable via `SDI_GGUF_MODEL_PATH` env var
- `gguf` module required for W_ref extraction — fails with actionable error if missing
- Local reproducibility: requires USB model mount or pre-cached artifacts

---

## Recommended Next Phase
**Phase 31AK** — Full MLP artifact budget/economics fix, only if explicitly requested.
