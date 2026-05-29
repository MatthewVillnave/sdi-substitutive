# Phase 31G: Sparse Residual k-Sweep / Memory-Viable Policy Selection

**Classification:** `PASS_GLOBAL_MEMORY_VIABLE_POLICY`

## Metadata

- **Old HEAD:** `d21f814`
- **New HEAD:** `pending`
- **k values tested:** [5, 7.5, 8, 9, 10, 11, 12.5, 15]
- **Layers tested:** [0, 1, 2]
- **Tensor families:** ['ffn_up', 'ffn_down']
- **Prompts:** ['Hi', 'The capital of France is', '2+2=', 'def add(a, b):', 'Once upon a time']
- **Total combos per k:** 30 = 30 (5 prompts × 6 tensors)

## Encoding Policy

`dense_bitmap + top-k% + fp16 residual values`

## Summary

- **Best global k:** 7.5%
- **Adaptive policy:** N/A
- **Phase 31H unlocked:** True

## Table 1: k vs Memory Viability (per tensor)

| k% | L0.ffn_up | L0.ffn_down | L1.ffn_up | L1.ffn_down | L2.ffn_up | L2.ffn_down | ALL_VIABLE |
|---|---|---|---|---|---|---|
| 5% | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 7.5% | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 8% | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 9% | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 10% | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 11% | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 12.5% | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 15% | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

## Table 2: k vs Approximation Quality (delta cosine)

| k% | Mean ΔCos | Worst ΔCos | Improving% | All Memory Viable |
|-----|-----------|-----------|-----------|------------------|
| 5% | +0.000648 | +0.000002 | 100.0% | ✅ |
| 7.5% | +0.000936 | +0.000003 | 100.0% | ✅ |
| 8% | +0.000999 | +0.000003 | 100.0% | ✅ |
| 9% | +0.001120 | +0.000003 | 100.0% | ✅ |
| 10% | +0.001229 | +0.000004 | 100.0% | ❌ |
| 11% | +0.001343 | +0.000004 | 100.0% | ❌ |
| 12.5% | +0.001506 | +0.000004 | 100.0% | ❌ |
| 15% | +0.001754 | +0.000005 | 100.0% | ❌ |

## Best Global k Policy

- **k:** 7.5%
- **Mean ΔCos:** +0.000936
- **Worst ΔCos:** +0.000003
- **Improving:** 100.0% of combos
- **All Memory Viable:** True

## Decision Gate

```
classification: PASS_GLOBAL_MEMORY_VIABLE_POLICY
best_global_k: 7.5
adaptive_policy: {}
phase31h_unlocked: True
```

✅ **Phase 31H unlocked.** Proceed to compressed residual compute harness.