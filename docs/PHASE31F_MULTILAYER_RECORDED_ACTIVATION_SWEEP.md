# Phase 31F: Multi-Layer Recorded Activation Sweep Results

**Classification:** `PARTIAL_MEMORY_FAIL_ON_SOME_LAYERS`

## Metadata

- **Old HEAD:** `6f83e4a`
- **New HEAD:** `pending`
- **Layers tested:** [0, 1, 2]
- **Tensor families:** ['ffn_up', 'ffn_down']
- **Prompts:** ['Hi', 'The capital of France is', '2+2=', 'def add(a, b):', 'Once upon a time']
- **Total combos:** 30 (5 prompts × 6 tensors)

## Encoding Policy

Same as Phase 31D/31E: **dense_bitmap + global top-15% + fp16 residual values**

## Summary Metrics

- **Improving combos:** 30/30 (100.0%)
- **Mean delta cosine:** +0.001754
- **Worst-case delta:** +0.000005 (blk.2.ffn_down, prompt='Hi')
- **All memory viable:** False

## F. Memory Viability Table

| Tensor | Shape | Q4 (bytes) | Q2 (bytes) | Budget (bytes) | Encoded (bytes) | Viable? |
|--------|-------|-----------|-----------|----------------|-----------------|---------|
| blk.0.ffn_up | 4864×896 | 2,451,456 | 1,089,536 | 1,361,920 | 1,852,702 | ❌ |
| blk.0.ffn_down | 896×4864 | 2,451,456 | 1,089,536 | 1,361,920 | 1,852,216 | ❌ |
| blk.1.ffn_up | 4864×896 | 2,451,456 | 1,089,536 | 1,361,920 | 1,852,704 | ❌ |
| blk.1.ffn_down | 896×4864 | 2,451,456 | 1,089,536 | 1,361,920 | 1,852,280 | ❌ |
| blk.2.ffn_up | 4864×896 | 2,451,456 | 1,089,536 | 1,361,920 | 1,852,296 | ❌ |
| blk.2.ffn_down | 896×4864 | 2,451,456 | 1,089,536 | 1,361,920 | 1,852,332 | ❌ |

## A. Per-Prompt Table (delta cosine averaged across 6 tensors)

| Prompt | Mean ΔCos | Min ΔCos | Max ΔCos | Improving |
|--------|----------|---------|---------|----------|
| 'Hi' | +0.001598 | +0.000005 | +0.002500 | 6/6 (100%) |
| 'The capital of France is' | +0.001794 | +0.000005 | +0.003433 | 6/6 (100%) |
| '2+2=' | +0.002056 | +0.000006 | +0.004540 | 6/6 (100%) |
| 'def add(a, b):' | +0.001654 | +0.000006 | +0.002657 | 6/6 (100%) |
| 'Once upon a time' | +0.001668 | +0.000005 | +0.002657 | 6/6 (100%) |

## B. Per-Layer Table (delta cosine averaged across prompts & families)

| Layer | Mean ΔCos | Min ΔCos | ffn_up Δ | ffn_down Δ | Improving |
|-------|----------|---------|----------|------------|----------|
| 0 | +0.001995 | +0.000557 | +0.003054 | +0.000936 | 10/10 |
| 1 | +0.002196 | +0.001230 | +0.002879 | +0.001513 | 10/10 |
| 2 | +0.001071 | +0.000005 | +0.002137 | +0.000005 | 10/10 |

## C. Per-Tensor-Family Table (averaged across all layers & prompts)

| Family | Mean ΔCos | Min ΔCos | Max ΔCos | Improving |
|---------|----------|---------|---------|----------|
| ffn_up | +0.002690 | +0.001991 | +0.004540 | 15/15 |
| ffn_down | +0.000818 | +0.000005 | +0.001823 | 15/15 |

## D. Worst-Case Result

- **Layer:** 2
- **Family:** ffn_down
- **Prompt:** 'Hi'
- **delta_cosine:** +0.000005
- **cosine_ref_low:** 0.999986
- **cosine_ref_sub:** 0.999992

## E. Mean Delta

- **Mean delta cosine (all 30 combos):** +0.001754

## Decision Gate

```
memory_viable (all 6 tensors): False
improving_pct: 100.0% (need ≥80%)
mean_ffn_up_delta: +0.002690
mean_ffn_down_delta: +0.000818
Classification: PARTIAL_MEMORY_FAIL_ON_SOME_LAYERS
```

❌ **Memory budget exceeded for some layers**
  Recommendation: reduce k_pct or use row-wise sparsity