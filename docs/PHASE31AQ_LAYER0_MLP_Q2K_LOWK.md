# Phase 31AQ — Layer0 Full MLP Q2_K + Low-k Residual

## Classification
**`PASS_LAYER0_MLP_Q2K_LOWK_POLICY_FOUND`**

## Status
Phase 31AQ complete.

## Source
Reused 31AP's official llama.cpp Q2_K quantizer path via ctypes.

## Reproducibility Artifact
`src/phase31aq_layer0_mlp_q2k_lowk.py` is a stub — see file for explanation.
Authoritative results: `docs/PHASE31AQ_LAYER0_MLP_Q2K_LOWK.md`
Machine-readable results: `results/PHASE31AQ_LAYER0_MLP_Q2K_LOWK.json`

## Metric Sign Conventions

| Metric | Formula | Positive means | Negative means |
|--------|---------|----------------|----------------|
| `delta_cos` | cos_sub − cos_low | cosine improved | cosine degraded |
| `MAE_delta` | MAE_sub − MAE_low | MAE worsened (higher error) | MAE improved (lower error) |

## Layer0 Q2_K Byte Table

| Family | Shape | n_elements | n_blocks | bytes | bpe | Q4 budget | margin | cos | MAE |
|--------|-------|-----------|---------|-------|-----|-----------|--------|-----|-----|
| ffn_up | 4864x896 | 4,358,144 | 17,024 | 1,430,016 | 2.6250 | 2,179,072 | 749,056 | 0.956674 | 0.003604 |
| ffn_gate | 4864x896 | 4,358,144 | 17,024 | 1,430,016 | 2.6250 | 2,179,072 | 749,056 | 0.954447 | 0.003210 |
| ffn_down | 896x4864 | 4,358,144 | 17,024 | 1,430,016 | 2.6250 | 2,179,072 | 749,056 | 0.954944 | 0.003004 |

## Per-Family Residual Table

| k | ffn_up margin | ffn_gate margin | ffn_down margin | ffn_up improves | ffn_gate improves | ffn_down improves |
|---|---------------|-----------------|-----------------|-----------------|-------------------|-------------------|
| 0 | 749,056 | 749,056 | 749,056 | NO | NO | NO |
| 0.5 | 160,708 | 160,708 | 160,708 | YES | YES | YES |
| 1 | 117,126 | 117,126 | 117,126 | YES | YES | YES |
| 2 | 29,964 | 29,964 | 29,964 | YES | YES | YES |
| 3 | -57,200 | -57,200 | -57,200 | YES | YES | YES |

## Full MLP Policy Table (uniform k%)

| k | total bytes | margin | cos_low | delta_cos | MAE_sub | improves | memory_positive |
|---|-------------|--------|---------|-----------|---------|----------|----------------|
| 0 | 4,290,048 | 2,247,168 | 0.886206 | +0.000000 | 0.038051 | NO | YES |
| 0.5 | 6,055,092 | 482,124 | 0.886206 | +0.008291 | 0.036402 | YES | YES |
| 1 | 6,185,838 | 351,378 | 0.886206 | +0.011172 | 0.036106 | YES | YES |
| 2 | 6,447,324 | 89,892 | 0.886206 | +0.016521 | 0.035156 | YES | YES |
| 3 | 6,708,816 | -171,600 | 0.886206 | +0.022604 | 0.033782 | YES | NO |

## Family Ablation (k=1%)

| config | margin | cos_low | delta_cos | MAE_delta | improves |
|--------|--------|---------|-----------|-----------|----------|
| ffn_up | 117,126 | 0.886206 | +0.001258 | -0.000122 | YES |
| ffn_gate | 117,126 | 0.886206 | +0.003135 | -0.000573 | YES |
| ffn_down | 117,126 | 0.886206 | +0.005035 | -0.001153 | YES |
| ffn_up+ffn_gate | 234,252 | 0.886206 | +0.004857 | -0.000781 | YES |
| ffn_up+ffn_down | 234,252 | 0.886206 | +0.006836 | -0.001390 | YES |
| ffn_gate+ffn_down | 234,252 | 0.886206 | +0.008961 | -0.001673 | YES |

## Classification Reasoning

- Classification: `PASS_LAYER0_MLP_Q2K_LOWK_POLICY_FOUND`
- Q2_K encode byte-exact for all three families: YES
- All families memory-positive at k=0: YES
- MLP residual improves cosine at some k: YES
- Aggregate MLP memory-positive at k≤2%: YES

## Best Passing Policy

- **k=2%** — margin=+89,892, delta_cos=+0.0165, MAE_delta=−0.00290 (improves)
- k=3% fails memory-positive (margin=−171,600)