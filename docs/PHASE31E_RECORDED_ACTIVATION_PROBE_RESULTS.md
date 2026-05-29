# Phase 31E: Recorded Activation Probe Results

**Classification:** `PASS_RECORDED_ACTIVATION_IMPROVEMENT`

## Executive Summary

Tests whether Phase 31D sparse residual encoding (bitmap + top-15% + fp16) improves over W_low under **real activation distributions** (not just seeded random X).

| Tensor | Control Delta (random X) | Recorded Delta (real activations) | PASS? |
|--------|------------------------|-----------------------------------|-------|
| ffn_up | +0.002501 | +0.003054 | ✅ |
| ffn_down | +0.002471 | +0.000936 | ✅ |

## Activation Capture

- **Method:** transformers hooks on Qwen2.5-0.5B layer 0 (hf model)
- **Gate input:** forward_pre_hook on layer0.mlp.gate_proj
- **Down proj output:** forward_hook on layer0.mlp.down_proj
- **CPU only:** yes
- **Prompts:** ['Hi', 'The capital of France is', '2+2=', 'def add(a, b):', 'Once upon a time']

## Memory Budget (31D encoding)

- **Encoding:** dense_bitmap_fp16 @ k=15%
- **ffn_up:** nnz=653,721, total_bytes=1,852,702
- **ffn_down:** nnz=653,721, total_bytes=1,852,216

## Per-Prompt Breakdown

| Prompt | Tensor | W_low Cosine | W_sub Cosine | Delta | MAE_low | MAE_sub | MAE Imprv |
|--------|--------|-------------|-------------|-------|---------|---------|----------|
| 'Hi' | ffn_up | 0.993443 | 0.995943 | +0.002500 | 0.000545 | 0.000427 | +0.000118 |
| 'Hi' | ffn_down | 0.997044 | 0.998218 | +0.001174 | 0.011829 | 0.009307 | +0.002522 |
| 'The capital of France is' | ffn_up | 0.992352 | 0.995268 | +0.002916 | 0.000512 | 0.000403 | +0.000108 |
| 'The capital of France is' | ffn_down | 0.998570 | 0.999128 | +0.000557 | 0.010429 | 0.007999 | +0.002430 |
| '2+2=' | ffn_up | 0.988491 | 0.993031 | +0.004540 | 0.000750 | 0.000586 | +0.000164 |
| '2+2=' | ffn_down | 0.998019 | 0.998846 | +0.000827 | 0.011947 | 0.009210 | +0.002737 |
| 'def add(a, b):' | ffn_up | 0.992765 | 0.995422 | +0.002657 | 0.000634 | 0.000506 | +0.000128 |
| 'def add(a, b):' | ffn_down | 0.997810 | 0.998609 | +0.000800 | 0.009146 | 0.007257 | +0.001889 |
| 'Once upon a time' | ffn_up | 0.993164 | 0.995821 | +0.002657 | 0.000588 | 0.000459 | +0.000130 |
| 'Once upon a time' | ffn_down | 0.996807 | 0.998128 | +0.001321 | 0.011123 | 0.008361 | +0.002762 |

## Mean & Worst-Case Across Prompts

- **ffn_up random control delta:** +0.002501
- **ffn_up recorded delta:** +0.003054
- **ffn_down random control delta:** +0.002471
- **ffn_down recorded delta:** +0.000936
- **ffn_up MAE improvement (recorded):** +0.000130
- **ffn_down MAE improvement (recorded):** +0.002468
- **ffn_up max-error improvement (recorded):** +0.000091
- **ffn_down max-error improvement (recorded):** +0.014139

## Decision Gate

```
ffn_up improvement (recorded):   +0.003054  > 0 → PASS
ffn_down improvement (recorded): +0.000936  > 0 → PASS
Classification: PASS_RECORDED_ACTIVATION_IMPROVEMENT
```

## Recommendation

✅ **Proceed to Phase 31F:** multi-layer recorded activation sweep (layers 0–2)
✅ **Proceed to Phase 31G:** substitutive compute prototype design