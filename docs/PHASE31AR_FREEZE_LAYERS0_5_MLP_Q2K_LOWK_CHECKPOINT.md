# Phase 31AR-FREEZE — Layers 0–5 Full MLP Q2_K + Low-k Checkpoint

## Checkpoint Status: ACCEPTED

| Item | Value |
|------|-------|
| **HEAD** | `cc9fa391bc8f92525053c9a6f4298861d8434873` |
| **Tag** | `phase31ar-layers0-5-mlp-q2k-lowk-checkpoint` |
| **Classification** | `PASS_LAYERS0_5_MLP_Q2K_LOWK_POLICY_FOUND` |
| **Regression** | `PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN` |

---

## Accepted Checkpoint Claim

Standalone layers 0–5 full MLP substitutive prototype passes under the tested harness using official llama.cpp Q2_K W_low plus k=2% sparse residual via `encode_sdir` bitmap+fp16 format. All 18 Q2_K tensor checks pass byte-exactly (1,430,016 bytes/family), all 6 layers are memory-positive at k=2% (worst_layer=+89,524), and all 6 layers improve cosine and MAE over Q2_K-only.

---

## Forbidden Claims

- no model quality recovery
- no behavior recovery
- no speedup
- no full-model memory savings
- no llama.cpp integration claim
- no production readiness
- no all-layer/full-model claim beyond layers 0–5
- no checkpoint/tag until this phase completes

---

## Q2_K Byte Verification Summary

All 18 tensor checks PASS (byte_match=True for all layers 0–5, families up/gate/down):

| Layer | Family | Shape | bytes | bpe | margin | cos | MAE |
|-------|--------|-------|-------|-----|--------|-----|-----|
| 0 | ffn_up | 4864×896 | 1,430,016 | 2.625 | +749,056 | 0.957374 | 0.003565 |
| 0 | ffn_gate | 4864×896 | 1,430,016 | 2.625 | +749,056 | 0.955809 | 0.003114 |
| 0 | ffn_down | 896×4864 | 1,430,016 | 2.625 | +749,056 | 0.954944 | 0.003004 |
| 1 | ffn_up | 4864×896 | 1,430,016 | 2.625 | +749,056 | 0.957565 | 0.004486 |
| 1 | ffn_gate | 4864×896 | 1,430,016 | 2.625 | +749,056 | 0.956508 | 0.004118 |
| 1 | ffn_down | 896×4864 | 1,430,016 | 2.625 | +749,056 | 0.956474 | 0.002737 |
| 2 | ffn_up | 4864×896 | 1,430,016 | 2.625 | +749,056 | 0.957303 | 0.029311 |
| 2 | ffn_gate | 4864×896 | 1,430,016 | 2.625 | +749,056 | 0.956275 | 0.008963 |
| 2 | ffn_down | 896×4864 | 1,430,016 | 2.625 | +749,056 | 0.959742 | 0.000903 |
| 3 | ffn_up | 4864×896 | 1,430,016 | 2.625 | +749,056 | 0.957605 | 0.008311 |
| 3 | ffn_gate | 4864×896 | 1,430,016 | 2.625 | +749,056 | 0.957260 | 0.002563 |
| 3 | ffn_down | 896×4864 | 1,430,016 | 2.625 | +749,056 | 0.961716 | 0.001002 |
| 4 | ffn_up | 4864×896 | 1,430,016 | 2.625 | +749,056 | 0.956769 | 0.007022 |
| 4 | ffn_gate | 4864×896 | 1,430,016 | 2.625 | +749,056 | 0.955022 | 0.005245 |
| 4 | ffn_down | 896×4864 | 1,430,016 | 2.625 | +749,056 | 0.955833 | 0.002463 |
| 5 | ffn_up | 4864×896 | 1,430,016 | 2.625 | +749,056 | 0.957396 | 0.008858 |
| 5 | ffn_gate | 4864×896 | 1,430,016 | 2.625 | +749,056 | 0.954356 | 0.007294 |
| 5 | ffn_down | 896×4864 | 1,430,016 | 2.625 | +749,056 | 0.957133 | 0.002206 |

---

## Selected Policy: k=2%

| Metric | Value |
|--------|-------|
| **Per-layer margin** | +89,524 to +89,796 |
| **Aggregate margin** | +537,844 |
| **Worst layer margin** | +89,524 |
| **Avg delta_cos** | +0.01184 |
| **n_memory_positive** | 6/6 |
| **MAE** | Improves on all 6 layers |
| **Cosine** | Improves on all 6 layers |

---

## Per-Layer Results at k=2%

| Layer | margin | delta_cos | MAE_delta | memory_positive |
|-------|--------|-----------|-----------|-----------------|
| 0 | +89,586 | +0.01102 | −0.002306 | YES |
| 1 | +89,646 | +0.01671 | −0.003398 | YES |
| 2 | +89,680 | +0.00520 | −0.019328 | YES |
| 3 | +89,524 | +0.01120 | −0.001328 | YES |
| 4 | +89,796 | +0.01580 | −0.009643 | YES |
| 5 | +89,612 | +0.01111 | −0.011059 | YES |

---

## Aggregate Policy Table (all k values)

| k | agg_margin | worst_margin | n_mem_pos/6 | avg_delta_cos |
|---|------------|--------------|-------------|---------------|
| 0 | +13,483,008 | +2,247,168 | 6/6 | +0.00000 |
| 0.5 | +2,891,956 | +481,950 | 6/6 | +0.00583 |
| 1 | +2,106,902 | +351,076 | 6/6 | +0.00806 |
| **2** | **+537,844** | **+89,524** | **6/6** | **+0.01184** |
| 3 | −1,031,746 | −172,098 | 0/6 | +0.01708 |

---

## Known Limitations

- Checkpoint applies only to layers 0–5 in the standalone MLP toy harness
- No full-model (32 layers) claim
- No production readiness claim
- No llama.cpp integration claim beyond Q2_K encode/decode via ctypes
- Residual encoding uses `encode_sdir` (bitmap + fp16) — not a llama.cpp native format
- Results are harness-specific; actual inference behavior not tested
- Only Qwen2.5-0.5B shapes tested (4864×896 / 896×4864)

---

## Next Allowed Phase

**Phase 31AS — Full Model (all 32 layers) Q2_K + Low-k Residual Sweep, only if explicitly requested.**

---

## SOURCE_OF_TRUTH Reference

See `docs/PHASE31AR_LAYERS0_5_MLP_Q2K_LOWK_SWEEP.md` and `results/PHASE31AR_LAYERS0_5_MLP_Q2K_LOWK_SWEEP.json` for full sweep details.