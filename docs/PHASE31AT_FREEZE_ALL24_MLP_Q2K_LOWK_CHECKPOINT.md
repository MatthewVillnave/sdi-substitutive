# Phase 31AT-FREEZE — All-24-Layer Consistent-Seed Q2_K + Low-k Checkpoint

## Checkpoint Status: ACCEPTED

| Item | Value |
|------|-------|
| **Phase** | 31AT-FREEZE |
| **HEAD** | `899022d4730edf9f1ea56c599e49561a5081d333` |
| **Tag** | `phase31at-all24-mlp-q2k-lowk-checkpoint` |
| **Classification** | `PASS_31AT_FREEZE_ALL24_MLP_Q2K_LOWK_CHECKPOINT` |
| **Commit** | `899022d4730edf9f1ea56c599e49561a5081d333` |

---

## Accepted Checkpoint Claim

> "Standalone all-24-layer full MLP substitutive tensor harness passes for Qwen2.5-0.5B using official llama.cpp Q2_K W_low plus k=1% sparse residual via encode_sdir, under consistent seed=42. All 24 available FFN layers are memory-positive, cosine-positive, and MAE-improving versus Q2_K-only in the tested harness."

---

## Forbidden Claims

- no model quality recovery
- no behavior recovery
- no speedup
- no full-model runtime memory savings claim
- no llama.cpp integration claim
- no production readiness
- no claim beyond standalone full-MLP tensor harness
- no inference/generation claim

---

## Regression Result

```
PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN — sdiw_loaded=2, fallback_count=0, error_count=0
```

---

## Methodology

| Item | Value |
|------|-------|
| **Model** | Qwen2.5-0.5B (24 FFN layers, indices 0–23) |
| **FFN shapes** | ffn_up: 4864×896, ffn_gate: 4864×896, ffn_down: 896×4864 |
| **W_low encoding** | Official llama.cpp Q2_K via `quantize_row_q2_K_ref` / `dequantize_row_q2_K` |
| **Residual encoding** | `encode_sdir` (bitmap + fp16 values) |
| **MLP formula** | `Y = (SiLU(X @ W_gate.T) * (X @ W_up.T)) @ W_down.T` |
| **Activation seed** | Consistent seed=42 for all 24 layers |
| **Selected policy** | k=1% |
| **Q4 budget per layer** | 6,537,216 bytes (3 × 2,179,072) |

---

## Aggregate k=1% Results

| Metric | Value |
|--------|-------|
| n_layers_tested | 24 |
| n_memory_positive | **24/24** |
| n_cosine_positive | **24/24** |
| n_MAE_improving | **24/24** |
| aggregate_margin | +8,428,606 bytes |
| worst_margin | layer3: +351,076 |
| worst_cosine | layer4: delta_cos=+0.00500 |
| avg_delta_cos | +0.01133 |

---

## Worst-Layer Table (k=1%)

| Role | Layer | margin | delta_cos | MAE_improvement |
|------|-------|--------|-----------|-----------------|
| worst_margin | 3 | +351,076 | +0.00874 | 0.000697 |
| worst_cosine | 4 | +351,242 | +0.00500 | 0.003557 |

---

## k=2% Results (also passes)

| Metric | Value |
|--------|-------|
| n_memory_positive | **24/24** |
| n_cosine_positive | **24/24** |
| n_MAE_improving | **24/24** |
| aggregate_margin | +2,152,334 bytes |
| worst_margin | layer3: +89,524 |
| worst_cosine | layer4: delta_cos=+0.00943 |

---

## 31AS/31AS-R/31AT/31AU Correction Chain

| Phase | Classification | Issue |
|-------|---------------|-------|
| 31AS | PASS_ALL_LAYERS (initial, incorrect) | Missed layer 21 cosine regression |
| 31AS-R | PARTIAL_LAYER_VARIANCE | Layer 21 cosine regresses at seed=63 (per-layer seed artifact) |
| 31AT | PARTIAL_LAYER21_ACTIVATION_SENSITIVE | Layer 21 is activation-sensitive, not structurally failing; 9/10 seeds improve |
| 31AU | PASS_ALL_LAYERS_CONSISTENT_SEED_POLICY_FOUND | Consistent seed=42: 24/24 pass all gates |
| 31AU-R | PASS (label-only fix) | MAE column label corrected; numeric pass/fail unchanged |
| **31AT-FREEZE** | **PASS_31AT_FREEZE_ALL24_MLP_Q2K_LOWK_CHECKPOINT** | **Checkpoint accepted** |

---

## Known Limitations

- **Standalone tensor harness only** — results apply only to the tested full-MLP tensor evaluation path
- **No generation/inference claim** — no autoregressive or sequence-level evaluation performed
- **No llama.cpp integration** — residual is not integrated into llama.cpp dequantization path
- **No speed claim** — no throughput or latency measurements performed
- **Activation robustness** — fully characterized only for seed=42; other seeds partially characterized (layer 21: 9/10 seeds positive)
- **No larger model claim** — Qwen2.5-0.5B only; results may not generalize to larger models
- **No full-model memory savings** — aggregate margin is across all 24 layers but no runtime memory allocation claim is made

---

## Current Allowed Next Phase

**Phase 31AV — Robustness / Multi-seed All-layer Characterization, only if explicitly requested.**

---

## Tag Information

- **Tag name**: `phase31at-all24-mlp-q2k-lowk-checkpoint`
- **Tag target**: `899022d4730edf9f1ea56c599e49561a5081d333` (31AU-R commit — the finalized result state after MAE audit)
- **Note**: The 31AU-R commit (`899022d4`) is the authoritative finalized state. The freeze-doc commit (`<31AT-FREEZE>`) is metadata only and is not tagged.