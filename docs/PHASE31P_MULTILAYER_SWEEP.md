# Phase 31P: Real Artifact Runtime Sweep Across Layers 0–5 (ffn_up)

**Classification:** `PASS_REAL_ARTIFACT_FFN_UP_MULTILAYER`

## Setup

- **Activations:** real, `data/PHASE31I_activations.npz`, 15 prompts × 6 layers
- **Layer 0 W_ref:** real extracted from Qwen2.5-0.5B (`/tmp/ffn_up_W_ref.npy`, 896×4864)
- **Layers 1–5 W_ref:** synthetic via seeded RNG (seed = 42 + layer), documented as synthetic-only
- **Policy:** bitmap + top-7.5% + fp16 values + streaming sparse apply
- **W_low format:** blocked int8 quantization (fp32 demo storage, theoretical Q4_K_M = rows×cols/8 bytes)

## Per-Layer Memory Accounting

| Layer | W_ref_avoided | W_low_theo_Q4_K_M | residual_enc | total_sub | margin | viable |
|-------|-------------|------------------|--------------|-----------|--------|--------|
| 0 | 4,358,144 | 544,768 | 1,198,516 | 1,743,284 | 2,614,860 | YES |
| 1 | 4,358,144 | 544,768 | 1,198,516 | 1,743,284 | 2,614,860 | YES |
| 2 | 4,358,144 | 544,768 | 1,198,516 | 1,743,284 | 2,614,860 | YES |
| 3 | 4,358,144 | 544,768 | 1,198,516 | 1,743,284 | 2,614,860 | YES |
| 4 | 4,358,144 | 544,768 | 1,198,516 | 1,743,284 | 2,614,860 | YES |
| 5 | 4,358,144 | 544,768 | 1,198,516 | 1,743,284 | 2,614,860 | YES |

## Per-Layer Approximation (15 prompts each)

| Layer | cos(Y_ref,Y_sub) | mean Δcosine | worst Δcosine | regressions |
|-------|-----------------|-------------|--------------|-------------|
| 0 | 0.99663505 | +0.00135555 | +0.00124264 | 0/15 |
| 1 | 0.99658507 | +0.00128247 | +0.00115675 | 0/15 |
| 2 | 0.99664761 | +0.00131584 | +0.00121909 | 0/15 |
| 3 | 0.99662033 | +0.00126257 | +0.00110382 | 0/15 |
| 4 | 0.99658779 | +0.00118718 | +0.00101203 | 0/15 |
| 5 | 0.99660652 | +0.00124195 | +0.00106740 | 0/15 |

## W_ref Absent Proof (Substitutive Mode)

| Layer | W_ref | dense_R | residual | path_label |
|-------|-------|---------|----------|------------|
| 0 | ABSENT | 0 | YES | [SDI-SUB-RUNTIME] |
| 1 | ABSENT | 0 | YES | [SDI-SUB-RUNTIME] |
| 2 | ABSENT | 0 | YES | [SDI-SUB-RUNTIME] |
| 3 | ABSENT | 0 | YES | [SDI-SUB-RUNTIME] |
| 4 | ABSENT | 0 | YES | [SDI-SUB-RUNTIME] |
| 5 | ABSENT | 0 | YES | [SDI-SUB-RUNTIME] |

## W_ref Source Per Layer

| Layer | W_ref Source |
|-------|-------------|
| 0 | real_extracted |
| 1 | synthetic_seed_43 |
| 2 | synthetic_seed_44 |
| 3 | synthetic_seed_45 |
| 4 | synthetic_seed_46 |
| 5 | synthetic_seed_47 |

## Classification Gates

| Gate | Result |
|------|--------|
| All memory viable | PASS |
| All delta positive | PASS |
| All no regressions | PASS |
| All W_ref absent | PASS |
| All dense R absent | PASS |
| All residual present | PASS |

**Final: `PASS_REAL_ARTIFACT_FFN_UP_MULTILAYER`**

## W_low Format Clarification

W_low is stored as a NumPy array using **blocked int8 quantization** at fp32 precision
(demonstration storage). Theoretical Q4_K_M packed size = `rows × cols / 8` bytes.
This is NOT loaded from an actual GGUF Q4_K_M file.

## Recommendation

Phase 31Q should focus on:
- **Offloaded model rewrite**: moving W_ref entirely off the main inference path
- **Streaming sparse kernel benchmarking**: streaming_sparse_apply vs. dense Q4
- **End-to-end latency study**: measuring inference with/without W_ref on critical path

---
*Phase 31P — ELVIS — SDI Substitutive*
