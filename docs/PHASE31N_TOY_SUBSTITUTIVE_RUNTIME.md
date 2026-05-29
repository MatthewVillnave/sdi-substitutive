# Phase 31N: Toy Substitutive Runtime Harness

**Classification:** `PASS_TOY_SUBSTITUTIVE_RUNTIME`

> **Note:** Synthetic weights used (GGUF loading ~4s header parse — too slow for standalone demo).
> Real GGUF extraction path documented for production use.

## Memory Counter Table

| Mode | W_ref_loaded | Dense R materialized | Encoded R | path_label |
|------|-------------|---------------------|-----------|------------|
| reference    | True  | 0 | 0   | [REF] |
| low_only     | False | 0 | 0   | [LOW-ONLY]  |
| substitutive | False | 0 | 1,198,516 | [SDI-SUB-RUNTIME] |

## Approximation Quality

| Metric | Value |
|--------|-------|
| cosine(Y_ref, Y_low)  | 0.99553198 |
| cosine(Y_ref, Y_sub)  | 0.99670637 |
| delta_cosine          | +0.00117439 |
| MAE_low               | 0.224378 |
| MAE_sub               | 0.192840 |
| max_error_low         | 1.108324 |
| max_error_sub         | 0.832080 |

## Classification Checks

- W_ref absent in substitutive:  ✅ (True)
- Dense R absent in substitutive: ✅ (0 bytes)
- delta_cosine > 0:             ✅ (+0.00117439)
- Fail-fast on missing residual: ✅

## Interpretation

- **delta_cosine > 0**: substitutive mode recovers 0.1174% more reference cosine than low-only baseline
- Substitutive mode uses **0 bytes** of dense residual (bitmap + fp16 sparse only)
- Substitutive path **[SDI-SUB-RUNTIME]** confirms W_ref never enters scope

## Decision Gate

**PASS_TOY_SUBSTITUTIVE_RUNTIME**

Next: Phase 31O (offline model artifact design) or Phase 31P (runtime across layers 0–5 ffn_up).

---
*Phase 31N — ELVIS — SDI Substitutive*
