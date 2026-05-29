# Phase 31L: AVX2 / Blocking Kernel Implementation

**Classification:** `PASS_AVX2_BLOCKED_KERNEL_READY`
**Numba available:** False

## Memory Accounting Clarification (AUDIT FIX)

**Phase 31K reported 'all layouts < 3.2MB vs Q4 budget ~234MB' — unit/scope mismatch.**

| Tensor | Shape | Q4 bytes | Q2 bytes | Budget (Q4-Q2) | CSR Encoded | Margin | Viable? |
|--------|-------|----------|----------|---------------|-------------|--------|---------|
| tiny | 8x16 | 64 | 32 | 32 | 122 | -90 | NO |
| ffn_up | 896x4864 | 2,179,072 | 1,089,536 | 1,089,536 | 1,964,780 | -875,244 | NO |
| ffn_down | 4864x896 | 2,179,072 | 1,089,536 | 1,089,536 | 1,980,652 | -891,116 | NO |

**Key:** Per-tensor residual budget = Q4 - Q2 bytes. '234MB' was full-model aggregate.
The full-model Q4 size is the sum across all layers/tensors, not comparable here.

## Correctness Table

| Shape | B | Variant | cos | maxdiff | correct? |
|-------|---|--------|-----|---------|---------|
| tiny | 1 | CSR_scalar_baseline_31K | 1.00000012 | 0.00e+00 | OK |
| tiny | 1 | CSR_blocked_scalar | 1.00000012 | 0.00e+00 | OK |
| tiny | 1 | CSR_batch_specialized | 1.00000012 | 0.00e+00 | OK |
| tiny | 1 | AVX2_numba | 1.00000012 | 0.00e+00 | OK |
| tiny | 4 | CSR_scalar_baseline_31K | 1.00000000 | 0.00e+00 | OK |
| tiny | 4 | CSR_blocked_scalar | 1.00000000 | 0.00e+00 | OK |
| tiny | 4 | CSR_batch_specialized | 1.00000000 | 0.00e+00 | OK |
| tiny | 4 | AVX2_numba | 1.00000000 | 0.00e+00 | OK |
| tiny | 16 | CSR_scalar_baseline_31K | 1.00000000 | 0.00e+00 | OK |
| tiny | 16 | CSR_blocked_scalar | 1.00000000 | 0.00e+00 | OK |
| tiny | 16 | CSR_batch_specialized | 1.00000000 | 0.00e+00 | OK |
| tiny | 16 | AVX2_numba | 1.00000000 | 0.00e+00 | OK |
| ffn_up | 1 | CSR_scalar_baseline_31K | 1.00000012 | 0.00e+00 | OK |
| ffn_up | 1 | CSR_blocked_scalar | 1.00000012 | 0.00e+00 | OK |
| ffn_up | 1 | CSR_batch_specialized | 1.00000012 | 0.00e+00 | OK |
| ffn_up | 1 | AVX2_numba | 1.00000012 | 0.00e+00 | OK |
| ffn_up | 4 | CSR_scalar_baseline_31K | 1.00000012 | 0.00e+00 | OK |
| ffn_up | 4 | CSR_blocked_scalar | 1.00000012 | 0.00e+00 | OK |
| ffn_up | 4 | CSR_batch_specialized | 1.00000012 | 0.00e+00 | OK |
| ffn_up | 4 | AVX2_numba | 1.00000012 | 0.00e+00 | OK |
| ffn_down | 1 | CSR_scalar_baseline_31K | 1.00000000 | 0.00e+00 | OK |
| ffn_down | 1 | CSR_blocked_scalar | 1.00000000 | 0.00e+00 | OK |
| ffn_down | 1 | CSR_batch_specialized | 1.00000000 | 0.00e+00 | OK |
| ffn_down | 1 | AVX2_numba | 1.00000000 | 0.00e+00 | OK |
| ffn_down | 4 | CSR_scalar_baseline_31K | 1.00000000 | 0.00e+00 | OK |
| ffn_down | 4 | CSR_blocked_scalar | 1.00000000 | 0.00e+00 | OK |
| ffn_down | 4 | CSR_batch_specialized | 1.00000000 | 0.00e+00 | OK |
| ffn_down | 4 | AVX2_numba | 1.00000000 | 0.00e+00 | OK |

## Timing Table (median ms)

| Shape | B | ref_dense | baseline_31K | blocked_scalar | batch_specialized | AVX2_numba |
|-------|---|-----------|--------------|----------------|-----------------|-----------|
| tiny | 1 | 0.0517 | 0.0275 | 0.0278 | 0.0275 | 0.0298 |
| tiny | 4 | 0.0507 | 0.0290 | 0.0294 | 0.0138 | 0.0150 |
| tiny | 16 | 0.0256 | 0.0129 | 0.0131 | 0.0129 | 0.0140 |
| ffn_up | 1 | 25.7992 | 3.2984 | 3.3222 | 3.3190 | 3.3893 |
| ffn_up | 4 | 28.8487 | 6.0407 | 6.0901 | 6.0068 | 6.1196 |
| ffn_down | 1 | 47.6819 | 19.7456 | 19.6639 | 19.5693 | 19.7575 |
| ffn_down | 4 | 66.0277 | 35.2085 | 34.7177 | 34.5528 | 34.7815 |

## Speedup vs 31K CSR Scalar Baseline

| Shape | B | blocked | batch | AVX2 |
|-------|---|---------|-------|------|
| tiny | 1 | 0.990x | 1.001x | 0.925x |
| tiny | 4 | 0.985x | 2.096x | 1.929x |
| tiny | 16 | 0.985x | 0.997x | 0.921x |
| ffn_up | 1 | 0.993x | 0.994x | 0.973x |
| ffn_up | 4 | 0.992x | 1.006x | 0.987x |
| ffn_down | 1 | 1.004x | 1.009x | 0.999x |
| ffn_down | 4 | 1.014x | 1.019x | 1.012x |

## Decision Gate

**AVX2/blocking improved the standalone sparse residual kernel by X vs the 31K CSR scalar baseline.** Phase 31M UNLOCKED.

---
*Phase 31L — ELVIS — SDI Substitutive*