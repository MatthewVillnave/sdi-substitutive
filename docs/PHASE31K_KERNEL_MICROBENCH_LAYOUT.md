# Phase 31K: Kernel Microbenchmark + Layout Optimization

## Summary

- **Classification:** `PARTIAL_BITMAP_MEMORY_OK_SPEED_BAD`
- **Benchmark shapes:** tiny (8×16), attn_out (896×896), ffn_up (896×4864), ffn_down (4864×896)
- **k_pct:** 7.5%
- **Warmup/Timed iters:** 5/30-50 (varies by shape)
- **Seed:** 42
- **Memory budget source:** Phase 31G/31H real GGUF measurements (attn_out: Q4=551936/Q2=451584/spare=100352; ffn: Q4=2451456/Q2=1089536/spare=1361920)

## Layouts Tested

| Letter | Name | Description | Bitmap Scan? |
|--------|------|-------------|--------------|
| A | bitmap scan (baseline) | Unpack bitmap, scan all bits, consume fp16 values for set bits | ✅ (reference) |
| B | index list | Precomputed flat uint32 indices, no bitmap scan during compute | ❌ |
| C | row-wise sparse lists | Per-row offsets + column indices + fp16 values | ❌ |
| D | block sparse 4×4 | 4×4 block occupancy + sub-bitmaps (optional, correctness issues) | Partial |

## Memory Viability Table

**Budget source:** Real GGUF measurements from Phase 31G (attn_out) and Phase 31H (ffn_up/down).
"Memory viable" means residual encoding fits within the spare budget between Q2 and Q4 storage.

| Shape | Layout | Total Bytes | Budget | Margin | Viable? |
|-------|--------|-------------|--------|--------|---------|
| tiny (8×16) | A_bitmap | 66 | 16 | -50 | ❌ |
| tiny (8×16) | B_index_list | 86 | 16 | -70 | ❌ |
| tiny (8×16) | C_row_wise | 122 | 16 | -106 | ❌ |
| tiny (8×16) | D_block_sparse_4x4 | 92 | 16 | -76 | ❌ |
| attn_out (896×896) | A_bitmap | 220,806 | 100,352 | -120,454 | ❌ |
| attn_out (896×896) | B_index_list | 361,298 | 100,352 | -260,946 | ❌ |
| attn_out (896×896) | C_row_wise | 364,886 | 100,352 | -264,534 | ❌ |
| attn_out (896×896) | D_block_sparse_4x4 | 387,641 | 100,352 | -287,289 | ❌ |
| ffn_up (896×4864) | A_bitmap | 1,198,520 | 1,361,920 | +163,400 | ✅ |
| ffn_up (896×4864) | B_index_list | 1,961,192 | 1,361,920 | -599,272 | ❌ |
| ffn_up (896×4864) | C_row_wise | 1,964,780 | 1,361,920 | -602,860 | ❌ |
| ffn_up (896×4864) | D_block_sparse_4x4 | 2,104,196 | 1,361,920 | -742,276 | ❌ |
| ffn_down (4864×896) | A_bitmap | 1,198,520 | 1,361,920 | +163,400 | ✅ |
| ffn_down (4864×896) | B_index_list | 1,961,192 | 1,361,920 | -599,272 | ❌ |
| ffn_down (4864×896) | C_row_wise | 1,980,652 | 1,361,920 | -618,732 | ❌ |
| ffn_down (4864×896) | D_block_sparse_4x4 | 2,104,196 | 1,361,920 | -742,276 | ❌ |

**Key finding:** Layout A (bitmap) is the **only** memory-viable layout, and only on the larger FFN shapes (ffn_up, ffn_down). The smaller shapes (tiny, attn_out) have no memory-viable layout at k=7.5% under these budgets.

## Timing Table (median ms, B=1)

| Shape | A_ref (ms) | B_index (ms) | B vs A | C_row (ms) | C vs A | D_block (ms) | D vs A |
|-------|------------|-------------|--------|------------|--------|--------------|--------|
| tiny | 0.0312 | 0.0413 | 0.755× | 0.0186 | **1.682×** | 0.0316 | 0.987× |
| attn_out | 7.3761 | 27.6331 | 0.267× | 2.3872 | **3.090×** | 213.27 | 0.035× |
| ffn_up | 26.7459 | 161.1307 | 0.166× | 3.4603 | **7.729×** | 1157.29 | 0.023× |
| ffn_down | 50.2539 | 917.9067 | 0.055× | 20.7147 | **2.426×** | 1169.74 | 0.043× |

**Key findings:**
- **Layout C (row-wise)** is consistently 1.7-7.7× faster than Layout A across all shapes
- **Layout B (index list)** is 3-20× slower than Layout A due to per-element range checks on large matrices
- **Layout D (block sparse)** is extremely slow in Python due to the double-nested block iteration

## Correctness Table (vs Layout A reference)

| Shape | Layout | max_abs_diff | cosine | correct? |
|-------|--------|-------------|--------|---------|
| tiny | B_index_list | 0.00e+00 | 1.00000000 | ✅ |
| tiny | C_row_wise | 0.00e+00 | 1.00000000 | ✅ |
| tiny | D_block_sparse_4x4 | 2.31e-03 | -0.126 | ❌ |
| attn_out | B_index_list | 0.00e+00 | 1.00000000 | ✅ |
| attn_out | C_row_wise | 0.00e+00 | 1.00000000 | ✅ |
| attn_out | D_block_sparse_4x4 | 4.43e-02 | 0.048 | ❌ |
| ffn_up | B_index_list | 0.00e+00 | 1.00000000 | ✅ |
| ffn_up | C_row_wise | 0.00e+00 | 1.00000000 | ✅ |
| ffn_up | D_block_sparse_4x4 | 9.72e-02 | -0.014 | ❌ |
| ffn_down | B_index_list | 0.00e+00 | 1.00000000 | ✅ |
| ffn_down | C_row_wise | 0.00e+00 | 1.00000000 | ✅ |
| ffn_down | D_block_sparse_4x4 | 9.55e-02 | 0.005 | ❌ |

**Main layouts (A, B, C):** All correct on all shapes. max_abs_diff=0.0 for B and C vs A reference.
**Layout D:** Fails correctness on all shapes — excluded from main analysis.

## Timing Matrix (all batch sizes, median ms)

| Shape | B | A_ref | B_index | C_row | D_block |
|-------|---|-------|---------|-------|---------|
| tiny | 1 | 0.0312 | 0.0413 | 0.0186 | 0.0316 |
| tiny | 4 | 0.0329 | 0.0430 | 0.0201 | 0.0325 |
| tiny | 16 | 0.0330 | 0.0420 | 0.0200 | 0.0305 |
| tiny | 32 | 0.0332 | 0.0419 | 0.0201 | 0.0305 |
| attn_out | 1 | 7.3761 | 27.6331 | 2.3872 | 213.2731 |
| attn_out | 4 | 8.3720 | 28.4341 | 3.0622 | 192.8485 |
| ffn_up | 1 | 26.7459 | 161.1307 | 3.4603 | 1157.2906 |
| ffn_up | 4 | 29.4399 | 161.7959 | 6.0002 | 1055.9261 |
| ffn_down | 1 | 50.2539 | 917.9067 | 20.7147 | 1169.7382 |
| ffn_down | 4 | 65.8075 | 884.7293 | 35.7170 | 1075.0052 |

**Batch scaling:** Layout A scales linearly with batch size. Layout C shows better scaling on larger shapes.

## Memory Viability vs Q4 Analysis

**For shapes where Layout A is viable (ffn_up, ffn_down):**

| Shape | Q4 bytes | Q2 bytes | Residual budget | A total | Margin vs Q4 |
|-------|----------|----------|----------------|---------|--------------|
| ffn_up (896×4864) | 2,451,456 | 1,089,536 | 1,361,920 | 1,198,520 | -163,400 |
| ffn_down (4864×896) | 2,451,456 | 1,089,536 | 1,361,920 | 1,198,520 | -163,400 |

Layout A is 163KB below Q4 baseline on both FFN layers. W_low+Q2+residual = 2,288,056 bytes vs Q4 alone = 2,451,456 bytes — **net savings even before any quantization of W_low**.

**For shapes where no layout is viable (tiny, attn_out):**

The attn_out layer (896×896) has a tight 100KB budget. At 7.5% k, the bitmap alone (100,352 bytes = 100% of budget) leaves no room for values. This indicates k_pct needs to be lower for attn_out to be viable.

## Best Layout Recommendation

**Winner for AVX2/blocking implementation (Phase 31L): Layout A (bitmap scan)**

Rationale:
1. **Only memory-viable layout** on the production-relevant shapes (ffn_up, ffn_down) — all others (B, C, D) exceed budget
2. **All layouts (A, B, C) are correct** — no correctness regression
3. **Layout C is faster** but not memory-viable — can't be used
4. **Layout A is the safe, viable baseline** for AVX2 optimization

**Note on Layout C:** Row-wise sparse lists are 2-8× faster than A in Python, suggesting significant cache locality benefit. If Phase 31L can optimize the memory layout to fit within budget (e.g., by reducing k_pct or using a denser index format), Layout C should be reconsidered as it would provide a substantial speedup.

**Note on Layout B:** Index list is 3-20× slower than bitmap scan due to the per-element range check overhead. This layout only makes sense if the bitmap scan itself is the bottleneck, which it is not in Python — the bottleneck is the per-nonzero dot product. In C with AVX2, the bitmap scan overhead would be much lower relative to compute, making B even less attractive.

## Decision Gate

⚠️ **Classification: `PARTIAL_BITMAP_MEMORY_OK_SPEED_BAD`**

Layout A (bitmap) is the only memory-viable option on production shapes. No alternative layout simultaneously satisfies memory + correctness + speed criteria.

- ✅ A_bitmap: memory-viable on FFN shapes, correct, slow (baseline)
- ❌ B_index_list: not memory-viable, correct, slowest
- ❌ C_row_wise: not memory-viable, correct, fastest
- ❌ D_block_sparse_4x4: not memory-viable, incorrect

**Phase 31L is UNLOCKED** — Layout A serves as the baseline for AVX2/blocking optimization. Layout C's row-wise structure suggests that inner-loop SIMD vectorization over nonzeros within a row could yield significant speedup. AVX2 implementation should target row-wise processing with contiguous value access.

## Diagnostic: Dense Matmul Reference (not comparable to sparse)

| Shape | B | Sparse A (ms) | Dense (ms) | Ratio |
|-------|---|---------------|------------|-------|
| tiny | 1 | 0.0312 | ~0.0001 | — |
| attn_out | 1 | 7.3761 | ~0.1 | — |
| ffn_up | 1 | 26.7459 | ~1.0 | — |
| ffn_down | 1 | 50.2539 | ~2.0 | — |

Dense BLAS is not comparable — it operates on a different representation (full dense R) and uses hardware-accelerated matmul. This diagnostic confirms the sparse overhead is real but not directly quantifiable against dense without matching the representation.

---
*Phase 31K — ELVIS/SparkCascade — SDI Substitutive*