#!/usr/bin/env python3
"""
bench_kernel.py — Phase 31K: Kernel Microbenchmark + Layout Optimization

Profiles the scalar compressed residual kernel across multiple encoding layouts
and reports timing, correctness, and memory viability.

Shapes tested:
  - tiny:    8×16
  - attn_out: 896×896
  - ffn_up:  896×4864
  - ffn_down: 4864×896

Batch sizes: 1, 4, 16, 32
Layouts: A=bitmap scan, B=index list, C=row-wise, D=block sparse 4×4
No dense R materialized. No speedup claims.
"""

import sys
import json
import time
import pathlib
import resource
import argparse
import statistics
import numpy as np
from typing import List, Dict, Any, Tuple

REPO_DIR = pathlib.Path.home() / "sdi-substitutive"
RESULTS_DIR = REPO_DIR / "results"
sys.path.insert(0, str(REPO_DIR / "src"))
from residual_encode import EncodedResidual, encode_residual

# =============================================================================
# Config
# =============================================================================
BENCHMARK_SHAPES = {
    "tiny":      (8,   16),
    "attn_out":  (896, 896),
    "ffn_up":    (896, 4864),
    "ffn_down":  (4864, 896),
}
# Reduced for large-shape speed in Python
LARGE_SHAPE_BATCH_SIZES = [1, 4]
SMALL_SHAPE_BATCH_SIZES = [1, 4, 16, 32]
WARMUP = 5
TIMED_TINY = 50
TIMED_LARGE = 30
SEED = 42
MEMORY_BUDGET_BPW = 1.0

np.random.seed(SEED)


# =============================================================================
# Layout A: Scalar bitmap scan (baseline — same as Phase 31J reference)
# =============================================================================

def layout_a_apply(X: np.ndarray, R_enc: EncodedResidual,
                   X_on_R_cols: bool = True) -> np.ndarray:
    """Layout A: bitmap scan. Unpack, scan bits, consume fp16 values."""
    if X.ndim == 1:
        X = X[np.newaxis, :]
        squeeze = True
    else:
        squeeze = False

    batch_size = X.shape[0]
    rows, cols = R_enc.rows, R_enc.cols

    vals_f16 = R_enc.values.view(np.float16)
    vals_f32 = vals_f16.astype(np.float32)
    flat_bitmap = np.unpackbits(R_enc.bitmap, count=R_enc.n_elements)
    set_indices = np.where(flat_bitmap)[0]
    pos_map = np.full(R_enc.n_elements, -1, dtype=np.int32)
    pos_map[set_indices] = np.arange(R_enc.nnz, dtype=np.int32)

    if X_on_R_cols:
        # X: (batch, d_in), R: (d_out, d_in) -> Y: (batch, d_out)
        Y_delta = np.zeros((batch_size, rows), dtype=np.float32)
        for j in range(rows):
            rs = j * cols
            re = rs + cols
            row_bitmap = flat_bitmap[rs:re]
            row_set = np.where(row_bitmap)[0]
            if len(row_set) == 0:
                continue
            gi = rs + row_set
            vp = pos_map[gi]
            v = vals_f32[vp]
            Y_delta[:, j] = np.dot(X[:, row_set], v)
    else:
        # X: (batch, d_out), R: (d_out, d_in) -> Y: (batch, d_in)
        Y_delta = np.zeros((batch_size, cols), dtype=np.float32)
        for j in range(rows):
            rs = j * cols
            re = rs + cols
            row_bitmap = flat_bitmap[rs:re]
            row_set = np.where(row_bitmap)[0]
            if len(row_set) == 0:
                continue
            gi = rs + row_set
            vp = pos_map[gi]
            v = vals_f32[vp]
            np.add.at(Y_delta, (slice(None), row_set),
                      np.outer(X[:, j], v))

    if squeeze:
        return Y_delta[0]
    return Y_delta


# =============================================================================
# Layout B: Precomputed nonzero index list (uint32 flat indices)
# =============================================================================

class LayoutB:
    """Layout B: flat uint32 indices. No bitmap scan. Same nnz/indices as Layout A."""
    def __init__(self, rows, cols, k_pct, indices, values):
        self.rows = rows
        self.cols = cols
        self.k_pct = k_pct
        self.indices = indices
        self.values = values
        self.nnz = len(indices)

    @property
    def total_bytes(self):
        return 32 + self.nnz * 4 + self.nnz * 2

    @property
    def ebpw(self):
        return self.total_bytes * 8.0 / (self.rows * self.cols)

    @classmethod
    def from_encoded(cls, enc: EncodedResidual):
        """Build from EncodedResidual — same indices/values as Layout A."""
        flat_bitmap = np.unpackbits(enc.bitmap, count=enc.n_elements)
        set_indices = np.where(flat_bitmap)[0].astype(np.uint32)
        vals = enc.values.copy()
        return cls(enc.rows, enc.cols, enc.k_pct, set_indices, vals)

    def apply(self, X, X_on_R_cols=True):
        if X.ndim == 1:
            X = X[np.newaxis, :]
            squeeze = True
        else:
            squeeze = False

        batch_size = X.shape[0]
        vals_f32 = self.values.view(np.float16).astype(np.float32)

        if X_on_R_cols:
            Y_delta = np.zeros((batch_size, self.rows), dtype=np.float32)
            for j in range(self.rows):
                rs = j * self.cols
                re = rs + self.cols
                mask = (self.indices >= rs) & (self.indices < re)
                lc = self.indices[mask] - rs
                if len(lc) == 0:
                    continue
                v = vals_f32[mask]
                Y_delta[:, j] = np.dot(X[:, lc.astype(int)], v)
        else:
            Y_delta = np.zeros((batch_size, self.cols), dtype=np.float32)
            for j in range(self.rows):
                rs = j * self.cols
                re = rs + self.cols
                mask = (self.indices >= rs) & (self.indices < re)
                lc = self.indices[mask] - rs
                if len(lc) == 0:
                    continue
                v = vals_f32[mask]
                np.add.at(Y_delta, (slice(None), lc.astype(int)),
                          np.outer(X[:, j], v))

        if squeeze:
            return Y_delta[0]
        return Y_delta


# =============================================================================
# Layout C: Row-wise sparse lists (same global indices as Layout A)
# =============================================================================

class LayoutC:
    """Layout C: per-row nnz list with offsets. Same global indices as Layout A."""
    def __init__(self, rows, cols, k_pct, offsets, col_indices, values):
        self.rows = rows
        self.cols = cols
        self.k_pct = k_pct
        self.offsets = offsets
        self.col_indices = col_indices
        self.values = values
        self.nnz = len(col_indices)

    @property
    def total_bytes(self):
        return 32 + self.nnz * 4 + self.nnz * 2 + (self.rows + 1) * 4

    @property
    def ebpw(self):
        return self.total_bytes * 8.0 / (self.rows * self.cols)

    @classmethod
    def from_encoded(cls, enc: EncodedResidual):
        """Build from EncodedResidual — same indices/values as Layout A."""
        rows, cols = enc.rows, enc.cols
        flat_bitmap = np.unpackbits(enc.bitmap, count=enc.n_elements)
        set_indices = np.where(flat_bitmap)[0]
        vals_f16 = enc.values.view(np.float16)

        offsets = [0]
        col_indices = []
        values_list = []

        for r in range(rows):
            rs = r * cols
            row_bitmap = flat_bitmap[rs:rs + cols]
            row_set = np.where(row_bitmap)[0]
            for c in row_set:
                col_indices.append(c)
                gi = rs + c
                pos = np.where(set_indices == gi)[0]
                if len(pos) > 0:
                    values_list.append(float(vals_f16[pos[0]]))
            offsets.append(len(col_indices))

        col_indices = np.array(col_indices, dtype=np.uint32)
        values_arr = np.array(values_list, dtype=np.float16).view(np.uint16)
        offsets_arr = np.array(offsets, dtype=np.uint32)
        return cls(rows, cols, enc.k_pct, offsets_arr, col_indices, values_arr)

    def apply(self, X, X_on_R_cols=True):
        if X.ndim == 1:
            X = X[np.newaxis, :]
            squeeze = True
        else:
            squeeze = False

        batch_size = X.shape[0]
        vals_f32 = self.values.view(np.float16).astype(np.float32)

        if X_on_R_cols:
            Y_delta = np.zeros((batch_size, self.rows), dtype=np.float32)
            for j in range(self.rows):
                s = int(self.offsets[j])
                e = int(self.offsets[j + 1])
                if s == e:
                    continue
                lc = self.col_indices[s:e].astype(int)
                v = vals_f32[s:e]
                Y_delta[:, j] = np.dot(X[:, lc], v)
        else:
            Y_delta = np.zeros((batch_size, self.cols), dtype=np.float32)
            for j in range(self.rows):
                s = int(self.offsets[j])
                e = int(self.offsets[j + 1])
                if s == e:
                    continue
                lc = self.col_indices[s:e].astype(int)
                v = vals_f32[s:e]
                np.add.at(Y_delta, (slice(None), lc), np.outer(X[:, j], v))

        if squeeze:
            return Y_delta[0]
        return Y_delta


# =============================================================================
# Layout D: Block sparse 4×4 (same global indices as Layout A)
# =============================================================================

class LayoutD:
    """Layout D: 4×4 block occupancy + sub-bitmaps. Same indices as Layout A."""
    def __init__(self, rows, cols, k_pct, block_occ, block_bitmaps,
                 block_row_ptr, values):
        self.rows = rows
        self.cols = cols
        self.k_pct = k_pct
        self.block_occ = block_occ
        self.block_bitmaps = block_bitmaps
        self.block_row_ptr = block_row_ptr
        self.values = values
        self.nnz = len(values) // 2 if len(values) > 0 else 0

    @property
    def total_bytes(self):
        BLOCK = 4
        br = (self.rows + BLOCK - 1) // BLOCK
        bc = (self.cols + BLOCK - 1) // BLOCK
        n_blocks = br * bc
        block_occ_bytes = (n_blocks + 7) // 8
        return (32 + block_occ_bytes + len(self.block_bitmaps) +
                len(self.block_row_ptr) * 4 + self.nnz * 2)

    @property
    def ebpw(self):
        return self.total_bytes * 8.0 / (self.rows * self.cols)

    @classmethod
    def from_encoded(cls, enc: EncodedResidual):
        """Build from EncodedResidual — same indices/values as Layout A."""
        rows, cols = enc.rows, enc.cols
        BLOCK = 4
        br = (rows + BLOCK - 1) // BLOCK
        bc = (cols + BLOCK - 1) // BLOCK
        n_blocks = br * bc

        flat_bitmap = np.unpackbits(enc.bitmap, count=enc.n_elements)
        set_indices = np.where(flat_bitmap)[0]
        vals_f16 = enc.values.view(np.float16)

        block_occ_bits = np.zeros(n_blocks, dtype=np.uint8)
        block_bitmaps_list = []
        block_row_ptr = [0]
        values_list = []
        ptr = 0

        for bi in range(br):
            for bj in range(bc):
                occ_idx = bi * bc + bj
                rs = bi * BLOCK
                cs = bj * BLOCK
                re = min(rs + BLOCK, rows)
                ce = min(cs + BLOCK, cols)
                b_r, b_c = re - rs, ce - cs
                b_size = b_r * b_c

                # Build sub-bitmap for this block
                sub_bitmap = np.zeros(b_size, dtype=np.uint8)
                for i in range(b_r):
                    for j in range(b_c):
                        gi = (rs + i) * cols + (cs + j)
                        if gi < enc.n_elements and flat_bitmap[gi]:
                            li = i * b_c + j
                            if li < b_size:
                                sub_bitmap[li] = 1

                n_set = np.sum(sub_bitmap)
                if n_set > 0:
                    block_occ_bits[occ_idx] = 1
                    packed = np.packbits(sub_bitmap)
                    block_bitmaps_list.append(packed.tobytes())
                    ptr += len(packed)

                    # Collect values in row-major order within block
                    for i in range(b_r):
                        for j in range(b_c):
                            li = i * b_c + j
                            if li < b_size and sub_bitmap[li]:
                                gi = (rs + i) * cols + (cs + j)
                                pos = np.where(set_indices == gi)[0]
                                if len(pos) > 0:
                                    values_list.append(float(vals_f16[pos[0]]))

                block_row_ptr.append(ptr)

        block_occ_packed = np.packbits(block_occ_bits)
        block_bitmap_bytes = b''.join(block_bitmaps_list)
        block_bitmap_arr = np.frombuffer(block_bitmap_bytes, dtype=np.uint8) if block_bitmap_bytes else np.zeros(0, dtype=np.uint8)
        values_arr = np.array(values_list, dtype=np.float16).view(np.uint8)
        row_ptr_arr = np.array(block_row_ptr, dtype=np.uint32)

        return cls(rows, cols, enc.k_pct, block_occ_packed, block_bitmap_arr,
                   row_ptr_arr, values_arr)

    def apply(self, X, X_on_R_cols=True):
        BLOCK = 4
        if X.ndim == 1:
            X = X[np.newaxis, :]
            squeeze = True
        else:
            squeeze = False

        batch_size = X.shape[0]
        rows, cols = self.rows, self.cols
        br = (rows + BLOCK - 1) // BLOCK
        bc = (cols + BLOCK - 1) // BLOCK
        flat_occ = np.unpackbits(self.block_occ, count=br * bc)
        occ_blocks = np.where(flat_occ)[0]
        vals_f32 = self.values.view(np.float16).astype(np.float32)

        if X_on_R_cols:
            Y_delta = np.zeros((batch_size, rows), dtype=np.float32)
            for bidx, occ in enumerate(occ_blocks):
                bi = occ // bc
                bj = occ % bc
                rs = bi * BLOCK
                cs = bj * BLOCK
                re = min(rs + BLOCK, rows)
                ce = min(cs + BLOCK, cols)
                b_r, b_c = re - rs, ce - cs
                b_size = b_r * b_c

                s_ptr = int(self.block_row_ptr[bidx])
                e_ptr = int(self.block_row_ptr[bidx + 1])

                if e_ptr > s_ptr:
                    sub_bmap = self.block_bitmaps[s_ptr:e_ptr]
                    if len(sub_bmap) > 0:
                        mask = np.unpackbits(sub_bmap, count=b_size)
                        val_ptr = 0
                        for i in range(b_r):
                            for j in range(b_c):
                                li = i * b_c + j
                                if li < b_size and mask[li]:
                                    gr = rs + i
                                    gc = cs + j
                                    if gr < rows and gc < cols:
                                        Y_delta[:, gr] += X[:, gc] * vals_f32[val_ptr]
                                    val_ptr += 1
        else:
            Y_delta = np.zeros((batch_size, cols), dtype=np.float32)
            for bidx, occ in enumerate(occ_blocks):
                bi = occ // bc
                bj = occ % bc
                rs = bi * BLOCK
                cs = bj * BLOCK
                re = min(rs + BLOCK, rows)
                ce = min(cs + BLOCK, cols)
                b_r, b_c = re - rs, ce - cs
                b_size = b_r * b_c

                s_ptr = int(self.block_row_ptr[bidx])
                e_ptr = int(self.block_row_ptr[bidx + 1])

                if e_ptr > s_ptr:
                    sub_bmap = self.block_bitmaps[s_ptr:e_ptr]
                    if len(sub_bmap) > 0:
                        mask = np.unpackbits(sub_bmap, count=b_size)
                        val_ptr = 0
                        for i in range(b_r):
                            for j in range(b_c):
                                li = i * b_c + j
                                if li < b_size and mask[li]:
                                    gr = rs + i
                                    gc = cs + j
                                    if gr < rows and gc < cols:
                                        Y_delta[:, gc] += X[:, gr] * vals_f32[val_ptr]
                                    val_ptr += 1

        if squeeze:
            return Y_delta[0]
        return Y_delta


# =============================================================================
# Utilities
# =============================================================================

def cosine(a, b):
    af = a.ravel()
    bf = b.ravel()
    dot = np.dot(af, bf)
    na = np.linalg.norm(af)
    nb = np.linalg.norm(bf)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def verify(test, ref):
    return {
        "max_abs_diff": float(np.abs(test - ref).max()),
        "cosine": cosine(test, ref),
        "has_nan": bool(np.any(np.isnan(test)) or np.any(np.isnan(ref))),
        "has_inf": bool(np.any(np.isinf(test)) or np.any(np.isinf(ref))),
        "correct": bool(np.abs(test - ref).max() < 1e-3 and
                        cosine(test, ref) > 0.9999 and
                        not (np.any(np.isnan(test)) or np.any(np.isnan(ref))) and
                        not (np.any(np.isinf(test)) or np.any(np.isinf(ref)))),
    }


def mem_accounting(rows, cols, nnz, layout_key):
    n = rows * cols
    BLOCK = 4
    br = (rows + BLOCK - 1) // BLOCK
    bc = (cols + BLOCK - 1) // BLOCK
    n_blocks = br * bc
    bitmap_bytes = (n + 7) // 8
    budget = int(n * MEMORY_BUDGET_BPW / 8.0)
    q4_bytes = int(n * 4.0 / 8.0)

    if layout_key == "A":
        total = 32 + bitmap_bytes + nnz * 2
        r = dict(bitmap=bitmap_bytes, index=0, offset=0, values=nnz*2, metadata=32, total=total)
    elif layout_key == "B":
        total = 32 + nnz * 4 + nnz * 2
        r = dict(bitmap=0, index=nnz*4, offset=0, values=nnz*2, metadata=32, total=total)
    elif layout_key == "C":
        off = (rows + 1) * 4
        total = 32 + nnz * 4 + nnz * 2 + off
        r = dict(bitmap=0, index=nnz*4, offset=off, values=nnz*2, metadata=32, total=total)
    elif layout_key == "D":
        block_occ = (n_blocks + 7) // 8
        # Block indices: worst case all blocks have at least 1 entry
        # Sub-bitmap bytes: per-nnz overhead for block's sub-bitmap
        sub_bmap_bytes = nnz  # approx 1 byte per nnz for sub-bitmap overhead
        block_idx_bytes = n_blocks * 4
        total = 32 + block_occ + sub_bmap_bytes + block_idx_bytes + nnz * 2
        r = dict(bitmap=block_occ, block_idx=block_idx_bytes, sub_bmap=sub_bmap_bytes,
                 values=nnz*2, metadata=32, total=total)
    else:
        r = dict(total=0)

    r["ebpw"] = r["total"] * 8.0 / n
    r["budget"] = budget
    r["q4_approx"] = q4_bytes
    r["margin"] = budget - r["total"]
    r["memory_viable"] = r["total"] <= budget
    return r


# =============================================================================
# Benchmark
# =============================================================================

def run_benchmark(shape_name, rows, cols, X_on_R_cols, k_pct, batch_sizes, timed_iters):
    """Benchmark all layouts for one shape."""

    # Generate R and X
    R_f32 = np.random.randn(rows, cols).astype(np.float32) * 0.01
    feat_dim = cols if X_on_R_cols else rows
    X_base = np.random.randn(32, feat_dim).astype(np.float32) * 0.05

    # Build all layouts from same EncodedResidual
    enc = encode_residual(R_f32, k_pct=k_pct)
    lay_b = LayoutB.from_encoded(enc)
    lay_c = LayoutC.from_encoded(enc)
    lay_d = LayoutD.from_encoded(enc)

    nnz = enc.nnz
    n_elements = rows * cols

    # Memory accounting
    mem = {
        "A_bitmap": mem_accounting(rows, cols, nnz, "A"),
        "B_index_list": mem_accounting(rows, cols, nnz, "B"),
        "C_row_wise": mem_accounting(rows, cols, nnz, "C"),
        "D_block_sparse_4x4": mem_accounting(rows, cols, nnz, "D"),
    }

    result = {
        "shape_name": shape_name,
        "shape": [rows, cols],
        "k_pct": k_pct,
        "nnz": nnz,
        "n_elements": n_elements,
        "sparsity_pct": f"{100.0 * (1.0 - nnz / n_elements):.2f}%",
        "X_on_R_cols": X_on_R_cols,
        "memory": mem,
        "batch_sizes": {},
    }

    for B in batch_sizes:
        X = X_base[:B, :].copy()

        # Layout A (reference)
        for _ in range(WARMUP):
            layout_a_apply(X, enc, X_on_R_cols)
        times_a = []
        for _ in range(timed_iters):
            t0 = time.perf_counter()
            Y_a = layout_a_apply(X, enc, X_on_R_cols)
            t1 = time.perf_counter()
            times_a.append((t1 - t0) * 1000.0)

        med_a = statistics.median(times_a)

        # Layout B
        for _ in range(WARMUP):
            lay_b.apply(X, X_on_R_cols)
        t0 = time.perf_counter()
        Y_b = lay_b.apply(X, X_on_R_cols)
        t1 = time.perf_counter()
        time_b = (t1 - t0) * 1000.0

        # Layout C
        for _ in range(WARMUP):
            lay_c.apply(X, X_on_R_cols)
        t0 = time.perf_counter()
        Y_c = lay_c.apply(X, X_on_R_cols)
        t1 = time.perf_counter()
        time_c = (t1 - t0) * 1000.0

        # Layout D
        for _ in range(WARMUP):
            lay_d.apply(X, X_on_R_cols)
        t0 = time.perf_counter()
        Y_d = lay_d.apply(X, X_on_R_cols)
        t1 = time.perf_counter()
        time_d = (t1 - t0) * 1000.0

        # Correctness vs Layout A
        corr_b = verify(Y_b, Y_a)
        corr_c = verify(Y_c, Y_a)
        corr_d = verify(Y_d, Y_a)

        batch_result = {
            "batch_size": B,
            "reference_A": {
                "median_ms": med_a,
                "mean_ms": statistics.mean(times_a),
                "p95_ms": sorted(times_a)[int(len(times_a) * 0.95)],
                "min_ms": min(times_a),
                "max_ms": max(times_a),
                "all_times_ms": sorted(times_a),
            },
            "layouts": {
                "B_index_list": {
                    "median_ms": time_b,
                    "vs_A_speedup": med_a / time_b if time_b > 0 else None,
                    "time_per_nonzero_us": time_b / nnz * 1000 if nnz > 0 else 0,
                    "correctness": corr_b,
                },
                "C_row_wise": {
                    "median_ms": time_c,
                    "vs_A_speedup": med_a / time_c if time_c > 0 else None,
                    "time_per_nonzero_us": time_c / nnz * 1000 if nnz > 0 else 0,
                    "correctness": corr_c,
                },
                "D_block_sparse_4x4": {
                    "median_ms": time_d,
                    "vs_A_speedup": med_a / time_d if time_d > 0 else None,
                    "time_per_nonzero_us": time_d / nnz * 1000 if nnz > 0 else 0,
                    "correctness": corr_d,
                },
            },
        }
        result["batch_sizes"][str(B)] = batch_result

        print(f"  B={B}: A={med_a:.4f}ms B={time_b:.4f}ms C={time_c:.4f}ms D={time_d:.4f}ms "
              f"| corr_B={corr_b['correct']} corr_C={corr_c['correct']} corr_D={corr_d['correct']}")

    return result


def classify(results):
    """Determine classification."""
    any_correct = False
    any_faster_than_a = False
    any_memory_viable = False

    for sname, sdata in results["shapes"].items():
        for bs, bdata in sdata["batch_sizes"].items():
            for lk, ldata in bdata["layouts"].items():
                corr = ldata.get("correctness", {})
                if corr.get("correct"):
                    any_correct = True
                sp = ldata.get("vs_A_speedup")
                if sp is not None and sp > 1.05:
                    any_faster_than_a = True
        for lk, mdata in sdata["memory"].items():
            if mdata.get("memory_viable"):
                any_memory_viable = True

    if not any_correct:
        return "BLOCKED_CORRECTNESS_REGRESSION"
    if any_faster_than_a and any_memory_viable:
        return "PASS_LAYOUT_OPTIMIZED_SCALAR"
    if any_faster_than_a and not any_memory_viable:
        return "PARTIAL_INDEX_LAYOUT_FAST_BUT_MEMORY_FAILS"
    if not any_faster_than_a and any_memory_viable:
        return "PARTIAL_BITMAP_MEMORY_OK_SPEED_BAD"
    return "PARTIAL_LAYOUT_TRADEOFF"


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", nargs="+",
                        default=["tiny", "attn_out", "ffn_up", "ffn_down"])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 32])
    parser.add_argument("--k-pct", type=float, default=7.5)
    parser.add_argument("--output-json",
                        default=str(RESULTS_DIR / "PHASE31K_KERNEL_MICROBENCH_LAYOUT.json"))
    parser.add_argument("--output-md",
                        default=str(REPO_DIR / "docs" / "PHASE31K_KERNEL_MICROBENCH_LAYOUT.md"))
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 31K: Kernel Microbenchmark + Layout Optimization")
    print("=" * 70)

    # Determine which batch sizes to use per shape
    large_shapes = {"attn_out", "ffn_up", "ffn_down"}

    results = {
        "phase": "31K",
        "classification": "IN_PROGRESS",
        "benchmark_shapes": args.shapes,
        "batch_sizes": args.batch_sizes,
        "k_pct": args.k_pct,
        "warmup_iters": WARMUP,
        "seed": SEED,
        "memory_budget_bpw": MEMORY_BUDGET_BPW,
        "shapes": {},
    }

    for sname in args.shapes:
        if sname not in BENCHMARK_SHAPES:
            print(f"Skipping: {sname}")
            continue
        rows, cols = BENCHMARK_SHAPES[sname]
        is_large = sname in large_shapes
        bs_list = LARGE_SHAPE_BATCH_SIZES if is_large else SMALL_SHAPE_BATCH_SIZES
        n_timed = TIMED_LARGE if is_large else TIMED_TINY

        # Determine orientation
        if sname == "ffn_down":
            X_on_R_cols = False
        else:
            X_on_R_cols = True

        print(f"\n--- Shape: {sname} ({rows}×{cols}), X_on_R_cols={X_on_R_cols} ---")
        r = run_benchmark(sname, rows, cols, X_on_R_cols,
                           k_pct=args.k_pct,
                           batch_sizes=bs_list,
                           timed_iters=n_timed)
        results["shapes"][sname] = r

    classification = classify(results)
    results["classification"] = classification
    print(f"\nClassification: {classification}")

    # Write JSON
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote: {args.output_json}")

    # Write MD
    write_md(results, args.output_md)
    print(f"Wrote: {args.output_md}")

    return results


def write_md(results, path):
    lines = [
        "# Phase 31K: Kernel Microbenchmark + Layout Optimization",
        "",
        "## Summary",
        "",
        f"- **Classification:** `{results['classification']}`",
        f"- **Benchmark shapes:** {results['benchmark_shapes']}",
        f"- **k_pct:** {results['k_pct']}%",
        f"- **Warmup/Timed:** {results['warmup_iters']}/{results.get('timed_iters', 'varies')}",
        f"- **Memory budget:** {results['memory_budget_bpw']} bpw",
        "",
        "## Memory Viability",
        "",
        "| Shape | Layout | Total Bytes | EBPW | Budget | Margin | Viable? |",
        "|-------|--------|-------------|------|--------|--------|---------|",
    ]

    for sname, sdata in results["shapes"].items():
        for lk, m in sdata["memory"].items():
            margin_str = f"{m.get('margin', 'N/A')}"
            if isinstance(m.get('margin'), int):
                margin_str = f"{m['margin']:,}"
            viable = "✅" if m.get("memory_viable") else "❌"
            lines.append(
                f"| {sname} | {lk} | {m.get('total', 'N/A'):,} | "
                f"{m.get('ebpw', 0):.4f} | {m.get('budget', 'N/A'):,} | "
                f"{margin_str} | {viable} |"
            )

    lines += ["", "## Timing Table (median ms)", "",
              "| Shape | B | A_ref | B_index | C_row | D_block |",
              "|-------|---|-------|---------|-------|---------|"]

    for sname, sdata in results["shapes"].items():
        for bs, bdata in sdata["batch_sizes"].items():
            ref = bdata["reference_A"]["median_ms"]
            b_ms = bdata["layouts"]["B_index_list"]["median_ms"]
            c_ms = bdata["layouts"]["C_row_wise"]["median_ms"]
            d_ms = bdata["layouts"]["D_block_sparse_4x4"]["median_ms"]
            lines.append(f"| {sname} | {bs} | {ref:.4f} | {b_ms:.4f} | {c_ms:.4f} | {d_ms:.4f} |")

    lines += ["", "## Correctness (vs Layout A reference)", "",
              "| Shape | Layout | max_abs_diff | cosine | correct? |",
              "|-------|--------|-------------|--------|---------|"]

    for sname, sdata in results["shapes"].items():
        for bs, bdata in sdata["batch_sizes"].items():
            for lk, ldata in bdata["layouts"].items():
                c = ldata["correctness"]
                ok = "✅" if c["correct"] else "❌"
                lines.append(
                    f"| {sname} | {lk} | {c['max_abs_diff']:.2e} | "
                    f"{c['cosine']:.8f} | {ok} |"
                )

    lines += ["", "## Speedup vs Layout A (B=1)", "",
              "| Shape | B_index | C_row_wise | D_block_4x4 |",
              "|-------|---------|------------|------------|"]

    for sname, sdata in results["shapes"].items():
        b1 = sdata["batch_sizes"].get("1", {})
        ref_a = b1.get("reference_A", {}).get("median_ms", 1.0)
        for lk, ldata in b1["layouts"].items():
            sp = ldata.get("vs_A_speedup", 0)
            if sp:
                ldata["_speedup_str"] = f"{sp:.3f}×"
            else:
                ldata["_speedup_str"] = "N/A"
        sb = b1["layouts"]["B_index_list"].get("_speedup_str", "N/A")
        sc = b1["layouts"]["C_row_wise"].get("_speedup_str", "N/A")
        sd = b1["layouts"]["D_block_sparse_4x4"].get("_speedup_str", "N/A")
        lines.append(f"| {sname} | {sb} | {sc} | {sd} |")

    lines += ["", "## Decision Gate", ""]
    cls = results["classification"]
    if cls == "PASS_LAYOUT_OPTIMIZED_SCALAR":
        lines += ["✅ **Phase 31L UNLOCKED** — layout optimization improves scalar timing, memory-viable. Proceed to AVX2/blocking."]
    elif "PARTIAL" in cls:
        lines += [f"⚠️ **Classification: {cls}** — tradeoffs documented. See tables above."]
        lines += [""]
        lines += ["Key observations:"]
        for sname, sdata in results["shapes"].items():
            viable = [lk for lk, m in sdata["memory"].items() if m.get("memory_viable")]
            b1 = sdata["batch_sizes"].get("1", {})
            faster = [lk for lk, ldata in b1["layouts"].items()
                      if ldata.get("vs_A_speedup", 0) and ldata["vs_A_speedup"] > 1.05]
            lines.append(f"- **{sname}**: memory_viable={viable}, faster_than_A={faster}")
    else:
        lines += [f"**Classification: {cls}**"]

    lines += ["", "---", "*Phase 31K — ELVIS/SparkCascade — SDI Substitutive*"]
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
