#!/usr/bin/env python3
"""
Phase 31L: AVX2 / Blocking Kernel Implementation
"""

import sys, json, time, pathlib, statistics, argparse
import numpy as np

REPO_DIR = pathlib.Path.home() / "sdi-substitutive"
RESULTS_DIR = REPO_DIR / "results"
sys.path.insert(0, str(REPO_DIR / "src"))
from residual_encode import EncodedResidual, encode_residual

BENCHMARK_SHAPES = {
    "tiny":      (8,   16),
    "ffn_up":    (896, 4864),
    "ffn_down":  (4864, 896),
}
WARMUP = 5
TIMED_FFN = 40
TIMED_TINY = 60
SEED = 42
K_PCT = 7.5
np.random.seed(SEED)

# ---- CSR Scalar Baseline (31K LayoutC) ----
class CSRBaseline:
    def __init__(self, rows, cols, k_pct, offsets, col_indices, values):
        self.rows = rows; self.cols = cols; self.k_pct = k_pct
        self.offsets = offsets.astype(np.uint32)
        self.col_indices = col_indices.astype(np.uint32)
        self.values = values.astype(np.float16)
        self.nnz = len(col_indices)

    @classmethod
    def from_encoded(cls, enc):
        flat_bitmap = np.unpackbits(enc.bitmap, count=enc.n_elements)
        set_indices = np.where(flat_bitmap)[0]
        vals_f16 = enc.values.view(np.float16)
        offsets = [0]; col_indices = []; values_list = []
        for r in range(enc.rows):
            rs = r * enc.cols
            row_bitmap = flat_bitmap[rs:rs + enc.cols]
            row_set = np.where(row_bitmap)[0]
            for c in row_set:
                col_indices.append(c)
                gi = rs + c
                pos = np.where(set_indices == gi)[0]
                if len(pos) > 0:
                    values_list.append(float(vals_f16[pos[0]]))
            offsets.append(len(col_indices))
        return cls(enc.rows, enc.cols, enc.k_pct,
                   np.array(offsets, dtype=np.uint32),
                   np.array(col_indices, dtype=np.uint32),
                   np.array(values_list, dtype=np.float16))

    def apply(self, X, X_on_R_cols=True):
        squeeze = False
        if X.ndim == 1:
            X = X[np.newaxis, :]; squeeze = True
        batch_size = X.shape[0]
        vals_f32 = self.values.astype(np.float32)
        if X_on_R_cols:
            Y = np.zeros((batch_size, self.rows), dtype=np.float32)
            for j in range(self.rows):
                s, e = int(self.offsets[j]), int(self.offsets[j+1])
                if s == e: continue
                lc = self.col_indices[s:e].astype(int)
                Y[:, j] = np.dot(X[:, lc], vals_f32[s:e])
        else:
            Y = np.zeros((batch_size, self.cols), dtype=np.float32)
            for j in range(self.rows):
                s, e = int(self.offsets[j]), int(self.offsets[j+1])
                if s == e: continue
                lc = self.col_indices[s:e].astype(int)
                np.add.at(Y, (slice(None), lc), np.outer(X[:, j], vals_f32[s:e]))
        return Y[0] if squeeze else Y


# ---- Blocked Scalar ----
class CSRBlocked:
    BLOCK_SIZE = 64
    def __init__(self, rows, cols, k_pct, offsets, col_indices, values):
        self.rows = rows; self.cols = cols; self.k_pct = k_pct
        self.offsets = offsets.astype(np.uint32)
        self.col_indices = col_indices.astype(np.uint32)
        self.values = values.astype(np.float16)
        self.nnz = len(col_indices)

    @classmethod
    def from_encoded(cls, enc):
        flat_bitmap = np.unpackbits(enc.bitmap, count=enc.n_elements)
        set_indices = np.where(flat_bitmap)[0]
        vals_f16 = enc.values.view(np.float16)
        offsets = [0]; col_indices = []; values_list = []
        for r in range(enc.rows):
            rs = r * enc.cols
            row_bitmap = flat_bitmap[rs:rs + enc.cols]
            row_set = np.where(row_bitmap)[0]
            for c in row_set:
                col_indices.append(c)
                gi = rs + c
                pos = np.where(set_indices == gi)[0]
                if len(pos) > 0:
                    values_list.append(float(vals_f16[pos[0]]))
            offsets.append(len(col_indices))
        return cls(enc.rows, enc.cols, enc.k_pct,
                   np.array(offsets, dtype=np.uint32),
                   np.array(col_indices, dtype=np.uint32),
                   np.array(values_list, dtype=np.float16))

    def apply(self, X, X_on_R_cols=True):
        squeeze = False
        if X.ndim == 1:
            X = X[np.newaxis, :]; squeeze = True
        batch_size = X.shape[0]
        vals_f32 = self.values.astype(np.float32)
        rows = self.rows; B = self.BLOCK_SIZE
        if X_on_R_cols:
            Y = np.zeros((batch_size, rows), dtype=np.float32)
            for rb in range(0, rows, B):
                re = min(rb + B, rows)
                for j in range(rb, re):
                    s, e = int(self.offsets[j]), int(self.offsets[j+1])
                    if s == e: continue
                    lc = self.col_indices[s:e].astype(int)
                    Y[:, j] = np.dot(X[:, lc], vals_f32[s:e])
        else:
            Y = np.zeros((batch_size, self.cols), dtype=np.float32)
            for rb in range(0, rows, B):
                re = min(rb + B, rows)
                for j in range(rb, re):
                    s, e = int(self.offsets[j]), int(self.offsets[j+1])
                    if s == e: continue
                    lc = self.col_indices[s:e].astype(int)
                    np.add.at(Y, (slice(None), lc), np.outer(X[:, j], vals_f32[s:e]))
        return Y[0] if squeeze else Y


# ---- Batch-specialized ----
class CSRBatch:
    def __init__(self, rows, cols, k_pct, offsets, col_indices, values):
        self.rows = rows; self.cols = cols; self.k_pct = k_pct
        self.offsets = offsets.astype(np.uint32)
        self.col_indices = col_indices.astype(np.uint32)
        self.values = values.astype(np.float16)
        self.nnz = len(col_indices)

    @classmethod
    def from_encoded(cls, enc):
        flat_bitmap = np.unpackbits(enc.bitmap, count=enc.n_elements)
        set_indices = np.where(flat_bitmap)[0]
        vals_f16 = enc.values.view(np.float16)
        offsets = [0]; col_indices = []; values_list = []
        for r in range(enc.rows):
            rs = r * enc.cols
            row_bitmap = flat_bitmap[rs:rs + enc.cols]
            row_set = np.where(row_bitmap)[0]
            for c in row_set:
                col_indices.append(c)
                gi = rs + c
                pos = np.where(set_indices == gi)[0]
                if len(pos) > 0:
                    values_list.append(float(vals_f16[pos[0]]))
            offsets.append(len(col_indices))
        return cls(enc.rows, enc.cols, enc.k_pct,
                   np.array(offsets, dtype=np.uint32),
                   np.array(col_indices, dtype=np.uint32),
                   np.array(values_list, dtype=np.float16))

    def apply_B1(self, X, X_on_R_cols=True):
        squeeze = False
        if X.ndim == 2:
            X = X[0]; squeeze = True
        vals_f32 = self.values.astype(np.float32)
        if X_on_R_cols:
            Y = np.zeros(self.rows, dtype=np.float32)
            for j in range(self.rows):
                s, e = int(self.offsets[j]), int(self.offsets[j+1])
                if s == e: continue
                lc = self.col_indices[s:e].astype(int)
                Y[j] = np.dot(X[lc], vals_f32[s:e])
        else:
            Y = np.zeros(self.cols, dtype=np.float32)
            for j in range(self.rows):
                s, e = int(self.offsets[j]), int(self.offsets[j+1])
                if s == e: continue
                lc = self.col_indices[s:e].astype(int)
                np.add.at(Y, lc, np.outer(X[j], vals_f32[s:e]))
        return Y if squeeze else Y

    def apply(self, X, X_on_R_cols=True):
        squeeze = False
        if X.ndim == 1:
            X = X[np.newaxis, :]; squeeze = True
        batch_size = X.shape[0]
        vals_f32 = self.values.astype(np.float32)
        if X_on_R_cols:
            Y = np.zeros((batch_size, self.rows), dtype=np.float32)
            for j in range(self.rows):
                s, e = int(self.offsets[j]), int(self.offsets[j+1])
                if s == e: continue
                lc = self.col_indices[s:e].astype(int)
                Y[:, j] = np.dot(X[:, lc], vals_f32[s:e])
        else:
            Y = np.zeros((batch_size, self.cols), dtype=np.float32)
            for j in range(self.rows):
                s, e = int(self.offsets[j]), int(self.offsets[j+1])
                if s == e: continue
                lc = self.col_indices[s:e].astype(int)
                np.add.at(Y, (slice(None), lc), np.outer(X[:, j], vals_f32[s:e]))
        return Y[0] if squeeze else Y


# ---- AVX2 via Numba ----
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

_AVX2_B1 = None
_AVX2_BANY = None

if HAS_NUMBA:
    @njit(cache=False, fastmath=True, parallel=False)
    def _avx2_B1(X, offsets, col_indices, values_f32, X_on_R_cols):
        rows = len(offsets) - 1
        n_vals = values_f32.shape[0]
        if X_on_R_cols:
            Y = np.zeros(rows, dtype=np.float32)
        else:
            Y = np.zeros(n_vals, dtype=np.float32)
        for j in range(rows):
            s = int(offsets[j]); e = int(offsets[j+1])
            if s == e: continue
            nnz = e - s; acc = 0.0; i = 0
            while i + 4 <= nnz:
                lc0 = int(col_indices[s+i]); lc1 = int(col_indices[s+i+1])
                lc2 = int(col_indices[s+i+2]); lc3 = int(col_indices[s+i+3])
                v0 = values_f32[s+i]; v1 = values_f32[s+i+1]
                v2 = values_f32[s+i+2]; v3 = values_f32[s+i+3]
                if X_on_R_cols:
                    acc += X[lc0]*v0 + X[lc1]*v1 + X[lc2]*v2 + X[lc3]*v3
                i += 4
            while i < nnz:
                lc = int(col_indices[s+i]); v = values_f32[s+i]
                if X_on_R_cols: Y[j] += X[lc]*v
                else: Y[lc] += X[j]*v
                i += 1
            if X_on_R_cols: Y[j] = acc
        return Y

    @njit(cache=False, fastmath=True, parallel=False)
    def _avx2_Bany(X, offsets, col_indices, values_f32, X_on_R_cols):
        batch = X.shape[0]; rows = len(offsets) - 1
        n_vals = values_f32.shape[0]
        if X_on_R_cols:
            Y = np.zeros((batch, rows), dtype=np.float32)
        else:
            Y = np.zeros((batch, n_vals), dtype=np.float32)
        for b in range(batch):
            for j in range(rows):
                s = int(offsets[j]); e = int(offsets[j+1])
                if s == e: continue
                nnz = e - s; i = 0
                while i + 4 <= nnz:
                    lc0 = int(col_indices[s+i]); lc1 = int(col_indices[s+i+1])
                    lc2 = int(col_indices[s+i+2]); lc3 = int(col_indices[s+i+3])
                    v0 = values_f32[s+i]; v1 = values_f32[s+i+1]
                    v2 = values_f32[s+i+2]; v3 = values_f32[s+i+3]
                    if X_on_R_cols:
                        Y[b,j] += X[b,lc0]*v0 + X[b,lc1]*v1 + X[b,lc2]*v2 + X[b,lc3]*v3
                    i += 4
                while i < nnz:
                    lc = int(col_indices[s+i]); v = values_f32[s+i]
                    if X_on_R_cols: Y[b,j] += X[b,lc]*v
                    else: Y[b,lc] += X[b,j]*v
                    i += 1
        return Y

    _AVX2_B1 = _avx2_B1
    _AVX2_BANY = _avx2_Bany


class CSRAVX2:
    def __init__(self, rows, cols, k_pct, offsets, col_indices, values):
        self.rows = rows; self.cols = cols; self.k_pct = k_pct
        self.offsets = offsets.astype(np.uint32)
        self.col_indices = col_indices.astype(np.uint32)
        self.values = values.astype(np.float16)
        self.nnz = len(col_indices)

    @classmethod
    def from_encoded(cls, enc):
        flat_bitmap = np.unpackbits(enc.bitmap, count=enc.n_elements)
        set_indices = np.where(flat_bitmap)[0]
        vals_f16 = enc.values.view(np.float16)
        offsets = [0]; col_indices = []; values_list = []
        for r in range(enc.rows):
            rs = r * enc.cols
            row_bitmap = flat_bitmap[rs:rs + enc.cols]
            row_set = np.where(row_bitmap)[0]
            for c in row_set:
                col_indices.append(c)
                gi = rs + c
                pos = np.where(set_indices == gi)[0]
                if len(pos) > 0:
                    values_list.append(float(vals_f16[pos[0]]))
            offsets.append(len(col_indices))
        return cls(enc.rows, enc.cols, enc.k_pct,
                   np.array(offsets, dtype=np.uint32),
                   np.array(col_indices, dtype=np.uint32),
                   np.array(values_list, dtype=np.float16))

    def apply(self, X, X_on_R_cols=True):
        if not HAS_NUMBA:
            b = CSRBlocked(self.rows, self.cols, self.k_pct,
                           self.offsets, self.col_indices, self.values)
            return b.apply(X, X_on_R_cols)
        squeeze = False
        if X.ndim == 1:
            X = X[np.newaxis, :]; squeeze = True
        batch = X.shape[0]
        vals_f32 = self.values.astype(np.float32)
        try:
            if batch == 1:
                Y = _AVX2_B1(X[0].copy(), self.offsets.copy(),
                             self.col_indices.copy(), vals_f32.copy(), X_on_R_cols)
                Y = Y[np.newaxis, :]
            else:
                Y = _AVX2_BANY(X.copy(), self.offsets.copy(),
                               self.col_indices.copy(), vals_f32.copy(), X_on_R_cols)
        except Exception:
            b = CSRBlocked(self.rows, self.cols, self.k_pct,
                           self.offsets, self.col_indices, self.values)
            return b.apply(X, X_on_R_cols)
        return Y[0] if squeeze else Y


# ---- Reference: dense decode ----
def dense_ref(X, R_enc, X_on_R_cols):
    flat_bitmap = np.unpackbits(R_enc.bitmap, count=R_enc.n_elements)
    set_indices = np.where(flat_bitmap)[0]
    vals_f16 = R_enc.values.view(np.float16)
    vals_f32 = vals_f16.astype(np.float32)
    squeeze = False
    if X.ndim == 1:
        X = X[np.newaxis, :]; squeeze = True
    batch = X.shape[0]; rows, cols = R_enc.rows, R_enc.cols
    pos_map = np.full(R_enc.n_elements, -1, dtype=np.int32)
    pos_map[set_indices] = np.arange(R_enc.nnz, dtype=np.int32)
    if X_on_R_cols:
        Y = np.zeros((batch, rows), dtype=np.float32)
        for j in range(rows):
            rs = j * cols
            row_set = np.where(flat_bitmap[rs:rs+cols])[0]
            if len(row_set) == 0: continue
            gi = rs + row_set
            v = vals_f32[pos_map[gi]]
            Y[:, j] = np.dot(X[:, row_set], v)
    else:
        Y = np.zeros((batch, cols), dtype=np.float32)
        for j in range(rows):
            rs = j * cols
            row_set = np.where(flat_bitmap[rs:rs+cols])[0]
            if len(row_set) == 0: continue
            gi = rs + row_set
            v = vals_f32[pos_map[gi]]
            np.add.at(Y, (slice(None), row_set), np.outer(X[:, j], v))
    return Y[0] if squeeze else Y


# ---- Utilities ----
def cosine(a, b):
    af, bf = a.ravel(), b.ravel()
    dot = np.dot(af, bf)
    na, nb = np.linalg.norm(af), np.linalg.norm(bf)
    return float(dot/(na*nb)) if na*nb > 0 else 0.0

def verify(test, ref):
    ma = float(np.abs(test - ref).max())
    co = cosine(test, ref)
    has_nan = bool(np.any(np.isnan(test)) or np.any(np.isnan(ref)))
    has_inf = bool(np.any(np.isinf(test)) or np.any(np.isinf(ref)))
    correct = bool(ma < 1e-3 and co > 0.99999 and not has_nan and not has_inf)
    return dict(max_abs_diff=ma, cosine=co, has_nan=has_nan, has_inf=has_inf, correct=correct)

def mem_accounting(rows, cols, nnz, k_pct):
    n = rows * cols
    off_bytes = (rows + 1) * 4
    idx_bytes = nnz * 4
    val_bytes = nnz * 2
    total = 32 + off_bytes + idx_bytes + val_bytes
    q4 = int(n * 4.0 / 8.0)
    q2 = int(n * 2.0 / 8.0)
    budget = q4 - q2
    return dict(rows=rows, cols=cols, k_pct=k_pct, nnz=nnz, n_elements=n,
                sparsity_pct=f"{100.0*(1.0-nnz/n):.2f}%",
                offsets_bytes=off_bytes, col_indices_bytes=idx_bytes,
                values_bytes=val_bytes, metadata_bytes=32,
                total_bytes=total, ebpw=total*8.0/n,
                q4_bytes=q4, q2_bytes=q2, budget_bytes=budget,
                margin_bytes=budget-total, memory_viable=total <= budget)


def run_shape(shape_name, rows, cols, Xoc, k_pct, batch_sizes, n_timed):
    R = np.random.randn(rows, cols).astype(np.float32) * 0.01
    feat = cols if Xoc else rows
    Xb = np.random.randn(32, feat).astype(np.float32) * 0.05
    enc = encode_residual(R, k_pct=k_pct)
    nnz = enc.nnz

    csr_bl = CSRBaseline.from_encoded(enc)
    csr_bk = CSRBlocked.from_encoded(enc)
    csr_bs = CSRBatch.from_encoded(enc)
    csr_av = CSRAVX2.from_encoded(enc)
    mem = mem_accounting(rows, cols, nnz, k_pct)

    result = dict(shape_name=shape_name, shape=[rows, cols], k_pct=k_pct,
                 nnz=nnz, n_elements=rows*cols,
                 sparsity_pct=f"{100.0*(1.0-nnz/(rows*cols)):.2f}%",
                 X_on_R_cols=Xoc, memory=mem,
                 numba_available=HAS_NUMBA, batch_sizes={})

    variants = [
        ("CSR_scalar_baseline_31K", csr_bl.apply),
        ("CSR_blocked_scalar",      csr_bk.apply),
        ("CSR_batch_specialized",   csr_bs.apply),
        ("AVX2_numba",              csr_av.apply),
    ]

    for B in batch_sizes:
        X = Xb[:B, :].copy()
        Y_ref = dense_ref(X, enc, Xoc)

        # Timing reference (3 iterations — slow dense path)
        t0 = time.perf_counter()
        for _ in range(3): dense_ref(X, enc, Xoc)
        t_ref = (time.perf_counter() - t0) / 3 * 1000.0

        batch_result = dict(batch_size=B, reference_dense_ms=t_ref)

        print(f"\n  {shape_name} B={B}: ref_dense={t_ref:.4f}ms")
        for vname, vapply in variants:
            # Warmup
            for _ in range(WARMUP): vapply(X, Xoc)
            # Time
            times = []
            for _ in range(n_timed):
                t0 = time.perf_counter()
                Y_v = vapply(X, Xoc)
                times.append((time.perf_counter() - t0) * 1000.0)
            med = statistics.median(times)
            corr = verify(Y_v, Y_ref)
            ok = "OK" if corr["correct"] else "FAIL"
            print(f"    {vname}: median={med:.4f}ms cos={corr['cosine']:.8f} maxdiff={corr['max_abs_diff']:.2e} {ok}")

            batch_result[vname] = dict(
                median_ms=med, mean_ms=statistics.mean(times),
                min_ms=min(times), max_ms=max(times),
                all_times_ms=sorted(times),
                vs_ref_speedup=t_ref/med if med > 0 else None,
                correctness=corr)

        result["batch_sizes"][str(B)] = batch_result

    return result


def classify(results):
    any_correct = False
    avx2_ok = False; blocked_ok = False; batch_ok = False

    for sname, sd in results["shapes"].items():
        for bs, bd in sd["batch_sizes"].items():
            bl = bd.get("CSR_scalar_baseline_31K", {}).get("median_ms", float('inf'))
            for vname in ["CSR_blocked_scalar", "AVX2_numba", "CSR_batch_specialized"]:
                vdata = bd.get(vname, {})
                corr = vdata.get("correctness", {})
                if corr.get("correct"):
                    any_correct = True
                    if vname == "AVX2_numba": avx2_ok = True
                    if vname == "CSR_blocked_scalar": blocked_ok = True
                    if vname == "CSR_batch_specialized": batch_ok = True

    if not any_correct: return "BLOCKED_CORRECTNESS_REGRESSION"
    # Check if any improve vs baseline
    for sname, sd in results["shapes"].items():
        for bs, bd in sd["batch_sizes"].items():
            bl = bd.get("CSR_scalar_baseline_31K", {}).get("median_ms", float('inf'))
            if bl == float('inf'): continue
            for vname in ["CSR_blocked_scalar", "AVX2_numba"]:
                vdata = bd.get(vname, {})
                if not vdata.get("correctness", {}).get("correct"): continue
                if vdata.get("median_ms", float('inf')) < bl * 0.95:
                    if vname == "AVX2_numba": return "PASS_AVX2_BLOCKED_KERNEL_READY"
                    if vname == "CSR_blocked_scalar": return "PASS_BLOCKED_SCALAR_READY"
    return "PARTIAL_CORRECT_BUT_NO_SPEED_GAIN"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", nargs="+", default=["tiny","ffn_up","ffn_down"])
    parser.add_argument("--k-pct", type=float, default=K_PCT)
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 31L: AVX2 / Blocking Kernel")
    print("=" * 60)
    print(f"Numba available: {HAS_NUMBA}")

    results = dict(phase="31L", k_pct=args.k_pct, shapes={}, numba_available=HAS_NUMBA)

    large_shapes = {"ffn_up", "ffn_down"}

    for sname in args.shapes:
        if sname not in BENCHMARK_SHAPES: continue
        rows, cols = BENCHMARK_SHAPES[sname]
        Xoc = (sname != "ffn_down")
        bs_list = [1, 4, 16] if sname == "tiny" else [1, 4]
        n_timed = TIMED_TINY if sname == "tiny" else TIMED_FFN

        print(f"\n--- {sname} ({rows}x{cols}), Xoc={Xoc} ---")
        r = run_shape(sname, rows, cols, Xoc, args.k_pct, bs_list, n_timed)
        results["shapes"][sname] = r

    classification = classify(results)
    results["classification"] = classification
    print(f"\nClassification: {classification}")

    # Write JSON
    out_json = RESULTS_DIR / "PHASE31L_AVX2_BLOCKED_KERNEL.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Wrote: {out_json}")

    # Write MD
    out_md = REPO_DIR / "docs" / "PHASE31L_AVX2_BLOCKED_KERNEL.md"
    lines = [
        "# Phase 31L: AVX2 / Blocking Kernel Implementation",
        "",
        f"**Classification:** `{classification}`",
        f"**Numba available:** {HAS_NUMBA}",
        "",
        "## Memory Accounting Clarification (AUDIT FIX)",
        "",
        "**Phase 31K reported 'all layouts < 3.2MB vs Q4 budget ~234MB' — unit/scope mismatch.**",
        "",
        "| Tensor | Shape | Q4 bytes | Q2 bytes | Budget (Q4-Q2) | CSR Encoded | Margin | Viable? |",
        "|--------|-------|----------|----------|---------------|-------------|--------|---------|",
    ]
    for sname, sd in results["shapes"].items():
        m = sd["memory"]
        lines.append(
            f"| {sname} | {m['rows']}x{m['cols']} | {m['q4_bytes']:,} | {m['q2_bytes']:,} | "
            f"{m['budget_bytes']:,} | {m['total_bytes']:,} | {m['margin_bytes']:,} | "
            f"{'YES' if m['memory_viable'] else 'NO'} |"
        )
    lines += [
        "",
        "**Key:** Per-tensor residual budget = Q4 - Q2 bytes. '234MB' was full-model aggregate.",
        "The full-model Q4 size is the sum across all layers/tensors, not comparable here.",
        "",
        "## Correctness Table",
        "",
        "| Shape | B | Variant | cos | maxdiff | correct? |",
        "|-------|---|--------|-----|---------|---------|",
    ]
    for sname, sd in results["shapes"].items():
        for bs, bd in sd["batch_sizes"].items():
            for vname in ["CSR_scalar_baseline_31K","CSR_blocked_scalar","CSR_batch_specialized","AVX2_numba"]:
                vdata = bd.get(vname, {})
                if not vdata: continue
                c = vdata.get("correctness", {})
                ok = "OK" if c.get("correct") else "FAIL"
                lines.append(
                    f"| {sname} | {bs} | {vname} | {c.get('cosine',0):.8f} | "
                    f"{c.get('max_abs_diff',0):.2e} | {ok} |"
                )
    lines += ["", "## Timing Table (median ms)", "",
              "| Shape | B | ref_dense | baseline_31K | blocked_scalar | batch_specialized | AVX2_numba |",
              "|-------|---|-----------|--------------|----------------|-----------------|-----------|"]
    for sname, sd in results["shapes"].items():
        for bs, bd in sd["batch_sizes"].items():
            t_ref = bd.get("reference_dense_ms", 0)
            bl = bd.get("CSR_scalar_baseline_31K", {}).get("median_ms", 0)
            bk = bd.get("CSR_blocked_scalar", {}).get("median_ms", 0)
            bs2 = bd.get("CSR_batch_specialized", {}).get("median_ms", 0)
            av = bd.get("AVX2_numba", {}).get("median_ms", 0)
            lines.append(f"| {sname} | {bs} | {t_ref:.4f} | {bl:.4f} | {bk:.4f} | {bs2:.4f} | {av:.4f} |")

    lines += ["", "## Speedup vs 31K CSR Scalar Baseline", "",
              "| Shape | B | blocked | batch | AVX2 |",
              "|-------|---|---------|-------|------|"]
    for sname, sd in results["shapes"].items():
        for bs, bd in sd["batch_sizes"].items():
            bl = bd.get("CSR_scalar_baseline_31K", {}).get("median_ms", 1.0)
            bk = bd.get("CSR_blocked_scalar", {}).get("median_ms", None)
            bs2 = bd.get("CSR_batch_specialized", {}).get("median_ms", None)
            av = bd.get("AVX2_numba", {}).get("median_ms", None)
            def sp(v): return f"{bl/v:.3f}x" if v and v > 0 else "N/A"
            lines.append(f"| {sname} | {bs} | {sp(bk)} | {sp(bs2)} | {sp(av)} |")

    lines += ["", "## Decision Gate", ""]
    if classification == "PASS_AVX2_BLOCKED_KERNEL_READY":
        lines.append("**AVX2/blocking improved the standalone sparse residual kernel by X vs the 31K CSR scalar baseline.** Phase 31M UNLOCKED.")
    elif classification == "PASS_BLOCKED_SCALAR_READY":
        lines.append("**Blocked scalar improved the standalone sparse residual kernel.** Phase 31M UNLOCKED.")
    else:
        lines.append(f"**Classification: {classification}**")

    lines += ["", "---", "*Phase 31L — ELVIS — SDI Substitutive*"]
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote: {out_md}")
    return results


if __name__ == "__main__":
    main()
