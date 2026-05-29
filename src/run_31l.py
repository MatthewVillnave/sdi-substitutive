#!/usr/bin/env python3
"""Phase 31L benchmark: scalar kernel variants, pure Python."""
import sys, json, time, statistics
sys.path.insert(0, '.')
import numpy as np
np.random.seed(42)

from residual_encode import encode_residual
from phase31l_avx2_blocked_kernel import (
    CSRBaseline, CSRBlocked, CSRBatch,
    dense_ref, verify, mem_accounting)

WARMUP = 5
TIMED_TINY = 40
TIMED_FFN = 25

RESULTS = {}

def benchmark(vapply, X, Xoc, warmup=WARMUP, timed=TIMED_TINY):
    for _ in range(warmup): vapply(X, Xoc)
    t0 = time.perf_counter()
    for _ in range(timed): vapply(X, Xoc)
    return (time.perf_counter() - t0) / timed * 1000.0

def run_shape(name, rows, cols, Xoc, batch_sizes, n_timed):
    print(f"\n=== {name} ({rows}x{cols}) Xoc={Xoc} ===", flush=True)
    R = np.random.randn(rows, cols).astype(np.float32) * 0.01
    feat = cols if Xoc else rows
    X_full = np.random.randn(32, feat).astype(np.float32) * 0.05
    enc = encode_residual(R, k_pct=7.5)
    nnz = enc.nnz
    mem = mem_accounting(rows, cols, nnz, 7.5)
    print(f"nnz={nnz} encoded={mem['total_bytes']:,}/{mem['budget_bytes']:,} "
          f"viable={mem['memory_viable']}", flush=True)

    csr_bl = CSRBaseline.from_encoded(enc)
    csr_bk = CSRBlocked.from_encoded(enc)
    csr_bs = CSRBatch.from_encoded(enc)

    shape_result = {"shape_name": name, "shape": [rows, cols], "nnz": nnz,
                    "memory": mem, "batch_sizes": {}}
    RESULTS[name] = shape_result

    for B in batch_sizes:
        X = X_full[:B, :].copy()
        Y_ref = dense_ref(X, enc, Xoc)
        print(f"\n  B={B}:", flush=True)
        bdata = {"batch_size": B}
        RESULTS[name][B] = bdata

        for vn, va in [
            ("CSR_scalar_baseline_31K", csr_bl.apply),
            ("CSR_blocked_scalar",      csr_bk.apply),
            ("CSR_batch_specialized",   csr_bs.apply),
        ]:
            t = benchmark(va, X, Xoc, WARMUP, n_timed)
            Y_v = va(X, Xoc)
            corr = verify(Y_v, Y_ref)
            ok = "OK" if corr["correct"] else "FAIL"
            print(f"    {vn}: {t:.4f}ms cos={corr['cosine']:.8f} {ok}", flush=True)
            bdata[vn] = {"median_ms": t, "correctness": corr}

    return shape_result

# === TINY ===
run_shape("tiny", 8, 16, True,  [1, 4, 16], TIMED_TINY)

# === FFN_UP ===
run_shape("ffn_up", 896, 4864, True,  [1, 4], TIMED_FFN)

# === FFN_DOWN ===
run_shape("ffn_down", 4864, 896, False, [1, 4], TIMED_FFN)

# === CLASSIFICATION ===
print("\n=== CLASSIFICATION ===", flush=True)
classification = "PASS_BLOCKED_SCALAR_READY"
# Verify all scalar variants correct
all_correct = True
for name in ["tiny", "ffn_up", "ffn_down"]:
    for B in RESULTS[name]:
        if not isinstance(RESULTS[name][B], dict): continue
        for vn in ["CSR_scalar_baseline_31K", "CSR_blocked_scalar", "CSR_batch_specialized"]:
            v = RESULTS[name][B].get(vn, {})
            if not v.get("correctness", {}).get("correct"):
                all_correct = False
                print(f"  INCORRECT: {name} B={B} {vn}", flush=True)

if not all_correct:
    classification = "BLOCKED_CORRECTNESS_REGRESSION"
else:
    # Check if blocked improves over baseline
    improved = False
    for name in ["ffn_up", "ffn_down"]:
        for B in [1, 4]:
            bl = RESULTS[name][B].get("CSR_scalar_baseline_31K", {}).get("median_ms", float('inf'))
            bk = RESULTS[name][B].get("CSR_blocked_scalar", {}).get("median_ms", float('inf'))
            if bk < bl * 0.95:
                improved = True
                print(f"  IMPROVED: {name} B={B} blocked={bk:.4f}ms baseline={bl:.4f}ms", flush=True)
    if improved:
        classification = "PASS_BLOCKED_SCALAR_READY"
    else:
        classification = "PARTIAL_CORRECT_BUT_NO_SPEED_GAIN"

print(f"Classification: {classification}", flush=True)
RESULTS["_classification"] = classification

# Write JSON
out_json = "/home/matthew-villnave/sdi-substitutive/results/PHASE31L_AVX2_BLOCKED_KERNEL.json"
with open(out_json, "w") as f:
    json.dump(RESULTS, f, indent=2, default=float)
print(f"Wrote: {out_json}", flush=True)
