#!/usr/bin/env python3
"""
Phase 31Y-R: Manifest Runtime Approximation Discrepancy Audit

Compares Phase 31Y (manifest path, U@V W_ref construction) vs 
Phase 31T (direct path, rng.randn W_ref construction) to find why
31Y produces cos_sub ~0.789 vs 31T's cos_sub ~0.9956.

Finds: W_ref construction METHOD mismatch (not just seed).
Root cause: 31Y uses rank-448 U@V decomposition; 31T uses full-rank randn.
"""
import os, sys, json, struct
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src'))

from phase31x_manifest_runtime import (
    ManifestRuntime, pack_wlow, encode_sdir, sdiw_streaming_apply,
    sdir_streaming_apply, cosine, BLOCK_SIZE, RSC_MAGIC
)
from wlow_pack import pack_wlow as wlow_pack_official
from residual_encode import encode_residual

ROWS, COLS = 896, 4864
LAYERS = [0, 1, 2, 3, 4, 5]

# Phase 31T seeds
T_SEEDS = {0: 0, 1: 43, 2: 44, 3: 45, 4: 46, 5: 47}
# Phase 31Y seeds
Y_SEEDS = {l: l * 43 + 7 for l in LAYERS}

def cosine(a, b):
    a, b = a.ravel(), b.ravel()
    n = np.linalg.norm(a) * np.linalg.norm(b)
    if n == 0: return 0.0
    return float(np.dot(a, b) / n)

print("=" * 70)
print("Phase 31Y-R: Approximation Discrepancy Audit")
print("=" * 70)

# ── Step 1: W_ref construction method comparison ──────────────────────────
print("\n### STEP 1: W_ref Construction Method Comparison ###\n")

X_ones = np.ones(ROWS, dtype=np.float32)

for layer in LAYERS:
    t_seed = T_SEEDS[layer]
    y_seed = Y_SEEDS[layer]

    # 31T style: rng.randn(direct)
    rng_t = np.random.RandomState(t_seed)
    W_ref_T = rng_t.randn(ROWS, COLS).astype(np.float32) * 0.1

    # 31Y style: U @ V * 0.1
    np.random.seed(y_seed)
    U_y = np.random.randn(ROWS, min(ROWS, COLS)//2).astype(np.float32)
    V_y = np.random.randn(min(ROWS, COLS)//2, COLS).astype(np.float32)
    W_ref_Y = U_y @ V_y * 0.1

    # Correlation between the two W_ref matrices
    corr_W_ref = cosine(W_ref_T, W_ref_Y)
    
    # Norms
    norm_T = float(np.linalg.norm(W_ref_T))
    norm_Y = float(np.linalg.norm(W_ref_Y))
    
    # Output for X=ones
    Y_ref_T = X_ones @ W_ref_T
    Y_ref_Y = X_ones @ W_ref_Y
    corr_Y_ref = cosine(Y_ref_T, Y_ref_Y)
    
    # Matrix rank estimate
    rank_T = np.linalg.matrix_rank(W_ref_T)
    rank_Y = np.linalg.matrix_rank(W_ref_Y)

    print(f"  Layer {layer}: 31T seed={t_seed}, 31Y seed={y_seed}")
    print(f"    31T W_ref: norm={norm_T:.4f}, rank={rank_T}")
    print(f"    31Y W_ref: norm={norm_Y:.4f}, rank={rank_Y}")
    print(f"    W_ref cosine(corr): {corr_W_ref:.6f}")
    print(f"    Y_ref cosine(corr): {corr_Y_ref:.6f}")
    print(f"    ⚠️  SAME SEED? {'NO' if t_seed != y_seed else 'YES'} — BUT METHOD DIFFERS ANYWAY")

# ── Step 2: W_low comparison for layer 0 ─────────────────────────────────
print("\n### STEP 2: W_low Construction Comparison (Layer 0) ###\n")

layer = 0
t_seed = T_SEEDS[layer]
y_seed = Y_SEEDS[layer]

# 31T W_ref
rng_t = np.random.RandomState(t_seed)
W_ref_T = rng_t.randn(ROWS, COLS).astype(np.float32) * 0.1

# 31Y W_ref
np.random.seed(y_seed)
U_y = np.random.randn(ROWS, min(ROWS, COLS)//2).astype(np.float32)
V_y = np.random.randn(min(ROWS, COLS)//2, COLS).astype(np.float32)
W_ref_Y = U_y @ V_y * 0.1

# 31T W_low (q4 style, scale / 7.0)
def q4_qdq(W, block_size=32):
    flat = W.flatten()
    n = len(flat)
    nb = (n + block_size - 1) // block_size
    out = np.zeros(n, dtype=np.float32)
    for b in range(nb):
        s, e = b * block_size, min((b+1) * block_size, n)
        block = flat[s:e]
        scale = float(np.abs(block).max()) / 7.0
        if scale < 1e-8: scale = 1.0
        q = np.clip(np.round(block / scale), -8, 7).astype(np.int8)
        out[s:e] = q * scale
    return out.reshape(W.shape)

# 31Y W_low (scale / 7.5, block32)
def wlow_qdq_31y(W, block_size=32):
    flat = W.flatten()
    n = len(flat)
    nb = n // block_size
    out = np.zeros(n, dtype=np.float32)
    for b in range(nb):
        s, e = b * block_size, (b+1) * block_size
        block = flat[s:e]
        scale = float(np.abs(block).max()) / 7.5
        if scale < 1e-8: scale = 1.0
        q = np.clip(np.round(block / scale), -8, 7).astype(np.int8)
        out[s:e] = q * scale
    return out.reshape(W.shape)

W_low_T = q4_qdq(W_ref_T)
W_low_Y = wlow_qdq_31y(W_ref_Y)

cos_W_low = cosine(W_ref_T, W_low_T)
cos_W_low_Y = cosine(W_ref_Y, W_low_Y)
print(f"  31T: W_ref norm={float(np.linalg.norm(W_ref_T)):.4f}, cos(W_ref, W_low)={cos_W_low:.6f}")
print(f"  31Y: W_ref norm={float(np.linalg.norm(W_ref_Y)):.4f}, cos(W_ref, W_low_Y)={cos_W_low_Y:.6f}")

# ── Step 3: Residual apply comparison ────────────────────────────────────
print("\n### STEP 3: Residual Apply Comparison (Layer 0) ###\n")

R_T = W_ref_T - W_low_T
R_Y = W_ref_Y - W_low_Y

# 31T residual: encode with 7.5% k
enc_T = encode_residual(R_T, k_pct=7.5)

# 31Y residual: encode with encode_sdir (same k_pct)
sdir_T = enc_T
sdir_Y_bytes = encode_sdir(R_Y, k_pct=7.5)

# Dense residual apply: X @ R
Y_delta_T_dense = X_ones @ R_T
Y_delta_Y_dense = X_ones @ R_Y

print(f"  31T R: norm={float(np.linalg.norm(R_T)):.4f}, nnz_top7.5%={int(np.sum(np.abs(R_T) > 1e-9))}")
print(f"  31Y R: norm={float(np.linalg.norm(R_Y)):.4f}, nnz_top7.5%={int(np.sum(np.abs(R_Y) > 1e-9))}")

# sdir_streaming_apply for 31Y
Y_delta_Y_stream, vp_Y = sdir_streaming_apply(sdir_Y_bytes, X_ones, ROWS, COLS)

# Dense reference for 31Y
Y_delta_Y_ref = X_ones @ R_Y

max_diff_stream = float(np.max(np.abs(Y_delta_Y_dense - Y_delta_Y_stream)))
cos_stream = cosine(Y_delta_Y_dense, Y_delta_Y_stream)
print(f"\n  Y_delta stream vs dense (31Y R): max_diff={max_diff_stream:.2e}, cos={cos_stream:.8f}")
print(f"  Stream output: norm={float(np.linalg.norm(Y_delta_Y_stream)):.4f}")
print(f"  Dense output:  norm={float(np.linalg.norm(Y_delta_Y_dense)):.4f}")

# ── Step 4: Full substitutive comparison ─────────────────────────────────
print("\n### STEP 4: Full Substitutive Path Comparison (Layer 0) ###\n")

# 31T-style
Y_ref_T = X_ones @ W_ref_T
Y_low_T_dense = X_ones @ W_low_T

# 31Y-style via manifest runtime (rebuild everything inline)
packed_Y, scales_Y = wlow_pack_official(W_low_Y)
Y_low_Y_stream = sdiw_streaming_apply(packed_Y, scales_Y, X_ones, ROWS, COLS)
Y_delta_Y_stream, vp = sdir_streaming_apply(sdir_Y_bytes, X_ones, ROWS, COLS)
Y_sub_Y_stream = Y_low_Y_stream + Y_delta_Y_stream

# Dense 31Y (reconstructed)
Y_low_Y_dense = X_ones @ W_low_Y
Y_sub_Y_dense = Y_low_Y_dense + (X_ones @ R_Y)

cos_sub_stream_vs_ref = cosine(Y_ref_T, Y_sub_Y_stream)  # compare against 31T's ref
cos_sub_stream_vs_Yref = cosine(Y_ref_Y, Y_sub_Y_stream)  # compare against 31Y's ref
cos_sub_dense_vs_Yref = cosine(Y_ref_Y, Y_sub_Y_dense)

mae_stream_vs_ref = float(np.abs(Y_ref_T - Y_sub_Y_stream).mean())
mae_stream_vs_Yref = float(np.abs(Y_ref_Y - Y_sub_Y_stream).mean())

print(f"  Y_ref_T (31T reference): norm={float(np.linalg.norm(Y_ref_T)):.4f}")
print(f"  Y_ref_Y (31Y reference): norm={float(np.linalg.norm(Y_ref_Y)):.4f}")
print(f"  Ratio norms: {float(np.linalg.norm(Y_ref_Y)) / float(np.linalg.norm(Y_ref_T)):.4f}")
print()
print(f"  31Y stream vs 31T ref: cos={cos_sub_stream_vs_ref:.6f}, MAE={mae_stream_vs_ref:.4e}")
print(f"  31Y stream vs 31Y ref: cos={cos_sub_stream_vs_Yref:.6f}, MAE={mae_stream_vs_Yref:.4e}")
print(f"  31Y dense  vs 31Y ref: cos={cos_sub_dense_vs_Yref:.6f}")

# Key diagnostic: Is the residual being applied?
Y_low_stream_only = Y_low_Y_stream
cos_low_only = cosine(Y_ref_Y, Y_low_stream_only)
cos_with_delta = cosine(Y_ref_Y, Y_sub_Y_stream)
print(f"\n  cos(Y_ref_Y, Y_low_only)  = {cos_low_only:.6f}")
print(f"  cos(Y_ref_Y, Y_sub)        = {cos_with_delta:.6f}")
print(f"  delta_cos (31Y path)       = {cos_with_delta - cos_low_only:+.6f}")
print(f"  31Y manifest delta_cos    ≈ +0.00005 (reported)")
print(f"  31T delta_cos             ≈ +0.00131 (reported)")

# ── Step 5: Compare manifest path vs direct dense for 31Y (same W_ref) ───
print("\n### STEP 5: Manifest Path vs Direct Dense (31Y W_ref, Layer 0) ###\n")

# Both use 31Y W_ref, compare streaming vs dense
Y_sub_direct = Y_low_Y_dense + (X_ones @ R_Y)
max_diff = float(np.max(np.abs(Y_sub_Y_stream - Y_sub_direct)))
cos_diff = cosine(Y_sub_Y_stream, Y_sub_direct)
print(f"  Manifest vs Direct: max_abs_diff={max_diff:.4e}, cos={cos_diff:.8f}")

# ── Step 6: Key stats table per layer ────────────────────────────────────
print("\n### STEP 6: Per-Layer W_ref Norm & Y_ref Comparison ###\n")

print(f"  {'Layer':>5} | {'31T Y_ref norm':>14} | {'31Y Y_ref norm':>14} | {'Y_ref cos':>10} | {'31T cos_sub':>12} | {'31Y cos_sub':>12}")
print(f"  {'-'*5}-+-{'-'*14}-+-{'-'*14}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}")

for layer in LAYERS:
    t_seed = T_SEEDS[layer]
    y_seed = Y_SEEDS[layer]
    
    rng_t = np.random.RandomState(t_seed)
    W_ref_T = rng_t.randn(ROWS, COLS).astype(np.float32) * 0.1
    Y_ref_T = X_ones @ W_ref_T
    
    np.random.seed(y_seed)
    U_y = np.random.randn(ROWS, min(ROWS, COLS)//2).astype(np.float32)
    V_y = np.random.randn(min(ROWS, COLS)//2, COLS).astype(np.float32)
    W_ref_Y = U_y @ V_y * 0.1
    Y_ref_Y = X_ones @ W_ref_Y
    
    corr_Y = cosine(Y_ref_T, Y_ref_Y)
    
    # Get 31T cos_sub from known results
    t_cos_sub = [0.9957165678, 0.9955961665, 0.9957480153, 0.9956667264, 0.9956202467, 0.9956248164][layer]
    
    # Get 31Y cos_sub from known results
    y_cos_sub = [0.7993956208, 0.7792413235, 0.8002089858, 0.7856018543, 0.7773125768, 0.7933056355][layer]
    
    print(f"  {layer:>5} | {float(np.linalg.norm(Y_ref_T)):>14.4f} | {float(np.linalg.norm(Y_ref_Y)):>14.4f} | {corr_Y:>10.6f} | {t_cos_sub:>12.6f} | {y_cos_sub:>12.6f}")

# ── Step 7: Residual magnitude analysis ──────────────────────────────────
print("\n### STEP 7: Residual Magnitude Analysis ###\n")

# For 31Y, with X=ones, what's the expected Y_delta norm?
np.random.seed(Y_SEEDS[0])
U_y = np.random.randn(ROWS, min(ROWS, COLS)//2).astype(np.float32)
V_y = np.random.randn(min(ROWS, COLS)//2, COLS).astype(np.float32)
W_ref_Y = U_y @ V_y * 0.1
W_low_Y = wlow_qdq_31y(W_ref_Y)
R_Y = W_ref_Y - W_low_Y

nnz_R = int(np.sum(np.abs(R_Y) > 1e-9))
k_pct_actual = nnz_R / (ROWS * COLS) * 100
print(f"  31Y Layer 0 residual nnz: {nnz_R:,} ({k_pct_actual:.2f}%)")
print(f"  R norm: {float(np.linalg.norm(R_Y)):.4f}")
print(f"  R mean abs: {float(np.mean(np.abs(R_Y))):.6f}")
print(f"  Y_delta norm (X=ones): {float(np.linalg.norm(X_ones @ R_Y)):.4f}")

# Compare: what fraction of Y_ref does Y_delta represent?
Y_ref_Y = X_ones @ W_ref_Y
print(f"  Y_ref norm: {float(np.linalg.norm(Y_ref_Y)):.4f}")
print(f"  Y_delta / Y_ref ratio: {float(np.linalg.norm(X_ones @ R_Y)) / float(np.linalg.norm(Y_ref_Y)):.6f}")

# What cosine does adding this delta produce?
cos_low_only = cosine(Y_ref_Y, X_ones @ W_low_Y)
cos_sub = cosine(Y_ref_Y, X_ones @ W_low_Y + X_ones @ R_Y)
print(f"\n  cos(Y_ref, Y_low)  = {cos_low_only:.6f}")
print(f"  cos(Y_ref, Y_sub)   = {cos_sub:.6f}")
print(f"  delta_cos            = {cos_sub - cos_low_only:+.6f}")

print("\n" + "=" * 70)
print("FINDINGS SUMMARY")
print("=" * 70)
