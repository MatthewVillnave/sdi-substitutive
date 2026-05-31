#!/usr/bin/env python3
"""
Phase 31BC — Alternative Residual Formulation / Output-Encoding Probe
Repo: sdi-substitutive

Tests whether direct output-residual encoding (Y_ref - Y_low) fixes the
layer21-seed9 severe cosine regression better than per-family weight residual.

IMPORTANT: Output residual is activation-specific. This is a diagnostic oracle
probe only. No runtime-ready claim is made.

Pairs: L21-S9 (severe), L2-S7 (mild), L21-S0/S5/S14 (safe), L20-S9, L22-S9
k sweep on output residual: 1%, 2%, 5%, 10%, 20%, 50%, 100%
"""

import ctypes, json, os, sys, time
import numpy as np
sys.path.insert(0, '/home/matthew-villnave/sdi-substitutive/src')
sys.path.insert(0, '/home/matthew-villnave/llama.cpp/gguf-py')
from gguf import GGUFReader
from gguf.quants import dequantize
from phase31x_manifest_runtime import encode_sdir, decode_sdir

LIB = '/home/matthew-villnave/llama.cpp/build/bin/libggml-base.so'
Q4_BUDGET_FAMILY = 2179072
Q4_BUDGET_LAYER  = Q4_BUDGET_FAMILY * 3
QK_K = 256
Q2_BLOCK_BYTES = 84

lib = ctypes.CDLL(LIB)
lib.quantize_row_q2_K_ref.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_int]
lib.quantize_row_q2_K_ref.restype = None
lib.dequantize_row_q2_K.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int]
lib.dequantize_row_q2_K.restype = None

def q2_encode(flat):
    n = flat.size
    buf = np.zeros(n // QK_K * Q2_BLOCK_BYTES, dtype=np.uint8)
    lib.quantize_row_q2_K_ref(
        flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(buf.ctypes.data, ctypes.POINTER(ctypes.c_void_p)),
        n)
    return buf

def q2_decode(buf, n):
    out = np.zeros(n, dtype=np.float32)
    lib.dequantize_row_q2_K(
        ctypes.cast(buf.ctypes.data, ctypes.POINTER(ctypes.c_void_p)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n)
    return out

def silu(x): return x / (1.0 + np.exp(-np.clip(x, -709, 709)))

def cos_sim(a, b):
    a = a.flatten(); b = b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def topk_sparse(flat, k_frac):
    """Return (indices, values, n_nnz) for top-k fraction by magnitude."""
    n = flat.size
    n_nnz = max(1, int(round(n * k_frac)))
    absv = np.abs(flat)
    indices = np.argpartition(absv.flatten(), -n_nnz)[-n_nnz:]
    indices = indices[np.argsort(-absv.flatten()[indices])]
    vals = flat.flatten()[indices]
    return indices, vals, n_nnz

def dense_residual_bytes(vec_size, dtype):
    if dtype == 'fp16': return vec_size * 2
    if dtype == 'int8':  return vec_size * 1
    if dtype == 'fp32':  return vec_size * 4
    return 0

print("Loading model...")
reader = GGUFReader('/media/matthew-villnave/VL_usb/models/qwen2.5-0.5b-official/qwen2.5-0.5b-instruct-q4_k_m.gguf')

families_all = {}
for li in range(24):
    families_all[li] = {}
    for fam in ['ffn_up', 'ffn_gate', 'ffn_down']:
        t = next(x for x in reader.tensors if x.name == f'blk.{li}.{fam}.weight')
        W_ref = dequantize(t.data, t.tensor_type).astype(np.float32)
        buf = q2_encode(W_ref.flatten())
        W_low = q2_decode(buf, W_ref.size).reshape(W_ref.shape)
        families_all[li][fam] = {
            'W_ref': W_ref, 'W_low': W_low,
            'R': W_ref - W_low, 'base_bytes': buf.nbytes}

def mlp(X, Wg, Wu, Wd):
    return (silu(X @ Wg.T) * (X @ Wu.T)) @ Wd.T

def run_layer_weight_residual(X, families, k, alpha=1.0):
    """Standard per-family weight residual at sparsity k."""
    Y_ref = mlp(X,
                families['ffn_gate']['W_ref'],
                families['ffn_up']['W_ref'],
                families['ffn_down']['W_ref'])
    Y_low = mlp(X,
                families['ffn_gate']['W_low'],
                families['ffn_up']['W_low'],
                families['ffn_down']['W_low'])
    if k == 0:
        Y_sub = Y_low
        res_bytes = 0
    else:
        dec = {fam: decode_sdir(encode_sdir(alpha * families[fam]['R'], k))
               for fam in families}
        Y_sub = mlp(X,
                    families['ffn_gate']['W_low'] + dec['ffn_gate'],
                    families['ffn_up']['W_low']   + dec['ffn_up'],
                    families['ffn_down']['W_low']  + dec['ffn_down'])
        res_bytes = sum(len(encode_sdir(families[f]['R'], k)) for f in families)
    base_bytes = sum(families[f]['base_bytes'] for f in families)
    margin = Q4_BUDGET_LAYER - (base_bytes + res_bytes)
    cos_low = cos_sim(Y_ref, Y_low)
    cos_sub = cos_sim(Y_ref, Y_sub)
    mae_low = float(np.abs(Y_ref - Y_low).mean())
    mae_sub = float(np.abs(Y_ref - Y_sub).mean())
    mae_delta = mae_sub - mae_low
    return {
        'Y_ref': Y_ref, 'Y_low': Y_low, 'Y_sub': Y_sub,
        'cos_low': round(cos_low, 8), 'cos_sub': round(cos_sub, 8),
        'delta_cos': round(cos_sub - cos_low, 8),
        'cosine_improved': cos_sub > cos_low,
        'cosine_nonnegative': cos_sub >= cos_low,
        'severe_regression': (cos_sub - cos_low) < -0.05,
        'mae_low': round(mae_low, 8), 'mae_sub': round(mae_sub, 8),
        'mae_delta': round(mae_delta, 8),
        'mae_improved': mae_delta < 0,
        'memory_positive': margin >= 0,
        'margin': margin,
        'residual_bytes': res_bytes,
        'base_bytes': base_bytes,
    }

def apply_output_residual(Y_low, indices, vals, dtype='fp16'):
    """Apply sparse output residual to Y_low.

    IMPORTANT: Must use Y_out.flat[indices] (view) not Y_out.flatten()[indices] (copy).
    np.flatten() returns a copy, so assignments to it are silently dropped.
    """
    Y_out = Y_low.copy()
    if dtype == 'fp16':
        Y_out.flat[indices] = Y_low.flat[indices] + vals
    elif dtype == 'int8':
        Y_out.flat[indices] = Y_low.flat[indices] + vals.astype(np.int8).astype(np.float32)
    else:
        Y_out.flat[indices] = Y_low.flat[indices] + vals
    return Y_out

print("Computing baselines for all target pairs...")
TARGET_PAIRS = [
    (21, 9,  'L21_seed9_SEVERE'),
    (2,  7,  'L2_seed7_mild'),
    (21, 0,  'L21_seed0_safe'),
    (21, 5,  'L21_seed5_safe'),
    (21, 14, 'L21_seed14_safe'),
    (20, 9,  'L20_seed9'),
    (22, 9,  'L22_seed9'),
]

OUTPUT_K_FRACS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]

all_results = {}

for li, seed, label in TARGET_PAIRS:
    print(f"\n=== {label} (layer={li}, seed={seed}) ===")
    families = families_all[li]
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((1, 896)).astype(np.float32)

    # Q2-only baseline
    base = run_layer_weight_residual(X, families, k=0)
    print(f"  Q2-only:  cos_low={base['cos_low']:.6f}  mae_low={base['mae_low']:.6f}")

    # Q2 + weight residual at k=1 (current)
    cur = run_layer_weight_residual(X, families, k=1.0)
    print(f"  WR k=1:   delta_cos={cur['delta_cos']:+.6f}  MAE_delta={cur['mae_delta']:+.6f}  "
          f"cos_impr={cur['cosine_improved']}  severe={cur['severe_regression']}  "
          f"mem_pos={cur['memory_positive']}")

    # Dense output residual oracle
    R_y = base['Y_ref'] - base['Y_low']
    Y_oracle = base['Y_low'] + R_y
    cos_oracle = cos_sim(base['Y_ref'], Y_oracle)
    mae_oracle = float(np.abs(base['Y_ref'] - Y_oracle).mean())
    vec_size = R_y.size
    dense_fp16 = dense_residual_bytes(vec_size, 'fp16')
    dense_int8 = dense_residual_bytes(vec_size, 'int8')
    dense_fp32 = dense_residual_bytes(vec_size, 'fp32')
    print(f"  Oracle:   cos={cos_oracle:.8f}  mae={mae_oracle:.8f}  "
          f"bytes_fp16={dense_fp16}  bytes_int8={dense_int8}")

    pair_results = {
        'label': label,
        'layer': li,
        'seed': seed,
        'q2_only': {k: v for k, v in base.items()
                    if k in ('cos_low', 'mae_low')},
        'weight_residual_k1': {k: v for k, v in cur.items()
                               if k in ('delta_cos', 'cosine_improved',
                                        'severe_regression', 'mae_delta',
                                        'mae_improved', 'memory_positive',
                                        'residual_bytes', 'margin')},
        'dense_oracle': {
            'cos': round(cos_oracle, 8),
            'mae': round(mae_oracle, 8),
            'bytes_fp16': dense_fp16,
            'bytes_int8': dense_int8,
            'bytes_fp32': dense_fp32,
            'vec_size': vec_size,
        },
        'sparse_output': [],
    }

    # Sparse output residual sweep
    for kf in OUTPUT_K_FRACS:
        indices, vals, n_nnz = topk_sparse(R_y, kf)
        # fp16
        Y_out_fp16 = apply_output_residual(base['Y_low'], indices, vals, 'fp16')
        cos_fp16 = cos_sim(base['Y_ref'], Y_out_fp16)
        mae_fp16 = float(np.abs(base['Y_ref'] - Y_out_fp16).mean())
        bytes_fp16 = n_nnz * 2 + n_nnz * 4  # fp16 indices + fp16 values

        # int8
        Y_out_int8 = apply_output_residual(base['Y_low'], indices, vals, 'int8')
        cos_int8 = cos_sim(base['Y_ref'], Y_out_int8)
        mae_int8 = float(np.abs(base['Y_ref'] - Y_out_int8).mean())
        bytes_int8 = n_nnz * 2 + n_nnz * 1  # fp16 indices + int8 values

        for dtype, cos_val, mae_val, bbytes in [
            ('fp16', cos_fp16, mae_fp16, bytes_fp16),
            ('int8', cos_int8, mae_int8, bytes_int8),
        ]:
            dcos = cos_val - base['cos_low']
            dmae = mae_val - base['mae_low']
            pair_results['sparse_output'].append({
                'k_frac': kf,
                'n_nnz': int(n_nnz),
                'dtype': dtype,
                'residual_bytes': int(bbytes),
                'cos': round(cos_val, 8),
                'delta_cos': round(dcos, 8),
                'cosine_improved': bool(cos_val > base['cos_low']),
                'severe_regression': bool(dcos < -0.05),
                'mae': round(mae_val, 8),
                'mae_delta': round(dmae, 8),
                'mae_improved': bool(mae_val < base['mae_low']),
            })

        print(f"  R_y k={kf*100:5.2f}%: fp16 cos={cos_fp16:.6f} dcos={cos_fp16-base['cos_low']:+.6f}  "
              f"int8 cos={cos_int8:.6f} dcos={cos_int8-base['cos_low']:+.6f}  "
              f"bytes={bytes_fp16}/{bytes_int8}")

    all_results[label] = pair_results

# Optional aggregate mini-sweep: layers 2 and 21, seeds 0-15
print("\n\n=== OPTIONAL AGGREGATE MINI-SWEEP (layers 2+21, seeds 0-15) ===")
agg_k_values = [0.01, 0.05, 0.10]
agg_results = {kf: {'cos_fail': 0, 'severe': 0, 'mae_fail': 0, 'total': 0}
               for kf in agg_k_values}

for li in [21, 2]:
    families = families_all[li]
    for seed in range(16):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((1, 896)).astype(np.float32)
        base = run_layer_weight_residual(X, families, k=0)
        R_y = base['Y_ref'] - base['Y_low']
        for kf in agg_k_values:
            indices, vals, n_nnz = topk_sparse(R_y, kf)
            Y_out = apply_output_residual(base['Y_low'], indices, vals, 'fp16')
            cos_out = cos_sim(base['Y_ref'], Y_out)
            mae_out = float(np.abs(base['Y_ref'] - Y_out).mean())
            dcos = cos_out - base['cos_low']
            dmae = mae_out - base['mae_low']
            agg_results[kf]['total'] += 1
            if dcos < 0:     agg_results[kf]['cos_fail'] += 1
            if dcos < -0.05: agg_results[kf]['severe'] += 1
            if dmae > 0:     agg_results[kf]['mae_fail'] += 1

print("\nAggregate (layers 2+21, seeds 0-15):")
for kf, st in agg_results.items():
    print(f"  k={kf*100:5.2f}%: cos_fail={st['cos_fail']}/{st['total']}  "
          f"severe={st['severe']}  mae_fail={st['mae_fail']}")

# Write JSON
out = {
    'classification': 'PENDING',
    'target_pairs': all_results,
    'aggregate_sample': agg_results,
}
json_path = '/home/matthew-villnave/sdi-substitutive/results/PHASE31BC_OUTPUT_RESIDUAL_PROBE.json'
with open(json_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nJSON written: {json_path}")
