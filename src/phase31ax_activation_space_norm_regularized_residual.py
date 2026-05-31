#!/usr/bin/env python3
"""
Phase 31AX — Activation-Space / Norm-Regularized Residual Probe
Repo: sdi-substitutive
Layer 21 / seed=9. Tests: norm matching, output interpolation, oracle projection.
"""

import ctypes, json, os, sys, time
import numpy as np
sys.path.insert(0, '/home/matthew-villnave/sdi-substitutive/src')
sys.path.insert(0, '/home/matthew-villnave/llama.cpp/gguf-py')
from gguf import GGUFReader
from gguf.quants import dequantize
from phase31x_manifest_runtime import encode_sdir, decode_sdir

LIB = '/home/matthew-villnave/llama.cpp/build/bin/libggml-base.so'
Q4_BUDGET_FAMILY = 2179072; Q4_BUDGET_LAYER = Q4_BUDGET_FAMILY * 3
QK_K = 256; Q2_BLOCK_BYTES = 84

lib = ctypes.CDLL(LIB)
lib.quantize_row_q2_K_ref.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int]
lib.quantize_row_q2_K_ref.restype = None
lib.dequantize_row_q2_K.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.dequantize_row_q2_K.restype = None

def q2_encode(flat):
    n = flat.size
    buf = np.zeros(n // QK_K * Q2_BLOCK_BYTES, dtype=np.uint8)
    lib.quantize_row_q2_K_ref(flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                               ctypes.cast(buf.ctypes.data, ctypes.c_void_p), n)
    return buf

def q2_decode(buf, n):
    out = np.zeros(n, dtype=np.float32)
    lib.dequantize_row_q2_K(ctypes.cast(buf.ctypes.data, ctypes.c_void_p),
                             out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n)
    return out

def silu(x): return x / (1.0 + np.exp(-np.clip(x, -709, 709)))
def cos_sim(a, b):
    a = a.flatten(); b = b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
def norm_vec(v): return float(np.linalg.norm(v.flatten()))

print("Loading model...")
reader = GGUFReader('/media/matthew-villnave/VL_usb/models/qwen2.5-0.5b-official/qwen2.5-0.5b-instruct-q4_k_m.gguf')

LI = 21
families = {}
for fam in ['ffn_up', 'ffn_gate', 'ffn_down']:
    t = next(x for x in reader.tensors if x.name == f'blk.{LI}.{fam}.weight')
    W_ref = dequantize(t.data, t.tensor_type).astype(np.float32)
    buf = q2_encode(W_ref.flatten())
    W_low = q2_decode(buf, W_ref.size).reshape(W_ref.shape)
    families[fam] = {'W_ref': W_ref, 'W_low': W_low, 'R': W_ref - W_low, 'base_bytes': buf.nbytes}

def mlp(X, Wg, Wu, Wd):
    return (silu(X @ Wg.T) * (X @ Wu.T)) @ Wd.T

def run_layer(X, families, k, alpha):
    Y_ref = mlp(X, families['ffn_gate']['W_ref'], families['ffn_up']['W_ref'], families['ffn_down']['W_ref'])
    Y_low = mlp(X, families['ffn_gate']['W_low'], families['ffn_up']['W_low'], families['ffn_down']['W_low'])
    if k == 0:
        Y_sub = Y_low
        res_bytes = 0
    else:
        dec = {fam: decode_sdir(encode_sdir(alpha * families[fam]['R'], k)) for fam in families}
        Y_sub = mlp(X,
                    families['ffn_gate']['W_low'] + dec['ffn_gate'],
                    families['ffn_up']['W_low']   + dec['ffn_up'],
                    families['ffn_down']['W_low']  + dec['ffn_down'])
        res_bytes = sum(len(encode_sdir(families[f]['R'], k)) for f in families)
    base_bytes = sum(families[f]['base_bytes'] for f in families)
    margin = Q4_BUDGET_LAYER - (base_bytes + res_bytes)
    return Y_ref, Y_low, Y_sub, margin

# Seed 9 activation
rng9 = np.random.default_rng(9)
X9 = rng9.standard_normal((1, 896)).astype(np.float32)
rng0 = np.random.default_rng(0)
X0 = rng0.standard_normal((1, 896)).astype(np.float32)
rng5 = np.random.default_rng(5)
X5 = rng5.standard_normal((1, 896)).astype(np.float32)

# ================================================================
# STEP 1: Baseline reproduction
# ================================================================
print("\n=== STEP 1: Baseline (layer 21, seed=9, k=1%, alpha=1.0) ===\n")
Y_ref9, Y_low9, Y_sub9, margin9 = run_layer(X9, families, k=1.0, alpha=1.0)
b_cos_low  = cos_sim(Y_ref9, Y_low9)
b_cos_sub  = cos_sim(Y_ref9, Y_sub9)
b_mae_low = float(np.abs(Y_ref9 - Y_low9).mean())
b_mae_sub = float(np.abs(Y_ref9 - Y_sub9).mean())
b_norm_ref = norm_vec(Y_ref9)
b_norm_low = norm_vec(Y_low9)
b_norm_sub = norm_vec(Y_sub9)
print(f"  margin={margin9:,}, memory_positive={margin9>0}")
print(f"  cos_low={b_cos_low:.6f}, cos_sub={b_cos_sub:.6f}, delta_cos={b_cos_sub-b_cos_low:+.6f}")
print(f"  MAE_low={b_mae_low:.6f}, MAE_sub={b_mae_sub:.6f}, MAE_delta={b_mae_sub-b_mae_low:+.6f}")
print(f"  norm_ref={b_norm_ref:.6f}, norm_low={b_norm_low:.6f}, norm_sub={b_norm_sub:.6f}")
print(f"  cosine_improved={b_cos_sub>b_cos_low}, MAE_improved={b_mae_sub<b_mae_low}")
assert abs(b_cos_sub - b_cos_low - (-0.146059)) < 0.01, f"Baseline mismatch: {b_cos_sub-b_cos_low}"
print("  Baseline confirmed.")

# ================================================================
# STEP 2: Norm matching
# ================================================================
print("\n=== STEP 2: Norm matching ===\n")

# Y_sub scaled to match norm of Y_ref
scale_to_ref = b_norm_ref / b_norm_sub
Y_norm_ref = Y_sub9 * scale_to_ref
cos_norm_ref = cos_sim(Y_ref9, Y_norm_ref)
mae_norm_ref = float(np.abs(Y_ref9 - Y_norm_ref).mean())

# Y_sub scaled to match norm of Y_low
scale_to_low = b_norm_low / b_norm_sub
Y_norm_low = Y_sub9 * scale_to_low
cos_norm_low = cos_sim(Y_ref9, Y_norm_low)
mae_norm_low = float(np.abs(Y_ref9 - Y_norm_low).mean())

print(f"  Scale to ref (||Y_sub|| * {scale_to_ref:.4f} = ||Y_ref||):")
print(f"    cos(Y_ref, Y_norm_ref)={cos_norm_ref:.6f}, delta_cos vs low={cos_norm_ref-b_cos_low:+.6f}")
print(f"    MAE(Y_ref, Y_norm_ref)={mae_norm_ref:.6f}, MAE_delta vs low={mae_norm_ref-b_mae_low:+.6f}")
print(f"    cos_improved={cos_norm_ref>b_cos_low}, MAE_improved={mae_norm_ref<b_mae_low}")

print(f"\n  Scale to low (||Y_sub|| * {scale_to_low:.4f} = ||Y_low||):")
print(f"    cos(Y_ref, Y_norm_low)={cos_norm_low:.6f}, delta_cos vs low={cos_norm_low-b_cos_low:+.6f}")
print(f"    MAE(Y_ref, Y_norm_low)={mae_norm_low:.6f}, MAE_delta vs low={mae_norm_low-b_mae_low:+.6f}")
print(f"    cos_improved={cos_norm_low>b_cos_low}, MAE_improved={mae_norm_low<b_mae_low}")

# Grid of scale factors
print("\n  Scale factor grid:")
print(f"  {'scale':>8} | {'target_norm':>12} | {'cos':>10} | {'delta_cos':>10} | {'MAE':>10} | {'MAE_delta':>10} | {'cos_pos':>8} | {'MAE_pos':>8}")
print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")
norm_targets = {'ref': b_norm_ref, 'low': b_norm_low}
best_norm_cos = None; best_norm_mae = None
for label, target_norm in norm_targets.items():
    for scale in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]:
        s = target_norm / b_norm_sub * scale
        Y_scaled = Y_sub9 * s
        c = cos_sim(Y_ref9, Y_scaled)
        m = float(np.abs(Y_ref9 - Y_scaled).mean())
        dc = c - b_cos_low
        dm = m - b_mae_low
        cp = c > b_cos_low
        mp = m < b_mae_low
        if best_norm_cos is None and cp and mp: best_norm_cos = (scale, target_norm, c, dc, m, dm)
        if best_norm_mae is None and cp and mp: best_norm_mae = (scale, target_norm, c, dc, m, dm)
        print(f"  {scale:>8.2f} | {target_norm:>12.4f} | {c:>10.6f} | {dc:>+10.6f} | {m:>10.6f} | {dm:>+10.6f} | {'YES' if cp else 'NO':>8} | {'YES' if mp else 'NO':>8}")

# ================================================================
# STEP 3: Output interpolation
# ================================================================
print("\n=== STEP 3: Output interpolation ===\n")
print(f"  {'beta':>6} | {'cos':>10} | {'delta_cos':>10} | {'MAE':>10} | {'MAE_delta':>10} | {'norm':>10} | {'cos_pos':>8} | {'MAE_pos':>8}")
print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")
best_interp = None
for beta in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
    Y_interp = Y_low9 + beta * (Y_sub9 - Y_low9)
    c = cos_sim(Y_ref9, Y_interp)
    m = float(np.abs(Y_ref9 - Y_interp).mean())
    n = norm_vec(Y_interp)
    dc = c - b_cos_low
    dm = m - b_mae_low
    cp = c > b_cos_low
    mp = m < b_mae_low
    if best_interp is None and cp and mp: best_interp = (beta, c, dc, m, dm)
    print(f"  {beta:>6.2f} | {c:>10.6f} | {dc:>+10.6f} | {m:>10.6f} | {dm:>+10.6f} | {n:>10.4f} | {'YES' if cp else 'NO':>8} | {'YES' if mp else 'NO':>8}")

# ================================================================
# STEP 4: Oracle projection (DIAGNOSTIC — uses Y_ref, not runtime available)
# ================================================================
print("\n=== STEP 4: Oracle projection (DIAGNOSTIC / ORACLE) ===\n")
D = Y_sub9 - Y_low9
T = Y_ref9 - Y_low9
T_flat = T.flatten()
T_norm = np.linalg.norm(T_flat)
D_flat = D.flatten()
# proj_T(D) = (D.T / ||T||) * T
D_proj = (np.dot(D_flat, T_flat) / (T_norm ** 2)) * T
Y_proj = Y_low9 + D_proj
cos_proj = cos_sim(Y_ref9, Y_proj)
mae_proj = float(np.abs(Y_ref9 - Y_proj).mean())
norm_proj = norm_vec(Y_proj)
dc_proj = cos_proj - b_cos_low
dm_proj = mae_proj - b_mae_low
print(f"  D = Y_sub - Y_low")
print(f"  T = Y_ref - Y_low  (reference correction direction)")
print(f"  D_proj = proj_T(D)")
print(f"  Y_proj = Y_low + D_proj")
print(f"    cos(Y_ref, Y_proj)={cos_proj:.6f}, delta_cos={dc_proj:+.6f}")
print(f"    MAE(Y_ref, Y_proj)={mae_proj:.6f}, MAE_delta={dm_proj:+.6f}")
print(f"    norm(Y_proj)={norm_proj:.4f}")
print(f"    cos_improved={cos_proj>b_cos_low}, MAE_improved={mae_proj<b_mae_low}")

# Direction-scaled: Y_dir = Y_low + gamma * D_proj
print(f"\n  Direction-scaled variants Y_dir = Y_low + gamma * D_proj:")
print(f"  {'gamma':>6} | {'cos':>10} | {'delta_cos':>10} | {'MAE':>10} | {'MAE_delta':>10} | {'norm':>10} | {'cos_pos':>8} | {'MAE_pos':>8}")
print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")
best_oracle = None
for gamma in [0.5, 1.0, 1.5]:
    Y_dir = Y_low9 + gamma * D_proj
    c = cos_sim(Y_ref9, Y_dir)
    m = float(np.abs(Y_ref9 - Y_dir).mean())
    n = norm_vec(Y_dir)
    dc = c - b_cos_low
    dm = m - b_mae_low
    cp = c > b_cos_low
    mp = m < b_mae_low
    if best_oracle is None and cp and mp: best_oracle = (gamma, c, dc, m, dm)
    print(f"  {gamma:>6.2f} | {c:>10.6f} | {dc:>+10.6f} | {m:>10.6f} | {dm:>+10.6f} | {n:>10.4f} | {'YES' if cp else 'NO':>8} | {'YES' if mp else 'NO':>8}")

# ================================================================
# STEP 5: Transfer to safe cases
# ================================================================
print("\n=== STEP 5: Transfer to safe cases ===\n")

# Best norm-matching candidate: scale Y_sub to ref norm
best_scale = scale_to_ref  # from Step 2
Y_candidate = Y_sub9 * best_scale

def eval_candidate(Y_ref_x, Y_low_x, Y_candidate_x, label):
    c = cos_sim(Y_ref_x, Y_candidate_x)
    c_low = cos_sim(Y_ref_x, Y_low_x)
    m = float(np.abs(Y_ref_x - Y_candidate_x).mean())
    m_low = float(np.abs(Y_ref_x - Y_low_x).mean())
    dc = c - c_low
    dm = m - m_low
    cp = c > c_low
    mp = m < m_low
    print(f"  {label}: cos={c:.6f} (low={c_low:.6f}, d={dc:+.6f}), MAE={m:.6f} (low={m_low:.6f}, d={dm:+.6f}), cos_pos={'Y' if cp else 'N'}, MAE_pos={'Y' if mp else 'N'}")
    return {'cos': round(c,6), 'cos_low': round(c_low,6), 'delta_cos': round(dc,6),
            'MAE': round(m,6), 'MAE_low': round(m_low,6), 'MAE_delta': round(dm,6),
            'cosine_improved': cp, 'MAE_improved': mp}

print("  Transfer of norm-to-ref candidate (scale Y_sub to ||Y_ref||):")
safe_results = []
for seed, X in [(0, X0), (5, X5)]:
    Y_ref_s, Y_low_s, Y_sub_s, _ = run_layer(X, families, k=1.0, alpha=1.0)
    Y_cand_s = Y_sub_s * best_scale
    r = eval_candidate(Y_ref_s, Y_low_s, Y_cand_s, f"layer21 seed={seed}")
    safe_results.append({'seed': seed, **r})

# Also test what the best oracle (gamma=1.0) does to safe seeds
print("\n  Transfer of oracle gamma=1.0 candidate:")
oracle_gamma = 1.0
D_safe = {seed: None for seed in [0, 5]}
Y_oracle_safe = {}
for seed, X in [(0, X0), (5, X5)]:
    Y_ref_s, Y_low_s, Y_sub_s, _ = run_layer(X, families, k=1.0, alpha=1.0)
    D_s = Y_sub_s - Y_low_s
    T_s = Y_ref_s - Y_low_s
    T_flat_s = T_s.flatten()
    T_norm_s = np.linalg.norm(T_flat_s)
    D_flat_s = D_s.flatten()
    D_proj_s = (np.dot(D_flat_s, T_flat_s) / (T_norm_s ** 2)) * T_s
    Y_oracle_s = Y_low_s + oracle_gamma * D_proj_s
    r = eval_candidate(Y_ref_s, Y_low_s, Y_oracle_s, f"layer21 seed={seed} oracle")
    safe_results.append({'seed': seed, 'oracle': True, **r})
    D_safe[seed] = D_s; Y_oracle_safe[seed] = Y_oracle_s

# ================================================================
# STEP 6: Classification
# ================================================================
print("\n=== STEP 6: Classification ===\n")

norm_fix_cos   = best_norm_cos   is not None
norm_fix_mae  = best_norm_mae   is not None
interp_fix    = best_interp     is not None
oracle_fix    = best_oracle     is not None

# Check if any fix breaks safe cases
fix_breaks_safe = False
for r in safe_results:
    if not r.get('oracle', False):
        if not (r['cosine_improved'] and r['MAE_improved']):
            fix_breaks_safe = True

if oracle_fix and not fix_breaks_safe:
    classification = "PARTIAL_ORACLE_DIRECTION_FIX_FOUND"
elif oracle_fix and fix_breaks_safe:
    classification = "PARTIAL_FIX_BREAKS_SAFE_CASES"
elif best_interp is not None and not fix_breaks_safe:
    classification = "PARTIAL_OUTPUT_INTERPOLATION_FIX_FOUND"
elif best_norm_cos is not None and not fix_breaks_safe:
    classification = "PASS_NORM_REG_FIX_FOUND"
elif norm_fix_cos or norm_fix_mae or interp_fix:
    classification = "PARTIAL_NO_FIX_METRIC_CONFLICT_CONFIRMED"
else:
    classification = "PARTIAL_NO_FIX_METRIC_CONFLICT_CONFIRMED"

print(f"  norm_fix_cos={norm_fix_cos}, norm_fix_mae={norm_fix_mae}")
print(f"  interp_fix (beta)={best_interp}")
print(f"  oracle_fix (gamma)={best_oracle}")
print(f"  fix_breaks_safe={fix_breaks_safe}")
print(f"  Classification: {classification}")

# ================================================================
# WRITE JSON
# ================================================================
result_out = {
    'phase': '31AX',
    'classification': classification,
    'baseline': {
        'layer': 21, 'seed': 9, 'k': 1.0, 'alpha': 1.0,
        'margin': margin9, 'memory_positive': margin9 > 0,
        'cos_low': round(b_cos_low, 6), 'cos_sub': round(b_cos_sub, 6),
        'delta_cos': round(b_cos_sub - b_cos_low, 6),
        'MAE_low': round(b_mae_low, 6), 'MAE_sub': round(b_mae_sub, 6),
        'MAE_delta': round(b_mae_sub - b_mae_low, 6),
        'norm_ref': round(b_norm_ref, 6), 'norm_low': round(b_norm_low, 6), 'norm_sub': round(b_norm_sub, 6),
    },
    'norm_matching': {
        'scale_to_ref': round(scale_to_ref, 6),
        'scale_to_low': round(scale_to_low, 6),
        'cos_norm_ref': round(cos_norm_ref, 6), 'MAE_norm_ref': round(mae_norm_ref, 6),
        'cos_norm_low': round(cos_norm_low, 6), 'MAE_norm_low': round(mae_norm_low, 6),
        'best_cos_positive_with_MAE_improvement': dict(best_norm_cos) if best_norm_cos else None,
        'best_MAE_positive_with_cos_improvement': dict(best_norm_mae) if best_norm_mae else None,
    },
    'output_interpolation': {
        'best_beta': best_interp[0] if best_interp else None,
        'best': {k: round(v, 6) if isinstance(v, float) else v for k, v in zip(['beta','cos','delta_cos','MAE','MAE_delta'], best_interp)} if best_interp else None,
    },
    'oracle_projection': {
        'note': 'DIAGNOSTIC — uses Y_ref, not runtime available',
        'D_proj_magnitude': round(float(np.linalg.norm(D_proj.flatten())), 6),
        'cos_Y_proj': round(cos_proj, 6), 'MAE_Y_proj': round(mae_proj, 6),
        'delta_cos': round(dc_proj, 6), 'MAE_delta': round(dm_proj, 6),
        'best_gamma': best_oracle[0] if best_oracle else None,
        'best': {k: round(v, 6) if isinstance(v, float) else v for k, v in zip(['gamma','cos','delta_cos','MAE','MAE_delta'], best_oracle)} if best_oracle else None,
    },
    'safe_case_transfer': safe_results,
    'mae_convention': {
        'MAE_delta_formula': 'MAE_sub - MAE_low; negative = MAE improved',
        'MAE_improvement': 'abs(MAE_delta); positive = MAE improved',
    },
}

json_path = '/home/matthew-villnave/sdi-substitutive/results/PHASE31AX_ACTIVATION_SPACE_NORM_REGULARIZED_RESIDUAL.json'
with open(json_path, 'w') as f:
    json.dump(result_out, f, indent=2)
print(f"\nJSON: {json_path} ({os.path.getsize(json_path)/1024:.1f} KB)")
print(f"\nDONE. classification={classification}")