#!/usr/bin/env python3
"""
Phase 31AT — Layer 21 Cosine Regression Diagnosis
Repo: sdi-substitutive
"""

import ctypes, json, os, sys
import numpy as np

sys.path.insert(0, '/home/matthew-villnave/sdi-substitutive/src')
sys.path.insert(0, '/home/matthew-villnave/llama.cpp/gguf-py')

from gguf import GGUFReader
from gguf.quants import dequantize
from phase31x_manifest_runtime import encode_sdir, decode_sdir

REPO = '/home/matthew-villnave/sdi-substitutive'
USB_BASE = '/media/matthew-villnave/VL_usb/models/qwen2.5-0.5b-official'
Q4KM_PATH = f'{USB_BASE}/qwen2.5-0.5b-instruct-q4_k_m.gguf'
LIB = '/home/matthew-villnave/llama.cpp/build/bin/libggml-base.so'

QK_K = 256; Q2_BLOCK_BYTES = 84
Q4_BUDGET_FAMILY = 2179072; Q4_BUDGET_LAYER = Q4_BUDGET_FAMILY * 3

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

def norm(a):
    return float(np.linalg.norm(a.flatten()))

def to_native(obj):
    if isinstance(obj, dict): return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [to_native(x) for x in obj]
    elif isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.ndarray): return None
    else: return obj

print("=== Phase 31AT: Layer 21 Cosine Regression Diagnosis ===\n")

reader = GGUFReader(Q4KM_PATH)
names = [t.name for t in reader.tensors]

def load_layer(li):
    families = {}
    for fam in ['ffn_up', 'ffn_gate', 'ffn_down']:
        t = next(x for x in reader.tensors if x.name == f'blk.{li}.{fam}.weight')
        W_ref = dequantize(t.data, t.tensor_type).astype(np.float32)
        W_flat = W_ref.flatten()
        buf = q2_encode(W_flat)
        W_low = q2_decode(buf, W_flat.size).reshape(W_ref.shape)
        families[fam] = {'W_ref': W_ref, 'W_low': W_low}
    return families

def mlp_output(X, families):
    return (silu(X @ families['ffn_gate']['W_ref'].T) * (X @ families['ffn_up']['W_ref'].T)) @ families['ffn_down']['W_ref'].T

def mlp_output_low(X, families):
    return (silu(X @ families['ffn_gate']['W_low'].T) * (X @ families['ffn_up']['W_low'].T)) @ families['ffn_down']['W_low'].T

def mlp_output_sub(X, families, k):
    dec = {}
    for fam in ['ffn_up', 'ffn_gate', 'ffn_down']:
        R = families[fam]['W_ref'] - families[fam]['W_low']
        enc = encode_sdir(R, k)
        dec[fam] = decode_sdir(enc)
    W_gate_sub = families['ffn_gate']['W_low'] + dec['ffn_gate']
    W_up_sub   = families['ffn_up']['W_low']   + dec['ffn_up']
    W_down_sub = families['ffn_down']['W_low']  + dec['ffn_down']
    return (silu(X @ W_gate_sub.T) * (X @ W_up_sub.T)) @ W_down_sub.T

def run_trial(X, families, k):
    Y_ref = mlp_output(X, families)
    Y_low = mlp_output_low(X, families)
    Y_sub = mlp_output_sub(X, families, k) if k > 0 else Y_low

    base_bytes = sum(q2_encode(families[f]['W_ref'].flatten()).shape[0] * Q2_BLOCK_BYTES // (families[f]['W_ref'].size // QK_K)
                     for f in ['ffn_up', 'ffn_gate', 'ffn_down'])
    base_bytes = sum(
        q2_encode(families[f]['W_ref'].flatten()).nbytes for f in ['ffn_up', 'ffn_gate', 'ffn_down']
    )
    res_bytes = 0
    if k > 0:
        res_bytes = sum(len(encode_sdir(families[f]['W_ref'] - families[f]['W_low'], k)) for f in ['ffn_up', 'ffn_gate', 'ffn_down'])
    total_bytes = base_bytes + res_bytes
    margin = Q4_BUDGET_LAYER - total_bytes

    return {
        'k_pct': k,
        'margin': int(margin),
        'memory_positive': bool(margin > 0),
        'norm_ref': round(norm(Y_ref), 8),
        'norm_low': round(norm(Y_low), 8),
        'norm_sub': round(norm(Y_sub), 8) if k > 0 else None,
        'cos_low': round(cos_sim(Y_ref, Y_low), 8),
        'cos_sub': round(cos_sim(Y_ref, Y_sub), 8) if k > 0 else round(cos_sim(Y_ref, Y_low), 8),
        'delta_cos': round(cos_sim(Y_ref, Y_sub) - cos_sim(Y_ref, Y_low), 8) if k > 0 else 0.0,
        'MAE_low': round(float(np.abs(Y_ref - Y_low).mean()), 8),
        'MAE_sub': round(float(np.abs(Y_ref - Y_sub).mean()), 8) if k > 0 else round(float(np.abs(Y_ref - Y_low).mean()), 8),
        'MAE_delta': round(float(np.abs(Y_ref - Y_sub).mean() - np.abs(Y_ref - Y_low).mean()), 8) if k > 0 else 0.0,
        'MAE_improvement': round(float(abs(np.abs(Y_ref - Y_sub).mean() - np.abs(Y_ref - Y_low).mean())), 8) if k > 0 else 0.0,
    }

# === TASK 1: Reproduce layer 21 for all k ===
print("=== TASK 1: Layer 21 Reproduction ===")
families21 = load_layer(21)
np.random.seed(42)
X = np.random.randn(1, families21['ffn_up']['W_ref'].shape[1]).astype(np.float32)

print("Layer 21 policies k=0/0.5/1/2/3:")
print(f"{'k':>4} | {'margin':>12} | {'cos_low':>10} | {'cos_sub':>10} | {'delta_cos':>10} | {'MAE_low':>10} | {'MAE_sub':>10} | {'MAE_delta':>10} | {'mem_pos':>8}")
print("-" * 105)

layer21_results = {}
for k in [0, 0.5, 1, 2, 3]:
    r = run_trial(X, families21, k)
    layer21_results[k] = r
    print(f"{k:4.1f} | {r['margin']:+,} | {r['cos_low']:.6f} | {r['cos_sub']:.6f} | {r['delta_cos']:+.5f} | {r['MAE_low']:.6f} | {r['MAE_sub']:.6f} | {r['MAE_delta']:+.6f} | {'YES' if r['memory_positive'] else 'NO':>8}")

# === TASK 2: Family Ablation on layer 21 at k=1 and k=2 ===
print("\n=== TASK 2: Family Ablation ===")

def family_ablated(X, families, which_fams, k):
    dec = {}
    for fam in ['ffn_up', 'ffn_gate', 'ffn_down']:
        if fam in which_fams:
            R = families[fam]['W_ref'] - families[fam]['W_low']
            dec[fam] = decode_sdir(encode_sdir(R, k))
        else:
            dec[fam] = np.zeros_like(families[fam]['W_ref'])

    Y_ref = (silu(X @ families['ffn_gate']['W_ref'].T) * (X @ families['ffn_up']['W_ref'].T)) @ families['ffn_down']['W_ref'].T
    Y_low = (silu(X @ families['ffn_gate']['W_low'].T)  * (X @ families['ffn_up']['W_low'].T))  @ families['ffn_down']['W_low'].T
    W_gate_sub = families['ffn_gate']['W_low'] + dec['ffn_gate']
    W_up_sub = families['ffn_up']['W_low'] + dec['ffn_up']
    W_down_sub = families['ffn_down']['W_low'] + dec['ffn_down']
    Y_sub = (silu(X @ W_gate_sub.T) * (X @ W_up_sub.T)) @ W_down_sub.T

    base_bytes = sum(q2_encode(families[f]['W_ref'].flatten()).nbytes for f in ['ffn_up', 'ffn_gate', 'ffn_down'])
    res_bytes = sum(len(encode_sdir(families[f]['W_ref'] - families[f]['W_low'], k)) for f in which_fams)
    total_bytes = base_bytes + res_bytes
    margin = Q4_BUDGET_LAYER - total_bytes

    return {
        'ablation': '+'.join(which_fams),
        'margin': int(margin),
        'memory_positive': bool(margin > 0),
        'cos_low': round(cos_sim(Y_ref, Y_low), 8),
        'cos_sub': round(cos_sim(Y_ref, Y_sub), 8),
        'delta_cos': round(cos_sim(Y_ref, Y_sub) - cos_sim(Y_ref, Y_low), 8),
        'MAE_low': round(float(np.abs(Y_ref - Y_low).mean()), 8),
        'MAE_sub': round(float(np.abs(Y_ref - Y_sub).mean()), 8),
        'MAE_delta': round(float(np.abs(Y_ref - Y_sub).mean() - np.abs(Y_ref - Y_low).mean()), 8),
        'MAE_improvement': round(float(abs(np.abs(Y_ref - Y_sub).mean() - np.abs(Y_ref - Y_low).mean())), 8),
        'norm_ref': round(norm(Y_ref), 8),
        'norm_low': round(norm(Y_low), 8),
        'norm_sub': round(norm(Y_sub), 8),
    }

ablations = [
    ['ffn_up'],
    ['ffn_gate'],
    ['ffn_down'],
    ['ffn_up', 'ffn_gate'],
    ['ffn_up', 'ffn_down'],
    ['ffn_gate', 'ffn_down'],
    ['ffn_up', 'ffn_gate', 'ffn_down'],
]

print("\nFamily ablation at k=1%:")
print(f"{'ablation':>25} | {'margin':>10} | {'cos_low':>8} | {'cos_sub':>8} | {'delta_cos':>9} | {'MAE_low':>8} | {'MAE_sub':>8} | {'MAE_delta':>9}")
print("-" * 105)
ablation_k1 = {}
for which in ablations:
    r = family_ablated(X, families21, which, 1.0)
    ablation_k1['+'.join(which)] = r
    flag = " ← best" if (r['delta_cos'] > r['MAE_delta'] and r['memory_positive']) else ""
    print(f"{'+'.join(which):>25} | {r['margin']:+,} | {r['cos_low']:.6f} | {r['cos_sub']:.6f} | {r['delta_cos']:+.5f} | {r['MAE_low']:.6f} | {r['MAE_sub']:.6f} | {r['MAE_delta']:+.6f}{flag}")

print("\nFamily ablation at k=2%:")
print(f"{'ablation':>25} | {'margin':>10} | {'cos_low':>8} | {'cos_sub':>8} | {'delta_cos':>9} | {'MAE_low':>8} | {'MAE_sub':>8} | {'MAE_delta':>9}")
print("-" * 105)
ablation_k2 = {}
for which in ablations:
    r = family_ablated(X, families21, which, 2.0)
    ablation_k2['+'.join(which)] = r
    print(f"{'+'.join(which):>25} | {r['margin']:+,} | {r['cos_low']:.6f} | {r['cos_sub']:.6f} | {r['delta_cos']:+.5f} | {r['MAE_low']:.6f} | {r['MAE_sub']:.6f} | {r['MAE_delta']:+.6f}")

# === TASK 3: Alpha Sweep on layer 21 ===
print("\n=== TASK 3: Alpha Sweep (k=1%) ===")
print(f"{'alpha':>6} | {'delta_cos':>10} | {'MAE_delta':>10} | {'MAE_impr':>10} | {'norm_ref':>10} | {'norm_sub':>10} | {'margin':>12}")
print("-" * 85)
alpha_k1 = {}
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
    dec_up   = alpha * decode_sdir(encode_sdir(families21['ffn_up']['W_ref']   - families21['ffn_up']['W_low'],   1.0))
    dec_gate = alpha * decode_sdir(encode_sdir(families21['ffn_gate']['W_ref'] - families21['ffn_gate']['W_low'], 1.0))
    dec_down = alpha * decode_sdir(encode_sdir(families21['ffn_down']['W_ref'] - families21['ffn_down']['W_low'], 1.0))

    Y_ref = (silu(X @ families21['ffn_gate']['W_ref'].T) * (X @ families21['ffn_up']['W_ref'].T)) @ families21['ffn_down']['W_ref'].T
    Y_low = (silu(X @ families21['ffn_gate']['W_low'].T)  * (X @ families21['ffn_up']['W_low'].T))  @ families21['ffn_down']['W_low'].T
    W_gate_sub = families21['ffn_gate']['W_low'] + dec_gate
    W_up_sub = families21['ffn_up']['W_low'] + dec_up
    W_down_sub = families21['ffn_down']['W_low'] + dec_down
    Y_sub = (silu(X @ W_gate_sub.T) * (X @ W_up_sub.T)) @ W_down_sub.T

    base_bytes = sum(q2_encode(families21[f]['W_ref'].flatten()).nbytes for f in ['ffn_up', 'ffn_gate', 'ffn_down'])
    res_bytes = sum(len(encode_sdir(families21[f]['W_ref'] - families21[f]['W_low'], 1.0)) for f in ['ffn_up', 'ffn_gate', 'ffn_down'])
    total_bytes = base_bytes + res_bytes
    margin = Q4_BUDGET_LAYER - total_bytes

    mae_delta = float(np.abs(Y_ref - Y_sub).mean() - np.abs(Y_ref - Y_low).mean())

    r = {
        'alpha': alpha, 'margin': int(margin),
        'delta_cos': round(cos_sim(Y_ref, Y_sub) - cos_sim(Y_ref, Y_low), 8),
        'MAE_delta': round(mae_delta, 8),
        'MAE_improvement': round(float(abs(mae_delta)), 8),
        'norm_ref': round(norm(Y_ref), 8),
        'norm_low': round(norm(Y_low), 8),
        'norm_sub': round(norm(Y_sub), 8),
    }
    alpha_k1[alpha] = r
    print(f"{alpha:6.2f} | {r['delta_cos']:+.5f} | {r['MAE_delta']:+.6f} | {r['MAE_improvement']:+.6f} | {r['norm_ref']:10.4f} | {r['norm_sub']:10.4f} | {r['margin']:+,}")

# === TASK 4: Activation Sensitivity ===
print("\n=== TASK 4: Activation Sensitivity (10 seeds) ===")
activation_sensitivity = {}
for target_layer in [20, 21, 22]:
    families_layer = load_layer(target_layer)
    seed_results = []
    for seed in range(10):
        np.random.seed(seed)
        X_s = np.random.randn(1, families_layer['ffn_up']['W_ref'].shape[1]).astype(np.float32)
        r_k1 = run_trial(X_s, families_layer, 1.0)
        seed_results.append({
            'seed': seed,
            'delta_cos': r_k1['delta_cos'],
            'MAE_improvement': r_k1['MAE_improvement'],
            'margin': r_k1['margin'],
            'memory_positive': r_k1['memory_positive'],
        })

    n_cos_pos = sum(1 for x in seed_results if x['delta_cos'] > 0)
    n_mae_imp = sum(1 for x in seed_results if x['MAE_improvement'] > 0)
    dcs = [x['delta_cos'] for x in seed_results]
    mis = [x['MAE_improvement'] for x in seed_results]

    activation_sensitivity[f'layer{target_layer}'] = {
        'n_cos_positive': n_cos_pos,
        'n_MAE_improving': n_mae_imp,
        'mean_delta_cos': round(sum(dcs)/10, 8),
        'min_delta_cos': round(min(dcs), 8),
        'max_delta_cos': round(max(dcs), 8),
        'mean_MAE_improvement': round(sum(mis)/10, 8),
        'seed_results': seed_results,
    }
    print(f"Layer {target_layer}: n_cos_pos={n_cos_pos}/10, n_mae_imp={n_mae_imp}/10, "
          f"mean_delta_cos={sum(dcs)/10:+.5f}, min={min(dcs):+.5f}, max={max(dcs):+.5f}")

# === TASK 5: Compare Error Geometry with layers 20 and 22 ===
print("\n=== TASK 5: Error Geometry Comparison ===")
for target_layer in [20, 21, 22]:
    families_layer = load_layer(target_layer)
    np.random.seed(42)
    X_s = np.random.randn(1, families_layer['ffn_up']['W_ref'].shape[1]).astype(np.float32)
    Y_ref = mlp_output(X_s, families_layer)
    Y_low = mlp_output_low(X_s, families_layer)
    Y_sub = mlp_output_sub(X_s, families_layer, 1.0)

    print(f"\nLayer {target_layer} at k=1%:")
    print(f"  norm_ref={norm(Y_ref):.4f}, norm_low={norm(Y_low):.4f}, norm_sub={norm(Y_sub):.4f}")
    print(f"  cos(Y_ref, Y_low)={cos_sim(Y_ref, Y_low):.6f}, cos(Y_ref, Y_sub)={cos_sim(Y_ref, Y_sub):.6f}")
    print(f"  delta_cos={cos_sim(Y_ref, Y_sub)-cos_sim(Y_ref, Y_low):+.5f}")
    print(f"  MAE_low={np.abs(Y_ref-Y_low).mean():.6f}, MAE_sub={np.abs(Y_ref-Y_sub).mean():.6f}")
    print(f"  MAE_improvement={abs(np.abs(Y_ref-Y_sub).mean()-np.abs(Y_ref-Y_low).mean()):.6f}")

    # Direction change analysis
    err_low = (Y_ref - Y_low).flatten()
    err_sub = (Y_ref - Y_sub).flatten()
    cos_err = cos_sim(err_low, err_sub) if np.linalg.norm(err_low) > 1e-12 and np.linalg.norm(err_sub) > 1e-12 else 0.0
    print(f"  cos(err_low, err_sub)={cos_err:.6f} (angle between error vectors)")

# === Classification ===
# Check if any alpha improves both cosine and MAE
any_cosine_fix = any(r['delta_cos'] > 0 and r['MAE_improvement'] > 0 for r in alpha_k1.values())
any_family_ablated_cosine_fix = any(r['delta_cos'] > 0 and r['MAE_improvement'] > 0 for r in ablation_k1.values())

# Check activation sensitivity
layer21_sensitivity = activation_sensitivity['layer21']
layer21_cos_consistent = layer21_sensitivity['n_cos_positive'] == 0  # 0 seeds improve cosine

best_alpha_cos = max(alpha_k1.values(), key=lambda x: x['delta_cos'])
best_alpha_mae = max(alpha_k1.values(), key=lambda x: x['MAE_improvement'])
best_ablation = max(ablation_k1.values(), key=lambda x: x['delta_cos'])

print(f"\n=== DIAGNOSIS SUMMARY ===")
print(f"any_alpha_cosine_positive: {any(r['delta_cos']>0 for r in alpha_k1.values())}")
print(f"any_family_ablated_cosine_positive: {any(r['delta_cos']>0 for r in ablation_k1.values())}")
print(f"layer21_n_cos_positive_at_k1: {layer21_sensitivity['n_cos_positive']}/10")
print(f"best_alpha: {best_alpha_cos['alpha']} -> delta_cos={best_alpha_cos['delta_cos']:+.5f}, MAE_impr={best_alpha_cos['MAE_improvement']:+.6f}")
print(f"best_ablation: {best_ablation['ablation']} -> delta_cos={best_ablation['delta_cos']:+.5f}, MAE_impr={best_ablation['MAE_improvement']:+.6f}")

# Classify
if any_cosine_fix:
    classification = "PASS_LAYER21_POLICY_FOUND"
elif layer21_cos_consistent:
    classification = "PARTIAL_LAYER21_ACTIVATION_SENSITIVE"
    if best_ablation['delta_cos'] > 0:
        classification = "PARTIAL_LAYER21_FAMILY_SPECIFIC"
else:
    classification = "PARTIAL_LAYER21_MAE_ONLY"

# If cosine regression is metric artifact (MAE improves consistently but cosine worsens)
# and layer21 cosine never positive across any alpha, it's metric conflict
if not any_cosine_fix and not any(r['delta_cos'] > 0 for r in ablation_k1.values()) and layer21_cos_consistent:
    classification = "PARTIAL_LAYER21_METRIC_CONFLICT"

print(f"\nClassification: {classification}")

# === Write results ===
result = {
    'phase': '31AT',
    'classification': classification,
    'layer21_reproduction': {str(k): to_native(v) for k, v in layer21_results.items()},
    'family_ablation_k1': {k: to_native(v) for k, v in ablation_k1.items()},
    'family_ablation_k2': {k: to_native(v) for k, v in ablation_k2.items()},
    'alpha_sweep_k1': {str(a): to_native(v) for a, v in alpha_k1.items()},
    'activation_sensitivity': to_native(activation_sensitivity),
    'mae_convention': {
        'MAE_delta_formula': 'MAE_sub - MAE_low',
        'MAE_delta_positive_means': 'MAE worsened',
        'MAE_delta_negative_means': 'MAE improved',
        'MAE_improvement': 'abs(MAE_delta); positive means improvement',
    },
}

json_path = f'{REPO}/results/PHASE31AT_LAYER21_COSINE_REGRESSION_DIAGNOSIS.json'
with open(json_path, 'w') as f:
    json.dump(result, f, indent=2)
size = os.path.getsize(json_path)
print(f"\nJSON: {json_path} — {size/1024:.1f} KB")

# MD summary
md = [
    "# Phase 31AT — Layer 21 Cosine Regression Diagnosis\n",
    f"## Classification: **`{classification}`**\n",
    "\n## Layer 21 Reproduction (seed=42)\n",
    f"| k | margin | cos_low | cos_sub | delta_cos | MAE_low | MAE_sub | MAE_delta | mem_pos |\n",
    "|---|--------|---------|---------|-----------|---------|---------|-----------|--------|\n",
]
for k, r in layer21_results.items():
    md.append(f"| {k} | {r['margin']:+,} | {r['cos_low']:.6f} | {r['cos_sub']:.6f} | {r['delta_cos']:+.5f} | {r['MAE_low']:.6f} | {r['MAE_sub']:.6f} | {r['MAE_delta']:+.6f} | {r['memory_positive']} |")

md += ["\n## Family Ablation k=1%\n",
       f"| ablation | margin | cos_low | cos_sub | delta_cos | MAE_delta | MAE_impr |\n",
       "|----------|--------|---------|---------|-----------|-----------|----------|\n"]
for name, r in ablation_k1.items():
    flag = " ← best cosine" if r['delta_cos'] == max(x['delta_cos'] for x in ablation_k1.values()) else ""
    md.append(f"| {name} | {r['margin']:+,} | {r['cos_low']:.6f} | {r['cos_sub']:.6f} | {r['delta_cos']:+.5f} | {r['MAE_delta']:+.6f} | {r['MAE_improvement']:+.6f} |{flag}")

md += ["\n## Alpha Sweep k=1%\n",
       f"| alpha | delta_cos | MAE_delta | norm_ref | norm_sub |\n",
       "|-------|-----------|-----------|----------|----------|\n"]
for a, r in sorted(alpha_k1.items()):
    md.append(f"| {a} | {r['delta_cos']:+.5f} | {r['MAE_delta']:+.6f} | {r['norm_ref']:.4f} | {r['norm_sub']:.4f} |")

md += ["\n## Activation Sensitivity (k=1%)\n",
       f"| layer | n_cos_pos/10 | n_mae_imp/10 | mean_delta_cos | min | max |\n",
       "|-------|--------------|--------------|----------------|-----|-----|\n"]
for li in [20, 21, 22]:
    s = activation_sensitivity[f'layer{li}']
    md.append(f"| {li} | {s['n_cos_positive']} | {s['n_MAE_improving']} | {s['mean_delta_cos']:+.5f} | {s['min_delta_cos']:+.5f} | {s['max_delta_cos']:+.5f} |")

md_path = f'{REPO}/docs/PHASE31AT_LAYER21_COSINE_REGRESSION_DIAGNOSIS.md'
with open(md_path, 'w') as f:
    f.write("\n".join(md))

print(f"MD: {md_path}")
print(f"\nDONE. classification={classification}")