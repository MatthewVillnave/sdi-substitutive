#!/usr/bin/env python3
"""
Phase 31AR — Layers 0-5 Full MLP Q2_K + Low-k Residual Sweep
Repo: sdi-substitutive, HEAD: 1b2e971936ef4fbedb91cc375a20479add9008bf

Goal: Test whether the layer0 result from 31AQ generalizes to layers 0-5.
Core question: Do layers 0-5 full MLPs remain memory-positive and
approximation-improving using official Q2_K W_low plus low-k residual?

Official llama.cpp Q2_K quantize_row_q2_K_ref + dequantize_row_q2_K via ctypes.
Residual encoding uses source-of-truth encode_sdir (bitmap + fp16 values).

Classification targets:
PASS_LAYERS0_5_MLP_Q2K_LOWK_POLICY_FOUND
PARTIAL_LAYER_VARIANCE
BLOCKED_Q2K_QUANTIZER
"""

import ctypes, json, math, os, struct, sys
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{REPO}/src")
sys.path.insert(0, "/home/matthew-villnave/llama.cpp/gguf-py")

from gguf import GGUFReader
from gguf.quants import dequantize
from phase31x_manifest_runtime import encode_sdir, decode_sdir

USB_BASE = "/media/matthew-villnave/VL_usb/models/qwen2.5-0.5b-official"
Q4KM_PATH = f"{USB_BASE}/qwen2.5-0.5b-instruct-q4_k_m.gguf"
LIB_PATH = "/home/matthew-villnave/llama.cpp/build/bin/libggml-base.so"

QK_K = 256
Q2_BLOCK_BYTES = 84
Q4_BUDGET_FAMILY = 2179072
Q4_BUDGET_LAYER = Q4_BUDGET_FAMILY * 3  # 6,537,216

# --- llama.cpp ctypes wrappers ---
lib = ctypes.CDLL(LIB_PATH)
lib.quantize_row_q2_K_ref.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int
]
lib.quantize_row_q2_K_ref.restype = None

lib.dequantize_row_q2_K.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int
]
lib.dequantize_row_q2_K.restype = None

def q2_encode(W_flat):
    n = W_flat.size
    buf = np.zeros(n // QK_K * Q2_BLOCK_BYTES, dtype=np.uint8)
    lib.quantize_row_q2_K_ref(
        W_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(buf.ctypes.data, ctypes.c_void_p), n)
    return buf

def q2_decode(buf, n):
    out = np.zeros(n, dtype=np.float32)
    lib.dequantize_row_q2_K(
        ctypes.cast(buf.ctypes.data, ctypes.c_void_p),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n)
    return out

def silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -709, 709)))

def cos_sim(a, b):
    a = a.flatten(); b = b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def load_layer_family(layer_idx, family, reader4):
    """Load and Q2_K encode a single family tensor."""
    name = f'blk.{layer_idx}.{family}.weight'
    t = next(t for t in reader4.tensors if t.name == name)
    W_ref = dequantize(t.data, t.tensor_type).astype(np.float32)
    W_flat = W_ref.flatten()
    buf = q2_encode(W_flat)
    W_low = q2_decode(buf, W_flat.size)
    n = W_flat.size
    n_blocks = n // QK_K
    expected_bytes = n_blocks * Q2_BLOCK_BYTES
    return {
        'W_ref': W_ref,
        'W_low': W_low.reshape(W_ref.shape),
        'buf': buf,
        'shape': list(W_ref.shape),
        'n_elements': n,
        'n_blocks': n_blocks,
        'expected_bytes': expected_bytes,
        'actual_bytes': len(buf),
        'byte_match': len(buf) == expected_bytes,
        'bpe': len(buf) * 8 / n,
        'Q4_budget': Q4_BUDGET_FAMILY,
        'margin': Q4_BUDGET_FAMILY - len(buf),
        'cos': cos_sim(W_ref, W_low.reshape(W_ref.shape)),
        'MAE': float(np.abs(W_ref - W_low.reshape(W_ref.shape)).mean()),
        'max_error': float(np.abs(W_ref - W_low.reshape(W_ref.shape)).max()),
        'has_nan': bool(np.isnan(W_low).any()),
        'has_inf': bool(np.isinf(W_low).any()),
    }

def residual_encode(R, k_pct):
    """Encode residual using source-of-truth encode_sdir."""
    return encode_sdir(R, k_pct=k_pct)

def mlp_forward(X, W_up, W_gate, W_down):
    """Y = (SiLU(X @ W_gate.T) * (X @ W_up.T)) @ W_down.T"""
    return (silu(X @ W_gate.T) * (X @ W_up.T)) @ W_down.T

def run_layer_sweep(layer_idx, reader4):
    """Run full MLP Q2_K + residual sweep for one layer."""
    # Load all three families
    up = load_layer_family(layer_idx, 'ffn_up', reader4)
    gate = load_layer_family(layer_idx, 'ffn_gate', reader4)
    down = load_layer_family(layer_idx, 'ffn_down', reader4)

    families = {'ffn_up': up, 'ffn_gate': gate, 'ffn_down': down}
    base_bytes = sum(f['actual_bytes'] for f in families.values())

    # Fixed X for reproducibility
    np.random.seed(42 + layer_idx)
    X = np.random.randn(1, up['shape'][1]).astype(np.float32)

    # Reference MLP
    Y_ref = mlp_forward(X, up['W_ref'], gate['W_ref'], down['W_ref'])

    # Q2-only MLP
    Y_q2 = mlp_forward(X, up['W_low'], gate['W_low'], down['W_low'])
    cos_low = cos_sim(Y_ref, Y_q2)
    mae_low = float(np.abs(Y_ref - Y_q2).mean())

    results = []
    for k in [0, 0.5, 1, 2, 3]:
        if k == 0:
            # No residual — use Q2 directly
            Y_sub = Y_q2
            res_bytes = 0
        else:
            # Encode residual for each family using source-of-truth format
            R_up = up['W_ref'] - up['W_low']
            R_gate = gate['W_ref'] - gate['W_low']
            R_down = down['W_ref'] - down['W_low']

            enc_up = residual_encode(R_up, k)
            enc_gate = residual_encode(R_gate, k)
            enc_down = residual_encode(R_down, k)
            res_bytes = len(enc_up) + len(enc_gate) + len(enc_down)

            # Decode residuals
            dec_up = decode_sdir(enc_up)
            dec_gate = decode_sdir(enc_gate)
            dec_down = decode_sdir(enc_down)

            Y_sub = mlp_forward(X,
                                up['W_low']   + dec_up,
                                gate['W_low'] + dec_gate,
                                down['W_low'] + dec_down)

        total_bytes = base_bytes + res_bytes
        margin = Q4_BUDGET_LAYER - total_bytes
        cos_sub = cos_sim(Y_ref, Y_sub)
        mae_sub = float(np.abs(Y_ref - Y_sub).mean())
        mae_delta = mae_sub - mae_low
        delta_cos = cos_sub - cos_low

        results.append({
            'k_pct': k,
            'res_bytes': res_bytes,
            'total_bytes': total_bytes,
            'Q4_budget': Q4_BUDGET_LAYER,
            'margin': margin,
            'memory_positive': margin > 0,
            'cos_low': cos_low,
            'cos_sub': cos_sub,
            'delta_cos': delta_cos,
            'MAE_low': mae_low,
            'MAE_sub': mae_sub,
            'MAE_delta': mae_delta,
            'MAE_improvement': abs(mae_delta) if mae_delta != 0 else 0.0,
            'residual_on_improves': delta_cos > 0,
        })

    return families, results

def main():
    print("=== Phase 31AR: Layers 0-5 Full MLP Q2_K + Low-k Residual Sweep ===\n")

    for p in [Q4KM_PATH, LIB_PATH]:
        exists = os.path.exists(p)
        size = os.path.getsize(p) / 1024 / 1024 if exists else 0
        print(f"{'FOUND' if exists else 'MISSING'}: {os.path.basename(p)} — {size:.1f} MB")

    reader4 = GGUFReader(Q4KM_PATH)

    # Verify layer shapes
    layer_shapes = {}
    for layer_idx in range(6):
        for family in ['ffn_up', 'ffn_gate', 'ffn_down']:
            name = f'blk.{layer_idx}.{family}.weight'
            t = next(t for t in reader4.tensors if t.name == name)
            layer_shapes[f'l{layer_idx}_{family}'] = list(t.shape)
    print(f"\nVerified {len(layer_shapes)} tensor shapes")

    # Q2_K byte verification
    print("\n=== Q2_K BYTE VERIFICATION ===")
    all_families = {}
    for layer_idx in range(6):
        print(f"\nLayer {layer_idx}:")
        families, _ = run_layer_sweep(layer_idx, reader4)
        all_families[layer_idx] = families
        for family in ['ffn_up', 'ffn_gate', 'ffn_down']:
            f = families[family]
            mark = "✓" if f['byte_match'] else "✗"
            print(f"  {family}: expected={f['expected_bytes']:,}, actual={f['actual_bytes']:,}, "
                  f"match={f['byte_match']} {mark}, cos={f['cos']:.6f}, MAE={f['MAE']:.6f}")

    # Per-layer MLP policy sweep
    print("\n=== PER-LAYER MLP POLICY SWEEP ===")
    per_layer_results = {}
    for layer_idx in range(6):
        print(f"\n  Layer {layer_idx}:")
        _, results = run_layer_sweep(layer_idx, reader4)
        per_layer_results[layer_idx] = results
        for r in results:
            print(f"    k={r['k_pct']}%: margin={r['margin']:+}, delta_cos={r['delta_cos']:+.5f}, "
                  f"MAE_delta={r['MAE_delta']:+.6f}, mem_pos={r['memory_positive']}")

    # Aggregate policy table
    print("\n=== AGGREGATE POLICY TABLE ===")
    Q4_BUDGET_TOTAL = Q4_BUDGET_LAYER * 6  # 39,223,296 for all 6 layers

    aggregate = {}
    for k in [0, 0.5, 1, 2, 3]:
        layer_margins = []
        layer_delta_cos = []
        layer_mae_imps = []
        layer_mem_pos = []
        total_bytes = 0

        for layer_idx in range(6):
            r = next(x for x in per_layer_results[layer_idx] if x['k_pct'] == k)
            layer_margins.append(r['margin'])
            layer_delta_cos.append(r['delta_cos'])
            layer_mae_imps.append(r['MAE_improvement'])
            layer_mem_pos.append(r['memory_positive'])
            total_bytes += r['total_bytes']

        agg_margin = sum(layer_margins)
        worst_margin = min(layer_margins)
        n_mem_pos = sum(layer_mem_pos)

        aggregate[k] = {
            'k_pct': k,
            'n_layers': 6,
            'n_memory_positive': n_mem_pos,
            'total_bytes': total_bytes,
            'agg_budget': Q4_BUDGET_TOTAL,
            'aggregate_margin': agg_margin,
            'worst_margin': worst_margin,
            'avg_delta_cos': sum(layer_delta_cos) / 6,
            'total_mae_improvement': sum(layer_mae_imps),
            'all_memory_positive': all(layer_mem_pos),
            'layer_margins': layer_margins,
            'layer_delta_cos': layer_delta_cos,
            'layer_mae_improvement': layer_mae_imps,
        }

        print(f"  k={k}%: agg_margin={agg_margin:+,}, worst={worst_margin:+,}, "
              f"n_mem_pos={n_mem_pos}/6, avg_delta_cos={sum(layer_delta_cos)/6:+.5f}")

    # Policy selection
    print("\n=== POLICY SELECTION ===")
    best_k = None
    classification = None

    for k in [2, 1, 0.5]:
        r = aggregate[k]
        if r['all_memory_positive']:
            best_k = k
            classification = "PASS_LAYERS0_5_MLP_Q2K_LOWK_POLICY_FOUND"
            print(f"  Selected: k={k}% — {classification}")
            break

    if best_k is None:
        for k in [2, 1, 0.5]:
            r = aggregate[k]
            if r['aggregate_margin'] > 0:
                best_k = k
                classification = "PARTIAL_LAYER_VARIANCE"
                print(f"  Selected (PARTIAL_LAYER_VARIANCE): k={k}%")
                break

    if best_k is None:
        classification = "BLOCKED_ALL_POLICIES_FAIL_MEMORY"
        print(f"  No policy selected — {classification}")

    print(f"\nClassification: {classification}")

    # Build result
    result = {
        'phase': '31AR',
        'classification': classification,
        'best_policy_k': best_k,
        'layer_shapes': layer_shapes,
        'q2_byte_table': {},
        'per_layer_results': {},
        'aggregate': {k: aggregate[k] for k in [0, 0.5, 1, 2, 3]},
        'mae_convention': {
            'MAE_delta_formula': 'MAE_sub - MAE_low',
            'MAE_delta_positive_means': 'MAE worsened (higher error)',
            'MAE_delta_negative_means': 'MAE improved (lower error)',
            'MAE_improvement': 'abs(MAE_delta); positive means MAE improved',
        },
    }

    for layer_idx in range(6):
        families = all_families[layer_idx]
        result['q2_byte_table'][f'layer{layer_idx}'] = {}
        for family in ['ffn_up', 'ffn_gate', 'ffn_down']:
            f = families[family]
            result['q2_byte_table'][f'layer{layer_idx}'][family] = {
                'shape': f['shape'],
                'n_elements': f['n_elements'],
                'n_blocks': f['n_blocks'],
                'expected_bytes': f['expected_bytes'],
                'actual_bytes': f['actual_bytes'],
                'byte_match': f['byte_match'],
                'bpe': round(f['bpe'], 6),
                'Q4_budget': f['Q4_budget'],
                'margin': f['margin'],
                'cos': round(f['cos'], 8),
                'MAE': round(f['MAE'], 8),
                'max_error': round(f['max_error'], 8),
            }
        result['per_layer_results'][f'layer{layer_idx}'] = per_layer_results[layer_idx]

    # Write outputs
    json_path = f"{REPO}/results/PHASE31AR_LAYERS0_5_MLP_Q2K_LOWK_SWEEP.json"
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)

    # Markdown doc
    md = [
        "# Phase 31AR — Layers 0-5 Full MLP Q2_K + Low-k Residual Sweep",
        "",
        f"## Classification",
        f"**`{classification}`**",
        "",
        f"## Best Policy: k={best_k}%",
        "",
        "## Q2_K Byte Verification",
        "",
        "| Layer | Family | Shape | n_elements | n_blocks | bytes | bpe | margin | cos | MAE |",
        "|------|--------|-------|-----------|---------|-------|-----|--------|-----|-----|",
    ]
    for layer_idx in range(6):
        for family in ['ffn_up', 'ffn_gate', 'ffn_down']:
            d = result['q2_byte_table'][f'layer{layer_idx}'][family]
            mark = "✓" if d['byte_match'] else "✗"
            md.append(
                f"| {layer_idx} | {family} | {'×'.join(map(str,d['shape']))} | "
                f"{d['n_elements']:,} | {d['n_blocks']:,} | {d['actual_bytes']:,} | "
                f"{d['bpe']:.4f} | {d['margin']:,} | {d['cos']:.6f} | {d['MAE']:.6f} | {mark}"
            )

    md += ["", "## Per-Layer Policy Table", ""]
    md.append("| k | Layer | margin | delta_cos | MAE_delta | mem_pos |")
    md.append("|---|-------|--------|-----------|-----------|---------|")
    for layer_idx in range(6):
        for r in per_layer_results[layer_idx]:
            md.append(
                f"| {r['k_pct']} | {layer_idx} | {r['margin']:+,} | "
                f"{r['delta_cos']:+.5f} | {r['MAE_delta']:+.6f} | {r['memory_positive']} |"
            )

    md += ["", "## Aggregate Policy Table", ""]
    md.append("| k | agg_margin | worst_margin | n_mem_pos/6 | avg_delta_cos |")
    md.append("|---|------------|--------------|-------------|---------------|")
    for k in [0, 0.5, 1, 2, 3]:
        r = aggregate[k]
        md.append(
            f"| {k} | {r['aggregate_margin']:+,} | {r['worst_margin']:+,} | "
            f"{r['n_memory_positive']}/6 | {r['avg_delta_cos']:+.5f} |"
        )

    md_path = f"{REPO}/docs/PHASE31AR_LAYERS0_5_MLP_Q2K_LOWK_SWEEP.md"
    with open(md_path, 'w') as f:
        f.write("\n".join(md))

    print(f"\nResults written to:\n  {json_path}\n  {md_path}")
    print(f"Classification: {classification}")
    print(f"Best policy k={best_k}%")
    return result

if __name__ == "__main__":
    result = main()