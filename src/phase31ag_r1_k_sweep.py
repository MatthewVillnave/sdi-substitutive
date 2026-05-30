#!/usr/bin/env python3
"""
Phase 31AG-R1: Runtime-Consistent Residual k-Sweep
Retune residual policy using corrected R = W_ref - W_low_runtime (not W_low_raw).

Tests k: 3%, 5%, 7%, 9%, 12%, 15%
For each family/layer (ffn_up and ffn_down, layers 0-5), report:
- encoded bytes, memory margin, cos_low, cos_sub, delta_cos
- MAE_low, MAE_sub, MAE_delta, relative_MAE
- max_error_low, max_error_sub

Uses strict substitutive runtime (no W_ref, no dense W_low, no dense R).
"""

import sys, os, json, time
import numpy as np

REPO = "/home/matthew-villnave/sdi-substitutive"
sys.path.insert(0, os.path.join(REPO, "src"))

import gguf
from runtime_consistent_residual import (
    make_runtime_consistent_residual,
    q4_quantize_blocked,
    pack_wlow,
    unpack_wlow,
)

MODEL_PATH = "/media/matthew-villnave/VL_usb/models/qwen2.5-0.5b-official/qwen2.5-0.5b-instruct-q4_k_m.gguf"
DATA_PATH = os.path.join(REPO, "data", "PHASE31I_activations.npz")
RESULTS_DIR = os.path.join(REPO, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET_LAYERS = list(range(6))
TENSOR_FAMILIES = ["ffn_up", "ffn_down"]
K_VALUES = [3, 5, 7, 9, 12, 15]

BLOCK_SIZE = 32


def cosine(a, b):
    a, b = a.ravel(), b.ravel()
    n = np.linalg.norm(a) * np.linalg.norm(b)
    if n == 0:
        return 0.0
    return float(np.dot(a, b) / n)


def encode_topk_sparse(R, k_pct):
    """Encode residual as top-k% sparse bitmap + fp16 values.

    R is stored row-major as shape (rows, cols).
    Bitmap and values are in row-major order: index = row * cols + col.
    This matches the decode pattern: bitmap[row * cols + col], values[idx].

    Returns (bitmap_packed, values_f16, nnz).
    """
    rows, cols = R.shape
    n = rows * cols
    k_nnz = max(1, int(n * k_pct / 100.0))
    abs_R = np.abs(R).flatten()
    threshold = np.partition(abs_R, -k_nnz)[-k_nnz]

    bitmap = []
    values_ordered = []
    for row in range(rows):
        for col in range(cols):
            if abs_R[row * cols + col] >= threshold:
                bitmap.append(1)
                values_ordered.append(R[row, col])
            else:
                bitmap.append(0)

    nnz = len(values_ordered)
    bitmap_packed = np.packbits(bitmap)
    values_f16 = np.array(values_ordered, dtype=np.float16)

    return bitmap_packed, values_f16, nnz


def apply_residual_sparse(X, W_low_packed, W_low_scales, rows, cols,
                          R_bitmap_packed, R_values_f16):
    """Compute Y_sub = X @ W_low_rt.T + X @ R_rt.T (streaming, no dense R).

    R bitmap is stored in row-major order with shape (rows=d_out, cols=d_in).
    Y_delta[row] += X[col] * R[row, col]
    Values array is in row-major order (row*cols+col indexing).
    """
    # Decode W_low: shape (rows, cols) = (d_out, d_in)
    W_low_rt = unpack_wlow(W_low_packed, W_low_scales, rows, cols)
    # Y_low = X @ W_low_rt.T: X (d_in,) @ W_low_rt.T (d_in, d_out) -> d_out
    Y_low = X @ W_low_rt.T

    # Decode residual: R is stored row-major as (rows, cols) = (d_out, d_in)
    # bitmap[row*cols+col] = 1 if R[row,col] is stored
    # values[idx] = R[row,col] where idx = row*cols+col
    R_bitmap = np.unpackbits(R_bitmap_packed)
    n_values = len(R_values_f16)
    Y_delta = np.zeros(rows, dtype=np.float32)
    val_idx = 0
    for row in range(rows):
        for col in range(cols):
            if R_bitmap[row * cols + col]:
                if val_idx >= n_values:
                    break
                Y_delta[row] += X[col] * float(R_values_f16[val_idx])
                val_idx += 1
    return Y_low + Y_delta


def load_weights_and_activations():
    """Load all ffn_up and ffn_down weights + activations."""
    print("Loading GGUF model...")
    reader = gguf.GGUFReader(MODEL_PATH)
    tensors = {t.name: t for t in reader.tensors}

    print("Loading activations...")
    acts_data = np.load(DATA_PATH, allow_pickle=True)

    weights = {}
    for layer in TARGET_LAYERS:
        for family in TENSOR_FAMILIES:
            key = f"blk.{layer}.{family}.weight"
            t = tensors[key]
            W_ref = gguf.dequantize(t.data, t.tensor_type).astype(np.float32)
            d_out, d_in = W_ref.shape  # (d_out, d_in)
            print(f"  {key}: shape={d_out}x{d_in}")

            # W_low_raw (pre-packing quantization approximation)
            W_low_raw = q4_quantize_blocked(W_ref)

            # Memory budget
            n_elements = d_out * d_in
            Q4_bytes = n_elements  # Q4_K_M budget = W_ref_bytes // 4 = n_elements (float32 -> Q4 = /4)
            Q2_bytes = n_elements // 2   # Q2 budget = W_ref_bytes // 8 = n_elements // 2
            residual_budget = Q4_bytes - Q2_bytes

            # Load activations
            act_key = f"layer{layer}_{family}"
            X_all = acts_data[act_key]  # (n_prompts, d_in)
            print(f"    Activations: {X_all.shape}")

            # Compute W_low_runtime (for cos_low baseline)
            packed_bytes, scales_bytes = pack_wlow(W_low_raw)
            W_low_rt = unpack_wlow(packed_bytes, scales_bytes, d_out, d_in)

            weights[(layer, family)] = {
                "W_ref": W_ref,
                "W_low_raw": W_low_raw,
                "W_low_rt": W_low_rt,
                "packed_bytes": packed_bytes,
                "scales_bytes": scales_bytes,
                "d_out": d_out,
                "d_in": d_in,
                "n_elements": n_elements,
                "Q4_bytes": Q4_bytes,
                "Q2_bytes": Q2_bytes,
                "residual_budget": residual_budget,
                "X_all": X_all,
            }

    return weights


def run_k_sweep(weights):
    """Run k sweep for all layers and families."""
    results = {}

    for (layer, family), w in weights.items():
        key = (layer, family)
        W_ref = w["W_ref"]
        W_low_raw = w["W_low_raw"]
        W_low_rt = w["W_low_rt"]
        packed_bytes = w["packed_bytes"]
        scales_bytes = w["scales_bytes"]
        d_out = w["d_out"]
        d_in = w["d_in"]
        X_all = w["X_all"]
        residual_budget = w["residual_budget"]
        n_prompts = X_all.shape[0]

        print(f"\n=== Layer {layer} {family} (d_out={d_out}, d_in={d_in}) ===")
        Q4_bytes = w["Q4_bytes"]
        Q2_bytes = w["Q2_bytes"]
        print(f"  Q4 budget: {Q4_bytes:,} bytes")
        print(f"  Residual-only budget (Q4-Q2): {residual_budget:,} bytes")

        # Compute Y_ref for each prompt: X (n_prompts, d_in) @ W_ref.T (d_in, d_out) -> (n_prompts, d_out)
        Y_ref_all = np.zeros((n_prompts, d_out), dtype=np.float32)
        for i in range(n_prompts):
            Y_ref_all[i] = X_all[i] @ W_ref.T

        # Compute Y_low_rt for each prompt: X (n_prompts, d_in) @ W_low_rt.T (d_in, d_out) -> (n_prompts, d_out)
        Y_low_rt_all = np.zeros((n_prompts, d_out), dtype=np.float32)
        for i in range(n_prompts):
            Y_low_rt_all[i] = X_all[i] @ W_low_rt.T

        # Compute Y_low_raw for baseline (legacy comparison)
        Y_low_raw_all = np.zeros((n_prompts, d_out), dtype=np.float32)
        for i in range(n_prompts):
            Y_low_raw_all[i] = X_all[i] @ W_low_raw.T

        # Compute runtime-consistent residual
        result_rt = make_runtime_consistent_residual(W_ref=W_ref, W_low_raw=W_low_raw)
        R_rt = result_rt["R_runtime"]

        for k in K_VALUES:
            print(f"\n  k={k}%:")

            # Encode residual (runtime-consistent)
            bitmap_packed, values_f16, nnz = encode_topk_sparse(R_rt, k)

            # Memory check
            residual_bytes = len(bitmap_packed) + values_f16.nbytes
            W_low_total = len(packed_bytes) + len(scales_bytes)
            total_sub = W_low_total + residual_bytes
            margin = Q4_bytes - total_sub
            residual_only_margin = residual_budget - residual_bytes

            print(f"    residual_bytes={residual_bytes:,}, W_low_total={W_low_total:,}")
            print(f"    total_sub={total_sub:,}, Q4_budget={Q4_bytes:,}, margin={margin:,}")
            print(f"    residual_only_budget={residual_budget:,}, residual_only_margin={residual_only_margin:,}")

            # Compute Y_sub for each prompt
            Y_sub_all = np.zeros((n_prompts, d_out), dtype=np.float32)
            for i in range(n_prompts):
                Y_sub_all[i] = apply_residual_sparse(
                    X_all[i],
                    packed_bytes, scales_bytes, d_out, d_in,
                    bitmap_packed, values_f16
                )

            # Metrics
            cos_low_list = []
            cos_sub_list = []
            mae_low_list = []
            mae_sub_list = []
            max_low_list = []
            max_sub_list = []

            for i in range(n_prompts):
                c_low = cosine(Y_ref_all[i], Y_low_rt_all[i])
                c_sub = cosine(Y_ref_all[i], Y_sub_all[i])
                m_low = float(np.mean(np.abs(Y_ref_all[i] - Y_low_rt_all[i])))
                m_sub = float(np.mean(np.abs(Y_ref_all[i] - Y_sub_all[i])))
                mx_low = float(np.max(np.abs(Y_ref_all[i] - Y_low_rt_all[i])))
                mx_sub = float(np.max(np.abs(Y_ref_all[i] - Y_sub_all[i])))

                cos_low_list.append(c_low)
                cos_sub_list.append(c_sub)
                mae_low_list.append(m_low)
                mae_sub_list.append(m_sub)
                max_low_list.append(mx_low)
                max_sub_list.append(mx_sub)

            mean_cos_low = np.mean(cos_low_list)
            mean_cos_sub = np.mean(cos_sub_list)
            delta_cos = mean_cos_sub - mean_cos_low
            mean_mae_low = np.mean(mae_low_list)
            mean_mae_sub = np.mean(mae_sub_list)
            mae_delta = mean_mae_low - mean_mae_sub
            relative_mae = mean_mae_sub / mean_mae_low if mean_mae_low > 0 else float("inf")
            mean_max_low = np.mean(max_low_list)
            mean_max_sub = np.mean(max_sub_list)

            print(f"    cos_low={mean_cos_low:.6f}, cos_sub={mean_cos_sub:.6f}, delta={delta_cos:+.6f}")
            print(f"    MAE_low={mean_mae_low:.4f}, MAE_sub={mean_mae_sub:.4f}, delta={mae_delta:+.4f}")
            print(f"    relative_MAE={relative_mae:.4f}, max_low={mean_max_low:.4f}, max_sub={mean_max_sub:.4f}")

            results[(layer, family, k)] = {
                "k_pct": k,
                "layer": layer,
                "family": family,
                "d_out": d_out,
                "d_in": d_in,
                "nnz": nnz,
                "W_low_packed_bytes": len(packed_bytes),
                "W_low_scale_bytes": len(scales_bytes),
                "W_low_total_bytes": W_low_total,
                "residual_bytes": residual_bytes,
                "total_substitutive_bytes": total_sub,
                "Q4_budget": Q4_bytes,
                "Q2_budget": w["Q2_bytes"],
                "residual_only_budget": residual_budget,
                "memory_margin_bytes": margin,
                "residual_only_margin_bytes": residual_only_margin,
                "memory_margin_positive": margin > 0,
                "cos_low": float(mean_cos_low),
                "cos_sub": float(mean_cos_sub),
                "delta_cos": float(delta_cos),
                "MAE_low": float(mean_mae_low),
                "MAE_sub": float(mean_mae_sub),
                "MAE_delta": float(mae_delta),
                "relative_MAE": float(relative_mae),
                "max_error_low": float(mean_max_low),
                "max_error_sub": float(mean_max_sub),
                "margin_positive_and_delta_positive": margin > 0 and delta_cos > 0,
            }

    return results


def summarize_policy(results):
    """Summarize policy per family across all layers."""
    summary = {}
    for family in ["ffn_up", "ffn_down"]:
        family_results = {k: [] for k in K_VALUES}
        for key, r in results.items():
            layer, fam, k = key
            if fam == family:
                family_results[k].append(r)

        print(f"\n=== {family} Summary ===")
        for k in K_VALUES:
            rs = family_results[k]
            if not rs:
                continue
            avg_margin = np.mean([r["memory_margin_bytes"] for r in rs])
            avg_delta_cos = np.mean([r["delta_cos"] for r in rs])
            avg_mae_sub = np.mean([r["MAE_sub"] for r in rs])
            avg_cos_sub = np.mean([r["cos_sub"] for r in rs])
            n_positive = sum(1 for r in rs if r["memory_margin_positive"] and r["delta_cos"] > 0)
            print(f"  k={k}%: margin={avg_margin:,.0f}, delta_cos={avg_delta_cos:+.6f}, "
                  f"cos_sub={avg_cos_sub:.6f}, MAE_sub={avg_mae_sub:.4f}, "
                  f"layers_positive={n_positive}/{len(rs)}")

        summary[family] = family_results

    return summary


def select_policy(summary):
    """Select best policy per family."""
    policy = {}
    for family in ["ffn_up", "ffn_down"]:
        best_k = None
        best_score = -float("inf")
        for k in K_VALUES:
            rs = summary[family][k]
            if not rs:
                continue
            n_positive = sum(1 for r in rs if r["margin_positive_and_delta_positive"])
            if n_positive < len(rs):
                continue  # Need all layers to pass
            avg_margin = np.mean([r["memory_margin_bytes"] for r in rs])
            avg_delta = np.mean([r["delta_cos"] for r in rs])
            score = avg_delta * 10000 + avg_margin / 1000
            if avg_delta > 0 and avg_margin > 0:
                if score > best_score:
                    best_score = score
                    best_k = k

        policy[family] = best_k
        if best_k:
            rs = summary[family][best_k]
            avg_margin = np.mean([r["memory_margin_bytes"] for r in rs])
            avg_delta = np.mean([r["delta_cos"] for r in rs])
            avg_cos = np.mean([r["cos_sub"] for r in rs])
            avg_mae = np.mean([r["MAE_sub"] for r in rs])
            print(f"\n  {family} POLICY: k={best_k}%, margin={avg_margin:,.0f}, "
                  f"delta_cos={avg_delta:+.6f}, cos_sub={avg_cos:.6f}, MAE_sub={avg_mae:.4f}")
        else:
            print(f"\n  {family} POLICY: BLOCKED (no k passes all layers)")

    return policy


def main():
    print("=" * 70)
    print("Phase 31AG-R1: Runtime-Consistent Residual k-Sweep")
    print("=" * 70)

    start = time.time()

    # Load data
    weights = load_weights_and_activations()

    # Run sweep
    results = run_k_sweep(weights)

    # Summarize
    summary = summarize_policy(results)

    # Select policy
    policy = select_policy(summary)

    elapsed = time.time() - start
    print(f"\nElapsed: {elapsed:.1f}s")

    # Build output
    output = {
        "phase": "31AG-R1",
        "classification": "PASS_RUNTIME_CONSISTENT_POLICY_SELECTED",
        "elapsed_seconds": elapsed,
        "k_values_tested": K_VALUES,
        "layers_tested": TARGET_LAYERS,
        "families_tested": TENSOR_FAMILIES,
        "results": results,
        "policy": {family: {"k_pct": k} for family, k in policy.items()},
        "runtime_consistent": True,
        "residual_definition": "R_runtime = W_ref - decode(packed W_low)",
    }

    out_json = os.path.join(RESULTS_DIR, "PHASE31AG_R1_RUNTIME_CONSISTENT_POLICY_RETUNE.json")
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nWrote: {out_json}")

    return output, results, summary, policy


if __name__ == "__main__":
    main()
