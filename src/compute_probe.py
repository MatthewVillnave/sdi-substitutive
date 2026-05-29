#!/usr/bin/env python3
"""
compute_probe.py — Probe harness for Phase 31H compressed residual compute.

Tests the compressed residual compute path on real GGUF weights and recorded
activations. Verifies that streaming sparse apply matches dense reference decode,
and reports memory/approximation metrics.

Key insight on matmul direction:
  Weights are stored as W_ref (d_out, d_in).
  Activations X_batch are (n_prompts, d_in).
  For the residual contribution: Y_delta = X @ R (where R = W_ref - W_low).
  Since R has shape (d_out, d_in), we need Y_delta = X @ R.T = (n_prompts, d_in) @ (d_in, d_out) = (n_prompts, d_out).

  So compute as: Y_delta = X_batch @ R.T (transpose R because X is (batch, d_in))
  W_low.T and R.T are both (d_in, d_out).

Usage:
  python compute_probe.py

Output:
  results/PHASE31H_COMPRESSED_RESIDUAL_COMPUTE.json
  docs/PHASE31H_COMPRESSED_RESIDUAL_COMPUTE.md
"""

import sys
import os
import json
import time
import pathlib
import resource

import numpy as np

REPO_DIR = pathlib.Path.home() / "sdi-substitutive"
RESULTS_DIR = REPO_DIR / "results"
DATA_DIR = REPO_DIR / "data"
sys.path.insert(0, str(REPO_DIR / "src"))
import gguf

MODEL_Q4 = "/media/matthew-villnave/VL_usb/models/qwen2.5-0.5b-official/qwen2.5-0.5b-instruct-q4_k_m.gguf"
ACTIVATION_NPZ = DATA_DIR / "PHASE31G_activations.npz"
PROMPTS = ["Hi", "The capital of France is", "2+2=", "def add(a, b):", "Once upon a time"]
TARGET_LAYERS = [0, 1, 2]
TENSOR_FAMILIES = ["ffn_up", "ffn_down"]
K_PCT = 7.5

os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# Helpers
# =============================================================================

def cosine_batch(Y_ref, Y):
    return float(np.sum(Y_ref * Y) / (
        np.linalg.norm(Y_ref.ravel()) * np.linalg.norm(Y.ravel()) + 1e-10))


def mae_batch(Y_ref, Y):
    return float(np.mean(np.abs(Y_ref - Y)))


def maxae_batch(Y_ref, Y):
    return float(np.max(np.abs(Y_ref - Y)))


def get_peak_rss_mb():
    """Get current process RSS in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def compute_W_low_from_W_ref(W_ref):
    """Q2-style per-block quantization."""
    BLOCK_SIZE = 256
    rows, cols = W_ref.shape
    W_low = np.zeros_like(W_ref)
    for r in range(rows):
        for c_start in range(0, cols, BLOCK_SIZE):
            block = W_ref[r, c_start:c_start + BLOCK_SIZE]
            if block.size == 0:
                continue
            scale = np.abs(block).max()
            if scale > 0:
                block_q = np.round(block / scale)
                W_low[r, c_start:c_start + BLOCK_SIZE] = block_q * scale
            else:
                W_low[r, c_start:c_start + BLOCK_SIZE] = block
    return W_low


# =============================================================================
# Main probe
# =============================================================================

def main():
    print("=" * 70, flush=True)
    print("Phase 31H: Compressed Residual Compute Probe", flush=True)
    print("=" * 70, flush=True)

    # Import from our modules
    from residual_encode import encode_residual, EncodedResidual
    from residual_compute import (
        reference_compute_Y_sub,
        streaming_compute_Y_sub,
        verify_equivalence,
    )

    # ---- Step 1: Load GGUF weights ----
    print("\n[1] Loading GGUF weights for layers 0-2...", flush=True)
    reader = gguf.GGUFReader(MODEL_Q4)
    tensors = {t.name: t for t in reader.tensors}

    # stored_weights: (layer, family) -> W_ref (d_out, d_in)
    stored_weights = {}
    for layer in TARGET_LAYERS:
        for family in TENSOR_FAMILIES:
            key = f"blk.{layer}.{family}.weight"
            t = tensors[key]
            W_ref = gguf.dequantize(t.data, t.tensor_type).astype(np.float32)
            d_out, d_in = W_ref.shape

            W_low = compute_W_low_from_W_ref(W_ref)
            R_f32 = (W_ref - W_low).astype(np.float32)

            n_elements = d_out * d_in
            Q4_bytes = int(n_elements * 0.5625)
            Q2_bytes = int(n_elements * 0.25)
            residual_budget = Q4_bytes - Q2_bytes

            stored_weights[(layer, family)] = {
                "W_ref": W_ref,        # (d_out, d_in)
                "W_low": W_low,        # (d_out, d_in)
                "R_f32": R_f32,        # (d_out, d_in)
                "d_out": d_out,
                "d_in": d_in,
                "Q4_bytes": Q4_bytes,
                "Q2_bytes": Q2_bytes,
                "residual_budget": residual_budget,
            }
            print(f"  blk.{layer}.{family}: {d_out}×{d_in}, "
                  f"Q4={Q4_bytes:,} bytes, budget={residual_budget:,} bytes", flush=True)

    # ---- Step 2: Encode residuals ----
    print(f"\n[2] Encoding residuals at k={K_PCT}%...", flush=True)

    encoded = {}
    for key in stored_weights:
        w = stored_weights[key]
        R = w["R_f32"]  # (d_out, d_in)
        enc = encode_residual(R, k_pct=K_PCT, global_topk=True)

        header_bytes = 32
        bitmap_bytes = enc.bitmap_bytes
        value_bytes = enc.value_bytes
        total_encoded_bytes = header_bytes + bitmap_bytes + value_bytes
        margin = w["residual_budget"] - total_encoded_bytes

        encoded[key] = {
            "enc": enc,
            "header_bytes": header_bytes,
            "bitmap_bytes": bitmap_bytes,
            "value_bytes": value_bytes,
            "total_encoded_bytes": total_encoded_bytes,
            "memory_viable": total_encoded_bytes <= w["residual_budget"],
            "memory_margin": margin,
            "ebpw": enc.effective_bits_per_weight,
            "nnz": enc.nnz,
        }
        e = encoded[key]
        print(f"  blk.{key[0]}.{key[1]}: nnz={enc.nnz:,}, "
              f"bitmap={bitmap_bytes:,}, values={value_bytes:,}, "
              f"total={total_encoded_bytes:,}, budget={w['residual_budget']:,}, "
              f"margin={margin:,}, viable={e['memory_viable']}", flush=True)

    # ---- Step 3: Load activations ----
    print("\n[3] Loading recorded activations...", flush=True)
    data = np.load(ACTIVATION_NPZ, allow_pickle=True)
    activations = {}
    for (layer, family) in [(l, f) for l in TARGET_LAYERS for f in TENSOR_FAMILIES]:
        key = f"layer{layer}_{family}"
        activations[(layer, family)] = data[key]
        print(f"  {key}: shape={activations[(layer, family)].shape}", flush=True)

    # ---- Step 4: Compute correctness and approximation metrics ----
    print(f"\n[4] Running compute probe...", flush=True)

    all_results = []
    correctness_per_tensor = {}

    for key in stored_weights:
        layer, family = key
        w = stored_weights[key]
        e = encoded[key]
        X_batch = activations[key]  # (5, d_in)
        n_prompts = X_batch.shape[0]

        W_ref = w["W_ref"]    # (d_out, d_in)
        W_low = w["W_low"]    # (d_out, d_in)
        R_f32 = w["R_f32"]    # (d_out, d_in)
        enc = e["enc"]

        d_out, d_in = W_ref.shape
        assert X_batch.shape == (n_prompts, d_in), \
            f"X_batch shape {X_batch.shape} != ({n_prompts}, {d_in})"

        # Reference: Y_ref = X @ W_ref.T  (batch, d_out)
        #   X_batch: (n_prompts, d_in), W_ref: (d_out, d_in)
        #   X @ W_ref.T: (n_prompts, d_in) @ (d_in, d_out) = (n_prompts, d_out) ✓
        Y_ref = X_batch @ W_ref.T

        # W_low only: Y_low = X @ W_low.T
        Y_low = X_batch @ W_low.T

        # Sub using dense decode reference: Y_sub_dense = X @ (W_low + R_dense).T
        Y_sub_dense = reference_compute_Y_sub(X_batch, W_low.T, enc)  # W_low is (d_out, d_in), no transpose

        # Sub using streaming sparse apply: Y_sub_stream = X @ (W_low + R_sparse).T
        Y_sub_stream = streaming_compute_Y_sub(X_batch, W_low.T, enc)

        # Correctness: dense ref vs streaming
        verify = verify_equivalence(Y_sub_dense, Y_sub_stream)
        max_diff = verify["max_abs_diff"]
        cos_correct = verify["cosine"]
        within_tol = verify["within_tolerance"]

        correctness_per_tensor[f"blk.{layer}.{family}"] = {
            "cosine": cos_correct,
            "max_diff": max_diff,
            "within_tolerance": within_tol,
        }

        print(f"  blk.{layer}.{family}: correctness_cos={cos_correct:.8f}, "
              f"max_diff={max_diff:.2e}, within_tol={within_tol}", flush=True)

        # Approximation metrics per prompt
        for pi in range(n_prompts):
            Y_ref_p = Y_ref[pi:pi+1]   # (1, d_out)
            Y_low_p = Y_low[pi:pi+1]   # (1, d_out)
            Y_sub_dense_p = Y_sub_dense[pi:pi+1]  # (1, d_out)
            Y_sub_stream_p = Y_sub_stream[pi:pi+1]  # (1, d_out)

            cos_low = cosine_batch(Y_ref_p, Y_low_p)
            cos_sub_stream = cosine_batch(Y_ref_p, Y_sub_stream_p)
            delta_cos = cos_sub_stream - cos_low

            mae_low = mae_batch(Y_ref_p, Y_low_p)
            mae_sub_stream = mae_batch(Y_ref_p, Y_sub_stream_p)
            mae_delta = mae_low - mae_sub_stream

            maxe_low = maxae_batch(Y_ref_p, Y_low_p)
            maxe_sub_stream = maxae_batch(Y_ref_p, Y_sub_stream_p)
            maxe_delta = maxe_low - maxe_sub_stream

            all_results.append({
                "layer": layer,
                "family": family,
                "prompt_idx": pi,
                "prompt": PROMPTS[pi],
                "cosine_low": round(cos_low, 6),
                "cosine_sub_stream": round(cos_sub_stream, 6),
                "delta_cosine": round(delta_cos, 6),
                "MAE_low": round(mae_low, 8),
                "MAE_sub_stream": round(mae_sub_stream, 8),
                "MAE_improvement": round(mae_delta, 8),
                "max_error_low": round(maxe_low, 8),
                "max_error_sub_stream": round(maxe_sub_stream, 8),
                "max_error_improvement": round(maxe_delta, 8),
                "correctness_cosine": round(cos_correct, 8),
                "correctness_max_diff": round(max_diff, 12),
                "correctness_within_tolerance": within_tol,
                "memory_viable": e["memory_viable"],
            })

    # ---- Step 5: Collect statistics ----
    print("\n[5] Collecting statistics...", flush=True)

    correctness_cosines = [r["correctness_cosine"] for r in all_results]
    deltas = [r["delta_cosine"] for r in all_results]
    improving = sum(1 for d in deltas if d > 0)
    worst_delta = min(deltas)
    best_delta = max(deltas)
    mean_delta = round(float(np.mean(deltas)), 6)
    worst_combo = min(all_results, key=lambda c: c["delta_cosine"])

    mean_mae_improvement = round(float(np.mean([r["MAE_improvement"] for r in all_results])), 8)
    worst_mae_improvement = round(float(np.min([r["MAE_improvement"] for r in all_results])), 8)
    regression_count = sum(1 for r in all_results if r["delta_cosine"] < 0)

    print(f"  Correctness: max_diff max={max(r['correctness_max_diff'] for r in all_results):.2e}, "
          f"cosine min={min(correctness_cosines):.8f}", flush=True)
    print(f"  Approximation: mean_delta={mean_delta:+.6f}, "
          f"worst={worst_delta:+.6f}, improving={improving}/{len(all_results)}", flush=True)
    print(f"  Regressions: {regression_count}", flush=True)

    # ---- Step 6: Memory accounting table ----
    print("\n[6] Memory accounting...", flush=True)

    memory_table = []
    for key in stored_weights:
        layer, family = key
        w = stored_weights[key]
        e = encoded[key]
        enc = e["enc"]

        memory_table.append({
            "tensor": f"blk.{layer}.{family}.weight",
            "shape": [w["d_out"], w["d_in"]],
            "W_ref_Q4_bytes": w["Q4_bytes"],
            "W_low_Q2_bytes": w["Q2_bytes"],
            "encoded_total_bytes": e["total_encoded_bytes"],
            "encoded_bitmap_bytes": e["bitmap_bytes"],
            "encoded_value_bytes": e["value_bytes"],
            "encoded_header_bytes": e["header_bytes"],
            "residual_budget": w["residual_budget"],
            "memory_margin": e["memory_margin"],
            "memory_viable": e["memory_viable"],
            "ebpw": round(e["ebpw"], 4),
            "nnz": enc.nnz,
            "W_low_plus_encoded": w["Q2_bytes"] + e["total_encoded_bytes"],
            "margin_vs_Q4": w["Q2_bytes"] + e["total_encoded_bytes"] - w["Q4_bytes"],
        })

    # ---- Step 7: Classification ----
    print("\n[7] Classification...", flush=True)

    all_correct = all(r["correctness_within_tolerance"] for r in all_results)
    all_viable = all(e["memory_viable"] for e in encoded.values())
    all_improve = all(r["delta_cosine"] > 0 for r in all_results)
    min_cos = min(correctness_cosines)

    if all_correct and all_viable and all_improve:
        classification = "PASS_COMPRESSED_COMPUTE_EQUIVALENT"
    elif all_correct and all_viable:
        classification = "PARTIAL_APPROXIMATION_WEAK"
    elif all_correct:
        classification = "PARTIAL_COMPUTE_CORRECT_MEMORY_THEORETICAL"
    elif min_cos > 0.9999:
        classification = "PARTIAL_STREAMING_NUMERICAL_DRIFT"
    else:
        classification = "BLOCKED_ENCODING_BUG"

    print(f"  classification: {classification}", flush=True)
    print(f"  all_correct={all_correct}, all_viable={all_viable}, "
          f"all_improve={all_improve}, min_cos={min_cos:.8f}", flush=True)

    # ---- Step 8: Build results JSON ----
    print("\n[8] Writing results JSON...", flush=True)

    results_json = {
        "phase": "31H",
        "classification": classification,
        "old_head": "1e195b3",
        "new_head": None,
        "encoded_format": {
            "type": "dense_bitmap + global_topk + fp16_values",
            "header_bytes": 32,
            "bitmap_encoding": "dense_bitmap (1 bit per weight, row-major, LSB first)",
            "value_dtype": "fp16",
            "k_pct": K_PCT,
            "magic": "RSC\\x00",
            "version": 1,
        },
        "memory_accounting": memory_table,
        "correctness_check": {
            "all_within_tolerance": all_correct,
            "max_abs_diff_max": round(max(r["correctness_max_diff"] for r in all_results), 12),
            "cosine_min": round(min(correctness_cosines), 8),
            "dense_materialized_reference": True,
            "dense_materialized_stream": False,
            "per_tensor": correctness_per_tensor,
        },
        "approximation_metrics": {
            "mean_delta_cosine": mean_delta,
            "worst_delta_cosine": round(worst_delta, 6),
            "best_delta_cosine": round(best_delta, 6),
            "improving_count": improving,
            "total_combos": len(all_results),
            "mean_mae_improvement": mean_mae_improvement,
            "worst_mae_improvement": worst_mae_improvement,
            "regression_count": regression_count,
        },
        "per_combo_results": all_results,
        "summary": {
            "phase31i_unlocked": classification.startswith("PASS_"),
            "streaming_no_dense_materialized": True,
            "reference_path_materializes_dense": True,
        },
    }

    json_path = RESULTS_DIR / "PHASE31H_COMPRESSED_RESIDUAL_COMPUTE.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"  [Wrote] {json_path}", flush=True)

    # ---- Step 9: Print summary tables ----
    print("\n" + "=" * 70, flush=True)
    print("CORRECTNESS TABLE", flush=True)
    print(f"{'Tensor':<20} | {'Cosine(dense,stream)':>20} | {'MaxDiff':>12} | {'OK?':>4}", flush=True)
    print("-" * 65, flush=True)
    for k, v in correctness_per_tensor.items():
        ok = "✅" if v["within_tolerance"] else "❌"
        print(f"{k:<20} | {v['cosine']:>20.8f} | {v['max_diff']:>12.2e} | {ok:>4}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("MEMORY ACCOUNTING TABLE", flush=True)
    print(f"{'Tensor':<18} | {'Q4':>8} | {'Q2':>8} | {'Bitmap':>8} | {'Values':>8} | "
          f"{'TotalEnc':>8} | {'Budget':>8} | {'Margin':>8} | {'OK?':>4}", flush=True)
    print("-" * 105, flush=True)
    for row in memory_table:
        ok = "✅" if row["memory_viable"] else "❌"
        print(f"{row['tensor']:<18} | {row['W_ref_Q4_bytes']:>8,} | {row['W_low_Q2_bytes']:>8,} | "
              f"{row['encoded_bitmap_bytes']:>8,} | {row['encoded_value_bytes']:>8,} | "
              f"{row['encoded_total_bytes']:>8,} | {row['residual_budget']:>8,} | "
              f"{row['memory_margin']:>8,} | {ok:>4}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("APPROXIMATION METRICS (per tensor)", flush=True)
    print(f"{'Tensor':<18} | {'cos_low':>8} | {'cos_stream':>8} | {'ΔCos':>9} | "
          f"{'MAE_low':>10} | {'MAE_stream':>10} | {'ΔMAE':>10}", flush=True)
    print("-" * 90, flush=True)
    for key in stored_weights:
        layer, family = key
        tensor_results = [r for r in all_results
                          if r["layer"] == layer and r["family"] == family]
        cos_low_vals = [r["cosine_low"] for r in tensor_results]
        cos_sub_vals = [r["cosine_sub_stream"] for r in tensor_results]
        delta_vals = [r["delta_cosine"] for r in tensor_results]
        mae_low_vals = [r["MAE_low"] for r in tensor_results]
        mae_sub_vals = [r["MAE_sub_stream"] for r in tensor_results]
        mae_improvement_vals = [r["MAE_improvement"] for r in tensor_results]

        print(f"blk.{layer}.{family}: "
              f"{np.mean(cos_low_vals):>8.6f} | {np.mean(cos_sub_vals):>8.6f} | "
              f"{np.mean(delta_vals):>+9.6f} | "
              f"{np.mean(mae_low_vals):>10.6f} | {np.mean(mae_sub_vals):>10.6f} | "
              f"{np.mean(mae_improvement_vals):>+10.6f}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"Classification: {classification}", flush=True)
    print(f"Correctness: all_correct={all_correct}, min_cosine={min(correctness_cosines):.8f}", flush=True)
    print(f"Memory: all_viable={all_viable}", flush=True)
    print(f"Approximation: mean_delta={mean_delta:+.6f}, worst={worst_delta:+.6f}, "
          f"regressions={regression_count}", flush=True)
    print(f"Phase 31I unlocked: {classification.startswith('PASS_')}", flush=True)
    print("=" * 70, flush=True)

    return results_json


if __name__ == "__main__":
    main()