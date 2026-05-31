#!/usr/bin/env python3
"""
Phase 31BJ — Corrected Q2_K Mode Rebaseline

Rebaseline static tensor-harness results under technically corrected Q2_K mode.

Environment variables required:
    SDI_GGUF_MODEL_PATH   — Qwen2.5 GGUF file
    SDI_LLAMA_CPP_ROOT    — llama.cpp source tree
    SDI_LLAMA_CPP_LIB     — path to libggml-base.so

Run:
    SDI_GGUF_MODEL_PATH=/path/to/gguf \
    SDI_LLAMA_CPP_ROOT=/path/to/llama.cpp \
    SDI_LLAMA_CPP_LIB=/path/to/libggml-base.so \
        .venv/bin/python src/phase31bj_corrected_q2k_rebaseline.py
"""

import json
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA_CPP_ROOT = os.environ.get("SDI_LLAMA_CPP_ROOT", "")
sys.path.insert(0, os.path.join(REPO_DIR, "src"))
sys.path.insert(0, os.path.join(LLAMA_CPP_ROOT, "gguf-py"))

from gguf import GGUFReader
from gguf.quants import dequantize

from phase31x_manifest_runtime import (
    cosine,
    encode_sdir,
    sdir_streaming_apply,
    decode_sdir,
)
from q2k_backend import (
    quantize_q2k_f32_to_bytes,
    dequantize_q2k_bytes_to_f32,
    q2k_expected_nbytes,
    is_available as q2k_is_available,
    lib as q2k_lib,
)

# ─── Constants ────────────────────────────────────────────────────────────────
QK_K = 256          # Q2_K block size
Q2_BLOCK_BYTES = 84  # Q2_K block size in bytes
D_HIDDEN = 896
D_FFN_UP = 4864
D_FFN_DOWN = 896
K_PCT = 1.0         # residual sparsity
ALPHA = 1.0

# Q4 budget (per family, from schema)
Q4_BUDGET_FAMILY = 2_179_072

# Accepted historical values (from Phase 31AY/31BA, historical_floor_flat mode)
HISTORICAL = {
    (21, 9):  {"cos_low": 0.794913, "delta_cos": -0.146059, "MAE_delta": -0.000479, "severe": True},
    (21, 0):  {"cos_low": 0.855612, "delta_cos": +0.017845, "MAE_delta": -0.003169, "severe": False},
    (2, 7):   {"cos_low": None,    "delta_cos": None,        "MAE_delta": None,       "severe": None},  # not in historical
}


# ─── GGUF loading ──────────────────────────────────────────────────────────────
def load_w_ref_from_gguf(gguf_path, layer, family):
    """Load W_ref (float32) from GGUF tensor for a given layer and family."""
    reader = GGUFReader(gguf_path)
    tensor_name = f"blk.{layer}.{family}.weight"
    for tensor in reader.tensors:
        if tensor.name == tensor_name:
            W = dequantize(tensor.data, tensor.tensor_type).astype(np.float32)
            return W  # shape (d_out, d_in) — dequantize returns canonical orientation
    raise ValueError(f"Tensor {tensor_name} not found in GGUF")


# ─── MLP math ─────────────────────────────────────────────────────────────────
def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


def mlp_full(X_2d, W_gate, W_up, W_down):
    """Full MLP: Y = SiLU(X @ W_gate.T) * (X @ W_up.T) @ W_down.T"""
    if X_2d.ndim == 1:
        X_2d = X_2d[np.newaxis, :]
    hidden = silu(X_2d @ W_gate.T) * (X_2d @ W_up.T)
    return hidden @ W_down.T


# ─── Run one anchor in one mode ───────────────────────────────────────────────
def run_anchor(layer, seed, mode, tmpdir, GGUF_PATH):
    """
    Run corrected_ceil_per_row or historical_floor_flat for one layer/seed.
    Returns metrics dict.
    """
    # X vector — FIXED: np.random.default_rng (matches Phase 31AY/31BA path)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((1, D_HIDDEN)).astype(np.float32)

    # Load W_ref from GGUF (raw, dequantized)
    W_ref_up = load_w_ref_from_gguf(GGUF_PATH, layer, "ffn_up")
    W_ref_gate = load_w_ref_from_gguf(GGUF_PATH, layer, "ffn_gate")
    W_ref_down = load_w_ref_from_gguf(GGUF_PATH, layer, "ffn_down")

    # Q2_K encode each family
    q2k_up = quantize_q2k_f32_to_bytes(W_ref_up, mode=mode)
    q2k_gate = quantize_q2k_f32_to_bytes(W_ref_gate, mode=mode)
    q2k_down = quantize_q2k_f32_to_bytes(W_ref_down, mode=mode)

    # Q2_K decode back to get W_low
    W_low_up = dequantize_q2k_bytes_to_f32(q2k_up, D_FFN_UP, D_HIDDEN, mode=mode)
    W_low_gate = dequantize_q2k_bytes_to_f32(q2k_gate, D_FFN_UP, D_HIDDEN, mode=mode)
    W_low_down = dequantize_q2k_bytes_to_f32(q2k_down, D_FFN_DOWN, D_FFN_UP, mode=mode)

    # Residuals per family
    R_up = W_ref_up - W_low_up
    R_gate = W_ref_gate - W_low_gate
    R_down = W_ref_down - W_low_down

    # SDIR encode/decode (verify roundtrip)
    sdir_up = encode_sdir(R_up, k_pct=K_PCT)
    sdir_gate = encode_sdir(R_gate, k_pct=K_PCT)
    sdir_down = encode_sdir(R_down, k_pct=K_PCT)

    # MLP forward (low)
    Y_low = mlp_full(X, W_low_gate, W_low_up, W_low_down)
    Y_ref = mlp_full(X, W_ref_gate, W_ref_up, W_ref_down)

    # Streaming residual: apply via MLP formula (residual to weights, not activations)
    # dec_* = decode_sdir(encode_sdir(R_*, k_pct)) to match streaming path
    dec_up = decode_sdir(sdir_up)
    dec_gate = decode_sdir(sdir_gate)
    dec_down = decode_sdir(sdir_down)
    Y_sub_simple = mlp_full(X,
                            W_low_gate + dec_gate,
                            W_low_up   + dec_up,
                            W_low_down + dec_down)

    # Metrics
    cos_low = cosine(Y_ref.ravel(), Y_low.ravel())
    cos_sub = cosine(Y_ref.ravel(), Y_sub_simple.ravel())

    MAE_low = float(np.abs(Y_ref - Y_low).mean())
    MAE_sub = float(np.abs(Y_ref - Y_sub_simple).mean())

    delta_cos = cos_sub - cos_low
    MAE_delta = MAE_sub - MAE_low

    severe = delta_cos < -0.05

    # Memory
    q2k_bytes_up = len(q2k_up)
    q2k_bytes_gate = len(q2k_gate)
    q2k_bytes_down = len(q2k_down)
    sdir_bytes_up = len(sdir_up)
    sdir_bytes_gate = len(sdir_gate)
    sdir_bytes_down = len(sdir_down)

    per_layer_bytes = (q2k_bytes_up + sdir_bytes_up +
                       q2k_bytes_gate + sdir_bytes_gate +
                       q2k_bytes_down + sdir_bytes_down)
    per_layer_margin = 3 * Q4_BUDGET_FAMILY - per_layer_bytes

    return {
        "layer": layer,
        "seed": seed,
        "mode": mode,
        "cos_low": cos_low,
        "cos_sub": cos_sub,
        "delta_cos": delta_cos,
        "MAE_low": MAE_low,
        "MAE_sub": MAE_sub,
        "MAE_delta": MAE_delta,
        "severe": severe,
        "q2k_bytes_up": q2k_bytes_up,
        "q2k_bytes_gate": q2k_bytes_gate,
        "q2k_bytes_down": q2k_bytes_down,
        "sdir_bytes_up": sdir_bytes_up,
        "sdir_bytes_gate": sdir_bytes_gate,
        "sdir_bytes_down": sdir_bytes_down,
        "per_layer_bytes": per_layer_bytes,
        "per_layer_margin": per_layer_margin,
        "q2k_total": q2k_bytes_up + q2k_bytes_gate + q2k_bytes_down,
        "sdir_total": sdir_bytes_up + sdir_bytes_gate + sdir_bytes_down,
    }


# ─── Small aggregate run ───────────────────────────────────────────────────────
def run_small_aggregate(layers, seeds, mode, tmpdir, GGUF_PATH):
    """Run corrected_ceil_per_row for multiple layers × seeds."""
    results = []
    for layer in layers:
        for seed in seeds:
            r = run_anchor(layer, seed, mode, tmpdir, GGUF_PATH)
            results.append(r)
    return results


def aggregate_summary(results):
    n = len(results)
    delta_cos_vals = [r["delta_cos"] for r in results]
    mae_delta_vals = [r["MAE_delta"] for r in results]
    margin_vals = [r["per_layer_margin"] for r in results]
    severe_vals = [r["severe"] for r in results]

    n_cos_pos = sum(1 for d in delta_cos_vals if d >= 0)
    n_cos_neg = sum(1 for d in delta_cos_vals if d < 0)
    n_mae_imp = sum(1 for m in mae_delta_vals if m < 0)  # lower MAE is improvement
    n_mem_pos = sum(1 for m in margin_vals if m >= 0)
    n_severe = sum(1 for s in severe_vals if s)

    worst_pair = min(results, key=lambda r: r["delta_cos"])

    return {
        "total_pairs": n,
        "n_cosine_positive": n_cos_pos,
        "n_cosine_negative": n_cos_neg,
        "n_MAE_improving": n_mae_imp,
        "n_MAE_worsening": n - n_mae_imp,
        "n_memory_positive": n_mem_pos,
        "n_memory_negative": n - n_mem_pos,
        "n_severe_regressions": n_severe,
        "worst_pair": {
            "layer": worst_pair["layer"],
            "seed": worst_pair["seed"],
            "delta_cos": worst_pair["delta_cos"],
            "MAE_delta": worst_pair["MAE_delta"],
            "per_layer_margin": worst_pair["per_layer_margin"],
        },
        "mean_delta_cos": float(np.mean(delta_cos_vals)),
        "median_delta_cos": float(np.median(delta_cos_vals)),
        "mean_MAE_improvement": float(np.mean(mae_delta_vals)),
        "worst_margin": min(margin_vals),
        "total_margin": sum(margin_vals),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    GGUF_PATH = os.environ.get("SDI_GGUF_MODEL_PATH")
    if not GGUF_PATH or not os.path.isfile(GGUF_PATH):
        print("ERROR: Set SDI_GGUF_MODEL_PATH to a GGUF file")
        sys.exit(1)

    LIB_PATH = os.environ.get("SDI_LLAMA_CPP_LIB")
    if not LIB_PATH or not os.path.isfile(LIB_PATH):
        print("ERROR: Set SDI_LLAMA_CPP_LIB to libggml-base.so")
        sys.exit(1)

    print(f"GGUF: {GGUF_PATH}")
    print(f"lib:  {LIB_PATH}")
    print(f"Q2_K backend available: {q2k_is_available()}")
    print()

    if not q2k_is_available():
        print("ERROR: Q2_K backend not available")
        sys.exit(1)

    tmpdir = tempfile.mkdtemp(prefix="phase31bj_")
    print(f"Temp dir: {tmpdir}")
    print()

    # ── Anchor table ─────────────────────────────────────────────────────────
    print("=" * 80)
    print("ANCHOR REPRODUCTION TABLE")
    print("=" * 80)
    print()
    print(f"{'Layer':<6} {'Seed':<5} {'Mode':<30} {'cos_low':>9} {'delta_cos':>11} {'MAE_delta':>11} {'severe':>7} {'margin':>12}")
    print("-" * 100)

    anchor_results = {}
    anchor_layers = [21, 21, 2]
    anchor_seeds  = [9, 0, 7]
    modes         = ["historical_floor_flat", "corrected_ceil_per_row"]

    for layer, seed in zip(anchor_layers, anchor_seeds):
        key = f"L{layer}_S{seed}"
        anchor_results[key] = {}
        for mode in modes:
            r = run_anchor(layer, seed, mode, tmpdir, GGUF_PATH)
            anchor_results[key][mode] = r
            hist = HISTORICAL.get((layer, seed), {})
            print(
                f"L{layer}   S{seed}   {mode:<30} "
                f"{r['cos_low']:>9.6f} "
                f"{r['delta_cos']:>+11.6f} "
                f"{r['MAE_delta']:>+11.6f} "
                f"{str(r['severe']):>7} "
                f"{r['per_layer_margin']:>+12,}"
            )
        print()

    # ── Small aggregate (L2 + L21, seeds 0-15) ─────────────────────────────
    print("=" * 80)
    print("SMALL AGGREGATE: corrected_ceil_per_row, k=1%, L2 + L21, seeds 0-15")
    print("=" * 80)
    print()

    agg_corr = run_small_aggregate([2, 21], list(range(16)), "corrected_ceil_per_row", tmpdir, GGUF_PATH)
    agg_hist = run_small_aggregate([2, 21], list(range(16)), "historical_floor_flat", tmpdir, GGUF_PATH)

    sum_corr = aggregate_summary(agg_corr)
    sum_hist = aggregate_summary(agg_hist)

    print(f"{'Mode':<30} {'pairs':>6} {'cos_pos':>8} {'cos_neg':>8} {'mae_imp':>8} {'mem_pos':>8} {'severe':>7}")
    print("-" * 80)
    print(
        f"{'corrected_ceil_per_row':<30} "
        f"{sum_corr['total_pairs']:>6} "
        f"{sum_corr['n_cosine_positive']:>8} "
        f"{sum_corr['n_cosine_negative']:>8} "
        f"{sum_corr['n_MAE_improving']:>8} "
        f"{sum_corr['n_memory_positive']:>8} "
        f"{sum_corr['n_severe_regressions']:>7}"
    )
    print(
        f"{'historical_floor_flat':<30} "
        f"{sum_hist['total_pairs']:>6} "
        f"{sum_hist['n_cosine_positive']:>8} "
        f"{sum_hist['n_cosine_negative']:>8} "
        f"{sum_hist['n_MAE_improving']:>8} "
        f"{sum_hist['n_memory_positive']:>8} "
        f"{sum_hist['n_severe_regressions']:>7}"
    )
    print()
    print(f"corrected: mean_delta_cos={sum_corr['mean_delta_cos']:+.6f}, median={sum_corr['median_delta_cos']:+.6f}, "
          f"mean_MAE_imp={sum_corr['mean_MAE_improvement']:+.6f}, worst_margin={sum_corr['worst_margin']:,}")
    print(f"historical: mean_delta_cos={sum_hist['mean_delta_cos']:+.6f}, median={sum_hist['median_delta_cos']:+.6f}, "
          f"mean_MAE_imp={sum_hist['mean_MAE_improvement']:+.6f}, worst_margin={sum_hist['worst_margin']:,}")
    print()
    print(f"Worst corrected pair: L{sum_corr['worst_pair']['layer']}_S{sum_corr['worst_pair']['seed']}, "
          f"delta_cos={sum_corr['worst_pair']['delta_cos']:+.6f}, margin={sum_corr['worst_pair']['per_layer_margin']:,}")
    print(f"Worst historical pair: L{sum_hist['worst_pair']['layer']}_S{sum_hist['worst_pair']['seed']}, "
          f"delta_cos={sum_hist['worst_pair']['delta_cos']:+.6f}, margin={sum_hist['worst_pair']['per_layer_margin']:,}")

    # ── Memory accounting ────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("MEMORY ACCOUNTING (corrected_ceil_per_row)")
    print("=" * 80)
    print()
    r_ex = agg_corr[0]  # use L2-S0 as example
    print(f"Per-family Q2_K bytes: up={r_ex['q2k_bytes_up']:,}, gate={r_ex['q2k_bytes_gate']:,}, down={r_ex['q2k_bytes_down']:,}")
    print(f"Per-family SDIR bytes: up={r_ex['sdir_bytes_up']:,}, gate={r_ex['sdir_bytes_gate']:,}, down={r_ex['sdir_bytes_down']:,}")
    print(f"Per-layer Q2_K total: {r_ex['q2k_total']:,} bytes ({r_ex['q2k_total']/1024:.1f} KB)")
    print(f"Per-layer SDIR total: {r_ex['sdir_total']:,} bytes ({r_ex['sdir_total']/1024:.1f} KB)")
    print(f"Per-layer all-family total: {r_ex['per_layer_bytes']:,} bytes ({r_ex['per_layer_bytes']/1024:.1f} KB)")
    print(f"3 × Q4_BUDGET_FAMILY = {3 * Q4_BUDGET_FAMILY:,} ({3 * Q4_BUDGET_FAMILY/1024:.1f} KB)")
    print(f"Per-layer margin (corrected): {r_ex['per_layer_margin']:,} ({r_ex['per_layer_margin']/1024:.1f} KB)")
    print()
    print(f"Q4_budget per family: {Q4_BUDGET_FAMILY:,} ({Q4_BUDGET_FAMILY/1024:.1f} KB)")
    print(f"Corrected Q2_K + SDIR per layer: {r_ex['per_layer_bytes']:,} ({r_ex['per_layer_bytes']/1024:.1f} KB)")
    print(f"Memory-positive: {'YES' if r_ex['per_layer_margin'] >= 0 else 'NO'}")

    # ── Classification decision ───────────────────────────────────────────────
    print()
    print("=" * 80)
    print("CLASSIFICATION DECISION")
    print("=" * 80)

    # Check corrected mode status
    corr_mem_pos = sum_corr["n_memory_positive"]
    corr_cos_pos = sum_corr["n_cosine_positive"]
    corr_severe = sum_corr["n_severe_regressions"]

    if corr_mem_pos == 0:
        classification = "PARTIAL_31BJ_CORRECTED_Q2K_MEMORY_FAIL"
        reason = "corrected mode loses memory positivity (0/32 memory-positive)"
    elif corr_severe == 0 and corr_cos_pos >= 30:
        classification = "PASS_31BJ_CORRECTED_Q2K_SMALL_AGGREGATE_STRONG"
        reason = f"corrected mode: {corr_cos_pos}/32 cos-positive, {corr_severe} severe, {corr_mem_pos}/32 mem-positive"
    else:
        classification = "PARTIAL_31BJ_CORRECTED_Q2K_METRICS_SHIFT"
        reason = f"corrected mode: {corr_cos_pos}/32 cos-positive, {corr_severe} severe, {corr_mem_pos}/32 mem-positive"

    print(f"Classification: {classification}")
    print(f"Reason: {reason}")

    # ── Save results ──────────────────────────────────────────────────────────
    out = {
        "classification": classification,
        "reason": reason,
        "k_pct": K_PCT,
        "alpha": ALPHA,
        "anchor_results": anchor_results,
        "small_aggregate": {
            "corrected_ceil_per_row": sum_corr,
            "historical_floor_flat": sum_hist,
        },
        "memory_accounting": {
            "q4_budget_per_family": Q4_BUDGET_FAMILY,
            "per_layer_total_bytes": r_ex["per_layer_bytes"],
            "per_layer_margin": r_ex["per_layer_margin"],
            "memory_positive": r_ex["per_layer_margin"] >= 0,
            "q2k_total_per_layer": r_ex["q2k_total"],
            "sdir_total_per_layer": r_ex["sdir_total"],
        },
        "pairs_tested": {
            "layers": [2, 21],
            "seeds": list(range(16)),
            "total": 32,
        },
    }

    out_path = os.path.join(REPO_DIR, "src", "results", "PHASE31BJ_CORRECTED_Q2K_REBASELINE.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nResults saved: {out_path}")

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"\nPhase 31BJ complete.")


if __name__ == "__main__":
    main()
