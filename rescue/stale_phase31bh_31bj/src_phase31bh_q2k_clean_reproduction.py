#!/usr/bin/env python3
"""
Phase 31BH-R2 — Q2_K Anchor Reproduction Fix / Floor-vs-Ceil Decision

Fixes:
1. X RNG: use np.random.default_rng(seed) matching OLD (31AY/31BA) path
2. Q2_K modes: support both "historical_floor_flat" and "corrected_ceil_per_row"

Environment variables required:
    SDI_GGUF_MODEL_PATH   — Qwen2.5 GGUF file
    SDI_LLAMA_CPP_ROOT    — llama.cpp source tree (for gguf-py)
    SDI_LLAMA_CPP_LIB     — path to libggml-base.so

Run:
    SDI_GGUF_MODEL_PATH=/path/to/gguf \
    SDI_LLAMA_CPP_ROOT=/path/to/llama.cpp \
    SDI_LLAMA_CPP_LIB=/path/to/libggml-base.so \
        .venv/bin/python src/phase31bh_q2k_clean_reproduction.py
"""

import hashlib
import json
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

# ─── Env resolution ────────────────────────────────────────────────────────────
GGUF_PATH = os.environ.get("SDI_GGUF_MODEL_PATH")
LLAMA_CPP_ROOT = os.environ.get("SDI_LLAMA_CPP_ROOT", "")
GGUF_PY_PATH = os.path.join(LLAMA_CPP_ROOT, "gguf-py") if LLAMA_CPP_ROOT else ""
LIB_PATH = os.environ.get("SDI_LLAMA_CPP_LIB", "")

MISSING_ENV = []
if not GGUF_PATH:
    MISSING_ENV.append("SDI_GGUF_MODEL_PATH")
if not LIB_PATH:
    if not LLAMA_CPP_ROOT:
        MISSING_ENV.append("SDI_LLAMA_CPP_ROOT or SDI_LLAMA_CPP_LIB")
    elif not os.path.isfile(os.path.join(LLAMA_CPP_ROOT, "build", "bin", "libggml-base.so")):
        MISSING_ENV.append("SDI_LLAMA_CPP_LIB (auto-resolve failed)")

if MISSING_ENV:
    print("ERROR: Required environment variables not set: " + ", ".join(MISSING_ENV))
    print("Hint: export SDI_GGUF_MODEL_PATH=/media/matthew-villnave/VL_usb/models/qwen2.5-0.5b-official/qwen2.5-0.5b-instruct-q4_k_m.gguf")
    print("       export SDI_LLAMA_CPP_ROOT=/home/matthew-villnave/llama.cpp")
    sys.exit(1)

if not os.path.isfile(GGUF_PATH):
    print(f"ERROR: GGUF not found: {GGUF_PATH}")
    sys.exit(1)

if not os.path.isfile(LIB_PATH):
    auto = os.path.join(LLAMA_CPP_ROOT, "build", "bin", "libggml-base.so")
    if os.path.isfile(auto):
        LIB_PATH = auto
    else:
        print(f"ERROR: llama.cpp library not found at {LIB_PATH} (auto: {auto})")
        sys.exit(1)

if os.path.isdir(GGUF_PY_PATH):
    sys.path.insert(0, GGUF_PY_PATH)

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "src"))

from bundle_manifest import ManifestLoader, sha256_bytes
from phase31x_manifest_runtime import (
    cosine, encode_sdir, sdir_streaming_apply, decode_sdir,
)
from gguf import GGUFReader
from gguf.quants import dequantize
from q2k_backend import (
    quantize_q2k_f32_to_bytes, dequantize_q2k_bytes_to_f32,
    q2k_expected_nbytes, lib as q2k_lib, is_available as q2k_is_available,
    QK_K, Q2_BLOCK_BYTES,
)

# ─── Accepted anchor data (from committed Phase 31AY results) ─────────────────
# These were produced with OLD path: default_rng(seed), floor-flat Q2_K
ACCEPTED = {
    "L21_S9": {
        "seed": 9,
        "layer": 21,
        "family": "ffn_up",
        "cos_low": 0.79491341,
        "cos_sub_expected": 0.64885426,
        "delta_cos_expected": -0.14605916,
        "MAE_delta_expected": -0.00047889,
        "memory_positive": True,
        "severe_expected": True,
        "margin": 351192,
        "k_pct": 1.0,
        "d_out": 4864,
        "d_in": 896,
    },
    "L21_S0": {
        "seed": 0,
        "layer": 21,
        "family": "ffn_up",
        "cos_low": 0.85561174,
        "cos_sub_expected": 0.87345678,
        "delta_cos_expected": 0.01784503,
        "MAE_delta_expected": -0.00316946,
        "memory_positive": True,
        "severe_expected": False,
        "margin": 351192,
        "k_pct": 1.0,
        "d_out": 4864,
        "d_in": 896,
    },
}

# ─── Helpers ───────────────────────────────────────────────────────────────────

def load_w_ref_from_gguf(gguf_path, layer, family):
    """Load W_ref (float32) from GGUF tensor."""
    reader = GGUFReader(gguf_path)
    tensor_name = f"blk.{layer}.{family}.weight"
    t = next((x for x in reader.tensors if x.name == tensor_name))
    return dequantize(t.data, t.tensor_type).astype(np.float32)


def x_fingerprint(X):
    """Compute fingerprint dict for an activation vector X."""
    flat = X.astype(np.float32).ravel()
    return {
        "shape": list(X.shape),
        "dtype": str(X.dtype),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "norm": float(np.linalg.norm(flat)),
        "first_8": [float(v) for v in flat[:8]],
        "sha256": hashlib.sha256(flat.tobytes()).hexdigest()[:32],
    }


def w_low_fingerprint(W_low):
    """Compute fingerprint dict for a W_low tensor."""
    flat = W_low.astype(np.float32).ravel()
    return {
        "shape": list(W_low.shape),
        "dtype": str(W_low.dtype),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "norm": float(np.linalg.norm(flat)),
        "first_8": [float(v) for v in flat[:8]],
        "sha256": hashlib.sha256(flat.tobytes()).hexdigest()[:32],
    }


def build_q2k_bundle(bundle_dir, family, d_out, d_in, W_ref, k_pct, layer, seed, q2k_mode):
    """
    Build a schema-v1.0 manifest + Q2_K W_low + sdir residual.

    W_ref is the FP32 reference (from GGUF dequantization).
    W_low is W_ref quantized to Q2_K via llama.cpp.
    Residual R = W_ref - W_low is encoded as sdir.

    q2k_mode: "historical_floor_flat" or "corrected_ceil_per_row"
    """
    os.makedirs(os.path.join(bundle_dir, "tensors"), exist_ok=True)

    # Quantize W_ref → Q2_K bytes
    q2k_bytes = quantize_q2k_f32_to_bytes(W_ref, mode=q2k_mode)
    q2k_nbytes = len(q2k_bytes)

    # Dequantize back to get W_low
    W_low = dequantize_q2k_bytes_to_f32(q2k_bytes, d_out, d_in, mode=q2k_mode)

    # Residual
    R = W_ref - W_low
    sdir_bytes = encode_sdir(R, k_pct=k_pct)

    # Write W_low as raw Q2_K binary
    wlow_path = os.path.join(bundle_dir, "tensors", f"blk.{layer}.{family}.q2_k.W_low")
    with open(wlow_path, "wb") as f:
        f.write(q2k_bytes)

    # Write residual
    sdir_path = os.path.join(bundle_dir, "tensors", f"blk.{layer}.{family}.residual.sdir")
    with open(sdir_path, "wb") as f:
        f.write(sdir_bytes)

    total_sub_bytes = q2k_nbytes + len(sdir_bytes)
    entry = {
        "tensor_name": f"blk.{layer}.{family}.weight",
        "layer": layer,
        "family": family,
        "shape": [d_out, d_in],
        "orientation": "canonical_d_out_d_in",
        "W_ref_bytes": d_out * d_in * 4,
        "W_ref_Q4_budget_bytes": 2_179_072 + 351_192,
        "W_low_packed_bytes": q2k_nbytes,
        "W_low_scale_bytes": 0,
        "residual_bytes": len(sdir_bytes),
        "total_substitutive_bytes": total_sub_bytes,
        "memory_margin_bytes": 351_192,
        "formats": {
            "W_low_format": "q2_k",
            "residual_format": "sdir_v1",
            "k_percent": k_pct,
            "value_dtype": "fp16",
            "mask_encoding": "dense_bitmap",
            "scale_policy": "block32_fp16",
            "q2_k_backend_mode": q2k_mode,
        },
        "paths": {
            "W_low_path": f"tensors/blk.{layer}.{family}.q2_k.W_low",
            "sdir_path": f"tensors/blk.{layer}.{family}.residual.sdir",
        },
        "checksums": {
            "W_low": sha256_bytes(q2k_bytes),
            "residual": sha256_bytes(sdir_bytes),
        },
    }
    return entry, W_low


def build_manifest(bundle_dir, entries, layer, seed, k_pct, q2k_mode):
    manifest = {
        "schema_version": "1.0",
        "package_id": f"phase31bh-q2k-L{layer}-S{seed}-{q2k_mode}",
        "bundle_type": "ffn_up_q2k_substitutive",
        "source_model": {
            "model_name": "qwen2.5-0.5b",
            "architecture": "ffn",
            "quantization": "Q4_K_M_reference",
        },
        "substitution_policy": {
            "k_percent": k_pct,
            "alpha": 1.0,
            "W_low_format": "q2_k",
            "residual_encoding": "sdir_v1",
            "residual_policy": "always_on",
            "scale_policy": "block32_fp16",
            "q2_k_backend_mode": q2k_mode,
        },
        "layers": entries,
    }
    manifest_path = os.path.join(bundle_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_q2k_weights(gguf_path, layer):
    """
    Load Q2_K dequantized weights for all 3 MLP families.
    OLD path (31AY/31BA): W_ref = dequantize(Q2_K(gguf_weights)) — NOT raw GGUF weights.
    This is because the "reference" in OLD runs was itself a Q2_K artifact.
    """
    reader = GGUFReader(gguf_path)
    result = {}
    for fam in ['ffn_up', 'ffn_gate', 'ffn_down']:
        t = next(x for x in reader.tensors if x.name == f'blk.{layer}.{fam}.weight')
        W_raw = dequantize(t.data, t.tensor_type).astype(np.float32)
        buf = quantize_q2k_f32_to_bytes(W_raw, mode="historical_floor_flat")
        W_q2k = dequantize_q2k_bytes_to_f32(buf, W_raw.shape[0], W_raw.shape[1],
                                            mode="historical_floor_flat")
        result[fam] = {'W_ref': W_q2k, 'W_raw': W_raw}
    return result


def run_anchor(anchor_key, anchor, bundle_dir, X, q2k_mode):
    """Reproduce one anchor using a specific Q2_K mode."""
    layer = anchor["layer"]
    family = anchor["family"]
    seed = anchor["seed"]
    k_pct = anchor["k_pct"]
    d_out = anchor["d_out"]
    d_in = anchor["d_in"]

    # Load raw GGUF weights for all 3 families
    reader = GGUFReader(GGUF_PATH)
    W_ref_up_raw = dequantize(
        next(x for x in reader.tensors if x.name == f'blk.{layer}.ffn_up.weight').data,
        next(x for x in reader.tensors if x.name == f'blk.{layer}.ffn_up.weight').tensor_type
    ).astype(np.float32)
    W_ref_gate_raw = dequantize(
        next(x for x in reader.tensors if x.name == f'blk.{layer}.ffn_gate.weight').data,
        next(x for x in reader.tensors if x.name == f'blk.{layer}.ffn_gate.weight').tensor_type
    ).astype(np.float32)
    W_ref_down_raw = dequantize(
        next(x for x in reader.tensors if x.name == f'blk.{layer}.ffn_down.weight').data,
        next(x for x in reader.tensors if x.name == f'blk.{layer}.ffn_down.weight').tensor_type
    ).astype(np.float32)

    # Build bundle with Q2_K for ffn_up only (other families not in bundle)
    q2k_bytes = quantize_q2k_f32_to_bytes(W_ref_up_raw, mode=q2k_mode)
    W_low_up = dequantize_q2k_bytes_to_f32(q2k_bytes, d_out, d_in, mode=q2k_mode)

    entry, _ = build_q2k_bundle(bundle_dir, family, d_out, d_in, W_ref_up_raw,
                                  k_pct, layer, seed, q2k_mode)
    manifest = build_manifest(bundle_dir, [entry], layer, seed, k_pct, q2k_mode)

    # Validate schema
    try:
        loader = ManifestLoader(bundle_dir)
        loader.load()
        validation = loader.validate_bundle(bundle_dir)
        schema_ok = validation["error_count"] == 0
        validation_error = None
    except Exception as exc:
        schema_ok = False
        validation_error = str(exc)

    if not schema_ok:
        return {
            "anchor": anchor_key,
            "layer": layer,
            "seed": seed,
            "family": family,
            "q2k_mode": q2k_mode,
            "status": "SCHEMA_VALIDATION_FAILED",
            "error": validation_error,
            "passed": False,
        }

    # MLP evaluation: Y = silu(X @ W_gate.T) * (X @ W_up.T) @ W_down.T
    # OLD path (31AY/31BA):
    #   Reference: RAW weights (W_ref = raw GGUF dequantized)
    #   W_low: Q2_K roundtrip for ALL 3 families (ffn_up, ffn_gate, ffn_down)
    #   Residual: W_ref - W_low for each family, SDIR-encoded
    #   Y_sub = mlp(W_low + dec_residual) for ALL families
    X_2d = X.reshape(1, -1) if X.ndim == 1 else X

    def silu(x):
        return x / (1.0 + np.exp(-np.clip(x, -709, 709)))

    def mlp_full(X, Wg, Wu, Wd):
        return (silu(X @ Wg.T) * (X @ Wu.T)) @ Wd.T

    # W_ref: RAW weights (gguf dequantized, no Q2_K roundtrip)
    # W_low_q2k: Q2_K roundtrip weights for ALL families using specified mode
    # (ffn_up from bundle artifact, ffn_gate and ffn_down computed here)
    W_ref_up = W_ref_up_raw
    W_ref_gate = W_ref_gate_raw
    W_ref_down = W_ref_down_raw

    # For gate and down, we need Q2_K roundtrip in the same mode
    # (the bundle only stores ffn_up; gate/down are always Q2_K artifacts)
    q2k_bytes_gate = quantize_q2k_f32_to_bytes(W_ref_gate, mode=q2k_mode)
    W_low_gate = dequantize_q2k_bytes_to_f32(q2k_bytes_gate, W_ref_gate.shape[0], W_ref_gate.shape[1], mode=q2k_mode)
    q2k_bytes_down = quantize_q2k_f32_to_bytes(W_ref_down, mode=q2k_mode)
    W_low_down = dequantize_q2k_bytes_to_f32(q2k_bytes_down, W_ref_down.shape[0], W_ref_down.shape[1], mode=q2k_mode)

    # Y_ref: RAW reference MLP
    Y_ref = mlp_full(X_2d, W_ref_gate, W_ref_up, W_ref_down)

    # Y_low: Q2_K MLP (all families Q2_K roundtrip)
    Y_low = mlp_full(X_2d, W_low_gate, W_low_up, W_low_down)

    # Y_sub: apply SDIR residual to ALL families (OLD formula from 31AY/31BA)
    # Residual = W_ref - W_low for each family
    R_up = W_ref_up - W_low_up
    R_gate = W_ref_gate - W_low_gate
    R_down = W_ref_down - W_low_down
    dec_up = decode_sdir(encode_sdir(R_up, k_pct))
    dec_gate = decode_sdir(encode_sdir(R_gate, k_pct))
    dec_down = decode_sdir(encode_sdir(R_down, k_pct))
    Y_sub = mlp_full(X_2d,
                     W_low_gate + dec_gate,
                     W_low_up   + dec_up,
                     W_low_down + dec_down)

    # Metrics — flatten for cosine (works for both 1D and 2D)
    cos_low = cosine(Y_ref.ravel(), Y_low.ravel())
    cos_sub = cosine(Y_ref.ravel(), Y_sub.ravel())
    mae_low = float(np.abs(Y_ref - Y_low).mean())
    mae_sub = float(np.abs(Y_ref - Y_sub).mean())
    delta_cos = float(cos_sub - cos_low)
    mae_delta = float(mae_sub - mae_low)

    # Tolerances
    cos_tol = 1e-3
    mae_tol = 1e-3
    delta_cos_tol = 5e-4
    mae_delta_tol = 5e-4

    cos_low_match = abs(cos_low - anchor["cos_low"]) < cos_tol
    delta_cos_match = abs(delta_cos - anchor["delta_cos_expected"]) < delta_cos_tol
    mae_delta_match = abs(mae_delta - anchor["MAE_delta_expected"]) < mae_delta_tol
    severe_match = (delta_cos < -0.05) == anchor["severe_expected"]

    passed = cos_low_match and delta_cos_match and mae_delta_match and severe_match

    # Schema version check
    with open(os.path.join(bundle_dir, "manifest.json")) as f:
        mfest = json.load(f)
    q2k_format = mfest["layers"][0]["formats"]["W_low_format"]
    schema_ok = mfest["schema_version"] in ("1.0", "0.2.0")

    # Q2_K size check
    q2k_path = os.path.join(bundle_dir, entry["paths"]["W_low_path"])
    actual_q2k_bytes = os.path.getsize(q2k_path)
    if q2k_mode == "historical_floor_flat":
        expected_q2k_bytes = (d_out * d_in) // QK_K * Q2_BLOCK_BYTES
    else:
        expected_q2k_bytes = q2k_expected_nbytes(d_out, d_in) * d_out
    size_ok = actual_q2k_bytes == expected_q2k_bytes

    return {
        "anchor": anchor_key,
        "layer": layer,
        "seed": seed,
        "family": family,
        "q2k_mode": q2k_mode,
        "W_low_format": q2k_format,
        "Q2_K_bytes": actual_q2k_bytes,
        "Q2_K_bytes_expected": expected_q2k_bytes,
        "Q2_K_bytes_match": size_ok,
        "W_low_fingerprint": w_low_fingerprint(W_low_up),
        "X_fingerprint": None,   # filled by caller
        "cos_low": cos_low,
        "cos_low_expected": anchor["cos_low"],
        "cos_low_match": cos_low_match,
        "cos_sub": cos_sub,
        "cos_sub_expected": anchor.get("cos_sub_expected"),
        "delta_cos": delta_cos,
        "delta_cos_expected": anchor["delta_cos_expected"],
        "delta_cos_match": delta_cos_match,
        "MAE_low": mae_low,
        "MAE_sub": mae_sub,
        "MAE_delta": mae_delta,
        "MAE_delta_expected": anchor["MAE_delta_expected"],
        "MAE_delta_match": mae_delta_match,
        "severe_expected": anchor["severe_expected"],
        "severe_observed": delta_cos < -0.05,
        "severe_match": severe_match,
        "schema_ok": schema_ok,
        "q2k_size_ok": size_ok,
        "passed": passed,
        "status": "REPRODUCED" if passed else "PARTIAL",
    }


def main():
    print(f"GGUF: {GGUF_PATH}")
    print(f"lib:  {LIB_PATH}")
    print(f"Q2_K backend available: {q2k_is_available()}")
    print()

    if not q2k_is_available():
        print("ERROR: Q2_K backend not available")
        sys.exit(1)

    tmpdir = tempfile.mkdtemp(prefix="phase31bh_r2_")
    print(f"Temp bundle dir: {tmpdir}")
    print()

    # ── X vector fingerprints (using FIXED default_rng) ──────────────────────
    print("=== X VECTOR FINGERPRINTS (FIXED: np.random.default_rng) ===\n")
    HIDDEN = 896
    x_fprints = {}
    for seed in [0, 9]:
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((1, HIDDEN)).astype(np.float32)
        x_fprints[seed] = x_fingerprint(X)
        print(f"  seed={seed}: shape={x_fprints[seed]['shape']}, dtype={x_fprints[seed]['dtype']}")
        print(f"    mean={x_fprints[seed]['mean']:+.6f}, std={x_fprints[seed]['std']:.6f}")
        print(f"    norm={x_fprints[seed]['norm']:.6f}")
        print(f"    first_8: {[round(v,6) for v in x_fprints[seed]['first_8']]}")
        print(f"    SHA256: {x_fprints[seed]['sha256']}")
        print()

    # ── Run both modes for both anchors ───────────────────────────────────────
    Q2K_MODES = ["historical_floor_flat", "corrected_ceil_per_row"]
    results = {}

    print("=== ANCHOR REPRODUCTION TABLE ===\n")

    for anchor_key, anchor in ACCEPTED.items():
        results[anchor_key] = {}
        seed = anchor["seed"]

        # Fixed X (default_rng, shape (1, 896))
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((1, HIDDEN)).astype(np.float32)

        for mode in Q2K_MODES:
            subdir = os.path.join(tmpdir, f"bundle_{anchor_key}_{mode}")
            os.makedirs(subdir, exist_ok=True)

            result = run_anchor(anchor_key, anchor, subdir, X, mode)
            result["X_fingerprint"] = x_fprints[seed]
            results[anchor_key][mode] = result

            cos = result["cos_low"]
            dc = result["delta_cos"]
            md = result["MAE_delta"]
            exp_cos = result["cos_low_expected"]
            exp_dc = result["delta_cos_expected"]
            exp_md = result["MAE_delta_expected"]
            passed = result["passed"]

            print(f"[{anchor_key}] {mode}")
            print(f"  Q2_K bytes:   {result['Q2_K_bytes']:,} (expected match={result['Q2_K_bytes_match']})")
            print(f"  cos_low:      {cos:.6f}  expected={exp_cos:.6f}  match={result['cos_low_match']}")
            print(f"  cos_sub:      {result['cos_sub']:.6f}  expected={result['cos_sub_expected']:.6f}")
            print(f"  delta_cos:    {dc:+.6f}  expected={exp_dc:+.6f}  match={result['delta_cos_match']}")
            print(f"  MAE_delta:    {md:+.6f}  expected={exp_md:+.6f}  match={result['MAE_delta_match']}")
            print(f"  severe:       observed={result['severe_observed']}  expected={result['severe_expected']}  match={result['severe_match']}")
            print(f"  W_low SHA256: {result['W_low_fingerprint']['sha256']}")
            print(f"  status:       {result['status']}  passed={passed}")
            print()

    # ── Summary table ─────────────────────────────────────────────────────────
    print("=== SUMMARY TABLE ===\n")
    print(f"{'Anchor':<12} {'Mode':<28} {'cos_low':>10} {'delta_cos':>12} {'MAE_delta':>12} {'Passed':>7}")
    print("-" * 85)
    for anchor_key in ACCEPTED:
        for mode in Q2K_MODES:
            r = results[anchor_key][mode]
            print(
                f"{anchor_key:<12} {mode:<28} "
                f"{r['cos_low']:>10.6f} "
                f"{r['delta_cos']:>+12.6f} "
                f"{r['MAE_delta']:>+12.6f} "
                f"{str(r['passed']):>7}"
            )
    print()

    # ── Classification ─────────────────────────────────────────────────────────
    # Check which modes reproduce historical anchors
    hist_repro = {
        key: results[key]["historical_floor_flat"]["passed"]
        for key in results
    }
    corr_repro = {
        key: results[key]["corrected_ceil_per_row"]["passed"]
        for key in results
    }

    all_hist = all(hist_repro.values())
    all_corr = all(corr_repro.values())
    none_hist = not any(hist_repro.values())
    none_corr = not any(corr_repro.values())

    if all_hist and not all_corr:
        classification = "PASS_31BH_R2_HISTORICAL_ANCHORS_REPRODUCED"
    elif all_corr and not all_hist:
        classification = "PASS_31BH_R2_CORRECTED_Q2K_ANCHORS_REPRODUCED"
    elif all_hist and all_corr:
        classification = "PASS_31BH_R2_BOTH_MODES_REPRODUCE"
    elif all_hist and not all_corr:
        classification = "PARTIAL_31BH_R2_HISTORICAL_REPRO_CORRECTED_DIFFERS"
    elif none_hist and none_corr:
        classification = "BLOCKED_31BH_R2_ANCHORS_STILL_MISMATCH"
    else:
        # Some anchors pass, some don't — partial
        classification = "PARTIAL_31BH_R2_MIXED_REPRODUCTION"

    print(f"Classification: {classification}")
    print(f"  historical_floor_flat: L21-S0={hist_repro['L21_S0']}, L21-S9={hist_repro['L21_S9']}")
    print(f"  corrected_ceil_per_row: L21-S0={corr_repro['L21_S0']}, L21-S9={corr_repro['L21_S9']}")

    # ── W_low byte comparison ───────────────────────────────────────────────────
    print("\n=== W_LOW BYTE COMPARISON ===\n")
    for anchor_key in ACCEPTED:
        r_hist = results[anchor_key]["historical_floor_flat"]
        r_corr = results[anchor_key]["corrected_ceil_per_row"]
        print(f"{anchor_key}:")
        print(f"  historical_floor_flat: {r_hist['Q2_K_bytes']:,} bytes, SHA256={r_hist['W_low_fingerprint']['sha256']}")
        print(f"  corrected_ceil_per_row: {r_corr['Q2_K_bytes']:,} bytes, SHA256={r_corr['W_low_fingerprint']['sha256']}")
        print(f"  difference: {r_corr['Q2_K_bytes'] - r_hist['Q2_K_bytes']:,} bytes ({(r_corr['Q2_K_bytes']/r_hist['Q2_K_bytes']-1)*100:+.1f}%)")
        print()

    # ── Output JSON ────────────────────────────────────────────────────────────
    output = {
        "phase": "31BH-R2",
        "classification": classification,
        "schema_version": "1.0",
        "q2k_backend_modes": Q2K_MODES,
        "x_fingerprints": x_fprints,
        "anchor_results": results,
        "historical_reproduction": hist_repro,
        "corrected_reproduction": corr_repro,
        "gguf_path": os.path.basename(GGUF_PATH),
    }

    out_path = os.path.join(REPO, "results", "PHASE31BH_R2_Q2K_ANCHOR_REPRO_FIX.json")
    os.makedirs(os.path.join(REPO, "results"), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults: {out_path}")
    return 0 if (all_hist or all_corr) else 1


if __name__ == "__main__":
    try:
        ret = main()
    finally:
        # Keep tempdir for inspection if needed; comment out to auto-cleanup:
        pass
        # shutil.rmtree(tmpdir, ignore_errors=True)
    raise SystemExit(ret)
