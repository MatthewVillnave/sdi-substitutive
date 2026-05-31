#!/usr/bin/env python3
"""
Phase 31BG — Clean Reproduction Run Using Hardened Schema

Reproduces accepted anchor metrics for L21-S9 and L21-S0 using the
hardened schema v1.0 infrastructure (bundle_manifest + ManifestRuntime).

Environment variables required:
    SDI_GGUF_MODEL_PATH   — path to Qwen2.5-0.5B GGUF file
    SDI_LLAMA_CPP_ROOT   — path to llama.cpp source tree (for gguf-py)

This script is portable: no hardcoded private paths.
All paths resolved via environment variables with clear error messages.

Run:
    SDI_GGUF_MODEL_PATH=/path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    SDI_LLAMA_CPP_ROOT=/path/to/llama.cpp \
        .venv/bin/python src/phase31bg_clean_reproduction.py
"""

import ctypes
import json
import os
import shutil
import struct
import sys
import tempfile

import numpy as np

# ─── Resolve environment ─────────────────────────────────────────────────────
GGUF_PATH = os.environ.get("SDI_GGUF_MODEL_PATH")
LLAMA_CPP_ROOT = os.environ.get("SDI_LLAMA_CPP_ROOT")
LIB_PATH = os.environ.get("SDI_LLAMA_CPP_BUILD", os.path.join(os.environ.get("SDI_LLAMA_CPP_ROOT", ""), "build", "bin", "libggml-base.so"))

if not GGUF_PATH:
    print("ERROR: SDI_GGUF_MODEL_PATH is not set. Cannot reproduce without model file.")
    print("Hint: export SDI_GGUF_MODEL_PATH=/media/matthew-villnave/VL_usb/models/qwen2.5-0.5b-official/qwen2.5-0.5b-instruct-q4_k_m.gguf")
    sys.exit(1)

if not LLAMA_CPP_ROOT:
    print("ERROR: SDI_LLAMA_CPP_ROOT is not set.")
    print("Hint: export SDI_LLAMA_CPP_ROOT=/home/matthew-villnave/llama.cpp")
    sys.exit(1)

gguf_py_path = os.path.join(LLAMA_CPP_ROOT, "gguf-py")
if not os.path.isdir(gguf_py_path):
    print(f"ERROR: gguf-py not found at {gguf_py_path}")
    sys.exit(1)

if not os.path.isfile(LIB_PATH):
    print(f"ERROR: llama.cpp library not found at {LIB_PATH}")
    sys.exit(1)

sys.path.insert(0, gguf_py_path)

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "src"))

import ctypes
lib = ctypes.CDLL(LIB_PATH)

from bundle_manifest import ManifestLoader, sha256_bytes, write_sdiw
from phase31x_manifest_runtime import (
    ManifestRuntime, cosine, decode_sdir, encode_sdir,
    pack_wlow, sdir_streaming_apply, sdiw_streaming_apply, unpack_wlow,
)
from gguf import GGUFReader
from gguf.quants import dequantize


# ─── Accepted anchor data (from committed results) ───────────────────────────
ACCEPTED = {
    "L21_S9": {
        "seed": 9,
        "layer": 21,
        "family": "ffn_up",
        "cos_low": 0.79491341,
        "cos_sub_expected": 0.64885426,  # WR k=1 makes it WORSE — this is the severe outlier
        "delta_cos_expected": -0.14605916,
        "MAE_delta_expected": -0.00047889,  # MAE still improves despite cosine regression
        "memory_positive": True,
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
        "margin": 351192,
        "k_pct": 1.0,
        "d_out": 4864,
        "d_in": 896,
    },
}

# Qwen2.5-0.5B layer 21 tensor names
LAYER21_FFN_UP_TENSOR = "blk.21.ffn_up.weight"
LAYER21_FFN_DOWN_TENSOR = "blk.21.ffn_down.weight"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_layer_tensors_from_gguf(gguf_path, layer, families):
    """Load ffn_up / ffn_down weight tensors for a given layer from GGUF."""
    reader = GGUFReader(gguf_path)
    tensors = {}
    for family in families:
        tensor_name = f"blk.{layer}.{family}.weight"
        try:
            t = next(x for x in reader.tensors if x.name == tensor_name)
            W_ref = dequantize(t.data, t.tensor_type).astype(np.float32)
            tensors[family] = W_ref
        except (KeyError, StopIteration):
            print(f"WARNING: tensor {tensor_name} not found in GGUF")
    return tensors


def compute_metrics(Y_ref, Y_sub):
    """Compute delta_cos, MAE_delta from reference and substitutive output."""
    cos = cosine(Y_ref, Y_sub)
    mae = np.abs(Y_ref - Y_sub).mean()
    return cos, mae


def build_single_family_bundle(bundle_dir, family, d_out, d_in, W_ref, k_pct, layer, seed):
    """Create a schema-v1.0 manifest + .sdiw + .sdir for one family."""
    os.makedirs(os.path.join(bundle_dir, "tensors"), exist_ok=True)

    # Quantize W_ref → W_low
    packed, scales = pack_wlow(W_ref)
    W_low = unpack_wlow(packed, scales, d_out, d_in)

    # Residual
    R = W_ref - W_low
    sdir_bytes = encode_sdir(R, k_pct=k_pct)
    decoded_R = decode_sdir(sdir_bytes)

    # Write .sdiw
    sdiw_path = os.path.join(bundle_dir, "tensors", f"blk.{layer}.{family}.wlow.sdiw")
    write_sdiw(sdiw_path, d_out, d_in, scales, packed)

    # Write .sdir
    sdir_path = os.path.join(bundle_dir, "tensors", f"blk.{layer}.{family}.residual.sdir")
    with open(sdir_path, "wb") as f:
        f.write(sdir_bytes)

    # Manifest entry
    total_sub_bytes = len(packed) + len(scales) + len(sdir_bytes)
    entry = {
        "tensor_name": f"blk.{layer}.{family}.weight",
        "layer": layer,
        "family": family,
        "shape": [d_out, d_in],
        "orientation": "canonical_d_out_d_in",
        "W_ref_bytes": d_out * d_in * 4,
        "W_ref_Q4_budget_bytes": 2_179_072 + 351_192,  # budget + margin
        "W_low_packed_bytes": len(packed),
        "W_low_scale_bytes": len(scales),
        "residual_bytes": len(sdir_bytes),
        "total_substitutive_bytes": len(packed) + len(scales) + len(sdir_bytes),
        "memory_margin_bytes": 351_192,
        "formats": {
            "W_low_format": "sdiw_v1",
            "residual_format": "sdir_v1",
            "k_percent": k_pct,
            "value_dtype": "fp16",
            "mask_encoding": "dense_bitmap",
            "scale_policy": "block32_fp16",
        },
        "paths": {
            "sdiw_path": f"tensors/blk.{layer}.{family}.wlow.sdiw",
            "sdir_path": f"tensors/blk.{layer}.{family}.residual.sdir",
        },
        "checksums": {
            "wlow": sha256_bytes(packed + scales),
            "residual": sha256_bytes(sdir_bytes),
        },
    }
    return entry


def build_manifest(bundle_dir, entries, layer, seed, k_pct):
    manifest = {
        "schema_version": "1.0",
        "package_id": f"phase31bg-repro-L{layer}-S{seed}",
        "bundle_type": "ffn_up_substitutive",
        "source_model": {
            "model_name": "qwen2.5-0.5b",
            "architecture": "ffn",
            "quantization": "Q4_K_M_reference",
        },
        "substitution_policy": {
            "k_percent": k_pct,
            "alpha": 1.0,
            "W_low_format": "sdiw_v1",
            "residual_encoding": "sdir_v1",
            "residual_policy": "always_on",
            "scale_policy": "block32_fp16",
        },
        "layers": entries,
    }
    manifest_path = os.path.join(bundle_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def run_anchor_reproduction(anchor_key, anchor, bundle_dir, X):
    """Reproduce one anchor: extract, build bundle, validate via ManifestRuntime."""
    layer = anchor["layer"]
    family = anchor["family"]
    seed = anchor["seed"]
    k_pct = anchor["k_pct"]
    d_out = anchor["d_out"]
    d_in = anchor["d_in"]

    # Load real W_ref from GGUF
    tensors = load_layer_tensors_from_gguf(GGUF_PATH, layer, [family])
    if family not in tensors:
        return {"anchor": anchor_key, "status": "BLOCKED_MISSING_TENSOR", "passed": False}

    W_ref = tensors[family]

    # Build bundle
    entry = build_single_family_bundle(bundle_dir, family, d_out, d_in, W_ref, k_pct, layer, seed)
    manifest = build_manifest(bundle_dir, [entry], layer, seed, k_pct)

    # Validate via ManifestLoader (schema v1.0)
    loader = ManifestLoader(bundle_dir)
    try:
        loader.load()
        validation = loader.validate_bundle(bundle_dir)
    except Exception as exc:
        return {"anchor": anchor_key, "status": "SCHEMA_VALIDATION_FAILED", "error": str(exc), "passed": False}

    # Execute via ManifestRuntime
    runtime = ManifestRuntime()
    runtime.load_and_validate_manifest(bundle_dir)
    resolved_entry = runtime.loader.select_tensor(family, layer)

    # Reference: Q4_K only output
    sdiw_loaded = runtime.loader.load_sdiw(resolved_entry, bundle_dir)
    Y_low = sdiw_streaming_apply(sdiw_loaded["packed_bytes"], sdiw_loaded["scale_bytes"], X, d_out, d_in)

    # Reference: Q4_K + WR output
    Y_sub, runtime_info = runtime.execute_substitutive_path(resolved_entry, X)

    # Compute reference Y_ref for the same X
    Y_ref = X @ W_ref.T

    cos_low, mae_low = compute_metrics(Y_ref, Y_low)
    cos_sub, mae_sub = compute_metrics(Y_ref, Y_sub)
    delta_cos = float(cos_sub - cos_low)
    mae_delta = float(mae_sub - mae_low)
    cos_low = float(cos_low)
    cos_sub = float(cos_sub)
    mae_low = float(mae_low)
    mae_sub = float(mae_sub)

    # Check against accepted anchor
    delta_cos_exp = anchor["delta_cos_expected"]
    mae_delta_exp = anchor["MAE_delta_expected"]
    cos_low_exp = anchor["cos_low"]

    cos_match = abs(cos_low - cos_low_exp) < 0.001
    delta_cos_match = abs(delta_cos - delta_cos_exp) < 0.005
    mae_delta_match = abs(mae_delta - mae_delta_exp) < 0.001

    # The severe regression is expected: WR makes cosine WORSE for L21-S9
    cos_regression_expected = delta_cos_exp < -0.05
    cos_regression_observed = delta_cos < -0.05

    passed = cos_match and delta_cos_match and mae_delta_match

    result = {
        "anchor": anchor_key,
        "layer": layer,
        "seed": seed,
        "family": family,
        "k_pct": k_pct,
        # Computed
        "cos_low": cos_low,
        "cos_low_expected": cos_low_exp,
        "cos_low_match": cos_match,
        "cos_sub": cos_sub,
        "delta_cos": delta_cos,
        "delta_cos_expected": delta_cos_exp,
        "delta_cos_match": delta_cos_match,
        "MAE_delta": mae_delta,
        "MAE_delta_expected": mae_delta_exp,
        "MAE_delta_match": mae_delta_match,
        # Severe regression check
        "severe_regression_expected": cos_regression_expected,
        "severe_regression_observed": cos_regression_observed,
        "severe_match": cos_regression_expected == cos_regression_observed,
        # Runtime
        "runtime_counters": runtime.get_counters(),
        "schema_validation_passed": validation["error_count"] == 0,
        "passed": passed and cos_regression_expected == cos_regression_observed,
        "status": "REPRODUCED" if passed and cos_regression_expected == cos_regression_observed else "PARTIAL_REPRODUCTION",
    }

    return result


def main():
    print(f"GGUF model: {GGUF_PATH}")
    print(f"llama.cpp root: {LLAMA_CPP_ROOT}")
    print()

    # Verify GGUF is accessible
    if not os.path.isfile(GGUF_PATH):
        print(f"ERROR: GGUF file not found: {GGUF_PATH}")
        sys.exit(1)

    tmpdir = tempfile.mkdtemp(prefix="phase31bg_repro_")
    print(f"Temp bundle: {tmpdir}")
    print()

    try:
        # Use a fixed seed for X to get the same activation vector used in accepted results
        # The original runs used seed=seed_value for the RNG that generated X
        results = {}

        for anchor_key, anchor in ACCEPTED.items():
            # Use a fixed X vector (same across runs for reproducibility)
            rng = np.random.RandomState(anchor["seed"])
            X = rng.randn(anchor["d_in"]).astype(np.float32)

            subdir = os.path.join(tmpdir, f"bundle_{anchor_key}")
            os.makedirs(subdir, exist_ok=True)

            result = run_anchor_reproduction(anchor_key, anchor, subdir, X)
            results[anchor_key] = result

            print(f"[{anchor_key}] status={result['status']}")
            print(f"  cos_low:  {result['cos_low']:.6f} (expected {result['cos_low_expected']:.6f}) match={result['cos_low_match']}")
            print(f"  cos_sub:  {result['cos_sub']:.6f}")
            print(f"  delta_cos: {result['delta_cos']:+.6f} (expected {result['delta_cos_expected']:+.6f}) match={result['delta_cos_match']}")
            print(f"  MAE_delta: {result['MAE_delta']:+.6f} (expected {result['MAE_delta_expected']:+.6f}) match={result['MAE_delta_match']}")
            print(f"  severe_expected={result['severe_regression_expected']} severe_observed={result['severe_regression_observed']} match={result['severe_match']}")
            print(f"  passed={result['passed']}")
            print()

        # Overall result
        all_passed = all(r["passed"] for r in results.values())
        schema_validation_passed = all(r.get("schema_validation_passed", False) for r in results.values())

        classification = (
            "PASS_31BG_CLEAN_REPRODUCTION_ANCHORS"
            if all_passed
            else "PARTIAL_31BG_CLEAN_REPRODUCTION_ANCHORS"
        )

        output = {
            "phase": "31BG",
            "classification": classification,
            "schema_version": "1.0",
            "anchors": results,
            "schema_validation_passed": schema_validation_passed,
            "all_passed": all_passed,
            "gguf_path": os.path.basename(GGUF_PATH),
            "environment": {
                "SDI_GGUF_MODEL_PATH": "SDI_GGUF_MODEL_PATH",
                "SDI_LLAMA_CPP_ROOT": "SDI_LLAMA_CPP_ROOT",
            },
            "note": "pack_wlow custom quantization != llama.cpp Q2_K; accepted anchor values require Q2_K dequantization path",
        }

        os.makedirs(os.path.join(REPO, "results"), exist_ok=True)
        with open(os.path.join(REPO, "results", "PHASE31BG_CLEAN_REPRODUCTION_HARDENED_SCHEMA.json"), "w") as f:
            json.dump(output, f, indent=2)

        print(f"Classification: {classification}")
        print(f"Schema validation: {'PASS' if schema_validation_passed else 'FAIL'}")
        print(f"All anchors reproduced: {'PASS' if all_passed else 'PARTIAL'}")
        print(f"Results: {os.path.join(REPO, 'results', 'PHASE31BG_CLEAN_REPRODUCTION_HARDENED_SCHEMA.json')}")

        return 0 if all_passed else 1

    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    raise SystemExit(main())
