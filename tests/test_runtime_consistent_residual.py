#!/usr/bin/env python3
"""
Regression test for runtime-consistent residual generation.

Bug: older scripts computed R_old = W_ref - W_low_raw, but the runtime
does not use W_low_raw directly. It uses decode(pack(W_low_raw)).

Correct residual:
    R_runtime = W_ref - decode(pack(W_low_raw))
"""

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from runtime_consistent_residual import (  # noqa: E402
    make_legacy_residual,
    make_runtime_consistent_residual,
    q4_quantize_blocked,
)


def cosine(a, b):
    a = a.ravel()
    b = b.ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denom == 0 else float(np.dot(a, b) / denom)


def test_runtime_vs_legacy_residual():
    rows, cols = 896, 4864
    rng = np.random.RandomState(42)

    W_ref = (rng.randn(rows, cols).astype(np.float32) * 0.1)
    W_low_raw = q4_quantize_blocked(W_ref)

    rt = make_runtime_consistent_residual(W_ref=W_ref, W_low_raw=W_low_raw)
    W_low_runtime = rt["W_low_runtime"]
    R_runtime = rt["R_runtime"]
    R_old = make_legacy_residual(W_ref, W_low_raw)

    X = (rng.randn(12, rows).astype(np.float32) * 0.5)
    Y_ref = X @ W_ref
    Y_runtime_correct = X @ W_low_runtime + X @ R_runtime
    Y_legacy_on_runtime = X @ W_low_runtime + X @ R_old

    mae_runtime = float(np.mean(np.abs(Y_ref - Y_runtime_correct)))
    mae_legacy = float(np.mean(np.abs(Y_ref - Y_legacy_on_runtime)))
    max_runtime = float(np.max(np.abs(Y_ref - Y_runtime_correct)))
    max_legacy = float(np.max(np.abs(Y_ref - Y_legacy_on_runtime)))
    cos_runtime = cosine(Y_ref, Y_runtime_correct)
    cos_legacy = cosine(Y_ref, Y_legacy_on_runtime)

    wlow_diff = float(np.max(np.abs(W_low_runtime - W_low_raw)))
    residual_diff = float(np.max(np.abs(R_runtime - R_old)))

    results = {
        "classification": "PASS_RUNTIME_CONSISTENT_RESIDUAL",
        "residual_definition": "R_runtime = W_ref - decode(pack(W_low_raw))",
        "shape": [rows, cols],
        "metrics": {
            "mae_runtime_correct": mae_runtime,
            "mae_legacy_on_runtime": mae_legacy,
            "max_error_runtime_correct": max_runtime,
            "max_error_legacy_on_runtime": max_legacy,
            "cos_runtime_correct": cos_runtime,
            "cos_legacy_on_runtime": cos_legacy,
        },
        "bug_magnitude": {
            "W_low_runtime_vs_raw_max_diff": wlow_diff,
            "R_runtime_vs_R_old_max_diff": residual_diff,
        },
    }

    out_path = os.path.join(REPO, "results", "test_runtime_consistent_residual.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    assert wlow_diff > 0.0
    assert residual_diff > 0.0
    assert mae_runtime < 1e-5
    assert max_runtime < 1e-4
    assert mae_runtime < mae_legacy
    assert max_runtime < max_legacy
    assert cos_runtime >= cos_legacy


if __name__ == "__main__":
    test_runtime_vs_legacy_residual()
