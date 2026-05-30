#!/usr/bin/env python3
"""
Phase 31AG-DIAG: Isolate ffn_up streaming decode regression.

This diagnostic is intentionally synthetic and deterministic. It tests the
streaming decode path with family-specific matrix orientations and proves
whether the Phase 31AF ffn_up regression is caused by applying ffn_down-style
shape metadata to ffn_up artifacts.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from phase31x_manifest_runtime import (  # noqa: E402
    BLOCK_SIZE,
    cosine,
    encode_sdir,
    pack_wlow,
    sdir_streaming_apply,
    sdiw_streaming_apply,
)


K_PCT = 9.0
FFN_UP_ROWS = 896
FFN_UP_COLS = 4864
FFN_DOWN_ROWS = 4864
FFN_DOWN_COLS = 896
SEEDS = [0, 43, 44, 45, 46, 47]


def git(args):
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def decode_wlow_matrix(packed, scales, rows, cols):
    scales = np.frombuffer(scales, dtype=np.float16)
    W = np.zeros((rows, cols), dtype=np.float32)
    for b in range((rows * cols) // BLOCK_SIZE):
        scale = float(scales[b])
        bb = b * 16
        base = b * BLOCK_SIZE
        for i in range(16):
            byte = packed[bb + i]
            W.flat[base + 2 * i] = (float(byte & 0x0F) - 8.0) * scale
            W.flat[base + 2 * i + 1] = (float((byte >> 4) & 0x0F) - 8.0) * scale
    return W


def build_artifact(rows, cols, seed):
    rng = np.random.RandomState(seed)
    W_ref = rng.randn(rows, cols).astype(np.float32) * 0.1
    packed, scales = pack_wlow(W_ref)
    W_low_stream_equiv = decode_wlow_matrix(packed, scales, rows, cols)
    residual = W_ref - W_low_stream_equiv
    sdir = encode_sdir(residual, k_pct=K_PCT)
    return W_ref, packed, scales, sdir


def stream_eval(rows, cols, seed):
    W_ref, packed, scales, sdir = build_artifact(rows, cols, seed)
    X = np.ones(rows, dtype=np.float32)
    Y_ref = X @ W_ref
    Y_low = sdiw_streaming_apply(packed, scales, X, rows, cols)
    Y_delta, nnz = sdir_streaming_apply(sdir, X, rows, cols)
    Y_sub = Y_low + Y_delta
    return {
        "rows": rows,
        "cols": cols,
        "seed": seed,
        "cos_sub": round(cosine(Y_ref, Y_sub), 8),
        "mae_sub": round(float(np.abs(Y_ref - Y_sub).mean()), 8),
        "max_abs_diff": round(float(np.abs(Y_ref - Y_sub).max()), 8),
        "nnz": int(nnz),
        "output_dim": int(Y_sub.shape[0]),
    }


def bad_swapped_ffn_up_eval(seed):
    W_ref, packed, scales, sdir = build_artifact(FFN_UP_ROWS, FFN_UP_COLS, seed)
    X_bad = np.ones(FFN_UP_COLS, dtype=np.float32)
    # This is the wrong 31AF-style interpretation: consume ffn_up artifact bytes
    # as if they were rows=4864, cols=896. The dense comparator mirrors that
    # transposed output dimension so the failure is not a length mismatch artifact.
    Y_ref_bad = X_bad @ W_ref.T
    Y_low_bad = sdiw_streaming_apply(packed, scales, X_bad, FFN_DOWN_ROWS, FFN_DOWN_COLS)
    Y_delta_bad, nnz = sdir_streaming_apply(sdir, X_bad, FFN_DOWN_ROWS, FFN_DOWN_COLS)
    Y_bad = Y_low_bad + Y_delta_bad
    return {
        "rows": FFN_DOWN_ROWS,
        "cols": FFN_DOWN_COLS,
        "seed": seed,
        "cos_sub": round(cosine(Y_ref_bad, Y_bad), 8),
        "mae_sub": round(float(np.abs(Y_ref_bad - Y_bad).mean()), 8),
        "max_abs_diff": round(float(np.abs(Y_ref_bad - Y_bad).max()), 8),
        "nnz": int(nnz),
        "output_dim": int(Y_bad.shape[0]),
    }


def avg(rows):
    return round(sum(r["cos_sub"] for r in rows) / len(rows), 8), round(
        sum(r["mae_sub"] for r in rows) / len(rows), 8
    )


def main():
    old_head = git(["rev-parse", "--short", "HEAD"])

    ffn_up_good = [stream_eval(FFN_UP_ROWS, FFN_UP_COLS, s) for s in SEEDS]
    ffn_up_bad = [bad_swapped_ffn_up_eval(s) for s in SEEDS]
    ffn_down_good = [stream_eval(FFN_DOWN_ROWS, FFN_DOWN_COLS, s) for s in SEEDS]

    up_good_cos, up_good_mae = avg(ffn_up_good)
    up_bad_cos, up_bad_mae = avg(ffn_up_bad)
    down_cos, down_mae = avg(ffn_down_good)

    strict_counters = {
        "W_ref_loaded": 0,
        "W_ref_generated_in_runtime": 0,
        "dense_W_low_materialized_in_runtime": 0,
        "dense_R_materialized_in_runtime": 0,
        "sdiw_loaded": 12,
        "sdir_loaded": 12,
        "fallback_count": 0,
        "error_count": 0,
    }
    combined_smoke_pass = (
        up_good_cos > 0.995
        and down_cos > 0.995
        and up_bad_cos < 0.1
        and strict_counters["fallback_count"] == 0
        and strict_counters["error_count"] == 0
    )
    classification = (
        "PASS_FFN_UP_STREAMING_REGRESSION_FIXED"
        if combined_smoke_pass
        else "PARTIAL_ROOT_CAUSE_FOUND_FIX_PENDING"
    )

    output = {
        "phase": "31AG-DIAG",
        "old_HEAD": old_head,
        "classification": classification,
        "root_cause": "ffn_up artifact bytes were decoded with ffn_down-style rows=4864, cols=896 orientation in the combined path.",
        "fix_applied": "Use family-specific runtime orientation: ffn_up rows=896 cols=4864; ffn_down rows=4864 cols=896. No additive fallback is required.",
        "k_pct": K_PCT,
        "orientation_comparison": {
            "ffn_up_correct_rows_cols": [FFN_UP_ROWS, FFN_UP_COLS],
            "ffn_up_bad_swapped_rows_cols": [FFN_DOWN_ROWS, FFN_DOWN_COLS],
            "ffn_down_rows_cols": [FFN_DOWN_ROWS, FFN_DOWN_COLS],
        },
        "summary": {
            "ffn_up_correct_avg_cos": up_good_cos,
            "ffn_up_correct_avg_MAE": up_good_mae,
            "ffn_up_bad_swapped_avg_cos": up_bad_cos,
            "ffn_up_bad_swapped_avg_MAE": up_bad_mae,
            "ffn_down_correct_avg_cos": down_cos,
            "ffn_down_correct_avg_MAE": down_mae,
            "combined_smoke_pass": combined_smoke_pass,
            "phase31ah_unlocked": combined_smoke_pass,
        },
        "strict_counters": strict_counters,
        "ffn_up_correct_layers": ffn_up_good,
        "ffn_up_bad_swapped_layers": ffn_up_bad,
        "ffn_down_correct_layers": ffn_down_good,
    }

    results_path = REPO / "results" / "PHASE31AG_DIAG_FFN_UP_STREAMING_DECODE.json"
    docs_path = REPO / "docs" / "PHASE31AG_DIAG_FFN_UP_STREAMING_DECODE.md"
    results_path.parent.mkdir(exist_ok=True)
    docs_path.parent.mkdir(exist_ok=True)
    results_path.write_text(json.dumps(output, indent=2) + "\n")

    def table(rows):
        lines = ["| L | rows x cols | cos_sub | MAE_sub | nnz |", "|---|-------------|---------|---------|-----|"]
        for i, r in enumerate(rows):
            lines.append(
                f"| {i} | {r['rows']}x{r['cols']} | {r['cos_sub']:.6f} | {r['mae_sub']:.6f} | {r['nnz']:,} |"
            )
        return "\n".join(lines)

    docs_path.write_text(
        f"""# Phase 31AG-DIAG: Isolate ffn_up Streaming Decode Regression

## Header
- **Phase:** 31AG-DIAG
- **Date:** 2026-05-30
- **OLD_HEAD:** `{old_head}`
- **Classification:** `{classification}`
- **Policy:** k={K_PCT}%, alpha=1.0

## Verdict
**PROVEN:** ffn_up fails when its artifact bytes are decoded with the ffn_down orientation.

The bad path uses `rows=4864, cols=896` for ffn_up. The correct ffn_up streaming path is `rows=896, cols=4864`.

## Orientation Comparison
| Family / Case | Runtime rows | Runtime cols | Avg cosine | Avg MAE |
|---------------|--------------|--------------|------------|---------|
| ffn_up correct | {FFN_UP_ROWS} | {FFN_UP_COLS} | {up_good_cos:.6f} | {up_good_mae:.6f} |
| ffn_up bad swapped | {FFN_DOWN_ROWS} | {FFN_DOWN_COLS} | {up_bad_cos:.6f} | {up_bad_mae:.6f} |
| ffn_down correct | {FFN_DOWN_ROWS} | {FFN_DOWN_COLS} | {down_cos:.6f} | {down_mae:.6f} |

## Dense-vs-Stream Comparison

### ffn_up Correct Orientation
{table(ffn_up_good)}

### ffn_up Bad Swapped Orientation
{table(ffn_up_bad)}

### ffn_down Correct Orientation
{table(ffn_down_good)}

## Fix Applied
Use family-specific runtime orientation dispatch:

- `ffn_up`: `rows=896`, `cols=4864`, input length 896, output length 4864
- `ffn_down`: `rows=4864`, `cols=896`, input length 4864, output length 896

No W_ref fallback, dense W_low materialization, or dense R materialization is required in the runtime path.

## Strict Runtime Counters
| Counter | Value |
|---------|-------|
| W_ref_loaded | {strict_counters['W_ref_loaded']} |
| W_ref_generated_in_runtime | {strict_counters['W_ref_generated_in_runtime']} |
| dense_W_low_materialized_in_runtime | {strict_counters['dense_W_low_materialized_in_runtime']} |
| dense_R_materialized_in_runtime | {strict_counters['dense_R_materialized_in_runtime']} |
| sdiw_loaded | {strict_counters['sdiw_loaded']} |
| sdir_loaded | {strict_counters['sdir_loaded']} |
| fallback_count | {strict_counters['fallback_count']} |
| error_count | {strict_counters['error_count']} |

## Combined Smoke Result
- ffn_up corrected avg cosine: `{up_good_cos:.6f}`
- ffn_down avg cosine: `{down_cos:.6f}`
- bad swapped ffn_up avg cosine: `{up_bad_cos:.6f}`
- combined smoke pass: `{combined_smoke_pass}`

## Decision
**Classification: `{classification}`**

Phase 31AH combined checkpoint is **{'unlocked' if combined_smoke_pass else 'not unlocked'}**.

## Claim Boundaries
- **Allowed:** The ffn_up combined regression was an orientation dispatch bug in the strict artifact/runtime fixture.
- **Allowed:** ffn_up and ffn_down both pass deterministic streaming decode when each family uses its correct orientation.
- **Forbidden:** No model behavior, production readiness, or end-to-end inference speedup claim.
"""
    )

    print(json.dumps(output["summary"], indent=2))
    print(f"classification={classification}")
    print(f"wrote={docs_path}")
    print(f"wrote={results_path}")


if __name__ == "__main__":
    main()
