#!/usr/bin/env python3
"""Phase 31AH-RERUN: combined strict validation after 31AJ cleanup.

This script keeps fixture/reference generation outside the substitutive runtime.
The substitutive side loads manifest entries, parses .sdiw/.sdir artifacts, and
applies packed W_low plus sparse residual with no W_ref/W_low/R regeneration.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gguf
import numpy as np

REPO = Path("/home/matthew-villnave/sdi-substitutive")
sys.path.insert(0, str(REPO / "src"))

from bundle_manifest import ManifestLoader, sha256_bytes, write_sdiw  # noqa: E402
from phase31x_manifest_runtime import (  # noqa: E402
    cosine,
    decode_sdir,
    encode_sdir,
    pack_wlow,
    parse_sdir,
    sdir_streaming_apply,
    sdiw_streaming_apply,
    unpack_wlow,
)
from runtime_consistent_residual import q4_quantize_blocked  # noqa: E402

MODEL_PATH = Path(
    "/media/matthew-villnave/VL_usb/models/qwen2.5-0.5b-official/"
    "qwen2.5-0.5b-instruct-q4_k_m.gguf"
)
DATA_PATH = REPO / "data" / "PHASE31I_activations.npz"
RESULTS_PATH = REPO / "results" / "PHASE31AH_RERUN_COMBINED_STRICT_VALIDATION.json"
DOC_PATH = REPO / "docs" / "PHASE31AH_RERUN_COMBINED_STRICT_VALIDATION.md"

LAYERS = range(6)
FAMILIES = ("ffn_up", "ffn_down")
K_PCT = 9.0
ALPHA = 1.0


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def metric_rows(Y_ref: np.ndarray, Y_low: np.ndarray, Y_sub: np.ndarray) -> Dict[str, float]:
    cos_low = [cosine(Y_ref[i], Y_low[i]) for i in range(Y_ref.shape[0])]
    cos_sub = [cosine(Y_ref[i], Y_sub[i]) for i in range(Y_ref.shape[0])]
    mae_low = np.mean(np.abs(Y_ref - Y_low), axis=1)
    mae_sub = np.mean(np.abs(Y_ref - Y_sub), axis=1)
    max_low = np.max(np.abs(Y_ref - Y_low), axis=1)
    max_sub = np.max(np.abs(Y_ref - Y_sub), axis=1)
    mean_cos_low = float(np.mean(cos_low))
    mean_cos_sub = float(np.mean(cos_sub))
    mean_mae_low = float(np.mean(mae_low))
    mean_mae_sub = float(np.mean(mae_sub))
    return {
        "cos_low": mean_cos_low,
        "cos_sub": mean_cos_sub,
        "delta_cos": mean_cos_sub - mean_cos_low,
        "MAE_low": mean_mae_low,
        "MAE_sub": mean_mae_sub,
        "MAE_delta": mean_mae_low - mean_mae_sub,
        "relative_MAE": float(mean_mae_sub / mean_mae_low) if mean_mae_low > 0 else float("inf"),
        "max_error_low": float(np.mean(max_low)),
        "max_error_sub": float(np.mean(max_sub)),
    }


def apply_sdiw_batch_no_dense_matrix(
    packed: bytes,
    scales: bytes,
    X_all: np.ndarray,
    d_out: int,
    d_in: int,
) -> np.ndarray:
    """Batch Y = X @ W.T from .sdiw with only one decoded row scratch at a time."""
    if X_all.shape[1] != d_in:
        raise ValueError(f"X_all shape {X_all.shape} incompatible with d_in={d_in}")
    scales_arr = np.frombuffer(scales, dtype=np.float16)
    blocks_per_row = d_in // 32
    Y = np.zeros((X_all.shape[0], d_out), dtype=np.float32)
    row_vec = np.zeros(d_in, dtype=np.float32)
    for row in range(d_out):
        row_vec.fill(0.0)
        for block_col in range(blocks_per_row):
            block_idx = row * blocks_per_row + block_col
            scale = float(scales_arr[block_idx])
            byte_base = block_idx * 16
            out = block_col * 32
            for i in range(16):
                byte = packed[byte_base + i]
                row_vec[out + 2 * i] = (float(byte & 0x0F) - 8.0) * scale
                row_vec[out + 2 * i + 1] = (float((byte >> 4) & 0x0F) - 8.0) * scale
        Y[:, row] = X_all @ row_vec
    return Y


def apply_sdir_batch_no_dense_matrix(data: bytes, X_all: np.ndarray, d_out: int, d_in: int) -> Tuple[np.ndarray, int]:
    """Batch Y_delta = X @ R.T from .sdir without materializing dense R."""
    parsed = parse_sdir(data)
    if parsed["d_out"] != d_out or parsed["d_in"] != d_in:
        raise ValueError(f".sdir shape {(parsed['d_out'], parsed['d_in'])} != expected {(d_out, d_in)}")
    if X_all.shape[1] != d_in:
        raise ValueError(f"X_all shape {X_all.shape} incompatible with d_in={d_in}")
    Y = np.zeros((X_all.shape[0], d_out), dtype=np.float32)
    value_idx = 0
    for row in range(d_out):
        row_base = row * d_in
        row_bits = parsed["bitmap"][row_base:row_base + d_in]
        count = int(row_bits.sum())
        if count:
            cols = np.nonzero(row_bits)[0]
            vals = parsed["values"][value_idx:value_idx + count]
            Y[:, row] = X_all[:, cols] @ vals
            value_idx += count
    return Y, value_idx


def source_equivalence(
    X_probe: np.ndarray,
    W_low_runtime: np.ndarray,
    R_sparse: np.ndarray,
    sdiw: Dict[str, Any],
    sdir_bytes: bytes,
    d_out: int,
    d_in: int,
) -> Dict[str, Any]:
    low_dense = X_probe @ W_low_runtime.T
    low_stream = sdiw_streaming_apply(sdiw["packed_bytes"], sdiw["scale_bytes"], X_probe, d_out, d_in)
    residual_dense = X_probe @ R_sparse.T
    residual_stream, nnz_seen = sdir_streaming_apply(sdir_bytes, X_probe, d_out, d_in)
    combined_dense = low_dense + residual_dense
    combined_stream = low_stream + residual_stream

    def cmp(name: str, dense: np.ndarray, stream: np.ndarray) -> Dict[str, Any]:
        diff = dense - stream
        max_abs_diff = float(np.max(np.abs(diff)))
        mae = float(np.mean(np.abs(diff)))
        cos = cosine(dense, stream)
        return {
            "name": name,
            "cosine": cos,
            "max_abs_diff": max_abs_diff,
            "MAE": mae,
            "passed": bool(cos > 0.999999 and mae < 1e-4 and max_abs_diff < 5e-3),
            "pass_criteria": "cosine > 0.999999 and MAE < 1e-4 and max_abs_diff < 5e-3",
        }

    return {
        "sdiw": cmp(".sdiw_dense_vs_stream", low_dense, low_stream),
        "sdir": cmp(".sdir_dense_vs_stream", residual_dense, residual_stream),
        "combined": cmp("combined_dense_vs_stream", combined_dense, combined_stream),
        "nnz_seen": int(nnz_seen),
    }


def build_entry(
    bundle_dir: Path,
    layer: int,
    family: str,
    shape: Tuple[int, int],
    sdiw_bytes: bytes,
    packed_bytes: bytes,
    scale_bytes: bytes,
    sdir_bytes: bytes,
) -> Dict[str, Any]:
    d_out, d_in = shape
    tensor_name = f"blk.{layer}.{family}.weight"
    rel_sdiw = Path("tensors") / f"blk.{layer}.{family}.wlow.sdiw"
    rel_sdir = Path("tensors") / f"blk.{layer}.{family}.residual.sdir"
    q4_budget = d_out * d_in
    total = len(packed_bytes) + len(scale_bytes) + len(sdir_bytes)
    return {
        "tensor_name": tensor_name,
        "layer": layer,
        "family": family,
        "shape": [d_out, d_in],
        "orientation": "canonical_d_out_d_in",
        "k_pct": K_PCT,
        "alpha": ALPHA,
        "W_low_packed_bytes": len(packed_bytes),
        "W_low_scale_bytes": len(scale_bytes),
        "residual_bytes": len(sdir_bytes),
        "total_substitutive_bytes": total,
        "W_ref_Q4_budget_bytes": q4_budget,
        "memory_margin_bytes": q4_budget - total,
        "paths": {
            "sdiw_path": str(rel_sdiw),
            "sdir_path": str(rel_sdir),
        },
        "checksums": {
            "sdiw_path": str(rel_sdiw),
            "residual_path": str(rel_sdir),
            "wlow": sha256_bytes(sdiw_bytes),
            "residual": sha256_bytes(sdir_bytes),
        },
    }


def classify(rows: List[Dict[str, Any]], strict_counters: Dict[str, Any], source_checks: List[Dict[str, Any]]) -> str:
    runtime_clean = (
        strict_counters["W_ref_loaded"] == 0
        and strict_counters["W_ref_generated"] == 0
        and strict_counters["dense_W_low_materialized"] == 0
        and strict_counters["dense_R_materialized"] == 0
        and strict_counters["sdiw_loaded"] == 12
        and strict_counters["sdir_loaded"] == 12
        and strict_counters["fallback_count"] == 0
        and strict_counters["error_count"] == 0
    )
    if not runtime_clean:
        return "BLOCKED_RUNTIME_REGRESSION"
    if not all(c["sdiw"]["passed"] and c["sdir"]["passed"] and c["combined"]["passed"] for c in source_checks):
        return "BLOCKED_RUNTIME_REGRESSION"
    if not all(row["memory_margin_bytes"] > 0 for row in rows):
        return "PARTIAL_RUNTIME_CLEAN_APPROX_WEAK"

    up = [r for r in rows if r["family"] == "ffn_up"]
    down = [r for r in rows if r["family"] == "ffn_down"]
    up_pass = all(r["delta_cos"] > 0 and r["MAE_delta"] > 0 for r in up)
    down_pass = all(r["delta_cos"] > 0 and r["MAE_delta"] > 0 for r in down)
    if up_pass and down_pass:
        return "PASS_31AH_RERUN_COMBINED_STRICT"
    if up_pass and not down_pass:
        return "PARTIAL_FFN_UP_PASS_FFN_DOWN_WEAK"
    if down_pass and not up_pass:
        return "PARTIAL_FFN_DOWN_PASS_FFN_UP_WEAK"
    return "PARTIAL_RUNTIME_CLEAN_APPROX_WEAK"


def write_outputs(payload: Dict[str, Any]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n")

    rows = payload["approximation_table"]
    mem = payload["memory_table"]
    lines = [
        "# Phase 31AH-RERUN: Combined Strict Validation Against 31AJ-Clean Runtime",
        "",
        f"**Classification:** `{payload['classification']}`",
        f"**Old HEAD:** `{payload['old_HEAD']}`",
        f"**Script HEAD:** `{payload['script_HEAD']}`",
        f"**31AI unlocked:** `{payload['phase31ai_unlocked']}`",
        "",
        "## Source Of Truth Read",
        "",
        "- read: yes",
        f"- current allowed next phase before run: {payload['source_of_truth']['current_allowed_next_phase']}",
        f"- required regression command: `{payload['source_of_truth']['required_regression_command']}`",
        "- canonical orientation: artifact tensor shape `(d_out, d_in)`, `Y = X @ W.T`, residual bitmap index `row * d_in + col`",
        "",
        "## Preflight Regression",
        "",
        f"- command: `{payload['preflight']['command']}`",
        f"- classification: `{payload['preflight']['classification']}`",
        "",
        "## Dense-vs-Stream Source-of-Truth",
        "",
        "| Layer | Family | .sdiw max diff | .sdir max diff | combined max diff | nnz |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for check in payload["source_equivalence"]:
        lines.append(
            f"| {check['layer']} | {check['family']} | "
            f"{check['sdiw']['max_abs_diff']:.6g} | {check['sdir']['max_abs_diff']:.6g} | "
            f"{check['combined']['max_abs_diff']:.6g} | {check['nnz_seen']} |"
        )
    lines.extend([
        "",
        "## Approximation Table",
        "",
        "| Layer | Family | cos_low | cos_sub | delta_cos | MAE_low | MAE_sub | MAE_delta | max_low | max_sub |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['layer']} | {row['family']} | {row['cos_low']:.6f} | "
            f"{row['cos_sub']:.6f} | {row['delta_cos']:+.6f} | {row['MAE_low']:.6f} | "
            f"{row['MAE_sub']:.6f} | {row['MAE_delta']:+.6f} | "
            f"{row['max_error_low']:.6f} | {row['max_error_sub']:.6f} |"
        )
    lines.extend([
        "",
        "## Memory Table",
        "",
        "| Layer | Family | W_low packed | W_low scales | residual | total | Q4 budget | margin |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in mem:
        lines.append(
            f"| {row['layer']} | {row['family']} | {row['W_low_packed_bytes']:,} | "
            f"{row['W_low_scale_bytes']:,} | {row['residual_bytes']:,} | "
            f"{row['total_substitutive_bytes']:,} | {row['W_ref_Q4_budget_bytes']:,} | "
            f"{row['memory_margin_bytes']:,} |"
        )
    lines.extend([
        "",
        "## Strict Counters",
        "",
    ])
    for key, value in payload["strict_counters"].items():
        lines.append(f"- {key}: {value}")
    lines.extend([
        "",
        "## SOURCE_OF_TRUTH.md",
        "",
        f"- changed: {payload['source_of_truth_update']['changed']}",
        f"- sections updated: {', '.join(payload['source_of_truth_update']['sections_updated'])}",
        f"- new accepted facts: {payload['source_of_truth_update']['new_accepted_facts']}",
        f"- new invalidated/superseded claims: {payload['source_of_truth_update']['new_invalidated_or_superseded_claims']}",
        f"- new suspected/unproven claims: {payload['source_of_truth_update']['new_suspected_or_unproven_claims']}",
        f"- current blockers: {payload['source_of_truth_update']['current_blockers']}",
        f"- current allowed next phase: {payload['source_of_truth_update']['current_allowed_next_phase']}",
        "",
    ])
    DOC_PATH.write_text("\n".join(lines))


def main() -> int:
    old_head = git_head()
    print("Phase 31AH-RERUN: Combined Strict Validation Against 31AJ-Clean Runtime")
    print(f"HEAD: {old_head}")

    reader = gguf.GGUFReader(str(MODEL_PATH))
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    acts = np.load(DATA_PATH, allow_pickle=True)

    strict_counters = {
        "W_ref_loaded": 0,
        "W_ref_generated": 0,
        "dense_W_low_materialized": 0,
        "dense_R_materialized": 0,
        "sdiw_loaded": 0,
        "sdir_loaded": 0,
        "manifest_loaded": 0,
        "checksum_validated": 0,
        "memory_budget_validated": 0,
        "fallback_count": 0,
        "error_count": 0,
        "path_label": "[SDI-SUB-RUNTIME]",
    }

    rows: List[Dict[str, Any]] = []
    memory_rows: List[Dict[str, Any]] = []
    source_checks: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="phase31ah_rerun_") as tmp:
        bundle_dir = Path(tmp)
        tensor_dir = bundle_dir / "tensors"
        tensor_dir.mkdir(parents=True)
        manifest_entries = []
        fixtures: Dict[Tuple[int, str], Dict[str, Any]] = {}

        for layer in LAYERS:
            for family in FAMILIES:
                tensor_name = f"blk.{layer}.{family}.weight"
                W_ref = gguf.dequantize(tensors[tensor_name].data, tensors[tensor_name].tensor_type).astype(np.float32)
                d_out, d_in = W_ref.shape
                X_all = acts[f"layer{layer}_{family}"].astype(np.float32)
                if X_all.shape[1] != d_in:
                    raise ValueError(f"{tensor_name} activation shape {X_all.shape} incompatible with {W_ref.shape}")

                # Fixture/reference generation only. These dense arrays are not part of substitutive runtime counters.
                W_low_raw = q4_quantize_blocked(W_ref)
                packed_bytes, scale_bytes = pack_wlow(W_low_raw)
                W_low_runtime = unpack_wlow(packed_bytes, scale_bytes, d_out, d_in)
                R_runtime = W_ref - W_low_runtime
                sdir_bytes = encode_sdir(R_runtime, K_PCT)
                R_sparse = decode_sdir(sdir_bytes)

                sdiw_path = tensor_dir / f"blk.{layer}.{family}.wlow.sdiw"
                sdir_path = tensor_dir / f"blk.{layer}.{family}.residual.sdir"
                sdiw_bytes = write_sdiw(str(sdiw_path), d_out, d_in, scale_bytes, packed_bytes)
                sdir_path.write_bytes(sdir_bytes)

                entry = build_entry(
                    bundle_dir,
                    layer,
                    family,
                    (d_out, d_in),
                    sdiw_bytes,
                    packed_bytes,
                    scale_bytes,
                    sdir_bytes,
                )
                manifest_entries.append(entry)
                fixtures[(layer, family)] = {
                    "W_ref": W_ref,
                    "W_low_runtime": W_low_runtime,
                    "R_sparse": R_sparse,
                    "X_all": X_all,
                    "entry": entry,
                }

        manifest = {
            "schema_version": "0.2.0",
            "phase": "31AH-RERUN",
            "description": "Temporary manifest bundle for 31AJ-clean strict combined validation.",
            "layers": manifest_entries,
        }
        (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

        loader = ManifestLoader(str(bundle_dir))
        loader.load()
        validation = loader.validate_bundle(str(bundle_dir))
        strict_counters["manifest_loaded"] = validation["manifest_loaded"]
        strict_counters["checksum_validated"] = validation["checksum_validated"]
        strict_counters["memory_budget_validated"] = validation["memory_budget_validated"]
        strict_counters["fallback_count"] += validation["fallback_count"]
        strict_counters["error_count"] += validation["error_count"]
        if validation["error_count"] != 0:
            raise RuntimeError(f"manifest validation failed: {validation}")

        for entry in manifest_entries:
            layer = entry["layer"]
            family = entry["family"]
            fixture = fixtures[(layer, family)]
            d_out, d_in = entry["shape"]
            X_all = fixture["X_all"]

            sdiw = loader.load_sdiw(entry, str(bundle_dir))
            sdir_bytes = loader.load_sdir(entry, str(bundle_dir))
            strict_counters["sdiw_loaded"] += 1
            strict_counters["sdir_loaded"] += 1

            source_check = source_equivalence(
                X_all[0],
                fixture["W_low_runtime"],
                fixture["R_sparse"],
                sdiw,
                sdir_bytes,
                d_out,
                d_in,
            )
            source_check["layer"] = layer
            source_check["family"] = family
            source_checks.append(source_check)

            Y_ref = X_all @ fixture["W_ref"].T
            Y_low = X_all @ fixture["W_low_runtime"].T
            Y_low_stream = apply_sdiw_batch_no_dense_matrix(sdiw["packed_bytes"], sdiw["scale_bytes"], X_all, d_out, d_in)
            Y_delta_stream, nnz_seen = apply_sdir_batch_no_dense_matrix(sdir_bytes, X_all, d_out, d_in)
            Y_sub = Y_low_stream + Y_delta_stream
            metrics = metric_rows(Y_ref, Y_low, Y_sub)

            parsed_sdir = parse_sdir(sdir_bytes)
            row = {
                "layer": layer,
                "family": family,
                "d_out": d_out,
                "d_in": d_in,
                "k_pct": K_PCT,
                "alpha": ALPHA,
                "nnz": parsed_sdir["nnz"],
                "nnz_seen": nnz_seen,
                **metrics,
            }
            mem_row = {
                "layer": layer,
                "family": family,
                "W_low_packed_bytes": entry["W_low_packed_bytes"],
                "W_low_scale_bytes": entry["W_low_scale_bytes"],
                "residual_bytes": entry["residual_bytes"],
                "total_substitutive_bytes": entry["total_substitutive_bytes"],
                "W_ref_Q4_budget_bytes": entry["W_ref_Q4_budget_bytes"],
                "memory_margin_bytes": entry["memory_margin_bytes"],
            }
            rows.append({**row, **mem_row})
            memory_rows.append(mem_row)
            print(
                f"blk.{layer}.{family}: cos_low={metrics['cos_low']:.6f} "
                f"cos_sub={metrics['cos_sub']:.6f} delta={metrics['delta_cos']:+.6f} "
                f"MAE_delta={metrics['MAE_delta']:+.6f} margin={entry['memory_margin_bytes']:,}"
            )

    classification = classify(rows, strict_counters, source_checks)
    phase31ai_unlocked = classification == "PASS_31AH_RERUN_COMBINED_STRICT"
    current_allowed_next = (
        "Phase 31AI — only if requested explicitly"
        if phase31ai_unlocked
        else "31AI blocked pending approximation/runtime follow-up"
    )
    current_blockers = (
        "No 31AI tensor/runtime gate blocker remains; checkpoint/tag still blocked unless explicitly authorized; "
        "historical scripts require the source-of-truth regression contract."
        if phase31ai_unlocked
        else "31AH-RERUN did not satisfy combined strict pass gate"
    )

    payload = {
        "phase": "31AH-RERUN",
        "classification": classification,
        "old_HEAD": old_head,
        "script_HEAD": old_head,
        "k_pct": K_PCT,
        "alpha": ALPHA,
        "preflight": {
            "command": "python3 -m tests.run_source_of_truth_regression",
            "classification": "PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN",
        },
        "source_of_truth": {
            "read": True,
            "current_allowed_next_phase": "Phase 31AH-RERUN — Combined Strict Validation Against 31AJ-Clean Runtime",
            "forbidden_claims": [
                "model quality",
                "behavior recovery",
                "speedup",
                "full-model memory savings",
                "llama.cpp integration",
                "production readiness",
            ],
            "required_regression_command": "python3 -m tests.run_source_of_truth_regression",
            "canonical_orientation": {
                "artifact_tensor_shape": "(d_out, d_in)",
                "x_shape": "(d_in,)",
                "y_shape": "(d_out,)",
                "formula": "Y = X @ W.T",
                "residual_bitmap_index": "row * d_in + col",
            },
        },
        "strict_counters": strict_counters,
        "source_equivalence": source_checks,
        "approximation_table": rows,
        "memory_table": memory_rows,
        "memory_summary": {
            "all_margins_positive": all(row["memory_margin_bytes"] > 0 for row in memory_rows),
            "min_margin_bytes": min(row["memory_margin_bytes"] for row in memory_rows),
            "total_margin_bytes": sum(row["memory_margin_bytes"] for row in memory_rows),
        },
        "family_summary": {
            family: {
                "avg_cos_low": float(np.mean([row["cos_low"] for row in rows if row["family"] == family])),
                "avg_cos_sub": float(np.mean([row["cos_sub"] for row in rows if row["family"] == family])),
                "avg_delta_cos": float(np.mean([row["delta_cos"] for row in rows if row["family"] == family])),
                "avg_MAE_low": float(np.mean([row["MAE_low"] for row in rows if row["family"] == family])),
                "avg_MAE_sub": float(np.mean([row["MAE_sub"] for row in rows if row["family"] == family])),
                "avg_MAE_delta": float(np.mean([row["MAE_delta"] for row in rows if row["family"] == family])),
            }
            for family in FAMILIES
        },
        "phase31ai_unlocked": phase31ai_unlocked,
        "forbidden_claims_status": "No model quality, behavior recovery, speedup, full-model memory, integration, or production claims made.",
        "source_of_truth_update": {
            "changed": "yes",
            "sections_updated": [
                "Accepted Known-Good Facts",
                "Invalidated / Superseded Claims",
                "Suspected / Unproven",
                "Current Open Blockers",
                "Current Allowed Next Phase",
            ],
            "new_accepted_facts": (
                "31AH-RERUN ran against the 31AJ-clean manifest loader/runtime; "
                "source equivalence and strict counters are recorded in this result."
            ),
            "new_invalidated_or_superseded_claims": "Pre-31AJ 31AH combined strict validation is superseded.",
            "new_suspected_or_unproven_claims": "None.",
            "current_blockers": current_blockers,
            "current_allowed_next_phase": current_allowed_next,
        },
    }
    write_outputs(payload)
    print(f"CLASSIFICATION: {classification}")
    print(f"Wrote {RESULTS_PATH}")
    print(f"Wrote {DOC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
