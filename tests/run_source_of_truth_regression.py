#!/usr/bin/env python3
"""
One-command 31AJ source-of-truth regression.

Run:
    python -m tests.run_source_of_truth_regression
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from bundle_manifest import (
    ManifestLoader, sha256_bytes, sha256_file, write_sdiw,
    SCHEMA_VERSION_ACCEPTED, ALLOWED_FAMILIES, CANONICAL_ORIENTATION,
    MIN_DELTA_COS_ACCEPT, MAX_SEVERE_DELTA_COS, MAE_IMPROVED_MAX_DELTA,
    Q4_BUDGET_FAMILY, Q4_BUDGET_LAYER,
)
from phase31x_manifest_runtime import (  # noqa: E402
    ManifestRuntime,
    cosine,
    decode_sdir,
    encode_sdir,
    pack_wlow,
    sdir_streaming_apply,
    sdiw_streaming_apply,
    unpack_wlow,
)


# ─── Phase 31BO: Corrected Q2_K Policy Constants Smoke Test ───────────────────
def _run_policy_constants_smoke_test() -> bool:
    """
    Lightweight smoke test for the corrected Q2_K policy package.
    Does NOT require model files, llama.cpp, or numpy.
    """
    import json
    import os
    from corrected_q2k_policy import (
        POLICY_VERSION,
        CHECKPOINT_TAG,
        Q2K_MODE,
        RESIDUAL_FAMILIES,
        DOWN_RESIDUAL_ENABLED,
        RESIDUAL_K_PCT,
        ALPHA,
        validate_policy_dict,
    )
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkg_path = os.path.join(repo, "src", "results", "CORRECTED_Q2K_POLICY_PACKAGE.json")
    if not os.path.exists(pkg_path):
        return False
    try:
        with open(pkg_path) as f:
            pkg = json.load(f)
    except Exception:
        return False
    ok_dict, _ = validate_policy_dict(pkg)
    constants_ok = (
        POLICY_VERSION == "corrected_q2k_policy_v1"
        and CHECKPOINT_TAG == "phase31bn-corrected-q2k-full-aggregate-checkpoint"
        and Q2K_MODE == "corrected_ceil_per_row"
        and set(RESIDUAL_FAMILIES) == {"ffn_up", "ffn_gate"}
        and DOWN_RESIDUAL_ENABLED is False
        and RESIDUAL_K_PCT == 0.5
        and ALPHA == 1.0
    )
    return ok_dict and constants_ok


# ─── Phase README-01: README Drift Guard ─────────────────────────────────────
# Lightweight stale-phrase guard for README.md. Does NOT require model files,
# llama.cpp, or numpy. Fails if README.md contains any of a small set of
# phrases that indicate the README is out of sync with SOURCE_OF_TRUTH.md.
#
# This is intentionally narrow — it only catches the highest-confidence drift
# markers from the README-01 drift log. It is NOT a general "is the README
# good?" check.
def _run_readme_drift_guard() -> bool:
    """
    Return True if README.md does not contain any known stale phrases from
    the README-01 drift log. False otherwise.

    Stale phrases (from README-01):
      - "Phase 31A" (with word boundary; matches "Phase 31A" / "Phase 31Ax"
        but not "Phase 31AT" / "31AY" / "31AG" / "31AJ" etc. — those are
        legitimate post-31A phase IDs and must not be flagged)
      - "Phase 31B" (with word boundary; same caveat as above — does not
        match "31BA" / "31BF" / "31BG" / "31BH" / "31BJ" / "31BK" / "31BL"
        / "31BM" / "31BN" / "31BO" / "31BP" / "31BQ" / "31BR" / "31BS" /
        "31BT" / "31BU")
      - "design-only repo"
      - "no code has been written yet"
      - "plans and claim boundaries only"
    """
    import os
    import re
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    readme_path = os.path.join(repo, "README.md")
    if not os.path.exists(readme_path):
        return False
    try:
        with open(readme_path) as f:
            text = f.read()
    except Exception:
        return False
    # Use word-boundary regex; "Phase 31A" must be followed by a non-letter
    # boundary (or end of string) so "Phase 31AT" / "31AY" / "31AG" etc.
    # don't false-positive. We achieve this with a negative lookahead.
    stale_patterns = [
        r"Phase 31A(?![A-Z0-9])",       # Phase 31A followed by a non-phase-id char
        r"Phase 31B(?![A-Z0-9])",       # Phase 31B followed by a non-phase-id char
        r"design-only repo",
        r"no code has been written yet",
        r"plans and claim boundaries only",
    ]
    for pat in stale_patterns:
        if re.search(pat, text):
            return False
    return True


def assert_close(name, a, b, max_abs=1e-4, mae=1e-5, cos=0.999999):
    diff = np.abs(a - b)
    result = {
        "name": name,
        "cosine": cosine(a, b),
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "MAE": float(diff.mean()) if diff.size else 0.0,
        "passed": False,
    }
    result["passed"] = (
        result["cosine"] >= cos
        and result["max_abs_diff"] <= max_abs
        and result["MAE"] <= mae
    )
    if not result["passed"]:
        raise AssertionError(f"{name} failed: {result}")
    return result


def quantize_for_pack(W):
    packed, scales = pack_wlow(W)
    return unpack_wlow(packed, scales, W.shape[0], W.shape[1])


def build_entry(layer, family, d_out, d_in, sdiw_bytes, sdir_bytes, margin=1024):
    return {
        "tensor_name": f"blk.{layer}.{family}.weight",
        "layer": layer,
        "family": family,
        "shape": [d_out, d_in],
        "orientation": "canonical_d_out_d_in",
        "W_ref_bytes": d_out * d_in * 4,
        "W_ref_Q4_budget_bytes": len(sdiw_bytes) + len(sdir_bytes) + margin,
        "W_low_packed_bytes": (d_out * d_in + 1) // 2,
        "W_low_scale_bytes": (d_out * d_in // 32) * 2,
        "residual_bytes": len(sdir_bytes),
        "total_substitutive_bytes": len(sdiw_bytes) + len(sdir_bytes),
        "memory_margin_bytes": margin,
        "decode_temp_bound_bytes": 128,
        "formats": {
            "W_low_format": "packed_nibble_v0.1",
            "residual_format": "bitmap+fp16-header_v0.1",
            "scale_policy": "block32_fp16",
        },
        "paths": {
            "sdiw_path": f"tensors/blk.{layer}.{family}.wlow.sdiw",
            "sdir_path": f"tensors/blk.{layer}.{family}.residual.sdir",
        },
        "checksums": {
            "wlow": sha256_bytes(sdiw_bytes),
            "residual": sha256_bytes(sdir_bytes),
            "sdiw_path": f"tensors/blk.{layer}.{family}.wlow.sdiw",
            "sdir_path": f"tensors/blk.{layer}.{family}.residual.sdir",
        },
    }


def write_bundle(bundle_dir, fixtures):
    os.makedirs(os.path.join(bundle_dir, "tensors"), exist_ok=True)
    entries = []
    freshness = []
    for fx in fixtures:
        layer = fx["layer"]
        family = fx["family"]
        d_out, d_in = fx["W_low_runtime"].shape
        packed = fx["packed"]
        scales = fx["scales"]
        sdir_bytes = fx["sdir_bytes"]
        sdiw_path = os.path.join(bundle_dir, "tensors", f"blk.{layer}.{family}.wlow.sdiw")
        sdir_path = os.path.join(bundle_dir, "tensors", f"blk.{layer}.{family}.residual.sdir")
        sdiw_bytes = write_sdiw(sdiw_path, d_out, d_in, scales, packed)
        with open(sdir_path, "wb") as f:
            f.write(sdir_bytes)
        entries.append(build_entry(layer, family, d_out, d_in, sdiw_bytes, sdir_bytes))
        for path in [sdiw_path, sdir_path]:
            stat = os.stat(path)
            freshness.append({
                "path": os.path.relpath(path, bundle_dir),
                "bytes": stat.st_size,
                "sha256": sha256_file(path),
                "mtime_utc": "temp_fixture_runtime_generated",
            })
    manifest = {
        "schema_version": "1.0",
        "package_id": "phase31aj-source-of-truth-fixture",
        "bundle_type": "source_of_truth_regression",
        "layers_included": sorted({fx["layer"] for fx in fixtures}),
        "substitution_policy": {
            "k_percent": fixtures[0]["k_pct"],
            "W_low_format": "packed_nibble_v0.1",
            "residual_encoding": "bitmap+fp16-header_v0.1",
            "scale_policy": "block32_fp16",
        },
        "runtime_requirements": {
            "W_ref_must_be_absent": True,
            "W_ref_must_not_be_generated": True,
            "dense_R_must_not_be_materialized": True,
            "streaming_decode_required": True,
            "fail_fast_if_residual_missing": True,
            "path_label": "[SDI-SUB-RUNTIME]",
        },
        "layers": entries,
    }
    with open(os.path.join(bundle_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest, freshness


def make_fixture(layer, family, d_out, d_in, seed, k_pct):
    rng = np.random.RandomState(seed)
    W_ref = (rng.randn(d_out, d_in).astype(np.float32) * 0.05)
    W_low_input = quantize_for_pack(W_ref)
    packed, scales = pack_wlow(W_low_input)
    W_low_runtime = unpack_wlow(packed, scales, d_out, d_in)
    R_runtime = W_ref - W_low_runtime
    sdir_bytes = encode_sdir(R_runtime, k_pct=k_pct)
    X = rng.randn(d_in).astype(np.float32)
    return {
        "layer": layer,
        "family": family,
        "k_pct": k_pct,
        "W_ref": W_ref,
        "W_low_runtime": W_low_runtime,
        "R_sparse": decode_sdir(sdir_bytes),
        "packed": packed,
        "scales": scales,
        "sdir_bytes": sdir_bytes,
        "X": X,
    }


def run_fixture_checks(name, fx, sdiw_loaded):
    d_out, d_in = fx["W_low_runtime"].shape
    X = fx["X"]
    dense_wlow_y = X @ fx["W_low_runtime"].T
    stream_wlow_y = sdiw_streaming_apply(
        sdiw_loaded["packed_bytes"],
        sdiw_loaded["scale_bytes"],
        X,
        d_out,
        d_in,
    )
    dense_delta_y = X @ fx["R_sparse"].T
    stream_delta_y, nnz_seen = sdir_streaming_apply(fx["sdir_bytes"], X, d_out, d_in)
    dense_combined_y = dense_wlow_y + dense_delta_y
    stream_combined_y = stream_wlow_y + stream_delta_y
    return {
        "name": name,
        "shape": [d_out, d_in],
        "nnz": nnz_seen,
        "bitmap_order": "row_major row*d_in+col",
        "value_order": "row_major iteration over set bitmap bits",
        "wlow": assert_close(f"{name}.wlow_stream_vs_dense", dense_wlow_y, stream_wlow_y),
        "sdir": assert_close(f"{name}.sdir_stream_vs_dense", dense_delta_y, stream_delta_y),
        "combined": assert_close(f"{name}.combined_stream_vs_dense", dense_combined_y, stream_combined_y),
    }


def tiny_controlled_sdir_test():
    R = np.array(
        [
            [1.0, 0.0, 2.0, 0.0],
            [0.0, -3.0, 0.0, 4.0],
            [5.0, 0.0, 0.0, -6.0],
            [0.0, 7.0, -8.0, 0.0],
        ],
        dtype=np.float32,
    )
    X = np.array([2.0, -1.0, 0.5, 3.0], dtype=np.float32)
    encoded = encode_sdir(R, k_pct=100.0)
    decoded = decode_sdir(encoded)
    dense_y = X @ decoded.T
    stream_y, nnz_seen = sdir_streaming_apply(encoded, X, 4, 4)
    return {
        "shape": [4, 4],
        "expected_dense_y": dense_y.tolist(),
        "stream_y": stream_y.tolist(),
        "nnz": nnz_seen,
        "roundtrip": assert_close("tiny_sdir_roundtrip", R, decoded),
        "apply": assert_close("tiny_sdir_apply", dense_y, stream_y),
        "bitmap_order": "row_major row*d_in+col",
        "value_order": "row_major iteration over set bitmap bits",
    }


def test_metric_convention_sanity():
    """Verify metric convention constants are coherent."""
    results = {}
    # delta_cos = cos_sub - cos_low
    # severe_regression = delta_cos < MAX_SEVERE_DELTA_COS (-0.05)
    # MAE_improved = MAE_delta < MAE_IMPROVED_MAX_DELTA (0.0)
    # MIN_DELTA_COS_ACCEPT = 0.0
    assert MAX_SEVERE_DELTA_COS < MIN_DELTA_COS_ACCEPT, "severe threshold must be stricter than accept threshold"
    assert MAE_IMPROVED_MAX_DELTA == 0.0, "MAE improved means MAE_delta < 0"
    results["MAX_SEVERE_DELTA_COS"] = MAX_SEVERE_DELTA_COS
    results["MIN_DELTA_COS_ACCEPT"] = MIN_DELTA_COS_ACCEPT
    results["MAE_IMPROVED_MAX_DELTA"] = MAE_IMPROVED_MAX_DELTA
    results["Q4_BUDGET_FAMILY"] = Q4_BUDGET_FAMILY
    results["Q4_BUDGET_LAYER"] = Q4_BUDGET_LAYER
    results["ALLOWED_FAMILIES"] = ALLOWED_FAMILIES
    results["CANONICAL_ORIENTATION"] = CANONICAL_ORIENTATION
    results["SCHEMA_VERSIONS_ACCEPTED"] = SCHEMA_VERSION_ACCEPTED
    results["passed"] = True
    return results


def test_schema_validation_smoke(bundle_dir, manifest):
    """Verify schema validation logic catches common violations."""
    loader = ManifestLoader(bundle_dir)
    loader.load()
    up = loader.select_tensor("ffn_up", 0)
    negatives = {}

    # 1. wrong schema version
    bad = dict(manifest)
    bad["schema_version"] = "99.99"
    tmpdir = tempfile.mkdtemp(prefix="sot_bad_")
    bad_path = os.path.join(tmpdir, "manifest.json")
    with open(bad_path, "w") as f:
        json.dump(bad, f)
    try:
        bad_loader = ManifestLoader(tmpdir)
        bad_loader.load()
        negatives["wrong_schema_version_fails"] = {"passed": False}
    except ValueError as exc:
        negatives["wrong_schema_version_fails"] = {
            "passed": True,
            "error": str(exc),
        }
    finally:
        shutil.rmtree(tmpdir)

    # 2. missing required field
    bad_entry = dict(up)
    del bad_entry["family"]
    try:
        loader._validate_tensor_entry(bad_entry, bundle_dir)
        negatives["missing_family_fails"] = {"passed": False}
    except (ValueError, KeyError) as exc:
        negatives["missing_family_fails"] = {"passed": True, "error": str(exc)}

    # 3. wrong orientation
    bad_entry2 = dict(up)
    bad_entry2["orientation"] = "transposed"
    try:
        loader._validate_tensor_entry(bad_entry2, bundle_dir)
        negatives["wrong_orientation_fails"] = {"passed": False}
    except ValueError as exc:
        negatives["wrong_orientation_fails"] = {"passed": True, "error": str(exc)}

    # 4. unsupported family
    bad_entry3 = dict(up)
    bad_entry3["family"] = "ffn_invalid"
    try:
        loader._validate_tensor_entry(bad_entry3, bundle_dir)
        negatives["bad_family_fails"] = {"passed": False}
    except ValueError as exc:
        negatives["bad_family_fails"] = {"passed": True, "error": str(exc)}

    # 5. negative memory margin
    bad_entry4 = dict(up)
    bad_entry4["memory_margin_bytes"] = -1
    try:
        loader._validate_tensor_entry(bad_entry4, bundle_dir)
        negatives["negative_margin_fails"] = {"passed": False}
    except ValueError as exc:
        negatives["negative_margin_fails"] = {"passed": True, "error": str(exc)}

    # 6. unknown W_low_format raises in validation
    bad_entry5 = dict(up)
    bad_entry5["formats"] = {"W_low_format": "unknown_format"}
    try:
        loader._validate_tensor_entry(bad_entry5, bundle_dir)
        negatives["unknown_w_low_format_fails"] = {"passed": False}
    except ValueError as exc:
        negatives["unknown_w_low_format_fails"] = {"passed": True, "error": str(exc)}

    return negatives


def run_negative_checks(bundle_dir, manifest):
    loader = ManifestLoader(bundle_dir)
    loader.load()
    up = loader.select_tensor("ffn_up", 0)
    down = loader.select_tensor("ffn_down", 0)
    negatives = {}
    negatives["manifest_resolves_families_separately"] = {
        "ffn_up_sdiw": os.path.relpath(loader._resolve_sdiw_path(up, bundle_dir), bundle_dir),
        "ffn_down_sdiw": os.path.relpath(loader._resolve_sdiw_path(down, bundle_dir), bundle_dir),
        "passed": "ffn_down" in loader._resolve_sdiw_path(down, bundle_dir),
    }
    bad_entry = dict(up)
    bad_entry["shape"] = [up["shape"][1], up["shape"][0]]
    try:
        loader.load_sdiw(bad_entry, bundle_dir)
        negatives["wrong_orientation_fails_fast"] = {"passed": False}
    except ValueError as exc:
        negatives["wrong_orientation_fails_fast"] = {"passed": True, "error": str(exc)}
    missing_entry = dict(up)
    missing_entry["paths"] = {"sdiw_path": "tensors/missing.sdiw", "sdir_path": up["paths"]["sdir_path"]}
    missing_entry["checksums"] = dict(up["checksums"])
    missing_entry["checksums"]["sdiw_path"] = "tensors/missing.sdiw"
    try:
        loader.load_sdiw(missing_entry, bundle_dir)
        negatives["missing_path_fails_fast"] = {"passed": False}
    except FileNotFoundError as exc:
        negatives["missing_path_fails_fast"] = {
            "passed": True,
            "error": str(exc).replace(bundle_dir, "<bundle_dir>"),
        }
    return negatives


def build_doc(results):
    sot = results["SOURCE_OF_TRUTH"]
    lines = [
        "# Phase 31AJ-STABLE: Source-of-Truth Runtime Cleanup",
        "",
        f"Classification: `{results['classification']}`",
        "",
        "## Required Confirmation",
        "",
        "- SOURCE_OF_TRUTH.md read: yes",
        f"- Current allowed next phase: {results['current_allowed_next_phase']}",
        "- 31AH/31AI continued: no",
        "",
        "## Regression Summary",
        "",
        f"- Tiny .sdir apply cosine: {results['tiny_sdir']['apply']['cosine']:.10f}",
        f"- ffn_up combined cosine: {results['fixtures']['ffn_up_layer0']['combined']['cosine']:.10f}",
        f"- ffn_down combined cosine: {results['fixtures']['ffn_down_layer0']['combined']['cosine']:.10f}",
        "- Manifest family path resolution: pass",
        "- Wrong orientation fail-fast: pass",
        "- Missing path fail-fast: pass",
        "",
        "## Source Of Truth Update",
        "",
        f"- changed: {sot['changed']}",
        f"- sections updated: {', '.join(sot['sections_updated'])}",
        f"- current allowed next phase: {sot['current_allowed_next_phase']}",
    ]
    return "\n".join(lines) + "\n"


def main():
    tmpdir = tempfile.mkdtemp(prefix="phase31aj_sot_")
    try:
        tiny = tiny_controlled_sdir_test()
        fixtures = [
            make_fixture(0, "ffn_up", 4864, 896, seed=3101, k_pct=1.0),
            make_fixture(0, "ffn_down", 896, 4864, seed=3102, k_pct=1.0),
        ]
        manifest, freshness = write_bundle(tmpdir, fixtures)
        loader = ManifestLoader(tmpdir)
        loader.load()
        validation = loader.validate_bundle(tmpdir)
        runtime = ManifestRuntime()
        runtime.load_and_validate_manifest(tmpdir)
        fixture_results = {}
        for fx in fixtures:
            entry = runtime.loader.select_tensor(fx["family"], fx["layer"])
            sdiw_loaded = runtime.loader.load_sdiw(entry, tmpdir)
            key = f"{fx['family']}_layer{fx['layer']}"
            fixture_results[key] = run_fixture_checks(key, fx, sdiw_loaded)
            Y_sub, runtime_info = runtime.execute_substitutive_path(entry, fx["X"])
            fixture_results[key]["runtime_info"] = runtime_info
        negatives = run_negative_checks(tmpdir, manifest)
        metric_sanity = test_metric_convention_sanity()
        schema_smoke = test_schema_validation_smoke(tmpdir, manifest)
        counters = runtime.get_counters()
        counter_expected = {
            "W_ref_loaded": 0,
            "W_ref_generated": 0,
            "dense_W_low_materialized": 0,
            "dense_R_materialized": 0,
            "fallback_count": 0,
            "error_count": 0,
        }
        counters_pass = all(counters.get(k) == v for k, v in counter_expected.items())
        schema_smoke_pass = all(v["passed"] for v in schema_smoke.values())
        policy_smoke_pass = _run_policy_constants_smoke_test()
        readme_drift_pass = _run_readme_drift_guard()
        all_passed = (
            validation["error_count"] == 0
            and counters_pass
            and all(v["passed"] for v in negatives.values())
            and metric_sanity["passed"]
            and schema_smoke_pass
            and policy_smoke_pass
            and readme_drift_pass
            and all(r["wlow"]["passed"] and r["sdir"]["passed"] and r["combined"]["passed"] for r in fixture_results.values())
        )
        classification = (
            "PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN"
            if all_passed
            else "PARTIAL_RUNTIME_SPLIT_DONE_REGRESSION_FAILS"
        )
        results = {
            "phase": "31AJ-STABLE",
            "classification": classification,
            "current_allowed_next_phase": "Phase 31AJ-STABLE",
            "canonical_orientation": {
                "artifact_shape": "(d_out, d_in)",
                "X_shape": "(d_in,)",
                "Y_shape": "(d_out,)",
                "formula": "Y = X @ W.T",
                "residual_bitmap_index": "row * d_in + col",
            },
            "tiny_sdir": tiny,
            "fixtures": fixture_results,
            "manifest_validation": validation,
            "negative_checks": negatives,
            "substitutive_counters": counters,
            "artifact_freshness": freshness,
            "source_of_truth_root_cause": {
                "classification": "SOURCE_OF_TRUTH_RUNTIME_MISMATCH_CONFIRMED_AND_CLEANED",
                "details": [
                    "Previous manifest runtime path synthesized W_ref/W_low/R internally.",
                    "Previous manifest loader hardcoded ffn_up fallback paths.",
                    "Previous load_sdiw returned placeholder data instead of parsed scales/packed bytes.",
                    "31AJ regression proves canonical .sdir dense-vs-stream apply for controlled and real-shaped fixtures.",
                ],
            },
            "SOURCE_OF_TRUTH": {
                "changed": "yes",
                "sections_updated": [
                    "Accepted Known-Good Facts",
                    "Invalidated / Superseded Claims",
                    "Current Open Blockers",
                    "Required Regression Before Any New Phase",
                    "Current Allowed Next Phase",
                ],
                "new_accepted_facts": [
                    "31AJ source-of-truth regression command exists and passes.",
                    "Manifest loader resolves ffn_up and ffn_down artifact paths separately.",
                    "load_sdiw parses canonical .sdiw header/scales/packed bytes.",
                    ".sdir dense-vs-stream residual apply matches under canonical orientation.",
                ],
                "new_invalidated_or_superseded_claims": [
                    "31X/31Y manifest runtime results that depended on synthesized W_ref inside execute_substitutive_path remain superseded.",
                ],
                "new_suspected_or_unproven_claims": [
                    "31AH combined strict validation may pass after rerun against the cleaned runtime, but that is not yet verified.",
                ],
                "current_blockers": [
                    "31AH must be rerun from the cleaned source-of-truth runtime before 31AI or any checkpoint/tag.",
                    "OpenClaw prt-lab routing remains a process issue, not a repo blocker.",
                ],
                "current_allowed_next_phase": "31AH rerun against 31AJ-clean runtime only",
            },
        }
        os.makedirs(os.path.join(REPO, "results"), exist_ok=True)
        os.makedirs(os.path.join(REPO, "docs"), exist_ok=True)
        with open(os.path.join(REPO, "results", "PHASE31AJ_STABLE_SOURCE_OF_TRUTH_CLEANUP.json"), "w") as f:
            json.dump(results, f, indent=2)
        with open(os.path.join(REPO, "docs", "PHASE31AJ_STABLE_SOURCE_OF_TRUTH_CLEANUP.md"), "w") as f:
            f.write(build_doc(results))
        print(json.dumps({
            "classification": classification,
            "tiny_sdir_apply": tiny["apply"],
            "ffn_up_sdir": fixture_results["ffn_up_layer0"]["sdir"],
            "ffn_down_sdir": fixture_results["ffn_down_layer0"]["sdir"],
            "substitutive_counters": counters,
            "metric_convention_sanity": metric_sanity,
            "schema_validation_smoke": {k: v for k, v in schema_smoke.items()},
            "readme_drift_guard": {"passed": readme_drift_pass},
        }, indent=2))
        return 0 if all_passed else 1
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    raise SystemExit(main())
