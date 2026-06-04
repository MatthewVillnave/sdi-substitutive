#!/usr/bin/env python3
"""
phase31bw_broader_layer_probe_planning.py

Phase 31BW — Qwen2.5-1.5B Corrected Q2_K Broader Layer Probe Planning.

PLANNING-ONLY. This script does NOT run validation. It does NOT generate
Q2_K / SDIR artifacts. It does NOT load the 1.5B model. It only:
- loads the 31BU and 31BV result JSONs to ground cost estimates
- evaluates four candidate next-executable scopes (A, B, C, D)
- picks one recommended next phase (Option A — 7-layer stratified, per Matt's
  stated preference unless data argues otherwise)
- writes the planning JSON
- does not commit, not push, not stage any other files

Forbidden: no validation execution, no generation, no inference, no runtime
integration, no quality/performance claims, no commit/push/tag without approval.
"""

import json
import os
from pathlib import Path

REPO_DIR = "/home/matthew-villnave/sdi-substitutive"
RES_31BU = os.path.join(REPO_DIR, "src/results/PHASE31BU_1_5B_CORRECTED_Q2K_ANCHOR_PROBE.json")
RES_31BV = os.path.join(REPO_DIR, "src/results/PHASE31BV_1_5B_CORRECTED_Q2K_SMALL_MULTILAYER_PROBE.json")
OUT_JSON = os.path.join(REPO_DIR, "src/results/PHASE31BW_BROADER_LAYER_PROBE_PLANNING.json")

# ─── 31BW planning inputs (from 31BU / 31BV, no new validation) ──────────
# Observed 31BV per-pair time (wall-clock from background process logs):
# 31BV ran 6 pairs in ~240 sec wall clock (started 14:01:11, exited 14:05:11).
# 31BV also did a final load_layer_mlp pass for the per-layer summary block
# (~3 layers x ~1 sec each), so the per-pair cost includes some serial
# Q2_K encode overhead. Net: ~30 sec/pair for the Q2_K + SDIR + MLP pipeline
# on the optiplex. 31BU (3 pairs) took similar per-pair time.
OBSERVED_31BV_PAIRS = 6
OBSERVED_31BV_TOTAL_SEC = 240
OBSERVED_PER_PAIR_SEC = OBSERVED_31BV_TOTAL_SEC / OBSERVED_31BV_PAIRS  # ~40 sec/pair
# Memory footprint (1.5B dequant in float32):
# ffn_up/ffn_gate 8960 x 1536 float32 = 55 MB each
# ffn_down 1536 x 8960 float32 = 55 MB
# 3 dequant W_ref families resident at once = ~165 MB
# Plus 3 W_low families = ~165 MB
# Plus 3 R families = ~165 MB
# Plus dec_up/dec_gate after SDIR decode = ~165 MB
# Per-anchor working set ~ 660 MB; pipeline peak ~1.5 GB
PEAK_PER_PAIR_RESIDENT_MB = 1500

# Q4_budget per layer (1.5B): 3 * 6,881,280 = 20,643,840 bytes
# Observed per-layer margin at L0/L14/L27: +3,380,350 to +3,380,376 bytes
# Variance across selected layers: 26 bytes (negligible)
OBSERVED_PER_LAYER_MARGIN_BYTES = 3_380_350
Q4_BUDGET_LAYER_BYTES = 20_643_840

# 1.5B has 28 FFN layers (0-27), per 31BS metadata
N_LAYERS_1_5B = 28


def option_a():
    return {
        "name": "Option A — 7-layer stratified probe",
        "layers": [0, 4, 8, 12, 16, 20, 27],
        "seeds": [0, 9],
        "total_pairs": 7 * 2,
        "purpose": "broad coverage without full sweep; stratified ~every 4 layers + final layer (27) preserves top-of-network, mid, and end positions",
    }


def option_b():
    return {
        "name": "Option B — 14-layer half-model probe",
        "layers": list(range(0, 28, 2)),   # 0, 2, 4, …, 26
        "seeds": [0, 9],
        "total_pairs": 14 * 2,
        "purpose": "stronger coverage (every other layer); still not full aggregate",
    }


def option_c():
    return {
        "name": "Option C — full 28-layer aggregate",
        "layers": list(range(0, 28)),
        "seeds": [0, 9],
        "total_pairs": 28 * 2,
        "purpose": "full layer coverage at two seeds; higher runtime and scope risk",
    }


def option_d():
    return {
        "name": "Option D — seed sensitivity probe",
        "layers": [0, 14, 27],
        "seeds": [0, 1, 2, 3, 4, 5, 9, 13],
        "total_pairs": 3 * 8,
        "purpose": "seed/activation sensitivity before broader layers; uses 31BV's three proven layers with a wider seed set",
    }


def estimate(option, per_pair_sec=OBSERVED_PER_PAIR_SEC, peak_mb=PEAK_PER_PAIR_RESIDENT_MB):
    """Return cost/risk table for an option. Pure computation, no execution."""
    pairs = option["total_pairs"]
    runtime_sec = pairs * per_pair_sec
    runtime_min = runtime_sec / 60.0
    peak_resident_gb = peak_mb / 1024.0
    # Temp artifacts per pair (Q2_K blobs in memory; SDIR blobs in memory; no disk writes)
    # All in tempfile.mkdtemp + numpy arrays; cleaned at runner exit
    # Resident peak = pairs * peak_per_pair_resident (serial, not parallel)
    return {
        "total_pairs": pairs,
        "estimated_runtime_sec": round(runtime_sec, 1),
        "estimated_runtime_min": round(runtime_min, 1),
        "estimated_runtime_human": f"~{runtime_min:.0f} min ({pairs} pairs × ~{per_pair_sec:.0f} sec/pair from 31BV observation)",
        "peak_resident_mb_per_pair": peak_mb,
        "peak_resident_gb_per_pair": round(peak_resident_gb, 2),
        "serial_execution": True,
        "disk_artifacts_committed": "none (all in tempfile.mkdtemp + memory; cleaned at exit, per 31BU/31BV pattern)",
        "is_aggregate_validation": pairs >= 28,  # 28+ pairs * 28 layers = aggregate-like
        "scope_creep_risk": (
            "low" if pairs <= 14 else
            "medium" if pairs <= 28 else
            "high"
        ),
        "claim_boundary_if_pass": (
            "anchor probe result only" if pairs <= 7 else
            "small stratified probe result only" if pairs <= 14 else
            "approaching aggregate — would qualify as 'broader probe', not full aggregate validation"
        ),
        "claim_boundary_if_fail": (
            "no claim; results preserved as audit context for next phase planning"
        ),
    }


def main():
    # ─── Grounding: 31BU + 31BV data ─────────────────────────────────────
    with open(RES_31BU) as f:
        r_31bu = json.load(f)
    with open(RES_31BV) as f:
        r_31bv = json.load(f)

    # ─── Candidate options ──────────────────────────────────────────────
    options = {
        "A": option_a(),
        "B": option_b(),
        "C": option_c(),
        "D": option_d(),
    }
    for k, opt in options.items():
        opt["estimate"] = estimate(opt)

    # ─── Selection logic ────────────────────────────────────────────────
    # Decision drivers (data-driven, not just preference):
    # 1. 31BV worst pair: L0-S9 dc=+0.001090. L14 has the smallest mean_dc
    #    (+0.002659) but still positive. L27 has the largest (+0.004909).
    #    This suggests activation-sensitivity is layer-dependent (L14 is
    #    closer to Q2_K's ceiling than L0/L27 at these seeds).
    # 2. Per-layer margin is essentially constant across the 3 selected
    #    layers (variance 26 bytes). This suggests memory economics are
    #    NOT the binding constraint for broader sweeps.
    # 3. 31BV runner took ~240 sec for 6 pairs (40 sec/pair) on optiplex.
    #    That scaling gives us:
    #       A: 14 pairs * 40 sec = ~9 min
    #       B: 28 pairs * 40 sec = ~19 min
    #       C: 56 pairs * 40 sec = ~37 min
    #       D: 24 pairs * 40 sec = ~16 min
    # 4. Claim boundary is the binding constraint. Option C is "full
    #    layer coverage at 2 seeds" which crosses into aggregate territory
    #    per the planning rubric. Options A and D stay clearly under.
    #    Option B is borderline (14 of 28 layers = half-model coverage).
    # 5. 31BV used 3 layers (top, mid, end). Option D's seed sensitivity
    #    probe is useful but doesn't add layer coverage — would re-confirm
    #    what 31BV already shows (L0 worst, L14 mildest, L27 best).
    # 6. Option A gives 7 layers (top, ~1/4, ~1/3, mid, ~2/3, ~3/4, end)
    #    which covers the same positional diversity 31BV had, but with
    #    more granularity. Each new layer adds 2 pairs (40 sec each =
    #    80 sec/layer = ~80 sec * 7 = ~9 min total).

    selection = {
        "selected_option": "A",
        "selected_name": options["A"]["name"],
        "selected_layers": options["A"]["layers"],
        "selected_seeds": options["A"]["seeds"],
        "total_pairs": options["A"]["total_pairs"],
        "rationale": [
            "31BU + 31BV together show the policy is robust at layer 0, layer 14, and layer 27 (3 of 28 layers = ~10.7% coverage, all pass).",
            "Per-layer margin variance is 26 bytes across 3 layers → memory economics are NOT the binding constraint; broader sweeps are safe on memory.",
            "31BV ran 6 pairs in ~240 sec wall clock (~40 sec/pair). Option A = 14 pairs ≈ 9 min, well within tolerance for a single-session run.",
            "Option A is stratified sampling: 0, 4, 8, 12, 16, 20, 27 = covers positions L0 (top), L4, L8, L12, L16, L20, L27 (end). 7 layers with stride 4 plus the final layer preserves top/mid/end sampling.",
            "Option A's claim boundary stays clearly 'stratified probe' (14 pairs, 7 of 28 layers = 25% layer coverage). It does NOT cross into aggregate territory (which would require 28+ layers at multiple seeds).",
            "Option B (14 layers) would cross into 'half-model probe' territory — useful but a larger scope step from 31BV (3 layers → 14 layers = 4.7x). 31BV was 2x scope step from 31BU. Stepping 2x → 2x → 4.7x is uneven.",
            "Option C (full 28 layers) would be aggregate validation territory — explicitly forbidden by 31BW spec.",
            "Option D (seed sensitivity) does not add layer coverage; it re-tests the same 3 layers 31BV already tested. Useful as a follow-up, but lower priority than layer coverage.",
        ],
        "rejected_options": {
            "B": "Step from 3 → 14 layers is too large for one phase; 31BV was 1 → 3 layers. Save B for after A passes, as a follow-up.",
            "C": "Full 28-layer sweep = aggregate validation, explicitly forbidden by 31BW spec. Would also be ~37 min wall clock and cross the claim boundary.",
            "D": "Seed sensitivity probe does not add layer coverage; re-tests 3 layers 31BV already covered. Lower priority than layer coverage step.",
        },
    }

    # ─── Success criteria for next phase (31BX) ─────────────────────────
    success_criteria = {
        "regression_passes": "PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN, error_count=0, fallback_count=0, readme_drift_guard.passed=true (before AND after)",
        "all_pairs_finite": "All 14 pairs produce finite Y_ref, Y_low, Y_sub",
        "all_pairs_memory_positive": "All 14 pairs memory-positive (per-layer margin >= 0)",
        "no_severe_regressions": "0 pairs with delta_cos < -0.05",
        "cosine_improved_majority": ">= 12/14 pairs have delta_cos >= 0",
        "MAE_improved_majority": ">= 12/14 pairs have MAE_delta < 0",
        "no_model_files_committed": "model file remains outside repo",
        "no_blobs_committed": "no Q2_K / SDIR / temp blobs in commit",
        "no_scope_creep": "layers and seeds exactly as approved; no layer added, no seed added",
    }

    # ─── Classification rules for next phase (31BX) ─────────────────────
    classification_rules = {
        "PASS_31BX_1_5B_Q2K_STRATIFIED_LAYER_PROBE_CLEAN": (
            "all 14 pairs pass all criteria; no severe regressions; "
            "all finite; all memory-positive; >= 12/14 cosine-improved; "
            ">= 12/14 MAE-improved"
        ),
        "PARTIAL_31BX_1_5B_Q2K_STRATIFIED_LAYER_PROBE_MINOR_FAILURES": (
            "all pairs finite and memory-positive; some pairs have minor "
            "cosine/MAE failures but no severe regressions"
        ),
        "PARTIAL_31BX_1_5B_Q2K_STRATIFIED_LAYER_PROBE_TRADEOFF": (
            "cosine/MAE tradeoffs present (some layers improve cosine, "
            "some improve MAE, no layer severely regresses both)"
        ),
        "PARTIAL_31BX_1_5B_Q2K_STRATIFIED_LAYER_PROBE_MEMORY_FAIL": (
            "any selected layer is memory-negative under current accounting"
        ),
        "BLOCKED_31BX_SDI_MODEL_DIR_UNSET": "env var missing",
        "BLOCKED_31BX_MODEL_FILE_MISSING": "model file missing",
        "BLOCKED_31BX_Q2K_BACKEND_FAIL": "corrected Q2_K generation/dequantization fails",
        "BLOCKED_31BX_SDIR_FAIL": "SDIR residual generation/application fails",
        "BLOCKED_SOURCE_OF_TRUTH_REGRESSION": "regression fails",
    }

    # ─── Stop conditions for next phase (31BX) ──────────────────────────
    stop_conditions = [
        "regression failure at any point (pre-flight or post-edit)",
        "model file tracking issue (file missing, wrong file, different quantization)",
        "non-finite Y_ref / Y_low / Y_sub on any pair",
        "severe regression (delta_cos < -0.05) on any pair",
        "memory-negative on any layer",
        "scope creep (extra layer, extra seed, extra family, aggregate-style sweep)",
        "any claim trigger: inference, generation, runtime, production, speedup, quality, behavior, llama.cpp integration, larger-model validation",
        "Q2_K / SDIR / temp blobs committed",
    ]

    # ─── Allowed claims if 31BX passes ──────────────────────────────────
    allowed_claims_if_pass = {
        "primary": "Corrected Q2_K policy remains memory-positive and improves cosine/MAE across a stratified 1.5B standalone tensor-harness probe (7 layers: 0, 4, 8, 12, 16, 20, 27; 2 seeds: 0, 9; 14 pairs).",
        "secondary": [
            "Per-layer margin is consistent across the stratified layer set (variance bounded by 31BV's 26 bytes/layer observation).",
            "L0 result reproduces 31BU and 31BV (cross-runner reproducibility).",
            "The mixed source-GGUF ffn_down quant types (L0/L27 Q6_K, L14 Q4_K) do not affect corrected Q2_K memory accounting.",
        ],
        "forbidden_still": [
            "no full larger-model validation claim (7 of 28 layers ≠ 28 layers)",
            "no inference / generation claim",
            "no speedup claim",
            "no quality / behavior recovery claim",
            "no runtime claim",
            "no production claim",
            "no claim that 1.5B behaves like 0.5B (no 0.5B comparison)",
        ],
    }

    # ─── Assemble planning JSON ─────────────────────────────────────────
    result = {
        "phase": "31BW",
        "title": "Phase 31BW — Qwen2.5-1.5B Corrected Q2_K Broader Layer Probe Planning",
        "scope": "planning-only. No validation execution in 31BW. No model load. No Q2_K / SDIR artifacts. No generation / inference / runtime integration. No commit/push/tag without explicit Matt approval.",
        "forbidden_actions_upheld": [
            "no aggregate validation execution",
            "no full 28-layer sweep execution",
            "no generation / inference / sampling",
            "no llama.cpp runtime integration",
            "no performance claim",
            "no model quality claim",
            "no model files committed",
            "no Q2_K / SDIR blobs generated or committed",
            "no commit/push/tag without explicit Matt approval",
        ],
        "planning_inputs": {
            "31BU_n_pairs": r_31bu["summary"]["n_pairs"],
            "31BU_classification": r_31bu["classification"],
            "31BU_min_margin_bytes": r_31bu["summary"]["min_per_layer_margin"],
            "31BV_n_pairs": r_31bv["summary"]["n_pairs"],
            "31BV_n_layers": r_31bv["summary"]["n_layers"],
            "31BV_classification": r_31bv["classification"],
            "31BV_min_margin_bytes": r_31bv["summary"]["min_per_layer_margin"],
            "31BV_per_layer_margin_variance_bytes": (
                max(ps["min_per_layer_margin"] for ps in r_31bv["per_layer_summary"].values())
                - min(ps["min_per_layer_margin"] for ps in r_31bv["per_layer_summary"].values())
            ),
            "31BV_worst_pair": r_31bv["summary"]["worst_pair"],
            "31BV_mean_dc": r_31bv["summary"]["mean_delta_cos"],
            "31BV_mean_MAE_imp": r_31bv["summary"]["mean_MAE_improvement"],
            "observed_per_pair_sec_31BV": OBSERVED_PER_PAIR_SEC,
            "observed_31BV_total_sec": OBSERVED_31BV_TOTAL_SEC,
            "Q4_budget_layer_bytes_1_5B": Q4_BUDGET_LAYER_BYTES,
            "n_layers_1_5B": N_LAYERS_1_5B,
            "model_path_redacted": "$SDI_MODEL_DIR/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        },
        "options_considered": options,
        "cost_risk_table": {
            k: opt["estimate"] for k, opt in options.items()
        },
        "selection": selection,
        "selected_next_phase": {
            "name": "Phase 31BX — Qwen2.5-1.5B Corrected Q2_K Stratified Layer Probe",
            "scope": "small stratified probe, not aggregate",
            "layers": selection["selected_layers"],
            "seeds": selection["selected_seeds"],
            "total_pairs": selection["total_pairs"],
            "policy": "corrected_q2k_policy_v1 (corrected_ceil_per_row Q2_K, ffn_up+ffn_gate SDIR k=0.5%, alpha=1.0, ffn_down W_low only — no ffn_down residual)",
            "W_ref": "Qwen2.5-1.5B-Instruct Q4_K_M dequantized (NOT FP16)",
            "no_generation": True,
            "no_runtime_integration": True,
            "no_full_aggregate": True,
            "no_quality_performance_claims": True,
        },
        "success_criteria_for_31BX": success_criteria,
        "classification_rules_for_31BX": classification_rules,
        "stop_conditions_for_31BX": stop_conditions,
        "allowed_claims_if_31BX_passes": allowed_claims_if_pass,
        "interpretation": (
            "31BW is a planning phase: it does not run validation. It evaluates four "
            "candidate next-scope options (A=7-layer stratified, B=14-layer half-model, "
            "C=full 28-layer aggregate, D=seed sensitivity probe) using grounded cost "
            "estimates from 31BV's observed per-pair time (~40 sec) and per-layer margin "
            "variance (26 bytes). It recommends Option A (7 layers × 2 seeds = 14 pairs) "
            "as the safest next executable phase: stratified layer coverage with stride 4 "
            "+ final layer, scope step 2x from 31BV (3 → 7 layers), runtime ~9 min, and a "
            "claim boundary that stays clearly 'stratified probe' rather than approaching "
            "aggregate. Options B, C, D are explicitly rejected with rationale. The next "
            "allowed phase (if 31BW is approved) is 31BX."
        ),
        "classification": "PASS_31BW_BROADER_LAYER_PROBE_PLAN_SELECTED",
        "classification_reason": (
            f"Option A selected cleanly: 7 layers (0, 4, 8, 12, 16, 20, 27) × 2 seeds = 14 pairs, "
            f"estimated ~9 min wall clock (14 pairs × ~40 sec/pair from 31BV observation), "
            f"no aggregate territory, no scope creep, claim boundary stays 'stratified probe'."
        ),
        "next_allowed_phase_if_clean": (
            "Phase 31BX — Qwen2.5-1.5B Corrected Q2_K Stratified Layer Probe, only if explicitly requested"
        ),
        "next_allowed_phase_if_blocked": (
            "Phase 31BW-R — Broader Probe Planning Repair, only if explicitly requested"
        ),
    }

    # ─── Write planning JSON (no commit) ─────────────────────────────────
    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"wrote {OUT_JSON}")
    print(f"classification: {result['classification']}")
    print(f"selected: Option A — {result['selected_next_phase']['name']}")
    print(f"layers: {result['selected_next_phase']['layers']}")
    print(f"seeds: {result['selected_next_phase']['seeds']}")
    print(f"total pairs: {result['selected_next_phase']['total_pairs']}")


if __name__ == "__main__":
    main()
