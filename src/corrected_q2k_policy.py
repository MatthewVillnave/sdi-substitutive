#!/usr/bin/env python3
"""
Corrected Q2_K Policy Package — Constants Helper

Phase: 31BO
Package version: corrected_q2k_policy_v1
Checkpoint: phase31bn-corrected-q2k-full-aggregate-checkpoint (0304590c)

This module exposes selected policy constants and a lightweight describe_policy()
function. No model loading, no llama.cpp, no external dependencies beyond stdlib.
"""

# ─── Package Identity ──────────────────────────────────────────────────────────
POLICY_VERSION = "corrected_q2k_policy_v1"
POLICY_CREATED_PHASE = "31BO"

# ─── Checkpoint ────────────────────────────────────────────────────────────────
CHECKPOINT_TAG = "phase31bn-corrected-q2k-full-aggregate-checkpoint"
CHECKPOINT_TARGET_COMMIT = "0304590c92d43fdf48d3d28998255d39c9a20c07"

# ─── Selected Policy ──────────────────────────────────────────────────────────
Q2K_MODE = "corrected_ceil_per_row"
RESIDUAL_FAMILIES = ["ffn_up", "ffn_gate"]
DOWN_RESIDUAL_ENABLED = False          # ffn_down has no residual under selected policy
RESIDUAL_K_PCT = 0.5                   # 0.5% of channel elements per family
ALPHA = 1.0

# ─── Tensor Orientation ───────────────────────────────────────────────────────
TENSOR_SHAPE = ("d_out", "d_in")       # artifact shape; computation is Y = X @ W.T

# ─── Required Environment Variables ────────────────────────────────────────────
REQUIRED_ENV_VARS = [
    "SDI_GGUF_MODEL_PATH",
    "SDI_LLAMA_CPP_ROOT",
    "SDI_LLAMA_CPP_LIB",
]

# ─── Forbidden Claims ─────────────────────────────────────────────────────────
FORBIDDEN_CLAIMS = [
    "no model quality recovery claim",
    "no behavior recovery claim",
    "no speedup claim",
    "no full-model runtime memory savings claim",
    "no llama.cpp integration claim",
    "no production readiness claim",
    "no inference/generation claim",
    "no larger-model claim",
    "no runtime-ready output-residual claim",
    "no claim beyond standalone tensor harness",
    "no claim that 31AY/31BA exact anchors are current canonical metrics",
]

# ─── Validation Scope ─────────────────────────────────────────────────────────
VALIDATION_MODEL = "Qwen2.5-0.5B"
VALIDATION_LAYERS = list(range(24))    # L0 – L23
VALIDATION_SEEDS = list(range(16))     # 0 – 15
VALIDATION_TOTAL_PAIRS = 384           # 24 layers × 16 seeds


# ─── Policy Describe ───────────────────────────────────────────────────────────
def describe_policy() -> str:
    """Return a human-readable description of the selected policy."""
    return "\n".join([
        f"Corrected Q2_K Policy Package — {POLICY_VERSION}",
        f"Checkpoint: {CHECKPOINT_TAG} @ {CHECKPOINT_TARGET_COMMIT}",
        f"",
        f"  Q2_K mode:        {Q2K_MODE}",
        f"  Residual families: {RESIDUAL_FAMILIES}",
        f"  Residual k:       {RESIDUAL_K_PCT}%",
        f"  Alpha:             {ALPHA}",
        f"  ffn_down residual: {'enabled' if DOWN_RESIDUAL_ENABLED else 'disabled'}",
        f"",
        f"  Validation scope: {VALIDATION_MODEL}, L{VALIDATION_LAYERS[0]}–L{VALIDATION_LAYERS[-1]}, seeds 0–15",
        f"  Total pairs:       {VALIDATION_TOTAL_PAIRS}",
    ])


# ─── Policy Validate ───────────────────────────────────────────────────────────
def validate_policy_dict(policy: dict) -> tuple[bool, str]:
    """
    Lightweight validation of a policy dict loaded from JSON.

    Returns (ok, reason) where ok is True if valid, False otherwise.
    Does not load models or external libraries.
    """
    errors = []

    # version
    if policy.get("package_version") != POLICY_VERSION:
        errors.append(
            f"package_version mismatch: got {policy.get('package_version')!r}, "
            f"expected {POLICY_VERSION!r}"
        )

    sp = policy.get("selected_policy", {})
    if sp.get("q2k_mode") != Q2K_MODE:
        errors.append(f"q2k_mode: got {sp.get('q2k_mode')!r}, expected {Q2K_MODE!r}")
    if sp.get("residual_families") != RESIDUAL_FAMILIES:
        errors.append(
            f"residual_families: got {sp.get('residual_families')!r}, "
            f"expected {RESIDUAL_FAMILIES!r}"
        )
    if sp.get("down_family_residual") != DOWN_RESIDUAL_ENABLED:
        errors.append(
            f"down_family_residual: got {sp.get('down_family_residual')!r}, "
            f"expected {DOWN_RESIDUAL_ENABLED!r}"
        )
    if sp.get("k_pct") != RESIDUAL_K_PCT:
        errors.append(
            f"k_pct: got {sp.get('k_pct')!r}, expected {RESIDUAL_K_PCT!r}"
        )
    if sp.get("alpha") != ALPHA:
        errors.append(f"alpha: got {sp.get('alpha')!r}, expected {ALPHA!r}")

    if errors:
        return False, "; ".join(errors)
    return True, "ok"


if __name__ == "__main__":
    print(describe_policy())