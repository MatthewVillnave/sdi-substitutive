#!/usr/bin/env python3
"""
Lightweight policy constants smoke test.
Does NOT require model files, llama.cpp, or numpy.

Run standalone:
    python tests/test_corrected_q2k_policy.py

Or import and call validate():
    from tests.test_corrected_q2k_policy import validate; validate()
"""

import json
import os
import sys

# ─── Paths ─────────────────────────────────────────────────────────────────────
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO, "src")
RESULTS_DIR = os.path.join(REPO, "src", "results")
sys.path.insert(0, SRC_DIR)

from corrected_q2k_policy import (
    POLICY_VERSION,
    CHECKPOINT_TAG,
    CHECKPOINT_TARGET_COMMIT,
    Q2K_MODE,
    RESIDUAL_FAMILIES,
    DOWN_RESIDUAL_ENABLED,
    RESIDUAL_K_PCT,
    ALPHA,
    describe_policy,
    validate_policy_dict,
)

# ─── Known-good values ─────────────────────────────────────────────────────────
EXPECTED = {
    "package_version": "corrected_q2k_policy_v1",
    "checkpoint_tag": "phase31bn-corrected-q2k-full-aggregate-checkpoint",
    "checkpoint_target_commit": "0304590c92d43fdf48d3d28998255d39c9a20c07",
    "q2k_mode": "corrected_ceil_per_row",
    "residual_families": ["ffn_up", "ffn_gate"],
    "down_family_residual": False,
    "k_pct": 0.5,
    "alpha": 1.0,
}


def validate() -> tuple[bool, str]:
    """Run all validations. Returns (ok, reason)."""
    errors = []

    # 1. Constants match expected values
    for key, expected_val in EXPECTED.items():
        actual = locals().get(key) or globals().get(key)
        if actual != expected_val:
            errors.append(f"constant {key}: got {actual!r}, expected {expected_val!r}")

    # 2. RESIDUAL_FAMILIES is exactly two items, no ffn_down
    if set(RESIDUAL_FAMILIES) != {"ffn_up", "ffn_gate"}:
        errors.append(f"RESIDUAL_FAMILIES must be exactly ['ffn_up', 'ffn_gate'], got {RESIDUAL_FAMILIES}")

    # 3. DOWN_RESIDUAL_ENABLED must be False
    if DOWN_RESIDUAL_ENABLED is not False:
        errors.append(f"DOWN_RESIDUAL_ENABLED must be False, got {DOWN_RESIDUAL_ENABLED}")

    # 4. k must be 0.5 and alpha must be 1.0
    if RESIDUAL_K_PCT != 0.5:
        errors.append(f"RESIDUAL_K_PCT must be 0.5, got {RESIDUAL_K_PCT}")
    if ALPHA != 1.0:
        errors.append(f"ALPHA must be 1.0, got {ALPHA}")

    # 5. Checkpoint tag must be non-empty string
    if not CHECKPOINT_TAG or not isinstance(CHECKPOINT_TAG, str):
        errors.append(f"CHECKPOINT_TAG must be a non-empty string, got {CHECKPOINT_TAG!r}")

    # 6. validate_policy_dict must pass on the package JSON
    pkg_path = os.path.join(RESULTS_DIR, "CORRECTED_Q2K_POLICY_PACKAGE.json")
    if os.path.exists(pkg_path):
        with open(pkg_path) as f:
            pkg = json.load(f)
        ok_dict, reason_dict = validate_policy_dict(pkg)
        if not ok_dict:
            errors.append(f"validate_policy_dict failed: {reason_dict}")
    else:
        errors.append(f"policy package JSON not found at {pkg_path}")

    if errors:
        return False, "; ".join(errors)
    return True, "ok"


def main() -> int:
    print(describe_policy())
    print()
    ok, reason = validate()
    print(f"Policy constants smoke test: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"  FAIL reason: {reason}")
        return 1

    # Pretty-print key values
    print(f"  ✓ POLICY_VERSION={POLICY_VERSION!r}")
    print(f"  ✓ CHECKPOINT_TAG={CHECKPOINT_TAG!r}")
    print(f"  ✓ CHECKPOINT_TARGET_COMMIT={CHECKPOINT_TARGET_COMMIT!r}")
    print(f"  ✓ Q2K_MODE={Q2K_MODE!r}")
    print(f"  ✓ RESIDUAL_FAMILIES={RESIDUAL_FAMILIES!r}")
    print(f"  ✓ DOWN_RESIDUAL_ENABLED={DOWN_RESIDUAL_ENABLED}")
    print(f"  ✓ RESIDUAL_K_PCT={RESIDUAL_K_PCT}")
    print(f"  ✓ ALPHA={ALPHA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())