#!/usr/bin/env python3
"""
Phase 31AQ — Layer0 Full MLP Q2_K + Low-k Residual

REPRODUCIBILITY NOTE:
This file is a STUB. No standalone reproduction script was committed for Phase 31AQ.

How 31AQ was produced:
  - Phase 31AQ reused the 31AP official llama.cpp Q2_K quantizer path via ctypes.
  - The actual test runner lived in .venv/lib/python*/site-packages/ (not committed).
  - Authoritative results: docs/PHASE31AQ_LAYER0_MLP_Q2K_LOWK.md
  - Machine-readable results: results/PHASE31AQ_LAYER0_MLP_Q2K_LOWK.json

MAE sign convention:
  MAE_delta = MAE_sub - MAE_low
  Negative MAE_delta means residual-on IMproved MAE (lower error).
  Positive MAE_delta means residual-on Worsened MAE (higher error).

This stub cannot reproduce 31AQ. Do not run it.
"""
import sys

AUTHORITATIVE_RESULTS = "docs/PHASE31AQ_LAYER0_MLP_Q2K_LOWK.md"
MACHINE_RESULTS = "results/PHASE31AQ_LAYER0_MLP_Q2K_LOWK.json"

def main():
    print("Phase 31AQ stub — cannot reproduce.")
    print(f"See results: {AUTHORITATIVE_RESULTS}")
    print(f"See data:   {MACHINE_RESULTS}")
    sys.exit(1)

if __name__ == "__main__":
    main()