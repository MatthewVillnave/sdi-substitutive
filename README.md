# SDI-Substitutive — Substitutive Tensor Replacement Research

**Public orientation page.** For the current project state, accepted claims, invalidated claims, forbidden claims, and the currently allowed next phase, read [`SOURCE_OF_TRUTH.md`](./SOURCE_OF_TRUTH.md) — Section 0 (Current Working State) and Section 9 (Current Allowed Next Phase).

> **README is a high-level public orientation page, not the full audit trail.**
> If `README.md` and `SOURCE_OF_TRUTH.md` disagree, **`SOURCE_OF_TRUTH.md` wins**.

## Background

This repo follows the additive sidecar experiment (`llama.cpp experimental/prt-phase19a-alt-sidecar-backed`, tag `SDI_PRT_EXPERIMENTAL_ARCHIVE_CHECKPOINT`). That experiment produced a clear negative result: additive sidecar injection for Qwen2.5-0.5B on CPU increased memory usage rather than reducing it.

The substitutive direction tests a different thesis: instead of adding a sidecar alongside the resident tensor, replace the resident tensor with a lower-bit version plus a compressed residual that corrects the approximation error.

## Core Principle

> **Sidecar correction must replace resident cost, not add beside it.**

## Status / Drift Guard

- `README.md` is intentionally high-level. For current phase, accepted claims, invalidated claims, forbidden claims, and next allowed phase, read `SOURCE_OF_TRUTH.md` Section 0 and Section 9.
- If `README.md` and `SOURCE_OF_TRUTH.md` disagree, **`SOURCE_OF_TRUTH.md` wins**.
- This README was last updated in the **README-01** documentation maintenance interlude. The current scientific next phase (per `SOURCE_OF_TRUTH.md` Section 9) is **Phase 31CC — Qwen2.5-1.5B Real-Activation Capture Planning, only if explicitly requested** (recommended default after the 31CB stale file provenance cleanup). README-01 did not advance, replace, or invalidate the current scientific next phase. 31CB (Stale File Provenance Cleanup) was a working-tree hygiene + provenance phase, not a new scientific phase. 31CA (Qwen2.5-1.5B Corrected Q2_K Aggregate Freeze / Package Update) was a documentation + provenance + handoff phase, not a new scientific phase.

## Current State (defer to SOURCE_OF_TRUTH.md Section 0 for full details)

- **SOT version / state label:** v2 (post-31BT, SOT-01 hardened).
- **Current selected policy package:** `corrected_q2k_policy_v1` — corrected_ceil_per_row Q2_K, ffn_up + ffn_gate SDIR k=0.5%, alpha=1.0, no ffn_down residual. Documented in `docs/CORRECTED_Q2K_POLICY_PACKAGE.md` and `src/results/CORRECTED_Q2K_POLICY_PACKAGE.json`.
- **Frozen 0.5B checkpoint:** `phase31bn-corrected-q2k-full-aggregate-checkpoint` (commit `0304590c92d43fdf48d3d28998255d39c9a20c07`). 0.5B result scope: **standalone tensor harness only** (Qwen2.5-0.5B-Instruct, all 24 FFN layers, 384/384 memory-positive, 383/384 cosine-improved, 383/384 MAE-improved).
- **Frozen 1.5B checkpoint:** annotated tag `phase31ca-1_5b-corrected-q2k-aggregate-checkpoint` (target commit `a433875aa20431da8749e42c3449494434fa9f23`, the 31CA package commit; tag was created after the 31CA package commit and before 31CB). 1.5B result scope: **standalone tensor harness only** (Qwen2.5-1.5B-Instruct Q4_K_M, all 28 FFN layers × seeds {0, 9}, 56/56 memory-positive, 56/56 cosine-improved, 56/56 MAE-improved, 0 severe, all finite — Phase 31BZ). The 1.5B model file lives outside the repo at `$SDI_MODEL_DIR/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf` and is not committed.
- **Current scientific next phase:** **Phase 31CC — Qwen2.5-1.5B Real-Activation Capture Planning, only if explicitly requested.**

## What this repo contains

- **Phase docs** under `docs/` — per-phase architecture, planning, checkpoint, download, metadata, and orientation parity reports. Each phase has a corresponding `docs/PHASE31xx_*.md` and `src/results/PHASE31xx_*.json`.
- **Result JSON / Markdown reports** under `src/results/` — the audit trail of per-phase classifications and metrics.
- **Source scripts and harness code** under `src/` — phase scripts, tensor harness, manifest runtime, policy package helpers, and probe runners. Includes a regression suite at `tests/run_source_of_truth_regression.py`.
- **Policy package** — `corrected_q2k_policy_v1` constants, doc, and JSON; smoke-tested via the regression suite.
- **Regression tests** — `python3 -m tests.run_source_of_truth_regression` must report `PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN`, `error_count=0`, `fallback_count=0` before any new phase.
- **Claim-boundary docs** — `SOURCE_OF_TRUTH.md` Section 0.A (Master Forbidden Claims) and Section 4 (Invalidated / Superseded Claims).

## What this repo does not contain / does not claim

- **No model weights** committed to the repo. The 1.5B model lives outside the repo, referenced via `$SDI_MODEL_DIR/...`. The 0.5B model is also not committed.
- **No production runtime** — this is research code, not a shipped product.
- **No speedup claim** — no inference latency, throughput, or wall-clock numbers are claimed.
- **No quality recovery claim** — no claim that the substitutive path recovers the original model's output quality.
- **No behavior recovery claim** — no claim that behavior is preserved end-to-end.
- **No inference / generation validation claim** — no end-to-end text generation results are claimed.
- **No full llama.cpp integration claim** — no claim of a fully integrated, runtime-ready llama.cpp path.
- **No broad larger-model claim** — only Qwen2.5-0.5B and Qwen2.5-1.5B are validated as frozen evidence tiers under `corrected_q2k_policy_v1`. 3B / 7B / 14B / 32B / 72B / 110B+ Qwen2.5 are not tested. Other families (Llama / Mistral / etc.) are not tested. There is no 0.5B-vs-1.5B comparison, no "1.5B behaves like 0.5B" generalization, and no broader-family claim.
- **No runtime-ready output-residual claim** — the substitutive path is a standalone tensor harness, not a runtime.
- **No claim that 31AY / 31BA exact anchors are current canonical metrics** — they are historical only.

## Memory Math (background, not a claim)

| Representation | Bits/Weight |
|----------------|-------------|
| Q4 reference | 4 |
| Q2 base | 2 |
| Q2 + dense INT8 residual | 10 ❌ |
| Q2 + dense INT2 residual | ~4 + metadata; ≈ or > Q4 due to scales/alignment/decode buffers |
| Q2 + top-k sparse residual | variable — must test |

For Q2 + residual to beat Q4: residual must average **< 2 bits/weight**.
For Q3 + residual to beat Q4: residual must average **< 1 bit/weight**.

## Repository Layout (high-level)

```
sdi-substitutive/
├── README.md                              # this file (public orientation, low-frequency updates)
├── SOURCE_OF_TRUTH.md                     # canonical project state (SOT-01 hardened, post-31BT)
├── docs/                                  # per-phase architecture/planning/checkpoint reports
│   ├── CORRECTED_Q2K_POLICY_PACKAGE.md
│   ├── PHASE31A_ARCHITECTURE_PLAN.md      # historical — superseded by later phases
│   ├── PHASE31B_RESIDUAL_ECONOMICS_PLAN.md
│   ├── PHASE31BN_CORRECTED_Q2K_FULL_AGGREGATE_CHECKPOINT.md
│   ├── PHASE31BS_1_5B_DOWNLOAD_METADATA_VERIFICATION.md
│   ├── PHASE31BT_1_5B_ORIENTATION_PARITY_MICRO_PROBE.md
│   └── ... (per-phase docs)
├── src/                                   # phase scripts, harness, manifest runtime, policy helpers
│   ├── corrected_q2k_policy.py            # corrected_q2k_policy_v1 constants
│   ├── bundle_manifest.py
│   ├── phase31x_manifest_runtime.py
│   ├── phase31bt_1_5b_orientation_parity_micro_probe.py
│   └── ... (per-phase scripts)
├── src/results/                           # curated JSON reports (per-phase audit trail)
├── tests/                                 # regression suite
│   ├── run_source_of_truth_regression.py
│   └── test_corrected_q2k_policy.py
└── ... (other repo files)
```

## How to verify the current state

1. Read `SOURCE_OF_TRUTH.md` Section 0 (Current Working State).
2. Read `SOURCE_OF_TRUTH.md` Section 9 (Current Allowed Next Phase).
3. Run `python3 -m tests.run_source_of_truth_regression` and confirm `PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN`, `error_count=0`, `fallback_count=0`.

If any of these disagree with this README, **`SOURCE_OF_TRUTH.md` wins**.
