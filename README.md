# SDI-Substitutive — Substitutive Tensor Replacement Research

**Status:** Phase 31A (architecture plan) complete. Phase 31B (residual economics harness) next.

## Background

This repo follows the additive sidecar experiment (llama.cpp `experimental/prt-phase19a-alt-sidecar-backed`, tag `SDI_PRT_EXPERIMENTAL_ARCHIVE_CHECKPOINT`). That experiment produced a clear negative result: additive sidecar injection for Qwen2.5-0.5B on CPU increased memory usage rather than reducing it.

The substitutive direction tests a different thesis: instead of adding a sidecar alongside the resident tensor, replace the resident tensor with a lower-bit version plus a compressed residual that corrects the approximation error.

## Core Principle

> **Sidecar correction must replace resident cost, not add beside it.**

## Current Phase

**Phase 31A:** Architecture plan — complete. See `docs/PHASE31A_ARCHITECTURE_PLAN.md`.

**Phase 31B:** Residual economics harness — design complete. See `docs/PHASE31B_RESIDUAL_ECONOMICS_PLAN.md`.

## Key Memory Math

| Representation | Bits/Weight |
|----------------|-------------|
| Q4 reference | 4 |
| Q2 base | 2 |
| Q2 + dense INT8 residual | 10 ❌ |
| Q2 + dense INT2 residual | ~4 + metadata | ≈ or > Q4; scales/alignment/decode buffers may erase savings |
| Q2 + top-k sparse residual | variable — must test |

For Q2 + residual to beat Q4: residual must average **< 2 bits/weight**.

For Q3 + residual to beat Q4: residual must average **< 1 bit/weight**.

## Repository Layout

```
sdi-substitutive/
├── docs/
│   ├── PHASE31A_ARCHITECTURE_PLAN.md
│   └── PHASE31B_RESIDUAL_ECONOMICS_PLAN.md
├── src/
├── results/                 # curated JSON/MD reports only; raw/temp/log outputs ignored
├── CLAIMS.md
├── FORBIDDEN_CLAIMS.md
└── README.md
```

## Public Status

This repository is an early research scaffold. It contains plans and claim boundaries only. It does not contain model weights, extracted tensors, sidecars, benchmarks, or production code.

## Claim Boundaries

See `CLAIMS.md` and `FORBIDDEN_CLAIMS.md`. No hype, no broad claims. Every claim must match its proof level.

## What's Here

This is a design-only repo at this point. No code has been written yet. Phase 31B will build the residual economics harness to determine whether any residual representation can beat Q4 memory while preserving useful tensor approximation.
