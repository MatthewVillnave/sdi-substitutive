# FORBIDDEN_CLAIMS.md — Substitutive Tensor Replacement Project

The following claims are **forbidden** until explicitly proven in a future phase and reviewed by Matt before publication.

## Hard Boundaries

- **"SDI is solved"** — This project is a tensor-level prototype. It does not solve SDI at any level.
- **"Quality recovery achieved"** — Tensor cosine, MAE, max error, and logit equivalence can support approximation evidence only. Quality recovery requires explicit behavioral evaluation comparing Q_low, Q_low + residual, and Q_ref on defined prompts and metrics.
- **"Memory savings on full model"** — Phase 31 only tests single tensor families.
- **"Speedup demonstrated"** — No compute timing is measured in Phase 31.
- **"Production ready"** — This is experimental research. No production claims ever without explicit future phase and Matt's approval.
- **"Q2+sidecar ≈ Q4 behavior"** — Tensor-level approximation does not claim behavioral equivalence.
- **"Residual approach scales to full model"** — Economics may not hold at model scale.
- **"The prior additive-sidecar archive results are superseded"** — The prior additive-sidecar negative result remains valid. This repo tests a different substitutive architecture.
- **"This repo is the answer to the SDI problem"** — It is a single experimental direction, not a solution.

## Why These Are Forbidden

The additive sidecar path produced misleading positive signals in early phases (activation success, injection success) before the memory overhead was measured. That path was only correctly classified as a failure after proper memory accounting.

The same error mode applies here: a residual representation that looks accurate (high cosine) may still fail on memory economics, or may fail to generalize, or may fail at the compute level. Each claim must match its proof level.

## Amendment Process

These forbidden claims can become allowed claims only after:
1. Explicit future phase demonstrates the claim
2. Matt reviews and approves the claim before publication
3. Documentation is updated in CLAIMS.md with the new approved scope
