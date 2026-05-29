# CLAIMS.md — Substitutive Tensor Replacement Project

## Allowed Claims (Phase 31)

- Tensor-level substitutive math prototype produces output
- Memory accounting shows W_low + R_compressed < W_q4 for a specific residual representation
- Cosine similarity, MAE, and max error reported for tensor approximation quality
- attn_output chosen as minimal proof-of-mechanics target
- ffn_up/ffn_down chosen as economically meaningful targets (after attn_out)
- Residual compression level required to beat Q4 residency is reported
- Specific residual representation (sparse, low-rank, etc.) passes memory viability gate
- Phase 31B economics gate: PASS, PARTIAL, or BLOCKED

## Claims That Require Explicit Proof Before Making

- Approximation evidence: tensor cosine, MAE, max error, and logit-level equivalence can support a tensor approximation claim, scoped to the tested tensor and prompt set
- Quality recovery: requires explicit behavioral evaluation comparing Q_low, Q_low + residual, and Q_ref on defined prompts and metrics
- Generalization from attn_out to ffn_up/ffn_down: requires separate measurement
- llama.cpp integration feasibility: requires Phase 31E demonstration
- Production readiness: never, without explicit future phase
- Memory savings on full model: never, without explicit future phase

## Principle

Claims must be scoped to the smallest unit of proof actually demonstrated. A result at the attn_output tensor level does not claim ffn_up behavior. A tensor approximation result does not claim model-level behavior or quality recovery.
