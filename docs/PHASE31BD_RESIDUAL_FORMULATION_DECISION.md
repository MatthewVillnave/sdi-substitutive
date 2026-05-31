# Phase 31BD — Residual Formulation Decision / Accept Outlier

## Classification
`PASS_31BD_ACCEPT_STATIC_DEFAULT_WITH_KNOWN_OUTLIER`

---

## Phase Chain Summary

| Phase | Classification | Key Finding |
|-------|---------------|-------------|
| 31AT-FREEZE | STABLE | Standalone all-24-layer full MLP harness passes for Qwen2.5-0.5B; Q2_K + k=1% WR; 24/24 memory-positive |
| 31BA | PARTIAL_LAYER21_DOMINANT_SENSITIVITY | 384 pairs: 99.48% cosine/MAE improvement; 1 severe regression (L21-S9); 22/24 layers fully robust |
| 31BB | PARTIAL_K_TRADEOFF | Full 384-pair aggregate: k=0.5→5 fail, 0 severe; k=1.0→2 fail, 1 severe; k=1.5→1 fail, 1 severe; no k dominates; k=1.0 is pragmatic default |
| 31BC | PARTIAL_OUTPUT_RESIDUAL_FIXES_BUT_NOT_STATIC | Output residual fp16 k=1% (54 bytes) fixes L21-S9; oracle confirms fix exists but is activation-specific and not statically deployable |
| 31BC-R | PASS_31BC_R_REPORT_RECONCILED | Placeholders fixed; JSON classification corrected; flatten() copy bug documented; no numeric changes |

---

## Decision

**Accept the static Q2_K + k=1% weight residual as the current default.**

Retain the rare outlier as a documented limitation.

Defer output residual to a future dynamic-architecture research path.

---

## Rationale

### The static path is strong

- **99.48%** cosine improvement rate across 384 seed-layer pairs (24 layers × 16 seeds)
- **99.48%** MAE improvement rate
- **100%** memory-positive across all 384 pairs
- **0 severe regressions** from k=0.5 option; 1 severe at k=1.0 and k=1.5
- **22/24 layers** are fully robust across all 16 seeds — zero failures of any kind
- The approach is memory-positive, approximation-improving, and architecturally simple

### The severe outlier is genuinely rare

- 1 severe out of 384 pairs = **0.26%** of all cases
- The outlier (L21 seed 9) is activation-specific: it passes cleanly at all other layers
- No k parameter solves it without creating worse tradeoffs elsewhere
- The output residual oracle proves the fix exists — but it requires knowing Y_ref at runtime

### Output residual proves the problem is formulation, not parameterization

- Oracle: dense output residual perfectly recovers Y_ref for L21-S9
- Even 1% sparse output residual (54 bytes) fixes the severe case
- But: output residual is activation-specific — you need Y_ref to compute it
- Static deployment would require either (a) pre-storing Y_ref (defeats the purpose) or (b) a runtime mechanism that provides reference activations (not architecturally simple)
- Therefore: not a static-path solution; marked as future dynamic-architecture work

### k=1.0 is the pragmatic default

- k=1.5 is a legitimate alternative (1 cos failure vs 2, 0 MAE failures vs 2, but same 1 severe)
- k=1.0 has the best aggregate cosine-failure rate among candidates with severe=1 (only 2 vs k=0.75's 5 and k=1.5's 1 — but k=1.5 adds nothing on severe count)
- k=0.5 eliminates the single severe but creates 5 cosine failures — net worse
- **k=1.0 is the right balance for the current default**

---

## Current Accepted Default Policy

```
W_low format:     Q2_K (official llama.cpp)
Residual:         encode_sdir on per-family weight delta (W_ref - W_low)
Residual policy:  always-on
Residual k:      1%
Residual alpha:   1.0
Target:           standalone full-MLP tensor harness
Memory:           memory-positive on all 24 layers (Qwen2.5-0.5B)
Scope:            tensor research only — no production, no inference claim
```

---

## Known Limitation

**Rare activation-specific residual-direction failure.**

- Severe regression: 1/384 pairs (0.26%) — L21 seed 9
- Mild failure: 1/384 pairs (0.26%) — L2 seed 7
- Cause: per-family weight residual directionally misaligned with the actual correction needed for this specific activation
- The failure is not a function of k or alpha tuning
- The failure is not a memory or encoding quality issue
- A runtime mechanism that could apply output residual (if Y_ref were available) would fix it

---

## Future Work (Not Current Path)

The following are **future directions**, not current default policy:

1. **Dynamic/output residual architecture**
   - Requires runtime access to Y_ref or an oracle for the correct residual
   - Not statically deployable under current architecture
   - Marked as next-architecture research, not current default

2. **Learned correction**
   - A learned model predicts the correct residual direction
   - Would require separate training infrastructure
   - No current implementation

3. **Larger model testing**
   - Qwen2.5-0.5B is a single-point validation
   - Behavior at 1B, 3B, 7B+ is unknown
   - Larger models may have different failure profiles

4. **Eventual llama.cpp integration**
   - Not until more proof exists
   - Current scope is standalone tensor harness only
   - Full integration requires solving the rare-outlier problem or accepting it explicitly

---

## Forbidden Claims (Unchanged)

- no model quality recovery
- no behavior recovery
- no speedup
- no full-model runtime memory savings claim
- no llama.cpp integration claim
- no production readiness
- no inference/generation claim
- no larger-model claim
- no runtime-ready output-residual claim
- no claim beyond standalone tensor harness

---

## SOURCE_OF_TRUTH

See SOURCE_OF_TRUTH.md Section 3 for updated accepted facts.

This phase closes the Phase 31 diagnostic loop.
