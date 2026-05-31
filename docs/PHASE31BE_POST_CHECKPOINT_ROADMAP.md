# Phase 31BE — Post-Checkpoint Architecture Roadmap / Next Validation Target

## Classification
`PASS_31BE_ROADMAP_STATIC_ARTIFACT_HARDENING_SELECTED`

---

## Current Accepted Default

```
W_low format:     Q2_K (official llama.cpp)
Residual:         encode_sdir on per-family weight delta (W_ref - W_low)
Residual policy:  always-on
Residual k:       1%
Residual alpha:   1.0
Target:           standalone full-MLP tensor harness
Memory:           memory-positive on all 24 layers (Qwen2.5-0.5B)
```

---

## Accepted Limitations

- **Rare outlier:** L21-seed9 severe regression (0.26% of 384 pairs) — accepted as documented limitation
- **Static path only:** no runtime-ready output-residual claim
- **Qwen2.5-0.5B only:** larger models untested
- **No llama.cpp integration:** not current path

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

## What Phase 31 Proved

1. **The substitutive tensor harness works** — Q2_K + k=1% encode_sdir residual improves cosine and MAE for 99.48% of 384 seed-layer pairs on Qwen2.5-0.5B
2. **The approach is memory-positive** — 100% of pairs stay within Q4 budget
3. **k=1% is the right default** — 1% cosine failures, 1 severe regression (L21-seed9), 0 full failures
4. **The problem is residual formulation, not parameterization** — output residual oracle confirms a fix exists but is not statically deployable
5. **The static weight residual path is sound** — accepted as current default
6. **The rare outlier is real but rare** — 0.26% severe regression rate is acceptable with documentation
7. **The diagnostics loop is closed** — no unresolved numerical contradictions

---

## What Phase 31 Did Not Prove

1. **Generalization to larger models** — Qwen2.5-1.5B, 3B, 7B behavior unknown
2. **Production readiness** — no inference runtime integration
3. **llama.cpp integration** — static path not yet designed for GGUF integration
4. **Artifact format stability** — schema is implicit in scripts, not formally specified
5. **Robustness under diverse activations** — 16 seeds × 24 layers is good but prompt-derived activations untested
6. **Post-quantization behavior** — Q2_K quantization vs raw residual quality gap not fully characterized

---

## Candidate Next Lanes

### Lane 1: Larger-Model Tensor Harness Validation
**Goal:** Test Q2_K + k=1% encode_sdir on Qwen2.5-1.5B or 3B.

| | |
|---|---|
| **Pros** | Validates generality beyond 0.5B; identifies whether failure profile changes at scale |
| **Risks** | Extraction/quantization runtime cost; disk/RAM pressure; private model path issues |
| **Constraints** | Requires model access; Qwen2.5-1.5B or 3B GGUF must be available |
| **Forbidden claim risk** | High — larger-model claim is explicitly forbidden until validated |

**Verdict:** Valuable but should not precede artifact hardening. Cannot easily reproduce results on new hardware without fixed artifact schema.

---

### Lane 2: Static Artifact Spec + Regression Hardening ✓ SELECTED
**Goal:** Formalize the artifact schema, manifest format, and regression suite so the static path is reproducible and script-independent.

| | |
|---|---|
| **Pros** | Prevents future drift; enables later 1.5B/3B validation to use same schema; prepares eventual llama.cpp integration; reduces private-path leakage; strengthens SOURCE_OF_TRUTH contract |
| **Risks** | Less exciting; labor-intensive spec work |
| **Constraints** | Must not change accepted numeric results; must preserve regression pass/fail contract |
| **Reward** | High — foundational infrastructure that makes all future phases more reliable |

**Verdict:** Selected. This is the correct next step before larger models or integration.

---

### Lane 3: Runtime Integration Design (Not Implementation)
**Goal:** Write architecture doc for how the static default could plug into llama.cpp.

| | |
|---|---|
| **Pros** | Prepares eventual real path; clarifies integration constraints |
| **Risks** | Premature without hardened artifact format; design may be wrong without 1.5B validation |
| **Constraints** | No implementation; design doc only |
| **Forbidden claim risk** | Low — design doc is not integration |

**Verdict:** Useful as an internal note but not a Phase milestone. Can be done in parallel or as part of 31BF.

---

### Lane 4: Robustness Extension
**Goal:** More seeds, more activations, potentially prompt-derived activations.

| | |
|---|---|
| **Pros** | Strengthens 0.5B evidence base; already strong at 384 pairs |
| **Risks** | Diminishing returns; 16 seeds already shows 22/24 layers fully robust |
| **Constraints** | Gaussian activations only currently; prompt activations would need new infrastructure |
| **Reward** | Medium — good hygiene but low scientific upside after 384-pair sweep |

**Verdict:** Good for 31BF mini-task, not a standalone phase.

---

### Lane 5: Public/Internal Technical Writeup
**Goal:** Document Phase 31 as a clean milestone.

| | |
|---|---|
| **Pros** | Preserves institutional knowledge; supports eventual public post |
| **Risks** | Does not advance implementation |
| **Reward** | Low for implementation timeline |

**Verdict:** Writeup can be a 31BF sub-artifact, not a standalone phase.

---

## Recommended Next Phase

**Phase 31BF — Static Artifact Spec + Regression Hardening**

### Why not larger model first?

Before testing on Qwen2.5-1.5B or 3B, the artifact schema must be hardened:
- Current phase scripts encode private paths (e.g., `/home/matthew-villnave/llama.cpp/...`)
- The manifest loader has no formal schema doc
- The regression suite is minimal — a single pass/fail contract
- Without a formal spec, 1.5B validation would add a new script with the same private-path and drift problems

### Why not llama.cpp integration first?

Integration requires a stable artifact format. You cannot integrate a moving target.

### Why not just continue with phase scripts?

Because each phase script has unique private paths, ad-hoc artifact naming, and implicit conventions. After 31 phases, the technical debt is significant. 31BF cleans this up once.

---

## Phase 31BF — Scope

### Artifact Schema Definition
- Define a stable artifact schema for `.sdiw` (packed W_low) and `.sdir` (sparse residual) files
- Formalize the manifest schema: fields, types, required vs optional
- No private paths in schema — all paths relative to manifest location
- Schema versioning: `schema_version = "1.0"`

### Manifest Loader Hardening
- `bundle_manifest.py` already exists — audit it for schema compliance
- Add schema validation on load
- Add field documentation
- Remove ad-hoc path assumptions

### Regression Suite Strengthening
- Current: single pass/fail contract (`run_source_of_truth_regression.py`)
- Add: explicit schema-validation step
- Add: per-family byte-budget enforcement
- Add: cosine and MAE thresholds as explicit constants (not magic numbers in scripts)
- Ensure regression can run from a clean clone with only public inputs

### Private Path Audit
- Scan existing phase scripts for hardcoded private paths
- Replace with manifest-relative or environment-variable-based paths
- No `/home/matthew-villnave/...` paths in committed code

### Phase Script Hygiene
- Phase scripts should be runnable from `results/` output location
- All scripts must pass regression before committing
- Document the canonical run order

### Success Criteria for Phase 31BF
1. Artifact schema documented in `SPEC.md` or `SCHEMA.md`
2. Manifest loader passes schema validation tests
3. Regression suite has explicit cosine/MAE thresholds as constants
4. No hardcoded private paths in any committed phase script
5. All existing phase scripts still pass regression unchanged (no behavioral change to results)
6. SOURCE_OF_TRUTH updated with schema version fact

---

## Risk/Benefit Summary Table

| Lane | Benefit | Risk | Precedence |
|------|---------|------|------------|
| Larger-model validation | High (generality proof) | High (wrong without spec) | After 31BF |
| Artifact spec + hardening | High (foundation) | Low (spec work) | **SELECTED** |
| Runtime integration design | Medium | Low (design only) | After 31BF |
| Robustness extension | Medium | Low (diminishing returns) | Mini-task in 31BF |
| Technical writeup | Low | None | Sub-artifact in 31BF |
