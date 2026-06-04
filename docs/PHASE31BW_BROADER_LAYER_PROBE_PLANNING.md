# Phase 31BW — Qwen2.5-1.5B Corrected Q2_K Broader Layer Probe Planning

**Classification:** `PASS_31BW_BROADER_LAYER_PROBE_PLAN_SELECTED`

**Phase:** 31BW
**Scope:** Planning-only. No validation execution. No model load. No Q2_K / SDIR artifacts. No generation / inference / runtime integration. No commit/push/tag without explicit Matt approval.
**Status:** Artifacts prepared — **not yet committed** (PRE-COMMIT REPORT; awaiting Matt approval).

---

## 1. Goal

Decide the safest next validation scope after successful 31BU and 31BV, **without running it yet**. Answer:

- Should the next executable phase be a 7-layer stratified probe, a 14-layer half-model probe, or a full 28-layer aggregate?
- What layer/seed set is scientifically useful while still bounded?
- What runtime/disk/memory risk does each option carry?
- What stop conditions should gate the next phase?
- What claims would be allowed if the next phase passes?

**31BW does not execute the next validation.** It produces a planning JSON, a planning doc, and a SOT update that points to the selected next-allowed phase.

---

## 2. Proof trail reviewed (no new validation run)

Read from committed files (not re-derived):

- `SOURCE_OF_TRUTH.md` Section 0 (current-state header)
- `SOURCE_OF_TRUTH.md` Section 3 entries for 31BU and 31BV
- `docs/PHASE31BU_1_5B_CORRECTED_Q2K_ANCHOR_PROBE.md`
- `docs/PHASE31BV_1_5B_CORRECTED_Q2K_SMALL_MULTILAYER_PROBE.md`
- `src/results/PHASE31BU_1_5B_CORRECTED_Q2K_ANCHOR_PROBE.json`
- `src/results/PHASE31BV_1_5B_CORRECTED_Q2K_SMALL_MULTILAYER_PROBE.json`

Key numbers pulled from these (no rounding, no projection):

| metric | value | source |
|---|---:|---|
| 31BU classification | `PASS_31BU_1_5B_Q2K_ANCHOR_PROBE_CLEAN` | 31BU result JSON |
| 31BU n_pairs | 3 | 31BU result JSON |
| 31BU min per-layer margin | +3,380,374 bytes | 31BU result JSON |
| 31BU mean delta_cos | +0.004139 | 31BU result JSON |
| 31BU mean MAE improvement | −0.004712 | 31BU result JSON |
| 31BV classification | `PASS_31BV_1_5B_Q2K_SMALL_MULTILAYER_CLEAN` | 31BV result JSON |
| 31BV n_pairs | 6 (3 layers × 2 seeds) | 31BV result JSON |
| 31BV n_layers | 3 (0, 14, 27) | 31BV result JSON |
| 31BV min per-layer margin | +3,380,350 bytes | 31BV result JSON |
| 31BV per-layer margin variance (max − min) | 26 bytes | 31BV result JSON |
| 31BV worst pair | L0-S9, dc=+0.001090, MAE_delta=−0.003686 | 31BV result JSON |
| 31BV mean delta_cos | +0.003872 | 31BV result JSON |
| 31BV mean MAE improvement | −0.005923 | 31BV result JSON |
| 31BV wall-clock (observed) | ~240 sec for 6 pairs (~40 sec/pair) | background process log |
| 31BV per-pair peak resident memory | ~1.5 GB (3 W_ref + 3 W_low + 3 R + 2 dec, all f32) | 31BV runner observation |
| 1.5B FFN layer count | 28 (block_count=28) | 31BS metadata |
| 1.5B Q4_budget per layer | 20,643,840 bytes (3 × 6,881,280) | computed from shapes |

**No validation was run in 31BW.** The 31BU/31BV numbers are read-only inputs to the planning logic.

---

## 3. Options considered

### Option A — 7-layer stratified probe (RECOMMENDED)

- **Layers:** `[0, 4, 8, 12, 16, 20, 27]`
- **Seeds:** `[0, 9]`
- **Total pairs:** 14
- **Purpose:** Broad coverage without full sweep. Stratified ~every 4 layers + final layer (27) preserves top-of-network, mid, and end positions.

### Option B — 14-layer half-model probe

- **Layers:** `[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]` (every other)
- **Seeds:** `[0, 9]`
- **Total pairs:** 28
- **Purpose:** Stronger coverage, still not full aggregate.

### Option C — Full 28-layer aggregate

- **Layers:** `[0, 1, 2, …, 27]`
- **Seeds:** `[0, 9]`
- **Total pairs:** 56
- **Purpose:** Full layer coverage at two seeds. **Higher runtime and scope risk.** Crosses into aggregate validation territory.

### Option D — Seed sensitivity probe

- **Layers:** `[0, 14, 27]`
- **Seeds:** `[0, 1, 2, 3, 4, 5, 9, 13]`
- **Total pairs:** 24
- **Purpose:** Seed/activation sensitivity before broader layers. Re-tests 31BV's 3 layers with 4x more seeds.

---

## 4. Cost / risk table

Estimated from 31BV's observed per-pair time (~40 sec) and per-pair peak resident memory (~1.5 GB). Serial execution (one pair at a time; not parallel). All temp artifacts in `tempfile.mkdtemp` + numpy arrays, cleaned at runner exit — **no disk artifacts committed**.

| option | total pairs | est. runtime | est. peak resident | disk artifacts committed | scope-creep risk | claim boundary if pass |
|---|---:|---|---|---|---|---|
| A — 7-layer stratified | 14 | ~9 min | ~1.5 GB | none | **low** | small stratified probe (7 of 28 layers = 25% coverage) |
| B — 14-layer half-model | 28 | ~19 min | ~1.5 GB | none | medium | half-model probe (14 of 28 layers = 50% coverage; approach aggregate boundary) |
| C — full 28-layer aggregate | 56 | ~37 min | ~1.5 GB | none | **high** | aggregate validation territory (28 of 28 layers = 100% coverage) — **forbidden by 31BW spec** |
| D — seed sensitivity | 24 | ~16 min | ~1.5 GB | none | low | seed sensitivity at 3 layers (no new layer coverage) |

---

## 5. Selection: Option A — 7-layer stratified probe

**Selected next phase (if 31BW is approved):**

**Phase 31BX — Qwen2.5-1.5B Corrected Q2_K Stratified Layer Probe**

- Layers: `[0, 4, 8, 12, 16, 20, 27]`
- Seeds: `[0, 9]`
- Total pairs: 14
- Policy: `corrected_q2k_policy_v1` (corrected_ceil_per_row Q2_K, ffn_up+ffn_gate SDIR k=0.5%, alpha=1.0, ffn_down W_low only — no ffn_down residual)
- W_ref: Qwen2.5-1.5B-Instruct Q4_K_M dequantized (NOT FP16)
- No generation / inference / runtime integration
- No full aggregate validation
- No quality / performance claims

### Rationale (data-driven, not just preference)

1. **Robustness established at top, mid, end:** 31BU (layer 0) and 31BV (layers 0, 14, 27) together show the policy is memory-positive and quality-improving at 3 of 28 layers (~10.7% coverage), all with no severe regressions. Per-layer margin variance is 26 bytes — **memory economics are NOT the binding constraint** for broader sweeps.
2. **Runtime scale is acceptable:** 31BV ran 6 pairs in ~240 sec wall clock (~40 sec/pair). Option A = 14 pairs × 40 sec = **~9 min**, well within a single-session tolerance.
3. **Stratified coverage:** Option A samples positions 0 (top), 4, 8, 12, 16, 20, 27 (end) — stride 4 plus the final layer. This preserves the same positional diversity 31BV tested (top, mid, end) but with 4x more granularity in the middle.
4. **Claim boundary stays clean:** 7 of 28 layers = 25% layer coverage. This is **clearly stratified probe**, not approaching aggregate. Option C (28/28 = 100%) would cross into aggregate territory. Option B (14/28 = 50%) is borderline.
5. **Even scope step from 31BV:** 31BU was 1 layer (3 pairs); 31BV was 3 layers (6 pairs, 2x step); 31BX would be 7 layers (14 pairs, 2.3x step). Even doubling. Option B would be a 4.7x step — too large for one phase.
6. **Option D (seed sensitivity) does not add layer coverage.** It re-tests 31BV's 3 layers 4x. Useful as a follow-up after 31BX, but not the right next step.

### Rejected options

- **Option B (14-layer half-model):** Step from 3 → 14 layers is too large for one phase. 31BV was 1 → 3 layers. Save B for after A passes, as a follow-up scope step.
- **Option C (full 28-layer aggregate):** Full 28-layer sweep = aggregate validation, **explicitly forbidden by 31BW spec**. Would also be ~37 min wall clock and cross the claim boundary.
- **Option D (seed sensitivity):** Re-tests 3 layers 31BV already covered. Lower priority than layer coverage step.

---

## 6. Success criteria for the next phase (31BX)

At minimum:

- `python3 -m tests.run_source_of_truth_regression` passes before AND after, with `PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN`, `error_count=0`, `fallback_count=0`, `readme_drift_guard.passed=true`
- All 14 pairs produce finite Y_ref, Y_low, Y_sub
- All 14 pairs memory-positive (per-layer margin >= 0)
- 0 pairs with severe regression (`delta_cos < -0.05`)
- >= 12/14 pairs have `delta_cos >= 0` (cosine-improved majority)
- >= 12/14 pairs have `MAE_delta < 0` (MAE-improved majority)
- No model files committed (model remains outside repo)
- No Q2_K / SDIR / temp blobs in the commit
- No scope creep: layers and seeds exactly as approved; no layer added, no seed added, no family added, no aggregate-style sweep

---

## 7. Classification rules for next phase (31BX)

| classification | trigger |
|---|---|
| `PASS_31BX_1_5B_Q2K_STRATIFIED_LAYER_PROBE_CLEAN` | all 14 pairs pass all criteria; no severe regressions; all finite; all memory-positive; >= 12/14 cosine-improved; >= 12/14 MAE-improved |
| `PARTIAL_31BX_1_5B_Q2K_STRATIFIED_LAYER_PROBE_MINOR_FAILURES` | all pairs finite and memory-positive; some pairs have minor cosine/MAE failures but no severe regressions |
| `PARTIAL_31BX_1_5B_Q2K_STRATIFIED_LAYER_PROBE_TRADEOFF` | cosine/MAE tradeoffs present (some layers improve cosine, some improve MAE, no layer severely regresses both) |
| `PARTIAL_31BX_1_5B_Q2K_STRATIFIED_LAYER_PROBE_MEMORY_FAIL` | any selected layer is memory-negative under current accounting |
| `BLOCKED_31BX_SDI_MODEL_DIR_UNSET` | env var missing |
| `BLOCKED_31BX_MODEL_FILE_MISSING` | model file missing |
| `BLOCKED_31BX_Q2K_BACKEND_FAIL` | corrected Q2_K generation/dequantization fails |
| `BLOCKED_31BX_SDIR_FAIL` | SDIR residual generation/application fails |
| `BLOCKED_SOURCE_OF_TRUTH_REGRESSION` | regression fails |

---

## 8. Stop conditions for the next phase (31BX)

- Regression failure at any point (pre-flight or post-edit)
- Model file tracking issue (file missing, wrong file, different quantization)
- Non-finite Y_ref / Y_low / Y_sub on any pair
- Severe regression (`delta_cos < -0.05`) on any pair
- Memory-negative on any layer
- Scope creep (extra layer, extra seed, extra family, aggregate-style sweep)
- Any claim trigger: inference, generation, runtime, production, speedup, quality, behavior, llama.cpp integration, larger-model validation
- Q2_K / SDIR / temp blobs committed

---

## 9. Allowed claims if 31BX passes (conservative)

**Primary claim:**

> "Corrected Q2_K policy remains memory-positive and improves cosine/MAE across a stratified 1.5B standalone tensor-harness probe (7 layers: 0, 4, 8, 12, 16, 20, 27; 2 seeds: 0, 9; 14 pairs)."

**Secondary claims (also allowed):**

- Per-layer margin is consistent across the stratified layer set (variance bounded by 31BV's 26 bytes/layer observation)
- L0 result reproduces 31BU and 31BV (cross-runner reproducibility)
- The mixed source-GGUF ffn_down quant types (L0/L27 Q6_K, L14 Q4_K) do not affect corrected Q2_K memory accounting

**Still forbidden:**

- ~~"full larger-model validation"~~ (7 of 28 layers ≠ 28 layers)
- ~~"inference / generation"~~
- ~~"speedup"~~
- ~~"quality / behavior recovery"~~
- ~~"runtime"~~
- ~~"production"~~
- ~~"1.5B behaves like 0.5B"~~ (no 0.5B comparison in 31BX scope)

---

## 10. What did NOT run in 31BW (upheld)

- ❌ No validation execution (no Q2_K encode, no SDIR, no MLP forward)
- ❌ No model load (the 1.5B model was not opened; runner only reads the 31BU/31BV result JSONs)
- ❌ No Q2_K / SDIR artifacts generated
- ❌ No temp tensor dumps
- ❌ No generation / inference / sampling
- ❌ No llama.cpp runtime integration
- ❌ No performance claim
- ❌ No model quality claim
- ❌ No model files committed
- ❌ No Q2_K / SDIR blobs committed
- ❌ No commit/push/tag without explicit Matt approval

---

## 11. Forbidden claims preserved

Full canonical master list in `SOURCE_OF_TRUTH.md` Section 0.A. Per-phase notes in 31BW spec are audit context. Active 31BW-specific forbidden items:

- no model quality recovery claim
- no behavior recovery claim
- no speedup claim
- no full-model runtime memory savings claim
- no llama.cpp integration claim
- no production readiness claim
- no inference / generation claim
- no full larger-model validation claim
- no runtime-ready output-residual claim
- no claim beyond standalone tensor harness
- no aggregate validation execution in this phase
- no full 28-layer sweep execution in this phase
- no model files committed
- no large generated artifacts committed
- no Q2_K/SDIR blobs generated or committed
- no commit/push/tag without explicit Matt approval

---

## 12. Limitations

- **Planning only:** 31BW did not run validation. The selected next phase (31BX) must be explicitly requested to be entered.
- **Cost estimates are linear extrapolations from 31BV's observed per-pair time (~40 sec).** Real per-pair time may vary slightly with layer / seed / machine load.
- **No disk artifacts in any option** — all temp work in memory + `tempfile.mkdtemp`, cleaned at runner exit, per the 31BU/31BV pattern.
- **Memory estimates assume serial execution** (one pair at a time, not parallel). Parallel execution is not used in this project.
- **31BW's selection logic is conservative.** It does not propose a novel layer sampling strategy or a new metric. It chooses among the four spec options using grounded cost data.

---

## 13. Artifacts (untracked, prepared, not committed)

| Path | Purpose |
|---|---|
| `docs/PHASE31BW_BROADER_LAYER_PROBE_PLANNING.md` | This document |
| `src/phase31bw_broader_layer_probe_planning.py` | Planning-only runner (no model load; only reads 31BU/31BV result JSONs; writes planning JSON) |
| `src/results/PHASE31BW_BROADER_LAYER_PROBE_PLANNING.json` | Planning output: 4 options considered, cost/risk table, selection rationale, success criteria, classification rules, stop conditions, claim boundaries |
| `SOURCE_OF_TRUTH.md` | New accepted fact appended; Section 0 / Section 9 updated to point next allowed phase to 31BX |

**Not produced / not in this phase:**

- model files
- HF cache
- generated Q2_K blobs
- generated SDIR blobs
- temp tensor dumps
- large artifacts
- aggregate validation output
- generation / inference output
- runtime integration output

---

## 14. SOT update (to be applied if approved)

Append to `SOURCE_OF_TRUTH.md` Section 3, then update Section 0 / Section 9 to point next allowed phase to:

- **if clean (this phase)**: Phase 31BX — Qwen2.5-1.5B Corrected Q2_K Stratified Layer Probe, only if explicitly requested
- **if blocked**: Phase 31BW-R — Broader Probe Planning Repair, only if explicitly requested

---

## 15. Pre-commit status

| field | value |
|---|---|
| Runner | `src/phase31bw_broader_layer_probe_planning.py` (no model load; planning-only; lint-clean) |
| Result JSON | `src/results/PHASE31BW_BROADER_LAYER_PROBE_PLANNING.json` (~17 KB) |
| Doc | `docs/PHASE31BW_BROADER_LAYER_PROBE_PLANNING.md` (this file) |
| SOT | new accepted fact to append (draft below) |
| Pre-commit regression | must be re-run after SOT edit, must still pass with same contract |
| Git status (after edits, before commit) | 3 new files in `docs/`, `src/`, `src/results/`; 1 modified file (`SOURCE_OF_TRUTH.md`) |
| Any private paths in commit? | **No** — runner only reads env-var-redacted model path from result JSONs; no model refs in runner code |
| Any large / generated artifacts in commit? | **No** — planning output is small JSON; no blobs |
| Pre-existing untracked 31BH-R2 / 31BJ files disposition | **untouched** per explicit standing instruction; not in this commit |
| Proposed commit message | `Phase 31BW: plan 1.5B broader layer probe` |

---

## 16. Draft SOT Section 3 entry (to append if approved)

```markdown
    - **Phase 31BW — Qwen2.5-1.5B Corrected Q2_K Broader Layer Probe Planning:** Classification `PASS_31BW_BROADER_LAYER_PROBE_PLAN_SELECTED`.
      - Scope: planning-only. No validation execution in 31BW. No model load. No Q2_K / SDIR artifacts. No generation / inference / runtime integration. No commit/push/tag without explicit Matt approval.
      - Proof trail reviewed (read-only): 31BU (3 pairs, layer 0, PASSED), 31BV (6 pairs, layers 0/14/27, PASSED). 31BV observed per-pair time ~40 sec, per-layer margin variance 26 bytes, worst pair L0-S9 (dc=+0.001090, MAE_delta=−0.003686).
      - Four candidate next-executable scopes considered:
        - Option A — 7-layer stratified probe (layers 0, 4, 8, 12, 16, 20, 27; seeds 0, 9; 14 pairs; ~9 min)
        - Option B — 14-layer half-model probe (every other layer; 28 pairs; ~19 min)
        - Option C — full 28-layer aggregate (56 pairs; ~37 min; crosses aggregate territory — forbidden by 31BW spec)
        - Option D — seed sensitivity probe (3 layers × 8 seeds = 24 pairs; ~16 min; no new layer coverage)
      - Cost / risk table grounded in 31BV observations (per-pair time, per-layer margin, peak resident memory).
      - **Selected: Option A (7-layer stratified).** Rationale: 2x scope step from 31BV (3 → 7 layers), preserves top/mid/end sampling, claim boundary stays 'stratified probe' (7 of 28 layers = 25%), no aggregate territory, runtime well within tolerance.
      - Selected next phase: **Phase 31BX — Qwen2.5-1.5B Corrected Q2_K Stratified Layer Probe**, only if explicitly requested. Layers [0, 4, 8, 12, 16, 20, 27] × seeds [0, 9] = 14 pairs. Same `corrected_q2k_policy_v1`. W_ref = 1.5B Q4_K_M dequantized. No generation / inference / runtime integration.
      - Success criteria for 31BX: regression passes before/after; all 14 pairs finite; all 14 pairs memory-positive; 0 severe regressions; >= 12/14 cosine-improved; >= 12/14 MAE-improved; no model files committed; no blobs committed; no scope creep.
      - Classification rules for 31BX: PASS / PARTIAL (minor failures / tradeoff / memory fail) / BLOCKED (env / model / Q2_K backend / SDIR / regression).
      - Stop conditions for 31BX: regression fail, model tracking issue, non-finite outputs, severe regression, memory-negative, scope creep, claim trigger, blob commit.
      - Allowed claims if 31BX passes (conservative): "Corrected Q2_K policy remains memory-positive and improves cosine/MAE across a stratified 1.5B standalone tensor-harness probe (7 layers × 2 seeds = 14 pairs)". Plus L0-cross-runner-reproducibility + mixed source-GGUF quant types observation. Forbidden: full larger-model validation, inference/generation, speedup, quality/behavior, runtime, production, "1.5B behaves like 0.5B".
      - Runner: `src/phase31bw_broader_layer_probe_planning.py` (no model load; planning-only; reads 31BU/31BV result JSONs; writes planning JSON; lint-clean).
      - Result JSON: `src/results/PHASE31BW_BROADER_LAYER_PROBE_PLANNING.json` (~17 KB; no blobs).
      - Doc: `docs/PHASE31BW_BROADER_LAYER_PROBE_PLANNING.md`.
      - Pre-flight regression: `python3 -m tests.run_source_of_truth_regression` → `PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN`, `error_count=0`, `fallback_count=0`, `readme_drift_guard.passed=true` (run before AND after SOT/doc/result edits; must remain clean for commit approval).
      - Pre-existing untracked 31BH-R2 / 31BJ files: untouched per explicit instruction; not in this commit; will be handled in a dedicated stale-file cleanup / provenance phase.
      - Next allowed phase: **Phase 31BX — Qwen2.5-1.5B Corrected Q2_K Stratified Layer Probe**, only if explicitly requested. (Not entered — explicit request required.)
      - **Accepted claim:** 31BW successfully selected Option A (7-layer stratified) as the safest next executable phase after 31BU + 31BV. Selected scope: layers [0, 4, 8, 12, 16, 20, 27] × seeds [0, 9] = 14 pairs, under `corrected_q2k_policy_v1`. Estimated runtime ~9 min (14 pairs × ~40 sec/pair from 31BV observation). Claim boundary stays 'stratified probe' (7 of 28 layers = 25%), does not approach aggregate. Success criteria, classification rules, stop conditions, and allowed claims defined for 31BX.
      - **Valid as long as:**
        - 31BU and 31BV accepted facts remain unchanged (the planning inputs)
        - the 1.5B Q4_K_M file at `$SDI_MODEL_DIR/...` remains unmodified
        - the next phase (31BX) is entered only on explicit Matt request
        - the planning options (A/B/C/D) and their cost/risk estimates are recorded and unchanged in the planning JSON
        - the canonical orientation convention (Section 7) remains unchanged
        - no later phase invalidates or supersedes this plan
```

**Do not commit. Do not push. Awaiting explicit Matt approval.**

---

## 17. Question

**Approve commit / push of 31BW artifacts (planning runner + planning JSON + this doc + SOT update)?**
