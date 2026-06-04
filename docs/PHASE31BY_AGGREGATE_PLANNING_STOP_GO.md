# Phase 31BY — Qwen2.5-1.5B Corrected Q2_K Aggregate Planning / Stop-Go Decision

> **PLANNING-ONLY phase.** No aggregate validation, no full 28-layer sweep, no generation/inference, no llama.cpp runtime integration, no Q2_K/SDIR artifact generation. This document decides whether the next executable phase should run a full 28-layer × 2-seed aggregate and defines the exact scope, stop conditions, classifications, and claim boundaries.

---

## 1. Scope

This phase is a stop/go decision for the next executable phase. It does **not** run validation.

It answers:

- Is the evidence from 31BU / 31BV / 31BX strong enough to justify a full 28-layer × 2-seed aggregate?
- What exact scope should that aggregate use?
- What runtime / disk / memory risk does it carry?
- What stop conditions should gate the next phase?
- What claims would be allowed if it passes?
- What claims would still be forbidden even if it passes?

Companion artifact: `src/results/PHASE31BY_AGGREGATE_PLANNING_STOP_GO.json` (this document's structured source of truth).

---

## 2. Prior evidence reviewed

| phase | classification | scope | pairs | memory+ | cos+ | MAE+ | severe | finite | mean Δcos | mean MAE Δ |
|---|---|---|---:|---:|---:|---:|:---:|:---:|---:|---:|
| 31BU | PASS_31BU_1_5B_Q2K_ANCHOR_PROBE_CLEAN | L0 only, seeds {0, 9, 13} | 3 | 3 | 3 | 3 | 0 | 3 | +0.004139 | −0.004712 |
| 31BV | PASS_31BV_1_5B_Q2K_SMALL_MULTILAYER_CLEAN | L0/L14/L27, seeds {0, 9} | 6 | 6 | 6 | 6 | 0 | 6 | +0.003872 | −0.005923 |
| 31BW | PASS_31BW_BROADER_LAYER_PROBE_PLAN_SELECTED | (planning only) | — | — | — | — | — | — | — | — |
| 31BX | PASS_31BX_1_5B_Q2K_STRATIFIED_LAYER_CLEAN | L0/L4/L8/L12/L16/L20/L27, seeds {0, 9} | 14 | 14 | 14 | 14 | 0 | 14 | +0.004641 | −0.010987 |

**Cumulative:** 23 pairs across 3 independent runners. 0 severe, 0 memory-fail, 0 finite-fail. L0 result is bit-identical across 31BU, 31BV, and 31BX (reproducibility verified, not assumed). 31BX per-layer margin variance is 32 bytes (3,380,342 to 3,380,374) — numerically stable across the layers observed. Worst observed pair is L16-S9 (Δcos = +0.000652, MAE Δ = −0.004184) — still strongly positive.

---

## 3. Selected policy (unchanged from 31BX)

| field | value |
|---|---|
| policy package | `corrected_q2k_policy_v1` |
| Q2_K mode | `corrected_ceil_per_row` |
| residual families | `ffn_up` + `ffn_gate` |
| residual k | 0.5% |
| alpha | 1.0 |
| ffn_down residual | **none** (W_low only) |
| W_ref | Qwen2.5-1.5B-Instruct Q4_K_M, **dequantized** (NOT FP16) |
| hidden | 1536 |
| intermediate | 8960 |
| batch | 1 |
| RNG | `np.random.default_rng(seed).standard_normal((1, 1536))` |
| MLP formula (31BT canonical) | `Y = (silu(X @ W_gate.T) * (X @ W_up.T)) @ W_down.T` |

No policy changes. The 31BY recommendation reuses the exact policy that passed 31BU, 31BV, and 31BX.

---

## 4. Options considered

| id | name | layers | seeds | pairs | crosses into aggregate? |
|---|---|---|---|---:|:---:|
| **A** | **GO full 28-layer × 2-seed aggregate** | 0–27 | {0, 9} | **56** | **yes** |
| B | GO 28-layer × 1-seed aggregate first | 0–27 | {9} | 28 | yes |
| C | MORE SEED SENSITIVITY before full aggregate | {0, 8, 16, 27} | {0,1,2,3,4,5,9,13} | 32 | no (still stratified by layer subset) |
| D | STOP / FREEZE current 1.5B probe evidence | — | — | 0 | no |

---

## 5. Cost / risk table

**Cost model caveat.** 31BX wall clock was observed at ~9 minutes for 14 pairs in a single process. The per-pair cost is dominated by the one-time model dequantize on first layer touch; subsequent layers amortize the load because the GGUF file is hot in the page cache. So per-run cost grows sub-linearly with pair count when pairs are run in a single process. Estimates below are:

- **working** = realistic best estimate based on 31BX observation
- **upper** = conservative ceiling (1.4× working to absorb variance)

| option | pairs | runtime (min, working) | runtime (min, upper) | peak temp disk / layer | peak resident mem | artifact size | scope-creep risk |
|---|---:|---:|---:|---:|---:|---:|:---:|
| A | 56 | **25** | **35** | 17,263,466 B | ~120 MB | ~60 KB | low (locked scope) |
| B | 28 | 16 | 22 | 17,263,466 B | ~120 MB | ~40 KB | low (budget-cut A) |
| C | 32 | 14 | 20 | 17,263,466 B | ~120 MB | ~40 KB | moderate (no layer-coverage gain) |
| D | 0 | 0 | 0 | 0 | 0 | 0 | N/A |

Disk estimate per layer: Q2_K blob (3 × 4.5 MB) + SDIR residual (~1.9 MB × 2 families) + Python overhead ≈ 17.3 MB peak, freed after the layer's pair(s) complete. Single-process runner never holds more than one layer's worth on disk at a time.

---

## 6. Stop / go recommendation: **Option A — GO**

Selected next phase: **Phase 31BZ — Qwen2.5-1.5B Corrected Q2_K Full-Layer Two-Seed Aggregate**.

Scope: **layers 0–27 × seeds {0, 9} = 56 pairs**, same `corrected_q2k_policy_v1`, same Q4_K_M dequantized W_ref, standalone tensor-harness aggregate, no generation, no inference, no llama.cpp runtime integration, no model files committed, no Q2_K/SDIR blobs committed.

### Rationale

1. **Three prior accepted phases, all clean.** 31BU (L0), 31BV (L0/L14/L27), and 31BX (L0/L4/L8/L12/L16/L20/L27) all PASS. 23 cumulative pairs, 0 severe, 0 memory-fail, 0 finite-fail.
2. **Worst pair is still strongly positive.** 31BX worst is L16-S9 (Δcos = +0.000652, MAE Δ = −0.004184) — the smallest gain in the set, but still in the green by a wide margin. There is no current signal of a layer where the policy is misbehaving.
3. **Reproducibility is verified, not assumed.** L0 result is bit-identical across 31BU, 31BV, and 31BX (3 independent runners). The same X vector, same corrected_ceil_per_row Q2_K, same SDIR k=0.5% reproduces to the bit. This rules out "the previous passes were lucky" as a plausible concern.
4. **The step ladder is controlled.** Anchor (1 layer, 3 seeds) → small multi-layer (3 layers, 2 seeds) → stratified (7 layers, 2 seeds) → full-layer aggregate (28 layers, 2 seeds). Each rung ran under the same policy, same runner family, same harness, same W_ref source. There is no rung in the ladder where the policy failed and was retried.
5. **The two seeds 0 and 9 are sufficient.** 31BV and 31BX used seeds 0 and 9; 31BU added seed 13 to triple-check L0. The two-seed choice catches the "all-bad-on-this-seed" failure mode for a corrected Q2_K policy on Gaussian-activation harness inputs. Adding more seeds (Option C) is strictly weaker than increasing layer coverage at this point.
6. **Option B is strictly weaker than A at comparable cost.** 16 min (28 pairs) vs 25 min (56 pairs) — 56% more pairs for 56% more time, but the second seed is what lets you distinguish "this layer is bad regardless of activation noise" from "this layer is sensitive to activation direction."
7. **Option C does not advance the project.** Going from 7 of 28 layers to 4 of 28 layers with 8 seeds is a regression in layer coverage (14% vs 25%) for a 2.3× increase in pairs. It does not move the project from stratified to full-layer evidence.
8. **Option D is unjustified.** Freezing now would leave us with 7-of-28 layer coverage — exactly the gap that 31BX was designed to quantify the cost of filling, and 31BX passed. The current evidence is not pointing at a problem; it is pointing at a clear next bounded step.

---

## 7. Success criteria for 31BZ

### CLEAN — strict definition

CLEAN classification requires **ALL** of the following simultaneously. **The 2-pair allowance applies ONLY to the two soft thresholds in §7.2. All other criteria are HARD — 1 violation of any of them disqualifies CLEAN.**

### 7.1 Hard criteria (no allowance — 1 violation = not CLEAN)

| criterion | required value |
|---|---:|
| regression passes (before AND after) | `PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN`, `error_count=0`, `fallback_count=0`, `readme_drift_guard.passed=true` |
| finite pairs | **56/56** (n_finite = 56) — any non-finite pair disqualifies CLEAN |
| memory-positive pairs | **56/56** (n_memory_positive = 56) — **any memory-negative pair/layer disqualifies CLEAN → `PARTIAL_31BZ_1_5B_Q2K_AGGREGATE_MEMORY_FAIL`, period** |
| severe regressions | **0** (n_severe = 0) — any severe regression disqualifies CLEAN |
| model files committed | **none** |
| Q2_K / SDIR blobs committed | **none** (all temp under `tempfile.mkdtemp`) |
| scope creep | **none** beyond layers 0–27, seeds {0, 9}, no ffn_down residual, no FP16 W_ref |

### 7.2 Soft thresholds (2-pair allowance applies ONLY here)

| threshold | value | allowance |
|---|---:|---|
| cosine-improved pairs | ≥ 54/56 | up to 2 pairs may fail cos+, as long as they are finite, memory-positive, and non-severe |
| MAE-improved pairs | ≥ 54/56 | up to 2 pairs may fail MAE+, as long as they are finite, memory-positive, and non-severe |

**Independence rule:** the 2-pair allowance for cos+ and the 2-pair allowance for MAE+ are **independent**. A single pair may fail cos+ AND fail MAE+ and still count as 1 toward each allowance. CLEAN is met if at most 2 pairs fail cos+ AND at most 2 pairs fail MAE+ (independent of overlap).

### 7.3 Strictness justification

54/56 (96.4%) is the same proportional allowance as 31BX's 14/14 (100% — but only 14 pairs) and 31BV's 6/6 (100% — but only 6 pairs). Option A's 56 pairs are a 4× increase over 31BX, so a 2-pair allowance reflects the proportional increase in sample size, **not a relaxation of the per-pair pass criterion**. The HARD criteria (finite, memory-positive, non-severe) carry no allowance because they are pre-conditions for the per-pair result to be meaningful at all — a non-finite or memory-negative or severe pair is a **structural failure**, not a soft one.

---

## 8. Stop conditions for 31BZ

### 8.1 Severity routing (which PARTIAL/BLOCKED to use)

| failure mode | hard/soft | CLEAN-eligible? | routing |
|---|:---:|:---:|---|
| any non-finite pair | hard | **no** | 1–2 non-finite pairs → `PARTIAL_31BZ_1_5B_Q2K_AGGREGATE_TRADEOFF`; **≥ 3 non-finite pairs → BLOCKED** (structural harness failure) |
| any memory-negative pair/layer | hard | **no** | **ALWAYS `PARTIAL_31BZ_1_5B_Q2K_AGGREGATE_MEMORY_FAIL`**, regardless of count or severity. Memory is the core accounting claim — any single memory-negative result breaks the claim. |
| any severe regression (Δcos < −0.05) | hard | **no** | ALWAYS `PARTIAL_31BZ_1_5B_Q2K_AGGREGATE_TRADEOFF` (unless the severe pair is also memory-negative, in which case `PARTIAL_MEMORY_FAIL` takes precedence) |
| 1–2 pairs not cos+ (and finite, mem+, non-severe) | soft | **yes** (under 2-pair allowance) | if 3+ pairs not cos+ → `PARTIAL_31BZ_1_5B_Q2K_AGGREGATE_MINOR_FAILURES` |
| 1–2 pairs not MAE+ (and finite, mem+, non-severe) | soft | **yes** (under 2-pair allowance) | if 3+ pairs not MAE+ → `PARTIAL_31BZ_1_5B_Q2K_AGGREGATE_MINOR_FAILURES` |

### 8.2 Stop immediately and classify BLOCKED if

- regression fails (`BLOCKED_SOURCE_OF_TRUTH_REGRESSION`)
- `SDI_MODEL_DIR` unset (`BLOCKED_31BZ_SDI_MODEL_DIR_UNSET`)
- model file missing (`BLOCKED_31BZ_MODEL_FILE_MISSING`)
- corrected Q2_K encode/dequantize backend fails on any layer (`BLOCKED_31BZ_Q2K_BACKEND_FAIL`)
- SDIR residual encode/decode fails on any pair (`BLOCKED_31BZ_SDIR_FAIL`)
- **≥ 3 non-finite pairs** (BLOCKED; structural harness failure)

### 8.3 Stop and classify `PARTIAL_31BZ_1_5B_Q2K_AGGREGATE_MEMORY_FAIL` if

- **ANY single memory-negative pair/layer, regardless of count or other criteria** — this is a HARD criterion with no allowance

### 8.4 Stop and classify `PARTIAL_31BZ_1_5B_Q2K_AGGREGATE_TRADEOFF` if

- 1–2 non-finite pairs (and no memory-negative, no severe)
- any severe regression (Δcos < −0.05) on any pair — unless memory-negative, in which case `PARTIAL_MEMORY_FAIL` takes precedence

### 8.5 Stop and classify `PARTIAL_31BZ_1_5B_Q2K_AGGREGATE_MINOR_FAILURES` if

- all hard criteria met, 0 severe, all finite, all memory-positive, regression passes, no scope-creep / model-file / blob violations — **but** 3+ pairs fail cos+ **or** 3+ pairs fail MAE+ (allowances exceeded)

### 8.6 Classify `CLEAN` if and only if

**ALL of:**
- 56/56 finite
- 56/56 memory-positive
- 0 severe
- regression passes before AND after
- no model files / blobs committed
- no scope creep
- AND independently at most 2 pairs fail cos+ AND at most 2 pairs fail MAE+

**The 2-pair allowance applies ONLY to the cos+ and MAE+ soft thresholds.** Any violation of a hard criterion is CLEAN-disqualifying regardless of cos+/MAE+ counts.

---

## 9. Classifications for 31BZ

| classification | meaning |
|---|---|
| `PASS_31BZ_1_5B_Q2K_FULL_LAYER_TWO_SEED_AGGREGATE_CLEAN` | All hard criteria met (regression passes, 56/56 finite, 56/56 memory-positive, 0 severe, no model files, no blobs, no scope creep) **AND** at most 2 pairs fail cos+ **AND** at most 2 pairs fail MAE+ (the two allowances are independent) |
| `PARTIAL_31BZ_1_5B_Q2K_AGGREGATE_MINOR_FAILURES` | All hard criteria met (56/56 finite, 56/56 memory-positive, 0 severe, regression passes) AND no scope-creep / model-file / blob violations, but **3+ pairs fail cos+ OR 3+ pairs fail MAE+** (the allowances are exceeded) |
| `PARTIAL_31BZ_1_5B_Q2K_AGGREGATE_TRADEOFF` | 1–2 non-finite pairs (and no memory-negative, no severe), **OR** any severe regression (Δcos < −0.05) on any pair (unless memory-negative, in which case `PARTIAL_MEMORY_FAIL` takes precedence) |
| `PARTIAL_31BZ_1_5B_Q2K_AGGREGATE_MEMORY_FAIL` | **ANY single memory-negative pair or layer**, regardless of count or other criteria. Memory-positive is a HARD criterion with no allowance. |
| `BLOCKED_31BZ_SDI_MODEL_DIR_UNSET` | env var missing |
| `BLOCKED_31BZ_MODEL_FILE_MISSING` | model file missing |
| `BLOCKED_31BZ_Q2K_BACKEND_FAIL` | corrected Q2_K generation or dequantization fails |
| `BLOCKED_31BZ_SDIR_FAIL` | SDIR residual generation or application fails |
| `BLOCKED_SOURCE_OF_TRUTH_REGRESSION` | regression fails before or after the run |

---

## 10. Claim boundaries for 31BZ

### Allowed if 31BZ passes

> **"Corrected Q2_K policy remained memory-positive and improved cosine/MAE across a full-layer, two-seed Qwen2.5-1.5B standalone tensor-harness aggregate (28 layers × 2 seeds = 56 pairs)."**

That is the **only** allowed headline. No other phrasing is sanctioned.

### Still forbidden even if 31BZ passes

- no inference / generation / sampling claim
- no runtime / speedup / latency claim
- no model quality claim
- no behavior recovery claim
- no production readiness claim
- no llama.cpp integration claim
- no FP16 recovery claim
- no 0.5B-vs-1.5B generalization claim
- no broader model-family claim
- no claim that the result transfers to real activations or to a serving stack
- no claim that the result transfers to 7B / 14B / 32B / 72B / 110B+ Qwen2.5 or any other family

---

## 11. Artifacts planned for 31BZ (pending explicit approval)

- `docs/PHASE31BZ_1_5B_CORRECTED_Q2K_FULL_LAYER_TWO_SEED_AGGREGATE.md`
- `src/phase31bz_1_5b_corrected_q2k_full_layer_two_seed_aggregate.py`
- `src/results/PHASE31BZ_1_5B_CORRECTED_Q2K_FULL_LAYER_TWO_SEED_AGGREGATE.json`

No model files. No Q2_K / SDIR blobs. No HF cache. No temp tensor dumps. No aggregate outputs committed (only the explicit JSON above).

---

## 12. Limitations of this plan

1. **Single activation regime.** All 31BU/31BV/31BX inputs are `np.random.default_rng(seed).standard_normal((1, 1536))` — zero-mean unit-variance Gaussian on a single token. Real activations have non-zero mean, structured correlations, and longer sequences. **No claim about real-activation behavior is sanctioned** even if 31BZ passes.
2. **Single model.** Qwen2.5-1.5B-Instruct only. No claim transfers to 0.5B, 7B, 14B, 32B, 72B, 110B+, or to any other Qwen / Llama / Mistral / etc. family member.
3. **Single policy.** Only `corrected_q2k_policy_v1` is being tested. No claim transfers to any other k value, alpha, residual family set, or Q2_K mode.
4. **No ffn_down residual.** The policy intentionally uses W_low-only for ffn_down. No claim transfers to a policy that does add an ffn_down residual.
5. **Tensor-harness only.** This is a standalone tensor harness running in pure NumPy + ctypes against llama.cpp's Q2_K encode/dequantize. No claim transfers to an end-to-end inference run, a serving stack, or a llama.cpp integration.
6. **Memory accounting is the harness's accounting.** `memory_positive` here means "per-layer bytes under the harness's 3 × Q4_budget_family comparison is positive" — it is **not** a claim about runtime memory in any production system.

---

## 13. Classification

**`PASS_31BY_AGGREGATE_STOP_GO_PLAN_SELECTED`** — three prior accepted phases, 23 cumulative pairs all clean, L0 reproducible bit-identical across three runners, 7-of-28 layer coverage is the natural next-step gap to fill, Option A is the bounded step that fills it at the same two seeds with the same policy.

Next allowed phase if approved: **Phase 31BZ — Qwen2.5-1.5B Corrected Q2_K Full-Layer Two-Seed Aggregate**, only if explicitly requested.
