# Phase 31CB — Stale File Provenance Cleanup

> **Provenance / working-tree hygiene phase only.** No new validation. No generation. No inference. No llama.cpp runtime integration. No modification of scientific results. No tags. The 5 stale untracked 31BH-R2 / 31BJ files are moved to `rescue/stale_phase31bh_31bj/`. No commit, push, tag, delete, or move occurs without explicit Matt approval.

---

## 1. Goal

Resolve the 5 stale untracked 31BH-R2 / 31BJ files that have been carried through 31BU → 31BV → 31BW → 31BX → 31BY → 31BZ → 31BZ → 31CA → 31CB without being committed, deleted, or formally archived. Apply the **conservative cleanup disposition** that preserves auditability and removes working-tree noise.

---

## 2. Classification

**`PASS_31CB_STALE_FILE_PROVENANCE_CLEANED`** — 5 stale files inspected, classified, and moved to `rescue/stale_phase31bh_31bj/` with a provenance README. No file was deleted. No file was committed. No scientific result was modified. No model file was touched. No tag was created. Regression passes before and after.

---

## 3. Stale file inventory (as found, before 31CB)

| file | bytes | lines | mtime | classification (as-found) |
|---|---:|---:|---|---|
| `docs/PHASE31BH_R2_Q2K_ANCHOR_REPRO_FIX.md` | 5.7K | 154 | 2026-05-31 15:17 | `PASS_31BH_R2_HISTORICAL_ANCHORS_REPRODUCED` (MD header; contradicted by JSON) |
| `src/phase31bh_q2k_clean_reproduction.py` | 24K | 606 | 2026-05-31 15:15 | n/a (runner code; supports both `historical_floor_flat` and `corrected_ceil_per_row` Q2_K modes) |
| `src/results/PHASE31BH_Q2K_RUNTIME_QUANT_PATH.json` | 2.1K | 74 | 2026-05-31 13:41 | `PARTIAL_31BH_Q2K_BACKEND_ADDED_ANCHOR_MISMATCH` |
| `src/results/PHASE31BH_R2_Q2K_ANCHOR_REPRO_FIX.json` | 11K | 379 | 2026-05-31 15:32 | `BLOCKED_31BH_R2_ANCHORS_STILL_MISMATCH` (`historical_reproduction: {L21_S9: false, L21_S0: false}`) |
| `src/results/PHASE31BJ_ANCHOR_ONLY.json` | 2.1K | 73 | 2026-05-31 16:23 | n/a (raw single-pair L21-S9 / L21-S0 numbers; one entry has `margin: -261638`) |

---

## 4. Disposition table (per-file)

| file | disposition | rationale |
|---|---|---|
| `docs/PHASE31BH_R2_Q2K_ANCHOR_REPRO_FIX.md` | **MOVE_TO_RESCUE** | the MD header claims PASS for historical anchor reproduction, but the corresponding JSON (also in this rescue set) says BLOCKED with `historical_reproduction: {false, false}`. The MD-vs-JSON classification drift is real. Cannot be committed as-is (would mislead future agents). Cannot be safely deleted (has some audit value). MOVE is the only safe action. |
| `src/phase31bh_q2k_clean_reproduction.py` | **MOVE_TO_RESCUE** | runner code that supports BOTH `historical_floor_flat` and `corrected_ceil_per_row` Q2_K modes. The historical_floor_flat mode is not used by any current accepted phase. Cannot be committed to `src/` (it would be cited as a usable runner, and its "both modes" capability is no longer relevant). Cannot be safely deleted (the corrected_ceil_per_row branch overlaps with the current policy code). MOVE is the only safe action. |
| `src/results/PHASE31BH_Q2K_RUNTIME_QUANT_PATH.json` | **MOVE_TO_RESCUE** | classification `PARTIAL_31BH_Q2K_BACKEND_ADDED_ANCHOR_MISMATCH`. The Q2_K backend it added is now hardened and documented in `src/q2k_backend.py` + `docs/CORRECTED_Q2K_POLICY_PACKAGE.md`. Cannot be committed to `src/results/` (would appear as a current accepted result). Cannot be safely deleted (the Q2_K symbol-inventory data is the first proof that the Q2_K backend was loadable, which is a real audit fact). MOVE is the only safe action. |
| `src/results/PHASE31BH_R2_Q2K_ANCHOR_REPRO_FIX.json` | **MOVE_TO_RESCUE** | classification `BLOCKED_31BH_R2_ANCHORS_STILL_MISMATCH`. The historical-reproduction approach did not converge as of 31BH-R2. Cannot be committed (would appear as a current accepted result, which it is not). Cannot be safely deleted (the BLOCKED classification is a real audit fact that future agents investigating the 31BK evolution would benefit from). MOVE is the only safe action. |
| `src/results/PHASE31BJ_ANCHOR_ONLY.json` | **MOVE_TO_RESCUE** | raw single-pair L21-S9 / L21-S0 numbers. The `corrected_ceil_per_row` row of L21-S0 shows `margin: -261638` (memory-NEGATIVE for that single Q2_K W_low variant only). Cannot be committed to `src/results/` (would appear as a current accepted result, and the negative margin could be misread as contradicting the 0.5B/1.5B memory-positive aggregate without context). Cannot be safely deleted (the numbers are the basis for the 31BK/31BL/31BM re-baselining). MOVE is the only safe action. |

**Summary:** 5 of 5 files → `MOVE_TO_RESCUE`. No file → `COMMIT_AS_HISTORICAL_RECORD`. No file → `DELETE_AFTER_CONFIRMATION`. No file → `LEAVE_UNTRACKED_WITH_REASON`.

---

## 5. Rescue folder layout

```
rescue/
└── stale_phase31bh_31bj/
    ├── README.md                                                  (12.9 KB; this rescue README)
    ├── docs_PHASE31BH_R2_Q2K_ANCHOR_REPRO_FIX.md                  (5.7 KB; was docs/PHASE31BH_R2_Q2K_ANCHOR_REPRO_FIX.md)
    ├── src_phase31bh_q2k_clean_reproduction.py                    (24 KB; was src/phase31bh_q2k_clean_reproduction.py)
    ├── src_results_PHASE31BH_Q2K_RUNTIME_QUANT_PATH.json          (2.1 KB; was src/results/PHASE31BH_Q2K_RUNTIME_QUANT_PATH.json)
    ├── src_results_PHASE31BH_R2_Q2K_ANCHOR_REPRO_FIX.json         (11 KB; was src/results/PHASE31BH_R2_Q2K_ANCHOR_REPRO_FIX.json)
    └── src_results_PHASE31BJ_ANCHOR_ONLY.json                      (2.1 KB; was src/results/PHASE31BJ_ANCHOR_ONLY.json)
```

The original relative path is encoded in the filename (`docs/` → `docs_`, `src/results/` → `src_results_`) so future agents can recover the original location.

---

## 6. Rescue README — key contents

The `rescue/stale_phase31bh_31bj/README.md` (12.9 KB) is the canonical warning to future agents. It documents:

1. **What each stale file actually says** (the as-found classification).
2. **The specific contradictions found** (e.g. the 31BH-R2 MD claims PASS while the 31BH-R2 JSON says BLOCKED).
3. **Why `MOVE_TO_RESCUE` instead of `COMMIT_AS_HISTORICAL_RECORD`** (no unique historical evidence; contradictory classifications; cannot be retroactively marked superseded).
4. **Why `MOVE_TO_RESCUE` instead of `DELETE_AFTER_CONFIRMATION`** (some audit-trail value; the BLOCKED classification is a real signal; the raw single-pair numbers are the basis for later re-baselining).
5. **What the rescue folder is NOT** (not a commit; not canonical evidence; not a re-validation; not a source of new claims; not a clean-up of misleading content).
6. **What the rescue folder IS** (provenance record; forensic landing zone; clean working tree; explicit warning).
7. **Re-validation rule** (if a future phase needs to re-validate historical anchors, it must re-derive from scratch using the current `src/corrected_q2k_policy.py` constants and the current `src/q2k_backend.py` + `src/phase31x_manifest_runtime.py` implementations; not import stale runner code).
8. **Audit trail** (step-by-step 31CB actions; regression results; SOT changes).
9. **No scientific / numeric claim changes** (table showing every claim unchanged).

---

## 7. What 31CB did NOT do (upheld)

- ❌ Did not run new validation
- ❌ Did not run generation / inference / sampling
- ❌ Did not run llama.cpp runtime integration
- ❌ Did not modify scientific results
- ❌ Did not delete any file (all 5 moved, none destroyed)
- ❌ Did not commit any file (5 moved + 1 README + 1 31CB doc + 1 SOT edit = 8 uncommitted, awaiting approval)
- ❌ Did not push anything
- ❌ Did not create a tag
- ❌ Did not touch any model file (model is at `$SDI_MODEL_DIR/...` outside the repo)
- ❌ Did not modify the 0.5B or 1.5B frozen evidence tiers
- ❌ Did not modify the `corrected_q2k_policy_v1` package
- ❌ Did not modify any prior phase's committed artifacts

---

## 8. No scientific / numeric claim changes

| claim | before 31CB | after 31CB |
|---|---|---|
| 0.5B `corrected_q2k_policy_v1` is memory-positive (384/384) | accepted via 31BN/31BM | **unchanged** |
| 0.5B `corrected_q2k_policy_v1` is cos-improved (383/384) | accepted via 31BN/31BM | **unchanged** |
| 0.5B `corrected_q2k_policy_v1` is MAE-improved (383/384) | accepted via 31BN/31BM | **unchanged** |
| 1.5B `corrected_q2k_policy_v1` is memory-positive (56/56) | accepted via 31BZ/31CA | **unchanged** |
| 1.5B `corrected_q2k_policy_v1` is cos-improved (56/56) | accepted via 31BZ/31CA | **unchanged** |
| 1.5B `corrected_q2k_policy_v1` is MAE-improved (56/56) | accepted via 31BZ/31CA | **unchanged** |
| 31AY/31BA L21-S9 / L21-S0 historical anchor values | historical-only, NOT current canonical metrics (per SOT Section 0.A) | **unchanged** |
| 31BJ single-pair L21-S9 / L21-S0 raw numbers | rescue-only, NOT current canonical evidence | **unchanged** |
| 31BH-R2 PASS/BLOCKED classification drift | rescue-only, NOT current canonical evidence | **unchanged** |

The 31CB cleanup is **provenance and working-tree hygiene only**. It does not validate, invalidate, supersede, or re-classify any prior phase. The 0.5B and 1.5B frozen evidence tiers remain exactly as 31CA froze them.

---

## 9. SOT updates performed in 31CB

| section | change |
|---|---|
| **Section 0** (current blockers / provenance debt) | "Current blockers: none" → "Current blockers: none (provenance debt resolved in 31CB: 5 stale 31BH-R2 / 31BJ files moved to `rescue/stale_phase31bh_31bj/` with provenance README; not committed, not deleted, not promoted to canonical evidence)." |
| **Section 0** (current allowed scientific next phase) | 31CB → **31CC** (recommended default per 31CB reasoning: real-activation capture planning) |
| **Section 0** (model under test line) | appended 31CB cleanup note |
| **Section 0** (validation scope line) | appended 31CB cleanup note |
| **Section 3** | new 31CB accepted fact appended after 31CA (full entry with scope, classification, per-file disposition table, rescue folder contents, audit trail, no-claim-changes table, "valid as long as") |
| **Section 9** | current allowed next phase: 31CB → **31CC**; rationale rewritten to reference 31CB's PASSED state and outline 31CC as the recommended next (real-activation capture planning) |

---

## 10. Recommended next phase (31CC)

Per the 31CB spec, the recommended default is **Phase 31CC — Qwen2.5-1.5B Real-Activation Capture Planning, only if explicitly requested**. Rationale: after stale provenance cleanup, the next scientific gap is real prompt-derived activation behavior, not more random-activation tensor harness work. Alternative: 31CC — Runtime Artifact Format / Loader Planning.

---

## 11. Forbidden claims (upheld)

31CB did **not** run, did not validate, did not invalidate, did not supersede, did not re-classify. The forbidden claims from 31CA remain forbidden:

- no model quality recovery claim
- no behavior recovery claim
- no speedup claim
- no full-model runtime memory savings claim
- no llama.cpp integration claim
- no production readiness claim
- no inference / generation / sampling claim
- no runtime-ready output-residual claim
- no FP16 recovery claim
- no 0.5B-vs-1.5B generalization claim
- no "1.5B behaves like 0.5B" claim
- no broader model-family claim
- no real-activation-transfer claim
- no larger-model claim
- no claim that 31AY / 31BA exact anchors are current canonical metrics
- no claim that 31BH-R2 / 31BJ numbers are current canonical evidence (they are rescue-only)

---

## 12. Files 31CB creates (uncommitted, awaiting approval)

- `rescue/stale_phase31bh_31bj/README.md` (12.9 KB)
- `rescue/stale_phase31bh_31bj/docs_PHASE31BH_R2_Q2K_ANCHOR_REPRO_FIX.md` (5.7 KB; moved)
- `rescue/stale_phase31bh_31bj/src_phase31bh_q2k_clean_reproduction.py` (24 KB; moved)
- `rescue/stale_phase31bh_31bj/src_results_PHASE31BH_Q2K_RUNTIME_QUANT_PATH.json` (2.1 KB; moved)
- `rescue/stale_phase31bh_31bj/src_results_PHASE31BH_R2_Q2K_ANCHOR_REPRO_FIX.json` (11 KB; moved)
- `rescue/stale_phase31bh_31bj/src_results_PHASE31BJ_ANCHOR_ONLY.json` (2.1 KB; moved)
- `docs/PHASE31CB_STALE_FILE_PROVENANCE_CLEANUP.md` (this file; 31CB phase doc)
- `SOURCE_OF_TRUTH.md` (modified; 31CB accepted fact + Section 0/9 updates)

No file is committed. Commit requires explicit Matt approval.

---

## 13. What future agents should do

1. Read this 31CB doc + `rescue/stale_phase31bh_31bj/README.md` if you encounter any reference to 31BH-R2, 31BJ, or the historical-anchor reproduction work.
2. **Do not cite the rescue files as current evidence.** They are forensic landing zone only.
3. **Do not import the rescue runner code** (`src_phase31bh_q2k_clean_reproduction.py`). Use the current `src/corrected_q2k_policy.py` constants and the current `src/q2k_backend.py` + `src/phase31x_manifest_runtime.py` implementations.
4. The canonical evidence is in `SOURCE_OF_TRUTH.md` Section 0 + Section 3 (31BN/31BM, 31BZ/31CA) and in `docs/CORRECTED_Q2K_POLICY_PACKAGE.md`.
5. The 0.5B and 1.5B frozen evidence tiers are the only current canonical evidence. All 31BH-R2 / 31BJ claims are historical-only unless re-validated under a future phase.
