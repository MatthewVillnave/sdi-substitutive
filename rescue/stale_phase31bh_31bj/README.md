# Rescue: Stale Phase 31BH-R2 / 31BJ Artifacts

> **Status:** RESCUE / PROVENANCE / NOT CANONICAL EVIDENCE
> **Origin:** these files were untracked working-tree artifacts that were carried through 31BU → 31BV → 31BW → 31BX → 31BY → 31BZ → 31BZ (post-31BZ → 31CA → 31CB) without ever being committed, deleted, or formally archived.
> **Cleanup phase:** Phase 31CB — Stale File Provenance Cleanup.
> **Disposition:** `MOVE_TO_RESCUE`. NOT `COMMIT_AS_HISTORICAL_RECORD`. NOT `DELETE_AFTER_CONFIRMATION`.

---

## 1. Why these files were moved (not committed, not deleted)

The 5 files in this folder were inspected during 31CB and found to contain **conflicting / partially-superseded / partially-misleading** content relative to the now-canonical `corrected_q2k_policy_v1` package and the two frozen evidence tiers (0.5B 31BN/31BM/31BO and 1.5B 31BZ/31CA).

### Specific contradictions found

| stale file | what it says | what SOT / current evidence says | conflict severity |
|---|---|---|---|
| `docs_PHASE31BH_R2_Q2K_ANCHOR_REPRO_FIX.md` | classification `PASS_31BH_R2_HISTORICAL_ANCHORS_REPRODUCED` | the corresponding JSON `src_results_PHASE31BH_R2_Q2K_ANCHOR_REPRO_FIX.json` (in this same folder) says classification `BLOCKED_31BH_R2_ANCHORS_STILL_MISMATCH` and shows `historical_reproduction: {L21_S9: false, L21_S0: false}` | **HIGH** — the MD claims PASS, the JSON says BLOCKED. Future agents reading only the MD would be misled. |
| `docs_PHASE31BH_R2_Q2K_ANCHOR_REPRO_FIX.md` (header) | claims historical-anchor reproduction | the historical anchors (31AY/31BA L21-S9 and L21-S0) were later **explicitly re-baselined** by 31BJ/31BK/31BL/31BM/31BN under the corrected `corrected_ceil_per_row` policy, which is the now-canonical evidence chain | superseded |
| `src_phase31bh_q2k_clean_reproduction.py` | 606-line runner supporting BOTH `historical_floor_flat` AND `corrected_ceil_per_row` Q2_K modes; fixes X-vector RNG to match old 31AY/31BA path | superseded by the 31BK runner (which established the canonical `corrected_ceil_per_row` policy) and the 31BM / 31BZ runners (which executed the full-aggregate validation). The "both modes" capability is no longer needed — the historical_floor_flat mode is not used by any current accepted phase. | superseded |
| `src_results_PHASE31BH_Q2K_RUNTIME_QUANT_PATH.json` | classification `PARTIAL_31BH_Q2K_BACKEND_ADDED_ANCHOR_MISMATCH` — first attempt to add the Q2_K backend; anchor mismatch | superseded. The Q2_K backend was hardened and the anchor mismatch resolved by 31BJ/31BK/31BL/31BM. The current accepted Q2_K backend is documented in `src/q2k_backend.py` and `docs/CORRECTED_Q2K_POLICY_PACKAGE.md`. | superseded |
| `src_results_PHASE31BH_R2_Q2K_ANCHOR_REPRO_FIX.json` | classification `BLOCKED_31BH_R2_ANCHORS_STILL_MISMATCH` — anchor reproduction still failed even after the fix | contradicted by the MD in the same folder. Note: the SOT 31BH-R2 entry uses a different classification name (`PASS_31BH_R2_FIX_RUNTIME_DISPATCH_REPAIRED`) which refers to the runtime-dispatch fix specifically, not to anchor reproduction. The MD-vs-JSON classification drift is real. | **HIGH** — contradictory and superseded |
| `src_results_PHASE31BJ_ANCHOR_ONLY.json` | L21-S9 / L21-S0 raw numbers with `historical_floor_flat` and `corrected_ceil_per_row`; one `corrected_ceil_per_row` entry shows `margin: -261638` (memory-NEGATIVE for that single Q2_K W_low variant) | superseded. The corrected_ceil_per_row memory-positive claim is now established across 384 pairs in 31BN/31BM (0.5B) and 56 pairs in 31BZ (1.5B). The single-pair -261638 margin in this file is **NOT** a contradiction of those aggregate claims (it was a single Q2_K W_low variant for L21-S0, not the full policy), but it could be misread as contradicting the aggregate without context. | low–medium — needs context to interpret safely |

### Why `MOVE_TO_RESCUE` instead of `COMMIT_AS_HISTORICAL_RECORD`

`COMMIT_AS_HISTORICAL_RECORD` would require the file to:
1. contain unique historical evidence that is **not** in SOT or other committed artifacts, AND
2. be clearly marked superseded/historical in its own header

Neither condition is met:
1. Every fact in these files is either (a) superseded by 31BJ/31BK/31BL/31BM/31BN/31BO (0.5B) or 31BS/31BT/31BU/31BV/31BW/31BX/31BY/31BZ/31CA (1.5B), or (b) contradicted within the stale files themselves.
2. The MD file (`docs_PHASE31BH_R2_Q2K_ANCHOR_REPRO_FIX.md`) is **not** clearly marked superseded — its header says `PASS_31BH_R2_HISTORICAL_ANCHORS_REPRODUCED` (the contradicted PASS claim). Marking it as superseded retroactively would be editing the historical record.

### Why `MOVE_TO_RESCUE` instead of `DELETE_AFTER_CONFIRMATION`

These files do contain some audit-trail value:
- The `BLOCKED_31BH_R2_ANCHORS_STILL_MISMATCH` classification in the 31BH-R2 JSON is a real signal that the anchor-reproduction approach had not yet converged as of 31BH-R2. Deleting it would erase that signal from the repo.
- The 31BJ `ANCHOR_ONLY` JSON contains the raw single-pair numbers that were the basis for the 31BK/31BL/31BM re-baselining. The numbers themselves are reproducible (they're just `np.random.default_rng(seed).standard_normal((1, hidden))` × corrected Q2_K × Q4_K_M W_ref at a specific seed/layer).
- The 31BH-R2 runner code (`phase31bh_q2k_clean_reproduction.py`) shows the "both modes" pattern that was tried before the policy converged on `corrected_ceil_per_row` only.

Moving to rescue preserves the audit trail without letting the files be cited as current evidence.

---

## 2. Files in this rescue folder

| file (rescue path) | original path | bytes | lines | original mtime | original classification | rescue disposition |
|---|---|---:|---:|---|---|---|
| `docs_PHASE31BH_R2_Q2K_ANCHOR_REPRO_FIX.md` | `docs/PHASE31BH_R2_Q2K_ANCHOR_REPRO_FIX.md` | 5.7K | 154 | 2026-05-31 15:17 | `PASS_31BH_R2_HISTORICAL_ANCHORS_REPRODUCED` (contradicted by JSON) | moved to rescue |
| `src_phase31bh_q2k_clean_reproduction.py` | `src/phase31bh_q2k_clean_reproduction.py` | 24K | 606 | 2026-05-31 15:15 | n/a (runner code) | moved to rescue |
| `src_results_PHASE31BH_Q2K_RUNTIME_QUANT_PATH.json` | `src/results/PHASE31BH_Q2K_RUNTIME_QUANT_PATH.json` | 2.1K | 74 | 2026-05-31 13:41 | `PARTIAL_31BH_Q2K_BACKEND_ADDED_ANCHOR_MISMATCH` | moved to rescue |
| `src_results_PHASE31BH_R2_Q2K_ANCHOR_REPRO_FIX.json` | `src/results/PHASE31BH_R2_Q2K_ANCHOR_REPRO_FIX.json` | 11K | 379 | 2026-05-31 15:32 | `BLOCKED_31BH_R2_ANCHORS_STILL_MISMATCH` | moved to rescue |
| `src_results_PHASE31BJ_ANCHOR_ONLY.json` | `src/results/PHASE31BJ_ANCHOR_ONLY.json` | 2.1K | 73 | 2026-05-31 16:23 | n/a (raw single-pair numbers) | moved to rescue |

The original relative path is encoded in the filename (with `/` replaced by `_`) so future agents can find the original location. The 5 files were moved with `git mv` is **not** applicable here because they were untracked (no HEAD reference to move from). They were moved with `mv` and the new paths are untracked.

---

## 3. What this rescue folder is NOT

- ❌ Not a commit. None of these files are in the git tree.
- ❌ Not canonical evidence. `SOURCE_OF_TRUTH.md` is canonical. `docs/CORRECTED_Q2K_POLICY_PACKAGE.md` is canonical. The two frozen evidence tiers (0.5B 31BN/31BM and 1.5B 31BZ/31CA) are canonical.
- ❌ Not a re-validation. Nothing in this folder was re-run, re-validated, or re-classified. The classifications shown in §1 are the **as-found** classifications from the stale files themselves, surfaced for forensic visibility only.
- ❌ Not a source of new claims. The "claims" in these files (PASS / BLOCKED / PARTIAL / etc.) are not added to SOT, not added to the policy package, and not cited anywhere in 31CA or 31CB.
- ❌ Not a clean-up-of-misleading-content. The contradicting classifications remain in the files as-found. 31CB does not edit them.

---

## 4. What this rescue folder IS

- ✅ A provenance record that the 5 stale files existed in the working tree, what they said, and why they were not promoted to canonical evidence.
- ✅ A forensic landing zone: if a future agent needs to investigate the 31BH-R2 → 31BK evolution (e.g. to understand why the policy converged on `corrected_ceil_per_row`), the raw artifacts are here, not deleted.
- ✅ A clean working tree: the 5 stale untracked files no longer appear in `git status --short` at their original paths.
- ✅ An explicit warning: the files in this folder **must not be cited as current evidence** by any future phase. The rescue README is the canonical warning.

---

## 5. Re-validation rule (for any future phase)

If a future phase needs to re-validate the historical anchors (e.g. to refresh the 31AY/31BA reference metrics or to re-test the `historical_floor_flat` Q2_K mode for any reason), the future phase **must**:

1. Read this rescue README first.
2. Re-derive the metrics from scratch using the current `src/corrected_q2k_policy.py` constants and the current `src/q2k_backend.py` + `src/phase31x_manifest_runtime.py` implementations. Do not import the stale runner code from this folder.
3. Record the re-validation under a new phase name (e.g. `Phase 31CC` or later) with a fresh result JSON, not by editing the stale files in this folder.
4. Not cite the stale files in this folder as current evidence, even after re-validation.

---

## 6. Audit trail for the 31CB cleanup

| step | command / action | result |
|---|---|---|
| 1 | `git status --short` (pre-31CB) | 5 stale untracked files at original paths |
| 2 | `ls -lh`, `wc -l`, `head`, `tail` on each file | full content captured (see 31CB PRE-COMMIT REPORT) |
| 3 | `git ls-files`, `grep` for stale-file references in SOT and committed docs | all SOT references are meta-references (about the stale files), not into the stale files |
| 4 | `python3 -m tests.run_source_of_truth_regression` (pre-cleanup) | `PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN`, `error_count=0`, `fallback_count=0`, `readme_drift_guard.passed=true` |
| 5 | `mkdir -p rescue/stale_phase31bh_31bj` | folder created (untracked) |
| 6 | `mv` each stale file to `rescue/stale_phase31bh_31bj/` with renamed filename | 5 files moved (untracked) |
| 7 | this README written | `rescue/stale_phase31bh_31bj/README.md` (untracked) |
| 8 | SOURCE_OF_TRUTH.md updated (31CB accepted fact appended; Section 0 + Section 9 updated) | SOT modified (uncommitted, awaiting approval) |
| 9 | `python3 -m tests.run_source_of_truth_regression` (post-cleanup) | must remain `PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN` |
| 10 | `git status --short` (post-cleanup) | no stale files at original paths; 5 files at new rescue paths (untracked); SOT modified (uncommitted) |

The 5 files are moved but **not committed** in 31CB. Commit requires explicit Matt approval.

---

## 7. No scientific / numeric claim changes

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

## 8. Related references

- `SOURCE_OF_TRUTH.md` Section 0 (Current Working State)
- `SOURCE_OF_TRUTH.md` Section 3 (Accepted Known-Good Facts) — 31BH (line 382), 31BH-R2 (line 401), 31BJ (line 418), 31BK (line 431), 31BL (line 448), 31BM (line 455), 31BN (line 464), 31CA (line ~815)
- `SOURCE_OF_TRUTH.md` Section 9 (Current Allowed Next Phase) — points to 31CB during this phase, then to 31CC after commit
- `docs/CORRECTED_Q2K_POLICY_PACKAGE.md` — canonical policy manifest with 0.5B and 1.5B evidence tiers
- `docs/PHASE31CA_1_5B_AGGREGATE_FREEZE_PACKAGE.md` — 31CA freeze document
- `docs/PHASE31CB_STALE_FILE_PROVENANCE_CLEANUP.md` — 31CB cleanup report (written during this phase)

This rescue folder was created during Phase 31CB — Stale File Provenance Cleanup. It is part of the 31CB commit pending Matt's explicit approval.
