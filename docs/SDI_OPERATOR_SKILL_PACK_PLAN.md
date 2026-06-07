# SDI Operator Skill Pack — Planning Document

> **Classification:** `PASS_SDI_OPERATOR_SKILL_PACK_PLAN_SELECTED`
> **Status:** Planning only. **Skills NOT implemented yet.**
> **Scope:** Metadata-only planning doc. No scientific code modified. No results modified. No SOT modified. No commit. No push. No tag.
> **Repo:** `sdi-substitutive` (operator: matthew-villnave)
> **Current HEAD (read-only preflight):** `27be90df1988035c2fa294f753e3990aaf4cf369` (Phase 31CN post-push)

---

## 1. Purpose

The SDI (Substitutive Decoding Infrastructure) project has a stable, repeated **scientific phase workflow** that has now been executed end-to-end six times in a row: 31CH → 31CI → 31CJ → 31CL → 31CM → 31CN. Each phase followed the same shape:

1. **Preflight**: branch / HEAD / status / log / fetch / SOT-drift scan / regression
2. **Read SOT** first, report current allowed next phase, confirm no conflict
3. **Read inputs** (previous phase's outputs + plan + design)
4. **Implement** (Python validator / C++ capture / planner — metadata-only or runtime-isolated)
5. **Run self-tests** (embedded, no model load)
6. **Write doc** (`docs/PHASE31xx_…md`)
7. **Update SOT** (Section 0 line 5 state label, line 11 next-allowed-phase, line 15 journal entry)
8. **Run post-edit regression** (`tests/run_source_of_truth_regression`)
9. **PRE-COMMIT REPORT** (per-spec checklist, fences A-F, no fabrication)
10. **Wait for explicit approval** before commit / push
11. **Post-push verification** (real commands: `git log`, `git show`, `git rev-parse`, `git ls-remote`, `git tag`, fence scans)

This is a lot of repeated structure, with a few high-risk pitfalls that have already been hit:

| mistake | where it happened | how it could have been prevented |
|---|---|---|
| **SOT drift** (state label said "no commit yet" after commit) | 31CL → 31CL hotfix cycle, 31CM line 5 wording | a `sot_drift_scan` skill that runs *before* the post-edit regression would have caught it |
| **Fabricated or imprecise commit/push reports** (forgotten fence, claimed clean when not) | every pre-commit report; the 31CM scan-scope bug was caught only by 31CM's own self-test on real data | a `precommit_report_builder` skill that runs the *actual* required commands (no human memory) and a `postpush_verifier` skill that double-checks fences A-F would prevent this |
| **Self-test scope drift** (negative mutations targeting the wrong field, e.g. `planned_path` instead of `planned_filename`) | 31CM self-tests, 31CN self-tests — both required multiple iterations to fix | an `artifact_hygiene_scan` skill that audits the planned_fields-vs-scan_fields contract would have caught this in the first self-test run |
| **Validator-API mismatch** (bundle_type hard-restricted; R-provenance hard-requires `'31CI'`) | 31CN (both bundle_type and created_by_phase) | a `phase_gate_check` skill that pre-flights the validator API for the new phase would have surfaced this before implementation |
| **Forbidden-scan false positives** (scanning inherited sample-layer fields, scanning rule-message strings) | 31CM scan, 31CN scan — both required scope narrowing | a `claim_boundary_scan` skill whose scope rules are versioned with the phase would prevent re-discovering this |
| **Next-phase prompt rephrasing** (each request needed a 200-line re-spec; the request can drift from the SOT's named next phase) | every request | a `next_phase_prompt_builder` skill that pulls the exact next-phase block from SOT line 11 and templates the standard request shape would prevent drift |

The **SDI Operator Skill Pack** turns this workflow into 8 small, composable Hermes skills. The pack is metadata-only (no scientific code change) and ships as planning doc + future skill files under `~/.hermes/skills/software-development/sdi-*/SKILL.md` (user-local, since these are operator workflow skills specific to the matthew-villnave operator's SDI project setup).

---

## 2. Composition (the 8 skills)

For each skill, the spec requires: **purpose, trigger condition, required commands/files, required output format, failure conditions, what it must never do, how it prevents previous mistakes**.

### 2.1 `sdi_preflight_check`

| field | value |
|---|---|
| **purpose** | Run the standard preflight before any SDI scientific phase work. Returns a green/red preflight report. |
| **trigger condition** | User starts a new SDI phase (any of: user explicitly requests a phase ID, or `sdi_phase_gate_check` is about to be called, or `sdi_precommit_report_builder` is about to be called for a phase work product). |
| **required commands/files** | `git branch --show-current`, `git rev-parse HEAD`, `git status --short`, `git log --oneline -5`, `git fetch origin`, `git rev-parse origin/master`, `python3 -m tests.run_source_of_truth_regression` |
| **required output format** | A preflight report with: `branch`, `HEAD`, `working_tree_clean` (bool), `local_eq_origin` (bool), `last_5_commits` (list), `regression_classification` (string), `regression_error_count` (int), `regression_fallback_count` (int), `readme_drift_guard_passed` (bool), `preflight_status` (PASS / FAIL). |
| **failure conditions** | working tree dirty; HEAD != origin/master; regression classification != PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN; error_count > 0; fallback_count > 0; readme_drift_guard.passed != true. |
| **must never do** | Modify any file. Auto-clean the working tree. Pull. Push. Reset. Cherry-pick. Rebase. Any of those without explicit operator approval. |
| **prevents previous mistakes** | 31CL hotfix scenario: the `sot_drift_scan` was skipped after 31CL commit because the preflight pattern wasn't standardized. By always running `sdi_preflight_check` first, drift detection becomes automatic. Also prevents commit-on-dirty-tree (one of the failure conditions). |

### 2.2 `sdi_phase_gate_check`

| field | value |
|---|---|
| **purpose** | Verify that the requested phase is the **current allowed scientific next phase** in `SOURCE_OF_TRUTH.md` Section 0 line 11, AND that the SOT has no wording drift on prior phases' state labels. Returns gate pass/fail. |
| **trigger condition** | User explicitly requests an SDI phase (e.g. "I am explicitly requesting Phase 31CO. Proceed to Phase 31CO — …"). Must run BEFORE any implementation. |
| **required commands/files** | `awk 'NR==11' SOURCE_OF_TRUTH.md` (parse current allowed next phase); `grep -E "no commit yet|planning artifacts staged" SOURCE_OF_TRUTH.md` (drift scan); read `docs/PHASE31xx_…md` (previous phase doc) for cross-reference. |
| **required output format** | A gate report with: `requested_phase` (string), `current_allowed_next_phase_per_sot` (string), `gate_status` (PASS / FAIL with reason), `sot_drift_findings` (list of strings), `conflicts_with_sot` (bool), `alternative_phases_listed` (list), `classification_phrase_to_speak_first` (the literal phrase the operator must say first: `"I have read SOURCE_OF_TRUTH.md. The current allowed next phase is X. I will not proceed if this request conflicts with the source of truth."`). |
| **failure conditions** | requested_phase != current_allowed_next_phase_per_sot (or not in alternative_phases_listed); sot_drift_findings non-empty (e.g. prior phase still says "no commit yet" after it was committed). |
| **must never do** | Modify SOURCE_OF_TRUTH.md. Skip the "I have read SOT" preamble. Allow the user to start the requested phase if the gate fails (instead, present the gate report and ask the user to either re-spec the request or fix the drift). |
| **prevents previous mistakes** | 31CM line 5 "no commit yet" wording drift: the gate would have flagged it and demanded the SOT fix land first. Also prevents the user from requesting a phase that's not yet allowed (e.g. requesting 31D before 31C is complete). The 3-strike patch review pattern is reinforced by the gate's mandatory "I have read SOT" preamble. |

### 2.3 `sdi_artifact_hygiene_scan`

| field | value |
|---|---|
| **purpose** | Audit a phase's new files (Python source, JSON results, Markdown doc, C++ capture) for: hardcoded operator paths, `__pycache__` / `.pyc` presence, model-file extensions in the diff, large-file warnings, lint failures, missing env-var redacted forms, and missing self-tests. |
| **trigger condition** | After any new file is created in `src/`, `src/results/`, or `docs/` for an SDI phase (i.e. after the "implement" step, before the "PRE-COMMIT REPORT" step). |
| **required commands/files** | `git status --short` (find new files); `python3 -m py_compile <new-py-files>`; `grep -E "/home/matthew-villnave\|/tmp/\|/Users/\|/root/" <new-files>`; `grep -E "\.pyc$\|__pycache__" <new-files>`; `wc -c <new-files>`; scan JSON files for `__pycache__` and `model` substrings; verify each Python file has at least one self-test. |
| **required output format** | An audit report with: `n_files_audited`, `n_hardcoded_paths_found` (with file:line:match), `n_pyc_or_pycache_in_diff` (with file paths), `n_model_extensions_in_diff` (with file paths), `n_lint_failures` (with file:line), `n_missing_self_tests` (with file names), `n_large_files` (>100 KB, with file:size), `audit_status` (PASS / FAIL). |
| **failure conditions** | Any hardcoded operator path; any `.pyc` in the diff; any model-file extension in the diff; any lint failure; any Python file without self-tests; any file >100 KB without explicit operator approval. |
| **must never do** | Auto-fix the failures (just report them). Modify the file in place. Stage anything. Run any pre-commit hooks. |
| **prevents previous mistakes** | 31CN self-test scope drift: a self-test that mutates `planned_path` but the scan reads `planned_filename` would have been caught here (the audit would ask "does each self-test mutate the field that the corresponding check actually reads?"). Also catches accidental hardcoded paths (none of the 31CN/31CM/31CL implementations had any, but the audit is defense-in-depth). |

### 2.4 `sdi_claim_boundary_scan`

| field | value |
|---|---|
| **purpose** | Verify a phase's output files (Python source, JSON results, Markdown doc) do not contain over-claims or forbidden claims. Specifically: scan for phrases that violate the metadata-only / no-runtime-loader / no-binary-blob boundary, and verify that the phase's classification is consistent with its actual behavior. |
| **trigger condition** | Before the PRE-COMMIT REPORT is generated for a phase, AND after any SOT or doc edit. |
| **required commands/files** | `grep -E "speedup\|production[- ]readiness\|runtime[- ]loadable\|live[- ]runtime memory savings" <new-files>`; `grep -E "PASS_.*" <new-files>`; compare classification string with the spec's allowed-classification list. |
| **required output format** | A claim-boundary report with: `n_over_claims_found` (with file:line:phrase), `n_classification_drift_found` (with file:line), `n_forbidden_claim_substrings_found` (with file:line), `claim_boundary_status` (PASS / FAIL). |
| **failure conditions** | Any over-claim phrase; any classification string not in the spec's allowed list for that phase; any forbidden-claim substring present in non-error-message contexts. |
| **must never do** | Auto-edit out the over-claims. Modify the file. The scan is informational; the operator decides what to do. |
| **prevents previous mistakes** | Defends against the 3-strike patch review pattern's most common cause: "over-claims, contradictions in technical phrasing". The scan is a pre-flight check that catches over-claims before the operator sees them. |

### 2.5 `sdi_precommit_report_builder`

| field | value |
|---|---|
| **purpose** | Generate the standard SDI PRE-COMMIT REPORT by running all 6 fence checks (A-F) and the post-edit SOT regression, in the exact format the operator's 3-strike review pattern expects. The report is built from **real command output**, not human memory. |
| **trigger condition** | After the "Update SOT" step, before the operator approval step. |
| **required commands/files** | `git status --short`, `git diff --stat`, `git log -1 --oneline`, `git show --name-status --stat HEAD~1`, `python3 -m tests.run_source_of_truth_regression`, `git rev-parse HEAD`, `git ls-files --others --exclude-standard`, the new file list (from `git status --short` lines starting with `??`). |
| **required output format** | A PRE-COMMIT REPORT in the exact section structure the operator has approved: current branch, current HEAD, git status --short, git diff --stat, files changed, regression result, classification, confirmation SOURCE_OF_TRUTH was read, confirmation <phase> allowed, implementation inputs reviewed, implementation deliverables created, <phase-specific-check-results>, scope assertions, whether prior accepted numeric results changed, SOURCE_OF_TRUTH sections changed, README update status, proposed commit message, whether any private/generated/large files would be committed, explicit question: "Approve commit/push?". |
| **failure conditions** | Working tree dirty before the report; HEAD != origin/master; regression classification != PASS; any fence check fails; any over-claim detected (delegated to `sdi_claim_boundary_scan`). |
| **must never do** | Fabricate any field. Use stale cache. Modify any file. Run any pre-commit hook. Stage anything. Push. |
| **prevents previous mistakes** | **This is the most important skill in the pack.** The 31CM pre-commit report and the 31CN pre-commit report were both generated by hand; both had the same risk: any field claimed clean could be wrong if the human mis-typed or skipped a step. By running the actual commands, the report is grounded in real output. The "explicit question: Approve commit/push?" footer is mandatory and prevents premature commit. |

### 2.6 `sdi_postpush_verifier`

| field | value |
|---|---|
| **purpose** | After the operator has approved commit and push, run the post-push verification suite: 7 required real commands + 5 fence checks (A-E), with no fabrication. The verifier double-checks that the pre-commit report's claims are actually true post-push. |
| **trigger condition** | After `git push origin master` returns success. |
| **required commands/files** | `git log -1 --oneline`, `git show --name-status --stat HEAD`, `git status --short`, `git rev-parse HEAD`, `git rev-parse origin/master`, `git ls-remote origin master`, `git tag --points-at HEAD`; the 5 fence scans (commit scope, diff stat, forbidden patterns, upstream byte-identity, no untracked); the post-push regression. |
| **required output format** | A POST-PUSH REPORT in the exact section structure the operator has approved: 7 real-command outputs verbatim, 5 fence results, 20-item post-push confirmation checklist, push evidence, summary. |
| **failure conditions** | Any of the 7 required commands returns an unexpected value; any of the 5 fences fails; any of the 20 confirmation checklist items is false. |
| **must never do** | Trust the pre-commit report's claims without re-running the commands. Fabricate. Modify. Revert. Force-push. |
| **prevents previous mistakes** | The 31CM and 31CN post-push verifications were generated by hand; both correctly re-ran the 7 commands and the 5 fences. The skill makes this re-verification automatic and consistent across all future phases. |

### 2.7 `sdi_sot_drift_scan`

| field | value |
|---|---|
| **purpose** | Scan SOURCE_OF_TRUTH.md for wording drift between the actual git state and the SOT's narrative. Specifically: detect "no commit yet" / "planning artifacts staged, not yet committed" for any phase that has been committed. Detect "no commit yet" anywhere. Detect "next-allowed-phase" name mismatches between SOT line 11 and the most recent committed phase. |
| **trigger condition** | Before every PRE-COMMIT REPORT, AND before every post-edit regression. Optionally as part of `sdi_preflight_check`. |
| **required commands/files** | `awk 'NR==5' SOURCE_OF_TRUTH.md` (parse state label tail); `grep -E "no commit yet\|planning artifacts staged" SOURCE_OF_TRUTH.md` (drift scan); `git log --oneline -10` (cross-reference committed phase IDs); `git rev-parse HEAD` (cross-reference current HEAD with the latest state label). |
| **required output format** | A drift report with: `n_drift_findings` (int), `drift_findings` (list of `phase_id`, `current_sot_wording`, `expected_wording`, `severity`), `drift_status` (PASS / FAIL). |
| **failure conditions** | Any "no commit yet" or "planning artifacts staged" wording for a phase that has a non-empty commit hash; any phase ID in SOT line 11 that doesn't match the operator's explicit request; any state label that says "PASSED" but the classification string is missing or wrong. |
| **must never do** | Auto-edit the SOT. Stage any changes. The skill only reports drift; the operator decides whether to patch. |
| **prevents previous mistakes** | The 31CL "no commit yet" drift was caught only because the operator (matthew-villnave) explicitly asked me to scan for it. The 31CM "no commit yet" drift was caught only because I ran the drift scan during 31CN's preflight. Without the skill, every future phase has the same risk of unnoticed drift. The skill makes drift detection automatic. |

### 2.8 `sdi_next_phase_prompt_builder`

| field | value |
|---|---|
| **purpose** | Generate the standard operator request template for the **current allowed next phase** in SOT line 11. The template includes: the pre-flight commands, the spec section structure, the "I have read SOT" preamble, the deliverable list, the allowed/forbidden claims, the recommended next-next-phase, and the explicit question at the end. The output is a single copy-pasteable markdown block the operator can edit and send. |
| **trigger condition** | When the operator says "I am explicitly requesting Phase X" or "what's the next phase?" or "build the request for the next phase." |
| **required commands/files** | `awk 'NR==11' SOURCE_OF_TRUTH.md` (extract current allowed next phase); `awk 'NR==5,NR==15' SOURCE_OF_TRUTH.md` (extract state label and journal entry for the most recent phase); `find docs/ -name "PHASE31xx_*.md" | sort -r | head -3` (read previous 3 phase docs for the spec template). |
| **required output format** | A copy-pasteable request template with: phase ID, phase full name, classification, allowed scope, preflight commands, implementation inputs list, implementation deliverables (with file paths), required checker/normalizer behavior, output JSONs required, self-test list, classification phrase, partial/blocked classifications, allowed/forbidden claims, SOT edit instructions, recommended next-next-phase. |
| **failure conditions** | SOT line 11 doesn't name a phase (e.g. all phases complete); requested phase ID doesn't match SOT line 11; operator has not approved the previous phase (no commit in the most recent state label). |
| **must never do** | Auto-send the request to the operator. Auto-start the next phase. Modify the SOT. The skill only generates the template; the operator edits and sends. |
| **prevents previous mistakes** | Each of the 6 phase requests so far was 200+ lines of inline re-specification; each one was hand-crafted and had small drift from the previous. By templating the request, drift between requests is minimized. The "I have read SOT" preamble is mandatory in the template, which reinforces the 3-strike patch review pattern. |

---

## 3. Workflow (how the 8 skills compose)

The 8 skills form a **linear pipeline** that the operator follows for every phase:

```
[operator sends phase request]
   |
   v
[sdi_next_phase_prompt_builder] (only if operator asked for the template)
   |
   v
[sdi_phase_gate_check]  --- FAIL ---> [operator fixes SOT drift or re-specs]
   | PASS
   v
[sdi_preflight_check]    --- FAIL ---> [operator cleans working tree / pulls / fixes regression]
   | PASS
   v
[operator implements]
   |
   v
[sdi_artifact_hygiene_scan] --- FAIL ---> [operator fixes hardcoded paths / missing self-tests / etc.]
   | PASS
   v
[sdi_claim_boundary_scan]  --- FAIL ---> [operator edits out over-claims]
   | PASS
   v
[operator updates SOT]
   |
   v
[sdi_sot_drift_scan]      --- FAIL ---> [operator patches SOT wording]
   | PASS
   v
[sdi_precommit_report_builder]
   |
   v
[operator approves]
   |
   v
[operator commits + pushes]
   |
   v
[sdi_postpush_verifier]  --- FAIL ---> [operator investigates the post-push inconsistency]
   | PASS
   v
[phase complete]
```

The skills are **independent** (each can be invoked standalone) but **compose linearly** as a pipeline. Each skill is a small, focused, single-purpose tool. This is the same shape as the `hermes-agent-skill-authoring` peer skills: 8-14k chars each, peer-matched frontmatter, single-purpose.

---

## 4. Skill implementation details (for the future, not now)

When the operator approves the plan, each skill will be implemented as a **user-local** skill at `~/.hermes/skills/software-development/sdi-<name>/SKILL.md` (per `hermes-agent-skill-authoring` guidance, user-local is correct for operator-specific workflow skills that don't ship with hermes-agent). The skills will follow the standard frontmatter:

```yaml
---
name: sdi-preflight-check
description: Use when starting a new SDI scientific phase. Runs git preflight + SOT regression and returns a PASS/FAIL preflight report.
version: 1.0.0
author: Hermes Agent (operator: matthew-villnave)
license: MIT
metadata:
  hermes:
    tags: [sdi, preflight, scientific-phase, gate]
    related_skills: [sdi-phase-gate-check, sdi-sot-drift-scan]
---
```

Each skill's body will follow the `## Overview` → `## When to Use` → body (with explicit commands) → `## Common Pitfalls` → `## Verification Checklist` structure, matching the peer-matched template.

The skills will be **Python stdlib-only** where they run shell commands (via `subprocess.run` or `terminal`), and **never import model-related libraries**. The skills will **never** modify scientific code, results, or SOT — they are read-only workflow tools.

The skills will be **composable**: each one returns a structured result that the next one consumes. The output format for each skill is the `required output format` defined above. If the operator later wants to chain these into a single `sdi_phase_pipeline` skill, the structured outputs make that possible.

---

## 5. Failure model (the "must never do" recap)

Every skill has a "must never do" section. The union of all 8 skills' "must never do" lists forms the **safety boundary** of the entire pack:

| forbidden action | which skills enforce |
|---|---|
| modify scientific code | `sdi_preflight_check`, `sdi_sot_drift_scan`, `sdi_claim_boundary_scan` |
| modify results | `sdi_preflight_check`, `sdi_sot_drift_scan`, `sdi_claim_boundary_scan` |
| modify SOT | `sdi_phase_gate_check`, `sdi_sot_drift_scan`, `sdi_precommit_report_builder` |
| auto-commit | `sdi_precommit_report_builder`, `sdi_postpush_verifier` |
| auto-push | `sdi_precommit_report_builder`, `sdi_postpush_verifier` |
| auto-tag | `sdi_precommit_report_builder`, `sdi_postpush_verifier` |
| fabricate report fields | `sdi_precommit_report_builder`, `sdi_postpush_verifier` |
| load model files | (N/A — these are workflow skills, no model code) |
| generate Q2_K/SDIR blobs | (N/A — these are workflow skills, no artifact code) |
| run inference | (N/A — these are workflow skills, no inference code) |

The pack is **read-only** with respect to scientific code / results / SOT. The only writes the pack ever does are:
- `sdi_precommit_report_builder` writes a report file (in the operator's chosen location, defaults to a temp file the operator can review)
- `sdi_postpush_verifier` writes a report file (same)
- `sdi_next_phase_prompt_builder` writes a template file (in `~/.hermes/scratch/` or similar)

All writes are non-destructive and operator-reviewed.

---

## 6. Why these 8 skills, not more / fewer

**Why not just one `sdi_phase_pipeline` skill?** A single mega-skill would be 30k+ chars, hard to maintain, and would conflate 8 distinct concerns. The 3-strike patch review pattern benefits from **explicit, granular** checks; one skill that does all 8 checks is harder to debug when one of them fails.

**Why not 20+ skills?** The pack is intentionally minimal. Each skill must be reused at least once per phase. Skills that fire only occasionally (e.g. "handle a corrupted git history") are out of scope.

**Why these 8 specifically?** They map 1:1 to the 8 stages of the current workflow:
1. **preflight** → 1 skill (`sdi_preflight_check`)
2. **read SOT + confirm allowed** → 1 skill (`sdi_phase_gate_check`)
3. **read inputs** → covered by `sdi_phase_gate_check` (it reads the SOT) and `sdi_next_phase_prompt_builder` (it reads previous docs)
4. **implement** → no skill (operator-driven)
5. **self-tests** → covered by `sdi_artifact_hygiene_scan` (verifies self-tests exist) and `sdi_claim_boundary_scan` (verifies over-claims)
6. **write doc** → covered by `sdi_artifact_hygiene_scan` (verifies doc file size, paths) and `sdi_claim_boundary_scan` (verifies doc language)
7. **update SOT** → 1 skill (`sdi_sot_drift_scan`)
8. **PRE-COMMIT REPORT** → 1 skill (`sdi_precommit_report_builder`)
9. **commit + push** → 1 skill (`sdi_postpush_verifier`)
10. **next-phase prompt** → 1 skill (`sdi_next_phase_prompt_builder`)

8 skills, 8 stages, 1:1 mapping. Clean.

---

## 7. Estimated effort to implement (when approved)

Each skill is 8-14k chars (per the peer-matched template). At ~10k chars per skill × 8 skills = ~80k chars total. Each skill takes ~1-2 hours to draft + 1-2 review cycles with the operator = ~16-24 hours total. This is a multi-day project.

**Suggested implementation order** (so the highest-value skills are usable first):
1. `sdi_phase_gate_check` (catches the worst mistake: SOT drift + wrong phase)
2. `sdi_precommit_report_builder` (catches fabricated reports — the most operator-visible mistake)
3. `sdi_postpush_verifier` (catches post-push drift)
4. `sdi_sot_drift_scan` (catches the SOT drift class of mistake)
5. `sdi_preflight_check` (catches dirty-tree / regression-failed mistakes)
6. `sdi_artifact_hygiene_scan` (catches hardcoded-path / missing-self-test mistakes)
7. `sdi_claim_boundary_scan` (catches over-claim mistakes)
8. `sdi_next_phase_prompt_builder` (templating convenience; not safety-critical)

---

## 8. Limitations of the pack

- The pack is **operator-specific** (matthew-villnave on `optiplex TheForgeHQ` with `sdi-substitutive`). Another operator on another host would need to adapt the paths, the model-check command (currently `git fetch` + `git rev-parse`), and the SOT-line-11 parser. The skills are not generalizable across operators without a `sdi_operator_config` skill (out of scope for this planning doc).
- The pack does **not** automate the actual scientific implementation (e.g. writing the validator code, running the regression). The operator still drives the implementation. The pack wraps the workflow, not the science.
- The pack does **not** enforce that the operator's request matches the SOT. The `sdi_phase_gate_check` reports a mismatch; it does not auto-reject the request. The operator must still choose to honor the gate.
- The pack is **read-only** with respect to the operator's data. It does not archive phase results, do backups, or maintain a state machine. (A future `sdi_phase_archive` skill could do that; out of scope here.)

---

## 9. End of planning doc

This document is the **planning doc only**. No skills are implemented. No files in `~/.hermes/skills/` have been created. No SOT has been modified. No scientific code has been modified. No results have been modified. No commit. No push. No tag.

The next step (when the operator approves) is to implement the 8 skills in the order suggested in §7.

---

## Appendix A — relationship to the 3-strike patch review pattern

The operator's 3-strike patch review pattern (per memory) is:
- Operator reviews the pre-commit report
- Operator either approves, OR "do not commit yet" with specific wording/claim contradictions to fix
- The agent patches and re-presents the report
- Revised reports must include grep/readback evidence that patches landed
- Process repeats until approved (up to 3 strikes, but in practice the operator has approved on strike 1 or 2)

The pack directly supports this pattern:
- `sdi_precommit_report_builder` produces the report the operator reviews
- `sdi_claim_boundary_scan` catches the over-claims and contradictions before the operator sees them (reducing strikes)
- `sdi_sot_drift_scan` catches the wording-drift class of contradiction before the operator sees them
- `sdi_artifact_hygiene_scan` catches the hardcoded-path / missing-self-test contradictions before the operator sees them
- The "explicit question: Approve commit/push?" footer in `sdi_precommit_report_builder` enforces the operator's approval gate

In short: the pack makes the 3-strike pattern **faster** (fewer strikes per phase) and **more reliable** (fewer missed mistakes).

---

## Appendix B — known preflight state at the time of this planning doc

Per the preflight run at the top of this report:

- **branch:** `master`
- **HEAD:** `27be90df1988035c2fa294f753e3990aaf4cf369` (Phase 31CN post-push)
- **working tree:** DIRTY (`M src/results/PHASE31CN_NORMALIZED_PLANNED_MANIFEST.json`, `M src/results/PHASE31CN_PLANNED_MANIFEST_NORMALIZER_RESULT.json`) — these were modified by the post-push re-run of the normalizer (a verification step, not a substantive change). The operator can `git checkout` them if desired; this planning doc does not modify them.
- **SOT:** 2 mentions of "31CN" (line 5 + line 15, as expected post-31CN)
- **31CN result files:** present, classification `PASS_31CN_PLANNED_MANIFEST_NORMALIZER_PROVENANCE_ADAPTER_CLEAN`
- **planning doc target location:** `docs/SDI_OPERATOR_SKILL_PACK_PLAN.md` (this file)

---

## Appendix C — classification

**`PASS_SDI_OPERATOR_SKILL_PACK_PLAN_SELECTED`**

The plan is selected (not yet implemented). The skills are well-defined, the failure modes are bounded, the safety boundary is documented, the implementation order is suggested, and the relationship to the operator's 3-strike patch review pattern is explicit. No skills have been implemented yet, per the operator's request.
