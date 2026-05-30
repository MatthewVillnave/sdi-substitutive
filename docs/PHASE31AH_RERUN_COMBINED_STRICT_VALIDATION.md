# Phase 31AH-RERUN: Combined Strict Validation Against 31AJ-Clean Runtime

**Classification:** `PASS_31AH_RERUN_COMBINED_STRICT`
**Old HEAD:** `2766d78c3d80e3a28cda275f27695b1bdb964d4e`
**Script HEAD:** `2766d78c3d80e3a28cda275f27695b1bdb964d4e`
**31AI unlocked:** `True`

## Source Of Truth Read

- read: yes
- current allowed next phase before run: Phase 31AH-RERUN — Combined Strict Validation Against 31AJ-Clean Runtime
- required regression command: `python3 -m tests.run_source_of_truth_regression`
- canonical orientation: artifact tensor shape `(d_out, d_in)`, `Y = X @ W.T`, residual bitmap index `row * d_in + col`

## Preflight Regression

- command: `python3 -m tests.run_source_of_truth_regression`
- classification: `PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN`

## Dense-vs-Stream Source-of-Truth

| Layer | Family | .sdiw max diff | .sdir max diff | combined max diff | nnz |
|---:|---|---:|---:|---:|---:|
| 0 | ffn_up | 2.98023e-07 | 1.49012e-08 | 2.98023e-07 | 392402 |
| 0 | ffn_down | 4.76837e-07 | 1.11759e-08 | 4.76837e-07 | 392277 |
| 1 | ffn_up | 7.15256e-07 | 2.98023e-08 | 7.15256e-07 | 392386 |
| 1 | ffn_down | 9.53674e-07 | 4.84288e-08 | 9.53674e-07 | 392234 |
| 2 | ffn_up | 1.52588e-05 | 4.76837e-07 | 1.90735e-05 | 392889 |
| 2 | ffn_down | 0.00146484 | 6.10352e-05 | 0.0012207 | 392323 |
| 3 | ffn_up | 1.52588e-05 | 9.53674e-07 | 1.52588e-05 | 392547 |
| 3 | ffn_down | 0.000488281 | 0.00012207 | 0.000488281 | 392426 |
| 4 | ffn_up | 2.38419e-06 | 1.19209e-07 | 2.38419e-06 | 392388 |
| 4 | ffn_down | 2.38419e-07 | 5.96046e-08 | 1.49012e-07 | 392243 |
| 5 | ffn_up | 6.67572e-06 | 3.57628e-07 | 6.67572e-06 | 392500 |
| 5 | ffn_down | 4.57764e-05 | 2.86102e-06 | 4.19617e-05 | 392251 |

## Approximation Table

| Layer | Family | cos_low | cos_sub | delta_cos | MAE_low | MAE_sub | MAE_delta | max_low | max_sub |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | ffn_up | 0.991052 | 0.995758 | +0.004707 | 0.028738 | 0.020668 | +0.008070 | 0.242289 | 0.100915 |
| 0 | ffn_down | 0.996764 | 0.998753 | +0.001989 | 0.007969 | 0.005392 | +0.002576 | 0.063061 | 0.022813 |
| 1 | ffn_up | 0.991999 | 0.995221 | +0.003223 | 0.040226 | 0.031351 | +0.008875 | 0.285843 | 0.150452 |
| 1 | ffn_down | 0.995337 | 0.997858 | +0.002522 | 0.008372 | 0.005635 | +0.002737 | 0.052439 | 0.023444 |
| 2 | ffn_up | 0.992283 | 0.995830 | +0.003548 | 0.338181 | 0.254918 | +0.083264 | 2.520261 | 1.231233 |
| 2 | ffn_down | 0.993647 | 0.997504 | +0.003857 | 0.280293 | 0.034851 | +0.245442 | 68.415329 | 0.162794 |
| 3 | ffn_up | 0.989687 | 0.992858 | +0.003171 | 0.187107 | 0.161956 | +0.025151 | 1.402896 | 0.614660 |
| 3 | ffn_down | 0.996449 | 0.998840 | +0.002391 | 0.217680 | 0.046037 | +0.171644 | 76.440948 | 0.171245 |
| 4 | ffn_up | 0.991328 | 0.994794 | +0.003466 | 0.093106 | 0.072041 | +0.021066 | 0.639740 | 0.337654 |
| 4 | ffn_down | 0.994345 | 0.997212 | +0.002866 | 0.010718 | 0.007640 | +0.003078 | 0.120063 | 0.032132 |
| 5 | ffn_up | 0.991596 | 0.994807 | +0.003211 | 0.147127 | 0.116157 | +0.030970 | 1.164451 | 0.504188 |
| 5 | ffn_down | 0.994066 | 0.998012 | +0.003946 | 0.018150 | 0.011650 | +0.006500 | 1.400914 | 0.042499 |

## Memory Table

| Layer | Family | W_low packed | W_low scales | residual | total | Q4 budget | margin |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | ffn_up | 2,179,072 | 272,384 | 1,329,600 | 3,781,056 | 4,358,144 | 577,088 |
| 0 | ffn_down | 2,179,072 | 272,384 | 1,329,350 | 3,780,806 | 4,358,144 | 577,338 |
| 1 | ffn_up | 2,179,072 | 272,384 | 1,329,568 | 3,781,024 | 4,358,144 | 577,120 |
| 1 | ffn_down | 2,179,072 | 272,384 | 1,329,264 | 3,780,720 | 4,358,144 | 577,424 |
| 2 | ffn_up | 2,179,072 | 272,384 | 1,330,574 | 3,782,030 | 4,358,144 | 576,114 |
| 2 | ffn_down | 2,179,072 | 272,384 | 1,329,442 | 3,780,898 | 4,358,144 | 577,246 |
| 3 | ffn_up | 2,179,072 | 272,384 | 1,329,890 | 3,781,346 | 4,358,144 | 576,798 |
| 3 | ffn_down | 2,179,072 | 272,384 | 1,329,648 | 3,781,104 | 4,358,144 | 577,040 |
| 4 | ffn_up | 2,179,072 | 272,384 | 1,329,572 | 3,781,028 | 4,358,144 | 577,116 |
| 4 | ffn_down | 2,179,072 | 272,384 | 1,329,282 | 3,780,738 | 4,358,144 | 577,406 |
| 5 | ffn_up | 2,179,072 | 272,384 | 1,329,796 | 3,781,252 | 4,358,144 | 576,892 |
| 5 | ffn_down | 2,179,072 | 272,384 | 1,329,298 | 3,780,754 | 4,358,144 | 577,390 |

## Strict Counters

- W_ref_loaded: 0
- W_ref_generated: 0
- dense_W_low_materialized: 0
- dense_R_materialized: 0
- sdiw_loaded: 12
- sdir_loaded: 12
- manifest_loaded: 1
- checksum_validated: 12
- memory_budget_validated: 12
- fallback_count: 0
- error_count: 0
- path_label: [SDI-SUB-RUNTIME]

## SOURCE_OF_TRUTH.md

- changed: yes
- sections updated: Accepted Known-Good Facts, Invalidated / Superseded Claims, Suspected / Unproven, Current Open Blockers, Current Allowed Next Phase
- new accepted facts: 31AH-RERUN ran against the 31AJ-clean manifest loader/runtime; source equivalence and strict counters are recorded in this result.
- new invalidated/superseded claims: Pre-31AJ 31AH combined strict validation is superseded.
- new suspected/unproven claims: None.
- current blockers: No 31AI tensor/runtime gate blocker remains; checkpoint/tag still blocked unless explicitly authorized; historical scripts require the source-of-truth regression contract.
- current allowed next phase: Phase 31AI — only if requested explicitly
