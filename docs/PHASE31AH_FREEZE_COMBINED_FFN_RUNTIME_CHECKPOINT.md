# Phase 31AH-FREEZE: Combined FFN Substitutive Runtime Prototype Checkpoint

## Metadata

| Field | Value |
|-------|-------|
| old HEAD | `5b2c1e32ace76cfd3a1b7883578207f384fb9231` |
| new HEAD | _(after commit)_ |
| tag | `phase31ah-combined-ffn-runtime-checkpoint` |
| tag target | same as new HEAD |
| classification | `PASS_31AH_FREEZE_COMBINED_FFN_RUNTIME_CHECKPOINT` |
| k_pct | 9.0% |
| alpha | 1.0 |
| scope | ffn_up layers 0–5 + ffn_down layers 0–5 |
| date | 2026-05-30 |

---

## Accepted Checkpoint Claim

> "Standalone manifest-driven combined FFN substitutive runtime prototype validated for ffn_up and ffn_down layers 0–5 under corrected source-of-truth harness. Runtime-consistent residuals improve cosine and MAE over packed W_low while keeping W_ref absent, W_ref not generated, dense W_low absent, dense R absent, and memory margins positive."

---

## Forbidden Claims (Unchanged)

- no model quality recovery
- no behavior recovery
- no speedup
- no full-model memory savings
- no llama.cpp integration
- no production readiness
- no full MLP replacement claim
- no ffn_gate claim

---

## Phase Proven Chain

### Phase 31AJ-STABLE (HEAD `2766d78`)
- manifest/runtime source-of-truth cleanup
- true runtime separated from fixture/reference generation
- family-aware manifest path resolution
- real `.sdiw` parser (header, rows, cols, scale bytes, packed nibble bytes)
- one-command source-of-truth regression harness

### Phase 31AH-RERUN (HEAD `5b2c1e3`)
- combined ffn_up + ffn_down strict validation
- runtime-consistent residuals (`R = W_ref - decode(packed W_low)`)
- canonical orientation enforced
- all strict counters clean
- memory margins positive
- ffn_up and ffn_down both improve cosine and MAE vs W_low

---

## Metrics Table

### Approximation (averages, layers 0–5)

| family | avg cos_low | avg cos_sub | avg delta | avg MAE_low | avg MAE_sub | avg MAE_delta |
|--------|------------|-------------|-----------|-------------|-------------|---------------|
| ffn_up | 0.991324 | 0.994878 | +0.003554 | 0.139081 | 0.109515 | +0.029566 |
| ffn_down | 0.995101 | 0.998030 | +0.002929 | 0.090530 | 0.018534 | +0.071996 |

### Strict Counters

| counter | value |
|---------|-------|
| W_ref_loaded | 0 |
| W_ref_generated | 0 |
| dense_W_low_materialized | 0 |
| dense_R_materialized | 0 |
| sdiw_loaded | 12 |
| sdir_loaded | 12 |
| fallback_count | 0 |
| error_count | 0 |

### Memory Margins

| metric | value |
|--------|-------|
| min margin | 576,114 bytes |
| total margin | 6,924,972 bytes |
| Q4 budget per row | 4,358,144 bytes |
| all rows positive | yes |

### Source Equivalence Gates

| gate | status |
|------|--------|
| .sdiw dense-vs-stream | PASS |
| .sdir dense-vs-stream | PASS |
| combined stream-vs-dense | PASS |

---

## Known Limitations

- qwen2.5-0.5b-instruct only (0.5B parameter model)
- layers 0–5 only (ffn_up + ffn_down)
- k=9% only
- No llama.cpp integration
- No production readiness claim
- No behavior/model quality recovery claim

---

## Next Allowed Phase After Freeze

**Phase 31AI — only if explicitly requested**

---

## SOURCE_OF_TRUTH.md Status

- changed: yes
- sections updated: accepted facts, suspected/unproven, blockers, allowed next phase
- new accepted facts:
  - 31AH-RERUN passed against 31AJ-clean runtime
  - Combined ffn_up + ffn_down standalone strict runtime is checkpointed
  - Current checkpoint tag points to 31AH-FREEZE commit
- suspected/unproven: cleared (31AI moved to accepted after31AH-RERUN)
- blockers: updated (checkpoint/tag restriction lifted by explicit authorization)
- current allowed next phase: Phase 31AI — only if explicitly requested
