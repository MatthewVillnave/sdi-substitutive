# Phase 31BQ — Larger-Model Local Availability / Metadata Probe

**Classification:** `PARTIAL_31BQ_NO_LARGER_QWEN_LOCAL_USABLE`
**Date:** Phase 31BQ
**Repo:** sdi-substitutive
**Current HEAD:** `84328d92` (Phase 31BP)
**Policy package:** `corrected_q2k_policy_v1`
**Frozen checkpoint:** `phase31bn-corrected-q2k-full-aggregate-checkpoint` → `0304590c`

> **Important:** This phase performed metadata-only probing. No larger-model validation was executed. No models were downloaded. No 1.5B/3B/7B GGUF is locally available for validation.

---

## Policy Package Verified

The `corrected_q2k_policy_v1` package was verified present:
- `docs/CORRECTED_Q2K_POLICY_PACKAGE.md` ✓
- `src/results/CORRECTED_Q2K_POLICY_PACKAGE.json` ✓
- `src/corrected_q2k_policy.py` ✓
- `tests/test_corrected_q2k_policy.py` ✓
- Regression suite includes policy constants smoke test ✓

---

## Local Availability Scan (Bounded)

**Sources scanned (no downloads):**
- `/media/matthew-villnave/VL_usb/models/`
- `~/.cache/huggingface/hub/`
- Env vars: `SDI_GGUF_MODEL_PATH`, `SDI_MODEL_DIR`, `SDI_LLAMA_CPP_ROOT` (all unset)

**Qwen2.5 family — local presence:**

| Model | Local? | Path | Status |
|-------|--------|------|--------|
| Qwen2.5-0.5B (Q4_K_M) | ✓ Yes | `/media/matthew-villnave/VL_usb/models/qwen2.5-0.5b-instruct-q4_k_m.gguf` | Already validated (31BN) |
| Qwen2.5-0.5B (official set) | ✓ Yes | `/media/matthew-villnave/VL_usb/models/qwen2.5-0.5b-official/` | Source of 31BN validation |
| Qwen2.5-0.5B (HF safetensors) | ✓ Yes | `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/` | Not GGUF |
| Qwen2.5-0.5B (HF GGUF) | ✓ Yes | `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct-GGUF/` | Already validated |
| Qwen2.5-1.5B | ✗ No | — | Not present anywhere |
| Qwen2.5-3B | ✗ No | — | Not present anywhere |
| Qwen2.5-7B | ✗ No | — | Not present anywhere |
| Qwen2.5-14B | ✗ No | — | Not present anywhere |
| Qwen2.5-32B (Q4_K_M, sharded) | ⚠ Partial | `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct-GGUF/` | **Shard 1 of 5 only** — 4/5 missing |

**Conclusion:** No larger Qwen GGUF is fully local and usable for validation. The 32B candidate is incomplete (1 of 5 shards present, ~3.69 GB of ~20 GB total). The 0.5B is the only fully validated model.

---

## Qwen2.5-32B Partial Metadata (Verified from Shard 1)

GGUFReader opened shard 1 successfully and exposed the following fields and tensors:

### Architecture fields
| Key | Value |
|-----|-------|
| `general.architecture` | `qwen2` |
| `general.name` | `qwen2.5-32b-instruct` |
| `qwen2.block_count` | **64** |
| `qwen2.embedding_length` | **5120** |
| `qwen2.attention.head_count` | 40 |
| `qwen2.attention.head_count_kv` | 8 |
| `qwen2.feed_forward_length` | **27648** |
| `qwen2.context_length` | 131072 |
| `qwen2.rope.freq_base` | 1000000.0 |
| `qwen2.attention.layer_norm_rms_epsilon` | 1e-06 |
| `general.file_type` | Q4_K_M (from filename) |
| `split.count` | 5 |
| `split.tensors.count` | 771 (across all 5 shards) |
| `split.no` | 0 (this is shard 1) |

### Tensor shape orientation (from shard 1, layer 0)

| Tensor | Raw GGUFReader shape | Canonical interpretation |
|--------|---------------------|--------------------------|
| `blk.0.ffn_up.weight` | `[5120, 27648]` | `(d_out=27648, d_in=5120)` — but see note below |
| `blk.0.ffn_gate.weight` | `[5120, 27648]` | `(d_out=27648, d_in=5120)` — but see note below |
| `blk.0.ffn_down.weight` | `[27648, 5120]` | `(d_out=5120, d_in=27648)` — but see note below |

**Important — orientation caveat:**

For Qwen2.5-0.5B (verified previously), GGUFReader reported ffn_up as `[896, 4864]`. The canonical MLP computation `Y = X @ W.T` with `X ∈ R^{*, 896}` requires `W_up ∈ R^{4864, 896}` for `Y ∈ R^{*, 4864}`. So `[896, 4864]` in GGUFReader was treated as a **storage/representation format** (transposed view), and canonical orientation is `(d_out=4864, d_in=896)`.

For Qwen2.5-32B, GGUFReader reports:
- ffn_up: `[5120, 27648]`
- ffn_down: `[27648, 5120]`

The raw shape pattern in 32B is **different from 0.5B** — ffn_up displays as `[hidden, intermediate]` and ffn_down as `[intermediate, hidden]`.

**FFN raw tensor shape display from GGUFReader is model/file/reader-specific and must not be used directly as canonical orientation.** Canonical MLP orientation must be derived from tensor role and dimensions:
- ffn_up / ffn_gate map hidden → intermediate → canonical artifact orientation is `(d_out=intermediate, d_in=hidden)`
- ffn_down maps intermediate → hidden → canonical artifact orientation is `(d_out=hidden, d_in=intermediate)`

For 32B metadata (hidden=5120, intermediate=27648):
- Canonical ffn_up/ffn_gate would be `(27648, 5120)`
- Canonical ffn_down would be `(5120, 27648)`

The GGUFReader raw display `[5120, 27648]` for ffn_up/gate should be treated as **storage/reader representation** until validated by a model-specific orientation parity test. Whether the raw display matches canonical orientation directly, requires a transpose, or follows some other convention cannot be determined from the raw shape alone — it must be empirically verified per model.

> **This is a critical finding for 31BR+ planning.** Cannot reuse the 0.5B harness code without verifying orientation per model.

---

## 3B Intermediate Conflict Status

| Source | Value | Status |
|--------|-------|--------|
| Public Qwen2.5 spec | 8192 | Documented |
| Prior project evidence (14D) | 11008 | Documented |
| Local GGUF verification | — | **No Qwen2.5-3B GGUF locally available** |

**Conclusion:** 3B intermediate_size conflict **remains UNRESOLVED**. No local 3B GGUF exists to read metadata from. Cannot confirm 8192 vs 11008 from local data.

---

## Memory Estimates (32B, Selected Policy)

Based on verified 32B metadata, with the **caveat that orientation interpretation must be re-derived for 32B**:

### Verified shape elements
| Component | Shape | Elements |
|-----------|-------|----------|
| ffn_up | `[5120, 27648]` | 141,557,760 |
| ffn_gate | `[5120, 27648]` | 141,557,760 |
| ffn_down | `[27648, 5120]` | 141,557,760 |

### Estimated memory per layer (corrected Q2_K row-ceil)

| Component | Bytes/family/layer | Formula |
|-----------|-------------------|---------|
| Q2_K ffn_up (row-ceil) | ~57.1 MB | estimated, linear scaling from 0.5B |
| Q2_K ffn_gate (row-ceil) | ~57.1 MB | estimated, linear scaling from 0.5B |
| Q2_K ffn_down (row-ceil) | ~57.1 MB | estimated, linear scaling from 0.5B |
| SDIR ffn_up @ k=0.5% | ~0.71 MB | bitmap = 141M / 200 |
| SDIR ffn_gate @ k=0.5% | ~0.71 MB | bitmap = 141M / 200 |
| SDIR ffn_down | none | disabled per policy |
| **Total selected policy per layer** | **~172.7 MB** | |
| Q4_K baseline per layer | ~195 MB | estimated |
| **Margin per layer** | **~22 MB** | — |

> **Caveat:** These are linear-scaling estimates. Row-ceil overhead at larger `d_out` (5120 vs 896) may shift the actual margin. Cannot be verified without real measurement.

### Estimated aggregate for 64 layers
- **Aggregate margin (estimated):** ~1.4 GB across 64 layers
- **Total selected policy:** ~11.05 GB
- **Q4 baseline:** ~12.48 GB

> **Reality check:** 32B at Q2_K is roughly 11 GB of selected policy. This is the full model size at 4-bit. Tensor harness memory would be 2-3× this during execution. Likely needs ~30-40 GB RAM. Do not attempt without explicit approval.

---

## Risk Notes

1. **No larger-model validation executed.** This is a metadata probe only.
2. **No model downloaded.** 32B is only 1/5 shards locally.
3. **3B conflict unresolved.** No local 3B GGUF to verify against.
4. **Availability does not imply feasibility.** 32B exists (partially) but resource requirements are huge.
5. **Metadata does not prove tensor-harness success.** Just because shapes can be read doesn't mean the corrected Q2_K policy scales.
6. **Selected policy may not scale.** Row-ceil overhead at `d_out=5120` is unverified.
7. **FFN orientation must be re-derived and parity-tested per model.** Raw GGUFReader display alone is not canonical; storage/reader representation may or may not match canonical artifact orientation. Future phases must verify per model.
8. **Temp artifacts must not be committed.** 32B validation would generate ~11 GB of temp data.
9. **Private path debt remains** in older scripts (31AG–31AW) but package path is clean.

---

## Recommended Next Phase

**Phase 31BR — Larger-Model Acquisition Plan / Download Approval, only if explicitly requested**

**Rationale:**
- No larger Qwen GGUF is fully local
- 32B is incomplete (1/5 shards)
- Cannot run validation without a complete local model
- Cannot make a smaller-than-32B candidate available without Matt's explicit download approval
- 32B at Q2_K is ~20 GB total, ~11 GB working set, ~30-40 GB RAM during tensor harness

**31BR must include:**
1. Explicit download approval list (which model, which quant, from where)
2. Storage and RAM budget estimate
3. Per-model orientation re-derivation plan (do not reuse 0.5B harness)
4. Staged download + metadata verification + small-anchor validation approach
5. Do NOT proceed to actual validation without Matt's explicit "go" at each step

**31BR forbidden claims:** Same as all phases — no larger-model result, no production readiness, no runtime integration.

---

## Classification

**`PARTIAL_31BQ_NO_LARGER_QWEN_LOCAL_USABLE`**

- 32B metadata partially verified (shard 1 only) — usable for reference but not for full validation
- 0.5B is the only fully usable larger-model benchmark
- 3B conflict remains unresolved (no local 3B GGUF)
- No validation executed
- No download performed
- Recommended next: 31BR — Acquisition Plan / Download Approval