# Phase 31BR — Larger-Model Acquisition Plan / Download Approval

**Classification:** `PASS_31BR_ACQUISITION_PLAN_READY`
**Date:** Phase 31BR
**Repo:** sdi-substitutive
**Current HEAD:** `097b1143` (Phase 31BQ)
**Policy package:** `corrected_q2k_policy_v1`
**Frozen checkpoint:** `phase31bn-corrected-q2k-full-aggregate-checkpoint` → `0304590c`

> **Important:** This is a planning/approval phase only. No model was downloaded. No validation was executed. No Q2_K or SDIR artifacts were generated.

---

## Policy Package Verified

`corrected_q2k_policy_v1` package verified present:
- `docs/CORRECTED_Q2K_POLICY_PACKAGE.md` ✓
- `src/results/CORRECTED_Q2K_POLICY_PACKAGE.json` ✓
- `src/corrected_q2k_policy.py` ✓
- `tests/test_corrected_q2k_policy.py` ✓
- Regression smoke test included ✓

---

## Disk and Workspace Availability

| Filesystem | Size | Used | Free | Use% |
|------------|------|------|------|------|
| `/` (NVMe) | 233 GB | 126 GB | **96 GB** | 57% |
| `/media/matthew-villnave/VL_usb` | 115 GB | 74 GB | **42 GB** | 64% |
| `~/.cache/huggingface/hub` | — | 5.1 GB | — | — |
| `/tmp` | — | 5.9 GB | — | — |
| `/home/matthew-villnave/sdi-substitutive` | — | 1.3 GB | — | — |

**Current local model footprint:**
- VL_usb models directory: 6.6 GB (Qwen2.5-0.5B, LFM2.5-8B, etc.)
- HF cache: 5.1 GB (0.5B variants + partial 32B shard 1 at 3.69 GB)
- Repo: 1.3 GB

**Cleanup risks:** None identified. No large stale files in repo (`PHASE31BH_*` JSONs are KB-scale). 32B partial shard in HF cache is 3.69 GB of dead weight (incomplete, unusable) — could be freed if Matt approves.

---

## Candidate Model Files

The exact remote filenames for Qwen2.5-1.5B, 3B, 7B GGUF files require either Hugging Face lookup or user confirmation. Conservative size estimates based on Qwen2.5 family scaling from known 0.5B (491 MB Q4_K_M, 396 MB Q2_K) and 32B (Q4_K_M = ~20 GB sharded).

| Model | Quant | Estimated Size | Sharded? | Local? |
|-------|-------|----------------|----------|--------|
| Qwen2.5-1.5B-Instruct | Q4_K_M | ~1.0-1.4 GB | No (single file) | ✗ No |
| Qwen2.5-1.5B-Instruct | Q2_K | ~0.7-1.0 GB | No (single file) | ✗ No |
| Qwen2.5-1.5B-Instruct | Q3_K_M | ~0.9-1.2 GB | No (single file) | ✗ No |
| Qwen2.5-3B-Instruct | Q4_K_M | ~1.8-2.2 GB | No (single file) | ✗ No |
| Qwen2.5-3B-Instruct | Q2_K | ~1.3-1.7 GB | No (single file) | ✗ No |
| Qwen2.5-7B-Instruct | Q4_K_M | ~4.0-4.8 GB | No (single file) | ✗ No |
| Qwen2.5-7B-Instruct | Q3_K_M | ~3.3-4.0 GB | No (single file) | ✗ No |
| Qwen2.5-32B-Instruct | Q4_K_M | ~20 GB | **Yes (5 shards)** | ⚠ Partial (1/5) |

> **Filename caveat:** The exact Hugging Face repo filename pattern is typically `qwen2.5-{size}b-instruct-{quant}.gguf` (or similar). Final filename must be confirmed at download time.

---

## Selected Acquisition Target

**Qwen2.5-1.5B-Instruct, Q4_K_M quantization** (primary recommendation)

### Why 1.5B
- Cleanest step up from 0.5B (already validated in 31BN)
- Lowest resource risk in 1.5B-32B range
- Best chance to debug orientation / metadata / scaling before larger models
- 3B has unresolved intermediate_size conflict (8192 vs 11008)
- 7B/32B too large for current validation resource budget

### Why Q4_K_M
- 0.5B baseline used Q2_K + Q4_K_M comparison — Q4_K_M gives the comparator reference
- Q2_K could be downloaded as secondary target if Matt wants Q2_K validation specifically
- Q3_K_M and Q5_K_M not needed at this step
- Single file, no sharding complexity

### Why alternatives deferred
- **3B:** intermediate_size conflict unresolved; 1.5B must succeed first before 3B makes sense
- **7B:** 2-3× the memory budget; 1.5B is sufficient probe
- **32B:** 5-shard incomplete in local cache; 20 GB total; not appropriate for next validation

---

## Recommended Download Destination

**Primary destination (placeholder):** `$SDI_MODEL_DIR/qwen2.5-1.5b-official/`

**Rationale:**
- Same `qwen2.5-...-official/` naming pattern as existing 0.5B model directory
- Local operator sets `SDI_MODEL_DIR` env var to their real model-storage root (e.g. `/media/matthew-villnave/VL_usb/models` on this machine)
- Avoids hardcoding any operator-specific absolute path in committed artifacts
- Symlink-friendly if additional download mirrors needed

**Alternative destination:** `~/.cache/huggingface/hub/` if using `huggingface-cli download` (auto-managed by HF cache)

**Estimated disk budget for 1.5B Q4_K_M:**
- Model file: ~1.0-1.4 GB
- HF cache metadata: ~50-100 MB
- Temp validation artifacts (if 31BS+ runs): ~3-5 GB
- **Total: ~5-7 GB** — fits in VL_usb with 35+ GB remaining

---

## Staged Acquisition Plan

### Stage 0 — Approval (NOT YET — requires Matt's explicit "go")
Matt must approve:
- Exact model: `Qwen2.5-1.5B-Instruct`
- Exact quantization: `Q4_K_M` (or alternate if Matt prefers)
- Destination: `$SDI_MODEL_DIR/qwen2.5-1.5b-official/` (local operator sets `SDI_MODEL_DIR` to a real path)
- Max disk budget: 7 GB
- Method: `huggingface-cli download` OR manual `wget`/browser download (Matt's choice)

### Stage 1 — Download (31BS)
- Download selected file(s) to approved destination
- Verify file size matches Hugging Face expected size
- If checksum available (e.g. SHA256 in HF metadata), verify
- **Do not commit the model file**
- Save GGUF only, no extra files in destination

### Stage 2 — Metadata Probe (31BS or 31BT)
- Read GGUFReader metadata
- Verify `n_layers`, `hidden`, `intermediate`, `context_length`
- Verify `ffn_up`, `ffn_gate`, `ffn_down` raw tensor shapes for layer 0
- Resolve orientation using tensor role and dimensions (per 31BQ caveat)
- **No tensor validation yet**

### Stage 3 — Orientation Parity Micro-Probe (31BU or similar)
- Tiny layer 0 only, single seed
- No aggregate
- Validate orientation formula
- **No claims beyond orientation**

### Stage 4 — Anchor Probe (31BV or similar, only after Stage 3 passes)
- Selected few layers, few seeds
- Only if metadata and orientation pass
- Memory-positive, no severe regressions required

### Stage 5 — Aggregate Validation (31BW+ if warranted)
- Full 384-pair aggregate (or smaller if memory-constrained)
- Comparison to 0.5B baseline
- **Still standalone tensor harness only** — no inference/generation

---

## Stop Conditions

Acquisition/validation phases must STOP and report BLOCKED if any of the following:

1. **Disk space** drops below 5 GB free during any stage
2. **Model is sharded** unexpectedly (1.5B/3B/7B should be single-file)
3. **Checksum/size mismatch** between local file and expected size
4. **GGUFReader cannot open** the downloaded file
5. **Metadata fields missing** (`n_layers`, `hidden`, `intermediate`, etc.)
6. **Orientation ambiguous** — raw shape does not match canonical for any reasonable interpretation
7. **Architecture mismatch** — model is not Qwen2/Qwen2.5
8. **File too large** for available disk
9. **Regression fails** in any stage

---

## Exact Next Phase After 31BR

**Phase 31BS — Approved Larger-Model Download / Metadata Verification, only if explicitly requested**

This is a **download + metadata probe only** phase. It must NOT execute validation. The next phase after 31BS depends on 31BS outcome:
- If download + metadata probe succeeds → 31BT (orientation parity micro-probe)
- If download fails or metadata is missing → 31BR-R (acquisition plan repair)
- If orientation ambiguous → 31BR-R or 31BQ-R (deeper metadata investigation)

---

## Proposed Matt Approval Phrase for 31BS

When Matt is ready to authorize the download, send a message like:

> "I approve Phase 31BS to download Qwen2.5-1.5B-Instruct Q4_K_M into $SDI_MODEL_DIR/qwen2.5-1.5b-official/, max 7 GB budget, huggingface-cli download, metadata-probe only, no tensor validation."

Alternative shorter version:

> "Approve 31BS: download Qwen2.5-1.5B-Instruct Q4_K_M to $SDI_MODEL_DIR/qwen2.5-1.5b-official/, max 7 GB, metadata only, no validation."

Matt can also specify:
- Different quantization (e.g. Q2_K instead of Q4_K_M)
- Different destination
- Different model (e.g. 3B if 1.5B unavailable)
- `wget` or browser-download instead of `huggingface-cli`

---

## Risk Notes

1. No model downloaded in 31BR — pure planning.
2. No validation executed.
3. 3B intermediate_size conflict remains UNRESOLVED.
4. Disk has 96 GB free on root + 42 GB on VL_usb — sufficient for 1.5B.
5. Exact Hugging Face filenames require lookup or user confirmation — assumed standard Qwen2.5 pattern.
6. 32B partial shard (3.69 GB) in HF cache is unusable; could be freed if Matt approves.
7. Private path debt remains in older phase scripts (31AG–31AW) but package path is clean.
8. Orientation parity test (Stage 3) is mandatory before any aggregate validation.

---

## Classification

**`PASS_31BR_ACQUISITION_PLAN_READY`**

- Model target selected: Qwen2.5-1.5B-Instruct, Q4_K_M
- Destination selected: `$SDI_MODEL_DIR/qwen2.5-1.5b-official/` (local operator sets `SDI_MODEL_DIR` to a real path, e.g. `/media/matthew-villnave/VL_usb/models`)
- Disk budget estimated: ~7 GB
- Approval path clear
- Staged plan defined
- Stop conditions defined
- No download, no validation executed