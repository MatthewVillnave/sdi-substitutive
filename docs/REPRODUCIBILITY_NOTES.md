# Reproducibility Notes

> How to reproduce Phase 31 results, run regression, and maintain schema compliance.

---

## Quick Start

### Run Regression

```bash
cd /path/to/sdi-substitutive
.venv/bin/python -m tests.run_source_of_truth_regression
```

The regression is self-contained: it generates temporary fixture bundles at runtime and does not require external model files.

### Expected Output

```json
{
  "classification": "PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN",
  "tiny_sdir_apply": { "cosine": 0.99999994, "passed": true },
  "ffn_up_sdir": { "cosine": 1.00000012, "passed": true },
  "ffn_down_sdir": { "cosine": 0.99999988, "passed": true },
  "substitutive_counters": {
    "sdiw_loaded": 2, "sdir_loaded": 2,
    "W_ref_loaded": 0, "W_ref_generated": 0,
    "fallback_count": 0, "error_count": 0
  }
}
```

---

## Required Environment Variables

The following environment variables control runtime paths. If unset, the system attempts to use defaults, but explicit setting is recommended.

| variable           | description                                    | default (if any) |
|-------------------|------------------------------------------------|-----------------|
| `SDI_GGUF_MODEL_PATH` | Path to GGUF model file (Qwen2.5-0.5B etc.) | none — required for model-dependent scripts |
| `SDI_LLAMA_CPP_ROOT` | Root of llama.cpp source tree                  | none — required for model-dependent scripts |
| `SDI_LLAMA_CPP_BUILD` | Path to llama.cpp build directory              | none — required for model-dependent scripts |
| `SDI_ARTIFACT_ROOT`   | Root directory for artifact bundles             | `./data/`       |

**Important:** The regression suite (`tests/run_source_of_truth_regression.py`) does NOT use these variables — it generates temp fixtures entirely in-memory.

---

## What Files Are NOT Committed

The following types of files must NOT be committed to the repo:

```
# Model files
*.gguf          # GGUF model binaries
*.bin           # Raw weight binaries
*.pt / *.npz    # PyTorch/NumPy tensors
*.npy           # NumPy arrays

# Virtual environment
.venv/          # Python venv (use .gitignore)

# Large blobs
*.sdiw          # Packed weight artifacts (generated)
*.sdir          # Sparse residual artifacts (generated)
*.npz           # Tensor archives

# Build artifacts
libtorch*/      # LibTorch checkout
__pycache__/    # Python bytecode cache (handled by gitignore)
*.pyc           # Python compiled files

# Logs (large)
*.log           # Phase execution logs
```

The `.gitignore` file in the repo root handles most of these.

---

## Artifact Location Rules

1. **Bundle directory** = directory containing `manifest.json`
2. **Artifacts** should be stored relative to the bundle directory (e.g., `tensors/blk.21.ffn_up.wlow.sdiw`)
3. **Never** store absolute paths in committed `manifest.json` files
4. **Model files** stay on the external USB mount or wherever they live — never commit them

---

## Updating SOURCE_OF_TRUTH.md

After every phase, update `SOURCE_OF_TRUTH.md` following these rules:

### Required fields per phase update

1. **Phase classification** — the exact classification string
2. **Accepted facts** — verified facts only, no guesses
3. **Invalidated claims** — what was superseded
4. **New suspected/unproven claims** — if any discovered
5. **Current blockers** — what is blocking next phases
6. **Next allowed phase** — who can go next

### Format

```markdown
- **Phase 31XX name:** Classification `EXACT_CLASSIFICATION_STRING`.
  - Finding 1
  - Finding 2
  - Next allowed phase: Phase 31YY — ...
```

### Rules
- Do not add guesses as accepted facts
- If a claim is suspected, put it in the "Suspected / Unproven" section
- If a claim is invalidated, move it to "Invalidated / Superseded Claims"
- Do not delete history — mark it superseded
- If code and SOURCE_OF_TRUTH disagree, stop and resolve

---

## Forbidden Claims

The following claims are **never allowed** in any phase doc, commit message, or artifact metadata:

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

## Schema Version

Current canonical schema version: **"1.0"**

Defined in: `docs/STATIC_ARTIFACT_SCHEMA.md`

The manifest loader (`src/bundle_manifest.py`) validates schema version on load. Any manifest with `schema_version` not matching an accepted version will raise an error.

Accepted schema versions: `"0.2.0"`, `"1.0"`

---

## Canonical Run Order for Phase Scripts

Phase scripts in `src/phase31*.py` are numbered roughly in execution order. To reproduce results:

1. Ensure `.venv/bin/python` has numpy and scipy available
2. Set required environment variables (see above)
3. Ensure llama.cpp library is built at `SDI_LLAMA_CPP_BUILD`
4. Ensure GGUF model file exists at `SDI_GGUF_MODEL_PATH`
5. Run the desired phase script: `.venv/bin/python src/phase31xx_*.py`
6. Always run regression afterward: `.venv/bin/python -m tests.run_source_of_truth_regression`

**Note:** Phase scripts (as opposed to the regression suite) DO contain hardcoded private paths from the development machine. The schema hardening task (Phase 31BF) is intended to address this. Until then, phase scripts may require local path adjustment to run.
