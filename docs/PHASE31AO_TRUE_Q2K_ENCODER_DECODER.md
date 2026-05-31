# Phase 31AO — True Q2_K Encoder/Decoder Correctness Prototype

## Classification
**`BLOCKED_Q2K_ENCODER`**

---

## Q2_K Format Source

| Item | Value |
|------|-------|
| Format source | `llama.cpp/ggml/src/ggml-common.h` |
| block_q2_K layout | d(2) + dmin(2) + scales(16) + qs(64) = **84 bytes** |
| block_size | **256 elements** |
| bpe | **2.625** |

Confirmed from `ggml-common.h`:
```c
typedef struct {
    ggml_half d;        // 2 bytes - delta scale
    ggml_half dmin;     // 2 bytes - min offset
    uint8_t scales[QK_K/16];  // 16 bytes - 4-bit pairs
    uint8_t qs[QK_K/4];       // 64 bytes - 2-bit quads
} block_q2_K;
static_assert(sizeof(block_q2_K) == 84, "wrong q2_K block size/padding");
```

---

## Expected Byte Size for ffn_up

| Item | Value |
|------|-------:|
| ffn_up elements | 4,358,144 |
| Q2_K blocks | 17,024 (256 per block) |
| Expected Q2_K bytes | 1,430,016 (1,396.5 KB) |
| Q4 budget | 2,179,072 (2,128.0 KB) |
| Expected margin | +749,056 (+731.5 KB) |

**True Q2_K would be memory-positive** with +731.5 KB margin per ffn_up tensor.

---

## Encoder/Decoder Status

| Component | Status |
|-----------|--------|
| Q2_K format layout | ✓ Verified from ggml-common.h |
| Q2_K decode formula | ✓ Matches llama.cpp `dequantize_row_q2_K` |
| gguf-py `Q2_K.dequantize_blocks` | ✓ Available |
| gguf-py `Q2_K.quantize_blocks` | ✗ **NotImplemented** |
| Custom Q2_K encoder | ✗ **Blocked** — divide-by-zero, overflow issues |
| Type-10 Q2_K tensors available | ✗ **0** — Q2_K GGUF contains IQ4_NL/Q3_K |

---

## Key Finding

The file `qwen2.5-0.5b-instruct-q2_k.gguf` does **not** contain type-10 (Q2_K) encoded tensors. It contains:
- IQ4_NL (4.5 bpe) for ffn_up/ffn_gate
- Q3_K (3.44 bpe) for ffn_down
- No actual Q2_K type=10 tensors

Without type-10 Q2_K tensors to validate against, and with `gguf-py`'s `Q2_K.quantize_blocks = NotImplemented`, the encoder cannot be reliably implemented.

---

## Classification Rationale

`BLOCKED_Q2K_ENCODER` — three blockers:

1. **No validation reference** — no type-10 Q2_K tensors in any available GGUF
2. **Encoder not implemented** — gguf-py quantize_blocks is NotImplemented, custom encoder has numerical issues
3. **Format known but correctness unproven** — block layout confirmed from header, but cannot validate encode/decode roundtrip

---

## Forbidden Claims (Not Made)

- ❌ No model quality recovery claimed
- ❌ No true Q2_K encode/decode working
- ❌ No memory-positive result yet
- ❌ No llama.cpp integration claimed
- ❌ No production readiness claimed

---

## Recommended Next Phase

**Phase 31AP — Q2_K Encode Implementation via llama.cpp quantize_row_q2_k, only if explicitly requested.**

Options:
1. **Call llama.cpp `quantize_row_q2_k` via ctypes** — if compiled binary available
2. **Use IQ4_NL/Q3_K as proxy** — 31AN showed residuals improve with actual decoded IQ4_NL, memory fails due to 4.5 bpe
3. **Wait for gguf-py Q2_K quantize implementation** — not currently available

Do not proceed without explicit user request.

---

## Artifacts

- **Results JSON:** `results/PHASE31AO_TRUE_Q2K_ENCODER_DECODER.json`
- **Probe script:** `src/phase31ao_true_q2k_encoder_decoder.py`

---

*Phase 31AO — probe run 2026-05-30. Classification: BLOCKED_Q2K_ENCODER.*