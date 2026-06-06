// phase31cfs_capture.cpp
//
// Phase 31CF-S: minimal standalone C++ harness to capture the exact
// Q4_K_M GGUF runtime FFN-input activation X for Qwen2.5-1.5B-Instruct
// at L0, last prefill token of the prompt "The capital of France is".
//
// Pattern: follows llama.cpp's own examples/eval-callback/eval-callback.cpp
// to use the public common_params::cb_eval hook (a ggml_backend_sched_eval_callback)
// with a tensor-name filter that matches the per-arch graph construction
// tensor label "ffn_inp-0" (set by llama_context::graph_get_cb at qwen.cpp:67).
//
// Constraints (per Phase 31CF-S user approval):
//   - no ~/llama.cpp source modifications
//   - no llama.cpp rebuild (links against existing build/bin/*.so + build/common/libcommon.a)
//   - no Option A / HF fallback
//   - no raw activation arrays committed to sdi-substitutive (output goes to /tmp)
//   - no model files committed
//   - no generation-quality evaluation (prefill only, n_predict=0 / no decode beyond prefill)
//
// SDIX output format (binary file at SDI_CAPTURE_OUT, default /tmp/phase31cfs_p0_l0.bin):
//   64-byte header:
//     0-3   magic   = "SDIX" (0x58494453, little-endian)
//     4-7   version = 0x00000001
//     8-11  dtype   = 0x00000001 = float32
//     12-15 n_dim   = 0x00000002
//     16-23 dim[0]  = 1 (batch=1)
//     24-31 dim[1]  = 1536 (Qwen2.5-1.5B hidden_size)
//     32-39 token_position = 5 (last prefill token of "The capital of France is" with add_bos=true → 7 tokens, index 6; 31CD used add_special_tokens=False; we report the index we actually used)
//     40-47 il      = 0
//     48-55 shape_logical = 1536
//     56-59 prompt_sha256_first4 = first 4 bytes of SHA256 of prompt bytes
//     60-63 reserved = 0
//   64+ raw float32 data (1536 * 4 = 6144 bytes)

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <vector>
#include <string>
#include <regex>
#include <cstdint>
#include <atomic>

// ============================================================================
// LINKER WORKAROUND: pre-existing link-time bug in operator's libllama.so
// ============================================================================
// The operator's ~/llama.cpp/ has a pre-existing link-time bug:
//
//   src/llama.cpp:1648  extern int g_prt_sidecar_root_set;
//   src/llama.cpp:1649  return g_prt_sidecar_root_set ? 1 : 0;
//
// `g_prt_sidecar_root_set` is declared as `static bool` in llama-graph.cpp:196,
// which gives it internal linkage (file-local, mangled as `_ZL22g_prt_sidecar_root_set`).
// The `extern "C"` declaration in llama.cpp references an unmangled C-linkage
// symbol `g_prt_sidecar_root_set` that is NEVER DEFINED as a global, so linking
// any executable against libllama.so produces:
//
//   /usr/bin/ld: libllama.so: undefined reference to `g_prt_sidecar_root_set'
//
// This is a pre-existing build issue in the operator's local llama.cpp checkout
// (NOT introduced by 31CF-S, NOT a modification of ~/llama.cpp source).
//
// Workaround: provide the missing symbol here, in sdi-substitutive's harness.
// The value is initialized to 0 (= "no PRT sidecar data set"), which matches
// the runtime behavior of the static bool in llama-graph.cpp (which starts
// at false and is only set to true in PRT sidecar code paths, which are guarded
// by #ifdef PRT_SIDECAR_PAGER_EXPERIMENTAL — undefined in the operator's build,
// so the static is never set to true at runtime). Therefore providing
// `int g_prt_sidecar_root_set = 0;` here is observably equivalent to the
// static's runtime behavior.
//
// This is NOT a modification of ~/llama.cpp source. The harness remains
// fully in sdi-substitutive/src/. The operator can rebuild ~/llama.cpp to
// fix the underlying bug at any time; this workaround is independent.
extern "C" int g_prt_sidecar_root_set = 0;
// ============================================================================

// SDIX format constants
static constexpr char     SDIX_MAGIC[4]   = {'S', 'D', 'I', 'X'};
static constexpr uint32_t SDIX_VERSION    = 0x00000001;
static constexpr uint32_t SDIX_DTYPE_F32  = 0x00000001;
static constexpr uint32_t SDIX_N_DIM      = 2;

// Capture state
struct capture_state {
    std::atomic<bool>        captured{false};     // single-shot
    std::vector<uint8_t>     data;                // raw tensor bytes
    ggml_type                type{GGML_TYPE_F32};
    int64_t                  ne[GGML_MAX_DIMS]{};
    size_t                   nb[GGML_MAX_DIMS]{};
    std::string              captured_name;       // e.g. "ffn_inp-0"
    int                      captured_il{-1};
    size_t                   n_bytes{0};
    std::string              prompt_text;
    std::vector<llama_token> tokens;
};

// Compute SHA256 of the prompt bytes (first 4 bytes only, for traceability)
static uint32_t prompt_sha256_first4(const std::string & prompt) {
    // minimal FNV-1a 32-bit (NOT a real SHA256 — for traceability header only)
    // we use a real SHA256 if available, but to avoid linking openssl, use a simple
    // 32-bit hash. The header is for traceability, not cryptographic security.
    uint32_t h = 0x811c9dc5u;
    for (unsigned char c : prompt) {
        h ^= c;
        h *= 0x01000193u;
    }
    return h;
}

// The eval callback: fires for every ggml tensor computation. We filter on
// t->name matching "ffn_inp-0" exactly, then copy the data to host memory
// and stop further callbacks (single-shot).
static bool sdi_capture_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * state = static_cast<capture_state *>(user_data);

    if (ask) {
        // Always return true so the tensor is computed and we can see its data
        return true;
    }

    if (state->captured.load()) {
        return true;  // single-shot: ignore subsequent
    }

    // Filter: exact match on "ffn_inp-0" (set by graph_get_cb via ggml_format_name)
    // The tensor name format is "%s-%d" for il >= 0, so "ffn_inp" at il=0 becomes "ffn_inp-0"
    if (t->name == nullptr || std::string(t->name) != "ffn_inp-0") {
        return true;
    }

    // Capture metadata
    state->captured_name = t->name;
    state->captured_il = 0;  // we filtered on -0 suffix
    state->type = t->type;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        state->ne[i] = t->ne[i];
        state->nb[i] = t->nb[i];
    }
    state->n_bytes = ggml_nbytes(t);

    // Copy data to host memory
    state->data.resize(state->n_bytes);
    if (ggml_backend_buffer_is_host(t->buffer)) {
        // already on host
        std::memcpy(state->data.data(), t->data, state->n_bytes);
    } else {
        // offloaded; copy back
        ggml_backend_tensor_get(t, state->data.data(), 0, state->n_bytes);
    }

    state->captured.store(true);
    LOG_INF("captured tensor name=%s type=%s ne=[%lld,%lld,%lld,%lld] n_bytes=%zu\n",
            t->name, ggml_type_name(t->type),
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
            state->n_bytes);
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    capture_state state;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    // Force CPU-only for the operator's no-GPU box
    if (params.n_gpu_layers < 0) {
        params.n_gpu_layers = 0;
    }

    // Default prompt if not provided
    if (params.prompt.empty()) {
        params.prompt = "The capital of France is";
    }
    state.prompt_text = params.prompt;

    // Output path
    const char * out_path = std::getenv("SDI_CAPTURE_OUT");
    if (out_path == nullptr) {
        out_path = "/tmp/phase31cfs_p0_l0.bin";
    }

    // Force n_predict=0: prefill only, no generation
    params.n_predict = 0;

    // No warmup: skip the dummy eval
    params.warmup = false;

    // Hook the eval callback
    params.cb_eval = sdi_capture_cb_eval;
    params.cb_eval_user_data = &state;

    // Init backend
    llama_backend_init();
    llama_numa_init(params.numa);

    // Load model
    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("failed to init model/context\n");
        return 1;
    }

    // Tokenize (matches 31CD: add_special_tokens=False via common_tokenize signature)
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    // 31CD used add_special_tokens=False; the equivalent llama.cpp path is:
    //   std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, /*add_bos=*/false, /*parse_special=*/true);
    // 31CD's prompt "The capital of France is" → 6 BPE tokens.
    // For Option B, we use add_bos=false to match 31CD's tokenization (since
    // Qwen2.5-1.5B does not add BOS automatically for instruct prompts; we want
    // the same 6 tokens as 31CD, not 7).
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, /*add_bos=*/false, /*parse_special=*/true);
    state.tokens = tokens;

    LOG_INF("prompt = %s\n", params.prompt.c_str());
    LOG_INF("add_bos = %s\n", add_bos ? "true" : "false");
    LOG_INF("number of input tokens = %zu\n", tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        char buf[256];
        // Use llama_token_to_piece if available, otherwise just print the id
        int n = llama_token_to_piece(vocab, tokens[i], buf, sizeof(buf), /*lstrip=*/0, /*special=*/true);
        if (n < 0) {
            LOG_INF("  token[%zu] = %d (decode failed: %d)\n", i, tokens[i], n);
        } else {
            LOG_INF("  token[%zu] = %d (\"%s\")\n", i, tokens[i], buf);
        }
    }

    if (tokens.empty()) {
        LOG_ERR("no input tokens\n");
        return 1;
    }

    // Run the prefill (single batch, all tokens at once)
    // Use the same pattern as examples/eval-callback: llama_batch_get_one creates
    // a batch on the stack. For "The capital of France is" with add_bos=false
    // (matches 31CD), we have 6 tokens. The batch's n_tokens is 6, sequence_id 0,
    // logits=false for the prompt tokens. The last token (index 5) has logits=true
    // (we want the model output for the last position).
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size())) != 0) {
        LOG_ERR("llama_decode failed\n");
        return 1;
    }

    if (!state.captured.load()) {
        LOG_ERR("ffn_inp-0 tensor was NOT captured. This means the upstream 'cb_eval' did not see the tensor, OR the tensor name is not 'ffn_inp-0' for this model/build.\n");
        LOG_ERR("Possible causes:\n");
        LOG_ERR("  1. The operator's llama-server build was made with a different llama.cpp source where qwen.cpp does not have cb(ffn_inp, ...) at the expected position.\n");
        LOG_ERR("  2. The graph optimization fused the ffn_inp add into another op and 'ffn_inp' name was not preserved through optimization.\n");
        LOG_ERR("  3. The model is not Qwen2 (e.g. Qwen2MoE or Qwen2VL which use a different graph construction).\n");
        return 2;
    }

    // Verify the captured tensor matches expected shape
    // Expected: ne[0] = 1536 (hidden_size), ne[1] = N_tokens (or 1 if batched)
    // For "The capital of France is" with add_bos=false: 6 tokens → ne[1] = 6
    // The standalone harness's last-prefill-token is index 5; we'd want to
    // take the last token's slice.
    //
    // However, the SDIX format requires [1, 1536] (last prefill token only).
    // We need to slice ne[1] down to 1 by taking the last token's row.
    //
    // In the GGUF Q4_K_M GGUF / llama.cpp runtime, the FFN input activation
    // tensor is f32. We can slice the captured buffer.

    const int64_t dim0 = state.ne[0];
    const int64_t dim1 = state.ne[1];
    LOG_INF("captured tensor raw: ne=[%lld, %lld, %lld, %lld] type=%s\n",
            (long long) dim0, (long long) dim1, (long long) state.ne[2], (long long) state.ne[3],
            ggml_type_name(state.type));

    // For 31CF-S, we want shape [1, 1536] (last prefill token only).
    // The captured buffer layout (contiguous in row-major order) is:
    //   ne[0] = 1536 (hidden) is the inner dim
    //   ne[1] = N_tokens is the outer dim
    //   bytes per row = ne[0] * sizeof(float) = 1536 * 4 = 6144
    // Last token = row index (dim1 - 1).
    //
    // We take the last token's row and write it as shape [1, 1536].

    const int64_t last_token_index = dim1 - 1;
    LOG_INF("last prefill token index = %lld\n", (long long) last_token_index);

    if (state.type != GGML_TYPE_F32) {
        LOG_ERR("captured tensor type is %s, expected GGML_TYPE_F32\n", ggml_type_name(state.type));
        return 3;
    }

    const size_t bytes_per_row = (size_t) dim0 * sizeof(float);
    const size_t last_row_offset = (size_t) last_token_index * bytes_per_row;

    if (last_row_offset + bytes_per_row > state.data.size()) {
        LOG_ERR("internal error: last_row_offset + bytes_per_row > data.size()\n");
        return 4;
    }

    const float * last_row = reinterpret_cast<const float *>(state.data.data() + last_row_offset);

    // Compute SHA256 of the prompt (for the header)
    uint32_t psha = prompt_sha256_first4(state.prompt_text);

    // Compute SHA256 of the captured activation row (for the result JSON)
    // We use a simple FNV-1a here too (32-bit), for the result JSON's "activation_sha256" field.
    uint32_t act_sha = 0x811c9dc5u;
    for (size_t i = 0; i < bytes_per_row; ++i) {
        act_sha ^= state.data[last_row_offset + i];
        act_sha *= 0x01000193u;
    }

    // Compute activation statistics
    float x_min = last_row[0], x_max = last_row[0], x_sum = 0.0f;
    int finite_count = 0, nan_count = 0, inf_count = 0;
    for (int64_t i = 0; i < dim0; ++i) {
        const float v = last_row[i];
        if (std::isnan(v)) { nan_count++; continue; }
        if (std::isinf(v)) { inf_count++; continue; }
        finite_count++;
        if (v < x_min) x_min = v;
        if (v > x_max) x_max = v;
        x_sum += v;
    }
    const float x_mean = (finite_count > 0) ? (x_sum / (float) finite_count) : 0.0f;
    const float x_abs_sum_max = std::max(std::abs(x_min), std::abs(x_max));

    LOG_INF("activation last-row stats: finite=%d nan=%d inf=%d min=%.6e max=%.6e mean=%.6e max_abs=%.6e\n",
            finite_count, nan_count, inf_count, x_min, x_max, x_mean, x_abs_sum_max);

    // Write SDIX file
    std::ofstream ofs(out_path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        LOG_ERR("failed to open %s for writing\n", out_path);
        return 5;
    }

    char header[64];
    std::memcpy(header + 0,  SDIX_MAGIC, 4);
    uint32_t version = SDIX_VERSION;
    uint32_t dtype   = SDIX_DTYPE_F32;
    uint32_t n_dim   = SDIX_N_DIM;
    uint64_t dim0_u  = (uint64_t) 1;            // [1, 1536]
    uint64_t dim1_u  = (uint64_t) dim0;         // 1536
    uint64_t tokpos  = (uint64_t) last_token_index;
    uint64_t il      = (uint64_t) 0;
    uint64_t shape_logical = (uint64_t)(1 * dim0);
    uint32_t psha32  = psha;
    uint32_t reserved = 0;
    std::memcpy(header +  4, &version, 4);
    std::memcpy(header +  8, &dtype, 4);
    std::memcpy(header + 12, &n_dim, 4);
    std::memcpy(header + 16, &dim0_u, 8);
    std::memcpy(header + 24, &dim1_u, 8);
    std::memcpy(header + 32, &tokpos, 8);
    std::memcpy(header + 40, &il, 8);
    std::memcpy(header + 48, &shape_logical, 8);
    std::memcpy(header + 56, &psha32, 4);
    std::memcpy(header + 60, &reserved, 4);
    ofs.write(header, 64);

    ofs.write(reinterpret_cast<const char *>(last_row), bytes_per_row);
    ofs.close();

    LOG_INF("wrote SDIX file: %s (header=64 bytes, payload=%zu bytes, total=%zu bytes)\n",
            out_path, bytes_per_row, 64 + bytes_per_row);
    LOG_INF("  shape: [1, %lld] dtype=f32 token_position=%lld il=0\n",
            (long long) dim0, (long long) last_token_index);
    LOG_INF("  prompt_sha256_first4=0x%08x (truncated FNV-1a, traceability only)\n", psha);
    LOG_INF("  activation_sha256=0x%08x (truncated FNV-1a, traceability only)\n", act_sha);
    LOG_INF("  x_min=%.6e x_max=%.6e x_mean=%.6e max_abs=%.6e\n", x_min, x_max, x_mean, x_abs_sum_max);
    LOG_INF("  finite=%d nan=%d inf=%d (expected finite=1536, nan=0, inf=0)\n", finite_count, nan_count, inf_count);

    // Sanity: print first 8 and last 8 values for verification
    LOG_INF("  first 8 values: %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n",
            last_row[0], last_row[1], last_row[2], last_row[3],
            last_row[4], last_row[5], last_row[6], last_row[7]);
    LOG_INF("  last 8 values:  %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n",
            last_row[dim0 - 8], last_row[dim0 - 7], last_row[dim0 - 6], last_row[dim0 - 5],
            last_row[dim0 - 4], last_row[dim0 - 3], last_row[dim0 - 2], last_row[dim0 - 1]);

    // Persist result metadata to a sibling JSON for downstream replay.
    // We use a minimal hand-rolled JSON to avoid linking json.hpp.
    {
        const std::string json_path = std::string(out_path) + ".meta.json";
        std::ofstream jf(json_path);
        if (jf) {
            jf << "{\n";
            jf << "  \"phase\": \"31CF-S\",\n";
            jf << "  \"option\": \"B\",\n";
            jf << "  \"method\": \"standalone_cpp_harness\",\n";
            jf << "  \"model_path_env_redacted\": \"$SDI_MODEL_DIR/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf\",\n";
            jf << "  \"prompt\": \"" << state.prompt_text << "\",\n";
            jf << "  \"n_tokens\": " << tokens.size() << ",\n";
            jf << "  \"tokens\": [";
            for (size_t i = 0; i < tokens.size(); ++i) { jf << (i ? "," : "") << tokens[i]; }
            jf << "],\n";
            jf << "  \"layer\": 0,\n";
            jf << "  \"token_position\": " << last_token_index << ",\n";
            jf << "  \"shape_captured\": [1, " << dim0 << "],\n";
            jf << "  \"dtype\": \"f32\",\n";
            jf << "  \"activation_sha256_truncated_fnv1a\": \"0x" << std::hex << act_sha << std::dec << "\",\n";
            jf << "  \"prompt_sha256_first4_truncated_fnv1a\": \"0x" << std::hex << psha << std::dec << "\",\n";
            jf << "  \"x_min\": " << x_min << ",\n";
            jf << "  \"x_max\": " << x_max << ",\n";
            jf << "  \"x_mean\": " << x_mean << ",\n";
            jf << "  \"x_max_abs\": " << x_abs_sum_max << ",\n";
            jf << "  \"finite_count\": " << finite_count << ",\n";
            jf << "  \"nan_count\": " << nan_count << ",\n";
            jf << "  \"inf_count\": " << inf_count << ",\n";
            jf << "  \"n_bytes_payload\": " << bytes_per_row << ",\n";
            jf << "  \"n_bytes_total\": " << (64 + bytes_per_row) << ",\n";
            jf << "  \"sdix_magic\": \"SDIX\",\n";
            jf << "  \"sdix_version\": 1,\n";
            jf << "  \"sdix_dtype\": \"f32\",\n";
            jf << "  \"hook_strategy\": \"public params.cb_eval + tensor-name filter 'ffn_inp-0' (per-arch graph construction label set by llama_context::graph_get_cb at qwen.cpp:67 via ggml_format_name('%s-%d', il=0))\",\n";
            jf << "  \"hook_modified_llama_cpp_source\": false,\n";
            jf << "  \"hook_rebuilt_llama_cpp\": false,\n";
            jf << "  \"n_gpu_layers\": " << params.n_gpu_layers << ",\n";
            jf << "  \"no_bos_added\": true,\n";
            jf << "  \"raw_x_path\": \"" << out_path << "\"\n";
            jf << "}\n";
            jf.close();
            LOG_INF("wrote meta JSON: %s\n", json_path.c_str());
        }
    }

    llama_backend_free();
    return 0;
}
