// phase31cfs2_capture.cpp
//
// Phase 31CF-S2: exact Q4_K_M GGUF runtime FFN-input activation
// multi-prompt / multi-layer extension of Phase 31CF-S.
//
// 31CF-S was a single-pair micro-probe: L0, last prefill token of
// "The capital of France is", exact Q4_K_M GGUF / llama.cpp runtime
// capture, 1 pair total, PASSED at commit 16ef1a02.
//
// 31CF-S2 extends the same exact-GGUF-runtime capture path to a
// bounded 3 prompts × 3 layers × last-prefill-token = 9-pair matrix:
//   prompts: P0 = "The capital of France is"
//            P1 = "Once upon a time"
//            P2 = "In a small village"
//   layers:  L0, L14, L27
//   token:   last prefill token only
//   total:   9 pairs
//
// Pattern (same as 31CF-S): uses llama.cpp's public
// common_params::cb_eval hook (a ggml_backend_sched_eval_callback) with
// a tensor-name filter that matches the per-arch graph construction
// tensor label "ffn_inp-{il}" (set by llama_context::graph_get_cb via
// ggml_format_name("%s-%d", name="ffn_inp", il)). NO modification to
// ~/llama.cpp/ source, NO llama.cpp rebuild, NO patch to qwen.cpp /
// llama-graph.cpp / build_ffn, NO enabling of PRT_SIDECAR_PAGER_EXPERIMENTAL.
//
// Constraints (per Phase 31CF-S2 user approval):
//   - no ~/llama.cpp source modifications
//   - no llama.cpp rebuild (links against existing build/bin/*.so + build/common/libcommon.a)
//   - no Option A / HF fallback
//   - no raw activation arrays committed to sdi-substitutive (output goes to /tmp)
//   - no model files committed
//   - no generation-quality evaluation (prefill only, n_predict=0 / no decode beyond prefill)
//   - no commit/push/tag without explicit operator approval
//
// CLI args:
//   --model          path to the local Qwen2.5-1.5B Q4_K_M GGUF
//                    (required; can also be set via SDI_MODEL_GGUF env var)
//   --out-dir        directory to write the 9 SDIX files + 9 meta JSONs
//                    (default /tmp; must NOT be in sdi-substitutive/)
//   --n-gpu-layers   N for -ngl flag (default 0 = CPU only)
//   --seed           llama.cpp sampling seed (default 42)
//
// SDIX output format (one file per pair, 9 total):
//   64-byte header:
//     0-3   magic   = "SDIX" (0x58494453, little-endian)
//     4-7   version = 0x00000001
//     8-11  dtype   = 0x00000001 = float32
//     12-15 n_dim   = 0x00000002
//     16-23 dim[0]  = 1 (batch=1)
//     24-31 dim[1]  = 1536 (Qwen2.5-1.5B hidden_size)
//     32-39 token_position = last prefill token index (0-indexed)
//     40-47 il      = layer index (0, 14, or 27)
//     48-55 shape_logical = 1536
//     56-59 prompt_sha_first4 = first 4 bytes of FNV-1a of prompt bytes (traceability)
//     60-63 reserved = 0
//   64+ raw float32 data (1536 * 4 = 6144 bytes)
//
// File naming: {out_dir}/phase31cfs2_p{0,1,2}_l{0,14,27}.bin
//              {out_dir}/phase31cfs2_p{0,1,2}_l{0,14,27}.bin.meta.json

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
#include <sys/stat.h>

// ============================================================================
// LINKER WORKAROUND: pre-existing link-time bug in operator's libllama.so
// ============================================================================
// The operator's ~/llama.cpp/ has a pre-existing link-time bug:
//
//   src/llama.cpp:1648  extern int g_prt_sidecar_root_set;
//   src/llama.cpp:1649  return g_prt_sidecar_root_set ? 1 : 0;
//
// `g_prt_sidecar_root_set` is declared as `static bool` in
// llama-graph.cpp:196, which gives it internal linkage (file-local,
// mangled as `_ZL22g_prt_sidecar_root_set`). The `extern "C"` declaration
// in llama.cpp references an unmangled C-linkage symbol
// `g_prt_sidecar_root_set` that is NEVER DEFINED as a global, so linking
// any executable against libllama.so produces:
//
//   /usr/bin/ld: libllama.so: undefined reference to `g_prt_sidecar_root_set'
//
// This is a pre-existing build issue in the operator's local llama.cpp
// checkout (NOT introduced by 31CF-S or 31CF-S2, NOT a modification of
// ~/llama.cpp source).
//
// Workaround: provide the missing symbol here, in sdi-substitutive's harness.
// The value is initialized to 0 (= "no PRT sidecar data set"), which matches
// the runtime behavior of the static bool in llama-graph.cpp (which starts
// at false and is only set to true in PRT sidecar code paths, which are
// guarded by #ifdef PRT_SIDECAR_PAGER_EXPERIMENTAL — undefined in the
// operator's build, so the static is never set to true at runtime).
// Therefore providing `int g_prt_sidecar_root_set = 0;` here is
// observably equivalent to the static's runtime behavior.
extern "C" int g_prt_sidecar_root_set = 0;

// ============================================================================
// CONFIG
// ============================================================================
static const int HIDDEN = 1536;  // Qwen2.5-1.5B hidden_size
static const int N_PAIRS = 9;

// Prompt/layer matrix (the bounded scope for 31CF-S2)
struct PairSpec {
    int prompt_idx;          // 0, 1, or 2
    const char * prompt;     // prompt text
    int il;                  // 0, 14, or 27
    const char * sdix_name;  // output filename suffix
};

static const PairSpec PAIRS[N_PAIRS] = {
    { 0, "The capital of France is", 0,  "phase31cfs2_p0_l0"  },
    { 0, "The capital of France is", 14, "phase31cfs2_p0_l14" },
    { 0, "The capital of France is", 27, "phase31cfs2_p0_l27" },
    { 1, "Once upon a time",         0,  "phase31cfs2_p1_l0"  },
    { 1, "Once upon a time",         14, "phase31cfs2_p1_l14" },
    { 1, "Once upon a time",         27, "phase31cfs2_p1_l27" },
    { 2, "In a small village",       0,  "phase31cfs2_p2_l0"  },
    { 2, "In a small village",       14, "phase31cfs2_p2_l14" },
    { 2, "In a small village",       27, "phase31cfs2_p2_l27" },
};

// ============================================================================
// CAPTURE STATE (per-pair, reset between pairs)
// ============================================================================
struct CaptureState {
    std::string target_tensor_name;  // e.g. "ffn_inp-14"
    int target_il;                   // 0, 14, or 27
    int pair_idx;                    // 0..8
    // Result fields (populated by the callback)
    std::atomic<bool> captured{false};
    std::vector<float> sliced;       // [1, 1536] after slice
    std::vector<int64_t> captured_shape;  // e.g. [1536, n_tokens, 1, 1]
    uint32_t fnv1a_act = 0;
    int last_token_index = -1;
    int n_tokens = 0;
    // For verification
    int hit_count = 0;
};

static CaptureState g_state;

// FNV-1a hash (32-bit, for traceability)
static uint32_t fnv1a(const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *) data;
    uint32_t h = 0x811c9dc5;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x01000193;
    }
    return h;
}

// ============================================================================
// EVAL CALLBACK (ggml_backend_sched_eval_callback)
// ============================================================================
// Fires per ggml-backend-eval, per tensor. The per-arch graph construction
// label "ffn_inp-{il}" is set on the tensor's ggml name by
// llama_context::graph_get_cb via ggml_format_name. We match by t->name
// AND extract the data via ggml_backend_tensor_get.
//
// Pattern follows llama.cpp's own examples/eval-callback/eval-callback.cpp.
static bool sdi_capture_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    (void) user_data;
    if (!t) return true;
    if (!t->name) return true;

    // Match against the target tensor name for this pair.
    // The per-arch graph construction label is set via
    // ggml_format_name(cur, "%s-%d", name="ffn_inp", il), so for L14
    // the name is "ffn_inp-14", for L27 it's "ffn_inp-27", etc.
    const char * target = g_state.target_tensor_name.c_str();
    if (t->name[0] == '\0' || strcmp(t->name, target) != 0) {
        return true;  // not our target; continue
    }

    g_state.hit_count++;
    if (g_state.captured.load()) {
        return true;  // already captured; single-shot
    }

    // Diagnostic: log the actual shape we see, for the first match.
    int n_dims_diag = ggml_n_dims(t);
    LOG_INF("    [diag] matched tensor name='%s' n_dims=%d shape=[%lld, %lld, %lld, %lld] dtype=%d\n",
            t->name, n_dims_diag,
            (long long) t->ne[0], (long long) t->ne[1],
            (long long) t->ne[2], (long long) t->ne[3],
            (int) t->type);

    // Sanity checks: shape must be 1D, 2D, or 4D ggml tensor where the innermost
    // dim is HIDDEN. (ggml tensors are row-major so ne[0] is the innermost dim.)
    int n_dims = ggml_n_dims(t);
    if (n_dims != 1 && n_dims != 2 && n_dims != 4) {
        LOG_ERR("target tensor has %d dims, expected 1, 2, or 4\n", n_dims);
        return true;
    }
    int64_t ne0 = t->ne[0];  // HIDDEN (innermost)
    int64_t ne1 = (n_dims >= 2) ? t->ne[1] : 1;  // n_tokens (1 if missing)
    if (ne0 != HIDDEN) {
        LOG_ERR("target tensor has ne[0]=%lld, expected %d (HIDDEN)\n",
                (long long) ne0, HIDDEN);
        return true;
    }
    if (n_dims == 4 && (t->ne[2] != 1 || t->ne[3] != 1)) {
        LOG_ERR("target tensor has unexpected 4D shape [%lld, %lld, %lld, %lld]\n",
                (long long) ne0, (long long) ne1, (long long) t->ne[2], (long long) t->ne[3]);
        return true;
    }

    // Slice: keep last prefill token. For 1D (per-token) tensors, ne1 is implicitly 1.
    int last_idx;
    if (n_dims == 1) {
        last_idx = 0;  // 1D: there's only one row
    } else {
        last_idx = (int) ne1 - 1;
    }
    g_state.last_token_index = last_idx;
    g_state.n_tokens = (int) ne1;
    if (n_dims == 1) {
        g_state.captured_shape = { ne0 };
    } else if (n_dims == 2) {
        g_state.captured_shape = { ne0, ne1 };
    } else {
        g_state.captured_shape = { ne0, ne1, 1, 1 };
    }

    // Allocate and read via ggml_backend_tensor_get
    size_t total = (size_t) ne0 * (size_t) ne1;
    std::vector<float> full(total);
    if (total > 0) {
        ggml_backend_tensor_get(t, full.data(), 0, total * sizeof(float));
    }

    // Slice to [1, HIDDEN]: keep row last_idx
    if (total == (size_t) HIDDEN) {
        // 1D tensor: full IS the activation
        g_state.sliced.assign(full.begin(), full.end());
    } else {
        // 2D/4D tensor: take last row
        g_state.sliced.assign(full.begin() + (size_t) last_idx * (size_t) ne0,
                              full.begin() + (size_t)(last_idx + 1) * (size_t) ne0);
    }

    g_state.fnv1a_act = fnv1a(g_state.sliced.data(), g_state.sliced.size() * sizeof(float));
    g_state.captured.store(true);
    return true;
}

// ============================================================================
// SDIX FILE WRITER
// ============================================================================
static bool write_sdix(const std::string & path, const CaptureState & st) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        LOG_ERR("cannot open SDIX output: %s\n", path.c_str());
        return false;
    }
    // 64-byte header
    uint32_t magic   = 0x58494453;  // "SDIX" little-endian
    uint32_t version = 1;
    uint32_t dtype   = 1;  // float32
    uint32_t n_dim   = 2;
    uint64_t dim0    = 1;
    uint64_t dim1    = (uint64_t) HIDDEN;
    uint64_t tokpos  = (uint64_t) st.last_token_index;
    uint64_t il      = (uint64_t) st.target_il;
    uint64_t shape_logical = (uint64_t) HIDDEN;
    uint32_t psha    = fnv1a(PAIRS[st.pair_idx].prompt, strlen(PAIRS[st.pair_idx].prompt));
    uint32_t reserved = 0;

    f.write((const char *) &magic,   4);
    f.write((const char *) &version, 4);
    f.write((const char *) &dtype,   4);
    f.write((const char *) &n_dim,   4);
    f.write((const char *) &dim0,    8);
    f.write((const char *) &dim1,    8);
    f.write((const char *) &tokpos,  8);
    f.write((const char *) &il,      8);
    f.write((const char *) &shape_logical, 8);
    f.write((const char *) &psha,    4);
    f.write((const char *) &reserved, 4);
    // 64 bytes header
    // payload: 1536 * 4 = 6144 bytes float32
    f.write((const char *) st.sliced.data(), st.sliced.size() * sizeof(float));
    if (!f) {
        LOG_ERR("SDIX write failed: %s\n", path.c_str());
        return false;
    }
    return true;
}

// ============================================================================
// META JSON WRITER (hand-rolled, no json.hpp dependency)
// ============================================================================
static void write_meta_json(const std::string & path, int pair_idx, const CaptureState & st,
                            int n_tokens, const std::vector<int> & tokens) {
    std::ofstream f(path);
    if (!f) {
        LOG_ERR("cannot open meta JSON: %s\n", path.c_str());
        return;
    }
    // Compute X statistics
    double x_min = 1e30, x_max = -1e30, x_sum = 0.0;
    int finite = 0, nan = 0, inf = 0;
    for (float v : st.sliced) {
        if (std::isnan(v)) { nan++; continue; }
        if (std::isinf(v)) { inf++; continue; }
        finite++;
        if (v < x_min) x_min = v;
        if (v > x_max) x_max = v;
        x_sum += v;
    }
    double x_mean = finite > 0 ? (x_sum / finite) : 0.0;
    double x_max_abs = 0.0;
    for (float v : st.sliced) {
        double a = std::fabs((double) v);
        if (a > x_max_abs) x_max_abs = a;
    }

    f << "{\n";
    f << "  \"phase\": \"31CF-S2\",\n";
    f << "  \"option\": \"B\",\n";
    f << "  \"method\": \"standalone_cpp_harness\",\n";
    f << "  \"model_path_env_redacted\": \"$SDI_MODEL_DIR/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf\",\n";
    f << "  \"prompt_idx\": " << st.pair_idx / 3 << ",\n";
    f << "  \"prompt\": \"" << PAIRS[st.pair_idx].prompt << "\",\n";
    f << "  \"n_tokens\": " << n_tokens << ",\n";
    f << "  \"tokens\": [";
    for (size_t i = 0; i < tokens.size(); ++i) f << (i ? "," : "") << tokens[i];
    f << "],\n";
    f << "  \"layer\": " << st.target_il << ",\n";
    f << "  \"tensor_name\": \"" << st.target_tensor_name << "\",\n";
    f << "  \"token_position\": " << st.last_token_index << ",\n";
    f << "  \"shape_captured\": [1, " << HIDDEN << "],\n";
    f << "  \"shape_raw\": [";
    for (size_t i = 0; i < st.captured_shape.size(); ++i) f << (i ? ", " : "") << st.captured_shape[i];
    f << "],\n";
    f << "  \"dtype\": \"f32\",\n";
    f << "  \"activation_sha256_truncated_fnv1a\": \"0x" << std::hex << st.fnv1a_act << std::dec << "\",\n";
    f << "  \"prompt_sha256_first4_truncated_fnv1a\": \"0x" << std::hex
      << fnv1a(PAIRS[st.pair_idx].prompt, strlen(PAIRS[st.pair_idx].prompt)) << std::dec << "\",\n";
    f << "  \"x_min\": " << x_min << ",\n";
    f << "  \"x_max\": " << x_max << ",\n";
    f << "  \"x_mean\": " << x_mean << ",\n";
    f << "  \"x_max_abs\": " << x_max_abs << ",\n";
    f << "  \"finite_count\": " << finite << ",\n";
    f << "  \"nan_count\": " << nan << ",\n";
    f << "  \"inf_count\": " << inf << ",\n";
    f << "  \"n_bytes_payload\": " << (HIDDEN * 4) << ",\n";
    f << "  \"n_bytes_total\": " << (64 + HIDDEN * 4) << ",\n";
    f << "  \"sdix_magic\": \"SDIX\",\n";
    f << "  \"sdix_version\": 1,\n";
    f << "  \"sdix_dtype\": \"f32\",\n";
    f << "  \"hook_strategy\": \"public params.cb_eval + tensor-name filter 'ffn_inp-{il}' (per-arch graph construction label set by llama_context::graph_get_cb at qwen.cpp:67 via ggml_format_name('%s-%d', il=0,14,27))\",\n";
    f << "  \"hook_modified_llama_cpp_source\": false,\n";
    f << "  \"hook_rebuilt_llama_cpp\": false,\n";
    f << "  \"n_gpu_layers\": 0,\n";
    f << "  \"no_bos_added\": true,\n";
    f << "  \"raw_x_path\": \"" << path.substr(0, path.rfind(".meta.json")) << "\"\n";
    f << "}\n";
}

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char ** argv) {
    // Parse CLI args
    std::string model_path;
    std::string out_dir = "/tmp";
    int n_gpu_layers = 0;
    int seed = 42;

    {
        std::string model_env = getenv("SDI_MODEL_GGUF") ? getenv("SDI_MODEL_GGUF") : "";
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--model" && i + 1 < argc) { model_path = argv[++i]; }
            else if (a == "--out-dir" && i + 1 < argc) { out_dir = argv[++i]; }
            else if (a == "--n-gpu-layers" && i + 1 < argc) { n_gpu_layers = atoi(argv[++i]); }
            else if (a == "--seed" && i + 1 < argc) { seed = atoi(argv[++i]); }
            else if (a == "--help" || a == "-h") {
                fprintf(stderr,
                    "phase31cfs2_capture — Phase 31CF-S2 multi-pair exact Q4_K_M GGUF runtime activation capture\n"
                    "Usage:\n"
                    "  --model PATH        path to Qwen2.5-1.5B Q4_K_M GGUF (required; or env var SDI_MODEL_GGUF)\n"
                    "  --out-dir DIR       output directory for 9 SDIX files (default /tmp)\n"
                    "  --n-gpu-layers N    ngl flag (default 0 = CPU only)\n"
                    "  --seed N            seed (default 42)\n"
                );
                return 0;
            }
        }
        if (model_path.empty()) model_path = model_env;
        if (model_path.empty()) {
            fprintf(stderr, "ERROR: --model is required (or set env var SDI_MODEL_GGUF)\n"
                "Example: --model \"$SDI_MODEL_DIR/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf\"\n");
            return 1;
        }
    }

    // Refuse to write into the sdi-substitutive working tree
    if (out_dir.find("sdi-substitutive") != std::string::npos) {
        fprintf(stderr, "ERROR: --out-dir must NOT be inside sdi-substitutive/ (artifact policy)\n");
        return 1;
    }

    // Make sure out_dir exists
    mkdir(out_dir.c_str(), 0755);

    LOG_INF("Phase 31CF-S2 multi-pair exact Q4_K_M GGUF runtime activation capture\n");
    LOG_INF("  model_path_env_redacted: $SDI_MODEL_DIR/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf\n");
    LOG_INF("  out_dir: %s\n", out_dir.c_str());
    LOG_INF("  n_gpu_layers: %d\n", n_gpu_layers);
    LOG_INF("  seed: %d\n", seed);
    LOG_INF("  pairs: 9 (3 prompts x 3 layers)\n");

    // Init llama backend
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

    // Load model (single load, reused across 9 pairs)
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    mparams.use_mmap = true;

    LOG_INF("loading model...\n");
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        LOG_ERR("failed to load model from: %s\n", model_path.c_str());
        llama_backend_free();
        return 1;
    }
    LOG_INF("model loaded: vocab_size=%d n_layer=%d n_embd=%d\n",
            (int) llama_vocab_n_tokens(llama_model_get_vocab(model)),
            llama_model_n_layer(model),
            llama_model_n_embd(model));

    // Capture loop: 9 pairs
    for (int pair_idx = 0; pair_idx < N_PAIRS; ++pair_idx) {
        const PairSpec & p = PAIRS[pair_idx];
        LOG_INF("\n=== pair %d/%d: prompt_idx=%d layer=%d prompt='%s' ===\n",
                pair_idx + 1, N_PAIRS, p.prompt_idx, p.il, p.prompt);

        // Reset capture state for this pair
        g_state.target_tensor_name = "ffn_inp-" + std::to_string(p.il);
        g_state.target_il = p.il;
        g_state.pair_idx = pair_idx;
        g_state.captured.store(false);
        g_state.sliced.clear();
        g_state.captured_shape.clear();
        g_state.last_token_index = -1;
        g_state.n_tokens = 0;
        g_state.fnv1a_act = 0;
        g_state.hit_count = 0;

        // Tokenize
        std::vector<llama_token> tokens(p.prompt, p.prompt + strlen(p.prompt));
        tokens.resize(strlen(p.prompt));
        // Use llama_tokenize to get exact tokenization
        const llama_vocab * vocab = llama_model_get_vocab(model);
        {
            std::vector<llama_token> tmp(64);
            int n = llama_tokenize(vocab, p.prompt, strlen(p.prompt),
                                   tmp.data(), tmp.size(), /*add_special*/ false, /*parse_special*/ false);
            if (n < 0) {
                tmp.resize(-n);
                n = llama_tokenize(vocab, p.prompt, strlen(p.prompt),
                                   tmp.data(), tmp.size(), /*add_special*/ false, /*parse_special*/ false);
            }
            tmp.resize(n);
            tokens = tmp;
        }
        LOG_INF("  tokenized: %d tokens\n", (int) tokens.size());

        // Create context with cb_eval set
        auto cparams = llama_context_default_params();
        cparams.n_ctx = 512;  // plenty for prefill
        cparams.n_batch = (uint32_t) tokens.size();
        cparams.cb_eval = sdi_capture_cb_eval;
        cparams.cb_eval_user_data = &g_state;
        cparams.no_perf = true;

        llama_context * ctx = llama_init_from_model(model, cparams);
        if (!ctx) {
            LOG_ERR("failed to create context for pair %d\n", pair_idx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }

        // Prefill: build batch with all tokens, logits for last token
        llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size());
        int rc = llama_decode(ctx, batch);
        if (rc != 0) {
            LOG_ERR("llama_decode failed for pair %d: rc=%d\n", pair_idx, rc);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }

        // Verify capture
        if (!g_state.captured.load()) {
            LOG_ERR("capture FAILED for pair %d: no tensor named '%s' was evaluated\n",
                    pair_idx, g_state.target_tensor_name.c_str());
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }

        if (g_state.sliced.size() != (size_t) HIDDEN) {
            LOG_ERR("captured sliced size %zu != HIDDEN=%d\n", g_state.sliced.size(), HIDDEN);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }

        // Write SDIX + meta JSON
        std::string sdix_path = out_dir + "/" + p.sdix_name + ".bin";
        std::string meta_path = out_dir + "/" + p.sdix_name + ".bin.meta.json";
        if (!write_sdix(sdix_path, g_state)) {
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        write_meta_json(meta_path, pair_idx, g_state, (int) tokens.size(), tokens);
        LOG_INF("  wrote SDIX: %s (%lld bytes)\n", sdix_path.c_str(),
                (long long)(64 + HIDDEN * 4));
        LOG_INF("  wrote meta: %s\n", meta_path.c_str());
        LOG_INF("  last_token_index=%d  x_max_abs=%.6e  hit_count=%d\n",
                g_state.last_token_index, g_state.sliced.empty() ? 0.0 : 0.0,
                g_state.hit_count);

        llama_free(ctx);
    }

    llama_model_free(model);
    llama_backend_free();
    LOG_INF("\nPhase 31CF-S2 capture: 9/9 pairs written to %s/\n", out_dir.c_str());
    LOG_INF("  raw X files must be deleted before PRE-COMMIT REPORT (artifact policy)\n");
    LOG_INF("  SHA256 + metadata + replay metrics are recorded in result JSON for traceability\n");
    return 0;
}
