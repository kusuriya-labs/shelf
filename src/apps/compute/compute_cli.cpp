#include "compute_cli.h"
#include <shelf/kernels/kv_types.h>

#include <cerrno>
#include <cmath>
#include <cstdlib>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace shelf::app {
namespace {

bool parse_uint(const char* s, unsigned long& out) {
    if (s == nullptr || *s == '\0') return false;
    char* end = nullptr;
    out = std::strtoul(s, &end, 10);
    return end != s && *end == '\0';
}

bool parse_float(const char* s, float& out) {
    if (s == nullptr || *s == '\0') return false;
    char* end = nullptr;
    out = std::strtof(s, &end);
    return end != s && *end == '\0';
}

bool parse_u64(const char* s, uint64_t& out) {
    if (s == nullptr || *s == '\0') return false;
    char* end = nullptr;
    out = std::strtoull(s, &end, 10);
    return end != s && *end == '\0';
}

bool parse_devices(const char* s, std::vector<int>& out) {
    if (s == nullptr || *s == '\0') return false;
    out.clear();
    const char* p = s;
    while (*p != '\0') {
        const char* comma = std::strchr(p, ',');
        const std::string token(p, comma == nullptr ? std::string::size_type(std::strlen(p))
                                                    : static_cast<std::string::size_type>(comma - p));
        unsigned long value = 0;
        if (!parse_uint(token.c_str(), value) || value > 65535ul) return false;
        out.push_back(static_cast<int>(value));
        if (comma == nullptr) break;
        p = comma + 1;
    }
    return !out.empty();
}

}  // namespace

void compute_usage(const char* prog) {
    printf("usage: %s [options]\n", prog);
    printf("\nShelf inference runtime for Qwen 3.5 dense and grouped-expert models"
           " on AMD RDNA4\n");
    printf("options:\n");
    printf("  --model-dir PATH       BF16 or quantized model directory (auto-detected via\n");
    printf("                         shelf_quantization.json) (required)\n");
    printf("  --input-ids-file PATH  whitespace separated or PSKLDTOK binary prompt token ids\n");
    printf("  --max-new-tokens N     max tokens to generate (default 32)\n");
    printf("  --max-seq-len N        max sequence length (default 512)\n");
    printf("  --arena-gib N|auto|-N  arena capacity in GiB; auto = free VRAM minus 0.5 GiB,\n"
           "                         -N = total VRAM minus N GiB\n");
    printf("                         (default auto)\n");
    printf("  --page-tokens N        paged KV page size in tokens (default 16)\n");
    printf("  --max-concurrent-requests N  max in-flight requests (default 4)\n");
    printf("  --max-num-batched-tokens N   token budget per scheduled step; also sets\n");
    printf("                         the executor row bucket (default 2048)\n");
    printf("  --device N|N,M         GPU device(s); count must match --tensor-parallel or --pipeline-parallel (default 0)\n");
    printf("  --tensor-parallel N    tensor parallel degree 1 | 2 | 4 (default 1)\n");
    printf("  --pipeline-parallel N  pipeline-parallel stage count (default 1; requires --tensor-parallel 1)\n");
    printf("  --kv-cache-dtype TYPE  bf16 | fp8 | octave:cKvV | octave:bRnNvV (default bf16)\n");
    printf("  --rope-factor MODE     auto | 1 | 2 | 4; static YaRN rope factor\n");
    printf("                         (default auto = model config: native unless the\n");
    printf("                         checkpoint requests yarn)\n");
    printf("  --decode-backend TYPE  hip (default)\n");
    printf("  --verify-weights 0|1      verify quantized payload CRC32 on load (default 0)\n");
    printf("  --dump-logits PATH      append per-step sampled logits rows (raw f32) to PATH\n");
    printf("                          row k aligns with GENERATED_IDS[k] (prefill row first)\n");
    printf("  --hidden-taps LIST      comma separated layer ids (max 8) for intermediate\n");
    printf("                          hidden capture; dump to SHELF_DEBUG_HIDDEN_FILE\n");
    printf("  --tensor-parallel-combine-f32 0|1  keep row-parallel combine output in f32 (default 0 =\n");
    printf("                          bf16 round as today; tp only)\n");
    printf("  --temperature F        sampling temperature (default 0 = greedy)\n");
    printf("  --top-p F              nucleus sampling threshold in (0, 1] (default 1)\n");
    printf("  --min-p F              min-p threshold in [0, 1] (default 0 = disabled)\n");
    printf("  --top-k N              keep the N highest logits (default 0 = disabled)\n");
    printf("  --seed N               sampling seed (default 0)\n");
    printf("  --dflash2-model-dir PATH  DFlash2 draft model directory; enables\n");
    printf("                            speculative decoding (greedy / stochastic)\n");
    printf("  --dflash2-tree-budget N|auto  tree verify rows (default auto; 0 disables)\n");
    printf("                         auto: 10 for one decoding request if speculative, else 8\n");
    printf("  --dflash2-tree-extra B  enable experimental tree verification (B=0,1,2,4)\n");
    printf("  --dflash2-drafts N     speculative draft tokens per round (default 7)\n");
    printf("  --dflash2-adaptive     control speculation with a batch policy\n");
    printf("  --dflash2-max-sessions N  requests that may speculate at once (serve; default\n");
    printf("                            min(--max-concurrent-requests, 8)); the rest decode\n");
    printf("                            without speculation\n");
    printf("  --dflash2-stats 0|1    print speculative decode statistics (default 1)\n");
    printf("  --mtp                  enable the checkpoint's MTP block as the drafter\n");
    printf("  --mtp-drafts N         speculative draft tokens per round (default 7,\n");
    printf("                         max %u)\n", shelf::qwen35::runtime::kMtpMaxDrafts);
    printf("  --mtp-stats 0|1        print MTP speculative decode statistics (default 1)\n");
    printf("  --prefix-cache 0|1     enable prefix cache (default 1)\n");
    printf("  --working-set-reserve-gib N    arena headroom kept free for the working set\n");
    printf("                                 (executor workspace allocated after the\n");
    printf("                                 memory plan; 0 = size it from the row\n");
    printf("                                 bucket, default 0)\n");
    printf("  --prefix-cache-concurrency N   max concurrent prefix captures (default 4)\n");
    printf("  --serve-stdio          serve JSON Lines generation requests on stdin/stdout\n");
    printf("  --help                 show this help\n");
}

std::vector<int32_t> read_input_ids(const std::string& path) {
    std::vector<int32_t> ids;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return ids;
    char magic[8]{};
    const std::size_t prefix_size = std::fread(magic, 1, sizeof(magic), f);
    if (prefix_size == sizeof(magic) && std::memcmp(magic, "PSKLDTOK", sizeof(magic)) == 0) {
        uint32_t version = 0;
        uint64_t count = 0;
        bool ok = std::fread(&version, sizeof(version), 1, f) == 1 && version == 1u &&
            std::fread(&count, sizeof(count), 1, f) == 1 && count > 0 && count < (1u << 22);
        if (ok) {
            ids.resize(static_cast<std::size_t>(count));
            ok = std::fread(ids.data(), sizeof(int32_t), ids.size(), f) == ids.size();
        }
        std::fclose(f);
        if (!ok) ids.clear();
        return ids;
    }
    // /dev/stdin can be a pipe, so fseek/rewind is not available. Return the
    // probe bytes in reverse order before using the existing text parser.
    for (std::size_t i = prefix_size; i > 0; --i) {
        if (std::ungetc(static_cast<unsigned char>(magic[i - 1]), f) == EOF) {
            std::fclose(f);
            return {};
        }
    }
    long v = 0;
    while (std::fscanf(f, "%ld", &v) == 1) {
        ids.push_back(static_cast<int32_t>(v));
    }
    std::fclose(f);
    return ids;
}

std::size_t safetensors_dir_bytes(const std::string& dir) {
    std::size_t total = 0;
    std::error_code ec;
    const std::filesystem::path root(dir);
    if (!std::filesystem::is_directory(root, ec)) {
        return 0;
    }
    for (const auto& entry : std::filesystem::directory_iterator(root, ec)) {
        if (ec) break;
        if (!entry.is_regular_file(ec)) continue;
        const std::string name = entry.path().filename().string();
        constexpr std::size_t kSuffix = 12;
        if (name.size() <= kSuffix ||
            name.compare(name.size() - kSuffix, kSuffix, ".safetensors") != 0) {
            continue;
        }
        total += static_cast<std::size_t>(entry.file_size(ec));
    }
    return total;
}

int parse_compute_options(int argc, char** argv, ComputeOptions& opt) {
    auto& model_dir = opt.model_dir;
    auto& input_ids_file = opt.input_ids_file;
    auto& max_new_tokens = opt.max_new_tokens;
    auto& max_seq_len = opt.max_seq_len;
    auto& arena_gib = opt.arena_gib;
    auto& arena_gib_auto = opt.arena_gib_auto;
    auto& page_tokens = opt.page_tokens;
        auto& max_num_batched_tokens = opt.max_num_batched_tokens;
auto& max_concurrent_requests = opt.max_concurrent_requests;
    auto& devices = opt.devices;
    auto& tp = opt.tp;
    auto& pipeline_parallel = opt.pipeline_parallel;
    auto& kv_cache_dtype = opt.kv_cache_dtype;
    shelf::qwen35::runtime::DecodeBackend& decode_backend = opt.decode_backend;
    auto& verify_weights = opt.verify_weights;
    auto& dump_logits_path = opt.dump_logits_path;
    auto& hidden_taps = opt.hidden_taps;
    auto& tp_combine_f32 = opt.tp_combine_f32;
    auto& serve_stdio = opt.serve_stdio;
    shelf::qwen35::runtime::SamplingConfig& sampling = opt.sampling;
    auto& dflash2_model_dir = opt.dflash2_model_dir;
    auto& dflash2_drafts = opt.dflash2_drafts;
    auto& dflash2_max_sessions = opt.dflash2_max_sessions;
    auto& dflash2_stats = opt.dflash2_stats;
    auto& mtp = opt.mtp;
    auto& mtp_drafts = opt.mtp_drafts;
    auto& mtp_stats = opt.mtp_stats;
    auto& prefix_cache_enabled = opt.prefix_cache_enabled;
    auto& prefix_cache_concurrency = opt.prefix_cache_concurrency;
    auto& working_set_reserve_gib = opt.working_set_reserve_gib;
    auto& rope_factor = opt.rope_factor;
    unsigned long top_k = 0;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 ||
            (argv[i][0] == '-' && argv[i][1] == 'h' && argv[i][2] == '\0')) {
            compute_usage(argv[0]);
            return kComputeParseHelp;
        } else if (std::strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--input-ids-file") == 0 && i + 1 < argc) {
            input_ids_file = argv[++i];
        } else if (std::strcmp(argv[i], "--max-new-tokens") == 0 && i + 1 < argc) {
            if (!parse_uint(argv[++i], max_new_tokens)) { fprintf(stderr, "bad --max-new-tokens\n"); return kComputeParseError; }
        } else if (std::strcmp(argv[i], "--max-seq-len") == 0 && i + 1 < argc) {
            if (!parse_uint(argv[++i], max_seq_len)) { fprintf(stderr, "bad --max-seq-len\n"); return kComputeParseError; }
        } else if (std::strcmp(argv[i], "--arena-gib") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            if (std::strcmp(value, "auto") == 0) {
                arena_gib_auto = true;
            } else {
                char* end = nullptr;
                errno = 0;
                const double gib = std::strtod(value, &end);
                if (errno != 0 || end == value || *end != '\0' || !std::isfinite(gib) ||
                    gib == 0.0) {
                    fprintf(stderr, "bad --arena-gib (expected N, -N or auto)\n");
                    return kComputeParseError;
                }
                arena_gib_auto = false;
                arena_gib = gib;
            }
        } else if (std::strcmp(argv[i], "--page-tokens") == 0 && i + 1 < argc) {
            if (!parse_uint(argv[++i], page_tokens) || page_tokens == 0) { fprintf(stderr, "bad --page-tokens\n"); return kComputeParseError; }
        } else if (std::strcmp(argv[i], "--max-concurrent-requests") == 0 && i + 1 < argc) {
            if (!parse_uint(argv[++i], max_concurrent_requests) || max_concurrent_requests == 0) {
                fprintf(stderr, "bad --max-concurrent-requests\n"); return kComputeParseError;
            }
        } else if (std::strcmp(argv[i], "--max-num-batched-tokens") == 0 && i + 1 < argc) {
            if (!parse_uint(argv[++i], max_num_batched_tokens) || max_num_batched_tokens < 16u ||
                max_num_batched_tokens > 8192u) {
                fprintf(stderr, "bad --max-num-batched-tokens (16..8192)\n");
                return kComputeParseError;
            }
        } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            if (!parse_devices(argv[++i], devices)) { fprintf(stderr, "bad --device\n"); return kComputeParseError; }
        } else if (std::strcmp(argv[i], "--tensor-parallel") == 0 && i + 1 < argc) {
            if (!parse_uint(argv[++i], tp) || (tp != 1 && tp != 2 && tp != 4)) {
                fprintf(stderr, "bad --tensor-parallel (expected 1, 2 or 4)\n");
                return kComputeParseError;
            }
        } else if (std::strcmp(argv[i], "--pipeline-parallel") == 0 && i + 1 < argc) {
            if (!parse_uint(argv[++i], pipeline_parallel) || pipeline_parallel < 1) {
                fprintf(stderr, "bad --pipeline-parallel (expected N >= 1)\n");
                return kComputeParseError;
            }
        } else if (std::strcmp(argv[i], "--kv-cache-dtype") == 0 && i + 1 < argc) {
            kv_cache_dtype = argv[++i];
            shelf::kernel::OctaveKVCacheFormat octave_format{};
            if (kv_cache_dtype != "bf16" && kv_cache_dtype != "fp8" &&
                !shelf::kernel::parse_octave_kv_cache_format(kv_cache_dtype, octave_format)) {
                fprintf(stderr, "bad --kv-cache-dtype: %s\n", kv_cache_dtype.c_str());
                return kComputeParseError;
            }
        } else if (std::strcmp(argv[i], "--rope-factor") == 0 && i + 1 < argc) {
            rope_factor = argv[++i];
            if (rope_factor != "auto" && rope_factor != "1" && rope_factor != "2" &&
                rope_factor != "4") {
                fprintf(stderr,
                        "bad --rope-factor: %s (expected auto | 1 | 2 | 4)\n",
                        rope_factor.c_str());
                return kComputeParseError;
            }
        } else if (std::strcmp(argv[i], "--decode-backend") == 0 && i + 1 < argc) {
            const char* backend_text = argv[++i];
            if (!shelf::qwen35::runtime::parse_decode_backend(backend_text,
                                                          &decode_backend)) {
                fprintf(stderr, "bad --decode-backend: %s (expected hip)\n",
                        backend_text);
                return kComputeParseError;
            }
        } else if (std::strcmp(argv[i], "--verify-weights") == 0 && i + 1 < argc) {
            verify_weights = std::atoi(argv[++i]);
            if (verify_weights != 0 && verify_weights != 1) {
                fprintf(stderr, "bad --verify-weights (expected 0 or 1)\n");
                return kComputeParseError;
            }
        } else if (std::strcmp(argv[i], "--dump-logits") == 0 && i + 1 < argc) {
            dump_logits_path = argv[++i];
        } else if (std::strcmp(argv[i], "--hidden-taps") == 0 && i + 1 < argc) {
            hidden_taps = argv[++i];
        } else if (std::strcmp(argv[i], "--tensor-parallel-combine-f32") == 0 && i + 1 < argc) {
            tp_combine_f32 = std::atoi(argv[++i]);
            if (tp_combine_f32 != 0 && tp_combine_f32 != 1) {
                fprintf(stderr, "bad --tensor-parallel-combine-f32 (expected 0 or 1)\n");
                return kComputeParseError;
            }
        } else if (std::strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            if (!parse_float(argv[++i], sampling.temperature)) { fprintf(stderr, "bad --temperature\n"); return kComputeParseError; }
        } else if (std::strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            if (!parse_float(argv[++i], sampling.top_p)) { fprintf(stderr, "bad --top-p\n"); return kComputeParseError; }
        } else if (std::strcmp(argv[i], "--min-p") == 0 && i + 1 < argc) {
            if (!parse_float(argv[++i], sampling.min_p)) { fprintf(stderr, "bad --min-p\n"); return kComputeParseError; }
        } else if (std::strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            if (!parse_uint(argv[++i], top_k)) { fprintf(stderr, "bad --top-k\n"); return kComputeParseError; }
            sampling.top_k = static_cast<uint32_t>(top_k);
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            if (!parse_u64(argv[++i], sampling.seed)) { fprintf(stderr, "bad --seed\n"); return kComputeParseError; }
        } else if (std::strcmp(argv[i], "--dflash2-model-dir") == 0 && i + 1 < argc) {
            dflash2_model_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--dflash2-adaptive") == 0) {
            opt.dflash2_adaptive = true;
        } else if (std::strcmp(argv[i], "--dflash2-drafts") == 0 && i + 1 < argc) {
            if (!parse_uint(argv[++i], dflash2_drafts) || dflash2_drafts == 0) {
                fprintf(stderr, "bad --dflash2-drafts\n");
                return kComputeParseError;
            }
        } else if (std::strcmp(argv[i], "--dflash2-tree-budget") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            opt.dflash2_tree_budget_explicit = true;
            opt.dflash2_tree_budget_auto = std::strcmp(value, "auto") == 0;
            if (opt.dflash2_tree_budget_auto) opt.dflash2_tree_budget = 10;
            if (opt.dflash2_tree || (!opt.dflash2_tree_budget_auto && !parse_uint(value, opt.dflash2_tree_budget)) ||
                (opt.dflash2_tree_budget != 0 && opt.dflash2_tree_budget < 2) || opt.dflash2_tree_budget > 512) {
                fprintf(stderr, "invalid or conflicting --dflash2-tree-budget\n");
                return kComputeParseError;
            }
            opt.dflash2_tree = opt.dflash2_tree_budget != 0;
        } else if (std::strcmp(argv[i], "--dflash2-tree-extra") == 0 && i + 1 < argc) {
            if (opt.dflash2_tree) { fprintf(stderr, "tree options are exclusive\n"); return kComputeParseError; }
            if (!parse_uint(argv[++i], opt.dflash2_tree_extra) ||
                (opt.dflash2_tree_extra != 0 && opt.dflash2_tree_extra != 1 &&
                 opt.dflash2_tree_extra != 2 && opt.dflash2_tree_extra != 4)) {
                fprintf(stderr, "--dflash2-tree-extra expects 0, 1, 2, or 4\n");
                return kComputeParseError;
            }
            opt.dflash2_tree = true;
        } else if (std::strcmp(argv[i], "--dflash2-max-sessions") == 0 && i + 1 < argc) {
            if (!parse_uint(argv[++i], dflash2_max_sessions) || dflash2_max_sessions == 0) {
                fprintf(stderr, "bad --dflash2-max-sessions\n");
                return kComputeParseError;
            }
        } else if (std::strcmp(argv[i], "--dflash2-stats") == 0 && i + 1 < argc) {
            dflash2_stats = std::atoi(argv[++i]);
            if (dflash2_stats != 0 && dflash2_stats != 1) {
                fprintf(stderr, "bad --dflash2-stats (expected 0 or 1)\n");
                return kComputeParseError;
            }
        } else if (std::strcmp(argv[i], "--mtp") == 0) {
            mtp = true;
        } else if (std::strcmp(argv[i], "--mtp-drafts") == 0 && i + 1 < argc) {
            if (!parse_uint(argv[++i], mtp_drafts) || mtp_drafts == 0 ||
                mtp_drafts > shelf::qwen35::runtime::kMtpMaxDrafts) {
                fprintf(stderr, "bad --mtp-drafts (expected 1..%u)\n",
                        shelf::qwen35::runtime::kMtpMaxDrafts);
                return kComputeParseError;
            }
        } else if (std::strcmp(argv[i], "--mtp-stats") == 0 && i + 1 < argc) {
            mtp_stats = std::atoi(argv[++i]);
            if (mtp_stats != 0 && mtp_stats != 1) {
                fprintf(stderr, "bad --mtp-stats (expected 0 or 1)\n");
                return kComputeParseError;
            }
        } else if (std::strcmp(argv[i], "--prefix-cache") == 0 && i + 1 < argc) {
            prefix_cache_enabled = std::atoi(argv[++i]);
            if (prefix_cache_enabled != 0 && prefix_cache_enabled != 1) {
                fprintf(stderr, "bad --prefix-cache (expected 0 or 1)\n");
                return kComputeParseError;
            }
        } else if (std::strcmp(argv[i], "--working-set-reserve-gib") == 0 && i + 1 < argc) {
            if (!parse_uint(argv[++i], working_set_reserve_gib)) {
                fprintf(stderr, "bad --working-set-reserve-gib\n");
                return kComputeParseError;
            }
        } else if (std::strcmp(argv[i], "--prefix-cache-concurrency") == 0 && i + 1 < argc) {
            if (!parse_uint(argv[++i], prefix_cache_concurrency) ||
                prefix_cache_concurrency == 0 || prefix_cache_concurrency > 65535) {
                fprintf(stderr, "bad --prefix-cache-concurrency (expected 1..65535)\n");
                return kComputeParseError;
            }
        } else if (std::strcmp(argv[i], "--serve-stdio") == 0) {
            serve_stdio = true;
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            compute_usage(argv[0]);
            return kComputeParseError;
        }
    }

    if (!dflash2_model_dir.empty() && !opt.dflash2_tree && opt.dflash2_tree_budget == 0 &&
        !opt.dflash2_tree_budget_explicit && dflash2_drafts == 7 && tp == 1 &&
        (serve_stdio || sampling.temperature == 0.0f)) {
        opt.dflash2_tree_budget_auto = true;
        opt.dflash2_tree_budget = 10;
        opt.dflash2_tree = true;
    }
    return kComputeParseOk;
}

int validate_compute_options(const ComputeOptions& opt) {
    const auto& model_dir = opt.model_dir;
    const auto& input_ids_file = opt.input_ids_file;
    const auto& max_new_tokens = opt.max_new_tokens;
    const auto& max_seq_len = opt.max_seq_len;
    const auto& arena_gib = opt.arena_gib;
    const auto& page_tokens = opt.page_tokens;
    const auto& max_concurrent_requests = opt.max_concurrent_requests;
    const auto& devices = opt.devices;
    const auto& tp = opt.tp;
    const auto& pipeline_parallel = opt.pipeline_parallel;
    const auto& kv_cache_dtype = opt.kv_cache_dtype;
    const shelf::qwen35::runtime::DecodeBackend& decode_backend = opt.decode_backend;
    const auto& verify_weights = opt.verify_weights;
    const auto& dump_logits_path = opt.dump_logits_path;
    const auto& serve_stdio = opt.serve_stdio;
    const shelf::qwen35::runtime::SamplingConfig& sampling = opt.sampling;
    const auto& dflash2_model_dir = opt.dflash2_model_dir;
    const auto& dflash2_drafts = opt.dflash2_drafts;
    const auto& dflash2_stats = opt.dflash2_stats;
    const auto& mtp = opt.mtp;
    const auto& mtp_drafts = opt.mtp_drafts;
    const auto& mtp_stats = opt.mtp_stats;
    const auto& prefix_cache_enabled = opt.prefix_cache_enabled;
    const auto& prefix_cache_concurrency = opt.prefix_cache_concurrency;
    const auto& working_set_reserve_gib = opt.working_set_reserve_gib;
    if (opt.dflash2_adaptive && (dflash2_model_dir.empty() || tp != 1)) {
        fprintf(stderr, "--dflash2-adaptive requires --dflash2-model-dir and one device\n");
        return kComputeParseError;
    }
    if (model_dir.empty()) {
        fprintf(stderr, "model-dir required\n");
        return kComputeParseError;
    }
    if (tp != 1 && tp != 2 && tp != 4) {
        fprintf(stderr, "--tensor-parallel must be 1, 2 or 4\n");
        return kComputeParseError;
    }
    if (pipeline_parallel > 1) {
        if (tp != 1) {
            fprintf(stderr, "--pipeline-parallel > 1 requires --tensor-parallel 1\n");
            return kComputeParseError;
        }
        if (devices.size() != pipeline_parallel) {
            fprintf(stderr,
                    "--device expects %lu entries for --pipeline-parallel %lu, got %zu\n",
                    pipeline_parallel, pipeline_parallel, devices.size());
            return kComputeParseError;
        }
        if (!dflash2_model_dir.empty()) {
            fprintf(stderr, "--pipeline-parallel does not support DFlash2 speculative decoding\n");
            return kComputeParseError;
        }
        if (!dump_logits_path.empty()) {
            fprintf(stderr, "--pipeline-parallel does not support --dump-logits\n");
            return kComputeParseError;
        }
        if (!opt.hidden_taps.empty()) {
            fprintf(stderr, "--pipeline-parallel does not support --hidden-taps\n");
            return kComputeParseError;
        }
    } else if (devices.size() != tp) {
        fprintf(stderr, "--device expects %lu entries for --tensor-parallel %lu, got %zu\n",
                tp, tp, devices.size());
        return kComputeParseError;
    }
    {
        std::vector<int> unique_devices = devices;
        std::sort(unique_devices.begin(), unique_devices.end());
        if (std::adjacent_find(unique_devices.begin(), unique_devices.end()) !=
            unique_devices.end()) {
            fprintf(stderr, "--device entries must be unique\n");
            return kComputeParseError;
        }
    }

    const bool dflash_enabled = !dflash2_model_dir.empty();
    if (opt.dflash2_tree_budget_auto && dflash2_drafts != 7) {
        fprintf(stderr, "auto tree budget requires 7 drafts\n");
        return kComputeParseError;
    }
    shelf::kernel::OctaveKVCacheFormat octave_format{};
    const bool octave = shelf::kernel::parse_octave_kv_cache_format(
        kv_cache_dtype, octave_format);
    if (octave && (tp != 1 || pipeline_parallel != 1)) {
        fprintf(stderr,
                "Octave KV cache currently requires TP=1 and PP=1\n");
        return kComputeParseError;
    }
    if (opt.dflash2_tree_budget != 0 && opt.dflash2_tree_budget < dflash2_drafts + 1) {
        fprintf(stderr, "tree budget must include anchor and all draft positions\n");
        return kComputeParseError;
    }
    if (serve_stdio && opt.dflash2_tree_budget > 32) {
        fprintf(stderr, "serve dflash2 tree budget must be at most 32 rows\n");
        return kComputeParseError;
    }
    if (serve_stdio && opt.dflash2_tree_budget > dflash2_drafts + 1 &&
        std::getenv("SHELF_GDN_VERIFY_CHUNKED") != nullptr &&
        (std::getenv("SHELF_GDN_VERIFY_CHUNKED")[0] == '\0' ||
         std::getenv("SHELF_GDN_VERIFY_CHUNKED")[0] == '0')) {
        fprintf(stderr, "serve dflash2 tree requires chunked GDN verify\n");
        return kComputeParseError;
    }
    if (opt.dflash2_tree && (!dflash_enabled || (serve_stdio && opt.dflash2_tree_budget == 0) || tp != 1 || sampling.temperature != 0.0f)) {
        fprintf(stderr, "--dflash2-tree-extra requires single-request DFlash2, TP=1, and greedy sampling\n");
        return kComputeParseError;
    }
    if (dflash_enabled) {
        if (!dump_logits_path.empty()) {
            fprintf(stderr, "dflash2 speculative decoding does not support --dump-logits\n");
            return kComputeParseError;
        }
    }
    if (mtp) {
        if (dflash_enabled) {
            fprintf(stderr, "--mtp and --dflash2-model-dir are mutually exclusive\n");
            return kComputeParseError;
        }
        if (pipeline_parallel != 1) {
            fprintf(stderr, "--mtp currently requires --pipeline-parallel 1\n");
            return kComputeParseError;
        }
        if (!dump_logits_path.empty()) {
            fprintf(stderr, "MTP speculative decoding does not support --dump-logits\n");
            return kComputeParseError;
        }
        if (mtp_drafts > shelf::qwen35::runtime::kMtpMaxDrafts) {
            fprintf(stderr, "--mtp-drafts exceeds %u\n",
                    shelf::qwen35::runtime::kMtpMaxDrafts);
            return kComputeParseError;
        }
    }


    {
        const shelf::Status sampling_status =
            shelf::qwen35::runtime::validate_sampling_config(sampling);
        if (!sampling_status.ok()) {
            fprintf(stderr, "invalid sampling option: %s\n",
                    sampling_status.message().c_str());
            return kComputeParseError;
        }
    }

    return kComputeParseOk;
}

}  // namespace shelf::app
