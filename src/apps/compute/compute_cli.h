#pragma once

#include <shelf/core/status.h>
#include <shelf/models/qwen35/runtime/decode_backend.h>
#include <shelf/models/qwen35/runtime/mtp_drafter.h>
#include <shelf/models/qwen35/runtime/scheduling_types.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace shelf::app {

struct ComputeOptions {
    std::string model_dir;
    std::string input_ids_file;
    unsigned long max_new_tokens = 32;
    unsigned long max_seq_len = 512;
    unsigned long arena_gib = 16;
    bool arena_gib_auto = true;
    unsigned long page_tokens = 16;
    unsigned long max_concurrent_requests = 4;
    unsigned long max_num_batched_tokens = 2048;
    std::vector<int> devices = {0};
    unsigned long tp = 1;
    unsigned long pipeline_parallel = 1;
    std::string kv_cache_dtype = "bf16";
    shelf::qwen35::runtime::DecodeBackend decode_backend =
        shelf::qwen35::runtime::DecodeBackend::Hip;
    int verify_weights = 0;
    std::string dump_logits_path;
    std::string hidden_taps;
    int tp_combine_f32 = 0;
    bool serve_stdio = false;
    shelf::qwen35::runtime::SamplingConfig sampling;
    std::string dflash2_model_dir;
    unsigned long dflash2_drafts = 7;
    int dflash2_stats = 1;
    bool mtp = false;
    unsigned long mtp_drafts = 7;
    int mtp_stats = 1;
    int prefix_cache_enabled = 1;
    unsigned long prefix_cache_concurrency = 4;
    unsigned long working_set_reserve_gib = 0;
    std::string rope_factor = "auto";
};

inline constexpr int kComputeParseOk = 0;
inline constexpr int kComputeParseHelp = 1;
inline constexpr int kComputeParseError = 2;

inline bool compute_dflash_enabled(const ComputeOptions& opt) {
    return !opt.dflash2_model_dir.empty();
}

void compute_usage(const char* prog);

int parse_compute_options(int argc, char** argv, ComputeOptions& opt);

int validate_compute_options(const ComputeOptions& opt);

std::vector<int32_t> read_input_ids(const std::string& path);

std::size_t safetensors_dir_bytes(const std::string& dir);

}  // namespace shelf::app
