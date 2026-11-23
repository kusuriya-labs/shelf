#include "single_request.h"

#include "compute_common.h"

#include <shelf/engine/speculation.h>
#include <shelf/models/model_info.h>
#include <shelf/runtime/runtime_request.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace shelf::app {
namespace {

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

void print_model_geometry(const shelf::models::ModelInfo& info) {
    printf("MODEL_HIDDEN_SIZE=%llu\n", static_cast<unsigned long long>(info.hidden_size));
    printf("MODEL_FAMILY=%s\n", info.family_name.c_str());
    if (info.has_intermediate_size) {
        printf("MODEL_INTERMEDIATE_SIZE=%llu\n",
               static_cast<unsigned long long>(info.intermediate_size));
    }
    if (info.has_experts) {
        printf("MODEL_EXPERTS=%llu\n", static_cast<unsigned long long>(info.num_experts));
        printf("MODEL_EXPERTS_PER_TOK=%llu\n",
               static_cast<unsigned long long>(info.num_experts_per_tok));
        printf("MODEL_MOE_INTERMEDIATE_SIZE=%llu\n",
               static_cast<unsigned long long>(info.moe_intermediate_size));
        printf("MODEL_SHARED_EXPERT_INTERMEDIATE_SIZE=%llu\n",
               static_cast<unsigned long long>(info.shared_expert_intermediate_size));
    }
    printf("MODEL_LAYERS=%llu\n", static_cast<unsigned long long>(info.num_hidden_layers));
    printf("MODEL_GDN_LAYERS=%llu\n", static_cast<unsigned long long>(info.gdn_layers));
    printf("MODEL_ATTENTION_LAYERS=%llu\n",
           static_cast<unsigned long long>(info.attention_layers));
    printf("MODEL_Q_HEADS=%llu\n", static_cast<unsigned long long>(info.num_attention_heads));
    printf("MODEL_KV_HEADS=%llu\n", static_cast<unsigned long long>(info.num_key_value_heads));
    printf("MODEL_HEAD_DIM=%llu\n", static_cast<unsigned long long>(info.attention_head_dim));
    printf("MODEL_GDN_KEY_HEADS=%llu\n",
           static_cast<unsigned long long>(info.linear_num_key_heads));
    printf("MODEL_GDN_VALUE_HEADS=%llu\n",
           static_cast<unsigned long long>(info.gdn_num_value_heads));
    printf("MODEL_GDN_KEY_HEAD_DIM=%llu\n",
           static_cast<unsigned long long>(info.gdn_key_head_dim));
    printf("MODEL_GDN_VALUE_HEAD_DIM=%llu\n",
           static_cast<unsigned long long>(info.gdn_value_head_dim));
    printf("MODEL_GDN_CONV_DIM=%llu\n", static_cast<unsigned long long>(info.gdn_conv_dim));
    printf("MODEL_GDN_CONV_HISTORY=%llu\n",
           static_cast<unsigned long long>(info.gdn_conv_history));
    fflush(stdout);
}

void dump_mem_debug(shelf::engine::ComputeSession& runtime) {
    const shelf::models::ModelInfo info = runtime.model_info();
    const shelf::engine::MemoryDebugInfo debug = runtime.memory_debug();
    const shelf::engine::PoolUsage usage = runtime.pool_usage();
    for (std::size_t bucket = 0; bucket < debug.row_bucket_workspace_bytes.size(); ++bucket) {
        printf("ROWBUCKET_%zu_WORKSPACE_BYTES=%zu\n", bucket,
               static_cast<std::size_t>(debug.row_bucket_workspace_bytes[bucket]));
    }
    printf("WEIGHTS_BF16_BYTES=%llu\n",
           static_cast<unsigned long long>(debug.dense_weights.bf16_bytes));
    printf("WEIGHTS_W4_CODES_BYTES=%llu\n",
           static_cast<unsigned long long>(debug.dense_weights.w4_code_bytes));
    printf("WEIGHTS_W4_STORAGE_SCALE_BYTES=%llu\n",
           static_cast<unsigned long long>(debug.dense_weights.w4_storage_scale_bytes));
    printf("WEIGHTS_W4_COMPUTE_SCALE_BYTES=%llu\n",
           static_cast<unsigned long long>(debug.dense_weights.w4_compute_scale_bytes));
    printf("WEIGHTS_W8_CODES_BYTES=%llu\n",
           static_cast<unsigned long long>(debug.dense_weights.w8_code_bytes));
    printf("WEIGHTS_W8_STORAGE_SCALE_BYTES=%llu\n",
           static_cast<unsigned long long>(debug.dense_weights.w8_storage_scale_bytes));
    printf("WEIGHTS_W8_COMPUTE_SCALE_BYTES=%llu\n",
           static_cast<unsigned long long>(debug.dense_weights.w8_compute_scale_bytes));
    printf("WEIGHTS_FP8_CODES_BYTES=%llu\n",
           static_cast<unsigned long long>(debug.dense_weights.fp8_code_bytes));
    printf("WEIGHTS_FP8_SCALE_BYTES=%llu\n",
           static_cast<unsigned long long>(debug.dense_weights.fp8_scale_bytes));
    printf("DISPATCH_STAGING_DEVICE_BYTES=%llu\n",
           static_cast<unsigned long long>(debug.staging_device_bytes));
    printf("DISPATCH_STAGING_HOST_BYTES=%llu\n",
           static_cast<unsigned long long>(debug.staging_host_bytes));
    printf("STAGING_DISPATCH_COUNT=%llu\n",
           static_cast<unsigned long long>(debug.staging_dispatch_count));
    printf("STAGING_MAX_IMAGE_BYTES=%llu\n",
           static_cast<unsigned long long>(debug.staging_max_image_bytes));
    printf("STAGING_SHARED_PROGRAM_BYTES=%llu\n",
           static_cast<unsigned long long>(debug.staging_shared_program_bytes));
    printf("EXECUTION_WORKSPACE_BYTES=%llu\n",
           static_cast<unsigned long long>(debug.execution_workspace_bytes));
    printf("GDN_STATE_BYTES=%llu\n",
           static_cast<unsigned long long>(debug.gdn_state_bytes));
    printf("ARENA_USED_BYTES=%llu\n",
           static_cast<unsigned long long>(usage.arena_used_bytes));
    printf("ARENA_CAPACITY_BYTES=%llu\n",
           static_cast<unsigned long long>(usage.arena_capacity_bytes));
    printf("ARENA_COMMITTED_BYTES=%llu\n",
           static_cast<unsigned long long>(usage.arena_committed_bytes));
    printf("ARENA_VMM=%d\n", usage.arena_vmm ? 1 : 0);
    (void)info;
}

struct LogitsDumper {
    FILE* file = nullptr;
    std::size_t vocab = 0;
    std::vector<std::size_t> step_rows;
    std::vector<float> host;

    bool open(const std::string& path, std::size_t vocab_size) {
        vocab = vocab_size;
        file = std::fopen(path.c_str(), "wb");
        return file != nullptr;
    }

    bool step(shelf::engine::ComputeSession& runtime) {
        if (file == nullptr) return true;
        std::size_t rows = 0;
        const shelf::Status status = runtime.copy_logits(host, rows);
        if (!status.ok()) {
            fprintf(stderr, "dump-logits: %s\n", status.message().c_str());
            return false;
        }
        if (std::fwrite(host.data(), sizeof(float), rows * vocab, file) != rows * vocab) {
            fprintf(stderr, "dump-logits write failed\n");
            return false;
        }
        step_rows.push_back(rows);
        return true;
    }

    void finish() {
        if (file == nullptr) return;
        std::fclose(file);
        file = nullptr;
        std::size_t total_rows = 0;
        printf("LOGITS_VOCAB=%zu\n", vocab);
        printf("LOGITS_STEP_ROWS=");
        for (std::size_t i = 0; i < step_rows.size(); ++i) {
            if (i > 0) printf(",");
            printf("%zu", step_rows[i]);
            total_rows += step_rows[i];
        }
        printf("\n");
        printf("LOGITS_TOTAL_ROWS=%zu\n", total_rows);
    }
};

void dump_hidden_debug(shelf::engine::ComputeSession& runtime, const char* tag) {
    const char* h = std::getenv("SHELF_DEBUG_HIDDEN");
    if (h == nullptr || h[0] == '\0') return;
    for (std::size_t t = 0; t < 2; ++t) {
        std::vector<float> values;
        std::size_t rows = 0;
        std::size_t cols = 0;
        const shelf::Status status = runtime.copy_hidden(t, values, rows, cols);
        if (!status.ok()) {
            fprintf(stderr, "HIDDENDBG %s t=%zu %s\n", tag, t, status.message().c_str());
            continue;
        }
        double mx = 0.0;
        std::size_t nz = 0;
        for (float value : values) {
            const double a = std::abs(value);
            if (a > mx) mx = a;
            if (value != 0.0f) ++nz;
        }
        fprintf(stderr, "HIDDENDBG %s t=%zu rows=%zu cols=%zu max=%.6e nonzero=%zu\n",
                tag, t, rows, cols, mx, nz);
        const char* dbg_file = std::getenv("SHELF_DEBUG_HIDDEN_FILE");
        if (dbg_file != nullptr && dbg_file[0] != '\0') {
            char path[512];
            std::snprintf(path, sizeof(path), "%s/%s_t%zu.bin", dbg_file, tag, t);
            FILE* hf = std::fopen(path, "wb");
            if (hf != nullptr) {
                std::fwrite(values.data(), sizeof(float), values.size(), hf);
                std::fclose(hf);
            }
        }
    }
}

void dump_tap_hidden(shelf::engine::ComputeSession& runtime, const char* tag,
                     std::size_t rows) {
    if (!runtime.has_target_hidden_taps()) return;
    const char* dbg_file = std::getenv("SHELF_DEBUG_HIDDEN_FILE");
    for (std::size_t t = 0;; ++t) {
        std::vector<float> values;
        std::size_t tap_rows = 0;
        std::size_t cols = 0;
        std::size_t layer = 0;
        const shelf::Status status =
            runtime.copy_target_hidden(t, values, tap_rows, cols, layer);
        if (!status.ok()) break;
        const std::size_t use_rows = rows < tap_rows ? rows : tap_rows;
        double mx = 0.0;
        for (std::size_t i = 0; i < use_rows * cols; ++i) {
            const double a = std::abs(values[i]);
            if (a > mx) mx = a;
        }
        fprintf(stderr, "TAPDBG %s layer=%zu rows=%zu cols=%zu max=%.6e\n",
                tag, layer, use_rows, cols, mx);
        if (dbg_file != nullptr && dbg_file[0] != '\0') {
            char path[512];
            std::snprintf(path, sizeof(path), "%s/%s_L%zu.bin", dbg_file, tag, layer);
            FILE* hf = std::fopen(path, "wb");
            if (hf != nullptr) {
                std::fwrite(values.data(), sizeof(float), use_rows * cols, hf);
                std::fclose(hf);
            }
        }
    }
}

void print_speculation_stats(const std::vector<shelf::engine::SpeculationMetric>& stats) {
    for (const auto& metric : stats) {
        const bool space = metric.key == "DFLASH2_ADAPTIVE" ||
                           metric.key == "DFLASH2_ADAPTIVE_K";
        if (space) {
            printf("%s %s\n", metric.key.c_str(), metric.value.c_str());
        } else {
            printf("%s=%s\n", metric.key.c_str(), metric.value.c_str());
        }
    }
    fflush(stdout);
}

void print_adaptive_stats(const std::vector<shelf::engine::SpeculationMetric>& stats) {
    for (const auto& metric : stats) {
        if (metric.key.rfind("DFLASH2_ADAPTIVE", 0) != 0) continue;
        printf("%s %s\n", metric.key.c_str(), metric.value.c_str());
    }
    fflush(stdout);
}

void print_generated_ids(const std::vector<int32_t>& generated) {
    printf("GENERATED_IDS=");
    for (std::size_t i = 0; i < generated.size(); ++i) {
        if (i > 0) printf(",");
        printf("%d", generated[i]);
    }
    printf("\n");
}

bool check_pools(shelf::engine::ComputeSession& runtime) {
    const shelf::engine::PoolUsage usage = runtime.pool_usage();
    if (usage.kv_used_pages != 0 || usage.sequence_used_slots != 0 ||
        usage.gdn_used_slots != 0) {
        fprintf(stderr,
                "pool leak after generation: kv_pages=%u seq_slots=%u gdn_slots=%u\n",
                usage.kv_used_pages, usage.sequence_used_slots, usage.gdn_used_slots);
        return false;
    }
    return true;
}

void print_common_tail(shelf::engine::ComputeSession& runtime,
                       const shelf::models::ModelInfo& info,
                       std::size_t prompt_tokens,
                       const std::vector<int32_t>& generated,
                       uint32_t decode_tokens, double prefill_ms, double decode_ms,
                       const std::string& kv_cache_dtype, bool print_layers) {
    const shelf::engine::PoolUsage usage = runtime.pool_usage();
    const double decode_tps =
        decode_ms > 0.0 ? decode_tokens * 1000.0 / decode_ms : 0.0;
    if (print_layers) {
        printf("MODEL_LAYERS=%llu\n",
               static_cast<unsigned long long>(info.num_hidden_layers));
    }
    printf("PROMPT_TOKENS=%zu\n", prompt_tokens);
    printf("GENERATED_TOKENS=%zu\n", generated.size());
    printf("ARENA_USED_GB=%.3f\n",
           static_cast<double>(usage.arena_used_bytes) / (1024.0 * 1024.0 * 1024.0));
    printf("ARENA_COMMITTED_GB=%.3f\n",
           static_cast<double>(usage.arena_committed_bytes) / (1024.0 * 1024.0 * 1024.0));
    printf("MODEL_LOAD_MS=%.1f\n", runtime.model_load_ms());
    printf("PREFILL_MS=%.1f\n", prefill_ms);
    printf("DECODE_MS=%.1f\n", decode_ms);
    printf("DECODE_TOKENS=%u\n", decode_tokens);
    printf("DECODE_TOKENS_PER_SEC=%.2f\n", decode_tps);
    printf("KV_CACHE_DTYPE=%s\n", kv_cache_dtype.c_str());
    printf("KV_POOL_RESERVED_BYTES=%llu\n",
           static_cast<unsigned long long>(usage.kv_reserved_bytes));
    print_generated_ids(generated);
    (void)info;
}

}  // namespace

int run_single_shot(shelf::engine::ComputeSession& runtime,
                    const std::vector<int32_t>& prompt_ids,
                    unsigned long max_new_tokens,
                    const shelf::runtime::SamplingConfig& sampling,
                    const std::string& kv_cache_dtype,
                    const std::string& dump_logits_path) {
    const shelf::models::ModelInfo info = runtime.model_info();
    print_model_geometry(info);
    if (std::getenv("SHELF_MEM_DEBUG") != nullptr) {
        dump_mem_debug(runtime);
    }

    LogitsDumper dumper;
    if (!dump_logits_path.empty()) {
        if (!dumper.open(dump_logits_path, info.vocab_size)) {
            fprintf(stderr, "cannot open --dump-logits path %s\n", dump_logits_path.c_str());
            return 1;
        }
    }

    auto submit_result = runtime.submit(prompt_ids, static_cast<uint32_t>(max_new_tokens), sampling);
    if (!submit_result.ok()) {
        fprintf(stderr, "submit failed: %s\n", submit_result.status().message().c_str());
        return 1;
    }
    const uint64_t request_id = submit_result.release();

    auto t_prefill_start = std::chrono::steady_clock::now();
    uint32_t prefill_chunks = 0;
    for (;;) {
        const shelf::engine::RequestView prefill_request = runtime.request(request_id);
        if (!prefill_request.found ||
            prefill_request.state == shelf::runtime::RequestState::Finished ||
            prefill_request.sequence_position >= prefill_request.input_tokens_size) {
            break;
        }
        auto prefill_step = runtime.step();
        if (!prefill_step.ok()) {
            fprintf(stderr, "prefill step failed: %s\n",
                    prefill_step.status().message().c_str());
            auto cleanup_st = runtime.cancel_all();
            if (!cleanup_st.ok()) {
                fprintf(stderr, "cleanup after prefill failure: %s\n",
                        cleanup_st.message().c_str());
            }
            return 1;
        }
        if (!prefill_step.value().executed) {
            fprintf(stderr, "prefill step did not execute\n");
            auto cleanup_st = runtime.cancel_all();
            if (!cleanup_st.ok()) {
                fprintf(stderr, "cleanup after prefill failure: %s\n",
                        cleanup_st.message().c_str());
            }
            return 1;
        }
        ++prefill_chunks;
        const shelf::Status sync = runtime.synchronize();
        if (!sync.ok()) {
            fprintf(stderr, "prefill sync failed: %s\n", sync.message().c_str());
            return 1;
        }
        if (dumper.file != nullptr) {
            if (!dumper.step(runtime)) return 1;
        }
    }
    const shelf::Status prefill_sync = runtime.synchronize();
    if (!prefill_sync.ok()) {
        fprintf(stderr, "prefill sync failed: %s\n", prefill_sync.message().c_str());
        return 1;
    }
    if (dumper.file != nullptr) {
        if (!dumper.step(runtime)) return 1;
    }
    dump_hidden_debug(runtime, "prefill");
    dump_tap_hidden(runtime, "prefill", prompt_ids.size());
    auto t_prefill_end = std::chrono::steady_clock::now();

    auto t_decode_start = std::chrono::steady_clock::now();
    uint32_t decode_tokens = 0;
    while (runtime.has_pending()) {
        auto step_result = runtime.step();
        if (!step_result.ok()) {
            fprintf(stderr, "decode step failed: %s\n", step_result.status().message().c_str());
            auto cleanup_st = runtime.cancel_all();
            if (!cleanup_st.ok()) {
                fprintf(stderr, "cleanup after decode failure: %s\n", cleanup_st.message().c_str());
            }
            return 1;
        }
        if (!step_result.value().executed) {
            fprintf(stderr, "decode stuck: pending request remains but step did not execute\n");
            auto cleanup_st = runtime.cancel_all();
            if (!cleanup_st.ok()) {
                fprintf(stderr, "cleanup after stuck decode: %s\n", cleanup_st.message().c_str());
            }
            return 1;
        }
        if (dumper.file != nullptr) {
            const shelf::Status sync = runtime.synchronize();
            if (!sync.ok()) {
                fprintf(stderr, "decode sync failed: %s\n", sync.message().c_str());
                return 1;
            }
            if (!dumper.step(runtime)) return 1;
        }
        if (std::getenv("SHELF_DEBUG_HIDDEN") != nullptr ||
            runtime.has_target_hidden_taps()) {
            const shelf::Status sync = runtime.synchronize();
            if (!sync.ok()) {
                fprintf(stderr, "decode sync failed: %s\n", sync.message().c_str());
                return 1;
            }
            char tag[32];
            std::snprintf(tag, sizeof(tag), "decode%d", static_cast<int>(decode_tokens));
            dump_hidden_debug(runtime, tag);
            dump_tap_hidden(runtime, tag, step_result.value().num_tokens);
        }
        decode_tokens += step_result.value().num_tokens;
    }
    const shelf::Status decode_sync = runtime.synchronize();
    if (!decode_sync.ok()) {
        fprintf(stderr, "decode sync failed: %s\n", decode_sync.message().c_str());
        return 1;
    }
    auto t_decode_end = std::chrono::steady_clock::now();

    const shelf::engine::RequestView request = runtime.request(request_id);
    if (!request.found ||
        request.state != shelf::runtime::RequestState::Finished) {
        fprintf(stderr, "request did not finish\n");
        return 1;
    }
    std::vector<int32_t> generated(request.generated, request.generated + request.generated_size);

    const auto prefill_ms =
        std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();
    const auto decode_ms =
        std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
    const double decode_tps = decode_ms > 0.0 ? decode_tokens * 1000.0 / decode_ms : 0.0;

    printf("MODEL_LAYERS=%llu\n", static_cast<unsigned long long>(info.num_hidden_layers));
    printf("PROMPT_TOKENS=%zu\n", prompt_ids.size());
    printf("GENERATED_TOKENS=%zu\n", generated.size());
    const shelf::engine::PoolUsage usage = runtime.pool_usage();
    printf("ARENA_USED_GB=%.3f\n",
           static_cast<double>(usage.arena_used_bytes) / (1024.0 * 1024.0 * 1024.0));
    printf("ARENA_COMMITTED_GB=%.3f\n",
           static_cast<double>(usage.arena_committed_bytes) / (1024.0 * 1024.0 * 1024.0));
    printf("MODEL_LOAD_MS=%.1f\n", runtime.model_load_ms());
    printf("PREFILL_MS=%.1f\n", prefill_ms);
    printf("PREFILL_CHUNKS=%u\n", prefill_chunks);
    printf("DECODE_MS=%.1f\n", decode_ms);
    printf("DECODE_TOKENS=%u\n", decode_tokens);
    printf("DECODE_TOKENS_PER_SEC=%.2f\n", decode_tps);
    printf("KV_CACHE_DTYPE=%s\n", kv_cache_dtype.c_str());
    printf("KV_POOL_RESERVED_BYTES=%llu\n",
           static_cast<unsigned long long>(usage.kv_reserved_bytes));

    print_generated_ids(generated);
    dumper.finish();
    fflush(stdout);

    if (!check_pools(runtime)) return 1;
    auto shutdown_st = runtime.shutdown();
    if (!shutdown_st.ok()) {
        fprintf(stderr, "runtime shutdown failed: %s\n", shutdown_st.message().c_str());
        return 1;
    }
    return 0;
}

int run_mtp_shot(shelf::engine::ComputeSession& runtime,
                 const std::vector<int32_t>& prompt_ids,
                 unsigned long max_new_tokens,
                 uint32_t mtp_drafts,
                 int mtp_stats,
                 const std::string& kv_cache_dtype,
                 const shelf::runtime::SamplingConfig& sampling) {
    const shelf::models::ModelInfo info = runtime.model_info();
    print_model_geometry(info);
    if (std::getenv("SHELF_MEM_DEBUG") != nullptr) {
        dump_mem_debug(runtime);
    }
    auto result = shelf::engine::SpeculationRunner::run_mtp(
        runtime, prompt_ids, max_new_tokens, mtp_drafts, sampling);
    if (!result.ok()) {
        fprintf(stderr, "mtp failed: %s\n", result.status().message().c_str());
        return 1;
    }
    if (mtp_stats != 0) print_speculation_stats(result.value().stats);
    print_common_tail(runtime, info, prompt_ids.size(), result.value().generated_ids,
                      result.value().decode_tokens, result.value().prefill_ms,
                      result.value().decode_ms, kv_cache_dtype, false);
    fflush(stdout);
    if (!check_pools(runtime)) return 1;
    auto shutdown_st = runtime.shutdown();
    if (!shutdown_st.ok()) {
        fprintf(stderr, "runtime shutdown failed: %s\n", shutdown_st.message().c_str());
        return 1;
    }
    return 0;
}

int run_dflash2_shot(shelf::engine::ComputeSession& runtime,
                     const std::vector<int32_t>& prompt_ids,
                     unsigned long max_new_tokens,
                     const ::shelf::runtime::DFlash2Config& dflash_config,
                     const std::string& dflash2_model_dir,
                     unsigned long dflash2_drafts,
                     int dflash2_stats,
                     bool adaptive,
                     bool dflash2_tree,
                     unsigned long dflash2_tree_extra,
                     unsigned long dflash2_tree_budget,
                     const std::string& kv_cache_dtype,
                     const shelf::runtime::SamplingConfig& sampling) {
    const shelf::models::ModelInfo info = runtime.model_info();
    print_model_geometry(info);
    if (std::getenv("SHELF_MEM_DEBUG") != nullptr) {
        dump_mem_debug(runtime);
    }
    auto result = shelf::engine::SpeculationRunner::run_dflash2(
        runtime, prompt_ids, max_new_tokens, dflash_config, dflash2_drafts, adaptive,
        dflash2_tree, dflash2_tree_extra, dflash2_tree_budget, sampling);
    if (!result.ok()) {
        fprintf(stderr, "dflash2 failed: %s\n", result.status().message().c_str());
        return 1;
    }
    if (dflash2_stats != 0) {
        print_speculation_stats(result.value().stats);
    } else {
        print_adaptive_stats(result.value().stats);
    }
    print_common_tail(runtime, info, prompt_ids.size(), result.value().generated_ids,
                      result.value().decode_tokens, result.value().prefill_ms,
                      result.value().decode_ms, kv_cache_dtype, true);
    fflush(stdout);
    if (!check_pools(runtime)) return 1;
    auto shutdown_st = runtime.shutdown();
    if (!shutdown_st.ok()) {
        fprintf(stderr, "runtime shutdown failed: %s\n", shutdown_st.message().c_str());
        return 1;
    }
    return 0;
}

}  // namespace shelf::app
