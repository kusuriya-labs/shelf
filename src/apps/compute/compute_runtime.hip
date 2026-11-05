#include "compute_runtime.h"
#include "private_target_weights.h"

#include <shelf/core/memory/size_format.h>
#include <shelf/runtime/execution/row_bucket.h>
#include <shelf/models/common/qwen_hybrid/stage_range.h>
#include <shelf/models/pipeline_stage_model.h>
#include <shelf/models/qwen35/model/gdn_state_layout.h>
#include <shelf/models/qwen35/runtime/dflash2_footprint.h>
#include <shelf/models/qwen35/runtime/executor_footprint.h>
#include <shelf/models/qwen35/runtime/mtp_drafter.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <utility>

namespace shelf {
namespace app {

namespace {

constexpr std::size_t kAutoArenaMarginBytes = 1ull * 1024ull * 1024ull * 1024ull;
constexpr std::size_t kAutoWorkspaceMinimumBytes = 128ull * 1024ull * 1024ull;

uint32_t ceil_div(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

uint32_t count_attention_layers(const qwen_hybrid::QwenHybridConfig& tc) {
    return static_cast<uint32_t>(qwen_hybrid::count_layer_topology(tc).attention_layers);
}

struct MemoryPlan {
    bool auto_mode = false;
    std::size_t available_bytes = 0;
    std::size_t arena_capacity = 0;
    std::size_t target_weights = 0;
    std::size_t draft_weights = 0;
    std::size_t drafter_runtime = 0;
    std::size_t workspace = 0;
    std::size_t kv = 0;
    std::size_t prefix = 0;
    uint32_t kv_tokens = 0;
    uint32_t prefix_tokens = 0;
    bool prefix_enabled = false;
};

void print_memory_plan(const MemoryPlan& plan) {
    const std::size_t required = plan.target_weights + plan.draft_weights +
        plan.drafter_runtime + plan.workspace + plan.kv + plan.prefix;
    const std::size_t left = plan.arena_capacity > required
        ? plan.arena_capacity - required
        : 0;
    fprintf(stderr,
            "MEMORY_PLAN mode=%s\n"
            "  target weights    %s\n"
            "  draft weights     %s\n"
            "  drafter runtime   %s\n"
            "  runtime workspace %s\n"
            "  KV budget         %s (%u tokens)\n"
            "  prefix cache      %s (%u tokens)\n"
            "  ------------------------------\n"
            "  required          %s\n"
            "  available         %s\n"
            "  arena capacity    %s\n"
            "  left              %s\n",
            plan.auto_mode ? "auto" : "fixed",
            gpu::format_gib(plan.target_weights).c_str(),
            gpu::format_gib(plan.draft_weights).c_str(),
            gpu::format_gib(plan.drafter_runtime).c_str(),
            gpu::format_gib(plan.workspace).c_str(),
            gpu::format_gib(plan.kv).c_str(), plan.kv_tokens,
            plan.prefix_enabled ? gpu::format_gib(plan.prefix).c_str() : "0.00 GiB",
            plan.prefix_tokens,
            gpu::format_gib(required).c_str(),
            gpu::format_gib(plan.available_bytes).c_str(),
            gpu::format_gib(plan.arena_capacity).c_str(),
            gpu::format_gib(left).c_str());
    fflush(stderr);
}

}  // namespace

Result<std::unique_ptr<Qwen35ComputeRuntime>> Qwen35ComputeRuntime::create(
    const Qwen35RuntimeConfig& config) {
    std::unique_ptr<Qwen35ComputeRuntime> runtime(new Qwen35ComputeRuntime());
    Status status = runtime->initialize(config);
    if (!status.ok()) {
        return status;
    }
    return runtime;
}

Qwen35ComputeRuntime::~Qwen35ComputeRuntime() noexcept {
    if (!shutdown_) {
        auto st = shutdown();
        (void)st;
    }
    if (stream_ != nullptr) {
        (void)hipStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

Status Qwen35ComputeRuntime::initialize(const Qwen35RuntimeConfig& config) {
    config_ = config;
    if (config_.max_seq_len == 0) {
        return Status::invalid_argument("max_seq_len must be positive", __FILE__, __LINE__);
    }
    if (config_.max_concurrent_requests == 0) {
        return Status::invalid_argument(
            "max_concurrent_requests must be positive", __FILE__, __LINE__);
    }
    if (config_.tp != 1 && config_.tp != 2) {
        return Status::unsupported("--tensor-parallel supports 1 or 2", __FILE__, __LINE__);
    }
    if (config_.pipeline_parallel == 0) {
        return Status::invalid_argument(
            "pipeline_parallel must be positive", __FILE__, __LINE__);
    }
    if (config_.pipeline_parallel > 1 && config_.tp != 1) {
        return Status::unsupported(
            "pipeline parallel requires tp 1", __FILE__, __LINE__);
    }

    int dev_count = 0;
    if (hipGetDeviceCount(&dev_count) != hipSuccess || dev_count == 0) {
        return Status::invalid_state("no GPU available", __FILE__, __LINE__);
    }
    const std::size_t parallel_width = config_.pipeline_parallel > 1
        ? config_.pipeline_parallel
        : config_.tp;
    if (config_.devices.size() != parallel_width) {
        return Status::invalid_argument(
            "--device count must match --tensor-parallel or --pipeline-parallel", __FILE__, __LINE__);
    }
    for (int device : config_.devices) {
        if (device < 0 || device >= dev_count) {
            return Status::out_of_range("device out of range", __FILE__, __LINE__);
        }
    }

    hipError_t herr = hipSetDevice(config_.devices[0]);
    if (herr != hipSuccess) {
        return Status::hip_error("hipSetDevice", hipGetErrorString(herr), __FILE__, __LINE__);
    }

    const bool auto_arena = config_.arena_bytes == 0;
    auto_arena_ = auto_arena;
    if (auto_arena) {
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        herr = hipMemGetInfo(&free_bytes, &total_bytes);
        if (herr != hipSuccess) {
            return Status::hip_error("hipMemGetInfo", hipGetErrorString(herr), __FILE__,
                                     __LINE__);
        }
        startup_available_bytes_ = free_bytes;
        if (free_bytes == 0) {
            return Status::insufficient_memory(
                "no VRAM available for the arena", __FILE__, __LINE__);
        }
    } else {
        startup_available_bytes_ = config_.arena_bytes;
    }

    exec_max_tokens_ = config_.max_num_batched_tokens;
    if (exec_max_tokens_ == 0) {
        exec_max_tokens_ = config_.max_seq_len;
    }
    const uint32_t row_bucket_max = shelf::runtime::row_bucket_limit(
        shelf::runtime::RowBucket::R8192);
    if (exec_max_tokens_ > row_bucket_max) {
        exec_max_tokens_ = row_bucket_max;
    }
    if (exec_max_tokens_ == 0) {
        exec_max_tokens_ = 1;
    }

    if (config_.pipeline_parallel > 1) {
        return initialize_pipeline_parallel();
    }
    if (config_.tp > 1) {
        return initialize_tensor_parallel();
    }

    herr = hipStreamCreate(&stream_);
    if (herr != hipSuccess) {
        return Status::hip_error("hipStreamCreate", hipGetErrorString(herr), __FILE__, __LINE__);
    }

    if (auto_arena_) {
        const std::size_t free_bytes = startup_available_bytes_;
        const std::size_t available = free_bytes > kAutoArenaMarginBytes
            ? free_bytes - kAutoArenaMarginBytes
            : free_bytes / 2u;
        if (available == 0) {
            return Status::insufficient_memory(
                "no VRAM available for the arena", __FILE__, __LINE__);
        }
        config_.arena_bytes = available;
        fprintf(stderr, "ARENA auto: free=%s capacity=%s\n",
                gpu::format_gib(free_bytes).c_str(),
                gpu::format_gib(config_.arena_bytes).c_str());
        fflush(stderr);
    }
    auto arena_result = shelf::gpu::GpuArena::create(config_.devices[0], config_.arena_bytes);
    if (!arena_result.ok()) {
        return arena_result.status();
    }
    arena_.emplace(arena_result.release());

    const auto t_load_start = std::chrono::steady_clock::now();
    auto model_result = shelf::models::LoadedModel::load(
        config_.model_dir, *arena_, stream_, config_.verify_weights, config_.mtp);
    if (!model_result.ok()) {
        return model_result.status();
    }
    model_ = std::make_unique<shelf::models::LoadedModel>(model_result.release());
    const auto t_load_end = std::chrono::steady_clock::now();
    model_load_ms_ =
        std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();

    const auto& text_config = model_->hybrid_config();
    const qwen_hybrid::StopTokens& generation_stop_tokens = text_config.stop_tokens;
    const std::size_t used_after_weights = arena_->used();
    const uint32_t kv_heads = static_cast<uint32_t>(text_config.num_key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(text_config.attention_head_dim);
    const uint32_t num_attn_layers = count_attention_layers(text_config);
    const auto gdn_layout = qwen35::make_qwen35_gdn_state_layout(text_config);

    const uint32_t max_seq = config_.max_seq_len;
    const uint32_t max_blocks_per_sequence =
        ceil_div(max_seq + 1, config_.page_tokens) + 1;
    const uint32_t max_concurrent = config_.max_concurrent_requests;

    auto slot_pool_result =
        state::SequenceSlotPool::create(*arena_, max_concurrent, max_blocks_per_sequence);
    if (!slot_pool_result.ok()) {
        return slot_pool_result.status();
    }
    slot_pool_.emplace(std::move(slot_pool_result.release()));

    auto gdn_pool_result = state::GdnStatePool::create(*arena_, max_concurrent, gdn_layout);
    if (!gdn_pool_result.ok()) {
        return gdn_pool_result.status();
    }
    gdn_pool_.emplace(std::move(gdn_pool_result.release()));
    const std::size_t used_after_pools = arena_->used();

    qwen35::runtime::PrefixCacheConfig prefix_config;
    prefix_config.hot_capacity_tokens = config_.prefix_cache_hot_tokens;
    prefix_config.cold_capacity_tokens = config_.prefix_cache_cold_tokens;
    prefix_config.max_concurrent = config_.prefix_cache_concurrency;
    prefix_config.cold_state_slots = config_.prefix_cache_cold_state_slots;
    prefix_config.spec_ring_bytes = config_.prefix_cache_spec_ring_bytes;

    const qwen35::runtime::PrefixCacheGeometry prefix_geometry{
        .page_tokens = config_.page_tokens,
        .num_attention_layers = num_attn_layers,
        .kv_heads = kv_heads,
        .head_dim = head_dim,
        .dtype = config_.kv_cache_dtype,
    };
    auto prefix_bytes_for = [&](uint32_t hot_tokens) {
        qwen35::runtime::PrefixCacheConfig candidate = prefix_config;
        candidate.hot_capacity_tokens = hot_tokens;
        return qwen35::runtime::prefix_cache_bytes(candidate, gdn_layout, prefix_geometry);
    };

    qwen35::runtime::PrefixCacheBytes prefix_bytes{};
    if (config_.prefix_cache) {
        prefix_bytes = prefix_bytes_for(config_.prefix_cache_hot_tokens);
    }

    const std::size_t kv_page_bytes = state::PagedKVPool::page_bytes(
        config_.page_tokens, num_attn_layers, kv_heads, head_dim, config_.kv_cache_dtype);
    if (kv_page_bytes == 0u) {
        return Status::invalid_state("KV page geometry is zero", __FILE__, __LINE__);
    }
    const uint32_t required_pages = ceil_div(max_seq + 1u, config_.page_tokens) + 1u;
    const std::size_t kv_required_bytes = required_pages * kv_page_bytes;

    apply_auto_working_set_reserve();
    const std::size_t drafter_total_bytes = drafter_reserve_bytes(*gdn_pool_);
    const std::size_t drafter_runtime_bytes = drafter_total_bytes > config_.draft_bytes
        ? drafter_total_bytes - config_.draft_bytes
        : 0u;

    if (auto_arena) {
        const std::size_t arena_free_now = arena_->capacity() - arena_->used();
        const std::size_t fixed_bytes = kv_required_bytes + drafter_total_bytes;
        const std::size_t available_for_runtime =
            arena_free_now > fixed_bytes ? arena_free_now - fixed_bytes : 0u;
        if (config_.working_set_reserve_bytes > available_for_runtime) {
            if (available_for_runtime < kAutoWorkspaceMinimumBytes) {
                fprintf(stderr,
                        "MEMORY_PLAN auto: %s free after weights, but KV %s + drafter %s + "
                        "minimum workspace %s already exceed it\n",
                        gpu::format_gib(arena_free_now).c_str(),
                        gpu::format_gib(kv_required_bytes).c_str(),
                        gpu::format_gib(drafter_total_bytes).c_str(),
                        gpu::format_gib(kAutoWorkspaceMinimumBytes).c_str());
                fflush(stderr);
                return Status::insufficient_memory(
                    "free VRAM after weights cannot hold the required KV and workspace "
                    "(lower --max-seq-len or --max-concurrent-requests, or disable "
                    "--dflash2-model-dir)",
                    __FILE__, __LINE__);
            }
            fprintf(stderr, "MEMORY_PLAN auto: workspace %s -> %s\n",
                    gpu::format_gib(config_.working_set_reserve_bytes).c_str(),
                    gpu::format_gib(available_for_runtime).c_str());
            fflush(stderr);
            config_.working_set_reserve_bytes = available_for_runtime;
        }
        if (config_.prefix_cache) {
            const std::size_t hot_available =
                available_for_runtime > config_.working_set_reserve_bytes
                    ? available_for_runtime - config_.working_set_reserve_bytes
                    : 0u;
            if (prefix_bytes.hot_total() > hot_available) {
                const uint32_t page_tokens = config_.page_tokens;
                uint32_t lo = 0u;
                uint32_t hi = config_.prefix_cache_hot_tokens / page_tokens;
                uint32_t best_pages = 0u;
                while (lo < hi) {
                    const uint32_t mid = lo + (hi - lo + 1u) / 2u;
                    if (prefix_bytes_for(mid * page_tokens).hot_total() <= hot_available) {
                        lo = mid;
                        best_pages = mid;
                    } else {
                        hi = mid - 1u;
                    }
                }
                if (best_pages >= 1u) {
                    const uint32_t best_tokens = best_pages * page_tokens;
                    fprintf(stderr,
                            "MEMORY_PLAN auto: prefix cache hot tokens %u -> %u (%s available)\n",
                            config_.prefix_cache_hot_tokens, best_tokens,
                            gpu::format_gib(hot_available).c_str());
                    fflush(stderr);
                    config_.prefix_cache_hot_tokens = best_tokens;
                    prefix_config.hot_capacity_tokens = best_tokens;
                    prefix_bytes = prefix_bytes_for(best_tokens);
                } else {
                    fprintf(stderr,
                            "MEMORY_PLAN auto: prefix cache disabled (requested %s, "
                            "available %s)\n",
                            gpu::format_gib(prefix_bytes.hot_total()).c_str(),
                            gpu::format_gib(hot_available).c_str());
                    fflush(stderr);
                    config_.prefix_cache = false;
                    prefix_bytes = qwen35::runtime::PrefixCacheBytes{};
                }
            }
        }
    }

    auto make_plan = [&](std::size_t kv_bytes, uint32_t kv_tokens) {
        MemoryPlan plan;
        plan.auto_mode = auto_arena;
        plan.available_bytes = auto_arena ? startup_available_bytes_ : arena_->capacity();
        plan.arena_capacity = arena_->capacity();
        plan.target_weights = used_after_weights;
        plan.draft_weights = config_.draft_bytes;
        plan.drafter_runtime = drafter_runtime_bytes;
        plan.workspace = config_.working_set_reserve_bytes;
        plan.kv = kv_bytes;
        plan.kv_tokens = kv_tokens;
        plan.prefix = prefix_bytes.hot_total();
        plan.prefix_tokens = config_.prefix_cache
            ? prefix_bytes.hot_pages * config_.page_tokens
            : 0u;
        plan.prefix_enabled = config_.prefix_cache;
        return plan;
    };

    const std::size_t reserved_bytes = prefix_bytes.hot_total() +
        config_.working_set_reserve_bytes + drafter_total_bytes;
    const std::size_t arena_free = arena_->capacity() - arena_->used();
    if (arena_free <= reserved_bytes) {
        print_memory_plan(make_plan(kv_required_bytes, max_seq + 1u));
        return Status::insufficient_memory(
            "arena has " + gpu::format_gib(arena_free) + " free after weights but the runtime "
            "reserves " + gpu::format_gib(reserved_bytes)
            + " for the prefix cache, the working set and the drafter weights (raise "
              "--arena-gib, lower --working-set-reserve-gib or --prefix-cache-hot-tokens)",
            __FILE__, __LINE__);
    }
    const uint32_t pages_fit =
        static_cast<uint32_t>((arena_free - reserved_bytes) / kv_page_bytes);

    if (pages_fit <= 1u) {
        print_memory_plan(make_plan(kv_required_bytes, max_seq + 1u));
        return Status::insufficient_memory(
            "arena budget cannot hold any KV page", __FILE__, __LINE__);
    }
    const uint32_t fit_pages = pages_fit - 1u;
    const uint64_t single_request_tokens = static_cast<uint64_t>(max_seq) + 1u;
    const uint32_t min_context_pages =
        ceil_div(static_cast<uint32_t>(single_request_tokens), config_.page_tokens);
    if (fit_pages < min_context_pages) {
        const uint32_t max_supportable = fit_pages * config_.page_tokens - 1u;
        print_memory_plan(make_plan(kv_required_bytes, max_seq + 1u));
        return Status::insufficient_memory(
            "the arena budget fits " + std::to_string(fit_pages * config_.page_tokens)
            + " KV tokens but max_seq_len " + std::to_string(max_seq)
            + " needs " + std::to_string(single_request_tokens)
            + " (max supportable max_seq_len " + std::to_string(max_supportable)
            + " with --arena-gib "
            + std::to_string(config_.arena_bytes / (1024ull * 1024ull * 1024ull))
            + ")",
            __FILE__, __LINE__);
    }
    const uint32_t physical_capacity_tokens = fit_pages * config_.page_tokens;
    const uint32_t kv_pool_pages = fit_pages + 1u;
    print_memory_plan(make_plan(
        static_cast<std::size_t>(kv_pool_pages) * kv_page_bytes, physical_capacity_tokens));

    auto kv_pool_result = state::PagedKVPool::create(
        *arena_, kv_pool_pages, config_.page_tokens, num_attn_layers, kv_heads, head_dim,
        config_.kv_cache_dtype);
    if (!kv_pool_result.ok()) {
        return kv_pool_result.status();
    }
    kv_pool_.emplace(std::move(kv_pool_result.release()));
    const std::size_t used_after_kv = arena_->used();

    qwen35::ExecutorConfig exec_config;
    exec_config.max_batch_rows = exec_max_tokens_;
    exec_config.max_scheduled_requests = max_concurrent;
    exec_config.max_scheduled_output_rows = config_.max_scheduled_output_rows;
    exec_config.target_hidden_taps = config_.target_hidden_taps;
    exec_config.target_hidden_tap_count = config_.target_hidden_tap_count;
    exec_config.tp_combine_f32 = config_.tp_combine_f32;
    exec_config.backend = config_.decode_backend;
    exec_config.emit_normed_token_hidden = config_.mtp;
    if (config_.rope_profile.has_value()) {
        exec_config.rope_profile = *config_.rope_profile;
    }

    const std::size_t executor_before = arena_->used();
    auto exec_result = model_->create_executor(
        *slot_pool_, *gdn_pool_, *kv_pool_, *arena_, exec_config);
    fprintf(stderr,
            "MEMORY_PLAN executor footprint: %s (reserve %s)\n",
            gpu::format_gib(arena_->used() - executor_before).c_str(),
            gpu::format_gib(config_.working_set_reserve_bytes).c_str());
    fflush(stderr);
    if (!exec_result.ok()) {
        return exec_result.status();
    }
    executor_ = std::move(exec_result.release());

    qwen35::runtime::ContinuousBatcherConfig batcher_config;
    batcher_config.max_num_batched_tokens = exec_max_tokens_;
    batcher_config.max_scheduled_requests = exec_config.max_scheduled_requests;
    batcher_config.max_seq_len = max_seq;
    batcher_config.page_tokens = config_.page_tokens;
    batcher_config.eos_token_ids = text_config.stop_tokens;

    batcher_ = std::make_unique<qwen35::runtime::ContinuousBatcher>(
        executor_, *slot_pool_, *gdn_pool_, *kv_pool_, batcher_config, stream_);

    if (config_.prefix_cache) {
        auto prefix_result = qwen35::runtime::PrefixCache::create(
            *arena_, *kv_pool_, *gdn_pool_, prefix_config);
        if (prefix_result.ok()) {
            prefix_cache_.emplace(prefix_result.release());
            batcher_->set_prefix_cache(&prefix_cache_.value());
        } else {
            prefix_cache_error_ = prefix_result.status().message();
        }
    }

    if (std::getenv("SHELF_BATCH_TRACE") != nullptr) {
        const std::size_t left = arena_->capacity() - arena_->used();
        fprintf(stderr,
                "SHELF_ARENA_PLAN capacity=%zu weights=%zu pools=%zu kv=%zu "
                "kv_tokens=%u prefix_est=%zu rest=%zu left=%zu reserve=%zu\n",
                arena_->capacity(), used_after_weights,
                used_after_pools - used_after_weights, used_after_kv - used_after_pools,
                physical_capacity_tokens, prefix_bytes.hot_total(),
                arena_->used() - used_after_kv, left, config_.working_set_reserve_bytes);
        fflush(stderr);
    }

    return Status::make_ok();
}

std::size_t Qwen35ComputeRuntime::drafter_reserve_bytes(
    const ::shelf::state::GdnStatePool& pool) const {
    if (config_.mtp) {
        qwen35::runtime::MtpDrafterConfig mtp_config;
        mtp_config.num_drafts = config_.mtp_drafts == 0u ? 7u : config_.mtp_drafts;
        mtp_config.max_seq_len = config_.max_seq_len;
        mtp_config.page_tokens = config_.page_tokens;
        mtp_config.max_batch_rows = std::min(
            exec_max_tokens_, qwen35::runtime::kMtpMaxPrefillRows);
        mtp_config.kv_dtype = config_.kv_cache_dtype;
        if (config_.rope_profile.has_value()) {
            mtp_config.rope_profile = *config_.rope_profile;
        }
        if (model_ != nullptr) return model_->mtp_drafter_reserve_bytes(mtp_config);
        return 0u;
    }
    if (!config_.draft_config_ready) {
        return 0u;
    }
    qwen35::runtime::DFlash2SpecDecoderConfig spec_config;
    spec_config.num_drafts = config_.dflash2_drafts == 0u ? 7u : config_.dflash2_drafts;
    std::size_t lm_head_k = 0u;
    if (model_) {
        lm_head_k = static_cast<std::size_t>(model_->lm_head().k_padded);
    }
    return qwen35::runtime::dflash2_footprint(
        config_.draft_config, spec_config, pool, config_.draft_bytes, lm_head_k,
        config_.max_concurrent_requests).total();
}

void Qwen35ComputeRuntime::apply_auto_working_set_reserve() {
    if (config_.working_set_reserve_bytes != 0u) {
        return;
    }
    config_.working_set_reserve_bytes = model_->executor_reserve_bytes(
        exec_max_tokens_, config_.max_concurrent_requests, config_.max_scheduled_output_rows,
        config_.target_hidden_tap_count, config_.tp);
    fprintf(stderr, "MEMORY_PLAN auto: working set reserve %s (row bucket %u)\n",
            gpu::format_gib(config_.working_set_reserve_bytes).c_str(),
            static_cast<unsigned>(exec_max_tokens_));
    fflush(stderr);
}

Status Qwen35ComputeRuntime::initialize_pipeline_parallel() {
    auto config_result = qwen_hybrid::read_qwen_hybrid_config(config_.model_dir);
    if (!config_result.ok()) return config_result.status();
    const qwen_hybrid::QwenHybridConfig& text_config = config_result.value();
    auto builder_result = shelf::models::select_pipeline_stage_model(config_.model_dir);
    if (!builder_result.ok()) return builder_result.status();
    const auto ranges = qwen_hybrid::even_stage_ranges(
        static_cast<std::uint32_t>(text_config.num_hidden_layers),
        static_cast<std::uint32_t>(config_.pipeline_parallel));
    if (ranges.size() != config_.pipeline_parallel) {
        return Status::invalid_argument(
            "--pipeline-parallel must not exceed the model layer count",
            __FILE__, __LINE__);
    }

    qwen35::runtime::PipelineParallelConfig pp_config;
    pp_config.model_dir = config_.model_dir;
    pp_config.devices = config_.devices;
    pp_config.stages = ranges;
    pp_config.max_batch_rows = exec_max_tokens_;
    pp_config.max_scheduled_requests = config_.max_concurrent_requests;
    pp_config.max_scheduled_output_rows = config_.max_scheduled_output_rows;
    pp_config.max_seq_len = config_.max_seq_len;
    pp_config.page_tokens = config_.page_tokens;
    pp_config.kv_cache_dtype = config_.kv_cache_dtype;
    pp_config.arena_bytes = config_.arena_bytes;
    pp_config.working_set_reserve_bytes = config_.working_set_reserve_bytes;
    if (config_.rope_profile.has_value()) {
        pp_config.rope_profile = *config_.rope_profile;
    }
    pp_config.load_options.verify_quantized_payload_crc = config_.verify_weights;
    pp_config.build_stage_model = builder_result.value();

    const auto t_load_start = std::chrono::steady_clock::now();
    auto coordinator_result =
        qwen35::runtime::PipelineParallelCoordinator::create(pp_config);
    if (!coordinator_result.ok()) return coordinator_result.status();
    pp_coordinator_ = coordinator_result.release();
    const auto t_load_end = std::chrono::steady_clock::now();
    model_load_ms_ =
        std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();

    qwen35::runtime::PipelineStageRuntime& stage0 = pp_coordinator_->stage(0);
    pp_model_ = std::make_unique<shelf::models::LoadedModel>(
        stage0.model().dense_model() != nullptr
            ? shelf::models::LoadedModel::borrow(*stage0.model().dense_model())
            : shelf::models::LoadedModel::borrow(*stage0.model().grouped_model()));
    fprintf(stderr,
            "MEMORY_PLAN mode=%s pipeline=%u arena=%s used=%s available=%s\n",
            auto_arena_ ? "auto" : "fixed",
            static_cast<unsigned>(config_.pipeline_parallel),
            gpu::format_gib(stage0.arena().capacity()).c_str(),
            gpu::format_gib(stage0.arena().used()).c_str(),
            gpu::format_gib(startup_available_bytes_).c_str());
    fflush(stderr);
    std::fprintf(stderr, "%s", pp_coordinator_->debug_report().c_str());

    qwen35::runtime::ContinuousBatcherConfig batcher_config;
    batcher_config.max_num_batched_tokens = exec_max_tokens_;
    batcher_config.max_scheduled_requests = config_.max_concurrent_requests;
    batcher_config.max_seq_len = config_.max_seq_len;
    batcher_config.page_tokens = config_.page_tokens;
    batcher_config.eos_token_ids = stage0.model().hybrid_config().stop_tokens;

    batcher_ = std::make_unique<qwen35::runtime::ContinuousBatcher>(
        stage0.executor(), stage0.sequence_pool(), stage0.gdn_pool(), stage0.kv_pool(),
        batcher_config, stage0.stream());
    batcher_->set_tp_batch_hook(pp_coordinator_.get());
    return Status::make_ok();
}

Status Qwen35ComputeRuntime::initialize_tensor_parallel() {
    qwen35::runtime::TpCoordinatorConfig tp_config;
    tp_config.model_dir = config_.model_dir;
    tp_config.devices = config_.devices;
    tp_config.max_batch_rows = exec_max_tokens_;
    tp_config.max_scheduled_requests = config_.max_concurrent_requests;
    tp_config.max_scheduled_output_rows = config_.max_scheduled_output_rows;
    tp_config.max_seq_len = config_.max_seq_len;
    tp_config.page_tokens = config_.page_tokens;
    tp_config.kv_cache_dtype = config_.kv_cache_dtype;
    tp_config.arena_bytes = config_.arena_bytes;
    tp_config.working_set_reserve_bytes = config_.working_set_reserve_bytes;
    tp_config.prefix_cache = config_.prefix_cache;
    tp_config.prefix_hot_tokens = config_.prefix_cache_hot_tokens;
    tp_config.prefix_cold_tokens = config_.prefix_cache_cold_tokens;
    tp_config.prefix_concurrency = config_.prefix_cache_concurrency;
    tp_config.prefix_cold_state_slots = config_.prefix_cache_cold_state_slots;
    tp_config.prefix_spec_ring_bytes = config_.prefix_cache_spec_ring_bytes;
    if (config_.draft_config_ready) {
        tp_config.drafter_config = &config_.draft_config;
        tp_config.drafter_weights_bytes = config_.draft_bytes;
        tp_config.drafter_drafts = config_.dflash2_drafts == 0u ? 7u : config_.dflash2_drafts;
        tp_config.drafter_sessions = config_.max_concurrent_requests;
    }
    if (config_.tp > 1 && config_.private_target_weights) {
        const bool load_lm_head = config_.private_target_lm_head;
        tp_config.load_private_weights =
            [this, load_lm_head](gpu::GpuArena& rank_arena, hipStream_t rank_stream,
                                 weights::MatrixWeight& lm_head,
                                 weights::MatrixWeight& embed_tokens) {
                return load_private_target_weights(
                    config_.model_dir, rank_arena, rank_stream, lm_head, embed_tokens,
                    load_lm_head, true);
            };
    }
    tp_config.target_hidden_taps = config_.target_hidden_taps;
    tp_config.target_hidden_tap_count = config_.target_hidden_tap_count;
    tp_config.tp_combine_f32 = config_.tp_combine_f32;
    if (config_.rope_profile.has_value()) {
        tp_config.rope_profile = *config_.rope_profile;
    }
    tp_config.load_options.verify_quantized_payload_crc = config_.verify_weights;

    const auto t_load_start = std::chrono::steady_clock::now();
    auto group_result = shelf::models::TpGroup::create(tp_config);
    if (!group_result.ok()) {
        return group_result.status();
    }
    tp_group_ = std::make_unique<shelf::models::TpGroup>(group_result.release());
    tp_coordinator_ = &tp_group_->coordinator();
    const auto t_load_end = std::chrono::steady_clock::now();
    model_load_ms_ =
        std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();

    qwen35::runtime::TpRankRuntime& rank0 = tp_coordinator_->rank(0);
    tp_executor_ = &rank0.executor();
    tp_slots_ = &rank0.sequence_pool();
    tp_gdn_ = &rank0.gdn_pool();
    tp_kv_ = &rank0.kv_pool();
    tp_arena_ = &rank0.arena();
    tp_stream_ = rank0.stream();

    fprintf(stderr,
            "MEMORY_PLAN mode=%s tp=2 arena=%s used=%s available=%s\n",
            auto_arena_ ? "auto" : "fixed",
            gpu::format_gib(tp_arena_->capacity()).c_str(),
            gpu::format_gib(tp_arena_->used()).c_str(),
            gpu::format_gib(startup_available_bytes_).c_str());
    fflush(stderr);

    qwen35::runtime::ContinuousBatcherConfig batcher_config;
    batcher_config.max_num_batched_tokens = exec_max_tokens_;
    batcher_config.max_scheduled_requests = config_.max_concurrent_requests;
    batcher_config.max_seq_len = config_.max_seq_len;
    batcher_config.page_tokens = config_.page_tokens;
    batcher_config.eos_token_ids = rank0.hybrid_config().stop_tokens;

    batcher_ = std::make_unique<qwen35::runtime::ContinuousBatcher>(
        *tp_executor_, *tp_slots_, *tp_gdn_, *tp_kv_, batcher_config, tp_stream_);
    batcher_->set_tp_batch_hook(tp_coordinator_);

    if (config_.prefix_cache) {
        qwen35::runtime::PrefixCacheConfig prefix_config;
        prefix_config.hot_capacity_tokens = config_.prefix_cache_hot_tokens;
        prefix_config.cold_capacity_tokens = config_.prefix_cache_cold_tokens;
        prefix_config.max_concurrent = config_.prefix_cache_concurrency;
        prefix_config.cold_state_slots = config_.prefix_cache_cold_state_slots;
        prefix_config.spec_ring_bytes = config_.prefix_cache_spec_ring_bytes;

        const auto& tc = rank0.hybrid_config();
        const qwen35::runtime::PrefixCacheGeometry geometry{
            .page_tokens = config_.page_tokens,
            .num_attention_layers = count_attention_layers(tc),
            .kv_heads = rank0.context().local_key_value_heads,
            .head_dim = static_cast<uint32_t>(tc.attention_head_dim),
            .dtype = config_.kv_cache_dtype,
        };
        const std::size_t arena_free = arena().capacity() - arena().used();
        constexpr std::size_t kDraftRuntimeReserve = 1536ull * 1024ull * 1024ull;
        const std::size_t prefix_budget =
            arena_free > kDraftRuntimeReserve ? arena_free - kDraftRuntimeReserve : 0u;
        auto hot_bytes = [&](uint32_t tokens) {
            auto candidate = prefix_config;
            candidate.hot_capacity_tokens = tokens;
            return qwen35::runtime::prefix_cache_bytes(
                       candidate, rank0.gdn_pool().layout(), geometry)
                .hot_total();
        };
        if (hot_bytes(prefix_config.hot_capacity_tokens) > prefix_budget) {
            const uint32_t page_tokens = config_.page_tokens;
            uint32_t lo = 0u;
            uint32_t hi = config_.prefix_cache_hot_tokens / page_tokens;
            uint32_t best = 0u;
            while (lo < hi) {
                const uint32_t mid = (lo + hi + 1u) / 2u;
                if (hot_bytes(mid * page_tokens) <= prefix_budget) {
                    lo = mid;
                    best = mid;
                } else {
                    hi = mid - 1u;
                }
            }
            if (best == 0u) {
                fprintf(stderr,
                        "MEMORY_PLAN auto: prefix cache disabled (arena free %s, reserve %s)\n",
                        gpu::format_gib(arena_free).c_str(),
                        gpu::format_gib(kDraftRuntimeReserve).c_str());
                fflush(stderr);
                config_.prefix_cache = false;
            } else {
                fprintf(stderr, "MEMORY_PLAN auto: prefix cache hot tokens %u -> %u\n",
                        config_.prefix_cache_hot_tokens, best * page_tokens);
                fflush(stderr);
                prefix_config.hot_capacity_tokens = best * page_tokens;
            }
        }
        std::vector<qwen35::runtime::PrefixCacheRankContext> extra_ranks;
        for (int r = 1; r < tp_coordinator_->tp_size(); ++r) {
            auto& rank = tp_coordinator_->rank(static_cast<std::size_t>(r));
            extra_ranks.push_back(qwen35::runtime::PrefixCacheRankContext{
                &rank.kv_pool(), &rank.gdn_pool(), &rank.arena(), rank.stream(),
                rank.device_id()});
        }
        if (config_.prefix_cache) {
            auto prefix_result = qwen35::runtime::PrefixCache::create(
                arena(), kv_pool(), gdn_pool(), prefix_config, extra_ranks);
            if (prefix_result.ok()) {
                prefix_cache_.emplace(prefix_result.release());
                (void)prefix_cache_->set_tp_hook(tp_coordinator_);
                batcher_->set_prefix_cache(&prefix_cache_.value());
            } else {
                prefix_cache_error_ = prefix_result.status().message();
            }
        }
    }

    return Status::make_ok();
}

Result<uint64_t> Qwen35ComputeRuntime::submit(
    std::vector<int32_t> input_ids,
    uint32_t max_new_tokens,
    const qwen35::runtime::SamplingConfig& sampling) {
    return submit(std::move(input_ids), max_new_tokens, sampling,
                  qwen35::runtime::ThinkingBudget{});
}

Result<uint64_t> Qwen35ComputeRuntime::submit(
    std::vector<int32_t> input_ids,
    uint32_t max_new_tokens,
    const qwen35::runtime::SamplingConfig& sampling,
    const qwen35::runtime::ThinkingBudget& thinking) {
    if (batcher_ == nullptr) {
        return Status::invalid_state("runtime not initialized", __FILE__, __LINE__);
    }
    return batcher_->submit(std::move(input_ids), max_new_tokens, sampling, thinking);
}

Status Qwen35ComputeRuntime::set_dflash2_batcher(
    const qwen35::runtime::DFlash2BatcherBinding& binding) {
    if (batcher_ == nullptr) {
        return Status::invalid_state("runtime not initialized", __FILE__, __LINE__);
    }
    return batcher_->set_dflash2(binding);
}

Result<qwen35::runtime::StepResult> Qwen35ComputeRuntime::step() {
    if (batcher_ == nullptr) {
        return Status::invalid_state("runtime not initialized", __FILE__, __LINE__);
    }
    return batcher_->step();
}

bool Qwen35ComputeRuntime::has_pending() const {
    return batcher_ != nullptr && batcher_->has_pending();
}

qwen35::runtime::KVCapacitySnapshot Qwen35ComputeRuntime::kv_capacity_snapshot() const {
    if (batcher_ == nullptr) {
        return {};
    }
    return batcher_->kv_capacity_snapshot();
}

std::size_t Qwen35ComputeRuntime::queued_request_count() const noexcept {
    return batcher_ == nullptr ? 0 : batcher_->queued_count();
}

const qwen35::runtime::RuntimeRequest* Qwen35ComputeRuntime::request(uint64_t id) const {
    if (batcher_ == nullptr) {
        return nullptr;
    }
    return batcher_->request(id);
}

Status Qwen35ComputeRuntime::cancel_all() {
    if (batcher_ == nullptr) {
        return Status::make_ok();
    }
    return batcher_->cancel_all();
}

Status Qwen35ComputeRuntime::cancel(uint64_t id) {
    if (batcher_ == nullptr) {
        return Status::invalid_state("runtime not initialized", __FILE__, __LINE__);
    }
    return batcher_->cancel(id);
}

Status Qwen35ComputeRuntime::retire(uint64_t id) {
    if (batcher_ == nullptr) {
        return Status::invalid_state("runtime not initialized", __FILE__, __LINE__);
    }
    return batcher_->retire(id);
}

Result<qwen35::dflash2::DFlash2Weights> Qwen35ComputeRuntime::load_dflash2_weights(
    const qwen35::dflash2::DFlash2Config& config,
    const shelf::weights::WeightLoadOptions& options) {
    if (config_.dflash2_model_dir.empty()) {
        return Status::invalid_state("dflash2 model dir missing", __FILE__, __LINE__);
    }
    return qwen35::dflash2::load_dflash2_weights(
        config_.dflash2_model_dir, config, arena(), stream(), options);
}

Status Qwen35ComputeRuntime::shutdown() {
    if (shutdown_) {
        return Status::make_ok();
    }
    Status first_error = Status::make_ok();

    if (batcher_ != nullptr) {
        Status st = batcher_->cancel_all();
        if (!st.ok() && first_error.ok()) first_error = st;
    }

    auto exec_st = (tp_coordinator_ != nullptr || pp_coordinator_ != nullptr)
                       ? Status::make_ok()
                       : qwen35::executor_shutdown(executor_);
    if (!exec_st.ok() && first_error.ok()) first_error = exec_st;

    if (batcher_ != nullptr) {
        batcher_->set_prefix_cache(nullptr);
    }
    if (prefix_cache_.has_value()) {
        Status st = prefix_cache_->shutdown();
        if (!st.ok() && first_error.ok()) first_error = st;
        prefix_cache_.reset();
    }

    if (stream_ != nullptr) {
        hipError_t herr = hipStreamDestroy(stream_);
        if (herr != hipSuccess && first_error.ok()) {
            first_error = Status::hip_error(
                "hipStreamDestroy", hipGetErrorString(herr), __FILE__, __LINE__);
        }
        stream_ = nullptr;
    }

    if (arena_.has_value()) {
        auto arena_st = arena_->shutdown();
        if (!arena_st.ok() && first_error.ok()) first_error = arena_st;
    }

    batcher_.reset();
    tp_executor_ = nullptr;
    tp_slots_ = nullptr;
    tp_gdn_ = nullptr;
    tp_kv_ = nullptr;
    tp_arena_ = nullptr;
    tp_stream_ = nullptr;
    tp_coordinator_ = nullptr;
    tp_group_.reset();
    pp_model_ = nullptr;
    pp_coordinator_.reset();

    shutdown_ = true;
    return first_error;
}

}  // namespace app
}  // namespace shelf
