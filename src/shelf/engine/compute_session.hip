#include <shelf/engine/compute_session.h>

#include "compute_session_impl.h"

#include <shelf/core/memory/size_format.h>
#include <shelf/inference/execution/gemm_lt.h>
#include <shelf/inference/state/ple_conv_state_pool.h>
#include <shelf/inference/state/qsa_state_pool.h>
#include <shelf/models/loaded_model.h>
#include <shelf/models/model_policy.h>
#include <shelf/models/pipeline_stage_model.h>
#include <shelf/models/private_target_weights.h>
#include <shelf/models/tp_rank_model.h>
#include <shelf/runtime/continuous_batcher.h>
#include <shelf/runtime/dflash2/dflash2_footprint.h>
#include <shelf/runtime/execution/row_bucket.h>
#include <shelf/runtime/executor.h>
#include <shelf/runtime/gdn_state_layout.h>
#include <shelf/runtime/prefix_cache.h>
#include <shelf/runtime/spec_decode.h>
#include <shelf/runtime/tp/pipeline_parallel.h>
#include <shelf/runtime/tp/tensor_parallel.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <utility>

namespace shelf {
namespace engine {

namespace {

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
    bool prefix_enabled = false;
    bool measured_left = false;
    std::size_t left_bytes = 0;
};

void print_memory_plan(const MemoryPlan& plan, const char* scope = "") {
    const std::size_t required = plan.target_weights + plan.draft_weights +
        plan.drafter_runtime + plan.workspace + plan.kv + plan.prefix;
    const std::size_t derived_left = plan.arena_capacity > required
        ? plan.arena_capacity - required
        : 0;
    const std::size_t left = plan.measured_left ? plan.left_bytes : derived_left;
    fprintf(stderr,
            "MEMORY_PLAN mode=%s%s\n"
            "  target weights    %s\n"
            "  draft weights     %s\n"
            "  drafter runtime   %s\n"
            "  runtime workspace %s\n"
            "  KV budget         %s (%u tokens)\n"
            "  prefix cache      %s\n"
            "  ------------------------------\n"
            "  required          %s\n"
            "  available         %s\n"
            "  arena capacity    %s\n"
            "  left              %s\n",
            plan.auto_mode ? "auto" : "fixed",
            scope,
            gpu::format_gib(plan.target_weights).c_str(),
            gpu::format_gib(plan.draft_weights).c_str(),
            gpu::format_gib(plan.drafter_runtime).c_str(),
            gpu::format_gib(plan.workspace).c_str(),
            gpu::format_gib(plan.kv).c_str(), plan.kv_tokens,
            plan.prefix_enabled ? gpu::format_gib(plan.prefix).c_str() : "0.00 GiB",
            gpu::format_gib(required).c_str(),
            gpu::format_gib(plan.available_bytes).c_str(),
            gpu::format_gib(plan.arena_capacity).c_str(),
            gpu::format_gib(left).c_str());
    fflush(stderr);
}

void print_host_memory(const std::vector<std::size_t>& embeddings,
                       std::size_t ple_bytes, std::size_t offload_bytes,
                       std::size_t prefix_bytes, const char* scope = "",
                       const char* unit = "rank") {
    std::size_t total = prefix_bytes + ple_bytes + offload_bytes;
    for (std::size_t bytes : embeddings) total += bytes;
    if (total == 0u) return;
    fprintf(stderr, "MEMORY_PLAN host memory%s\n", scope);
    if (embeddings.size() <= 1u) {
        const std::size_t embedding = embeddings.empty() ? 0u : embeddings[0];
        fprintf(stderr, "  embedding (pinned)   %s\n",
                gpu::format_gib(embedding).c_str());
    } else {
        for (std::size_t index = 0; index < embeddings.size(); ++index) {
            fprintf(stderr, "  embedding %s %zu     %s\n", unit, index,
                    gpu::format_gib(embeddings[index]).c_str());
        }
    }
    if (ple_bytes != 0u) {
        fprintf(stderr, "  PLE table            %s\n", gpu::format_gib(ple_bytes).c_str());
    }
    if (offload_bytes != 0u) {
        fprintf(stderr, "  offload experts      %s\n",
                gpu::format_gib(offload_bytes).c_str());
    }
    if (prefix_bytes != 0u) {
        fprintf(stderr, "  prefix checkpoint    %s\n",
                gpu::format_gib(prefix_bytes).c_str());
    }
    fprintf(stderr, "  total                %s\n", gpu::format_gib(total).c_str());
    fflush(stderr);
}

}  // namespace


Result<std::unique_ptr<ComputeSession>> ComputeSession::create(
    const ComputeConfig& config) {
    std::unique_ptr<ComputeSession> session(new ComputeSession());
    Status status = session->initialize(config);
    if (!status.ok()) {
        return status;
    }
    return session;
}

ComputeSession::~ComputeSession() noexcept {
    if (impl_ != nullptr && !impl_->shutdown_) {
        auto st = shutdown();
        (void)st;
    }
}

Status ComputeSession::initialize(const ComputeConfig& config) {
    impl_.reset(new Impl());
    Impl& state = *impl_;
    state.config_ = config;
    if (!state.config_.resolved_ready) {
        auto resolved = models::resolve_model_options(state.config_.requested);
        if (!resolved.ok()) return resolved.status();
        state.config_.resolved = resolved.release();
        state.config_.resolved_ready = true;
    }
    for (const auto& notice : state.config_.resolved.notices) {
        std::fprintf(stderr, "%s\n", notice.c_str());
    }
    state.config_.dflash2_host_token_ids = state.config_.resolved.dflash2_host_token_ids;
    state.config_.private_target_weights = state.config_.resolved.private_target_weights;
    state.config_.private_target_lm_head = state.config_.resolved.private_target_lm_head;
    state.config_.dflash2_drafts = state.config_.resolved.dflash2_drafts;
    if (state.config_.resolved.has_rope_profile) {
        state.config_.rope_profile = state.config_.resolved.rope_profile;
    }
    const Status policy = models::validate_model_runtime_options(
        state.config_.requested, state.config_.resolved);
    if (!policy.ok()) return policy;

    if (state.config_.max_seq_len == 0) {
        return Status::invalid_argument("max_seq_len must be positive", __FILE__, __LINE__);
    }
    if (state.config_.max_concurrent_requests == 0) {
        return Status::invalid_argument(
            "max_concurrent_requests must be positive", __FILE__, __LINE__);
    }
    if (state.config_.tp != 1 && state.config_.tp != 2 && state.config_.tp != 4) {
        return Status::unsupported("--tensor-parallel supports 1, 2 or 4", __FILE__, __LINE__);
    }
    if (state.config_.pipeline_parallel == 0) {
        return Status::invalid_argument(
            "pipeline_parallel must be positive", __FILE__, __LINE__);
    }
    if (state.config_.mtp &&
        state.config_.kv_cache_dtype == kernel::KVCacheDType::OCTAVE)
        std::fprintf(stderr,
                     "MTP with octave KV: the drafter runs with BF16 KV "
                     "(the target keeps octave)\n");

    int dev_count = 0;
    if (hipGetDeviceCount(&dev_count) != hipSuccess || dev_count == 0) {
        return Status::invalid_state("no GPU available", __FILE__, __LINE__);
    }
    for (int device : state.config_.devices) {
        if (device < 0 || device >= dev_count) {
            return Status::out_of_range("device out of range", __FILE__, __LINE__);
        }
    }
    runtime::ParallelShape topology_shape;
    topology_shape.pp = state.config_.pipeline_parallel;
    topology_shape.ep = state.config_.expert_parallel;
    topology_shape.tp = state.config_.tp;
    auto topology_result =
        runtime::ParallelTopology::create(topology_shape, state.config_.devices);
    if (!topology_result.ok()) {
        return Status::invalid_argument(
            (std::string("--device count must match pipeline_parallel * expert_parallel * "
                         "tensor_parallel: ") + topology_result.status().message()).c_str(),
            __FILE__, __LINE__);
    }
    state.topology_.emplace(topology_result.release());

    hipError_t herr = hipSetDevice(state.config_.devices[0]);
    if (herr != hipSuccess) {
        return Status::hip_error("hipSetDevice", hipGetErrorString(herr), __FILE__, __LINE__);
    }

    const bool auto_arena = state.config_.arena_mode == ArenaBudgetMode::Auto;
    state.auto_arena_ = auto_arena;
    std::size_t memory_capacity = state.config_.arena_bytes;
    if (state.config_.arena_mode == ArenaBudgetMode::Fixed) {
        if (state.config_.arena_bytes == 0u) {
            return Status::insufficient_memory(
                "arena capacity is zero", __FILE__, __LINE__);
        }
        fprintf(stderr, "ARENA fixed: capacity=%s (includes 0.3 GiB external overhead adjustment)\n",
                gpu::format_gib(memory_capacity).c_str());
    } else if (!auto_arena) {
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        herr = hipMemGetInfo(&free_bytes, &total_bytes);
        if (herr != hipSuccess) {
            return Status::hip_error("hipMemGetInfo", hipGetErrorString(herr), __FILE__,
                                     __LINE__);
        }
        const std::size_t margin = state.config_.arena_bytes;
        if (margin >= total_bytes) {
            return Status::invalid_argument(
                "--arena-gib margin including 0.3 GiB overhead must be less than total VRAM", __FILE__, __LINE__);
        }
        memory_capacity = total_bytes - margin;
        fprintf(stderr, "ARENA total-minus: total=%s margin=%s capacity=%s\n",
                gpu::format_gib(total_bytes).c_str(),
                gpu::format_gib(margin).c_str(),
                gpu::format_gib(memory_capacity).c_str());
    }

    std::vector<int> memory_devices;
    if (state.config_.pipeline_parallel > 1u || state.config_.expert_parallel > 1u ||
        state.config_.tp > 1u) {
        memory_devices = state.config_.devices;
    } else {
        memory_devices.push_back(state.config_.devices[0]);
    }
    auto memory_result = gpu::GpuMemoryManager::create(memory_devices, memory_capacity);
    if (!memory_result.ok()) return memory_result.status();
    state.memory_.emplace(memory_result.release());
    for (int device : memory_devices) {
        Status prepare_status = shelf::execution::prepare_gemm_lt(device);
        if (!prepare_status.ok()) return prepare_status;
    }
    auto arena0 = state.memory_->arena_for(state.config_.devices[0]);
    if (!arena0.ok()) return arena0.status();
    state.startup_available_bytes_ = arena0.value()->capacity();
    fflush(stderr);

    state.exec_max_tokens_ = state.config_.max_num_batched_tokens;
    if (state.exec_max_tokens_ == 0) {
        state.exec_max_tokens_ = state.config_.max_seq_len;
    }
    const uint32_t row_bucket_max =
        runtime::row_bucket_limit(runtime::RowBucket::R8192);
    if (state.exec_max_tokens_ > row_bucket_max) {
        state.exec_max_tokens_ = row_bucket_max;
    }
    if (state.exec_max_tokens_ == 0) {
        state.exec_max_tokens_ = 1;
    }

    if (state.config_.pipeline_parallel > 1) {
        return initialize_pipeline_parallel();
    }
    if (state.config_.tp > 1 || state.config_.expert_parallel > 1u) {
        return initialize_tensor_parallel();
    }

    auto arena_ptr = state.memory_->arena_for(state.config_.devices[0]);
    if (!arena_ptr.ok()) return arena_ptr.status();
    state.arena_ = arena_ptr.value();
    auto stream_ptr = state.memory_->stream_for(state.config_.devices[0]);
    if (!stream_ptr.ok()) return stream_ptr.status();
    state.stream_ = stream_ptr.value();

    const auto t_load_start = std::chrono::steady_clock::now();
    auto model_result = models::LoadedModel::load(
        state.config_.model_dir, *state.arena_, state.stream_, state.config_.verify_weights,
        state.config_.mtp);
    if (!model_result.ok()) {
        return model_result.status();
    }
    state.model_ = std::make_unique<models::LoadedModel>(model_result.release());
    const auto t_load_end = std::chrono::steady_clock::now();
    state.model_load_ms_ =
        std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();

    const auto& text_config = state.model_->hybrid_config();
    const std::size_t used_after_weights = state.arena_->used();
    const uint32_t kv_heads = static_cast<uint32_t>(text_config.num_key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(text_config.attention_head_dim);
    const uint32_t num_attn_layers = count_attention_layers(text_config);
    const auto gdn_layout = runtime::make_gdn_state_layout(text_config);

    const uint32_t max_seq = state.config_.max_seq_len;
    const uint32_t max_blocks_per_sequence =
        ceil_div(max_seq + 1, state.config_.page_tokens) + 1;
    const uint32_t max_concurrent = state.config_.max_concurrent_requests;

    auto slot_pool_result =
        state::SequenceSlotPool::create(*state.arena_, max_concurrent, max_blocks_per_sequence);
    if (!slot_pool_result.ok()) {
        return slot_pool_result.status();
    }
    state.slot_pool_.emplace(slot_pool_result.release());

    auto gdn_pool_result = state::GdnStatePool::create(*state.arena_, max_concurrent, gdn_layout);
    if (!gdn_pool_result.ok()) {
        return gdn_pool_result.status();
    }
    state.gdn_pool_.emplace(gdn_pool_result.release());
    const std::size_t used_after_pools = state.arena_->used();

    runtime::PrefixCacheConfig prefix_config;
    prefix_config.checkpoints = state.config_.prefix_cache_checkpoints;
    prefix_config.capture_slots = state.config_.max_concurrent_requests;
    prefix_config.spec_ring_bytes = state.config_.prefix_cache_spec_ring_bytes;
    prefix_config.strategy = state.config_.prefix_cache_strategy;
    prefix_config.slot_count = state.config_.prefix_cache_slot_count;
    prefix_config.save_interval = state.config_.prefix_cache_save_interval;
    if (prefix_config.strategy == runtime::PrefixCacheStrategy::Slot) {
        prefix_config.checkpoints = prefix_config.slot_count + 1u;
        prefix_config.capture_slots = 0u;
    }

    auto prefix_bytes_for =
        [&](const shelf::state::QsaKeyStatePoolLayout* qsa,
            const shelf::state::PleConvStatePoolLayout* ple) {
            return runtime::prefix_cache_bytes(prefix_config, gdn_layout, qsa, ple);
        };

    runtime::PrefixCacheBytes prefix_bytes{};
    std::size_t prefix_capture_bytes = 0u;
    if (state.config_.prefix_cache) {
        prefix_bytes = prefix_bytes_for(nullptr, nullptr);
        prefix_capture_bytes = state.model_->prefix_capture_state_bytes(
            prefix_config.capture_slots, max_seq + 1u);
    }

    const std::size_t kv_page_bytes = state::PagedKVPool::page_bytes(
        state.config_.page_tokens, num_attn_layers, kv_heads, head_dim,
        state.config_.kv_cache_dtype, state.config_.octave_kv_format);
    if (kv_page_bytes == 0u) {
        return Status::invalid_state("KV page geometry is zero", __FILE__, __LINE__);
    }
    const uint32_t required_pages = ceil_div(max_seq + 1u, state.config_.page_tokens) + 1u;
    const std::size_t kv_required_bytes = required_pages * kv_page_bytes;

    apply_auto_working_set_reserve();
    const std::size_t drafter_total_bytes = drafter_reserve_bytes(*state.gdn_pool_);
    std::size_t drafter_runtime_bytes = drafter_total_bytes > state.config_.draft_bytes
        ? drafter_total_bytes - state.config_.draft_bytes
        : 0u;

    if (auto_arena) {
        const std::size_t arena_free_now = state.arena_->capacity() - state.arena_->used();
        const std::size_t fixed_bytes = kv_required_bytes + drafter_total_bytes;
        const std::size_t available_for_runtime =
            arena_free_now > fixed_bytes ? arena_free_now - fixed_bytes : 0u;
        if (state.config_.working_set_reserve_bytes > available_for_runtime) {
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
                    gpu::format_gib(state.config_.working_set_reserve_bytes).c_str(),
                    gpu::format_gib(available_for_runtime).c_str());
            fflush(stderr);
            state.config_.working_set_reserve_bytes = available_for_runtime;
        }
    }

    auto make_plan = [&](std::size_t kv_bytes, uint32_t kv_tokens) {
        MemoryPlan plan;
        plan.auto_mode = auto_arena;
        plan.available_bytes = auto_arena ? state.startup_available_bytes_ : state.arena_->capacity();
        plan.arena_capacity = state.arena_->capacity();
        plan.target_weights = used_after_weights;
        plan.draft_weights = state.config_.draft_bytes;
        plan.drafter_runtime = drafter_runtime_bytes;
        plan.workspace = state.config_.working_set_reserve_bytes;
        plan.kv = kv_bytes;
        plan.kv_tokens = kv_tokens;
        plan.prefix = prefix_bytes.total() + prefix_capture_bytes;
        plan.prefix_enabled = state.config_.prefix_cache;
        return plan;
    };

    auto create_kv_pool = [&](std::size_t pages) -> Status {
        state.kv_pool_ = std::nullopt;
        auto result = state::PagedKVPool::create(
            *state.arena_, static_cast<std::uint32_t>(pages), state.config_.page_tokens,
            num_attn_layers, kv_heads, head_dim, state.config_.kv_cache_dtype,
            state.config_.octave_kv_format);
        if (!result.ok()) return result.status();
        state.kv_pool_.emplace(result.release());
        return Status::make_ok();
    };

    runtime::ExecutorConfig exec_config;
    exec_config.max_batch_rows = state.exec_max_tokens_;
    exec_config.max_scheduled_requests = max_concurrent;
    exec_config.max_scheduled_output_rows = state.config_.max_scheduled_output_rows;
    exec_config.target_hidden_taps = state.config_.target_hidden_taps;
    exec_config.target_hidden_tap_count = state.config_.target_hidden_tap_count;
    exec_config.tp_combine_f32 = state.config_.tp_combine_f32;
    exec_config.max_context_tokens = state.config_.max_seq_len + 1u;
    exec_config.backend = state.config_.decode_backend;
    exec_config.emit_normed_token_hidden = state.config_.mtp;
    if (exec_config.emit_normed_token_hidden) {
        const std::uint32_t width = state.model_->mtp_token_hidden_width();
        if (width != 0u) exec_config.normed_token_hidden_width = width;
    }
    if (state.config_.rope_profile.has_value()) {
        exec_config.rope_profile = *state.config_.rope_profile;
    }

    auto create_executor_now = [&]() -> Status {
        auto result = state.model_->create_executor(
            *state.slot_pool_, *state.gdn_pool_, *state.kv_pool_, *state.arena_, exec_config);
        if (!result.ok()) return result.status();
        state.executor_ = result.release();
        return Status::make_ok();
    };

    Status probe_status = create_kv_pool(1u);
    if (!probe_status.ok()) return probe_status;
    const std::size_t executor_before = state.arena_->used();
    probe_status = create_executor_now();
    if (!probe_status.ok()) return probe_status;
    const std::size_t executor_bytes = state.arena_->used() - executor_before;
    if (executor_bytes > state.config_.working_set_reserve_bytes) {
        state.config_.working_set_reserve_bytes = executor_bytes;
    }
    fprintf(stderr,
            "MEMORY_PLAN executor footprint: %s (reserve %s)\n",
            gpu::format_gib(executor_bytes).c_str(),
            gpu::format_gib(state.config_.working_set_reserve_bytes).c_str());
    fflush(stderr);

    if (state.config_.prefix_cache) {
        prefix_bytes = prefix_bytes_for(
            state.executor_.qsa_state_pool != nullptr ? &state.executor_.qsa_state_pool->layout() : nullptr,
            state.executor_.ple_conv_state_pool != nullptr
                ? &state.executor_.ple_conv_state_pool->layout()
                : nullptr);
    }

    const std::size_t arena_free = state.arena_->capacity() - state.arena_->used();
    const std::size_t base_reserve =
        prefix_bytes.total() + prefix_capture_bytes + drafter_total_bytes;
    const char* coarse = std::getenv("SHELF_DFLASH2_COARSE_HEAD");
    if (state.config_.draft_config_ready && state.config_.tp == 1 &&
        (coarse == nullptr || std::strcmp(coarse, "0") != 0)) {
        const auto bytes = runtime::dflash2_coarse_head_bytes(state.model_->lm_head());
        if (arena_free >= base_reserve + kv_required_bytes + bytes) {
            state.config_.dflash2_coarse_bytes = bytes;
            drafter_runtime_bytes += bytes;
        }
    }
    const std::size_t reserved_bytes = base_reserve + state.config_.dflash2_coarse_bytes;
    if (arena_free <= reserved_bytes) {
        print_memory_plan(make_plan(kv_required_bytes, max_seq + 1u));
        return Status::insufficient_memory(
            "arena has " + gpu::format_gib(arena_free) + " free after weights but the runtime "
            "reserves " + gpu::format_gib(reserved_bytes)
            + " for the prefix state and the drafter weights (raise "
              "--arena-gib or lower --max-concurrent-requests)",
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
        ceil_div(static_cast<uint32_t>(single_request_tokens), state.config_.page_tokens);
    if (fit_pages < min_context_pages) {
        const uint32_t max_supportable = fit_pages * state.config_.page_tokens - 1u;
        print_memory_plan(make_plan(kv_required_bytes, max_seq + 1u));
        return Status::insufficient_memory(
            "the arena budget fits " + std::to_string(fit_pages * state.config_.page_tokens)
            + " KV tokens but max_seq_len " + std::to_string(max_seq)
            + " needs " + std::to_string(single_request_tokens)
            + " (max supportable max_seq_len " + std::to_string(max_supportable)
            + " with --arena-gib "
            + std::to_string(state.startup_available_bytes_ / (1024ull * 1024ull * 1024ull))
            + ")",
            __FILE__, __LINE__);
    }
    const uint32_t physical_capacity_tokens = fit_pages * state.config_.page_tokens;
    const uint32_t kv_pool_pages = fit_pages + 1u;

    Status kv_status = create_kv_pool(kv_pool_pages);
    if (!kv_status.ok()) return kv_status;
    state.executor_.bind_kv_pool(*state.kv_pool_);
    const std::size_t used_after_kv = state.arena_->used();

    runtime::ContinuousBatcherConfig batcher_config;
    batcher_config.max_num_batched_tokens = state.exec_max_tokens_;
    batcher_config.max_scheduled_requests = exec_config.max_scheduled_requests;
    batcher_config.max_seq_len = max_seq;
    batcher_config.page_tokens = state.config_.page_tokens;
    batcher_config.eos_token_ids = text_config.stop_tokens;
    batcher_config.ngram = state.config_.ngram;

    state.batcher_ = std::make_unique<runtime::ContinuousBatcher>(
        state.executor_, *state.slot_pool_, *state.gdn_pool_, *state.kv_pool_, batcher_config,
        state.stream_);
    if (state.config_.ngram.enabled && !state.config_.draft_config_ready && !state.config_.mtp) {
        const Status ngram_bind = state.batcher_->set_ngram_verify(state.arena_);
        if (!ngram_bind.ok()) return ngram_bind;
    }

    if (state.config_.prefix_cache) {
        auto bindings_res = runtime::make_prefix_state_bindings(state.executor_);
        if (!bindings_res.ok()) return bindings_res.status();
        auto prefix_result = runtime::PrefixCache::create(
            *state.arena_, bindings_res.value(), prefix_config);
        if (!prefix_result.ok())
            return prefix_result.status();
        state.prefix_cache_.emplace(prefix_result.release());
        state.batcher_->set_prefix_cache(&state.prefix_cache_.value());
    }

    MemoryPlan final_plan = make_plan(
        static_cast<std::size_t>(kv_pool_pages) * kv_page_bytes, physical_capacity_tokens);
    final_plan.measured_left = true;
    final_plan.left_bytes = state.arena_->capacity() - state.arena_->used();
    print_memory_plan(final_plan);

    const models::HostMemoryBytes host_bytes = state.model_->host_memory_bytes();
    print_host_memory({static_cast<std::size_t>(host_bytes.embedding)},
                      static_cast<std::size_t>(host_bytes.ple_table),
                      static_cast<std::size_t>(host_bytes.offload_experts),
                      state.prefix_cache_.has_value() ? state.prefix_cache_->reserved_bytes() : 0u);

    if (std::getenv("SHELF_BATCH_TRACE") != nullptr) {
        const std::size_t left = state.arena_->capacity() - state.arena_->used();
        fprintf(stderr,
                "SHELF_ARENA_PLAN capacity=%zu weights=%zu pools=%zu kv=%zu "
                "kv_tokens=%u prefix_est=%zu rest=%zu left=%zu reserve=%zu\n",
                state.arena_->capacity(), used_after_weights,
                used_after_pools - used_after_weights, used_after_kv - used_after_pools,
                physical_capacity_tokens, prefix_bytes.total() + prefix_capture_bytes,
                state.arena_->used() - used_after_kv, left,
                state.config_.working_set_reserve_bytes);
        fflush(stderr);
    }

    return Status::make_ok();
}

std::size_t ComputeSession::drafter_reserve_bytes(
    const ::shelf::state::GdnStatePool& pool) const {
    const Impl& state = *impl_;
    if (state.config_.mtp) {
        models::MtpReserveRequest mtp_request;
        mtp_request.num_drafts =
            state.config_.mtp_drafts == 0u ? 7u : state.config_.mtp_drafts;
        mtp_request.max_seq_len = state.config_.max_seq_len;
        mtp_request.page_tokens = state.config_.page_tokens;
        mtp_request.max_batch_rows = std::min(
            state.exec_max_tokens_, models::max_mtp_prefill_rows());
        mtp_request.kv_dtype =
            state.config_.kv_cache_dtype == kernel::KVCacheDType::OCTAVE
                ? kernel::KVCacheDType::BF16 : state.config_.kv_cache_dtype;
        if (state.config_.rope_profile.has_value()) {
            mtp_request.rope_profile = *state.config_.rope_profile;
        }
        if (state.model_ == nullptr) return 0u;
        return models::mtp_drafter_reserve_bytes(
            *state.model_, mtp_request, state.config_.mtp_serve,
            mtp_batch_requests(mtp_request.num_drafts));
    }
    if (!state.config_.draft_config_ready) {
        if (state.config_.ngram.enabled && pool.num_gdn_states() > 0u &&
            pool.device_view().recurrent_base != nullptr && runtime::supports_ngram_history(pool.device_view())) {
            const uint32_t max_reqs = std::max(1u, state.config_.max_concurrent_requests);
            const uint32_t ngram_k =
                state.config_.ngram.num_drafts != 0u ? state.config_.ngram.num_drafts : 7u;
            const uint32_t rows = runtime::standalone_ngram_max_rows(max_reqs, ngram_k);
            return static_cast<std::size_t>(rows) *
                (runtime::spec_gdn_conv_bytes(pool) +
                 runtime::spec_gdn_recurrent_bytes(pool)) +
                4096u;
        }
        return 0u;
    }
    runtime::DFlash2SpecDecoderConfig spec_config;
    spec_config.num_drafts = state.config_.dflash2_drafts == 0u ? 7u : state.config_.dflash2_drafts;
    spec_config.tree = state.config_.dflash2_tree;
    spec_config.tree_extra = state.config_.dflash2_tree_extra;
    spec_config.tree_budget = state.config_.dflash2_tree_budget;
    std::size_t lm_head_k = 0u;
    bool exl3_mul24_head = false;
    if (state.model_) {
        lm_head_k = static_cast<std::size_t>(state.model_->lm_head().k_padded);
        exl3_mul24_head = state.model_->lm_head().quant_spec ==
            quantization::QuantSpecId::EXL3_MUL24_INT8_V1;
    }
    return runtime::dflash2_footprint(
        state.config_.draft_config, spec_config, pool, state.config_.draft_bytes, lm_head_k,
        state.config_.dflash2_sessions(), state.config_.dflash2_prefill_rows,
        exl3_mul24_head).total();
}

std::size_t ComputeSession::mtp_batch_requests(uint32_t num_drafts) const {
    const Impl& state = *impl_;
    const uint32_t verify_rows = num_drafts + 1u;
    uint32_t rows = std::min(state.exec_max_tokens_, models::max_mtp_prefill_rows());
    uint32_t slots = state.config_.max_concurrent_requests;
    if (slots == 0u) slots = 1u;
    if (verify_rows == 0u) return slots;
    if (rows / verify_rows < slots) slots = rows / verify_rows;
    return slots == 0u ? 1u : slots;
}

void ComputeSession::apply_auto_working_set_reserve() {
    Impl& state = *impl_;
    if (state.config_.working_set_reserve_bytes != 0u) {
        return;
    }
    state.config_.working_set_reserve_bytes = state.model_->executor_reserve_bytes(
        state.exec_max_tokens_, state.config_.max_concurrent_requests,
        state.config_.max_scheduled_output_rows, state.config_.target_hidden_tap_count,
        state.config_.tp, static_cast<std::size_t>(state.config_.max_seq_len) + 1u);
    fprintf(stderr, "MEMORY_PLAN auto: working set reserve %s (row bucket %u)\n",
            gpu::format_gib(state.config_.working_set_reserve_bytes).c_str(),
            static_cast<unsigned>(state.exec_max_tokens_));
    fflush(stderr);
}

Status ComputeSession::initialize_pipeline_parallel() {
    Impl& state = *impl_;
    auto config_result = qwen_hybrid::read_qwen_hybrid_config(state.config_.model_dir);
    if (!config_result.ok()) return config_result.status();
    const qwen_hybrid::QwenHybridConfig& text_config = config_result.value();
    auto builder_result = models::select_pipeline_stage_model(state.config_.model_dir);
    if (!builder_result.ok()) return builder_result.status();
    const auto ranges = qwen_hybrid::even_stage_ranges(
        static_cast<std::uint32_t>(text_config.num_hidden_layers),
        static_cast<std::uint32_t>(state.config_.pipeline_parallel));
    if (ranges.size() != state.config_.pipeline_parallel) {
        return Status::invalid_argument(
            "--pipeline-parallel must not exceed the model layer count",
            __FILE__, __LINE__);
    }

    runtime::PipelineParallelConfig pp_config;
    pp_config.model_dir = state.config_.model_dir;
    std::vector<int> stage_devices;
    stage_devices.reserve(state.config_.pipeline_parallel);
    for (uint32_t stage = 0; stage < state.config_.pipeline_parallel; ++stage) {
        const uint32_t rank = state.topology_->rank(runtime::ParallelCoord{stage, 0u, 0u});
        stage_devices.push_back(state.topology_->device(rank));
    }
    pp_config.devices = std::move(stage_devices);
    pp_config.stages = ranges;
    pp_config.max_batch_rows = state.exec_max_tokens_;
    pp_config.max_scheduled_requests = state.config_.max_concurrent_requests;
    pp_config.max_scheduled_output_rows = state.config_.max_scheduled_output_rows;
    pp_config.max_seq_len = state.config_.max_seq_len;
    pp_config.page_tokens = state.config_.page_tokens;
    pp_config.pipeline_hidden_width = models::pipeline_hidden_width(
        state.config_.model_dir, state.config_.resolved.model_class);
    pp_config.kv_cache_dtype = state.config_.kv_cache_dtype;
    pp_config.octave_kv_format = state.config_.octave_kv_format;
    pp_config.memory = &*state.memory_;
    pp_config.working_set_reserve_bytes = state.config_.working_set_reserve_bytes;
    if (state.config_.rope_profile.has_value()) {
        pp_config.rope_profile = *state.config_.rope_profile;
    }
    pp_config.load_options.verify_quantized_payload_crc = state.config_.verify_weights;
    pp_config.build_stage_model = builder_result.value();

    if (state.config_.tp > 1u || state.config_.expert_parallel > 1u) {
        pp_config.stage_tp.resize(state.config_.pipeline_parallel);
        for (std::uint32_t stage = 0; stage < state.config_.pipeline_parallel; ++stage) {
            runtime::TpCoordinatorConfig tp;
            tp.model_dir = state.config_.model_dir;
            tp.pipeline_hidden_width = pp_config.pipeline_hidden_width;
            tp.tensor_parallel = state.config_.tp;
            tp.expert_parallel = state.config_.expert_parallel;
            const std::vector<std::uint32_t> stage_ranks = state.topology_->stage_ranks(stage);
            tp.devices.reserve(stage_ranks.size());
            for (std::uint32_t rank : stage_ranks) {
                tp.devices.push_back(state.topology_->device(rank));
            }
            tp.max_batch_rows = state.exec_max_tokens_;
            tp.max_scheduled_requests = state.config_.max_concurrent_requests;
            tp.max_scheduled_output_rows = state.config_.max_scheduled_output_rows;
            tp.max_seq_len = state.config_.max_seq_len;
            tp.page_tokens = state.config_.page_tokens;
            tp.kv_cache_dtype = state.config_.kv_cache_dtype;
            tp.octave_kv_format = state.config_.octave_kv_format;
            tp.memory = &*state.memory_;
            tp.working_set_reserve_bytes = state.config_.working_set_reserve_bytes;
            tp.prefix_cache = state.config_.prefix_cache;
            tp.prefix_checkpoints = state.config_.prefix_cache_checkpoints;
            tp.prefix_spec_ring_bytes = state.config_.prefix_cache_spec_ring_bytes;
            tp.prefix_strategy = state.config_.prefix_cache_strategy;
            tp.prefix_slot_count = state.config_.prefix_cache_slot_count;
            tp.prefix_save_interval = state.config_.prefix_cache_save_interval;
            tp.tp_combine_f32 = state.config_.tp_combine_f32;
            tp.load_options.verify_quantized_payload_crc = state.config_.verify_weights;
            tp.pipeline_stage.enabled = true;
            tp.pipeline_stage.layers = ranges[stage];
            if (const char* env = std::getenv("SHELF_KV_POOL_PAGES")) {
                tp.kv_pool_pages = static_cast<std::uint32_t>(std::strtoul(env, nullptr, 10));
            }
            if (state.config_.rope_profile.has_value()) {
                tp.rope_profile = *state.config_.rope_profile;
            }
            Status selected = models::select_tp_rank_model(tp);
            if (!selected.ok()) return selected;
            pp_config.stage_tp[stage] = std::move(tp);
        }
    }

    const auto t_load_start = std::chrono::steady_clock::now();
    auto coordinator_result =
        runtime::PipelineParallelCoordinator::create(pp_config);
    if (!coordinator_result.ok()) return coordinator_result.status();
    state.pp_coordinator_ = coordinator_result.release();
    const auto t_load_end = std::chrono::steady_clock::now();
    state.model_load_ms_ =
        std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();

    runtime::PipelineStageRuntime& stage0 = state.pp_coordinator_->stage(0);
    const shelf::models::TpRankModel* tp_rank_model =
        stage0.tp() != nullptr ? stage0.tp()->rank(0).rank_model() : nullptr;
    state.pp_model_ = std::make_unique<models::LoadedModel>(
        tp_rank_model != nullptr
            ? models::LoadedModel::borrow_rank_model(*tp_rank_model)
            : models::LoadedModel::borrow_stage_model(stage0.model()));
    fprintf(stderr,
            "MEMORY_PLAN mode=%s pipeline=%u arena=%s used=%s available=%s\n",
            state.auto_arena_ ? "auto" : "fixed",
            static_cast<unsigned>(state.config_.pipeline_parallel),
            gpu::format_gib(stage0.arena().capacity()).c_str(),
            gpu::format_gib(stage0.arena().used()).c_str(),
            gpu::format_gib(state.startup_available_bytes_).c_str());
    fflush(stderr);
    std::fprintf(stderr, "%s", state.pp_coordinator_->debug_report().c_str());

    std::vector<std::size_t> host_embeddings;
    host_embeddings.reserve(state.pp_coordinator_->stage_count());
    for (std::size_t index = 0; index < state.pp_coordinator_->stage_count(); ++index) {
        host_embeddings.push_back(state.pp_coordinator_->stage(index).host_pinned_bytes());
    }
    char host_scope[32];
    std::snprintf(host_scope, sizeof(host_scope), " pipeline=%u",
                  static_cast<unsigned>(state.config_.pipeline_parallel));
    print_host_memory(host_embeddings, 0u, 0u, 0u, host_scope, "stage");

    runtime::ContinuousBatcherConfig batcher_config;
    batcher_config.max_num_batched_tokens = state.exec_max_tokens_;
    batcher_config.max_scheduled_requests = state.config_.max_concurrent_requests;
    batcher_config.max_seq_len = state.config_.max_seq_len;
    batcher_config.page_tokens = state.config_.page_tokens;
    batcher_config.eos_token_ids = stage0.hybrid_config().stop_tokens;
    batcher_config.ngram = state.config_.ngram;

    state.batcher_ = std::make_unique<runtime::ContinuousBatcher>(
        stage0.executor(), stage0.sequence_pool(), stage0.gdn_pool(), stage0.kv_pool(),
        batcher_config, stage0.stream());
    state.batcher_->set_tp_batch_hook(state.pp_coordinator_.get());
    return Status::make_ok();
}

Status ComputeSession::initialize_tensor_parallel() {
    Impl& state = *impl_;
    runtime::TpCoordinatorConfig tp_config;
    tp_config.model_dir = state.config_.model_dir;
    tp_config.devices = state.config_.devices;
    tp_config.tensor_parallel = state.config_.tp;
    tp_config.expert_parallel = state.config_.expert_parallel;
    tp_config.max_batch_rows = state.exec_max_tokens_;
    tp_config.max_scheduled_requests = state.config_.max_concurrent_requests;
    tp_config.max_scheduled_output_rows = state.config_.max_scheduled_output_rows;
    tp_config.max_seq_len = state.config_.max_seq_len;
    tp_config.page_tokens = state.config_.page_tokens;
    tp_config.kv_cache_dtype = state.config_.kv_cache_dtype;
    tp_config.octave_kv_format = state.config_.octave_kv_format;
    tp_config.memory = &*state.memory_;
    tp_config.working_set_reserve_bytes = state.config_.working_set_reserve_bytes;
    tp_config.prefix_cache = state.config_.prefix_cache;
    tp_config.prefix_checkpoints = state.config_.prefix_cache_checkpoints;
    tp_config.prefix_spec_ring_bytes = state.config_.prefix_cache_spec_ring_bytes;
    tp_config.prefix_strategy = state.config_.prefix_cache_strategy;
    tp_config.prefix_slot_count = state.config_.prefix_cache_slot_count;
    tp_config.prefix_save_interval = state.config_.prefix_cache_save_interval;
    if (state.config_.draft_config_ready) {
        tp_config.drafter_config = &state.config_.draft_config;
        tp_config.drafter_footprint = [](const shelf::runtime::DFlash2Config& cfg,
                                         const shelf::runtime::DFlash2SpecDecoderConfig& spec,
                                         shelf::state::GdnStatePool& pool,
                                         std::size_t weights, std::size_t lm_head_k,
                                         uint32_t sessions, std::size_t prefill_rows,
                                         bool exl3_mul24_head) {
            return runtime::dflash2_footprint(
                cfg, spec, pool, weights, lm_head_k, sessions, prefill_rows,
                exl3_mul24_head);
        };
        tp_config.drafter_weights_bytes = state.config_.draft_bytes;
        tp_config.drafter_drafts =
            state.config_.dflash2_drafts == 0u ? 7u : state.config_.dflash2_drafts;
        tp_config.drafter_sessions = state.config_.dflash2_sessions();
    }
    if (state.config_.tp > 1 && state.config_.private_target_weights) {
        const bool load_lm_head = state.config_.private_target_lm_head;
        const std::string model_dir = state.config_.model_dir;
        tp_config.load_private_weights =
            [model_dir, load_lm_head](gpu::GpuArena& rank_arena, hipStream_t rank_stream,
                                      weights::MatrixWeight& lm_head,
                                      weights::MatrixWeight& embed_tokens) {
                return models::load_private_target_weights(
                    model_dir, rank_arena, rank_stream, lm_head, embed_tokens,
                    load_lm_head, true);
            };
    }
    tp_config.target_hidden_taps = state.config_.target_hidden_taps;
    tp_config.target_hidden_tap_count = state.config_.target_hidden_tap_count;
    tp_config.tp_combine_f32 = state.config_.tp_combine_f32;
    tp_config.emit_normed_token_hidden = state.config_.mtp;
    if (state.config_.rope_profile.has_value()) {
        tp_config.rope_profile = *state.config_.rope_profile;
    }
    tp_config.load_options.verify_quantized_payload_crc = state.config_.verify_weights;
    tp_config.load_mtp_layers = state.config_.mtp;
    tp_config.mtp_split = state.config_.mtp && state.config_.mtp_serve && state.config_.tp > 1u;
    if (state.config_.mtp) {
        const uint32_t num_drafts =
            state.config_.mtp_drafts == 0u ? 7u : state.config_.mtp_drafts;
        const uint32_t max_batch_rows = std::min(
            state.exec_max_tokens_, models::max_mtp_prefill_rows());
        tp_config.mtp_batch.enabled = true;
        tp_config.mtp_batch.num_drafts = num_drafts;
        tp_config.mtp_batch.max_seq_len = state.config_.max_seq_len;
        tp_config.mtp_batch.page_tokens = state.config_.page_tokens;
        tp_config.mtp_batch.max_batch_rows = max_batch_rows;
        tp_config.mtp_batch.kv_dtype =
            state.config_.kv_cache_dtype == kernel::KVCacheDType::OCTAVE
                ? kernel::KVCacheDType::BF16 : state.config_.kv_cache_dtype;
        tp_config.mtp_batch.max_requests =
            state.config_.mtp_serve ? mtp_batch_requests(num_drafts) : 1u;
        tp_config.mtp_reserve = [](const shelf::models::TpRankModel& model,
                                   const shelf::qwen_hybrid::QwenHybridConfig& tc,
                                   const runtime::MtpBatchReserveConfig& cfg,
                                   const shelf::state::GdnStatePoolLayout& layout,
                                   uint32_t tp_rank) {
            return models::mtp_batch_reserve_bytes(model, tc, cfg, layout, tp_rank);
        };
    }

    const auto t_load_start = std::chrono::steady_clock::now();
    auto group_result = models::TpGroup::create(tp_config);
    if (!group_result.ok()) {
        return group_result.status();
    }
    state.tp_group_ = std::make_unique<models::TpGroup>(group_result.release());
    state.tp_coordinator_ = &state.tp_group_->coordinator();
    const auto t_load_end = std::chrono::steady_clock::now();
    state.model_load_ms_ =
        std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();

    runtime::TpRankRuntime& rank0 = state.tp_coordinator_->rank(0);
    state.tp_executor_ = &rank0.executor();
    state.tp_slots_ = &rank0.sequence_pool();
    state.tp_gdn_ = &rank0.gdn_pool();
    state.tp_kv_ = &rank0.kv_pool();
    state.tp_arena_ = &rank0.arena();
    state.tp_stream_ = rank0.stream();

    for (int r = 0; r < state.tp_coordinator_->tp_size(); ++r) {
        runtime::TpRankRuntime& rank =
            state.tp_coordinator_->rank(static_cast<std::size_t>(r));
        const runtime::TpRankMemoryPlan& rank_plan = rank.memory_plan();
        MemoryPlan plan;
        plan.auto_mode = state.auto_arena_;
        plan.available_bytes = state.startup_available_bytes_;
        plan.arena_capacity = rank.arena().capacity();
        plan.target_weights = rank_plan.target_weights;
        plan.draft_weights = rank_plan.draft_weights;
        plan.drafter_runtime = rank_plan.drafter_runtime;
        plan.workspace = rank_plan.workspace;
        plan.kv = rank_plan.kv;
        plan.kv_tokens = rank_plan.kv_tokens;
        plan.prefix = rank_plan.prefix;
        plan.prefix_enabled = state.config_.prefix_cache;
        char scope[32];
        std::snprintf(scope, sizeof(scope), " tp=%d rank=%d", state.config_.tp, r);
        print_memory_plan(plan, scope);
    }

    runtime::ContinuousBatcherConfig batcher_config;
    batcher_config.max_num_batched_tokens = state.exec_max_tokens_;
    batcher_config.max_scheduled_requests = state.config_.max_concurrent_requests;
    batcher_config.max_seq_len = state.config_.max_seq_len;
    batcher_config.page_tokens = state.config_.page_tokens;
    batcher_config.eos_token_ids = rank0.hybrid_config().stop_tokens;
    batcher_config.ngram = state.config_.ngram;

    state.batcher_ = std::make_unique<runtime::ContinuousBatcher>(
        *state.tp_executor_, *state.tp_slots_, *state.tp_gdn_, *state.tp_kv_, batcher_config,
        state.tp_stream_);
    state.batcher_->set_tp_batch_hook(state.tp_coordinator_);
    if (state.config_.ngram.enabled && !state.config_.draft_config_ready && !state.config_.mtp) {
        std::vector<runtime::NgramVerifyRankBinding> rank_bindings;
        rank_bindings.reserve(static_cast<std::size_t>(state.tp_coordinator_->tp_size()));
        for (int r = 0; r < state.tp_coordinator_->tp_size(); ++r) {
            auto& rank = state.tp_coordinator_->rank(static_cast<std::size_t>(r));
            rank_bindings.push_back(runtime::NgramVerifyRankBinding{
                &rank.gdn_pool(), &rank.arena(), rank.stream(), rank.device_id()});
        }
        const Status ngram_bind = state.batcher_->set_ngram_verify(&state.arena(), &rank_bindings);
        if (!ngram_bind.ok()) return ngram_bind;
    }

    if (state.config_.prefix_cache) {
        runtime::PrefixCacheConfig prefix_config;
        prefix_config.checkpoints = state.config_.prefix_cache_checkpoints;
        prefix_config.capture_slots = state.config_.max_concurrent_requests;
        prefix_config.spec_ring_bytes = state.config_.prefix_cache_spec_ring_bytes;
        prefix_config.strategy = state.config_.prefix_cache_strategy;
        prefix_config.slot_count = state.config_.prefix_cache_slot_count;
        prefix_config.save_interval = state.config_.prefix_cache_save_interval;
        if (prefix_config.strategy == runtime::PrefixCacheStrategy::Slot) {
            prefix_config.checkpoints = prefix_config.slot_count + 1u;
            prefix_config.capture_slots = 0u;
        }

        std::vector<runtime::PrefixCacheRankContext> extra_ranks;
        for (int r = 1; r < state.tp_coordinator_->tp_size(); ++r) {
            auto& rank = state.tp_coordinator_->rank(static_cast<std::size_t>(r));
            extra_ranks.push_back(runtime::PrefixCacheRankContext{
                &rank.kv_pool(), &rank.gdn_pool(), &rank.arena(), rank.stream(),
                rank.device_id(), rank.executor().qsa_state_pool.get(),
                rank.executor().ple_conv_state_pool.get()});
        }
        auto bindings_res = runtime::make_prefix_state_bindings(*state.tp_executor_);
        if (!bindings_res.ok()) return bindings_res.status();
        auto prefix_result = runtime::PrefixCache::create(
            state.arena(), bindings_res.value(), prefix_config, extra_ranks);
        if (!prefix_result.ok())
            return prefix_result.status();
        state.prefix_cache_.emplace(prefix_result.release());
        (void)state.prefix_cache_->set_tp_hook(state.tp_coordinator_);
        state.batcher_->set_prefix_cache(&state.prefix_cache_.value());
    }

    std::vector<std::size_t> host_embeddings;
    host_embeddings.reserve(static_cast<std::size_t>(state.tp_coordinator_->tp_size()));
    std::uint64_t host_ple = 0u;
    std::uint64_t host_offload = 0u;
    for (int r = 0; r < state.tp_coordinator_->tp_size(); ++r) {
        const shelf::models::TpRankModel* rank_model =
            state.tp_coordinator_->rank(static_cast<std::size_t>(r)).rank_model();
        host_embeddings.push_back(rank_model != nullptr ? rank_model->host_pinned_bytes() : 0u);
        if (rank_model == nullptr) continue;
        host_offload += rank_model->host_offload_expert_bytes();
        if (r == 0) host_ple = rank_model->host_ple_table_bytes();
    }
    char host_scope[32];
    std::snprintf(host_scope, sizeof(host_scope), " tp=%d", state.config_.tp);
    print_host_memory(host_embeddings, static_cast<std::size_t>(host_ple),
                      static_cast<std::size_t>(host_offload),
                      state.prefix_cache_.has_value() ? state.prefix_cache_->reserved_bytes() : 0u,
                      host_scope);

    return Status::make_ok();
}

Result<uint64_t> ComputeSession::submit(
    std::vector<int32_t> input_ids,
    uint32_t max_new_tokens,
    const runtime::SamplingConfig& sampling) {
    return submit(std::move(input_ids), max_new_tokens, sampling,
                  runtime::ThinkingBudget{});
}

Result<uint64_t> ComputeSession::submit(
    std::vector<int32_t> input_ids,
    uint32_t max_new_tokens,
    const runtime::SamplingConfig& sampling,
    const runtime::ThinkingBudget& thinking) {
    Impl& state = *impl_;
    if (state.batcher_ == nullptr) {
        return Status::invalid_state("runtime not initialized", __FILE__, __LINE__);
    }
    if (state.config_.dflash2_tree_budget_auto && sampling.temperature != 0.0f) {
        return state.batcher_->submit(std::move(input_ids), max_new_tokens, sampling, false,
                                      thinking);
    }
    return state.batcher_->submit(std::move(input_ids), max_new_tokens, sampling, thinking);
}

Result<uint64_t> ComputeSession::submit_unconstrained_decode(
    std::vector<int32_t> input_ids, uint32_t max_new_tokens,
    const runtime::SamplingConfig& sampling,
    const runtime::ThinkingBudget& thinking) {
    Impl& state = *impl_;
    if (state.batcher_ == nullptr) {
        return Status::invalid_state("runtime not initialized", __FILE__, __LINE__);
    }
    return state.batcher_->submit(std::move(input_ids), max_new_tokens, sampling, false,
                                  thinking);
}

Status ComputeSession::set_token_bitmask(uint64_t id, const uint32_t* mask,
                                         bool waiting) {
    Impl& state = *impl_;
    if (state.batcher_ == nullptr) {
        return Status::invalid_state("runtime not initialized", __FILE__, __LINE__);
    }
    return state.batcher_->set_token_bitmask(id, mask, waiting);
}

Result<uint64_t> ComputeSession::submit_structured_dflash(
    std::vector<int32_t> input_ids, uint32_t max_new_tokens,
    const runtime::SamplingConfig& sampling,
    const runtime::ThinkingBudget& thinking) {
    Impl& state = *impl_;
    if (state.batcher_ == nullptr) {
        return Status::invalid_state("runtime not initialized", __FILE__, __LINE__);
    }
    return state.batcher_->submit(std::move(input_ids), max_new_tokens, sampling, true,
                                  thinking);
}

Status ComputeSession::set_structured_dflash(uint64_t id, bool delayed,
                                             int32_t close_token) {
    return impl_->batcher_->set_structured_dflash(id, delayed, close_token);
}

Status ComputeSession::activate_grammar(uint64_t id) {
    return impl_->batcher_->activate_grammar(id);
}

Status ComputeSession::set_draft_masks(uint64_t id, uint64_t round, const uint32_t* mask,
                                       uint32_t stride, uint32_t valid_prefix) {
    return impl_->batcher_->set_draft_masks(id, round, mask, stride, valid_prefix);
}

Result<runtime::StepResult> ComputeSession::step() {
    Impl& state = *impl_;
    if (state.batcher_ == nullptr) {
        return Status::invalid_state("runtime not initialized", __FILE__, __LINE__);
    }
    return state.batcher_->step();
}

bool ComputeSession::has_pending() const {
    return impl_->batcher_ != nullptr && impl_->batcher_->has_pending();
}

bool ComputeSession::has_active_requests() const {
    if (impl_->batcher_ == nullptr) return false;
    return impl_->batcher_->active_count() != 0u || impl_->batcher_->queued_count() != 0u;
}

RequestView ComputeSession::request(uint64_t id) const {
    RequestView view;
    if (impl_->batcher_ == nullptr) return view;
    const runtime::RuntimeRequest* request = impl_->batcher_->request(id);
    if (request == nullptr) return view;
    view.found = true;
    view.state = request->state;
    view.finish_reason = request->finish_reason;
    view.generated = request->generated.data();
    view.generated_size = request->generated.size();
    view.input_tokens = request->input_tokens.data();
    view.input_tokens_size = request->input_tokens.size();
    view.sequence_position = request->sequence.position;
    view.restored_tokens = request->restored_tokens;
    view.generated_tokens = request->generated_tokens;
    view.dflash_used = request->dflash_used;
    view.structured_dflash = request->structured_dflash;
    view.draft_mask_state = request->draft_mask_state;
    view.draft_round_id = request->draft_round_id;
    view.pending_drafts = request->pending_drafts.data();
    view.pending_drafts_size = request->pending_drafts.size();
    view.prefill_ms = request->prefill_ms;
    view.error_message = request->error_message.c_str();
    return view;
}

Status ComputeSession::set_external_request_id(uint64_t id,
                                               int64_t external_request_id) {
    Impl& state = *impl_;
    if (state.batcher_ == nullptr) {
        return Status::invalid_state("runtime not initialized", __FILE__, __LINE__);
    }
    return state.batcher_->set_external_request_id(id, external_request_id);
}

Status ComputeSession::cancel_all() {
    if (impl_->batcher_ == nullptr) {
        return Status::make_ok();
    }
    return impl_->batcher_->cancel_all();
}

Status ComputeSession::cancel(uint64_t id) {
    Impl& state = *impl_;
    if (state.batcher_ == nullptr) {
        return Status::invalid_state("runtime not initialized", __FILE__, __LINE__);
    }
    return state.batcher_->cancel(id);
}

Status ComputeSession::retire(uint64_t id) {
    Impl& state = *impl_;
    if (state.batcher_ == nullptr) {
        return Status::invalid_state("runtime not initialized", __FILE__, __LINE__);
    }
    return state.batcher_->retire(id);
}

Status ComputeSession::shutdown() {
    Impl& state = *impl_;
    if (state.shutdown_) {
        return Status::make_ok();
    }
    Status first_error = Status::make_ok();

    if (state.batcher_ != nullptr) {
        Status st = state.batcher_->cancel_all();
        if (!st.ok() && first_error.ok()) first_error = st;
    }

    auto exec_st = (state.tp_coordinator_ != nullptr || state.pp_coordinator_ != nullptr)
                       ? Status::make_ok()
                       : runtime::executor_shutdown(state.executor_);
    if (!exec_st.ok() && first_error.ok()) first_error = exec_st;

    if (state.batcher_ != nullptr) {
        state.batcher_->set_prefix_cache(nullptr);
    }
    if (state.prefix_cache_.has_value()) {
        Status st = state.prefix_cache_->shutdown();
        if (!st.ok() && first_error.ok()) first_error = st;
        state.prefix_cache_.reset();
    }

    state.batcher_.reset();
    state.tp_executor_ = nullptr;
    state.tp_slots_ = nullptr;
    state.tp_gdn_ = nullptr;
    state.tp_kv_ = nullptr;
    state.tp_arena_ = nullptr;
    state.tp_stream_ = nullptr;
    state.tp_coordinator_ = nullptr;
    state.tp_group_.reset();
    state.pp_model_ = nullptr;
    state.pp_coordinator_.reset();

    if (state.memory_.has_value()) {
        Status st = state.memory_->shutdown();
        if (!st.ok() && first_error.ok()) first_error = st;
    }
    state.arena_ = nullptr;

    state.shutdown_ = true;
    return first_error;
}

bool ComputeSession::is_shutdown() const noexcept { return impl_->shutdown_; }

const ComputeConfig& ComputeSession::config() const noexcept { return impl_->config_; }

double ComputeSession::model_load_ms() const noexcept { return impl_->model_load_ms_; }

uint32_t ComputeSession::max_num_batched_tokens() const noexcept {
    return impl_->exec_max_tokens_;
}

bool ComputeSession::dflash2_host_token_ids() const noexcept {
    return impl_->config_.dflash2_host_token_ids;
}

bool ComputeSession::dflash2_batcher_active() const noexcept {
    return impl_->batcher_ != nullptr && impl_->batcher_->dflash2_available();
}

bool ComputeSession::prefix_cache_active() const noexcept {
    return impl_->prefix_cache_.has_value();
}

runtime::OnlineNgramStats ComputeSession::ngram_stats() const noexcept {
    if (impl_ == nullptr || impl_->batcher_ == nullptr) {
        return runtime::OnlineNgramStats{};
    }
    return impl_->batcher_->ngram_stats();
}

std::string ComputeSession::prefix_cache_error() const {
    return impl_->prefix_cache_error_;
}

runtime::KVCapacitySnapshot ComputeSession::kv_capacity_snapshot() const {
    if (impl_->batcher_ == nullptr) {
        return {};
    }
    return impl_->batcher_->kv_capacity_snapshot();
}

std::size_t ComputeSession::queued_request_count() const noexcept {
    return impl_->batcher_ == nullptr ? 0 : impl_->batcher_->queued_count();
}

models::ModelInfo ComputeSession::model_info() const {
    return models::make_model_info(impl_->model());
}

PoolUsage ComputeSession::pool_usage() const {
    PoolUsage usage;
    const Impl& state = *impl_;
    usage.arena_used_bytes = state.arena().used();
    usage.arena_committed_bytes = state.arena().committed();
    usage.arena_capacity_bytes = state.arena().capacity();
    usage.arena_vmm = state.arena().is_vmm();
    usage.kv_reserved_bytes = state.kv_pool().reserved_bytes();
    usage.kv_used_pages = state.kv_pool().num_used_pages();
    usage.kv_total_pages = state.kv_pool().num_pages();
    usage.sequence_used_slots = state.slot_pool().num_used_slots();
    usage.gdn_used_slots = state.gdn_pool().num_used_slots();
    return usage;
}

MemoryDebugInfo ComputeSession::memory_debug() {
    Impl& state = *impl_;
    MemoryDebugInfo info;
    info.has_dense_weight_view = models::has_dense_weight_view(state.model());
    info.dense_weights = models::dense_weight_bytes(state.model());
    runtime::Executor& executor = state.executor();
    info.row_bucket_workspace_bytes.reserve(runtime::ProgramSet::kRowBucketCount);
    for (std::size_t bucket = 0; bucket < runtime::ProgramSet::kRowBucketCount; ++bucket) {
        const auto& program = executor.program_set.programs[bucket];
        info.row_bucket_workspace_bytes.push_back(program.workspace.total_bytes);
        info.staging_dispatch_count += program.dispatches.size();
        const uint64_t slot_base =
            executor.staging_pool.slot(static_cast<uint32_t>(bucket), 0);
        for (std::size_t index = 0; index < program.dispatches.size(); ++index) {
            const uint64_t slot = slot_base + index;
            if (slot < executor.staging_pool.bytes.size()) {
                const uint64_t image = executor.staging_pool.bytes[slot];
                if (image > info.staging_max_image_bytes) info.staging_max_image_bytes = image;
            }
        }
        if (bucket < executor.program_staging_meta.size()) {
            info.staging_shared_program_bytes +=
                executor.program_staging_meta[bucket].device_bytes;
        }
    }
    info.staging_device_bytes = executor.staging_pool.pool_bytes;
    info.staging_host_bytes = executor.dispatch_staging_bytes;
    info.execution_workspace_bytes = executor.execution_workspace.allocation_bytes();
    info.gdn_state_bytes = state.gdn_pool().reserved_bytes();
    return info;
}

bool ComputeSession::has_target_hidden_taps() const noexcept {
    return impl_->executor().config.target_hidden_tap_count > 0;
}

Status ComputeSession::synchronize() const {
    const hipError_t error = hipStreamSynchronize(impl_->stream());
    if (error != hipSuccess) {
        return Status::hip_error("hipStreamSynchronize", hipGetErrorString(error), __FILE__,
                                 __LINE__);
    }
    return Status::make_ok();
}

Status ComputeSession::copy_logits(std::vector<float>& out, std::size_t& rows) {
    const auto& logits = impl_->executor().logits;
    const std::size_t vocab = impl_->model().hybrid_config().vocab_size;
    if (logits.ndim() != 2 || logits.dim(1) != vocab ||
        logits.element_size() != sizeof(float)) {
        char message[256];
        std::snprintf(
            message, sizeof(message),
            "unexpected logits shape ndim=%zu dim0=%zu dim1=%zu elem=%zu (vocab=%zu)",
            logits.ndim(), logits.ndim() > 0 ? logits.dim(0) : 0,
            logits.ndim() > 1 ? logits.dim(1) : 0, logits.element_size(), vocab);
        return Status::invalid_state(message, __FILE__, __LINE__);
    }
    rows = logits.dim(0);
    out.resize(rows * vocab);
    const hipError_t error = hipMemcpy(out.data(), logits.data<float>(),
                                       out.size() * sizeof(float), hipMemcpyDeviceToHost);
    if (error != hipSuccess) {
        return Status::hip_error("hipMemcpy", hipGetErrorString(error), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status ComputeSession::copy_hidden(std::size_t which, std::vector<float>& out,
                                   std::size_t& rows, std::size_t& cols) {
    runtime::Executor& executor = impl_->executor();
    const auto& tensor = which == 0 ? executor.output_hidden : executor.token_hidden;
    if (tensor.ndim() != 2 || tensor.element_size() != sizeof(std::uint16_t)) {
        return Status::invalid_state("hidden tensor has an unexpected shape", __FILE__,
                                     __LINE__);
    }
    rows = tensor.dim(0);
    cols = tensor.dim(1);
    std::vector<std::uint16_t> raw(rows * cols);
    const hipError_t error = hipMemcpy(raw.data(), tensor.data<void>(),
                                       raw.size() * sizeof(std::uint16_t),
                                       hipMemcpyDeviceToHost);
    if (error != hipSuccess) {
        return Status::hip_error("hipMemcpy", hipGetErrorString(error), __FILE__, __LINE__);
    }
    out.resize(raw.size());
    for (std::size_t index = 0; index < raw.size(); ++index) {
        const std::uint32_t bits = static_cast<std::uint32_t>(raw[index]) << 16;
        std::memcpy(&out[index], &bits, sizeof(float));
    }
    return Status::make_ok();
}

Status ComputeSession::copy_target_hidden(std::size_t index, std::vector<float>& out,
                                          std::size_t& rows, std::size_t& cols,
                                          std::size_t& layer) {
    runtime::Executor& executor = impl_->executor();
    if (index >= executor.config.target_hidden_tap_count) {
        return Status::out_of_range("target hidden tap index out of range", __FILE__,
                                    __LINE__);
    }
    const auto& tensor = executor.dflash_target_hidden[index];
    layer = executor.config.target_hidden_taps[index];
    if (tensor.ndim() != 2 || tensor.element_size() != sizeof(std::uint16_t)) {
        return Status::invalid_state("target hidden tensor has an unexpected shape",
                                     __FILE__, __LINE__);
    }
    rows = tensor.dim(0);
    cols = tensor.dim(1);
    std::vector<std::uint16_t> raw(rows * cols);
    const hipError_t error = hipMemcpy(raw.data(), tensor.data<void>(),
                                       raw.size() * sizeof(std::uint16_t),
                                       hipMemcpyDeviceToHost);
    if (error != hipSuccess) {
        return Status::hip_error("hipMemcpy", hipGetErrorString(error), __FILE__, __LINE__);
    }
    out.resize(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) {
        const std::uint32_t bits = static_cast<std::uint32_t>(raw[i]) << 16;
        std::memcpy(&out[i], &bits, sizeof(float));
    }
    return Status::make_ok();
}

}  // namespace engine
}  // namespace shelf
