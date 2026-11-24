#include <shelf/runtime/tp/tensor_parallel.h>

#include <shelf/core/gpu/cleanup.h>
#include <shelf/weights/weight_bytes.h>
#include <shelf/core/gpu/scoped_device.h>
#include <shelf/core/memory/size_format.h>
#include <shelf/runtime/request_handle.h>
#include <shelf/runtime/gdn_state_layout.h>
#include <shelf/runtime/dflash2_config.h>
#include <shelf/runtime/dflash2_spec_config.h>
#include <shelf/runtime/executor_footprint.h>
#include <shelf/runtime/kv_pages.h>
#include <shelf/runtime/prefix_cache.h>
#include <shelf/models/tp_rank_model.h>

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <thread>
#include <utility>

namespace shelf {
namespace runtime {

namespace {

using ::shelf::weights::add_matrix_bytes;

std::uint32_t ceil_div(std::uint32_t a, std::uint32_t b) {
    return (a + b - 1u) / b;
}

std::uint32_t count_attention_layers(const shelf::qwen_hybrid::QwenHybridConfig& tc) {
    return static_cast<std::uint32_t>(
        shelf::qwen_hybrid::count_layer_topology(tc).attention_layers);
}

std::size_t tp_inbox_bytes(const Executor& executor) {
    std::size_t max_bytes = 0;
    const std::size_t count =
        std::min(executor.tp_schedules.size(),
                 static_cast<std::size_t>(::shelf::runtime::ProgramSet::kRowBucketCount));
    for (std::size_t i = 0; i < count; ++i) {
        const auto& program = executor.program_set.programs[i];
        const std::uint32_t bucket_rows =
            ::shelf::runtime::row_bucket_rows(program.row_bucket);
        const std::uint32_t token_rows =
            std::min(bucket_rows, executor.config.max_batch_rows);
        for (const auto& segment : executor.tp_schedules[i].segments) {
            if (segment.barrier_after == ::shelf::runtime::TpBarrierKind::None) continue;
            const auto* value = program.find_value(segment.combine_value);
            if (value == nullptr) continue;
            const std::uint32_t rows =
                value->row_domain == ::shelf::runtime::ValueRowDomain::OUTPUT_ROWS
                    ? executor.config.max_scheduled_output_rows
                    : token_rows;
            const std::size_t bytes =
                static_cast<std::size_t>(rows) * value->row_stride *
                ::shelf::runtime::value_dtype_bytes(value->dtype);
            max_bytes = std::max(max_bytes, bytes);
        }
    }
    return max_bytes;
}

}

TpRankRuntime::~TpRankRuntime() noexcept {
    if (device_id_ >= 0) {
        auto scope = shelf::gpu::ScopedDevice::create(device_id_);
        (void)scope;
        if (executor_ready_) {
            (void)executor_shutdown(executor_);
            executor_ready_ = false;
        }
    }
}

Result<std::unique_ptr<TpRankRuntime>> TpRankRuntime::create(
    const TpRankRuntimeConfig& config) {
    if (config.tp_size < 1) {
        return Status::invalid_argument("tp_size must be at least 1", __FILE__, __LINE__);
    }
    if (config.tp_rank >= config.tp_size) {
        return Status::invalid_argument("tp_rank out of range", __FILE__, __LINE__);
    }
    if (config.max_scheduled_requests == 0 || config.max_seq_len == 0) {
        return Status::invalid_argument(
            "max_scheduled_requests and max_seq_len must be > 0", __FILE__, __LINE__);
    }
    if (config.arena == nullptr || config.stream == nullptr) {
        return Status::invalid_argument(
            "tp rank runtime requires a borrowed arena and stream", __FILE__, __LINE__);
    }

    auto scope = shelf::gpu::ScopedDevice::create(config.device_id);
    if (!scope.ok()) return scope.status();

    std::unique_ptr<TpRankRuntime> rank(new TpRankRuntime());
    rank->device_id_ = config.device_id;
    rank->stream_ = config.stream;

    gpu::GpuArena& arena = *config.arena;
    rank->arena_ = config.arena;

    // The rank's model is the caller's choice: which family the checkpoint is, and therefore
    // which weights this rank loads and which local geometry it runs with, is not something
    // the rank runtime can decide.
    if (config.build_rank_model == nullptr) {
        return Status::invalid_argument(
            "tp rank runtime requires a rank model builder", __FILE__, __LINE__);
    }
    auto model_result = config.build_rank_model(config, arena, rank->stream_);
    if (!model_result.ok()) return model_result.status();
    rank->model_ = model_result.release();

    const shelf::qwen_hybrid::QwenHybridConfig& tc = rank->hybrid_config();
    const std::uint32_t max_blocks_per_sequence =
        kv_pages_for_context(config.max_seq_len, config.page_tokens);
    const std::uint32_t attention_layers =
        stage_attention_layers(tc, config.pipeline_stage);
    const std::uint32_t local_kv_heads = rank->context().local_key_value_heads;
    const std::uint32_t head_dim = static_cast<std::uint32_t>(tc.attention_head_dim);

    const std::size_t kv_page_bytes = PagedKVPool::page_bytes(
        config.page_tokens, attention_layers, local_kv_heads, head_dim,
        config.kv_cache_dtype, config.octave_kv_format);
    if (kv_page_bytes == 0u) {
        return Status::invalid_state("KV page geometry is zero", __FILE__, __LINE__);
    }

    const auto& rank_context = rank->context();
    const auto gdn_layout_result = make_gdn_state_layout_local(
        tc, stage_gdn_layers(tc, config.pipeline_stage), rank_context.local_gdn_key_heads,
        rank_context.local_gdn_value_heads);

    if (config.load_private_weights && config.tp_rank == 0u) {
        Status private_status = config.load_private_weights(
            arena, rank->stream_, rank->private_lm_head_,
            rank->private_embed_tokens_);
        if (!private_status.ok()) return private_status;
    }
    const std::size_t used_after_weights = rank->arena_->used();
    TpRankMemoryPlan plan;
    plan.target_weights = used_after_weights;
    plan.draft_weights = config.drafter_weights_bytes;

    auto seq_result = SequenceSlotPool::create(
        arena, config.max_scheduled_requests, max_blocks_per_sequence);
    if (!seq_result.ok()) return seq_result.status();
    rank->seq_pool_.emplace(seq_result.release());

    auto gdn_result = GdnStatePool::create(arena,
                                           config.max_scheduled_requests,
                                           gdn_layout_result);
    if (!gdn_result.ok()) return gdn_result.status();
    rank->gdn_pool_.emplace(gdn_result.release());

    const shelf::weights::Exl3LinearShape exl3_linear_shape =
        rank->model_->exl3_linear_shape();
    const std::size_t arena_free_now =
        rank->arena_->capacity() - rank->arena_->used();
    std::size_t reserved = config.working_set_reserve_bytes;
    if (reserved == 0u) {
        ExecutorConfig reserve_config;
        reserve_config.max_batch_rows = config.max_batch_rows;
        reserve_config.max_scheduled_requests = config.max_scheduled_requests;
        reserve_config.max_scheduled_output_rows = config.max_scheduled_output_rows;
        reserve_config.target_hidden_tap_count = config.target_hidden_tap_count;
        reserve_config.exl3_linear_shape = exl3_linear_shape;
        // The dense FFN term of the shared estimate stands in for the FFN activations that
        // still sit at token rows: for a model with grouped experts those are its shared
        // expert. The routed experts add their own group-row term.
        reserved = executor_reserve_bytes(
            tc, rank->model_->token_row_ffn_size(), reserve_config, config.tp_size,
            rank->model_->uses_exl3_linear(),
            rank->model_->uses_qsa_state());
        reserved += rank->model_->ffn_workspace_reserve_bytes(config.max_batch_rows);
        const std::size_t state_pools =
            rank->model_->executor_state_reserve_bytes(config.max_scheduled_requests);
        reserved += state_pools;
        // The capacity formula is an estimate, and it is short of the executor's real footprint
        // because the real workspace is the max program workspace, which the estimate only
        // approximates. The gap grows with the row bucket (measured: 10 - 130 MiB on a 40 layer
        // model, ~210 MiB on a 48 layer EXL3 model at a 2048 row bucket). The KV pool is sized
        // from what is left, so an estimate that is too small turns into a failed allocation
        // after the pool already took the arena. The auto path keeps slack back, scaled to the
        // estimate so it tracks the row bucket; an explicit --working-set-reserve is taken as
        // given. The scale is capped because the measured gap does not grow with the estimate
        // past a few hundred MiB, and every MiB held back here is a MiB the KV pool cannot use.
        const std::size_t slack = std::min<std::size_t>(
            std::max<std::size_t>(64ull << 20ull, reserved / 8ull), 256ull << 20ull);
        reserved += slack;
        std::fprintf(stderr,
                     "MEMORY_PLAN auto tp rank %u: working set reserve %s (row bucket %u, "
                     "incl. %s slack)\n",
                     static_cast<unsigned>(config.tp_rank),
                     gpu::format_gib(reserved).c_str(),
                     static_cast<unsigned>(config.max_batch_rows),
                     gpu::format_gib(slack).c_str());
    }
    plan.workspace = reserved;
    if (config.prefix_cache) {
        PrefixCacheConfig prefix_config;
        prefix_config.checkpoints = config.prefix_checkpoints;
        prefix_config.capture_slots = config.max_scheduled_requests;
        prefix_config.spec_ring_bytes = config.prefix_spec_ring_bytes;
        prefix_config.strategy = config.prefix_strategy;
        prefix_config.slot_count = config.prefix_slot_count;
        prefix_config.save_interval = config.prefix_save_interval;
        if (prefix_config.strategy == PrefixCacheStrategy::Slot) {
            prefix_config.checkpoints = prefix_config.slot_count + 1u;
            prefix_config.capture_slots = 0u;
        }
        const PrefixCacheBytes prefix_bytes =
            prefix_cache_bytes(prefix_config, gdn_layout_result);
        reserved += prefix_bytes.total();
        const std::size_t capture_state =
            prefix_config.capture_slots == 0u
                ? 0u
                : rank->model_->executor_state_reserve_bytes(
                      prefix_config.capture_slots);
        reserved += capture_state;
        plan.prefix = prefix_bytes.total() + capture_state;
        std::fprintf(stderr, "MEMORY_PLAN tp rank %u weights: %s\n", config.tp_rank,
                     gpu::format_gib(plan.target_weights).c_str());
        std::fprintf(stderr, "MEMORY_PLAN tp prefix cache: %s (capture state %s)\n",
                     gpu::format_gib(prefix_bytes.total()).c_str(),
                     gpu::format_gib(capture_state).c_str());
    }
    if (config.drafter_config != nullptr) {
        DFlash2SpecDecoderConfig spec_config;
        spec_config.num_drafts = config.drafter_drafts == 0u ? 7u : config.drafter_drafts;
        const auto* q4 = rank->rank_model() != nullptr
            ? rank->rank_model()->qwen4exp_model()
            : nullptr;
        const weights::MatrixWeight* drafter_head = q4 != nullptr
            ? &q4->weights().lm_head
            : &rank->private_lm_head();
        const std::size_t lm_head_k = config.drafter_lm_head_k != 0u
            ? config.drafter_lm_head_k
            : static_cast<std::size_t>(drafter_head->k_padded);
        const bool exl3_mul24_head = drafter_head->quant_spec ==
            quantization::QuantSpecId::EXL3_MUL24_INT8_V1;
        const DFlash2Footprint footprint = config.drafter_footprint(
            *config.drafter_config, spec_config, *rank->gdn_pool_,
            config.drafter_weights_bytes, lm_head_k,
            config.drafter_sessions == 0u ? config.max_scheduled_requests
                                          : config.drafter_sessions,
            0u, exl3_mul24_head);
        reserved += footprint.total();
        plan.drafter_runtime = footprint.total() - footprint.draft_weights;
        std::fprintf(stderr,
                     "MEMORY_PLAN tp drafter: weights %s scratch %s spec batch %s "
                     "session slot %s x %u\n",
                     gpu::format_gib(footprint.draft_weights).c_str(),
                     gpu::format_gib(footprint.draft_scratch).c_str(),
                     gpu::format_gib(footprint.spec_batch).c_str(),
                     gpu::format_gib(footprint.session_slot).c_str(),
                     static_cast<unsigned>(footprint.sessions));
    }
    if (config.mtp_batch.enabled && config.mtp_batch.max_requests != 0u) {
        const std::size_t batch_bytes = config.mtp_reserve(
            *rank->model_, tc, config.mtp_batch, gdn_layout_result, config.tp_rank);
        reserved += batch_bytes;
        plan.drafter_runtime += batch_bytes;
        std::fprintf(stderr,
                     "MEMORY_PLAN tp mtp batch reserve: %s (%u slots)\n",
                     gpu::format_gib(batch_bytes).c_str(),
                     static_cast<unsigned>(config.mtp_batch.max_requests));
    }
    std::fprintf(stderr,
                 "MEMORY_PLAN tp rank %u kv budget: free_after_weights=%s reserve=%s "
                 "kv_budget=%s page=%s\n",
                 static_cast<unsigned>(config.tp_rank),
                 gpu::format_gib(arena_free_now).c_str(),
                 gpu::format_gib(reserved).c_str(),
                 gpu::format_gib(arena_free_now > reserved ? arena_free_now - reserved : 0u)
                     .c_str(),
                 gpu::format_gib(kv_page_bytes).c_str());
    if (arena_free_now <= reserved) {
        return Status::insufficient_memory(
            "tp rank arena has " + gpu::format_gib(arena_free_now) +
                " free after weights but the runtime reserves " +
                gpu::format_gib(reserved) +
                " for the working set (raise --arena-gib or lower "
                "--working-set-reserve-gib)",
            __FILE__, __LINE__);
    }
    const std::uint32_t pages_fit = static_cast<std::uint32_t>(
        (arena_free_now - reserved) / kv_page_bytes);
    if (pages_fit <= 1u) {
        return Status::insufficient_memory(
            "tp rank arena cannot hold any KV page after reserving the working set",
            __FILE__, __LINE__);
    }
    const std::uint32_t fit_pages = pages_fit - 1u;
    const std::uint64_t single_request_tokens =
        static_cast<std::uint64_t>(config.max_seq_len) + 1u;
    const std::uint32_t required_pages =
        ceil_div(static_cast<std::uint32_t>(single_request_tokens), config.page_tokens);
    if (fit_pages < required_pages) {
        const std::uint32_t max_supportable =
            fit_pages * config.page_tokens - 1u;
        return Status::insufficient_memory(
            "tp rank arena fits " +
                std::to_string(fit_pages * config.page_tokens) +
                " KV tokens but max_seq_len " + std::to_string(config.max_seq_len) +
                " needs " + std::to_string(single_request_tokens) +
                " (max supportable max_seq_len " + std::to_string(max_supportable) +
                ")",
            __FILE__, __LINE__);
    }
    const std::uint32_t kv_pool_pages =
        config.kv_pool_pages != 0u ? config.kv_pool_pages : fit_pages + 1u;
    plan.kv = static_cast<std::size_t>(kv_pool_pages) * kv_page_bytes;
    plan.kv_tokens = (kv_pool_pages - 1u) * config.page_tokens;
    rank->plan_ = plan;

    auto kv_result = PagedKVPool::create(
        arena, kv_pool_pages, config.page_tokens, attention_layers,
        local_kv_heads, head_dim, config.kv_cache_dtype, config.octave_kv_format);
    if (!kv_result.ok()) return kv_result.status();
    rank->kv_pool_.emplace(kv_result.release());

    std::uint32_t exec_tokens = config.max_batch_rows;
    if (exec_tokens == 0) exec_tokens = config.max_seq_len;
    const std::uint32_t row_bucket_max =
        shelf::runtime::row_bucket_limit(shelf::runtime::RowBucket::R8192);
    if (exec_tokens > row_bucket_max) exec_tokens = row_bucket_max;
    if (exec_tokens == 0) exec_tokens = 1;

    ExecutorConfig exec_config;
    exec_config.max_batch_rows = exec_tokens;
    exec_config.max_scheduled_requests = config.max_scheduled_requests;
    exec_config.max_scheduled_output_rows = config.max_scheduled_output_rows;
    exec_config.target_hidden_taps = config.target_hidden_taps;
    exec_config.target_hidden_tap_count = config.target_hidden_tap_count;
    exec_config.tp_combine_f32 = config.tp_combine_f32;
    exec_config.emit_normed_token_hidden = config.emit_normed_token_hidden;
    if (config.emit_normed_token_hidden) {
        const std::uint32_t width = rank->model_->normed_token_hidden_width();
        if (width != 0u) exec_config.normed_token_hidden_width = width;
    }
    exec_config.max_context_tokens = config.max_seq_len + 1u;
    exec_config.pipeline_hidden_width = config.pipeline_hidden_width;
    exec_config.rope_profile = config.rope_profile;
    exec_config.stage = config.pipeline_stage;
    exec_config.tp = &rank->context();
    exec_config.exl3_linear_shape = exl3_linear_shape;

    const std::size_t executor_before = rank->arena_->used();
    auto exec_result = rank->model_->create_executor(
        rank->seq_pool_.value(), rank->gdn_pool_.value(), rank->kv_pool_.value(),
        arena, exec_config);
    std::fprintf(stderr,
                 "MEMORY_PLAN tp rank %u executor footprint: %s (reserve %s)\n",
                 static_cast<unsigned>(config.tp_rank),
                 gpu::format_gib(rank->arena_->used() - executor_before).c_str(),
                 gpu::format_gib(reserved).c_str());
    if (!exec_result.ok()) return exec_result.status();
    rank->executor_ = exec_result.release();
    rank->executor_ready_ = true;

    return rank;
}


const shelf::models::TpRankModel* TpRankRuntime::rank_model() const { return model_.get(); }

const shelf::qwen_hybrid::QwenHybridConfig& TpRankRuntime::hybrid_config() const {
    return model_->hybrid_config();
}

const TensorParallelContext& TpRankRuntime::context() const { return model_->context(); }

std::uint64_t TpRankRuntime::weight_bytes() const {
    return model_ ? model_->weight_bytes() : 0u;
}

std::uint64_t TpRankRuntime::kv_bytes() const {
    if (!kv_pool_) return 0;
    const PagedKVPool& pool = kv_pool_.value();
    const std::uint64_t body =
        static_cast<std::uint64_t>(pool.num_attention_layers()) * pool.num_pages() *
        pool.page_tokens() * pool.kv_heads() * pool.head_dim();
    switch (pool.dtype()) {
        case KVCacheDType::BF16:
            return body * 4u;
        case KVCacheDType::FP8_E4M3:
            return body * 2u + (body / pool.head_dim()) * 8u;
        case KVCacheDType::OCTAVE:
            return static_cast<std::uint64_t>(pool.num_attention_layers()) *
                   pool.num_pages() * pool.page_tokens() * pool.kv_heads() *
                   (::shelf::kernel::octave_k_slot_bytes(pool.octave_format()) +
                    ::shelf::kernel::octave_v_slot_bytes(pool.octave_format()));
    }
    return 0;
}

std::uint64_t TpRankRuntime::gdn_state_bytes() const {
    if (!gdn_pool_) return 0;
    const GdnStatePoolLayout& layout = gdn_pool_->layout();
    const std::uint64_t sequences = gdn_pool_->max_sequences();
    const std::uint64_t conv =
        sequences * layout.num_gdn_states * layout.conv_history *
        layout.conv_history_stride * 2u;
    const std::uint64_t recurrent =
        sequences * layout.num_gdn_states * layout.num_v_heads * layout.head_k *
        layout.head_v * 4u;
    return conv + recurrent;
}

const char* TpCoordinator::transport_name() const noexcept {
    return !tp_groups_.empty() && tp_groups_.front().transport != nullptr
        ? tp_groups_.front().transport->name()
        : "none";
}

Result<std::unique_ptr<TpCoordinator>> TpCoordinator::create(
    const TpCoordinatorConfig& config) {
    if (config.devices.size() < 2) {
        return Status::invalid_argument("tp coordinator requires at least 2 devices",
                                        __FILE__, __LINE__);
    }

    std::unique_ptr<TpCoordinator> coordinator(new TpCoordinator());
    Status init = coordinator->initialize(config);
    if (!init.ok()) return init;
    return coordinator;
}

Status TpCoordinator::initialize(const TpCoordinatorConfig& config) {
    devices_ = config.devices;
    const std::uint32_t tp_size = config.tensor_parallel == 0u
        ? static_cast<std::uint32_t>(devices_.size())
        : config.tensor_parallel;
    // TP and EP are independent axes. EP=1 keeps every expert on every TP rank and lets TP
    // split each expert's matrix; EP>1 first splits the expert IDs, then TP splits each local
    // expert's matrix inside the EP group.
    const bool explicit_ep = config.expert_parallel > 1u;
    const std::uint32_t ep_size = config.expert_parallel;
    const std::size_t expected_devices = static_cast<std::size_t>(ep_size) * tp_size;
    if (devices_.empty() || tp_size == 0u || ep_size == 0u ||
        devices_.size() != expected_devices) {
        return Status::invalid_argument(
            "tp coordinator device count does not match tensor_parallel * expert_parallel",
            __FILE__, __LINE__);
    }
    if (config.memory == nullptr) {
        return Status::invalid_argument(
            "tp coordinator requires a GpuMemoryManager", __FILE__, __LINE__);
    }
    std::vector<gpu::GpuArena*> arenas(devices_.size(), nullptr);
    std::vector<hipStream_t> streams(devices_.size(), nullptr);
    for (std::size_t r = 0; r < devices_.size(); ++r) {
        auto arena = config.memory->arena_for(devices_[r]);
        if (!arena.ok()) return arena.status();
        arenas[r] = arena.value();
        auto stream = config.memory->stream_for(devices_[r]);
        if (!stream.ok()) return stream.status();
        streams[r] = stream.value();
    }

    std::jthread model_prefetch;
    if (config.prefetch_rank_model != nullptr) {
        model_prefetch = std::jthread(
            [model_dir = config.model_dir, prefetch = config.prefetch_rank_model]() {
                prefetch(model_dir);
            });
    }

    struct RankSlot {
        std::unique_ptr<TpRankRuntime> runtime;
        Status status = Status::make_ok();
    };
    std::vector<RankSlot> slots(devices_.size());
    std::vector<std::thread> loaders;
    loaders.reserve(devices_.size());
    for (std::size_t r = 0; r < devices_.size(); ++r) {
        TpRankRuntimeConfig rank_config;
        rank_config.model_dir = config.model_dir;
        rank_config.tp_size = tp_size;
        rank_config.tp_rank = static_cast<std::uint32_t>(r % tp_size);
        rank_config.ep_size = ep_size;
        rank_config.ep_rank = static_cast<std::uint32_t>(r / tp_size);
        rank_config.ep_enabled = explicit_ep;
        rank_config.expert_partition = explicit_ep
            ? ExpertPartitionMode::AcrossEP
            : ExpertPartitionMode::Replicated;
        rank_config.device_id = devices_[r];
        rank_config.max_batch_rows = config.max_batch_rows;
        rank_config.max_scheduled_requests = config.max_scheduled_requests;
        rank_config.max_scheduled_output_rows = config.max_scheduled_output_rows;
        rank_config.max_seq_len = config.max_seq_len;
        rank_config.page_tokens = config.page_tokens;
        rank_config.pipeline_hidden_width = config.pipeline_hidden_width;
        rank_config.kv_cache_dtype = config.kv_cache_dtype;
        rank_config.octave_kv_format = config.octave_kv_format;
        rank_config.arena = arenas[r];
        rank_config.stream = streams[r];
        rank_config.working_set_reserve_bytes = config.working_set_reserve_bytes;
        rank_config.kv_pool_pages = config.kv_pool_pages;
        rank_config.drafter_config = config.drafter_config;
        rank_config.drafter_footprint = config.drafter_footprint;
        rank_config.mtp_reserve = config.mtp_reserve;
        rank_config.drafter_weights_bytes = config.drafter_weights_bytes;
        rank_config.drafter_drafts = config.drafter_drafts;
        rank_config.drafter_sessions = config.drafter_sessions;
        rank_config.drafter_lm_head_k = config.drafter_lm_head_k;
        rank_config.prefix_cache = config.prefix_cache;
        rank_config.prefix_checkpoints = config.prefix_checkpoints;
        rank_config.prefix_spec_ring_bytes = config.prefix_spec_ring_bytes;
        rank_config.prefix_strategy = config.prefix_strategy;
        rank_config.prefix_slot_count = config.prefix_slot_count;
        rank_config.prefix_save_interval = config.prefix_save_interval;
        rank_config.load_private_weights = config.load_private_weights;
        rank_config.target_hidden_taps = config.target_hidden_taps;
        rank_config.target_hidden_tap_count = config.target_hidden_tap_count;
        rank_config.emit_normed_token_hidden = config.emit_normed_token_hidden;
        rank_config.tp_combine_f32 = config.tp_combine_f32;
        rank_config.rope_profile = config.rope_profile;
        rank_config.build_rank_model = config.build_rank_model;
        rank_config.load_options = config.load_options;
        rank_config.load_mtp_layers = config.load_mtp_layers;
        rank_config.mtp_split = config.mtp_split;
        rank_config.pipeline_stage = config.pipeline_stage;
        rank_config.mtp_batch = config.mtp_batch;
        loaders.emplace_back(
            [&slots, r, rank_config = std::move(rank_config)]() {
                auto rank_result = TpRankRuntime::create(rank_config);
                if (rank_result.ok()) {
                    slots[r].runtime = rank_result.release();
                } else {
                    slots[r].status = rank_result.status();
                }
            });
    }
    for (std::thread& loader : loaders) loader.join();
    if (model_prefetch.joinable()) model_prefetch.join();
    if (config.release_rank_model != nullptr) config.release_rank_model();
    for (RankSlot& slot : slots) {
        if (!slot.status.ok()) return slot.status;
        ranks_.push_back(std::move(slot.runtime));
    }

    explicit_ep_ = explicit_ep;
    ParallelShape shape;
    shape.pp = 1u;
    shape.ep = config.expert_parallel;
    shape.tp = tp_size;
    auto topology_result = ParallelTopology::create(shape, devices_);
    if (!topology_result.ok()) return topology_result.status();
    topology_.emplace(topology_result.release());

    auto make_group = [&](const RankGroup& rank_group) -> Result<ReduceGroup> {
        ReduceGroup group;
        for (std::uint32_t member : rank_group.ranks) {
            group.devices.push_back(topology_->device(member));
        }
        if (group.devices.size() < 2u) return group;
        auto transport_result = ::shelf::runtime::create_tp_transport(group.devices);
        if (!transport_result.ok()) return transport_result.status();
        group.transport = std::move(transport_result.value().transport);
        auto verify_status = group.transport->verify();
        if (!verify_status.ok()) {
            std::fprintf(stderr,
                         "[tp] transport %s failed verification (%s); falling back to "
                         "host-mediated reference transport\n",
                         group.transport->name(), verify_status.message().c_str());
            group.transport =
                std::make_unique<::shelf::runtime::HostMediatedTpTransport>(group.devices);
        }
        group.barrier = std::make_unique<::shelf::runtime::TpBarrierGroup>(
            static_cast<int>(group.devices.size()), group.devices, group.transport.get());
        Status barrier_status = group.barrier->initialize();
        if (!barrier_status.ok()) return barrier_status;
        return group;
    };

    for (std::uint32_t e = 0; e < topology_->shape().ep; ++e) {
        auto group = make_group(topology_->tp_group(0u, e));
        if (!group.ok()) return group.status();
        tp_groups_.push_back(group.release());
    }
    if (explicit_ep) {
        ep_groups_.reserve(topology_->shape().tp);
        for (std::uint32_t t = 0; t < topology_->shape().tp; ++t) {
            auto group = make_group(topology_->ep_group(0u, t));
            if (!group.ok()) return group.status();
            ep_groups_.push_back(group.release());
        }
    }

    axes_overlap_ = explicit_ep && topology_->shape().tp == 1u &&
                    topology_->shape().ep > 1u;
    for (std::size_t r = 0; r < ranks_.size(); ++r) {
        const ParallelCoord coord = topology_->coord(static_cast<std::uint32_t>(r));
        auto& executor = ranks_[r]->executor();
        if (axes_overlap_) {
            auto* barrier = ep_groups_[coord.tp].barrier.get();
            executor.tp_barrier = barrier;
            executor.tp_rank = static_cast<int>(coord.ep);
            executor.ep_barrier = barrier;
            executor.ep_rank = static_cast<int>(coord.ep);
        } else if (explicit_ep) {
            executor.tp_barrier = tp_groups_[coord.ep].barrier.get();
            executor.tp_rank = static_cast<int>(coord.tp);
            executor.ep_barrier = ep_groups_[coord.tp].barrier.get();
            executor.ep_rank = static_cast<int>(coord.ep);
        } else {
            executor.tp_barrier = tp_groups_.front().barrier.get();
            executor.tp_rank = static_cast<int>(coord.tp);
        }
    }

    std::size_t reserve_bytes = 0;
    for (auto& rank : ranks_) {
        reserve_bytes = std::max(reserve_bytes, tp_inbox_bytes(rank->executor()));
    }
    if (reserve_bytes != 0) {
        std::fprintf(stderr, "TP_WORKSPACE inbox_bytes=%zu groups=%zu\n", reserve_bytes,
                     tp_groups_.size() + ep_groups_.size());
        for (auto& group : tp_groups_) {
            Status st = group.barrier->reserve(reserve_bytes);
            if (!st.ok()) return st;
        }
        for (auto& group : ep_groups_) {
            Status st = group.barrier->reserve(reserve_bytes);
            if (!st.ok()) return st;
        }
    }

    workers_.reserve(ranks_.size());
    for (std::size_t r = 0; r < ranks_.size(); ++r) {
        auto worker = std::make_unique<Worker>();
        Worker* raw = worker.get();
        workers_.push_back(std::move(worker));
        raw->thread = std::thread([this, r] { worker_loop(r); });
    }
    return Status::make_ok();
}

TpCoordinator::~TpCoordinator() noexcept {
    abort(Status::invalid_state("tp coordinator shutting down", __FILE__, __LINE__));
    for (auto& worker : workers_) {
        if (!worker->thread.joinable()) continue;
        {
            std::lock_guard<std::mutex> lock(worker->mutex);
            worker->stop = true;
        }
        worker->cv.notify_all();
        worker->thread.join();
    }
    workers_.clear();
    mirrors_.clear();
    tp_groups_.clear();
    ep_groups_.clear();
    for (auto& rank : ranks_) rank.reset();
    ranks_.clear();
}

void TpCoordinator::abort(const Status& reason) noexcept {
    for (auto& group : tp_groups_) {
        if (group.barrier) group.barrier->abort(reason);
    }
    for (auto& group : ep_groups_) {
        if (group.barrier) group.barrier->abort(reason);
    }
}

Status TpCoordinator::on_sequence_created(const PagedSequenceState& source) {
    const std::uint64_t key = ::shelf::runtime::request_handle_key(source.request_handle());
    if (mirrors_.find(key) != mirrors_.end()) {
        return Status::invalid_state("tp mirror sequence already exists", __FILE__,
                                     __LINE__);
    }
    MirrorSequences mirror;
    mirror.states.reserve(ranks_.size() - 1);
    for (std::size_t r = 1; r < ranks_.size(); ++r) {
        TpRankRuntime& rank = *ranks_[r];
        auto scope = shelf::gpu::ScopedDevice::create(rank.device_id());
        if (!scope.ok()) {
            mirror.states.clear();
            return scope.status();
        }
        auto created = create_paged_sequence_state(
            rank.sequence_pool(), rank.gdn_pool(), rank.kv_pool(), source.max_seq_len,
            rank.stream());
        if (!created.ok()) {
            mirror.states.clear();
            return created.status();
        }
        PagedSequenceState state = created.release();
        if (::shelf::runtime::request_handle_key(state.request_handle()) != key) {
            mirror.states.clear();
            return Status::invalid_state(
                "tp mirror sequence slot diverged from rank0", __FILE__, __LINE__);
        }
        mirror.states.push_back(std::move(state));
    }
    mirrors_.emplace(key, std::move(mirror));
    return Status::make_ok();
}

Status TpCoordinator::on_prefix_restored(const PagedSequenceState& source,
                                         std::span<const int32_t> tokens) {
    Status first_error = Status::make_ok();
    for (auto& rank_ptr : ranks_) {
        TpRankRuntime& rank = *rank_ptr;
        const ExternalPrefixRestoreHandler& handler =
            rank.executor().external_prefix_restore;
        if (!handler) continue;
        auto scope = shelf::gpu::ScopedDevice::create(rank.device_id());
        if (!scope.ok()) {
            if (first_error.ok()) first_error = scope.status();
            continue;
        }
        const Status st = handler(source.slot, tokens);
        if (!st.ok() && first_error.ok()) first_error = st;
    }
    return first_error;
}

uint32_t TpCoordinator::on_prefix_rank_count() {
    return static_cast<uint32_t>(ranks_.size());
}

PagedSequenceState* TpCoordinator::on_prefix_rank_sequence(
    const PagedSequenceState& source, uint32_t rank) {
    if (rank == 0u) return const_cast<PagedSequenceState*>(&source);
    const std::uint64_t key = ::shelf::runtime::request_handle_key(source.request_handle());
    auto it = mirrors_.find(key);
    if (it == mirrors_.end()) return nullptr;
    if (static_cast<std::size_t>(rank - 1u) >= it->second.states.size()) return nullptr;
    return &it->second.states[rank - 1u];
}

Status TpCoordinator::on_sequence_released(const PagedSequenceState& source) {
    const std::uint64_t key = ::shelf::runtime::request_handle_key(source.request_handle());
    auto it = mirrors_.find(key);
    if (it == mirrors_.end()) {
        return Status::invalid_state("tp mirror sequence not found", __FILE__, __LINE__);
    }
    Status first_error = Status::make_ok();
    for (std::size_t r = 1; r < ranks_.size(); ++r) {
        TpRankRuntime& rank = *ranks_[r];
        auto scope = shelf::gpu::ScopedDevice::create(rank.device_id());
        if (!scope.ok()) {
            if (first_error.ok()) first_error = scope.status();
            continue;
        }
        Status st =
            release_paged_sequence_state(it->second.states[r - 1], rank.stream());
        if (!st.ok() && first_error.ok()) first_error = st;
    }
    mirrors_.erase(it);
    return first_error;
}

Status TpCoordinator::sync_mirror_state(const ScheduledBatch& batch) {
    for (const ScheduledRequest& req : batch.requests) {
        if (req.sequence == nullptr) continue;
        const std::uint64_t key = ::shelf::runtime::request_handle_key(req.handle);
        auto it = mirrors_.find(key);
        if (it == mirrors_.end()) continue;
        const std::size_t want_blocks = req.sequence->block_table.size();
        for (std::size_t i = 0; i < it->second.states.size(); ++i) {
            PagedSequenceState& mirror = it->second.states[i];
            TpRankRuntime& rank = *ranks_[i + 1];
            if (mirror.block_table.size() > want_blocks) {
                auto scope = shelf::gpu::ScopedDevice::create(rank.device_id());
                if (!scope.ok()) return scope.status();
                Status st = rollback_sequence_append(
                    mirror, static_cast<std::uint32_t>(want_blocks), rank.stream());
                if (!st.ok()) return st;
            }
            mirror.position = req.sequence->position;
        }
    }
    return Status::make_ok();
}

Status TpCoordinator::attach_aux_executor(Executor& executor, std::uint32_t rank) {
    if (!topology_.has_value() || rank >= ranks_.size()) {
        return Status::invalid_argument("tp aux executor rank out of range", __FILE__,
                                        __LINE__);
    }
    const ParallelCoord coord = topology_->coord(rank);
    if (axes_overlap_) {
        auto* barrier = ep_groups_[coord.tp].barrier.get();
        executor.tp_barrier = barrier;
        executor.tp_rank = static_cast<int>(coord.ep);
        executor.ep_barrier = barrier;
        executor.ep_rank = static_cast<int>(coord.ep);
    } else if (explicit_ep_) {
        executor.tp_barrier = tp_groups_[coord.ep].barrier.get();
        executor.tp_rank = static_cast<int>(coord.tp);
        executor.ep_barrier = ep_groups_[coord.tp].barrier.get();
        executor.ep_rank = static_cast<int>(coord.ep);
    } else {
        executor.tp_barrier = tp_groups_.front().barrier.get();
        executor.tp_rank = static_cast<int>(coord.tp);
    }
    const std::size_t reserve_bytes = tp_inbox_bytes(executor);
    if (reserve_bytes != 0 && executor.tp_barrier != nullptr) {
        Status st = executor.tp_barrier->reserve(reserve_bytes);
        if (!st.ok()) return st;
        if (executor.ep_barrier != nullptr && executor.ep_barrier != executor.tp_barrier) {
            st = executor.ep_barrier->reserve(reserve_bytes);
            if (!st.ok()) return st;
        }
    }
    return Status::make_ok();
}

Result<BatchExecutionOutput> TpCoordinator::on_execute(const ScheduledBatch& batch) {
    return on_execute(batch, nullptr);
}

Result<BatchExecutionOutput> TpCoordinator::on_execute(
    const ScheduledBatch& batch, const ExecuteBatchOptions* per_rank) {
    for (const auto& request : batch.requests) {
        if (request.token_bitmask != nullptr) {
            return Status::unsupported("token bitmask requires single-device execution", __FILE__, __LINE__);
        }
    }
    if (ranks_.size() < 2) {
        return Status::invalid_state("tp coordinator has no ranks", __FILE__, __LINE__);
    }
    if (batch.token_ids_location != TokenIdsLocation::Host) {
        return Status::unsupported(
            "tp execution requires host token ids", __FILE__, __LINE__);
    }
    if (batch.speculative_verify && per_rank == nullptr) {
        return Status::unsupported(
            "tp speculative verify requires per-rank options", __FILE__, __LINE__);
    }
    const auto* aborted_barrier = explicit_ep_ ? ep_groups_.front().barrier.get()
                                               : tp_groups_.front().barrier.get();
    if (aborted_barrier != nullptr && aborted_barrier->aborted()) {
        const Status reason = aborted_barrier->abort_status();
        if (!reason.ok()) return reason;
        return Status::invalid_state("tp barrier group is aborted", __FILE__, __LINE__);
    }
    const Status sync_st = sync_mirror_state(batch);
    if (!sync_st.ok()) return sync_st;

    const std::size_t rank_count = ranks_.size();
    std::vector<std::vector<ScheduledRequest>> request_storage(rank_count);
    std::vector<ScheduledBatch> rank_batches(rank_count);
    for (std::size_t r = 0; r < rank_count; ++r) {
        rank_batches[r] = batch;
        request_storage[r].assign(batch.requests.begin(), batch.requests.end());
        if (r > 0) {
            for (ScheduledRequest& request : request_storage[r]) {
                const std::uint64_t key =
                    ::shelf::runtime::request_handle_key(request.handle);
                auto it = mirrors_.find(key);
                if (it == mirrors_.end()) {
                    return Status::invalid_state(
                        "tp mirror sequence missing for scheduled request",
                        __FILE__, __LINE__);
                }
                if (r - 1 >= it->second.states.size()) {
                    return Status::invalid_state(
                        "tp mirror sequence rank out of range", __FILE__, __LINE__);
                }
                request.sequence = &it->second.states[r - 1];
            }
        }
        rank_batches[r].requests =
            std::span<const ScheduledRequest>(request_storage[r]);
    }

    for (std::size_t r = 0; r < workers_.size(); ++r) {
        Worker* worker = workers_[r].get();
        {
            std::lock_guard<std::mutex> lock(worker->mutex);
            worker->job = &rank_batches[r];
            worker->executor = nullptr;
            worker->options = per_rank != nullptr ? &per_rank[r] : nullptr;
            worker->job_pending = true;
            worker->done = false;
            worker->status = Status::make_ok();
        }
        worker->cv.notify_all();
    }

    Status first_error = Status::make_ok();
    BatchExecutionOutput rank0_output;
    for (std::size_t r = 0; r < workers_.size(); ++r) {
        Worker* worker = workers_[r].get();
        std::unique_lock<std::mutex> lock(worker->mutex);
        worker->cv.wait(lock, [&] { return worker->done; });
        if (!worker->status.ok() && first_error.ok()) {
            first_error = worker->status;
        }
        if (r == 0) rank0_output = std::move(worker->output);
    }
    if (!first_error.ok()) {
        abort(first_error);
        return first_error;
    }
    return rank0_output;
}

Result<BatchExecutionOutput> TpCoordinator::on_execute_aux(
    const ScheduledBatch& batch, std::span<const TpAuxRank> aux,
    const ExecuteBatchOptions* per_rank) {
    if (aux.size() != ranks_.size()) {
        return Status::invalid_argument("tp aux executor count does not match ranks",
                                        __FILE__, __LINE__);
    }
    for (const TpAuxRank& rank : aux) {
        if (rank.executor == nullptr ||
            rank.sequences.size() != batch.requests.size()) {
            return Status::invalid_argument("tp aux executor entry is incomplete",
                                            __FILE__, __LINE__);
        }
        for (PagedSequenceState* sequence : rank.sequences) {
            if (sequence == nullptr) {
                return Status::invalid_argument("tp aux sequence entry is empty",
                                                __FILE__, __LINE__);
            }
        }
    }
    for (const auto& request : batch.requests) {
        if (request.token_bitmask != nullptr) {
            return Status::unsupported("token bitmask requires single-device execution",
                                       __FILE__, __LINE__);
        }
    }
    if (ranks_.size() < 2) {
        return Status::invalid_state("tp coordinator has no ranks", __FILE__, __LINE__);
    }
    if (batch.token_ids_location != TokenIdsLocation::Host) {
        return Status::unsupported("tp execution requires host token ids", __FILE__,
                                   __LINE__);
    }
    const auto* aborted_barrier = explicit_ep_ ? ep_groups_.front().barrier.get()
                                               : tp_groups_.front().barrier.get();
    if (aborted_barrier != nullptr && aborted_barrier->aborted()) {
        const Status reason = aborted_barrier->abort_status();
        if (!reason.ok()) return reason;
        return Status::invalid_state("tp barrier group is aborted", __FILE__, __LINE__);
    }

    const std::size_t rank_count = ranks_.size();
    std::vector<std::vector<ScheduledRequest>> request_storage(rank_count);
    std::vector<ScheduledBatch> rank_batches(rank_count);
    for (std::size_t r = 0; r < rank_count; ++r) {
        rank_batches[r] = batch;
        request_storage[r].assign(batch.requests.begin(), batch.requests.end());
        for (std::size_t i = 0; i < request_storage[r].size(); ++i) {
            ScheduledRequest& request = request_storage[r][i];
            request.sequence = aux[r].sequences[i];
            request.handle = aux[r].sequences[i]->request_handle();
        }
        rank_batches[r].requests =
            std::span<const ScheduledRequest>(request_storage[r]);
    }

    for (std::size_t r = 0; r < workers_.size(); ++r) {
        Worker* worker = workers_[r].get();
        {
            std::lock_guard<std::mutex> lock(worker->mutex);
            worker->job = &rank_batches[r];
            worker->executor = aux[r].executor;
            worker->options = per_rank != nullptr ? &per_rank[r] : nullptr;
            worker->job_pending = true;
            worker->done = false;
            worker->status = Status::make_ok();
        }
        worker->cv.notify_all();
    }

    Status first_error = Status::make_ok();
    BatchExecutionOutput rank0_output;
    for (std::size_t r = 0; r < workers_.size(); ++r) {
        Worker* worker = workers_[r].get();
        std::unique_lock<std::mutex> lock(worker->mutex);
        worker->cv.wait(lock, [&] { return worker->done; });
        if (!worker->status.ok() && first_error.ok()) {
            first_error = worker->status;
        }
        if (r == 0) rank0_output = std::move(worker->output);
    }
    if (!first_error.ok()) {
        abort(first_error);
        return first_error;
    }
    return rank0_output;
}

Status TpCoordinator::draft_sum(const shelf::runtime::TpSumTarget& target,
                                hipStream_t stream, int rank) {
    const ParallelCoord coord = topology_->coord(static_cast<std::uint32_t>(rank));
    const std::size_t group_index = explicit_ep_ ? coord.ep : 0u;
    const int local_rank = static_cast<int>(axes_overlap_ ? coord.ep : coord.tp);
    auto* barrier = axes_overlap_ ? ep_groups_[coord.tp].barrier.get()
                                  : tp_groups_[group_index].barrier.get();
    if (barrier == nullptr) {
        return Status::invalid_state("tp barrier group missing", __FILE__, __LINE__);
    }
    return barrier->arrive(local_rank, target, stream, false);
}

void TpCoordinator::worker_loop(std::size_t rank) {
    Worker* worker = workers_[rank].get();
    auto scope = shelf::gpu::ScopedDevice::create(ranks_[rank]->device_id());
    if (!scope.ok()) {
        std::lock_guard<std::mutex> lock(worker->mutex);
        worker->status = scope.status();
        worker->done = true;
        worker->cv.notify_all();
        return;
    }
    while (true) {
        const ScheduledBatch* job = nullptr;
        Executor* job_executor = nullptr;
        const ExecuteBatchOptions* options = nullptr;
        {
            std::unique_lock<std::mutex> lock(worker->mutex);
            worker->cv.wait(lock, [&] { return worker->job_pending || worker->stop; });
            if (worker->stop) break;
            worker->job_pending = false;
            job = worker->job;
            job_executor = worker->executor;
            options = worker->options;
        }
        BatchExecutionOutput output;
        Status st = Status::make_ok();
        if (job != nullptr) {
            st = run_rank_step(rank, *job, output, job_executor, options);
        }
        if (!st.ok()) abort(st);
        {
            std::lock_guard<std::mutex> lock(worker->mutex);
            worker->status = st;
            worker->output = std::move(output);
            worker->done = true;
        }
        worker->cv.notify_all();
    }
}

Status TpCoordinator::run_rank_step(std::size_t rank, const ScheduledBatch& batch,
                                    BatchExecutionOutput& output, Executor* executor,
                                    const ExecuteBatchOptions* options) {
    TpRankRuntime& runtime = *ranks_[rank];
    Executor& target = executor != nullptr ? *executor : runtime.executor();
    auto pending = options != nullptr
                       ? submit_batch(target, batch, runtime.stream(), *options)
                       : submit_batch(target, batch, runtime.stream());
    if (!pending.ok()) return pending.status();
    auto completed = complete_batch(target, pending.release(), runtime.stream());
    if (!completed.ok()) return completed.status();
    output = completed.release();
    return Status::make_ok();
}

std::string TpCoordinator::debug_report() const {
    std::ostringstream out;
    out << "tp_size=" << ranks_.size() << " transport="
        << transport_name() << "\n";
    for (std::size_t r = 0; r < ranks_.size(); ++r) {
        const TpRankRuntime& runtime = *ranks_[r];
        out << "  rank" << r << " device=" << runtime.device_id()
            << " weight_bytes=" << runtime.weight_bytes()
            << " kv_bytes=" << runtime.kv_bytes()
            << " gdn_state_bytes=" << runtime.gdn_state_bytes() << "\n";
    }
    return out.str();
}

}
}
