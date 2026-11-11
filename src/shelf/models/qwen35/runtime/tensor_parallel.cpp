#include <shelf/models/qwen35/runtime/tensor_parallel.h>

#include <shelf/core/gpu/cleanup.h>
#include <shelf/weights/weight_bytes.h>
#include <shelf/core/gpu/scoped_device.h>
#include <shelf/core/memory/size_format.h>
#include <shelf/runtime/request_handle.h>
#include <shelf/models/qwen35/model/gdn_state_layout.h>
#include <shelf/models/qwen35/dflash2/config.h>
#include <shelf/models/qwen35/runtime/dflash2_footprint.h>
#include <shelf/models/qwen35/runtime/dflash2_spec_decoder.h>
#include <shelf/models/qwen35/runtime/mtp_drafter.h>
#include <shelf/models/qwen35/runtime/executor_footprint.h>
#include <shelf/models/qwen35/runtime/prefix_cache.h>
#include <shelf/models/tp_rank_model.h>

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <thread>
#include <utility>

namespace shelf {
namespace qwen35 {
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

std::uint64_t tensor_bytes(const gpu::Tensor& t) {
    if (t.ndim() == 0) return 0;
    std::size_t bytes = 0;
    if (!t.required_span_bytes(bytes).ok()) return 0;
    return bytes;
}

}

TpRankRuntime::~TpRankRuntime() noexcept {
    if (executor_ready_) {
        shelf::qwen35::executor_shutdown(executor_);
        executor_ready_ = false;
    }
    if (stream_ != nullptr) {
        shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream_));
        stream_ = nullptr;
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

    auto scope = shelf::gpu::ScopedDevice::create(config.device_id);
    if (!scope.ok()) return scope.status();

    std::unique_ptr<TpRankRuntime> rank(new TpRankRuntime());
    rank->device_id_ = config.device_id;

    const hipError_t stream_err = hipStreamCreate(&rank->stream_);
    if (stream_err != hipSuccess) {
        return Status::hip_error("hipStreamCreate tp rank", hipGetErrorString(stream_err),
                                 __FILE__, __LINE__);
    }

    std::size_t arena_budget = config.arena_bytes;
    if (arena_budget == 0) {
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        const hipError_t info_err = hipMemGetInfo(&free_bytes, &total_bytes);
        if (info_err != hipSuccess) {
            return Status::hip_error("hipMemGetInfo tp rank", hipGetErrorString(info_err),
                                     __FILE__, __LINE__);
        }
        constexpr std::size_t kMarginBytes = 1ull * 1024ull * 1024ull * 1024ull;
        const std::size_t available = free_bytes > kMarginBytes
            ? free_bytes - kMarginBytes
            : free_bytes / 2u;
        if (available == 0) {
            return Status::insufficient_memory(
                "no VRAM available for the tp rank arena", __FILE__, __LINE__);
        }
        arena_budget = available;
        std::fprintf(stderr,
                     "ARENA auto: tp_rank=%u free=%s capacity=%s\n",
                     config.tp_rank, gpu::format_gib(free_bytes).c_str(),
                     gpu::format_gib(arena_budget).c_str());
    }

    auto arena_result = gpu::GpuArena::create(config.device_id, arena_budget);
    if (!arena_result.ok()) return arena_result.status();
    rank->arena_.emplace(arena_result.release());

    // The rank's model is the caller's choice: which family the checkpoint is, and therefore
    // which weights this rank loads and which local geometry it runs with, is not something
    // the rank runtime can decide.
    if (config.build_rank_model == nullptr) {
        return Status::invalid_argument(
            "tp rank runtime requires a rank model builder", __FILE__, __LINE__);
    }
    auto model_result = config.build_rank_model(config, rank->arena_.value(), rank->stream_);
    if (!model_result.ok()) return model_result.status();
    rank->model_ = model_result.release();

    const shelf::qwen_hybrid::QwenHybridConfig& tc = rank->hybrid_config();
    const std::uint32_t max_blocks_per_sequence =
        ceil_div(config.max_seq_len + 1u, config.page_tokens) + 1u;
    const std::uint32_t attention_layers = count_attention_layers(tc);
    const std::uint32_t local_kv_heads = rank->context().local_key_value_heads;
    const std::uint32_t head_dim = static_cast<std::uint32_t>(tc.attention_head_dim);

    const std::size_t kv_page_bytes = PagedKVPool::page_bytes(
        config.page_tokens, attention_layers, local_kv_heads, head_dim,
        config.kv_cache_dtype, config.octave_kv_format);
    if (kv_page_bytes == 0u) {
        return Status::invalid_state("KV page geometry is zero", __FILE__, __LINE__);
    }

    const auto& rank_context = rank->context();
    const auto gdn_layout_result = make_qwen35_gdn_state_layout_local(
        tc, qwen35_stage_gdn_layers(tc, {}), rank_context.local_gdn_key_heads,
        rank_context.local_gdn_value_heads);

    if (config.load_private_weights && config.tp_rank == 0u) {
        Status private_status = config.load_private_weights(
            rank->arena_.value(), rank->stream_, rank->private_lm_head_,
            rank->private_embed_tokens_);
        if (!private_status.ok()) return private_status;
    }

    auto seq_result = SequenceSlotPool::create(
        rank->arena_.value(), config.max_scheduled_requests, max_blocks_per_sequence);
    if (!seq_result.ok()) return seq_result.status();
    rank->seq_pool_.emplace(seq_result.release());

    auto gdn_result = GdnStatePool::create(rank->arena_.value(),
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
        qwen35::ExecutorConfig reserve_config;
        reserve_config.max_batch_rows = config.max_batch_rows;
        reserve_config.max_scheduled_requests = config.max_scheduled_requests;
        reserve_config.max_scheduled_output_rows = config.max_scheduled_output_rows;
        reserve_config.target_hidden_tap_count = config.target_hidden_tap_count;
        reserve_config.exl3_linear_shape = exl3_linear_shape;
        // The dense FFN term of the shared estimate stands in for the FFN activations that
        // still sit at token rows: for a model with grouped experts those are its shared
        // expert. The routed experts add their own group-row term.
        reserved = qwen35::runtime::qwen35_executor_reserve_bytes(
            tc, rank->model_->token_row_ffn_size(), reserve_config, config.tp_size,
            rank->model_->uses_exl3_linear(),
            rank->model_->qwen4exp_model() != nullptr);
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
    if (config.prefix_cache) {
        PrefixCacheConfig prefix_config;
        prefix_config.max_concurrent = config.prefix_concurrency;
        prefix_config.spec_ring_bytes = config.prefix_spec_ring_bytes;
        const PrefixCacheBytes prefix_bytes =
            prefix_cache_bytes(prefix_config, gdn_layout_result);
        reserved += prefix_bytes.total();
        const std::size_t family_shadow =
            rank->model_->executor_state_reserve_bytes(
                2u * static_cast<std::size_t>(config.prefix_concurrency));
        reserved += family_shadow;
        std::fprintf(stderr, "MEMORY_PLAN tp prefix cache: %s (family shadow %s)\n",
                     gpu::format_gib(prefix_bytes.total()).c_str(),
                     gpu::format_gib(family_shadow).c_str());
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
        const DFlash2Footprint footprint = dflash2_footprint(
            *config.drafter_config, spec_config, *rank->gdn_pool_,
            config.drafter_weights_bytes, lm_head_k,
            config.drafter_sessions == 0u ? config.max_scheduled_requests
                                          : config.drafter_sessions,
            0u, exl3_mul24_head);
        reserved += footprint.total();
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
        qwen35::runtime::MtpDrafterConfig mtp_options;
        mtp_options.num_drafts = config.mtp_batch.num_drafts;
        mtp_options.max_seq_len = config.mtp_batch.max_seq_len;
        mtp_options.page_tokens = config.mtp_batch.page_tokens;
        mtp_options.max_batch_rows = config.mtp_batch.max_batch_rows;
        mtp_options.kv_dtype = config.mtp_batch.kv_dtype;
        std::size_t drafter_bytes = 0u;
        if (config.tp_rank == 0u) {
            drafter_bytes = qwen35::runtime::mtp_drafter_reserve_bytes(
                                tc, mtp_options, rank->model_->token_row_ffn_size(),
                                rank->model_->mtp_grouped_reserve_bytes(
                                    qwen35::runtime::mtp_drafter_rows(mtp_options),
                                    mtp_options.max_seq_len + 1u),
                                rank->model_->uses_exl3_linear(),
                                rank->model_->mtp_exl3_linear_shape(),
                                &gdn_layout_result) *
                            config.mtp_batch.max_requests;
        }
        const std::size_t state_bytes =
            qwen35::runtime::mtp_batch_rank_state_reserve_bytes(
                gdn_layout_result,
                qwen35::runtime::mtp_spec_rows(config.mtp_batch.max_requests,
                                               mtp_options.num_drafts),
                config.mtp_batch.max_requests);
        const std::size_t batch_bytes = drafter_bytes + state_bytes;
        reserved += batch_bytes;
        std::fprintf(stderr,
                     "MEMORY_PLAN tp mtp batch reserve: %s (%u slots, drafter %s)\n",
                     gpu::format_gib(batch_bytes).c_str(),
                     static_cast<unsigned>(config.mtp_batch.max_requests),
                     gpu::format_gib(drafter_bytes).c_str());
    }
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
    const std::uint32_t kv_pool_pages = fit_pages + 1u;

    auto kv_result = PagedKVPool::create(
        rank->arena_.value(), kv_pool_pages, config.page_tokens, attention_layers,
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
    exec_config.rope_profile = config.rope_profile;
    exec_config.tp = &rank->context();
    exec_config.exl3_linear_shape = exl3_linear_shape;

    const std::size_t executor_before = rank->arena_->used();
    auto exec_result = rank->model_->create_executor(
        rank->seq_pool_.value(), rank->gdn_pool_.value(), rank->kv_pool_.value(),
        rank->arena_.value(), exec_config);
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

namespace {

class DenseTpRankModel final : public shelf::models::TpRankModel {
public:
    DenseTpRankModel(Qwen35Model&& model, Qwen35TensorParallelContext&& context)
        : model_(std::move(model)), context_(context) {}

    const shelf::qwen_hybrid::QwenHybridConfig& hybrid_config() const override {
        return model_.text_config();
    }
    const Qwen35TensorParallelContext& context() const override { return context_; }

    std::uint64_t weight_bytes() const override {
        const Qwen35ModelWeights& w = model_.weights();
        std::uint64_t total = 0;
        add_matrix_bytes(total, w.embed_tokens);
        if (!w.lm_head_tied) add_matrix_bytes(total, w.lm_head);
        for (const Qwen35LayerWeights& layer : w.layers) {
            add_matrix_bytes(total, layer.ffn.gate);
            add_matrix_bytes(total, layer.ffn.up);
            add_matrix_bytes(total, layer.ffn.down);
            if (layer.is_gdn) {
                add_matrix_bytes(total, layer.attn_in_proj_qkv);
                add_matrix_bytes(total, layer.attn_in_proj_z);
                add_matrix_bytes(total, layer.attn_in_proj_a);
                add_matrix_bytes(total, layer.attn_in_proj_b);
                add_matrix_bytes(total, layer.attn_out_proj);
            } else {
                add_matrix_bytes(total, layer.attn_q_proj);
                add_matrix_bytes(total, layer.attn_k_proj);
                add_matrix_bytes(total, layer.attn_v_proj);
                add_matrix_bytes(total, layer.attn_o_proj);
            }
        }
        return total;
    }

    std::size_t token_row_ffn_size() const override {
        return model_.text_config().ffn.intermediate_size;
    }

    std::size_t ffn_workspace_reserve_bytes(std::size_t) const override { return 0u; }

    bool uses_exl3_linear() const override {
        return ::shelf::qwen35::uses_exl3_linear(model_.weights());
    }

    shelf::weights::Exl3LinearShape exl3_linear_shape() const override {
        return ::shelf::qwen35::exl3_linear_shape(model_.weights());
    }

    Result<Executor> create_executor(SequenceSlotPool& sequence_pool,
                                     GdnStatePool& gdn_state_pool, PagedKVPool& kv_pool,
                                     gpu::GpuArena& arena,
                                     const ExecutorConfig& config) const override {
        return create_model_executor(model_, sequence_pool, gdn_state_pool, kv_pool, arena,
                                     config);
    }

    const Qwen35Model* dense_model() const override { return &model_; }

private:
    Qwen35Model model_;
    Qwen35TensorParallelContext context_;
};

}  // namespace

Result<std::unique_ptr<shelf::models::TpRankModel>> build_dense_tp_rank_model(
    const TpRankRuntimeConfig& config, gpu::GpuArena& arena, hipStream_t stream) {
    auto model_result = Qwen35Model::load_tensor_parallel_from_safetensors(
        config.model_dir, config.tp_size, config.tp_rank, arena, stream, config.load_options);
    if (!model_result.ok()) return model_result.status();

    auto context_result = make_qwen35_tensor_parallel_context(
        model_result.value().text_config(), config.tp_size, config.tp_rank,
        config.load_options.vocab_parallel_lm_head);
    if (!context_result.ok()) return context_result.status();

    return std::unique_ptr<shelf::models::TpRankModel>(std::make_unique<DenseTpRankModel>(
        model_result.release(), context_result.release()));
}

const shelf::models::TpRankModel* TpRankRuntime::rank_model() const { return model_.get(); }

const Qwen35Model* TpRankRuntime::dense_model() const {
    return model_ ? model_->dense_model() : nullptr;
}

const shelf::qwen_hybrid::QwenHybridConfig& TpRankRuntime::hybrid_config() const {
    return model_->hybrid_config();
}

const Qwen35TensorParallelContext& TpRankRuntime::context() const { return model_->context(); }

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
    return transport_ != nullptr ? transport_->name() : "none";
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

    std::jthread model_prefetch;
    if (config.prefetch_rank_model != nullptr) {
        model_prefetch = std::jthread(
            [model_dir = config.model_dir, prefetch = config.prefetch_rank_model]() {
                prefetch(model_dir);
            });
    }

    auto transport_result = ::shelf::runtime::create_tp_transport(devices_);
    if (!transport_result.ok()) return transport_result.status();
    transport_ = std::move(transport_result.value().transport);

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
        rank_config.tp_size = static_cast<std::uint32_t>(devices_.size());
        rank_config.tp_rank = static_cast<std::uint32_t>(r);
        rank_config.device_id = devices_[r];
        rank_config.max_batch_rows = config.max_batch_rows;
        rank_config.max_scheduled_requests = config.max_scheduled_requests;
        rank_config.max_scheduled_output_rows = config.max_scheduled_output_rows;
        rank_config.max_seq_len = config.max_seq_len;
        rank_config.page_tokens = config.page_tokens;
        rank_config.kv_cache_dtype = config.kv_cache_dtype;
        rank_config.octave_kv_format = config.octave_kv_format;
        rank_config.arena_bytes = config.arena_bytes;
        rank_config.working_set_reserve_bytes = config.working_set_reserve_bytes;
        rank_config.drafter_config = config.drafter_config;
        rank_config.drafter_weights_bytes = config.drafter_weights_bytes;
        rank_config.drafter_drafts = config.drafter_drafts;
        rank_config.drafter_sessions = config.drafter_sessions;
        rank_config.drafter_lm_head_k = config.drafter_lm_head_k;
        rank_config.prefix_cache = config.prefix_cache;
        rank_config.prefix_concurrency = config.prefix_concurrency;
        rank_config.prefix_spec_ring_bytes = config.prefix_spec_ring_bytes;
        rank_config.load_private_weights = config.load_private_weights;
        rank_config.target_hidden_taps = config.target_hidden_taps;
        rank_config.target_hidden_tap_count = config.target_hidden_tap_count;
        rank_config.emit_normed_token_hidden = config.emit_normed_token_hidden;
        rank_config.tp_combine_f32 = config.tp_combine_f32;
        rank_config.rope_profile = config.rope_profile;
        rank_config.build_rank_model = config.build_rank_model;
        rank_config.load_options = config.load_options;
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

    Status verify_status = transport_->verify();
    if (!verify_status.ok()) {
        std::fprintf(stderr,
                     "[tp] transport %s failed verification (%s); falling back to "
                     "host-mediated reference transport\n",
                     transport_->name(), verify_status.message().c_str());
        transport_ = std::make_unique<::shelf::runtime::HostMediatedTpTransport>(devices_);
    }

    barrier_group_ = std::make_unique<::shelf::runtime::TpBarrierGroup>(
        static_cast<int>(devices_.size()), devices_, transport_.get());
    Status barrier_status = barrier_group_->initialize();
    if (!barrier_status.ok()) return barrier_status;

    for (std::size_t r = 0; r < ranks_.size(); ++r) {
        ranks_[r]->executor().tp_barrier = barrier_group_.get();
        ranks_[r]->executor().tp_rank = static_cast<int>(r);
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
    if (barrier_group_) barrier_group_->shutdown();
    barrier_group_.reset();
    transport_.reset();
    for (auto& rank : ranks_) rank.reset();
    ranks_.clear();
}

void TpCoordinator::abort(const Status& reason) noexcept {
    if (barrier_group_) barrier_group_->abort(reason);
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
    if (barrier_group_->aborted()) {
        const Status reason = barrier_group_->abort_status();
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
    if (barrier_group_ == nullptr) {
        return Status::invalid_state("tp barrier group missing", __FILE__, __LINE__);
    }
    return barrier_group_->arrive(rank, target, stream, false);
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
        const ExecuteBatchOptions* options = nullptr;
        {
            std::unique_lock<std::mutex> lock(worker->mutex);
            worker->cv.wait(lock, [&] { return worker->job_pending || worker->stop; });
            if (worker->stop) break;
            worker->job_pending = false;
            job = worker->job;
            options = worker->options;
        }
        BatchExecutionOutput output;
        Status st = Status::make_ok();
        if (job != nullptr) {
            st = run_rank_step(rank, *job, output, options);
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
                                    BatchExecutionOutput& output,
                                    const ExecuteBatchOptions* options) {
    TpRankRuntime& runtime = *ranks_[rank];
    auto pending = options != nullptr
                       ? submit_batch(runtime.executor(), batch, runtime.stream(), *options)
                       : submit_batch(runtime.executor(), batch, runtime.stream());
    if (!pending.ok()) return pending.status();
    auto completed =
        complete_batch(runtime.executor(), pending.release(), runtime.stream());
    if (!completed.ok()) return completed.status();
    output = completed.release();
    return Status::make_ok();
}

std::string TpCoordinator::debug_report() const {
    std::ostringstream out;
    out << "tp_size=" << ranks_.size() << " transport="
        << (transport_ != nullptr ? transport_->name() : "none") << "\n";
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
}
