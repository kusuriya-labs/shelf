#include <shelf/models/tp_rank_model.h>

#include <shelf/models/model_weights_factory.h>
#include <shelf/models/qwen35moe/model/lower_to_primitives.h>
#include <shelf/models/qwen35moe/model/qwen35moe_config.h>
#include <shelf/models/qwen35moe/model/tensor_parallel_context.h>
#include <shelf/models/qwen35moe/runtime/executor_footprint.h>
#include <shelf/models/qwen35moe/weights/moe_tensor_parallel_plan.h>
#include <shelf/models/qwen4exp/model/lower_to_primitives.h>
#include <shelf/models/qwen4exp/runtime/executor_footprint.h>
#include <shelf/inference/state/ple_conv_state_pool.h>
#include <shelf/inference/state/qsa_state_pool.h>
#include <shelf/kernels/optimized/qsa.h>
#include <shelf/models/qwen35/runtime/executor_footprint.h>
#include <shelf/models/qwen4exp/runtime/ple_input.h>
#include <shelf/models/qwen4exp/runtime/moe_expert_cache.h>
#include <shelf/runtime/execution/row_bucket.h>
#include <shelf/runtime/graph/grouped_row_shape.h>
#include <shelf/runtime/program/moe_plan_scratch.h>

#include <memory>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace shelf::models {

namespace {

namespace q35 = shelf::qwen35;
namespace q35r = shelf::qwen35::runtime;
namespace qhy = shelf::qwen_hybrid;
namespace qmoe = shelf::qwen35moe;

class MoeTpRankModel final : public TpRankModel {
public:
    MoeTpRankModel(qmoe::Qwen35MoeModel&& model,
                   qhy::TensorParallelContext&& context)
        : model_(std::move(model)), context_(context) {}

    const qhy::QwenHybridConfig& hybrid_config() const override { return model_.text_config(); }
    const q35::Qwen35TensorParallelContext& context() const override { return context_; }
    const shelf::weights::MatrixWeight* lm_head() const override {
        return &model_.weights().lm_head;
    }
    const shelf::weights::MatrixWeight* embed_tokens() const override {
        return &model_.weights().embed_tokens;
    }

    std::uint64_t weight_bytes() const override {
        const qmoe::Qwen35MoeModelWeights& w = model_.weights();
        std::uint64_t total = 0;
        add_matrix_bytes(total, w.embed_tokens);
        if (!w.lm_head_tied) add_matrix_bytes(total, w.lm_head);
        for (const qmoe::Qwen35MoeLayerWeights& layer : w.layers) {
            add_matrix_bytes(total, layer.ffn.router);
            add_matrix_bytes(total, layer.ffn.shared_gate);
            add_matrix_bytes(total, layer.ffn.shared_gate_proj);
            add_matrix_bytes(total, layer.ffn.shared_up_proj);
            add_matrix_bytes(total, layer.ffn.shared_down_proj);
            add_matrix_bytes(total, layer.ffn.expert_w13);
            add_matrix_bytes(total, layer.ffn.expert_w2);
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

    std::uint64_t host_pinned_bytes() const override {
        return model_.weights().pinned_host.used();
    }

    // A model with grouped experts has no token-row FFN of the routed kind: what still sits
    // at token rows is its shared expert, and that is already this rank's local width.
    std::size_t token_row_ffn_size() const override {
        return context_.local_shared_expert_intermediate_size;
    }

    std::size_t ffn_workspace_reserve_bytes(std::size_t max_batch_rows) const override {
        return qmoe::moe_executor_grouped_bytes(model_.text_config(), max_batch_rows,
                                                context_.tp_size, context_.ep_size);
    }

    bool uses_exl3_linear() const override { return qmoe::uses_exl3_linear(model_.weights()); }

    std::size_t mtp_grouped_reserve_bytes(std::size_t rows,
                                          std::uint32_t) const override {
        return qmoe::moe_executor_grouped_bytes(model_.text_config(), rows, 1u);
    }

    shelf::weights::Exl3LinearShape exl3_linear_shape() const override {
        return qmoe::exl3_linear_shape(model_.weights());
    }

    shelf::Result<q35::Executor> create_executor(
        q35::SequenceSlotPool& sequence_pool, q35::GdnStatePool& gdn_state_pool,
        q35::PagedKVPool& kv_pool, gpu::GpuArena& arena,
        const q35::ExecutorConfig& config) const override {
        return create_moe_model_executor(model_, sequence_pool, gdn_state_pool, kv_pool, arena,
                                         config);
    }

    const qmoe::Qwen35MoeModel* grouped_model() const override { return &model_; }

private:
    static void add_matrix_bytes(std::uint64_t& total, const weights::MatrixWeight& w) {
        if (w.encoding == weights::MatrixEncoding::Bf16) {
            total += static_cast<std::uint64_t>(w.rows) * w.cols * 2u;
            return;
        }
        const auto add = [&total](const gpu::Tensor& t) {
            if (t.ndim() == 0) return;
            std::size_t bytes = 0;
            if (!t.required_span_bytes(bytes).ok()) return;
            total += bytes;
        };
        add(w.codes);
        add(w.scales);
        add(w.row_ref);
        add(w.compute_codes);
        add(w.compute_scales_bf16);
    }

    qmoe::Qwen35MoeModel model_;
    qhy::TensorParallelContext context_;
};

std::shared_ptr<const qwen4exp::PleTable> shared_ple_table(
    const std::string& model_dir, const qwen4exp::Qwen4ExpTextConfig& config) {
    static std::mutex mutex;
    static std::unordered_map<std::string, std::weak_ptr<const qwen4exp::PleTable>> cache;
    std::lock_guard<std::mutex> lock(mutex);
    auto& entry = cache[model_dir];
    if (auto existing = entry.lock()) return existing;
    auto loaded = qwen4exp::PleTable::load(model_dir, config);
    if (!loaded.ok()) return nullptr;
    auto table = std::make_shared<qwen4exp::PleTable>(loaded.release());
    entry = table;
    return table;
}

std::mutex prefetched_ple_mutex;
std::unordered_map<std::string, std::shared_ptr<const qwen4exp::PleTable>> prefetched_ple;

struct Qwen4ExpStatePoolLayouts {
    bool uses_qsa = false;
    bool uses_ple_conv = false;
    ::shelf::state::QsaKeyStatePoolLayout qsa{};
    ::shelf::state::PleConvStatePoolLayout ple_conv{};
};

Qwen4ExpStatePoolLayouts qwen4exp_state_pool_layouts(
    const qwen4exp::Qwen4ExpTextConfig& cfg, std::uint32_t max_context_tokens,
    const qhy::PipelineStage& stage = {}) {
    const auto counts = qhy::count_layer_topology(cfg);
    std::uint32_t attention_layers = counts.attention_layers;
    std::uint32_t ple_layers = static_cast<std::uint32_t>(cfg.ple.layer_ids.size());
    if (stage.enabled) {
        attention_layers = 0u;
        ple_layers = 0u;
        for (std::uint32_t layer = stage.layer_lo(); layer < stage.layer_hi(); ++layer) {
            if (!qhy::layer_is_gdn(cfg, layer)) ++attention_layers;
            if (cfg.layer_has_ple(layer)) ++ple_layers;
        }
    }
    Qwen4ExpStatePoolLayouts layouts;
    layouts.uses_qsa = attention_layers != 0u;
    if (layouts.uses_qsa) {
        const std::uint32_t ratio = static_cast<std::uint32_t>(cfg.indexer.compress_ratio);
        const std::uint32_t blocks = (max_context_tokens + ratio - 1u) / ratio;
        layouts.qsa = ::shelf::state::QsaKeyStatePoolLayout::make(
            attention_layers, ::shelf::kernel::kQsaHeadDim, ratio, blocks);
    }
    layouts.uses_ple_conv = ple_layers != 0u;
    if (layouts.uses_ple_conv) {
        layouts.ple_conv = ::shelf::state::PleConvStatePoolLayout::make(
            ple_layers, static_cast<std::uint32_t>(cfg.hc_count * cfg.hidden_size));
    }
    return layouts;
}

class Qwen4ExpTpRankModel final : public TpRankModel {
public:
    Qwen4ExpTpRankModel(qwen4exp::Qwen4ExpModel&& model,
                        qhy::TensorParallelContext&& context,
                        std::uint32_t max_context_tokens,
                        std::shared_ptr<const qwen4exp::PleTable> ple_table)
        : model_(std::move(model)), context_(context),
          max_context_tokens_(max_context_tokens), ple_table_(std::move(ple_table)) {
        if (model_.weights().host_experts != 0u) {
            cache_ = qwen4exp::MoeExpertCache::create(model_.weights());
        }
    }

    const qhy::QwenHybridConfig& hybrid_config() const override {
        return model_.text_config();
    }
    const q35::Qwen35TensorParallelContext& context() const override { return context_; }
    const shelf::weights::MatrixWeight* lm_head() const override {
        return &model_.weights().lm_head;
    }
    const shelf::weights::MatrixWeight* embed_tokens() const override {
        return &model_.weights().embed_tokens;
    }

    std::uint64_t weight_bytes() const override {
        const qwen4exp::Qwen4ExpRankWeights& w = model_.weights();
        std::uint64_t total = 0;
        add_matrix(total, w.embed_tokens);
        add_matrix(total, w.lm_head);
        add_hyper(total, w.mixer);
        for (const auto& layer : w.layers) {
            add_hyper(total, layer.attn_hyper_connection);
            add_hyper(total, layer.mlp_hyper_connection);
            if (layer.attention.is_gdn) {
                add_matrix(total, layer.attention.in_proj_qkv);
                add_matrix(total, layer.attention.in_proj_z);
                add_matrix(total, layer.attention.in_proj_b);
                add_matrix(total, layer.attention.in_proj_a);
                add_matrix(total, layer.attention.out_proj);
            } else {
                add_matrix(total, layer.attention.q_proj);
                add_matrix(total, layer.attention.k_proj);
                add_matrix(total, layer.attention.v_proj);
                add_matrix(total, layer.attention.o_proj);
                add_matrix(total, layer.attention.index_qk_proj);
            }
            add_matrix(total, layer.mlp.router);
            add_matrix(total, layer.mlp.shared_gate);
            add_matrix(total, layer.mlp.shared_gate_proj);
            add_matrix(total, layer.mlp.shared_up_proj);
            add_matrix(total, layer.mlp.shared_down_proj);
            add_matrix(total, layer.mlp.gate_experts);
            add_matrix(total, layer.mlp.up_experts);
            add_matrix(total, layer.mlp.down_experts);
            if (layer.ple.present) {
                add_matrix(total, layer.ple.key_proj);
                add_matrix(total, layer.ple.value_proj);
            }
        }
        return total;
    }

    std::uint64_t host_pinned_bytes() const override { return 0u; }

    std::uint64_t host_offload_expert_bytes() const override {
        return model_.weights().host_expert_bytes();
    }

    std::uint64_t host_ple_table_bytes() const override {
        return ple_table_ != nullptr ? ple_table_->host_bytes() : 0u;
    }

    std::size_t token_row_ffn_size() const override {
        return model_.text_config().ffn.shared_expert_intermediate_size;
    }

    std::size_t ffn_workspace_reserve_bytes(std::size_t max_batch_rows) const override {
        const auto& cfg = model_.text_config();
        const std::uint32_t local_experts = context_.local_num_experts;
        const std::uint32_t top_k = static_cast<std::uint32_t>(cfg.ffn.num_experts_per_tok);
        const std::uint32_t tokens = static_cast<std::uint32_t>(max_batch_rows);
        const std::uint32_t hidden = static_cast<std::uint32_t>(cfg.hidden_size);
        const std::uint32_t inter = context_.local_moe_intermediate_size;
        const std::uint32_t group_rows = ::shelf::runtime::grouped_row_capacity(
            tokens, ::shelf::runtime::GroupedRowShape{local_experts, top_k});
        std::size_t total = static_cast<std::size_t>(group_rows) * inter * 2u * 2u;
        total += static_cast<std::size_t>(group_rows) * hidden * 2u;
        const std::size_t rows = ::shelf::qwen35::runtime::qwen35_padded_row_count(tokens);
        const std::size_t blocks = qsa_blocks(cfg);
        const std::size_t mask_bytes = ((blocks + 31u) / 32u) * 4u;
        total += rows * ((mask_bytes + 15u) / 16u) * 16u;
        total += ::shelf::runtime::MoePlanScratch::make(tokens, local_experts, top_k).bytes();
        return total;
    }

    std::size_t executor_state_reserve_bytes(
        std::size_t max_scheduled_requests) const override {
        const auto layouts = qwen4exp_state_pool_layouts(model_.text_config(),
                                                         max_context_tokens_);
        const std::uint32_t slots = static_cast<std::uint32_t>(max_scheduled_requests);
        std::size_t total = 0u;
        if (layouts.uses_qsa)
            total += ::shelf::state::qsa_key_state_pool_bytes(slots, layouts.qsa).total();
        if (layouts.uses_ple_conv)
            total += ::shelf::state::ple_conv_state_pool_bytes(slots, layouts.ple_conv);
        return total;
    }

    bool uses_qsa_state() const override {
        return qwen4exp_state_pool_layouts(model_.text_config(), max_context_tokens_).uses_qsa;
    }

private:
    std::size_t qsa_blocks(const qwen4exp::Qwen4ExpTextConfig& cfg) const {
        const std::size_t ratio = cfg.indexer.compress_ratio != 0u
                                      ? cfg.indexer.compress_ratio
                                      : 1u;
        return (max_context_tokens_ + ratio - 1u) / ratio;
    }

public:
    bool uses_exl3_linear() const override { return true; }

    std::size_t mtp_grouped_reserve_bytes(std::size_t rows,
                                          std::uint32_t max_context_tokens) const override {
        return qwen4exp::mtp_grouped_executor_bytes(
            model_.text_config(), rows, max_context_tokens, context_.local_num_experts);
    }

    shelf::weights::Exl3LinearShape exl3_linear_shape() const override {
        return qwen4exp::exl3_linear_shape(model_.weights());
    }

    shelf::weights::Exl3LinearShape mtp_exl3_linear_shape() const override {
        return qwen4exp::mtp_exl3_linear_shape(model_.weights());
    }

    shelf::Result<q35r::MtpDrafterInput> mtp_drafter_input(
        const q35r::MtpDrafterConfig& options) const override {
        const auto& cfg = model_.text_config();
        qwen4exp::Qwen4ExpLowerOptions lower_options;
        lower_options.max_context_tokens = options.max_seq_len + 1u;
        lower_options.tp_size = context_.tp_size;
        lower_options.tp_rank = context_.tp_rank;
        lower_options.ep_size = context_.ep_size;
        lower_options.ep_rank = context_.ep_rank;
        lower_options.ep_enabled = context_.ep_enabled;
        lower_options.rope_attention_factor = options.rope_profile.attention_factor;
        auto lowered = qwen4exp::lower_qwen4exp_mtp_to_primitives(
            cfg, model_.weights(), lower_options);
        if (!lowered.ok()) return lowered.status();
        q35r::MtpDrafterInput input;
        input.config = &hybrid_config();
        input.embed_tokens = &model_.weights().embed_tokens;
        input.lowered = lowered.release();
        input.token_row_intermediate =
            static_cast<std::size_t>(cfg.ffn.shared_expert_intermediate_size);
        input.grouped_executor_bytes = qwen4exp::mtp_grouped_executor_bytes(
            cfg, q35r::mtp_drafter_rows(options), 0u, context_.local_num_experts);
        input.exl3_linear_shape = mtp_exl3_linear_shape();
        input.hidden_width =
            static_cast<std::uint32_t>(cfg.hc_count * cfg.hidden_size);
        input.wide_hidden = true;
        return input;
    }

    std::uint32_t normed_token_hidden_width() const override {
        return static_cast<std::uint32_t>(model_.text_config().hc_count *
                                          model_.text_config().hidden_size);
    }

    Result<q35::Executor> create_executor(
        q35::SequenceSlotPool& sequence_pool, q35::GdnStatePool& gdn_state_pool,
        q35::PagedKVPool& kv_pool, gpu::GpuArena& arena,
        const q35::ExecutorConfig& config) const override {
        auto created = create_qwen4exp_model_executor(
            model_, ple_table_, sequence_pool, gdn_state_pool, kv_pool, arena, config);
        if (created.ok() && cache_ != nullptr) {
            auto cache = cache_;
            created.value().after_batch = [cache](hipStream_t stream) {
                return cache->step(stream);
            };
        }
        return created;
    }

    const qwen4exp::Qwen4ExpModel* qwen4exp_model() const override { return &model_; }

private:
    static void add_matrix(std::uint64_t& total, const weights::MatrixWeight& w) {
        if (w.encoding == weights::MatrixEncoding::Bf16) {
            total += static_cast<std::uint64_t>(w.rows) * w.cols * 2u;
            return;
        }
        const auto add = [&total](const gpu::Tensor& t) {
            if (t.ndim() == 0) return;
            std::size_t bytes = 0;
            if (t.required_span_bytes(bytes).ok()) total += bytes;
        };
        add(w.codes);
        add(w.scales);
        add(w.row_ref);
        add(w.compute_codes);
        add(w.compute_scales_bf16);
        add(w.exl3_suh);
        add(w.exl3_svh);
        add(w.exl3_marker);
    }
    static void add_hyper(std::uint64_t& total,
                          const qwen4exp::Qwen4ExpHyperConnectionWeights& h) {
        add_matrix(total, h.input_mix_down);
        add_matrix(total, h.input_mix_up);
        if (h.block_inject.rows != 0u) add_matrix(total, h.block_inject);
        total += h.hc_norm.ndim() == 0u ? 0u : h.hc_norm.dim(0) * 2u;
    }
    qwen4exp::Qwen4ExpModel model_;
    qhy::TensorParallelContext context_;
    std::uint32_t max_context_tokens_ = 512u;
    std::shared_ptr<const qwen4exp::PleTable> ple_table_;
    std::shared_ptr<qwen4exp::MoeExpertCache> cache_;
};

}  // namespace

void prefetch_qwen4exp_ple_table(const std::string& model_dir) {
    auto config_result = qwen4exp::read_qwen4exp_text_config(model_dir);
    if (!config_result.ok()) return;
    auto table = shared_ple_table(model_dir, config_result.value());
    if (table == nullptr) return;
    std::lock_guard<std::mutex> lock(prefetched_ple_mutex);
    prefetched_ple[model_dir] = std::move(table);
}

void release_prefetched_qwen4exp_ple_table() {
    std::lock_guard<std::mutex> lock(prefetched_ple_mutex);
    prefetched_ple.clear();
}

std::shared_ptr<const qwen4exp::PleTable> get_qwen4exp_ple_table(
    const std::string& model_dir) {
    auto config_result = qwen4exp::read_qwen4exp_text_config(model_dir);
    if (!config_result.ok()) return nullptr;
    return shared_ple_table(model_dir, config_result.value());
}

Result<q35::Executor> create_qwen4exp_model_executor(
    const qwen4exp::Qwen4ExpModel& model,
    const std::shared_ptr<const qwen4exp::PleTable>& ple_table,
    q35::SequenceSlotPool& sequence_pool,
    q35::GdnStatePool& gdn_state_pool,
    q35::PagedKVPool& kv_pool,
    gpu::GpuArena& arena,
    const q35::ExecutorConfig& config) {
    const auto& cfg = model.text_config();
    qwen4exp::Qwen4ExpLowerOptions options;
    options.max_context_tokens =
        config.max_context_tokens != 0u ? config.max_context_tokens : 512u;
    options.tp_size = config.tp != nullptr ? config.tp->tp_size : 1u;
    options.tp_rank = config.tp != nullptr ? config.tp->tp_rank : 0u;
    options.ep_size = config.tp != nullptr ? config.tp->ep_size : 1u;
    options.ep_rank = config.tp != nullptr ? config.tp->ep_rank : 0u;
    options.ep_enabled = config.tp != nullptr && config.tp->ep_enabled;
    options.expert_tp_size =
        config.tp != nullptr ? config.tp->expert_tp_size : options.tp_size;
    options.expert_tp_rank =
        config.tp != nullptr ? config.tp->expert_tp_rank : options.tp_rank;
    options.rope_attention_factor = static_cast<float>(config.rope_profile.attention_factor);
    options.hidden_taps = std::span<const std::uint32_t>(
        config.target_hidden_taps.data(), config.target_hidden_tap_count);
    options.expose_wide_hidden = config.emit_normed_token_hidden;
    options.stage = config.stage;
    auto lowered = qwen4exp::lower_qwen4exp_to_primitives(cfg, model.weights(), options);
    if (!lowered.ok()) return lowered.status();

    q35::ExecutorModelInput input;
    input.model.config = &cfg;
    input.model.embed_tokens = &model.weights().embed_tokens;
    input.lowered = lowered.release();
    const auto state_layouts =
        qwen4exp_state_pool_layouts(cfg, options.max_context_tokens, options.stage);
    input.uses_qsa_state = state_layouts.uses_qsa;
    input.qsa_layers = state_layouts.qsa.num_qsa_states;
    input.qsa_head_dim = state_layouts.qsa.head_dim;
    input.qsa_compress_ratio = state_layouts.qsa.compress_ratio;
    input.qsa_max_blocks = state_layouts.qsa.max_blocks;
    input.uses_ple_conv_state = state_layouts.uses_ple_conv;
    input.ple_conv_layers = state_layouts.ple_conv.num_ple_states;
    input.ple_conv_channels = state_layouts.ple_conv.channels;

    auto executor = q35::create_executor(std::move(input), sequence_pool, gdn_state_pool,
                                         kv_pool, arena, config);
    if (!executor.ok()) return executor.status();
    if (state_layouts.uses_ple_conv && ple_table != nullptr) {
        auto runtime = qwen4exp::PleInputRuntime::create(
            ple_table, static_cast<std::uint32_t>(cfg.hidden_size));
        if (runtime == nullptr) {
            return Status::invalid_state("PLE input runtime is empty", __FILE__, __LINE__);
        }
        executor.value().external_input_preparer =
            [runtime](const ::shelf::scheduling::ScheduledBatch& batch,
                      std::vector<bf16_t>& out) {
                return runtime->prepare(batch, out);
            };
        executor.value().external_prefix_restore =
            [runtime](::shelf::state::SequenceSlotId slot,
                      std::span<const int32_t> tokens) {
                return runtime->seed(slot, tokens);
            };
        executor.value().external_input_stateless =
            [ple_table](std::span<const std::int32_t> prefix_tail,
                        std::span<const std::int32_t> tokens,
                        std::vector<bf16_t>& out) {
                std::vector<std::int32_t> tail(prefix_tail.begin(), prefix_tail.end());
                auto rows = ple_table->gather_bf16_stateless(tail, tokens);
                if (!rows.ok()) return rows.status();
                out = std::move(rows.value());
                return Status::make_ok();
            };
    }
    return executor;
}

Result<q35::Executor> create_moe_model_executor(
    const qmoe::Qwen35MoeModel& model,
    q35::SequenceSlotPool& sequence_pool,
    q35::GdnStatePool& gdn_state_pool,
    q35::PagedKVPool& kv_pool,
    gpu::GpuArena& arena,
    const q35::ExecutorConfig& config) {
    std::uint32_t tp_size = 1u;
    std::uint32_t tp_rank = 0u;
    std::uint32_t ep_size = 1u;
    std::uint32_t ep_rank = 0u;
    bool ep_enabled = false;
    if (config.tp != nullptr) {
        tp_size = config.tp->tp_size;
        tp_rank = config.tp->tp_rank;
        ep_size = config.tp->ep_size;
        ep_rank = config.tp->ep_rank;
        ep_enabled = config.tp->ep_enabled;
    }
    if ((tp_size > 1u || ep_size > 1u) && config.tp != nullptr &&
        config.tp->global_num_experts == 0u) {
        return Status::unsupported(
            "a model with grouped experts needs a tensor parallel context that states its "
            "expert range: the expert planes are not part of the matrix partition contract",
            __FILE__, __LINE__);
    }

    std::vector<q35::LayerOverrideSource> overrides =
        q35::read_layer_override_sources(static_cast<std::uint32_t>(model.text_config().hidden_size));
    std::vector<std::uint32_t> override_layers;
    override_layers.reserve(overrides.size());
    for (const auto& ov : overrides) override_layers.push_back(ov.layer);

    qmoe::Qwen35MoeLowerOptions lower_options;
    lower_options.hidden_taps = std::span<const std::uint32_t>(
        config.target_hidden_taps.data(), config.target_hidden_tap_count);
    lower_options.override_layers =
        std::span<const std::uint32_t>(override_layers.data(), override_layers.size());
    lower_options.rope_attention_factor = static_cast<float>(config.rope_profile.attention_factor);
    lower_options.emit_normed_token_hidden = config.emit_normed_token_hidden;
    lower_options.tp_size = tp_size;
    lower_options.tp_rank = tp_rank;
    lower_options.ep_size = ep_size;
    lower_options.ep_rank = ep_rank;
    lower_options.ep_enabled = ep_enabled;
    lower_options.stage = config.stage;

    auto lowered = qmoe::lower_qwen35_moe_to_primitives(model.text_config(), model.weights(),
                                                        lower_options);
    if (!lowered.ok()) return lowered.status();

    q35::ExecutorModelInput input;
    input.model.config = &model.text_config();
    input.model.embed_tokens = &model.weights().embed_tokens;
    input.lowered = lowered.release();
    input.layer_overrides = std::move(overrides);
    return q35::create_executor(std::move(input), sequence_pool, gdn_state_pool, kv_pool, arena,
                                config);
}

Result<std::unique_ptr<TpRankModel>> build_qwen4exp_tp_rank_model(
    const q35r::TpRankRuntimeConfig& config, gpu::GpuArena& arena, hipStream_t stream) {
    auto text_config = qwen4exp::read_qwen4exp_text_config(config.model_dir);
    if (!text_config.ok()) return text_config.status();
    const auto& cfg = text_config.value();
    const bool heads_divide_ep =
        config.ep_size > 1u &&
        (cfg.num_attention_heads % config.ep_size) == 0u &&
        (cfg.num_key_value_heads % config.ep_size) == 0u &&
        (cfg.linear_num_key_heads % config.ep_size) == 0u &&
        (cfg.linear_num_value_heads % config.ep_size) == 0u;
    const bool axes_overlap =
        config.ep_enabled && config.tp_size == 1u && heads_divide_ep;
    const std::uint32_t backbone_tp_size =
        axes_overlap ? config.ep_size : config.tp_size;
    const std::uint32_t backbone_tp_rank =
        axes_overlap ? config.ep_rank : config.tp_rank;
    qwen4exp::Qwen4ExpLoadOptions load_options;
    load_options.load_mtp = config.load_mtp_layers;
    load_options.mtp_split = config.mtp_split;
    load_options.ep_size = config.ep_size;
    load_options.ep_rank = config.ep_rank;
    load_options.ep_enabled = config.ep_enabled;
    load_options.backbone_tp_size = backbone_tp_size;
    load_options.backbone_tp_rank = backbone_tp_rank;
    if (const char* host_experts = std::getenv("SHELF_MOE_HOST_EXPERTS")) {
        load_options.moe_host_experts =
            static_cast<std::uint32_t>(std::strtoul(host_experts, nullptr, 10));
    }
    if (const char* lm_head_int8 = std::getenv("SHELF_LM_HEAD_INT8")) {
        load_options.quantize_lm_head_int8_w32 =
            std::strtoul(lm_head_int8, nullptr, 10) != 0u;
    }
    auto model = qwen4exp::Qwen4ExpModel::load_from_safetensors(
        config.model_dir, config.tp_size, config.tp_rank, arena, stream,
        load_options, config.ep_size, config.ep_rank);
    if (!model.ok()) return model.status();

    qhy::TensorParallelContext context;
    context.tp_size = backbone_tp_size;
    context.tp_rank = backbone_tp_rank;
    context.local_attention_heads =
        static_cast<std::uint32_t>(cfg.num_attention_heads / backbone_tp_size);
    context.local_key_value_heads =
        static_cast<std::uint32_t>(cfg.num_key_value_heads / backbone_tp_size);
    context.local_gdn_key_heads =
        static_cast<std::uint32_t>(cfg.linear_num_key_heads / backbone_tp_size);
    context.local_gdn_value_heads =
        static_cast<std::uint32_t>(cfg.linear_num_value_heads / backbone_tp_size);
    context.local_vocab_size = static_cast<std::uint32_t>(cfg.vocab_size);
    context.vocab_parallel_lm_head = false;
    context.ep_size = config.ep_size;
    context.ep_rank = config.ep_rank;
    context.ep_enabled = config.ep_enabled;
    context.axes_overlap = axes_overlap;
    context.expert_tp_size = config.tp_size;
    context.expert_tp_rank = config.tp_rank;
    context.global_num_experts = static_cast<std::uint32_t>(cfg.ffn.num_experts);
    if (config.ep_size == 0u || config.ep_rank >= config.ep_size ||
        (cfg.ffn.num_experts % config.ep_size) != 0u) {
        return Status::unsupported("Qwen4Exp experts must divide by ep_size", __FILE__,
                                   __LINE__);
    }
    context.local_num_experts =
        static_cast<std::uint32_t>(cfg.ffn.num_experts / config.ep_size);
    context.expert_begin = config.ep_rank * context.local_num_experts;
    context.local_shared_expert_intermediate_size =
        static_cast<std::uint32_t>(cfg.ffn.shared_expert_intermediate_size);
    if ((cfg.ffn.moe_intermediate_size % config.tp_size) != 0u) {
        return Status::unsupported(
            "Qwen4Exp routed expert intermediate size must divide by tp_size", __FILE__,
            __LINE__);
    }
    context.local_moe_intermediate_size =
        static_cast<std::uint32_t>(cfg.ffn.moe_intermediate_size / config.tp_size);

    auto ple_table = shared_ple_table(config.model_dir, cfg);
    if (ple_table == nullptr) {
        return Status::invalid_argument("Qwen4Exp PLE table could not be loaded", __FILE__,
                                        __LINE__);
    }
    return std::unique_ptr<TpRankModel>(std::make_unique<Qwen4ExpTpRankModel>(
        model.release(), std::move(context), config.max_seq_len + 1u, ple_table));
}

Result<std::unique_ptr<TpRankModel>> build_moe_tp_rank_model(
    const q35r::TpRankRuntimeConfig& config, gpu::GpuArena& arena, hipStream_t stream) {
    if (config.pipeline_stage.enabled) {
        return Status::unsupported(
            "a model with grouped experts has no pipeline stage weight load", __FILE__,
            __LINE__);
    }
    if (config.load_mtp_layers && config.tp_size > 1u) {
        return Status::unsupported(
            "an MTP block is not partitioned across tensor parallel ranks", __FILE__, __LINE__);
    }

    auto config_result = qmoe::read_qwen35_moe_text_config(config.model_dir);
    if (!config_result.ok()) return config_result.status();
    const qmoe::Qwen35MoeTextConfig& text_config = config_result.value();

    // The routed experts split along the expert axis, which a matrix partition plan cannot
    // state, so the plan carries the backbone and the shared expert only. The MoE lm_head
    // stays replicated: the vocab-parallel head is implemented for the dense family.
    qmoe::Qwen35MoeLoadOptions load_options;
    static_cast<qhy::HybridLoadOptions&>(load_options) =
        static_cast<const qhy::HybridLoadOptions&>(config.load_options);
    load_options.load_mtp_layers = config.load_mtp_layers;
    load_options.ep_size = config.ep_size;
    load_options.ep_rank = config.ep_rank;
    load_options.ep_enabled = config.ep_enabled;
    // The plan outlives the load: the loader keeps a pointer to it, so it cannot be a
    // temporary inside the branch.
    std::optional<weights::TensorPartitionPlan> plan;
    if (config.tp_size > 1u) {
        auto plan_result = qmoe::build_qwen35moe_tensor_partition_plan(
            text_config, config.tp_size, config.tp_rank, false);
        if (!plan_result.ok()) return plan_result.status();
        plan.emplace(plan_result.release());
        load_options.weights.partition_plan = &plan.value();
    }

    auto model_result = qmoe::Qwen35MoeModel::load_from_safetensors(config.model_dir, arena,
                                                                    stream, load_options);
    if (!model_result.ok()) return model_result.status();

    auto context_result = qmoe::make_qwen35moe_tensor_parallel_context(
        text_config, config.tp_size, config.tp_rank, config.ep_size, config.ep_rank,
        config.ep_enabled, false);
    if (!context_result.ok()) return context_result.status();

    return std::unique_ptr<TpRankModel>(
        std::make_unique<MoeTpRankModel>(model_result.release(), context_result.release()));
}

namespace {

::shelf::state::QsaKeyStatePoolLayout qwen35qsa_state_layout(
    const qwen35qsa::Qwen35QsaTextConfig& config, std::uint32_t max_context_tokens) {
    const auto counts = qhy::count_layer_topology(config);
    const std::uint32_t ratio = static_cast<std::uint32_t>(config.indexer.block_size);
    const std::uint32_t blocks = ratio == 0u
        ? 0u
        : (max_context_tokens + ratio - 1u) / ratio;
    return ::shelf::state::QsaKeyStatePoolLayout::make(
        static_cast<std::uint32_t>(counts.attention_layers),
        static_cast<std::uint32_t>(config.indexer.head_dim), ratio, blocks);
}

class Qwen35QsaTpRankModel final : public TpRankModel {
public:
    Qwen35QsaTpRankModel(qwen35qsa::Qwen35QsaModel&& model,
                         qhy::TensorParallelContext&& context,
                         std::uint32_t max_context_tokens)
        : model_(std::move(model)), context_(context),
          max_context_tokens_(max_context_tokens) {}

    const qhy::QwenHybridConfig& hybrid_config() const override {
        return model_.text_config();
    }
    const q35::Qwen35TensorParallelContext& context() const override { return context_; }
    const shelf::weights::MatrixWeight* lm_head() const override {
        return &model_.weights().backbone.lm_head;
    }
    const shelf::weights::MatrixWeight* embed_tokens() const override {
        return &model_.weights().backbone.embed_tokens;
    }

    std::uint64_t weight_bytes() const override {
        const q35::Qwen35ModelWeights& w = model_.weights().backbone;
        std::uint64_t total = 0;
        add_matrix_bytes(total, w.embed_tokens);
        if (!w.lm_head_tied) add_matrix_bytes(total, w.lm_head);
        for (const q35::Qwen35LayerWeights& layer : w.layers) {
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
        for (const qwen35qsa::QsaIndexerLayerWeights& indexer : model_.weights().indexer) {
            add_matrix_bytes(total, indexer.q_proj);
            add_matrix_bytes(total, indexer.k_proj);
        }
        return total;
    }

    std::uint64_t host_pinned_bytes() const override {
        return model_.weights().backbone.pinned_host.used();
    }

    std::size_t token_row_ffn_size() const override {
        return model_.text_config().ffn.intermediate_size;
    }

    std::size_t ffn_workspace_reserve_bytes(std::size_t) const override { return 0u; }

    bool uses_exl3_linear() const override {
        return ::shelf::qwen35::uses_exl3_linear(model_.weights().backbone);
    }

    shelf::weights::Exl3LinearShape exl3_linear_shape() const override {
        return ::shelf::qwen35::exl3_linear_shape(model_.weights().backbone);
    }

    bool uses_qsa_state() const override {
        return max_context_tokens_ > model_.text_config().indexer.budget;
    }

    std::size_t executor_state_reserve_bytes(
        std::size_t max_scheduled_requests) const override {
        if (!uses_qsa_state()) return 0u;
        const std::uint32_t requested = static_cast<std::uint32_t>(max_scheduled_requests);
        const std::uint32_t slots = requested == 0u ? 1u : requested;
        const auto layout = qwen35qsa_state_layout(model_.text_config(), max_context_tokens_);
        return ::shelf::state::qsa_key_state_pool_bytes(slots, layout).total();
    }

    shelf::Result<q35::Executor> create_executor(
        q35::SequenceSlotPool& sequence_pool, q35::GdnStatePool& gdn_state_pool,
        q35::PagedKVPool& kv_pool, gpu::GpuArena& arena,
        const q35::ExecutorConfig& config) const override {
        return qwen35qsa::create_qwen35qsa_model_executor(
            model_, sequence_pool, gdn_state_pool, kv_pool, arena, config);
    }

    const qwen35qsa::Qwen35QsaModel* qsa_model() const override { return &model_; }

private:
    static void add_matrix_bytes(std::uint64_t& total, const weights::MatrixWeight& w) {
        if (w.rows == 0) return;
        if (w.encoding == weights::MatrixEncoding::Bf16) {
            total += static_cast<std::uint64_t>(w.rows) * w.cols * 2u;
            return;
        }
        const auto add = [&total](const gpu::Tensor& t) {
            if (t.ndim() == 0) return;
            std::size_t bytes = 0;
            if (!t.required_span_bytes(bytes).ok()) return;
            total += bytes;
        };
        add(w.codes);
        add(w.scales);
        add(w.row_ref);
        add(w.compute_codes);
        add(w.compute_scales_bf16);
    }

    qwen35qsa::Qwen35QsaModel model_;
    qhy::TensorParallelContext context_;
    std::uint32_t max_context_tokens_ = 0;
};

Result<std::unique_ptr<TpRankModel>> build_qwen35qsa_tp_rank_model(
    const q35r::TpRankRuntimeConfig& config, gpu::GpuArena& arena, hipStream_t stream) {
    auto text_config = qwen35qsa::read_qwen35qsa_text_config(config.model_dir);
    if (!text_config.ok()) return text_config.status();
    qwen35qsa::Qwen35QsaLoadOptions load_options(config.load_options);
    load_options.stage = config.pipeline_stage;
    auto model = qwen35qsa::Qwen35QsaModel::load_tensor_parallel_from_safetensors(
        config.model_dir, config.tp_size, config.tp_rank, arena, stream, load_options);
    if (!model.ok()) return model.status();

    auto context = q35::make_qwen35_tensor_parallel_context(
        text_config.value(), config.tp_size, config.tp_rank,
        config.load_options.vocab_parallel_lm_head);
    if (!context.ok()) return context.status();

    return std::unique_ptr<TpRankModel>(
        std::make_unique<Qwen35QsaTpRankModel>(
            model.release(), context.release(), config.max_seq_len + 1u));
}

}  // namespace

Status select_tp_rank_model(q35r::TpCoordinatorConfig& config) {
    if (config.build_rank_model != nullptr) {
        return Status::invalid_argument("tp coordinator already has a rank model builder",
                                        __FILE__, __LINE__);
    }
    auto family = probe_model_dir(config.model_dir);
    if (!family.ok()) return family.status();
    if (family.value() == ModelFamily::Plamo3)
        return Status::unsupported("plamo3: runtime path is not connected", __FILE__, __LINE__);
    if (family.value() == ModelFamily::Qwen4Exp) {
        config.build_rank_model = &build_qwen4exp_tp_rank_model;
        config.prefetch_rank_model = &prefetch_qwen4exp_ple_table;
        config.release_rank_model = &release_prefetched_qwen4exp_ple_table;
        return Status::make_ok();
    }
    if (family.value() == ModelFamily::Qsa) {
        config.build_rank_model = &build_qwen35qsa_tp_rank_model;
        return Status::make_ok();
    }
    config.build_rank_model = family.value() == ModelFamily::Moe
        ? &build_moe_tp_rank_model
        : &q35r::build_dense_tp_rank_model;
    return Status::make_ok();
}

Result<TpGroup> TpGroup::create(const q35r::TpCoordinatorConfig& config_in) {
    q35r::TpCoordinatorConfig config = config_in;
    Status selected = select_tp_rank_model(config);
    if (!selected.ok()) return selected;

    auto coordinator_result = q35r::TpCoordinator::create(config);
    if (!coordinator_result.ok()) return coordinator_result.status();

    TpGroup group;
    group.coordinator_ = coordinator_result.release();
    const q35r::TpRankRuntime& rank0 = group.coordinator_->rank(0);
    const TpRankModel* model = rank0.rank_model();
    if (model == nullptr) {
        return Status::invalid_state("tp rank 0 built no model", __FILE__, __LINE__);
    }
    if (const auto* q4 = model->qwen4exp_model()) {
        group.model_.emplace(LoadedModel::borrow(*q4));
    } else if (const auto* qsa = model->qsa_model()) {
        group.model_.emplace(LoadedModel::borrow(*qsa));
    } else if (const auto* grouped = model->grouped_model()) {
        group.model_.emplace(LoadedModel::borrow(*grouped));
    } else if (const auto* dense = model->dense_model()) {
        group.model_.emplace(LoadedModel::borrow(*dense));
    } else {
        return Status::invalid_state("tp rank 0's model names no family", __FILE__, __LINE__);
    }
    return group;
}

}  // namespace shelf::models
