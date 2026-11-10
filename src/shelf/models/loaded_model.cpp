#include <shelf/models/loaded_model.h>
#include <shelf/models/plamo3/model/lower_to_primitives.h>
#include <shelf/models/qwen35/model/lower_to_primitives.h>
#include <shelf/models/qwen35/runtime/executor_footprint.h>
#include <shelf/models/tp_rank_model.h>
#include <shelf/models/qwen35moe/model/lower_to_primitives.h>
#include <shelf/models/qwen35moe/runtime/executor_footprint.h>
#include <shelf/models/qwen4exp/model/lower_to_primitives.h>
#include <shelf/models/qwen4exp/runtime/executor_footprint.h>

#include <algorithm>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace shelf::models {

const shelf::qwen_hybrid::QwenHybridConfig& LoadedModel::hybrid_config() const noexcept {
    if (plamo_ != nullptr) return plamo_->hybrid;
    if (qwen4exp_ != nullptr) {
        return static_cast<const shelf::qwen_hybrid::QwenHybridConfig&>(
            qwen4exp_->text_config());
    }
    return moe_ != nullptr ? static_cast<const shelf::qwen_hybrid::QwenHybridConfig&>(
                                 moe_->text_config())
                           : static_cast<const shelf::qwen_hybrid::QwenHybridConfig&>(
                                 dense_->text_config());
}

const shelf::weights::MatrixWeight& LoadedModel::embed_tokens() const noexcept {
    if (plamo_ != nullptr) return plamo_->weights.embed_tokens;
    if (qwen4exp_ != nullptr) return qwen4exp_->weights().embed_tokens;
    return moe_ != nullptr ? moe_->weights().embed_tokens : dense_->weights().embed_tokens;
}

const shelf::weights::MatrixWeight& LoadedModel::lm_head() const noexcept {
    if (plamo_ != nullptr) return plamo_->weights.lm_head;
    if (qwen4exp_ != nullptr) return qwen4exp_->weights().lm_head;
    return moe_ != nullptr ? moe_->weights().lm_head : dense_->weights().lm_head;
}

Result<LoadedModel> LoadedModel::load(
    const std::string& model_dir,
    gpu::GpuArena& arena,
    hipStream_t stream,
    bool verify_payload_crc,
    bool load_mtp) {

    auto family_result = probe_model_dir(model_dir);
    if (!family_result.ok()) return family_result.status();
    LoadedModel out;
    if (family_result.value() == ModelFamily::Plamo3) {
        if (load_mtp) return Status::unsupported("plamo3 has no MTP block", __FILE__, __LINE__);
        auto cfg = shelf::plamo3::read_config(model_dir);
        if (!cfg.ok()) return cfg.status();
        auto weights = shelf::plamo3::load_weights(model_dir, cfg.value(), arena, stream);
        if (!weights.ok()) return weights.status();
        out.plamo_ = std::make_unique<PlamoModel>();
        out.plamo_->config = cfg.release();
        out.plamo_->weights = weights.release();
        const auto& c = out.plamo_->config;
        auto& h = out.plamo_->hybrid;
        h.vocab_size = c.vocab_size; h.hidden_size = c.hidden_size;
        h.num_hidden_layers = c.num_hidden_layers;
        h.num_attention_heads = c.num_attention_heads; h.num_key_value_heads = c.num_key_value_heads;
        h.attention_head_dim = c.head_dim; h.rms_norm_eps = static_cast<float>(c.rms_norm_eps);
        h.rope_theta = static_cast<float>(c.local_rope.theta); h.partial_rotary_factor = 1.0f;
        h.layer_types.assign(c.num_hidden_layers, 1); h.stop_tokens = {1, 16};
        h.declared_model_type = "plamo3";
        return out;
    }
    if (family_result.value() == ModelFamily::Qwen4Exp) {
        auto loaded = shelf::qwen4exp::Qwen4ExpModel::load_from_safetensors(
            model_dir, 1u, 0u, arena, stream, shelf::qwen4exp::Qwen4ExpLoadOptions{});
        if (!loaded.ok()) return loaded.status();
        out.owns_qwen4exp_ =
            std::make_unique<shelf::qwen4exp::Qwen4ExpModel>(loaded.release());
        out.qwen4exp_ = out.owns_qwen4exp_.get();
        return out;
    }
    if (family_result.value() == ModelFamily::Moe) {
        shelf::qwen35moe::Qwen35MoeLoadOptions options;
        options.verify_quantized_payload_crc = verify_payload_crc;
        options.load_mtp_layers = load_mtp;
        auto loaded = shelf::qwen35moe::Qwen35MoeModel::load_from_safetensors(
            model_dir, arena, stream, options);
        if (!loaded.ok()) return loaded.status();
        out.owns_moe_ =
            std::make_unique<shelf::qwen35moe::Qwen35MoeModel>(loaded.release());
        out.moe_ = out.owns_moe_.get();
        return out;
    }

    shelf::qwen35::Qwen35LoadOptions options;
    options.verify_quantized_payload_crc = verify_payload_crc;
    options.load_mtp_layers = load_mtp;
    auto loaded =
        shelf::qwen35::Qwen35Model::load_from_safetensors(model_dir, arena, stream, options);
    if (!loaded.ok()) return loaded.status();
    out.owns_dense_ = std::make_unique<shelf::qwen35::Qwen35Model>(loaded.release());
    out.dense_ = out.owns_dense_.get();
    return out;
}

LoadedModel LoadedModel::borrow(const shelf::qwen35::Qwen35Model& model) {
    LoadedModel out;
    out.dense_ = &model;
    return out;
}

LoadedModel LoadedModel::borrow(const shelf::qwen35moe::Qwen35MoeModel& model) {
    LoadedModel out;
    out.moe_ = &model;
    return out;
}

LoadedModel LoadedModel::borrow(const shelf::qwen4exp::Qwen4ExpModel& model) {
    LoadedModel out;
    out.qwen4exp_ = &model;
    return out;
}

shelf::weights::Exl3LinearShape LoadedModel::exl3_linear_shape() const {
    if (qwen4exp_ != nullptr) {
        return shelf::qwen4exp::exl3_linear_shape(qwen4exp_->weights());
    }
    if (moe_ != nullptr) {
        return shelf::qwen35moe::exl3_linear_shape(moe_->weights());
    }
    if (dense_ != nullptr) {
        return shelf::qwen35::exl3_linear_shape(dense_->weights());
    }
    return {};
}

std::size_t LoadedModel::executor_reserve_bytes(std::size_t max_batch_rows,
                                                std::size_t max_scheduled_requests,
                                                std::size_t max_scheduled_output_rows,
                                                std::size_t target_hidden_tap_count,
                                                std::size_t tp_size) const {
    shelf::qwen35::ExecutorConfig reserve_config;
    reserve_config.max_batch_rows = static_cast<std::uint32_t>(max_batch_rows);
    reserve_config.max_scheduled_requests = static_cast<std::uint32_t>(max_scheduled_requests);
    reserve_config.max_scheduled_output_rows =
        static_cast<std::uint32_t>(max_scheduled_output_rows);
    reserve_config.target_hidden_tap_count = static_cast<std::uint32_t>(target_hidden_tap_count);
    reserve_config.exl3_linear_shape = exl3_linear_shape();

    // The dense FFN term of the shared estimate stands in for the FFN activations that
    // still sit at token rows: for a model with grouped experts those are its shared expert.
    const std::size_t token_row_ffn =
        plamo_ != nullptr ? plamo_->config.intermediate_size
        : qwen4exp_ != nullptr ? qwen4exp_->text_config().ffn.shared_expert_intermediate_size
        : moe_ != nullptr ? moe_->text_config().ffn.shared_expert_intermediate_size
                          : dense_->text_config().ffn.intermediate_size;
    const bool exl3 =
        plamo_ != nullptr ? false
        : qwen4exp_ != nullptr ? true
        : moe_ != nullptr ? shelf::qwen35moe::uses_exl3_linear(moe_->weights())
                          : shelf::qwen35::uses_exl3_linear(dense_->weights());
    std::size_t total = shelf::qwen35::runtime::qwen35_executor_reserve_bytes(
        hybrid_config(), token_row_ffn, reserve_config, tp_size, exl3);
    // PLaMo's combined projections and BF16 intermediate values need more scratch
    // than the Qwen estimate at small row buckets. Keep room after the KV plan.
    if (plamo_ != nullptr) total = std::max<std::size_t>(total, 1ull << 30);
    if (moe_ != nullptr)
        total += shelf::qwen35moe::moe_executor_grouped_bytes(moe_->text_config(),
                                                              max_batch_rows, tp_size);
    return total;
}

Result<shelf::qwen35::Executor> LoadedModel::create_executor(
    shelf::qwen35::SequenceSlotPool& sequence_pool,
    shelf::qwen35::GdnStatePool& gdn_state_pool,
    shelf::qwen35::PagedKVPool& kv_pool,
    gpu::GpuArena& arena,
    const shelf::qwen35::ExecutorConfig& config) const {

    if (plamo_ != nullptr) {
        if (config.tp != nullptr || config.stage.enabled || config.target_hidden_tap_count != 0 ||
            config.emit_normed_token_hidden || config.external_hidden_input)
            return Status::unsupported("plamo3: requested executor mode is not supported", __FILE__, __LINE__);
        auto lowered = shelf::plamo3::lower_to_primitives(plamo_->config, plamo_->weights);
        if (!lowered.ok()) return lowered.status();
        shelf::qwen35::ExecutorModelInput input;
        input.model.config = &plamo_->hybrid;
        input.model.embed_tokens = &plamo_->weights.embed_tokens;
        input.lowered = lowered.release();
        shelf::qwen35::ExecutorConfig resolved = config;
        const auto& c = plamo_->config;
        shelf::qwen35::EffectiveRopeProfile local;
        local.theta = c.local_rope.theta; local.rotary_dim = static_cast<uint32_t>(c.head_dim);
        local.float32_frequencies = true;
        shelf::qwen35::EffectiveRopeProfile full;
        full.type = c.full_rope.factor > 1.0
            ? shelf::qwen35::EffectiveRopeProfile::Type::Yarn
            : shelf::qwen35::EffectiveRopeProfile::Type::Default;
        full.theta = c.full_rope.theta; full.rotary_dim = static_cast<uint32_t>(c.head_dim);
        full.factor = c.full_rope.factor;
        full.original_max_position_embeddings = static_cast<uint32_t>(c.full_rope.original_max_position_embeddings);
        full.beta_fast = c.full_rope.beta_fast; full.beta_slow = c.full_rope.beta_slow;
        full.truncate = c.full_rope.truncate; full.float32_frequencies = true;
        resolved.rope_profiles = {local, full};
        return shelf::qwen35::create_executor(std::move(input), sequence_pool, gdn_state_pool,
                                               kv_pool, arena, resolved);
    }
    if (qwen4exp_ != nullptr) {
        return Status::unsupported(
            "Qwen4Exp runs through the tensor-parallel rank model, not LoadedModel",
            __FILE__, __LINE__);
    }
    shelf::qwen35::ExecutorConfig resolved = config;
    resolved.exl3_linear_shape = exl3_linear_shape();
    if (dense_ != nullptr) {
        return shelf::qwen35::create_model_executor(*dense_, sequence_pool, gdn_state_pool,
                                                    kv_pool, arena, resolved);
    }

    // The MoE family's lowering options, including the rank's expert range, are built by one
    // helper so a tensor parallel rank reaches the executor the same way.
    if (config.tp != nullptr && config.tp->tp_size > 1u && config.tp->global_num_experts == 0u) {
        return Status::unsupported(
            "a model with grouped experts needs a tensor parallel context that states its "
            "expert range: the expert planes are not part of the matrix partition contract",
            __FILE__, __LINE__);
    }
    return create_moe_model_executor(
        *moe_, sequence_pool, gdn_state_pool, kv_pool, arena, resolved);
}

namespace {

// The executor workspace terms of one drafter block: the FFN activations that still sit at
// token rows, and, for a model that runs on grouped expert rows, the group-row planes and
// routing plan the shared estimate cannot take from a dense intermediate width.
struct MtpExecutorTerms {
    std::size_t token_row_intermediate = 0;
    std::size_t grouped_executor_bytes = 0;
    bool exl3_linear_workspace = false;
};

MtpExecutorTerms mtp_executor_terms(const LoadedModel& model, std::size_t rows) {
    MtpExecutorTerms terms;
    if (model.qwen4exp_model() != nullptr) {
        const auto& q4 = *model.qwen4exp_model();
        terms.token_row_intermediate =
            q4.text_config().ffn.shared_expert_intermediate_size;
        terms.grouped_executor_bytes = shelf::qwen4exp::mtp_grouped_executor_bytes(
            q4.text_config(), rows, 0u);
        terms.exl3_linear_workspace = true;
    } else if (model.moe_model() != nullptr) {
        const auto& moe = *model.moe_model();
        terms.token_row_intermediate = moe.text_config().ffn.shared_expert_intermediate_size;
        terms.grouped_executor_bytes =
            shelf::qwen35moe::moe_executor_grouped_bytes(moe.text_config(), rows);
        terms.exl3_linear_workspace = shelf::qwen35moe::uses_exl3_linear(moe.weights());
    } else {
        terms.token_row_intermediate = model.dense_model()->text_config().ffn.intermediate_size;
        terms.exl3_linear_workspace =
            shelf::qwen35::uses_exl3_linear(model.dense_model()->weights());
    }
    return terms;
}

}  // namespace

bool LoadedModel::has_mtp_block() const noexcept {
    if (qwen4exp_ != nullptr) return qwen4exp_->weights().mtp.present;
    if (moe_ != nullptr) return moe_->weights().mtp.present;
    return dense_ != nullptr && dense_->weights().mtp.present;
}

std::size_t LoadedModel::mtp_drafter_reserve_bytes(
    const shelf::qwen35::runtime::MtpDrafterConfig& options) const {
    if (!has_mtp_block()) return 0u;
    const std::size_t rows = shelf::qwen35::runtime::mtp_drafter_rows(options);
    const MtpExecutorTerms terms = mtp_executor_terms(*this, rows);
    return shelf::qwen35::runtime::mtp_drafter_reserve_bytes(
        hybrid_config(), options, terms.token_row_intermediate,
        terms.grouped_executor_bytes, terms.exl3_linear_workspace);
}

std::size_t LoadedModel::mtp_spec_batch_reserve_bytes(
    const shelf::qwen35::runtime::MtpDrafterConfig& options,
    std::size_t max_requests) const {
    if (!has_mtp_block()) return 0u;
    const std::size_t rows = shelf::qwen35::runtime::mtp_drafter_rows(options);
    const MtpExecutorTerms terms = mtp_executor_terms(*this, rows);
    return shelf::qwen35::runtime::mtp_spec_batch_reserve_bytes(
        hybrid_config(), options, max_requests, terms.token_row_intermediate,
        terms.grouped_executor_bytes, terms.exl3_linear_workspace);
}

Result<shelf::qwen35::runtime::MtpDrafterInput> LoadedModel::mtp_drafter_input(
    const shelf::qwen35::runtime::MtpDrafterConfig& options,
    const shelf::weights::MatrixWeight* lm_head_override) const {
    if (!has_mtp_block()) {
        return Status::unsupported(
            "the loaded model has no MTP block: load it with the MTP option", __FILE__,
            __LINE__);
    }
    shelf::qwen35::runtime::MtpDrafterInput input;
    input.config = &hybrid_config();
    input.embed_tokens = &embed_tokens();
    const std::size_t rows = shelf::qwen35::runtime::mtp_drafter_rows(options);
    const MtpExecutorTerms terms = mtp_executor_terms(*this, rows);
    input.token_row_intermediate = terms.token_row_intermediate;
    input.grouped_executor_bytes = terms.grouped_executor_bytes;

    if (qwen4exp_ != nullptr) {
        shelf::qwen4exp::Qwen4ExpLowerOptions lower_options;
        lower_options.max_context_tokens = options.max_seq_len + 1u;
        auto lowered = shelf::qwen4exp::lower_qwen4exp_mtp_to_primitives(
            qwen4exp_->text_config(), qwen4exp_->weights(), lower_options);
        if (!lowered.ok()) return lowered.status();
        input.lowered = lowered.release();
        input.hidden_width = static_cast<std::uint32_t>(
            qwen4exp_->text_config().hc_count * qwen4exp_->text_config().hidden_size);
        input.wide_hidden = true;
        return input;
    }
    if (moe_ != nullptr) {
        auto lowered = shelf::qwen35moe::lower_qwen35_moe_mtp_to_primitives(
            moe_->text_config(), moe_->weights(),
            static_cast<float>(options.rope_profile.attention_factor));
        if (!lowered.ok()) return lowered.status();
        input.lowered = lowered.release();
        return input;
    }

    auto lowered = shelf::qwen35::lower_qwen35_mtp_to_primitives(
        dense_->text_config(), dense_->weights(), options.rope_profile,
        lm_head_override);
    if (!lowered.ok()) return lowered.status();
    input.lowered = lowered.release();
    return input;
}

}  // namespace shelf::models
