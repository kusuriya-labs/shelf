#include <shelf/models/loaded_model.h>
#include <shelf/models/qwen35/model/lower_to_primitives.h>
#include <shelf/models/qwen35/runtime/executor_footprint.h>
#include <shelf/models/qwen35moe/model/lower_to_primitives.h>
#include <shelf/models/qwen35moe/runtime/executor_footprint.h>

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace shelf::models {

const shelf::qwen_hybrid::QwenHybridConfig& LoadedModel::hybrid_config() const noexcept {
    return moe_ != nullptr ? static_cast<const shelf::qwen_hybrid::QwenHybridConfig&>(
                                 moe_->text_config())
                           : static_cast<const shelf::qwen_hybrid::QwenHybridConfig&>(
                                 dense_->text_config());
}

const shelf::weights::MatrixWeight& LoadedModel::embed_tokens() const noexcept {
    return moe_ != nullptr ? moe_->weights().embed_tokens : dense_->weights().embed_tokens;
}

const shelf::weights::MatrixWeight& LoadedModel::lm_head() const noexcept {
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

    // The dense FFN term of the shared estimate stands in for the FFN activations that
    // still sit at token rows: for a model with grouped experts those are its shared expert.
    const std::size_t token_row_ffn =
        moe_ != nullptr ? moe_->text_config().ffn.shared_expert_intermediate_size
                        : dense_->text_config().ffn.intermediate_size;
    std::size_t total = shelf::qwen35::runtime::qwen35_executor_reserve_bytes(
        hybrid_config(), token_row_ffn, reserve_config, tp_size);
    if (moe_ != nullptr)
        total += shelf::qwen35moe::moe_executor_grouped_bytes(moe_->text_config(),
                                                              max_batch_rows);
    return total;
}

Result<shelf::qwen35::Executor> LoadedModel::create_executor(
    shelf::qwen35::SequenceSlotPool& sequence_pool,
    shelf::qwen35::GdnStatePool& gdn_state_pool,
    shelf::qwen35::PagedKVPool& kv_pool,
    gpu::GpuArena& arena,
    const shelf::qwen35::ExecutorConfig& config) const {

    if (dense_ != nullptr) {
        return shelf::qwen35::create_model_executor(*dense_, sequence_pool, gdn_state_pool,
                                                    kv_pool, arena, config);
    }

    if (config.tp != nullptr && config.tp->tp_size > 1u) {
        return Status::unsupported(
            "a model with grouped experts runs on one device: the expert planes are not "
            "part of the tensor partition contract",
            __FILE__, __LINE__);
    }

    std::vector<shelf::qwen35::LayerOverrideSource> overrides =
        shelf::qwen35::read_layer_override_sources(
            static_cast<std::uint32_t>(moe_->text_config().hidden_size));
    std::vector<std::uint32_t> override_layers;
    override_layers.reserve(overrides.size());
    for (const auto& ov : overrides) override_layers.push_back(ov.layer);

    shelf::qwen35moe::Qwen35MoeLowerOptions lower_options;
    lower_options.hidden_taps = std::span<const std::uint32_t>(
        config.target_hidden_taps.data(), config.target_hidden_tap_count);
    lower_options.override_layers = std::span<const std::uint32_t>(
        override_layers.data(), override_layers.size());
    lower_options.rope_attention_factor = static_cast<float>(config.rope_profile.attention_factor);
    lower_options.emit_normed_token_hidden = config.emit_normed_token_hidden;

    auto lowered = shelf::qwen35moe::lower_qwen35_moe_to_primitives(
        moe_->text_config(), moe_->weights(), lower_options);
    if (!lowered.ok()) return lowered.status();

    shelf::qwen35::ExecutorModelInput input;
    input.model.config = &moe_->text_config();
    input.model.embed_tokens = &moe_->weights().embed_tokens;
    input.lowered = lowered.release();
    input.layer_overrides = std::move(overrides);
    return shelf::qwen35::create_executor(std::move(input), sequence_pool, gdn_state_pool,
                                          kv_pool, arena, config);
}

namespace {

// The executor workspace terms of one drafter block: the FFN activations that still sit at
// token rows, and, for a model that runs on grouped expert rows, the group-row planes and
// routing plan the shared estimate cannot take from a dense intermediate width.
struct MtpExecutorTerms {
    std::size_t token_row_intermediate = 0;
    std::size_t grouped_executor_bytes = 0;
};

MtpExecutorTerms mtp_executor_terms(const LoadedModel& model, std::size_t rows) {
    MtpExecutorTerms terms;
    if (model.moe_model() != nullptr) {
        const auto& moe = *model.moe_model();
        terms.token_row_intermediate = moe.text_config().ffn.shared_expert_intermediate_size;
        terms.grouped_executor_bytes =
            shelf::qwen35moe::moe_executor_grouped_bytes(moe.text_config(), rows);
    } else {
        terms.token_row_intermediate = model.dense_model()->text_config().ffn.intermediate_size;
    }
    return terms;
}

}  // namespace

bool LoadedModel::has_mtp_block() const noexcept {
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
        terms.grouped_executor_bytes);
}

Result<shelf::qwen35::runtime::MtpDrafterInput> LoadedModel::mtp_drafter_input(
    const shelf::qwen35::runtime::MtpDrafterConfig& options) const {
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

    if (moe_ != nullptr) {
        auto lowered = shelf::qwen35moe::lower_qwen35_moe_mtp_to_primitives(
            moe_->text_config(), moe_->weights(),
            static_cast<float>(options.rope_profile.attention_factor));
        if (!lowered.ok()) return lowered.status();
        input.lowered = lowered.release();
        return input;
    }

    auto lowered = shelf::qwen35::lower_qwen35_mtp_to_primitives(
        dense_->text_config(), dense_->weights(), options.rope_profile);
    if (!lowered.ok()) return lowered.status();
    input.lowered = lowered.release();
    return input;
}

}  // namespace shelf::models
