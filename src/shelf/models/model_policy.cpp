#include <shelf/models/model_policy.h>

#include <shelf/models/model_weights_factory.h>
#include <shelf/models/loaded_model.h>
#include <shelf/models/hunyuan_dense/model/config.h>
#include <shelf/models/plamo3/model/config.h>
#include <shelf/models/qwen35/model/rope_profile.h>
#include <shelf/runtime/mtp/mtp_drafter.h>
#include <shelf/models/qwen4exp/model/qwen4exp_config.h>
#include <shelf/models/tp_rank_model.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace shelf::models {
namespace {

ModelClass to_model_class(ModelFamily family) noexcept {
    switch (family) {
        case ModelFamily::Dense: return ModelClass::Dense;
        case ModelFamily::Moe: return ModelClass::Moe;
        case ModelFamily::Qsa: return ModelClass::Qsa;
        case ModelFamily::Qwen4Exp: return ModelClass::Qwen4Exp;
        case ModelFamily::Plamo3: return ModelClass::Plamo3;
        case ModelFamily::HunyuanDense: return ModelClass::HunyuanDense;
    }
    return ModelClass::Dense;
}

const char* display_family(ModelClass model_class) noexcept {
    switch (model_class) {
        case ModelClass::Dense: return "dense";
        case ModelClass::Moe: return "moe";
        case ModelClass::Qsa: return "qsa";
        case ModelClass::Qwen4Exp: return "moe";
        case ModelClass::Plamo3: return "plamo3";
        case ModelClass::HunyuanDense: return "hunyuan_v1_dense";
    }
    return "dense";
}

shelf::Status resolve_plamo(const ModelRuntimeRequest& request) {
    if (request.rope_factor != "auto") {
        return Status::invalid_argument(
            "plamo3: --rope-factor override is not supported", __FILE__, __LINE__);
    }
    if (request.max_seq_len > 2048u) {
        return Status::invalid_argument(
            "plamo3: --max-seq-len must be at most 2048 (sliding attention limit)",
            __FILE__, __LINE__);
    }
    return Status::make_ok();
}

shelf::Status resolve_hunyuan(const ModelRuntimeRequest& request, ResolvedModelOptions& out) {
    auto cfg = shelf::hunyuan_dense::read_config(request.model_dir);
    if (!cfg.ok()) return cfg.status();
    if (cfg.value().max_position_embeddings > 0u &&
        request.max_seq_len > cfg.value().max_position_embeddings) {
        return Status::invalid_argument(
            "hunyuan_v1_dense: --max-seq-len exceeds max_position_embeddings",
            __FILE__, __LINE__);
    }
    auto profile = shelf::hunyuan_dense::resolve_rope_profile(cfg.value(), request.rope_factor);
    if (!profile.ok()) return profile.status();
    out.rope_profile = profile.value();
    out.has_rope_profile = true;
    return Status::make_ok();
}

}  // namespace

Result<ResolvedModelOptions> resolve_model_options(const ModelRuntimeRequest& request) {
    auto plamo = shelf::plamo3::is_plamo3_config(request.model_dir);
    if (!plamo.ok()) return plamo.status();

    ResolvedModelOptions out;
    if (plamo.value()) {
        out.model_class = ModelClass::Plamo3;
    } else {
        auto family = probe_model_dir(request.model_dir);
        if (!family.ok()) return family.status();
        out.model_class = to_model_class(family.value());
    }
    out.display_family = display_family(out.model_class);
    out.max_seq_len = request.max_seq_len;
    out.private_target_weights = request.private_target_weights;
    out.private_target_lm_head = request.private_target_lm_head;
    out.dflash2_drafts = request.dflash2_drafts;

    if (out.model_class == ModelClass::Plamo3) {
        const Status checked = resolve_plamo(request);
        if (!checked.ok()) return checked;
        return out;
    }
    if (out.model_class == ModelClass::HunyuanDense) {
        const Status checked = resolve_hunyuan(request, out);
        if (!checked.ok()) return checked;
        return out;
    }

    auto selection = shelf::qwen35::parse_rope_factor_selection(request.rope_factor);
    if (!selection.ok()) {
        const std::string message =
            "rope factor resolve failed: " + selection.status().message();
        return Status::invalid_argument(message.c_str(), __FILE__, __LINE__);
    }
    auto params = shelf::qwen35::read_qwen35_rope_parameters(request.model_dir);
    if (!params.ok()) {
        const std::string message =
            "rope config read failed: " + params.status().message();
        return Status::invalid_argument(message.c_str(), __FILE__, __LINE__);
    }
    auto profile = shelf::qwen35::resolve_effective_rope_profile(
        params.value(), selection.value());
    if (!profile.ok()) {
        const std::string message =
            "rope profile resolve failed: " + profile.status().message();
        return Status::invalid_argument(message.c_str(), __FILE__, __LINE__);
    }
    out.rope_profile = profile.value();
    out.has_rope_profile = true;
    const std::uint32_t cap = shelf::qwen35::rope_max_seq_len_cap(out.rope_profile);
    if (cap != 0u && request.max_seq_len > cap) {
        char message[512];
        std::snprintf(message, sizeof(message),
                      "--max-seq-len %llu exceeds the rope factor limit %u "
                      "(--rope-factor %s, model original length %u)",
                      static_cast<unsigned long long>(request.max_seq_len), cap,
                      request.rope_factor.c_str(),
                      out.rope_profile.original_max_position_embeddings);
        return Status::invalid_argument(message, __FILE__, __LINE__);
    }

    if (out.model_class == ModelClass::Qwen4Exp) {
        out.dflash2_host_token_ids = true;
        // The Qwen4Exp rank already holds a complete replicated lm_head / embedding, so
        // the drafter borrows those instead of loading a second private copy.
        out.private_target_weights = false;
        out.private_target_lm_head = false;
        // The 8-row verify path (block_size 8, drafts 7) is not yet lossless for this
        // family; 6 drafts stays on the validated 7-row verify path.
        if (request.dflash2 && out.dflash2_drafts > 6u) {
            out.dflash2_drafts = 6u;
            out.notices.push_back(
                "Qwen4Exp DFlash2: drafts capped at 6 "
                "(8-row verify is not validated)");
        }
    }
    return out;
}

Status validate_model_runtime_options(const ModelRuntimeRequest& request,
                                      const ResolvedModelOptions& resolved) {
    if (resolved.model_class == ModelClass::HunyuanDense) {
        if (request.tensor_parallel != 1u || request.pipeline_parallel != 1u ||
            request.mtp || request.dflash2) {
            return Status::unsupported(
                "hunyuan_v1_dense: parallel and speculative execution are not supported",
                __FILE__, __LINE__);
        }
        return Status::make_ok();
    }
    if (resolved.model_class != ModelClass::Plamo3) return Status::make_ok();
    if (request.max_seq_len > 2048u) {
        return Status::invalid_argument(
            "plamo3: max_seq_len must not exceed 2048", __FILE__, __LINE__);
    }
    if (request.tensor_parallel != 1u || request.pipeline_parallel != 1u ||
        request.mtp || request.dflash2) {
        return Status::unsupported(
            "plamo3: parallel and speculative execution are not supported", __FILE__,
            __LINE__);
    }
    return Status::make_ok();
}

std::uint32_t pipeline_hidden_width(const std::string& model_dir, ModelClass model_class) {
    if (model_class != ModelClass::Qwen4Exp) return 0u;
    auto config = shelf::qwen4exp::read_qwen4exp_text_config(model_dir);
    if (!config.ok()) return 0u;
    return static_cast<std::uint32_t>(config.value().hc_count * config.value().hidden_size);
}

std::size_t mtp_batch_reserve_bytes(
    const TpRankModel& model,
    const qwen_hybrid::QwenHybridConfig& text_config,
    const runtime::MtpBatchReserveConfig& config,
    const state::GdnStatePoolLayout& layout,
    std::uint32_t tp_rank) {
    shelf::runtime::mtp::MtpDrafterConfig options;
    options.num_drafts = config.num_drafts;
    options.max_seq_len = config.max_seq_len;
    options.page_tokens = config.page_tokens;
    options.max_batch_rows = config.max_batch_rows;
    options.kv_dtype = config.kv_dtype;
    if (model.qwen4exp_model() != nullptr && !model.qwen4exp_model()->weights().mtp.present) {
        return std::size_t{0};
    }
    std::size_t drafter_bytes = 0u;
    std::size_t state_bytes = 0u;
    if (model.context().tp_size > 1u) {
        options.local_kv_heads =
            static_cast<std::uint32_t>(model.context().local_key_value_heads);
    }
    const char* fused_env = std::getenv("SHELF_MTP_FUSED_VERIFY");
    const bool fused_verify =
        fused_env == nullptr || (fused_env[0] != '\0' && fused_env[0] != '0');
    if (model.uses_qsa_state()) {
        const std::uint32_t batch_rows = std::max(
            options.max_batch_rows, config.max_requests * (options.num_drafts + 1u));
        drafter_bytes = shelf::runtime::mtp::mtp_shared_draft_reserve_bytes(
            text_config, options, model.token_row_ffn_size(),
            model.mtp_grouped_reserve_bytes(batch_rows, options.max_seq_len + 1u),
            model.uses_exl3_linear(), model.mtp_exl3_linear_shape(),
            config.max_requests);
        if (fused_verify) {
            state_bytes = shelf::runtime::mtp::mtp_batch_rank_state_reserve_bytes(
                layout,
                shelf::runtime::mtp::mtp_spec_rows(config.max_requests,
                                                      options.num_drafts),
                config.max_requests);
        }
    } else if (tp_rank == 0u) {
        drafter_bytes = shelf::runtime::mtp::mtp_drafter_reserve_bytes(
            text_config, options, model.token_row_ffn_size(),
            model.mtp_grouped_reserve_bytes(
                shelf::runtime::mtp::mtp_drafter_rows(options),
                options.max_seq_len + 1u),
            model.uses_exl3_linear(), model.mtp_exl3_linear_shape(), &layout) *
            config.max_requests;
        state_bytes = shelf::runtime::mtp::mtp_batch_rank_state_reserve_bytes(
            layout,
            shelf::runtime::mtp::mtp_spec_rows(config.max_requests, options.num_drafts),
            config.max_requests);
    }
    return drafter_bytes + state_bytes;
}

std::uint32_t max_mtp_drafts() noexcept {
    return shelf::runtime::mtp::kMtpMaxDrafts;
}

std::uint32_t max_mtp_prefill_rows() noexcept {
    return shelf::runtime::mtp::kMtpMaxPrefillRows;
}

std::size_t mtp_drafter_reserve_bytes(const LoadedModel& model,
                                      const MtpReserveRequest& request,
                                      bool batched, std::size_t max_requests) {
    shelf::runtime::mtp::MtpDrafterConfig options;
    options.num_drafts = request.num_drafts;
    options.max_seq_len = request.max_seq_len;
    options.page_tokens = request.page_tokens;
    options.max_batch_rows = request.max_batch_rows;
    options.kv_dtype = request.kv_dtype;
    options.rope_profile = request.rope_profile;
    if (batched) {
        return model.mtp_spec_batch_reserve_bytes(options, max_requests);
    }
    return model.mtp_drafter_reserve_bytes(options);
}

}  // namespace shelf::models
