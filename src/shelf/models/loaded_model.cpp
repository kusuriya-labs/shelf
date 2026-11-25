#include <shelf/models/loaded_model.h>
#include <shelf/models/hunyuan_dense/model/lower_to_primitives.h>
#include <shelf/models/plamo3/model/lower_to_primitives.h>
#include <shelf/models/qwen35/model/gdn_state_layout.h>
#include <shelf/models/qwen35/model/lower_to_primitives.h>
#include <shelf/models/qwen35/runtime/executor_footprint.h>
#include <shelf/models/tp_rank_model.h>
#include <shelf/models/pipeline_stage_model.h>
#include <shelf/models/qwen35moe/model/lower_to_primitives.h>
#include <shelf/models/qwen35moe/runtime/executor_footprint.h>
#include <shelf/models/qwen4exp/model/lower_to_primitives.h>
#include <shelf/models/qwen4exp/runtime/executor_footprint.h>
#include <shelf/models/qwen4exp/weights/ple_table.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <utility>
#include <vector>

namespace shelf::models {

const shelf::qwen_hybrid::QwenHybridConfig& LoadedModel::hybrid_config() const noexcept {
    if (hunyuan_ != nullptr) return hunyuan_->hybrid;
    if (plamo_ != nullptr) return plamo_->hybrid;
    if (qwen4exp_ != nullptr) {
        return static_cast<const shelf::qwen_hybrid::QwenHybridConfig&>(
            qwen4exp_->text_config());
    }
    if (qsa_ != nullptr) {
        return static_cast<const shelf::qwen_hybrid::QwenHybridConfig&>(
            qsa_->text_config());
    }
    return moe_ != nullptr ? static_cast<const shelf::qwen_hybrid::QwenHybridConfig&>(
                                 moe_->text_config())
                           : static_cast<const shelf::qwen_hybrid::QwenHybridConfig&>(
                                 dense_->text_config());
}

const shelf::weights::MatrixWeight& LoadedModel::embed_tokens() const noexcept {
    if (hunyuan_ != nullptr) return hunyuan_->weights.embed_tokens;
    if (plamo_ != nullptr) return plamo_->weights.embed_tokens;
    if (qwen4exp_ != nullptr) return qwen4exp_->weights().embed_tokens;
    if (qsa_ != nullptr) return qsa_->weights().backbone.embed_tokens;
    return moe_ != nullptr ? moe_->weights().embed_tokens : dense_->weights().embed_tokens;
}

const shelf::weights::MatrixWeight& LoadedModel::lm_head() const noexcept {
    if (hunyuan_ != nullptr) return hunyuan_->weights.lm_head;
    if (plamo_ != nullptr) return plamo_->weights.lm_head;
    if (qwen4exp_ != nullptr) return qwen4exp_->weights().lm_head;
    if (qsa_ != nullptr) return qsa_->weights().backbone.lm_head;
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
    if (family_result.value() == ModelFamily::HunyuanDense) {
        if (load_mtp) {
            return Status::unsupported("hunyuan_v1_dense has no MTP block", __FILE__, __LINE__);
        }
        auto cfg = shelf::hunyuan_dense::read_config(model_dir);
        if (!cfg.ok()) return cfg.status();
        shelf::hunyuan_dense::LoadOptions load_options;
        load_options.verify_quantized_payload_crc = verify_payload_crc;
        if (const char* lm_head_int8 = std::getenv("SHELF_LM_HEAD_INT8")) {
            load_options.quantize_lm_head_int8_w32 =
                std::strtoul(lm_head_int8, nullptr, 10) != 0u;
        }
        auto weights = shelf::hunyuan_dense::load_weights(
            model_dir, cfg.value(), arena, stream, load_options);
        if (!weights.ok()) return weights.status();
        out.hunyuan_ = std::make_unique<HunyuanDenseModel>();
        out.hunyuan_->config = cfg.release();
        out.hunyuan_->weights = weights.release();
        const auto& c = out.hunyuan_->config;
        auto& h = out.hunyuan_->hybrid;
        h.vocab_size = c.vocab_size;
        h.hidden_size = c.hidden_size;
        h.num_hidden_layers = c.num_hidden_layers;
        h.num_attention_heads = c.num_attention_heads;
        h.num_key_value_heads = c.num_key_value_heads;
        h.attention_head_dim = c.head_dim;
        h.rms_norm_eps = static_cast<float>(c.rms_norm_eps);
        h.rope_theta = static_cast<float>(c.rope_theta);
        h.partial_rotary_factor = 1.0f;
        h.tie_word_embeddings = c.tie_word_embeddings;
        h.layer_types.assign(c.num_hidden_layers, 1);
        h.stop_tokens = c.stop_tokens;
        h.declared_model_type = "hunyuan_v1_dense";
        return out;
    }
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
        shelf::qwen4exp::Qwen4ExpLoadOptions load_options;
        load_options.load_mtp = load_mtp;
        if (const char* host_experts = std::getenv("SHELF_MOE_HOST_EXPERTS")) {
            load_options.moe_host_experts =
                static_cast<std::uint32_t>(std::strtoul(host_experts, nullptr, 10));
        }
        if (const char* lm_head_int8 = std::getenv("SHELF_LM_HEAD_INT8")) {
            load_options.quantize_lm_head_int8_w32 =
                std::strtoul(lm_head_int8, nullptr, 10) != 0u;
        }
        auto loaded = shelf::qwen4exp::Qwen4ExpModel::load_from_safetensors(
            model_dir, 1u, 0u, arena, stream, load_options);
        if (!loaded.ok()) return loaded.status();
        out.owns_qwen4exp_ =
            std::make_unique<shelf::qwen4exp::Qwen4ExpModel>(loaded.release());
        out.qwen4exp_ = out.owns_qwen4exp_.get();
        if (out.qwen4exp_->weights().host_experts != 0u) {
            out.moe_cache_ =
                shelf::qwen4exp::MoeExpertCache::create(out.qwen4exp_->weights());
        }
        return out;
    }
    if (family_result.value() == ModelFamily::Qsa) {
        shelf::qwen35qsa::Qwen35QsaLoadOptions options;
        options.verify_quantized_payload_crc = verify_payload_crc;
        options.load_mtp_layers = load_mtp;
        auto loaded = shelf::qwen35qsa::Qwen35QsaModel::load_from_safetensors(
            model_dir, arena, stream, options);
        if (!loaded.ok()) return loaded.status();
        out.owns_qsa_ =
            std::make_unique<shelf::qwen35qsa::Qwen35QsaModel>(loaded.release());
        out.qsa_ = out.owns_qsa_.get();
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

LoadedModel LoadedModel::borrow(const shelf::qwen35qsa::Qwen35QsaModel& model) {
    LoadedModel out;
    out.qsa_ = &model;
    return out;
}

LoadedModel LoadedModel::borrow_rank_model(const TpRankModel& model) {
    if (const auto* q4 = model.qwen4exp_model()) return borrow(*q4);
    if (const auto* qsa = model.qsa_model()) return borrow(*qsa);
    if (const auto* grouped = model.grouped_model()) return borrow(*grouped);
    return borrow(*model.dense_model());
}

LoadedModel LoadedModel::borrow_stage_model(const PipelineStageModel& model) {
    if (const auto* q4 = model.qwen4exp_model()) return borrow(*q4);
    if (const auto* qsa = model.qsa_model()) return borrow(*qsa);
    if (const auto* grouped = model.grouped_model()) return borrow(*grouped);
    return borrow(*model.dense_model());
}

shelf::weights::Exl3LinearShape LoadedModel::exl3_linear_shape() const {
    if (hunyuan_ != nullptr) {
        shelf::weights::Exl3LinearShape shape;
        shelf::weights::note_exl3_int8_decode_accum(hunyuan_->weights.lm_head, shape);
        for (const auto& layer : hunyuan_->weights.layers) {
            for (const auto* w : {&layer.q_proj, &layer.k_proj, &layer.v_proj, &layer.o_proj,
                                  &layer.gate_proj, &layer.up_proj, &layer.down_proj}) {
                shelf::weights::note_exl3_linear(*w, shape);
            }
        }
        return shape;
    }
    if (plamo_ != nullptr) {
        shelf::weights::Exl3LinearShape shape;
        shelf::weights::note_exl3_linear(plamo_->weights.lm_head, shape);
        for (const auto& layer : plamo_->weights.layers) {
            for (const auto* w : {&layer.qkv, &layer.q, &layer.k, &layer.v, &layer.o_proj,
                                  &layer.gate_up, &layer.gate, &layer.up, &layer.down})
                shelf::weights::note_exl3_linear(*w, shape);
        }
        return shape;
    }
    if (qwen4exp_ != nullptr) {
        return shelf::qwen4exp::exl3_linear_shape(qwen4exp_->weights());
    }
    if (qsa_ != nullptr) {
        return shelf::qwen35::exl3_linear_shape(qsa_->weights().backbone);
    }
    if (moe_ != nullptr) {
        return shelf::qwen35moe::exl3_linear_shape(moe_->weights());
    }
    if (dense_ != nullptr) {
        return shelf::qwen35::exl3_linear_shape(dense_->weights());
    }
    return {};
}

std::uint64_t LoadedModel::host_pinned_bytes() const noexcept {
    if (qwen4exp_ != nullptr) return qwen4exp_->weights().host_pinned_bytes();
    if (qsa_ != nullptr) return qsa_->weights().backbone.pinned_host.used();
    if (moe_ != nullptr) return moe_->weights().pinned_host.used();
    if (dense_ != nullptr) return dense_->weights().pinned_host.used();
    return 0u;
}

HostMemoryBytes LoadedModel::host_memory_bytes() const noexcept {
    HostMemoryBytes out;
    out.embedding = host_pinned_bytes();
    if (qwen4exp_ != nullptr) {
        out.offload_experts = qwen4exp_->weights().host_expert_bytes();
        if (const auto& table = qwen4exp_->ple_table()) out.ple_table = table->host_bytes();
    }
    return out;
}

std::size_t LoadedModel::executor_reserve_bytes(std::size_t max_batch_rows,
                                                std::size_t max_scheduled_requests,
                                                std::size_t max_scheduled_output_rows,
                                                std::size_t target_hidden_tap_count,
                                                std::size_t tp_size,
                                                std::size_t max_context_tokens) const {
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
        hunyuan_ != nullptr ? hunyuan_->config.intermediate_size
        : plamo_ != nullptr ? plamo_->config.intermediate_size
        : qwen4exp_ != nullptr ? qwen4exp_->text_config().ffn.shared_expert_intermediate_size
        : qsa_ != nullptr ? qsa_->text_config().ffn.intermediate_size
        : moe_ != nullptr ? moe_->text_config().ffn.shared_expert_intermediate_size
                          : dense_->text_config().ffn.intermediate_size;
    if (qsa_ != nullptr) {
        reserve_config.max_context_tokens = static_cast<std::uint32_t>(max_context_tokens);
        return shelf::qwen35qsa::qwen35qsa_executor_reserve_bytes(
            qsa_->text_config(), token_row_ffn, reserve_config, tp_size,
            reserve_config.exl3_linear_shape.present(), max_scheduled_requests);
    }
    const bool exl3 =
        hunyuan_ != nullptr ? reserve_config.exl3_linear_shape.present()
        : plamo_ != nullptr ? reserve_config.exl3_linear_shape.present()
        : qwen4exp_ != nullptr ? true
        : moe_ != nullptr ? shelf::qwen35moe::uses_exl3_linear(moe_->weights())
                          : shelf::qwen35::uses_exl3_linear(dense_->weights());
    std::size_t total = shelf::qwen35::runtime::qwen35_executor_reserve_bytes(
        hybrid_config(), token_row_ffn, reserve_config, tp_size, exl3,
        qwen4exp_ != nullptr);
    // PLaMo's combined projections and BF16 intermediate values need more scratch
    // than the Qwen estimate at small row buckets. Keep room after the KV plan.
    if (plamo_ != nullptr) total = std::max<std::size_t>(total, 1ull << 30);
    if (moe_ != nullptr)
        total += shelf::qwen35moe::moe_executor_grouped_bytes(moe_->text_config(),
                                                              max_batch_rows, tp_size);
    if (qwen4exp_ != nullptr) {
        const auto& text = qwen4exp_->text_config();
        shelf::qwen35moe::Qwen35MoeTextConfig grouped;
        static_cast<shelf::qwen_hybrid::QwenHybridConfig&>(grouped) = text;
        grouped.ffn.num_experts = text.ffn.num_experts;
        grouped.ffn.num_experts_per_tok = text.ffn.num_experts_per_tok;
        grouped.ffn.moe_intermediate_size = text.ffn.moe_intermediate_size;
        grouped.ffn.shared_expert_intermediate_size = text.ffn.shared_expert_intermediate_size;
        total += shelf::qwen35moe::moe_executor_grouped_bytes(grouped, max_batch_rows, tp_size);
    }
    return total;
}

std::size_t LoadedModel::prefix_capture_state_bytes(std::size_t capture_slots,
                                                    std::size_t max_context_tokens) const {
    if (qsa_ == nullptr) return 0u;
    return shelf::qwen35qsa::qwen35qsa_prefix_capture_bytes(
        qsa_->text_config(), static_cast<std::uint32_t>(max_context_tokens), capture_slots);
}

Result<shelf::qwen35::Executor> LoadedModel::create_executor(
    shelf::qwen35::SequenceSlotPool& sequence_pool,
    shelf::qwen35::GdnStatePool& gdn_state_pool,
    shelf::qwen35::PagedKVPool& kv_pool,
    gpu::GpuArena& arena,
    const shelf::qwen35::ExecutorConfig& config) const {

    if (hunyuan_ != nullptr) {
        if (config.tp != nullptr || config.stage.enabled || config.target_hidden_tap_count != 0 ||
            config.emit_normed_token_hidden || config.external_hidden_input) {
            return Status::unsupported(
                "hunyuan_v1_dense: requested executor mode is not supported", __FILE__, __LINE__);
        }
        auto profile = shelf::hunyuan_dense::resolve_rope_profile(hunyuan_->config);
        if (!profile.ok()) return profile.status();
        auto lowered = shelf::hunyuan_dense::lower_to_primitives(
            hunyuan_->config, hunyuan_->weights);
        if (!lowered.ok()) return lowered.status();
        shelf::qwen35::ExecutorModelInput input;
        input.model.config = &hunyuan_->hybrid;
        input.model.embed_tokens = &hunyuan_->weights.embed_tokens;
        input.lowered = lowered.release();
        shelf::qwen35::ExecutorConfig resolved = config;
        resolved.exl3_linear_shape = exl3_linear_shape();
        resolved.rope_profiles = {profile.value()};
        return shelf::qwen35::create_executor(
            std::move(input), sequence_pool, gdn_state_pool, kv_pool, arena, resolved);
    }
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
        resolved.exl3_linear_shape = exl3_linear_shape();
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
        auto created = shelf::models::create_qwen4exp_model_executor(
            *qwen4exp_, sequence_pool, gdn_state_pool, kv_pool, arena, config);
        if (created.ok() && moe_cache_ != nullptr) {
            auto cache = moe_cache_;
            created.value().after_batch = [cache](hipStream_t stream) {
                return cache->step(stream);
            };
        }
        return created;
    }
    if (qsa_ != nullptr) {
        if (config.tp != nullptr && config.tp->tp_size > 1u) {
            return Status::unsupported(
                "qwen35qsa: tensor parallel is not implemented", __FILE__, __LINE__);
        }
        return shelf::qwen35qsa::create_qwen35qsa_model_executor(
            *qsa_, sequence_pool, gdn_state_pool, kv_pool, arena, config);
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

MtpExecutorTerms mtp_executor_terms(const LoadedModel& model, std::size_t rows,
                                    std::uint32_t context_tokens) {
    MtpExecutorTerms terms;
    if (model.qwen4exp_model() != nullptr) {
        const auto& q4 = *model.qwen4exp_model();
        terms.token_row_intermediate =
            q4.text_config().ffn.shared_expert_intermediate_size;
        terms.grouped_executor_bytes = shelf::qwen4exp::mtp_grouped_executor_bytes(
            q4.text_config(), rows, context_tokens);
        terms.exl3_linear_workspace = true;
    } else if (model.qsa_model() != nullptr) {
        terms.token_row_intermediate =
            model.qsa_model()->text_config().ffn.intermediate_size;
        terms.exl3_linear_workspace =
            shelf::qwen35::uses_exl3_linear(model.qsa_model()->weights().backbone);
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
    // Speculative decode is not wired for the sparse family yet: its state pool has to be
    // rolled back with the KV cache when drafts are rejected.
    if (qsa_ != nullptr) return false;
    return dense_ != nullptr && dense_->weights().mtp.present;
}

std::uint32_t LoadedModel::mtp_token_hidden_width() const noexcept {
    if (qwen4exp_ == nullptr) return 0u;
    const auto& text = qwen4exp_->text_config();
    return static_cast<std::uint32_t>(text.hc_count * text.hidden_size);
}

std::size_t LoadedModel::mtp_drafter_reserve_bytes(
    const shelf::runtime::mtp::MtpDrafterConfig& options) const {
    if (!has_mtp_block()) return 0u;
    const std::size_t rows = shelf::runtime::mtp::mtp_drafter_rows(options);
    const MtpExecutorTerms terms = mtp_executor_terms(*this, rows, options.max_seq_len + 1u);
    shelf::weights::Exl3LinearShape shape;
    if (qwen4exp_ != nullptr) {
        shape = shelf::qwen4exp::mtp_exl3_linear_shape(qwen4exp_->weights());
    }
    return shelf::runtime::mtp::mtp_drafter_reserve_bytes(
        hybrid_config(), options, terms.token_row_intermediate,
        terms.grouped_executor_bytes, terms.exl3_linear_workspace, shape);
}

std::size_t LoadedModel::mtp_spec_batch_reserve_bytes(
    const shelf::runtime::mtp::MtpDrafterConfig& options,
    std::size_t max_requests) const {
    if (!has_mtp_block()) return 0u;
    const auto requests = static_cast<std::uint32_t>(max_requests);
    const auto spec_rows = shelf::runtime::mtp::mtp_spec_rows(requests, options.num_drafts);
    const std::size_t rows = qwen4exp_ != nullptr
        ? std::max(options.max_batch_rows, spec_rows)
        : shelf::runtime::mtp::mtp_drafter_rows(options);
    const MtpExecutorTerms terms = mtp_executor_terms(*this, rows, options.max_seq_len + 1u);
    shelf::runtime::mtp::MtpDrafterConfig per_request = options;
    per_request.allocate_spec_buffers = false;
    shelf::weights::Exl3LinearShape shape;
    if (qwen4exp_ != nullptr) {
        shape = shelf::qwen4exp::mtp_exl3_linear_shape(qwen4exp_->weights());
        const auto shared = shelf::runtime::mtp::mtp_shared_draft_reserve_bytes(
            hybrid_config(), per_request, terms.token_row_intermediate,
            terms.grouped_executor_bytes, terms.exl3_linear_workspace, shape,
            requests);
        const char* fused = std::getenv("SHELF_MTP_FUSED_VERIFY");
        if (fused != nullptr && (*fused == '0' || *fused == '\0')) return shared;
        return shared + shelf::runtime::mtp::mtp_batch_rank_state_reserve_bytes(
            shelf::qwen35::make_qwen35_gdn_state_layout(hybrid_config()), spec_rows, requests);
    }
    const std::size_t drafter = shelf::runtime::mtp::mtp_drafter_reserve_bytes(
        hybrid_config(), per_request, terms.token_row_intermediate,
        terms.grouped_executor_bytes, terms.exl3_linear_workspace, shape);
    const std::size_t state = shelf::runtime::mtp::mtp_batch_rank_state_reserve_bytes(
        shelf::qwen35::make_qwen35_gdn_state_layout(hybrid_config()),
        spec_rows, requests);
    return drafter * max_requests + state;
}

Result<shelf::runtime::mtp::MtpDrafterInput> LoadedModel::mtp_drafter_input(
    const shelf::runtime::mtp::MtpDrafterConfig& options,
    const shelf::weights::MatrixWeight* lm_head_override) const {
    if (!has_mtp_block()) {
        return Status::unsupported(
            "the loaded model has no MTP block: load it with the MTP option", __FILE__,
            __LINE__);
    }
    shelf::runtime::mtp::MtpDrafterInput input;
    input.config = &hybrid_config();
    input.embed_tokens = &embed_tokens();
    const std::size_t rows = shelf::runtime::mtp::mtp_drafter_rows(options);
    const MtpExecutorTerms terms = mtp_executor_terms(*this, rows, options.max_seq_len + 1u);
    input.token_row_intermediate = terms.token_row_intermediate;
    input.grouped_executor_bytes = terms.grouped_executor_bytes;

    if (qwen4exp_ != nullptr) {
        shelf::qwen4exp::Qwen4ExpLowerOptions lower_options;
        lower_options.max_context_tokens = options.max_seq_len + 1u;
        auto lowered = shelf::qwen4exp::lower_qwen4exp_mtp_to_primitives(
            qwen4exp_->text_config(), qwen4exp_->weights(), lower_options);
        if (!lowered.ok()) return lowered.status();
        input.lowered = lowered.release();
        input.exl3_linear_shape =
            shelf::qwen4exp::mtp_exl3_linear_shape(qwen4exp_->weights());
        input.hidden_width = static_cast<std::uint32_t>(
            qwen4exp_->text_config().hc_count * qwen4exp_->text_config().hidden_size);
        input.wide_hidden = true;
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
