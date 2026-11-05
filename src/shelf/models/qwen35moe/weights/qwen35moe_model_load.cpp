#include <shelf/models/qwen35moe/weights/qwen35moe_model_load.h>

#include <shelf/quantization/manifest/quantized_manifest.h>
#include <shelf/models/common/qwen_hybrid/weight_validation.h>
#include <shelf/weights/weight_loader.h>

#include <filesystem>
#include <string>
#include <vector>

namespace shelf::qwen35moe {
namespace {

namespace fs = std::filesystem;
namespace qhy = shelf::qwen_hybrid;
namespace qqb = shelf::quantization::bundle;
namespace qqm = shelf::quantization::manifest;
using shelf::weights::MatrixWeight;

// Router and shared expert, in the order a request list carries them.
constexpr const char* kMoeAuxSuffixes[] = {
    "gate.weight",
    "shared_expert_gate.weight",
    "shared_expert.gate_proj.weight",
    "shared_expert.up_proj.weight",
    "shared_expert.down_proj.weight",
};

constexpr std::size_t kRouter = 0;
constexpr std::size_t kSharedGate = 1;
constexpr std::size_t kSharedGateProj = 2;
constexpr std::size_t kSharedUpProj = 3;
constexpr std::size_t kSharedDownProj = 4;

Status load_aux_matrix(const qqb::QuantizedModelReader& reader, const std::string& mlp,
                       std::size_t aux, gpu::GpuArena& arena, hipStream_t stream,
                       const shelf::weights::WeightLoadOptions& weight_options,
                       MatrixWeight& dst) {
    auto m = shelf::weights::load_quantized_matrix(
        reader, mlp + kMoeAuxSuffixes[aux], arena, stream, weight_options);
    if (!m.ok()) return m.status();
    dst = m.release();
    return Status::make_ok();
}

Status check_plane(const MatrixWeight& w, uint32_t experts, uint32_t rows, uint32_t cols,
                   uint32_t k_padded, uint32_t codes_row_stride, uint32_t scales_row_stride,
                   uint64_t codes_batch_stride, uint64_t scales_batch_stride,
                   const char* tag) {
    if (w.encoding != shelf::weights::MatrixEncoding::Mxfp4)
        return Status::invalid_argument(
            (std::string(tag) + ": expert plane is not MXFP4").c_str(), __FILE__, __LINE__);
    if (!w.preshuffled)
        return Status::invalid_argument(
            (std::string(tag) + ": expert plane is not native layout").c_str(), __FILE__,
            __LINE__);
    if (w.experts != experts)
        return Status::invalid_argument(
            (std::string(tag) + ": expert count differs from the plan").c_str(), __FILE__,
            __LINE__);
    if (w.rows != rows || w.cols != cols || w.k_padded != k_padded)
        return Status::invalid_argument(
            (std::string(tag) + ": per-expert geometry differs from the plan").c_str(),
            __FILE__, __LINE__);
    if (w.codes_row_stride_bytes != codes_row_stride ||
        w.scale_row_stride_bytes != scales_row_stride)
        return Status::invalid_argument(
            (std::string(tag) + ": row strides differ from the plan").c_str(), __FILE__,
            __LINE__);
    if (w.codes_batch_stride_bytes != codes_batch_stride ||
        w.scales_batch_stride_bytes != scales_batch_stride)
        return Status::invalid_argument(
            (std::string(tag) + ": batch strides differ from the plan").c_str(), __FILE__,
            __LINE__);
    if (w.row_ref.ndim() != 2 || static_cast<uint32_t>(w.row_ref.dim(0)) != experts ||
        static_cast<uint32_t>(w.row_ref.dim(1)) != rows)
        return Status::invalid_argument(
            (std::string(tag) + ": row_ref plane is not [experts][rows]").c_str(), __FILE__,
            __LINE__);
    return Status::make_ok();
}

Status load_moe_mtp(gpu::GpuArena& arena, hipStream_t stream,
                    const qqb::QuantizedModelReader& reader,
                    const Qwen35MoeTextConfig& config,
                    const qhy::HybridLoadOptions& options,
                    Qwen35MoeMtpWeights& m) {
    auto plan = build_moe_mtp_expert_group_plan(config);
    if (!plan.ok()) return plan.status();
    const MoeExpertGroupPlan& group = plan.value();

    auto fc = shelf::weights::load_quantized_matrix(reader, "mtp.fc.weight", arena, stream,
                                                    options.weights);
    if (!fc.ok()) return fc.status();
    m.fc = fc.release();

    auto pre_embedding = shelf::weights::load_quantized_small(
        reader, "mtp.pre_fc_norm_embedding.weight", arena, stream);
    if (!pre_embedding.ok()) return pre_embedding.status();
    m.pre_fc_norm_embedding_weight = pre_embedding.release();

    auto pre_hidden = shelf::weights::load_quantized_small(
        reader, "mtp.pre_fc_norm_hidden.weight", arena, stream);
    if (!pre_hidden.ok()) return pre_hidden.status();
    m.pre_fc_norm_hidden_weight = pre_hidden.release();

    auto tail_norm = shelf::weights::load_quantized_small(reader, "mtp.norm.weight", arena,
                                                          stream);
    if (!tail_norm.ok()) return tail_norm.status();
    m.final_norm_weight = tail_norm.release();

    Status st = qhy::load_hybrid_layer_body_quantized_at(arena, stream, reader,
                                                         moe_mtp_layer_prefix(), false,
                                                         options, m.layer);
    if (!st.ok()) return st;

    auto planes = load_moe_expert_group(reader, group, arena, stream);
    if (!planes.ok()) return planes.status();
    m.layer.ffn.expert_w13 = planes.value().w13;
    m.layer.ffn.expert_w2 = planes.value().w2;

    st = load_aux_matrix(reader, group.mlp_prefix, kRouter, arena, stream, options.weights,
                         m.layer.ffn.router);
    if (!st.ok()) return st;
    st = load_aux_matrix(reader, group.mlp_prefix, kSharedGate, arena, stream, options.weights,
                         m.layer.ffn.shared_gate);
    if (!st.ok()) return st;
    st = load_aux_matrix(reader, group.mlp_prefix, kSharedGateProj, arena, stream,
                         options.weights, m.layer.ffn.shared_gate_proj);
    if (!st.ok()) return st;
    st = load_aux_matrix(reader, group.mlp_prefix, kSharedUpProj, arena, stream, options.weights,
                         m.layer.ffn.shared_up_proj);
    if (!st.ok()) return st;
    st = load_aux_matrix(reader, group.mlp_prefix, kSharedDownProj, arena, stream,
                         options.weights, m.layer.ffn.shared_down_proj);
    if (!st.ok()) return st;

    m.present = true;
    return moe_ffn_weights_match(m.layer.ffn, group, config.ffn);
}

}  // namespace

Status validate_qwen35moe_mtp_weights(const Qwen35MoeMtpWeights& mtp,
                                      const MoeExpertGroupPlan& plan,
                                      const Qwen35MoeTextConfig& config) {
    if (!mtp.present)
        return Status::invalid_argument("qwen35moe mtp: the block is not loaded", __FILE__,
                                        __LINE__);

    const uint32_t hidden = static_cast<uint32_t>(config.hidden_size);
    if (mtp.fc.rows == 0u || mtp.fc.cols == 0u)
        return Status::invalid_argument("qwen35moe mtp: fc is not loaded", __FILE__, __LINE__);
    if (mtp.fc.rows != hidden || mtp.fc.cols != 2u * hidden)
        return Status::invalid_argument("qwen35moe mtp: fc is not [hidden, 2 * hidden]",
                                        __FILE__, __LINE__);

    const shelf::gpu::Tensor block_norms[] = {
        mtp.pre_fc_norm_embedding_weight, mtp.pre_fc_norm_hidden_weight,
        mtp.final_norm_weight};
    for (const shelf::gpu::Tensor& norm : block_norms) {
        if (norm.ndim() != 1u || static_cast<uint32_t>(norm.dim(0)) != hidden)
            return Status::invalid_argument("qwen35moe mtp: a block norm is not [hidden]",
                                            __FILE__, __LINE__);
    }

    Status st = qhy::validate_hybrid_layer_body(mtp.layer, hidden, false,
                                                "qwen35moe mtp layer");
    if (!st.ok()) return st;
    return moe_ffn_weights_match(mtp.layer.ffn, plan, config.ffn);
}

std::vector<std::string> moe_aux_tensor_names(const std::string& mlp) {
    std::vector<std::string> names;
    names.reserve(sizeof(kMoeAuxSuffixes) / sizeof(kMoeAuxSuffixes[0]));
    for (const char* suffix : kMoeAuxSuffixes) names.push_back(mlp + suffix);
    return names;
}

std::vector<std::string> moe_aux_tensor_names(std::size_t layer) {
    return moe_aux_tensor_names(moe_mlp_prefix(layer));
}

std::vector<std::string> moe_layer_logical_names(const MoeExpertGroupPlan& plan) {
    std::vector<std::string> names = moe_aux_tensor_names(plan.mlp_prefix);
    if (plan.fused_rank3) {
        names.push_back(plan.w13_name);
        names.push_back(plan.w2_name);
        return names;
    }
    names.reserve(names.size() + plan.sources.size());
    for (const auto& src : plan.sources) names.push_back(src.name);
    return names;
}

uint64_t moe_expert_group_bytes(const MoeExpertGroupPlan& plan) {
    return plan.w13_codes_bytes + plan.w13_scales_bytes +
           static_cast<uint64_t>(plan.w13_row_ref_bytes_per_expert) * plan.experts +
           plan.w2_codes_bytes + plan.w2_scales_bytes +
           static_cast<uint64_t>(plan.w2_row_ref_bytes_per_expert) * plan.experts;
}

Status moe_ffn_weights_match(const MoeFfnWeights& w, const MoeExpertGroupPlan& plan,
                             const MoeFfnConfig& ffn) {
    if (plan.experts != ffn.num_experts ||
        plan.w13_gate_rows != ffn.moe_intermediate_size ||
        plan.w13_up_rows != ffn.moe_intermediate_size)
        return Status::invalid_argument(
            "moe plan: expert group geometry differs from the config", __FILE__, __LINE__);

    Status st = check_plane(w.expert_w13, plan.experts, plan.w13_rows, plan.w13_cols,
                            plan.w13_k_padded, plan.w13_codes_row_stride_bytes,
                            plan.w13_scales_row_stride_bytes, plan.w13_codes_batch_stride_bytes,
                            plan.w13_scales_batch_stride_bytes, "moe w13");
    if (!st.ok()) return st;
    st = check_plane(w.expert_w2, plan.experts, plan.w2_rows, plan.w2_cols, plan.w2_k_padded,
                     plan.w2_codes_row_stride_bytes, plan.w2_scales_row_stride_bytes,
                     plan.w2_codes_batch_stride_bytes, plan.w2_scales_batch_stride_bytes,
                     "moe w2");
    if (!st.ok()) return st;

    const uint32_t hidden = plan.w13_cols;
    if (w.router.rows != plan.experts || w.router.cols != hidden)
        return Status::invalid_argument(
            "moe router: shape is not [experts, hidden]", __FILE__, __LINE__);
    if (w.shared_gate.rows != 1u || w.shared_gate.cols != hidden)
        return Status::invalid_argument(
            "moe shared_expert_gate: shape is not [1, hidden]", __FILE__, __LINE__);

    // The shared expert has its own width: config shared_expert_intermediate_size, which a
    // checkpoint may set differently from the routed moe_intermediate_size.
    const uint32_t shared_inter =
        static_cast<uint32_t>(ffn.shared_expert_intermediate_size);
    if (w.shared_gate_proj.rows != shared_inter || w.shared_gate_proj.cols != hidden ||
        w.shared_up_proj.rows != shared_inter || w.shared_up_proj.cols != hidden ||
        w.shared_down_proj.rows != hidden || w.shared_down_proj.cols != shared_inter)
        return Status::invalid_argument(
            "moe shared expert: shapes are not [shared_inter, hidden] / [hidden, "
            "shared_inter]",
            __FILE__, __LINE__);
    return Status::make_ok();
}

Status validate_qwen35moe_weights(const Qwen35MoeModelWeights& weights,
                                  const Qwen35MoeTextConfig& config) {
    if (weights.layers.size() != config.num_hidden_layers) {
        return Status::invalid_argument(
            "Qwen35MoeModel: layer count does not match the config",
            __FILE__, __LINE__);
    }
    Status st = qhy::validate_hybrid_model_level(weights, config, 0u,
                                                 weights.layers.size());
    if (!st.ok()) return st;

    for (std::size_t l = 0; l < weights.layers.size(); ++l) {
        st = qhy::validate_hybrid_layer_body(weights.layers[l], config, l);
        if (!st.ok()) return st;
        auto plan = build_moe_expert_group_plan(config, l);
        if (!plan.ok()) return plan.status();
        st = moe_ffn_weights_match(weights.layers[l].ffn, plan.value(), config.ffn);
        if (!st.ok()) return st;
    }

    if (!weights.mtp.present) return Status::make_ok();
    auto mtp_plan = build_moe_mtp_expert_group_plan(config);
    if (!mtp_plan.ok()) return mtp_plan.status();
    return validate_qwen35moe_mtp_weights(weights.mtp, mtp_plan.value(), config);
}

Result<Qwen35MoeModelWeights> load_qwen35moe_weights(
    const std::string& model_dir,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const Qwen35MoeLoadOptions& options) {
    if (options.weights.partition_plan != nullptr) {
        return Status::unsupported(
            "qwen35moe single-GPU weight load does not accept a tensor partition plan",
            __FILE__, __LINE__);
    }
    if (fs::exists(fs::path(model_dir) / qqm::kQuantizedMetadataFile)) {
        return Status::unsupported(
            "qwen35moe reads an HF real-quantized model dir only: rank-3 expert planes are "
            "not part of the Shelf quantized bundle contract",
            __FILE__, __LINE__);
    }

    auto config_result = read_qwen35_moe_text_config(model_dir);
    if (!config_result.ok()) return config_result.status();
    const Qwen35MoeTextConfig& config = config_result.value();

    std::vector<std::string> names = qhy::hf_quantized_body_names(config);
    std::vector<MoeExpertGroupPlan> plans;
    plans.reserve(config.num_hidden_layers);
    for (std::size_t l = 0; l < config.num_hidden_layers; ++l) {
        auto plan = build_moe_expert_group_plan(config, l);
        if (!plan.ok()) return plan.status();
        const std::vector<std::string> layer_names = moe_layer_logical_names(plan.value());
        names.insert(names.end(), layer_names.begin(), layer_names.end());
        plans.push_back(plan.release());
    }

    if (options.load_mtp_layers) {
        auto mtp_plan = build_moe_mtp_expert_group_plan(config);
        if (!mtp_plan.ok()) return mtp_plan.status();
        const std::vector<std::string> ffn_names = moe_layer_logical_names(mtp_plan.value());
        names.insert(names.end(), ffn_names.begin(), ffn_names.end());
        const std::vector<std::string> body_names =
            qhy::hf_layer_body_names(moe_mtp_layer_prefix(), false);
        names.insert(names.end(), body_names.begin(), body_names.end());
        names.push_back("mtp.fc.weight");
        names.push_back("mtp.norm.weight");
        names.push_back("mtp.pre_fc_norm_embedding.weight");
        names.push_back("mtp.pre_fc_norm_hidden.weight");
    }

    auto reader_result =
        qhy::open_quantized_model_reader(model_dir, names, options.verify_quantized_payload_crc);
    if (!reader_result.ok()) return reader_result.status();
    qqb::QuantizedModelReader reader = reader_result.release();

    Status contract = shelf::weights::validate_quantized_model(reader);
    if (!contract.ok()) return contract;

    Qwen35MoeModelWeights out;
    out.layers.resize(config.num_hidden_layers);

    Status model_level =
        qhy::load_hybrid_model_level_quantized(reader, config, arena, stream, options, out);
    if (!model_level.ok()) return model_level;

    for (std::size_t l = 0; l < config.num_hidden_layers; ++l) {
        Status st = qhy::load_hybrid_layer_body_quantized(arena, stream, reader, config, l,
                                                          options, out.layers[l]);
        if (!st.ok()) return st;

        const MoeExpertGroupPlan& plan = plans[l];
        Qwen35MoeLayerWeights& layer = out.layers[l];

        auto group = load_moe_expert_group(reader, plan, arena, stream);
        if (!group.ok()) return group.status();
        layer.ffn.expert_w13 = group.value().w13;
        layer.ffn.expert_w2 = group.value().w2;

        st = load_aux_matrix(reader, plan.mlp_prefix, kRouter, arena, stream, options.weights,
                             layer.ffn.router);
        if (!st.ok()) return st;
        st = load_aux_matrix(reader, plan.mlp_prefix, kSharedGate, arena, stream, options.weights,
                             layer.ffn.shared_gate);
        if (!st.ok()) return st;
        st = load_aux_matrix(reader, plan.mlp_prefix, kSharedGateProj, arena, stream,
                             options.weights, layer.ffn.shared_gate_proj);
        if (!st.ok()) return st;
        st = load_aux_matrix(reader, plan.mlp_prefix, kSharedUpProj, arena, stream,
                             options.weights, layer.ffn.shared_up_proj);
        if (!st.ok()) return st;
        st = load_aux_matrix(reader, plan.mlp_prefix, kSharedDownProj, arena, stream,
                             options.weights, layer.ffn.shared_down_proj);
        if (!st.ok()) return st;

        st = moe_ffn_weights_match(layer.ffn, plan, config.ffn);
        if (!st.ok()) return st;
    }

    if (options.load_mtp_layers) {
        Status mtp = load_moe_mtp(arena, stream, reader, config, options, out.mtp);
        if (!mtp.ok()) return mtp;
    }

    hipError_t sync_err = hipStreamSynchronize(stream);
    if (sync_err != hipSuccess) {
        return Status::hip_error("hipStreamSynchronize after qwen35moe weight load",
                                 hipGetErrorString(sync_err), __FILE__, __LINE__);
    }
    return out;
}

}  // namespace shelf::qwen35moe
