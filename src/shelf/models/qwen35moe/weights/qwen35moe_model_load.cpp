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

Status load_aux_matrix(const qqb::QuantizedModelReader& reader, std::size_t layer,
                       std::size_t aux, gpu::GpuArena& arena, hipStream_t stream,
                       const shelf::weights::WeightLoadOptions& weight_options,
                       MatrixWeight& dst) {
    auto m = shelf::weights::load_quantized_matrix(
        reader, moe_mlp_tensor_name(layer, kMoeAuxSuffixes[aux]), arena, stream, weight_options);
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

}  // namespace

std::vector<std::string> moe_aux_tensor_names(std::size_t layer) {
    std::vector<std::string> names;
    names.reserve(sizeof(kMoeAuxSuffixes) / sizeof(kMoeAuxSuffixes[0]));
    for (const char* suffix : kMoeAuxSuffixes) names.push_back(moe_mlp_tensor_name(layer, suffix));
    return names;
}

std::vector<std::string> moe_layer_logical_names(const MoeExpertGroupPlan& plan) {
    std::vector<std::string> names = moe_aux_tensor_names(plan.layer);
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
    return Status::make_ok();
}

Result<Qwen35MoeModelWeights> load_qwen35moe_weights(
    const std::string& model_dir,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const qhy::HybridLoadOptions& options) {
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

        st = load_aux_matrix(reader, l, kRouter, arena, stream, options.weights, layer.ffn.router);
        if (!st.ok()) return st;
        st = load_aux_matrix(reader, l, kSharedGate, arena, stream, options.weights,
                             layer.ffn.shared_gate);
        if (!st.ok()) return st;
        st = load_aux_matrix(reader, l, kSharedGateProj, arena, stream, options.weights,
                             layer.ffn.shared_gate_proj);
        if (!st.ok()) return st;
        st = load_aux_matrix(reader, l, kSharedUpProj, arena, stream, options.weights,
                             layer.ffn.shared_up_proj);
        if (!st.ok()) return st;
        st = load_aux_matrix(reader, l, kSharedDownProj, arena, stream, options.weights,
                             layer.ffn.shared_down_proj);
        if (!st.ok()) return st;

        st = moe_ffn_weights_match(layer.ffn, plan, config.ffn);
        if (!st.ok()) return st;
    }

    hipError_t sync_err = hipStreamSynchronize(stream);
    if (sync_err != hipSuccess) {
        return Status::hip_error("hipStreamSynchronize after qwen35moe weight load",
                                 hipGetErrorString(sync_err), __FILE__, __LINE__);
    }
    return out;
}

}  // namespace shelf::qwen35moe
