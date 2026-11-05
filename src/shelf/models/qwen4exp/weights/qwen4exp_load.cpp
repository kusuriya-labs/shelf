#include <shelf/models/qwen4exp/weights/qwen4exp_weights.h>

#include <shelf/models/common/qwen_hybrid/tensor_partition.h>
#include <shelf/models/common/qwen_hybrid/weight_loader.h>
#include <shelf/quantization/bundle/quantized_model_reader.h>
#include <shelf/weights/tensor_partition.h>
#include <shelf/weights/weight_loader.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace shelf {
namespace qwen4exp {
namespace {

namespace qhy = shelf::qwen_hybrid;
namespace qqb = shelf::quantization::bundle;
namespace qqm = shelf::quantization::manifest;
using shelf::gpu::GpuArena;
using shelf::gpu::Tensor;
using shelf::weights::MatrixWeight;
using shelf::weights::TensorPartitionPlan;
using shelf::weights::WeightLoadOptions;

constexpr std::uint64_t kTensorAlignment = 256u;
constexpr const char* kLayerFormat = "model.language_model.layers.%zu.";

std::string layer_prefix(std::size_t layer) {
    char buffer[80];
    std::snprintf(buffer, sizeof(buffer), kLayerFormat, layer);
    return std::string(buffer);
}

std::uint64_t tensor_bytes(const Tensor& tensor) {
    if (tensor.ndim() == 0) return 0;
    std::size_t bytes = 0;
    if (!tensor.required_span_bytes(bytes).ok()) return 0;
    return bytes;
}

std::uint64_t matrix_bytes(const MatrixWeight& weight) {
    const Tensor* tensors[] = {
        &weight.bf16,          &weight.data,          &weight.codes,
        &weight.scales,        &weight.row_ref,       &weight.compute_codes,
        &weight.compute_scales_bf16, &weight.exl3_suh, &weight.exl3_svh,
        &weight.exl3_marker};
    std::uint64_t total = 0;
    for (const Tensor* tensor : tensors) total += tensor_bytes(*tensor);
    return total;
}

Status checked_add(std::uint64_t value, std::uint64_t& total) {
    if (value > UINT64_MAX - total) {
        return Status::overflow("Qwen4Exp resident byte count overflow", __FILE__, __LINE__);
    }
    total += value;
    return Status::make_ok();
}

Status add_matrix(std::uint64_t& total, const MatrixWeight& weight) {
    return checked_add(matrix_bytes(weight), total);
}

Status add_tensor(std::uint64_t& total, const Tensor& tensor) {
    return checked_add(tensor_bytes(tensor), total);
}

Status add_plane(std::uint64_t& total, const Qwen4ExpExpertPlane& plane) {
    return checked_add(plane.resident_bytes(), total);
}

Status add_hyper(std::uint64_t& total, const Qwen4ExpHyperConnectionWeights& hc) {
    Status st = add_matrix(total, hc.input_mix_down);
    if (!st.ok()) return st;
    st = add_matrix(total, hc.input_mix_up);
    if (!st.ok()) return st;
    st = add_matrix(total, hc.block_inject);
    if (!st.ok()) return st;
    return add_tensor(total, hc.hc_norm);
}

Tensor alloc_u8(std::uint64_t bytes, const std::vector<std::size_t>& shape,
                const std::vector<std::size_t>& strides, GpuArena& arena) {
    auto view = arena.allocate_aligned(static_cast<std::size_t>(bytes), kTensorAlignment);
    if (!view.ok()) return {};
    auto tensor = Tensor::create<std::uint8_t>(view.release(), shape, strides);
    if (!tensor.ok()) return {};
    return tensor.release();
}

Tensor alloc_u16(std::uint64_t bytes, const std::vector<std::size_t>& shape,
                 const std::vector<std::size_t>& strides, GpuArena& arena) {
    auto view = arena.allocate_aligned(static_cast<std::size_t>(bytes), kTensorAlignment);
    if (!view.ok()) return {};
    auto tensor = Tensor::create<std::uint16_t>(view.release(), shape, strides);
    if (!tensor.ok()) return {};
    return tensor.release();
}

Status copy_async(void* dst, const void* src, std::size_t bytes, hipStream_t stream,
                  const char* what) {
    const hipError_t error = hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice, stream);
    if (error != hipSuccess) {
        return Status::hip_error(what, hipGetErrorString(error), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Result<TensorPartitionPlan> build_tp_plan(const Qwen4ExpTextConfig& config,
                                          std::uint32_t tp_size,
                                          std::uint32_t tp_rank) {
    TensorPartitionPlan plan;
    plan.tp_size = tp_size;
    plan.tp_rank = tp_rank;
    if (tp_size == 1u) return plan;
    if (tp_size != 2u) {
        return Status::unsupported("Qwen4Exp load path currently supports TP=2 only",
                                   __FILE__, __LINE__);
    }
    if (config.ffn.num_experts == 0u || (config.ffn.num_experts % tp_size) != 0u) {
        return Status::unsupported("Qwen4Exp num_experts must divide by tp_size", __FILE__,
                                   __LINE__);
    }
    // EXL3 matrices are not part of the canonical partition contract yet. The first load
    // path therefore splits only the routed experts on the expert axis and replicates
    // every non-expert matrix on both ranks.
    return plan;
}

std::vector<std::string> hyper_names(const std::string& prefix) {
    return {
        prefix + "input_mix_weight_down.weight",
        prefix + "input_mix_weight_up.weight",
        prefix + "block_inject_weight.weight",
        prefix + "hc_norm.weight",
    };
}

std::vector<std::string> attention_names(const std::string& prefix, bool is_gdn) {
    std::vector<std::string> names;
    if (is_gdn) {
        names = {
            prefix + "linear_attn.in_proj_qkv.weight",
            prefix + "linear_attn.in_proj_z.weight",
            prefix + "linear_attn.in_proj_a.weight",
            prefix + "linear_attn.in_proj_b.weight",
            prefix + "linear_attn.out_proj.weight",
            prefix + "linear_attn.conv1d.weight",
            prefix + "linear_attn.norm.weight",
            prefix + "linear_attn.A_log",
            prefix + "linear_attn.dt_bias",
        };
    } else {
        names = {
            prefix + "self_attn.q_proj.weight",
            prefix + "self_attn.k_proj.weight",
            prefix + "self_attn.v_proj.weight",
            prefix + "self_attn.o_proj.weight",
            prefix + "self_attn.q_norm.weight",
            prefix + "self_attn.k_norm.weight",
            prefix + "self_attn.indexer.index_qk_proj.weight",
            prefix + "self_attn.indexer.q_layernorm.weight",
            prefix + "self_attn.indexer.k_layernorm.weight",
        };
    }
    return names;
}

std::vector<std::string> moe_names(const std::string& prefix, std::uint32_t expert_begin,
                                   std::uint32_t local_experts) {
    std::vector<std::string> names = {
        prefix + "mlp.gate.weight",
        prefix + "mlp.shared_expert_gate.weight",
        prefix + "mlp.shared_expert.gate_proj.weight",
        prefix + "mlp.shared_expert.up_proj.weight",
        prefix + "mlp.shared_expert.down_proj.weight",
    };
    names.reserve(names.size() + static_cast<std::size_t>(local_experts) * 3u);
    for (std::uint32_t local = 0; local < local_experts; ++local) {
        const std::uint32_t expert = expert_begin + local;
        const std::string expert_prefix =
            prefix + "mlp.experts." + std::to_string(expert) + ".";
        names.push_back(expert_prefix + "gate_proj.weight");
        names.push_back(expert_prefix + "up_proj.weight");
        names.push_back(expert_prefix + "down_proj.weight");
    }
    return names;
}

std::vector<std::string> ple_names(const std::string& prefix) {
    return {
        prefix + "ple.key_proj.weight",
        prefix + "ple.value_proj.weight",
        prefix + "ple.conv1d.weight",
        prefix + "ple.norm_key.weight",
        prefix + "ple.norm_query.weight",
        prefix + "ple.norm_conv.weight",
    };
}

Status load_matrix(const qqb::QuantizedModelReader& reader, const std::string& name,
                   GpuArena& arena, hipStream_t stream, const WeightLoadOptions& options,
                   MatrixWeight& out) {
    auto matrix = shelf::weights::load_quantized_matrix(reader, name, arena, stream, options);
    if (!matrix.ok()) {
        std::fprintf(stderr, "Qwen4Exp matrix load %s: %s\n", name.c_str(),
                     matrix.status().message().c_str());
        return matrix.status();
    }
    out = matrix.release();
    return Status::make_ok();
}

Status load_small(const qqb::QuantizedModelReader& reader, const std::string& name,
                  GpuArena& arena, hipStream_t stream, const WeightLoadOptions& options,
                  Tensor& out) {
    auto tensor = shelf::weights::load_quantized_small(reader, name, arena, stream, options);
    if (!tensor.ok()) {
        std::fprintf(stderr, "Qwen4Exp tensor load %s: %s\n", name.c_str(),
                     tensor.status().message().c_str());
        return tensor.status();
    }
    out = tensor.release();
    return Status::make_ok();
}

Status load_hyper(const qqb::QuantizedModelReader& reader, const std::string& prefix,
                  GpuArena& arena, hipStream_t stream, const WeightLoadOptions& options,
                  Qwen4ExpHyperConnectionWeights& out) {
    const std::vector<std::string> names = hyper_names(prefix);
    Status st = load_matrix(reader, names[0], arena, stream, options, out.input_mix_down);
    if (!st.ok()) return st;
    st = load_matrix(reader, names[1], arena, stream, options, out.input_mix_up);
    if (!st.ok()) return st;
    st = load_matrix(reader, names[2], arena, stream, options, out.block_inject);
    if (!st.ok()) return st;
    return load_small(reader, names[3], arena, stream, options, out.hc_norm);
}

Status load_mixer(const qqb::QuantizedModelReader& reader, GpuArena& arena,
                  hipStream_t stream, const WeightLoadOptions& options,
                  Qwen4ExpHyperConnectionWeights& out) {
    const std::string prefix = "model.language_model.hyper_connection_mixer.";
    Status st = load_matrix(reader, prefix + "input_mix_weight_down.weight", arena, stream,
                            options, out.input_mix_down);
    if (!st.ok()) return st;
    st = load_matrix(reader, prefix + "input_mix_weight_up.weight", arena, stream, options,
                     out.input_mix_up);
    if (!st.ok()) return st;
    return load_small(reader, prefix + "hc_norm.weight", arena, stream, options, out.hc_norm);
}

Status load_attention(const qqb::QuantizedModelReader& reader, const std::string& prefix,
                      bool is_gdn, GpuArena& arena, hipStream_t stream,
                      const WeightLoadOptions& options, Qwen4ExpAttentionWeights& out) {
    out.is_gdn = is_gdn;
    if (is_gdn) {
        Status st = load_matrix(reader, prefix + "linear_attn.in_proj_qkv.weight", arena,
                                stream, options, out.in_proj_qkv);
        if (!st.ok()) return st;
        st = load_matrix(reader, prefix + "linear_attn.in_proj_z.weight", arena, stream,
                         options, out.in_proj_z);
        if (!st.ok()) return st;
        st = load_matrix(reader, prefix + "linear_attn.in_proj_a.weight", arena, stream,
                         options, out.in_proj_a);
        if (!st.ok()) return st;
        st = load_matrix(reader, prefix + "linear_attn.in_proj_b.weight", arena, stream,
                         options, out.in_proj_b);
        if (!st.ok()) return st;
        st = load_matrix(reader, prefix + "linear_attn.out_proj.weight", arena, stream,
                         options, out.out_proj);
        if (!st.ok()) return st;
        st = load_small(reader, prefix + "linear_attn.conv1d.weight", arena, stream, options,
                        out.conv1d);
        if (!st.ok()) return st;
        st = load_small(reader, prefix + "linear_attn.norm.weight", arena, stream, options,
                        out.norm);
        if (!st.ok()) return st;
        st = load_small(reader, prefix + "linear_attn.A_log", arena, stream, options,
                        out.a_log);
        if (!st.ok()) return st;
        return load_small(reader, prefix + "linear_attn.dt_bias", arena, stream, options,
                          out.dt_bias);
    }

    Status st = load_matrix(reader, prefix + "self_attn.q_proj.weight", arena, stream,
                            options, out.q_proj);
    if (!st.ok()) return st;
    st = load_matrix(reader, prefix + "self_attn.k_proj.weight", arena, stream, options,
                     out.k_proj);
    if (!st.ok()) return st;
    st = load_matrix(reader, prefix + "self_attn.v_proj.weight", arena, stream, options,
                     out.v_proj);
    if (!st.ok()) return st;
    st = load_matrix(reader, prefix + "self_attn.o_proj.weight", arena, stream, options,
                     out.o_proj);
    if (!st.ok()) return st;
    st = load_small(reader, prefix + "self_attn.q_norm.weight", arena, stream, options,
                    out.q_norm);
    if (!st.ok()) return st;
    st = load_small(reader, prefix + "self_attn.k_norm.weight", arena, stream, options,
                    out.k_norm);
    if (!st.ok()) return st;
    st = load_matrix(reader, prefix + "self_attn.indexer.index_qk_proj.weight", arena, stream,
                     options, out.index_qk_proj);
    if (!st.ok()) return st;
    st = load_small(reader, prefix + "self_attn.indexer.q_layernorm.weight", arena, stream,
                    options, out.index_q_layernorm);
    if (!st.ok()) return st;
    return load_small(reader, prefix + "self_attn.indexer.k_layernorm.weight", arena, stream,
                      options, out.index_k_layernorm);
}

Result<Qwen4ExpExpertPlane> load_expert_plane(const qqb::QuantizedModelReader& reader,
                                               const std::string& prefix,
                                               const char* projection,
                                               std::uint32_t global_experts,
                                               std::uint32_t local_experts,
                                               std::uint32_t expert_begin,
                                               std::uint32_t k,
                                               std::uint32_t n,
                                               GpuArena& arena,
                                               hipStream_t stream) {
    const std::string first_name =
        prefix + "mlp.experts." + std::to_string(expert_begin) + "." + projection + ".weight";
    auto first = reader.resolve(first_name);
    if (!first.ok()) return first.status();
    const qqb::QuantizedTensorView& view = first.value();
    if (view.encoding != qqm::QuantizedEncoding::Exl3) {
        return Status::unsupported(
            ("Qwen4Exp routed expert is not EXL3: " + first_name).c_str(), __FILE__, __LINE__);
    }
    if (view.logical_shape.size() != 2 ||
        static_cast<std::uint32_t>(view.logical_shape[0]) != n ||
        static_cast<std::uint32_t>(view.logical_shape[1]) != k ||
        static_cast<std::uint32_t>(view.k_padded) != k) {
        return Status::invalid_argument(
            ("Qwen4Exp expert logical shape differs from the config: " + first_name).c_str(),
            __FILE__, __LINE__);
    }
    const std::uint32_t ka = view.exl3_ka;
    const bool half = view.exl3_half;
    if (ka < 1u || ka > 8u || (half && ka > 3u)) {
        return Status::invalid_argument(
            ("Qwen4Exp expert bit rate is invalid: " + first_name).c_str(), __FILE__,
            __LINE__);
    }
    if ((k % 128u) != 0u || (n % 128u) != 0u) {
        return Status::invalid_argument(
            ("Qwen4Exp expert shape is not 128-aligned: " + first_name).c_str(), __FILE__,
            __LINE__);
    }
    const std::uint32_t words = 16u * ka + (half ? 8u : 0u);
    const std::uint64_t trellis_bytes =
        static_cast<std::uint64_t>(k / 16u) * (n / 16u) * words * 2u;
    const std::uint64_t suh_bytes = static_cast<std::uint64_t>(k) * 2u;
    const std::uint64_t svh_bytes = static_cast<std::uint64_t>(n) * 2u;

    Qwen4ExpExpertPlane plane;
    plane.global_experts = global_experts;
    plane.local_experts = local_experts;
    plane.expert_begin = expert_begin;
    plane.rows = n;
    plane.cols = k;
    plane.k_padded = k;
    plane.ka = ka;
    plane.half = half;
    plane.trellis_bytes_per_expert = trellis_bytes;
    plane.suh_bytes_per_expert = suh_bytes;
    plane.svh_bytes_per_expert = svh_bytes;

    plane.codes = alloc_u8(trellis_bytes * local_experts,
                           {local_experts, static_cast<std::size_t>(trellis_bytes)},
                           {static_cast<std::size_t>(trellis_bytes), 1u}, arena);
    if (plane.codes.ndim() == 0) {
        return Status::insufficient_memory(
            "Qwen4Exp expert trellis plane allocation failed", __FILE__, __LINE__);
    }
    plane.suh = alloc_u16(suh_bytes * local_experts, {local_experts, k}, {k, 1u}, arena);
    if (plane.suh.ndim() == 0) {
        return Status::insufficient_memory(
            "Qwen4Exp expert suh plane allocation failed", __FILE__, __LINE__);
    }
    plane.svh = alloc_u16(svh_bytes * local_experts, {local_experts, n}, {n, 1u}, arena);
    if (plane.svh.ndim() == 0) {
        return Status::insufficient_memory(
            "Qwen4Exp expert svh plane allocation failed", __FILE__, __LINE__);
    }
    plane.marker = alloc_u8(4u * local_experts, {local_experts, 4u}, {4u, 1u}, arena);
    if (plane.marker.ndim() == 0) {
        return Status::insufficient_memory(
            "Qwen4Exp expert marker plane allocation failed", __FILE__, __LINE__);
    }

    for (std::uint32_t local = 0; local < local_experts; ++local) {
        const std::uint32_t expert = expert_begin + local;
        const std::string name =
            prefix + "mlp.experts." + std::to_string(expert) + "." + projection + ".weight";
        auto resolved = reader.resolve(name);
        if (!resolved.ok()) return resolved.status();
        const qqb::QuantizedTensorView& value = resolved.value();
        if (value.encoding != qqm::QuantizedEncoding::Exl3 || value.exl3_ka != ka ||
            value.exl3_half != half) {
            return Status::invalid_argument(
                ("Qwen4Exp expert encoding differs inside one layer: " + name).c_str(),
                __FILE__, __LINE__);
        }
        if (value.codes.size != trellis_bytes || value.suh.size != suh_bytes ||
            value.svh.size != svh_bytes || value.marker.size != 4u) {
            return Status::invalid_argument(
                ("Qwen4Exp expert payload size differs from the geometry: " + name).c_str(),
                __FILE__, __LINE__);
        }
        Status st = copy_async(plane.codes.data<std::uint8_t>() + local * trellis_bytes,
                               value.codes.data, static_cast<std::size_t>(trellis_bytes), stream,
                               "Qwen4Exp expert trellis upload");
        if (!st.ok()) return st;
        st = copy_async(plane.suh.data<std::uint16_t>() + static_cast<std::size_t>(local) * k,
                        value.suh.data, static_cast<std::size_t>(suh_bytes), stream,
                        "Qwen4Exp expert suh upload");
        if (!st.ok()) return st;
        st = copy_async(plane.svh.data<std::uint16_t>() + static_cast<std::size_t>(local) * n,
                        value.svh.data, static_cast<std::size_t>(svh_bytes), stream,
                        "Qwen4Exp expert svh upload");
        if (!st.ok()) return st;
        st = copy_async(plane.marker.data<std::uint8_t>() + static_cast<std::size_t>(local) * 4u,
                        value.marker.data, 4u, stream, "Qwen4Exp expert marker upload");
        if (!st.ok()) return st;
    }
    return plane;
}

Status load_moe(const qqb::QuantizedModelReader& reader, const std::string& prefix,
                const Qwen4ExpTextConfig& config, std::uint32_t tp_size,
                std::uint32_t tp_rank, GpuArena& arena, hipStream_t stream,
                const WeightLoadOptions& options, Qwen4ExpMoeWeights& out) {
    Status st = load_matrix(reader, prefix + "mlp.gate.weight", arena, stream, options,
                            out.router);
    if (!st.ok()) return st;
    st = load_matrix(reader, prefix + "mlp.shared_expert_gate.weight", arena, stream, options,
                     out.shared_gate);
    if (!st.ok()) return st;
    st = load_matrix(reader, prefix + "mlp.shared_expert.gate_proj.weight", arena, stream,
                     options, out.shared_gate_proj);
    if (!st.ok()) return st;
    st = load_matrix(reader, prefix + "mlp.shared_expert.up_proj.weight", arena, stream,
                     options, out.shared_up_proj);
    if (!st.ok()) return st;
    st = load_matrix(reader, prefix + "mlp.shared_expert.down_proj.weight", arena, stream,
                     options, out.shared_down_proj);
    if (!st.ok()) return st;

    const std::uint32_t global_experts = static_cast<std::uint32_t>(config.ffn.num_experts);
    const std::uint32_t local_experts = global_experts / tp_size;
    const std::uint32_t expert_begin = tp_rank * local_experts;
    const std::uint32_t hidden = static_cast<std::uint32_t>(config.hidden_size);
    const std::uint32_t intermediate =
        static_cast<std::uint32_t>(config.ffn.moe_intermediate_size);

    auto gate = load_expert_plane(reader, prefix, "gate_proj", global_experts, local_experts,
                                  expert_begin, hidden, intermediate, arena, stream);
    if (!gate.ok()) return gate.status();
    out.gate_experts = gate.release();
    auto up = load_expert_plane(reader, prefix, "up_proj", global_experts, local_experts,
                                expert_begin, hidden, intermediate, arena, stream);
    if (!up.ok()) return up.status();
    out.up_experts = up.release();
    auto down = load_expert_plane(reader, prefix, "down_proj", global_experts, local_experts,
                                  expert_begin, intermediate, hidden, arena, stream);
    if (!down.ok()) return down.status();
    out.down_experts = down.release();
    return Status::make_ok();
}

Status load_ple(const qqb::QuantizedModelReader& reader, const std::string& prefix,
                GpuArena& arena, hipStream_t stream, const WeightLoadOptions& options,
                Qwen4ExpPleWeights& out) {
    Status st = load_matrix(reader, prefix + "ple.key_proj.weight", arena, stream, options,
                            out.key_proj);
    if (!st.ok()) return st;
    st = load_matrix(reader, prefix + "ple.value_proj.weight", arena, stream, options,
                     out.value_proj);
    if (!st.ok()) return st;
    st = load_small(reader, prefix + "ple.conv1d.weight", arena, stream, options, out.conv1d);
    if (!st.ok()) return st;
    st = load_small(reader, prefix + "ple.norm_key.weight", arena, stream, options,
                    out.norm_key);
    if (!st.ok()) return st;
    st = load_small(reader, prefix + "ple.norm_query.weight", arena, stream, options,
                    out.norm_query);
    if (!st.ok()) return st;
    st = load_small(reader, prefix + "ple.norm_conv.weight", arena, stream, options,
                    out.norm_conv);
    if (!st.ok()) return st;
    out.present = true;
    return Status::make_ok();
}

}  // namespace

shelf::weights::MatrixWeight make_expert_matrix(const Qwen4ExpExpertPlane& plane) {
    shelf::weights::MatrixWeight w;
    w.encoding = shelf::weights::MatrixEncoding::Exl3;
    w.rows = plane.rows;
    w.cols = plane.cols;
    w.k_padded = plane.k_padded;
    w.experts = plane.local_experts;
    w.codes = plane.codes;
    w.exl3_suh = plane.suh;
    w.exl3_svh = plane.svh;
    w.exl3_marker = plane.marker;
    w.exl3_ka = plane.ka;
    w.exl3_half = plane.half;
    const std::uint64_t words = 16u * plane.ka + (plane.half ? 8u : 0u);
    w.codes_row_stride_bytes =
        static_cast<std::uint32_t>((static_cast<std::uint64_t>(plane.rows) / 16u) * words * 2u);
    w.codes_batch_stride_bytes = plane.trellis_bytes_per_expert;
    w.exl3_suh_batch_stride_bytes = plane.suh_bytes_per_expert;
    w.exl3_svh_batch_stride_bytes = plane.svh_bytes_per_expert;
    w.exl3_marker_batch_stride_bytes = 4u;
    w.preshuffled = false;
    shelf::weights::apply_default_compute_spec(w);
    return w;
}

std::uint64_t Qwen4ExpExpertPlane::resident_bytes() const noexcept {
    return static_cast<std::uint64_t>(local_experts) *
           (trellis_bytes_per_expert + suh_bytes_per_expert + svh_bytes_per_expert + 4u);
}

std::uint64_t Qwen4ExpRankWeights::resident_bytes() const noexcept {
    std::uint64_t total = 0;
    (void)add_matrix(total, embed_tokens);
    (void)add_matrix(total, lm_head);
    (void)add_hyper(total, mixer);
    for (const Qwen4ExpLayerWeights& layer : layers) {
        (void)add_hyper(total, layer.attn_hyper_connection);
        (void)add_hyper(total, layer.mlp_hyper_connection);
        (void)add_matrix(total, layer.attention.in_proj_qkv);
        (void)add_matrix(total, layer.attention.in_proj_z);
        (void)add_matrix(total, layer.attention.in_proj_a);
        (void)add_matrix(total, layer.attention.in_proj_b);
        (void)add_matrix(total, layer.attention.out_proj);
        (void)add_tensor(total, layer.attention.conv1d);
        (void)add_tensor(total, layer.attention.norm);
        (void)add_tensor(total, layer.attention.a_log);
        (void)add_tensor(total, layer.attention.dt_bias);
        (void)add_matrix(total, layer.attention.q_proj);
        (void)add_matrix(total, layer.attention.k_proj);
        (void)add_matrix(total, layer.attention.v_proj);
        (void)add_matrix(total, layer.attention.o_proj);
        (void)add_tensor(total, layer.attention.q_norm);
        (void)add_tensor(total, layer.attention.k_norm);
        (void)add_matrix(total, layer.attention.index_qk_proj);
        (void)add_tensor(total, layer.attention.index_q_layernorm);
        (void)add_tensor(total, layer.attention.index_k_layernorm);
        (void)add_matrix(total, layer.mlp.router);
        (void)add_matrix(total, layer.mlp.shared_gate);
        (void)add_matrix(total, layer.mlp.shared_gate_proj);
        (void)add_matrix(total, layer.mlp.shared_up_proj);
        (void)add_matrix(total, layer.mlp.shared_down_proj);
        (void)add_plane(total, layer.mlp.gate_experts);
        (void)add_plane(total, layer.mlp.up_experts);
        (void)add_plane(total, layer.mlp.down_experts);
        (void)add_matrix(total, layer.ple.key_proj);
        (void)add_matrix(total, layer.ple.value_proj);
        (void)add_tensor(total, layer.ple.conv1d);
        (void)add_tensor(total, layer.ple.norm_key);
        (void)add_tensor(total, layer.ple.norm_query);
        (void)add_tensor(total, layer.ple.norm_conv);
    }
    return total;
}

std::vector<std::string> qwen4exp_rank_logical_names(const Qwen4ExpTextConfig& config,
                                                     std::uint32_t tp_size,
                                                     std::uint32_t tp_rank) {
    std::vector<std::string> names = {
        "model.language_model.embed_tokens.weight",
        "model.language_model.lm_head.weight",
        "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
        "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
        "model.language_model.hyper_connection_mixer.hc_norm.weight",
    };
    const std::uint32_t local_experts =
        static_cast<std::uint32_t>(config.ffn.num_experts) / tp_size;
    const std::uint32_t expert_begin = tp_rank * local_experts;
    for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        const std::string prefix = layer_prefix(layer);
        const std::vector<std::string> hc_attn = hyper_names(prefix + "attn_hyper_connection.");
        const std::vector<std::string> hc_mlp = hyper_names(prefix + "mlp_hyper_connection.");
        names.insert(names.end(), hc_attn.begin(), hc_attn.end());
        names.insert(names.end(), hc_mlp.begin(), hc_mlp.end());
        const std::vector<std::string> attn =
            attention_names(prefix, qhy::layer_is_gdn(config, layer));
        names.insert(names.end(), attn.begin(), attn.end());
        const std::vector<std::string> moe = moe_names(prefix, expert_begin, local_experts);
        names.insert(names.end(), moe.begin(), moe.end());
        if (config.layer_has_ple(layer)) {
            const std::vector<std::string> ple = ple_names(prefix);
            names.insert(names.end(), ple.begin(), ple.end());
        }
    }
    return names;
}

Result<Qwen4ExpRankWeights> load_qwen4exp_rank(const std::string& model_dir,
                                               GpuArena& arena,
                                               hipStream_t stream,
                                               const Qwen4ExpLoadOptions& options) {
    if (options.tp_size != 1u && options.tp_size != 2u) {
        return Status::unsupported("Qwen4Exp load supports tp_size 1 or 2", __FILE__, __LINE__);
    }
    if (options.tp_rank >= options.tp_size) {
        return Status::invalid_argument("Qwen4Exp tp_rank out of range", __FILE__, __LINE__);
    }
    auto config_result = read_qwen4exp_text_config(model_dir);
    if (!config_result.ok()) return config_result.status();
    const Qwen4ExpTextConfig& config = config_result.value();

    auto plan_result = build_tp_plan(config, options.tp_size, options.tp_rank);
    if (!plan_result.ok()) return plan_result.status();
    const TensorPartitionPlan& plan = plan_result.value();

    const std::vector<std::string> names =
        qwen4exp_rank_logical_names(config, options.tp_size, options.tp_rank);
    auto reader_result = qhy::open_quantized_model_reader(
        model_dir, names, options.verify_quantized_payload_crc);
    if (!reader_result.ok()) return reader_result.status();
    const qqb::QuantizedModelReader reader = reader_result.release();

    WeightLoadOptions weight_options;
    weight_options.partition_plan = &plan;

    Qwen4ExpRankWeights out;
    out.layers.resize(config.num_hidden_layers);

    Status st = load_matrix(reader, "model.language_model.embed_tokens.weight", arena, stream,
                            weight_options, out.embed_tokens);
    if (!st.ok()) return st;
    st = load_matrix(reader, "model.language_model.lm_head.weight", arena, stream,
                     weight_options, out.lm_head);
    if (!st.ok()) return st;
    st = load_mixer(reader, arena, stream, weight_options, out.mixer);
    if (!st.ok()) return st;

    for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        const std::string prefix = layer_prefix(layer);
        Qwen4ExpLayerWeights& weights = out.layers[layer];
        st = load_hyper(reader, prefix + "attn_hyper_connection.", arena, stream,
                        weight_options, weights.attn_hyper_connection);
        if (!st.ok()) return st;
        st = load_hyper(reader, prefix + "mlp_hyper_connection.", arena, stream,
                        weight_options, weights.mlp_hyper_connection);
        if (!st.ok()) return st;
        st = load_attention(reader, prefix, qhy::layer_is_gdn(config, layer), arena, stream,
                            weight_options, weights.attention);
        if (!st.ok()) return st;
        st = load_moe(reader, prefix, config, options.tp_size, options.tp_rank, arena, stream,
                      weight_options, weights.mlp);
        if (!st.ok()) return st;
        if (config.layer_has_ple(layer)) {
            st = load_ple(reader, prefix, arena, stream, weight_options, weights.ple);
            if (!st.ok()) return st;
        }
        const hipError_t sync_error = hipStreamSynchronize(stream);
        if (sync_error != hipSuccess) {
            return Status::hip_error("Qwen4Exp layer upload", hipGetErrorString(sync_error),
                                     __FILE__, __LINE__);
        }
    }

    Status valid = validate_qwen4exp_rank(out, config, options.tp_size, options.tp_rank);
    if (!valid.ok()) return valid;
    return out;
}

namespace {

Status expect_matrix(const MatrixWeight& weight, std::uint32_t rows, std::uint32_t cols,
                     const char* what) {
    if (weight.rows != rows || weight.cols != cols) {
        return Status::invalid_argument(
            (std::string("Qwen4Exp weight geometry mismatch: ") + what).c_str(), __FILE__,
            __LINE__);
    }
    if (weight.encoding == shelf::weights::MatrixEncoding::Exl3) {
        if (weight.exl3_ka < 1u || weight.exl3_ka > 8u ||
            (weight.exl3_half && weight.exl3_ka > 3u)) {
            return Status::invalid_argument(
                (std::string("Qwen4Exp EXL3 rate is invalid: ") + what).c_str(), __FILE__,
                __LINE__);
        }
        if (weight.k_padded < cols || (weight.k_padded % 128u) != 0u) {
            return Status::invalid_argument(
                (std::string("Qwen4Exp EXL3 k_padded is invalid: ") + what).c_str(),
                __FILE__, __LINE__);
        }
    } else if (weight.encoding != shelf::weights::MatrixEncoding::Bf16) {
        return Status::invalid_argument(
            (std::string("Qwen4Exp matrix has an unexpected encoding: ") + what).c_str(),
            __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status expect_tensor_1d(const Tensor& tensor, std::size_t size, const char* what) {
    if (tensor.ndim() != 1u || tensor.dim(0) != size) {
        return Status::invalid_argument(
            (std::string("Qwen4Exp tensor is not [size]: ") + what).c_str(), __FILE__,
            __LINE__);
    }
    return Status::make_ok();
}

Status expect_conv(const Tensor& tensor, std::size_t features, std::size_t kernel,
                   const char* what) {
    if (tensor.ndim() != 3u || tensor.dim(0) != features || tensor.dim(1) != 1u ||
        tensor.dim(2) != kernel) {
        return Status::invalid_argument(
            (std::string("Qwen4Exp depthwise conv is not [features,1,kernel]: ") + what).c_str(),
            __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status expect_hyper(const Qwen4ExpHyperConnectionWeights& hc, std::uint32_t hidden,
                    std::uint32_t hc_count, std::uint32_t lowrank, const char* what) {
    const std::uint32_t stream_width = hc_count * hidden;
    Status st = expect_matrix(hc.input_mix_down, lowrank, stream_width, what);
    if (!st.ok()) return st;
    st = expect_matrix(hc.input_mix_up, stream_width, lowrank, what);
    if (!st.ok()) return st;
    st = expect_matrix(hc.block_inject, hc_count, stream_width, what);
    if (!st.ok()) return st;
    return expect_tensor_1d(hc.hc_norm, stream_width, what);
}

Status expect_mixer(const Qwen4ExpHyperConnectionWeights& hc, std::uint32_t hidden,
                    std::uint32_t hc_count, std::uint32_t lowrank, const char* what) {
    const std::uint32_t stream_width = hc_count * hidden;
    Status st = expect_matrix(hc.input_mix_down, lowrank, stream_width, what);
    if (!st.ok()) return st;
    st = expect_matrix(hc.input_mix_up, stream_width, lowrank, what);
    if (!st.ok()) return st;
    return expect_tensor_1d(hc.hc_norm, stream_width, what);
}

Status expect_plane(const Qwen4ExpExpertPlane& plane, std::uint32_t global_experts,
                    std::uint32_t local_experts, std::uint32_t expert_begin, std::uint32_t k,
                    std::uint32_t n, const char* what) {
    if (plane.global_experts != global_experts || plane.local_experts != local_experts ||
        plane.expert_begin != expert_begin || plane.cols != k || plane.rows != n ||
        plane.k_padded != k) {
        return Status::invalid_argument(
            (std::string("Qwen4Exp expert plane geometry mismatch: ") + what).c_str(),
            __FILE__, __LINE__);
    }
    const std::uint32_t words = 16u * plane.ka + (plane.half ? 8u : 0u);
    const std::uint64_t expected_trellis =
        static_cast<std::uint64_t>(k / 16u) * (n / 16u) * words * 2u;
    if (plane.trellis_bytes_per_expert != expected_trellis ||
        plane.suh_bytes_per_expert != k * 2u || plane.svh_bytes_per_expert != n * 2u) {
        return Status::invalid_argument(
            (std::string("Qwen4Exp expert plane stride mismatch: ") + what).c_str(), __FILE__,
            __LINE__);
    }
    if (plane.codes.ndim() != 2u || plane.codes.dim(0) != local_experts ||
        plane.codes.dim(1) != expected_trellis ||
        plane.suh.ndim() != 2u || plane.suh.dim(0) != local_experts ||
        plane.suh.dim(1) != k ||
        plane.svh.ndim() != 2u || plane.svh.dim(0) != local_experts ||
        plane.svh.dim(1) != n ||
        plane.marker.ndim() != 2u || plane.marker.dim(0) != local_experts ||
        plane.marker.dim(1) != 4u) {
        return Status::invalid_argument(
            (std::string("Qwen4Exp expert plane tensor shape mismatch: ") + what).c_str(),
            __FILE__, __LINE__);
    }
    return Status::make_ok();
}

}  // namespace

Status validate_qwen4exp_rank(const Qwen4ExpRankWeights& weights,
                              const Qwen4ExpTextConfig& config,
                              std::uint32_t tp_size,
                              std::uint32_t tp_rank) {
    if (tp_size != 1u && tp_size != 2u) {
        return Status::unsupported("Qwen4Exp validation supports tp_size 1 or 2", __FILE__,
                                   __LINE__);
    }
    if (tp_rank >= tp_size) {
        return Status::invalid_argument("Qwen4Exp validation tp_rank out of range", __FILE__,
                                        __LINE__);
    }
    if (weights.layers.size() != config.num_hidden_layers) {
        return Status::invalid_argument("Qwen4Exp layer count mismatch", __FILE__, __LINE__);
    }

    const std::uint32_t hidden = static_cast<std::uint32_t>(config.hidden_size);
    const std::uint32_t hc_count = static_cast<std::uint32_t>(config.hc_count);
    const std::uint32_t lowrank = static_cast<std::uint32_t>(config.hc_lowrank);
    const std::uint32_t stream_width = hc_count * hidden;
    const std::uint32_t vocab = static_cast<std::uint32_t>(config.vocab_size);
    const std::uint32_t global_experts = static_cast<std::uint32_t>(config.ffn.num_experts);
    const std::uint32_t local_experts = global_experts / tp_size;
    const std::uint32_t expert_begin = tp_rank * local_experts;
    const std::uint32_t intermediate =
        static_cast<std::uint32_t>(config.ffn.moe_intermediate_size);
    const std::uint32_t shared_intermediate =
        static_cast<std::uint32_t>(config.ffn.shared_expert_intermediate_size);
    const std::uint32_t local_shared = shared_intermediate;

    const std::uint32_t q_heads = static_cast<std::uint32_t>(config.num_attention_heads);
    const std::uint32_t kv_heads = static_cast<std::uint32_t>(config.num_key_value_heads);
    const std::uint32_t head_dim = static_cast<std::uint32_t>(config.attention_head_dim);
    const std::uint32_t q_features = q_heads * head_dim;
    const std::uint32_t kv_features = kv_heads * head_dim;

    const std::uint32_t gdn_key_heads =
        static_cast<std::uint32_t>(config.linear_num_key_heads);
    const std::uint32_t gdn_value_heads =
        static_cast<std::uint32_t>(config.linear_num_value_heads);
    const std::uint32_t gdn_key_features =
        gdn_key_heads * static_cast<std::uint32_t>(config.linear_key_head_dim);
    const std::uint32_t gdn_value_features =
        gdn_value_heads * static_cast<std::uint32_t>(config.linear_value_head_dim);
    const std::uint32_t gdn_conv_features =
        2u * gdn_key_features + gdn_value_features;

    Status st = expect_matrix(weights.embed_tokens, vocab, hidden, "embed_tokens");
    if (!st.ok()) return st;
    if (weights.embed_tokens.encoding != shelf::weights::MatrixEncoding::Bf16) {
        return Status::invalid_argument("Qwen4Exp embed_tokens is not BF16", __FILE__,
                                        __LINE__);
    }
    st = expect_matrix(weights.lm_head, vocab, hidden, "lm_head");
    if (!st.ok()) return st;
    if (weights.lm_head.encoding != shelf::weights::MatrixEncoding::Exl3) {
        return Status::invalid_argument("Qwen4Exp lm_head is not EXL3", __FILE__, __LINE__);
    }
    st = expect_mixer(weights.mixer, hidden, hc_count, lowrank, "hyper_connection_mixer");
    if (!st.ok()) return st;

    for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        const Qwen4ExpLayerWeights& w = weights.layers[layer];
        st = expect_hyper(w.attn_hyper_connection, hidden, hc_count, lowrank,
                          "attn_hyper_connection");
        if (!st.ok()) return st;
        st = expect_hyper(w.mlp_hyper_connection, hidden, hc_count, lowrank,
                          "mlp_hyper_connection");
        if (!st.ok()) return st;

        const bool is_gdn = qhy::layer_is_gdn(config, layer);
        if (w.attention.is_gdn != is_gdn) {
            return Status::invalid_argument("Qwen4Exp attention topology mismatch", __FILE__,
                                            __LINE__);
        }
        if (is_gdn) {
            st = expect_matrix(w.attention.in_proj_qkv, gdn_conv_features, hidden,
                               "in_proj_qkv");
            if (!st.ok()) return st;
            st = expect_matrix(w.attention.in_proj_z, gdn_value_features, hidden,
                               "in_proj_z");
            if (!st.ok()) return st;
            st = expect_matrix(w.attention.in_proj_a, gdn_value_heads, hidden,
                               "in_proj_a");
            if (!st.ok()) return st;
            st = expect_matrix(w.attention.in_proj_b, gdn_value_heads, hidden,
                               "in_proj_b");
            if (!st.ok()) return st;
            st = expect_matrix(w.attention.out_proj, hidden, gdn_value_features,
                               "out_proj");
            if (!st.ok()) return st;
            st = expect_conv(w.attention.conv1d, gdn_conv_features,
                             config.linear_conv_kernel_dim, "linear_attn.conv1d");
            if (!st.ok()) return st;
            st = expect_tensor_1d(w.attention.norm, config.linear_value_head_dim,
                                  "linear_attn.norm");
            if (!st.ok()) return st;
            st = expect_tensor_1d(w.attention.a_log, gdn_value_heads,
                                  "linear_attn.A_log");
            if (!st.ok()) return st;
            st = expect_tensor_1d(w.attention.dt_bias, gdn_value_heads,
                                  "linear_attn.dt_bias");
            if (!st.ok()) return st;
        } else {
            st = expect_matrix(w.attention.q_proj, 2u * q_features, hidden, "q_proj");
            if (!st.ok()) return st;
            st = expect_matrix(w.attention.k_proj, kv_features, hidden, "k_proj");
            if (!st.ok()) return st;
            st = expect_matrix(w.attention.v_proj, kv_features, hidden, "v_proj");
            if (!st.ok()) return st;
            st = expect_matrix(w.attention.o_proj, hidden, q_features, "o_proj");
            if (!st.ok()) return st;
            st = expect_tensor_1d(w.attention.q_norm, head_dim, "q_norm");
            if (!st.ok()) return st;
            st = expect_tensor_1d(w.attention.k_norm, head_dim, "k_norm");
            if (!st.ok()) return st;
            const std::uint32_t indexer_rows =
                static_cast<std::uint32_t>(config.indexer.n_heads) *
                    static_cast<std::uint32_t>(config.indexer.head_dim) +
                static_cast<std::uint32_t>(config.indexer.kv_heads) *
                    static_cast<std::uint32_t>(config.indexer.head_dim);
            st = expect_matrix(w.attention.index_qk_proj, indexer_rows, hidden,
                               "indexer.index_qk_proj");
            if (!st.ok()) return st;
            st = expect_tensor_1d(w.attention.index_q_layernorm, config.indexer.head_dim,
                                  "indexer.q_layernorm");
            if (!st.ok()) return st;
            st = expect_tensor_1d(w.attention.index_k_layernorm, config.indexer.head_dim,
                                  "indexer.k_layernorm");
            if (!st.ok()) return st;
        }

        st = expect_matrix(w.mlp.router, global_experts, hidden, "mlp.gate");
        if (!st.ok()) return st;
        st = expect_matrix(w.mlp.shared_gate, 1u, hidden, "mlp.shared_expert_gate");
        if (!st.ok()) return st;
        st = expect_matrix(w.mlp.shared_gate_proj, local_shared, hidden,
                           "shared_expert.gate_proj");
        if (!st.ok()) return st;
        st = expect_matrix(w.mlp.shared_up_proj, local_shared, hidden,
                           "shared_expert.up_proj");
        if (!st.ok()) return st;
        st = expect_matrix(w.mlp.shared_down_proj, hidden, local_shared,
                           "shared_expert.down_proj");
        if (!st.ok()) return st;

        st = expect_plane(w.mlp.gate_experts, global_experts, local_experts, expert_begin,
                          hidden, intermediate, "gate_experts");
        if (!st.ok()) return st;
        st = expect_plane(w.mlp.up_experts, global_experts, local_experts, expert_begin,
                          hidden, intermediate, "up_experts");
        if (!st.ok()) return st;
        st = expect_plane(w.mlp.down_experts, global_experts, local_experts, expert_begin,
                          intermediate, hidden, "down_experts");
        if (!st.ok()) return st;

        const bool has_ple = config.layer_has_ple(layer);
        if (w.ple.present != has_ple) {
            return Status::invalid_argument("Qwen4Exp PLE presence mismatch", __FILE__,
                                            __LINE__);
        }
        if (has_ple) {
            st = expect_matrix(w.ple.key_proj, stream_width,
                               config.ple.embed_dim, "ple.key_proj");
            if (!st.ok()) return st;
            st = expect_matrix(w.ple.value_proj, hidden, config.ple.embed_dim,
                               "ple.value_proj");
            if (!st.ok()) return st;
            st = expect_conv(w.ple.conv1d, stream_width, config.ple.conv_kernel_size,
                             "ple.conv1d");
            if (!st.ok()) return st;
            st = expect_tensor_1d(w.ple.norm_key, stream_width, "ple.norm_key");
            if (!st.ok()) return st;
            st = expect_tensor_1d(w.ple.norm_query, stream_width, "ple.norm_query");
            if (!st.ok()) return st;
            st = expect_tensor_1d(w.ple.norm_conv, stream_width, "ple.norm_conv");
            if (!st.ok()) return st;
        }
    }
    return Status::make_ok();
}

}  // namespace qwen4exp
}  // namespace shelf
