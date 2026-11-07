#include <shelf/models/qwen4exp/weights/qwen4exp_weights.h>

#include <shelf/models/common/qwen_hybrid/tensor_partition.h>
#include <shelf/models/common/qwen_hybrid/weight_loader.h>
#include <shelf/quantization/bundle/quantized_model_reader.h>
#include <shelf/quantization/mxfp4/mxfp4.h>
#include <shelf/quantization/mxfp4/mxfp4_native.h>
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

Status copy_from_staging(void* dst, const void* src, std::size_t bytes, hipStream_t stream,
                         const char* what) {
    Status st = copy_async(dst, src, bytes, stream, what);
    if (!st.ok()) return st;
    const hipError_t error = hipStreamSynchronize(stream);
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
    if (tp_size != 2u && tp_size != 4u) {
        return Status::unsupported("Qwen4Exp load path supports TP=2 or TP=4",
                                   __FILE__, __LINE__);
    }
    if (config.ffn.num_experts == 0u || (config.ffn.num_experts % tp_size) != 0u) {
        return Status::unsupported("Qwen4Exp num_experts must divide by tp_size", __FILE__,
                                   __LINE__);
    }
    auto body_result = qhy::build_hybrid_partition(config, tp_size, tp_rank);
    if (!body_result.ok()) return body_result.status();
    const qhy::HybridPartition& body = body_result.value();
    for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        Status st = qhy::add_hybrid_layer_partitions(
            plan, body, layer_prefix(layer), qhy::layer_is_gdn(config, layer));
        if (!st.ok()) return st;
    }
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

std::vector<std::string> mtp_names(const Qwen4ExpTextConfig& config) {
    const std::uint32_t experts = static_cast<std::uint32_t>(config.ffn.num_experts);
    std::vector<std::string> names = {
        "mtp.fc_hidden.weight",
        "mtp.fc_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
        "mtp.hyper_connection_mixer.hc_norm.weight",
    };
    const std::string layer_prefix = "mtp.layers.0.";
    const std::vector<std::string> mtp_attn_hyper =
        hyper_names(layer_prefix + "attn_hyper_connection.");
    const std::vector<std::string> mtp_mlp_hyper =
        hyper_names(layer_prefix + "mlp_hyper_connection.");
    const std::vector<std::string> mtp_attention =
        attention_names(layer_prefix, false);
    names.insert(names.end(), mtp_attn_hyper.begin(), mtp_attn_hyper.end());
    names.insert(names.end(), mtp_mlp_hyper.begin(), mtp_mlp_hyper.end());
    names.insert(names.end(), mtp_attention.begin(), mtp_attention.end());
    const std::vector<std::string> moe = moe_names(layer_prefix, 0u, experts);
    names.insert(names.end(), moe.begin(), moe.end());
    return names;
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

Status load_mixer_prefix(const qqb::QuantizedModelReader& reader, const std::string& prefix,
                         GpuArena& arena, hipStream_t stream,
                         const WeightLoadOptions& options,
                         Qwen4ExpHyperConnectionWeights& out) {
    Status st = load_matrix(reader, prefix + "input_mix_weight_down.weight", arena, stream,
                            options, out.input_mix_down);
    if (!st.ok()) return st;
    st = load_matrix(reader, prefix + "input_mix_weight_up.weight", arena, stream, options,
                     out.input_mix_up);
    if (!st.ok()) return st;
    return load_small(reader, prefix + "hc_norm.weight", arena, stream, options, out.hc_norm);
}

Status load_mixer(const qqb::QuantizedModelReader& reader, GpuArena& arena,
                  hipStream_t stream, const WeightLoadOptions& options,
                  Qwen4ExpHyperConnectionWeights& out) {
    return load_mixer_prefix(reader, "model.language_model.hyper_connection_mixer.", arena,
                             stream, options, out);
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


Result<MatrixWeight> load_mxfp4_expert_matrix(
    const qqb::QuantizedModelReader& reader, const std::string& prefix,
    const std::string& projection, std::uint32_t local_experts,
    std::uint32_t expert_begin, std::uint32_t rows, std::uint32_t cols,
    GpuArena& arena, hipStream_t stream) {
    MatrixWeight w;
    w.encoding = shelf::weights::MatrixEncoding::Mxfp4;
    w.rows = rows;
    w.cols = cols;
    w.k_padded = cols;
    w.experts = local_experts;
    w.weight_scale_group = shelf::quantization::kMxfp4ScaleBlockK;
    w.preshuffled = true;

    shelf::quantization::mxfp4::Mxfp4NativeHost native;
    bool row_ref_exact = true;
    std::uint64_t codes_batch = 0;
    std::uint64_t scales_batch = 0;
    for (std::uint32_t local = 0; local < local_experts; ++local) {
        const std::uint32_t expert = expert_begin + local;
        const std::string name =
            prefix + "mlp.experts." + std::to_string(expert) + "." + projection + ".weight";
        auto resolved = reader.resolve(name);
        if (!resolved.ok()) return resolved.status();
        const qqb::QuantizedTensorView& v = resolved.value();
        if (v.encoding != qqm::QuantizedEncoding::Mxfp4 || v.logical_shape.size() != 2u ||
            static_cast<std::uint64_t>(v.logical_shape[0]) != rows ||
            static_cast<std::uint64_t>(v.logical_shape[1]) != cols ||
            v.k_padded != cols) {
            return Status::invalid_argument(
                ("Qwen4Exp expert is not a native MXFP4 matrix: " + name).c_str(),
                __FILE__, __LINE__);
        }
        const std::uint64_t packed_k = cols / 2u;
        const std::uint64_t scale_k = cols / shelf::quantization::kMxfp4ScaleBlockK;
        if (v.codes.data == nullptr || v.codes.size != rows * packed_k ||
            v.metadata1.data == nullptr || v.metadata1.size != rows * scale_k) {
            return Status::invalid_argument(
                ("Qwen4Exp MXFP4 expert payload size mismatch: " + name).c_str(), __FILE__,
                __LINE__);
        }
        shelf::quantization::mxfp4::Mxfp4CanonicalView cv;
        cv.batch = 1;
        cv.n = rows;
        cv.padded_k = cols;
        cv.codes = static_cast<const std::uint8_t*>(v.codes.data);
        cv.codes_bytes = v.codes.size;
        cv.scales = static_cast<const std::uint8_t*>(v.metadata1.data);
        cv.scales_bytes = v.metadata1.size;
        if (!cv.validate() ||
            !shelf::quantization::mxfp4::preshuffle_mxfp4_native(cv, native)) {
            return Status::invalid_argument(
                ("Qwen4Exp MXFP4 native preshuffle failed: " + name).c_str(), __FILE__,
                __LINE__);
        }
        row_ref_exact = row_ref_exact && native.row_ref_exact;
        if (local == 0u) {
            codes_batch = native.codes.size();
            scales_batch = native.scales.size();
            w.codes_row_stride_bytes =
                static_cast<std::uint32_t>(native.codes_row_stride_bytes);
            w.scale_row_stride_bytes =
                static_cast<std::uint32_t>(native.scale_row_stride_bytes);
            w.storage_scale_stride_bytes = w.scale_row_stride_bytes;
            w.codes_batch_stride_bytes = codes_batch;
            w.scales_batch_stride_bytes = scales_batch;
            const std::uint64_t tiles =
                rows / shelf::quantization::mxfp4::mxfp4_native_output_tile;
            w.codes = alloc_u8(
                codes_batch * local_experts,
                {local_experts, static_cast<std::size_t>(tiles),
                 static_cast<std::size_t>(w.codes_row_stride_bytes)},
                {static_cast<std::size_t>(codes_batch),
                 static_cast<std::size_t>(w.codes_row_stride_bytes), 1u},
                arena);
            w.scales = alloc_u8(
                scales_batch * local_experts,
                {local_experts, static_cast<std::size_t>(rows),
                 static_cast<std::size_t>(w.scale_row_stride_bytes)},
                {static_cast<std::size_t>(scales_batch),
                 static_cast<std::size_t>(w.scale_row_stride_bytes), 1u},
                arena);
            w.row_ref = alloc_u8(
                static_cast<std::uint64_t>(rows) * local_experts,
                {local_experts, static_cast<std::size_t>(rows)},
                {static_cast<std::size_t>(rows), 1u}, arena);
            if (w.codes.ndim() == 0 || w.scales.ndim() == 0 || w.row_ref.ndim() == 0) {
                return Status::insufficient_memory(
                    "Qwen4Exp MXFP4 expert plane allocation failed", __FILE__, __LINE__);
            }
        } else if (native.codes.size() != codes_batch ||
                   native.scales.size() != scales_batch ||
                   static_cast<std::uint32_t>(native.codes_row_stride_bytes) !=
                       w.codes_row_stride_bytes ||
                   static_cast<std::uint32_t>(native.scale_row_stride_bytes) !=
                       w.scale_row_stride_bytes ||
                   static_cast<std::uint64_t>(rows) != native.row_ref.size() / 1u) {
            return Status::invalid_argument(
                ("Qwen4Exp MXFP4 expert geometry differs inside one layer: " + name).c_str(),
                __FILE__, __LINE__);
        }
        Status st = copy_async(w.codes.data<std::uint8_t>() + local * codes_batch,
                               native.codes.data(), static_cast<std::size_t>(codes_batch),
                               stream, "Qwen4Exp MXFP4 expert codes upload");
        if (!st.ok()) return st;
        st = copy_async(w.scales.data<std::uint8_t>() + local * scales_batch,
                        native.scales.data(), static_cast<std::size_t>(scales_batch), stream,
                        "Qwen4Exp MXFP4 expert scales upload");
        if (!st.ok()) return st;
        st = copy_from_staging(
            w.row_ref.data<std::uint8_t>() + static_cast<std::uint64_t>(local) * rows,
            native.row_ref.data(), static_cast<std::size_t>(rows), stream,
            "Qwen4Exp MXFP4 expert row_ref upload");
        if (!st.ok()) return st;
    }
    w.row_ref_exact = row_ref_exact;
    shelf::weights::apply_default_compute_spec(w);
    return w;
}

Result<MatrixWeight> load_expert_matrix(const qqb::QuantizedModelReader& reader,
                                        const std::string& prefix,
                                        const char* projection,
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
    if (view.encoding == qqm::QuantizedEncoding::Mxfp4) {
        return load_mxfp4_expert_matrix(reader, prefix, projection, local_experts,
                                        expert_begin, n, k, arena, stream);
    }
    if (view.encoding != qqm::QuantizedEncoding::Exl3) {
        return Status::unsupported(
            ("Qwen4Exp routed expert is not EXL3 or MXFP4: " + first_name).c_str(),
            __FILE__, __LINE__);
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

    MatrixWeight w;
    w.encoding = shelf::weights::MatrixEncoding::Exl3;
    w.rows = n;
    w.cols = k;
    w.k_padded = k;
    w.experts = local_experts;
    w.exl3_ka = ka;
    w.exl3_half = half;
    w.preshuffled = false;
    w.codes_row_stride_bytes =
        static_cast<std::uint32_t>((static_cast<std::uint64_t>(n) / 16u) * words * 2u);
    w.codes_batch_stride_bytes = trellis_bytes;
    w.exl3_suh_batch_stride_bytes = suh_bytes;
    w.exl3_svh_batch_stride_bytes = svh_bytes;
    w.exl3_marker_batch_stride_bytes = 4u;

    w.codes = alloc_u8(trellis_bytes * local_experts,
                       {local_experts, static_cast<std::size_t>(trellis_bytes)},
                       {static_cast<std::size_t>(trellis_bytes), 1u}, arena);
    if (w.codes.ndim() == 0) {
        return Status::insufficient_memory(
            "Qwen4Exp expert trellis plane allocation failed", __FILE__, __LINE__);
    }
    w.exl3_suh = alloc_u16(suh_bytes * local_experts, {local_experts, k}, {k, 1u}, arena);
    if (w.exl3_suh.ndim() == 0) {
        return Status::insufficient_memory(
            "Qwen4Exp expert suh plane allocation failed", __FILE__, __LINE__);
    }
    w.exl3_svh = alloc_u16(svh_bytes * local_experts, {local_experts, n}, {n, 1u}, arena);
    if (w.exl3_svh.ndim() == 0) {
        return Status::insufficient_memory(
            "Qwen4Exp expert svh plane allocation failed", __FILE__, __LINE__);
    }
    w.exl3_marker = alloc_u8(4u * local_experts, {local_experts, 4u}, {4u, 1u}, arena);
    if (w.exl3_marker.ndim() == 0) {
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
        Status st = copy_async(w.codes.data<std::uint8_t>() + local * trellis_bytes,
                               value.codes.data, static_cast<std::size_t>(trellis_bytes), stream,
                               "Qwen4Exp expert trellis upload");
        if (!st.ok()) return st;
        st = copy_async(w.exl3_suh.data<std::uint16_t>() + static_cast<std::size_t>(local) * k,
                        value.suh.data, static_cast<std::size_t>(suh_bytes), stream,
                        "Qwen4Exp expert suh upload");
        if (!st.ok()) return st;
        st = copy_async(w.exl3_svh.data<std::uint16_t>() + static_cast<std::size_t>(local) * n,
                        value.svh.data, static_cast<std::size_t>(svh_bytes), stream,
                        "Qwen4Exp expert svh upload");
        if (!st.ok()) return st;
        st = copy_async(w.exl3_marker.data<std::uint8_t>() +
                            static_cast<std::size_t>(local) * 4u,
                        value.marker.data, 4u, stream, "Qwen4Exp expert marker upload");
        if (!st.ok()) return st;
    }
    shelf::weights::apply_default_compute_spec(w);
    return w;
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
    out.experts.global_experts = global_experts;
    out.experts.local_experts = local_experts;
    out.experts.expert_begin = expert_begin;

    auto gate = load_expert_matrix(reader, prefix, "gate_proj", local_experts, expert_begin,
                                   hidden, intermediate, arena, stream);
    if (!gate.ok()) return gate.status();
    out.gate_experts = gate.release();
    auto up = load_expert_matrix(reader, prefix, "up_proj", local_experts, expert_begin,
                                 hidden, intermediate, arena, stream);
    if (!up.ok()) return up.status();
    out.up_experts = up.release();
    auto down = load_expert_matrix(reader, prefix, "down_proj", local_experts, expert_begin,
                                   intermediate, hidden, arena, stream);
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

Status load_mtp(const qqb::QuantizedModelReader& reader, const Qwen4ExpTextConfig& config,
                GpuArena& arena, hipStream_t stream, const WeightLoadOptions& options,
                Qwen4ExpMtpWeights& out) {
    Status st = load_matrix(reader, "mtp.fc_hidden.weight", arena, stream, options,
                            out.fc_hidden);
    if (!st.ok()) return st;
    st = load_matrix(reader, "mtp.fc_embedding.weight", arena, stream, options,
                     out.fc_embedding);
    if (!st.ok()) return st;
    st = load_small(reader, "mtp.pre_fc_norm_hidden.weight", arena, stream, options,
                    out.pre_fc_norm_hidden);
    if (!st.ok()) return st;
    st = load_small(reader, "mtp.pre_fc_norm_embedding.weight", arena, stream, options,
                    out.pre_fc_norm_embedding);
    if (!st.ok()) return st;
    st = load_mixer_prefix(reader, "mtp.hyper_connection_mixer.", arena, stream, options,
                           out.mixer);
    if (!st.ok()) return st;

    const std::string layer_prefix = "mtp.layers.0.";
    st = load_hyper(reader, layer_prefix + "attn_hyper_connection.", arena, stream, options,
                    out.layer.attn_hyper_connection);
    if (!st.ok()) return st;
    st = load_hyper(reader, layer_prefix + "mlp_hyper_connection.", arena, stream, options,
                    out.layer.mlp_hyper_connection);
    if (!st.ok()) return st;
    st = load_attention(reader, layer_prefix, false, arena, stream, options,
                        out.layer.attention);
    if (!st.ok()) return st;
    st = load_moe(reader, layer_prefix, config, 1u, 0u, arena, stream, options,
                  out.layer.mlp);
    if (!st.ok()) return st;
    out.present = true;
    return Status::make_ok();
}

}  // namespace

std::uint64_t Qwen4ExpRankWeights::resident_bytes() const noexcept {
    std::uint64_t total = 0;
    (void)add_matrix(total, embed_tokens);
    (void)add_matrix(total, lm_head);
    (void)add_hyper(total, mixer);
    if (mtp.present) {
        (void)add_matrix(total, mtp.fc_hidden);
        (void)add_matrix(total, mtp.fc_embedding);
        (void)add_tensor(total, mtp.pre_fc_norm_hidden);
        (void)add_tensor(total, mtp.pre_fc_norm_embedding);
        (void)add_hyper(total, mtp.mixer);
        (void)add_hyper(total, mtp.layer.attn_hyper_connection);
        (void)add_hyper(total, mtp.layer.mlp_hyper_connection);
        (void)add_matrix(total, mtp.layer.attention.q_proj);
        (void)add_matrix(total, mtp.layer.attention.k_proj);
        (void)add_matrix(total, mtp.layer.attention.v_proj);
        (void)add_matrix(total, mtp.layer.attention.o_proj);
        (void)add_tensor(total, mtp.layer.attention.q_norm);
        (void)add_tensor(total, mtp.layer.attention.k_norm);
        (void)add_matrix(total, mtp.layer.attention.index_qk_proj);
        (void)add_tensor(total, mtp.layer.attention.index_q_layernorm);
        (void)add_tensor(total, mtp.layer.attention.index_k_layernorm);
        (void)add_matrix(total, mtp.layer.mlp.router);
        (void)add_matrix(total, mtp.layer.mlp.shared_gate);
        (void)add_matrix(total, mtp.layer.mlp.shared_gate_proj);
        (void)add_matrix(total, mtp.layer.mlp.shared_up_proj);
        (void)add_matrix(total, mtp.layer.mlp.shared_down_proj);
        (void)add_matrix(total, mtp.layer.mlp.gate_experts);
        (void)add_matrix(total, mtp.layer.mlp.up_experts);
        (void)add_matrix(total, mtp.layer.mlp.down_experts);
    }
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
        (void)add_matrix(total, layer.mlp.gate_experts);
        (void)add_matrix(total, layer.mlp.up_experts);
        (void)add_matrix(total, layer.mlp.down_experts);
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
                                                     std::uint32_t tp_rank,
                                                     bool load_mtp, bool mtp_only) {
    std::vector<std::string> names = {
        "model.language_model.embed_tokens.weight",
        "model.language_model.lm_head.weight",
        "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
        "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
        "model.language_model.hyper_connection_mixer.hc_norm.weight",
    };
    if (mtp_only) {
        names.resize(2u);
        if (load_mtp) {
            const std::vector<std::string> mtp = mtp_names(config);
            names.insert(names.end(), mtp.begin(), mtp.end());
        }
        return names;
    }
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
    if (load_mtp) {
        const std::vector<std::string> mtp = mtp_names(config);
        names.insert(names.end(), mtp.begin(), mtp.end());
    }
    return names;
}

Result<Qwen4ExpRankWeights> load_qwen4exp_rank(const std::string& model_dir,
                                               GpuArena& arena,
                                               hipStream_t stream,
                                               const Qwen4ExpLoadOptions& options) {
    if (options.tp_size != 1u && options.tp_size != 2u && options.tp_size != 4u) {
        return Status::unsupported("Qwen4Exp load supports tp_size 1, 2 or 4", __FILE__,
                                   __LINE__);
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
        qwen4exp_rank_logical_names(config, options.tp_size, options.tp_rank,
                                    options.load_mtp, options.mtp_only);
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
    if (!options.mtp_only) {
        st = load_mixer(reader, arena, stream, weight_options, out.mixer);
        if (!st.ok()) return st;
    }

    if (!options.mtp_only) {
        for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
            const std::string prefix = layer_prefix(layer);
            Qwen4ExpLayerWeights& weights = out.layers[layer];
            st = load_hyper(reader, prefix + "attn_hyper_connection.", arena, stream,
                            weight_options, weights.attn_hyper_connection);
            if (!st.ok()) return st;
            st = load_hyper(reader, prefix + "mlp_hyper_connection.", arena, stream,
                            weight_options, weights.mlp_hyper_connection);
            if (!st.ok()) return st;
            st = load_attention(reader, prefix, qhy::layer_is_gdn(config, layer), arena,
                                stream, weight_options, weights.attention);
            if (!st.ok()) return st;
            st = load_moe(reader, prefix, config, options.tp_size, options.tp_rank, arena,
                          stream, weight_options, weights.mlp);
            if (!st.ok()) return st;
            if (config.layer_has_ple(layer)) {
                st = load_ple(reader, prefix, arena, stream, weight_options, weights.ple);
                if (!st.ok()) return st;
            }
            const hipError_t sync_error = hipStreamSynchronize(stream);
            if (sync_error != hipSuccess) {
                return Status::hip_error("Qwen4Exp layer upload",
                                         hipGetErrorString(sync_error), __FILE__,
                                         __LINE__);
            }
        }
    }

    if (options.load_mtp && options.tp_rank == 0u) {
        st = load_mtp(reader, config, arena, stream, weight_options, out.mtp);
        if (!st.ok()) return st;
        const hipError_t sync_error = hipStreamSynchronize(stream);
        if (sync_error != hipSuccess) {
            return Status::hip_error("Qwen4Exp MTP upload", hipGetErrorString(sync_error),
                                     __FILE__, __LINE__);
        }
    }

    if (options.mtp_only) {
        return out;
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
    if (weight.encoding == shelf::weights::MatrixEncoding::Fp8PerChannel) {
        if (weight.k_padded != cols || weight.codes.ndim() == 0 ||
            weight.scales.ndim() == 0 || weight.scales.dim(0) != rows) {
            return Status::invalid_argument(
                (std::string("Qwen4Exp per-channel FP8 weight is invalid: ") + what).c_str(),
                __FILE__, __LINE__);
        }
    } else if (weight.encoding == shelf::weights::MatrixEncoding::Mxfp4) {
        if (weight.k_padded != cols || weight.codes.ndim() == 0 ||
            weight.scales.ndim() == 0 || weight.row_ref.ndim() == 0) {
            return Status::invalid_argument(
                (std::string("Qwen4Exp MXFP4 weight is invalid: ") + what).c_str(),
                __FILE__, __LINE__);
        }
    } else if (weight.encoding == shelf::weights::MatrixEncoding::Exl3) {
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

Status expect_experts(const MatrixWeight& w, std::uint32_t local_experts,
                      std::uint32_t rows, std::uint32_t cols, const char* what) {
    if (w.experts != local_experts || w.rows != rows || w.cols != cols ||
        w.k_padded != cols) {
        return Status::invalid_argument(
            (std::string("Qwen4Exp expert plane geometry mismatch: ") + what).c_str(),
            __FILE__, __LINE__);
    }
    if (w.encoding == shelf::weights::MatrixEncoding::Mxfp4) {
        if (w.codes.ndim() != 3u || w.codes.dim(0) != local_experts ||
            w.scales.ndim() != 3u || w.scales.dim(0) != local_experts ||
            w.row_ref.ndim() != 2u || w.row_ref.dim(0) != local_experts ||
            w.row_ref.dim(1) != rows) {
            return Status::invalid_argument(
                (std::string("Qwen4Exp MXFP4 expert plane shape mismatch: ") + what).c_str(),
                __FILE__, __LINE__);
        }
        return Status::make_ok();
    }
    if (w.encoding != shelf::weights::MatrixEncoding::Exl3) {
        return Status::invalid_argument(
            (std::string("Qwen4Exp expert plane has an unexpected encoding: ") + what)
                .c_str(),
            __FILE__, __LINE__);
    }
    if (w.exl3_ka < 1u || w.exl3_ka > 8u || (w.exl3_half && w.exl3_ka > 3u)) {
        return Status::invalid_argument(
            (std::string("Qwen4Exp EXL3 expert rate is invalid: ") + what).c_str(),
            __FILE__, __LINE__);
    }
    const std::uint32_t words = 16u * w.exl3_ka + (w.exl3_half ? 8u : 0u);
    const std::uint64_t expected_trellis =
        static_cast<std::uint64_t>(cols / 16u) * (rows / 16u) * words * 2u;
    if (w.codes_batch_stride_bytes != expected_trellis ||
        w.exl3_suh_batch_stride_bytes != static_cast<std::uint64_t>(cols) * 2u ||
        w.exl3_svh_batch_stride_bytes != static_cast<std::uint64_t>(rows) * 2u) {
        return Status::invalid_argument(
            (std::string("Qwen4Exp expert plane stride mismatch: ") + what).c_str(),
            __FILE__, __LINE__);
    }
    if (w.codes.ndim() != 2u || w.codes.dim(0) != local_experts ||
        w.codes.dim(1) != expected_trellis ||
        w.exl3_suh.ndim() != 2u || w.exl3_suh.dim(0) != local_experts ||
        w.exl3_suh.dim(1) != cols ||
        w.exl3_svh.ndim() != 2u || w.exl3_svh.dim(0) != local_experts ||
        w.exl3_svh.dim(1) != rows ||
        w.exl3_marker.ndim() != 2u || w.exl3_marker.dim(0) != local_experts ||
        w.exl3_marker.dim(1) != 4u) {
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
    if (tp_size == 0u || tp_size > 4u || (512u % tp_size) != 0u) {
        return Status::unsupported("Qwen4Exp validation supports tp_size 1, 2 or 4", __FILE__,
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
    const std::uint32_t local_q_heads = q_heads / tp_size;
    const std::uint32_t local_kv_heads = kv_heads / tp_size;
    const std::uint32_t local_q_features = local_q_heads * head_dim;
    const std::uint32_t local_kv_features = local_kv_heads * head_dim;

    const std::uint32_t gdn_key_heads =
        static_cast<std::uint32_t>(config.linear_num_key_heads);
    const std::uint32_t gdn_value_heads =
        static_cast<std::uint32_t>(config.linear_num_value_heads);
    const std::uint32_t local_gdn_key_heads = gdn_key_heads / tp_size;
    const std::uint32_t local_gdn_value_heads = gdn_value_heads / tp_size;
    const std::uint32_t gdn_key_features =
        local_gdn_key_heads * static_cast<std::uint32_t>(config.linear_key_head_dim);
    const std::uint32_t gdn_value_features =
        local_gdn_value_heads * static_cast<std::uint32_t>(config.linear_value_head_dim);
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
    if (weights.lm_head.encoding != shelf::weights::MatrixEncoding::Exl3 &&
        weights.lm_head.encoding != shelf::weights::MatrixEncoding::Bf16) {
        return Status::invalid_argument("Qwen4Exp lm_head is not EXL3 or BF16", __FILE__,
                                        __LINE__);
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
            st = expect_matrix(w.attention.in_proj_a, local_gdn_value_heads, hidden,
                               "in_proj_a");
            if (!st.ok()) return st;
            st = expect_matrix(w.attention.in_proj_b, local_gdn_value_heads, hidden,
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
            st = expect_tensor_1d(w.attention.a_log, local_gdn_value_heads,
                                  "linear_attn.A_log");
            if (!st.ok()) return st;
            st = expect_tensor_1d(w.attention.dt_bias, local_gdn_value_heads,
                                  "linear_attn.dt_bias");
            if (!st.ok()) return st;
        } else {
            st = expect_matrix(w.attention.q_proj, 2u * local_q_features, hidden,
                               "q_proj");
            if (!st.ok()) return st;
            st = expect_matrix(w.attention.k_proj, local_kv_features, hidden,
                               "k_proj");
            if (!st.ok()) return st;
            st = expect_matrix(w.attention.v_proj, local_kv_features, hidden,
                               "v_proj");
            if (!st.ok()) return st;
            st = expect_matrix(w.attention.o_proj, hidden, local_q_features,
                               "o_proj");
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

        if (w.mlp.experts.global_experts != global_experts ||
            w.mlp.experts.local_experts != local_experts ||
            w.mlp.experts.expert_begin != expert_begin) {
            return Status::invalid_argument("Qwen4Exp expert range mismatch", __FILE__,
                                            __LINE__);
        }
        st = expect_experts(w.mlp.gate_experts, local_experts, intermediate, hidden,
                            "gate_experts");
        if (!st.ok()) return st;
        st = expect_experts(w.mlp.up_experts, local_experts, intermediate, hidden,
                            "up_experts");
        if (!st.ok()) return st;
        st = expect_experts(w.mlp.down_experts, local_experts, hidden, intermediate,
                            "down_experts");
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
