#include <shelf/resident/resident_format.h>

#include <shelf/core/memory/arena.h>
#include <shelf/models/qwen35/model/qwen35_model.h>

#include <cstring>

namespace ps {
namespace resident {
namespace {

using qwen35::Qwen35LayerWeights;
using qwen35::Qwen35MtpWeights;
using qwen35::Qwen35ModelWeights;
using qwen35::Qwen35TextConfig;

constexpr std::uint64_t kArchiveSentinel = 0x52445350u;

#define PS_RSD_LAYER_TENSORS(F)            \
    F(input_layernorm_weight)              \
    F(post_attention_layernorm_weight)     \
    F(attn_q_norm_weight)                  \
    F(attn_k_norm_weight)                  \
    F(attn_conv1d_weight)                  \
    F(attn_norm_weight)                    \
    F(attn_dt_bias)                        \
    F(attn_A_log)

#define PS_RSD_LAYER_MATRICES(F)           \
    F(mlp_gate_proj)                       \
    F(mlp_up_proj)                         \
    F(mlp_down_proj)                       \
    F(attn_q_proj)                         \
    F(attn_k_proj)                         \
    F(attn_v_proj)                         \
    F(attn_o_proj)                         \
    F(attn_in_proj_a)                      \
    F(attn_in_proj_b)                      \
    F(attn_in_proj_qkv)                    \
    F(attn_in_proj_z)                      \
    F(attn_out_proj)

#define PS_RSD_DFLASH_LAYER_TENSORS(F)     \
    F(input_layernorm_weight)              \
    F(post_attention_layernorm_weight)     \
    F(attn_q_norm_weight)                  \
    F(attn_k_norm_weight)                  \
    F(attention_conv_base_kernel)          \
    F(mlp_conv_base_kernel)

#define PS_RSD_DFLASH_LAYER_MATRICES(F)    \
    F(attn_q_proj)                         \
    F(attn_k_proj)                         \
    F(attn_v_proj)                         \
    F(attn_o_proj)                         \
    F(mlp_gate_proj)                       \
    F(mlp_up_proj)                         \
    F(mlp_down_proj)                       \
    F(attention_conv_kernel_projection)    \
    F(mlp_conv_kernel_projection)

#define PS_RSD_MTP_TENSORS(F)              \
    F(pre_fc_norm_embedding_weight)        \
    F(pre_fc_norm_hidden_weight)           \
    F(final_norm_weight)

void write_qwen_layer(ArchiveWriter& w, const Qwen35LayerWeights& layer) {
    w.u8(layer.is_gdn ? 1u : 0u);
#define F(name) w.tensor(layer.name);
    PS_RSD_LAYER_TENSORS(F)
#undef F
#define F(name) w.matrix(layer.name);
    PS_RSD_LAYER_MATRICES(F)
#undef F
}

Qwen35LayerWeights read_qwen_layer(ArchiveReader& r) {
    Qwen35LayerWeights layer;
    layer.is_gdn = r.u8() != 0u;
#define F(name) layer.name = r.tensor();
    PS_RSD_LAYER_TENSORS(F)
#undef F
#define F(name) layer.name = r.matrix();
    PS_RSD_LAYER_MATRICES(F)
#undef F
    return layer;
}

void write_dflash_layer(ArchiveWriter& w,
                        const qwen35::dflash2::DFlash2LayerWeights& layer) {
#define F(name) w.tensor(layer.name);
    PS_RSD_DFLASH_LAYER_TENSORS(F)
#undef F
#define F(name) w.matrix(layer.name);
    PS_RSD_DFLASH_LAYER_MATRICES(F)
#undef F
}

qwen35::dflash2::DFlash2LayerWeights read_dflash_layer(ArchiveReader& r) {
    qwen35::dflash2::DFlash2LayerWeights layer;
#define F(name) layer.name = r.tensor();
    PS_RSD_DFLASH_LAYER_TENSORS(F)
#undef F
#define F(name) layer.name = r.matrix();
    PS_RSD_DFLASH_LAYER_MATRICES(F)
#undef F
    return layer;
}

void write_mtp(ArchiveWriter& w, const Qwen35MtpWeights& mtp) {
    w.u8(mtp.present ? 1u : 0u);
#define F(name) w.tensor(mtp.name);
    PS_RSD_MTP_TENSORS(F)
#undef F
    w.matrix(mtp.fc);
    write_qwen_layer(w, mtp.layer);
}

Qwen35MtpWeights read_mtp(ArchiveReader& r) {
    Qwen35MtpWeights mtp;
    mtp.present = r.u8() != 0u;
#define F(name) mtp.name = r.tensor();
    PS_RSD_MTP_TENSORS(F)
#undef F
    mtp.fc = r.matrix();
    mtp.layer = read_qwen_layer(r);
    return mtp;
}

void write_text_config(ArchiveWriter& w, const Qwen35TextConfig& config) {
    w.u64(config.vocab_size);
    w.u64(config.hidden_size);
    w.u64(config.intermediate_size);
    w.u64(config.num_hidden_layers);
    w.u64(config.full_attention_interval);
    w.u64(config.linear_num_key_heads);
    w.u64(config.linear_num_value_heads);
    w.u64(config.linear_key_head_dim);
    w.u64(config.linear_value_head_dim);
    w.u64(config.linear_conv_kernel_dim);
    w.u64(config.num_attention_heads);
    w.u64(config.num_key_value_heads);
    w.u64(config.attention_head_dim);
    w.i32_vec(config.stop_tokens);
    w.f32(config.rms_norm_eps);
    w.f32(config.rope_theta);
    w.f32(config.partial_rotary_factor);
    w.u8(config.tie_word_embeddings ? 1u : 0u);
    w.int_vec(config.layer_types);
    w.u8(config.output_gate_silu ? 1u : 0u);
}

Qwen35TextConfig read_text_config(ArchiveReader& r) {
    Qwen35TextConfig config;
    config.vocab_size = r.u64();
    config.hidden_size = r.u64();
    config.intermediate_size = r.u64();
    config.num_hidden_layers = r.u64();
    config.full_attention_interval = r.u64();
    config.linear_num_key_heads = r.u64();
    config.linear_num_value_heads = r.u64();
    config.linear_key_head_dim = r.u64();
    config.linear_value_head_dim = r.u64();
    config.linear_conv_kernel_dim = r.u64();
    config.num_attention_heads = r.u64();
    config.num_key_value_heads = r.u64();
    config.attention_head_dim = r.u64();
    config.stop_tokens = r.i32_vec();
    config.rms_norm_eps = r.f32();
    config.rope_theta = r.f32();
    config.partial_rotary_factor = r.f32();
    config.tie_word_embeddings = r.u8() != 0u;
    config.layer_types = r.int_vec();
    config.output_gate_silu = r.u8() != 0u;
    return config;
}

void write_dflash_config(ArchiveWriter& w,
                         const qwen35::dflash2::DFlash2Config& config) {
    w.str(config.architecture);
    w.u64(config.vocab_size);
    w.u64(config.hidden_size);
    w.u64(config.intermediate_size);
    w.u64(config.num_hidden_layers);
    w.u64(config.num_attention_heads);
    w.u64(config.num_key_value_heads);
    w.u64(config.head_dim);
    w.u64(config.sliding_window);
    w.u8(config.is_causal ? 1u : 0u);
    w.u64(config.block_size);
    w.u64(config.conv_group_size);
    w.u64(config.conv_kernel_size);
    w.u64(config.mask_token_id);
    w.u64(config.selector_rank);
    w.u64(config.selector_top_k);
    w.u64(config.num_target_layers);
    for (std::size_t i = 0; i < qwen35::dflash2::kMaxTargetLayerIds; ++i) {
        w.u64(config.target_layer_ids[i]);
    }
    w.u64(config.num_target_layer_ids);
    w.f32(config.rms_norm_eps);
    w.f32(config.rope_theta);
    w.u8(config.tie_word_embeddings ? 1u : 0u);
}

qwen35::dflash2::DFlash2Config read_dflash_config(ArchiveReader& r) {
    qwen35::dflash2::DFlash2Config config;
    config.architecture = r.str();
    config.vocab_size = r.u64();
    config.hidden_size = r.u64();
    config.intermediate_size = r.u64();
    config.num_hidden_layers = r.u64();
    config.num_attention_heads = r.u64();
    config.num_key_value_heads = r.u64();
    config.head_dim = r.u64();
    config.sliding_window = r.u64();
    config.is_causal = r.u8() != 0u;
    config.block_size = r.u64();
    config.conv_group_size = r.u64();
    config.conv_kernel_size = r.u64();
    config.mask_token_id = r.u64();
    config.selector_rank = r.u64();
    config.selector_top_k = r.u64();
    config.num_target_layers = r.u64();
    for (std::size_t i = 0; i < qwen35::dflash2::kMaxTargetLayerIds; ++i) {
        config.target_layer_ids[i] = r.u64();
    }
    config.num_target_layer_ids = r.u64();
    config.rms_norm_eps = r.f32();
    config.rope_theta = r.f32();
    config.tie_word_embeddings = r.u8() != 0u;
    return config;
}

void write_model_weights(ArchiveWriter& w, const Qwen35ModelWeights& weights) {
    w.matrix(weights.embed_tokens);
    w.tensor(weights.final_norm_weight);
    w.matrix(weights.lm_head);
    w.u8(weights.lm_head_tied ? 1u : 0u);
    w.u64(static_cast<std::uint64_t>(weights.layers.size()));
    for (const auto& layer : weights.layers) {
        write_qwen_layer(w, layer);
    }
    write_mtp(w, weights.mtp);
}

Qwen35ModelWeights read_model_weights(ArchiveReader& r) {
    Qwen35ModelWeights weights;
    weights.embed_tokens = r.matrix();
    weights.final_norm_weight = r.tensor();
    weights.lm_head = r.matrix();
    weights.lm_head_tied = r.u8() != 0u;
    const std::uint64_t layer_count = r.u64();
    if (!r.ok()) {
        return weights;
    }
    if (layer_count > 4096u) {
        r.fail("resident archive layer count");
        return weights;
    }
    weights.layers.reserve(static_cast<std::size_t>(layer_count));
    for (std::uint64_t i = 0; i < layer_count; ++i) {
        weights.layers.push_back(read_qwen_layer(r));
    }
    weights.mtp = read_mtp(r);
    return weights;
}

void write_dflash_weights(ArchiveWriter& w,
                          const qwen35::dflash2::DFlash2Weights& weights) {
    w.u8(weights.present ? 1u : 0u);
    w.u8(0u);
    w.tensor(weights.hidden_norm_weight);
    w.tensor(weights.final_norm_weight);
    w.matrix(weights.fc);
    w.matrix(weights.selector.hidden_projection);
    w.tensor(weights.selector.predecessor_codebook);
    w.tensor(weights.selector.successor_codebook);
    w.u64(static_cast<std::uint64_t>(weights.layers.size()));
    for (const auto& layer : weights.layers) {
        write_dflash_layer(w, layer);
    }
}

qwen35::dflash2::DFlash2Weights read_dflash_weights(ArchiveReader& r) {
    qwen35::dflash2::DFlash2Weights weights;
    weights.present = r.u8() != 0u;
    const std::uint8_t backbone_flag = r.u8();
    if (backbone_flag != 0u) {
        r.fail("resident archive dflash backbone encoding");
    }
    weights.hidden_norm_weight = r.tensor();
    weights.final_norm_weight = r.tensor();
    weights.fc = r.matrix();
    weights.selector.hidden_projection = r.matrix();
    weights.selector.predecessor_codebook = r.tensor();
    weights.selector.successor_codebook = r.tensor();
    const std::uint64_t layer_count = r.u64();
    if (!r.ok()) {
        return weights;
    }
    if (layer_count > 256u) {
        r.fail("resident archive dflash layer count");
        return weights;
    }
    weights.layers.reserve(static_cast<std::size_t>(layer_count));
    for (std::uint64_t i = 0; i < layer_count; ++i) {
        weights.layers.push_back(read_dflash_layer(r));
    }
    return weights;
}

#define PS_RSD_EAGLE4_LAYER_TENSORS(F)     \
    F(input_layernorm_weight)              \
    F(hidden_norm_weight)                  \
    F(post_attention_layernorm_weight)

#define PS_RSD_EAGLE4_LAYER_MATRICES(F)    \
    F(attn_q_proj)                         \
    F(attn_k_proj)                         \
    F(attn_v_proj)                         \
    F(attn_o_proj)                         \
    F(mlp_gate_proj)                       \
    F(mlp_up_proj)                         \
    F(mlp_down_proj)

void write_eagle4_config(ArchiveWriter& w,
                         const qwen35::eagle4::Eagle4Config& config) {
    w.str(config.architecture);
    w.u64(config.vocab_size);
    w.u64(config.hidden_size);
    w.u64(config.intermediate_size);
    w.u64(config.num_hidden_layers);
    w.u64(config.num_attention_heads);
    w.u64(config.num_key_value_heads);
    w.u64(config.head_dim);
    w.u64(config.sliding_window);
    w.u64(config.target_hidden_size);
    w.u64(config.num_aux_layers);
    for (std::size_t i = 0; i < qwen35::eagle4::kEagle4MaxAuxLayers; ++i) {
        w.u64(config.aux_layer_ids[i]);
    }
    w.u64(config.candidate_rank);
    w.u64(config.num_candidates);
    w.u8(config.fc_norm ? 1u : 0u);
    w.u8(config.norm_output ? 1u : 0u);
    w.f32(config.rms_norm_eps);
    w.f32(config.rope_theta);
    w.u8(config.tie_word_embeddings ? 1u : 0u);
}

qwen35::eagle4::Eagle4Config read_eagle4_config_archive(ArchiveReader& r) {
    qwen35::eagle4::Eagle4Config config;
    config.architecture = r.str();
    config.vocab_size = r.u64();
    config.hidden_size = r.u64();
    config.intermediate_size = r.u64();
    config.num_hidden_layers = r.u64();
    config.num_attention_heads = r.u64();
    config.num_key_value_heads = r.u64();
    config.head_dim = r.u64();
    config.sliding_window = r.u64();
    config.target_hidden_size = r.u64();
    config.num_aux_layers = r.u64();
    for (std::size_t i = 0; i < qwen35::eagle4::kEagle4MaxAuxLayers; ++i) {
        config.aux_layer_ids[i] = r.u64();
    }
    config.candidate_rank = r.u64();
    config.num_candidates = r.u64();
    config.fc_norm = r.u8() != 0u;
    config.norm_output = r.u8() != 0u;
    config.rms_norm_eps = r.f32();
    config.rope_theta = r.f32();
    config.tie_word_embeddings = r.u8() != 0u;
    return config;
}

void write_eagle4_layer(ArchiveWriter& w,
                        const qwen35::eagle4::Eagle4LayerWeights& layer) {
#define F(name) w.tensor(layer.name);
    PS_RSD_EAGLE4_LAYER_TENSORS(F)
#undef F
#define F(name) w.matrix(layer.name);
    PS_RSD_EAGLE4_LAYER_MATRICES(F)
#undef F
}

qwen35::eagle4::Eagle4LayerWeights read_eagle4_layer(ArchiveReader& r) {
    qwen35::eagle4::Eagle4LayerWeights layer;
#define F(name) layer.name = r.tensor();
    PS_RSD_EAGLE4_LAYER_TENSORS(F)
#undef F
#define F(name) layer.name = r.matrix();
    PS_RSD_EAGLE4_LAYER_MATRICES(F)
#undef F
    return layer;
}

void write_eagle4_weights(ArchiveWriter& w,
                          const qwen35::eagle4::Eagle4Weights& weights) {
    w.u8(weights.present ? 1u : 0u);
    w.matrix(weights.fc);
    for (std::size_t i = 0; i < qwen35::eagle4::kEagle4AuxLayers; ++i) {
        w.tensor(weights.fc_norm_weight[i]);
    }
    w.tensor(weights.final_norm_weight);
    w.matrix(weights.embed_tokens);
    w.matrix(weights.candidate.lm_head);
    w.matrix(weights.candidate.efficient_lm_head);
    w.matrix(weights.candidate.efficient_lm_head_down_proj);
    w.u64(static_cast<std::uint64_t>(weights.layers.size()));
    for (const auto& layer : weights.layers) {
        write_eagle4_layer(w, layer);
    }
}

qwen35::eagle4::Eagle4Weights read_eagle4_weights(ArchiveReader& r) {
    qwen35::eagle4::Eagle4Weights weights;
    weights.present = r.u8() != 0u;
    weights.fc = r.matrix();
    for (std::size_t i = 0; i < qwen35::eagle4::kEagle4AuxLayers; ++i) {
        weights.fc_norm_weight[i] = r.tensor();
    }
    weights.final_norm_weight = r.tensor();
    weights.embed_tokens = r.matrix();
    weights.candidate.lm_head = r.matrix();
    weights.candidate.efficient_lm_head = r.matrix();
    weights.candidate.efficient_lm_head_down_proj = r.matrix();
    const std::uint64_t layer_count = r.u64();
    if (!r.ok()) {
        return weights;
    }
    if (layer_count > 256u) {
        r.fail("resident archive eagle4 layer count");
        return weights;
    }
    weights.layers.reserve(static_cast<std::size_t>(layer_count));
    for (std::uint64_t i = 0; i < layer_count; ++i) {
        weights.layers.push_back(read_eagle4_layer(r));
    }
    return weights;
}

}

ArchiveWriter::ArchiveWriter(const void* base) : base_(base) {
    u64(kArchiveSentinel);
    u64(kManifestVersion);
}

void ArchiveWriter::raw(const void* data, std::size_t bytes) {
    const auto* p = static_cast<const std::byte*>(data);
    buf_.insert(buf_.end(), p, p + bytes);
}

void ArchiveWriter::u8(std::uint8_t value) { raw(&value, sizeof(value)); }
void ArchiveWriter::u32(std::uint32_t value) { raw(&value, sizeof(value)); }
void ArchiveWriter::i32(std::int32_t value) { raw(&value, sizeof(value)); }
void ArchiveWriter::u64(std::uint64_t value) { raw(&value, sizeof(value)); }
void ArchiveWriter::i64(std::int64_t value) { raw(&value, sizeof(value)); }
void ArchiveWriter::f32(float value) { raw(&value, sizeof(value)); }

void ArchiveWriter::str(const std::string& value) {
    u64(static_cast<std::uint64_t>(value.size()));
    raw(value.data(), value.size());
}

void ArchiveWriter::i32_vec(const std::vector<std::int32_t>& values) {
    u64(static_cast<std::uint64_t>(values.size()));
    for (std::int32_t v : values) i32(v);
}

void ArchiveWriter::int_vec(const std::vector<int>& values) {
    u64(static_cast<std::uint64_t>(values.size()));
    for (int v : values) i32(static_cast<std::int32_t>(v));
}

void ArchiveWriter::i64_vec(const std::vector<std::int64_t>& values) {
    u64(static_cast<std::uint64_t>(values.size()));
    for (std::int64_t v : values) i64(v);
}

void ArchiveWriter::tensor(const gpu::Tensor& tensor) {
    TensorRecord record{};
    if (tensor.ndim() == 0 || tensor.allocation_base() == nullptr) {
        record.present = 0;
        raw(&record, sizeof(record));
        return;
    }
    const auto* base = static_cast<const std::byte*>(base_);
    const auto* addr = static_cast<const std::byte*>(tensor.allocation_base());
    if (addr < base) {
        record.present = 0;
        raw(&record, sizeof(record));
        return;
    }
    record.present = 1;
    record.alloc_offset =
        static_cast<std::uint64_t>(addr - base) + tensor.view_offset_bytes();
    record.alloc_bytes =
        tensor.allocation_bytes() > tensor.view_offset_bytes()
            ? tensor.allocation_bytes() - tensor.view_offset_bytes()
            : 0;
    record.view_offset = 0;
    record.ndim = static_cast<std::uint32_t>(tensor.ndim());
    record.elem_size = static_cast<std::uint32_t>(tensor.element_size());
    record.device = tensor.physical_device();
    for (std::size_t i = 0; i < kMaxTensorDimsWire; ++i) {
        record.shape[i] = i < tensor.ndim() ? tensor.shape_data()[i] : 0;
        record.strides[i] = i < tensor.ndim() ? tensor.strides_data()[i] : 0;
    }
    raw(&record, sizeof(record));
}

void ArchiveWriter::partition(
    const std::optional<weights::TensorPartitionDesc>& partition) {
    if (!partition.has_value()) {
        u8(0);
        return;
    }
    u8(1);
    i32(partition->axis);
    u32(partition->index);
    u32(partition->count);
    i64_vec(partition->global_shape);
    u64(static_cast<std::uint64_t>(partition->ranges.size()));
    for (const auto& range : partition->ranges) {
        u64(range.global_offset);
        u64(range.extent);
    }
}

void ArchiveWriter::matrix(const weights::MatrixWeight& weight) {
    u32(static_cast<std::uint32_t>(weight.encoding));
    u32(static_cast<std::uint32_t>(weight.quant_spec));
    u32(static_cast<std::uint32_t>(weight.compute_spec));
    u32(weight.rows);
    u32(weight.cols);
    u32(weight.k_padded);
    u32(weight.storage_scale_stride_bytes);
    u32(weight.codes_row_stride_bytes);
    u32(weight.scale_row_stride_bytes);
    u32(weight.weight_scale_group);
    u8(weight.preshuffled ? 1u : 0u);
    tensor(weight.bf16);
    tensor(weight.data);
    tensor(weight.codes);
    tensor(weight.scales);
    tensor(weight.compute_codes);
    tensor(weight.compute_scales_bf16);
    partition(weight.partition);
}

ArchiveReader::ArchiveReader(const std::byte* data, std::size_t size, void* block,
                             std::size_t block_bytes, int device)
    : cur_(data),
      left_(size),
      block_(block),
      block_bytes_(block_bytes),
      device_(device),
      status_(Status::make_ok()) {}

std::size_t ArchiveReader::remaining() const noexcept { return left_; }

void ArchiveReader::fail(const std::string& message) {
    if (status_.ok()) {
        status_ = Status::invalid_state(message.c_str(), __FILE__, __LINE__);
    }
}

void ArchiveReader::raw(void* out, std::size_t bytes) {
    if (!status_.ok()) {
        return;
    }
    if (left_ < bytes) {
        status_ = Status::invalid_state("resident archive truncated", __FILE__, __LINE__);
        return;
    }
    std::memcpy(out, cur_, bytes);
    cur_ += bytes;
    left_ -= bytes;
    offset_ += bytes;
}

std::uint8_t ArchiveReader::u8() {
    std::uint8_t v = 0;
    raw(&v, sizeof(v));
    return v;
}

std::uint32_t ArchiveReader::u32() {
    std::uint32_t v = 0;
    raw(&v, sizeof(v));
    return v;
}

std::int32_t ArchiveReader::i32() {
    std::int32_t v = 0;
    raw(&v, sizeof(v));
    return v;
}

std::uint64_t ArchiveReader::u64() {
    std::uint64_t v = 0;
    raw(&v, sizeof(v));
    return v;
}

std::int64_t ArchiveReader::i64() {
    std::int64_t v = 0;
    raw(&v, sizeof(v));
    return v;
}

float ArchiveReader::f32() {
    float v = 0.0f;
    raw(&v, sizeof(v));
    return v;
}

std::string ArchiveReader::str() {
    const std::uint64_t n = u64();
    if (!status_.ok()) return {};
    if (n > (1ull << 32) || left_ < n) {
        status_ = Status::invalid_state("resident archive string", __FILE__, __LINE__);
        return {};
    }
    std::string out(reinterpret_cast<const char*>(cur_), static_cast<std::size_t>(n));
    cur_ += n;
    left_ -= n;
    return out;
}

std::vector<std::int32_t> ArchiveReader::i32_vec() {
    const std::uint64_t n = u64();
    if (!status_.ok()) return {};
    if (n > (1ull << 24)) {
        status_ = Status::invalid_state("resident archive vector", __FILE__, __LINE__);
        return {};
    }
    std::vector<std::int32_t> out;
    out.reserve(static_cast<std::size_t>(n));
    for (std::uint64_t i = 0; i < n && status_.ok(); ++i) out.push_back(i32());
    return out;
}

std::vector<int> ArchiveReader::int_vec() {
    const std::uint64_t n = u64();
    if (!status_.ok()) return {};
    if (n > (1ull << 24)) {
        status_ = Status::invalid_state("resident archive vector", __FILE__, __LINE__);
        return {};
    }
    std::vector<int> out;
    out.reserve(static_cast<std::size_t>(n));
    for (std::uint64_t i = 0; i < n && status_.ok(); ++i) {
        out.push_back(static_cast<int>(i32()));
    }
    return out;
}

std::vector<std::int64_t> ArchiveReader::i64_vec() {
    const std::uint64_t n = u64();
    if (!status_.ok()) return {};
    if (n > (1ull << 24)) {
        status_ = Status::invalid_state("resident archive vector", __FILE__, __LINE__);
        return {};
    }
    std::vector<std::int64_t> out;
    out.reserve(static_cast<std::size_t>(n));
    for (std::uint64_t i = 0; i < n && status_.ok(); ++i) out.push_back(i64());
    return out;
}

gpu::Tensor ArchiveReader::tensor() {
    TensorRecord record{};
    raw(&record, sizeof(record));
    if (!status_.ok()) return {};
    if (record.present == 0) return {};
    if (record.ndim == 0 || record.ndim > kMaxTensorDimsWire || record.elem_size == 0) {
        const std::string message =
            "resident archive tensor shape ndim=" + std::to_string(record.ndim) +
            " elem_size=" + std::to_string(record.elem_size) +
            " offset=" + std::to_string(offset_);
        fail(message);
        return {};
    }
    if (record.view_offset > record.alloc_bytes ||
        record.alloc_offset > block_bytes_ ||
        record.alloc_bytes > block_bytes_ - record.alloc_offset) {
        status_ = Status::invalid_state("resident archive tensor bounds", __FILE__, __LINE__);
        return {};
    }
    auto* addr = static_cast<std::byte*>(block_) + record.alloc_offset + record.view_offset;
    const std::size_t backing_bytes =
        static_cast<std::size_t>(record.alloc_bytes - record.view_offset);
    const auto view = gpu::GpuArena::make_view(addr, backing_bytes, device_);
    std::vector<std::size_t> shape(record.ndim);
    std::vector<std::size_t> strides(record.ndim);
    for (std::size_t i = 0; i < record.ndim; ++i) {
        shape[i] = static_cast<std::size_t>(record.shape[i]);
        strides[i] = static_cast<std::size_t>(record.strides[i]);
    }
    Result<gpu::Tensor> created = [&]() -> Result<gpu::Tensor> {
        switch (record.elem_size) {
            case 1:
                return gpu::Tensor::create<std::uint8_t>(view, shape, strides);
            case 2:
                return gpu::Tensor::create<std::uint16_t>(view, shape, strides);
            case 4:
                return gpu::Tensor::create<std::uint32_t>(view, shape, strides);
            case 8:
                return gpu::Tensor::create<std::uint64_t>(view, shape, strides);
            default:
                return Status::invalid_state("resident archive element size", __FILE__, __LINE__);
        }
    }();
    if (!created.ok()) {
        status_ = created.status();
        return {};
    }
    return created.release();
}

std::optional<weights::TensorPartitionDesc> ArchiveReader::partition() {
    if (u8() == 0) return std::nullopt;
    if (!status_.ok()) return std::nullopt;
    weights::TensorPartitionDesc desc;
    desc.axis = i32();
    desc.index = u32();
    desc.count = u32();
    desc.global_shape = i64_vec();
    const std::uint64_t ranges = u64();
    if (!status_.ok()) return std::nullopt;
    if (ranges > 4096u) {
        status_ = Status::invalid_state("resident archive partition", __FILE__, __LINE__);
        return std::nullopt;
    }
    desc.ranges.reserve(static_cast<std::size_t>(ranges));
    for (std::uint64_t i = 0; i < ranges && status_.ok(); ++i) {
        weights::TensorPartitionRange range;
        range.global_offset = u64();
        range.extent = u64();
        desc.ranges.push_back(range);
    }
    return desc;
}

weights::MatrixWeight ArchiveReader::matrix() {
    weights::MatrixWeight weight;
    const std::uint32_t encoding = u32();
    const std::uint32_t quant_spec = u32();
    const std::uint32_t compute_spec = u32();
    weight.encoding = static_cast<weights::MatrixEncoding>(
        static_cast<std::uint8_t>(encoding));
    weight.quant_spec = static_cast<quantization::QuantSpecId>(
        static_cast<std::uint8_t>(quant_spec));
    weight.compute_spec = static_cast<quantization::ComputeSpecId>(
        static_cast<std::uint8_t>(compute_spec));
    weight.rows = u32();
    weight.cols = u32();
    weight.k_padded = u32();
    weight.storage_scale_stride_bytes = u32();
    weight.codes_row_stride_bytes = u32();
    weight.scale_row_stride_bytes = u32();
    weight.weight_scale_group = u32();
    weight.preshuffled = u8() != 0u;
    weight.bf16 = tensor();
    weight.data = tensor();
    weight.codes = tensor();
    weight.scales = tensor();
    weight.compute_codes = tensor();
    weight.compute_scales_bf16 = tensor();
    weight.partition = partition();
    return weight;
}

Status write_qwen35_archive(
    const Qwen35TextConfig& text_config,
    const Qwen35ModelWeights& weights,
    const void* base,
    std::vector<std::byte>& out) {
    if (base == nullptr) {
        return Status::invalid_argument("resident archive base is null", __FILE__, __LINE__);
    }
    ArchiveWriter writer(base);
    write_text_config(writer, text_config);
    write_model_weights(writer, weights);
    out = writer.release();
    return Status::make_ok();
}

Result<qwen35::Qwen35Model> build_qwen35_model(
    const std::byte* data,
    std::size_t size,
    void* block,
    std::size_t block_bytes,
    int device) {
    if (data == nullptr || block == nullptr) {
        return Status::invalid_argument("resident archive block is null", __FILE__, __LINE__);
    }
    ArchiveReader reader(data, size, block, block_bytes, device);
    if (reader.u64() != kArchiveSentinel || reader.u64() != kManifestVersion) {
        return Status::invalid_state("resident archive header", __FILE__, __LINE__);
    }
    Qwen35TextConfig text_config = read_text_config(reader);
    Qwen35ModelWeights weights = read_model_weights(reader);
    if (!reader.ok()) {
        return reader.status();
    }
    if (reader.remaining() != 0) {
        return Status::invalid_state("resident archive trailing bytes", __FILE__, __LINE__);
    }
    return qwen35::Qwen35Model::adopt(std::move(weights), std::move(text_config));
}

Status write_dflash2_archive(
    const qwen35::dflash2::DFlash2Config& config,
    const qwen35::dflash2::DFlash2Weights& weights,
    const void* base,
    std::vector<std::byte>& out) {
    if (base == nullptr) {
        return Status::invalid_argument("resident archive base is null", __FILE__, __LINE__);
    }
    ArchiveWriter writer(base);
    write_dflash_config(writer, config);
    write_dflash_weights(writer, weights);
    out = writer.release();
    return Status::make_ok();
}

Status read_dflash2_archive(
    const std::byte* data,
    std::size_t size,
    void* block,
    std::size_t block_bytes,
    int device,
    qwen35::dflash2::DFlash2Config& config_out,
    qwen35::dflash2::DFlash2Weights& weights_out) {
    if (data == nullptr || block == nullptr) {
        return Status::invalid_argument("resident archive block is null", __FILE__, __LINE__);
    }
    ArchiveReader reader(data, size, block, block_bytes, device);
    if (reader.u64() != kArchiveSentinel || reader.u64() != kManifestVersion) {
        return Status::invalid_state("resident archive header", __FILE__, __LINE__);
    }
    config_out = read_dflash_config(reader);
    weights_out = read_dflash_weights(reader);
    if (!reader.ok()) {
        return reader.status();
    }
    if (reader.remaining() != 0) {
        return Status::invalid_state("resident archive trailing bytes", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status write_eagle4_archive(
    const qwen35::eagle4::Eagle4Config& config,
    const qwen35::eagle4::Eagle4Weights& weights,
    const void* base,
    std::vector<std::byte>& out) {
    if (base == nullptr) {
        return Status::invalid_argument("resident archive base is null", __FILE__, __LINE__);
    }
    ArchiveWriter writer(base);
    write_eagle4_config(writer, config);
    write_eagle4_weights(writer, weights);
    out = writer.release();
    return Status::make_ok();
}

Status read_eagle4_archive(
    const std::byte* data,
    std::size_t size,
    void* block,
    std::size_t block_bytes,
    int device,
    qwen35::eagle4::Eagle4Config& config_out,
    qwen35::eagle4::Eagle4Weights& weights_out) {
    if (data == nullptr || block == nullptr) {
        return Status::invalid_argument("resident archive block is null", __FILE__, __LINE__);
    }
    ArchiveReader reader(data, size, block, block_bytes, device);
    if (reader.u64() != kArchiveSentinel || reader.u64() != kManifestVersion) {
        return Status::invalid_state("resident archive header", __FILE__, __LINE__);
    }
    config_out = read_eagle4_config_archive(reader);
    weights_out = read_eagle4_weights(reader);
    if (!reader.ok()) {
        return reader.status();
    }
    if (reader.remaining() != 0) {
        return Status::invalid_state("resident archive trailing bytes", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

}
}
