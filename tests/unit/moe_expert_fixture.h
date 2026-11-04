#pragma once
// Synthetic HF Quark MXFP4 MoE checkpoint: unfused routed experts (rank-2 codes and
// E8M0 scales per expert, written in the checkpoint's file order) plus BF16 router and
// shared-expert tensors. The payload carries dead rows (zero codes, E8M0 byte 1), an
// E8M0 byte near the top of the exponent range in w2, and a per-row spread that keeps
// w13 inside the exact fold band while w2 falls outside it.
#include <shelf/core/memory/types.h>
#include <shelf/io/safetensors_writer.h>
#include <shelf/models/qwen35moe/model/qwen35moe_config.h>
#include <shelf/models/qwen35moe/weights/moe_expert_load.h>
#include <shelf/models/qwen35moe/weights/qwen35moe_model_load.h>
#include <shelf/models/qwen35moe/weights/moe_weight_plan.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace moe_expert_fixture {

namespace fs = std::filesystem;

inline constexpr std::size_t kVocab = 1024;
inline constexpr std::uint32_t kExperts = 4;
inline constexpr std::uint32_t kHidden = 256;
inline constexpr std::uint32_t kInter = 128;
inline constexpr std::size_t kLayers = 2;
// The GDN geometry of the linear_attention layer, stated so the checkpoint tensors and the
// config agree: in_proj_qkv carries the convolution features, in_proj_a / in_proj_b and
// dt_bias / A_log carry the value head count, and out_proj reads the value features.
inline constexpr std::uint32_t kKeyHeads = 2u;
inline constexpr std::uint32_t kKeyHeadDim = 128u;
inline constexpr std::uint32_t kValueHeads = 4u;
inline constexpr std::uint32_t kValueHeadDim = 128u;
inline constexpr std::uint32_t kConvKernel = 4u;
inline constexpr std::uint32_t kQkFeatures = kKeyHeads * kKeyHeadDim;
inline constexpr std::uint32_t kValueFeatures = kValueHeads * kValueHeadDim;
inline constexpr std::uint32_t kConvFeatures = 2u * kQkFeatures + kValueFeatures;
// The full_attention layer geometry, stated so the checkpoint tensors and the config agree:
// q_proj carries the output gate twice (the target checkpoint's convention), k_proj and
// v_proj carry the key-value head count, o_proj reads the head features once.
inline constexpr std::uint32_t kAttnHeads = 4u;
inline constexpr std::uint32_t kKvHeads = 2u;
inline constexpr std::uint32_t kAttnHeadDim = 128u;
inline constexpr std::uint32_t kAttnHeadFeatures = kAttnHeads * kAttnHeadDim;
inline constexpr std::uint32_t kAttnQRows = 2u * kAttnHeadFeatures;
inline constexpr std::uint32_t kAttnKvRows = kKvHeads * kAttnHeadDim;
inline constexpr std::uint8_t kE8m0DeadByte = 1u;
// High enough to leave the exact fold band (row spread 11 > 8) while staying within
// three powers of two of the other blocks, so the same payload serves the load test and
// the grouped GEMM numerics without making the f32 accumulation ill-conditioned.
inline constexpr std::uint8_t kE8m0HighByte = 123u;
inline constexpr std::uint8_t kE8m0NanByte = 255u;
inline constexpr std::uint8_t kE8m0Base = 112u;
inline constexpr std::uint8_t kE8m0Spread = 8u;
inline constexpr std::uint32_t kPoisonExpert = 1u;
inline constexpr std::uint32_t kPoisonRow = 2u;
inline constexpr std::uint32_t kPoisonBlock = 1u;

struct Fixture {
    fs::path dir;
    shelf::qwen35moe::Qwen35MoeTextConfig config;
    // Canonical on-disk payload per tensor name, exactly as written.
    std::map<std::string, std::vector<std::uint8_t>> codes;
    std::map<std::string, std::vector<std::uint8_t>> scales;
};

inline shelf::qwen35moe::Qwen35MoeTextConfig text_config(
    std::uint32_t shared_inter = kInter, std::uint32_t inter = kInter) {
    shelf::qwen35moe::Qwen35MoeTextConfig c;
    c.vocab_size = 1024;
    c.hidden_size = kHidden;
    c.num_hidden_layers = kLayers;
    c.ffn.num_experts = kExperts;
    c.ffn.num_experts_per_tok = 2;
    c.ffn.moe_intermediate_size = inter;
    c.ffn.shared_expert_intermediate_size = shared_inter;
    c.num_attention_heads = kAttnHeads;
    c.num_key_value_heads = kKvHeads;
    c.attention_head_dim = kAttnHeadDim;
    c.linear_num_key_heads = kKeyHeads;
    c.linear_key_head_dim = kKeyHeadDim;
    c.linear_num_value_heads = kValueHeads;
    c.linear_value_head_dim = kValueHeadDim;
    c.linear_conv_kernel_dim = kConvKernel;
    // Matches the config_json layer_types order (layer 0 linear_attention, layer 1
    // full_attention) the way the model-level topology check derives it.
    c.full_attention_interval = 2;
    c.layer_types = {0, 1};
    return c;
}

inline std::string config_json(std::uint32_t shared_inter = kInter,
                                std::uint32_t inter = kInter) {
    std::string json = R"json({"model_type":"qwen3_5_moe","tie_word_embeddings":false,"text_config":{)json";
    json += R"json("model_type":"qwen3_5_moe_text","vocab_size":1024,"hidden_size":)json";
    json += std::to_string(kHidden);
    json += R"json(,"num_hidden_layers":)json";
    json += std::to_string(kLayers);
    json += R"json(,"num_experts":)json";
    json += std::to_string(kExperts);
    json += R"json(,"num_experts_per_tok":2,"moe_intermediate_size":)json";
    json += std::to_string(inter);
    json += R"json(,"shared_expert_intermediate_size":)json";
    json += std::to_string(shared_inter);
    json += R"json(,"num_attention_heads":)json";
    json += std::to_string(kAttnHeads);
    json += R"json(,"num_key_value_heads":)json";
    json += std::to_string(kKvHeads);
    json += R"json(,"head_dim":)json";
    json += std::to_string(kAttnHeadDim);
    json += R"json(,"linear_num_key_heads":)json";
    json += std::to_string(kKeyHeads);
    json += R"json(,"linear_key_head_dim":)json";
    json += std::to_string(kKeyHeadDim);
    json += R"json(,"linear_num_value_heads":)json";
    json += std::to_string(kValueHeads);
    json += R"json(,"linear_value_head_dim":)json";
    json += std::to_string(kValueHeadDim);
    json += R"json(,"linear_conv_kernel_dim":)json";
    json += std::to_string(kConvKernel);
    json += R"json(,)json";
    json += R"json("full_attention_interval":2,"layer_types":["linear_attention","full_attention"]},)json";
    json += R"json("quantization_config":{"quant_method":"quark","version":"0.13",)json";
    json += R"json("global_quant_config":{"weight":{"dtype":"fp4","qscheme":"per_group",)json";
    json += R"json("group_size":32,"is_dynamic":false,"scale_format":"e8m0","ch_axis":-1},)json";
    json += R"json("input_tensors":{"dtype":"fp4","qscheme":"per_group","group_size":32,)json";
    json += R"json("is_dynamic":true,"scale_format":"e8m0"}},)json";
    json += R"json("export":{"weight_format":"real_quantized","pack_method":"reorder"}}})json";
    return json;
}

inline std::vector<std::uint8_t> make_codes(std::size_t rows, std::size_t packed_k,
                                            std::uint32_t seed, bool dead_first_row) {
    std::vector<std::uint8_t> out(rows * packed_k);
    for (std::size_t row = 0; row < rows; ++row) {
        const bool dead = dead_first_row && row == 0u;
        for (std::size_t col = 0; col < packed_k; ++col) {
            const std::size_t i = row * packed_k + col;
            out[i] = dead ? 0u
                          : static_cast<std::uint8_t>((i * 37u + row * 11u + seed) & 0xFFu);
        }
    }
    return out;
}

inline std::vector<std::uint8_t> make_scales(std::size_t rows, std::size_t blocks,
                                             std::uint32_t seed, bool dead_first_row,
                                             bool with_high_byte, bool with_nan_byte) {
    std::vector<std::uint8_t> out(rows * blocks);
    for (std::size_t row = 0; row < rows; ++row) {
        const bool dead = dead_first_row && row == 0u;
        for (std::size_t col = 0; col < blocks; ++col) {
            if (dead) {
                out[row * blocks + col] = kE8m0DeadByte;
                continue;
            }
            std::uint8_t value = static_cast<std::uint8_t>(kE8m0Base + ((row + col + seed) % kE8m0Spread));
            if (with_high_byte && row == 1u && col == 0u)
                value = kE8m0HighByte;
            if (with_nan_byte && row == kPoisonRow && col == kPoisonBlock)
                value = kE8m0NanByte;
            out[row * blocks + col] = value;
        }
    }
    return out;
}

inline std::vector<std::uint8_t> make_bf16(std::size_t count, std::uint32_t seed) {
    std::vector<std::uint8_t> out(count * 2u);
    for (std::size_t i = 0; i < count; ++i) {
        const float value = ((i + seed) % 7u == 0u) ? -1.5f : 0.25f;
        reinterpret_cast<std::uint16_t*>(out.data())[i] = shelf::f32_to_bf16_rne(value);
    }
    return out;
}

// The nan byte is injected into one gate_proj scale of expert kPoisonExpert so the
// w13 plane is the one that carries it.
inline bool poisons_plane(const char* projection, std::uint32_t expert) {
    return std::string(projection) == "gate_proj" && expert == kPoisonExpert;
}

struct PlannedTensor {
    std::string name;
    shelf::io::SType dtype;
    std::vector<std::size_t> shape;
    std::vector<std::uint8_t> bytes;
};

// Adds the tensors a full model load asks for: embed / lm_head / final norm and, per
// layer, the attention body. Shapes follow the synthetic config (4 heads, 2 kv heads,
// head_dim 64) and the layer_types order (layer 0 is GDN).
inline void add_model_level_tensors(std::vector<PlannedTensor>& planned,
                                    std::size_t layer) {
    const std::string prefix = "model.language_model.layers." + std::to_string(layer) + ".";
    planned.push_back({prefix + "input_layernorm.weight", shelf::io::SType::BF16, {kHidden},
                       make_bf16(kHidden, 31u + static_cast<std::uint32_t>(layer))});
    planned.push_back({prefix + "post_attention_layernorm.weight", shelf::io::SType::BF16,
                       {kHidden}, make_bf16(kHidden, 37u + static_cast<std::uint32_t>(layer))});
    const auto matrix = [&](const std::string& name, std::size_t rows, std::size_t cols,
                            std::uint32_t seed) {
        planned.push_back({prefix + name, shelf::io::SType::BF16, {rows, cols},
                           make_bf16(rows * cols, seed)});
    };
    if (layer % 2u == 0u) {
        matrix("linear_attn.in_proj_a.weight", kValueHeads, kHidden,
               41u + static_cast<std::uint32_t>(layer));
        matrix("linear_attn.in_proj_b.weight", kValueHeads, kHidden,
               43u + static_cast<std::uint32_t>(layer));
        matrix("linear_attn.in_proj_qkv.weight", kConvFeatures, kHidden,
               47u + static_cast<std::uint32_t>(layer));
        matrix("linear_attn.in_proj_z.weight", kValueFeatures, kHidden,
               53u + static_cast<std::uint32_t>(layer));
        planned.push_back({prefix + "linear_attn.conv1d.weight", shelf::io::SType::BF16,
                           {kConvFeatures, 1u, kConvKernel},
                           make_bf16(kConvFeatures * kConvKernel,
                                     59u + static_cast<std::uint32_t>(layer))});
        matrix("linear_attn.out_proj.weight", kHidden, kValueFeatures,
               61u + static_cast<std::uint32_t>(layer));
        planned.push_back({prefix + "linear_attn.norm.weight", shelf::io::SType::BF16,
                           {kValueHeadDim},
                           make_bf16(kValueHeadDim, 67u + static_cast<std::uint32_t>(layer))});
        planned.push_back({prefix + "linear_attn.dt_bias", shelf::io::SType::BF16, {kValueHeads},
                           make_bf16(kValueHeads, 71u + static_cast<std::uint32_t>(layer))});
        planned.push_back({prefix + "linear_attn.A_log", shelf::io::SType::BF16, {kValueHeads},
                           make_bf16(kValueHeads, 73u + static_cast<std::uint32_t>(layer))});
    } else {
        matrix("self_attn.q_proj.weight", kAttnQRows, kHidden,
               79u + static_cast<std::uint32_t>(layer));
        matrix("self_attn.k_proj.weight", kAttnKvRows, kHidden,
               83u + static_cast<std::uint32_t>(layer));
        matrix("self_attn.v_proj.weight", kAttnKvRows, kHidden,
               89u + static_cast<std::uint32_t>(layer));
        matrix("self_attn.o_proj.weight", kHidden, kAttnHeadFeatures,
               97u + static_cast<std::uint32_t>(layer));
        planned.push_back({prefix + "self_attn.q_norm.weight", shelf::io::SType::BF16,
                           {kAttnHeadDim},
                           make_bf16(kAttnHeadDim, 101u + static_cast<std::uint32_t>(layer))});
        planned.push_back({prefix + "self_attn.k_norm.weight", shelf::io::SType::BF16,
                           {kAttnHeadDim},
                           make_bf16(kAttnHeadDim, 103u + static_cast<std::uint32_t>(layer))});
    }
}

// Writes one safetensors shard plus config.json. Tensors are emitted in checkpoint
// file order (expert-major, then name order: down_proj, gate_proj, up_proj). With
// model_level the shard also carries the model-level and attention tensors, so the
// directory can go through a full model load.
// shared_inter sets the checkpoint's shared_expert_intermediate_size, which a real
// checkpoint may make different from the routed moe_intermediate_size.
inline bool write_fixture(Fixture& f, const fs::path& dir, bool with_nan_byte = false,
                          bool model_level = false, std::uint32_t shared_inter = kInter,
                          std::uint32_t inter = kInter) {
    f.dir = dir;
    f.config = text_config(shared_inter, inter);
    std::error_code ec;
    fs::create_directories(dir, ec);

    std::vector<PlannedTensor> planned;
    if (model_level) {
        planned.push_back({"model.language_model.embed_tokens.weight", shelf::io::SType::BF16,
                           {kVocab, kHidden}, make_bf16(kVocab * kHidden, 3u)});
        planned.push_back({"model.language_model.lm_head.weight", shelf::io::SType::BF16,
                           {kVocab, kHidden}, make_bf16(kVocab * kHidden, 5u)});
        planned.push_back({"model.language_model.norm.weight", shelf::io::SType::BF16, {kHidden},
                           make_bf16(kHidden, 11u)});
    }

    for (std::size_t layer = 0; layer < kLayers; ++layer) {
        for (std::uint32_t e = 0; e < kExperts; ++e) {
            const auto push = [&](const char* projection, std::size_t rows, std::size_t cols,
                                  std::uint32_t seed, bool dead_first_row, bool with_high_byte) {
                const std::string base =
                    shelf::qwen35moe::moe_expert_tensor_name(layer, e, projection, false);
                std::vector<std::uint8_t> codes = make_codes(rows, cols / 2u, seed, dead_first_row);
                std::vector<std::uint8_t> scales =
                    make_scales(rows, cols / 32u, seed, dead_first_row, with_high_byte,
                                with_nan_byte && poisons_plane(projection, e));
                planned.push_back({base, shelf::io::SType::U8, {rows, cols / 2u}, codes});
                planned.push_back({base + "_scale", shelf::io::SType::U8, {rows, cols / 32u},
                                   scales});
                f.codes[base] = std::move(codes);
                f.scales[base + "_scale"] = std::move(scales);
            };
            // down_proj: rows = hidden, cols = inter. w2 carries the high E8M0 byte so its
            // row spread leaves the exact fold band, while w13 stays inside it.
            push("down_proj", kHidden, inter, 100u + e, true, true);
            push("gate_proj", inter, kHidden, 200u + e, true, false);
            push("up_proj", inter, kHidden, 300u + e, true, false);
        }
        const std::string prefix =
            "model.language_model.layers." + std::to_string(layer) + ".mlp.";
        planned.push_back({prefix + "gate.weight", shelf::io::SType::BF16, {kExperts, kHidden},
                           make_bf16(kExperts * kHidden, 7u + static_cast<std::uint32_t>(layer))});
        planned.push_back({prefix + "shared_expert_gate.weight", shelf::io::SType::BF16,
                           {1, kHidden}, make_bf16(kHidden, 13u)});
        planned.push_back({prefix + "shared_expert.gate_proj.weight", shelf::io::SType::BF16,
                           {shared_inter, kHidden}, make_bf16(shared_inter * kHidden, 17u)});
        planned.push_back({prefix + "shared_expert.up_proj.weight", shelf::io::SType::BF16,
                           {shared_inter, kHidden}, make_bf16(shared_inter * kHidden, 19u)});
        planned.push_back({prefix + "shared_expert.down_proj.weight", shelf::io::SType::BF16,
                           {kHidden, shared_inter}, make_bf16(kHidden * shared_inter, 23u)});
        if (model_level) add_model_level_tensors(planned, layer);
    }

    auto writer = shelf::io::SafetensorsWriter::create((dir / "model.safetensors").string());
    if (!writer.ok())
        return false;
    for (const auto& t : planned) {
        if (!writer.value().plan_tensor(t.name, t.dtype, t.shape).ok())
            return false;
    }
    auto base = writer.value().write_header();
    if (!base.ok())
        return false;
    for (const auto& t : planned) {
        if (!writer.value().write_tensor(t.name, t.bytes.data(), t.bytes.size()).ok())
            return false;
    }
    if (!writer.value().finish().ok())
        return false;

    std::ofstream cfg(dir / "config.json", std::ios::binary | std::ios::trunc);
    cfg << config_json(shared_inter, inter);
    return cfg.good();
}

// Logical names of one layer's routed expert tensors, in checkpoint file order. The
// scale tensors are physical: they hang off these names with the `_scale` suffix.
inline std::vector<std::string> expert_names(std::size_t layer) {
    std::vector<std::string> names;
    for (std::uint32_t e = 0; e < kExperts; ++e) {
        for (const char* projection : {"down_proj", "gate_proj", "up_proj"})
            names.push_back(shelf::qwen35moe::moe_expert_tensor_name(layer, e, projection, false));
    }
    return names;
}

// Logical names of one layer's non-expert mlp tensors.
inline std::vector<std::string> mlp_aux_names(std::size_t layer) {
    return shelf::qwen35moe::moe_aux_tensor_names(layer);
}

}  // namespace moe_expert_fixture
