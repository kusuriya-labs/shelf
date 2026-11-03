#pragma once
// Synthetic HF Quark MXFP4 MoE checkpoint: unfused routed experts (rank-2 codes and
// E8M0 scales per expert, written in the checkpoint's file order) plus BF16 router and
// shared-expert tensors. The payload carries dead rows (zero codes, E8M0 byte 1), an
// E8M0 byte near the top of the exponent range in w2, and a per-row spread that keeps
// w13 inside the exact fold band while w2 falls outside it.
#include <shelf/core/memory/types.h>
#include <shelf/io/safetensors_writer.h>
#include <shelf/models/qwen35moe/model/qwen35moe_config.h>
#include <shelf/models/qwen35moe/weights/moe_weight_plan.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace moe_expert_fixture {

namespace fs = std::filesystem;

inline constexpr std::uint32_t kExperts = 4;
inline constexpr std::uint32_t kHidden = 256;
inline constexpr std::uint32_t kInter = 128;
inline constexpr std::size_t kLayers = 2;
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

inline shelf::qwen35moe::Qwen35MoeTextConfig text_config() {
    shelf::qwen35moe::Qwen35MoeTextConfig c;
    c.vocab_size = 1024;
    c.hidden_size = kHidden;
    c.num_hidden_layers = kLayers;
    c.num_experts = kExperts;
    c.num_experts_per_tok = 2;
    c.moe_intermediate_size = kInter;
    c.shared_expert_intermediate_size = kInter;
    c.num_attention_heads = 4;
    c.num_key_value_heads = 2;
    c.attention_head_dim = 64;
    return c;
}

inline std::string config_json() {
    std::string json = R"json({"model_type":"qwen3_5_moe","tie_word_embeddings":false,"text_config":{)json";
    json += R"json("model_type":"qwen3_5_moe_text","vocab_size":1024,"hidden_size":)json";
    json += std::to_string(kHidden);
    json += R"json(,"num_hidden_layers":)json";
    json += std::to_string(kLayers);
    json += R"json(,"num_experts":)json";
    json += std::to_string(kExperts);
    json += R"json(,"num_experts_per_tok":2,"moe_intermediate_size":)json";
    json += std::to_string(kInter);
    json += R"json(,"shared_expert_intermediate_size":)json";
    json += std::to_string(kInter);
    json += R"json(,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":64,)json";
    json += R"json("layer_types":["linear_attention","full_attention"]},)json";
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

// Writes one safetensors shard plus config.json. Tensors are emitted in checkpoint
// file order (expert-major, then name order: down_proj, gate_proj, up_proj).
inline bool write_fixture(Fixture& f, const fs::path& dir, bool with_nan_byte = false) {
    f.dir = dir;
    f.config = text_config();
    std::error_code ec;
    fs::create_directories(dir, ec);

    struct PlannedTensor {
        std::string name;
        shelf::io::SType dtype;
        std::vector<std::size_t> shape;
        std::vector<std::uint8_t> bytes;
    };
    std::vector<PlannedTensor> planned;

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
            push("down_proj", kHidden, kInter, 100u + e, true, true);
            push("gate_proj", kInter, kHidden, 200u + e, true, false);
            push("up_proj", kInter, kHidden, 300u + e, true, false);
        }
        const std::string prefix =
            "model.language_model.layers." + std::to_string(layer) + ".mlp.";
        planned.push_back({prefix + "gate.weight", shelf::io::SType::BF16, {kExperts, kHidden},
                           make_bf16(kExperts * kHidden, 7u + static_cast<std::uint32_t>(layer))});
        planned.push_back({prefix + "shared_expert_gate.weight", shelf::io::SType::BF16,
                           {1, kHidden}, make_bf16(kHidden, 13u)});
        planned.push_back({prefix + "shared_expert.gate_proj.weight", shelf::io::SType::BF16,
                           {kInter, kHidden}, make_bf16(kInter * kHidden, 17u)});
        planned.push_back({prefix + "shared_expert.up_proj.weight", shelf::io::SType::BF16,
                           {kInter, kHidden}, make_bf16(kInter * kHidden, 19u)});
        planned.push_back({prefix + "shared_expert.down_proj.weight", shelf::io::SType::BF16,
                           {kHidden, kInter}, make_bf16(kHidden * kInter, 23u)});
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
    cfg << config_json();
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
    const std::string prefix =
        "model.language_model.layers." + std::to_string(layer) + ".mlp.";
    return {prefix + "gate.weight", prefix + "shared_expert_gate.weight",
            prefix + "shared_expert.gate_proj.weight", prefix + "shared_expert.up_proj.weight",
            prefix + "shared_expert.down_proj.weight"};
}

}  // namespace moe_expert_fixture
