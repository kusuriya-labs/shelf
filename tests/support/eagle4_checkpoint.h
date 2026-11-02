#pragma once

#include <phaseshift/core/memory/types.h>

#include <phaseshift/io/safetensors_writer.h>

#include <phaseshift/models/qwen35/eagle4/config.h>


#include <array>
#include <filesystem>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>


namespace eagle4_ckpt {

inline void ckpt_fail(const std::string& msg) {
    std::printf("FAIL eagle4_checkpoint: %s\n", msg.c_str());
}

namespace fs = std::filesystem;

using ps::bf16_t;
using ps::f32_to_bf16_rne;


static constexpr uint32_t kHeads = 2u;
static constexpr uint32_t kHeadDim = 128u;
static constexpr uint32_t kHidden = kHeads * kHeadDim;
static constexpr uint32_t kIntermediate = 64u;
static constexpr uint32_t kVocab = 64u;
static constexpr uint32_t kRank = 8u;
static constexpr uint32_t kCandidates = 16u;
static constexpr uint32_t kWindow = 8u;
static constexpr uint32_t kTaps = 3u;
static constexpr uint32_t kTapFeatures = 256u;
static constexpr float kEps = 1.0e-6f;
static constexpr float kTheta = 10000000.0f;

static float bf16_to_f32(bf16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value.data) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static bf16_t to_bf16(float value) { return bf16_t{f32_to_bf16_rne(value)}; }

static float round_bf16(float value) { return bf16_to_f32(to_bf16(value)); }

struct HostTensor {
    std::vector<bf16_t> data;
    uint32_t rows = 0;
    uint32_t cols = 0;
};

struct HostModel {
    HostTensor fc;
    std::vector<bf16_t> fc_norm[3];
    std::vector<bf16_t> final_norm;
    std::vector<bf16_t> input_norm;
    std::vector<bf16_t> hidden_norm;
    std::vector<bf16_t> post_norm;
    HostTensor q;
    HostTensor k;
    HostTensor v;
    HostTensor o;
    HostTensor gate;
    HostTensor up;
    HostTensor down;
    HostTensor embed;
    HostTensor lm_head;
    HostTensor efficient_head;
    HostTensor efficient_down;
};

static uint32_t lcg(uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

static bf16_t random_bf16(uint32_t& state, float scale) {
    const float u = static_cast<float>(lcg(state) >> 8) / static_cast<float>(1u << 24);
    return to_bf16((u * 2.0f - 1.0f) * scale);
}

static HostTensor make_matrix(uint32_t rows, uint32_t cols, float scale, uint32_t& seed) {
    HostTensor out;
    out.rows = rows;
    out.cols = cols;
    out.data.resize(static_cast<std::size_t>(rows) * cols);
    for (bf16_t& value : out.data) value = random_bf16(seed, scale);
    return out;
}

static std::vector<bf16_t> make_norm(uint32_t features, uint32_t& seed) {
    std::vector<bf16_t> out(features);
    for (bf16_t& value : out) {
        const float u = static_cast<float>(lcg(seed) >> 8) / static_cast<float>(1u << 24);
        value = to_bf16(1.0f + 0.05f * (u * 2.0f - 1.0f));
    }
    return out;
}

static HostModel make_host_model(uint32_t tap_feature_size = kTaps * kTapFeatures,
                                 uint32_t vocab = kVocab, uint32_t hidden = kHidden) {
    uint32_t seed = 12345u;
    HostModel m;
    const float scale = 0.08f;
    m.fc = make_matrix(hidden, tap_feature_size, scale, seed);
    for (uint32_t t = 0; t < 3u; ++t) {
        m.fc_norm[t] = make_norm(tap_feature_size / kTaps, seed);
    }
    m.final_norm = make_norm(hidden, seed);
    m.input_norm = make_norm(hidden, seed);
    m.hidden_norm = make_norm(hidden, seed);
    m.post_norm = make_norm(hidden, seed);
    m.q = make_matrix(hidden, 2u * hidden, scale, seed);
    m.k = make_matrix(hidden, 2u * hidden, scale, seed);
    m.v = make_matrix(hidden, 2u * hidden, scale, seed);
    m.o = make_matrix(hidden, hidden, scale, seed);
    m.gate = make_matrix(kIntermediate, hidden, scale, seed);
    m.up = make_matrix(kIntermediate, hidden, scale, seed);
    m.down = make_matrix(hidden, kIntermediate, scale, seed);
    m.embed = make_matrix(vocab, hidden, 0.2f, seed);
    m.lm_head = make_matrix(vocab, hidden, 0.2f, seed);
    m.efficient_head = make_matrix(vocab, kRank, 0.3f, seed);
    m.efficient_down = make_matrix(kRank, hidden, 0.3f, seed);
    return m;
}

static const HostTensor& find(const HostModel& m, const std::string& name) {
    if (name == "fc.weight") return m.fc;
    if (name == "layers.0.self_attn.q_proj.weight") return m.q;
    if (name == "layers.0.self_attn.k_proj.weight") return m.k;
    if (name == "layers.0.self_attn.v_proj.weight") return m.v;
    if (name == "layers.0.self_attn.o_proj.weight") return m.o;
    if (name == "layers.0.mlp.gate_proj.weight") return m.gate;
    if (name == "layers.0.mlp.up_proj.weight") return m.up;
    if (name == "layers.0.mlp.down_proj.weight") return m.down;
    if (name == "embed_tokens.weight") return m.embed;
    if (name == "lm_head.weight") return m.lm_head;
    if (name == "efficient_lm_head.weight") return m.efficient_head;
    if (name == "efficient_lm_head_down_proj.weight") return m.efficient_down;
    return m.fc;
}

static std::vector<bf16_t> find_vector(const HostModel& m, const std::string& name) {
    if (name == "norm.weight") return m.final_norm;
    if (name == "layers.0.input_layernorm.weight") return m.input_norm;
    if (name == "layers.0.hidden_norm.weight") return m.hidden_norm;
    if (name == "layers.0.post_attention_layernorm.weight") return m.post_norm;
    if (name == "fc_norm.0.weight") return m.fc_norm[0];
    if (name == "fc_norm.1.weight") return m.fc_norm[1];
    if (name == "fc_norm.2.weight") return m.fc_norm[2];
    return m.final_norm;
}

static bool write_shard(
    const std::string& dir,
    const ps::qwen35::eagle4::Eagle4Config& config,
    const HostModel& model) {
    const fs::path path = fs::path(dir) / "model.safetensors";
    auto created = ps::io::SafetensorsWriter::create(path.string());
    if (!created.ok()) {
        ckpt_fail("writer create: " + created.status().message());
        return false;
    }
    ps::io::SafetensorsWriter writer = created.release();
    const auto expected = ps::qwen35::eagle4::eagle4_expected_tensors(config);
    for (const auto& e : expected) {
        if (!writer.plan_tensor(e.name, ps::io::SType::BF16, e.shape).ok()) {
            ckpt_fail("plan " + e.name);
            return false;
        }
    }
    if (!writer.write_header().ok()) {
        ckpt_fail("write_header");
        return false;
    }
    for (const auto& e : expected) {
        std::size_t count = 1;
        for (std::size_t d : e.shape) count *= d;
        std::vector<bf16_t> data;
        if (e.shape.size() == 2) {
            const HostTensor& source = find(model, e.name);
            data = source.data;
        } else {
            data = find_vector(model, e.name);
        }
        if (data.size() != count) {
            ckpt_fail("tensor size mismatch: " + e.name);
            return false;
        }
        if (!writer.write_tensor(e.name, data.data(), data.size() * sizeof(bf16_t)).ok()) {
            ckpt_fail("write " + e.name);
            return false;
        }
    }
    return writer.finish().ok();
}


inline bool write_eagle4_config_json(const std::string& dir, uint32_t target_hidden,
                                     uint32_t vocab,
                                     const std::array<int, 3>& aux_ids) {
    const fs::path path = fs::path(dir) / "config.json";
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.is_open()) return false;
    f << "{\n"
         "\"architectures\": [\"LlamaForCausalLMEagle4\"],\n"
         "\"eagle_config\": {\"eagle_aux_hidden_state_layer_ids\": [";
    f << aux_ids[0] << ", " << aux_ids[1] << ", " << aux_ids[2] << "]},\n";
    f << "\"efficient_lm_head_config\": {\"num_candidates\": 16, \"rank\": 8, "
         "\"type\": \"low_rank\"},\n"
         "\"fc_norm\": true,\n";
    f << "\"head_dim\": 128,\n"
         "\"hidden_size\": 256,\n";
    f << "\"intermediate_size\": 64,\n"
         "\"layer_types\": [\"sliding_attention\"],\n"
         "\"norm_output\": true,\n"
         "\"num_attention_heads\": 2,\n"
         "\"num_aux_hidden_states\": 3,\n"
         "\"num_hidden_layers\": 1,\n"
         "\"num_key_value_heads\": 2,\n"
         "\"rms_norm_eps\": 1e-06,\n"
         "\"rope_parameters\": {\"rope_theta\": 10000000.0, "
         "\"rope_type\": \"default\"},\n"
         "\"sliding_window\": 8,\n";
    f << "\"target_hidden_size\": " << target_hidden << ",\n";
    f << "\"tie_word_embeddings\": false,\n"
         "\"vocab_size\": " << vocab << "\n}\n";
    return f.good();
}

}  // namespace eagle4_ckpt

