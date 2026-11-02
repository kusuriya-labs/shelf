#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/arena.h>
#include <shelf/io/safetensors_writer.h>
#include <shelf/models/qwen35/eagle4/config.h>
#include <shelf/models/qwen35/eagle4/weights.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

using ps::io::SType;
using ps::qwen35::eagle4::Eagle4Config;
using ps::qwen35::eagle4::Eagle4WeightLoadOptions;
using ps::qwen35::eagle4::Eagle4Weights;
using ps::qwen35::eagle4::eagle4_expected_tensors;
using ps::qwen35::eagle4::load_eagle4_weights;
using ps::qwen35::eagle4::read_eagle4_config;
using ps::qwen35::eagle4::validate_eagle4_tensor_contract;

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

static fs::path g_root;

static const char* kSmallConfig = R"json({
  "architectures": ["LlamaForCausalLMEagle4"],
  "eagle_config": {"eagle_aux_hidden_state_layer_ids": [1, 2, 3]},
  "efficient_lm_head_config": {"num_candidates": 16, "rank": 8, "type": "low_rank"},
  "fc_norm": true,
  "head_dim": 128,
  "hidden_size": 256,
  "intermediate_size": 64,
  "layer_types": ["sliding_attention"],
  "norm_output": true,
  "num_attention_heads": 2,
  "num_aux_hidden_states": 3,
  "num_hidden_layers": 1,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"},
  "sliding_window": 8,
  "target_hidden_size": 256,
  "tie_word_embeddings": false,
  "vocab_size": 64
}
)json";

struct TensorPlan {
    std::string name;
    SType dtype = SType::BF16;
    std::vector<std::size_t> shape;
    std::vector<uint8_t> bytes;
};

static std::size_t numel(const std::vector<std::size_t>& shape) {
    std::size_t n = 1;
    for (std::size_t d : shape) n *= d;
    return n;
}

static std::vector<TensorPlan> expected_checkpoint(const Eagle4Config& config) {
    std::vector<TensorPlan> out;
    for (const auto& e : eagle4_expected_tensors(config)) {
        TensorPlan t;
        t.name = e.name;
        t.shape = e.shape;
        t.dtype = SType::BF16;
        const std::size_t n = numel(e.shape);
        t.bytes.resize(n * 2u);
        auto* u = reinterpret_cast<uint16_t*>(t.bytes.data());
        for (std::size_t i = 0; i < n; ++i) {
            u[i] = static_cast<uint16_t>(0x3C00u + static_cast<uint16_t>(i % 16u));
        }
        out.push_back(std::move(t));
    }
    return out;
}

static bool write_shard(const fs::path& path, std::vector<TensorPlan>& tensors) {
    auto w = ps::io::SafetensorsWriter::create(path.string());
    if (!w.ok()) {
        check(false, "writer create: " + w.status().message());
        return false;
    }
    ps::io::SafetensorsWriter writer = w.release();
    for (TensorPlan& t : tensors) {
        if (!writer.plan_tensor(t.name, t.dtype, t.shape).ok()) {
            check(false, "plan " + t.name);
            return false;
        }
    }
    if (!writer.write_header().ok()) {
        check(false, "write_header");
        return false;
    }
    for (TensorPlan& t : tensors) {
        if (!writer.write_tensor(t.name, t.bytes.data(), t.bytes.size()).ok()) {
            check(false, "write " + t.name);
            return false;
        }
    }
    return writer.finish().ok();
}

static bool write_model(
    const std::string& name,
    const std::string& config_text,
    std::vector<TensorPlan>& tensors) {
    const fs::path dir = g_root / name;
    fs::create_directories(dir);
    {
        std::ofstream f(dir / "config.json", std::ios::binary | std::ios::trunc);
        f << config_text;
    }
    return write_shard(dir / "model.safetensors", tensors);
}

static const TensorPlan* find_tensor(
    const std::vector<TensorPlan>& tensors, const std::string& name) {
    for (const TensorPlan& t : tensors) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

int main() {
    g_root = fs::temp_directory_path() /
             ("ps_eagle4_contract_" + std::to_string(static_cast<long>(getpid())));
    fs::remove_all(g_root);
    fs::create_directories(g_root);

    {
        std::ofstream f(g_root / "config.json", std::ios::binary | std::ios::trunc);
        f << kSmallConfig;
    }
    auto config_result = read_eagle4_config(g_root.string());
    check(config_result.ok(), "small config reads");
    if (!config_result.ok()) {
        std::printf("FAILED (%d)\n", g_fail);
        return 1;
    }
    const Eagle4Config config = config_result.release();

    std::vector<TensorPlan> base = expected_checkpoint(config);
    check(base.size() == 19u, "expected tensor count");

    {
        std::vector<TensorPlan> tensors = base;
        check(write_model("base", kSmallConfig, tensors), "base model writes");
    }
    {
        const std::string dir = (g_root / "base").string();
        check(validate_eagle4_tensor_contract(dir, config).ok(), "base contract passes");
    }

    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        std::printf("FAIL hipStreamCreate\n");
        return 1;
    }
    const char* real_dir = std::getenv("SHELF_EAGLE4_MODEL_DIR");
    const bool check_real = real_dir != nullptr && real_dir[0] != '\0';
    auto arena_result = ps::gpu::GpuArena::create(
        0, check_real ? 12ull * 1024ull * 1024ull * 1024ull : 256ull * 1024ull * 1024ull);
    if (!arena_result.ok()) {
        std::printf("SKIP: arena (%s)\n", arena_result.status().message().c_str());
        (void)hipStreamDestroy(stream);
        return 77;
    }
    ps::gpu::GpuArena arena = arena_result.release();
    {
        auto loaded = load_eagle4_weights((g_root / "base").string(), config, arena, stream);
        check(loaded.ok(), "weights load");
        if (loaded.ok()) {
            Eagle4Weights weights = loaded.release();
            check(weights.present, "weights present");
            check(weights.layers.size() == 1u, "layer count");
            check(weights.embed_tokens.rows == 64u, "embed rows");
            check(weights.candidate.lm_head.rows == 64u, "lm_head rows");
            check(weights.candidate.efficient_lm_head.rows == 64u, "efficient head rows");
            check(weights.candidate.efficient_lm_head.cols == 8u, "efficient head cols");
            check(weights.candidate.efficient_lm_head_down_proj.rows == 8u, "rank rows");
            check(weights.fc.rows == 256u && weights.fc.cols == 768u, "fc shape");
            check(weights.layers[0].attn_q_proj.cols == 512u, "q_proj input features");
        }
    }
    {
        Eagle4WeightLoadOptions options;
        options.load_embed_tokens = false;
        options.load_lm_head = false;
        auto loaded =
            load_eagle4_weights((g_root / "base").string(), config, arena, stream, options);
        check(loaded.ok(), "weights load without shared tables");
        if (loaded.ok()) {
            Eagle4Weights weights = loaded.release();
            check(weights.embed_tokens.rows == 0u, "embed skipped");
            check(weights.candidate.lm_head.rows == 0u, "lm_head skipped");
        }
    }

    {
        std::vector<TensorPlan> tensors = base;
        for (TensorPlan& t : tensors) {
            if (t.name == "fc.weight") t.name = "fc_bad.weight";
        }
        check(write_model("renamed", kSmallConfig, tensors), "renamed model writes");
        check(!validate_eagle4_tensor_contract((g_root / "renamed").string(), config).ok(),
              "missing tensor rejected");
    }
    {
        std::vector<TensorPlan> tensors = base;
        for (TensorPlan& t : tensors) {
            if (t.name == "layers.0.self_attn.q_proj.weight") {
                t.shape = {256u, 256u};
                t.bytes.resize(numel(t.shape) * 2u, 0u);
            }
        }
        check(write_model("shape", kSmallConfig, tensors), "shape model writes");
        check(!validate_eagle4_tensor_contract((g_root / "shape").string(), config).ok(),
              "shape mismatch rejected");
    }
    {
        std::vector<TensorPlan> tensors = base;
        for (TensorPlan& t : tensors) {
            if (t.name == "norm.weight") {
                t.dtype = SType::F32;
                const std::size_t n = numel(t.shape);
                t.bytes.resize(n * 4u, 0u);
            }
        }
        check(write_model("dtype", kSmallConfig, tensors), "dtype model writes");
        check(!validate_eagle4_tensor_contract((g_root / "dtype").string(), config).ok(),
              "dtype mismatch rejected");
    }
    {
        std::vector<TensorPlan> tensors = base;
        tensors.push_back(TensorPlan{"extra.weight", SType::BF16, {8u}, std::vector<uint8_t>(16u, 0u)});
        check(write_model("extra", kSmallConfig, tensors), "extra model writes");
        check(!validate_eagle4_tensor_contract((g_root / "extra").string(), config).ok(),
              "unexpected tensor rejected");
    }

    if (check_real) {
        auto real_config = read_eagle4_config(real_dir);
        check(real_config.ok(), "real config reads");
        if (real_config.ok()) {
            check(validate_eagle4_tensor_contract(real_dir, real_config.value()).ok(),
                  "real checkpoint contract passes");
            auto real_loaded =
                load_eagle4_weights(real_dir, real_config.value(), arena, stream);
            check(real_loaded.ok(), "real weights load");
        }
    }

    (void)hipStreamDestroy(stream);
    fs::remove_all(g_root);
    if (g_fail != 0) {
        std::printf("FAILED (%d)\n", g_fail);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
