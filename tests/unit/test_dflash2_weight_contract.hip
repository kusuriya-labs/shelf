#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/arena.h>
#include <shelf/io/safetensors_writer.h>
#include <shelf/models/qwen35/dflash2/config.h>
#include <shelf/models/qwen35/dflash2/weights.h>
#include <shelf/weights/weight_loader.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

using ps::io::SType;
using ps::qwen35::dflash2::DFlash2Config;
using ps::qwen35::dflash2::DFlash2Weights;
using ps::qwen35::dflash2::dflash2_expected_tensors;
using ps::qwen35::dflash2::load_dflash2_weights;
using ps::qwen35::dflash2::read_dflash2_config;
using ps::qwen35::dflash2::validate_dflash2_tensor_contract;

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

static fs::path g_root;

static const char* kSmallConfig = R"json({
  "architectures": ["DFlash2DraftModel"],
  "is_causal": false,
  "dflash_config": {
    "block_size": 4,
    "conv_group_size": 2,
    "conv_kernel_size": 2,
    "mask_token_id": 7,
    "selector_rank": 4,
    "selector_top_k": 2,
    "target_layer_ids": [1, 2]
  },
  "head_dim": 4,
  "hidden_size": 8,
  "intermediate_size": 16,
  "layer_types": ["sliding_attention"],
  "num_attention_heads": 2,
  "num_hidden_layers": 1,
  "num_key_value_heads": 1,
  "num_target_layers": 4,
  "rms_norm_eps": 1e-06,
  "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
  "sliding_window": 8,
  "tie_word_embeddings": false,
  "vocab_size": 32
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

static std::vector<TensorPlan> expected_checkpoint(const DFlash2Config& config) {
    std::vector<TensorPlan> out;
    for (const auto& e : dflash2_expected_tensors(config)) {
        TensorPlan t;
        t.name = e.name;
        t.shape = e.shape;
        t.dtype = SType::BF16;
        const std::size_t n = numel(e.shape);
        t.bytes.resize(n * 2u);
        auto* u = reinterpret_cast<uint16_t*>(t.bytes.data());
        for (std::size_t i = 0; i < n; ++i) {
            u[i] = static_cast<uint16_t>(0x3C00u + static_cast<uint16_t>(i % 64u));
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
    auto hdr = writer.write_header();
    if (!hdr.ok()) {
        check(false, "write_header");
        return false;
    }
    for (TensorPlan& t : tensors) {
        if (!writer.write_tensor(t.name, t.bytes.data(), t.bytes.size()).ok()) {
            check(false, "write " + t.name);
            return false;
        }
    }
    if (!writer.finish().ok()) {
        check(false, "finish");
        return false;
    }
    return true;
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

static DFlash2Config small_config() {
    auto r = read_dflash2_config((g_root / "valid").string());
    check(r.ok(), "small config must parse");
    return r.ok() ? r.value() : DFlash2Config{};
}

static std::vector<uint8_t> d2h(const void* dev, std::size_t bytes) {
    std::vector<uint8_t> host(bytes);
    if (bytes == 0) return host;
    const hipError_t e = hipMemcpy(host.data(), dev, bytes, hipMemcpyDeviceToHost);
    if (e != hipSuccess) {
        check(false, std::string("d2h ") + hipGetErrorString(e));
        host.assign(bytes, 0);
    }
    return host;
}

int main() {
    char tmpl[] = "/tmp/dflash2_weights_XXXXXX";
    const char* dir = mkdtemp(tmpl);
    if (dir == nullptr) {
        std::printf("FAIL mkdtemp\n");
        return 1;
    }
    g_root = fs::path(dir);

    {
        const fs::path d = g_root / "valid";
        fs::create_directories(d);
        std::ofstream f(d / "config.json", std::ios::binary | std::ios::trunc);
        f << kSmallConfig;
    }
    const DFlash2Config config = small_config();
    if (g_fail != 0) {
        std::printf("test_dflash2_weight_contract: failed=%d\n", g_fail);
        return 1;
    }

    {
        const auto expected = dflash2_expected_tensors(config);
        check(expected.size() == 21u, "expected tensor count 21");
        std::size_t fc_index = expected.size();
        std::size_t base_index = expected.size();
        std::size_t codebook_index = expected.size();
        for (std::size_t i = 0; i < expected.size(); ++i) {
            if (expected[i].name == "fc.weight") fc_index = i;
            if (expected[i].name == "layers.0.attention_conv.base_kernel") base_index = i;
            if (expected[i].name == "candidate_selector.successor_codebook") codebook_index = i;
        }
        check(fc_index < expected.size(), "fc.weight in expected list");
        check(base_index < expected.size(), "attention_conv.base_kernel in expected list");
        check(codebook_index < expected.size(), "successor_codebook in expected list");
        if (fc_index < expected.size()) {
            check(expected[fc_index].shape == std::vector<std::size_t>({8u, 16u}),
                  "fc.weight shape [8,16]");
        }
        if (base_index < expected.size()) {
            check(expected[base_index].shape == std::vector<std::size_t>({2u, 2u, 8u}),
                  "base_kernel shape [2,2,8]");
        }
        if (codebook_index < expected.size()) {
            check(expected[codebook_index].shape == std::vector<std::size_t>({32u, 4u}),
                  "successor_codebook shape [32,4]");
        }
        check(config.conv_projection_rows() == 16u, "conv_projection_rows 16");
        check(config.tap_feature_size() == 16u, "tap_feature_size 16");
    }

    auto arena_result = ps::gpu::GpuArena::create(0, 512ull * 1024ull * 1024ull);
    if (!arena_result.ok()) {
        std::printf("SKIP: arena (%s)\n", arena_result.status().message().c_str());
        return 77;
    }
    ps::gpu::GpuArena arena = arena_result.release();
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        std::printf("FAIL hipStreamCreate\n");
        return 1;
    }

    std::vector<TensorPlan> valid = expected_checkpoint(config);
    check(write_model("valid", kSmallConfig, valid), "write valid checkpoint");

    {
        auto st = validate_dflash2_tensor_contract((g_root / "valid").string(), config);
        check(st.ok(), "valid contract: " + (st.ok() ? "" : st.message()));
    }

    {
        const std::size_t used_before = arena.used();
        auto loaded = load_dflash2_weights((g_root / "valid").string(), config, arena, stream);
        check(loaded.ok(), "valid load: " +
                               (loaded.ok() ? "" : loaded.status().message()));
        if (loaded.ok()) {
            DFlash2Weights w = loaded.release();
            check(w.present, "present flag");
            check(w.layers.size() == 1u, "layer count");
            check(w.fc.rows == 8u && w.fc.cols == 16u, "fc rows/cols");
            check(w.fc.bf16.ndim() == 2u && w.fc.bf16.dim(0) == 8u && w.fc.bf16.dim(1) == 16u,
                  "fc tensor shape");
            check(w.selector.hidden_projection.rows == 4u &&
                      w.selector.hidden_projection.cols == 8u,
                  "selector hidden_projection rows/cols");
            check(w.selector.predecessor_codebook.ndim() == 2u &&
                      w.selector.predecessor_codebook.dim(0) == 32u &&
                      w.selector.predecessor_codebook.dim(1) == 4u,
                  "predecessor_codebook shape");
            check(w.selector.successor_codebook.ndim() == 2u &&
                      w.selector.successor_codebook.dim(0) == 32u &&
                      w.selector.successor_codebook.dim(1) == 4u,
                  "successor_codebook shape");
            check(w.hidden_norm_weight.ndim() == 1u && w.hidden_norm_weight.dim(0) == 8u,
                  "hidden_norm shape");
            check(w.final_norm_weight.ndim() == 1u && w.final_norm_weight.dim(0) == 8u,
                  "final_norm shape");
            if (!w.layers.empty()) {
                const auto& lw = w.layers[0];
                check(lw.attn_q_proj.rows == 8u && lw.attn_q_proj.cols == 8u,
                      "q_proj rows/cols");
                check(lw.attn_k_proj.rows == 4u && lw.attn_k_proj.cols == 8u,
                      "k_proj rows/cols");
                check(lw.attn_o_proj.rows == 8u && lw.attn_o_proj.cols == 8u,
                      "o_proj rows/cols");
                check(lw.mlp_down_proj.rows == 8u && lw.mlp_down_proj.cols == 16u,
                      "down_proj rows/cols");
                check(lw.attention_conv_base_kernel.ndim() == 3u &&
                          lw.attention_conv_base_kernel.dim(0) == 2u &&
                          lw.attention_conv_base_kernel.dim(1) == 2u &&
                          lw.attention_conv_base_kernel.dim(2) == 8u,
                      "attention_conv base_kernel shape");
                check(lw.attention_conv_kernel_projection.rows == 16u &&
                          lw.attention_conv_kernel_projection.cols == 8u,
                      "attention_conv kernel_projection rows/cols");
                check(lw.mlp_conv_kernel_projection.rows == 16u &&
                          lw.mlp_conv_kernel_projection.cols == 8u,
                      "mlp_conv kernel_projection rows/cols");
                check(lw.attn_q_norm_weight.ndim() == 1u && lw.attn_q_norm_weight.dim(0) == 4u,
                      "q_norm shape");
            }

            std::size_t fc_bytes = 0;
            for (const TensorPlan& t : valid) {
                if (t.name == "fc.weight") fc_bytes = t.bytes.size();
            }
            const std::vector<uint8_t> got_fc =
                d2h(w.fc.bf16.data<void>(), fc_bytes);
            std::vector<uint8_t> want_fc;
            for (const TensorPlan& t : valid) {
                if (t.name == "fc.weight") want_fc = t.bytes;
            }
            check(got_fc.size() == want_fc.size() &&
                      std::memcmp(got_fc.data(), want_fc.data(), want_fc.size()) == 0,
                  "fc payload round-trip");
        }
        check(arena.used() > used_before, "arena grew after load");
    }

    {
        std::vector<TensorPlan> tensors = expected_checkpoint(config);
        std::vector<TensorPlan> mutated;
        for (TensorPlan& t : tensors) {
            if (t.name != "layers.0.mlp.up_proj.weight") mutated.push_back(std::move(t));
        }
        check(mutated.size() + 1u == tensors.size(), "removed exactly one tensor");
        check(write_model("missing", kSmallConfig, mutated), "write missing checkpoint");
        auto st = validate_dflash2_tensor_contract((g_root / "missing").string(), config);
        check(!st.ok(), "missing tensor must fail the contract");
        check(st.message().find("layers.0.mlp.up_proj.weight") != std::string::npos,
              "missing tensor error names the tensor");
    }

    {
        std::vector<TensorPlan> tensors = expected_checkpoint(config);
        TensorPlan extra;
        extra.name = "layers.0.bogus.weight";
        extra.dtype = SType::BF16;
        extra.shape = {2u, 2u};
        extra.bytes.resize(8u, 0x11u);
        tensors.push_back(std::move(extra));
        check(write_model("unexpected", kSmallConfig, tensors), "write unexpected checkpoint");
        auto st = validate_dflash2_tensor_contract((g_root / "unexpected").string(), config);
        check(!st.ok(), "unexpected tensor must fail the contract");
        check(st.message().find("layers.0.bogus.weight") != std::string::npos,
              "unexpected tensor error names the tensor");
    }

    {
        std::vector<TensorPlan> tensors = expected_checkpoint(config);
        for (TensorPlan& t : tensors) {
            if (t.name == "fc.weight") {
                t.shape = {8u, 15u};
                t.bytes.resize(8u * 15u * 2u, 0x22u);
            }
        }
        check(write_model("bad_shape", kSmallConfig, tensors), "write bad shape checkpoint");
        auto st = validate_dflash2_tensor_contract((g_root / "bad_shape").string(), config);
        check(!st.ok(), "shape mismatch must fail the contract");
        check(st.message().find("fc.weight") != std::string::npos,
              "shape error names the tensor");
    }

    {
        std::vector<TensorPlan> tensors = expected_checkpoint(config);
        for (TensorPlan& t : tensors) {
            if (t.name == "layers.0.self_attn.q_norm.weight") {
                t.dtype = SType::F32;
                t.bytes.resize(4u * 4u, 0u);
            }
        }
        check(write_model("bad_dtype", kSmallConfig, tensors), "write bad dtype checkpoint");
        auto st = validate_dflash2_tensor_contract((g_root / "bad_dtype").string(), config);
        check(!st.ok(), "dtype mismatch must fail the contract");
        check(st.message().find("layers.0.self_attn.q_norm.weight") != std::string::npos,
              "dtype error names the tensor");
    }

    {
        std::vector<TensorPlan> tensors = expected_checkpoint(config);
        check(write_model("quantized", kSmallConfig, tensors), "write quantized checkpoint");
        {
            std::ofstream f(g_root / "quantized" / "shelf_quantization.json",
                            std::ios::binary | std::ios::trunc);
            f << "{\"format\":\"bogus\"}\n";
        }
        auto st = validate_dflash2_tensor_contract((g_root / "quantized").string(), config);
        check(!st.ok(), "quantized checkpoint must be rejected");
    }

    {
        std::string bad = kSmallConfig;
        const std::size_t pos = bad.find("\"hidden_size\": 8");
        check(pos != std::string::npos, "bad config pattern");
        if (pos != std::string::npos) bad.replace(pos, std::strlen("\"hidden_size\": 8"), "");
        const fs::path d = g_root / "bad_config";
        fs::create_directories(d);
        std::ofstream f(d / "config.json", std::ios::binary | std::ios::trunc);
        f << bad;
        auto r = read_dflash2_config(d.string());
        check(!r.ok(), "config missing hidden_size must fail");
    }

    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    {
        const ps::Status shutdown = arena.shutdown();
        check(shutdown.ok(), "arena shutdown: " + shutdown.message());
        check(arena.is_shutdown(), "arena reports shutdown");
    }

    std::printf("test_dflash2_weight_contract: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
