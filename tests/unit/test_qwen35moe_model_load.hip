#include <shelf/core/memory/arena.h>
#include <shelf/models/model_weights_factory.h>
#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/models/qwen35/weights/model_weights.h>
#include <shelf/models/qwen35moe/weights/moe_weight_plan.h>
#include <shelf/models/qwen35moe/weights/qwen35moe_model_load.h>

#include <hip/hip_runtime.h>

#include "moe_expert_fixture.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
namespace q35 = shelf::qwen35;
namespace moe = shelf::qwen35moe;
using namespace moe_expert_fixture;

namespace {

int g_fail = 0;

void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

bool message_contains(const shelf::Status& st, const std::string& needle) {
    return std::string(st.message()).find(needle) != std::string::npos;
}

std::uint64_t plane_bytes(const shelf::weights::MatrixWeight& w) {
    return static_cast<std::uint64_t>(w.codes.allocation_bytes() + w.scales.allocation_bytes() +
                                      w.row_ref.allocation_bytes());
}

// A config-only directory: enough for the probes and for the config-level rejections,
// which all happen before any tensor is read.
bool write_dense_config_only(const fs::path& dir) {
    fs::create_directories(dir);
    std::ofstream cfg(dir / "config.json", std::ios::binary | std::ios::trunc);
    cfg << R"json({"model_type":"qwen3_5","text_config":{"model_type":"qwen3_5_text",)json"
           R"json("vocab_size":1024,"hidden_size":256,"intermediate_size":128,)json"
           R"json("num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,)json"
           R"json("head_dim":64}})json";
    return cfg.good();
}

void check_layer(const moe::MoeLayerWeights& w, const moe::MoeExpertGroupPlan& plan,
                 const std::string& tag) {
    check(moe::moe_layer_weights_match_plan(w, plan).ok(), tag + ": planes match the plan");
    check(w.router.rows == kExperts && w.router.cols == kHidden, tag + ": router is [E, hidden]");
    check(w.expert_w13.experts == kExperts && w.expert_w2.experts == kExperts,
          tag + ": expert planes carry the expert count");
    check(w.expert_w13.codes_batch_stride_bytes == plan.w13_codes_batch_stride_bytes,
          tag + ": w13 batch stride survives the model load");
    check(w.expert_w2.scales_batch_stride_bytes == plan.w2_scales_batch_stride_bytes,
          tag + ": w2 scale batch stride survives the model load");
    check(w.shared_gate_proj.rows == kInter && w.shared_gate_proj.cols == kHidden &&
              w.shared_down_proj.rows == kHidden && w.shared_down_proj.cols == kInter,
          tag + ": shared expert shapes");
    check(w.shared_gate.rows == 1u, tag + ": shared expert gate is one row");
}

void check_body(const q35::Qwen35ModelWeights& body, const std::string& tag) {
    check(body.embed_tokens.rows == kVocab && body.embed_tokens.cols == kHidden,
          tag + ": embed_tokens geometry");
    check(body.lm_head.rows == kVocab && body.lm_head.cols == kHidden, tag + ": lm_head geometry");
    check(body.final_norm_weight.ndim() == 1 && body.final_norm_weight.dim(0) == kHidden,
          tag + ": final norm is a hidden vector");
    check(body.layers.size() == kLayers, tag + ": layer count");
    check(body.layers[0].is_gdn && !body.layers[1].is_gdn, tag + ": layer topology");
    for (std::size_t l = 0; l < body.layers.size(); ++l) {
        const q35::Qwen35LayerWeights& layer = body.layers[l];
        check(layer.input_layernorm_weight.ndim() == 1 &&
                  layer.post_attention_layernorm_weight.ndim() == 1,
              tag + "/layer" + std::to_string(l) + ": layernorms loaded");
        check(layer.mlp_gate_proj.rows == 0u && layer.mlp_up_proj.rows == 0u &&
                  layer.mlp_down_proj.rows == 0u,
              tag + "/layer" + std::to_string(l) + ": dense FFN stays empty");
    }
    check(body.layers[0].attn_in_proj_qkv.rows != 0u && body.layers[0].attn_out_proj.rows != 0u,
          tag + ": GDN layer body loaded");
    check(body.layers[1].attn_q_proj.rows == 4u * 64u && body.layers[1].attn_k_proj.rows == 2u * 64u,
          tag + ": attention layer body loaded");
}

}  // namespace

int main() {
    std::printf("=== qwen35moe model-level weight load (GPU) ===\n");
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess)
        return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess)
        return 1;

    std::error_code ec;
    const fs::path root =
        fs::temp_directory_path(ec) / ("shelf_moe_model_load_" + std::to_string(::getpid()));
    fs::remove_all(root, ec);

    Fixture f;
    if (!write_fixture(f, root / "moe", false, true)) {
        std::printf("FAIL moe model fixture write\n");
        return 1;
    }
    const fs::path dense_dir = root / "dense";
    if (!write_dense_config_only(dense_dir)) {
        std::printf("FAIL dense config write\n");
        return 1;
    }

    {
        auto probe = shelf::models::probe_model_dir((root / "moe").string());
        check(probe.ok() && probe.value() == shelf::models::ModelFamily::Moe,
              "probe_model_dir reads the MoE config as Moe");
        auto dense_probe = shelf::models::probe_model_dir(dense_dir.string());
        check(dense_probe.ok() && dense_probe.value() == shelf::models::ModelFamily::Dense,
              "probe_model_dir reads a dense config as Dense");
        check(!shelf::models::probe_model_dir((root / "missing").string()).ok(),
              "probe_model_dir fails when config.json is absent");
    }

    {
        auto plan = moe::build_moe_expert_group_plan(f.config, 0);
        check(plan.ok(), "plan for layer 0");
        const std::vector<std::string> names = moe::moe_layer_logical_names(plan.value());
        check(names.size() == 5u + kExperts * 3u,
              "one layer requests 5 aux tensors plus 3 per expert, got " + std::to_string(names.size()));
        bool no_physical_scale = true;
        for (const auto& n : names)
            if (n.size() > 6 && n.compare(n.size() - 6, 6, "_scale") == 0)
                no_physical_scale = false;
        check(no_physical_scale, "the request list never asks for a physical _scale tensor");
        check(moe::moe_aux_tensor_names(1).size() == 5u &&
                  moe::moe_aux_tensor_names(1) == mlp_aux_names(1),
              "aux tensor names match the checkpoint layout");
        check(plan.value().w13_codes_bytes + plan.value().w13_scales_bytes +
                      plan.value().w13_row_ref_bytes_per_expert * plan.value().experts +
                      plan.value().w2_codes_bytes + plan.value().w2_scales_bytes +
                      plan.value().w2_row_ref_bytes_per_expert * plan.value().experts ==
                  moe::moe_expert_group_bytes(plan.value()),
              "moe_expert_group_bytes sums the plan sizes");
    }

    auto arena = shelf::gpu::GpuArena::create(0, 2ull * 1024ull * 1024ull * 1024ull);
    check(arena.ok(), "arena create");
    shelf::gpu::GpuArena gpu_arena = arena.release();
    const std::uint64_t empty_used = gpu_arena.used();

    auto loaded = moe::load_qwen35moe_weights((root / "moe").string(), gpu_arena, stream);
    check(loaded.ok(), "load_qwen35moe_weights");
    if (loaded.ok()) {
        const moe::MoeModelWeights& w = loaded.value();
        check_body(w.body, "moe body");
        check(w.layers.size() == kLayers, "one MoeLayerWeights per layer");
        std::uint64_t expert_bytes = 0;
        for (std::size_t l = 0; l < kLayers; ++l) {
            auto plan = moe::build_moe_expert_group_plan(f.config, l);
            check(plan.ok(), "plan for layer " + std::to_string(l));
            check_layer(w.layers[l], plan.value(), "layer" + std::to_string(l));
            expert_bytes += plane_bytes(w.layers[l].expert_w13) + plane_bytes(w.layers[l].expert_w2);
            check(expert_bytes == 0 || expert_bytes <= gpu_arena.used() - empty_used,
                  "layer" + std::to_string(l) + ": expert bytes are inside the arena use");
        }
        check(expert_bytes == moe::moe_expert_group_bytes(moe::build_moe_expert_group_plan(f.config, 0).value()) * kLayers,
              "resident expert plane bytes equal the plan sum for both layers");
        std::printf("arena used %llu B, expert planes %llu B\n",
                    static_cast<unsigned long long>(gpu_arena.used() - empty_used),
                    static_cast<unsigned long long>(expert_bytes));
        check(q35::validate_qwen35_weights(w.body, f.config).ok(),
              "the MoE body passes the qwen35 model-level invariants");
    }

    {
        q35::Qwen35LoadOptions mtp;
        mtp.load_mtp_layers = true;
        auto st = moe::load_qwen35moe_weights((root / "moe").string(), gpu_arena, stream, mtp);
        check(!st.ok() && message_contains(st.status(), "does not load MTP layers"),
              "MTP layers are rejected for a MoE checkpoint");

        auto dense = q35::load_qwen35_weights_from_quantized_safetensors((root / "moe").string(),
                                                                        gpu_arena, stream);
        check(!dense.ok() && message_contains(dense.status(), "logical tensor not found"),
              "the dense loader still rejects a MoE checkpoint (dense FFN is absent)");

        auto not_moe = moe::load_qwen35moe_weights(dense_dir.string(), gpu_arena, stream);
        check(!not_moe.ok() && message_contains(not_moe.status(), "num_experts is required"),
              "the MoE loader rejects a dense config");
    }

    if (loaded.ok()) {
        moe::MoeLayerWeights tampered = loaded.value().layers[0];
        tampered.expert_w13.codes_batch_stride_bytes += 256u;
        auto plan = moe::build_moe_expert_group_plan(f.config, 0);
        check(!moe::moe_layer_weights_match_plan(tampered, plan.value()).ok(),
              "a tampered w13 batch stride is rejected by the plan check");
        moe::MoeLayerWeights tampered_router = loaded.value().layers[0];
        tampered_router.router.rows = kExperts - 1u;
        check(!moe::moe_layer_weights_match_plan(tampered_router, plan.value()).ok(),
              "a router that is not [experts, hidden] is rejected");
    }

    fs::remove_all(root, ec);
    gpu_arena.shutdown();
    if (hipStreamDestroy(stream) != hipSuccess)
        check(false, "stream destroy");

    std::printf("test_qwen35moe_model_load: %s (fail=%d)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
