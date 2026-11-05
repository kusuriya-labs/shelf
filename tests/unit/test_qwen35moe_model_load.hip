#include <shelf/core/memory/arena.h>
#include <shelf/models/common/qwen_hybrid/weight_loader.h>
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
#include <type_traits>
#include <unistd.h>
#include <variant>
#include <vector>

namespace fs = std::filesystem;
namespace q35 = shelf::qwen35;
namespace qhy = shelf::qwen_hybrid;
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

template <typename T>
concept has_dense_ffn = requires(const T& w) { w.ffn.gate; };

template <typename T>
concept has_moe_ffn = requires(const T& w) { w.ffn.expert_w13; };

template <typename T>
concept has_nested_body = requires(const T& w) { w.body; };

template <typename T>
concept has_mtp_option = requires(const T& o) { o.load_mtp_layers; };

// A dense layer cannot state an MoE FFN and an MoE layer cannot state a dense FFN, and a
// MoE model carries no second layer array. The load options of the shared backbone have no
// MTP field either, so an MoE load cannot ask for one. All of this is fixed in the types.
static_assert(std::is_base_of_v<qhy::HybridLayerWeights, q35::Qwen35LayerWeights>);
static_assert(std::is_base_of_v<qhy::HybridLayerWeights, moe::Qwen35MoeLayerWeights>);
static_assert(std::is_base_of_v<qhy::HybridModelLevelWeights, moe::Qwen35MoeModelWeights>);
static_assert(has_dense_ffn<q35::Qwen35LayerWeights> && !has_moe_ffn<q35::Qwen35LayerWeights>);
static_assert(has_moe_ffn<moe::Qwen35MoeLayerWeights> && !has_dense_ffn<moe::Qwen35MoeLayerWeights>);
static_assert(!has_nested_body<moe::Qwen35MoeModelWeights>);
static_assert(!has_mtp_option<qhy::HybridLoadOptions>);
static_assert(has_mtp_option<q35::Qwen35LoadOptions>);
static_assert(has_mtp_option<moe::Qwen35MoeLoadOptions>);

void check_ffn(const moe::MoeFfnWeights& ffn, const moe::MoeExpertGroupPlan& plan,
               std::uint32_t shared_inter, const std::string& tag) {
    check(ffn.router.rows == kExperts && ffn.router.cols == kHidden, tag + ": router is [E, hidden]");
    check(ffn.expert_w13.experts == kExperts && ffn.expert_w2.experts == kExperts,
          tag + ": expert planes carry the expert count");
    check(ffn.expert_w13.codes_batch_stride_bytes == plan.w13_codes_batch_stride_bytes,
          tag + ": w13 batch stride survives the model load");
    check(ffn.expert_w2.scales_batch_stride_bytes == plan.w2_scales_batch_stride_bytes,
          tag + ": w2 batch stride survives the model load");
    check(ffn.shared_gate_proj.rows == shared_inter && ffn.shared_gate_proj.cols == kHidden &&
              ffn.shared_up_proj.rows == shared_inter && ffn.shared_down_proj.cols == shared_inter &&
              ffn.shared_down_proj.rows == kHidden,
          tag + ": shared expert shapes follow shared_expert_intermediate_size");
    check(ffn.shared_gate.rows == 1u, tag + ": shared expert gate is one row");
}

void check_backbone(const qhy::HybridLayerWeights& layer, bool is_gdn, const std::string& tag) {
    check(layer.input_layernorm_weight.ndim() == 1 &&
              layer.post_attention_layernorm_weight.ndim() == 1,
          tag + ": layernorms loaded");
    if (is_gdn) {
        check(layer.attn_in_proj_qkv.rows != 0u && layer.attn_out_proj.rows != 0u,
              tag + ": GDN layer body loaded");
        check(layer.attn_q_proj.rows == 0u, tag + ": a GDN layer carries no self_attn weights");
    } else {
        check(layer.attn_q_proj.rows == kAttnQRows && layer.attn_k_proj.rows == kAttnKvRows,
              tag + ": attention layer body loaded");
        check(layer.attn_in_proj_a.rows == 0u,
              tag + ": an attention layer carries no linear_attn weights");
    }
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
    Fixture mf;
    if (!write_fixture(mf, root / "moe_mtp", false, true, kInter, kInter, true)) {
        std::printf("FAIL moe mtp fixture write\n");
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
        const moe::Qwen35MoeModelWeights& w = loaded.value();
        check(w.embed_tokens.rows == kVocab && w.embed_tokens.cols == kHidden,
              "moe model: embed_tokens geometry");
        check(w.lm_head.rows == kVocab && w.lm_head.cols == kHidden,
              "moe model: lm_head geometry");
        check(w.final_norm_weight.ndim() == 1 && w.final_norm_weight.dim(0) == kHidden,
              "moe model: final norm is a hidden vector");
        check(w.layers.size() == kLayers, "one MoE layer per transformer layer");
        std::uint64_t expert_bytes = 0;
        for (std::size_t l = 0; l < kLayers; ++l) {
            auto plan = moe::build_moe_expert_group_plan(f.config, l);
            check(plan.ok(), "plan for layer " + std::to_string(l));
            const std::string tag = "layer" + std::to_string(l);
            check(moe::moe_ffn_weights_match(w.layers[l].ffn, plan.value(), f.config.ffn).ok(),
                  tag + ": FFN weights match the plan");
            check_ffn(w.layers[l].ffn, plan.value(), kInter, tag);
            check_backbone(w.layers[l], l == 0u, tag);
            check(w.layers[l].is_gdn == (l == 0u), tag + ": layer topology");
            expert_bytes += plane_bytes(w.layers[l].ffn.expert_w13) +
                            plane_bytes(w.layers[l].ffn.expert_w2);
            check(expert_bytes == 0 || expert_bytes <= gpu_arena.used() - empty_used,
                  tag + ": expert bytes are inside the arena use");
        }
        check(expert_bytes == moe::moe_expert_group_bytes(moe::build_moe_expert_group_plan(f.config, 0).value()) * kLayers,
              "resident expert plane bytes equal the plan sum for both layers");
        std::printf("arena used %llu B, expert planes %llu B\n",
                    static_cast<unsigned long long>(gpu_arena.used() - empty_used),
                    static_cast<unsigned long long>(expert_bytes));
        check(moe::validate_qwen35moe_weights(w, f.config).ok(),
              "the MoE model passes its own invariants");
    }

    {
        auto dense = q35::load_qwen35_weights_from_quantized_safetensors((root / "moe").string(),
                                                                        gpu_arena, stream);
        check(!dense.ok() && message_contains(dense.status(), "intermediate_size"),
              "the dense loader rejects a MoE config (no dense FFN width)");

        auto not_moe = moe::load_qwen35moe_weights(dense_dir.string(), gpu_arena, stream);
        check(!not_moe.ok() && message_contains(not_moe.status(), "num_experts is required"),
              "the MoE loader rejects a dense config");
    }

    {
        // The shared expert has its own width: a checkpoint that makes it different from
        // the routed experts must still load, and one that carries the routed width there
        // must be rejected. plan.w13_gate_rows is not the shared expert's width.
        constexpr std::uint32_t kWideShared = 192u;
        Fixture wide;
        if (!write_fixture(wide, root / "moe_wide_shared", false, true, kWideShared)) {
            std::printf("FAIL wide shared expert fixture write\n");
            return 1;
        }
        auto loaded_wide = moe::load_qwen35moe_weights((root / "moe_wide_shared").string(),
                                                       gpu_arena, stream);
        check(loaded_wide.ok(),
              "shared_expert_intermediate_size differs from moe_intermediate_size: " +
                  std::string(loaded_wide.ok() ? "" : loaded_wide.status().message()));
        if (loaded_wide.ok()) {
            auto plan = moe::build_moe_expert_group_plan(wide.config, 0);
            check(plan.ok(), "plan for the wide shared fixture");
            check(moe::moe_ffn_weights_match(loaded_wide.value().layers[0].ffn, plan.value(),
                                             wide.config.ffn)
                      .ok(),
                  "the wide shared expert matches its own width");
            check_ffn(loaded_wide.value().layers[0].ffn, plan.value(), kWideShared, "wide shared");
            check(moe::validate_qwen35moe_weights(loaded_wide.value(), wide.config).ok(),
                  "the wide shared model passes its own invariants");

            moe::MoeFfnWeights as_routed = loaded_wide.value().layers[0].ffn;
            as_routed.shared_gate_proj.rows = kInter;
            check(!moe::moe_ffn_weights_match(as_routed, plan.value(), wide.config.ffn).ok(),
                  "a shared expert at the routed width is rejected");
            moe::MoeFfnWeights as_routed_down = loaded_wide.value().layers[0].ffn;
            as_routed_down.shared_down_proj.cols = kInter;
            check(!moe::moe_ffn_weights_match(as_routed_down, plan.value(), wide.config.ffn).ok(),
                  "a shared down_proj at the routed width is rejected");
        }
    }

    if (loaded.ok()) {
        auto plan = moe::build_moe_expert_group_plan(f.config, 0);
        moe::MoeFfnWeights tampered = loaded.value().layers[0].ffn;
        tampered.expert_w13.codes_batch_stride_bytes += 256u;
        check(!moe::moe_ffn_weights_match(tampered, plan.value(), f.config.ffn).ok(),
              "a tampered w13 batch stride is rejected by the plan check");

        moe::MoeFfnWeights tampered_router = loaded.value().layers[0].ffn;
        tampered_router.router.rows = kExperts - 1u;
        check(!moe::moe_ffn_weights_match(tampered_router, plan.value(), f.config.ffn).ok(),
              "a router that is not [experts, hidden] is rejected");

        moe::Qwen35MoeModelWeights bad_topology = loaded.value();
        bad_topology.layers[0].is_gdn = false;
        check(!moe::validate_qwen35moe_weights(bad_topology, f.config).ok(),
              "a layer whose topology disagrees with the config is rejected");

        moe::Qwen35MoeModelWeights missing_body = loaded.value();
        missing_body.layers[1].attn_q_proj = {};
        check(!moe::validate_qwen35moe_weights(missing_body, f.config).ok(),
              "an attention layer without its self_attn weights is rejected");

        moe::Qwen35MoeModelWeights extra_layer = loaded.value();
        extra_layer.layers.push_back({});
        check(!moe::validate_qwen35moe_weights(extra_layer, f.config).ok(),
              "a layer array that does not match the config length is rejected");
    }

    {
        // The factory holds exactly one family's weights, and an MTP request goes to the
        // MoE loader instead of being refused.
        auto via_factory = shelf::models::load_model_weights((root / "moe").string(), gpu_arena,
                                                             stream);
        check(via_factory.ok(), "load_model_weights picks the MoE loader");
        if (via_factory.ok()) {
            check(shelf::models::model_family(via_factory.value()) ==
                      shelf::models::ModelFamily::Moe,
                  "the factory reports the MoE family");
            check(std::get_if<q35::Qwen35ModelWeights>(&via_factory.value()) == nullptr,
                  "the factory holds no dense weights for a MoE checkpoint");
            const auto* m = std::get_if<moe::Qwen35MoeModelWeights>(&via_factory.value());
            check(m != nullptr && m->layers.size() == kLayers,
                  "the factory holds the MoE layer array");
            check(m != nullptr && !m->mtp.present,
                  "a load without the MTP option leaves the block absent");
        }

        q35::Qwen35LoadOptions mtp;
        mtp.load_mtp_layers = true;
        auto via_factory_mtp = shelf::models::load_model_weights((root / "moe_mtp").string(),
                                                                 gpu_arena, stream, mtp);
        check(via_factory_mtp.ok(), "the factory loads an MoE checkpoint's MTP block");
        if (via_factory_mtp.ok()) {
            const auto* m = std::get_if<moe::Qwen35MoeModelWeights>(&via_factory_mtp.value());
            check(m != nullptr && m->mtp.present, "the factory holds the MTP block");
        }

        auto no_mtp = shelf::models::load_model_weights((root / "moe").string(), gpu_arena,
                                                        stream, mtp);
        check(!no_mtp.ok(), "a checkpoint without the MTP block fails an MTP load");
    }

    {
        // The MTP block: the checkpoint stores its routed experts fused and bf16, and the
        // loader turns them into the same native planes a transformer layer gets.
        moe::Qwen35MoeLoadOptions options;
        options.load_mtp_layers = true;
        auto with_mtp = moe::load_qwen35moe_weights((root / "moe_mtp").string(), gpu_arena,
                                                    stream, options);
        check(with_mtp.ok(), "a MoE checkpoint with an MTP block loads");
        if (with_mtp.ok()) {
            const moe::Qwen35MoeModelWeights& w = with_mtp.value();
            check(w.mtp.present, "the MTP block is present");
            check(w.mtp.fc.rows == kHidden && w.mtp.fc.cols == 2u * kHidden,
                  "mtp fc is [hidden, 2 * hidden]");
            check(w.mtp.layer.is_gdn == false, "the MTP layer is a full attention body");
            check(w.mtp.layer.ffn.expert_w13.rows != 0u &&
                      w.mtp.layer.ffn.expert_w2.rows != 0u,
                  "the MTP block keeps its routed experts out of the transformer layer array");
            check_backbone(w.mtp.layer, false, "mtp layer");
            check(moe::validate_qwen35moe_weights(w, mf.config).ok(),
                  "the loaded MTP block passes validation");

            auto plan = moe::build_moe_mtp_expert_group_plan(mf.config);
            check(plan.ok(), "the MTP expert plan builds for the loaded config");
            if (plan.ok()) {
                check(moe::moe_ffn_weights_match(w.mtp.layer.ffn, plan.value(), mf.config.ffn)
                          .ok(),
                      "the MTP FFN matches the fused plan");
                moe::Qwen35MoeMtpWeights absent;
                check(!moe::validate_qwen35moe_mtp_weights(absent, plan.value(), mf.config).ok(),
                      "an mtp block that was not loaded is rejected");
            }

            moe::Qwen35MoeModelWeights short_fc = with_mtp.value();
            short_fc.mtp.fc.cols = kHidden;
            check(!moe::validate_qwen35moe_weights(short_fc, mf.config).ok(),
                  "an mtp fc that is not [hidden, 2 * hidden] is rejected");

            moe::Qwen35MoeModelWeights no_norm = with_mtp.value();
            no_norm.mtp.final_norm_weight = {};
            check(!moe::validate_qwen35moe_weights(no_norm, mf.config).ok(),
                  "an mtp block without its final norm is rejected");

            moe::Qwen35MoeModelWeights no_experts = with_mtp.value();
            no_experts.mtp.layer.ffn.expert_w13 = {};
            check(!moe::validate_qwen35moe_weights(no_experts, mf.config).ok(),
                  "an mtp block without its routed experts is rejected");

            moe::Qwen35MoeModelWeights no_attention = with_mtp.value();
            no_attention.mtp.layer.attn_q_proj = {};
            check(!moe::validate_qwen35moe_weights(no_attention, mf.config).ok(),
                  "an mtp body without its full attention weights is rejected");
        }
    }

    fs::remove_all(root, ec);
    gpu_arena.shutdown();
    if (hipStreamDestroy(stream) != hipSuccess)
        check(false, "stream destroy");

    std::printf("test_qwen35moe_model_load: %s (fail=%d)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
