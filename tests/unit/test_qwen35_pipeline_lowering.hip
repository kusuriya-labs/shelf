#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/types.h>
#include <shelf/models/qwen35/model/lower_to_primitives.h>
#include <shelf/models/qwen35/model/qwen35_config.h>
#include <shelf/models/qwen35/model/stage_range.h>
#include <shelf/models/qwen35/weights/model_weights.h>
#include <shelf/runtime/graph/primitive_graph.h>
#include <shelf/weights/matrix_weight.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace rt = ::shelf::runtime;
namespace gpu = ::shelf::gpu;

namespace {

int g_pass = 0;
int g_fail = 0;

void check(bool cond, const std::string& msg) {
    if (cond) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("FAIL: %s\n", msg.c_str());
    }
}

::shelf::Result<gpu::Tensor> alloc_bf16(gpu::GpuArena& arena,
                                       const std::vector<std::size_t>& shape) {
    std::size_t count = 1;
    for (std::size_t d : shape) count *= d;
    auto view = arena.allocate_aligned(count * sizeof(std::uint16_t), 128);
    if (!view.ok()) return view.status();
    std::vector<std::size_t> strides(shape.size(), 1);
    for (std::size_t i = 0; i + 1 < shape.size(); ++i)
        strides[i] = strides[i + 1] * shape[i + 1];
    return gpu::Tensor::create<shelf::bf16_t>(view.release(), shape, strides);
}

shelf::weights::MatrixWeight make_matrix(gpu::Tensor t, std::uint32_t rows,
                                         std::uint32_t cols) {
    shelf::weights::MatrixWeight w;
    w.bf16 = std::move(t);
    w.rows = rows;
    w.cols = cols;
    return w;
}

bool has_debug_prefix(const rt::PrimitiveGraph& g, const char* prefix) {
    for (const auto& n : g.nodes) {
        if (n.debug_name.rfind(prefix, 0) == 0) return true;
    }
    return false;
}

}  // namespace

int main() {
    constexpr std::uint32_t kV = 8;
    constexpr std::uint32_t kH = 16;
    constexpr std::uint32_t kI = 8;
    constexpr std::uint32_t kKh = 2;
    constexpr std::uint32_t kKd = 4;
    constexpr std::uint32_t kVh = 2;
    constexpr std::uint32_t kVd = 4;
    constexpr std::uint32_t kConv = 4;
    constexpr std::uint32_t kQk = kKh * kKd;
    constexpr std::uint32_t kVal = kVh * kVd;
    constexpr std::uint32_t kConvF = 2 * kQk + kVal;
    constexpr std::uint32_t kQHeads = 2;
    constexpr std::uint32_t kKvHeads = 1;

    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    auto ar = gpu::GpuArena::create(0, 8ull * 1024ull * 1024ull);
    if (!ar.ok()) {
        std::printf("SKIP: no GPU available (%s)\n", ar.status().message().c_str());
        return 77;
    }
    gpu::GpuArena arena = ar.release();

    bool alloc_ok = true;
    const auto alloc = [&](const std::vector<std::size_t>& shape, const char* name) -> gpu::Tensor {
        auto r = alloc_bf16(arena, shape);
        if (!r.ok()) {
            std::printf("FAIL: alloc %s: %s\n", name, r.status().message().c_str());
            alloc_ok = false;
            return gpu::Tensor{};
        }
        return r.release();
    };

    shelf::qwen35::Qwen35TextConfig cfg;
    cfg.vocab_size = kV;
    cfg.hidden_size = kH;
    cfg.ffn.intermediate_size = kI;
    cfg.num_hidden_layers = 2;
    cfg.linear_num_key_heads = kKh;
    cfg.linear_num_value_heads = kVh;
    cfg.linear_key_head_dim = kKd;
    cfg.linear_value_head_dim = kVd;
    cfg.linear_conv_kernel_dim = kConv;
    cfg.num_attention_heads = kQHeads;
    cfg.num_key_value_heads = kKvHeads;
    cfg.attention_head_dim = kKd;
    cfg.rms_norm_eps = 1e-6f;
    cfg.rope_theta = 1e7f;
    cfg.partial_rotary_factor = 0.25f;
    cfg.tie_word_embeddings = true;
    cfg.layer_types = {0, 1};

    shelf::qwen35::Qwen35ModelWeights w;
    w.lm_head_tied = true;
    w.embed_tokens = make_matrix(alloc({kV, kH}, "embed_tokens"), kV, kH);
    w.final_norm_weight = alloc({kH}, "final_norm");

    shelf::qwen35::Qwen35LayerWeights gdn;
    gdn.is_gdn = true;
    gdn.input_layernorm_weight = alloc({kH}, "gdn input_layernorm");
    gdn.post_attention_layernorm_weight = alloc({kH}, "gdn post_attention_layernorm");
    gdn.ffn.gate = make_matrix(alloc({kI, kH}, "gdn mlp gate"), kI, kH);
    gdn.ffn.up = make_matrix(alloc({kI, kH}, "gdn mlp up"), kI, kH);
    gdn.ffn.down = make_matrix(alloc({kH, kI}, "gdn mlp down"), kH, kI);
    gdn.attn_in_proj_qkv = make_matrix(alloc({kConvF, kH}, "gdn qkv"), kConvF, kH);
    gdn.attn_in_proj_z = make_matrix(alloc({kVal, kH}, "gdn z"), kVal, kH);
    gdn.attn_in_proj_b = make_matrix(alloc({kVh, kH}, "gdn b"), kVh, kH);
    gdn.attn_in_proj_a = make_matrix(alloc({kVh, kH}, "gdn a"), kVh, kH);
    gdn.attn_out_proj = make_matrix(alloc({kH, kVal}, "gdn out"), kH, kVal);
    gdn.attn_conv1d_weight = alloc({kConvF, kConv}, "gdn conv1d");
    gdn.attn_norm_weight = alloc({kVal}, "gdn norm");
    gdn.attn_dt_bias = alloc({kVh}, "gdn dt_bias");
    gdn.attn_A_log = alloc({kVh}, "gdn A_log");
    w.layers.push_back(std::move(gdn));

    shelf::qwen35::Qwen35LayerWeights full;
    full.is_gdn = false;
    full.input_layernorm_weight = alloc({kH}, "attn input_layernorm");
    full.post_attention_layernorm_weight = alloc({kH}, "attn post_attention_layernorm");
    full.ffn.gate = make_matrix(alloc({kI, kH}, "attn mlp gate"), kI, kH);
    full.ffn.up = make_matrix(alloc({kI, kH}, "attn mlp up"), kI, kH);
    full.ffn.down = make_matrix(alloc({kH, kI}, "attn mlp down"), kH, kI);
    full.attn_q_proj = make_matrix(
        alloc({2u * kQHeads * kKd, kH}, "attn q"), 2u * kQHeads * kKd, kH);
    full.attn_k_proj = make_matrix(
        alloc({kKvHeads * kKd, kH}, "attn k"), kKvHeads * kKd, kH);
    full.attn_v_proj = make_matrix(
        alloc({kKvHeads * kKd, kH}, "attn v"), kKvHeads * kKd, kH);
    full.attn_o_proj = make_matrix(
        alloc({kH, kQHeads * kKd}, "attn o"), kH, kQHeads * kKd);
    full.attn_q_norm_weight = alloc({kKd}, "attn q_norm");
    full.attn_k_norm_weight = alloc({kKd}, "attn k_norm");
    w.layers.push_back(std::move(full));

    if (!alloc_ok) {
        arena.shutdown();
        return 1;
    }

    auto full_lower = shelf::qwen35::lower_qwen35_to_primitives(cfg, w);
    if (!full_lower.ok()) {
        std::printf("FAIL: full lowering: %s\n", full_lower.status().message().c_str());
        arena.shutdown();
        return 1;
    }
    check(full_lower.value().graph.external_inputs.size() == 1,
          "full: one external input");
    check(full_lower.value().graph.external_outputs.size() == 4,
          "full: four external outputs");
    check(has_debug_prefix(full_lower.value().graph, "L0."), "full: layer 0");
    check(has_debug_prefix(full_lower.value().graph, "L1."), "full: layer 1");
    check(has_debug_prefix(full_lower.value().graph, "embedding"), "full: embedding");
    check(has_debug_prefix(full_lower.value().graph, "final_norm"), "full: final norm");

    shelf::qwen35::Qwen35LowerOptions stage0_options;
    stage0_options.stage.enabled = true;
    stage0_options.stage.layers = shelf::qwen35::StageRange{0, 1};
    auto stage0 = shelf::qwen35::lower_qwen35_to_primitives(cfg, w, stage0_options);
    if (!stage0.ok()) {
        std::printf("FAIL: stage0 lowering: %s\n", stage0.status().message().c_str());
        arena.shutdown();
        return 1;
    }
    const auto& g0 = stage0.value().graph;
    check(g0.external_inputs.size() == 1, "stage0: one external input");
    check(g0.external_outputs.size() == 1, "stage0: hidden handoff only");
    check(g0.value_specs[g0.external_inputs[0].id].dtype == rt::ValueDType::I32,
          "stage0: token input");
    check(has_debug_prefix(g0, "L0."), "stage0: layer 0");
    check(!has_debug_prefix(g0, "L1."), "stage0: no layer 1");
    check(has_debug_prefix(g0, "embedding"), "stage0: embedding");
    check(!has_debug_prefix(g0, "final_norm"), "stage0: no final norm");
    check(!has_debug_prefix(g0, "logits"), "stage0: no logits");

    shelf::qwen35::Qwen35LowerOptions stage1_options;
    stage1_options.stage.enabled = true;
    stage1_options.stage.layers = shelf::qwen35::StageRange{1, 2};
    auto stage1 = shelf::qwen35::lower_qwen35_to_primitives(cfg, w, stage1_options);
    if (!stage1.ok()) {
        std::printf("FAIL: stage1 lowering: %s\n", stage1.status().message().c_str());
        arena.shutdown();
        return 1;
    }
    const auto& g1 = stage1.value().graph;
    check(g1.external_inputs.size() == 1, "stage1: one external input");
    check(g1.external_outputs.size() == 4, "stage1: four external outputs");
    check(g1.value_specs[g1.external_inputs[0].id].dtype == rt::ValueDType::BF16,
          "stage1: hidden input");
    check(!has_debug_prefix(g1, "L0."), "stage1: no layer 0");
    check(has_debug_prefix(g1, "L1."), "stage1: layer 1");
    check(!has_debug_prefix(g1, "embedding"), "stage1: no embedding");
    check(has_debug_prefix(g1, "final_norm"), "stage1: final norm");
    check(has_debug_prefix(g1, "logits"), "stage1: logits");

    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    if (g_fail == 0) {
        std::printf("QWEN35_PIPELINE_LOWERING: PASS\n");
    }
    arena.shutdown();
    return g_fail == 0 ? 0 : 1;
}
