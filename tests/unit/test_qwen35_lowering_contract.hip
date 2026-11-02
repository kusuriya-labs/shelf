#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/model/lower_to_primitives.h>
#include <phaseshift/models/qwen35/model/qwen35_config.h>
#include <phaseshift/models/qwen35/weights/model_weights.h>
#include <phaseshift/runtime/graph/primitive_graph.h>
#include <phaseshift/runtime/graph/primitive_node.h>
#include <phaseshift/weights/matrix_weight.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace rt = ::ps::runtime;
namespace gpu = ::ps::gpu;

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

::ps::Result<gpu::Tensor> alloc_bf16(gpu::GpuArena& arena,
                                     const std::vector<std::size_t>& shape) {
    std::size_t count = 1;
    for (std::size_t d : shape) count *= d;
    auto view = arena.allocate_aligned(count * sizeof(std::uint16_t), 128);
    if (!view.ok()) return view.status();
    std::vector<std::size_t> strides(shape.size(), 1);
    for (std::size_t i = 0; i + 1 < shape.size(); ++i)
        strides[i] = strides[i + 1] * shape[i + 1];
    return gpu::Tensor::create<ps::bf16_t>(view.release(), shape, strides);
}

ps::weights::MatrixWeight make_matrix(gpu::Tensor t, std::uint32_t rows,
                                      std::uint32_t cols) {
    ps::weights::MatrixWeight w;
    w.bf16 = std::move(t);
    w.rows = rows;
    w.cols = cols;
    return w;
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
    constexpr float kEps = 1e-6f;

    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    auto ar = gpu::GpuArena::create(0, 4ull * 1024ull * 1024ull);
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

    ps::qwen35::Qwen35TextConfig cfg;
    cfg.vocab_size = kV;
    cfg.hidden_size = kH;
    cfg.intermediate_size = kI;
    cfg.num_hidden_layers = 1;
    cfg.linear_num_key_heads = kKh;
    cfg.linear_num_value_heads = kVh;
    cfg.linear_key_head_dim = kKd;
    cfg.linear_value_head_dim = kVd;
    cfg.linear_conv_kernel_dim = kConv;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 1;
    cfg.attention_head_dim = kKd;
    cfg.rms_norm_eps = kEps;
    cfg.rope_theta = 1e7f;
    cfg.partial_rotary_factor = 0.25f;
    cfg.tie_word_embeddings = true;

    ps::qwen35::Qwen35ModelWeights w;
    w.lm_head_tied = true;
    w.embed_tokens = make_matrix(alloc({kV, kH}, "embed_tokens"), kV, kH);
    w.final_norm_weight = alloc({kH}, "final_norm");

    ps::qwen35::Qwen35LayerWeights gdn;
    gdn.is_gdn = true;
    gdn.input_layernorm_weight = alloc({kH}, "gdn input_layernorm");
    gdn.post_attention_layernorm_weight = alloc({kH}, "gdn post_attention_layernorm");
    gdn.mlp_gate_proj = make_matrix(alloc({kI, kH}, "gdn mlp gate"), kI, kH);
    gdn.mlp_up_proj = make_matrix(alloc({kI, kH}, "gdn mlp up"), kI, kH);
    gdn.mlp_down_proj = make_matrix(alloc({kH, kI}, "gdn mlp down"), kH, kI);
    gdn.attn_in_proj_qkv =
        make_matrix(alloc({kConvF, kH}, "gdn qkv"), kConvF, kH);
    gdn.attn_in_proj_z = make_matrix(alloc({kVal, kH}, "gdn z"), kVal, kH);
    gdn.attn_in_proj_b = make_matrix(alloc({kVh, kH}, "gdn b"), kVh, kH);
    gdn.attn_in_proj_a = make_matrix(alloc({kVh, kH}, "gdn a"), kVh, kH);
    gdn.attn_out_proj = make_matrix(alloc({kH, kVal}, "gdn out"), kH, kVal);
    gdn.attn_conv1d_weight = alloc({kConvF, kConv}, "gdn conv1d");
    gdn.attn_norm_weight = alloc({kVal}, "gdn norm");
    gdn.attn_dt_bias = alloc({kVh}, "gdn dt_bias");
    gdn.attn_A_log = alloc({kVh}, "gdn A_log");
    w.layers.push_back(std::move(gdn));
    cfg.layer_types = {0};

    if (!alloc_ok) {
        arena.shutdown();
        return 1;
    }

    auto lower = ps::qwen35::lower_qwen35_to_primitives(cfg, w);
    if (!lower.ok()) {
        std::printf("FAIL: main lowering: %s\n", lower.status().message().c_str());
        arena.shutdown();
        return 1;
    }
    const auto& g = lower.value().graph;
    check(g.external_inputs.size() >= 1, "main: external input present");
    if (!g.external_inputs.empty()) {
        const auto& tok_spec = g.value_specs[g.external_inputs[0].id];
        check(tok_spec.dtype == rt::ValueDType::I32,
              "main: token external value is I32");
        check(tok_spec.features == 0, "main: token external value features 0");
        if (!g.nodes.empty()) {
            check(g.nodes[0].inputs.size() == 1 &&
                      g.nodes[0].inputs[0].id == g.external_inputs[0].id,
                  "main: embedding consumes the token external value");
        }
    }

    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    if (g_fail == 0) {
        std::printf("QWEN35_LOWERING_CONTRACT: PASS\n");
    }
    arena.shutdown();
    return g_fail == 0 ? 0 : 1;
}
