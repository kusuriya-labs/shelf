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
#include <cstdlib>
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

rt::PrimitiveKind kind_of(const rt::PrimitiveGraphNode& n) noexcept {
    return rt::primitive_kind_of(n.node);
}

bool same_value(const rt::ValueId& a, const rt::ValueId& b) noexcept {
    return a.id == b.id;
}

::ps::Result<gpu::Tensor> alloc_bf16(gpu::GpuArena& arena, const std::vector<std::size_t>& shape) {
    std::size_t count = 1;
    for (std::size_t d : shape) count *= d;
    auto view = arena.allocate_aligned(count * sizeof(std::uint16_t), 128);
    if (!view.ok()) return view.status();
    std::vector<std::size_t> strides(shape.size(), 1);
    for (std::size_t i = 0; i + 1 < shape.size(); ++i)
        strides[i] = strides[i + 1] * shape[i + 1];
    return gpu::Tensor::create<ps::bf16_t>(view.release(), shape, strides);
}

ps::weights::MatrixWeight make_matrix(gpu::Tensor t, std::uint32_t rows, std::uint32_t cols) {
    ps::weights::MatrixWeight w;
    w.bf16 = std::move(t);
    w.rows = rows;
    w.cols = cols;
    return w;
}

} // namespace

int main() {
    constexpr std::uint32_t kV = 8;
    constexpr std::uint32_t kH = 16;
    constexpr std::uint32_t kI = 8;
    constexpr std::uint32_t kQ = 2;
    constexpr std::uint32_t kKV = 1;
    constexpr std::uint32_t kHd = 4;
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
    cfg.num_attention_heads = kQ;
    cfg.num_key_value_heads = kKV;
    cfg.attention_head_dim = kHd;
    cfg.rms_norm_eps = kEps;
    cfg.rope_theta = 1e7f;
    cfg.partial_rotary_factor = 0.25f;
    cfg.tie_word_embeddings = true;

    ps::qwen35::Qwen35ModelWeights w;
    w.lm_head_tied = true;
    w.embed_tokens = make_matrix(alloc({kV, kH}, "embed_tokens"), kV, kH);
    w.mtp.present = true;
    w.mtp.fc = make_matrix(alloc({kH, 2 * kH}, "mtp.fc"), kH, 2 * kH);
    w.mtp.layer.attn_q_proj = make_matrix(alloc({2 * kQ * kHd, kH}, "mtp q_proj"), 2 * kQ * kHd, kH);
    w.mtp.layer.attn_k_proj = make_matrix(alloc({kKV * kHd, kH}, "mtp k_proj"), kKV * kHd, kH);
    w.mtp.layer.attn_v_proj = make_matrix(alloc({kKV * kHd, kH}, "mtp v_proj"), kKV * kHd, kH);
    w.mtp.layer.attn_o_proj = make_matrix(alloc({kH, kQ * kHd}, "mtp o_proj"), kH, kQ * kHd);
    w.mtp.layer.mlp_gate_proj = make_matrix(alloc({kI, kH}, "mtp mlp gate"), kI, kH);
    w.mtp.layer.mlp_up_proj = make_matrix(alloc({kI, kH}, "mtp mlp up"), kI, kH);
    w.mtp.layer.mlp_down_proj = make_matrix(alloc({kH, kI}, "mtp mlp down"), kH, kI);
    w.mtp.pre_fc_norm_embedding_weight = alloc({kH}, "pre_fc_norm_embedding");
    w.mtp.pre_fc_norm_hidden_weight = alloc({kH}, "pre_fc_norm_hidden");
    w.mtp.final_norm_weight = alloc({kH}, "mtp final_norm");
    w.mtp.layer.input_layernorm_weight = alloc({kH}, "mtp input_layernorm");
    w.mtp.layer.post_attention_layernorm_weight = alloc({kH}, "mtp post_attention_layernorm");
    w.mtp.layer.attn_q_norm_weight = alloc({kHd}, "mtp q_norm");
    w.mtp.layer.attn_k_norm_weight = alloc({kHd}, "mtp k_norm");
    if (!alloc_ok) {
        arena.shutdown();
        return 1;
    }

    auto lower = ps::qwen35::lower_qwen35_mtp_to_primitives(cfg, w);
    if (!lower.ok()) {
        std::printf("FAIL: lowering: %s\n", lower.status().message().c_str());
        arena.shutdown();
        return 1;
    }
    const auto& lowered = lower.value();
    const auto& g = lowered.graph;
    const auto& nodes = g.nodes;
    const auto spec_of = [&](rt::ValueId v) -> const rt::PrimitiveValueSpec& {
        return g.value_specs[v.id];
    };
    const auto out = [&](std::size_t i, std::size_t slot = 0) -> rt::ValueId {
        return nodes[i].outputs[slot];
    };

    check(nodes.size() == 30, "node count == 30");
    if (nodes.size() == 30) {
        constexpr rt::PrimitiveKind kExpected[30] = {
            rt::PrimitiveKind::EMBEDDING_LOOKUP, rt::PrimitiveKind::RMS_NORM,
            rt::PrimitiveKind::RMS_NORM, rt::PrimitiveKind::CONCAT,
            rt::PrimitiveKind::LINEAR, rt::PrimitiveKind::RMS_NORM,
            rt::PrimitiveKind::LINEAR, rt::PrimitiveKind::LINEAR,
            rt::PrimitiveKind::LINEAR, rt::PrimitiveKind::SPLIT,
            rt::PrimitiveKind::RMS_NORM, rt::PrimitiveKind::ROPE,
            rt::PrimitiveKind::RMS_NORM, rt::PrimitiveKind::ROPE,
            rt::PrimitiveKind::KV_APPEND, rt::PrimitiveKind::PAGED_ATTENTION,
            rt::PrimitiveKind::SIGMOID, rt::PrimitiveKind::MUL,
            rt::PrimitiveKind::LINEAR, rt::PrimitiveKind::RESIDUAL_ADD,
            rt::PrimitiveKind::RMS_NORM, rt::PrimitiveKind::LINEAR,
            rt::PrimitiveKind::LINEAR, rt::PrimitiveKind::SWIGLU,
            rt::PrimitiveKind::LINEAR, rt::PrimitiveKind::RESIDUAL_ADD,
            rt::PrimitiveKind::OUTPUT_GATHER, rt::PrimitiveKind::RMS_NORM,
            rt::PrimitiveKind::LINEAR, rt::PrimitiveKind::SAMPLING};
        for (std::size_t i = 0; i < 30; ++i) {
            if (kind_of(nodes[i]) != kExpected[i]) {
                std::printf("FAIL: node[%zu] kind=%s expected=%s\n", i,
                            rt::to_string(kind_of(nodes[i])), rt::to_string(kExpected[i]));
                ++g_fail;
            } else {
                ++g_pass;
            }
        }
    }

    const auto check_norm = [&](std::size_t i, std::uint32_t group, std::uint32_t param,
                                rt::ValueId in, const char* name) {
        const auto* n = std::get_if<rt::RmsNormNode>(&nodes[i].node);
        if (n == nullptr) {
            check(false, std::string(name) + ": not a RmsNormNode");
            return;
        }
        check(n->weight_mode == rt::RmsNormWeightMode::ONE_PLUS, std::string(name) + ": ONE_PLUS (Gemma)");
        check(n->group_size == group, std::string(name) + ": group_size");
        check(n->eps == kEps, std::string(name) + ": eps");
        check(n->parameter_index == param, std::string(name) + ": parameter_index");
        check(same_value(nodes[i].inputs[0], in), std::string(name) + ": input");
    };

    {
        const auto* n = std::get_if<rt::EmbeddingLookupNode>(&nodes[0].node);
        check(n != nullptr, "node0: EmbeddingLookupNode");
        if (n != nullptr) {
            check(n->vocab_size == kV, "embed: vocab_size");
            check(n->hidden_size == kH, "embed: hidden_size");
            check(n->weight_index == 0, "embed: weight_index == 0");
            check(n->weight_shape.output_features == kV &&
                      n->weight_shape.input_features == kH,
                  "embed: weight shape");
            check(same_value(nodes[0].inputs[0], g.external_inputs[0]), "embed: input == tokens");
        }
    }
    check_norm(1, kH, 0, out(0), "pre_fc_norm_embedding");
    check_norm(2, kH, 1, g.external_inputs[1], "pre_fc_norm_hidden");
    check(same_value(g.external_inputs[1], nodes[2].inputs[0]), "hidden ext is hnorm input");

    {
        const auto* n = std::get_if<rt::ConcatNode>(&nodes[3].node);
        check(n != nullptr, "node3: ConcatNode");
        if (n != nullptr) {
            check(n->shape.features == 2 * kH, "concat: out features 2H");
            check(n->input_shape_a.features == kH && n->input_shape_b.features == kH,
                  "concat: half shapes");
            check(same_value(nodes[3].inputs[0], out(1)), "concat: [0] == embedding_norm");
            check(same_value(nodes[3].inputs[1], out(2)), "concat: [1] == hidden_norm");
        }
    }

    const auto check_linear = [&](std::size_t i, std::uint32_t weight, std::uint32_t rows,
                                  std::uint32_t cols, rt::ValueId in, rt::ValueDType out_dtype,
                                  const char* name) {
        const auto* n = std::get_if<rt::LinearNode>(&nodes[i].node);
        if (n == nullptr) {
            check(false, std::string(name) + ": not a LinearNode");
            return;
        }
        check(n->weight_index == weight, std::string(name) + ": weight_index");
        check(n->shape.output_features == rows && n->shape.input_features == cols,
              std::string(name) + ": shape");
        check(n->out_dtype == out_dtype, std::string(name) + ": out dtype");
        check(same_value(nodes[i].inputs[0], in), std::string(name) + ": input");
    };
    check_linear(4, 1, kH, 2 * kH, out(3), rt::ValueDType::BF16, "fc");
    check_norm(5, kH, 2, out(4), "input_layernorm");
    check_linear(6, 2, 2 * kQ * kHd, kH, out(5), rt::ValueDType::BF16, "q_proj");
    check_linear(7, 3, kKV * kHd, kH, out(5), rt::ValueDType::BF16, "k_proj");
    check_linear(8, 4, kKV * kHd, kH, out(5), rt::ValueDType::BF16, "v_proj");

    {
        const auto* n = std::get_if<rt::SplitNode>(&nodes[9].node);
        check(n != nullptr, "node9: SplitNode");
        if (n != nullptr) {
            check(n->layout == rt::SplitLayout::INTERLEAVED_HEADS,
                  "split: INTERLEAVED_HEADS (per-head q|gate)");
            check(n->head_dim == kHd, "split: head_dim");
            check(n->split_dim == 1, "split: split_dim");
            check(n->input_shape.features == 2 * kQ * kHd, "split: input shape");
            check(n->output_shape_a.features == kQ * kHd &&
                      n->output_shape_b.features == kQ * kHd,
                  "split: output shapes");
            check(same_value(nodes[9].inputs[0], out(6)), "split: input == q_proj out");
        }
    }
    check_norm(10, kHd, 4, out(9, 0), "q_norm (per-head)");
    check(spec_of(out(10)).dtype == rt::ValueDType::F32, "q_norm: F32 out");
    {
        const auto* n = std::get_if<rt::RoPENode>(&nodes[11].node);
        check(n != nullptr, "node11: RoPENode");
        if (n != nullptr) {
            check(n->rotary_dim == kHd / 4, "q_rope: rotary_dim");
            check(n->head_dim == kHd, "q_rope: head_dim");
            check(n->theta == 1e7f, "q_rope: theta");
            check(same_value(nodes[11].inputs[0], out(10)), "q_rope: input");
        }
    }
    check_norm(12, kHd, 5, out(7), "k_norm (per-head)");
    check(spec_of(out(12)).dtype == rt::ValueDType::F32, "k_norm: F32 out");
    {
        const auto* n = std::get_if<rt::RoPENode>(&nodes[13].node);
        check(n != nullptr, "node13: RoPENode");
        if (n != nullptr) {
            check(n->rotary_dim == kHd / 4, "k_rope: rotary_dim");
            check(same_value(nodes[13].inputs[0], out(12)), "k_rope: input");
        }
    }

    check(same_value(nodes[14].inputs[0], out(13)), "kv_append: k_rope");
    check(same_value(nodes[14].inputs[1], out(8)), "kv_append: v_proj");
    check(nodes[14].outputs.empty(), "kv_append: no value outputs");

    check(same_value(nodes[15].inputs[0], out(11)), "paged_attention: input == q_rope");
    check(spec_of(out(15)).dtype == rt::ValueDType::F32, "paged_attention: F32 out");

    check(same_value(nodes[16].inputs[0], out(9, 1)), "sigmoid: input == split gate half");
    check(spec_of(out(16)).dtype == rt::ValueDType::F32, "sigmoid: F32 out");
    check(same_value(nodes[17].inputs[0], out(15)), "mul: [0] == attention core");
    check(same_value(nodes[17].inputs[1], out(16)), "mul: [1] == sigmoid(gate)");
    check(spec_of(out(17)).dtype == rt::ValueDType::BF16, "mul: BF16 out");

    check_linear(18, 5, kH, kQ * kHd, out(17), rt::ValueDType::BF16, "o_proj (gated)");
    check(same_value(nodes[19].inputs[0], out(18)), "residual1: [0] == o_proj out");
    check(same_value(nodes[19].inputs[1], out(4)), "residual1: [1] == fc out");
    check_norm(20, kH, 3, out(19), "post_attention_layernorm");
    check_linear(21, 6, kI, kH, out(20), rt::ValueDType::BF16, "mlp gate");
    check_linear(22, 7, kI, kH, out(20), rt::ValueDType::BF16, "mlp up");
    check(same_value(nodes[23].inputs[0], out(21)), "swiglu: [0] == gate");
    check(same_value(nodes[23].inputs[1], out(22)), "swiglu: [1] == up");
    check_linear(24, 8, kH, kI, out(23), rt::ValueDType::BF16, "mlp down");
    check(same_value(nodes[25].inputs[0], out(24)), "residual2: [0] == down out");
    check(same_value(nodes[25].inputs[1], out(19)), "residual2: [1] == residual1");

    check(same_value(nodes[26].inputs[0], out(25)), "output_gather: input");
    check(spec_of(out(26)).row_domain == rt::ValueRowDomain::OUTPUT_ROWS,
          "output_gather: OUTPUT_ROWS");
    check_norm(27, kH, 6, out(26), "mtp final_norm");
    check_linear(28, 0, kV, kH, out(27), rt::ValueDType::F32, "lm_head (tied to embed)");
    check(spec_of(out(28)).row_domain == rt::ValueRowDomain::OUTPUT_ROWS, "logits: OUTPUT_ROWS");
    check(same_value(nodes[29].inputs[0], out(28)), "sampling: input == logits");
    check(spec_of(out(29)).dtype == rt::ValueDType::I32, "sampling: I32 out");

    check(g.external_inputs.size() == 2, "external inputs == 2");
    check(g.external_outputs.size() == 4, "external outputs == 4");
    if (g.external_inputs.size() == 2) {
        check(spec_of(g.external_inputs[0]).features == 0, "tokens value: features 0");
        check(spec_of(g.external_inputs[0]).dtype == rt::ValueDType::I32,
              "tokens value: I32");
        check(spec_of(g.external_inputs[1]).features == kH, "hidden value: features H");
        check(spec_of(g.external_inputs[1]).dtype == rt::ValueDType::BF16, "hidden value: BF16");
    }

    check(lowered.weights.size() == 9, "weight table size 9 (tied: no separate lm slot)");
    if (lowered.weights.size() == 9) {
        struct Expect {
            std::uint32_t rows;
            std::uint32_t cols;
        };
        constexpr Expect kTable[9] = {
            {kV, kH}, {kH, 2 * kH}, {2 * kQ * kHd, kH}, {kKV * kHd, kH},
            {kKV * kHd, kH}, {kH, kQ * kHd}, {kI, kH}, {kI, kH}, {kH, kI}};
        for (std::size_t i = 0; i < 9; ++i) {
            check(lowered.weights[i].rows == kTable[i].rows &&
                      lowered.weights[i].cols == kTable[i].cols,
                  "weight table entry shape");
        }
    }
    check(lowered.parameters.size() == 7, "parameter table size 7");
    if (lowered.parameters.size() == 7) {
        constexpr std::uint32_t kParamElements[7] = {kH, kH, kH, kH, kHd, kHd, kH};
        for (std::size_t i = 0; i < 7; ++i) {
            check(lowered.parameters[i].elements == kParamElements[i], "parameter elements");
        }
    }

    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    if (g_fail == 0) {
        std::printf("QWEN35_MTP_LOWERING: PASS\n");
    }
    arena.shutdown();
    return g_fail == 0 ? 0 : 1;
}
