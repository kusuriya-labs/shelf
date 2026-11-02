#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/types.h>
#include <shelf/inference/execution/selectors/paged_attention_selector.h>
#include <shelf/models/qwen4exp/model/lower_to_primitives.h>
#include <shelf/models/qwen4exp/model/qwen4exp_config.h>
#include <shelf/models/qwen4exp/model/qwen4exp_weights.h>
#include <shelf/runtime/graph/primitive_graph.h>
#include <shelf/runtime/graph/primitive_node.h>
#include <shelf/runtime/program/program.h>
#include <shelf/weights/matrix_weight.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace rt = ::ps::runtime;
namespace gpu = ::ps::gpu;
namespace q4e = ::ps::qwen4exp;

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

::ps::weights::MatrixWeight make_matrix(gpu::Tensor t, std::uint32_t rows,
                                        std::uint32_t cols) {
    ::ps::weights::MatrixWeight w;
    w.bf16 = std::move(t);
    w.rows = rows;
    w.cols = cols;
    w.k_padded = cols;
    return w;
}

std::vector<rt::PrimitiveKind> hyper_read_kinds() {
    return {
        rt::PrimitiveKind::RMS_NORM,  rt::PrimitiveKind::LINEAR,
        rt::PrimitiveKind::SCALE,     rt::PrimitiveKind::SILU,
        rt::PrimitiveKind::LINEAR,    rt::PrimitiveKind::SIGMOID,
        rt::PrimitiveKind::MUL,       rt::PrimitiveKind::GROUP_MEAN,
        rt::PrimitiveKind::LINEAR,    rt::PrimitiveKind::SCALE,
        rt::PrimitiveKind::SIGMOID,   rt::PrimitiveKind::SCALE,
    };
}

std::vector<rt::PrimitiveKind> hyper_write_kinds() {
    return {
        rt::PrimitiveKind::GROUP_BROADCAST_MUL,
        rt::PrimitiveKind::RESIDUAL_ADD,
    };
}

std::vector<rt::PrimitiveKind> moe_kinds() {
    return {
        rt::PrimitiveKind::MOE_ROUTER,
        rt::PrimitiveKind::MOE_EXPERTS,
        rt::PrimitiveKind::LINEAR,
        rt::PrimitiveKind::LINEAR,
        rt::PrimitiveKind::SILU,
        rt::PrimitiveKind::MUL,
        rt::PrimitiveKind::LINEAR,
        rt::PrimitiveKind::LINEAR,
        rt::PrimitiveKind::SIGMOID,
        rt::PrimitiveKind::GROUP_BROADCAST_MUL,
        rt::PrimitiveKind::RESIDUAL_ADD,
    };
}

std::vector<rt::PrimitiveKind> collapse_kinds() {
    return {
        rt::PrimitiveKind::RMS_NORM,  rt::PrimitiveKind::LINEAR,
        rt::PrimitiveKind::SCALE,     rt::PrimitiveKind::SILU,
        rt::PrimitiveKind::LINEAR,    rt::PrimitiveKind::SIGMOID,
        rt::PrimitiveKind::MUL,       rt::PrimitiveKind::GROUP_MEAN,
    };
}

void append(std::vector<rt::PrimitiveKind>& dst,
            const std::vector<rt::PrimitiveKind>& src) {
    dst.insert(dst.end(), src.begin(), src.end());
}

const rt::PrimitiveGraphNode* find_by_name(const rt::PrimitiveGraph& g,
                                           const std::string& name) {
    for (const auto& n : g.nodes) {
        if (n.debug_name == name) return &n;
    }
    return nullptr;
}

std::size_t count_kind(const rt::PrimitiveGraph& g, rt::PrimitiveKind k) {
    std::size_t n = 0;
    for (const auto& node : g.nodes) {
        if (rt::primitive_kind_of(node.node) == k) ++n;
    }
    return n;
}

}  // namespace

int main() {
    constexpr std::uint32_t kVocab = 16;
    constexpr std::uint32_t kHidden = 32;
    constexpr std::uint32_t kStreams = 4;
    constexpr std::uint32_t kLowrank = 16;
    constexpr std::uint32_t kHcHidden = kStreams * kHidden;
    constexpr std::uint32_t kLayers = 3;
    constexpr std::uint32_t kKeyHeads = 2;
    constexpr std::uint32_t kValHeads = 4;
    constexpr std::uint32_t kHeadDimQ = 4;
    constexpr std::uint32_t kHeadDimV = 4;
    constexpr std::uint32_t kConvKernel = 4;
    constexpr std::uint32_t kQk = kKeyHeads * kHeadDimQ;
    constexpr std::uint32_t kVal = kValHeads * kHeadDimV;
    constexpr std::uint32_t kConvF = 2 * kQk + kVal;
    constexpr std::uint32_t kQsaHeads = 2;
    constexpr std::uint32_t kQsaKv = 1;
    constexpr std::uint32_t kQsaHeadDim = 16;
    constexpr std::uint32_t kExperts = 4;
    constexpr std::uint32_t kTopK = 2;
    constexpr std::uint32_t kMoeInter = 8;
    constexpr std::uint32_t kSharedInter = 8;
    constexpr float kEps = 1.0e-6f;

    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    auto ar = gpu::GpuArena::create(0, 64ull * 1024ull * 1024ull);
    if (!ar.ok()) {
        std::printf("SKIP: no GPU available (%s)\n", ar.status().message().c_str());
        return 77;
    }
    gpu::GpuArena arena = ar.release();

    bool alloc_ok = true;
    const auto alloc = [&](const std::vector<std::size_t>& shape,
                           const char* name) -> gpu::Tensor {
        auto r = alloc_bf16(arena, shape);
        if (!r.ok()) {
            std::printf("FAIL: alloc %s: %s\n", name, r.status().message().c_str());
            alloc_ok = false;
            return gpu::Tensor{};
        }
        return r.release();
    };
    const auto mat = [&](std::uint32_t rows, std::uint32_t cols,
                         const char* name) -> ::ps::weights::MatrixWeight {
        return make_matrix(alloc({rows, cols}, name), rows, cols);
    };

    q4e::Qwen4ExpTextConfig cfg;
    cfg.vocab_size = kVocab;
    cfg.hidden_size = kHidden;
    cfg.num_hidden_layers = kLayers;
    cfg.num_attention_heads = kQsaHeads;
    cfg.num_key_value_heads = kQsaKv;
    cfg.head_dim = kQsaHeadDim;
    cfg.rotary_dim = 4;
    cfg.full_attention_interval = 0;
    cfg.linear_num_key_heads = kKeyHeads;
    cfg.linear_num_value_heads = kValHeads;
    cfg.linear_key_head_dim = kHeadDimQ;
    cfg.linear_value_head_dim = kHeadDimV;
    cfg.linear_conv_kernel_dim = kConvKernel;
    cfg.num_experts = kExperts;
    cfg.num_experts_per_tok = kTopK;
    cfg.moe_intermediate_size = kMoeInter;
    cfg.shared_expert_intermediate_size = kSharedInter;
    cfg.hc_count = kStreams;
    cfg.hc_lowrank = kLowrank;
    cfg.rms_norm_eps = kEps;
    cfg.rope_theta = 10000000.0f;
    cfg.partial_rotary_factor = 0.25f;
    cfg.tie_word_embeddings = false;
    cfg.output_gate_silu = false;
    cfg.norm_topk_prob = true;
    cfg.layer_types = {0, 0, 1};
    cfg.ple_layer_index = 1;
    cfg.ple_slot = 0;

    q4e::Qwen4ExpModelWeights w;
    w.lm_head_tied = false;
    w.embed_tokens = mat(kVocab, kHidden, "embed_tokens");
    w.lm_head = mat(kVocab, kHidden, "lm_head");

    w.final_hyper.hc_norm = alloc({kHcHidden}, "final hc_norm");
    w.final_hyper.mix_down = mat(kLowrank, kHcHidden, "final mix down");
    w.final_hyper.mix_up = mat(kHcHidden, kLowrank, "final mix up");
    w.final_hyper.has_block_inject = false;

    w.layers.resize(kLayers);
    for (std::uint32_t li = 0; li < kLayers; ++li) {
        q4e::Qwen4ExpLayerWeights& lw = w.layers[li];
        lw.is_gdn = cfg.layer_types[li] == 0;
        for (q4e::Qwen4ExpHyperWeights* hyper : {&lw.attn_hyper, &lw.mlp_hyper}) {
            hyper->hc_norm = alloc({kHcHidden}, "layer hc_norm");
            hyper->mix_down = mat(kLowrank, kHcHidden, "layer mix down");
            hyper->mix_up = mat(kHcHidden, kLowrank, "layer mix up");
            hyper->block_inject = mat(kStreams, kHcHidden, "layer block inject");
            hyper->has_block_inject = true;
        }
        if (lw.is_gdn) {
            lw.gdn.qkv = mat(kConvF, kHidden, "gdn qkv");
            lw.gdn.z = mat(kVal, kHidden, "gdn z");
            lw.gdn.b = mat(kValHeads, kHidden, "gdn b");
            lw.gdn.a = mat(kValHeads, kHidden, "gdn a");
            lw.gdn.out = mat(kHidden, kVal, "gdn out");
            lw.gdn.conv1d = alloc({kConvF, 1, kConvKernel}, "gdn conv1d");
            lw.gdn.norm = alloc({kHeadDimV}, "gdn norm");
            lw.gdn.dt_bias = alloc({kValHeads}, "gdn dt bias");
            lw.gdn.a_log = alloc({kValHeads}, "gdn a log");
        } else {
            lw.qsa.q = mat(2 * kQsaHeads * kQsaHeadDim, kHidden, "qsa q");
            lw.qsa.k = mat(kQsaKv * kQsaHeadDim, kHidden, "qsa k");
            lw.qsa.v = mat(kQsaKv * kQsaHeadDim, kHidden, "qsa v");
            lw.qsa.o = mat(kHidden, kQsaHeads * kQsaHeadDim, "qsa o");
            lw.qsa.q_norm = alloc({kQsaHeadDim}, "qsa q norm");
            lw.qsa.k_norm = alloc({kQsaHeadDim}, "qsa k norm");
        }

        lw.moe.router = mat(kExperts, kHidden, "moe router");
        lw.moe.expert_count = kExperts;
        lw.moe.shared.gate = mat(kSharedInter, kHidden, "shared gate");
        lw.moe.shared.up = mat(kSharedInter, kHidden, "shared up");
        lw.moe.shared.down = mat(kHidden, kSharedInter, "shared down");
        lw.moe.shared.gate_scale = alloc({1, kHidden}, "shared gate scale");

        auto gate_up_fused = alloc({kExperts * 2 * kMoeInter, kHidden},
                                   "expert gate up fused");
        auto down_fused = alloc({kExperts * kHidden, kMoeInter},
                                "expert down fused");
        lw.moe.expert_gate_up.resize(kExperts);
        lw.moe.expert_down.resize(kExperts);
        for (std::uint32_t e = 0; e < kExperts; ++e) {
            auto gs = gate_up_fused.slice(0, static_cast<std::size_t>(e) * 2 * kMoeInter,
                                          2 * kMoeInter);
            auto ds = down_fused.slice(0, static_cast<std::size_t>(e) * kHidden,
                                       kHidden);
            if (!gs.ok() || !ds.ok()) {
                alloc_ok = false;
                break;
            }
            lw.moe.expert_gate_up[e] =
                make_matrix(gs.release(), 2 * kMoeInter, kHidden);
            lw.moe.expert_down[e] =
                make_matrix(ds.release(), kHidden, kMoeInter);
        }

        if (li == cfg.ple_layer_index) {
            lw.ple.present = true;
            lw.ple.key_proj = mat(kHcHidden, kHidden, "ple key proj");
        }
    }

    if (!alloc_ok) {
        arena.shutdown();
        return 1;
    }

    auto lower_full = q4e::lower_qwen4exp_to_primitives(cfg, w);
    check(lower_full.ok(), std::string("connect_moe=true: lowering succeeds") +
                              (lower_full.ok() ? ""
                                               : " (" + lower_full.status().message() +
                                                     ")"));
    if (!lower_full.ok()) {
        arena.shutdown();
        return 1;
    }
    const q4e::Qwen4ExpLoweredPrimitives& lowered = lower_full.value();
    const rt::PrimitiveGraph& g = lowered.graph;

    check(g.external_inputs.size() == 1, "one external input (tokens)");
    if (!g.external_inputs.empty()) {
        const rt::PrimitiveValueSpec* tok = g.value_spec(g.external_inputs[0]);
        check(tok != nullptr && tok->dtype == rt::ValueDType::I32 &&
                  tok->features == 0,
              "token input is I32 with 0 features");
    }
    const rt::PrimitiveGraphNode* embedding = find_by_name(g, "embedding");
    check(embedding != nullptr, "embedding node present");
    if (embedding != nullptr && !g.external_inputs.empty()) {
        check(embedding->inputs.size() == 1 &&
                  embedding->inputs[0] == g.external_inputs[0],
              "embedding consumes token input");
    }

    std::vector<rt::PrimitiveKind> want;
    want.push_back(rt::PrimitiveKind::EMBEDDING_LOOKUP);
    for (std::uint32_t s = 1; s < kStreams; ++s) {
        want.push_back(rt::PrimitiveKind::CONCAT);
    }
    for (std::uint32_t li = 0; li < kLayers; ++li) {
        append(want, hyper_read_kinds());
        if (cfg.layer_types[li] == 0) {
            const std::vector<rt::PrimitiveKind> gdn = {
                rt::PrimitiveKind::LINEAR,
                rt::PrimitiveKind::LINEAR,
                rt::PrimitiveKind::LINEAR,
                rt::PrimitiveKind::LINEAR,
                rt::PrimitiveKind::STATEFUL_CAUSAL_CONV1D,
                rt::PrimitiveKind::SILU,
                rt::PrimitiveKind::L2_NORMALIZE,
                rt::PrimitiveKind::L2_NORMALIZE,
                rt::PrimitiveKind::SCALE,
                rt::PrimitiveKind::GDN_RECURRENCE,
                rt::PrimitiveKind::RMS_NORM,
                rt::PrimitiveKind::SIGMOID,
                rt::PrimitiveKind::MUL,
                rt::PrimitiveKind::LINEAR,
            };
            append(want, gdn);
        } else {
            const std::vector<rt::PrimitiveKind> qsa = {
                rt::PrimitiveKind::LINEAR,
                rt::PrimitiveKind::LINEAR,
                rt::PrimitiveKind::LINEAR,
                rt::PrimitiveKind::SPLIT,
                rt::PrimitiveKind::RMS_NORM,
                rt::PrimitiveKind::ROPE,
                rt::PrimitiveKind::RMS_NORM,
                rt::PrimitiveKind::ROPE,
                rt::PrimitiveKind::KV_APPEND,
                rt::PrimitiveKind::PAGED_ATTENTION,
                rt::PrimitiveKind::SIGMOID,
                rt::PrimitiveKind::MUL,
                rt::PrimitiveKind::LINEAR,
            };
            append(want, qsa);
        }
        append(want, hyper_write_kinds());
        append(want, hyper_read_kinds());
        append(want, moe_kinds());
        append(want, hyper_write_kinds());
    }
    append(want, collapse_kinds());
    want.push_back(rt::PrimitiveKind::OUTPUT_GATHER);
    want.push_back(rt::PrimitiveKind::LINEAR);
    want.push_back(rt::PrimitiveKind::SAMPLING);

    const std::vector<rt::PrimitiveKind> got = g.kind_sequence();
    bool kinds_ok = got.size() == want.size();
    if (kinds_ok) {
        for (std::size_t i = 0; i < want.size(); ++i) {
            if (got[i] != want[i]) {
                kinds_ok = false;
                std::printf("  kind mismatch at %zu: got %s want %s\n", i,
                            rt::to_string(got[i]), rt::to_string(want[i]));
                break;
            }
        }
    } else {
        std::printf("  kind count: got %zu want %zu\n", got.size(), want.size());
    }
    check(kinds_ok, "node kind sequence matches the layer contract");

    const std::size_t gdn_layers = 2;
    const std::size_t qsa_layers = 1;
    const std::size_t want_weights =
        1u + gdn_layers * (3u + 3u + 5u + (1u + 3u + 1u + kExperts * 2u)) +
        qsa_layers * (3u + 3u + 4u + (1u + 3u + 1u + kExperts * 2u)) + 1u + 2u;
    check(lowered.weights.size() == want_weights,
          "weight slot count matches (PLE excluded, experts split per rank-2)");
    check(g.weight_table.size() == lowered.weights.size(),
          "weight table matches weight slots");

    const std::size_t want_params =
        kLayers * (1u + 1u) + gdn_layers * 4u + qsa_layers * 2u + 1u;
    check(lowered.parameters.size() == want_params,
          "static parameter slot count matches");

    check(g.states.size() == gdn_layers * 2u + qsa_layers,
          "state count = conv + recurrent per GDN layer + KV per QSA layer");
    std::size_t conv_states = 0;
    std::size_t rec_states = 0;
    std::size_t kv_states = 0;
    bool state_indices_ok = true;
    for (const auto& s : g.states) {
        if (s.kind == rt::PrimitiveStateKind::GDN_CONV_STATE) ++conv_states;
        if (s.kind == rt::PrimitiveStateKind::GDN_RECURRENCE_STATE) ++rec_states;
        if (s.kind == rt::PrimitiveStateKind::KV_CACHE) ++kv_states;
        if (s.kind == rt::PrimitiveStateKind::KV_CACHE && s.state_index != 0u)
            state_indices_ok = false;
    }
    check(conv_states == gdn_layers, "one GDN conv state per GDN layer");
    check(rec_states == gdn_layers, "one GDN recurrent state per GDN layer");
    check(kv_states == qsa_layers, "one KV state per QSA layer");
    check(state_indices_ok, "KV state index is dense from zero");

    check(find_by_name(g, "L0.attn.hyper_hc_norm") != nullptr,
          "attention hyper read debug name");
    check(find_by_name(g, "L0.attn.gdn_qkv") != nullptr, "GDN debug name");
    check(find_by_name(g, "L1.attn.gdn_recurrence") != nullptr,
          "GDN recurrence debug name");
    check(find_by_name(g, "L2.attn.q_proj") != nullptr, "QSA debug name");
    check(find_by_name(g, "L2.attn.kv_append") != nullptr, "QSA kv append debug name");
    check(find_by_name(g, "L0.mlp.moe_router") != nullptr, "MoE router debug name");
    check(find_by_name(g, "L0.mlp.moe_experts") != nullptr, "MoE experts debug name");
    check(find_by_name(g, "L0.mlp.shared_gate_mul") != nullptr,
          "shared expert gate debug name");
    check(find_by_name(g, "final.hyper_mixed") != nullptr, "final collapse debug name");
    check(find_by_name(g, "selected_hidden") != nullptr, "selected hidden debug name");
    check(find_by_name(g, "logits") != nullptr, "logits debug name");
    check(find_by_name(g, "sampled") != nullptr, "sampled debug name");
    check(find_by_name(g, "hc_expand_0") != nullptr &&
              find_by_name(g, "hc_expand_2") != nullptr,
          "embedding repeat concat debug names");

    bool ple_absent = true;
    for (const auto& n : g.nodes) {
        if (n.debug_name.find(".ple.") != std::string::npos) ple_absent = false;
    }
    check(ple_absent, "PLE is not present in the graph");

    const rt::PrimitiveGraphNode* moe_router = find_by_name(g, "L0.mlp.moe_router");
    check(moe_router != nullptr, "L0 MoE router node present");
    if (moe_router != nullptr) {
        const auto& n = std::get<rt::MoERouterNode>(moe_router->node);
        check(n.experts == kExperts && n.top_k == kTopK && n.renorm,
              "router geometry experts/top_k/renorm");
        check(g.weight_table[n.weight_index] ==
                  rt::MatrixwiseShapeKey{kExperts, kHidden},
              "router weight geometry [E, hidden]");
        check(moe_router->inputs.size() == 1 && moe_router->outputs.size() == 2,
              "router tensor arity 1 -> 2");
    }

    const rt::PrimitiveGraphNode* moe_experts = find_by_name(g, "L0.mlp.moe_experts");
    check(moe_experts != nullptr, "L0 MoE experts node present");
    if (moe_experts != nullptr) {
        const auto& n = std::get<rt::MoEExpertsNode>(moe_experts->node);
        check(n.experts == kExperts && n.top_k == kTopK, "experts geometry");
        check(moe_experts->inputs.size() == 3 && moe_experts->outputs.size() == 1,
              "experts tensor arity 3 -> 1");
        bool slots_ok = true;
        for (std::uint32_t e = 0; e < kExperts; ++e) {
            if (g.weight_table[n.weight_index + e] !=
                rt::MatrixwiseShapeKey{2 * kMoeInter, kHidden}) {
                slots_ok = false;
            }
            if (g.weight_table[n.weight_index + kExperts + e] !=
                rt::MatrixwiseShapeKey{kHidden, kMoeInter}) {
                slots_ok = false;
            }
        }
        check(slots_ok,
              "expert slots contiguous: gate_up[E] then down[E] at weight_index");
    }

    const rt::PrimitiveGraphNode* logits = find_by_name(g, "logits");
    const rt::PrimitiveGraphNode* gather = find_by_name(g, "selected_hidden");
    const rt::PrimitiveGraphNode* collapse = find_by_name(g, "final.hyper_mixed");
    check(logits != nullptr && gather != nullptr && collapse != nullptr,
          "final chain nodes present");
    bool final_chain_ok = false;
    bool final_norm_absent = false;
    if (logits != nullptr && gather != nullptr && collapse != nullptr) {
        final_chain_ok = logits->inputs.size() == 1 &&
                         gather->outputs.size() == 1 &&
                         logits->inputs[0] == gather->outputs[0] &&
                         gather->inputs.size() == 1 &&
                         collapse->outputs.size() == 1 &&
                         gather->inputs[0] == collapse->outputs[0];
        const auto& lm = std::get<rt::LinearNode>(logits->node);
        check(lm.shape == rt::MatrixwiseShapeKey{kVocab, kHidden},
              "lm_head geometry [vocab, hidden]");
        final_norm_absent = true;
        bool after_collapse = false;
        for (const auto& n : g.nodes) {
            if (&n == collapse) after_collapse = true;
            if (after_collapse &&
                rt::primitive_kind_of(n.node) == rt::PrimitiveKind::RMS_NORM) {
                final_norm_absent = false;
            }
        }
    }
    check(final_chain_ok,
          "final norm absent: collapse feeds gather feeds lm_head directly");
    check(final_norm_absent, "no RMS_NORM between final collapse and lm_head");

    std::size_t rope_checked = 0;
    for (const auto& n : g.nodes) {
        if (rt::primitive_kind_of(n.node) != rt::PrimitiveKind::ROPE) continue;
        const auto& r = std::get<rt::RoPENode>(n.node);
        check(r.rotary_dim == 4 && r.theta == 10000000.0f, "rope rotary_dim/theta");
        ++rope_checked;
    }
    check(rope_checked == 2, "two RoPE nodes on the single QSA layer");

    rt::ProgramBuildOptions build_options;
    build_options.validate_workspace = true;
    build_options.max_token_rows = 8;
    build_options.max_output_rows = 2;
    const rt::WeightTableView weight_view{lowered.weights.data(),
                                           lowered.weights.size()};
    auto built = rt::build_program(g, weight_view, rt::RowBucket::R16,
                                   rt::ExecutionClass::PREFILL, build_options);
    check(built.ok(), std::string("build_program with MoE dispatches") +
                          (built.ok() ? ""
                                      : " (" + built.status().message() + ")"));
    if (built.ok()) {
        const rt::Program plan = built.value();
        bool router_dispatch = false;
        bool experts_dispatch = false;
        for (const auto& d : plan.dispatches) {
            if (d.kernel_id == rt::KernelId::MOE_ROUTER) router_dispatch = true;
            if (d.kernel_id == rt::KernelId::MOE_EXPERTS) experts_dispatch = true;
        }
        check(router_dispatch, "MOE_ROUTER dispatch emitted");
        check(experts_dispatch, "MOE_EXPERTS dispatch emitted");
        check(plan.validate().ok(), "program validates");
    }

    {
        q4e::Qwen4ExpModelWeights pw = w;
        const gpu::Tensor codes = alloc({1024}, "psq5 codes");
        const gpu::Tensor signs = alloc({256}, "psq5 sign");
        const gpu::Tensor scales = alloc({64}, "psq5 scales");
        for (q4e::Qwen4ExpLayerWeights& lw : pw.layers) {
            for (auto& mw : lw.moe.expert_gate_up) {
                mw.encoding = ::ps::weights::MatrixEncoding::Psq5;
                ::ps::weights::apply_default_compute_spec(mw);
                mw.codes = codes;
                mw.sign = signs;
                mw.scales = scales;
                mw.k_padded = mw.cols;
                mw.codes_row_stride_bytes = 256;
                mw.scale_row_stride_bytes = 256;
                mw.storage_scale_stride_bytes = 256;
                mw.weight_scale_group = 32;
                mw.preshuffled = true;
            }
            for (auto& mw : lw.moe.expert_down) {
                mw.encoding = ::ps::weights::MatrixEncoding::Psq5;
                ::ps::weights::apply_default_compute_spec(mw);
                mw.codes = codes;
                mw.sign = signs;
                mw.scales = scales;
                mw.k_padded = mw.cols;
                mw.codes_row_stride_bytes = 256;
                mw.scale_row_stride_bytes = 256;
                mw.storage_scale_stride_bytes = 256;
                mw.weight_scale_group = 32;
                mw.preshuffled = true;
            }
        }
        auto lower_psq5 = q4e::lower_qwen4exp_to_primitives(cfg, pw);
        check(lower_psq5.ok(),
              std::string("psq5 expert lowering succeeds") +
                  (lower_psq5.ok() ? "" : " (" + lower_psq5.status().message() + ")"));
        if (lower_psq5.ok()) {
            const q4e::Qwen4ExpLoweredPrimitives& lp = lower_psq5.value();
            const rt::PrimitiveGraphNode* lp_experts =
                find_by_name(lp.graph, "L0.mlp.moe_experts");
            check(lp_experts != nullptr, "psq5: MoE experts node present");
            if (lp_experts != nullptr) {
                const auto& n = std::get<rt::MoEExpertsNode>(lp_experts->node);
                const rt::WeightSlot& gu = lp.weights[n.weight_index];
                const rt::WeightSlot& dn = lp.weights[n.weight_index + kExperts];
                const std::uint8_t psq5_enc =
                    static_cast<std::uint8_t>(::ps::weights::MatrixEncoding::Psq5);
                check(gu.encoding == psq5_enc && dn.encoding == psq5_enc,
                      "psq5 slots carry encoding 6");
                check(gu.device_ptr == nullptr && dn.device_ptr == nullptr,
                      "psq5 slots have no bf16 device pointer");
                check(gu.codes != nullptr && gu.sign != nullptr &&
                          gu.scales != nullptr,
                      "psq5 gate_up slot keeps magnitude/sign/scale pointers");
                check(dn.codes != nullptr && dn.sign != nullptr &&
                          dn.scales != nullptr,
                      "psq5 down slot keeps magnitude/sign/scale pointers");
                check(gu.preshuffled && dn.preshuffled, "psq5 slots preshuffled");
                check(gu.compute_spec ==
                          static_cast<std::uint8_t>(
                              ::ps::quantization::ComputeSpecId::PSQ5_W8A8_F32_BF16),
                      "psq5 slot compute spec");
                check(gu.rows == 2 * kMoeInter && gu.cols == kHidden &&
                          dn.rows == kHidden && dn.cols == kMoeInter,
                      "psq5 slot geometry");
            }
        }
    }

    q4e::Qwen4ExpLowerOptions no_moe;
    no_moe.connect_moe = false;
    const std::vector<std::uint32_t> tap_layers{1};
    const std::vector<std::uint32_t> override_layers{0};
    no_moe.hidden_taps = tap_layers;
    no_moe.override_layers = override_layers;
    auto lower_partial = q4e::lower_qwen4exp_to_primitives(cfg, w, no_moe);
    check(lower_partial.ok(),
          std::string("connect_moe=false: lowering succeeds") +
              (lower_partial.ok()
                   ? ""
                   : " (" + lower_partial.status().message() + ")"));
    if (lower_partial.ok()) {
        const q4e::Qwen4ExpLoweredPrimitives& part = lower_partial.value();
        const rt::PrimitiveGraph& pg = part.graph;
        check(count_kind(pg, rt::PrimitiveKind::MOE_ROUTER) == 0 &&
                  count_kind(pg, rt::PrimitiveKind::MOE_EXPERTS) == 0,
              "connect_moe=false: no MoE nodes");
        check(find_by_name(pg, "L0.mlp.hyper_hc_norm") == nullptr,
              "connect_moe=false: mlp sub-layer omitted");
        const std::size_t part_weights =
            1u + gdn_layers * (3u + 5u) + qsa_layers * (3u + 4u) + 1u + 2u;
        check(part.weights.size() == part_weights,
              "connect_moe=false: weight slot count");
        check(pg.external_outputs.size() == 5,
              "connect_moe=false: 4 base external outputs + 1 hidden tap");
        check(pg.external_inputs.size() == 1, "connect_moe=false: one external input");
        check(part.layer_input_values.size() == 1 &&
                  part.layer_input_values[0].first == 0u,
              "override layer recorded");
        if (!pg.external_outputs.empty() && pg.nodes.size() > 3) {
            const rt::PrimitiveValueSpec* tap =
                pg.value_spec(pg.external_outputs[4]);
            check(tap != nullptr && tap->features == kHcHidden &&
                      tap->row_domain == rt::ValueRowDomain::TOKEN_ROWS,
                  "hidden tap exposes 4-stream layer output");
        }
        bool ple_absent_partial = true;
        for (const auto& n : pg.nodes) {
            if (n.debug_name.find(".ple.") != std::string::npos)
                ple_absent_partial = false;
        }
        check(ple_absent_partial, "connect_moe=false: PLE absent");
        auto built_partial = rt::build_program(pg,
                                               rt::WeightTableView{
                                                   part.weights.data(),
                                                   part.weights.size()},
                                               rt::RowBucket::R16,
                                               rt::ExecutionClass::DECODE,
                                               build_options);
        check(built_partial.ok(),
              std::string("connect_moe=false: build_program") +
                  (built_partial.ok() ? ""
                                      : " (" + built_partial.status().message() +
                                            ")"));
    }

    const auto selector_input = [](std::uint32_t q, std::uint32_t kv,
                                   std::uint32_t max_visible) {
        ::ps::execution::PagedAttentionSelectorInput in;
        in.kv_dtype = ::ps::kernel::KVCacheDType::BF16;
        in.output_dtype = rt::ValueDType::F32;
        in.rows = 16;
        in.q_heads = q;
        in.kv_heads = kv;
        in.head_dim = 256;
        in.page_tokens = 16;
        in.max_visible_tokens = max_visible;
        return in;
    };
    check(::ps::execution::select_paged_attention_implementation(
              selector_input(24, 2, 262144)) ==
              ::ps::execution::PagedAttentionImplementation::Correctness,
          "QSA (24,2,256) selects correctness fallback (no optimized rule)");
    check(::ps::execution::select_paged_attention_implementation(
              selector_input(16, 4, 8192)) ==
              ::ps::execution::PagedAttentionImplementation::Optimized,
          "control: (16,4,256) selects optimized path");

    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    if (g_fail == 0) {
        std::printf("QWEN4EXP_LOWERING_CONTRACT: PASS\n");
    }
    const ps::Status shutdown = arena.shutdown();
    if (!shutdown.ok()) {
        std::printf("FAIL: arena shutdown\n");
        return 1;
    }
    return g_fail == 0 ? 0 : 1;
}
