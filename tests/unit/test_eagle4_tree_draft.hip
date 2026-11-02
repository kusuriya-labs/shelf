#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/io/safetensors_writer.h>
#include <phaseshift/models/qwen35/eagle4/config.h>
#include <phaseshift/models/qwen35/eagle4/context_state.h>
#include <phaseshift/models/qwen35/eagle4/executor.h>
#include <phaseshift/models/qwen35/eagle4/weights.h>
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

using ps::bf16_t;
using ps::f32_to_bf16_rne;

static constexpr uint32_t kHeads = 2u;
static constexpr uint32_t kHeadDim = 128u;
static constexpr uint32_t kHidden = kHeads * kHeadDim;
static constexpr uint32_t kIntermediate = 64u;
static constexpr uint32_t kVocab = 64u;
static constexpr uint32_t kRank = 8u;
static constexpr uint32_t kCandidates = 16u;
static constexpr uint32_t kTaps = 3u;
static constexpr uint32_t kTapFeatures = 256u;
static constexpr uint32_t kExtendRows = 2u;
static constexpr uint32_t kPosRoot = kExtendRows;

static int g_fail = 0;

static void fail(const std::string& msg) {
    ++g_fail;
    std::printf("FAIL %s\n", msg.c_str());
}

static void check(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

static float bf16_to_f32(bf16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value.data) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static bf16_t to_bf16(float value) { return bf16_t{f32_to_bf16_rne(value)}; }

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

static HostModel make_host_model() {
    uint32_t seed = 12345u;
    HostModel m;
    const float scale = 0.08f;
    m.fc = make_matrix(kHidden, kTaps * kTapFeatures, scale, seed);
    for (uint32_t t = 0; t < 3u; ++t) m.fc_norm[t] = make_norm(kTapFeatures, seed);
    m.final_norm = make_norm(kHidden, seed);
    m.input_norm = make_norm(kHidden, seed);
    m.hidden_norm = make_norm(kHidden, seed);
    m.post_norm = make_norm(kHidden, seed);
    m.q = make_matrix(kHidden, 2u * kHidden, scale, seed);
    m.k = make_matrix(kHidden, 2u * kHidden, scale, seed);
    m.v = make_matrix(kHidden, 2u * kHidden, scale, seed);
    m.o = make_matrix(kHidden, kHidden, scale, seed);
    m.gate = make_matrix(kIntermediate, kHidden, scale, seed);
    m.up = make_matrix(kIntermediate, kHidden, scale, seed);
    m.down = make_matrix(kHidden, kIntermediate, scale, seed);
    m.embed = make_matrix(kVocab, kHidden, 0.2f, seed);
    m.lm_head = make_matrix(kVocab, kHidden, 0.2f, seed);
    m.efficient_head = make_matrix(kVocab, kRank, 0.3f, seed);
    m.efficient_down = make_matrix(kRank, kHidden, 0.3f, seed);
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
        fail("writer create: " + created.status().message());
        return false;
    }
    ps::io::SafetensorsWriter writer = created.release();
    const auto expected = ps::qwen35::eagle4::eagle4_expected_tensors(config);
    for (const auto& e : expected) {
        if (!writer.plan_tensor(e.name, ps::io::SType::BF16, e.shape).ok()) {
            fail("plan " + e.name);
            return false;
        }
    }
    if (!writer.write_header().ok()) {
        fail("write_header");
        return false;
    }
    for (const auto& e : expected) {
        std::size_t count = 1;
        for (std::size_t d : e.shape) count *= d;
        std::vector<bf16_t> data;
        if (e.shape.size() == 2) {
            data = find(model, e.name).data;
        } else {
            data = find_vector(model, e.name);
        }
        if (data.size() != count) {
            fail("tensor size mismatch: " + e.name);
            return false;
        }
        if (!writer.write_tensor(e.name, data.data(), data.size() * sizeof(bf16_t)).ok()) {
            fail("write " + e.name);
            return false;
        }
    }
    return writer.finish().ok();
}

template <typename T>
static bool upload(ps::gpu::GpuArena& arena, ps::gpu::Tensor& out, const T* src,
                   const std::vector<std::size_t>& shape) {
    std::size_t count = 1;
    for (std::size_t d : shape) count *= d;
    auto view = arena.allocate_aligned(count * sizeof(T), 256u);
    if (!view.ok()) {
        fail("alloc");
        return false;
    }
    std::vector<std::size_t> strides(shape.size());
    std::size_t stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }
    auto created = ps::gpu::Tensor::create<T>(view.release(), shape, strides);
    if (!created.ok()) {
        fail("tensor create");
        return false;
    }
    out = created.release();
    if (hipMemcpy(out.data<T>(), src, count * sizeof(T), hipMemcpyHostToDevice) !=
        hipSuccess) {
        fail("upload copy");
        return false;
    }
    return true;
}

int main() {
    const fs::path root = fs::temp_directory_path() /
                          ("ps_eagle4_tree_draft_" + std::to_string(static_cast<long>(getpid())));
    fs::remove_all(root);
    fs::create_directories(root);
    {
        std::ofstream f(root / "config.json", std::ios::binary | std::ios::trunc);
        f << R"json({
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
    }
    auto config_result = ps::qwen35::eagle4::read_eagle4_config(root.string());
    check(config_result.ok(), "config reads");
    if (!config_result.ok()) return 1;
    const ps::qwen35::eagle4::Eagle4Config config = config_result.release();

    const HostModel host = make_host_model();
    check(write_shard(root.string(), config, host), "checkpoint writes");

    auto arena_result = ps::gpu::GpuArena::create(0, 256ull * 1024ull * 1024ull);
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

    auto weights_result =
        ps::qwen35::eagle4::load_eagle4_weights(root.string(), config, arena, stream);
    check(weights_result.ok(), "weights load");
    if (!weights_result.ok()) return 1;
    ps::qwen35::eagle4::Eagle4Weights weights = weights_result.release();

    ps::qwen35::eagle4::Eagle4ExecutorConfig exec_config;
    exec_config.num_candidates = kCandidates;
    auto executor_result =
        ps::qwen35::eagle4::create_eagle4_executor(config, weights, arena, exec_config);
    check(executor_result.ok(), "executor create");
    if (!executor_result.ok()) return 1;
    ps::qwen35::eagle4::Eagle4Executor executor = executor_result.release();

    auto context_result =
        ps::qwen35::eagle4::create_eagle4_context_state(config, arena, 0u);
    check(context_result.ok(), "context create");
    if (!context_result.ok()) return 1;
    ps::qwen35::eagle4::Eagle4ContextState context = context_result.release();

    uint32_t seed = 777u;
    std::vector<bf16_t> taps(static_cast<std::size_t>(kTaps) * kExtendRows * kTapFeatures);
    for (bf16_t& value : taps) value = random_bf16(seed, 1.0f);
    const std::vector<int32_t> host_tokens = {3, 5};

    ps::gpu::Tensor tap_tensor[kTaps];
    for (uint32_t t = 0; t < kTaps; ++t) {
        const std::size_t offset =
            static_cast<std::size_t>(t) * kExtendRows * kTapFeatures;
        check(upload<bf16_t>(arena, tap_tensor[t], taps.data() + offset,
                             {kExtendRows, kTapFeatures}),
              "tap upload");
    }
    ps::gpu::Tensor token_tensor;
    check(upload<int32_t>(arena, token_tensor, host_tokens.data(), {kExtendRows}),
          "token upload");
    ps::gpu::Tensor root_tokens;
    check(upload<int32_t>(arena, root_tokens, std::vector<int32_t>(kTaps, 0).data(), {kTaps}),
          "root upload");

    const bf16_t* tap_ptrs[kTaps] = {tap_tensor[0].data<bf16_t>(),
                                     tap_tensor[1].data<bf16_t>(),
                                     tap_tensor[2].data<bf16_t>()};
    auto do_extend = [&]() -> bool {
        auto st = ps::qwen35::eagle4::eagle4_extend_rows(
            executor, context, tap_ptrs, kTaps, token_tensor.data<int32_t>(), kExtendRows,
            0u, kTaps, root_tokens.data<int32_t>(), stream);
        if (!st.ok()) {
            fail("extend: " + st.message());
            return false;
        }
        return hipStreamSynchronize(stream) == hipSuccess;
    };

    check(do_extend(), "extend arm a");
    std::vector<int32_t> root_host(kTaps, -1);
    if (hipMemcpy(root_host.data(), root_tokens.data<int32_t>(), kTaps * sizeof(int32_t),
                  hipMemcpyDeviceToHost) != hipSuccess) {
        fail("root download");
        return 1;
    }
    check(root_host[0] >= 0 && root_host[0] < static_cast<int32_t>(kVocab),
          "root token valid");

    ps::gpu::Tensor chain_drafts;
    check(upload<int32_t>(arena, chain_drafts, std::vector<int32_t>(4, -1).data(), {4}),
          "chain drafts upload");
    if (hipMemcpy(chain_drafts.data<int32_t>(), &root_host[0], sizeof(int32_t),
                  hipMemcpyHostToDevice) != hipSuccess) {
        fail("chain root preload");
        return 1;
    }
    auto chain_st = ps::qwen35::eagle4::eagle4_draft_steps(
        executor, context, kPosRoot, 4u, chain_drafts.data<int32_t>(), stream);
    check(chain_st.ok(), "chain draft steps");
    if (hipStreamSynchronize(stream) != hipSuccess) {
        fail("chain sync");
        return 1;
    }
    std::vector<int32_t> chain_tokens(4, -1);
    if (hipMemcpy(chain_tokens.data(), chain_drafts.data<int32_t>(),
                  4 * sizeof(int32_t), hipMemcpyDeviceToHost) != hipSuccess) {
        fail("chain tokens download");
        return 1;
    }
    std::vector<bf16_t> chain_normed(kHidden);
    if (hipMemcpy(chain_normed.data(), executor.normed.data<bf16_t>(),
                  kHidden * sizeof(bf16_t), hipMemcpyDeviceToHost) != hipSuccess) {
        fail("chain normed download");
        return 1;
    }

    check(do_extend(), "extend arm b");

    ps::gpu::Tensor step_tokens;
    ps::gpu::Tensor row_nodes;
    ps::gpu::Tensor src_rows;
    ps::gpu::Tensor anc_offsets;
    ps::gpu::Tensor anc_nodes;
    check(upload<int32_t>(arena, step_tokens, std::vector<int32_t>(2, -1).data(), {2}),
          "step tokens upload");
    check(upload<int32_t>(arena, row_nodes, std::vector<int32_t>(2, -1).data(), {2}),
          "row nodes upload");
    check(upload<int32_t>(arena, src_rows, std::vector<int32_t>(2, -1).data(), {2}),
          "src rows upload");
    check(upload<int32_t>(arena, anc_offsets, std::vector<int32_t>(4, 0).data(), {4}),
          "anc offsets upload");
    check(upload<int32_t>(arena, anc_nodes, std::vector<int32_t>(16, -1).data(), {16}),
          "anc nodes upload");

    auto upload_i32 = [&](ps::gpu::Tensor& tensor, const std::vector<int32_t>& values) -> bool {
        if (hipMemcpy(tensor.data<int32_t>(), values.data(),
                      values.size() * sizeof(int32_t), hipMemcpyHostToDevice) !=
            hipSuccess) {
            fail("i32 upload");
            return false;
        }
        return true;
    };

    auto tree_step = [&](int32_t token, int32_t node, int32_t src_row, bool src_root,
                         const std::vector<int32_t>& offsets,
                         const std::vector<int32_t>& nodes, uint32_t position,
                         std::vector<int32_t>& next_token) -> bool {
        const uint32_t rows = static_cast<uint32_t>(offsets.size() - 1u);
        if (!upload_i32(step_tokens, std::vector<int32_t>(rows, token))) return false;
        if (!upload_i32(row_nodes, std::vector<int32_t>(rows, node))) return false;
        if (!upload_i32(src_rows, std::vector<int32_t>(rows, src_row))) return false;
        if (!upload_i32(anc_offsets, offsets)) return false;
        if (!upload_i32(anc_nodes, nodes)) return false;
        auto st = ps::qwen35::eagle4::eagle4_tree_draft_step(
            executor, context, step_tokens.data<int32_t>(), row_nodes.data<int32_t>(),
            src_rows.data<int32_t>(), src_root, anc_offsets.data<int32_t>(),
            anc_nodes.data<int32_t>(), rows, position, kPosRoot, 1u, stream);
        if (!st.ok()) {
            fail("tree step: " + st.message());
            return false;
        }
        if (hipStreamSynchronize(stream) != hipSuccess) {
            fail("tree step sync");
            return false;
        }
        next_token.resize(rows);
        if (hipMemcpy(next_token.data(), executor.topk_ids.data<int32_t>(),
                      rows * sizeof(int32_t), hipMemcpyDeviceToHost) != hipSuccess) {
            fail("tree topk download");
            return false;
        }
        return true;
    };

    std::vector<int32_t> tree_tokens;
    std::vector<int32_t> next;
    tree_tokens.push_back(root_host[0]);
    bool step_ok = tree_step(root_host[0], 0, 0, true, {0, 1}, {0}, kPosRoot, next);
    check(step_ok, "tree step 0");
    if (step_ok) tree_tokens.push_back(next[0]);
    step_ok = tree_step(next[0], 1, 0, false, {0, 2}, {0, 1}, kPosRoot + 1u, next);
    check(step_ok, "tree step 1");
    if (step_ok) tree_tokens.push_back(next[0]);
    step_ok = tree_step(next[0], 2, 0, false, {0, 3}, {0, 1, 2}, kPosRoot + 2u, next);
    check(step_ok, "tree step 2");
    if (step_ok) tree_tokens.push_back(next[0]);

    std::vector<bf16_t> tree_normed(kHidden);
    if (hipMemcpy(tree_normed.data(), executor.normed.data<bf16_t>(),
                  kHidden * sizeof(bf16_t), hipMemcpyDeviceToHost) != hipSuccess) {
        fail("tree normed download");
        return 1;
    }
    check(chain_tokens == tree_tokens, "degenerate tree drafts equal chain drafts");
    if (chain_tokens != tree_tokens) {
        std::printf("chain:");
        for (int32_t t : chain_tokens) std::printf(" %d", t);
        std::printf("\ntree:");
        for (int32_t t : tree_tokens) std::printf(" %d", t);
        std::printf("\n");
    }
    check(std::memcmp(chain_normed.data(), tree_normed.data(),
                      kHidden * sizeof(bf16_t)) == 0,
          "degenerate tree final normed equals chain");

    check(do_extend(), "extend siblings");
    step_ok = tree_step(root_host[0], 0, 0, true, {0, 1}, {0}, kPosRoot, next);
    check(step_ok, "sibling root step");
    const int32_t sibling_token = step_ok ? next[0] : 7;
    const std::vector<int32_t> sib_tokens = {sibling_token, sibling_token};
    if (!upload_i32(step_tokens, sib_tokens)) return 1;
    if (!upload_i32(row_nodes, std::vector<int32_t>{1, 2})) return 1;
    if (!upload_i32(src_rows, std::vector<int32_t>{0, 0})) return 1;
    if (!upload_i32(anc_offsets, std::vector<int32_t>{0, 2, 4})) return 1;
    if (!upload_i32(anc_nodes, std::vector<int32_t>{0, 1, 0, 2})) return 1;
    {
        auto st = ps::qwen35::eagle4::eagle4_tree_draft_step(
            executor, context, step_tokens.data<int32_t>(), row_nodes.data<int32_t>(),
            src_rows.data<int32_t>(), false, anc_offsets.data<int32_t>(),
            anc_nodes.data<int32_t>(), 2u, kPosRoot + 1u, kPosRoot, 1u, stream);
        if (!st.ok()) fail("sibling step: " + st.message());
        if (hipStreamSynchronize(stream) != hipSuccess) fail("sibling sync");
    }
    std::vector<bf16_t> sibling_rows(2u * kHidden);
    if (hipMemcpy(sibling_rows.data(), executor.normed.data<bf16_t>(),
                  sibling_rows.size() * sizeof(bf16_t), hipMemcpyDeviceToHost) !=
        hipSuccess) {
        fail("sibling normed download");
        return 1;
    }
    check(std::memcmp(sibling_rows.data(), sibling_rows.data() + kHidden,
                      kHidden * sizeof(bf16_t)) == 0,
          "identical sibling rows produce identical normed");

    auto garbage_round_slot = [&](uint32_t slot, uint8_t pattern) -> bool {
        const std::size_t kv_features = config.attention_kv_rows();
        std::vector<bf16_t> garbage(kv_features);
        std::memset(garbage.data(), static_cast<int>(pattern),
                    garbage.size() * sizeof(bf16_t));
        const std::size_t offset = static_cast<std::size_t>(slot) * kv_features;
        if (hipMemcpy(executor.round_k.data<bf16_t>() + offset, garbage.data(),
                      garbage.size() * sizeof(bf16_t), hipMemcpyHostToDevice) !=
            hipSuccess) {
            fail("garbage round k");
            return false;
        }
        if (hipMemcpy(executor.round_v.data<bf16_t>() + offset, garbage.data(),
                      garbage.size() * sizeof(bf16_t), hipMemcpyHostToDevice) !=
            hipSuccess) {
            fail("garbage round v");
            return false;
        }
        return true;
    };

    auto invisible_run = [&](uint8_t pattern, std::vector<bf16_t>& normed_out) -> bool {
        check(do_extend(), "extend invisible");
        if (!tree_step(root_host[0], 0, 0, true, {0, 1}, {0}, kPosRoot, next))
            return false;
        if (!tree_step(next[0], 1, 0, false, {0, 2}, {0, 1}, kPosRoot + 1u, next))
            return false;
        if (!garbage_round_slot(2u, pattern)) return false;
        if (!tree_step(next[0], 3, 0, false, {0, 3}, {0, 1, 3}, kPosRoot + 2u, next))
            return false;
        normed_out.resize(kHidden);
        if (hipMemcpy(normed_out.data(), executor.normed.data<bf16_t>(),
                      kHidden * sizeof(bf16_t), hipMemcpyDeviceToHost) != hipSuccess) {
            fail("invisible normed download");
            return false;
        }
        return true;
    };

    std::vector<bf16_t> invisible_a;
    std::vector<bf16_t> invisible_b;
    check(invisible_run(0x3C, invisible_a), "invisible run a");
    check(invisible_run(0x7E, invisible_b), "invisible run b");
    if (invisible_a.size() == kHidden && invisible_b.size() == kHidden) {
        check(std::memcmp(invisible_a.data(), invisible_b.data(),
                          kHidden * sizeof(bf16_t)) == 0,
              "non-ancestor round kv is invisible");
    }

    (void)ps::qwen35::eagle4::eagle4_context_shutdown(context);
    (void)ps::qwen35::eagle4::eagle4_executor_shutdown(executor);
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    (void)arena.shutdown();
    fs::remove_all(root);

    if (g_fail == 0) {
        std::printf("PASS\n");
        return 0;
    }
    std::printf("FAIL (%d)\n", g_fail);
    return 1;
}
