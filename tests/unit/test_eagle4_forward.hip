#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/io/safetensors_writer.h>
#include <phaseshift/models/qwen35/eagle4/config.h>
#include <phaseshift/models/qwen35/eagle4/context_state.h>
#include <phaseshift/models/qwen35/eagle4/executor.h>
#include <phaseshift/models/qwen35/eagle4/weights.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include "support/eagle4_checkpoint.h"
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
using namespace eagle4_ckpt;

using ps::bf16_t;
using ps::f32_to_bf16_rne;


static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}



static std::vector<float> cpu_rmsnorm(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    uint32_t rows,
    uint32_t features) {
    std::vector<float> out(input.size());
    for (uint32_t r = 0; r < rows; ++r) {
        double sum = 0.0;
        for (uint32_t i = 0; i < features; ++i) {
            const double v = input[static_cast<std::size_t>(r) * features + i];
            sum += v * v;
        }
        const float inv =
            1.0f / std::sqrt(static_cast<float>(sum) / static_cast<float>(features) + kEps);
        for (uint32_t i = 0; i < features; ++i) {
            const float norm = input[static_cast<std::size_t>(r) * features + i] * inv;
            const float norm_rounded = round_bf16(norm);
            out[static_cast<std::size_t>(r) * features + i] =
                round_bf16(weight[i] * norm_rounded);
        }
    }
    return out;
}

static std::vector<float> cpu_gemm(
    const std::vector<float>& input,
    uint32_t rows,
    uint32_t k,
    const HostTensor& weight) {
    std::vector<float> out(static_cast<std::size_t>(rows) * weight.rows);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t o = 0; o < weight.rows; ++o) {
            double acc = 0.0;
            for (uint32_t i = 0; i < k; ++i) {
                acc += static_cast<double>(input[static_cast<std::size_t>(r) * k + i]) *
                       static_cast<double>(bf16_to_f32(
                           weight.data[static_cast<std::size_t>(o) * weight.cols + i]));
            }
            out[static_cast<std::size_t>(r) * weight.rows + o] =
                round_bf16(static_cast<float>(acc));
        }
    }
    return out;
}

static std::vector<float> cpu_rope(
    const std::vector<float>& input,
    uint32_t rows,
    uint32_t features,
    uint32_t position_start) {
    std::vector<float> out(input.size());
    const uint32_t heads = features / kHeadDim;
    for (uint32_t r = 0; r < rows; ++r) {
        const float position = static_cast<float>(position_start + r);
        for (uint32_t h = 0; h < heads; ++h) {
            const std::size_t base = static_cast<std::size_t>(r) * features + h * kHeadDim;
            for (uint32_t pair = 0; pair < kHeadDim / 2u; ++pair) {
                const float exponent =
                    -2.0f * static_cast<float>(pair) / static_cast<float>(kHeadDim);
                const float inv_freq = std::pow(kTheta, exponent);
                const float angle = position * inv_freq;
                const float c = round_bf16(std::cos(angle));
                const float s = round_bf16(std::sin(angle));
                const float x1 = input[base + pair];
                const float x2 = input[base + pair + kHeadDim / 2u];
                const float p1 = round_bf16(x1 * c);
                const float p2 = round_bf16(x2 * s);
                const float p3 = round_bf16(x2 * c);
                const float p4 = round_bf16(x1 * s);
                out[base + pair] = round_bf16(p1 - p2);
                out[base + pair + kHeadDim / 2u] = round_bf16(p3 + p4);
            }
        }
    }
    return out;
}

static std::vector<float> cpu_attention(
    const std::vector<float>& q,
    const std::vector<float>& k_ring,
    const std::vector<float>& v_ring,
    uint32_t rows,
    uint32_t position_start,
    uint32_t capacity,
    uint32_t kv_features) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    std::vector<float> out(static_cast<std::size_t>(rows) * kv_features);
    for (uint32_t r = 0; r < rows; ++r) {
        const uint32_t qpos = position_start + r;
        const uint32_t lo = qpos + 1u > kWindow ? qpos + 1u - kWindow : 0u;
        const uint32_t keys = qpos - lo + 1u;
        for (uint32_t h = 0; h < kHeads; ++h) {
            std::vector<double> scores(keys);
            double max_score = -1.0e30;
            for (uint32_t i = 0; i < keys; ++i) {
                const uint32_t slot = (lo + i) % capacity;
                double acc = 0.0;
                for (uint32_t d = 0; d < kHeadDim; ++d) {
                    acc += static_cast<double>(
                               q[static_cast<std::size_t>(r) * kv_features + h * kHeadDim + d]) *
                           static_cast<double>(
                               k_ring[static_cast<std::size_t>(slot) * kv_features +
                                      h * kHeadDim + d]);
                }
                scores[i] = acc * scale;
                max_score = std::max(max_score, scores[i]);
            }
            double sum = 0.0;
            for (double& s : scores) {
                s = std::exp(s - max_score);
                sum += s;
            }
            for (uint32_t d = 0; d < kHeadDim; ++d) {
                double acc = 0.0;
                for (uint32_t i = 0; i < keys; ++i) {
                    const uint32_t slot = (lo + i) % capacity;
                    acc += scores[i] / sum *
                           static_cast<double>(
                               v_ring[static_cast<std::size_t>(slot) * kv_features +
                                      h * kHeadDim + d]);
                }
                out[static_cast<std::size_t>(r) * kv_features + h * kHeadDim + d] =
                    round_bf16(static_cast<float>(acc));
            }
        }
    }
    return out;
}

struct CpuDraft {
    const HostModel* model = nullptr;
    std::vector<float> k_ring;
    std::vector<float> v_ring;
    uint32_t capacity = kWindow;

    explicit CpuDraft(const HostModel& m)
        : model(&m),
          k_ring(static_cast<std::size_t>(kWindow) * kHidden, 0.0f),
          v_ring(static_cast<std::size_t>(kWindow) * kHidden, 0.0f) {}

    std::vector<float> forward(
        const std::vector<int32_t>& tokens,
        const std::vector<float>& hidden_in,
        const std::vector<float>& taps,
        uint32_t rows,
        uint32_t position_start,
        std::vector<float>* normed_out,
        std::vector<float>* summed_out) {
        const uint32_t tap_features = kTapFeatures;
        std::vector<float> concat(static_cast<std::size_t>(rows) * kTaps * tap_features);
        for (uint32_t t = 0; t < kTaps; ++t) {
            std::vector<float> tap(static_cast<std::size_t>(rows) * tap_features);
            for (uint32_t r = 0; r < rows; ++r) {
                for (uint32_t i = 0; i < tap_features; ++i) {
                    tap[static_cast<std::size_t>(r) * tap_features + i] =
                        taps[(static_cast<std::size_t>(t) * rows + r) * tap_features + i];
                }
            }
            const std::vector<float> normed =
                cpu_rmsnorm(tap, std::vector<float>(
                                     [&] {
                                         std::vector<float> w(tap_features);
                                         for (uint32_t i = 0; i < tap_features; ++i) {
                                             w[i] = bf16_to_f32(
                                                 model->fc_norm[t][i]);
                                         }
                                         return w;
                                     }()),
                            rows, tap_features);
            for (uint32_t r = 0; r < rows; ++r) {
                for (uint32_t i = 0; i < tap_features; ++i) {
                    concat[static_cast<std::size_t>(r) * kTaps * tap_features +
                           static_cast<std::size_t>(t) * tap_features + i] =
                        normed[static_cast<std::size_t>(r) * tap_features + i];
                }
            }
        }
        std::vector<float> block_hidden =
            cpu_gemm(concat, rows, kTaps * tap_features, model->fc);
        if (!hidden_in.empty()) block_hidden = hidden_in;

        std::vector<float> embed(static_cast<std::size_t>(rows) * kHidden);
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t i = 0; i < kHidden; ++i) {
                embed[static_cast<std::size_t>(r) * kHidden + i] =
                    bf16_to_f32(model->embed.data[static_cast<std::size_t>(tokens[r]) *
                                                      kHidden + i]);
            }
        }
        std::vector<float> embed_norm = cpu_rmsnorm(
            embed,
            [&] {
                std::vector<float> w(kHidden);
                for (uint32_t i = 0; i < kHidden; ++i) {
                    w[i] = bf16_to_f32(model->input_norm[i]);
                }
                return w;
            }(),
            rows, kHidden);
        std::vector<float> hidden_norm = cpu_rmsnorm(
            block_hidden,
            [&] {
                std::vector<float> w(kHidden);
                for (uint32_t i = 0; i < kHidden; ++i) {
                    w[i] = bf16_to_f32(model->hidden_norm[i]);
                }
                return w;
            }(),
            rows, kHidden);
        std::vector<float> layer_input(static_cast<std::size_t>(rows) * 2u * kHidden);
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t i = 0; i < kHidden; ++i) {
                layer_input[static_cast<std::size_t>(r) * 2u * kHidden + i] =
                    embed_norm[static_cast<std::size_t>(r) * kHidden + i];
                layer_input[static_cast<std::size_t>(r) * 2u * kHidden + kHidden + i] =
                    hidden_norm[static_cast<std::size_t>(r) * kHidden + i];
            }
        }
        std::vector<float> q_raw = cpu_gemm(layer_input, rows, 2u * kHidden, model->q);
        std::vector<float> k_raw = cpu_gemm(layer_input, rows, 2u * kHidden, model->k);
        std::vector<float> v_raw = cpu_gemm(layer_input, rows, 2u * kHidden, model->v);
        const std::vector<float> q_rope =
            cpu_rope(q_raw, rows, kHidden, position_start);
        const std::vector<float> k_rope =
            cpu_rope(k_raw, rows, kHidden, position_start);
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t slot = (position_start + r) % capacity;
            for (uint32_t i = 0; i < kHidden; ++i) {
                k_ring[static_cast<std::size_t>(slot) * kHidden + i] =
                    k_rope[static_cast<std::size_t>(r) * kHidden + i];
                v_ring[static_cast<std::size_t>(slot) * kHidden + i] =
                    v_raw[static_cast<std::size_t>(r) * kHidden + i];
            }
        }
        const std::vector<float> attn = cpu_attention(
            q_rope, k_ring, v_ring, rows, position_start, capacity, kHidden);
        const std::vector<float> attn_o = cpu_gemm(attn, rows, kHidden, model->o);
        std::vector<float> residual(static_cast<std::size_t>(rows) * kHidden);
        for (std::size_t i = 0; i < residual.size(); ++i) {
            residual[i] = round_bf16(block_hidden[i] + attn_o[i]);
        }
        std::vector<float> post_norm = cpu_rmsnorm(
            residual,
            [&] {
                std::vector<float> w(kHidden);
                for (uint32_t i = 0; i < kHidden; ++i) {
                    w[i] = bf16_to_f32(model->post_norm[i]);
                }
                return w;
            }(),
            rows, kHidden);
        const std::vector<float> gate = cpu_gemm(post_norm, rows, kHidden, model->gate);
        const std::vector<float> up = cpu_gemm(post_norm, rows, kHidden, model->up);
        std::vector<float> swiglu(gate.size());
        for (std::size_t i = 0; i < gate.size(); ++i) {
            const float sigmoid = 1.0f / (1.0f + std::exp(-gate[i]));
            const float silu = round_bf16(gate[i] * sigmoid);
            swiglu[i] = round_bf16(silu * up[i]);
        }
        const std::vector<float> down = cpu_gemm(swiglu, rows, kIntermediate, model->down);
        std::vector<float> summed(static_cast<std::size_t>(rows) * kHidden);
        for (std::size_t i = 0; i < summed.size(); ++i) {
            summed[i] = round_bf16(residual[i] + down[i]);
        }
        std::vector<float> normed(static_cast<std::size_t>(rows) * kHidden);
        {
            std::vector<float> w(kHidden);
            for (uint32_t i = 0; i < kHidden; ++i) w[i] = bf16_to_f32(model->final_norm[i]);
            const std::vector<float> local = cpu_rmsnorm(summed, w, rows, kHidden);
            normed = local;
        }
        if (normed_out != nullptr) *normed_out = normed;
        if (summed_out != nullptr) *summed_out = summed;
        (void)hidden_in;
        return normed;
    }
};

static std::vector<int32_t> cpu_candidate_topk(
    const HostModel& model,
    const float* hidden,
    uint32_t k) {
    std::vector<float> projected(kRank);
    for (uint32_t j = 0; j < kRank; ++j) {
        double acc = 0.0;
        for (uint32_t d = 0; d < kHidden; ++d) {
            acc += static_cast<double>(
                       bf16_to_f32(model.efficient_down.data[static_cast<std::size_t>(j) *
                                                                 kHidden + d])) *
                   static_cast<double>(hidden[d]);
        }
        projected[j] = round_bf16(static_cast<float>(acc));
    }
    std::vector<float> scores(kVocab);
    for (uint32_t v = 0; v < kVocab; ++v) {
        double acc = 0.0;
        for (uint32_t j = 0; j < kRank; ++j) {
            acc += static_cast<double>(
                       bf16_to_f32(model.efficient_head.data[static_cast<std::size_t>(v) *
                                                                 kRank + j])) *
                   static_cast<double>(projected[j]);
        }
        scores[v] = round_bf16(static_cast<float>(acc));
    }
    std::vector<uint32_t> order(kVocab);
    for (uint32_t v = 0; v < kVocab; ++v) order[v] = v;
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        if (scores[a] != scores[b]) return scores[a] > scores[b];
        return a < b;
    });
    std::vector<float> logits(kCandidates);
    std::vector<int32_t> ids(kCandidates);
    for (uint32_t c = 0; c < kCandidates; ++c) {
        ids[c] = static_cast<int32_t>(order[c]);
        double acc = 0.0;
        for (uint32_t d = 0; d < kHidden; ++d) {
            acc += static_cast<double>(
                       bf16_to_f32(model.lm_head.data[static_cast<std::size_t>(order[c]) *
                                                          kHidden + d])) *
                   static_cast<double>(hidden[d]);
        }
        logits[c] = static_cast<float>(acc);
    }
    std::vector<uint32_t> cand_order(kCandidates);
    for (uint32_t c = 0; c < kCandidates; ++c) cand_order[c] = c;
    std::sort(cand_order.begin(), cand_order.end(), [&](uint32_t a, uint32_t b) {
        if (logits[a] != logits[b]) return logits[a] > logits[b];
        return ids[a] < ids[b];
    });
    std::vector<int32_t> out(k);
    for (uint32_t i = 0; i < k; ++i) out[i] = ids[cand_order[i]];
    return out;
}

int main() {
    const fs::path root = fs::temp_directory_path() /
                          ("ps_eagle4_forward_" + std::to_string(static_cast<long>(getpid())));
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

    auto weights_result = ps::qwen35::eagle4::load_eagle4_weights(
        root.string(), config, arena, stream);
    check(weights_result.ok(), "weights load");
    if (!weights_result.ok()) return 1;
    ps::qwen35::eagle4::Eagle4Weights weights = weights_result.release();

    ps::qwen35::eagle4::Eagle4ExecutorConfig exec_config;
    exec_config.num_candidates = kCandidates;
    auto executor_result = ps::qwen35::eagle4::create_eagle4_executor(
        config, weights, arena, exec_config);
    check(executor_result.ok(), "executor create");
    if (!executor_result.ok()) return 1;
    ps::qwen35::eagle4::Eagle4Executor executor = executor_result.release();

    auto context_result =
        ps::qwen35::eagle4::create_eagle4_context_state(config, arena, 0u);
    check(context_result.ok(), "context create");
    if (!context_result.ok()) return 1;
    ps::qwen35::eagle4::Eagle4ContextState context = context_result.release();

    uint32_t seed = 777u;
    const uint32_t extend_rows = 2u;
    std::vector<bf16_t> taps(static_cast<std::size_t>(kTaps) * extend_rows * kTapFeatures);
    std::vector<float> host_taps(taps.size());
    for (std::size_t i = 0; i < taps.size(); ++i) {
        taps[i] = random_bf16(seed, 1.0f);
        host_taps[i] = bf16_to_f32(taps[i]);
    }
    std::vector<int32_t> host_tokens = {3, 5};
    std::vector<bf16_t> device_taps[3];
    ps::gpu::Tensor tap_tensor[3];
    for (uint32_t t = 0; t < kTaps; ++t) {
        device_taps[t].assign(
            taps.begin() + static_cast<std::size_t>(t) * extend_rows * kTapFeatures,
            taps.begin() + static_cast<std::size_t>(t + 1u) * extend_rows * kTapFeatures);
        auto view = arena.allocate_aligned(device_taps[t].size() * sizeof(bf16_t), 256u);
        check(view.ok(), "tap alloc");
        if (!view.ok()) return 1;
        auto created = ps::gpu::Tensor::create<bf16_t>(
            view.release(), {extend_rows, kTapFeatures}, {kTapFeatures, 1u});
        check(created.ok(), "tap tensor");
        if (!created.ok()) return 1;
        tap_tensor[t] = created.release();
        check(hipMemcpy(tap_tensor[t].data<bf16_t>(), device_taps[t].data(),
                        device_taps[t].size() * sizeof(bf16_t),
                        hipMemcpyHostToDevice) == hipSuccess,
              "tap upload");
    }
    ps::gpu::Tensor token_tensor;
    {
        auto view = arena.allocate_aligned(extend_rows * sizeof(int32_t), 256u);
        check(view.ok(), "token alloc");
        if (!view.ok()) return 1;
        auto created =
            ps::gpu::Tensor::create<int32_t>(view.release(), {extend_rows}, {1u});
        check(created.ok(), "token tensor");
        if (!created.ok()) return 1;
        token_tensor = created.release();
        check(hipMemcpy(token_tensor.data<int32_t>(), host_tokens.data(),
                        host_tokens.size() * sizeof(int32_t), hipMemcpyHostToDevice) ==
                  hipSuccess,
              "token upload");
    }
    ps::gpu::Tensor root_tokens;
    {
        auto view = arena.allocate_aligned(3u * sizeof(int32_t), 256u);
        check(view.ok(), "root alloc");
        if (!view.ok()) return 1;
        auto created = ps::gpu::Tensor::create<int32_t>(view.release(), {3u}, {1u});
        check(created.ok(), "root tensor");
        if (!created.ok()) return 1;
        root_tokens = created.release();
    }

    CpuDraft cpu(host);
    std::vector<float> cpu_normed;
    std::vector<float> cpu_summed;
    cpu.forward(host_tokens, {}, host_taps, extend_rows, 0u, &cpu_normed, &cpu_summed);

    const bf16_t* tap_ptrs[3] = {tap_tensor[0].data<bf16_t>(), tap_tensor[1].data<bf16_t>(),
                                 tap_tensor[2].data<bf16_t>()};
    auto extended = ps::qwen35::eagle4::eagle4_extend_rows(
        executor, context, tap_ptrs, kTaps, token_tensor.data<int32_t>(), extend_rows, 0u,
        3u, root_tokens.data<int32_t>(), stream);
    check(extended.ok(), "extend launches");
    check(hipStreamSynchronize(stream) == hipSuccess, "extend sync");

    {
        std::vector<bf16_t> device_normed(static_cast<std::size_t>(extend_rows) * kHidden);
        check(hipMemcpy(device_normed.data(), executor.normed.data<bf16_t>(),
                        device_normed.size() * sizeof(bf16_t), hipMemcpyDeviceToHost) ==
                  hipSuccess,
              "normed download");
        double worst = 0.0;
        double max_abs = 0.0;
        double rms = 0.0;
        for (std::size_t i = 0; i < device_normed.size(); ++i) {
            const double got = bf16_to_f32(device_normed[i]);
            const double want = cpu_normed[i];
            max_abs = std::max(max_abs, std::fabs(got - want));
            rms += want * want;
        }
        rms = std::sqrt(rms / static_cast<double>(device_normed.size()));
        worst = rms > 0.0 ? max_abs / rms : 0.0;
        check(worst < 3.0e-2, "extend normed matches the CPU reference (max_abs/rms=" +
                                 std::to_string(worst) + ", max_abs=" +
                                 std::to_string(max_abs) + ", rms=" + std::to_string(rms) +
                                 ")");
    }
    {
        std::vector<int32_t> device_root(3u);
        check(hipMemcpy(device_root.data(), root_tokens.data<int32_t>(),
                        3u * sizeof(int32_t), hipMemcpyDeviceToHost) == hipSuccess,
              "root download");
        const std::vector<int32_t> cpu_root = cpu_candidate_topk(
            host, cpu_normed.data() + static_cast<std::size_t>(extend_rows - 1u) * kHidden, 3u);
        for (uint32_t i = 0; i < 3u; ++i) {
            check(device_root[i] == cpu_root[i],
                  "root candidate " + std::to_string(i) + " matches (" +
                      std::to_string(device_root[i]) + " vs " +
                      std::to_string(cpu_root[i]) + ")");
        }
    }

    const uint32_t num_drafts = 3u;
    ps::gpu::Tensor drafts;
    {
        auto view = arena.allocate_aligned(num_drafts * sizeof(int32_t), 256u);
        check(view.ok(), "draft alloc");
        if (!view.ok()) return 1;
        auto created =
            ps::gpu::Tensor::create<int32_t>(view.release(), {num_drafts}, {1u});
        check(created.ok(), "draft tensor");
        if (!created.ok()) return 1;
        drafts = created.release();
        check(hipMemcpy(drafts.data<int32_t>(), root_tokens.data<int32_t>(),
                        sizeof(int32_t), hipMemcpyDeviceToDevice) == hipSuccess,
              "root relay");
    }
    auto stepped = ps::qwen35::eagle4::eagle4_draft_steps(
        executor, context, extend_rows, num_drafts, drafts.data<int32_t>(), stream);
    check(stepped.ok(), "draft steps launch");
    check(hipStreamSynchronize(stream) == hipSuccess, "draft steps sync");

    std::vector<int32_t> device_drafts(num_drafts);
    check(hipMemcpy(device_drafts.data(), drafts.data<int32_t>(),
                    num_drafts * sizeof(int32_t), hipMemcpyDeviceToHost) == hipSuccess,
          "drafts download");

    std::vector<int32_t> cpu_drafts(num_drafts);
    {
        std::vector<float> root_hidden(
            cpu_normed.begin() + static_cast<std::size_t>(extend_rows - 1u) * kHidden,
            cpu_normed.end());
        std::vector<float> prev_normed = root_hidden;
        std::vector<float> prev_summed(
            cpu_summed.begin() + static_cast<std::size_t>(extend_rows - 1u) * kHidden,
            cpu_summed.end());
        std::vector<int32_t> step_tokens(num_drafts);
        step_tokens[0] = cpu_candidate_topk(host, cpu_normed.data() +
                                                      static_cast<std::size_t>(extend_rows - 1u) *
                                                          kHidden,
                                            1)[0];
        for (uint32_t step = 0; step + 1u < num_drafts; ++step) {
            std::vector<float> hidden_in = (step == 0u) ? root_hidden : prev_normed;
            const std::vector<int32_t> tokens{step_tokens[step]};
            std::vector<float> step_normed;
            std::vector<float> step_summed;
            cpu.forward(tokens, hidden_in, host_taps, 1u, extend_rows + step, &step_normed,
                        &step_summed);
            step_tokens[step + 1u] = cpu_candidate_topk(host, step_normed.data(), 1)[0];
            prev_normed = step_normed;
            prev_summed = step_summed;
        }
        cpu_drafts = step_tokens;
    }
    for (uint32_t i = 0; i < num_drafts; ++i) {
        check(device_drafts[i] == cpu_drafts[i],
              "draft " + std::to_string(i) + " matches (" +
                  std::to_string(device_drafts[i]) + " vs " +
                  std::to_string(cpu_drafts[i]) + ")");
    }

    {
        uint32_t wrap_seed = 991u;
        for (uint32_t position = extend_rows; position < 12u; ++position) {
            std::vector<bf16_t> row_taps(static_cast<std::size_t>(kTaps) * kTapFeatures);
            std::vector<float> host_row_taps(row_taps.size());
            for (std::size_t i = 0; i < row_taps.size(); ++i) {
                row_taps[i] = random_bf16(wrap_seed, 1.0f);
                host_row_taps[i] = bf16_to_f32(row_taps[i]);
            }
            ps::gpu::Tensor row_tap_tensors[3];
            for (uint32_t t = 0; t < kTaps; ++t) {
                auto view = arena.allocate_aligned(kTapFeatures * sizeof(bf16_t), 256u);
                check(view.ok(), "wrap tap alloc");
                if (!view.ok()) return 1;
                auto created = ps::gpu::Tensor::create<bf16_t>(
                    view.release(), {1u, kTapFeatures}, {kTapFeatures, 1u});
                check(created.ok(), "wrap tap tensor");
                if (!created.ok()) return 1;
                row_tap_tensors[t] = created.release();
                check(hipMemcpy(row_tap_tensors[t].data<bf16_t>(),
                                row_taps.data() + static_cast<std::size_t>(t) * kTapFeatures,
                                kTapFeatures * sizeof(bf16_t), hipMemcpyHostToDevice) ==
                          hipSuccess,
                      "wrap tap upload");
            }
            const int32_t token = static_cast<int32_t>(position % kVocab);
            ps::gpu::Tensor row_token;
            {
                auto view = arena.allocate_aligned(sizeof(int32_t), 256u);
                check(view.ok(), "wrap token alloc");
                if (!view.ok()) return 1;
                auto created =
                    ps::gpu::Tensor::create<int32_t>(view.release(), {1u}, {1u});
                check(created.ok(), "wrap token tensor");
                if (!created.ok()) return 1;
                row_token = created.release();
                check(hipMemcpy(row_token.data<int32_t>(), &token, sizeof(int32_t),
                                hipMemcpyHostToDevice) == hipSuccess,
                      "wrap token upload");
            }
            const bf16_t* row_tap_ptrs[3] = {row_tap_tensors[0].data<bf16_t>(),
                                             row_tap_tensors[1].data<bf16_t>(),
                                             row_tap_tensors[2].data<bf16_t>()};
            auto extended_row = ps::qwen35::eagle4::eagle4_extend_rows(
                executor, context, row_tap_ptrs, kTaps, row_token.data<int32_t>(), 1u,
                position, 1u, root_tokens.data<int32_t>(), stream);
            check(extended_row.ok(), "wrap extend launches");
            check(hipStreamSynchronize(stream) == hipSuccess, "wrap extend sync");

            std::vector<float> cpu_row_normed;
            std::vector<float> cpu_row_summed;
            cpu.forward({token}, {}, host_row_taps, 1u, position, &cpu_row_normed,
                        &cpu_row_summed);
            std::vector<bf16_t> device_row(static_cast<std::size_t>(kHidden));
            check(hipMemcpy(device_row.data(), executor.normed.data<bf16_t>(),
                            device_row.size() * sizeof(bf16_t), hipMemcpyDeviceToHost) ==
                      hipSuccess,
                  "wrap normed download");
            double max_abs = 0.0;
            double rms = 0.0;
            for (uint32_t i = 0; i < kHidden; ++i) {
                const double got = bf16_to_f32(device_row[i]);
                const double want = cpu_row_normed[i];
                max_abs = std::max(max_abs, std::fabs(got - want));
                rms += want * want;
            }
            rms = std::sqrt(rms / static_cast<double>(kHidden));
            const double ratio = rms > 0.0 ? max_abs / rms : 0.0;
            check(ratio < 3.0e-2, "ring wrap extend matches at position " +
                                      std::to_string(position) + " (max_abs/rms=" +
                                      std::to_string(ratio) + ")");
            int32_t device_root = -1;
            check(hipMemcpy(&device_root, root_tokens.data<int32_t>(), sizeof(int32_t),
                            hipMemcpyDeviceToHost) == hipSuccess,
                  "wrap root download");
            check(device_root == cpu_candidate_topk(host, cpu_row_normed.data(), 1)[0],
                  "ring wrap root candidate at position " + std::to_string(position));
        }
    }

    check(ps::qwen35::eagle4::eagle4_context_shutdown(context).ok(), "context shutdown");
    check(ps::qwen35::eagle4::eagle4_executor_shutdown(executor).ok(), "executor shutdown");
    (void)hipStreamDestroy(stream);
    fs::remove_all(root);

    if (g_fail != 0) {
        std::printf("FAILED (%d)\n", g_fail);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
