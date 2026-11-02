#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/tensor.h>
#include <shelf/models/qwen35/eagle4/config.h>
#include <shelf/models/qwen35/eagle4/context_state.h>
#include <shelf/models/qwen35/eagle4/executor.h>
#include <shelf/models/qwen35/eagle4/weights.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using ps::bf16_t;

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

static float bf16_to_f32(bf16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value.data) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static uint32_t lcg(uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

static bf16_t random_bf16(uint32_t& state) {
    const float u = static_cast<float>(lcg(state) >> 8) / static_cast<float>(1u << 24);
    return bf16_t{ps::f32_to_bf16_rne((u * 2.0f - 1.0f) * 2.0f)};
}

int main() {
    const char* model_dir = std::getenv("SHELF_EAGLE4_MODEL_DIR");
    if (model_dir == nullptr || model_dir[0] == '\0') {
        std::printf("SKIP: SHELF_EAGLE4_MODEL_DIR is not set\n");
        return 77;
    }
    auto config_result = ps::qwen35::eagle4::read_eagle4_config(model_dir);
    check(config_result.ok(), "config reads");
    if (!config_result.ok()) return 1;
    const ps::qwen35::eagle4::Eagle4Config config = config_result.release();

    auto arena_result = ps::gpu::GpuArena::create(0, 12ull * 1024ull * 1024ull * 1024ull);
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
        ps::qwen35::eagle4::load_eagle4_weights(model_dir, config, arena, stream);
    check(weights_result.ok(), "weights load");
    if (!weights_result.ok()) return 1;
    ps::qwen35::eagle4::Eagle4Weights weights = weights_result.release();

    ps::qwen35::eagle4::Eagle4ExecutorConfig lowrank_config;
    lowrank_config.num_candidates = static_cast<uint32_t>(config.num_candidates);
    auto lowrank_result = ps::qwen35::eagle4::create_eagle4_executor(
        config, weights, arena, lowrank_config);
    check(lowrank_result.ok(), "lowrank executor create");
    if (!lowrank_result.ok()) return 1;
    ps::qwen35::eagle4::Eagle4Executor lowrank = lowrank_result.release();

    ps::qwen35::eagle4::Eagle4ExecutorConfig gemv_config = lowrank_config;
    gemv_config.use_lowrank_candidate_head = false;
    auto gemv_result =
        ps::qwen35::eagle4::create_eagle4_executor(config, weights, arena, gemv_config);
    check(gemv_result.ok(), "gemv executor create");
    if (!gemv_result.ok()) return 1;
    ps::qwen35::eagle4::Eagle4Executor gemv = gemv_result.release();

    auto lowrank_context_result =
        ps::qwen35::eagle4::create_eagle4_context_state(config, arena, 0u);
    check(lowrank_context_result.ok(), "lowrank context create");
    if (!lowrank_context_result.ok()) return 1;
    ps::qwen35::eagle4::Eagle4ContextState lowrank_context =
        lowrank_context_result.release();
    auto gemv_context_result =
        ps::qwen35::eagle4::create_eagle4_context_state(config, arena, 0u);
    check(gemv_context_result.ok(), "gemv context create");
    if (!gemv_context_result.ok()) return 1;
    ps::qwen35::eagle4::Eagle4ContextState gemv_context = gemv_context_result.release();

    constexpr uint32_t kRounds = 8u;
    const uint32_t tap_features = static_cast<uint32_t>(config.target_hidden_size);
    uint32_t seed = 4242u;
    std::vector<int32_t> tokens{7, 11, 19, 23, 31, 37, 41, 43};
    std::vector<int32_t> lowrank_ids(kRounds, -1);
    std::vector<int32_t> gemv_ids(kRounds, -1);
    for (uint32_t round = 0; round < kRounds; ++round) {
        std::vector<bf16_t> taps(static_cast<std::size_t>(config.num_aux_layers) *
                                 tap_features);
        for (bf16_t& value : taps) value = random_bf16(seed);
        ps::gpu::Tensor tap_tensors[ps::qwen35::eagle4::kEagle4AuxLayers];
        for (uint32_t t = 0; t < config.num_aux_layers; ++t) {
            auto view = arena.allocate_aligned(tap_features * sizeof(bf16_t), 256u);
            check(view.ok(), "tap alloc");
            if (!view.ok()) return 1;
            auto created = ps::gpu::Tensor::create<bf16_t>(
                view.release(), {1u, tap_features}, {tap_features, 1u});
            check(created.ok(), "tap tensor");
            if (!created.ok()) return 1;
            tap_tensors[t] = created.release();
            check(hipMemcpy(tap_tensors[t].data<bf16_t>(),
                            taps.data() + static_cast<std::size_t>(t) * tap_features,
                            tap_features * sizeof(bf16_t), hipMemcpyHostToDevice) ==
                      hipSuccess,
                  "tap upload");
        }
        ps::gpu::Tensor token_tensor;
        auto token_view = arena.allocate_aligned(sizeof(int32_t), 256u);
        check(token_view.ok(), "token alloc");
        if (!token_view.ok()) return 1;
        auto token_created =
            ps::gpu::Tensor::create<int32_t>(token_view.release(), {1u}, {1u});
        check(token_created.ok(), "token tensor");
        if (!token_created.ok()) return 1;
        token_tensor = token_created.release();
        check(hipMemcpy(token_tensor.data<int32_t>(), &tokens[round], sizeof(int32_t),
                        hipMemcpyHostToDevice) == hipSuccess,
              "token upload");
        ps::gpu::Tensor root_token;
        auto root_view = arena.allocate_aligned(sizeof(int32_t), 256u);
        check(root_view.ok(), "root alloc");
        if (!root_view.ok()) return 1;
        auto root_created =
            ps::gpu::Tensor::create<int32_t>(root_view.release(), {1u}, {1u});
        check(root_created.ok(), "root tensor");
        if (!root_created.ok()) return 1;
        root_token = root_created.release();

        const bf16_t* tap_ptrs[ps::qwen35::eagle4::kEagle4AuxLayers] = {
            tap_tensors[0].data<bf16_t>(), tap_tensors[1].data<bf16_t>(),
            tap_tensors[2].data<bf16_t>()};
        auto lowrank_extend = ps::qwen35::eagle4::eagle4_extend_rows(
            lowrank, lowrank_context, tap_ptrs, config.num_aux_layers,
            token_tensor.data<int32_t>(), 1u, round, 1u, root_token.data<int32_t>(),
            stream);
        check(lowrank_extend.ok(), "lowrank extend");
        if (!lowrank_extend.ok()) return 1;
        check(hipMemcpy(&lowrank_ids[round], root_token.data<int32_t>(), sizeof(int32_t),
                        hipMemcpyDeviceToHost) == hipSuccess,
              "lowrank root download");

        auto gemv_extend = ps::qwen35::eagle4::eagle4_extend_rows(
            gemv, gemv_context, tap_ptrs, config.num_aux_layers,
            token_tensor.data<int32_t>(), 1u, round, 1u, root_token.data<int32_t>(),
            stream);
        check(gemv_extend.ok(), "gemv extend");
        if (!gemv_extend.ok()) return 1;
        check(hipMemcpy(&gemv_ids[round], root_token.data<int32_t>(), sizeof(int32_t),
                        hipMemcpyDeviceToHost) == hipSuccess,
              "gemv root download");
    }

    uint32_t matches = 0u;
    for (uint32_t round = 0; round < kRounds; ++round) {
        if (lowrank_ids[round] == gemv_ids[round]) {
            ++matches;
        } else {
            std::printf("NOTE round %u: lowrank %d gemv %d (candidate set miss)\n", round,
                        lowrank_ids[round], gemv_ids[round]);
        }
    }
    std::printf("candidate head argmax agreement: %u / %u\n", matches, kRounds);

    {
        const uint32_t rows = 1u;
        const uint32_t candidates = static_cast<uint32_t>(config.num_candidates);
        std::vector<int32_t> ids(static_cast<std::size_t>(rows) * candidates);
        std::vector<float> lowrank_logits(static_cast<std::size_t>(rows) * candidates);
        std::vector<bf16_t> gemv_logits(static_cast<std::size_t>(rows) * config.vocab_size);
        check(hipMemcpy(ids.data(), lowrank.candidate_ids.data<int32_t>(),
                        ids.size() * sizeof(int32_t), hipMemcpyDeviceToHost) == hipSuccess,
              "candidate ids download");
        check(hipMemcpy(lowrank_logits.data(), lowrank.candidate_logits.data<float>(),
                        lowrank_logits.size() * sizeof(float), hipMemcpyDeviceToHost) ==
                  hipSuccess,
              "candidate logits download");
        check(hipMemcpy(gemv_logits.data(), gemv.gemv_logits.data<bf16_t>(),
                        gemv_logits.size() * sizeof(bf16_t), hipMemcpyDeviceToHost) ==
                  hipSuccess,
              "gemv logits download");
        double worst = 0.0;
        uint32_t compared = 0u;
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t c = 0; c < candidates; ++c) {
                const int32_t id = ids[static_cast<std::size_t>(r) * candidates + c];
                if (id < 0 || static_cast<uint32_t>(id) >= config.vocab_size) continue;
                const double lowrank_value =
                    lowrank_logits[static_cast<std::size_t>(r) * candidates + c];
                const double gemv_value = bf16_to_f32(
                    gemv_logits[static_cast<std::size_t>(r) * config.vocab_size + id]);
                worst = std::max(worst,
                                 std::fabs(lowrank_value - gemv_value) /
                                     std::max(std::fabs(lowrank_value), 1.0));
                ++compared;
            }
        }
        std::printf("candidate rescore compared %u entries, worst relative %.6f\n", compared,
                    worst);
        check(compared == static_cast<uint32_t>(rows) * candidates,
              "every candidate was rescored");
        check(worst < 1.0e-2, "candidate rescore matches the full-vocab gemv logits");
    }

    check(ps::qwen35::eagle4::eagle4_context_shutdown(lowrank_context).ok(),
          "lowrank context shutdown");
    check(ps::qwen35::eagle4::eagle4_context_shutdown(gemv_context).ok(),
          "gemv context shutdown");
    check(ps::qwen35::eagle4::eagle4_executor_shutdown(lowrank).ok(),
          "lowrank executor shutdown");
    check(ps::qwen35::eagle4::eagle4_executor_shutdown(gemv).ok(), "gemv executor shutdown");
    (void)hipStreamDestroy(stream);

    if (g_fail != 0) {
        std::printf("FAILED (%d)\n", g_fail);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
