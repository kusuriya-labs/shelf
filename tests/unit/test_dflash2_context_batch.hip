#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/types.h>
#include <shelf/models/qwen35/dflash2/config.h>
#include <shelf/models/qwen35/dflash2/context_state.h>
#include <shelf/models/qwen35/dflash2/executor.h>
#include <shelf/models/qwen35/dflash2/weights.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace df = shelf::qwen35::dflash2;

static float as_float(shelf::bf16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value.data) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

int main() {
    const char* model = std::getenv("SHELF_MODEL_DIR_DFLASH2");
    if (model == nullptr || *model == '\0') {
        std::printf("SKIP: SHELF_MODEL_DIR_DFLASH2 is not set\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 77;
    auto config_result = df::read_dflash2_config(model);
    if (!config_result.ok()) return 1;
    const auto config = config_result.value();
    auto arena_result = shelf::gpu::GpuArena::create(0, 12ull << 30);
    if (!arena_result.ok()) return 1;
    auto arena = arena_result.release();
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;
    auto weights_result = df::load_dflash2_weights(model, config, arena, stream);
    if (!weights_result.ok()) {
        std::printf("weight load failed: %s\n", weights_result.status().message().c_str());
        return 1;
    }
    auto weights = weights_result.release();
    df::DFlash2ExecutorConfig options;
    options.max_rows = 16u;
    options.max_batch_rows = 16u;
    options.max_context_rows = static_cast<uint32_t>(config.sliding_window);
    options.max_prefill_feature_rows = 16u;
    auto executor_result = df::create_dflash2_executor(config, weights, arena, options);
    if (!executor_result.ok()) {
        std::printf("executor create failed: %s\n", executor_result.status().message().c_str());
        return 1;
    }
    auto executor = executor_result.release();
    std::array<df::DFlash2ContextState, 4> contexts;
    const uint32_t capacity = static_cast<uint32_t>(config.sliding_window);
    const std::array<uint32_t, 2> start{0u, capacity - 1u};
    for (uint32_t i = 0; i < contexts.size(); ++i) {
        auto result = df::create_dflash2_context_state(config, arena, start[i % 2u]);
        if (!result.ok()) return 1;
        contexts[i] = result.release();
    }
    const uint32_t hidden = static_cast<uint32_t>(config.hidden_size);
    const uint32_t kv = static_cast<uint32_t>(config.attention_kv_rows());
    const uint32_t taps = shelf::kernel::kDFlash2TargetTaps;
    const std::size_t elements = static_cast<std::size_t>(2u) * taps * hidden;
    auto tap_allocation = arena.allocate_aligned(elements * sizeof(shelf::bf16_t), 256u);
    if (!tap_allocation.ok()) return 1;
    auto* device_taps = static_cast<shelf::bf16_t*>(tap_allocation.release().data());
    std::vector<shelf::bf16_t> host_taps(elements);
    std::array<df::DFlash2TargetTapAppendRequest, 2> requests;
    for (uint32_t i = 0; i < 2u; ++i) {
        requests[i].context = &contexts[i + 2u];
        for (uint32_t t = 0; t < taps; ++t) {
            requests[i].taps[t] = device_taps + (static_cast<std::size_t>(i) * taps + t) * hidden;
        }
    }
    int failures = 0;
    for (uint32_t step = 0; step < 2u; ++step) {
        if (step != 0u && hipStreamSynchronize(stream) != hipSuccess) return 1;
        for (std::size_t i = 0; i < elements; ++i) {
            const float value = static_cast<float>(static_cast<int>((i * 17u + step * 11u) % 101u) - 50) / 100.0f;
            host_taps[i].data = shelf::f32_to_bf16_rne(value);
        }
        if (hipMemcpyAsync(device_taps, host_taps.data(), elements * sizeof(shelf::bf16_t),
                           hipMemcpyHostToDevice, stream) != hipSuccess) return 1;
        for (uint32_t i = 0; i < 2u; ++i) {
            requests[i].position = start[i] + step;
            auto st = df::dflash2_append_target_taps(executor, contexts[i], requests[i].taps.data(),
                                                      taps, 1u, requests[i].position, stream);
            if (!st.ok()) { std::printf("serial append: %s\n", st.message().c_str()); return 1; }
        }
        auto st = df::dflash2_append_target_taps_batch(executor, requests.data(), 2u, stream);
        if (!st.ok()) { std::printf("batch append: %s\n", st.message().c_str()); return 1; }
    }
    if (hipStreamSynchronize(stream) != hipSuccess) return 1;
    double max_abs = 0.0, square = 0.0, reference_square = 0.0;
    uint64_t count = 0u;
    std::vector<shelf::bf16_t> a(kv), b(kv);
    for (uint32_t i = 0; i < 2u; ++i) {
        if (contexts[i].length != contexts[i + 2u].length ||
            contexts[i].next_position != contexts[i + 2u].next_position) ++failures;
        for (uint32_t layer = 0; layer < config.num_hidden_layers; ++layer) {
            for (uint32_t step = 0; step < 2u; ++step) {
                const std::size_t offset = (static_cast<std::size_t>(layer) * capacity +
                                            (start[i] + step) % capacity) * kv;
                for (uint32_t is_value = 0; is_value < 2u; ++is_value) {
                    const auto& src = is_value ? contexts[i].v_ring : contexts[i].k_ring;
                    const auto& dst = is_value ? contexts[i + 2u].v_ring : contexts[i + 2u].k_ring;
                    if (hipMemcpy(a.data(), src.data<shelf::bf16_t>() + offset, kv * sizeof(shelf::bf16_t), hipMemcpyDeviceToHost) != hipSuccess ||
                        hipMemcpy(b.data(), dst.data<shelf::bf16_t>() + offset, kv * sizeof(shelf::bf16_t), hipMemcpyDeviceToHost) != hipSuccess) return 1;
                    for (uint32_t j = 0; j < kv; ++j) {
                        const double x = as_float(a[j]), y = as_float(b[j]);
                        if (!std::isfinite(x) || !std::isfinite(y)) { ++failures; continue; }
                        const double delta = x - y;
                        max_abs = std::max(max_abs, std::abs(delta));
                        square += delta * delta;
                        reference_square += x * x;
                        ++count;
                    }
                }
            }
        }
    }
    const double relative_rms = reference_square > 0.0 ? std::sqrt(square / reference_square) : 0.0;
    if (count == 0u || relative_rms > 0.002) ++failures;
    auto duplicate = requests;
    duplicate[1].context = duplicate[0].context;
    duplicate[0].position = duplicate[0].context->next_position;
    duplicate[1].position = duplicate[0].position;
    if (df::dflash2_append_target_taps_batch(executor, duplicate.data(), 2u, stream).ok()) ++failures;
    for (auto& context : contexts) if (!df::dflash2_context_shutdown(context).ok()) ++failures;
    if (!df::dflash2_executor_shutdown(executor).ok()) ++failures;
    if (hipStreamDestroy(stream) != hipSuccess) ++failures;
    if (!arena.shutdown().ok()) ++failures;
    std::printf("count=%llu max_abs=%.6f relative_rms=%.6f failures=%d\n",
                static_cast<unsigned long long>(count), max_abs, relative_rms, failures);
    return failures ? 1 : 0;
}
