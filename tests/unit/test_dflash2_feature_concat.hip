#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/kernels/dflash2/feature_concat.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using ps::bf16_t;
using ps::f32_to_bf16_rne;

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

static float bf16_to_f32(bf16_t x) {
    const uint32_t u = static_cast<uint32_t>(x.data) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

static bool alloc_bf16(
    ps::gpu::GpuArena& arena,
    const std::vector<std::size_t>& shape,
    ps::gpu::Tensor& out) {
    std::size_t count = 1;
    for (std::size_t d : shape) count *= d;
    auto view = arena.allocate_aligned(count * sizeof(bf16_t), 128);
    if (!view.ok()) return false;
    auto backing = view.release();
    std::vector<std::size_t> strides(shape.size());
    std::size_t stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }
    auto t = ps::gpu::Tensor::create<bf16_t>(backing, shape, strides);
    if (!t.ok()) return false;
    out = t.release();
    return true;
}

static void fill_taps(
    std::vector<std::vector<bf16_t>>& taps,
    uint32_t rows,
    uint32_t features,
    uint32_t seed) {
    taps.assign(ps::kernel::kDFlash2TargetTaps, {});
    for (uint32_t t = 0; t < ps::kernel::kDFlash2TargetTaps; ++t) {
        taps[t].resize(static_cast<std::size_t>(rows) * features);
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t f = 0; f < features; ++f) {
                const float v = static_cast<float>(
                    (static_cast<int32_t>((r * 131u + f * 7u + t * 977u + seed) % 2001u)) - 1000);
                taps[t][static_cast<std::size_t>(r) * features + f] =
                    bf16_t{f32_to_bf16_rne(v * 0.001f)};
            }
        }
    }
}

static int run_case(
    ps::gpu::GpuArena& arena,
    hipStream_t stream,
    uint32_t rows,
    uint32_t features,
    uint32_t output_row_stride) {
    const uint32_t out_features = ps::kernel::kDFlash2TargetTaps * features;
    std::vector<std::vector<bf16_t>> host_taps;
    fill_taps(host_taps, rows, features, rows * 17u + features);

    ps::gpu::Tensor dev_taps[ps::kernel::kDFlash2TargetTaps];
    for (uint32_t t = 0; t < ps::kernel::kDFlash2TargetTaps; ++t) {
        if (!alloc_bf16(arena, {rows, features}, dev_taps[t])) {
            check(false, "tap alloc");
            return 1;
        }
        const hipError_t e = hipMemcpy(dev_taps[t].data<bf16_t>(), host_taps[t].data(),
                                       host_taps[t].size() * sizeof(bf16_t),
                                       hipMemcpyHostToDevice);
        check(e == hipSuccess, "tap upload");
    }

    ps::gpu::Tensor out;
    if (!alloc_bf16(arena, {rows, output_row_stride}, out)) {
        check(false, "out alloc");
        return 1;
    }
    std::vector<bf16_t> out_zero(static_cast<std::size_t>(rows) * output_row_stride,
                                 bf16_t{0});
    check(hipMemcpy(out.data<bf16_t>(), out_zero.data(),
                    out_zero.size() * sizeof(bf16_t), hipMemcpyHostToDevice) == hipSuccess,
          "out init");

    const hipError_t err = ps::kernel::launch_dflash2_feature_concat(
        dev_taps[0].data<bf16_t>(), dev_taps[1].data<bf16_t>(), dev_taps[2].data<bf16_t>(),
        dev_taps[3].data<bf16_t>(), dev_taps[4].data<bf16_t>(), features, rows,
        out.data<bf16_t>(), output_row_stride, stream);
    check(err == hipSuccess, "concat launch");
    if (err != hipSuccess) return 1;
    check(hipStreamSynchronize(stream) == hipSuccess, "concat sync");

    std::vector<bf16_t> host_out(static_cast<std::size_t>(rows) * output_row_stride);
    check(hipMemcpy(host_out.data(), out.data<bf16_t>(),
                    host_out.size() * sizeof(bf16_t), hipMemcpyDeviceToHost) == hipSuccess,
          "out readback");

    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t t = 0; t < ps::kernel::kDFlash2TargetTaps; ++t) {
            for (uint32_t f = 0; f < features; ++f) {
                const bf16_t got =
                    host_out[static_cast<std::size_t>(r) * output_row_stride +
                             static_cast<std::size_t>(t) * features + f];
                const bf16_t want = host_taps[t][static_cast<std::size_t>(r) * features + f];
                if (got.data != want.data) {
                    check(false, "concat mismatch rows=" + std::to_string(rows) +
                                     " features=" + std::to_string(features) + " row=" +
                                     std::to_string(r) + " tap=" + std::to_string(t) +
                                     " f=" + std::to_string(f));
                    return 1;
                }
            }
        }
        for (uint32_t f = out_features; f < output_row_stride; ++f) {
            check(host_out[static_cast<std::size_t>(r) * output_row_stride + f].data == 0u,
                  "padding untouched");
        }
    }
    for (const bf16_t v : host_out) {
        const float fv = bf16_to_f32(v);
        check(fv == fv && fv < 1.0e30f && fv > -1.0e30f, "finite");
    }
    return 0;
}

int main() {
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

    run_case(arena, stream, 1u, 5120u, 25600u);
    run_case(arena, stream, 2u, 5120u, 25600u);
    run_case(arena, stream, 8u, 5120u, 25600u);
    run_case(arena, stream, 32u, 5120u, 25600u);
    run_case(arena, stream, 1u, 8u, 40u);
    run_case(arena, stream, 3u, 12u, 60u);
    run_case(arena, stream, 8u, 4u, 20u);

    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    const ps::Status shutdown = arena.shutdown();
    check(shutdown.ok(), "arena shutdown: " + shutdown.message());

    std::printf("test_dflash2_feature_concat: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
