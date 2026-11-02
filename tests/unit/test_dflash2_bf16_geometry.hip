#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/tensor.h>
#include <shelf/core/memory/types.h>
#include <shelf/kernels/optimized/linear/bf16.h>
#include <shelf/inference/execution/selectors/linear_selector.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using shelf::bf16_t;
using shelf::f32_to_bf16_rne;

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
    shelf::gpu::GpuArena& arena,
    const std::vector<std::size_t>& shape,
    shelf::gpu::Tensor& out) {
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
    auto t = shelf::gpu::Tensor::create<bf16_t>(backing, shape, strides);
    if (!t.ok()) return false;
    out = t.release();
    return true;
}

static uint32_t lcg(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return s;
}

static void run_geometry(
    shelf::gpu::GpuArena& arena,
    hipStream_t stream,
    uint32_t out_features,
    uint32_t k,
    uint32_t rows,
    bool allow_dot2,
    uint32_t seed) {
    std::vector<bf16_t> host_weight(static_cast<std::size_t>(out_features) * k);
    std::vector<bf16_t> host_input(static_cast<std::size_t>(rows) * k);
    uint32_t s = seed;
    for (bf16_t& v : host_weight) {
        const float u = static_cast<float>(lcg(s) >> 8) / static_cast<float>(1u << 24);
        v = bf16_t{f32_to_bf16_rne((u * 2.0f - 1.0f) * 0.05f)};
    }
    for (bf16_t& v : host_input) {
        const float u = static_cast<float>(lcg(s) >> 8) / static_cast<float>(1u << 24);
        v = bf16_t{f32_to_bf16_rne((u * 2.0f - 1.0f) * 2.0f)};
    }

    std::vector<bf16_t> reference(static_cast<std::size_t>(rows) * out_features);
    for (uint32_t r = 0; r < rows; ++r) {
        const bf16_t* x_row = host_input.data() + static_cast<std::size_t>(r) * k;
        for (uint32_t o = 0; o < out_features; ++o) {
            const bf16_t* w_row = host_weight.data() + static_cast<std::size_t>(o) * k;
            float acc = 0.0f;
            for (uint32_t i = 0; i < k; ++i) {
                acc += bf16_to_f32(w_row[i]) * bf16_to_f32(x_row[i]);
            }
            reference[static_cast<std::size_t>(r) * out_features + o] =
                bf16_t{f32_to_bf16_rne(acc)};
        }
    }

    shelf::gpu::Tensor dev_weight;
    shelf::gpu::Tensor dev_input;
    shelf::gpu::Tensor dev_out;
    check(alloc_bf16(arena, {out_features, k}, dev_weight), "weight alloc");
    check(alloc_bf16(arena, {rows, k}, dev_input), "input alloc");
    check(alloc_bf16(arena, {rows, out_features}, dev_out), "out alloc");
    check(hipMemcpy(dev_weight.data<bf16_t>(), host_weight.data(),
                    host_weight.size() * sizeof(bf16_t), hipMemcpyHostToDevice) == hipSuccess,
          "weight upload");
    check(hipMemcpy(dev_input.data<bf16_t>(), host_input.data(),
                    host_input.size() * sizeof(bf16_t), hipMemcpyHostToDevice) == hipSuccess,
          "input upload");

    shelf::execution::Bf16GemmSelectorInput lin;
    lin.rows = rows;
    lin.out_features = out_features;
    lin.k = k;
    lin.input_row_stride = k;
    lin.weight_aligned16 =
        (reinterpret_cast<uintptr_t>(dev_weight.data<void>()) & 15u) == 0u;
    lin.input_aligned16 = (reinterpret_cast<uintptr_t>(dev_input.data<void>()) & 15u) == 0u;
    lin.allow_dot2_rows = allow_dot2;
    const auto config = shelf::execution::select_bf16_gemm_config(lin);
    check(config.has_value(), "geometry " + std::to_string(out_features) + "x" +
                                  std::to_string(k) + " is in the bf16 allowlist");
    if (!config.has_value()) return;

    const hipError_t err = shelf::kernel::launch_gemm_bf16(
        *config, dev_weight.data<bf16_t>(), dev_input.data<bf16_t>(), dev_out.data<bf16_t>(),
        shelf::kernel::Bf16GemmOutputDType::BF16, rows, out_features, k, k, out_features,
        stream);
    check(err == hipSuccess, "gemm launch");
    if (err != hipSuccess) return;
    check(hipStreamSynchronize(stream) == hipSuccess, "gemm sync");

    std::vector<bf16_t> got(static_cast<std::size_t>(rows) * out_features);
    check(hipMemcpy(got.data(), dev_out.data<bf16_t>(), got.size() * sizeof(bf16_t),
                    hipMemcpyDeviceToHost) == hipSuccess,
          "out readback");

    double diff = 0.0;
    double ref_norm = 0.0;
    double dot = 0.0;
    double got_norm = 0.0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const double a = bf16_to_f32(reference[i]);
        const double b = bf16_to_f32(got[i]);
        check(std::isfinite(a) && std::isfinite(b), "finite");
        const double d = a - b;
        diff += d * d;
        ref_norm += a * a;
        dot += a * b;
        got_norm += b * b;
    }
    const double rel_l2 = ref_norm > 0.0 ? std::sqrt(diff / ref_norm) : 0.0;
    const double cosine = (ref_norm > 0.0 && got_norm > 0.0)
                              ? dot / (std::sqrt(ref_norm) * std::sqrt(got_norm))
                              : 1.0;
    std::printf("  geometry %ux%u rows=%u rel_l2=%.3e cosine=%.6f\n", out_features, k, rows,
                rel_l2, cosine);
    check(rel_l2 <= 1.0e-2, "geometry rel_l2 " + std::to_string(rel_l2));
    check(cosine >= 0.9999, "geometry cosine " + std::to_string(cosine));
}

int main() {
    auto arena_result = shelf::gpu::GpuArena::create(0, 2ull * 1024ull * 1024ull * 1024ull);
    if (!arena_result.ok()) {
        std::printf("SKIP: arena (%s)\n", arena_result.status().message().c_str());
        return 77;
    }
    shelf::gpu::GpuArena arena = arena_result.release();
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        std::printf("FAIL hipStreamCreate\n");
        return 1;
    }

    run_geometry(arena, stream, 1280u, 5120u, 1u, 11u, false);
    run_geometry(arena, stream, 1280u, 5120u, 8u, 22u, false);
    run_geometry(arena, stream, 5120u, 25600u, 1u, 33u, false);
    run_geometry(arena, stream, 5120u, 25600u, 8u, 44u, false);
    run_geometry(arena, stream, 4096u, 5120u, 1u, 55u, false);
    run_geometry(arena, stream, 4096u, 5120u, 2u, 66u, false);
    run_geometry(arena, stream, 4096u, 5120u, 8u, 77u, false);
    run_geometry(arena, stream, 5120u, 4096u, 1u, 88u, false);
    run_geometry(arena, stream, 5120u, 4096u, 2u, 99u, false);
    run_geometry(arena, stream, 5120u, 4096u, 8u, 111u, false);
    run_geometry(arena, stream, 5120u, 25600u, 17u, 222u, true);
    run_geometry(arena, stream, 5120u, 25600u, 33u, 333u, true);
    run_geometry(arena, stream, 5120u, 25600u, 18u, 444u, true);

    shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    const shelf::Status shutdown = arena.shutdown();
    check(shutdown.ok(), "arena shutdown: " + shutdown.message());

    std::printf("test_dflash2_bf16_geometry: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
