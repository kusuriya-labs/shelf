#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/kernels/dflash2/rope.h>
#include <hip/hip_runtime.h>

#include <cmath>
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

static float round_to_bf16_f32(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    if ((u & 0x7F800000u) == 0x7F800000u) {
        u = ((u & 0x007FFFFFu) != 0u) ? 0x7FC00000u : (u & 0xFF800000u);
    } else {
        u += 0x7FFFu + ((u >> 16u) & 1u);
        u &= 0xFFFF0000u;
    }
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

static uint32_t lcg(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return s;
}

static void reference_rope(
    const std::vector<bf16_t>& input,
    uint32_t rows,
    uint32_t features,
    uint32_t head_dim,
    uint32_t position_start,
    float theta,
    std::vector<bf16_t>& out) {
    const uint32_t heads = features / head_dim;
    const uint32_t half = head_dim / 2u;
    out.assign(static_cast<std::size_t>(rows) * features, bf16_t{0});
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t h = 0; h < heads; ++h) {
            const bf16_t* src =
                input.data() + static_cast<std::size_t>(r) * features + h * head_dim;
            bf16_t* dst = out.data() + static_cast<std::size_t>(r) * features + h * head_dim;
            for (uint32_t i = 0; i < half; ++i) {
                const float exponent = -2.0f * static_cast<float>(i) / static_cast<float>(head_dim);
                const float inv_freq = std::pow(theta, exponent);
                const float angle = static_cast<float>(position_start + r) * inv_freq;
                const float c = round_to_bf16_f32(std::cos(angle));
                const float s = round_to_bf16_f32(std::sin(angle));
                const float x1 = bf16_to_f32(src[i]);
                const float x2 = bf16_to_f32(src[i + half]);
                const float p1 = round_to_bf16_f32(x1 * c);
                const float p2 = round_to_bf16_f32(x2 * s);
                const float p3 = round_to_bf16_f32(x2 * c);
                const float p4 = round_to_bf16_f32(x1 * s);
                dst[i] = bf16_t{f32_to_bf16_rne(p1 - p2)};
                dst[i + half] = bf16_t{f32_to_bf16_rne(p3 + p4)};
            }
        }
    }
}

static void compare(
    const std::vector<bf16_t>& got,
    const std::vector<bf16_t>& want,
    const std::string& tag) {
    check(got.size() == want.size(), tag + " size");
    if (got.size() != want.size()) return;
    double diff = 0.0;
    double ref_norm = 0.0;
    double dot = 0.0;
    double got_norm = 0.0;
    double max_abs = 0.0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        const double a = bf16_to_f32(want[i]);
        const double b = bf16_to_f32(got[i]);
        check(std::isfinite(a) && std::isfinite(b), tag + " finite");
        const double d = a - b;
        max_abs = std::fabs(d) > max_abs ? std::fabs(d) : max_abs;
        diff += d * d;
        ref_norm += a * a;
        dot += a * b;
        got_norm += b * b;
    }
    const double rel_l2 = ref_norm > 0.0 ? std::sqrt(diff / ref_norm) : 0.0;
    const double cosine = (ref_norm > 0.0 && got_norm > 0.0)
                              ? dot / (std::sqrt(ref_norm) * std::sqrt(got_norm))
                              : 1.0;
    std::printf("  %-28s rel_l2=%.3e cosine=%.6f max_abs=%.3e\n", tag.c_str(), rel_l2,
                cosine, max_abs);
    check(rel_l2 <= 1.0e-3, tag + " rel_l2 " + std::to_string(rel_l2));
    check(cosine >= 0.9999, tag + " cosine " + std::to_string(cosine));
}

static void run_case(
    ps::gpu::GpuArena& arena,
    hipStream_t stream,
    uint32_t rows,
    uint32_t heads,
    uint32_t head_dim,
    uint32_t position_start,
    float theta,
    uint32_t seed) {
    const uint32_t features = heads * head_dim;
    std::vector<bf16_t> input(static_cast<std::size_t>(rows) * features);
    uint32_t s = seed;
    for (bf16_t& v : input) {
        const float u = static_cast<float>(lcg(s) >> 8) / static_cast<float>(1u << 24);
        v = bf16_t{f32_to_bf16_rne((u * 2.0f - 1.0f) * 2.0f)};
    }
    std::vector<bf16_t> want;
    reference_rope(input, rows, features, head_dim, position_start, theta, want);

    ps::gpu::Tensor dev_in;
    ps::gpu::Tensor dev_out;
    check(alloc_bf16(arena, {rows, features}, dev_in), "input alloc");
    check(alloc_bf16(arena, {rows, features}, dev_out), "out alloc");
    check(hipMemcpy(dev_in.data<bf16_t>(), input.data(), input.size() * sizeof(bf16_t),
                    hipMemcpyHostToDevice) == hipSuccess,
          "input upload");

    const hipError_t err = ps::kernel::launch_dflash2_rope_bf16(
        dev_in.data<bf16_t>(), features, dev_out.data<bf16_t>(), features, rows, features,
        head_dim, position_start, theta, stream);
    check(err == hipSuccess, "rope launch");
    if (err != hipSuccess) return;
    check(hipStreamSynchronize(stream) == hipSuccess, "rope sync");

    std::vector<bf16_t> got(static_cast<std::size_t>(rows) * features);
    check(hipMemcpy(got.data(), dev_out.data<bf16_t>(), got.size() * sizeof(bf16_t),
                    hipMemcpyDeviceToHost) == hipSuccess,
          "out readback");

    compare(got, want,
            "rows=" + std::to_string(rows) + " heads=" + std::to_string(heads) +
                " pos=" + std::to_string(position_start));
}

static void run_half_split_check(ps::gpu::GpuArena& arena, hipStream_t stream) {
    const uint32_t head_dim = 8u;
    const uint32_t half = head_dim / 2u;
    std::vector<bf16_t> input(head_dim, bf16_t{0});
    input[0] = bf16_t{f32_to_bf16_rne(1.0f)};

    ps::gpu::Tensor dev_in;
    ps::gpu::Tensor dev_out;
    check(alloc_bf16(arena, {1u, head_dim}, dev_in), "half input alloc");
    check(alloc_bf16(arena, {1u, head_dim}, dev_out), "half out alloc");
    check(hipMemcpy(dev_in.data<bf16_t>(), input.data(), input.size() * sizeof(bf16_t),
                    hipMemcpyHostToDevice) == hipSuccess,
          "half input upload");
    check(ps::kernel::launch_dflash2_rope_bf16(
              dev_in.data<bf16_t>(), head_dim, dev_out.data<bf16_t>(), head_dim, 1u, head_dim,
              head_dim, 3u, 10000000.0f, stream) == hipSuccess,
          "half launch");
    check(hipStreamSynchronize(stream) == hipSuccess, "half sync");
    std::vector<bf16_t> got(head_dim);
    check(hipMemcpy(got.data(), dev_out.data<bf16_t>(), got.size() * sizeof(bf16_t),
                    hipMemcpyDeviceToHost) == hipSuccess,
          "half readback");

    check(got[0].data != 0u, "half-split: first half driven by x[0]");
    check(got[half].data != 0u, "half-split: second half driven by x[0]");
    for (uint32_t i = 1; i < half; ++i) {
        check(got[i].data == 0u, "half-split: no interleaved pair at " + std::to_string(i));
        check(got[half + i].data == 0u,
              "half-split: no interleaved pair at " + std::to_string(half + i));
    }

    std::vector<bf16_t> identity(head_dim);
    check(hipMemcpy(identity.data(), dev_in.data<bf16_t>(), identity.size() * sizeof(bf16_t),
                    hipMemcpyDeviceToHost) == hipSuccess,
          "identity readback");
    check(ps::kernel::launch_dflash2_rope_bf16(
              dev_in.data<bf16_t>(), head_dim, dev_out.data<bf16_t>(), head_dim, 1u, head_dim,
              head_dim, 0u, 10000000.0f, stream) == hipSuccess,
          "position0 launch");
    check(hipStreamSynchronize(stream) == hipSuccess, "position0 sync");
    std::vector<bf16_t> at_zero(head_dim);
    check(hipMemcpy(at_zero.data(), dev_out.data<bf16_t>(), at_zero.size() * sizeof(bf16_t),
                    hipMemcpyDeviceToHost) == hipSuccess,
          "position0 readback");
    for (uint32_t i = 0; i < head_dim; ++i) {
        check(at_zero[i].data == identity[i].data,
              "position 0 is identity at " + std::to_string(i));
    }
}

int main() {
    auto arena_result = ps::gpu::GpuArena::create(0, 512ull * 1024ull * 1024ull);
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

    const float theta = 10000000.0f;
    run_case(arena, stream, 1u, 1u, 128u, 0u, theta, 11u);
    run_case(arena, stream, 1u, 1u, 128u, 1u, theta, 22u);
    run_case(arena, stream, 1u, 1u, 128u, 127u, theta, 33u);
    run_case(arena, stream, 1u, 1u, 128u, 4096u, theta, 44u);
    run_case(arena, stream, 4u, 32u, 128u, 0u, theta, 55u);
    run_case(arena, stream, 2u, 32u, 128u, 4096u, theta, 66u);
    run_case(arena, stream, 8u, 32u, 128u, 2047u, theta, 77u);
    run_case(arena, stream, 2u, 4u, 8u, 3u, theta, 88u);
    run_case(arena, stream, 3u, 8u, 64u, 1000u, theta, 99u);
    run_half_split_check(arena, stream);

    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    const ps::Status shutdown = arena.shutdown();
    check(shutdown.ok(), "arena shutdown: " + shutdown.message());

    std::printf("test_dflash2_rope: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
