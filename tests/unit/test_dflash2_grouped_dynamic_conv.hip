#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/kernels/dflash2/grouped_dynamic_conv.h>
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

static uint32_t lcg(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return s;
}

static float rand_small(uint32_t& s, float scale) {
    const float u = static_cast<float>(lcg(s) >> 8) / static_cast<float>(1u << 24);
    return (u * 2.0f - 1.0f) * scale;
}

struct HostBuffers {
    std::vector<bf16_t> input;
    std::vector<bf16_t> dynamic;
    std::vector<bf16_t> base;
    std::vector<bf16_t> reference;
};

static void reference_conv(
    const std::vector<bf16_t>& input,
    const std::vector<bf16_t>& dynamic,
    const std::vector<bf16_t>& base,
    uint32_t rows,
    uint32_t hidden,
    uint32_t group_size,
    uint32_t phase,
    std::vector<bf16_t>& out) {
    const uint32_t groups = hidden / group_size;
    const uint32_t row_stride = 4u * groups;
    out.resize(static_cast<std::size_t>(rows) * hidden);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t f = 0; f < hidden; ++f) {
            const uint32_t group = f / group_size;
            const bf16_t* dyn = dynamic.data() + static_cast<std::size_t>(r) * row_stride +
                                static_cast<std::size_t>(phase) * 2u * groups + group;
            const bf16_t* base_phase =
                base.data() + static_cast<std::size_t>(phase) * 2u * hidden;
            const float v0 = bf16_to_f32(input[static_cast<std::size_t>(r) * hidden + f]);
            const float t0 = round_to_bf16_f32(bf16_to_f32(base_phase[f]) * v0);
            const float o0 = round_to_bf16_f32(t0 + bf16_to_f32(dyn[0]) * v0);
            float acc = o0;
            if (r >= 1u) {
                const float v1 =
                    bf16_to_f32(input[static_cast<std::size_t>(r - 1u) * hidden + f]);
                const float t1 =
                    round_to_bf16_f32(bf16_to_f32(base_phase[hidden + f]) * v1);
                const float o1 = round_to_bf16_f32(acc + t1);
                acc = round_to_bf16_f32(o1 + bf16_to_f32(dyn[groups]) * v1);
            }
            out[static_cast<std::size_t>(r) * hidden + f] = bf16_t{f32_to_bf16_rne(acc)};
        }
    }
}

static void compare(
    const std::vector<bf16_t>& got,
    const std::vector<bf16_t>& want,
    const std::string& tag,
    bool bit_exact,
    double rel_l2_max,
    double cosine_min) {
    check(got.size() == want.size(), tag + " size");
    if (got.size() != want.size()) return;
    double diff = 0.0;
    double ref_norm = 0.0;
    double dot = 0.0;
    double got_norm = 0.0;
    std::size_t worst = 0;
    double worst_abs = 0.0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        const double a = bf16_to_f32(want[i]);
        const double b = bf16_to_f32(got[i]);
        check(std::isfinite(a) && std::isfinite(b), tag + " finite");
        if (bit_exact && want[i].data != got[i].data) {
            check(false, tag + " bit-exact mismatch at " + std::to_string(i) + " got=" +
                              std::to_string(bf16_to_f32(got[i])) + " want=" +
                              std::to_string(bf16_to_f32(want[i])));
            return;
        }
        const double d = a - b;
        if (std::fabs(d) > worst_abs) {
            worst_abs = std::fabs(d);
            worst = i;
        }
        diff += d * d;
        ref_norm += a * a;
        dot += a * b;
        got_norm += b * b;
    }
    const double rel_l2 = ref_norm > 0.0 ? std::sqrt(diff / ref_norm) : 0.0;
    const double cosine = (ref_norm > 0.0 && got_norm > 0.0)
                              ? dot / (std::sqrt(ref_norm) * std::sqrt(got_norm))
                              : 1.0;
    std::printf("  %-34s rel_l2=%.3e cosine=%.6f max_abs=%.3e worst=%zu\n", tag.c_str(),
                rel_l2, cosine, worst_abs, worst);
    if (!bit_exact) {
        check(rel_l2 <= rel_l2_max, tag + " rel_l2 " + std::to_string(rel_l2));
        check(cosine >= cosine_min, tag + " cosine " + std::to_string(cosine));
    }
}

static void run_case(
    ps::gpu::GpuArena& arena,
    hipStream_t stream,
    uint32_t rows,
    uint32_t hidden,
    uint32_t group_size,
    uint32_t phase,
    bool exact_values,
    uint32_t seed) {
    const uint32_t groups = hidden / group_size;
    HostBuffers h;
    h.input.resize(static_cast<std::size_t>(rows) * hidden);
    h.dynamic.resize(static_cast<std::size_t>(rows) * 4u * groups);
    h.base.resize(static_cast<std::size_t>(2u) * 2u * hidden);

    uint32_t s = seed;
    for (bf16_t& v : h.input) {
        v = bf16_t{f32_to_bf16_rne(exact_values ? static_cast<float>(static_cast<int>(lcg(s) % 17u) - 8)
                                                : rand_small(s, 2.0f))};
    }
    for (bf16_t& v : h.dynamic) {
        v = bf16_t{f32_to_bf16_rne(exact_values ? static_cast<float>(static_cast<int>(lcg(s) % 9u) - 4)
                                                : rand_small(s, 0.5f))};
    }
    for (bf16_t& v : h.base) {
        v = bf16_t{f32_to_bf16_rne(exact_values ? static_cast<float>(static_cast<int>(lcg(s) % 7u) - 3)
                                                : rand_small(s, 1.0f))};
    }

    reference_conv(h.input, h.dynamic, h.base, rows, hidden, group_size, phase, h.reference);

    ps::gpu::Tensor dev_input;
    ps::gpu::Tensor dev_dynamic;
    ps::gpu::Tensor dev_base;
    ps::gpu::Tensor dev_out;
    check(alloc_bf16(arena, {rows, hidden}, dev_input), "input alloc");
    check(alloc_bf16(arena, {rows, 4u * groups}, dev_dynamic), "dynamic alloc");
    check(alloc_bf16(arena, {2u, 2u, hidden}, dev_base), "base alloc");
    check(alloc_bf16(arena, {rows, hidden}, dev_out), "out alloc");
    check(hipMemcpy(dev_input.data<bf16_t>(), h.input.data(),
                    h.input.size() * sizeof(bf16_t), hipMemcpyHostToDevice) == hipSuccess,
          "input upload");
    check(hipMemcpy(dev_dynamic.data<bf16_t>(), h.dynamic.data(),
                    h.dynamic.size() * sizeof(bf16_t), hipMemcpyHostToDevice) == hipSuccess,
          "dynamic upload");
    check(hipMemcpy(dev_base.data<bf16_t>(), h.base.data(), h.base.size() * sizeof(bf16_t),
                    hipMemcpyHostToDevice) == hipSuccess,
          "base upload");

    const hipError_t err = ps::kernel::launch_dflash2_grouped_dynamic_conv(
        dev_input.data<bf16_t>(), dev_dynamic.data<bf16_t>(), dev_base.data<bf16_t>(),
        dev_out.data<bf16_t>(), rows, hidden, group_size, phase, stream);
    check(err == hipSuccess, "conv launch");
    if (err != hipSuccess) return;
    check(hipStreamSynchronize(stream) == hipSuccess, "conv sync");

    std::vector<bf16_t> got(static_cast<std::size_t>(rows) * hidden);
    check(hipMemcpy(got.data(), dev_out.data<bf16_t>(), got.size() * sizeof(bf16_t),
                    hipMemcpyDeviceToHost) == hipSuccess,
          "out readback");

    const std::string tag = "rows=" + std::to_string(rows) + " hidden=" + std::to_string(hidden) +
                            " group=" + std::to_string(group_size) + " phase=" +
                            std::to_string(phase) + (exact_values ? " exact" : " random");
    compare(got, h.reference, tag, exact_values, 2.0e-3, 0.9999);
}

static void run_structural_checks(ps::gpu::GpuArena& arena, hipStream_t stream) {
    const uint32_t hidden = 16u;
    const uint32_t group_size = 4u;
    const uint32_t groups = hidden / group_size;

    HostBuffers h;
    h.input.assign(2u * hidden, bf16_t{0});
    h.dynamic.assign(2u * 4u * groups, bf16_t{0});
    h.base.assign(2u * 2u * hidden, bf16_t{0});

    for (uint32_t f = 0; f < hidden; ++f) {
        h.input[f] = bf16_t{f32_to_bf16_rne(2.0f)};
        h.base[0u * 2u * hidden + f] = bf16_t{f32_to_bf16_rne(1.0f)};
        h.base[0u * 2u * hidden + hidden + f] = bf16_t{f32_to_bf16_rne(1000.0f)};
    }
    for (uint32_t g = 0; g < groups; ++g) {
        for (uint32_t r = 0; r < 2u; ++r) {
            h.dynamic[static_cast<std::size_t>(r) * 4u * groups + 0u * groups + g] =
                bf16_t{f32_to_bf16_rne(1.0f)};
            h.dynamic[static_cast<std::size_t>(r) * 4u * groups + 1u * groups + g] =
                bf16_t{f32_to_bf16_rne(500.0f)};
        }
    }

    ps::gpu::Tensor dev_input;
    ps::gpu::Tensor dev_dynamic;
    ps::gpu::Tensor dev_base;
    ps::gpu::Tensor dev_out;
    check(alloc_bf16(arena, {2u, hidden}, dev_input), "struct input alloc");
    check(alloc_bf16(arena, {2u, 4u * groups}, dev_dynamic), "struct dynamic alloc");
    check(alloc_bf16(arena, {2u, 2u, hidden}, dev_base), "struct base alloc");
    check(alloc_bf16(arena, {2u, hidden}, dev_out), "struct out alloc");
    check(hipMemcpy(dev_input.data<bf16_t>(), h.input.data(),
                    h.input.size() * sizeof(bf16_t), hipMemcpyHostToDevice) == hipSuccess,
          "struct input upload");
    check(hipMemcpy(dev_dynamic.data<bf16_t>(), h.dynamic.data(),
                    h.dynamic.size() * sizeof(bf16_t), hipMemcpyHostToDevice) == hipSuccess,
          "struct dynamic upload");
    check(hipMemcpy(dev_base.data<bf16_t>(), h.base.data(), h.base.size() * sizeof(bf16_t),
                    hipMemcpyHostToDevice) == hipSuccess,
          "struct base upload");

    check(ps::kernel::launch_dflash2_grouped_dynamic_conv(
              dev_input.data<bf16_t>(), dev_dynamic.data<bf16_t>(), dev_base.data<bf16_t>(),
              dev_out.data<bf16_t>(), 2u, hidden, group_size, 0u, stream) == hipSuccess,
          "struct launch");
    check(hipStreamSynchronize(stream) == hipSuccess, "struct sync");

    std::vector<bf16_t> got(2u * hidden);
    check(hipMemcpy(got.data(), dev_out.data<bf16_t>(), got.size() * sizeof(bf16_t),
                    hipMemcpyDeviceToHost) == hipSuccess,
          "struct readback");

    for (uint32_t f = 0; f < hidden; ++f) {
        const bf16_t expect_row0 = bf16_t{f32_to_bf16_rne((1.0f + 1.0f) * 2.0f)};
        const bf16_t expect_row1 = bf16_t{f32_to_bf16_rne((1000.0f + 500.0f) * 2.0f)};
        check(got[f].data == expect_row0.data,
              "row0 ignores offset1 at f=" + std::to_string(f) + " got=" +
                  std::to_string(bf16_to_f32(got[f])));
        check(got[hidden + f].data == expect_row1.data,
              "row1 reads row0 at f=" + std::to_string(f) + " got=" +
                  std::to_string(bf16_to_f32(got[hidden + f])));
    }

    std::vector<bf16_t> single(1u * hidden);
    check(hipMemcpy(single.data(), dev_out.data<bf16_t>(), single.size() * sizeof(bf16_t),
                    hipMemcpyDeviceToHost) == hipSuccess,
          "row0 readback");
    check(ps::kernel::launch_dflash2_grouped_dynamic_conv(
              dev_input.data<bf16_t>(), dev_dynamic.data<bf16_t>(), dev_base.data<bf16_t>(),
              dev_out.data<bf16_t>(), 1u, hidden, group_size, 0u, stream) == hipSuccess,
          "rows1 launch");
    check(hipStreamSynchronize(stream) == hipSuccess, "rows1 sync");
    std::vector<bf16_t> got1(hidden);
    check(hipMemcpy(got1.data(), dev_out.data<bf16_t>(), got1.size() * sizeof(bf16_t),
                    hipMemcpyDeviceToHost) == hipSuccess,
          "rows1 readback");
    for (uint32_t f = 0; f < hidden; ++f) {
        check(got1[f].data == single[f].data,
              "rows=1 equals row0 at f=" + std::to_string(f));
    }

    HostBuffers h2 = h;
    for (std::size_t i = 0; i < h2.dynamic.size(); ++i) {
        const uint32_t within = static_cast<uint32_t>(i % (4u * groups));
        const uint32_t group = within % groups;
        const uint32_t offset = (within / groups) % 2u;
        if (group != 2u || offset != 0u) {
            h2.dynamic[i] = bf16_t{0};
        }
    }
    check(hipMemcpy(dev_dynamic.data<bf16_t>(), h2.dynamic.data(),
                    h2.dynamic.size() * sizeof(bf16_t), hipMemcpyHostToDevice) == hipSuccess,
          "group dynamic upload");
    check(ps::kernel::launch_dflash2_grouped_dynamic_conv(
              dev_input.data<bf16_t>(), dev_dynamic.data<bf16_t>(), dev_base.data<bf16_t>(),
              dev_out.data<bf16_t>(), 2u, hidden, group_size, 0u, stream) == hipSuccess,
          "group launch");
    check(hipStreamSynchronize(stream) == hipSuccess, "group sync");
    std::vector<bf16_t> got_group(2u * hidden);
    check(hipMemcpy(got_group.data(), dev_out.data<bf16_t>(),
                    got_group.size() * sizeof(bf16_t), hipMemcpyDeviceToHost) == hipSuccess,
          "group readback");
    for (uint32_t f = 0; f < hidden; ++f) {
        const uint32_t group = f / group_size;
        const bf16_t want = bf16_t{f32_to_bf16_rne((1.0f + (group == 2u ? 1.0f : 0.0f)) * 2.0f)};
        check(got_group[f].data == want.data,
              "group isolation at f=" + std::to_string(f) + " got=" +
                  std::to_string(bf16_to_f32(got_group[f])));
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

    for (uint32_t phase = 0; phase < 2u; ++phase) {
        run_case(arena, stream, 1u, 32u, 8u, phase, true, 11u + phase);
        run_case(arena, stream, 2u, 32u, 8u, phase, true, 22u + phase);
        run_case(arena, stream, 3u, 32u, 8u, phase, true, 33u + phase);
        run_case(arena, stream, 8u, 32u, 8u, phase, true, 44u + phase);
        run_case(arena, stream, 1u, 5120u, 16u, phase, true, 55u + phase);
        run_case(arena, stream, 2u, 5120u, 16u, phase, true, 66u + phase);
        run_case(arena, stream, 3u, 5120u, 16u, phase, true, 77u + phase);
        run_case(arena, stream, 8u, 5120u, 16u, phase, true, 88u + phase);
        run_case(arena, stream, 1u, 32u, 8u, phase, false, 111u + phase);
        run_case(arena, stream, 2u, 32u, 8u, phase, false, 122u + phase);
        run_case(arena, stream, 8u, 32u, 8u, phase, false, 133u + phase);
        run_case(arena, stream, 8u, 5120u, 16u, phase, false, 144u + phase);
    }
    run_structural_checks(arena, stream);

    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    const ps::Status shutdown = arena.shutdown();
    check(shutdown.ok(), "arena shutdown: " + shutdown.message());

    std::printf("test_dflash2_grouped_dynamic_conv: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
