#include <phaseshift/kernels/optimized/rmsnorm.h>
#include <phaseshift/kernels/optimized/elementwise.h>
#include <phaseshift/kernels/optimized/activation_quantize.h>
#include <phaseshift/kernels/optimized/resid_rmsnorm_quantize.h>
#include <phaseshift/runtime/program/int8_activation_workspace.h>
#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr uint32_t kF = 5120u;
constexpr float kEps = 1e-6f;

int g_passed = 0;
int g_failed = 0;

uint32_t h32(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return static_cast<uint32_t>(x);
}

uint16_t bf16_bits_of(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return static_cast<uint16_t>(u >> 16);
}

float elem_value(uint64_t i) {
    const uint32_t r = h32(i * 2654435761ull + 77ull);
    const int32_t v = static_cast<int32_t>(r % 4001u) - 2000;
    float f = static_cast<float>(v) / 128.0f;
    if (i % 1543u == 5u) f = 0.0f;
    if (i % 1543u == 17u) f = -0.0f;
    if (i % 6091u == 9u) f = 1.0e-28f;
    return f;
}

float nasty_value(uint64_t i) {
    const uint32_t r = h32(i * 40503ull + 99991ull);
    switch (r % 16u) {
        case 0:
            return std::numeric_limits<float>::quiet_NaN();
        case 1:
            return std::numeric_limits<float>::infinity();
        case 2:
            return -std::numeric_limits<float>::infinity();
        case 3:
            return (r & 32u) ? 234.5f : -187.25f;
        case 4:
            return 1.0e-30f;
        case 5:
            return 6.55e4f;
        default:
            break;
    }
    const int32_t v = static_cast<int32_t>(r % 4001u) - 2000;
    return static_cast<float>(v) / 128.0f;
}

struct Buf {
    void* dev = nullptr;
    size_t bytes = 0;
    ~Buf() {
        if (dev) ps::gpu::discard_cleanup_result(hipFree(dev));
    }
    bool alloc(size_t b) {
        bytes = b;
        return hipMalloc(&dev, b) == hipSuccess;
    }
};

struct Layout {
    uint32_t rows;
    uint32_t stride;
    uint32_t base_offset_elems;
    bool nasty_inputs;
};

void check_case(const Layout& lay, bool& ok) {
    const uint32_t rows = lay.rows;
    const uint32_t rs = lay.stride;
    const size_t off_e = lay.base_offset_elems;
    const size_t elems = off_e + static_cast<size_t>(rows) * rs;

    std::vector<uint16_t> h_a(elems);
    std::vector<uint16_t> h_b(elems);
    for (size_t i = 0; i < elems; ++i) {
        const float fa = lay.nasty_inputs ? nasty_value(i) : elem_value(i);
        const float fb = lay.nasty_inputs ? nasty_value(i + 777777u) : elem_value(i + 555555u);
        h_a[i] = bf16_bits_of(fa);
        h_b[i] = bf16_bits_of(fb * 0.5f);
    }
    std::vector<uint16_t> h_w(kF);
    for (uint32_t i = 0; i < kF; ++i)
        h_w[i] = bf16_bits_of(elem_value(i + 99991u) * 0.02f + 0.001f);

    const ::ps::runtime::Int8ActivationWorkspaceLayout aw =
        ::ps::runtime::Int8ActivationWorkspaceLayout::make(kF, rows);
    const size_t codes_bytes = aw.codes_bytes();
    const size_t scales_bytes = aw.scales_bytes();

    Buf a, b, w, resid_s, normed_s, resid_f, normed_f, codes_s, codes_f;
    if (!a.alloc(elems * 2) || !b.alloc(elems * 2) || !w.alloc(kF * 2) ||
        !resid_s.alloc(elems * 2) || !normed_s.alloc(elems * 2) ||
        !resid_f.alloc(elems * 2) || !normed_f.alloc(elems * 2) ||
        !codes_s.alloc(codes_bytes + scales_bytes) ||
        !codes_f.alloc(codes_bytes + scales_bytes)) {
        ok = false;
        return;
    }
    if (hipMemcpy(a.dev, h_a.data(), elems * 2, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(b.dev, h_b.data(), elems * 2, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(w.dev, h_w.data(), kF * 2, hipMemcpyHostToDevice) != hipSuccess) {
        ok = false;
        return;
    }
    const size_t off = off_e * 2u;
    auto* a_sr = reinterpret_cast<::ps::bf16_t*>(static_cast<uint8_t*>(a.dev) + off);
    auto* b_sr = reinterpret_cast<::ps::bf16_t*>(static_cast<uint8_t*>(b.dev) + off);
    auto* resid_sr =
        reinterpret_cast<::ps::bf16_t*>(static_cast<uint8_t*>(resid_s.dev) + off);
    auto* normed_sr =
        reinterpret_cast<::ps::bf16_t*>(static_cast<uint8_t*>(normed_s.dev) + off);
    auto* resid_fr =
        reinterpret_cast<::ps::bf16_t*>(static_cast<uint8_t*>(resid_f.dev) + off);
    auto* normed_fr =
        reinterpret_cast<::ps::bf16_t*>(static_cast<uint8_t*>(normed_f.dev) + off);
    auto* cs_s = static_cast<uint8_t*>(codes_s.dev);
    auto* cs_f = static_cast<uint8_t*>(codes_f.dev);
    float* scales_s = reinterpret_cast<float*>(cs_s + aw.scales_offset);
    float* scales_f = reinterpret_cast<float*>(cs_f + aw.scales_offset);

    if (hipMemset(resid_s.dev, 0, elems * 2) != hipSuccess ||
        hipMemset(normed_s.dev, 0, elems * 2) != hipSuccess ||
        hipMemset(resid_f.dev, 0, elems * 2) != hipSuccess ||
        hipMemset(normed_f.dev, 0, elems * 2) != hipSuccess ||
        hipMemset(codes_s.dev, 0, codes_bytes + scales_bytes) != hipSuccess ||
        hipMemset(codes_f.dev, 0, codes_bytes + scales_bytes) != hipSuccess) {
        ok = false;
        return;
    }

    hipStream_t s = nullptr;
    if (hipStreamCreate(&s) != hipSuccess) {
        ok = false;
        return;
    }

    const hipError_t e0 = ::ps::kernel::launch_residual_add_bf16(
        a_sr, rs, b_sr, rs, resid_sr, rs, rows, kF, s);
    const hipError_t e1 = ::ps::kernel::launch_rmsnorm(
        resid_sr, ::ps::kernel::RmsNormDataType::BF16, w.dev,
        ::ps::kernel::RmsNormDataType::BF16,
        ::ps::kernel::RmsNormWeightLayout::PER_FEATURE,
        ::ps::kernel::RmsNormWeightMode::ONE_PLUS, normed_sr,
        ::ps::kernel::RmsNormDataType::BF16, rows, kF, kF, rs, rs, kEps, s);
    const hipError_t e2 = ::ps::kernel::launch_activation_quantize_e4m3(
        normed_sr, cs_s, scales_s, rows, kF, kF, rs, aw.code_row_stride_bytes,
        aw.scale_row_stride_bytes, s, nullptr);

    ::ps::kernel::FusedResidRmsNormQuantizeArgs args{};
    args.resid_a = a_sr;
    args.resid_b = b_sr;
    args.residual = resid_fr;
    args.normed = normed_fr;
    args.weight = w.dev;
    args.codes = cs_f;
    args.scales = scales_f;
    args.rows = rows;
    args.features = kF;
    args.group_size = kF;
    args.resid_a_row_stride = rs;
    args.resid_b_row_stride = rs;
    args.residual_row_stride = rs;
    args.normed_row_stride = rs;
    args.code_row_stride = aw.code_row_stride_bytes;
    args.scale_row_stride = aw.scale_row_stride_bytes;
    args.eps = kEps;
    const hipError_t e3 = ::ps::kernel::launch_fused_resid_rmsnorm_quantize(args, s);

    bool launch_ok =
        e0 == hipSuccess && e1 == hipSuccess && e2 == hipSuccess && e3 == hipSuccess;
    if (!launch_ok) {
        std::printf("test_fused_resid_rmsnorm_quantize: launch fail rows=%u\n", rows);
        ok = false;
        ps::gpu::discard_cleanup_result(hipStreamDestroy(s));
        return;
    }
    if (hipStreamSynchronize(s) != hipSuccess) {
        ok = false;
        ps::gpu::discard_cleanup_result(hipStreamDestroy(s));
        return;
    }

    std::vector<uint8_t> got_res(elems * 2);
    std::vector<uint8_t> want_res(elems * 2);
    std::vector<uint8_t> got_out(elems * 2);
    std::vector<uint8_t> want_out(elems * 2);
    std::vector<uint8_t> got_cs(codes_bytes + scales_bytes);
    std::vector<uint8_t> want_cs(codes_bytes + scales_bytes);
    if (hipMemcpy(got_res.data(), resid_f.dev, elems * 2, hipMemcpyDeviceToHost) !=
            hipSuccess ||
        hipMemcpy(want_res.data(), resid_s.dev, elems * 2, hipMemcpyDeviceToHost) !=
            hipSuccess ||
        hipMemcpy(got_out.data(), normed_f.dev, elems * 2, hipMemcpyDeviceToHost) !=
            hipSuccess ||
        hipMemcpy(want_out.data(), normed_s.dev, elems * 2, hipMemcpyDeviceToHost) !=
            hipSuccess ||
        hipMemcpy(got_cs.data(), codes_f.dev, codes_bytes + scales_bytes,
                  hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(want_cs.data(), codes_s.dev, codes_bytes + scales_bytes,
                  hipMemcpyDeviceToHost) != hipSuccess) {
        ok = false;
        ps::gpu::discard_cleanup_result(hipStreamDestroy(s));
        return;
    }
    const bool res_same =
        std::memcmp(got_res.data(), want_res.data(), elems * 2) == 0;
    const bool out_same =
        std::memcmp(got_out.data(), want_out.data(), elems * 2) == 0;
    const bool cs_same =
        std::memcmp(got_cs.data(), want_cs.data(), codes_bytes + scales_bytes) == 0;
    if (!res_same || !out_same || !cs_same) {
        size_t first = elems * 2;
        for (size_t i = 0; i < elems * 2; ++i) {
            if (got_out[i] != want_out[i]) {
                first = i;
                break;
            }
        }
        size_t first_cs = codes_bytes + scales_bytes;
        for (size_t i = 0; i < codes_bytes + scales_bytes; ++i) {
            if (got_cs[i] != want_cs[i]) {
                first_cs = i;
                break;
            }
        }
        std::printf("test_fused_resid_rmsnorm_quantize: MISMATCH rows=%u stride=%u off=%u resid=%d normed=%d codes=%d first=%zu got=%02x want=%02x firstcs=%zu\n",
                    rows, rs, static_cast<uint32_t>(off_e), (int)res_same,
                    (int)out_same, (int)cs_same, first,
                    first < elems * 2 ? got_out[first] : 0,
                    first < elems * 2 ? want_out[first] : 0, first_cs);
        ok = false;
        ps::gpu::discard_cleanup_result(hipStreamDestroy(s));
        return;
    }
    ps::gpu::discard_cleanup_result(hipStreamDestroy(s));
}

}

int main() {
    bool ok = true;
    const Layout lays[] = {
        {1, 5120, 0, false}, {1, 5120, 8, false}, {2, 5120, 0, false},
        {5, 5120, 0, false}, {1, 5136, 0, false}, {3, 5200, 16, false},
        {1, 5120, 0, true}, {2, 5120, 0, true}, {5, 5120, 0, true},
    };
    for (const auto& lay : lays) check_case(lay, ok);
    if (ok) {
        ++g_passed;
        std::printf("test_fused_resid_rmsnorm_quantize: PASS (split == fused byte-exact)\n");
    } else {
        ++g_failed;
        std::printf("test_fused_resid_rmsnorm_quantize: FAIL\n");
    }
    return g_failed == 0 ? 0 : 1;
}
