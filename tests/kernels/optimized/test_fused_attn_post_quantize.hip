#include <phaseshift/kernels/optimized/elementwise.h>
#include <phaseshift/kernels/optimized/activation_quantize.h>
#include <phaseshift/kernels/optimized/attn_post_quantize.h>
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
            return 1.0e-40f;
        case 6:
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
    uint32_t features;
    uint32_t stride;
    uint32_t base_offset_elems;
    bool nasty_inputs;
    bool swap_mul;
};

void check_case(const Layout& lay, bool& ok) {
    const uint32_t rows = lay.rows;
    const uint32_t kF = lay.features;
    const uint32_t rs = lay.stride;
    const size_t off_e = lay.base_offset_elems;
    const size_t elems = off_e + static_cast<size_t>(rows) * rs;

    std::vector<uint16_t> h_gate(elems);
    std::vector<uint32_t> h_ctx(elems);
    for (size_t i = 0; i < elems; ++i) {
        const float fg = lay.nasty_inputs ? nasty_value(i) : elem_value(i);
        const float fc =
            lay.nasty_inputs ? nasty_value(i + 777777u) : elem_value(i + 555555u);
        h_gate[i] = bf16_bits_of(fg);
        std::memcpy(&h_ctx[i], &fc, 4);
    }

    const ::ps::runtime::Int8ActivationWorkspaceLayout aw =
        ::ps::runtime::Int8ActivationWorkspaceLayout::make(kF, rows);
    const size_t codes_bytes = aw.codes_bytes();
    const size_t scales_bytes = aw.scales_bytes();

    Buf gate, ctx, gs_s, att_s, att_f, codes_s, codes_f;
    if (!gate.alloc(elems * 2) || !ctx.alloc(elems * 4) || !gs_s.alloc(elems * 4) ||
        !att_s.alloc(elems * 2) || !att_f.alloc(elems * 2) ||
        !codes_s.alloc(codes_bytes + scales_bytes) ||
        !codes_f.alloc(codes_bytes + scales_bytes)) {
        ok = false;
        return;
    }
    if (hipMemcpy(gate.dev, h_gate.data(), elems * 2, hipMemcpyHostToDevice) !=
            hipSuccess ||
        hipMemcpy(ctx.dev, h_ctx.data(), elems * 4, hipMemcpyHostToDevice) !=
            hipSuccess) {
        ok = false;
        return;
    }
    auto* gate_r = reinterpret_cast<::ps::bf16_t*>(static_cast<uint8_t*>(gate.dev) + off_e * 2u);
    auto* ctx_r = reinterpret_cast<float*>(static_cast<uint8_t*>(ctx.dev) + off_e * 4u);
    auto* gs_sr = reinterpret_cast<float*>(static_cast<uint8_t*>(gs_s.dev) + off_e * 4u);
    auto* att_sr = reinterpret_cast<::ps::bf16_t*>(static_cast<uint8_t*>(att_s.dev) + off_e * 2u);
    auto* att_fr = reinterpret_cast<::ps::bf16_t*>(static_cast<uint8_t*>(att_f.dev) + off_e * 2u);
    auto* cs_s = static_cast<uint8_t*>(codes_s.dev);
    auto* cs_f = static_cast<uint8_t*>(codes_f.dev);
    float* scales_s = reinterpret_cast<float*>(cs_s + aw.scales_offset);
    float* scales_f = reinterpret_cast<float*>(cs_f + aw.scales_offset);

    if (hipMemset(gs_s.dev, 0, elems * 4) != hipSuccess ||
        hipMemset(att_s.dev, 0, elems * 2) != hipSuccess ||
        hipMemset(att_f.dev, 0, elems * 2) != hipSuccess ||
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

    const hipError_t e0 = ::ps::kernel::launch_sigmoid_bf16_f32(
        gate_r, rs, gs_sr, rs, rows, kF, s);
    const hipError_t e1 = lay.swap_mul
        ? ::ps::kernel::launch_mul_f32_f32_bf16(gs_sr, rs, ctx_r, rs, att_sr, rs,
                                               rows, kF, s)
        : ::ps::kernel::launch_mul_f32_f32_bf16(ctx_r, rs, gs_sr, rs, att_sr, rs,
                                               rows, kF, s);
    const hipError_t e2 = ::ps::kernel::launch_activation_quantize_e4m3(
        att_sr, cs_s, scales_s, rows, kF, kF, rs, aw.code_row_stride_bytes,
        aw.scale_row_stride_bytes, s, nullptr);

    ::ps::kernel::FusedAttnPostQuantizeArgs args{};
    args.gate = gate_r;
    args.gate_row_stride = rs;
    args.ctx = ctx_r;
    args.ctx_row_stride = rs;
    args.gated_out = att_fr;
    args.gated_row_stride = rs;
    args.codes = cs_f;
    args.scales = scales_f;
    args.code_row_stride = aw.code_row_stride_bytes;
    args.scale_row_stride = aw.scale_row_stride_bytes;
    args.rows = rows;
    args.features = kF;
    args.ctx_first = lay.swap_mul ? 0u : 1u;
    const hipError_t e3 = ::ps::kernel::launch_fused_attn_post_quantize(args, s);

    bool launch_ok =
        e0 == hipSuccess && e1 == hipSuccess && e2 == hipSuccess && e3 == hipSuccess;
    if (!launch_ok) {
        std::printf("test_fused_attn_post_quantize: launch fail rows=%u k=%u\n", rows,
                    kF);
        ok = false;
        ps::gpu::discard_cleanup_result(hipStreamDestroy(s));
        return;
    }
    if (hipStreamSynchronize(s) != hipSuccess) {
        ok = false;
        ps::gpu::discard_cleanup_result(hipStreamDestroy(s));
        return;
    }

    std::vector<uint8_t> got_att(elems * 2);
    std::vector<uint8_t> want_att(elems * 2);
    std::vector<uint8_t> got_cs(codes_bytes + scales_bytes);
    std::vector<uint8_t> want_cs(codes_bytes + scales_bytes);
    if (hipMemcpy(got_att.data(), att_f.dev, elems * 2, hipMemcpyDeviceToHost) !=
            hipSuccess ||
        hipMemcpy(want_att.data(), att_s.dev, elems * 2, hipMemcpyDeviceToHost) !=
            hipSuccess ||
        hipMemcpy(got_cs.data(), codes_f.dev, codes_bytes + scales_bytes,
                  hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(want_cs.data(), codes_s.dev, codes_bytes + scales_bytes,
                  hipMemcpyDeviceToHost) != hipSuccess) {
        ok = false;
        ps::gpu::discard_cleanup_result(hipStreamDestroy(s));
        return;
    }
    for (size_t i = 0; i < elems; ++i) {
        uint16_t g, w;
        std::memcpy(&g, &got_att[i * 2], 2);
        std::memcpy(&w, &want_att[i * 2], 2);
        if ((g & 0x7f80u) == 0x7f80u && (g & 0x007fu) != 0u) {
            g = 0x7fc0u;
            std::memcpy(&got_att[i * 2], &g, 2);
        }
        if ((w & 0x7f80u) == 0x7f80u && (w & 0x007fu) != 0u) {
            w = 0x7fc0u;
            std::memcpy(&want_att[i * 2], &w, 2);
        }
    }
    const bool att_same =
        std::memcmp(got_att.data(), want_att.data(), elems * 2) == 0;
    const bool cs_same =
        std::memcmp(got_cs.data(), want_cs.data(), codes_bytes + scales_bytes) == 0;
    if (!att_same || !cs_same) {
        size_t first = elems * 2;
        for (size_t i = 0; i < elems * 2; ++i) {
            if (got_att[i] != want_att[i]) {
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
        std::printf("test_fused_attn_post_quantize: MISMATCH rows=%u k=%u stride=%u off=%u swap=%d nasty=%d gated=%d codes=%d first=%zu got=%02x want=%02x firstcs=%zu\n",
                    rows, kF, rs, static_cast<uint32_t>(off_e), (int)lay.swap_mul,
                    (int)lay.nasty_inputs, (int)att_same, (int)cs_same, first,
                    first < elems * 2 ? got_att[first] : 0,
                    first < elems * 2 ? want_att[first] : 0, first_cs);
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
        {1, 6144, 6144, 0, false, false}, {1, 6144, 6144, 8, false, false},
        {2, 6144, 6144, 0, false, false}, {5, 6144, 6144, 0, false, false},
        {1, 6144, 6160, 0, false, false}, {3, 6144, 6160, 16, false, false},
        {1, 3072, 3072, 0, false, false}, {2, 3072, 3072, 0, false, false},
        {5, 3072, 3072, 0, false, false}, {1, 3072, 3088, 8, false, false},
        {1, 6144, 6144, 0, false, true}, {2, 3072, 3072, 0, false, true},
        {5, 6144, 6144, 0, false, true}, {1, 6144, 6144, 0, true, false},
        {2, 6144, 6144, 0, true, true}, {5, 6144, 6144, 0, true, false},
        {1, 3072, 3072, 0, true, false}, {2, 3072, 3072, 0, true, true},
        {5, 3072, 3072, 0, true, false},
    };
    for (const auto& lay : lays) check_case(lay, ok);
    if (ok) {
        ++g_passed;
        std::printf("test_fused_attn_post_quantize: PASS (split == fused byte-exact)\n");
    } else {
        ++g_failed;
        std::printf("test_fused_attn_post_quantize: FAIL\n");
    }
    return g_failed == 0 ? 0 : 1;
}
