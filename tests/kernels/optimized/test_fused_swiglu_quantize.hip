#include <shelf/kernels/optimized/elementwise.h>
#include <shelf/kernels/optimized/activation_quantize.h>
#include <shelf/kernels/optimized/swiglu_quantize.h>
#include <shelf/runtime/program/int8_activation_workspace.h>
#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr uint32_t kF = 5120u;

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

float elem_value(uint64_t i, uint32_t tag) {
    const uint32_t r = h32(i * 2654435761ull + 77ull + tag * 40503ull);
    const int32_t v = static_cast<int32_t>(r % 4001u) - 2000;
    float f = static_cast<float>(v) / 128.0f;
    if (i % 1543u == 5u) f = 0.0f;
    if (i % 1543u == 17u) f = -0.0f;
    if (i % 6091u == 9u) f = 1.0e-28f;
    if (i % 5099u == 13u && tag == 0u) f = -39.5f;
    return f;
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
    uint32_t gate_stride;
    uint32_t up_stride;
    uint32_t out_stride;
    uint32_t base_offset_elems;
};

void check_case(const Layout& lay, bool& ok) {
    const uint32_t rows = lay.rows;
    const size_t off_e = lay.base_offset_elems;
    const size_t gate_elems = off_e + static_cast<size_t>(rows) * lay.gate_stride;
    const size_t up_elems = off_e + static_cast<size_t>(rows) * lay.up_stride;
    const size_t out_elems = off_e + static_cast<size_t>(rows) * lay.out_stride;

    std::vector<uint16_t> h_gate(gate_elems);
    std::vector<uint16_t> h_up(up_elems);
    for (size_t i = 0; i < gate_elems; ++i)
        h_gate[i] = bf16_bits_of(elem_value(i, 0u));
    for (size_t i = 0; i < up_elems; ++i)
        h_up[i] = bf16_bits_of(elem_value(i, 1u));

    const ::ps::runtime::Int8ActivationWorkspaceLayout aw =
        ::ps::runtime::Int8ActivationWorkspaceLayout::make(kF, rows);
    const size_t codes_bytes = aw.codes_bytes();
    const size_t scales_bytes = aw.scales_bytes();

    Buf gate_s, up_s, out_s, out_f, codes_s, codes_f;
    if (!gate_s.alloc(gate_elems * 2) || !up_s.alloc(up_elems * 2) ||
        !out_s.alloc(out_elems * 2) || !out_f.alloc(out_elems * 2) ||
        !codes_s.alloc(codes_bytes + scales_bytes) ||
        !codes_f.alloc(codes_bytes + scales_bytes)) {
        ok = false;
        return;
    }
    if (hipMemcpy(gate_s.dev, h_gate.data(), gate_elems * 2, hipMemcpyHostToDevice) !=
            hipSuccess ||
        hipMemcpy(up_s.dev, h_up.data(), up_elems * 2, hipMemcpyHostToDevice) !=
            hipSuccess) {
        ok = false;
        return;
    }
    const size_t off = off_e * 2u;
    auto* gate_sr = reinterpret_cast<::ps::bf16_t*>(static_cast<uint8_t*>(gate_s.dev) + off);
    auto* up_sr = reinterpret_cast<::ps::bf16_t*>(static_cast<uint8_t*>(up_s.dev) + off);
    auto* out_sr = reinterpret_cast<::ps::bf16_t*>(static_cast<uint8_t*>(out_s.dev) + off);
    auto* out_fr = reinterpret_cast<::ps::bf16_t*>(static_cast<uint8_t*>(out_f.dev) + off);
    auto* cs_s = static_cast<uint8_t*>(codes_s.dev);
    auto* cs_f = static_cast<uint8_t*>(codes_f.dev);
    float* scales_s = reinterpret_cast<float*>(cs_s + aw.scales_offset);
    float* scales_f = reinterpret_cast<float*>(cs_f + aw.scales_offset);

    if (hipMemset(out_s.dev, 0, out_elems * 2) != hipSuccess ||
        hipMemset(out_f.dev, 0, out_elems * 2) != hipSuccess ||
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

    const hipError_t e0 = ::ps::kernel::launch_swiglu_bf16(
        gate_sr, lay.gate_stride, up_sr, lay.up_stride, out_sr, lay.out_stride, rows,
        kF, s);
    const hipError_t e1 = ::ps::kernel::launch_activation_quantize_e4m3(
        out_sr, cs_s, scales_s, rows, kF, kF, lay.out_stride,
        aw.code_row_stride_bytes, aw.scale_row_stride_bytes, s, nullptr);

    ::ps::kernel::FusedSwiGluQuantizeArgs args{};
    args.gate = gate_sr;
    args.up = up_sr;
    args.output = out_fr;
    args.codes = cs_f;
    args.scales = scales_f;
    args.rows = rows;
    args.k = kF;
    args.gate_row_stride = lay.gate_stride;
    args.up_row_stride = lay.up_stride;
    args.output_row_stride = lay.out_stride;
    args.code_row_stride = aw.code_row_stride_bytes;
    args.scale_row_stride = aw.scale_row_stride_bytes;
    const hipError_t e2 = ::ps::kernel::launch_fused_swiglu_quantize(args, s);

    const bool launch_ok = e0 == hipSuccess && e1 == hipSuccess && e2 == hipSuccess;
    const bool sync_ok = hipStreamSynchronize(s) == hipSuccess;
    (void)hipStreamDestroy(s);
    if (!launch_ok || !sync_ok) {
        std::printf("test_fused_swiglu_quantize: launch fail rows=%u e=%d/%d/%d\n",
                    rows, static_cast<int>(e0), static_cast<int>(e1),
                    static_cast<int>(e2));
        ok = false;
        return;
    }

    std::vector<uint8_t> out_sb(out_elems * 2), out_fb(out_elems * 2);
    std::vector<uint8_t> cs_sb(codes_bytes + scales_bytes), cs_fb(codes_bytes + scales_bytes);
    if (hipMemcpy(out_sb.data(), out_s.dev, out_elems * 2, hipMemcpyDeviceToHost) !=
            hipSuccess ||
        hipMemcpy(out_fb.data(), out_f.dev, out_elems * 2, hipMemcpyDeviceToHost) !=
            hipSuccess ||
        hipMemcpy(cs_sb.data(), codes_s.dev, codes_bytes + scales_bytes,
                  hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(cs_fb.data(), codes_f.dev, codes_bytes + scales_bytes,
                  hipMemcpyDeviceToHost) != hipSuccess) {
        ok = false;
        return;
    }
    bool out_match = true;
    for (uint32_t r = 0; r < rows && out_match; ++r) {
        const size_t o = (off_e + static_cast<size_t>(r) * lay.out_stride) * 2u;
        if (std::memcmp(out_sb.data() + o, out_fb.data() + o, kF * 2u) != 0)
            out_match = false;
    }
    const bool cs_match = cs_sb == cs_fb;
    if (!out_match || !cs_match) {
        std::printf(
            "test_fused_swiglu_quantize: MISMATCH rows=%u gate_stride=%u up_stride=%u "
            "out_stride=%u off=%u out=%d codes+scales=%d\n",
            rows, lay.gate_stride, lay.up_stride, lay.out_stride,
            lay.base_offset_elems, static_cast<int>(out_match),
            static_cast<int>(cs_match));
        ok = false;
    }
}

}

int main() {
    const Layout cases[] = {
        {1u, kF, kF, kF, 0u},
        {1u, kF, kF, kF, 8u},
        {2u, kF, kF, kF, 0u},
        {3u, kF + 64u, kF, kF, 8u},
        {5u, kF, kF + 64u, kF + 32u, 8u},
        {16u, kF, kF, kF, 0u},
        {129u, kF, kF, kF, 8u},
        {512u, kF, kF, kF, 0u},
        {2048u, kF, kF, kF, 0u},
    };
    bool ok = true;
    for (const Layout& lay : cases) check_case(lay, ok);
    if (ok) {
        ++g_passed;
        std::printf("test_fused_swiglu_quantize: PASS (split == fused byte-exact)\n");
    } else {
        ++g_failed;
        std::printf("test_fused_swiglu_quantize: FAIL\n");
    }
    std::printf("test_fused_swiglu_quantize: passed=%d failed=%d\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
