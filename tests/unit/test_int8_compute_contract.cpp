#include <shelf/runtime/program/int8_activation_workspace.h>
#include <shelf/quantization/quantized_compute_view.h>
#include <shelf/quantization/codec/ue4m3.h>
#include <shelf/core/memory/alignment.h>
#include <shelf/core/memory/types.h>
#include <cstdio>
#include <cstdint>

using shelf::runtime::Int8ActivationWorkspaceLayout;

namespace {
int g_passed = 0;
int g_failed = 0;

void check(bool cond, const char* msg) {
    if (cond) { ++g_passed; } else { ++g_failed; std::printf("FAIL: %s\n", msg); }
}

int run() {
    check(shelf::gpu::align_up_16(0) == 0, "align16(0)==0");
    check(shelf::gpu::align_up_16(1) == 16, "align16(1)==16");
    check(shelf::gpu::align_up_16(16) == 16, "align16(16)==16");
    check(shelf::gpu::align_up_16(17) == 32, "align16(17)==32");
    check(shelf::gpu::aligned_row_stride(7, 2) == 8, "bf16 stride 7->8");
    check(shelf::gpu::aligned_row_stride(9, 2) == 16, "bf16 stride 9->16");
    check(shelf::gpu::aligned_row_stride(3, 4) == 4, "fp32 stride 3->4");
    check(shelf::gpu::aligned_row_stride(5, 4) == 8, "fp32 stride 5->8");
    check(shelf::gpu::aligned_row_stride(15, 1) == 16, "int8 stride 15->16");
    check(shelf::gpu::aligned_row_stride(17, 1) == 32, "int8 stride 17->32");

    {
        const auto l = Int8ActivationWorkspaceLayout::make(5120, 4);
        check(l.k_padded == 5120, "a8 kp 5120");
        check(l.code_row_stride_bytes == 5120, "a8 code stride 5120");
        check(l.scale_row_stride_bytes == 160, "a8 scale stride 160 (group-128)");
        check(l.codes_offset % 16 == 0, "a8 codes off 16B");
        check(l.scales_offset % 16 == 0, "a8 scales off 16B");
        check(l.total_bytes % 16 == 0, "a8 total 16B");
        check(l.code_row_stride_bytes % 16 == 0, "a8 code row 16B");
        check(l.scale_row_stride_bytes % 16 == 0, "a8 scale row 16B");
    }
    {
        const auto l = Int8ActivationWorkspaceLayout::make(100, 3);
        check(l.k_padded == 128, "a8 kp 100->128");
        check(l.code_row_stride_bytes == 128, "a8 code stride 128");
        check(l.scale_row_stride_bytes == 16, "a8 scale stride 16");
        check(l.scales_offset % 16 == 0, "a8 odd scales off 16B");
        check(l.total_bytes % 16 == 0, "a8 odd total 16B");
    }
    {
        const auto l = Int8ActivationWorkspaceLayout::make(17408, 1);
        check(l.scale_row_stride_bytes == 544, "a8 scale stride 17408 (group-128)");
        check(l.scale_row_stride_bytes % 16 == 0, "a8 17408 scale row 16B");
    }

    {
        static_assert(alignof(shelf::quantization::QuantizedComputeView) >= 16);
        static_assert(sizeof(shelf::quantization::QuantizedComputeView) % 16 == 0);
        check(sizeof(shelf::quantization::QuantizedComputeView) % 16 == 0, "view size 16B");
    }

    {
        for (uint32_t e = 0; e < 256; ++e) {
            const float f = shelf::quantization::codec::ue4m3_to_fp32(
                static_cast<uint8_t>(e));
            const uint16_t bits = shelf::f32_to_bf16_rne(f);
            float back = 0.0f;
            uint32_t u = static_cast<uint32_t>(bits) << 16;
            __builtin_memcpy(&back, &u, sizeof(back));
            char msg[64];
            std::snprintf(msg, sizeof(msg), "ue4m3->bf16 roundtrip %u", e);
            if (f == 0.0f) {
                check(back == 0.0f, msg);
            } else {
                check(back > 0.0f && (back - f) / f < 0.01f &&
                          (f - back) / f < 0.01f,
                      msg);
            }
        }
    }
    return 0;
}

}

int main() {
    run();
    std::printf("test_int8_compute_contract: passed=%d failed=%d\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
