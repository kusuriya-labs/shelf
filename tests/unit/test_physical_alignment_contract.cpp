#include <shelf/runtime/program/int8_activation_workspace.h>
#include <shelf/models/qwen35/state/state_types.h>
#include <shelf/core/memory/alignment.h>
#include <cstdio>
#include <cstdint>

namespace rt = ps::runtime;

namespace {
int passed = 0, failed = 0;
void check(bool c, const char* m) { if (c) { ++passed; } else { ++failed; printf("FAIL: %s\n", m); } }
}

int main() {
    {
        const uint32_t cases[][2] = {{7, 1}, {9, 2}, {3, 4}, {5, 8}, {15, 1}, {17, 2}};
        for (const auto& cc : cases) {
            const uint32_t features = cc[0];
            const uint32_t bytes = cc[1];
            char msg[96];
            std::snprintf(
                msg, sizeof(msg), "aligned stride features=%u bytes=%u", features, bytes);
            check(ps::gpu::aligned_row_stride(features, bytes) * bytes % 16 == 0, msg);
        }
        check(ps::gpu::aligned_row_stride(5120, 2) == 5120, "bf16 5120 tight");
        check(ps::gpu::aligned_row_stride(17408, 2) == 17408, "bf16 17408 tight");
        check(ps::gpu::aligned_row_stride(248320, 4) % 4 == 0, "fp32 vocab stride");
    }
    {
        const auto l = rt::Int8ActivationWorkspaceLayout::make(7, 5);
        check(l.code_row_stride_bytes % 16 == 0, "a8 odd code row 16B");
        check(l.scale_row_stride_bytes % 16 == 0, "a8 odd scale row 16B");
        check(l.scales_offset % 16 == 0, "a8 odd scales off 16B");
    }
    {
        auto odd = ps::state::GdnStatePoolLayout::make(2, 100, 3, 4, 32, 32);
        check(odd.conv_dim_padded == 104, "gdn odd padded 100->104");
        check(odd.conv_history_stride == 104, "gdn odd hist stride");
        check((uint64_t)odd.conv_history_stride * 2 % 16 == 0, "gdn odd plane 16B");
        auto prod = ps::state::GdnStatePoolLayout::make(48, 10240, 3, 48, 128, 128);
        check((uint64_t)prod.conv_history_stride * 2 % 16 == 0, "gdn prod plane 16B");
        const uint64_t layer_elems =
            (uint64_t)prod.conv_history * prod.conv_history_stride;
        check(layer_elems * 2 % 16 == 0, "gdn prod layer 16B");
    }
    printf("test_physical_alignment_contract: passed=%d failed=%d\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
