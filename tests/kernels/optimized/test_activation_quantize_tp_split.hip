#include <shelf/kernels/optimized/activation_quantize.h>
#include <shelf/quantization/fp8/fp8_frag.h>
#include <shelf/runtime/program/int8_activation_workspace.h>
#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/types.h>
#include <shelf/core/gpu/cleanup.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace {

int passed = 0;
int failed = 0;

void check(bool cond, const char* name) {
    if (cond) {
        passed++;
        std::printf("PASS: %s\n", name);
    } else {
        failed++;
        std::printf("FAIL: %s\n", name);
    }
}

ps::bf16_t f32_to_bf16(float f) {
    uint32_t ui;
    std::memcpy(&ui, &f, 4);
    uint16_t bf = static_cast<uint16_t>((ui + 0x8000U) >> 16);
    return ps::bf16_t{bf};
}

std::vector<uint16_t> make_input(uint32_t rows, uint32_t full_k, uint32_t pattern) {
    std::vector<uint16_t> in(static_cast<size_t>(rows) * full_k);
    for (size_t idx = 0; idx < in.size(); ++idx) {
        const uint32_t h = static_cast<uint32_t>(idx * 2654435761u) + 12345u;
        uint16_t bits = 0u;
        switch (pattern) {
            case 0u: {
                const int base = static_cast<int>(h % 3001u) - 1500;
                bits = f32_to_bf16(static_cast<float>(base) / 1000.0f).data;
                break;
            }
            case 1u:
                bits = (h % 7u == 0u) ? 0x7FC0u : f32_to_bf16(1024.0f * (static_cast<float>(h % 100u) / 100.0f)).data;
                break;
            default:
                bits = f32_to_bf16(static_cast<float>(h % 2001u) - 1000.0f).data;
                break;
        }
        in[idx] = bits;
    }
    return in;
}

struct DeviceOut {
    std::vector<uint8_t> codes;
    std::vector<float> scales;
    bool ok = false;
};

bool run_local(const std::vector<uint16_t>& h_in, uint32_t rows, uint32_t full_k,
               uint32_t k, uint32_t col_off, ps::gpu::GpuArena& arena, hipStream_t stream,
               DeviceOut& out, std::string& err) {
    const auto layout = ::ps::runtime::Int8ActivationWorkspaceLayout::make(k, rows);
    auto ws_res = arena.allocate_aligned(layout.total_bytes, 256);
    auto in_res = arena.allocate_aligned(h_in.size() * sizeof(ps::bf16_t), 256);
    if (!ws_res.ok() || !in_res.ok()) {
        err = "alloc";
        return false;
    }
    uint8_t* ws = static_cast<uint8_t*>(ws_res.release().data());
    ps::bf16_t* in = static_cast<ps::bf16_t*>(in_res.release().data());
    if (hipMemcpy(in, h_in.data(), h_in.size() * 2, hipMemcpyHostToDevice) != hipSuccess) {
        err = "h2d";
        return false;
    }
    if (hipMemset(ws, 0xAB, layout.total_bytes) != hipSuccess) {
        err = "memset";
        return false;
    }
    uint8_t* codes = ws + layout.codes_offset;
    float* scales = reinterpret_cast<float*>(ws + layout.scales_offset);
    const hipError_t herr = ps::kernel::launch_activation_quantize_fp8_finegrained(
        in + col_off, codes, scales, rows, k, k, full_k,
        layout.code_row_stride_bytes, layout.scale_row_stride_bytes, stream, nullptr);
    if (herr != hipSuccess) {
        err = std::string("launch: ") + hipGetErrorString(herr);
        return false;
    }
    if (hipStreamSynchronize(stream) != hipSuccess) {
        err = "sync";
        return false;
    }
    const uint32_t rows_padded = (rows + 15u) & ~15u;
    out.codes.assign(static_cast<size_t>(rows_padded) * layout.code_row_stride_bytes, 0u);
    out.scales.assign(static_cast<size_t>(rows) * (layout.scale_row_stride_bytes / 4u), 0.f);
    if (hipMemcpy(out.codes.data(), ws, out.codes.size(), hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(out.scales.data(), scales, out.scales.size() * 4, hipMemcpyDeviceToHost) != hipSuccess) {
        err = "d2h";
        return false;
    }
    out.ok = true;
    return true;
}

const size_t code_pos(uint32_t code_stride, uint32_t m, uint32_t i) {
    return static_cast<size_t>(m >> 4) * static_cast<size_t>(code_stride) * 16u
        + static_cast<size_t>(i >> 6) * ps::quantization::fp8::kFragBlockBytes
        + ps::quantization::fp8::fp8_frag_offset(m & 15u, i & 63u);
}

bool compare_split(uint32_t rows, uint32_t full_k, uint32_t pattern,
                   ps::gpu::GpuArena& arena, hipStream_t stream) {
    const uint32_t local_k = full_k / 2u;
    const uint32_t groups_total = full_k / 128u;
    const uint32_t groups_local = local_k / 128u;
    const auto h_in = make_input(rows, full_k, pattern);

    DeviceOut tp1;
    DeviceOut rank0;
    DeviceOut rank1;
    std::string err;
    if (!run_local(h_in, rows, full_k, full_k, 0u, arena, stream, tp1, err)) {
        std::printf("FAIL: tp1 rows=%u k=%u: %s\n", rows, full_k, err.c_str());
        failed++;
        return false;
    }
    if (!run_local(h_in, rows, full_k, local_k, 0u, arena, stream, rank0, err)) {
        std::printf("FAIL: rank0 rows=%u k=%u: %s\n", rows, full_k, err.c_str());
        failed++;
        return false;
    }
    if (!run_local(h_in, rows, full_k, local_k, local_k, arena, stream, rank1, err)) {
        std::printf("FAIL: rank1 rows=%u k=%u: %s\n", rows, full_k, err.c_str());
        failed++;
        return false;
    }

    const auto layout1 = ::ps::runtime::Int8ActivationWorkspaceLayout::make(full_k, rows);
    const auto layout2 = ::ps::runtime::Int8ActivationWorkspaceLayout::make(local_k, rows);

    bool scales_ok = true;
    for (uint32_t m = 0; m < rows && scales_ok; ++m) {
        const float* s1 = tp1.scales.data() + static_cast<size_t>(m) * (layout1.scale_row_stride_bytes / 4u);
        const float* s0 = rank0.scales.data() + static_cast<size_t>(m) * (layout2.scale_row_stride_bytes / 4u);
        const float* sr = rank1.scales.data() + static_cast<size_t>(m) * (layout2.scale_row_stride_bytes / 4u);
        for (uint32_t g = 0; g < groups_local; ++g) {
            if (std::memcmp(&s1[g], &s0[g], 4) != 0) {
                std::printf("  scale mismatch m=%u g=%u tp1=%.9g rank0=%.9g (k=%u)\n",
                            m, g, s1[g], s0[g], full_k);
                scales_ok = false;
                break;
            }
            if (std::memcmp(&s1[groups_local + g], &sr[g], 4) != 0) {
                std::printf("  scale mismatch m=%u g=%u tp1=%.9g rank1=%.9g (k=%u)\n",
                            m, g, s1[groups_local + g], sr[g], full_k);
                scales_ok = false;
                break;
            }
        }
    }

    bool codes_ok = true;
    for (uint32_t m = 0; m < rows && codes_ok; ++m) {
        for (uint32_t g = 0; g < groups_local && codes_ok; ++g) {
            for (uint32_t li = 0; li < 128u; ++li) {
                const uint32_t g1 = g;
                const uint32_t g0 = g;
                const uint32_t gr = g;
                const uint32_t i1 = g1 * 128u + li;
                const uint32_t i0 = g0 * 128u + li;
                const uint32_t ir = gr * 128u + li;
                const size_t p1 = code_pos(layout1.code_row_stride_bytes, m, i1);
                const size_t p0 = code_pos(layout2.code_row_stride_bytes, m, i0);
                const size_t pr = code_pos(layout2.code_row_stride_bytes, m, ir);
                if (p1 >= tp1.codes.size() || p0 >= rank0.codes.size() ||
                    pr >= rank1.codes.size()) {
                    std::printf("  codes out of range m=%u g=%u (k=%u)\n", m, g, full_k);
                    codes_ok = false;
                    break;
                }
                if (tp1.codes[p1] != rank0.codes[p0]) {
                    std::printf("  codes mismatch rank0 m=%u g=%u li=%u tp1=0x%02x r0=0x%02x (k=%u)\n",
                                m, g, li, tp1.codes[p1], rank0.codes[p0], full_k);
                    codes_ok = false;
                    break;
                }
                const size_t p1r = code_pos(layout1.code_row_stride_bytes, m,
                                            (groups_local + g) * 128u + li);
                if (p1r >= tp1.codes.size()) {
                    std::printf("  codes out of range tp1 rank1 side m=%u g=%u (k=%u)\n", m, g, full_k);
                    codes_ok = false;
                    break;
                }
                if (tp1.codes[p1r] != rank1.codes[pr]) {
                    std::printf("  codes mismatch rank1 m=%u g=%u li=%u tp1=0x%02x r1=0x%02x (k=%u)\n",
                                m, g, li, tp1.codes[p1r], rank1.codes[pr], full_k);
                    codes_ok = false;
                    break;
                }
            }
        }
    }

    char label[160];
    std::snprintf(label, sizeof(label),
                  "tp-split rows=%-3u k=%-5u groups=%u/%u pattern=%u scales=%d codes=%d",
                  rows, full_k, groups_total, groups_local, pattern, scales_ok ? 1 : 0,
                  codes_ok ? 1 : 0);
    check(scales_ok && codes_ok, label);
    return scales_ok && codes_ok;
}

}  // namespace

int main() {
    std::printf("=== Activation Quantize TP Split Equivalence ===\n");
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;
    auto arena_result = ps::gpu::GpuArena::create(0, 1536u * 1024u * 1024u);
    if (!arena_result.ok()) {
        std::printf("FAIL: arena create: %s\n", arena_result.status().message().c_str());
        return 1;
    }
    ps::gpu::GpuArena arena = arena_result.release();

    const uint32_t pair_k[] = {17408u, 6144u};
    const uint32_t row_list[] = {1u, 16u, 32u};
    const uint32_t pattern_list[] = {0u, 1u, 2u};

    for (uint32_t rows : row_list) {
        for (uint32_t k : pair_k) {
            for (uint32_t p : pattern_list) {
                compare_split(rows, k, p, arena, stream);
            }
        }
    }

    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    std::printf("Results: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
