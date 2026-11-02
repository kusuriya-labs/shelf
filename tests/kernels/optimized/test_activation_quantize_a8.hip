#include <phaseshift/kernels/optimized/activation_quantize.h>
#include <phaseshift/runtime/program/int8_activation_workspace.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/core/gpu/cleanup.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
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

float bf16_to_f32(uint16_t bits) {
    uint32_t ui = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &ui, 4);
    return f;
}

ps::bf16_t f32_to_bf16(float f) {
    uint32_t ui;
    std::memcpy(&ui, &f, 4);
    uint16_t bf = static_cast<uint16_t>((ui + 0x8000U) >> 16);
    return ps::bf16_t{bf};
}

void cpu_reference(const std::vector<uint16_t>& in, uint32_t rows, uint32_t k, uint32_t kp,
                   uint32_t code_row_stride, uint32_t scale_row_stride,
                   std::vector<int8_t>& codes, std::vector<float>& scales) {
    const uint32_t groups = (kp + 127u) / 128u;
    const uint32_t scale_row_floats = scale_row_stride / 4u;
    const uint32_t rows_padded = (rows + 15u) & ~15u;
    codes.assign((size_t)rows_padded * code_row_stride, 0);
    scales.assign((size_t)rows * scale_row_floats, 0.f);
    for (uint32_t m = 0; m < rows; ++m) {
        for (uint32_t g = 0; g < groups; ++g) {
            const uint32_t i0 = g * 128u;
            float amax = 0.f;
            float vals[128];
            for (uint32_t i = 0; i < 128u; ++i) {
                const uint32_t idx = i0 + i;
                const float v = idx < k ? bf16_to_f32(in[(size_t)m * k + idx]) : 0.f;
                vals[i] = std::isfinite(v) ? v : 0.f;
                const float av = std::fabs(vals[i]);
                if (av > amax) amax = av;
            }
            const float scale = amax > 0.f ? amax / 127.0f : 0.f;
            scales[(size_t)m * scale_row_floats + g] = scale;
            const float inv = scale > 0.f ? 1.0f / scale : 0.f;
            for (uint32_t i = 0; i < 128u; ++i) {
                const uint32_t idx = i0 + i;
                if (idx >= kp) break;
                int c = 0;
                if (scale > 0.f) {
                    c = static_cast<int>(nearbyintf(vals[i] * inv));
                    if (c > 127) c = 127;
                    if (c < -127) c = -127;
                }
                codes[(size_t)(m >> 4) * (code_row_stride * 16u) +
                      (size_t)(idx >> 5u) * 512u + (size_t)((idx & 31u) >> 4u) * 256u +
                      (size_t)(m & 15u) * 16u + (size_t)(idx & 15u)] =
                    static_cast<int8_t>(c);
            }
        }
    }
}

std::vector<uint16_t> make_mixed_input(uint32_t rows, uint32_t k) {
    std::vector<uint16_t> in((size_t)rows * k);
    for (uint32_t idx = 0; idx < (size_t)rows * k; ++idx) {
        if (idx % 97u == 0u) in[idx] = f32_to_bf16(0.0f).data;
        else if (idx % 131u == 0u) in[idx] = 0x7F80u;
        else if (idx % 151u == 0u) in[idx] = 0xFF80u;
        else if (idx % 173u == 0u) in[idx] = 0x7FC0u;
        else {
            const uint32_t h = (idx * 2654435761u) + 12345u;
            const int base = static_cast<int>(h % 3001u) - 1500;
            in[idx] = f32_to_bf16(base / 1000.0f).data;
        }
    }
    return in;
}

bool run_case(uint32_t rows, uint32_t k, const std::vector<uint16_t>& h_in,
              ps::gpu::GpuArena& arena, hipStream_t stream, const char* label) {
    const uint32_t kp = (k + 31u) & ~31u;
    const auto layout = ::ps::runtime::Int8ActivationWorkspaceLayout::make(k, rows);
    auto ws_res = arena.allocate_aligned(layout.total_bytes, 16);
    auto in_res = arena.allocate_aligned((size_t)rows * k * sizeof(ps::bf16_t), 16);
    if (!ws_res.ok() || !in_res.ok()) {
        std::printf("FAIL: %s rows=%u k=%u alloc\n", label, rows, k);
        return false;
    }
    void* ws = ws_res.release().data();
    ps::bf16_t* in = static_cast<ps::bf16_t*>(in_res.release().data());
    if (hipMemcpy(in, h_in.data(), h_in.size() * 2, hipMemcpyHostToDevice) != hipSuccess) {
        std::printf("FAIL: %s rows=%u k=%u h2d\n", label, rows, k);
        return false;
    }
    int8_t* codes = static_cast<int8_t*>(ws) + layout.codes_offset;
    float* scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(ws) + layout.scales_offset);
    hipError_t herr = ps::kernel::launch_activation_quantize_a8(
        in, codes, scales, rows, k, kp, k,
        layout.code_row_stride_bytes, layout.scale_row_stride_bytes, stream);
    if (herr != hipSuccess) {
        std::printf("FAIL: %s rows=%u k=%u launch: %s\n", label, rows, k,
                    hipGetErrorString(herr));
        return false;
    }
    if (hipStreamSynchronize(stream) != hipSuccess) {
        std::printf("FAIL: %s rows=%u k=%u sync\n", label, rows, k);
        return false;
    }
    const uint32_t code_rs = layout.code_row_stride_bytes;
    const uint32_t scale_rs = layout.scale_row_stride_bytes;
    const uint32_t rows_padded = (rows + 15u) & ~15u;
    std::vector<int8_t> got_codes((size_t)rows_padded * code_rs, 0);
    std::vector<float> got_scales((size_t)rows * (scale_rs / 4u), 0.f);
    if (hipMemcpy(got_codes.data(), ws, (size_t)rows_padded * code_rs, hipMemcpyDeviceToHost) != hipSuccess)
        return false;
    if (hipMemcpy(got_scales.data(), scales, got_scales.size() * 4, hipMemcpyDeviceToHost) != hipSuccess)
        return false;
    std::vector<int8_t> ref_codes;
    std::vector<float> ref_scales;
    cpu_reference(h_in, rows, k, kp, code_rs, scale_rs, ref_codes, ref_scales);
    const bool codes_ok = (got_codes == ref_codes);
    const bool scales_ok = (got_scales == ref_scales);
    check(codes_ok && scales_ok, label);
    return codes_ok && scales_ok;
}

}  // namespace

int main() {
    std::printf("=== Activation Quantize A8 (optimized) Tests ===\n");
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;
    auto arena_result = ps::gpu::GpuArena::create(0, 256 * 1024 * 1024);
    if (!arena_result.ok()) {
        std::printf("FAIL: arena create: %s\n", arena_result.status().message().c_str());
        return 1;
    }
    ps::gpu::GpuArena arena = arena_result.release();

    const uint32_t row_list[] = {1, 2, 16, 128};
    const uint32_t k_list[] = {1, 31, 32, 33, 100, 2560, 9216};
    for (uint32_t rows : row_list) {
        for (uint32_t k : k_list) {
            const auto in = make_mixed_input(rows, k);
            char label[128];
            std::snprintf(label, sizeof(label), "mixed rows=%u k=%u", rows, k);
            run_case(rows, k, in, arena, stream, label);
        }
    }

    {
        const uint32_t rows = 4;
        const uint32_t k = 96;
        std::vector<uint16_t> in((size_t)rows * k);
        for (auto& v : in) v = f32_to_bf16(0.0f).data;
        char label[128];
        std::snprintf(label, sizeof(label), "allzero(amax=0) rows=%u k=%u", rows, k);
        run_case(rows, k, in, arena, stream, label);
    }

    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    std::printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
