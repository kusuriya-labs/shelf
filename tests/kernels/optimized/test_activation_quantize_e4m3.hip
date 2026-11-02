#include <shelf/kernels/optimized/activation_quantize.h>
#include <shelf/runtime/program/int8_activation_workspace.h>
#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/types.h>
#include <shelf/core/gpu/cleanup.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
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

float bf16_bits_to_f32(uint16_t bits) {
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

uint16_t f32_to_bf16_bits(float f) {
    return f32_to_bf16(f).data;
}

constexpr uint8_t kSentinel = 0xABu;

void cpu_reference(const std::vector<uint16_t>& in, uint32_t rows, uint32_t k, uint32_t kp,
                   uint32_t code_row_stride, uint32_t scale_row_stride,
                   std::vector<uint8_t>& codes, std::vector<float>& scales) {
    const uint32_t rows_padded = (rows + 15u) & ~15u;
    const size_t scale_count = (size_t)rows * (scale_row_stride / 4u);
    codes.assign((size_t)rows_padded * code_row_stride, kSentinel);
    std::vector<uint8_t> scale_bytes(scale_count * 4u, kSentinel);
    float* scale_out = reinterpret_cast<float*>(scale_bytes.data());
    for (uint32_t m = 0; m < rows; ++m) {
        uint8_t* crow = codes.data() +
            (size_t)(m >> 4) * (size_t)(code_row_stride * 16u) +
            (size_t)(m & 15u) * 16u;
        float* row_scales = scale_out + (size_t)m * (scale_row_stride / 4u);
        const uint32_t groups = (kp + 127u) / 128u;
        for (uint32_t g = 0; g < groups; ++g) {
            const uint32_t i0 = g * 128u;
            const uint32_t k_end = i0 + 128u < k ? i0 + 128u : k;
            float peak = 0.f;
            for (uint32_t i = i0; i < k_end; ++i) {
                const float v = bf16_bits_to_f32(in[(size_t)m * k + i]);
                if (std::isfinite(v) && std::fabs(v) > peak) peak = std::fabs(v);
            }
            const float scale = peak > 0.f ? peak / 448.0f : 0.f;
            const float inv = scale > 0.f ? 1.0f / scale : 0.f;
            row_scales[g] = scale;
            const uint32_t kp_end = i0 + 128u < kp ? i0 + 128u : kp;
            for (uint32_t i = i0; i < kp_end; ++i) {
                float v = 0.f;
                if (i < k) {
                    const float x = bf16_bits_to_f32(in[(size_t)m * k + i]);
                    v = std::isfinite(x) ? x : 0.f;
                }
                uint8_t c = 0u;
                if (scale > 0.f) {
                    const float sv = v * inv;
                    const __hip_fp8_storage_t enc =
                        __hip_cvt_float_to_fp8(sv, __HIP_SATFINITE, __HIP_E4M3);
                    c = static_cast<uint8_t>(enc);
                }
                crow[(size_t)(i >> 5) * 512u + (size_t)((i & 31u) >> 4) * 256u + (size_t)(i & 15u)] = c;
            }
        }
    }
    scales.assign(scale_out, scale_out + scale_count);
}

enum Pattern {
    kRandom = 0,
    kAllPosZero,
    kAllNegZero,
    kPositive,
    kNegative,
    kMixedNan,
    kPosInf,
    kNegInf,
    kTinySubnormal,
    kLargeFinite,
    kNear448,
    kE4m3Boundary,
    kMixedAll,
    kPatternCount
};

const char* pattern_name(uint32_t p) {
    static const char* names[] = {
        "random", "all+0", "all-0", "positive", "negative", "nan", "posinf",
        "neginf", "tiny-subnormal", "large-finite", "near-448", "e4m3-boundary",
        "mixed-all"};
    return p < kPatternCount ? names[p] : "?";
}

std::vector<uint16_t> make_input(uint32_t rows, uint32_t k, uint32_t pattern) {
    std::vector<uint16_t> in((size_t)rows * k);
    for (size_t idx = 0; idx < in.size(); ++idx) {
        const uint32_t h = static_cast<uint32_t>(idx * 2654435761u) + 12345u;
        uint16_t bits = 0u;
        switch (pattern) {
            case kRandom: {
                const int base = static_cast<int>(h % 3001u) - 1500;
                bits = f32_to_bf16_bits(static_cast<float>(base) / 1000.0f);
                break;
            }
            case kAllPosZero: bits = 0x0000u; break;
            case kAllNegZero: bits = 0x8000u; break;
            case kPositive: bits = f32_to_bf16_bits(0.5f + static_cast<float>(h % 100u)); break;
            case kNegative: bits = f32_to_bf16_bits(-0.5f - static_cast<float>(h % 100u)); break;
            case kMixedNan: bits = (h % 3u == 0u) ? 0x7FC0u : f32_to_bf16_bits(1.0f); break;
            case kPosInf: bits = 0x7F80u; break;
            case kNegInf: bits = 0xFF80u; break;
            case kTinySubnormal: bits = static_cast<uint16_t>(h % 7u); break;
            case kLargeFinite: bits = 0x7F7Fu; break;
            case kNear448: bits = f32_to_bf16_bits(448.0f * (0.99f + 0.02f * static_cast<float>(h % 100u) / 100.0f)); break;
            case kE4m3Boundary: {
                const float base = 1.0f + static_cast<float>(h % 64u) * 0.0078125f;
                bits = f32_to_bf16_bits(base);
                break;
            }
            case kMixedAll:
            default: {
                switch (h % 11u) {
                    case 0u: bits = 0x0000u; break;
                    case 1u: bits = 0x8000u; break;
                    case 2u: bits = 0x7FC0u; break;
                    case 3u: bits = 0x7F80u; break;
                    case 4u: bits = 0xFF80u; break;
                    case 5u: bits = 0x0001u; break;
                    case 6u: bits = 0x7F7Fu; break;
                    case 7u: bits = 0x43E0u; break;
                    case 8u: bits = f32_to_bf16_bits(-1.5f); break;
                    case 9u: bits = f32_to_bf16_bits(1024.0f); break;
                    default: {
                        const int base = static_cast<int>(h % 2001u) - 1000;
                        bits = f32_to_bf16_bits(static_cast<float>(base) / 1000.0f);
                        break;
                    }
                }
                break;
            }
        }
        in[idx] = bits;
    }
    return in;
}

struct Result {
    std::vector<uint8_t> codes;
    std::vector<float> scales;
    bool ok = false;
};

bool run_kernel(const std::vector<uint16_t>& h_in, uint32_t rows, uint32_t k,
                uint32_t kp, bool generic, ps::gpu::GpuArena& arena, hipStream_t stream,
                Result& out, std::string& err, bool expect_specialized = true) {
    const auto layout = ::ps::runtime::Int8ActivationWorkspaceLayout::make(k, rows);
    auto ws_res = arena.allocate_aligned(layout.total_bytes, 256);
    auto in_res = arena.allocate_aligned((size_t)rows * k * sizeof(ps::bf16_t), 256);
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
    bool specialized = false;
    const hipError_t herr = generic
        ? ps::kernel::launch_activation_quantize_e4m3_generic(
              in, codes, scales, rows, k, kp, k,
              layout.code_row_stride_bytes, layout.scale_row_stride_bytes, stream)
        : ps::kernel::launch_activation_quantize_e4m3(
              in, codes, scales, rows, k, kp, k,
              layout.code_row_stride_bytes, layout.scale_row_stride_bytes, stream,
              &specialized);
    if (herr != hipSuccess) {
        err = std::string("launch: ") + hipGetErrorString(herr);
        return false;
    }
    if (!generic && specialized != expect_specialized) {
        err = specialized ? "auto path did not take the specialized kernel"
                          : "auto path unexpectedly took the specialized kernel";
        return false;
    }
    if (hipStreamSynchronize(stream) != hipSuccess) {
        err = "sync";
        return false;
    }
    const uint32_t rows_padded = (rows + 15u) & ~15u;
    out.codes.assign((size_t)rows_padded * layout.code_row_stride_bytes, 0u);
    out.scales.assign((size_t)rows * (layout.scale_row_stride_bytes / 4u), 0.f);
    if (hipMemcpy(out.codes.data(), ws, out.codes.size(), hipMemcpyDeviceToHost) != hipSuccess) {
        err = "d2h codes";
        return false;
    }
    if (hipMemcpy(out.scales.data(), scales, out.scales.size() * 4, hipMemcpyDeviceToHost) != hipSuccess) {
        err = "d2h scales";
        return false;
    }
    out.ok = true;
    return true;
}

bool compare_cases(uint32_t rows, uint32_t k, uint32_t pattern, ps::gpu::GpuArena& arena,
                   hipStream_t stream) {
    const uint32_t kp = (k + 31u) & ~31u;
    const auto layout = ::ps::runtime::Int8ActivationWorkspaceLayout::make(k, rows);
    const auto h_in = make_input(rows, k, pattern);

    Result gen;
    Result vec;
    std::string err;
    if (!run_kernel(h_in, rows, k, kp, true, arena, stream, gen, err)) {
        std::printf("FAIL: generic rows=%u k=%u %s: %s\n", rows, k, pattern_name(pattern), err.c_str());
        failed++;
        return false;
    }
    if (!run_kernel(h_in, rows, k, kp, false, arena, stream, vec, err)) {
        std::printf("FAIL: vec rows=%u k=%u %s: %s\n", rows, k, pattern_name(pattern), err.c_str());
        failed++;
        return false;
    }

    std::vector<uint8_t> ref_codes;
    std::vector<float> ref_scales;
    cpu_reference(h_in, rows, k, kp, layout.code_row_stride_bytes,
                  layout.scale_row_stride_bytes, ref_codes, ref_scales);

    const bool cpu_codes_ok = (gen.codes == ref_codes);
    const bool cpu_scales_ok = (std::memcmp(gen.scales.data(), ref_scales.data(),
                                            gen.scales.size() * 4) == 0);
    const bool codes_ok = (vec.codes == gen.codes);
    const bool scales_ok = (std::memcmp(vec.scales.data(), gen.scales.data(),
                                        vec.scales.size() * 4) == 0);

    if (!cpu_codes_ok || !cpu_scales_ok) {
        for (size_t i = 0; i < gen.codes.size() && i < ref_codes.size(); ++i) {
            if (gen.codes[i] != ref_codes[i]) {
                std::printf("  [diag rows=%u k=%u %s] code idx=%zu gpu=0x%02x cpu=0x%02x\n",
                            rows, k, pattern_name(pattern), i, gen.codes[i], ref_codes[i]);
                break;
            }
        }
        for (size_t i = 0; i < gen.scales.size(); ++i) {
            if (std::memcmp(&gen.scales[i], &ref_scales[i], 4) != 0) {
                std::printf("  [diag rows=%u k=%u %s] scale idx=%zu gpu=%.9g cpu=%.9g\n",
                            rows, k, pattern_name(pattern), i, gen.scales[i], ref_scales[i]);
                break;
            }
        }
    }

    char label[224];
    std::snprintf(label, sizeof(label),
                  "rows=%-3u k=%-5u %-15s cpu-codes=%d cpu-scales=%d vec-codes=%d vec-scales=%d",
                  rows, k, pattern_name(pattern), cpu_codes_ok ? 1 : 0, cpu_scales_ok ? 1 : 0,
                  codes_ok ? 1 : 0, scales_ok ? 1 : 0);
    check(cpu_codes_ok && cpu_scales_ok && codes_ok && scales_ok, label);
    return cpu_codes_ok && cpu_scales_ok && codes_ok && scales_ok;
}

bool check_padding_generic(uint32_t rows, uint32_t k, ps::gpu::GpuArena& arena,
                           hipStream_t stream) {
    const uint32_t kp = (k + 31u) & ~31u;
    const auto layout = ::ps::runtime::Int8ActivationWorkspaceLayout::make(k, rows);
    const auto h_in = make_input(rows, k, kMixedAll);
    Result gen;
    std::string err;
    if (!run_kernel(h_in, rows, k, kp, true, arena, stream, gen, err)) {
        std::printf("FAIL: padding generic rows=%u k=%u: %s\n", rows, k, err.c_str());
        failed++;
        return false;
    }
    std::vector<uint8_t> ref_codes;
    std::vector<float> ref_scales;
    cpu_reference(h_in, rows, k, kp, layout.code_row_stride_bytes,
                  layout.scale_row_stride_bytes, ref_codes, ref_scales);
    const bool ok = (gen.codes == ref_codes) &&
                    (std::memcmp(gen.scales.data(), ref_scales.data(),
                                 gen.scales.size() * 4) == 0);
    char label[192];
    std::snprintf(label, sizeof(label), "padding-generic rows=%-3u k=%-5u kp=%-5u", rows, k, kp);
    check(ok, label);
    return ok;
}

bool check_fallback_shape(uint32_t rows, uint32_t k, ps::gpu::GpuArena& arena,
                          hipStream_t stream) {
    const uint32_t kp = (k + 31u) & ~31u;
    const auto layout = ::ps::runtime::Int8ActivationWorkspaceLayout::make(k, rows);
    const auto h_in = make_input(rows, k, kRandom);
    Result gen;
    Result vec;
    std::string err;
    if (!run_kernel(h_in, rows, k, kp, true, arena, stream, gen, err) ||
        !run_kernel(h_in, rows, k, kp, false, arena, stream, vec, err, false)) {
        std::printf("FAIL: fallback rows=%u k=%u: %s\n", rows, k, err.c_str());
        failed++;
        return false;
    }
    const bool codes_ok = (vec.codes == gen.codes);
    const bool scales_ok = (std::memcmp(vec.scales.data(), gen.scales.data(),
                                        vec.scales.size() * 4) == 0);
    char label[192];
    std::snprintf(label, sizeof(label), "fallback-equal rows=%-3u k=%-5u kp=%-5u", rows, k, kp);
    check(codes_ok && scales_ok, label);
    return codes_ok && scales_ok;
}


}  // namespace

int main() {
    std::printf("=== Activation Quantize E4M3 (specialized vs generic) Tests ===\n");
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

    const uint32_t production_k[] = {5120u, 6144u, 17408u, 8704u, 3072u};
    const uint32_t row_list[] = {1u, 2u, 16u, 128u};

    std::printf("\n-- production shapes: all patterns, specialized vs generic vs cpu --\n");
    for (uint32_t rows : row_list) {
        for (uint32_t k : production_k) {
            for (uint32_t p = 0u; p < kPatternCount; ++p) {
                compare_cases(rows, k, p, arena, stream);
            }
        }
    }

    std::printf("\n-- generic padding semantics (k not multiple of 32) --\n");
    {
        const uint32_t padding_k[] = {1u, 31u, 32u, 33u, 100u, 2560u, 9216u, 5000u};
        for (uint32_t rows : row_list) {
            for (uint32_t k : padding_k) {
                check_padding_generic(rows, k, arena, stream);
            }
        }
    }

    std::printf("\n-- non-production k: auto dispatch equals generic --\n");
    {
        const uint32_t other_k[] = {1024u, 10240u, 2048u, 4096u, 8192u};
        for (uint32_t rows : row_list) {
            for (uint32_t k : other_k) {
                check_fallback_shape(rows, k, arena, stream);
            }
        }
    }

    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    std::printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
