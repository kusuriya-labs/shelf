#include <shelf/core/gpu/gfx_arch.h>
#include <shelf/kernels/optimized/linear/bf16.h>
#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/types.h>
#include <shelf/core/gpu/cleanup.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <random>

static shelf::kernel::Bf16GemmConfig bf16_config_for(
    uint32_t rows, uint32_t out_features, uint32_t k) {
    using Id = shelf::kernel::Bf16GemmConfigId;
    if (rows == 1u && (k & 7u) == 0u)
        return shelf::kernel::Bf16GemmConfig{Id::ExactRows, 1u};
    if (out_features <= shelf::kernel::kBf16GemmSplitKMaxOutFeatures)
        return shelf::kernel::Bf16GemmConfig{Id::WmmaKPartition, 0u};
    if (rows >= shelf::kernel::kBf16GemmWideMinRows)
        return shelf::kernel::Bf16GemmConfig{Id::WmmaWide, 0u};
    return shelf::kernel::Bf16GemmConfig{Id::Wmma, 0u};
}

static hipError_t launch_bf16_wmma(
    const shelf::bf16_t* weight, const shelf::bf16_t* input, void* output,
    shelf::kernel::Bf16GemmOutputDType output_dtype, uint32_t rows, uint32_t out_features,
    uint32_t k, uint32_t input_row_stride, uint32_t output_row_stride, hipStream_t stream) {
    return shelf::kernel::launch_gemm_bf16(
        bf16_config_for(rows, out_features, k),
        weight, input, output, output_dtype, rows, out_features, k,
        input_row_stride, output_row_stride, stream);
}

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

float bf16bits_f32(uint16_t bits) {
    uint32_t ui = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &ui, 4);
    return f;
}

uint16_t f32_bf16(float f) {
    return __bfloat16_as_ushort(__float2bfloat16(f));
}

// weight [N, K] row-major (row stride = K), input [rows, in_stride] row-major.
// out[m][o] = sum_{k < K} weight[o][k] * input[m][k]
void cpu_ref(const std::vector<uint16_t>& w, uint32_t N, uint32_t K,
             const std::vector<uint16_t>& x, uint32_t in_stride, uint32_t rows,
             std::vector<float>& ref, std::vector<double>& cond) {
    ref.assign(static_cast<size_t>(rows) * N, 0.f);
    cond.assign(static_cast<size_t>(rows) * N, 0.0);
    for (uint32_t m = 0; m < rows; ++m)
        for (uint32_t o = 0; o < N; ++o) {
            float acc = 0.f;
            double c = 0.0;
            for (uint32_t k = 0; k < K; ++k) {
                const float t = bf16bits_f32(w[static_cast<size_t>(o) * K + k]) *
                                bf16bits_f32(x[static_cast<size_t>(m) * in_stride + k]);
                acc += t;
                c += std::fabs(t);
            }
            ref[static_cast<size_t>(m) * N + o] = acc;
            cond[static_cast<size_t>(m) * N + o] = c;
        }
}

bool run_case(uint32_t rows, uint32_t N, uint32_t K, uint32_t in_stride, uint32_t out_stride,
              uint32_t seed, shelf::gpu::GpuArena& arena, hipStream_t stream, const char* label) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> wd(-0.2f, 0.2f);
    std::uniform_real_distribution<float> xd(-1.0f, 1.0f);
    std::vector<uint16_t> w(static_cast<size_t>(N) * K);
    std::vector<uint16_t> x(static_cast<size_t>(rows) * in_stride, 0);
    for (auto& v : w) v = f32_bf16(wd(rng));
    for (uint32_t m = 0; m < rows; ++m)
        for (uint32_t c = 0; c < K; ++c)
            x[static_cast<size_t>(m) * in_stride + c] = f32_bf16(xd(rng));

    const size_t w_bytes = w.size() * 2;
    const size_t x_bytes = x.size() * 2;
    const size_t of32_bytes = static_cast<size_t>(rows) * out_stride * 4;
    const size_t obf16_bytes = static_cast<size_t>(rows) * out_stride * 2;
    auto wr = arena.allocate_aligned(w_bytes, 16);
    auto xr = arena.allocate_aligned(x_bytes, 16);
    auto of32r = arena.allocate_aligned(of32_bytes, 16);
    auto obf16r = arena.allocate_aligned(obf16_bytes, 16);
    if (!wr.ok() || !xr.ok() || !of32r.ok() || !obf16r.ok()) {
        std::printf("FAIL: %s alloc\n", label);
        check(false, label);
        return false;
    }
    uint16_t* d_w = static_cast<uint16_t*>(wr.release().data());
    uint16_t* d_x = static_cast<uint16_t*>(xr.release().data());
    float* d_of32 = static_cast<float*>(of32r.release().data());
    ::shelf::bf16_t* d_obf16 = static_cast<::shelf::bf16_t*>(obf16r.release().data());
    if (hipMemcpy(d_w, w.data(), w_bytes, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d_x, x.data(), x_bytes, hipMemcpyHostToDevice) != hipSuccess) {
        check(false, label);
        return false;
    }
    std::vector<float> ref;
    std::vector<double> cond;
    cpu_ref(w, N, K, x, in_stride, rows, ref, cond);

    const ::shelf::bf16_t* dw = reinterpret_cast<const ::shelf::bf16_t*>(d_w);
    const ::shelf::bf16_t* dx = reinterpret_cast<const ::shelf::bf16_t*>(d_x);
    hipError_t hf = launch_bf16_wmma(
        dw, dx, d_of32, shelf::kernel::Bf16GemmOutputDType::F32,
        rows, N, K, in_stride, out_stride, stream);
    hipError_t hb = launch_bf16_wmma(
        dw, dx, d_obf16, shelf::kernel::Bf16GemmOutputDType::BF16,
        rows, N, K, in_stride, out_stride, stream);
    if (hf != hipSuccess || hb != hipSuccess || hipStreamSynchronize(stream) != hipSuccess) {
        check(false, label);
        return false;
    }
    std::vector<float> got32(static_cast<size_t>(rows) * out_stride, 0.f);
    std::vector<uint16_t> gotbf16(static_cast<size_t>(rows) * out_stride, 0);
    if (hipMemcpy(got32.data(), d_of32, of32_bytes, hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(gotbf16.data(), d_obf16, obf16_bytes, hipMemcpyDeviceToHost) != hipSuccess) {
        check(false, label);
        return false;
    }
    bool f32_ok = true, bf16_ok = true, finite_ok = true;
    float max_f32 = 0.f, max_bf16 = 0.f;
    size_t arg = 0;
    for (uint32_t m = 0; m < rows; ++m)
        for (uint32_t o = 0; o < N; ++o) {
            const size_t gi = static_cast<size_t>(m) * out_stride + o;
            const size_t ri = static_cast<size_t>(m) * N + o;
            const float g = got32[gi];
            if (std::isnan(g) || std::isinf(g)) finite_ok = false;
            const float ad = std::fabs(g - ref[ri]);
            if (ad > max_f32) { max_f32 = ad; arg = gi; }
            const bool ok32 = ad <= 5e-4f || (ad / (std::fabs(ref[ri]) + 1e-30f) <= 5e-4f);
            if (!ok32) f32_ok = false;
            const float gb = bf16bits_f32(gotbf16[gi]);
            const float db = std::fabs(gb - ref[ri]);
            if (db > max_bf16) max_bf16 = db;
            const float btol = 1.5f * 0.00390625f * std::max(1.0f, std::fabs(ref[ri]));
            if (!(db <= btol)) bf16_ok = false;
        }
    const bool ok = f32_ok && bf16_ok && finite_ok;
    if (!ok)
        std::printf("  mismatch arg=%zu max_f32=%.4g max_bf16=%.4g f32_ok=%d bf16_ok=%d finite=%d\n",
                    arg, max_f32, max_bf16, f32_ok ? 1 : 0, bf16_ok ? 1 : 0, finite_ok ? 1 : 0);
    check(ok, label);
    return ok;
}

// 16x16x16 F32 with a non-symmetric integer pattern (mapping sanity / isolation).
bool run_mapping_sanity(shelf::gpu::GpuArena& arena, hipStream_t stream) {
    const uint32_t rows = 16, N = 16, K = 16;
    std::vector<uint16_t> w(static_cast<size_t>(N) * K);
    std::vector<uint16_t> x(static_cast<size_t>(rows) * K);
    for (uint32_t o = 0; o < N; ++o)
        for (uint32_t k = 0; k < K; ++k)
            w[static_cast<size_t>(o) * K + k] = f32_bf16(static_cast<float>((o * 3u + k) % 7u) - 3.0f);
    for (uint32_t m = 0; m < rows; ++m)
        for (uint32_t k = 0; k < K; ++k)
            x[static_cast<size_t>(m) * K + k] = f32_bf16(static_cast<float>((m * 2u + k) % 5u) - 2.0f);

    const size_t w_bytes = w.size() * 2, x_bytes = x.size() * 2, out_bytes = (size_t)rows * N * 4;
    auto wr = arena.allocate_aligned(w_bytes, 16);
    auto xr = arena.allocate_aligned(x_bytes, 16);
    auto ofr = arena.allocate_aligned(out_bytes, 16);
    if (!wr.ok() || !xr.ok() || !ofr.ok()) { check(false, "mapping_sanity 16x16x16"); return false; }
    uint16_t* d_w = static_cast<uint16_t*>(wr.release().data());
    uint16_t* d_x = static_cast<uint16_t*>(xr.release().data());
    float* d_out = static_cast<float*>(ofr.release().data());
    if (hipMemcpy(d_w, w.data(), w_bytes, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d_x, x.data(), x_bytes, hipMemcpyHostToDevice) != hipSuccess) {
        check(false, "mapping_sanity 16x16x16"); return false;
    }
    std::vector<float> ref;
    std::vector<double> cond;
    cpu_ref(w, N, K, x, K, rows, ref, cond);
    const ::shelf::bf16_t* dw = reinterpret_cast<const ::shelf::bf16_t*>(d_w);
    const ::shelf::bf16_t* dx = reinterpret_cast<const ::shelf::bf16_t*>(d_x);
    if (launch_bf16_wmma(
            dw, dx, d_out, shelf::kernel::Bf16GemmOutputDType::F32, rows, N, K, K, N, stream) != hipSuccess ||
        hipStreamSynchronize(stream) != hipSuccess) {
        check(false, "mapping_sanity 16x16x16");
        return false;
    }
    std::vector<float> got(static_cast<size_t>(rows) * N, 0.f);
    if (hipMemcpy(got.data(), d_out, out_bytes, hipMemcpyDeviceToHost) != hipSuccess) {
        check(false, "mapping_sanity 16x16x16");
        return false;
    }
    bool ok = true;
    float mx = 0.f;
    for (size_t i = 0; i < got.size(); ++i)
        if (std::fabs(got[i] - ref[i]) > 1e-4f) { ok = false; mx = std::fabs(got[i] - ref[i]); }
    if (!ok) std::printf("  max diff=%.4g\n", mx);
    check(ok, "mapping_sanity 16x16x16 (A/B/C mapping, non-symmetric)");
    return ok;
}

// 16x16x32: K16#0 and K16#1 with distinct patterns to catch acc reset / K offset bug.
bool run_k32_accum(shelf::gpu::GpuArena& arena, hipStream_t stream) {
    const uint32_t rows = 16, N = 16, K = 32;
    std::vector<uint16_t> w(static_cast<size_t>(N) * K);
    std::vector<uint16_t> x(static_cast<size_t>(rows) * K);
    for (uint32_t o = 0; o < N; ++o)
        for (uint32_t k = 0; k < K; ++k) {
            const bool hi = k >= 16;
            const float v = hi ? (static_cast<float>((o + (k - 16)) % 9u) + 10.0f)
                               : (static_cast<float>((o + k) % 5u) - 2.0f);
            w[static_cast<size_t>(o) * K + k] = f32_bf16(v);
        }
    for (uint32_t m = 0; m < rows; ++m)
        for (uint32_t k = 0; k < K; ++k) {
            const bool hi = k >= 16;
            const float v = hi ? (static_cast<float>((m + (k - 16)) % 7u) + 20.0f)
                               : (static_cast<float>((m * 2u + k) % 3u) - 1.0f);
            x[static_cast<size_t>(m) * K + k] = f32_bf16(v);
        }
    const size_t w_bytes = w.size() * 2, x_bytes = x.size() * 2, out_bytes = (size_t)rows * N * 4;
    auto wr = arena.allocate_aligned(w_bytes, 16);
    auto xr = arena.allocate_aligned(x_bytes, 16);
    auto ofr = arena.allocate_aligned(out_bytes, 16);
    if (!wr.ok() || !xr.ok() || !ofr.ok()) { check(false, "k32_accum 16x16x32"); return false; }
    uint16_t* d_w = static_cast<uint16_t*>(wr.release().data());
    uint16_t* d_x = static_cast<uint16_t*>(xr.release().data());
    float* d_out = static_cast<float*>(ofr.release().data());
    if (hipMemcpy(d_w, w.data(), w_bytes, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d_x, x.data(), x_bytes, hipMemcpyHostToDevice) != hipSuccess) {
        check(false, "k32_accum 16x16x32"); return false;
    }
    std::vector<float> ref;
    std::vector<double> cond;
    cpu_ref(w, N, K, x, K, rows, ref, cond);
    const ::shelf::bf16_t* dw = reinterpret_cast<const ::shelf::bf16_t*>(d_w);
    const ::shelf::bf16_t* dx = reinterpret_cast<const ::shelf::bf16_t*>(d_x);
    if (launch_bf16_wmma(
            dw, dx, d_out, shelf::kernel::Bf16GemmOutputDType::F32, rows, N, K, K, N, stream) != hipSuccess ||
        hipStreamSynchronize(stream) != hipSuccess) {
        check(false, "k32_accum 16x16x32");
        return false;
    }
    std::vector<float> got(static_cast<size_t>(rows) * N, 0.f);
    if (hipMemcpy(got.data(), d_out, out_bytes, hipMemcpyDeviceToHost) != hipSuccess) {
        check(false, "k32_accum 16x16x32");
        return false;
    }
    bool ok = true;
    float mx = 0.f;
    for (size_t i = 0; i < got.size(); ++i) {
        const float d = std::fabs(got[i] - ref[i]);
        if (d > 1e-3f) { ok = false; mx = d; }
    }
    if (!ok) std::printf("  max diff=%.4g\n", mx);
    check(ok, "k32_accum 16x16x32 (K16#0 + K16#1 accumulate)");
    return ok;
}

}  // namespace

int main() {
    std::printf("=== BF16 WMMA GEMM (Phase G) Tests ===\n");
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;
    auto arena_result = shelf::gpu::GpuArena::create(0, 1024 * 1024 * 1024);
    if (!arena_result.ok()) {
        std::printf("FAIL: arena create: %s\n", arena_result.status().message().c_str());
        return 1;
    }
    shelf::gpu::GpuArena arena = arena_result.release();

    run_mapping_sanity(arena, stream);
    run_k32_accum(arena, stream);

    // M/N/K edge/tail (representative combos, not full product)
    const uint32_t mn_list[] = {1, 15, 16, 17, 31, 32, 33};
    const uint32_t k_list[] = {1, 3, 4, 5, 15, 16, 17, 31, 32, 33, 63, 64, 65};
    for (uint32_t rows : mn_list)
        for (uint32_t N : mn_list)
            for (uint32_t k : k_list) {
                if (static_cast<size_t>(rows) * N * k > 1200000u) continue;
                char label[128];
                std::snprintf(label, sizeof(label), "tail rows=%u N=%u K=%u", rows, N, k);
                run_case(rows, N, k, k, N, 0xC0FFEEu + rows * 7 + N * 13 + k, arena, stream, label);
            }

    // input/output stride > logical (K / N)
    {
        run_case(17, 17, 24, 40, 33, 0x51DEu, arena, stream, "stride in_stride=40>K=24 out_stride=33>N=17");
        run_case(33, 33, 64, 96, 48, 0x51DE1u, arena, stream, "stride in_stride=96>K=64 out_stride=48>N=33");
    }

    // unaligned input/weight row (K=17, stride 19 -> next row base misaligned)
    {
        run_case(17, 17, 17, 19, 17, 0x0BAD5EEDu, arena, stream, "unaligned K=17 in_stride=19");
        run_case(33, 9, 17, 19, 9, 0x0BAD5EE1u, arena, stream, "unaligned K=17 in_stride=19 rows=33");
    }

    // larger representative (still small enough for CPU ref)
    {
        run_case(1, 256, 256, 256, 256, 0xB1600000u, arena, stream, "rep rows=1 N=256 K=256");
        run_case(4, 128, 512, 512, 128, 0xB1600001u, arena, stream, "rep rows=4 N=128 K=512");
        run_case(64, 64, 256, 256, 64, 0xB1600002u, arena, stream, "rep rows=64 N=64 K=256");
    }

    // Qwen3.8-27B TP=2 local geometries.
    {
        run_case(1, 8704, 5120, 5120, 8704, 0x7B200001u, arena, stream,
                 "tp2 rows=1 N=8704 K=5120");
        run_case(1, 512, 5120, 5120, 512, 0x7B200002u, arena, stream,
                 "tp2 rows=1 N=512 K=5120");
        run_case(1, 5120, 8704, 8704, 5120, 0x7B200003u, arena, stream,
                 "tp2 rows=1 N=5120 K=8704");
        run_case(64, 3072, 5120, 5120, 3072, 0x7B200004u, arena, stream,
                 "tp2 rows=64 N=3072 K=5120");
        run_case(1, 24, 5120, 5120, 24, 0x7B200005u, arena, stream,
                 "tp2 rows=1 N=24 K=5120");
    }

    shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    std::printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
