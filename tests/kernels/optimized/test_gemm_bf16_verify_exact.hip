#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/gpu/gfx_arch.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/kernels/optimized/linear/bf16.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

int passed = 0;
int failed = 0;

void check(bool cond, const char* name) {
    if (cond) {
        ++passed;
        std::printf("PASS: %s\n", name);
    } else {
        ++failed;
        std::printf("FAIL: %s\n", name);
    }
}

uint16_t f32_bf16(float f) {
    return __bfloat16_as_ushort(__float2bfloat16(f));
}

bool run_rows(
    uint32_t rows,
    uint32_t N,
    uint32_t K,
    uint32_t in_stride,
    uint32_t out_stride,
    uint32_t seed,
    ps::kernel::Bf16GemmOutputDType odt,
    ps::gpu::GpuArena& arena,
    hipStream_t stream,
    const char* label) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> wd(-0.3f, 0.3f);
    std::uniform_real_distribution<float> xd(-1.0f, 1.0f);
    std::vector<uint16_t> w(static_cast<size_t>(N) * K);
    std::vector<uint16_t> x(static_cast<size_t>(rows) * in_stride, 0);
    for (auto& v : w) v = f32_bf16(wd(rng));
    for (uint32_t m = 0; m < rows; ++m)
        for (uint32_t c = 0; c < K; ++c)
            x[static_cast<size_t>(m) * in_stride + c] = f32_bf16(xd(rng));

    const std::size_t elem = odt == ps::kernel::Bf16GemmOutputDType::BF16 ? 2u : 4u;
    const std::size_t out_bytes = static_cast<std::size_t>(rows) * out_stride * elem;
    auto wr = arena.allocate_aligned(w.size() * 2u, 256);
    auto xr = arena.allocate_aligned(x.size() * 2u, 256);
    auto exact_r = arena.allocate_aligned(out_bytes, 256);
    auto ref_r = arena.allocate_aligned(out_bytes, 256);
    if (!wr.ok() || !xr.ok() || !exact_r.ok() || !ref_r.ok()) {
        check(false, label);
        return false;
    }
    uint16_t* d_w = static_cast<uint16_t*>(wr.release().data());
    uint16_t* d_x = static_cast<uint16_t*>(xr.release().data());
    void* d_exact = exact_r.release().data();
    void* d_ref = ref_r.release().data();
    if (hipMemcpy(d_w, w.data(), w.size() * 2u, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d_x, x.data(), x.size() * 2u, hipMemcpyHostToDevice) != hipSuccess) {
        check(false, label);
        return false;
    }
    const ::ps::bf16_t* dw = reinterpret_cast<const ::ps::bf16_t*>(d_w);
    const ::ps::bf16_t* dx = reinterpret_cast<const ::ps::bf16_t*>(d_x);

    const ps::kernel::Bf16GemmConfig exact_cfg{
        ps::kernel::Bf16GemmConfigId::ExactRows, static_cast<uint8_t>(rows)};
    if (::ps::kernel::launch_gemm_bf16(
            exact_cfg, dw, dx, d_exact, odt, rows, N, K, in_stride, out_stride, stream) !=
        hipSuccess) {
        check(false, label);
        return false;
    }
    const ps::kernel::Bf16GemmConfig row_cfg{
        ps::kernel::Bf16GemmConfigId::ExactRows, 1u};
    for (uint32_t m = 0; m < rows; ++m) {
        void* dst = static_cast<uint8_t*>(d_ref) + static_cast<std::size_t>(m) * out_stride * elem;
        if (::ps::kernel::launch_gemm_bf16(
                row_cfg, dw, dx + static_cast<std::size_t>(m) * in_stride, dst, odt, 1u, N, K,
                in_stride, out_stride, stream) != hipSuccess) {
            check(false, label);
            return false;
        }
    }
    if (hipStreamSynchronize(stream) != hipSuccess) {
        check(false, label);
        return false;
    }
    std::vector<uint8_t> exact(out_bytes);
    std::vector<uint8_t> ref(out_bytes);
    if (hipMemcpy(exact.data(), d_exact, out_bytes, hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(ref.data(), d_ref, out_bytes, hipMemcpyDeviceToHost) != hipSuccess) {
        check(false, label);
        return false;
    }
    bool ok = true;
    std::size_t first = 0;
    for (std::size_t i = 0; i < out_bytes; ++i) {
        if (exact[i] != ref[i]) {
            ok = false;
            first = i;
            break;
        }
    }
    if (!ok)
        std::printf("  first differing byte=%zu (of %zu) rows=%u N=%u K=%u\n", first, out_bytes,
                    rows, N, K);
    check(ok, label);
    return ok;
}

}  // namespace

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;
    auto arena_result = ps::gpu::GpuArena::create(0, 2ull * 1024ull * 1024ull * 1024ull);
    if (!arena_result.ok()) {
        std::printf("FAIL: arena\n");
        return 1;
    }
    ps::gpu::GpuArena arena = arena_result.release();

    const uint32_t row_list[] = {2u, 3u, 5u, 6u, 9u, 16u};
    for (uint32_t rows : row_list) {
        char label[160];
        const uint32_t K = 256u;
        const uint32_t N = 256u;
        std::snprintf(label, sizeof(label), "exact vs M=1 gemv f32 rows=%u N=%u K=%u", rows, N, K);
        run_rows(rows, N, K, K, N, 0x1234u + rows, ps::kernel::Bf16GemmOutputDType::F32, arena,
                 stream, label);
        std::snprintf(label, sizeof(label), "exact vs M=1 gemv bf16 rows=%u N=%u K=%u", rows, N, K);
        run_rows(rows, N, K, K, N, 0x5678u + rows, ps::kernel::Bf16GemmOutputDType::BF16, arena,
                 stream, label);
    }

    {
        char label[160];
        std::snprintf(label, sizeof(label), "exact vs M=1 gemv f32 rows=9 N=2560 K=2560");
        run_rows(9u, 2560u, 2560u, 2560u, 2560u, 0xA000u, ps::kernel::Bf16GemmOutputDType::F32,
                 arena, stream, label);
        std::snprintf(label, sizeof(label), "exact vs M=1 gemv bf16 rows=9 N=5120 K=10240");
        run_rows(9u, 5120u, 10240u, 10240u, 5120u, 0xA001u, ps::kernel::Bf16GemmOutputDType::BF16,
                 arena, stream, label);
    }

    {
        char label[160];
        std::snprintf(label, sizeof(label), "exact vs M=1 gemv f32 rows=5 K tail=520");
        run_rows(5u, 512u, 520u, 520u, 512u, 0xB000u, ps::kernel::Bf16GemmOutputDType::F32, arena,
                 stream, label);
        std::snprintf(label, sizeof(label), "exact vs M=1 gemv bf16 rows=9 in_stride=4096>K=3072");
        run_rows(9u, 1024u, 3072u, 4096u, 1024u, 0xB001u, ps::kernel::Bf16GemmOutputDType::BF16,
                 arena, stream, label);
    }

    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    std::printf("VERIFY_EXACT_GEMM: %s (%d passed, %d failed)\n", failed == 0 ? "PASS" : "FAIL",
                passed, failed);
    return failed > 0 ? 1 : 0;
}
