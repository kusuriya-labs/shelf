#include <shelf/kernels/optimized/linear/exl3.h>
#include <shelf/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <hipblaslt/hipblaslt.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace k = shelf::kernel;
namespace shelf::kernel {
hipError_t launch_gemm_exl3_w16a16_dense_pre(const Exl3GemmArgs&, const void*, const void*,
                                             hipStream_t);
hipError_t launch_recon_tiles(const Exl3GemmArgs&, void*, hipStream_t);
}

static uint32_t g_rng = 0x12345678u;
static uint32_t next_u32() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return g_rng;
}

static double time_gemm(const k::Exl3GemmArgs& args, const void* xrot, const void* dense,
                        bool use_dense, int warmup, int samples) {
    for (int i = 0; i < warmup; ++i) {
        if (use_dense) {
            k::launch_gemm_exl3_w16a16_dense_pre(args, xrot, dense, nullptr);
        } else {
            k::launch_gemm_exl3_w16a16_pre(args, xrot, nullptr);
        }
    }
    hipDeviceSynchronize();
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    double ms = 0.0;
    for (int i = 0; i < samples; ++i) {
        hipEventRecord(start, nullptr);
        if (use_dense) {
            k::launch_gemm_exl3_w16a16_dense_pre(args, xrot, dense, nullptr);
        } else {
            k::launch_gemm_exl3_w16a16_pre(args, xrot, nullptr);
        }
        hipEventRecord(stop, nullptr);
        hipEventSynchronize(stop);
        float t = 0.0f;
        hipEventElapsedTime(&t, start, stop);
        ms += t;
    }
    hipEventDestroy(start);
    hipEventDestroy(stop);
    return ms / samples;
}

static void compare(const char* tag, const std::vector<float>& got, const std::vector<float>& ref) {
    double max_abs = 0.0;
    double sum_abs = 0.0;
    double max_rel = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = std::fabs(static_cast<double>(got[i]) - ref[i]);
        max_abs = d > max_abs ? d : max_abs;
        sum_abs += d;
        const double denom = 1e-6 + std::fabs(static_cast<double>(ref[i]));
        const double r = d / denom;
        max_rel = r > max_rel ? r : max_rel;
    }
    std::printf("CORRECT %s max_abs=%.6g mean_abs=%.6g max_rel=%.6g\n", tag, max_abs,
                sum_abs / static_cast<double>(got.size()), max_rel);
}

int main() {
    constexpr uint32_t M = 2048;
    constexpr uint32_t K = 5120;
    constexpr uint32_t N = 5120;
    constexpr uint32_t ka = 4;
    constexpr std::size_t kScratchFloats = 3145728;
    const uint32_t kt = K / 16u;
    const uint32_t nt = N / 16u;
    const std::size_t tcount = static_cast<std::size_t>(kt) * nt * 64u;

    void* d_trellis = nullptr;
    void* d_suh = nullptr;
    void* d_svh = nullptr;
    void* d_input = nullptr;
    void* d_xrot = nullptr;
    void* d_dense = nullptr;
    void* d_out_b0 = nullptr;
    void* d_out_b3 = nullptr;
    void* d_scratch = nullptr;
    if (hipMalloc(&d_trellis, tcount * 2u) != hipSuccess) return 1;
    if (hipMalloc(&d_suh, K * 2u) != hipSuccess) return 1;
    if (hipMalloc(&d_svh, N * 2u) != hipSuccess) return 1;
    if (hipMalloc(&d_input, static_cast<std::size_t>(M) * K * 2u) != hipSuccess) return 1;
    if (hipMalloc(&d_xrot, static_cast<std::size_t>(M) * K * 2u) != hipSuccess) return 1;
    if (hipMalloc(&d_dense, static_cast<std::size_t>(K) * N * 2u) != hipSuccess) return 1;
    if (hipMalloc(&d_out_b0, static_cast<std::size_t>(M) * N * 4u) != hipSuccess) return 1;
    if (hipMalloc(&d_out_b3, static_cast<std::size_t>(M) * N * 4u) != hipSuccess) return 1;
    if (hipMalloc(&d_scratch, kScratchFloats * 4u) != hipSuccess) return 1;

    std::vector<uint16_t> trellis(tcount);
    for (uint16_t& v : trellis) v = static_cast<uint16_t>(next_u32() & 0xFFFFu);
    std::vector<uint16_t> suh(K), svh(N);
    for (uint16_t& v : suh) v = shelf::f32_to_bf16_rne(0.01f * static_cast<float>(next_u32() & 0xFFu));
    for (uint16_t& v : svh) v = shelf::f32_to_bf16_rne(0.01f * static_cast<float>(next_u32() & 0xFFu));
    std::vector<shelf::bf16_t> input(static_cast<std::size_t>(M) * K);
    for (shelf::bf16_t& v : input) v.data = shelf::f32_to_bf16_rne(0.01f * static_cast<float>(next_u32() & 0xFFu));
    hipMemcpy(d_trellis, trellis.data(), tcount * 2u, hipMemcpyHostToDevice);
    hipMemcpy(d_suh, suh.data(), K * 2u, hipMemcpyHostToDevice);
    hipMemcpy(d_svh, svh.data(), N * 2u, hipMemcpyHostToDevice);
    hipMemcpy(d_input, input.data(), input.size() * 2u, hipMemcpyHostToDevice);

    k::Exl3GemmArgs args;
    args.trellis = d_trellis;
    args.suh = d_suh;
    args.svh = d_svh;
    args.input = d_input;
    args.output = d_out_b3;
    args.rows = M;
    args.out_features = N;
    args.k = K;
    args.input_row_stride = K;
    args.output_row_stride = N;
    args.output_dtype = 1u;
    args.ka = ka;
    args.half = 0u;
    args.splitk_scratch = static_cast<float*>(d_scratch);
    args.splitk_floats = static_cast<uint32_t>(kScratchFloats);
    args.splitk_splits = 1u;
    args.groups = 1u;

    k::launch_rotate_exl3(args, d_xrot, nullptr);
    k::launch_recon_tiles(args, d_dense, nullptr);
    hipDeviceSynchronize();

    args.output = d_out_b3;
    const double b3_ms = time_gemm(args, d_xrot, nullptr, false, 10, 50);
    args.output = d_out_b0;
    const double b0_ms = time_gemm(args, d_xrot, d_dense, true, 10, 50);

    std::vector<float> out_b3(static_cast<std::size_t>(M) * N);
    std::vector<float> out_b0(static_cast<std::size_t>(M) * N);
    hipMemcpy(out_b3.data(), d_out_b3, out_b3.size() * 4u, hipMemcpyDeviceToHost);
    hipMemcpy(out_b0.data(), d_out_b0, out_b0.size() * 4u, hipMemcpyDeviceToHost);
    compare("B0_vs_B3", out_b0, out_b3);

    void* dA = nullptr;
    void* dB = nullptr;
    void* dC = nullptr;
    if (hipMalloc(&dA, static_cast<std::size_t>(K) * M * 2u) != hipSuccess) return 1;
    if (hipMalloc(&dB, static_cast<std::size_t>(K) * N * 2u) != hipSuccess) return 1;
    if (hipMalloc(&dC, static_cast<std::size_t>(M) * N * 4u) != hipSuccess) return 1;
    hipMemset(dA, 0, static_cast<std::size_t>(K) * M * 2u);
    hipMemset(dB, 0, static_cast<std::size_t>(K) * N * 2u);
    rocblas_handle handle = nullptr;
    rocblas_create_handle(&handle);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    for (int i = 0; i < 10; ++i) {
        rocblas_gemm_ex(handle, rocblas_operation_none, rocblas_operation_none, M, N, K,
                        &alpha, dA, rocblas_datatype_f16_r, M, dB, rocblas_datatype_f16_r,
                        K, &beta, dC, rocblas_datatype_f32_r, M, dC,
                        rocblas_datatype_f32_r, M, rocblas_datatype_f32_r,
                        rocblas_gemm_algo_standard, 0, 0u);
    }
    hipDeviceSynchronize();
    hipEvent_t a0_start = nullptr;
    hipEvent_t a0_stop = nullptr;
    hipEventCreate(&a0_start);
    hipEventCreate(&a0_stop);
    double a0_ms = 0.0;
    constexpr int kSamples = 50;
    for (int i = 0; i < kSamples; ++i) {
        hipEventRecord(a0_start, nullptr);
        rocblas_gemm_ex(handle, rocblas_operation_none, rocblas_operation_none, M, N, K,
                        &alpha, dA, rocblas_datatype_f16_r, M, dB, rocblas_datatype_f16_r,
                        K, &beta, dC, rocblas_datatype_f32_r, M, dC,
                        rocblas_datatype_f32_r, M, rocblas_datatype_f32_r,
                        rocblas_gemm_algo_standard, 0, 0u);
        hipEventRecord(a0_stop, nullptr);
        hipEventSynchronize(a0_stop);
        float t = 0.0f;
        hipEventElapsedTime(&t, a0_start, a0_stop);
        a0_ms += t;
    }
    a0_ms /= kSamples;
    rocblas_destroy_handle(handle);

    hipblasLtHandle_t lt = nullptr;
    hipblasLtCreate(&lt);
    hipblasLtMatmulDesc_t lt_desc = nullptr;
    hipblasLtMatmulDescCreate(&lt_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F);
    hipblasOperation_t lt_op = HIPBLAS_OP_N;
    hipblasLtMatmulDescSetAttribute(lt_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &lt_op,
                                    sizeof(lt_op));
    hipblasLtMatmulDescSetAttribute(lt_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &lt_op,
                                    sizeof(lt_op));
    hipblasLtMatrixLayout_t lt_a = nullptr;
    hipblasLtMatrixLayout_t lt_b = nullptr;
    hipblasLtMatrixLayout_t lt_c = nullptr;
    hipblasLtMatrixLayoutCreate(&lt_a, HIP_R_16F, M, K, M);
    hipblasLtMatrixLayoutCreate(&lt_b, HIP_R_16F, K, N, K);
    hipblasLtMatrixLayoutCreate(&lt_c, HIP_R_32F, M, N, M);
    hipblasLtMatmulPreference_t lt_pref = nullptr;
    hipblasLtMatmulPreferenceCreate(&lt_pref);
    size_t lt_ws_size = 32u << 20;
    hipblasLtMatmulPreferenceSetAttribute(lt_pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                          &lt_ws_size, sizeof(lt_ws_size));
    hipblasLtMatmulHeuristicResult_t lt_heur{};
    int lt_got = 0;
    hipblasLtMatmulAlgoGetHeuristic(lt, lt_desc, lt_a, lt_b, lt_c, lt_c, lt_pref, 1,
                                    &lt_heur, &lt_got);
    void* lt_ws = nullptr;
    if (lt_heur.workspaceSize > 0) hipMalloc(&lt_ws, lt_heur.workspaceSize);
    for (int i = 0; i < 10; ++i) {
        hipblasLtMatmul(lt, lt_desc, &alpha, dA, lt_a, dB, lt_b, &beta, dC, lt_c, dC, lt_c,
                        &lt_heur.algo, lt_ws, lt_heur.workspaceSize, 0);
    }
    hipDeviceSynchronize();
    double lt_ms = 0.0;
    for (int i = 0; i < kSamples; ++i) {
        hipEventRecord(a0_start, nullptr);
        hipblasLtMatmul(lt, lt_desc, &alpha, dA, lt_a, dB, lt_b, &beta, dC, lt_c, dC, lt_c,
                        &lt_heur.algo, lt_ws, lt_heur.workspaceSize, 0);
        hipEventRecord(a0_stop, nullptr);
        hipEventSynchronize(a0_stop);
        float t = 0.0f;
        hipEventElapsedTime(&t, a0_start, a0_stop);
        lt_ms += t;
    }
    lt_ms /= kSamples;

    const double flops = 2.0 * M * K * N;
    auto report = [&](const char* stage, double ms) {
        std::printf("RESULT %s M=%u K=%u N=%u g=1 s=1 kernel_ms=%.4f tok_s=%.1f tflops=%.2f\n",
                    stage, M, K, N, ms, M * 1000.0 / ms, flops / (ms * 1e9));
    };
    report("A0_rocblas_f16", a0_ms);
    report("A0b_hipblaslt_f16", lt_ms);
    report("B0_wmma_fp16_direct", b0_ms);
    report("B3_exl3_current", b3_ms);
    return 0;
}
