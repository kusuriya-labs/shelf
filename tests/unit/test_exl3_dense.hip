#include <shelf/core/memory/types.h>
#include <shelf/kernels/optimized/linear/exl3.h>

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace k = shelf::kernel;

namespace {

uint32_t g_rng = 0x2468ace1u;

uint32_t next_u32() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return g_rng;
}

bool run_case(uint32_t rows, uint32_t n, uint32_t k, uint32_t ka, bool half, const char* tag) {
    const uint32_t kt = k / 16u;
    const uint32_t nt = n / 16u;
    const uint32_t words = 16u * ka + (half ? 8u : 0u);
    const std::size_t tile_words = static_cast<std::size_t>(kt) * nt * words;

    void* d_trellis = nullptr;
    void* d_suh = nullptr;
    void* d_svh = nullptr;
    void* d_input = nullptr;
    void* d_xrot = nullptr;
    void* d_dense = nullptr;
    void* d_accum = nullptr;
    void* d_fused = nullptr;
    void* d_dense_out = nullptr;
    if (hipMalloc(&d_trellis, tile_words * 2u) != hipSuccess) return false;
    if (hipMalloc(&d_suh, k * 2u) != hipSuccess) return false;
    if (hipMalloc(&d_svh, n * 2u) != hipSuccess) return false;
    if (hipMalloc(&d_input, static_cast<std::size_t>(rows) * k * 2u) != hipSuccess) return false;
    if (hipMalloc(&d_xrot, static_cast<std::size_t>(rows) * k * 2u) != hipSuccess) return false;
    if (hipMalloc(&d_dense, static_cast<std::size_t>(k) * n * 2u) != hipSuccess) return false;
    if (hipMalloc(&d_accum, static_cast<std::size_t>(rows) * n * 4u) != hipSuccess) return false;
    if (hipMalloc(&d_fused, static_cast<std::size_t>(rows) * n * 4u) != hipSuccess) return false;
    if (hipMalloc(&d_dense_out, static_cast<std::size_t>(rows) * n * 4u) != hipSuccess) {
        return false;
    }

    std::vector<uint16_t> trellis(tile_words);
    for (uint16_t& v : trellis) v = static_cast<uint16_t>(next_u32() & 0xFFFFu);
    std::vector<uint16_t> suh(k);
    std::vector<uint16_t> svh(n);
    for (uint16_t& v : suh) v = static_cast<uint16_t>(0x3800u | (next_u32() & 0x3FFu));
    for (uint16_t& v : svh) v = static_cast<uint16_t>(0x3800u | (next_u32() & 0x3FFu));
    std::vector<shelf::bf16_t> input(static_cast<std::size_t>(rows) * k);
    for (shelf::bf16_t& v : input) {
        v.data = static_cast<uint16_t>(0x3C00u | (next_u32() & 0x0Fu));
    }
    if (hipMemcpy(d_trellis, trellis.data(), tile_words * 2u, hipMemcpyHostToDevice) !=
        hipSuccess) {
        return false;
    }
    if (hipMemcpy(d_suh, suh.data(), k * 2u, hipMemcpyHostToDevice) != hipSuccess) {
        return false;
    }
    if (hipMemcpy(d_svh, svh.data(), n * 2u, hipMemcpyHostToDevice) != hipSuccess) {
        return false;
    }
    if (hipMemcpy(d_input, input.data(), input.size() * 2u, hipMemcpyHostToDevice) !=
        hipSuccess) {
        return false;
    }

    k::Exl3GemmArgs args;
    args.trellis = d_trellis;
    args.suh = d_suh;
    args.svh = d_svh;
    args.input = d_input;
    args.output = d_fused;
    args.rows = rows;
    args.out_features = n;
    args.k = k;
    args.input_row_stride = k;
    args.output_row_stride = n;
    args.output_dtype = 1u;
    args.ka = ka;
    args.half = half ? 1u : 0u;
    args.splitk_splits = 1u;
    args.groups = 0u;

    if (k::launch_rotate_exl3(args, d_xrot, nullptr) != hipSuccess) return false;
    if (k::launch_gemm_exl3_w16a16_pre(args, d_xrot, nullptr) != hipSuccess) return false;

    if (k::launch_recon_exl3_dense(args, d_dense, nullptr) != hipSuccess) return false;

    hipblasLtHandle_t lt = nullptr;
    if (hipblasLtCreate(&lt) != HIPBLAS_STATUS_SUCCESS) return false;
    hipblasLtMatmulDesc_t desc = nullptr;
    if (hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F) !=
        HIPBLAS_STATUS_SUCCESS) {
        return false;
    }
    hipblasOperation_t op = HIPBLAS_OP_N;
    (void) hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op,
                                           sizeof(op));
    (void) hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op,
                                           sizeof(op));
    hipblasLtMatrixLayout_t la = nullptr;
    hipblasLtMatrixLayout_t lb = nullptr;
    hipblasLtMatrixLayout_t lc = nullptr;
    (void) hipblasLtMatrixLayoutCreate(&la, HIP_R_16F, n, k, n);
    (void) hipblasLtMatrixLayoutCreate(&lb, HIP_R_16F, k, rows, k);
    (void) hipblasLtMatrixLayoutCreate(&lc, HIP_R_32F, n, rows, n);
    hipblasLtMatmulPreference_t pref = nullptr;
    (void) hipblasLtMatmulPreferenceCreate(&pref);
    const size_t workspace_limit = 0u;
    (void) hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_limit,
        sizeof(workspace_limit));
    hipblasLtMatmulHeuristicResult_t heuristic{};
    int found = 0;
    const hipblasStatus_t herr = hipblasLtMatmulAlgoGetHeuristic(
        lt, desc, la, lb, lc, lc, pref, 1, &heuristic, &found);
    bool ok = herr == HIPBLAS_STATUS_SUCCESS && found != 0;
    if (ok) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        ok = hipblasLtMatmul(lt, desc, &alpha, d_dense, la, d_xrot, lb, &beta, d_accum, lc,
                             d_accum, lc, &heuristic.algo, nullptr,
                             heuristic.workspaceSize, nullptr) == HIPBLAS_STATUS_SUCCESS;
    }
    if (ok) {
        args.output = d_dense_out;
        ok = k::launch_epilogue_exl3_dense(args, d_xrot, d_accum, nullptr) == hipSuccess;
    }
    (void) hipblasLtMatmulPreferenceDestroy(pref);
    (void) hipblasLtMatrixLayoutDestroy(la);
    (void) hipblasLtMatrixLayoutDestroy(lb);
    (void) hipblasLtMatrixLayoutDestroy(lc);
    (void) hipblasLtMatmulDescDestroy(desc);
    (void) hipblasLtDestroy(lt);
    if (!ok) {
        std::printf("FAIL %s launch\n", tag);
        return false;
    }
    if (hipDeviceSynchronize() != hipSuccess) {
        std::printf("FAIL %s sync\n", tag);
        return false;
    }

    const std::size_t count = static_cast<std::size_t>(rows) * n;
    std::vector<float> fused(count);
    std::vector<float> dense(count);
    if (hipMemcpy(fused.data(), d_fused, count * 4u, hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(dense.data(), d_dense_out, count * 4u, hipMemcpyDeviceToHost) != hipSuccess) {
        return false;
    }
    double max_abs = 0.0;
    double max_rel = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        const double d = std::fabs(static_cast<double>(fused[i]) - dense[i]);
        if (d > max_abs) max_abs = d;
        const double rel = d / (1e-6 + std::fabs(static_cast<double>(fused[i])));
        if (rel > max_rel) max_rel = rel;
    }
    const bool numeric = max_abs <= 2e-2 || max_rel <= 3e-3;
    if (!numeric) {
        std::printf("FAIL %s numeric max_abs=%.6g max_rel=%.6g\n", tag, max_abs, max_rel);
    }
    return numeric;
}

}  // namespace

int main() {
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
        std::printf("SKIP no GPU\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("FAIL hipSetDevice\n");
        return 1;
    }
    bool ok = true;
    ok = run_case(256, 1024, 512, 4, false, "dense_ka4_rows256") && ok;
    ok = run_case(320, 1024, 512, 4, false, "dense_ka4_rows320") && ok;
    ok = run_case(256, 1024, 512, 6, false, "dense_ka6_rows256") && ok;
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
