#include <shelf/inference/execution/exl3_dense_gemm.h>

#include <hipblaslt/hipblaslt.h>

#include <cstdint>

namespace shelf::execution {
namespace {

constexpr uint32_t kCacheCapacity = 16u;

struct AlgoKey {
    uint32_t rows = 0;
    uint32_t out_features = 0;
    uint32_t k = 0;
    hipblasLtMatmulAlgo_t algo{};
    bool valid = false;
};

struct LtState {
    hipblasLtHandle_t handle = nullptr;
    AlgoKey cache[kCacheCapacity];

    ~LtState() {
        if (handle != nullptr) {
            (void) hipblasLtDestroy(handle);
            handle = nullptr;
        }
    }
};

LtState& state() {
    static thread_local LtState value;
    return value;
}

const hipblasLtMatmulAlgo_t* find_algo(LtState& lt, uint32_t rows, uint32_t n, uint32_t k) {
    for (uint32_t i = 0u; i < kCacheCapacity; ++i) {
        const AlgoKey& entry = lt.cache[i];
        if (entry.valid && entry.rows == rows && entry.out_features == n && entry.k == k) {
            return &entry.algo;
        }
    }
    return nullptr;
}

void store_algo(LtState& lt, uint32_t rows, uint32_t n, uint32_t k,
                const hipblasLtMatmulAlgo_t& algo) {
    for (uint32_t i = 0u; i < kCacheCapacity; ++i) {
        if (!lt.cache[i].valid) {
            lt.cache[i] = AlgoKey{rows, n, k, algo, true};
            return;
        }
    }
}

}  // namespace

hipError_t exl3_dense_gemm(const shelf::kernel::Exl3GemmArgs& args, const void* dense,
                           void* accum, hipStream_t stream) {
    if (dense == nullptr || accum == nullptr || args.xrot == nullptr) {
        return hipErrorInvalidValue;
    }
    LtState& lt = state();
    if (lt.handle == nullptr) {
        if (hipblasLtCreate(&lt.handle) != HIPBLAS_STATUS_SUCCESS) {
            lt.handle = nullptr;
            return hipErrorInitializationError;
        }
    }

    hipblasLtMatmulDesc_t desc = nullptr;
    hipblasLtMatrixLayout_t la = nullptr;
    hipblasLtMatrixLayout_t lb = nullptr;
    hipblasLtMatrixLayout_t lc = nullptr;
    hipblasLtMatmulPreference_t pref = nullptr;
    if (hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F) !=
        HIPBLAS_STATUS_SUCCESS) {
        return hipErrorInitializationError;
    }
    hipblasOperation_t op = HIPBLAS_OP_N;
    (void) hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op,
                                           sizeof(op));
    (void) hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op,
                                           sizeof(op));
    const uint64_t m = args.rows;
    const uint64_t n = args.out_features;
    const uint64_t k = args.k;
    (void) hipblasLtMatrixLayoutCreate(&la, HIP_R_16F, n, k, n);
    (void) hipblasLtMatrixLayoutCreate(&lb, HIP_R_16F, k, m, k);
    (void) hipblasLtMatrixLayoutCreate(&lc, HIP_R_32F, n, m, n);
    (void) hipblasLtMatmulPreferenceCreate(&pref);
    const size_t workspace_limit = 0u;
    (void) hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_limit,
        sizeof(workspace_limit));

    const hipblasLtMatmulAlgo_t* cached = find_algo(lt, args.rows, args.out_features, args.k);
    hipblasLtMatmulAlgo_t algo{};
    size_t workspace_size = 0u;
    if (cached != nullptr) {
        algo = *cached;
    } else {
        hipblasLtMatmulHeuristicResult_t result{};
        int found = 0;
        if (hipblasLtMatmulAlgoGetHeuristic(lt.handle, desc, la, lb, lc, lc, pref, 1,
                                            &result, &found) != HIPBLAS_STATUS_SUCCESS ||
            found == 0) {
            (void) hipblasLtMatmulPreferenceDestroy(pref);
            (void) hipblasLtMatrixLayoutDestroy(la);
            (void) hipblasLtMatrixLayoutDestroy(lb);
            (void) hipblasLtMatrixLayoutDestroy(lc);
            (void) hipblasLtMatmulDescDestroy(desc);
            return hipErrorNotSupported;
        }
        algo = result.algo;
        workspace_size = result.workspaceSize;
        store_algo(lt, args.rows, args.out_features, args.k, algo);
    }

    void* workspace = nullptr;
    if (workspace_size > 0u) {
        if (hipMalloc(&workspace, workspace_size) != hipSuccess) {
            (void) hipblasLtMatmulPreferenceDestroy(pref);
            (void) hipblasLtMatrixLayoutDestroy(la);
            (void) hipblasLtMatrixLayoutDestroy(lb);
            (void) hipblasLtMatrixLayoutDestroy(lc);
            (void) hipblasLtMatmulDescDestroy(desc);
            return hipErrorOutOfMemory;
        }
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const hipblasStatus_t status =
        hipblasLtMatmul(lt.handle, desc, &alpha, dense, la, args.xrot, lb, &beta, accum, lc,
                        accum, lc, &algo, workspace, workspace_size, stream);
    if (workspace != nullptr) (void) hipFree(workspace);
    (void) hipblasLtMatmulPreferenceDestroy(pref);
    (void) hipblasLtMatrixLayoutDestroy(la);
    (void) hipblasLtMatrixLayoutDestroy(lb);
    (void) hipblasLtMatrixLayoutDestroy(lc);
    (void) hipblasLtMatmulDescDestroy(desc);
    return status == HIPBLAS_STATUS_SUCCESS ? hipSuccess : hipErrorNotSupported;
}

}  // namespace shelf::execution
