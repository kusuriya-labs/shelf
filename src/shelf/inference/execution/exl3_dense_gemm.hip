#include <shelf/inference/execution/exl3_dense_gemm.h>

#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <cstdlib>

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
    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (cached != nullptr) {
        algo = *cached;
    } else {
        constexpr int kCandidates = 16;
        hipblasLtMatmulHeuristicResult_t results[kCandidates]{};
        int found = 0;
        if (hipblasLtMatmulAlgoGetHeuristic(lt.handle, desc, la, lb, lc, lc, pref,
                                            kCandidates, results, &found) !=
                HIPBLAS_STATUS_SUCCESS ||
            found == 0) {
            (void) hipblasLtMatmulPreferenceDestroy(pref);
            (void) hipblasLtMatrixLayoutDestroy(la);
            (void) hipblasLtMatrixLayoutDestroy(lb);
            (void) hipblasLtMatrixLayoutDestroy(lc);
            (void) hipblasLtMatmulDescDestroy(desc);
            return hipErrorNotSupported;
        }
        bool tuned = false;
        static const bool tune_enabled = [] {
            const char* e = std::getenv("SHELF_EXL3_DENSE_TUNE");
            return e == nullptr || std::strtoul(e, nullptr, 10) != 0ul;
        }();
        if (tune_enabled && found > 1) {
            double best_ms = 1.0e30;
            int best = -1;
            for (int i = 0; i < found; ++i) {
                hipEvent_t start = nullptr;
                hipEvent_t stop = nullptr;
                if (hipEventCreate(&start) != hipSuccess ||
                    hipEventCreate(&stop) != hipSuccess) {
                    if (start != nullptr) (void) hipEventDestroy(start);
                    if (stop != nullptr) (void) hipEventDestroy(stop);
                    break;
                }
                hipblasStatus_t st = HIPBLAS_STATUS_SUCCESS;
                float elapsed = 0.0f;
                for (int rep = 0; rep < 2; ++rep) {
                    (void) hipEventRecord(start, stream);
                    st = hipblasLtMatmul(
                        lt.handle, desc, &alpha, dense, la, args.xrot, lb, &beta, accum,
                        lc, accum, lc, &results[i].algo, nullptr,
                        results[i].workspaceSize, stream);
                    (void) hipEventRecord(stop, stream);
                    (void) hipEventSynchronize(stop);
                    float rep_ms = 0.0f;
                    (void) hipEventElapsedTime(&rep_ms, start, stop);
                    if (rep == 0 || rep_ms < elapsed) elapsed = rep_ms;
                }
                (void) hipEventDestroy(start);
                (void) hipEventDestroy(stop);
                if (st != HIPBLAS_STATUS_SUCCESS) continue;
                if (elapsed < best_ms) {
                    best_ms = elapsed;
                    best = i;
                }
            }
            if (best >= 0) {
                algo = results[best].algo;
                workspace_size = results[best].workspaceSize;
                tuned = true;
            }
        }
        if (!tuned) {
            algo = results[0].algo;
            workspace_size = results[0].workspaceSize;
        }
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
