#include <shelf/inference/execution/exl3_dense_gemm.h>
#include <shelf/inference/execution/hipblaslt_abi.h>

#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace shelf::execution {
namespace {

constexpr uint32_t kCacheCapacity = 16u;

struct AlgoKey {
    uint32_t rows = 0;
    uint32_t out_features = 0;
    uint32_t k = 0;
    hipblasLtMatmulAlgo_t algo{};
    size_t workspace_size = 0u;
    bool valid = false;
};

struct LtState {
    hipblasLtHandle_t handle = nullptr;
    int device = -1;  // the device the handle was created on
    AlgoKey cache[kCacheCapacity];

    ~LtState() {
        if (handle != nullptr) {
            (void) hipblasLtDestroy(handle);
            handle = nullptr;
        }
    }
};

constexpr int kLtMaxDevices = 16;

LtState& state(int device) {
    static thread_local std::array<LtState, kLtMaxDevices> values;
    if (device < 0 || device >= kLtMaxDevices) device = 0;
    return values[static_cast<std::size_t>(device)];
}

const AlgoKey* find_algo(LtState& lt, uint32_t rows, uint32_t n, uint32_t k) {
    for (uint32_t i = 0u; i < kCacheCapacity; ++i) {
        const AlgoKey& entry = lt.cache[i];
        if (entry.valid && entry.rows == rows && entry.out_features == n && entry.k == k) {
            return &entry;
        }
    }
    return nullptr;
}

void store_algo(LtState& lt, uint32_t rows, uint32_t n, uint32_t k,
                const hipblasLtMatmulAlgo_t& algo, size_t workspace_size) {
    for (uint32_t i = 0u; i < kCacheCapacity; ++i) {
        if (!lt.cache[i].valid) {
            lt.cache[i] = AlgoKey{rows, n, k, algo, workspace_size, true};
            return;
        }
    }
}

}  // namespace

namespace {

hipError_t run_dense_gemm( uint32_t rows, uint32_t out_features, uint32_t kdim, const void* dense,
                          const void* xrot, void* accum, hipStream_t stream) {
    if (dense == nullptr || accum == nullptr || xrot == nullptr) {
        return hipErrorInvalidValue;
    }
    // A handle is bound to the device current at creation. PP stages switch the current
    // device on one thread, so each device keeps its own handle, workspace and cache.
    int cur_device = -1;
    if (hipGetDevice(&cur_device) != hipSuccess) return hipErrorInvalidDevice;
    LtState& lt = state(cur_device);
    if (lt.handle == nullptr) {
        if (!hipblaslt_abi_compatible()) return hipErrorInitializationError;
        lt.device = cur_device;
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
    if (hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32I,
                                  HIP_R_32I) !=
        HIPBLAS_STATUS_SUCCESS) {
        return hipErrorInitializationError;
    }
    hipblasOperation_t op_a = HIPBLAS_OP_T;
    hipblasOperation_t op_b = HIPBLAS_OP_N;
    (void) hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_a,
                                           sizeof(op_a));
    (void) hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_b,
                                           sizeof(op_b));
    const uint64_t m = rows;
    const uint64_t n = out_features;
    const uint64_t k = kdim;
    (void) hipblasLtMatrixLayoutCreate(&la, HIP_R_8I, k, n, k);
    (void) hipblasLtMatrixLayoutCreate(&lb, HIP_R_8I, k, m, k);
    (void) hipblasLtMatrixLayoutCreate(&lc, HIP_R_32I, n, m, n);
    (void) hipblasLtMatmulPreferenceCreate(&pref);
    const size_t workspace_limit = 256ull * 1024ull * 1024ull;
    (void) hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_limit,
        sizeof(workspace_limit));

    int prefill_slot = -1;
    int prefill_index = -1;
    if (rows == 2048u) {
        if (out_features == 5120u && kdim == 17408u) { prefill_slot = 0; prefill_index = 140674; }
        else if (out_features == 5120u && kdim == 6144u) { prefill_slot = 1; prefill_index = 140561; }
        else if (out_features == 6144u && kdim == 5120u) { prefill_slot = 2; prefill_index = 140561; }
    }
    bool use_prefill_choice = false;
    if (prefill_slot >= 0) {
        static const bool enabled = [] {
            const char* e = std::getenv("SHELF_EXL3_INT8_PREFILL_ALGO");
            return e != nullptr && std::strtoul(e, nullptr, 10) != 0ul;
        }();
        use_prefill_choice = enabled;
    }
    struct PrefillCache {
        int device = -1;
        hipblasLtMatmulAlgo_t algo{};
        size_t workspace_size = 0u;
        bool valid = false;
    };
    static thread_local PrefillCache prefill_cache[3];
    const PrefillCache* special = use_prefill_choice && prefill_cache[prefill_slot].valid &&
        prefill_cache[prefill_slot].device == cur_device ? &prefill_cache[prefill_slot] : nullptr;
    const AlgoKey* cached = use_prefill_choice ? nullptr : find_algo(lt, rows, out_features, kdim);
    hipblasLtMatmulAlgo_t algo{};
    size_t workspace_size = 0u;
    const int32_t alpha_i = 1;
    const int32_t beta_i = 0;
    const void* alpha_p = static_cast<const void*>(&alpha_i);
    const void* beta_p = static_cast<const void*>(&beta_i);
    if (special != nullptr) {
        algo = special->algo;
        workspace_size = special->workspace_size;
    } else if (cached != nullptr) {
        algo = cached->algo;
        workspace_size = cached->workspace_size;
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
        if (use_prefill_choice) {
            hipDeviceProp_t props{};
            char revision[64]{};
            if (hipGetDeviceProperties(&props, cur_device) == hipSuccess &&
                std::strncmp(props.gcnArchName, "gfx1201", 7u) == 0 &&
                hipblasLtGetGitRevision(lt.handle, revision) == HIPBLAS_STATUS_SUCCESS &&
                std::strncmp(revision, "cd957402", 8u) == 0) {
                for (int i = 0; i < found; ++i) {
                    if (results[i].state == HIPBLAS_STATUS_SUCCESS && results[i].workspaceSize == 0u &&
                        hipblaslt_ext::getIndexFromAlgo(results[i].algo) == prefill_index) {
                        algo = results[i].algo;
                        tuned = true;
                        break;
                    }
                }
            }
        }
        static const bool tune_int8 = [] {
            const char* e = std::getenv("SHELF_EXL3_INT8_TUNE");
            return e != nullptr && std::strtoul(e, nullptr, 10) != 0ul;
        }();
        if (!tuned && tune_int8 && found > 1) {
            double best_ms = 1.0e30;
            int best = -1;
            for (int i = 0; i < found; ++i) {
                if (results[i].workspaceSize != 0u) continue;
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
                        lt.handle, desc, alpha_p, dense, la, xrot, lb, beta_p, accum,
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
        if (use_prefill_choice) {
            prefill_cache[prefill_slot].device = cur_device;
            prefill_cache[prefill_slot].algo = algo;
            prefill_cache[prefill_slot].workspace_size = workspace_size;
            prefill_cache[prefill_slot].valid = true;
            if (std::getenv("SHELF_EXL3_INT8_PREFILL_ALGO_LOG") != nullptr)
                std::fprintf(stderr, "EXL3 prefill algo device=%d rows=%u n=%u k=%u index=%d workspace=%zu\n",
                             cur_device, rows, out_features, kdim, hipblaslt_ext::getIndexFromAlgo(algo), workspace_size);
        } else {
            store_algo(lt, rows, out_features, kdim, algo, workspace_size);
        }
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
        hipblasLtMatmul(lt.handle, desc, alpha_p, dense, la, xrot, lb, beta_p, accum, lc,
                        accum, lc, &algo, workspace, workspace_size, stream);
    if (workspace != nullptr) (void) hipFree(workspace);
    (void) hipblasLtMatmulPreferenceDestroy(pref);
    (void) hipblasLtMatrixLayoutDestroy(la);
    (void) hipblasLtMatrixLayoutDestroy(lb);
    (void) hipblasLtMatrixLayoutDestroy(lc);
    (void) hipblasLtMatmulDescDestroy(desc);
    if (status == HIPBLAS_STATUS_SUCCESS) return hipSuccess;
    if (status == HIPBLAS_STATUS_NOT_SUPPORTED ||
        status == HIPBLAS_STATUS_ARCH_MISMATCH)
        return hipErrorNotSupported;
    return hipErrorUnknown;
}

}  // namespace

hipError_t exl3_dense_gemm_int8(uint32_t rows, uint32_t out_features, uint32_t k, const void* dense,
                                const void* codes, void* accum, hipStream_t stream) {
    return run_dense_gemm(rows, out_features, k, dense, codes, accum, stream);
}

}  // namespace shelf::execution
