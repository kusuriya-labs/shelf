#include <shelf/inference/execution/gemm_lt.h>

#include <hipblaslt/hipblaslt.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace shelf::execution {
namespace {

enum class LtKind : uint32_t {
    Bf16 = 0,
    Bf16F32 = 1,
};

struct AlgoKey {
    LtKind kind = LtKind::Bf16;
    uint32_t rows = 0;
    uint32_t out_features = 0;
    uint32_t k = 0;
    uint32_t activation_row_stride = 0;
    uint32_t weight_row_stride = 0;
    uint32_t output_row_stride = 0;
    hipblasLtMatmulAlgo_t algo{};
    size_t workspace_size = 0;
    bool valid = false;
};

struct LtState {
    hipblasLtHandle_t handle = nullptr;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    std::vector<AlgoKey> cache;

    ~LtState() {
        if (workspace != nullptr) (void)hipFree(workspace);
        if (handle != nullptr) (void)hipblasLtDestroy(handle);
    }
};

LtState& state() {
    static thread_local LtState value;
    return value;
}

const AlgoKey* find_algo(const LtState& lt, const AlgoKey& key) {
    for (const AlgoKey& entry : lt.cache) {
        if (entry.valid && entry.kind == key.kind && entry.rows == key.rows &&
            entry.out_features == key.out_features && entry.k == key.k &&
            entry.activation_row_stride == key.activation_row_stride &&
            entry.weight_row_stride == key.weight_row_stride &&
            entry.output_row_stride == key.output_row_stride) {
            return &entry;
        }
    }
    return nullptr;
}

void store_algo(LtState& lt, const AlgoKey& key) {
    if (lt.cache.size() >= 64u) lt.cache.erase(lt.cache.begin());
    lt.cache.push_back(key);
}

hipError_t ensure_workspace(LtState& lt, size_t bytes) {
    if (bytes <= lt.workspace_bytes) return hipSuccess;
    if (lt.workspace != nullptr) (void)hipFree(lt.workspace);
    lt.workspace = nullptr;
    lt.workspace_bytes = 0;
    if (hipMalloc(&lt.workspace, bytes) != hipSuccess) {
        lt.workspace = nullptr;
        return hipErrorOutOfMemory;
    }
    lt.workspace_bytes = bytes;
    return hipSuccess;
}

hipError_t ensure_handle(LtState& lt) {
    if (lt.handle != nullptr) return hipSuccess;
    if (hipblasLtCreate(&lt.handle) != HIPBLAS_STATUS_SUCCESS) {
        lt.handle = nullptr;
        return hipErrorInitializationError;
    }
    return hipSuccess;
}

struct Descriptors {
    hipblasLtMatmulDesc_t desc = nullptr;
    hipblasLtMatrixLayout_t la = nullptr;
    hipblasLtMatrixLayout_t lb = nullptr;
    hipblasLtMatrixLayout_t lc = nullptr;
    hipblasLtMatmulPreference_t pref = nullptr;
    size_t workspace_limit = 0;

    ~Descriptors() {
        if (pref != nullptr) (void)hipblasLtMatmulPreferenceDestroy(pref);
        if (la != nullptr) (void)hipblasLtMatrixLayoutDestroy(la);
        if (lb != nullptr) (void)hipblasLtMatrixLayoutDestroy(lb);
        if (lc != nullptr) (void)hipblasLtMatrixLayoutDestroy(lc);
        if (desc != nullptr) (void)hipblasLtMatmulDescDestroy(desc);
    }
};

// Builds the column-major descriptors for output = weight * activation^T. A holds the
// weight so the transposed declaration lands the result in a row-major [rows,
// out_features] plane.
bool build_descriptors(Descriptors& d,
                       LtKind kind,
                       uint32_t rows,
                       uint32_t out_features,
                       uint32_t k,
                       uint32_t activation_row_stride,
                       uint32_t weight_row_stride,
                       uint32_t output_row_stride) {
    d.workspace_limit = 256ull * 1024ull * 1024ull;
    if (hipblasLtMatmulDescCreate(&d.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F) !=
        HIPBLAS_STATUS_SUCCESS) {
        return false;
    }
    hipblasOperation_t trans_a = HIPBLAS_OP_T;
    hipblasOperation_t trans_b = HIPBLAS_OP_N;
    if (hipblasLtMatmulDescSetAttribute(d.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a,
                                        sizeof(trans_a)) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatmulDescSetAttribute(d.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b,
                                        sizeof(trans_b)) != HIPBLAS_STATUS_SUCCESS) {
        return false;
    }
    const hipDataType output_type = kind == LtKind::Bf16F32 ? HIP_R_32F : HIP_R_16BF;
    const hipDataType input_type = HIP_R_16BF;
    if (hipblasLtMatrixLayoutCreate(&d.la, input_type, k, out_features,
                                    weight_row_stride) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutCreate(&d.lb, input_type, k, rows,
                                    activation_row_stride) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutCreate(&d.lc, output_type, out_features, rows,
                                    output_row_stride) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatmulPreferenceCreate(&d.pref) != HIPBLAS_STATUS_SUCCESS) {
        return false;
    }
    return hipblasLtMatmulPreferenceSetAttribute(
               d.pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &d.workspace_limit,
               sizeof(d.workspace_limit)) == HIPBLAS_STATUS_SUCCESS;
}

hipError_t run(const AlgoKey& key,
               Descriptors& d,
               const void* a,
               const void* b,
               void* c,
               hipStream_t stream) {
    LtState& lt = state();
    const AlgoKey* cached = find_algo(lt, key);
    hipblasLtMatmulAlgo_t algo{};
    size_t workspace_size = 0u;
    bool have_algo = false;
    if (cached != nullptr) {
        algo = cached->algo;
        workspace_size = cached->workspace_size;
        have_algo = true;
    } else {
        constexpr int kCandidates = 16;
        hipblasLtMatmulHeuristicResult_t results[kCandidates]{};
        int found = 0;
        if (hipblasLtMatmulAlgoGetHeuristic(lt.handle, d.desc, d.la, d.lb, d.lc, d.lc,
                                            d.pref, kCandidates, results,
                                            &found) != HIPBLAS_STATUS_SUCCESS ||
            found <= 0) {
            return hipErrorNotSupported;
        }
        AlgoKey stored = key;
        stored.algo = results[0].algo;
        stored.workspace_size = results[0].workspaceSize;
        stored.valid = true;
        store_algo(lt, stored);
        algo = stored.algo;
        workspace_size = stored.workspace_size;
        have_algo = true;
    }
    if (!have_algo) return hipErrorNotSupported;
    const hipError_t ws = ensure_workspace(lt, workspace_size);
    if (ws != hipSuccess) return ws;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const hipblasStatus_t status =
        hipblasLtMatmul(lt.handle, d.desc, &alpha, a, d.la, b, d.lb, &beta, c, d.lc, c,
                        d.lc, &algo, lt.workspace, workspace_size, stream);
    if (status != HIPBLAS_STATUS_SUCCESS) return hipErrorNotSupported;
    return hipSuccess;
}

}  // namespace

hipError_t gemm_bf16_lt(const shelf::bf16_t* weight,
                        const shelf::bf16_t* input,
                        void* output,
                        uint32_t rows,
                        uint32_t out_features,
                        uint32_t k,
                        uint32_t input_row_stride,
                        uint32_t output_row_stride,
                        bool output_bf16,
                        hipStream_t stream) {
    if (weight == nullptr || input == nullptr || output == nullptr || rows == 0u ||
        out_features == 0u || k == 0u || input_row_stride < k ||
        output_row_stride < out_features) {
        return hipErrorInvalidValue;
    }
    LtState& lt = state();
    const hipError_t handle = ensure_handle(lt);
    if (handle != hipSuccess) return handle;
    const LtKind kind = output_bf16 ? LtKind::Bf16 : LtKind::Bf16F32;
    Descriptors d;
    if (!build_descriptors(d, kind, rows, out_features, k, input_row_stride, k,
                           output_row_stride)) {
        return hipErrorNotSupported;
    }
    AlgoKey key;
    key.kind = kind;
    key.rows = rows;
    key.out_features = out_features;
    key.k = k;
    key.activation_row_stride = input_row_stride;
    key.weight_row_stride = k;
    key.output_row_stride = output_row_stride;
    return run(key, d, weight, input, output, stream);
}

}  // namespace shelf::execution
