// vendored from PhaseShift (own work); namespace/type adapted only.
//
// Fused per-head L2 normalize for q and k (eps INSIDE the rsqrt:
// y = x * rsqrtf(sum(x^2) + eps)), plus a second scaled copy for q
// (sq = normalized_q * scale; shelf contract passes scale = dk^-0.5 so the
// scale applies AFTER normalization, matching the shelf CPU reference
// qinv = 1/sqrt(sum + 1e-6), ql2 = q * qinv * dkinv).
#pragma once

#include <shelf/foundation/numeric.hpp>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::hipk {

using shelf::bf16_t;

struct FusedGdnQkPrepareArgs {
    const bf16_t* q = nullptr;
    uint32_t q_row_stride = 0;
    const bf16_t* k = nullptr;
    uint32_t k_row_stride = 0;
    float* nq_out = nullptr;
    uint32_t nq_row_stride = 0;
    float* nk_out = nullptr;
    uint32_t nk_row_stride = 0;
    float* sq_out = nullptr;
    uint32_t sq_row_stride = 0;
    float eps = 0.0f;
    float scale = 1.0f;
    uint32_t rows = 0;         // == key_heads
    uint32_t features = 0;     // == rows * features covers all heads; group = head dim
    uint32_t group_size = 0;   // == head_k (dk); features must be a multiple
};

inline constexpr const char* kFusedGdnQkPrepareSymbol =
    "phaseshift_fused_gdn_qk_prepare";

inline constexpr uint32_t kFusedGdnQkPrepareThreads = 32u;

constexpr uint32_t fused_gdn_qk_prepare_blocks_x(uint32_t features, uint32_t group_size) {
    return (group_size == 0u || features % group_size != 0u)
               ? 0u
               : 2u * (features / group_size);
}

hipError_t launch_fused_gdn_qk_prepare(
    const FusedGdnQkPrepareArgs& args,
    hipStream_t stream);

}  // namespace shelf::hipk
