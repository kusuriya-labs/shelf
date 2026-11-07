#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {

struct TpSumResidualArgs {
    void* out = nullptr;
    const void* own = nullptr;
    const void* const* peers = nullptr;
    uint32_t peer_count = 0;
    const void* residual = nullptr;
    uint64_t rows = 0;
    uint32_t features = 0;
    uint32_t combine_stride = 0;
    uint32_t residual_stride = 0;
    uint32_t output_stride = 0;
};

struct TpSumHyperApplyArgs {
    void* out = nullptr;
    const void* own = nullptr;
    const void* const* peers = nullptr;
    uint32_t peer_count = 0;
    const void* raw = nullptr;
    const void* injection = nullptr;
    uint64_t rows = 0;
    uint32_t hidden = 0;
    uint32_t hc = 0;
    uint32_t combine_stride = 0;
    uint32_t raw_stride = 0;
    uint32_t injection_stride = 0;
    uint32_t output_stride = 0;
};

hipError_t launch_tp_sum_residual_bf16(const TpSumResidualArgs& args, hipStream_t stream);

hipError_t launch_tp_sum_hyper_apply_bf16(const TpSumHyperApplyArgs& args,
                                          hipStream_t stream);

}  // namespace shelf::kernel
