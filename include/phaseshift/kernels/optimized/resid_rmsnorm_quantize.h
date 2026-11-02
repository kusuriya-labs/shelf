#pragma once
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ps::kernel {

struct FusedResidRmsNormQuantizeArgs {
    const void* resid_a = nullptr;
    const void* resid_b = nullptr;
    void* residual = nullptr;
    void* normed = nullptr;
    const void* weight = nullptr;
    uint8_t* codes = nullptr;
    float* scales = nullptr;
    uint32_t rows = 0;
    uint32_t features = 0;
    uint32_t group_size = 0;
    uint32_t resid_a_row_stride = 0;
    uint32_t resid_b_row_stride = 0;
    uint32_t residual_row_stride = 0;
    uint32_t normed_row_stride = 0;
    uint32_t code_row_stride = 0;
    uint32_t scale_row_stride = 0;
    float eps = 0.0f;
};

hipError_t launch_fused_resid_rmsnorm_quantize(const FusedResidRmsNormQuantizeArgs& args,
                                               hipStream_t stream);

}
