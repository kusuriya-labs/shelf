#pragma once

#include <shelf/core/memory/types.h>
#include <shelf/runtime/batch/device_batch_context.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::execution {

struct Mxfp4NormArgs {
    const ::shelf::bf16_t* input = nullptr;
    const ::shelf::bf16_t* weight = nullptr;
    uint8_t* codes = nullptr;
    float* scales = nullptr;
    uint32_t rows = 0;
    uint32_t k = 0;
    uint32_t k_padded = 0;
    uint32_t input_stride = 0;
    uint32_t code_stride = 0;
    uint32_t scale_stride = 0;
    float eps = 0.0f;
};

struct Mxfp4ResidualNormArgs {
    const ::shelf::bf16_t* a = nullptr;
    const ::shelf::bf16_t* b = nullptr;
    ::shelf::bf16_t* residual = nullptr;
    const ::shelf::bf16_t* weight = nullptr;
    uint8_t* codes = nullptr;
    float* scales = nullptr;
    uint32_t rows = 0;
    uint32_t k = 0;
    uint32_t k_padded = 0;
    uint32_t a_stride = 0;
    uint32_t b_stride = 0;
    uint32_t residual_stride = 0;
    uint32_t code_stride = 0;
    uint32_t scale_stride = 0;
    float eps = 0.0f;
};

struct Mxfp4SwigluArgs {
    const ::shelf::bf16_t* gate = nullptr;
    const ::shelf::bf16_t* up = nullptr;
    uint8_t* codes = nullptr;
    float* scales = nullptr;
    uint32_t rows = 0;
    uint32_t k = 0;
    uint32_t k_padded = 0;
    uint32_t gate_stride = 0;
    uint32_t up_stride = 0;
    uint32_t code_stride = 0;
    uint32_t scale_stride = 0;
    bool bf16_intermediate = false;
};

struct Mxfp4GdnGatedArgs {
    const float* rec = nullptr;
    const ::shelf::bf16_t* z = nullptr;
    const ::shelf::bf16_t* weight = nullptr;
    uint8_t* codes = nullptr;
    float* scales = nullptr;
    uint32_t rows = 0;
    uint32_t features = 0;
    uint32_t group_size = 0;
    uint32_t rec_stride = 0;
    uint32_t z_stride = 0;
    uint32_t code_stride = 0;
    uint32_t scale_stride = 0;
    float eps = 0.0f;
};

hipError_t launch_mxfp4_norm_quantize(const Mxfp4NormArgs& args, hipStream_t stream);
hipError_t launch_mxfp4_residual_norm_quantize(const Mxfp4ResidualNormArgs& args,
                                               hipStream_t stream);
hipError_t launch_mxfp4_swiglu_quantize(const Mxfp4SwigluArgs& args, hipStream_t stream);
hipError_t launch_mxfp4_gdn_gated_quantize(const Mxfp4GdnGatedArgs& args,
                                           hipStream_t stream);

}  // namespace shelf::execution
