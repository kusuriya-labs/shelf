#pragma once

#include <shelf/core/memory/types.h>
#include <shelf/runtime/batch/device_batch_context.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::execution {

struct Fp8ResidualNormArgs {
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

struct Fp8NormArgs {
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

struct Fp8FinalProducer {
    const ::shelf::bf16_t* input = nullptr;
    const ::shelf::bf16_t* weight = nullptr;
    ::shelf::bf16_t* gathered = nullptr;
    uint8_t* codes = nullptr;
    float* scales = nullptr;
    const ::shelf::runtime::DeviceBatchContext* batch = nullptr;
    uint32_t rows = 0;
    uint32_t k = 0;
    uint32_t k_padded = 0;
    uint32_t input_stride = 0;
    uint32_t gathered_stride = 0;
    uint32_t code_stride = 0;
    uint32_t scale_stride = 0;
    float eps = 0.0f;
};

hipError_t launch_fp8_residual_norm_quantize(const Fp8ResidualNormArgs& args,
                                             hipStream_t stream);
hipError_t launch_fp8_norm_quantize(const Fp8NormArgs& args, hipStream_t stream);
hipError_t launch_fp8_final_gather_norm_quantize(const Fp8FinalProducer& args,
                                                 hipStream_t stream);

}
