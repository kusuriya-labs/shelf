#pragma once
#include <shelf/core/memory/tensor.h>
#include <shelf/quantization/quantization_types.h>
#include <shelf/weights/tensor_partition.h>
#include <cstdint>
#include <optional>

namespace shelf::weights {

enum class MatrixEncoding : uint8_t {
    Bf16 = 0,
    Fp8Finegrained = 1,
    Mxfp4 = 3,
    Exl3 = 4,
};

struct MatrixWeight {
    quantization::QuantSpecId quant_spec = quantization::QuantSpecId::BF16_V1;
    quantization::ComputeSpecId compute_spec = quantization::ComputeSpecId::BF16_F32_BF16;
    MatrixEncoding encoding = MatrixEncoding::Bf16;
    gpu::Tensor bf16;
    gpu::Tensor data;
    gpu::Tensor codes;
    gpu::Tensor scales;
    // MXFP4 native only: one E8M0 reference byte per output row (the row's maximum
    // scale exponent). 0 when the payload is canonical or the plane is absent.
    gpu::Tensor row_ref;
    bool row_ref_exact = false;
    gpu::Tensor compute_codes;
    gpu::Tensor compute_scales_bf16;
    gpu::Tensor exl3_suh;
    gpu::Tensor exl3_svh;
    gpu::Tensor exl3_marker;
    uint32_t exl3_ka = 0;
    bool exl3_half = false;
    // Rank-3 expert groups. experts == 1 is the 2D path. When experts > 1, rows /
    // cols / k_padded and the row strides describe ONE expert, and the batch strides
    // separate experts inside the single arena allocation. row_ref is then a batched
    // [experts][rows] byte plane with row stride 1.
    uint32_t experts = 1;
    uint64_t codes_batch_stride_bytes = 0;
    uint64_t scales_batch_stride_bytes = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t k_padded = 0;
    uint32_t storage_scale_stride_bytes = 0;
    uint32_t codes_row_stride_bytes = 0;
    uint32_t scale_row_stride_bytes = 0;
    uint32_t weight_scale_group = 0;
    bool preshuffled = false;
    std::optional<TensorPartitionDesc> partition;
};

inline quantization::QuantSpecId encoding_to_quant_spec(MatrixEncoding enc) noexcept {
    switch (enc) {
        case MatrixEncoding::Bf16: return quantization::QuantSpecId::BF16_V1;
        case MatrixEncoding::Fp8Finegrained: return quantization::QuantSpecId::FP8_FINEGRAINED_F32_V1;
        case MatrixEncoding::Mxfp4: return quantization::QuantSpecId::MXFP4_E8M0_G32_V1;
        case MatrixEncoding::Exl3: return quantization::QuantSpecId::EXL3_W16A16_V1;
    }
    return quantization::QuantSpecId::BF16_V1;
}

inline quantization::ComputeSpecId encoding_to_compute_spec(MatrixEncoding enc) noexcept {
    switch (enc) {
        case MatrixEncoding::Bf16: return quantization::ComputeSpecId::BF16_F32_BF16;
        case MatrixEncoding::Fp8Finegrained: return quantization::ComputeSpecId::FP8_FINEGRAINED_W8A8_F32_BF16;
        case MatrixEncoding::Mxfp4: return quantization::ComputeSpecId::MXFP4_W4A8_F32_BF16;
        case MatrixEncoding::Exl3: return quantization::ComputeSpecId::EXL3_W16A16_F32_BF16;
    }
    return quantization::ComputeSpecId::BF16_F32_BF16;
}

inline void apply_default_compute_spec(MatrixWeight& w) noexcept {
    w.quant_spec = encoding_to_quant_spec(w.encoding);
    w.compute_spec = encoding_to_compute_spec(w.encoding);
}

inline MatrixEncoding quant_spec_to_encoding(quantization::QuantSpecId id) noexcept {
    switch (id) {
        case quantization::QuantSpecId::BF16_V1: return MatrixEncoding::Bf16;
        case quantization::QuantSpecId::FP8_FINEGRAINED_F32_V1: return MatrixEncoding::Fp8Finegrained;
        case quantization::QuantSpecId::MXFP4_E8M0_G32_V1: return MatrixEncoding::Mxfp4;
        case quantization::QuantSpecId::EXL3_W16A16_V1: return MatrixEncoding::Exl3;
        case quantization::QuantSpecId::EXL3_MUL24_INT8_V1: return MatrixEncoding::Exl3;
        default: break;
    }
    return MatrixEncoding::Bf16;
}

}
