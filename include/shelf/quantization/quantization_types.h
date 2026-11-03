#pragma once

#include <shelf/core/status.h>

#include <cstdint>

namespace shelf::quantization {

enum class StorageDType : uint8_t {
    NONE,
    BF16,
    U8,
    I8,
    FP16,
    FP32,
};

enum class QuantLayout : uint8_t {
    NONE,
    BF16_V1,
    FP8_E4M3_BLOCK128_V1,
    FP8_SCALE_F32_N128K128_V1,
    MXFP4_E2M1_BLOCK32_V1,
    MXFP4_SCALE_E8M0_PER_ROW_BLOCK32_V1,
};

struct QuantDescriptor {
    uint32_t group_size = 0;
    uint32_t block_size = 0;
    StorageDType storage_type = StorageDType::BF16;
    StorageDType scale_type = StorageDType::NONE;
    QuantLayout weight_layout = QuantLayout::BF16_V1;
    QuantLayout scale_layout = QuantLayout::NONE;
};

enum class QuantSpecId : uint8_t {
    BF16_V1,
    FP8_FINEGRAINED_F32_V1,
    MXFP4_E8M0_G32_V1,
};

enum class ComputeSpecId : uint8_t {
    BF16_F32_BF16,
    FP8_FINEGRAINED_W8A8_F32_BF16,
    MXFP4_W4A8_F32_BF16,
};

struct ResolvedQuantSpec {
    QuantDescriptor descriptor;
    QuantSpecId spec_id = QuantSpecId::BF16_V1;
};

inline Result<ComputeSpecId> preferred_gemm_compute_spec(QuantSpecId quant_spec) {
    switch (quant_spec) {
        case QuantSpecId::BF16_V1:
            return ComputeSpecId::BF16_F32_BF16;
        case QuantSpecId::FP8_FINEGRAINED_F32_V1:
            return ComputeSpecId::FP8_FINEGRAINED_W8A8_F32_BF16;
        case QuantSpecId::MXFP4_E8M0_G32_V1:
            return ComputeSpecId::MXFP4_W4A8_F32_BF16;
    }
    return Status::invalid_argument("unknown quant spec id", __FILE__, __LINE__);
}

inline bool is_gemm_compute_compatible(
    QuantSpecId quant_spec,
    ComputeSpecId compute_spec) noexcept {
    switch (quant_spec) {
        case QuantSpecId::BF16_V1:
            return compute_spec == ComputeSpecId::BF16_F32_BF16;
        case QuantSpecId::FP8_FINEGRAINED_F32_V1:
            return compute_spec == ComputeSpecId::FP8_FINEGRAINED_W8A8_F32_BF16;
        case QuantSpecId::MXFP4_E8M0_G32_V1:
            return compute_spec == ComputeSpecId::MXFP4_W4A8_F32_BF16;
    }
    return false;
}

} // namespace shelf::quantization
