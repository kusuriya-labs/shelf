#pragma once

#include <shelf/weights/matrix_weight.h>
#include <shelf/quantization/quant_spec.h>
#include <shelf/quantization/quantization_types.h>
#include <shelf/runtime/program/kernel_id.h>

#include <cstdint>

namespace shelf::runtime {

struct ResolvedLinearQuantContract {
    KernelId linear_kernel = KernelId::LINEAR_BF16;
    bool requires_activation_quant = false;
    quantization::ActivationQuantSpec activation{};
    uint32_t activation_k_alignment = 32u;
};

inline Result<ResolvedLinearQuantContract> resolve_linear_quant_contract(
    uint8_t weight_encoding, uint8_t compute_spec) {
    using shelf::weights::MatrixEncoding;
    using quantization::ComputeSpecId;

    ResolvedLinearQuantContract contract;
    if (static_cast<MatrixEncoding>(weight_encoding) == MatrixEncoding::Fp8Finegrained) {
        if (compute_spec != static_cast<uint8_t>(ComputeSpecId::FP8_FINEGRAINED_W8A8_F32_BF16))
            return Status::invalid_argument(
                "fp8 linear weight requires fp8 compute spec", __FILE__, __LINE__);
        contract.linear_kernel = KernelId::LINEAR_FP8_FINEGRAINED;
        contract.requires_activation_quant = true;
        contract.activation = quantization::kFp8FinegrainedActivationQuantSpec;
        contract.activation_k_alignment = 128u;
        return contract;
    }
    if (static_cast<MatrixEncoding>(weight_encoding) == MatrixEncoding::Mxfp4) {
        if (compute_spec != static_cast<uint8_t>(ComputeSpecId::MXFP4_W4A8_F32_BF16))
            return Status::invalid_argument(
                "mxfp4 linear weight requires mxfp4 w4a8 compute spec", __FILE__, __LINE__);
        contract.linear_kernel = KernelId::LINEAR_MXFP4_W4A8;
        contract.requires_activation_quant = true;
        contract.activation = quantization::kFp8FinegrainedActivationQuantSpec;
        contract.activation_k_alignment = 64u;
        return contract;
    }
    if (compute_spec != static_cast<uint8_t>(ComputeSpecId::BF16_F32_BF16))
        return Status::invalid_argument(
            "bf16 linear weight requires bf16 compute spec", __FILE__, __LINE__);
    contract.linear_kernel = KernelId::LINEAR_BF16;
    contract.requires_activation_quant = false;
    return contract;
}

inline uint32_t activation_k_padded_for(uint32_t alignment, uint32_t k) noexcept {
    return (k + alignment - 1u) & ~(alignment - 1u);
}

inline Result<KernelId> activation_quantize_kernel(
    const quantization::ActivationQuantSpec& activation) {
    if (activation.codes_dtype != quantization::QuantScalarType::FP8E4M3)
        return Status::unsupported(
            "activation quantize requires FP8E4M3 codes", __FILE__, __LINE__);
    return KernelId::ACTIVATION_QUANTIZE_FP8_FINEGRAINED;
}

inline bool activation_quant_matches(
    const quantization::ActivationQuantSpec& left,
    const quantization::ActivationQuantSpec& right) noexcept {
    return left.codes_dtype == right.codes_dtype && left.group_k == right.group_k &&
           left.scale_dtype == right.scale_dtype;
}

}
