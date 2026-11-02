#pragma once

#include <shelf/quantization/quant_spec.h>
#include <shelf/quantization/quantization_types.h>
#include <shelf/runtime/program/kernel_id.h>

#include <cstdint>

namespace ps::runtime {

struct ResolvedLinearQuantContract {
    KernelId linear_kernel = KernelId::LINEAR_BF16;
    bool requires_activation_quant = false;
    quantization::ActivationQuantSpec activation{};
    uint32_t activation_k_alignment = 32u;
};

inline Result<ResolvedLinearQuantContract> resolve_linear_quant_contract(
    uint8_t weight_encoding, uint8_t compute_spec) {
    using quantization::ComputeSpecId;
    using ps::quantization::manifest::QuantizedEncoding;

    ResolvedLinearQuantContract contract;
    switch (static_cast<QuantizedEncoding>(weight_encoding)) {
        case QuantizedEncoding::Fp8:
            if (compute_spec != static_cast<uint8_t>(ComputeSpecId::FP8_W8A8_F32_BF16))
                return Status::invalid_argument(
                    "fp8 linear weight requires fp8 compute spec", __FILE__, __LINE__);
            contract.linear_kernel = KernelId::LINEAR_FP8;
            contract.requires_activation_quant = true;
            contract.activation = quantization::kE4m3ActivationQuantSpec;
            contract.activation_k_alignment = 128u;
            return contract;
        case QuantizedEncoding::Psq5:
            if (compute_spec != static_cast<uint8_t>(ComputeSpecId::PSQ5_W8A8_F32_BF16))
                return Status::invalid_argument(
                    "psq5 linear weight requires psq5 compute spec", __FILE__, __LINE__);
            contract.linear_kernel = KernelId::LINEAR_PSQ5;
            contract.requires_activation_quant = true;
            contract.activation = quantization::kE4m3ActivationQuantSpec;
            contract.activation_k_alignment = 128u;
            return contract;
        case QuantizedEncoding::Bf16:
            break;
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

inline KernelId activation_quantize_kernel(
    const quantization::ActivationQuantSpec& activation) noexcept {
    return activation.codes_dtype == quantization::QuantScalarType::I8
               ? KernelId::ACTIVATION_QUANTIZE_FP8
               : KernelId::ACTIVATION_QUANTIZE_W4A8;
}

inline bool activation_quant_matches(
    const quantization::ActivationQuantSpec& left,
    const quantization::ActivationQuantSpec& right) noexcept {
    return left.codes_dtype == right.codes_dtype && left.group_k == right.group_k &&
           left.scale_dtype == right.scale_dtype;
}

}
