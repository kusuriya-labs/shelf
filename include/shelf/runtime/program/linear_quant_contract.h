#pragma once

#include <shelf/weights/matrix_weight.h>
#include <shelf/quantization/quant_spec.h>
#include <shelf/quantization/quantization_types.h>
#include <shelf/runtime/program/kernel_id.h>

#include <cstdint>

namespace shelf::runtime {

// DispatchBinding::flags bit 1 (bit 0 is the linear output dtype flag set by the
// lowering): the activation plane for this linear was produced in the row-reference
// form - E4M3 codes with the group E8M0 folded into them, and one f32 row reference
// at scales[row * scale_stride_floats + 0]. Producers and consumers must agree; the
// accumulation order of each path is its own.
inline constexpr uint32_t kActFlagRowRefFolded = 1u << 1;

struct ResolvedLinearQuantContract {
    KernelId linear_kernel = KernelId::LINEAR_BF16;
    bool requires_activation_quant = false;
    quantization::ActivationQuantSpec activation{};
    uint32_t activation_k_alignment = 32u;
    // Producer kernel for the activation plane. The FP8 W8A8 and MXFP4 W4A8 families
    // share the codes/scale plane layout but not its meaning: MXFP4 folds the group
    // E8M0 into the codes against a row reference and applies no activation scale in
    // the K loop.
    KernelId activation_kernel = KernelId::ACTIVATION_QUANTIZE_FP8_FINEGRAINED;
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
        contract.activation_kernel = KernelId::ACTIVATION_QUANTIZE_MXFP4_ROWREF;
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

inline bool activation_quant_matches(
    const quantization::ActivationQuantSpec& left,
    const quantization::ActivationQuantSpec& right) noexcept {
    return left.codes_dtype == right.codes_dtype && left.group_k == right.group_k &&
           left.scale_dtype == right.scale_dtype;
}

}
