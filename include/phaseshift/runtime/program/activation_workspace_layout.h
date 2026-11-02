#pragma once

#include <phaseshift/core/status.h>
#include <phaseshift/quantization/quant_spec.h>
#include <phaseshift/runtime/program/int8_activation_workspace.h>

#include <cstdint>

namespace ps::runtime {

inline Status make_activation_workspace_layout(
    const quantization::ActivationQuantSpec& activation,
    uint32_t k,
    uint32_t rows,
    Int8ActivationWorkspaceLayout& out) {
    const Status valid = quantization::validate_activation_quant_spec(activation);
    if (!valid.ok()) return valid;
    if ((activation.codes_dtype != quantization::QuantScalarType::FP8E4M3 &&
         activation.codes_dtype != quantization::QuantScalarType::I8) ||
        activation.group_k != Int8ActivationWorkspaceLayout::scale_group() ||
        activation.scale_dtype != quantization::QuantScalarType::F32) {
        return Status::unsupported(
            "activation workspace layout supports the production group-128 spec only",
            __FILE__, __LINE__);
    }
    if (rows == 0u)
        return Status::invalid_argument(
            "activation workspace layout needs rows > 0", __FILE__, __LINE__);
    out = Int8ActivationWorkspaceLayout::make(k, rows);
    return Status::make_ok();
}

}
