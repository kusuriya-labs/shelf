#pragma once

#include <phaseshift/quantization/manifest/quantized_manifest.h>
#include <phaseshift/quantization/payload_descriptor.h>
#include <phaseshift/quantization/quant_format.h>
#include <phaseshift/quantization/quant_spec.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace ps::quantization {

enum class QuantKernelKind : uint8_t {
    LinearBf16,
    LinearFp8,
    LinearPsq5,
};

inline manifest::QuantizedEncoding kernel_weight_encoding(QuantKernelKind kernel) {
    switch (kernel) {
        case QuantKernelKind::LinearBf16: return manifest::QuantizedEncoding::Bf16;
        case QuantKernelKind::LinearFp8: return manifest::QuantizedEncoding::Fp8;
        case QuantKernelKind::LinearPsq5: return manifest::QuantizedEncoding::Psq5;
    }
    return manifest::QuantizedEncoding::Bf16;
}

inline const char* canonical_layout_name(manifest::QuantizedEncoding enc) {
    switch (enc) {
        case manifest::QuantizedEncoding::Bf16: return "";
        case manifest::QuantizedEncoding::Fp8: return manifest::kFp8LayoutName;
        case manifest::QuantizedEncoding::Psq5: return manifest::kPsq5LayoutName;
    }
    return "";
}

inline uint32_t count_payload(const QuantizedTensorDescriptor& desc, QuantPayloadRole role) {
    uint32_t count = 0;
    for (const QuantPayloadRef& ref : desc.payloads) {
        if (ref.role == role) ++count;
    }
    return count;
}

inline bool payload_shape_equals(
    const std::vector<uint64_t>& actual, const std::vector<int64_t>& expected) {
    if (actual.size() != expected.size()) return false;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (expected[i] <= 0 || actual[i] != static_cast<uint64_t>(expected[i])) return false;
    }
    return true;
}

inline bool payload_shape_equals(
    const std::vector<uint64_t>& actual, const std::vector<uint64_t>& expected) {
    return actual.size() == expected.size() && actual == expected;
}

inline Status validate_weight_scale_spec(const WeightQuantSpec& spec) {
    const std::optional<WeightScaleSpec> fixed = default_weight_scale_spec(spec.encoding);
    if (!fixed.has_value()) {
        if (spec.scale.has_value()) {
            return Status::invalid_argument(
                "bf16 weight quant spec must not carry a scale", __FILE__, __LINE__);
        }
        return Status::make_ok();
    }
    if (!spec.scale.has_value()) {
        return Status::invalid_argument(
            "weight quant spec is missing its scale", __FILE__, __LINE__);
    }
    const WeightScaleSpec& scale = *spec.scale;
    if (scale.granularity != ScaleGranularity::MatrixBlock) {
        return Status::invalid_argument(
            "weight scale granularity must be matrix_block", __FILE__, __LINE__);
    }
    const uint32_t n = scale.block.n;
    const uint32_t k = scale.block.k;
    if (n == 0 || k == 0 || (n & (n - 1u)) != 0u || (k & (k - 1u)) != 0u ||
        k < kQuantBlockElements) {
        return Status::invalid_argument(
            "weight scale block shape must be powers of two with k >= 32", __FILE__, __LINE__);
    }
    if (scale.dtype != fixed->dtype) {
        return Status::invalid_argument(
            "weight scale dtype does not match the encoding", __FILE__, __LINE__);
    }
    if (scale.application != ScaleApplication::Multiply) {
        return Status::invalid_argument(
            "weight scale application must be multiply", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

inline Status validate_payload_contract(const QuantizedTensorDescriptor& desc) {
    if (desc.logical_shape.empty()) {
        return Status::invalid_argument(
            "quantized tensor logical_shape is empty", __FILE__, __LINE__);
    }
    for (const int64_t dim : desc.logical_shape) {
        if (dim <= 0) {
            return Status::invalid_argument(
                "quantized tensor logical_shape has non-positive dim", __FILE__, __LINE__);
        }
    }
    const uint64_t logical_k = static_cast<uint64_t>(desc.logical_shape.back());
    const uint64_t k_padded = desc.layout.k_padded;

    if (desc.quant.encoding == manifest::QuantizedEncoding::Bf16) {
        if (count_payload(desc, QuantPayloadRole::Data) != 1) {
            return Status::invalid_argument(
                "bf16 payload requires exactly one data stream", __FILE__, __LINE__);
        }
        if (count_payload(desc, QuantPayloadRole::Codes) != 0 ||
            count_payload(desc, QuantPayloadRole::Scale) != 0) {
            return Status::invalid_argument(
                "bf16 payload must not carry codes or scale", __FILE__, __LINE__);
        }
        if (k_padded != logical_k) {
            return Status::invalid_argument(
                "bf16 k_padded does not match logical k", __FILE__, __LINE__);
        }
        const QuantPayloadRef* data = find_payload(desc, QuantPayloadRole::Data);
        if (data->dtype != QuantScalarType::BF16) {
            return Status::invalid_argument(
                "bf16 data payload dtype is not BF16", __FILE__, __LINE__);
        }
        if (!payload_shape_equals(data->shape, desc.logical_shape)) {
            return Status::invalid_argument(
                "bf16 data payload shape mismatch", __FILE__, __LINE__);
        }
        return Status::make_ok();
    }

    const QuantFormatDesc* format = quant_format_desc(manifest::quant_format_id(desc.quant.encoding));
    if (format == nullptr) {
        return Status::invalid_argument("unknown quantized tensor encoding", __FILE__, __LINE__);
    }
    if (count_payload(desc, QuantPayloadRole::Codes) != 1) {
        return Status::invalid_argument(
            "quantized payload requires exactly one codes stream", __FILE__, __LINE__);
    }
    if (count_payload(desc, QuantPayloadRole::Scale) != 1) {
        return Status::invalid_argument(
            "quantized payload requires exactly one scale stream", __FILE__, __LINE__);
    }
    if (count_payload(desc, QuantPayloadRole::Data) != 0) {
        return Status::invalid_argument(
            "quantized payload must not carry a data stream", __FILE__, __LINE__);
    }

    const QuantPayloadRef* codes = find_payload(desc, QuantPayloadRole::Codes);
    if (codes->dtype != QuantScalarType::U8) {
        return Status::invalid_argument("codes payload dtype is not U8", __FILE__, __LINE__);
    }
    std::vector<uint64_t> expected_codes_shape(
        desc.logical_shape.begin(), desc.logical_shape.end());
    expected_codes_shape.back() = quant_codes_bytes(1, k_padded, *format);
    if (!payload_shape_equals(codes->shape, expected_codes_shape)) {
        return Status::invalid_argument("codes payload shape mismatch", __FILE__, __LINE__);
    }

    const WeightScaleSpec& scale_spec = *desc.quant.scale;
    const QuantPayloadRef* scale = find_payload(desc, QuantPayloadRole::Scale);
    if (scale->dtype != scale_spec.dtype) {
        return Status::invalid_argument(
            "scale payload dtype does not match the quant spec", __FILE__, __LINE__);
    }
    if (desc.logical_shape.size() < 2) {
        return Status::invalid_argument(
            "block scale requires a matrix shape", __FILE__, __LINE__);
    }
    const uint64_t n = static_cast<uint64_t>(desc.logical_shape[desc.logical_shape.size() - 2]);
    if ((k_padded % scale_spec.block.k) != 0) {
        return Status::invalid_argument(
            "quantized k_padded is not a multiple of the scale block", __FILE__, __LINE__);
    }
    const ScaleGrid grid = derive_scale_grid(n, k_padded, scale_spec.block);
    std::vector<uint64_t> expected_scale_shape(
        desc.logical_shape.begin(), desc.logical_shape.end() - 2);
    expected_scale_shape.push_back(grid.n);
    expected_scale_shape.push_back(grid.k);
    if (!payload_shape_equals(scale->shape, expected_scale_shape)) {
        return Status::invalid_argument("scale payload shape mismatch", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

inline Status validate_quantized_tensor(const QuantizedTensorDescriptor& desc) {
    if (desc.domain != QuantDomain::Weight) {
        return Status::unsupported(
            "only weight domain descriptors are supported", __FILE__, __LINE__);
    }
    if (desc.layout.storage_order != TensorStorageOrder::RowMajor) {
        return Status::unsupported(
            "only row-major weight layouts are supported", __FILE__, __LINE__);
    }
    Status spec_status = validate_weight_scale_spec(desc.quant);
    if (!spec_status.ok()) return spec_status;
    if (desc.layout.layout_id != canonical_layout_name(desc.quant.encoding)) {
        return Status::invalid_argument(
            "quantized layout_id does not match the encoding", __FILE__, __LINE__);
    }
    return validate_payload_contract(desc);
}

inline bool weight_block_supported(
    manifest::QuantizedEncoding enc, WeightBlockShape block) {
    const WeightBlockShape fixed = default_weight_block(enc);
    if (block.n != fixed.n) return false;
    switch (enc) {
        case manifest::QuantizedEncoding::Fp8:
            return block.k == kFp8ScaleBlockK || block.k == kFp8ScaleBlockK / 2u;
        case manifest::QuantizedEncoding::Psq5:
            return block.k == 32u;
        case manifest::QuantizedEncoding::Bf16:
            break;
    }
    return false;
}

inline bool weight_scale_group_supported(uint32_t weight_scale_group, QuantKernelKind kernel) {
    const manifest::QuantizedEncoding enc = kernel_weight_encoding(kernel);
    const std::optional<WeightScaleSpec> expected = default_weight_scale_spec(enc);
    if (!expected.has_value()) return weight_scale_group == 0u;
    if (enc == manifest::QuantizedEncoding::Fp8) {
        return weight_scale_group == kFp8ScaleBlockK ||
               weight_scale_group == kFp8ScaleBlockK / 2u;
    }
    return weight_scale_group == expected->block.k;
}

inline bool weight_scale_group_is_default(
    uint32_t weight_scale_group, QuantKernelKind kernel) noexcept {
    const manifest::QuantizedEncoding enc = kernel_weight_encoding(kernel);
    const std::optional<WeightScaleSpec> expected = default_weight_scale_spec(enc);
    const uint32_t expected_group = expected.has_value() ? expected->block.k : 0u;
    return weight_scale_group == expected_group;
}

inline Status validate_kernel_capability(
    const QuantizedTensorDescriptor& desc, QuantKernelKind kernel) {
    const manifest::QuantizedEncoding enc = kernel_weight_encoding(kernel);
    if (desc.quant.encoding != enc) {
        return Status::unsupported(
            "weight quant spec is not supported by the kernel", __FILE__, __LINE__);
    }
    const std::optional<WeightScaleSpec> expected = default_weight_scale_spec(enc);
    if (!expected.has_value()) return Status::make_ok();
    if (!desc.quant.scale.has_value()) {
        return Status::unsupported(
            "weight quant spec is missing its scale", __FILE__, __LINE__);
    }
    if (!weight_block_supported(enc, desc.quant.scale->block)) {
        return Status::unsupported(
            "kernel does not support the scale block shape", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

inline Status validate_weight_scale_group(uint32_t weight_scale_group, QuantKernelKind kernel) {
    if (!weight_scale_group_supported(weight_scale_group, kernel)) {
        return Status::invalid_state(
            "weight_scale_group does not match the kernel capability", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

inline bool weight_scale_group_matches(uint32_t weight_scale_group, QuantKernelKind kernel) noexcept {
    return validate_weight_scale_group(weight_scale_group, kernel).ok();
}

struct RuntimeWeightQuantDesc {
    manifest::QuantizedEncoding encoding = manifest::QuantizedEncoding::Bf16;
    uint32_t block_n = 1;
    uint32_t block_k = 0;
    QuantScalarType scale_dtype = QuantScalarType::F32;
    uint64_t scale_stride_n = 0;
    uint64_t k_padded = 0;
};

inline RuntimeWeightQuantDesc make_runtime_weight_quant_desc(
    manifest::QuantizedEncoding encoding, uint32_t weight_scale_group) {
    RuntimeWeightQuantDesc desc;
    desc.encoding = encoding;
    const WeightBlockShape block = default_weight_block(encoding);
    desc.block_n = block.n;
    desc.block_k = weight_scale_group;
    const std::optional<WeightScaleSpec> scale = default_weight_scale_spec(encoding);
    desc.scale_dtype = scale.has_value() ? scale->dtype : QuantScalarType::F32;
    return desc;
}

inline Status validate_runtime_kernel_capability(
    const RuntimeWeightQuantDesc& desc, QuantKernelKind kernel) {
    const manifest::QuantizedEncoding enc = kernel_weight_encoding(kernel);
    if (desc.encoding != enc) {
        return Status::invalid_state(
            "runtime weight quant spec is not supported by the kernel", __FILE__, __LINE__);
    }
    if (!weight_block_supported(enc, WeightBlockShape{desc.block_n, desc.block_k})) {
        return Status::invalid_state(
            "runtime weight scale group is not supported by the kernel", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

}
