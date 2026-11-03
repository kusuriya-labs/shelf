#pragma once

#include <shelf/core/status.h>
#include <shelf/quantization/quantization_types.h>
#include <shelf/quantization/quant_format.h>
#include <shelf/quantization/manifest/quantized_manifest.h>

#include <cstdint>

namespace shelf::quantization::manifest {

static_assert(shelf::quantization::kQuantBlockElements == 32);

inline Result<shelf::quantization::ResolvedQuantSpec> resolve_quant_spec(
    const QuantizedManifest& manifest,
    const QuantizedTensorMetadata& tensor)
{
    using shelf::quantization::QuantDescriptor;
    using shelf::quantization::QuantLayout;
    using shelf::quantization::QuantSpecId;
    using shelf::quantization::ResolvedQuantSpec;
    using shelf::quantization::StorageDType;

    if (manifest.format != kQuantizedSafetensorsFormat) {
        return Status::invalid_argument("quantized manifest format mismatch", __FILE__, __LINE__);
    }
    if (manifest.format_version < kQuantizedSafetensorsMinFormatVersion ||
        manifest.format_version > kQuantizedSafetensorsFormatVersion) {
        return Status::invalid_argument("quantized manifest format_version mismatch", __FILE__, __LINE__);
    }

    if (tensor.logical_shape.empty()) {
        return Status::invalid_argument("quantized tensor logical_shape is empty", __FILE__, __LINE__);
    }
    for (const int64_t d : tensor.logical_shape) {
        if (d <= 0) {
            return Status::invalid_argument(
                "quantized tensor logical_shape has non-positive dim", __FILE__, __LINE__);
        }
    }
    const uint64_t logical_k = static_cast<uint64_t>(tensor.logical_shape.back());

    ResolvedQuantSpec resolved;
    QuantDescriptor& d = resolved.descriptor;

    if (tensor.encoding == QuantizedEncoding::Bf16) {
        if (!tensor.data) {
            return Status::invalid_argument("bf16 quantized tensor missing data ref", __FILE__, __LINE__);
        }
        if (tensor.codes || tensor.metadata1) {
            return Status::invalid_argument(
                "bf16 quantized tensor must not carry codes or scales", __FILE__, __LINE__);
        }
        if (tensor.data->dtype != "BF16") {
            return Status::invalid_argument(
                "bf16 quantized tensor data dtype is not BF16", __FILE__, __LINE__);
        }
        if (tensor.k_padded != logical_k) {
            return Status::invalid_argument(
                "bf16 quantized tensor k_padded does not match logical k", __FILE__, __LINE__);
        }
        d.group_size = 0;
        d.block_size = 0;
        d.storage_type = StorageDType::BF16;
        d.scale_type = StorageDType::NONE;
        d.weight_layout = QuantLayout::BF16_V1;
        d.scale_layout = QuantLayout::NONE;
        resolved.spec_id = QuantSpecId::BF16_V1;
        return resolved;
    }

    if (tensor.encoding == QuantizedEncoding::Fp8) {
        if (!tensor.codes) {
            return Status::invalid_argument("quantized tensor missing codes ref", __FILE__, __LINE__);
        }
        if (!tensor.metadata1) {
            return Status::invalid_argument("quantized tensor missing scales ref", __FILE__, __LINE__);
        }
        if (tensor.data) {
            return Status::invalid_argument(
                "quantized tensor must not carry data ref", __FILE__, __LINE__);
        }
        if (!tensor.layout.empty() && tensor.layout != kFp8LayoutName) {
            return Status::invalid_argument("fp8 block128 layout mismatch", __FILE__, __LINE__);
        }
        if (tensor.codes->dtype != "U8") {
            return Status::invalid_argument(
                "quantized tensor codes dtype mismatch", __FILE__, __LINE__);
        }
        if (tensor.metadata1->dtype != "F32") {
            return Status::invalid_argument(
                "fp8 block128 scales dtype is not F32", __FILE__, __LINE__);
        }
        if (tensor.k_padded < logical_k ||
            (tensor.k_padded % shelf::quantization::kFp8ScaleBlockK) != 0) {
            return Status::invalid_argument("quantized tensor k_padded invalid", __FILE__, __LINE__);
        }
        d.group_size = shelf::quantization::kFp8ScaleBlockK;
        d.block_size = shelf::quantization::kFp8ScaleBlockK;
        d.storage_type = StorageDType::U8;
        d.scale_type = StorageDType::FP32;
        d.weight_layout = QuantLayout::FP8_E4M3_BLOCK128_V1;
        d.scale_layout = QuantLayout::FP8_SCALE_F32_N128K128_V1;
        resolved.spec_id = QuantSpecId::FP8_FINEGRAINED_F32_V1;
        return resolved;
    }

    if (tensor.encoding == QuantizedEncoding::Mxfp4) {
        if (!tensor.codes) {
            return Status::invalid_argument("mxfp4 quantized tensor missing codes ref", __FILE__, __LINE__);
        }
        if (!tensor.metadata1) {
            return Status::invalid_argument(
                "mxfp4 quantized tensor missing scale ref", __FILE__, __LINE__);
        }
        if (tensor.data) {
            return Status::invalid_argument(
                "mxfp4 quantized tensor must not carry data ref", __FILE__, __LINE__);
        }
        if (!tensor.layout.empty() && tensor.layout != kMxfp4LayoutName) {
            return Status::invalid_argument("mxfp4 layout mismatch", __FILE__, __LINE__);
        }
        if (tensor.codes->dtype != "U8") {
            return Status::invalid_argument(
                "mxfp4 quantized tensor codes dtype mismatch", __FILE__, __LINE__);
        }
        if (tensor.metadata1->dtype != "E8M0") {
            return Status::invalid_argument(
                "mxfp4 quantized tensor scales dtype is not E8M0", __FILE__, __LINE__);
        }
        if (tensor.k_padded < logical_k ||
            (tensor.k_padded % shelf::quantization::kMxfp4ScaleBlockK) != 0) {
            return Status::invalid_argument(
                "mxfp4 quantized tensor k_padded invalid", __FILE__, __LINE__);
        }
        d.group_size = shelf::quantization::kMxfp4ScaleBlockK;
        d.block_size = shelf::quantization::kMxfp4ScaleBlockK;
        d.storage_type = StorageDType::U8;
        d.scale_type = StorageDType::U8;
        d.weight_layout = QuantLayout::MXFP4_E2M1_BLOCK32_V1;
        d.scale_layout = QuantLayout::MXFP4_SCALE_E8M0_PER_ROW_BLOCK32_V1;
        resolved.spec_id = QuantSpecId::MXFP4_E8M0_G32_V1;
        return resolved;
    }

    return Status::invalid_argument("unsupported quantized tensor encoding", __FILE__, __LINE__);
}

inline Status validate_quantized_manifest_contract(const QuantizedManifest& manifest) {
    if (manifest.format != kQuantizedSafetensorsFormat) {
        return Status::invalid_argument("quantized manifest format mismatch", __FILE__, __LINE__);
    }
    if (manifest.format_version < kQuantizedSafetensorsMinFormatVersion ||
        manifest.format_version > kQuantizedSafetensorsFormatVersion) {
        return Status::invalid_argument("quantized manifest format_version mismatch", __FILE__, __LINE__);
    }
    if (manifest.tensors.empty()) {
        return Status::invalid_argument("quantized manifest has no tensors", __FILE__, __LINE__);
    }
    for (const auto& kv : manifest.tensors) {
        auto r = resolve_quant_spec(manifest, kv.second);
        if (!r.ok()) {
            return r.status();
        }
    }
    return Status::make_ok();
}

} // namespace shelf::quantization::manifest
