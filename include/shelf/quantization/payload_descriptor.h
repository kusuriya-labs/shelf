#pragma once

#include <shelf/quantization/quant_spec.h>

#include <cstdint>
#include <string>
#include <vector>

namespace shelf::quantization {

enum class QuantPayloadRole : uint8_t {
    Codes,
    Scale,
    ZeroPoint,
    Codebook,
    Auxiliary,
    Data,
};

enum class TensorStorageOrder : uint8_t {
    RowMajor,
    ColumnMajor,
    Packed,
};

struct QuantPayloadRef {
    QuantPayloadRole role = QuantPayloadRole::Codes;
    std::string tensor;
    QuantScalarType dtype = QuantScalarType::U8;
    std::vector<uint64_t> shape;
    std::string crc32;
};

struct QuantizedTensorLayout {
    std::string layout_id;
    TensorStorageOrder storage_order = TensorStorageOrder::RowMajor;
    std::vector<uint64_t> physical_shape;
    uint64_t k_padded = 0;
};

struct QuantizedTensorDescriptor {
    QuantDomain domain = QuantDomain::Weight;
    std::vector<int64_t> logical_shape;
    WeightQuantSpec quant;
    QuantizedTensorLayout layout;
    std::vector<QuantPayloadRef> payloads;
};

inline const QuantPayloadRef* find_payload(
    const QuantizedTensorDescriptor& desc, QuantPayloadRole role) {
    for (const QuantPayloadRef& ref : desc.payloads) {
        if (ref.role == role) return &ref;
    }
    return nullptr;
}

}
