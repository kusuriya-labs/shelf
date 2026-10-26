// shelf foundation — logical data types.
//
// foundation must stay independent of any model, container format or SYCL
// (see docs/architecture/dependency-rules.md). Quantized payload formats are
// NOT DataTypes here: physical quantization is a QuantSpec concern (quant/).
#pragma once

#include <cstddef>
#include <cstdint>

namespace shelf {

enum class DataType : std::uint8_t {
    unknown = 0,
    u8,
    u16,
    u32,
    i32,
    i64,
    bf16,
    fp16,
    f32,
    f64,
};

// Byte size of one element. `unknown` yields 0 (e.g. opaque payload handled
// through a QuantSpec, not through this enum).
constexpr std::size_t element_size(DataType dt) noexcept {
    switch (dt) {
        case DataType::u8:   return 1;
        case DataType::u16:  return 2;
        case DataType::u32:  return 4;
        case DataType::i32:  return 4;
        case DataType::i64:  return 8;
        case DataType::bf16: return 2;
        case DataType::fp16: return 2;
        case DataType::f32:  return 4;
        case DataType::f64:  return 8;
        case DataType::unknown:
        default:             return 0;
    }
}

constexpr bool is_floating_point(DataType dt) noexcept {
    return dt == DataType::bf16 || dt == DataType::fp16 ||
           dt == DataType::f32 || dt == DataType::f64;
}

} // namespace shelf
