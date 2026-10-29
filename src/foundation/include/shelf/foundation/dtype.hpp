// shelf foundation — logical data types.
//
// foundation must stay independent of any model, container format
// (see docs/architecture/dependency-rules.md). Quantized payload formats are
// NOT DataTypes here: physical quantization is a QuantSpec concern (quant/).
// FP8 E4M3 is the exception — it is a real 1-byte floating-point number type
// (not an opaque symbol stream), so it belongs here. The *meaning* of a tensor
// that uses it (which scale applies, at what granularity) stays in quant/.
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
    fp8_e4m3,   // S1E4M3 (bias 7, NaN = 0x7F, no infinities) — safetensors "F8_E4M3"
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
        case DataType::fp8_e4m3: return 1;
        case DataType::unknown:
        default:             return 0;
    }
}

constexpr bool is_floating_point(DataType dt) noexcept {
    return dt == DataType::bf16 || dt == DataType::fp16 ||
           dt == DataType::f32 || dt == DataType::f64 ||
           dt == DataType::fp8_e4m3;
}

} // namespace shelf
