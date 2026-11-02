#pragma once

#include <phaseshift/core/status.h>

#include <cstdint>
#include <memory>
#include <span>

namespace ps::quantization::psq5 {

inline constexpr uint32_t kBlockK = 32u;
inline constexpr uint32_t kTileRows = 16u;
inline constexpr uint32_t kMagnitudeBytesPerBlock = 16u;
inline constexpr uint32_t kSignBytesPerBlock = 4u;
inline constexpr uint32_t kScaleBytesPerBlock = 1u;
inline constexpr uint32_t kBytesPerBlock =
    kMagnitudeBytesPerBlock + kSignBytesPerBlock + kScaleBytesPerBlock;

inline constexpr uint8_t kCodebook[16] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28};

constexpr uint8_t cb28_to_e4m3(uint8_t q5) noexcept {
    const uint8_t magnitude = q5 & 0x0Fu;
    const uint8_t sign = static_cast<uint8_t>((q5 & 0x10u) << 3);
    const uint8_t value = static_cast<uint8_t>(magnitude +
                                              ((magnitude >> 3) * (magnitude & 0x07u)));
    return static_cast<uint8_t>(value | sign);
}

constexpr uint8_t magnitude_sign_to_e4m3(uint8_t magnitude, uint8_t sign_bit) noexcept {
    const uint8_t value = static_cast<uint8_t>(magnitude +
                                              ((magnitude >> 3) * (magnitude & 0x07u)));
    return static_cast<uint8_t>(value | static_cast<uint8_t>(sign_bit << 7));
}

float ue4m4_value(uint8_t stored) noexcept;

struct Psq5CanonicalView {
    uint64_t n = 0;
    uint64_t logical_k = 0;
    uint64_t padded_k = 0;
    const uint8_t* magnitude = nullptr;
    uint64_t magnitude_bytes = 0;
    const uint8_t* sign = nullptr;
    uint64_t sign_bytes = 0;
    const uint8_t* scale = nullptr;
    uint64_t scale_bytes = 0;

    uint64_t blocks_per_row() const noexcept { return padded_k / kBlockK; }
    bool validate() const noexcept;
};

struct Psq5NativeView {
    uint64_t rows = 0;
    uint64_t n = 0;
    uint64_t padded_k = 0;
    uint64_t blocks_per_row = 0;
    const uint8_t* magnitude = nullptr;
    uint64_t magnitude_bytes = 0;
    const uint8_t* sign = nullptr;
    uint64_t sign_bytes = 0;
    const uint8_t* scale = nullptr;
    uint64_t scale_bytes = 0;
    uint32_t magnitude_row_stride_bytes = 0;
    uint32_t sign_row_stride_bytes = 0;
};

struct Psq5NativeHost {
    uint64_t rows = 0;
    uint64_t n = 0;
    uint64_t padded_k = 0;
    uint64_t blocks_per_row = 0;
    uint32_t magnitude_row_stride_bytes = 0;
    uint32_t sign_row_stride_bytes = 0;
    std::unique_ptr<uint8_t[]> magnitude;
    uint64_t magnitude_bytes = 0;
    uint64_t magnitude_capacity = 0;
    std::unique_ptr<uint8_t[]> sign;
    uint64_t sign_bytes = 0;
    uint64_t sign_capacity = 0;
    std::unique_ptr<uint8_t[]> scale;
    uint64_t scale_bytes = 0;
    uint64_t scale_capacity = 0;

    std::span<const uint8_t> magnitude_span() const noexcept;
    std::span<const uint8_t> sign_span() const noexcept;
    std::span<const uint8_t> scale_span() const noexcept;
    Psq5NativeView view() const;
};

bool preshuffle_psq5_native(const Psq5CanonicalView& v, Psq5NativeHost& out);

void dequantize_psq5_canonical(const Psq5CanonicalView& v, float* dst);

}
