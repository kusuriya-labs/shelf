#pragma once
#include <hip/hip_fp8.h>
#include <cstdint>
#include <string_view>

namespace shelf {
namespace kernel {

using fp8e4m3_storage_t = __hip_fp8_storage_t;
static_assert(sizeof(fp8e4m3_storage_t) == 1, "FP8 E4M3 storage must be exactly 1 byte");

enum class KVCacheDType : uint8_t {
    BF16 = 0,
    FP8_E4M3 = 1,
    OCTAVE = 2,
};

enum class OctaveKMode : uint8_t {
    Compact = 0,
    Block = 1,
};

struct OctaveKVCacheFormat {
    OctaveKMode k_mode = OctaveKMode::Compact;
    uint8_t k_rope_bits = 4;
    uint8_t k_nope_bits = 4;
    uint8_t v_bits = 4;
};

constexpr bool operator==(const OctaveKVCacheFormat& a,
                          const OctaveKVCacheFormat& b) noexcept {
    return a.k_mode == b.k_mode && a.k_rope_bits == b.k_rope_bits &&
           a.k_nope_bits == b.k_nope_bits && a.v_bits == b.v_bits;
}

constexpr bool operator!=(const OctaveKVCacheFormat& a,
                          const OctaveKVCacheFormat& b) noexcept {
    return !(a == b);
}

constexpr bool octave_bits_supported(uint32_t bits) noexcept {
    return bits >= 3u && bits <= 6u;
}

constexpr bool octave_format_valid(const OctaveKVCacheFormat& format) noexcept {
    return octave_bits_supported(format.k_rope_bits) &&
           octave_bits_supported(format.k_nope_bits) &&
           octave_bits_supported(format.v_bits) &&
           (format.k_mode == OctaveKMode::Compact ||
            format.k_mode == OctaveKMode::Block) &&
           (format.k_mode != OctaveKMode::Compact ||
            format.k_rope_bits == format.k_nope_bits);
}

constexpr uint32_t octave_plane_bytes(uint32_t dimensions,
                                      uint32_t bits) noexcept {
    return dimensions * bits / 8u;
}

constexpr uint32_t octave_k_slot_bytes(
    const OctaveKVCacheFormat& format) noexcept {
    return format.k_mode == OctaveKMode::Compact
        ? octave_plane_bytes(256u, format.k_nope_bits) + 2u
        : octave_plane_bytes(64u, format.k_rope_bits) +
              3u * octave_plane_bytes(64u, format.k_nope_bits) + 8u;
}

constexpr uint32_t octave_v_slot_bytes(
    const OctaveKVCacheFormat& format) noexcept {
    return octave_plane_bytes(256u, format.v_bits) + 2u;
}

inline bool parse_octave_kv_cache_format(std::string_view text,
                                         OctaveKVCacheFormat& out) noexcept {
    if (text == "octave_k3v3_compact") text = "octave:c3v3";
    if (text == "octave_k3v3") text = "octave:b4n3v3";
    if (text == "octave_k3v4") text = "octave:b4n3v4";
    constexpr std::string_view prefix = "octave:";
    if (!text.starts_with(prefix)) return false;
    text.remove_prefix(prefix.size());
    OctaveKVCacheFormat parsed{};
    if (text.size() == 4u && text[0] == 'c' && text[2] == 'v') {
        parsed.k_mode = OctaveKMode::Compact;
        parsed.k_rope_bits = static_cast<uint8_t>(text[1] - '0');
        parsed.k_nope_bits = parsed.k_rope_bits;
        parsed.v_bits = static_cast<uint8_t>(text[3] - '0');
    } else if (text.size() == 6u && text[0] == 'b' && text[2] == 'n' &&
               text[4] == 'v') {
        parsed.k_mode = OctaveKMode::Block;
        parsed.k_rope_bits = static_cast<uint8_t>(text[1] - '0');
        parsed.k_nope_bits = static_cast<uint8_t>(text[3] - '0');
        parsed.v_bits = static_cast<uint8_t>(text[5] - '0');
    } else {
        return false;
    }
    if (!octave_format_valid(parsed)) return false;
    out = parsed;
    return true;
}

}
}
