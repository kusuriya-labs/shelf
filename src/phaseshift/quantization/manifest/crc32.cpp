#include <phaseshift/quantization/manifest/crc32.h>
#include <array>

namespace ps::quantization::manifest {

namespace {

constexpr std::array<uint32_t, 256> make_crc_table() {
    std::array<uint32_t, 256> t{};
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int k = 0; k < 8; ++k) {
            c = (c & 1u) ? 0xEDB88320u ^ (c >> 1) : (c >> 1);
        }
        t[i] = c;
    }
    return t;
}

const std::array<uint32_t, 256>& crc_table() {
    static const std::array<uint32_t, 256> t = make_crc_table();
    return t;
}

}

void Crc32Accumulator::update(const void* data, std::size_t len) {
    const auto* p = static_cast<const uint8_t*>(data);
    for (std::size_t i = 0; i < len; ++i) {
        state_ = crc_table()[(state_ ^ p[i]) & 0xFFu] ^ (state_ >> 8);
    }
}

uint32_t Crc32Accumulator::value() const {
    return state_ ^ 0xFFFFFFFFu;
}

uint32_t crc32_iso_hdlc(const void* data, std::size_t len) {
    Crc32Accumulator acc;
    acc.update(data, len);
    return acc.value();
}

}
