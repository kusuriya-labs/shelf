#pragma once
#include <cstddef>
#include <cstdint>

namespace shelf::quantization::manifest {

class Crc32Accumulator {
public:
    void update(const void* data, std::size_t len);
    uint32_t value() const;
private:
    uint32_t state_ = 0xFFFFFFFFu;
};

uint32_t crc32_iso_hdlc(const void* data, std::size_t len);

}
