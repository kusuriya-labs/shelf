#pragma once

#include <cstdint>

namespace shelf {
namespace qwen35 {
namespace runtime {

enum class DecodeBackend : uint8_t {
    Hip = 0,
};

const char* decode_backend_name(DecodeBackend backend);

bool parse_decode_backend(const char* text, DecodeBackend* out);

}  // namespace runtime
}  // namespace qwen35
}  // namespace shelf
