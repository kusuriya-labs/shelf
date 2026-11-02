#include <phaseshift/models/qwen35/runtime/decode_backend.h>

#include <strings.h>

namespace ps {
namespace qwen35 {
namespace runtime {

const char* decode_backend_name(DecodeBackend backend) {
    switch (backend) {
        case DecodeBackend::Hip:
            return "Hip";
    }
    return "unknown";
}

bool parse_decode_backend(const char* text, DecodeBackend* out) {
    if (text == nullptr || out == nullptr) {
        return false;
    }
    if (strcasecmp(text, "host") == 0) {
        *out = DecodeBackend::Hip;
        return true;
    }
    return false;
}

}  // namespace runtime
}  // namespace qwen35
}  // namespace ps
