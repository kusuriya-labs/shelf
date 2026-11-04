#include <shelf/inference/execution/weight_slot_validation.h>
#include <shelf/quantization/mxfp4/mxfp4_native.h>

#include <cstddef>
#include <string>

namespace shelf::execution {

namespace QMX = ::shelf::quantization::mxfp4;

constexpr uint32_t kNativeOutputTile = QMX::mxfp4_native_output_tile;
constexpr uint32_t kNativeKGroupAlignment = kNativeOutputTile * 8u;

Status validate_host_weight_slot(const ::shelf::runtime::WeightSlot& s, size_t index) {
    const uint8_t enc = s.encoding;
    if (s.rows == 0 || s.cols == 0) {
        return Status::invalid_argument(
            ("weight table slot has zero rows/cols: index " + std::to_string(index)).c_str(),
            __FILE__, __LINE__);
    }
    if (s.experts == 0u) {
        return Status::invalid_argument(
            ("weight table slot has zero experts: index " + std::to_string(index)).c_str(),
            __FILE__, __LINE__);
    }
    if (s.experts > 1u && enc != 3u) {
        return Status::invalid_argument(
            ("rank-3 weight table slot needs the mxfp4 encoding: index " + std::to_string(index))
                .c_str(),
            __FILE__, __LINE__);
    }
    if (enc == 0u) {
        if (s.device_ptr == nullptr || s.codes != nullptr || s.scales != nullptr) {
            return Status::invalid_argument(
                ("bf16 weight table slot needs device_ptr and no codes/scales: index " +
                 std::to_string(index))
                    .c_str(),
                __FILE__, __LINE__);
        }
        return Status::make_ok();
    }
    if (enc == 2u) {
        if (s.device_ptr != nullptr || s.codes == nullptr || s.scales == nullptr) {
            return Status::invalid_argument(
                ("rocm fp8 weight table slot needs codes+scales and no device_ptr: index " +
                 std::to_string(index))
                    .c_str(),
                __FILE__, __LINE__);
        }
        if (s.k_padded < s.cols || (s.k_padded & 31u) != 0u) {
            return Status::invalid_argument(
                ("rocm fp8 weight table slot k_padded invalid: index " + std::to_string(index))
                    .c_str(),
                __FILE__, __LINE__);
        }
        return Status::make_ok();
    }
    if (enc == 1u) {
        if (s.device_ptr != nullptr || s.codes == nullptr || s.scales == nullptr) {
            return Status::invalid_argument(
                ("block scaled weight table slot needs codes+scales and no device_ptr: index " +
                 std::to_string(index))
                    .c_str(),
                __FILE__, __LINE__);
        }
        const uint32_t k_align = 127u;
        if (s.k_padded < s.cols || (s.k_padded & k_align) != 0u) {
            return Status::invalid_argument(
                ("block scaled weight table slot k_padded invalid: index " +
                 std::to_string(index))
                    .c_str(),
                __FILE__, __LINE__);
        }
        return Status::make_ok();
    }
    if (enc == 3u) {
        if (s.device_ptr != nullptr || s.codes == nullptr || s.scales == nullptr) {
            return Status::invalid_argument(
                ("mxfp4 weight table slot needs codes+scales and no device_ptr: index " +
                 std::to_string(index))
                    .c_str(),
                __FILE__, __LINE__);
        }
        if (s.k_padded < s.cols || (s.k_padded & 63u) != 0u) {
            return Status::invalid_argument(
                ("mxfp4 weight table slot k_padded invalid: index " + std::to_string(index))
                    .c_str(),
                __FILE__, __LINE__);
        }
        if (s.weight_scale_group != 32u) {
            return Status::invalid_argument(
                ("mxfp4 weight table slot scale group must be 32: index " + std::to_string(index))
                    .c_str(),
                __FILE__, __LINE__);
        }
        const bool rank3 = s.experts > 1u || s.codes_batch_stride_bytes != 0u ||
                           s.scales_batch_stride_bytes != 0u;
        if (rank3) {
            if ((s.rows % kNativeOutputTile) != 0u ||
                (s.k_padded % kNativeKGroupAlignment) != 0u) {
                return Status::invalid_argument(
                    ("rank-3 mxfp4 weight table slot needs 16 output rows and 128 k: index " +
                     std::to_string(index))
                        .c_str(),
                    __FILE__, __LINE__);
            }
            if (s.codes_batch_stride_bytes !=
                    QMX::mxfp4_native_codes_batch_stride_bytes(s.rows, s.k_padded) ||
                s.scales_batch_stride_bytes !=
                    QMX::mxfp4_native_scale_batch_stride_bytes(s.rows, s.k_padded)) {
                return Status::invalid_argument(
                    ("rank-3 mxfp4 weight table slot batch strides do not match the native expert "
                     "plane layout: index " +
                     std::to_string(index))
                        .c_str(),
                    __FILE__, __LINE__);
            }
        }
        return Status::make_ok();
    }
    return Status::invalid_argument(
        ("unsupported weight table encoding: index " + std::to_string(index)).c_str(),
        __FILE__, __LINE__);
}

}
