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
    if (s.experts > 1u && enc != 3u && enc != 4u) {
        return Status::invalid_argument(
            ("rank-3 weight table slot needs the mxfp4 or exl3 encoding: index " +
             std::to_string(index))
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
    if (enc == 4u) {
        if (s.device_ptr != nullptr || s.codes == nullptr || s.exl3_suh == nullptr ||
            s.exl3_svh == nullptr || s.exl3_marker == nullptr) {
            return Status::invalid_argument(
                ("exl3 weight table slot needs trellis+suh+svh+marker and no device_ptr: index " +
                 std::to_string(index))
                    .c_str(),
                __FILE__, __LINE__);
        }
        if (s.preshuffled) {
            return Status::invalid_argument(
                ("exl3 weight table slot must not be preshuffled: index " +
                 std::to_string(index))
                    .c_str(),
                __FILE__, __LINE__);
        }
        if (s.exl3_ka < 1u || s.exl3_ka > 8u || (s.exl3_half && s.exl3_ka > 3u)) {
            return Status::invalid_argument(
                ("exl3 weight table slot bit rate invalid: index " + std::to_string(index))
                    .c_str(),
                __FILE__, __LINE__);
        }
        if (s.k_padded != s.cols || (s.cols & 127u) != 0u || (s.rows & 127u) != 0u) {
            return Status::invalid_argument(
                ("exl3 weight table slot shape must be 128-aligned: index " +
                 std::to_string(index))
                    .c_str(),
                __FILE__, __LINE__);
        }
        const uint64_t words = 16u * s.exl3_ka + (s.exl3_half ? 8u : 0u);
        const uint64_t row_stride = (static_cast<uint64_t>(s.rows) / 16u) * words * 2u;
        if (s.codes_row_stride_bytes != row_stride) {
            return Status::invalid_argument(
                ("exl3 weight table slot trellis stride mismatch: index " +
                 std::to_string(index))
                    .c_str(),
                __FILE__, __LINE__);
        }
        if (s.experts > 1u) {
            const uint64_t expert_trellis_bytes =
                (static_cast<uint64_t>(s.cols) / 16u) * row_stride;
            if (s.codes_batch_stride_bytes != expert_trellis_bytes ||
                s.exl3_suh_batch_stride_bytes != static_cast<uint64_t>(s.cols) * 2u ||
                s.exl3_svh_batch_stride_bytes != static_cast<uint64_t>(s.rows) * 2u ||
                s.exl3_marker_batch_stride_bytes != 4u) {
                return Status::invalid_argument(
                    ("rank-3 exl3 weight table slot batch strides do not match the expert "
                     "plane: index " +
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
