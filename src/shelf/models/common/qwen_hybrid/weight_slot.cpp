#include <shelf/models/common/qwen_hybrid/weight_slot.h>

namespace shelf {
namespace qwen_hybrid {

namespace RT = ::shelf::runtime;
namespace W = ::shelf::weights;

RT::WeightSlot make_weight_slot(const W::MatrixWeight& w) {
    RT::WeightSlot s{};
    s.rows = w.rows;
    s.cols = w.cols;
    s.k_padded = w.k_padded;
    s.encoding = static_cast<uint8_t>(w.encoding);
    s.compute_spec = static_cast<uint8_t>(w.compute_spec);
    s.experts = w.experts != 0u ? w.experts : 1u;
    s.codes_batch_stride_bytes = w.codes_batch_stride_bytes;
    s.scales_batch_stride_bytes = w.scales_batch_stride_bytes;
    if (w.encoding == W::MatrixEncoding::Bf16) {
        s.device_ptr = w.bf16.data<void>();
        s.bytes = static_cast<uint64_t>(s.experts) * w.rows * w.cols * sizeof(uint16_t);
        s.codes = nullptr;
        s.scales = nullptr;
    } else {
        s.device_ptr = nullptr;
        s.bytes = 0;
        s.codes = w.codes.ndim() != 0 ? w.codes.data<void>() : nullptr;
        s.scales = w.scales.ndim() != 0 ? w.scales.data<void>() : nullptr;
        s.row_ref = w.row_ref.ndim() != 0 ? w.row_ref.data<void>() : nullptr;
        s.row_ref_exact = w.row_ref_exact;
        s.storage_scale_stride_bytes = w.storage_scale_stride_bytes;
        s.compute_codes = w.compute_codes.ndim() != 0 ? w.compute_codes.data<void>() : nullptr;
        s.compute_scales_bf16 =
            w.compute_scales_bf16.ndim() != 0 ? w.compute_scales_bf16.data<void>() : nullptr;
        s.codes_row_stride_bytes = w.codes_row_stride_bytes;
        s.scale_row_stride_bytes = w.scale_row_stride_bytes;
        s.weight_scale_group = w.weight_scale_group;
        s.preshuffled = w.preshuffled;
    }
    return s;
}

}
}
