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
    s.exl3_suh_batch_stride_bytes = w.exl3_suh_batch_stride_bytes;
    s.exl3_svh_batch_stride_bytes = w.exl3_svh_batch_stride_bytes;
    s.exl3_marker_batch_stride_bytes = w.exl3_marker_batch_stride_bytes;
    if (w.encoding == W::MatrixEncoding::Bf16) {
        s.device_ptr = W::matrix_payload(w);
        if (w.partition.has_value() && w.partition->axis == 0) {
            s.vocab_offset = static_cast<uint32_t>(w.partition->index) * w.rows;
            s.vocab_sharded = true;
        }
        s.bytes = static_cast<uint64_t>(s.experts) * w.rows * w.cols * sizeof(uint16_t);
        s.codes = nullptr;
        s.scales = nullptr;
    } else {
        s.device_ptr = nullptr;
        s.bytes = 0;
        if (w.host_resident) {
            s.codes = w.host_codes;
            s.scales = w.host_scales;
            s.row_ref = w.host_row_ref;
            if (w.encoding == W::MatrixEncoding::Exl3) {
                s.exl3_suh = w.host_scales;
                s.exl3_svh = w.host_row_ref;
                s.exl3_marker = w.host_exl3_marker;
            }
        } else {
            s.codes = w.codes.ndim() != 0 ? w.codes.data<void>() : nullptr;
            s.scales = w.scales.ndim() != 0 ? w.scales.data<void>() : nullptr;
            s.row_ref = w.row_ref.ndim() != 0 ? w.row_ref.data<void>() : nullptr;
            s.exl3_suh = w.exl3_suh.ndim() != 0 ? w.exl3_suh.data<void>() : nullptr;
            s.exl3_svh = w.exl3_svh.ndim() != 0 ? w.exl3_svh.data<void>() : nullptr;
            s.exl3_marker = w.exl3_marker.ndim() != 0 ? w.exl3_marker.data<void>() : nullptr;
        }
        s.row_ref_exact = w.row_ref_exact;
        s.storage_scale_stride_bytes = w.storage_scale_stride_bytes;
        s.compute_codes = w.compute_codes.ndim() != 0 ? w.compute_codes.data<void>() : nullptr;
        s.compute_scales_bf16 =
            w.compute_scales_bf16.ndim() != 0 ? w.compute_scales_bf16.data<void>() : nullptr;
        s.exl3_ka = w.exl3_ka;
        s.exl3_half = w.exl3_half ? 1u : 0u;
        s.codes_row_stride_bytes = w.codes_row_stride_bytes;
        s.scale_row_stride_bytes = w.scale_row_stride_bytes;
        s.weight_scale_group = w.weight_scale_group;
        s.preshuffled = w.preshuffled;
    }
    return s;
}

}
}
