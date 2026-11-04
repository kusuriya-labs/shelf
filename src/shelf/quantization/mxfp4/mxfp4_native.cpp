#include <shelf/quantization/mxfp4/mxfp4_native.h>

#include <cstring>

namespace shelf::quantization::mxfp4 {

bool Mxfp4NativeView::validate() const {
    if (batch == 0u || n == 0u || padded_k == 0u || codes == nullptr || scales == nullptr)
        return false;
    if ((padded_k % kMxfp4ScaleBlockK) != 0u)
        return false;
    if (rows == 0u || (rows % 16u) != 0u || rows < n)
        return false;
    if (blocks != padded_k / kMxfp4ScaleBlockK)
        return false;
    if (codes_row_stride_bytes != blocks * mxfp4_native_block_bytes)
        return false;
    if (scale_row_stride_bytes != blocks)
        return false;
    if (codes_bytes != batch * mxfp4_native_codes_batch_stride_bytes(
                             static_cast<uint32_t>(rows), static_cast<uint32_t>(padded_k)))
        return false;
    return scales_bytes == batch * n * blocks;
}

Mxfp4NativeView Mxfp4NativeHost::view() const {
    Mxfp4NativeView v;
    v.batch = batch;
    v.rows = rows;
    v.n = n;
    v.padded_k = padded_k;
    v.blocks = blocks;
    v.codes_row_stride_bytes = codes_row_stride_bytes;
    v.scale_row_stride_bytes = scale_row_stride_bytes;
    v.codes = codes.data();
    v.codes_bytes = codes.size();
    v.scales = scales.data();
    v.scales_bytes = scales.size();
    v.row_ref_exact = row_ref_exact;
    v.row_ref = row_ref.data();
    v.row_ref_bytes = row_ref.size();
    return v;
}

bool preshuffle_mxfp4_native(const Mxfp4CanonicalView& v, Mxfp4NativeHost& out) {
    if (!v.validate())
        return false;
    if ((v.padded_k % kMxfp4ScaleBlockK) != 0u)
        return false;
    const uint64_t n = v.n;
    const uint64_t kp = v.padded_k;
    const uint64_t nb = kp / kMxfp4ScaleBlockK;
    const uint64_t tiles = (n + 15u) / 16u;

    out.batch = v.batch;
    out.n = n;
    out.padded_k = kp;
    out.blocks = nb;
    out.rows = tiles * 16u;
    out.codes_row_stride_bytes = nb * mxfp4_native_block_bytes;
    out.scale_row_stride_bytes = nb;
    out.codes.assign(out.batch * tiles * nb * mxfp4_native_block_bytes, 0u);
    out.scales.assign(v.scales, v.scales + v.scales_bytes);
    out.row_ref.assign(out.batch * out.n, 0u);
    out.row_ref_exact = true;

    const uint64_t canonical_row_bytes = kp / kMxfp4WeightsPerCode;
    for (uint64_t b = 0; b < out.batch; ++b) {
        const uint8_t* src_codes = v.codes + b * n * canonical_row_bytes;
        const uint8_t* src_scales = v.scales + b * n * nb;
        uint8_t* dst = out.codes.data() + b * tiles * nb * mxfp4_native_block_bytes;
        for (uint64_t t = 0; t < tiles; ++t) {
            for (uint64_t ib = 0; ib < nb; ++ib) {
                uint8_t* block = dst + (t * nb + ib) * mxfp4_native_block_bytes;
                for (uint32_t ol = 0; ol < 16u; ++ol) {
                    const uint64_t row = t * 16u + ol;
                    if (row >= n)
                        continue;
                    const uint8_t* srow = src_codes + row * canonical_row_bytes +
                        ib * (kMxfp4ScaleBlockK / kMxfp4WeightsPerCode);
                    uint8_t* drow = block + ol * 16u;
                    for (uint32_t j = 0; j < 16u; ++j) {
                        const uint8_t lo = static_cast<uint8_t>(
                            (srow[j >> 1u] >> (4u * (j & 1u))) & 0x0Fu);
                        const uint8_t hi = static_cast<uint8_t>(
                            (srow[(16u + j) >> 1u] >> (4u * ((16u + j) & 1u))) & 0x0Fu);
                        drow[j] = static_cast<uint8_t>(lo | (hi << 4u));
                    }
                }
            }
        }
        for (uint64_t r = 0; r < n; ++r) {
            uint8_t ref = 0u;
            uint8_t low = 255u;
            for (uint64_t ib = 0; ib < nb; ++ib) {
                const uint8_t value = src_scales[r * nb + ib];
                if (value > ref)
                    ref = value;
                if (value < low)
                    low = value;
            }
            out.row_ref[b * n + r] = ref;
            if (static_cast<uint32_t>(ref) - static_cast<uint32_t>(low) >
                kMxfp4FoldExactMaxDistance)
                out.row_ref_exact = false;
        }
    }
    return out.view().validate();
}

}  // namespace shelf::quantization::mxfp4
