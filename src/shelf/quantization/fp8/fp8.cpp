#include <shelf/quantization/fp8/fp8.h>
#include <shelf/quantization/codec/e4m3.h>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace ps::quantization::fp8 {

namespace {

const QuantFormatDesc* fp8_desc() {
    return quant_format_desc(QuantFormatId::Fp8);
}

uint64_t scale_n_for(uint64_t n) {
    return (n + kBlockN - 1u) / kBlockN;
}

}  // namespace

void quantize_fp8_matrix(
    const float* src,
    uint64_t src_stride,
    uint64_t n,
    uint64_t k,
    uint64_t k_padded,
    uint8_t* codes,
    float* scales,
    uint64_t scale_block_k) {
    const uint64_t bk = scale_block_k != 0 ? scale_block_k : kBlockK;
    const uint64_t sn = scale_n_for(n);
    const uint64_t sk = k_padded / bk;
    std::memset(codes, 0, n * k_padded);
    for (uint64_t tb = 0; tb < sn; ++tb) {
        const uint64_t r0 = tb * kBlockN;
        const uint64_t r1 = std::min(n, r0 + kBlockN);
        for (uint64_t kb = 0; kb < sk; ++kb) {
            const uint64_t k0 = kb * bk;
            const uint64_t k1 = std::min(k_padded, k0 + bk);
            float amax = 0.0f;
            for (uint64_t r = r0; r < r1; ++r) {
                const float* row = src + r * src_stride;
                for (uint64_t kk = k0; kk < k1; ++kk) {
                    if (kk >= k) continue;
                    const float a = std::fabs(row[kk]);
                    if (std::isfinite(a) && a > amax) amax = a;
                }
            }
            float scale = 0.0f;
            if (amax > 0.0f && std::isfinite(amax)) scale = amax / kE4m3Max;
            scales[tb * sk + kb] = scale;
            if (scale == 0.0f) continue;
            for (uint64_t r = r0; r < r1; ++r) {
                const float* row = src + r * src_stride;
                uint8_t* out = codes + r * k_padded + k0;
                for (uint64_t kk = k0; kk < k1; ++kk) {
                    if (kk >= k) break;
                    const float v = row[kk];
                    out[kk - k0] = std::isfinite(v) ? codec::e4m3_encode_u8(v / scale) : 0u;
                }
            }
        }
    }
}

void dequantize_fp8_matrix(
    const uint8_t* codes,
    const float* scales,
    uint64_t n,
    uint64_t k_padded,
    float* dst,
    uint64_t scale_block_k) {
    const uint64_t bk = scale_block_k != 0 ? scale_block_k : kBlockK;
    const uint64_t sk = k_padded / bk;
    for (uint64_t r = 0; r < n; ++r) {
        const uint64_t tb = r / kBlockN;
        const uint8_t* crow = codes + r * k_padded;
        float* drow = dst + r * k_padded;
        for (uint64_t kk = 0; kk < k_padded; ++kk) {
            const float s = scales[tb * sk + kk / bk];
            drow[kk] = codec::e4m3_decode_f32(crow[kk]) * s;
        }
    }
}

bool Fp8BlockCanonicalView::validate() const {
    const QuantFormatDesc* desc = fp8_desc();
    if (desc == nullptr) return false;
    const uint64_t bk = effective_scale_block_k();
    if (padded_k == 0 || padded_k % bk != 0) return false;
    if (scale_n != scale_n_for(n)) return false;
    if (scale_k != padded_k / bk) return false;
    if (codes_bytes != quant_codes_bytes(batch * n, padded_k, *desc)) return false;
    if (scales_bytes != batch * scale_n * scale_k * sizeof(float)) return false;
    return true;
}

void Fp8BlockCanonicalStore::init(uint64_t batch_in, uint64_t n_in, uint64_t logical_k_in, uint64_t padded_k_in) {
    batch = batch_in;
    n = n_in;
    logical_k = logical_k_in;
    padded_k = padded_k_in;
    const QuantFormatDesc* desc = fp8_desc();
    if (desc == nullptr) return;
    const uint64_t sn = scale_n_for(n);
    const uint64_t sk = padded_k / effective_scale_block_k();
    codes.resize(batch * n * padded_k);
    scales.resize(batch * sn * sk);
}

Fp8BlockCanonicalView Fp8BlockCanonicalStore::view() const {
    Fp8BlockCanonicalView v;
    v.batch = batch;
    v.n = n;
    v.padded_k = padded_k;
    v.codes = codes.data();
    v.codes_bytes = codes.size();
    v.scales = scales.data();
    v.scales_bytes = scales.size() * sizeof(float);
    v.scale_n = scale_n_for(n);
    v.scale_k = padded_k / effective_scale_block_k();
    v.scale_block_k = scale_block_k;
    return v;
}

bool Fp8BlockCanonicalStore::quantize_matrix(uint64_t matrix_index, const float* src, uint64_t src_stride) {
    if (matrix_index >= batch) return false;
    const uint64_t sn = scale_n_for(n);
    const uint64_t sk = padded_k / effective_scale_block_k();
    if (codes.size() != batch * n * padded_k) return false;
    if (scales.size() != batch * sn * sk) return false;
    quantize_fp8_matrix(
        src,
        src_stride,
        n,
        logical_k,
        padded_k,
        codes.data() + matrix_index * n * padded_k,
        scales.data() + matrix_index * sn * sk,
        effective_scale_block_k());
    return true;
}

void dequantize_fp8_canonical_matrix(
    const Fp8BlockCanonicalView& v, uint64_t matrix_index, float* dst, uint64_t dst_stride) {
    const uint64_t bk = v.effective_scale_block_k();
    const uint64_t sn = scale_n_for(v.n);
    const uint64_t sk = v.padded_k / bk;
    const uint8_t* codes = v.codes + matrix_index * v.n * v.padded_k;
    const float* scales = v.scales + matrix_index * sn * sk;
    for (uint64_t r = 0; r < v.n; ++r) {
        const uint64_t tb = r / kBlockN;
        const uint8_t* crow = codes + r * v.padded_k;
        float* drow = dst + r * dst_stride;
        for (uint64_t kk = 0; kk < v.padded_k; ++kk) {
            drow[kk] = codec::e4m3_decode_f32(crow[kk]) * scales[tb * sk + kk / bk];
        }
    }
}

void dequantize_fp8_canonical(const Fp8BlockCanonicalView& v, std::vector<float>& out) {
    out.assign(v.batch * v.n * v.padded_k, 0.0f);
    for (uint64_t m = 0; m < v.batch; ++m) {
        dequantize_fp8_canonical_matrix(
            v, m, out.data() + m * v.n * v.padded_k, v.padded_k);
    }
}

}  // namespace ps::quantization::fp8
