#include <hip/hip_fp16.h>
#include <rocprim/block/block_scan.hpp>
#include <shelf/models/hy4_flash/attention.h>
#include <shelf/models/hy4_flash/absorbed.h>
#include <shelf/models/hy4_flash/mla_prefill.h>
#include <shelf/kernels/fused/detail/octave_common.h>
#include <shelf/kernels/fused/detail/radix_select.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <algorithm>
#include <climits>
#include <numeric>
#include <stdexcept>

namespace shelf::hy4 {
namespace {
using namespace shelf::kernel::exl3_detail;
using namespace shelf::kernel::exl3_detail::mul24;
using shelf::kernel::detail::octave::sign_of;
__device__ int step_at(int value, const Hy4StepParams *params) { return params ? params->position : value; }
__device__ int kv_base(int first, int row_offset, bool sliding, const Hy4StepParams *params) {
    if (!params)
        return first;
    return sliding ? max(0, params->position + row_offset - 511) : 0;
}
__device__ __forceinline__ const uint8_t *kv_record_const(const uint8_t *base, const int32_t *page_table, int pos, int slots) {
    return page_table ? base + (size_t(page_table[pos >> 8]) * 256 + (pos & 255)) * 294
                      : base + size_t(pos % slots) * 294;
}
__device__ __forceinline__ uint8_t *kv_record(uint8_t *base, const int32_t *page_table, int pos, int slots) {
    return page_table ? base + (size_t(page_table[pos >> 8]) * 256 + (pos & 255)) * 294
                      : base + size_t(pos % slots) * 294;
}
__device__ __forceinline__ const uint16_t *index_group_const(const uint16_t *base, const int32_t *page_table, int group) {
    return page_table ? base + (size_t(page_table[group >> 6]) * 64 + (group & 63)) * 128
                      : base + size_t(group) * 128;
}
__device__ __forceinline__ uint16_t *index_group(uint16_t *base, const int32_t *page_table, int group) {
    return page_table ? base + (size_t(page_table[group >> 6]) * 64 + (group & 63)) * 128
                      : base + size_t(group) * 128;
}
__device__ constexpr float levels[] = {-2.73258948f, -2.06901717f, -1.61804640f, -1.25623119f,
                                       -0.94234043f, -0.65675914f, -0.38804829f, -0.12839504f,
                                       0.12839504f,  0.38804829f,  0.65675914f,  0.94234043f,
                                       1.25623119f,  1.61804640f,  2.06901717f,  2.73258948f};
__device__ float read(Param p, size_t i) {
    return p.type == io::SType::F32 ? static_cast<const float *>(p.p)[i]
                                    : bf16_bits_to_f32(static_cast<const uint16_t *>(p.p)[i]);
}
__device__ float bv(Mat x, int row, int col) { return bf16_bits_to_f32(x.p[size_t(row) * x.cols + col]); }
__device__ float wht(float v, float *scratch, int width) {
    int c = threadIdx.x;
    for (int step = 1; step < 32 && step < width; step *= 2) {
        float other = __shfl_xor(v, step, 32);
        v = (c & step) ? other - v : v + other;
    }
    if (width == 64) {
        scratch[c] = v;
        __syncthreads();
        float other = scratch[c ^ 32];
        v = (c & 32) ? other - v : v + other;
        return v * rsqrtf(64.f);
    }
    if (width == 256) {
        scratch[c] = v;
        __syncthreads();
        int lane = c & 31, wave = c >> 5;
        float u[8];
#pragma unroll
        for (int w = 0; w < 8; ++w)
            u[w] = scratch[w * 32 + lane];
        float a0 = u[0] + u[1], a1 = u[0] - u[1];
        float a2 = u[2] + u[3], a3 = u[2] - u[3];
        float a4 = u[4] + u[5], a5 = u[4] - u[5];
        float a6 = u[6] + u[7], a7 = u[6] - u[7];
        float b0 = a0 + a2, b1 = a1 + a3, b2 = a0 - a2, b3 = a1 - a3;
        float b4 = a4 + a6, b5 = a5 + a7, b6 = a4 - a6, b7 = a5 - a7;
        float out[8] = {b0 + b4, b1 + b5, b2 + b6, b3 + b7, b0 - b4, b1 - b5, b2 - b6, b3 - b7};
        return out[wave] * rsqrtf(256.f);
    }
    for (int step = 32; step < width; step *= 2) {
        scratch[c] = v;
        __syncthreads();
        float other = scratch[c ^ step];
        __syncthreads();
        v = (c & step) ? other - v : v + other;
    }
    return v * rsqrtf(float(width));
}
__device__ float wht_shared(float v, float *scratch, int width, int c, bool active) {
    for (int step = 1; step < width; step *= 2) {
        scratch[c] = active ? v : 0.f;
        __syncthreads();
        float other = scratch[c ^ step];
        __syncthreads();
        if (active)
            v = (c & step) ? other - v : v + other;
    }
    return v * rsqrtf(float(width));
}
template <bool Max> __device__ float reduce(float v, float *scratch, int width) {
    int c = threadIdx.x;
    scratch[c] = v;
    __syncthreads();
    for (int step = width / 2; step; step /= 2) {
        if (c < step)
            scratch[c] = Max ? fmaxf(scratch[c], scratch[c + step]) : scratch[c] + scratch[c + step];
        __syncthreads();
    }
    float result = scratch[0];
    __syncthreads();
    return result;
}
template <bool Max> __device__ float reduce_shared(float v, float *scratch, int width, int c, bool active) {
    scratch[c] = active ? v : (Max ? -INFINITY : 0.f);
    __syncthreads();
    for (int step = width / 2; step; step /= 2) {
        if (c < step && c < width)
            scratch[c] = Max ? fmaxf(scratch[c], scratch[c + step]) : scratch[c] + scratch[c + step];
        __syncthreads();
    }
    float result = active ? scratch[0] : 0.f;
    __syncthreads();
    return result;
}
template <int Width, int Salt>
__device__ void pack_part(Mat x, uint8_t *records, int offset, int sub, int row, int c, bool active, float *scratch,
                          uint8_t *codes) {
    uint8_t *dst = records + size_t(row) * 294 + offset + sub * (Width / 2 + 2);
    float v = wht_shared(active ? bv(x, row, sub * Width + c) * sign_of(c, Salt) : 0.f, scratch, Width, c, active);
    float ss = reduce_shared<false>(v * v, scratch, Width, c, active);
    float z = ss > 0 ? v * rsqrtf(ss / Width) : 0;
    int lo = 0, hi = 15;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (z > (levels[mid] + levels[mid + 1]) * 0.5f)
            lo = mid + 1;
        else
            hi = mid;
    }
    codes[c] = active && ss > 0 ? lo : 0;
    float dot = reduce_shared<false>(active ? v * levels[lo] : 0.f, scratch, Width, c, active);
    if (active && c < Width / 2)
        dst[c] = codes[c * 2] | (codes[c * 2 + 1] << 4);
    if (active && !c) {
        uint16_t h = __half_as_ushort(__float2half(dot > 0 ? ss / dot : 0));
        dst[Width / 2] = h & 255;
        dst[Width / 2 + 1] = h >> 8;
    }
}
__global__ void pack_kv_combined(Mat lat, Mat rot, uint8_t *records) {
    __shared__ float scratch[256];
    __shared__ uint8_t codes[256];
    int sub = blockIdx.x, row = blockIdx.y, c = threadIdx.x;
    if (sub < 2)
        pack_part<256, 29>(lat, records, 0, sub, row, c, true, scratch, codes);
    else
        pack_part<64, 11>(rot, records, 260, 0, row, c, c < 64, scratch, codes);
}
template <int Width, int Salt>
__global__ void unpack(const uint8_t *cache, const uint8_t *fresh, const int *ids, int first, int start, int slots,
                       Mat out, int offset, const Hy4StepParams *params, bool sliding, int row_offset, int chunk_rows,
                       const int *meta, const int32_t *page_table) {
    int unpack_row = blockIdx.y + blockIdx.z * gridDim.y;
    if (meta && unpack_row >= meta[512] + meta[513])
        return;
    __shared__ float scratch[Width];
    start = step_at(start, params);
    int base = kv_base(first, row_offset, sliding, params);
    int c = threadIdx.x, sub = blockIdx.x, row = blockIdx.y + blockIdx.z * gridDim.y;
    if (row >= out.rows)
        return;
    int id = ids ? ids[row] : base + row;
    if (!ids && id >= start + row_offset + chunk_rows) {
        out.p[size_t(row) * out.cols + sub * Width + c] = 0;
        return;
    }
    if (id < 0) {
        out.p[size_t(row) * out.cols + sub * Width + c] = 0;
        return;
    }
    const uint8_t *rec = id >= start ? fresh + size_t(id - start) * 294 : kv_record_const(cache, page_table, id, slots);
    const uint8_t *src = rec + offset + sub * (Width / 2 + 2);
    int code = (src[c / 2] >> (4 * (c % 2))) & 15;
    float scale = __half2float(__ushort_as_half(src[Width / 2] | (src[Width / 2 + 1] << 8)));
    float value = wht(levels[code] * scale, scratch, Width) * sign_of(c, Salt);
    out.p[size_t(row) * out.cols + sub * Width + c] = f32_to_bf16_rne_device(value);
}
__global__ void unpack_quant(const uint8_t *cache, const uint8_t *fresh, const int *ids, int first, int start,
                             int slots, const uint16_t *suh, _Float16 *out, const Hy4StepParams *params,
                             bool sliding, int row_offset, int chunk_rows, const int *meta, int max_rows,
                             const int32_t *page_table) {
    int row = blockIdx.y + blockIdx.z * gridDim.y, sub = blockIdx.x, c = threadIdx.x;
    if ((meta && row >= meta[512] + meta[513]) || row >= max_rows)
        return;
    __shared__ float scratch[256];
    __shared__ uint16_t decoded[256];
    __shared__ uint8_t s_src[128];
    start = step_at(start, params);
    int base = kv_base(first, row_offset, sliding, params);
    int id = ids ? ids[row] : base + row;
    if ((!ids && id >= start + row_offset + chunk_rows) || id < 0) {
        if (c < 64) {
            int col = sub * 256 + (c / 32) * 128 + (c % 32) * 4;
            for (int i = 0; i < 4; ++i)
                out[size_t(row) * 512 + col + i] = 0;
        }
        return;
    }
    const uint8_t *rec = id >= start ? fresh + size_t(id - start) * 294 : kv_record_const(cache, page_table, id, slots);
    const uint8_t *src = rec + sub * 130;
    if (c < 32)
        *reinterpret_cast<uint32_t *>(s_src + c * 4) = *reinterpret_cast<const uint32_t *>(src + c * 4);
    float scale = __half2float(__ushort_as_half(src[128] | (src[129] << 8)));
    __syncthreads();
    int code = (s_src[c >> 1] >> (4 * (c & 1))) & 15;
    float value = wht(levels[code] * scale, scratch, 256) * sign_of(c, 29);
    decoded[c] = f32_to_bf16_rne_device(value);
    __syncthreads();
    if (c < 64) {
        int local_off = (c / 32) * 128 + (c % 32) * 4;
        int col = sub * 256 + local_off;
        const uint2 d_raw = *reinterpret_cast<const uint2 *>(decoded + local_off);
        const uint2 s_raw = *reinterpret_cast<const uint2 *>(suh + col);
        float v[4] = {
            bf16_bits_to_f32(uint16_t(d_raw.x & 0xffffu)) * f16_bits_to_f32(uint16_t(s_raw.x & 0xffffu)),
            bf16_bits_to_f32(uint16_t(d_raw.x >> 16)) * f16_bits_to_f32(uint16_t(s_raw.x >> 16)),
            bf16_bits_to_f32(uint16_t(d_raw.y & 0xffffu)) * f16_bits_to_f32(uint16_t(s_raw.y & 0xffffu)),
            bf16_bits_to_f32(uint16_t(d_raw.y >> 16)) * f16_bits_to_f32(uint16_t(s_raw.y >> 16)),
        };
        fwht128(v[0], v[1], v[2], v[3], c % 32);
        float peak = 0;
        for (float &f : v) {
            f *= kInvSqrt128;
            peak = fmaxf(peak, fabsf(f));
        }
        for (int mask = 16; mask; mask /= 2)
            peak = fmaxf(peak, __shfl_xor(peak, mask, 32));
        float quant_scale = peak / 127;
        _Float16 qh[4];
        for (int i = 0; i < 4; ++i)
            qh[i] = quant_scale ? rintf(v[i] / quant_scale) * quant_scale : 0;
        *reinterpret_cast<uint2 *>(out + size_t(row) * 512 + col) = *reinterpret_cast<const uint2 *>(qh);
    }
}
template <int Width, int Salt>
__global__ void unpack_batch(const uint8_t *cache, const uint8_t *fresh, const int *slots,
                             const int *positions, int cache_slots, int stride, Mat out, int offset, bool sliding,
                             int block_rows, const int32_t *page_table, int page_stride) {
    __shared__ float scratch[Width];
    int c = threadIdx.x, sub = blockIdx.x, t = blockIdx.y, block = blockIdx.z;
    int row = block * block_rows, start = positions[row];
    int first = sliding ? max(0, start - 511) : 0;
    int id = first + t;
    if (id >= start + block_rows) {
        out.p[(size_t(block) * stride + t) * out.cols + sub * Width + c] = 0;
        return;
    }
    const int32_t *pt = page_table ? page_table + size_t(slots[row]) * page_stride : nullptr;
    const uint8_t *rec = id >= start
        ? fresh + size_t(row + id - start) * 294
        : (sliding ? cache + (size_t(slots[row]) * cache_slots + id % cache_slots) * 294
                   : kv_record_const(cache, pt, id, cache_slots));
    const uint8_t *src = rec + offset + sub * (Width / 2 + 2);
    int code = (src[c / 2] >> (4 * (c % 2))) & 15;
    float scale = __half2float(__ushort_as_half(src[Width / 2] | (src[Width / 2 + 1] << 8)));
    float value = wht(levels[code] * scale, scratch, Width) * sign_of(c, Salt);
    out.p[(size_t(block) * stride + t) * out.cols + sub * Width + c] = f32_to_bf16_rne_device(value);
}
__global__ void unpack_batch_quant(const uint8_t *cache, const uint8_t *fresh, const int *slots,
                                   const int *positions, int cache_slots, int stride, const uint16_t *suh,
                                   _Float16 *out, bool sliding, int block_rows,
                                   const int32_t *page_table, int page_stride) {
    __shared__ float scratch[256];
    __shared__ uint16_t decoded[256];
    int c = threadIdx.x, sub = blockIdx.x, t = blockIdx.y, block = blockIdx.z;
    int row = block * block_rows, start = positions[row];
    int first = sliding ? max(0, start - 511) : 0;
    int id = first + t;
    if (id >= start + block_rows) {
        if (c < 64) {
            int col = sub * 256 + (c / 32) * 128 + (c % 32) * 4;
            for (int i = 0; i < 4; ++i)
                out[(size_t(block) * stride + t) * 512 + col + i] = 0;
        }
        return;
    }
    const int32_t *pt = page_table ? page_table + size_t(slots[row]) * page_stride : nullptr;
    const uint8_t *rec = id >= start
        ? fresh + size_t(row + id - start) * 294
        : (sliding ? cache + (size_t(slots[row]) * cache_slots + id % cache_slots) * 294
                   : kv_record_const(cache, pt, id, cache_slots));
    const uint8_t *src = rec + sub * 130;
    int code = (src[c / 2] >> (4 * (c % 2))) & 15;
    float scale = __half2float(__ushort_as_half(src[128] | (src[129] << 8)));
    float value = wht(levels[code] * scale, scratch, 256) * sign_of(c, 29);
    decoded[c] = f32_to_bf16_rne_device(value);
    __syncthreads();
    if (c < 64) {
        int col = sub * 256 + (c / 32) * 128 + (c % 32) * 4;
        float v[4], peak = 0;
        for (int i = 0; i < 4; ++i)
            v[i] = bf16_bits_to_f32(decoded[(c / 32) * 128 + (c % 32) * 4 + i]) * f16_bits_to_f32(suh[col + i]);
        fwht128(v[0], v[1], v[2], v[3], c % 32);
        for (float &f : v) {
            f *= kInvSqrt128;
            peak = fmaxf(peak, fabsf(f));
        }
        for (int mask = 16; mask; mask /= 2)
            peak = fmaxf(peak, __shfl_xor(peak, mask, 32));
        float quant_scale = peak / 127;
        for (int i = 0; i < 4; ++i)
            out[(size_t(block) * stride + t) * 512 + col + i] = quant_scale ? rintf(v[i] / quant_scale) * quant_scale : 0;
    }
}
__global__ void batch_offsets(int *offsets, const int *positions, int stride, bool sliding, int block_rows) {
    int row = blockIdx.y, t = blockIdx.x * blockDim.x + threadIdx.x;
    int len = sliding ? min(positions[row] + 1, 512) : positions[row] + 1;
    int base = 0;
    if (block_rows > 1) {
        int block = row / block_rows;
        base = block * stride;
        if (sliding) base += max(0, positions[row] - 511) - max(0, positions[block * block_rows] - 511);
    }
    if (t < stride) offsets[size_t(row) * stride + t] = t < len ? base + t : -1;
}
__global__ void save_records_batch(uint8_t *cache, const uint8_t *fresh, const int *slots,
                                   const int *positions, int cache_slots, int rows,
                                   const int32_t *page_table, int page_stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * 294) {
        int row = i / 294;
        if (page_table) {
            kv_record(cache, page_table + size_t(slots[row]) * page_stride, positions[row], cache_slots)[i % 294] = fresh[i];
        } else {
            cache[(size_t(slots[row]) * cache_slots + positions[row] % cache_slots) * 294 + i % 294] = fresh[i];
        }
    }
}
__global__ void save_records(uint8_t *cache, const uint8_t *fresh, int rows, int start, int slots,
                             const Hy4StepParams *params, const int32_t *page_table) {
    start = step_at(start, params);
    int i = (page_table ? 0 : (rows - min(rows, slots)) * 294) + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 294 * rows)
        kv_record(cache, page_table, start + i / 294, slots)[i % 294] = fresh[i];
}
__global__ void pool(Mat keys, uint16_t *history, int first, int start, Param values, Param gates, Param ape,
                     Param norm, const uint16_t *pending, const Hy4StepParams *params, int rows,
                     const int32_t *page_table) {
    __shared__ float pooled[128], scratch[128];
    start = step_at(start, params);
    first = start / 4;
    if (blockIdx.x >= (start + rows) / 4 - first)
        return;
    int c = threadIdx.x, group = first + blockIdx.x;
    float v[4], g[4], peak = -INFINITY;
    for (int i = 0; i < 4; ++i) {
        int pos = 4 * group + i;
        auto x = pos >= start ? keys.p + size_t(pos - start) * 128 : pending + size_t(pos % 12) * 128;
        v[i] = 0;
        g[i] = read(ape, i * 128 + c);
        for (int j = 0; j < 128; ++j) {
            float a = bf16_bits_to_f32(x[j]);
            v[i] += a * read(values, c * 128 + j);
            g[i] += a * read(gates, c * 128 + j);
        }
        peak = fmaxf(peak, g[i]);
    }
    float sum = 0, total = 0;
    for (int i = 0; i < 4; ++i) {
        float p = expf(g[i] - peak);
        sum += p * v[i];
        total += p;
    }
    float out = sum / total;
    float inv = rsqrtf(reduce<false>(out * out, scratch, 128) / 128 + 1e-5f);
    pooled[c] = bf16_bits_to_f32(f32_to_bf16_rne_device(out * inv * read(norm, c)));
    __syncthreads();
    out = pooled[c];
    if (c >= 64) {
        float phase = group * 4 * powf(1e7f, -float((c - 64) / 2) / 32);
        float co = bf16_bits_to_f32(f32_to_bf16_rne_device(cosf(phase)));
        float si = bf16_bits_to_f32(f32_to_bf16_rne_device(sinf(phase)));
        out = out * co + pooled[c ^ 1] * ((c & 1) ? si : -si);
    }
    index_group(history, page_table, group)[c] = f32_to_bf16_rne_device(out);
}
__global__ void pool_batch(Mat keys, uint16_t *all_history, const uint16_t *all_pending, const int *slots,
                           const int *positions, Param values, Param gates, Param ape, Param norm, int block_rows, const int *row_starts,
                           const int32_t *page_table, int page_stride) {
    __shared__ float pooled[128], scratch[128];
    int row = blockIdx.x, start = positions[row];
    if ((start & 3) != 3) return;
    auto pending = all_pending + size_t(slots[row]) * 12 * 128;
    const int32_t *pt = page_table ? page_table + size_t(slots[row]) * page_stride : nullptr;
    int c = threadIdx.x, group = start / 4;
    int first = row_starts ? row_starts[row] : row - row % block_rows;
    start -= row - first;
    keys.p += size_t(first) * keys.cols;
    float v[4], g[4], peak = -INFINITY;
    for (int i = 0; i < 4; ++i) {
        int pos = 4 * group + i;
        auto x = pos >= start ? keys.p + size_t(pos - start) * 128 : pending + size_t(pos % 12) * 128;
        v[i] = 0;
        g[i] = read(ape, i * 128 + c);
        for (int j = 0; j < 128; ++j) {
            float a = bf16_bits_to_f32(x[j]);
            v[i] += a * read(values, c * 128 + j);
            g[i] += a * read(gates, c * 128 + j);
        }
        peak = fmaxf(peak, g[i]);
    }
    float sum = 0, total = 0;
    for (int i = 0; i < 4; ++i) {
        float p = expf(g[i] - peak);
        sum += p * v[i];
        total += p;
    }
    float out = sum / total;
    float inv = rsqrtf(reduce<false>(out * out, scratch, 128) / 128 + 1e-5f);
    pooled[c] = bf16_bits_to_f32(f32_to_bf16_rne_device(out * inv * read(norm, c)));
    __syncthreads();
    out = pooled[c];
    if (c >= 64) {
        float phase = group * 4 * powf(1e7f, -float((c - 64) / 2) / 32);
        float co = bf16_bits_to_f32(f32_to_bf16_rne_device(cosf(phase)));
        float si = bf16_bits_to_f32(f32_to_bf16_rne_device(sinf(phase)));
        out = out * co + pooled[c ^ 1] * ((c & 1) ? si : -si);
    }
    index_group(all_history, pt, group)[c] = f32_to_bf16_rne_device(out);
}
__global__ void save_pending_batch(Mat keys, uint16_t *all_pending, const int *slots, const int *positions) {
    int row = blockIdx.x;
    auto pending = all_pending + size_t(slots[row]) * 12 * 128;
    pending[(positions[row] % 12) * 128 + threadIdx.x] = keys.p[size_t(row) * 128 + threadIdx.x];
}
__global__ void save_pending(Mat k, int start, uint16_t *pending, const Hy4StepParams *params) {
    start = step_at(start, params);
    int row = k.rows - 1 - blockIdx.x;
    pending[((start + row) % 12) * 128 + threadIdx.x] = k.p[row * 128 + threadIdx.x];
}
template <bool Refine = false>
__global__ void index_scores(Mat q, Mat weights, const uint16_t *groups, int first, int start, int stride,
                             float *scores, const Hy4StepParams *params, const float *bounds = nullptr,
                             const int32_t *page_table = nullptr) {
    start = step_at(start, params);
    int row = first + blockIdx.y;
    int count = (start + row) / 4;
    int group = blockIdx.x * 4 + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (group >= count || group >= stride)
        return;
    if constexpr (Refine) {
        float score = scores[size_t(blockIdx.y) * stride + group];
        if (score < bounds[2 * blockIdx.y] || score > bounds[2 * blockIdx.y + 1]) return;
    }
    const uint16_t *group_ptr = index_group_const(groups, page_table, group);
    float sum = 0;
    for (int h = 0; h < 16; ++h) {
        float dot = 0;
        for (int j = lane; j < 128; j += 32)
            dot += bv(q, row, h * 128 + j) * bf16_bits_to_f32(group_ptr[j]);
        for (int dist = 16; dist; dist /= 2)
            dot += __shfl_down(dot, dist, 32);
        sum += fmaxf(dot, 0) * bv(weights, row, h);
    }
    if (!lane)
        scores[size_t(blockIdx.y) * stride + group] = sum;
}
using IndexFragment = short __attribute__((ext_vector_type(8)));
using IndexSum = float __attribute__((ext_vector_type(8)));

__device__ IndexFragment index_fragment(const uint16_t *src) {
    auto lo = *reinterpret_cast<const uint2 *>(src);
    auto hi = *reinterpret_cast<const uint2 *>(src + 8);
    uint4 packed{lo.x, lo.y, hi.x, hi.y};
    return __builtin_bit_cast(IndexFragment, packed);
}

__global__ void index_scores_wmma(Mat q, Mat weights, const uint16_t *groups, int first, int start,
                                  int stride, float *scores, const int32_t *page_table) {
    int row = first + blockIdx.y, count = min(stride, (start + row) / 4);
    int lane = threadIdx.x % 32, half = lane / 16, col = lane % 16;
    int tile = blockIdx.x * 64 + threadIdx.x / 32 * 16;
    if (tile >= count) return;
    int group = tile + col;
    const uint16_t *query = q.p + (size_t(row) * 16 + col) * 128 + half * 4;
    int clamped = min(group, count - 1);
    const uint16_t *key = (page_table
        ? groups + (size_t(page_table[tile >> 6]) * 64 + (clamped & 63)) * 128
        : groups + size_t(clamped) * 128) + half * 4;
    IndexSum dots{};
#pragma unroll
    for (int k = 0; k < 128; k += 16)
        dots = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(index_fragment(query + k),
                                                                  index_fragment(key + k), dots);
    float sum = 0;
#pragma unroll
    for (int h = 0; h < 8; ++h)
        sum += fmaxf(dots[h], 0.f) * bv(weights, row, half * 8 + h);
    sum += __shfl_xor(sum, 16, 32);
    if (!half && group < count)
        scores[size_t(blockIdx.y) * stride + group] = sum;
}

__global__ void index_key_peak(const uint16_t *keys, unsigned *peak, int elements, const int32_t *page_table) {
    __shared__ unsigned waves[8];
    unsigned bits = 0;
    for (int j = blockIdx.x * 256 + threadIdx.x; j < elements; j += gridDim.x * 256) {
        uint16_t val = page_table ? keys[size_t(page_table[j >> 13]) * 8192 + (j & 8191)] : keys[j];
        bits = max(bits, unsigned(val & 0x7fffu));
    }
    for (int d = 16; d; d /= 2)
        bits = max(bits, __shfl_down(bits, d));
    if (!(threadIdx.x % 32)) waves[threadIdx.x / 32] = bits;
    __syncthreads();
    if (!threadIdx.x) {
        for (int i = 1; i < 8; ++i) bits = max(bits, waves[i]);
        atomicMax(peak, bits);
    }
}
__global__ void __launch_bounds__(256) index_scores_refine(Mat q, Mat weights, const uint16_t *groups,
                                                             int first, int start, int stride, float *scores,
                                                             const float *bounds, const int32_t *page_table = nullptr) {
    __shared__ int cand[256];
    __shared__ int cand_count;
    int t = threadIdx.x, row = first + blockIdx.y;
    int count = min(stride, (start + row) / 4);
    int g0 = blockIdx.x * 256;
    if (!t)
        cand_count = 0;
    __syncthreads();
    int g = g0 + t;
    if (g < count) {
        float s = scores[size_t(blockIdx.y) * stride + g];
        if (s >= bounds[2 * blockIdx.y] && s <= bounds[2 * blockIdx.y + 1])
            cand[atomicAdd(&cand_count, 1)] = g;
    }
    __syncthreads();
    int total = cand_count;
    if (!total)
        return;
    int wave = t >> 5, lane = t & 31;
    float w_head[16];
    for (int h = 0; h < 16; ++h)
        w_head[h] = bv(weights, row, h);
    for (int idx = wave; idx < total; idx += 8) {
        int group = cand[idx];
        const uint16_t *gp = index_group_const(groups, page_table, group);
        float kg[4] = {
            bf16_bits_to_f32(gp[lane]),
            bf16_bits_to_f32(gp[lane + 32]),
            bf16_bits_to_f32(gp[lane + 64]),
            bf16_bits_to_f32(gp[lane + 96]),
        };
        float sum = 0;
        for (int h = 0; h < 16; ++h) {
            const uint16_t *qp = q.p + size_t(row) * q.cols + h * 128;
            float dot = bf16_bits_to_f32(qp[lane]) * kg[0] +
                        bf16_bits_to_f32(qp[lane + 32]) * kg[1] +
                        bf16_bits_to_f32(qp[lane + 64]) * kg[2] +
                        bf16_bits_to_f32(qp[lane + 96]) * kg[3];
            for (int dist = 16; dist; dist /= 2)
                dot += __shfl_down(dot, dist, 32);
            sum += fmaxf(dot, 0.f) * w_head[h];
        }
        if (!lane)
            scores[size_t(blockIdx.y) * stride + group] = sum;
    }
}

__device__ __forceinline__ uint32_t hy4_radix_key(float value) {
    const uint32_t bits = value == 0.f ? 0u : __float_as_uint(value);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ void hy4_scan_256(uint32_t val, uint32_t *wave_sums, uint32_t &exc, uint32_t &total) {
    int t = threadIdx.x, lane = t & 31, wave = t >> 5;
    uint32_t x = val;
    for (int d = 1; d < 32; d <<= 1) {
        uint32_t other = __shfl_up(x, d, 32);
        if (lane >= d)
            x += other;
    }
    if (lane == 31)
        wave_sums[wave] = x;
    __syncthreads();
    if (wave == 0) {
        uint32_t w = t < 8 ? wave_sums[t] : 0u;
        for (int d = 1; d < 8; d <<= 1) {
            uint32_t other = __shfl_up(w, d, 32);
            if (lane >= d)
                w += other;
        }
        if (t < 8)
            wave_sums[t] = w;
    }
    __syncthreads();
    uint32_t w_exc = wave > 0 ? wave_sums[wave - 1] : 0u;
    total = wave_sums[7];
    exc = w_exc + x - val;
}

__global__ void __launch_bounds__(256) index_topk(const float *scores, int first, int start, int stride, int *selected,
                                                  const Hy4StepParams *params) {
    __shared__ uint32_t wave_hist[8][256];
    __shared__ uint32_t cand_keys[2048];
    __shared__ int32_t cand_idx[2048];
    __shared__ uint32_t state[8], wave_sums[8];
    start = step_at(start, params);
    int row = blockIdx.x, t = threadIdx.x, wave = t >> 5;
    unsigned count = min(unsigned((start + first + row) / 4), unsigned(stride));
    const float *row_scores = scores + size_t(row) * stride;
    int *out = selected + row * 511;
    uint32_t rank = 511u, prefix = 0u, out_pos = 0u;
    bool in_lds = false;
    uint32_t cand_cnt = 0u;
    int pass = 0;
    for (; pass < 4; ++pass) {
        uint32_t shift = 24u - 8u * pass;
        for (int w = 0; w < 8; ++w)
            wave_hist[w][t] = 0u;
        __syncthreads();
        if (!in_lds) {
            for (uint32_t i = t; i < count; i += 256u) {
                uint32_t key = hy4_radix_key(row_scores[i]);
                if (pass == 0 || (key >> (shift + 8u)) == (prefix >> (shift + 8u)))
                    atomicAdd(&wave_hist[wave][(key >> shift) & 0xffu], 1u);
            }
        } else {
            for (uint32_t j = t; j < cand_cnt; j += 256u) {
                uint32_t key = cand_keys[j];
                if ((key >> (shift + 8u)) == (prefix >> (shift + 8u)))
                    atomicAdd(&wave_hist[wave][(key >> shift) & 0xffu], 1u);
            }
        }
        __syncthreads();
        uint32_t b_cnt = 0u;
        for (int w = 0; w < 8; ++w)
            b_cnt += wave_hist[w][255 - t];
        uint32_t exc, total_dummy;
        hy4_scan_256(b_cnt, wave_sums, exc, total_dummy);
        if (exc < rank && exc + b_cnt >= rank) {
            state[0] = uint32_t(255 - t);
            state[1] = rank - exc;
            state[2] = b_cnt;
        }
        __syncthreads();
        prefix |= state[0] << shift;
        rank = state[1];
        uint32_t bucket_cnt = state[2];
        if (!in_lds && bucket_cnt <= 2048u) {
            if (!t) {
                state[3] = out_pos;
                state[4] = 0u;
            }
            __syncthreads();
            uint32_t ptop = prefix >> shift;
            for (uint32_t i = t; i < count; i += 256u) {
                uint32_t key = hy4_radix_key(row_scores[i]);
                uint32_t top = key >> shift;
                if (top > ptop) {
                    uint32_t pos = atomicAdd(&state[3], 1u);
                    out[pos] = int(i);
                } else if (top == ptop) {
                    uint32_t cpos = atomicAdd(&state[4], 1u);
                    cand_keys[cpos] = key;
                    cand_idx[cpos] = int(i);
                }
            }
            __syncthreads();
            out_pos = state[3];
            cand_cnt = state[4];
            in_lds = true;
            if (rank == bucket_cnt) {
                for (uint32_t j = t; j < cand_cnt; j += 256u)
                    out[out_pos + j] = cand_idx[j];
                return;
            }
        } else {
            __syncthreads();
        }
    }
    uint32_t need = rank, index_prefix = 0u;
    for (int ip = 0; ip < 3 && need != 0u; ++ip) {
        uint32_t shift = 8u * (2 - ip);
        for (int w = 0; w < 8; ++w)
            wave_hist[w][t] = 0u;
        __syncthreads();
        if (in_lds) {
            for (uint32_t j = t; j < cand_cnt; j += 256u) {
                if (cand_keys[j] != prefix)
                    continue;
                uint32_t idx = uint32_t(cand_idx[j]);
                if (ip != 0 && (idx >> (shift + 8u)) != (index_prefix >> (shift + 8u)))
                    continue;
                atomicAdd(&wave_hist[wave][(idx >> shift) & 0xffu], 1u);
            }
        } else {
            for (uint32_t i = t; i < count; i += 256u) {
                if (hy4_radix_key(row_scores[i]) != prefix)
                    continue;
                if (ip != 0 && (i >> (shift + 8u)) != (index_prefix >> (shift + 8u)))
                    continue;
                atomicAdd(&wave_hist[wave][(i >> shift) & 0xffu], 1u);
            }
        }
        __syncthreads();
        uint32_t b_cnt = 0u;
        for (int w = 0; w < 8; ++w)
            b_cnt += wave_hist[w][t];
        uint32_t exc, total_dummy;
        hy4_scan_256(b_cnt, wave_sums, exc, total_dummy);
        if (exc < need && exc + b_cnt >= need) {
            state[0] = uint32_t(t);
            state[1] = need - exc;
        }
        __syncthreads();
        index_prefix |= state[0] << shift;
        need = state[1];
        __syncthreads();
    }
    if (!t)
        state[3] = out_pos;
    __syncthreads();
    if (in_lds) {
        for (uint32_t j = t; j < cand_cnt; j += 256u) {
            uint32_t key = cand_keys[j];
            uint32_t idx = uint32_t(cand_idx[j]);
            if (key > prefix || (key == prefix && idx <= index_prefix))
                out[atomicAdd(&state[3], 1u)] = int(idx);
        }
    } else {
        for (uint32_t i = t; i < count; i += 256u) {
            uint32_t key = hy4_radix_key(row_scores[i]);
            if (key > prefix || (key == prefix && i <= index_prefix))
                out[atomicAdd(&state[3], 1u)] = int(i);
        }
    }
}
__global__ void __launch_bounds__(256) index_bounds(Mat q, Mat w, const float *scores, int first, int start,
                                                    int stride, float *bounds, const unsigned *key_peak) {
    __shared__ uint32_t wave_hist[8][256];
    __shared__ uint32_t state[4], wave_sums[8];
    unsigned rank = 511;
    int row = blockIdx.x, t = threadIdx.x, wave = t >> 5;
    int count = min(stride, (start + first + row) / 4);
    const float *row_scores = scores + size_t(row) * stride;
    uint32_t prefix = 0u;
    for (int pass = 0; pass < 2; ++pass) {
        uint32_t shift = 24u - 8u * pass;
        for (int wi = 0; wi < 8; ++wi)
            wave_hist[wi][t] = 0u;
        __syncthreads();
        for (int i = t; i < count; i += 256) {
            uint32_t key = hy4_radix_key(row_scores[i]);
            if (pass == 0 || (key >> (shift + 8u)) == (prefix >> (shift + 8u)))
                atomicAdd(&wave_hist[wave][(key >> shift) & 0xffu], 1u);
        }
        __syncthreads();
        uint32_t b_cnt = 0u;
        for (int wi = 0; wi < 8; ++wi)
            b_cnt += wave_hist[wi][255 - t];
        uint32_t exc, total_dummy;
        hy4_scan_256(b_cnt, wave_sums, exc, total_dummy);
        if (exc < rank && exc + b_cnt >= rank) {
            state[0] = uint32_t(255 - t);
            state[1] = rank - exc;
        }
        __syncthreads();
        prefix |= state[0] << shift;
        rank = state[1];
        __syncthreads();
    }
    uint4 qv = *reinterpret_cast<const uint4 *>(q.p + size_t(first + row) * q.cols + t * 8);
    uint32_t qw[4] = {qv.x, qv.y, qv.z, qv.w};
    float peak = 0;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        peak = fmaxf(peak, fabsf(bf16_bits_to_f32(uint16_t(qw[p] & 0xffffu))));
        peak = fmaxf(peak, fabsf(bf16_bits_to_f32(uint16_t(qw[p] >> 16))));
    }
    peak = reduce<true>(peak, reinterpret_cast<float *>(wave_sums), 256);
    if (!t) {
        float weight = 0;
        for (int h = 0; h < 16; ++h)
            weight += fabsf(bv(w, first + row, h));
        float key = bf16_bits_to_f32(*key_peak);
        float margin = ldexpf(128.f * weight * peak * key, -14);
        auto value = [](unsigned bits) { return __uint_as_float((bits & 0x80000000u) ? bits ^ 0x80000000u : ~bits); };
        bounds[2 * row] = value(prefix) - margin;
        bounds[2 * row + 1] = value(prefix | 0xffffu) + margin;
    }
}
__global__ void contiguous_offsets(int *offsets, int n, int stride, int pos0, bool sliding, int first, int row_offset,
                                   const Hy4StepParams *params) {
    int r = blockIdx.y, t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= stride)
        return;
    int pos = step_at(pos0, params) + row_offset + r;
    int base = kv_base(first, row_offset, sliding, params);
    int len = sliding ? min(pos + 1, 512) : pos + 1;
    int first_r = pos + 1 - len;
    offsets[r * stride + t] = t < len ? first_r - base + t : -1;
}
__global__ void draft_contiguous_offsets(int *ids, int *offsets, int stride, bool sliding,
                                         const Hy4StepParams *params) {
    int r = blockIdx.y, t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= stride)
        return;
    int start = params->position;
    int history = sliding ? min(start, 512) : start;
    if (!r)
        ids[t] = t < history + int(gridDim.y) ? start - history + t : -1;
    offsets[r * stride + t] = t <= history + r ? t : -1;
}
__global__ void sparse_clear(int *mask, int mask_count, unsigned *bits, int word_count, int *meta, int *rev,
                             int rev_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < mask_count)
        mask[i] = 0;
    if (i < word_count)
        bits[i] = 0;
    if (i < 516)
        meta[i] = 0;
    if (i < rev_count)
        rev[i] = -1;
}
__global__ void sort_top(const int *top, int *groups, int *mask, int row_begin, int rows, int width,
                         const Hy4StepParams *draft = nullptr) {
    __shared__ int g[512];
    int r = blockIdx.x, t = threadIdx.x;
    if (draft)
        row_begin = draft->draft_last;
    g[t] = t < 511 ? top[(size_t(row_begin + r)) * 511 + t] : INT_MAX;
    __syncthreads();
    for (int k = 2; k <= 512; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            __syncthreads();
            int other = g[t ^ j], v = g[t];
            bool lower = (t & j) == 0;
            bool gt = lower ? v > other : other > v;
            bool lt = lower ? v < other : other < v;
            bool swap = (t & k) == 0 ? gt : lt;
            __syncthreads();
            g[t] = swap ? other : v;
        }
    }
    __syncthreads();
    groups[size_t(r) * 512 + t] = g[t];
    if (t < 511 && g[t] < width)
        atomicOr(reinterpret_cast<unsigned *>(mask) + (g[t] >> 5), 1u << (g[t] & 31));
}
__global__ void sparse_group_ids(int *mask, int rows, int width, int *meta, int *ids, int *rev,
                                 unsigned *bits, int pos0, int row_offset, const Hy4StepParams *params) {
    pos0 = step_at(pos0, params) + row_offset;
    using Scan = rocprim::block_scan<int, 256>;
    __shared__ typename Scan::storage_type scan;
    int words = (width + 31) / 32;
    int span = (words + 255) / 256;
    int first = min(int(threadIdx.x) * span, words);
    int end = min(first + span, words);
    auto group_bits = reinterpret_cast<const unsigned *>(mask);
    int count = 0;
    for (int w = first; w < end; ++w)
        count += __popc(group_bits[w]);
    int prefix;
    Scan().exclusive_scan(count, prefix, 0, scan);
    if (threadIdx.x == 255)
        meta[512] = (prefix + count) * 4;
    for (int w = first; w < end; ++w) {
        unsigned present = group_bits[w];
        while (present) {
            int g = w * 32 + __ffs(present) - 1;
            for (int k = 0; k < 4; ++k) {
                int token = 4 * g + k;
                ids[prefix * 4 + k] = token;
                rev[token] = prefix * 4 + k;
            }
            ++prefix;
            present &= present - 1;
        }
    }
    for (int r = threadIdx.x; r < rows; r += blockDim.x) {
        int pos = pos0 + r, base = pos & ~3;
        for (int k = base; k <= pos; ++k)
            atomicOr(bits + (k >> 5), 1u << (k & 31));
    }
}
__global__ void sparse_tail_emit(const unsigned *bits, int *ids, int *rev, int *meta, int lo, int hi, bool dedup,
                                 int start, int row_offset, int rows, const Hy4StepParams *params) {
    int pos = step_at(start, params) + row_offset;
    if (dedup)
        lo = pos & ~3, hi = pos + rows;
    else
        lo = pos & ~3, hi = lo + (pos & 3) + 1;
    int i = blockIdx.x * blockDim.x + threadIdx.x, token = lo + i;
    if (token >= hi)
        return;
    if ((!dedup || rev[token] < 0) && (bits[token >> 5] & (1u << (token & 31)))) {
        int p = meta[512] + (dedup ? atomicAdd(meta + 513, 1) : i);
        if (!dedup && i == 0)
            meta[513] = hi - lo;
        ids[p] = token;
        rev[token] = p;
    }
}
__global__ void draft_extra_ids(int start, int rows, int *ids, int *rev, int *meta,
                                const Hy4StepParams *params) {
    int i = threadIdx.x;
    start = step_at(start, params);
    if (i < rows) {
        int p = meta[512] + meta[513] + i, token = start + i;
        ids[p] = token;
        rev[token] = p;
    }
    __syncthreads();
    if (!i)
        atomicAdd(meta + 513, rows);
}
__global__ void sparse_offsets(const int *groups, const int *rev, int *offsets, int rows, int stride, int pos0,
                               int *ids, int *meta, int max_ids, int row_offset, const Hy4StepParams *params) {
    int r = blockIdx.y, t = blockIdx.x * blockDim.x + threadIdx.x;
    pos0 = step_at(pos0, params) + row_offset;
    if (t < max_ids && t >= meta[512] + meta[513])
        ids[t] = -1;
    if (t >= stride)
        return;
    int token;
    if (t < 2044) {
        int g = groups[size_t(r) * 512 + t / 4];
        token = 4 * g + t % 4;
    } else {
        int pos = pos0 + r;
        token = (pos & ~3) + (t - 2044);
        if (token > pos)
            token = -1;
    }
    offsets[r * stride + t] = token >= 0 && rev[token] >= 0 ? rev[token] : -1;
}
__global__ void draft_offsets(int *offsets, int rows, int stride, int base_len, int *ids, int *meta, int max_ids,
                              int start, const Hy4StepParams *params) {
    int r = blockIdx.y, t = blockIdx.x * blockDim.x + threadIdx.x;
    int pos = step_at(start, params) - 1;
    base_len = 2044 + (pos & 3) + 1;
    if (t < max_ids && t >= meta[512] + meta[513])
        ids[t] = -1;
    if (t >= stride)
        return;
    offsets[r * stride + t] = t < base_len + r + 1 ? t : -1;
}
struct SparseLaneArgs {
    int block[32];
    int slot[32];
    int draft_last[32];
};
__global__ void collect_sparse_meta(const int *slots, const int *positions, int block_rows, int num_blocks,
                                    bool draft, int *sparse_blocks, int *sparse_slots, int *sparse_positions) {
    int t = threadIdx.x;
    bool active = t < num_blocks;
    int pos0 = active ? positions[t * block_rows] : 0;
    bool is_sparse = active && (draft ? pos0 > 2048 : pos0 + block_rows > 2048);
    unsigned mask = __ballot(is_sparse);
    if (is_sparse) {
        int k = __popc(mask & ((1u << t) - 1u));
        sparse_blocks[k] = t;
        sparse_slots[k] = slots[t * block_rows];
        for (int r = 0; r < block_rows; ++r)
            sparse_positions[k * block_rows + r] = pos0 + r;
    }
}
__global__ void gather_sparse_inputs(Mat hidden, Mat sparse_hidden, Mat qres, Mat sparse_qres,
                                     const int *block_map, int block_rows) {
    int out_r = blockIdx.y, col = blockIdx.x * 256 + threadIdx.x;
    int src_r = block_map[out_r / block_rows] * block_rows + out_r % block_rows;
    if (col < hidden.cols)
        sparse_hidden.p[size_t(out_r) * hidden.cols + col] = hidden.p[size_t(src_r) * hidden.cols + col];
    if (col < qres.cols)
        sparse_qres.p[size_t(out_r) * qres.cols + col] = qres.p[size_t(src_r) * qres.cols + col];
}
__global__ void index_scores_sparse_batch(Mat q, Mat weights, const uint16_t *all_groups, size_t group_slot_stride,
                                          int stride, float *scores, const int *sparse_slots,
                                          const int *sparse_positions, SparseLaneArgs args, int block_rows,
                                          const int32_t *page_table, int page_stride) {
    int r = blockIdx.y, pos = sparse_positions[r];
    int count = min(stride, pos / 4);
    int group = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (pos < 2048 || group >= count)
        return;
    int slot = sparse_slots ? sparse_slots[r / block_rows] : args.slot[r / block_rows];
    const int32_t *pt = page_table ? page_table + size_t(slot) * page_stride : nullptr;
    const uint16_t *groups = pt ? all_groups : all_groups + size_t(slot) * group_slot_stride;
    const uint16_t *group_ptr = index_group_const(groups, pt, group);
    float sum = 0;
    for (int h = 0; h < 16; ++h) {
        float dot = 0;
        for (int j = lane; j < 128; j += 32)
            dot += bv(q, r, h * 128 + j) * bf16_bits_to_f32(group_ptr[j]);
        for (int dist = 16; dist; dist /= 2)
            dot += __shfl_down(dot, dist, 32);
        sum += fmaxf(dot, 0) * bv(weights, r, h);
    }
    if (!lane)
        scores[size_t(r) * stride + group] = sum;
}
__global__ void index_topk_sparse_batch(const float *scores, int stride, int *selected_buf, int *top_slot_base,
                                        const int *sparse_slots, const int *sparse_positions,
                                        SparseLaneArgs args, int block_rows) {
    __shared__ uint32_t histogram[256], state[4];
    int r = blockIdx.x, pos = sparse_positions[r];
    if (pos < 2048)
        return;
    unsigned count = min(unsigned(stride), unsigned(pos / 4));
    int *dst = selected_buf ? selected_buf + size_t(r) * 511
                            : top_slot_base + (size_t(sparse_slots ? sparse_slots[r / block_rows] : args.slot[r / block_rows]) * 6 +
                                               r % block_rows) * 511;
    ::shelf::kernel::detail::radix_select<3>(scores + size_t(r) * stride, count, 511, histogram, state, dst, nullptr, 0);
}
__device__ void bitonic_sort_512(int *g, int t) {
    for (int k = 2; k <= 512; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            __syncthreads();
            int other = g[t ^ j], v = g[t];
            bool lower = (t & j) == 0;
            bool gt = lower ? v > other : other > v;
            bool lt = lower ? v < other : other < v;
            bool swap = (t & k) == 0 ? gt : lt;
            __syncthreads();
            g[t] = swap ? other : v;
        }
    }
    __syncthreads();
}
__global__ void build_sparse_indices_batch(const int *top_slot_base, const int *selected_buf,
                                           const int *sparse_slots, const int *sparse_positions,
                                           SparseLaneArgs args, const BatchGraphInput *batch_input,
                                           int *all_ids, int *id_counts, int *all_offsets, int block_rows,
                                           int id_stride, int stride, bool draft) {
    using Scan = rocprim::block_scan<int, 512>;
    __shared__ typename Scan::storage_type scan;
    __shared__ int g[6][512];
    __shared__ int all_g[4096];
    __shared__ int uniq[3072];
    __shared__ int uniq_count;
    __shared__ int tail_map[12];
    int k = blockIdx.x, t = threadIdx.x;
    int slot = sparse_slots ? sparse_slots[k] : args.slot[k];
    int pos0 = sparse_positions[k * block_rows];
    int *ids = all_ids + size_t(k) * id_stride;
    int *offsets = all_offsets + size_t(k * block_rows) * stride;
    if (draft) {
        int last = batch_input ? batch_input->step.lanes[args.block[k]].draft_last : args.draft_last[k];
        const int *top = top_slot_base + (size_t(slot) * 6 + last) * 511;
        g[0][t] = t < 511 ? top[t] : INT_MAX;
        __syncthreads();
        bitonic_sort_512(g[0], t);
        for (int i = t; i < 2044; i += 512)
            ids[i] = 4 * g[0][i >> 2] + (i & 3);
        int pos = pos0 - 1;
        int tail_len = (pos & 3) + 1;
        int base_len = 2044 + tail_len;
        if (t < tail_len)
            ids[2044 + t] = (pos & ~3) + t;
        if (t < block_rows)
            ids[base_len + t] = pos0 + t;
        if (!t)
            id_counts[k] = base_len + block_rows;
        for (int idx = t; idx < block_rows * stride; idx += 512) {
            int r = idx / stride, col = idx % stride;
            offsets[idx] = col < base_len + r + 1 ? col : -1;
        }
        return;
    }
    auto top_of = [&](int r) {
        return top_slot_base ? top_slot_base + (size_t(slot) * 6 + r) * 511
                             : selected_buf + size_t(k * block_rows + r) * 511;
    };
    if (block_rows == 1) {
        const int *top = top_of(0);
        g[0][t] = t < 511 ? top[t] : INT_MAX;
        __syncthreads();
        bitonic_sort_512(g[0], t);
        for (int i = t; i < 2044; i += 512)
            ids[i] = 4 * g[0][i >> 2] + (i & 3);
        int tail_len = (pos0 & 3) + 1;
        if (t < tail_len)
            ids[2044 + t] = (pos0 & ~3) + t;
        if (!t)
            id_counts[k] = 2044 + tail_len;
        for (int col = t; col < stride; col += 512)
            offsets[col] = col < 2044 + tail_len ? col : -1;
        return;
    }
    if (pos0 < 2048) {
        int total_ids = pos0 + block_rows;
        for (int i = t; i < total_ids; i += 512)
            ids[i] = i;
        if (!t)
            id_counts[k] = total_ids;
        for (int r = 0; r < block_rows; ++r) {
            if (pos0 + r >= 2048) {
                const int *top = top_of(r);
                g[r][t] = t < 511 ? top[t] : INT_MAX;
                __syncthreads();
                bitonic_sort_512(g[r], t);
            }
        }
        for (int idx = t; idx < block_rows * stride; idx += 512) {
            int r = idx / stride, col = idx % stride, pos_r = pos0 + r;
            if (pos_r < 2048) {
                offsets[idx] = col <= pos_r ? col : -1;
            } else if (col < 2044) {
                offsets[idx] = 4 * g[r][col >> 2] + (col & 3);
            } else {
                int tok = (pos_r & ~3) + (col - 2044);
                offsets[idx] = tok <= pos_r ? tok : -1;
            }
        }
        return;
    }
    for (int r = 0; r < block_rows; ++r) {
        const int *top = top_of(r);
        g[r][t] = t < 511 ? top[t] : INT_MAX;
        __syncthreads();
        bitonic_sort_512(g[r], t);
    }
    for (int idx = t; idx < 4096; idx += 512) {
        int r = idx / 511, c = idx % 511;
        all_g[idx] = r < block_rows ? g[r][c] : INT_MAX;
    }
    __syncthreads();
    for (int len = 2; len <= 4096; len <<= 1) {
        for (int j = len >> 1; j > 0; j >>= 1) {
            for (int p = t; p < 2048; p += 512) {
                int a = ((p & ~(j - 1)) << 1) | (p & (j - 1));
                int b = a | j;
                int va = all_g[a], vb = all_g[b];
                if ((va > vb) == ((a & len) == 0)) {
                    all_g[a] = vb;
                    all_g[b] = va;
                }
            }
            __syncthreads();
        }
    }
    const int total_raw = block_rows * 511;
    int my_cnt = 0;
    bool keep[8]{};
    for (int i = 0; i < 8; ++i) {
        int idx = t * 8 + i;
        if (idx < total_raw) {
            keep[i] = idx == 0 || all_g[idx] != all_g[idx - 1];
            if (keep[i])
                ++my_cnt;
        }
    }
    int prefix;
    Scan().exclusive_scan(my_cnt, prefix, 0, scan);
    for (int i = 0; i < 8; ++i)
        if (keep[i])
            uniq[prefix++] = all_g[t * 8 + i];
    if (t == 511)
        uniq_count = prefix;
    __syncthreads();
    int U = uniq_count;
    for (int i = t; i < U; i += 512) {
        int grp = uniq[i];
        for (int c = 0; c < 4; ++c)
            ids[i * 4 + c] = 4 * grp + c;
    }
    int tail_lo = pos0 & ~3, tail_count = pos0 + block_rows - tail_lo;
    if (!t) {
        int next_id = 4 * U;
        for (int j = 0; j < tail_count; ++j) {
            int tok = tail_lo + j, grp = tok >> 2;
            int lo = 0, hi = U - 1, found = -1;
            while (lo <= hi) {
                int mid = (lo + hi) >> 1;
                if (uniq[mid] == grp) {
                    found = mid;
                    break;
                }
                if (uniq[mid] < grp)
                    lo = mid + 1;
                else
                    hi = mid - 1;
            }
            if (found >= 0) {
                tail_map[j] = found * 4 + (tok & 3);
            } else {
                ids[next_id] = tok;
                tail_map[j] = next_id++;
            }
        }
        id_counts[k] = next_id;
    }
    __syncthreads();
    for (int idx = t; idx < block_rows * stride; idx += 512) {
        int r = idx / stride, col = idx % stride;
        if (col < 2044) {
            int grp = g[r][col >> 2];
            int lo = 0, hi = U - 1;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (uniq[mid] < grp)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            offsets[idx] = lo * 4 + (col & 3);
        } else {
            int pos_r = pos0 + r;
            int tok = (pos_r & ~3) + (col - 2044);
            offsets[idx] = tok <= pos_r ? tail_map[tok - tail_lo] : -1;
        }
    }
}
__global__ void unpack_quant_sparse_batch(const uint8_t *cache, const uint8_t *fresh, const int *all_ids,
                                          const int *id_counts, const int *sparse_slots,
                                          const int *sparse_positions, const int *sparse_blocks,
                                          SparseLaneArgs args, int cache_slots, int block_rows, int id_stride,
                                          const uint16_t *suh, _Float16 *all_out,
                                          const int32_t *page_table, int page_stride) {
    __shared__ float scratch[256];
    __shared__ uint16_t decoded[256];
    int sub = blockIdx.x, t = blockIdx.y, k = blockIdx.z, c = threadIdx.x;
    if (t >= id_counts[k])
        return;
    int slot = sparse_slots ? sparse_slots[k] : args.slot[k];
    int orig_block = sparse_blocks ? sparse_blocks[k] : k;
    int start = sparse_positions[k * block_rows];
    int id = all_ids[size_t(k) * id_stride + t];
    if (id < 0) {
        if (c < 64) {
            int col = sub * 256 + (c / 32) * 128 + (c % 32) * 4;
            for (int i = 0; i < 4; ++i)
                all_out[(size_t(k) * id_stride + t) * 512 + col + i] = 0;
        }
        return;
    }
    const int32_t *pt = page_table ? page_table + size_t(slot) * page_stride : nullptr;
    const uint8_t *rec = id >= start ? fresh + size_t(orig_block * block_rows + (id - start)) * 294
                                     : (pt ? kv_record_const(cache, pt, id, cache_slots)
                                           : cache + (size_t(slot) * cache_slots + size_t(id % cache_slots)) * 294);
    const uint8_t *src = rec + sub * 130;
    int code = (src[c / 2] >> (4 * (c % 2))) & 15;
    float scale = __half2float(__ushort_as_half(src[128] | (src[129] << 8)));
    float value = wht(levels[code] * scale, scratch, 256) * sign_of(c, 29);
    decoded[c] = f32_to_bf16_rne_device(value);
    __syncthreads();
    if (c < 64) {
        int col = sub * 256 + (c / 32) * 128 + (c % 32) * 4;
        float v[4], peak = 0;
        for (int i = 0; i < 4; ++i)
            v[i] = bf16_bits_to_f32(decoded[(c / 32) * 128 + (c % 32) * 4 + i]) * f16_bits_to_f32(suh[col + i]);
        fwht128(v[0], v[1], v[2], v[3], c % 32);
        for (float &f : v) {
            f *= kInvSqrt128;
            peak = fmaxf(peak, fabsf(f));
        }
        for (int mask = 16; mask; mask /= 2)
            peak = fmaxf(peak, __shfl_xor(peak, mask, 32));
        float quant_scale = peak / 127;
        for (int i = 0; i < 4; ++i)
            all_out[(size_t(k) * id_stride + t) * 512 + col + i] =
                quant_scale ? rintf(v[i] / quant_scale) * quant_scale : 0;
    }
}
__global__ void unpack_rot_sparse_batch(const uint8_t *cache, const uint8_t *fresh, const int *all_ids,
                                        const int *id_counts, const int *sparse_slots,
                                        const int *sparse_positions, const int *sparse_blocks,
                                        SparseLaneArgs args, int cache_slots, int block_rows, int id_stride,
                                        uint16_t *all_out, const int32_t *page_table, int page_stride) {
    __shared__ float scratch[64];
    int t = blockIdx.y, k = blockIdx.z, c = threadIdx.x;
    if (t >= id_counts[k])
        return;
    int slot = sparse_slots ? sparse_slots[k] : args.slot[k];
    int orig_block = sparse_blocks ? sparse_blocks[k] : k;
    int start = sparse_positions[k * block_rows];
    int id = all_ids[size_t(k) * id_stride + t];
    if (id < 0) {
        all_out[(size_t(k) * id_stride + t) * 64 + c] = 0;
        return;
    }
    const int32_t *pt = page_table ? page_table + size_t(slot) * page_stride : nullptr;
    const uint8_t *rec = id >= start ? fresh + size_t(orig_block * block_rows + (id - start)) * 294
                                     : (pt ? kv_record_const(cache, pt, id, cache_slots)
                                           : cache + (size_t(slot) * cache_slots + size_t(id % cache_slots)) * 294);
    const uint8_t *src = rec + 260;
    int code = (src[c / 2] >> (4 * (c % 2))) & 15;
    float scale = __half2float(__ushort_as_half(src[32] | (src[33] << 8)));
    float value = wht(levels[code] * scale, scratch, 64) * sign_of(c, 11);
    all_out[(size_t(k) * id_stride + t) * 64 + c] = f32_to_bf16_rne_device(value);
}
Selection select(Ops &op, const std::string &p, int layer, int start, Mat x, Mat qres, bool index_ready) {
    int end = start + x.rows;
    int bucket = op.max_kv_len > 0 ? op.max_kv_len : std::max(2048, (end + 63) / 64 * 64);
    Selection selected(x.rows);
    bool sliding = layer != 0 && layer % 4 != 1;
    int begin = sliding ? x.rows : std::min(x.rows, std::max(0, 2048 - start));
    if (!sliding) {
        const int32_t *page_table = op.gpu.page_table[op.rank]
            ? op.gpu.page_table[op.rank] + size_t(op.slot) * op.gpu.kv_page_stride
            : nullptr;
        auto groups = page_table ? op.gpu.index_keys[op.rank][layer]
                                 : op.gpu.index_keys[op.rank][layer] + op.slot * (op.gpu.tokens / 4 + 12) * 128;
        auto pending = op.gpu.index_pending[op.rank][layer]
            ? op.gpu.index_pending[op.rank][layer] + size_t(op.slot) * 12 * 128
            : groups + (op.gpu.tokens / 4) * 128;
        if (!index_ready) {
            Mat k = op.norm(op.linear(p + "token_key_projection", x), p + "token_key_norm.weight");
            int first = start / 4, ng = end / 4 - first;
            int pool_blocks = op.params ? 2 : ng;
            if (pool_blocks) {
                pool<<<pool_blocks, 128, 0, op.stream>>>(k, groups, first, start, op.param(p + "key_value_projection.weight"),
                                                op.param(p + "key_gate_projection.weight"),
                                                op.param(p + "intra_group_ape"), op.param(p + "key_norm.weight"),
                                                pending, op.params, x.rows, page_table);
                checked(hipGetLastError());
            }
            save_pending<<<std::min(k.rows, 12), 128, 0, op.stream>>>(k, start, pending, op.params);
            checked(hipGetLastError());
        }
        if (end > 2048) {
            Mat q = op.rope(op.linear(p + "query_projection", qres), 16, start, 1e7f);
            Mat w = op.linear(p + "head_weight_projection", x);
            int stride = std::max(512, (bucket - 1) / 4);
            int batch = std::min(16, x.rows - begin);
            auto buffer = static_cast<float *>(op.arena.next(index_workspace_floats(batch, stride) * sizeof(float)));
            if (!op.gpu.sparse_top[op.rank][layer]) {
                size_t rows = op.gpu.prefill_rows() + (op.gpu.batch_draft ? 6 * op.gpu.max_slots : 0);
                op.gpu.sparse_top[op.rank][layer] = static_cast<int *>(op.reserve_persistent(rows * 511 * sizeof(int)));
            }
            auto top = op.gpu.sparse_top[op.rank][layer];
            op.sparse_last_layer[layer] = x.rows - 1;
            unsigned *shared_peak = nullptr;
            if (op.gpu.index_wmma && !op.params && q.rows > 6 && end > 4096) {
                shared_peak = static_cast<unsigned *>(op.arena.next(sizeof(unsigned)));
                checked(hipMemsetAsync(shared_peak, 0, sizeof(unsigned), op.stream));
                int elements = std::min(stride, (end - 1) / 4) * 128;
                index_key_peak<<<std::min(256, (elements + 255) / 256), 256, 0, op.stream>>>(
                    groups, shared_peak, elements, page_table);
            }
            for (int row = begin; row < x.rows; row += batch) {
                int rows = std::min(batch, x.rows - row);
                index_select(q, w, groups, row, rows, start, stride, buffer, top + size_t(row) * 511,
                             op.params, op.gpu.index_wmma && !op.params && q.rows > 6 && start + row >= 4096,
                             op.stream, shared_peak, page_table);
            }
        }
    }
    for (int i = 0; i < begin; ++i) {
        int pos = start + i, from = sliding ? std::max(0, pos - 511) : 0;
        selected[i].resize(pos - from + 1);
        std::iota(selected[i].begin(), selected[i].end(), from);
    }
    return selected;
}
struct DsaLayerPtrs {
    uint8_t *kv[12];
    uint16_t *idx[12];
};
__global__ void copy_dsa_page_prefix_kernel(DsaLayerPtrs ptrs, int32_t dst_page, int32_t src_page, int tokens) {
    int l = blockIdx.y;
    size_t kv_bytes = size_t(tokens) * 294;
    uint8_t *dst_kv = ptrs.kv[l] + size_t(dst_page) * kv_page_tokens * 294;
    const uint8_t *src_kv = ptrs.kv[l] + size_t(src_page) * kv_page_tokens * 294;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < kv_bytes; i += size_t(gridDim.x) * blockDim.x)
        dst_kv[i] = src_kv[i];
    size_t idx_elems = size_t(tokens / 4) * 128;
    uint16_t *dst_idx = ptrs.idx[l] + size_t(dst_page) * kv_page_groups * 128;
    const uint16_t *src_idx = ptrs.idx[l] + size_t(src_page) * kv_page_groups * 128;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < idx_elems; i += size_t(gridDim.x) * blockDim.x)
        dst_idx[i] = src_idx[i];
}
}
void copy_dsa_page_prefix(Residency &gpu, int rank, int32_t dst_page, int32_t src_page, int tokens, hipStream_t stream) {
    if (tokens <= 0 || dst_page < 0 || src_page < 0)
        return;
    DsaLayerPtrs ptrs{};
    int idx = 0;
    for (int layer = 0; layer < 42; ++layer) {
        if (Residency::is_sparse_layer(layer)) {
            ptrs.kv[idx] = gpu.cache[rank][layer];
            ptrs.idx[idx] = gpu.index_keys[rank][layer];
            ++idx;
        }
    }
    size_t kv_bytes = size_t(tokens) * 294;
    unsigned blocks_x = unsigned(std::min<size_t>(256, (kv_bytes + 255) / 256));
    copy_dsa_page_prefix_kernel<<<dim3(blocks_x, 12), 256, 0, stream>>>(ptrs, dst_page, src_page, tokens);
    checked(hipGetLastError());
}
void index_select(Mat q, Mat weights, const uint16_t *groups, int first, int rows, int start, int stride,
                  float *scores, int *selected, const Hy4StepParams *params, bool wmma, hipStream_t stream,
                  const unsigned *shared_peak, const int32_t *page_table) {
    if (wmma) {
        auto bounds = scores + size_t(rows) * stride;
        auto peak = shared_peak;
        if (!peak) {
            auto local_peak = reinterpret_cast<unsigned *>(bounds + 2 * rows);
            checked(hipMemsetAsync(local_peak, 0, sizeof(unsigned), stream));
            int elements = std::min(stride, (start + first + rows - 1) / 4) * 128;
            index_key_peak<<<std::min(256, (elements + 255) / 256), 256, 0, stream>>>(
                groups, local_peak, elements, page_table);
            peak = local_peak;
        }
        index_scores_wmma<<<dim3((stride + 63) / 64, rows), 128, 0, stream>>>(
            q, weights, groups, first, start, stride, scores, page_table);
        index_bounds<<<rows, 256, 0, stream>>>(q, weights, scores, first, start, stride, bounds, peak);
        index_scores_refine<<<dim3((stride + 255) / 256, rows), 256, 0, stream>>>(
            q, weights, groups, first, start, stride, scores, bounds, page_table);
    } else {
        index_scores<<<dim3((stride + 3) / 4, rows), 128, 0, stream>>>(
            q, weights, groups, first, start, stride, scores, params, nullptr, page_table);
    }
    checked(hipGetLastError());
    index_topk<<<rows, 256, 0, stream>>>(scores, first, start, stride, selected, params);
    checked(hipGetLastError());
}
Mat attention_inner(Ops &op, const std::string &p, int layer, int start, Mat hidden, Mat qres, Mat q, Mat lat,
                    Mat rot, Mat gate, Selection &selected, bool draft, int heads, bool sliding, bool index_ready = false) {
    int bucket = op.max_kv_len > 0 ? op.max_kv_len : std::max(2048, (start + hidden.rows + 63) / 64 * 64);
    Param sinks = op.param(p + "learnable_sink_param");
    if (op.gpu.tp_size > 1 && !draft) {
        const int first = (op.rank % op.gpu.tp_size) * heads;
        sinks.p = sinks.type == io::SType::F32
                      ? static_cast<const void *>(static_cast<const float *>(sinks.p) + first)
                      : static_cast<const void *>(static_cast<const uint16_t *>(sinks.p) + first);
    }
    auto fresh = static_cast<uint8_t *>(op.arena.next(size_t(hidden.rows) * 294));
    pack_kv_combined<<<dim3(3, hidden.rows), 256, 0, op.stream>>>(lat, rot, fresh);
    checked(hipGetLastError());
    if (!draft)
        selected = select(op, p + "compressed_indexer.", layer, start, hidden, qres, index_ready);
    Mat out = op.mat(hidden.rows, heads * 256);
    int slots = sliding ? std::min(op.gpu.tokens, size_t(518)) : op.gpu.tokens;
    const int32_t *page_table = (!sliding && op.gpu.page_table[op.rank])
        ? op.gpu.page_table[op.rank] + size_t(op.slot) * op.gpu.kv_page_stride
        : nullptr;
    auto cache = op.gpu.cache[op.rank][layer] + (sliding || !page_table ? size_t(op.slot) * slots * 294 : 0);
    bool contiguous = sliding || (draft ? start <= 2048 : start + hidden.rows <= 2048);
    int begin = std::min(hidden.rows, std::max(0, 2048 - start));
    bool batched_projection = !contiguous && !draft && start >= 2048 && start + hidden.rows > 4096 && hidden.rows > 16 && hidden.rows <= 1024;
    _Float16 *batch_query = nullptr, *batch_values = nullptr;
    if (batched_projection) {
        size_t bytes = size_t(hidden.rows) * heads * 512 * sizeof(_Float16);
        batch_query = static_cast<_Float16 *>(op.arena.next(bytes));
        batch_values = static_cast<_Float16 *>(op.arena.next(bytes));
        checked(shelf_hy4_mla_project_query(q.p, op.gpu.folded[op.rank].at(p + "kv_b_proj"), batch_query, hidden.rows, heads, op.stream));
    }
    for (int i = 0; i < hidden.rows;) {
        size_t save = op.arena.cursor;
        bool chunk_contiguous = contiguous;
        int n = std::min(1024, hidden.rows - i);
        if (!contiguous && !draft) {
            if (i < begin) {
                chunk_contiguous = true;
                n = std::min(n, begin - i);
            } else if (start + hidden.rows > 4096)
                n = std::min(n, 64);
        }
        int stride = 0;
        for (int r = i; r < i + n && !selected.empty(); ++r)
            stride = std::max(stride, int(selected[r].size()));
        bool dynamic_sizes = !op.params && chunk_contiguous && !draft;
        if (!dynamic_sizes)
            stride = chunk_contiguous ? (sliding ? 512 : 2048) : 2048;
        int first = chunk_contiguous && !selected.empty() ? selected[i][0] : 0;
        int count = chunk_contiguous && !selected.empty() ? selected[i + n - 1].back() + 1 - first : 0;
        int ids_count = dynamic_sizes ? count
                                      : (chunk_contiguous ? (sliding ? 518 : 2048) : (2044 * n + 4 * n + 8));
        int *ids_gpu = nullptr, *offsets_gpu = nullptr, *groups = nullptr;
        int *mask = nullptr, *meta = nullptr, *rev = nullptr;
        unsigned *bits = nullptr;
        if (draft && !chunk_contiguous && op.gpu.sparse_top[op.rank][layer]) {
            int last = op.slot_draft ? op.gpu.sparse_last[op.slot][layer] : op.sparse_last_layer[layer];
            int pos = start - 1;
            int tail_len = (pos & 3) + 1, tail_base = pos & ~3;
            int base_len = 2044 + tail_len;
            stride = (op.params ? 2048 : base_len) + n;
            int width = op.params ? bucket / 4 + 2 : (pos + 3) / 4 + 2;
            ids_count = std::min(4 * width, 2044) + 4 + n + 8;
            int capacity = op.params ? bucket : start + n;
            int words = (capacity + 31) / 32, rev_count = capacity + 8;
            groups = static_cast<int *>(op.arena.next(size_t(512) * 4));
            mask = static_cast<int *>(op.arena.next(size_t((width + 31) / 32) * 4));
            bits = static_cast<unsigned *>(op.arena.next(size_t(words) * 4));
            meta = static_cast<int *>(op.arena.next(size_t(516) * 4));
            rev = static_cast<int *>(op.arena.next(size_t(rev_count) * 4));
            ids_gpu = static_cast<int *>(op.arena.next(size_t(ids_count) * 4));
            offsets_gpu = static_cast<int *>(op.arena.next(size_t(n) * stride * 4));
            sparse_clear<<<(std::max({(width + 31) / 32, words, rev_count, 516}) + 255) / 256, 256, 0, op.stream>>>(
                mask, (width + 31) / 32, bits, words, meta, rev, rev_count);
            auto top = op.slot_draft ? op.gpu.sparse_top_slot(layer, op.slot) : op.gpu.sparse_top[op.rank][layer];
            sort_top<<<1, 512, 0, op.stream>>>(top, groups, mask, last, 1, width, op.params);
            sparse_group_ids<<<1, 256, 0, op.stream>>>(mask, 1, width, meta, ids_gpu, rev, bits,
                                                                       start, -1, op.params);
            sparse_tail_emit<<<(tail_len + 255) / 256, 256, 0, op.stream>>>(bits, ids_gpu, rev, meta, tail_base,
                                                                            tail_base + tail_len, false, start, -1, n,
                                                                            op.params);
            draft_extra_ids<<<1, 64, 0, op.stream>>>(start, n, ids_gpu, rev, meta, op.params);
            draft_offsets<<<dim3((std::max(stride, ids_count) + 255) / 256, n), 256, 0, op.stream>>>(
                offsets_gpu, n, stride, base_len, ids_gpu, meta, ids_count, start, op.params);
        } else if (draft && op.params) {
            ids_count = stride = sliding ? 518 : 2054;
            ids_gpu = static_cast<int *>(op.arena.next(size_t(stride) * sizeof(int)));
            offsets_gpu = static_cast<int *>(op.arena.next(size_t(n) * stride * sizeof(int)));
            draft_contiguous_offsets<<<dim3((stride + 255) / 256, n), 256, 0, op.stream>>>(
                ids_gpu, offsets_gpu, stride, sliding, op.params);
        } else if (draft) {
            stride = 0;
            for (int r = i; r < i + n; ++r)
                stride = std::max(stride, int(selected[r].size()));
            std::vector<int> ids, offsets(n * stride, -1);
            ids = selected[i + n - 1];
            for (int r = 0; r < n; ++r)
                std::iota(offsets.begin() + r * stride, offsets.begin() + r * stride + selected[i + r].size(), 0);
            ids_gpu = op.indices(ids);
            offsets_gpu = op.indices(offsets);
            ids_count = int(ids.size());
        } else if (chunk_contiguous) {
            offsets_gpu = static_cast<int *>(op.arena.next(size_t(n) * stride * 4));
            contiguous_offsets<<<dim3((stride + 255) / 256, n), 256, 0, op.stream>>>(
                offsets_gpu, n, stride, start, sliding, first, i, op.params);
        } else {
            int rows = n, pos0 = start + i;
            stride = 2048;
            int width = bucket / 4 + 8;
            ids_count = std::min(4 * width, 2044 * rows) + 4 * rows + 8;
            int words = (bucket + 31) / 32;
            int rev_count = bucket + 8;
            groups = static_cast<int *>(op.arena.next(size_t(rows) * 512 * 4));
            mask = static_cast<int *>(op.arena.next(size_t((width + 31) / 32) * 4));
            bits = static_cast<unsigned *>(op.arena.next(size_t(words) * 4));
            meta = static_cast<int *>(op.arena.next(size_t(516) * 4));
            rev = static_cast<int *>(op.arena.next(size_t(rev_count) * 4));
            ids_gpu = static_cast<int *>(op.arena.next(size_t(ids_count) * 4));
            offsets_gpu = static_cast<int *>(op.arena.next(size_t(rows) * stride * 4));
            int mask_count = (width + 31) / 32;
            sparse_clear<<<(std::max({mask_count, words, rev_count, 516}) + 255) / 256, 256, 0, op.stream>>>(
                mask, mask_count, bits, words, meta, rev, rev_count);
            sort_top<<<rows, 512, 0, op.stream>>>(op.gpu.sparse_top[op.rank][layer], groups, mask, i, rows, width);
            sparse_group_ids<<<1, 256, 0, op.stream>>>(mask, rows, width, meta, ids_gpu, rev, bits,
                                                                       start, i, op.params);
            sparse_tail_emit<<<(rows + 3 + 255) / 256, 256, 0, op.stream>>>(
                bits, ids_gpu, rev, meta, 2048, pos0 + rows, true, start, i, rows, op.params);
            sparse_offsets<<<dim3((std::max(stride, ids_count) + 255) / 256, rows), 256, 0, op.stream>>>(
                groups, rev, offsets_gpu, rows, stride, start, ids_gpu, meta, ids_count, i, op.params);
        }
        checked(hipGetLastError());
        bool fused = true;
        Mat a{nullptr, ids_count, 512};
        Mat b = op.mat(ids_count, 64);
        _Float16 *prequant = static_cast<_Float16 *>(op.arena.next(size_t(ids_count) * 512 * sizeof(_Float16)));
        unsigned id_slices = (ids_count + 65534) / 65535;
        unsigned id_rows = (ids_count + id_slices - 1) / id_slices;
        unpack_quant<<<dim3(2, id_rows, id_slices), 256, 0, op.stream>>>(
            cache, fresh, ids_gpu, first, start, slots,
            static_cast<const uint16_t *>(op.param(p + "kv_b_proj.suh").p), prequant, op.params, sliding, i, n,
            meta, ids_count, page_table);
        checked(hipGetLastError());
        unpack<64, 11><<<dim3(1, id_rows, id_slices), 64, 0, op.stream>>>(cache, fresh, ids_gpu, first, start, slots, b, 260,
                                                                  op.params, sliding, i, n, meta, page_table);
        checked(hipGetLastError());
        attend_absorbed(op, p + "kv_b_proj", a, b, q.row(i, n), gate.row(i, n), offsets_gpu, stride,
                        sinks, out.row(i, n), chunk_contiguous, sliding, i, n, meta, prequant,
                        batch_query ? batch_query + size_t(i) * heads * 512 : nullptr,
                        batch_values ? batch_values + size_t(i) * heads * 512 : nullptr);
        op.arena.cursor = save;
        i += n;
    }
    if (batched_projection)
        checked(shelf_hy4_mla_project_value(batch_values, op.gpu.folded[op.rank].at(p + "kv_b_proj"), gate.p, out.p, hidden.rows, heads, op.stream));
    if (!draft) {
        save_records<<<(std::min(hidden.rows, slots) * 294 + 255) / 256, 256, 0, op.stream>>>(
            cache, fresh, hidden.rows, start, slots, op.params, page_table);
        checked(hipGetLastError());
    }
    return out;
}
Mat attention(Ops &op, const std::string &prefix, int layer, int start, Mat hidden, Selection &selected,
              bool draft) {
    std::string p = prefix + "self_attn.";
    bool sliding = layer != 0 && layer % 4 != 1;
    const auto *projection = op.gpu.model.shape_of(p + "q_b_proj", op.rank);
    int heads = projection ? projection->at(1) / 256 : (sliding ? 64 : 32);
    float theta = sliding ? 1e4f : 1e7f;
    Mat qres = op.norm(op.linear(p + "q_a_proj", hidden), p + "q_a_layernorm.weight");
    Mat q = op.rope(op.linear(p + "q_b_proj", qres), heads, start, theta);
    Mat ka = op.linear(p + "kv_a_proj_with_mqa", hidden);
    Mat lat = op.norm_from(ka, 0, 512, p + "kv_a_layernorm.weight");
    Mat rot = op.rope_from(ka, 512, 64, 1, start, theta);
    Mat gate = op.linear(p + "linear_gate", hidden);
    Mat out = attention_inner(op, p, layer, start, hidden, qres, q, lat, rot, gate, selected, draft, heads, sliding);
    return op.norm(op.linear(p + "o_proj", out), p + "post_proj_norm.weight");
}
static void attend_batch_blocks(Ops &op, const std::string &p, int layer, bool sliding, bool draft,
                                std::span<const int> slots, std::span<const int> positions,
                                const int *device_slots, const int *device_positions, int block_rows,
                                Mat hidden, Mat qres, Mat q, Mat gate, uint8_t *fresh, Mat out) {
    size_t mark = op.arena.cursor;
    int cache_slots = sliding ? std::min(op.gpu.tokens, size_t(518)) : op.gpu.tokens;
    const int32_t *batch_pt = !sliding ? op.gpu.page_table[op.rank] : nullptr;
    const int batch_pt_stride = int(op.gpu.kv_page_stride);
    auto sparse = [&](int row) {
        return !sliding && (draft ? positions[row] > 2048 : positions[row] + block_rows > 2048);
    };
    std::vector<int> sparse_block_list;
    for (int first = 0, b = 0; first < hidden.rows; first += block_rows, ++b)
        if (sparse(first))
            sparse_block_list.push_back(b);
    for (int first = 0; first < hidden.rows;) {
        if (sparse(first)) {
            first += block_rows;
            continue;
        }
        int last = first + block_rows;
        while (last < hidden.rows && !sparse(last)) last += block_rows;
        int rows = last - first;
        int stride = *std::max_element(positions.begin() + first, positions.begin() + last) + 1;
        if (op.capture) stride = std::min(op.max_kv_len, 2048 + block_rows);
        if (sliding) stride = std::min(stride, 511 + block_rows);
        int blocks = rows / block_rows;
        Mat a{nullptr, blocks * stride, 512};
        Mat b = op.mat(blocks * stride, 64);
        auto prequant = static_cast<_Float16 *>(op.arena.next(size_t(blocks) * stride * 512 * sizeof(_Float16)));
        unpack_batch_quant<<<dim3(2, stride, blocks), 256, 0, op.stream>>>(
            op.gpu.cache[op.rank][layer], fresh + size_t(first) * 294, device_slots + first, device_positions + first,
            cache_slots, stride, static_cast<const uint16_t *>(op.param(p + "kv_b_proj.suh").p), prequant,
            sliding, block_rows, batch_pt, batch_pt_stride);
        checked(hipGetLastError());
        unpack_batch<64, 11><<<dim3(1, stride, blocks), 64, 0, op.stream>>>(
            op.gpu.cache[op.rank][layer], fresh + size_t(first) * 294, device_slots + first, device_positions + first,
            cache_slots, stride, b, 260, sliding, block_rows, batch_pt, batch_pt_stride);
        checked(hipGetLastError());
        auto offsets = static_cast<int *>(op.arena.next(size_t(rows) * stride * sizeof(int)));
        batch_offsets<<<dim3((stride + 255) / 256, rows), 256, 0, op.stream>>>(
            offsets, device_positions + first, stride, sliding, block_rows);
        checked(hipGetLastError());
        if (block_rows > 1)
            attend_absorbed(op, p + "kv_b_proj", a, b, q.row(first, rows), gate.row(first, rows), offsets, stride,
                            op.param(p + "learnable_sink_param"), out.row(first, rows), true, sliding, 0, 0,
                            nullptr, prequant);
        else
            attend_absorbed_batch(op, p + "kv_b_proj", a, b, q.row(first, rows), gate.row(first, rows), offsets,
                                  stride, device_positions + first, op.param(p + "learnable_sink_param"),
                                  out.row(first, rows), sliding, prequant);
        op.arena.cursor = mark;
        first = last;
    }
    if (!sparse_block_list.empty()) {
        int K = int(sparse_block_list.size()), num_blocks = hidden.rows / block_rows;
        bool all_sparse = K == num_blocks;
        int sparse_rows = K * block_rows;
        SparseLaneArgs args{};
        int max_end = 0;
        for (int k = 0; k < K; ++k) {
            int b = sparse_block_list[k], row = b * block_rows;
            args.block[k] = b;
            args.slot[k] = slots[row];
            args.draft_last[k] = draft ? op.gpu.sparse_last[slots[row]][layer] : -1;
            max_end = std::max(max_end, positions[row] + block_rows);
        }
        int bucket = op.max_kv_len > 0 ? op.max_kv_len : std::max(2048, (max_end + 63) / 64 * 64);
        int *sparse_blocks_dev = nullptr, *sparse_slots_dev = nullptr;
        const int *sparse_positions_dev = device_positions;
        if (!all_sparse) {
            sparse_blocks_dev = static_cast<int *>(op.arena.next(size_t(K * 2 + sparse_rows) * sizeof(int)));
            sparse_slots_dev = sparse_blocks_dev + K;
            int *sparse_positions_mut = sparse_slots_dev + K;
            collect_sparse_meta<<<1, 32, 0, op.stream>>>(
                device_slots, device_positions, block_rows, num_blocks, draft,
                sparse_blocks_dev, sparse_slots_dev, sparse_positions_mut);
            checked(hipGetLastError());
            sparse_positions_dev = sparse_positions_mut;
        }
        if (((!draft && op.gpu.batch_draft && layer >= 25) || draft) && !op.gpu.sparse_top[op.rank][layer]) {
            size_t total_rows = op.gpu.prefill_rows() + (op.gpu.batch_draft ? 6 * op.gpu.max_slots : 0);
            op.gpu.sparse_top[op.rank][layer] = static_cast<int *>(op.reserve_persistent(total_rows * 511 * sizeof(int)));
        }
        int *top_slot_base_write =
            !draft && op.gpu.batch_draft && layer >= 25
                ? op.gpu.sparse_top[op.rank][layer] + op.gpu.prefill_rows() * 511
                : nullptr;
        const int *top_slot_base_read =
            draft || (!draft && op.gpu.batch_draft && layer >= 25)
                ? op.gpu.sparse_top[op.rank][layer] + op.gpu.prefill_rows() * 511
                : nullptr;
        int *selected_buf = nullptr;
        if (!draft) {
            Mat sparse_hidden = hidden, sparse_qres = qres;
            if (!all_sparse) {
                sparse_hidden = op.mat(sparse_rows, hidden.cols);
                sparse_qres = op.mat(sparse_rows, qres.cols);
                gather_sparse_inputs<<<dim3(hidden.cols / 256, sparse_rows), 256, 0, op.stream>>>(
                    hidden, sparse_hidden, qres, sparse_qres, sparse_blocks_dev, block_rows);
                checked(hipGetLastError());
            }
            std::string ip = p + "compressed_indexer.";
            Mat q_idx = op.rope_batch(op.linear(ip + "query_projection", sparse_qres), 16, 1e7f, sparse_positions_dev);
            Mat w_idx = op.linear(ip + "head_weight_projection", sparse_hidden);
            int idx_stride = std::max(512, (bucket - 1) / 4);
            auto idx_scores = static_cast<float *>(op.arena.next(size_t(sparse_rows) * idx_stride * sizeof(float)));
            if (!top_slot_base_write)
                selected_buf = static_cast<int *>(op.arena.next(size_t(sparse_rows) * 511 * sizeof(int)));
            size_t group_slot_stride = (op.gpu.tokens / 4 + 12) * 128;
            index_scores_sparse_batch<<<dim3((idx_stride + 3) / 4, sparse_rows), 128, 0, op.stream>>>(
                q_idx, w_idx, op.gpu.index_keys[op.rank][layer], group_slot_stride, idx_stride, idx_scores,
                sparse_slots_dev, sparse_positions_dev, args, block_rows, batch_pt, batch_pt_stride);
            checked(hipGetLastError());
            index_topk_sparse_batch<<<sparse_rows, 256, 0, op.stream>>>(
                idx_scores, idx_stride, selected_buf, top_slot_base_write, sparse_slots_dev, sparse_positions_dev,
                args, block_rows);
            checked(hipGetLastError());
        }
        int attn_stride = 2048 + (draft ? block_rows : 0);
        int id_stride = draft ? (2048 + block_rows + 8)
                              : (block_rows == 1 ? 2056 : std::min(4 * (bucket / 4 + 8), 2044 * block_rows) + 4 * block_rows + 8);
        auto all_ids = static_cast<int *>(op.arena.next(size_t(K) * id_stride * sizeof(int)));
        auto id_counts = static_cast<int *>(op.arena.next(size_t(K) * sizeof(int)));
        auto all_offsets = static_cast<int *>(op.arena.next(size_t(sparse_rows) * attn_stride * sizeof(int)));
        build_sparse_indices_batch<<<K, 512, 0, op.stream>>>(
            top_slot_base_read, selected_buf, sparse_slots_dev, sparse_positions_dev, args,
            op.capture ? op.gpu.batch_input : nullptr, all_ids, id_counts, all_offsets, block_rows, id_stride,
            attn_stride, draft);
        checked(hipGetLastError());
        auto prequant_sparse = static_cast<_Float16 *>(op.arena.next(size_t(K) * id_stride * 512 * sizeof(_Float16)));
        auto rotary_sparse = static_cast<uint16_t *>(op.arena.next(size_t(K) * id_stride * 64 * sizeof(uint16_t)));
        unpack_quant_sparse_batch<<<dim3(2, id_stride, K), 256, 0, op.stream>>>(
            op.gpu.cache[op.rank][layer], fresh, all_ids, id_counts, sparse_slots_dev, sparse_positions_dev,
            sparse_blocks_dev, args, cache_slots, block_rows, id_stride,
            static_cast<const uint16_t *>(op.param(p + "kv_b_proj.suh").p), prequant_sparse,
            batch_pt, batch_pt_stride);
        checked(hipGetLastError());
        unpack_rot_sparse_batch<<<dim3(1, id_stride, K), 64, 0, op.stream>>>(
            op.gpu.cache[op.rank][layer], fresh, all_ids, id_counts, sparse_slots_dev, sparse_positions_dev,
            sparse_blocks_dev, args, cache_slots, block_rows, id_stride, rotary_sparse,
            batch_pt, batch_pt_stride);
        checked(hipGetLastError());
        attend_absorbed_sparse_batch(op, p + "kv_b_proj", prequant_sparse, rotary_sparse, q, gate, all_offsets,
                                     attn_stride, id_stride, sparse_positions_dev, sparse_blocks_dev,
                                     op.param(p + "learnable_sink_param"), out, K, block_rows, draft);
        op.arena.cursor = mark;
    }
}
Mat attention_batch(Ops &op, const std::string &prefix, int layer, const std::vector<int> &slots,
                    const std::vector<int> &positions, const int *device_slots,
                    const int *device_positions, Mat hidden, int block_rows, bool draft,
                    std::span<const int> row_offsets, const int *row_starts) {
    if (slots.size() != size_t(hidden.rows) || positions.size() != slots.size())
        throw std::invalid_argument("attention のバッチ長が不正です");
    std::string p = prefix + "self_attn.";
    bool sliding = layer != 0 && layer % 4 != 1;
    int heads = sliding ? 64 : 32;
    float theta = sliding ? 1e4f : 1e7f;
    Mat qres = op.norm(op.linear(p + "q_a_proj", hidden), p + "q_a_layernorm.weight");
    Mat raw = op.linear(p + "q_b_proj", qres);
    Mat ka = op.linear(p + "kv_a_proj_with_mqa", hidden);
    Mat lat = op.norm_from(ka, 0, 512, p + "kv_a_layernorm.weight");
    Mat gate = op.linear(p + "linear_gate", hidden);
    Mat q = op.rope_batch(raw, heads, theta, device_positions);
    Mat rot = op.rope_from_batch(ka, 512, 64, 1, theta, device_positions);
    auto fresh = static_cast<uint8_t *>(op.arena.next(size_t(hidden.rows) * 294));
    pack_kv_combined<<<dim3(3, hidden.rows), 256, 0, op.stream>>>(lat, rot, fresh);
    checked(hipGetLastError());
    if (!sliding && !draft) {
        std::string ip = p + "compressed_indexer.";
        Mat k = op.norm(op.linear(ip + "token_key_projection", hidden), ip + "token_key_norm.weight");
        pool_batch<<<hidden.rows, 128, 0, op.stream>>>(k, op.gpu.index_keys[op.rank][layer],
            op.gpu.index_pending[op.rank][layer], device_slots, device_positions,
            op.param(ip + "key_value_projection.weight"), op.param(ip + "key_gate_projection.weight"),
            op.param(ip + "intra_group_ape"), op.param(ip + "key_norm.weight"), block_rows, row_starts,
            op.gpu.page_table[op.rank], int(op.gpu.kv_page_stride));
        checked(hipGetLastError());
        save_pending_batch<<<hidden.rows, 128, 0, op.stream>>>(k, op.gpu.index_pending[op.rank][layer],
                                                               device_slots, device_positions);
        checked(hipGetLastError());
    }
    Mat out = op.mat(hidden.rows, heads * 256);
    if (row_offsets.empty()) {
        attend_batch_blocks(op, p, layer, sliding, draft, slots, positions, device_slots, device_positions,
                            block_rows, hidden, qres, q, gate, fresh, out);
    } else {
        for (size_t lane = 0; lane + 1 < row_offsets.size();) {
            int first = row_offsets[lane];
            int width = row_offsets[lane + 1] - first;
            do {
                ++lane;
            } while (lane + 1 < row_offsets.size() && row_offsets[lane + 1] - row_offsets[lane] == width);
            int count = row_offsets[lane] - first;
            attend_batch_blocks(op, p, layer, sliding, draft, std::span(slots).subspan(first, count),
                                std::span(positions).subspan(first, count), device_slots + first,
                                device_positions + first, width, hidden.row(first, count), qres.row(first, count),
                                q.row(first, count), gate.row(first, count), fresh + size_t(first) * 294,
                                out.row(first, count));
        }
    }
    if (!draft) {
        int cache_slots = sliding ? std::min(op.gpu.tokens, size_t(518)) : op.gpu.tokens;
        const auto *batch_pt = sliding ? nullptr : op.gpu.page_table[op.rank];
        save_records_batch<<<(hidden.rows * 294 + 255) / 256, 256, 0, op.stream>>>(
            op.gpu.cache[op.rank][layer], fresh, device_slots, device_positions, cache_slots, hidden.rows,
            batch_pt, int(op.gpu.kv_page_stride));
        checked(hipGetLastError());
    }
    op.slot = 0;
    return op.norm(op.linear(p + "o_proj", out), p + "post_proj_norm.weight");
}
}
