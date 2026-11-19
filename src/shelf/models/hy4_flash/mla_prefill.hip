#include <shelf/models/hy4_flash/mla_prefill.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>

namespace {
using namespace shelf::kernel::exl3_detail::mul24;
using B8 = short __attribute__((ext_vector_type(8)));
using F8 = float __attribute__((ext_vector_type(8)));
__device__ __forceinline__ B8 fragment(const uint16_t *source) {
    union {
        uint64_t u[2];
        B8 b;
    } value;
    __builtin_memcpy(&value.u[0], source, 8);
    __builtin_memcpy(&value.u[1], source + 8, 8);
    return value.b;
}
__global__ void score_tiles(const uint16_t *q, const uint16_t *kv, const uint16_t *rot, float *scores, int rows,
                            int heads, int slots, int pitch, int first, int start, bool sliding) {
    int lane = threadIdx.x % 32, lo = lane % 16, hi = lane / 16, h = blockIdx.z;
    int kb = blockIdx.x * 64 + (threadIdx.x / 32) * 16, rb = blockIdx.y * 16;
    int slot = kb + lo, row = rb + lo;
    if (kb >= pitch)
        return;
    if (first + kb > start + rb + 15 || (sliding && first + kb + 15 <= start + rb - 512)) {
        if (row < rows)
            for (int j = 0; j < 8; ++j)
                scores[(size_t(row) * heads + h) * pitch + kb + hi * 8 + j] = -INFINITY;
        return;
    }
    F8 acc{};
    for (int d = 0; d < 256; d += 16) {
        int offset = d + hi * 4;
        B8 a{}, b{};
        if (slot < slots)
            a = d < 192 ? fragment(kv + (size_t(slot) * heads + h) * 448 + offset)
                        : fragment(rot + slot * 64 + offset - 192);
        if (row < rows)
            b = fragment(q + (size_t(row) * heads + h) * 256 + offset);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, acc);
    }
    if (row >= rows)
        return;
    int pos = start + row;
    for (int j = 0; j < 8; ++j) {
        int k = kb + hi * 8 + j;
        scores[(size_t(row) * heads + h) * pitch + k] =
            (k < slots && first + k <= pos && (!sliding || first + k > pos - 512))
                ? round_bf16(acc[j]) * 0.0625f
                : -INFINITY;
    }
}
template <bool Max> __device__ float reduce(float x, float *parts) {
    int lane = threadIdx.x % 32;
    for (int d = 16; d; d /= 2) {
        float other = __shfl_down(x, d, 32);
        x = Max ? fmaxf(x, other) : x + other;
    }
    if (!lane)
        parts[threadIdx.x / 32] = x;
    __syncthreads();
    x = threadIdx.x < 8 ? parts[threadIdx.x] : (Max ? -INFINITY : 0.f);
    for (int d = 16; d; d /= 2) {
        float other = __shfl_down(x, d, 32);
        x = Max ? fmaxf(x, other) : x + other;
    }
    if (!threadIdx.x)
        parts[0] = x;
    __syncthreads();
    float result = parts[0];
    __syncthreads();
    return result;
}
__global__ void normalize(float *scores, uint16_t *probs, const void *sinks, bool fp32, int pitch) {
    __shared__ float parts[8];
    int h = blockIdx.x, r = blockIdx.y, t = threadIdx.x;
    size_t base = (size_t(r) * gridDim.x + h) * pitch;
    float sink =
        fp32 ? static_cast<const float *>(sinks)[h] : bf16_bits_to_f32(static_cast<const uint16_t *>(sinks)[h]);
    float peak = sink;
    for (int i = t; i < pitch; i += 256)
        peak = fmaxf(peak, scores[base + i]);
    peak = reduce<true>(peak, parts);
    float sum = t ? 0.f : expf(sink - peak);
    for (int i = t; i < pitch; i += 256) {
        float e = expf(scores[base + i] - peak);
        scores[base + i] = e;
        sum += e;
    }
    sum = reduce<false>(sum, parts);
    for (int i = t; i < pitch; i += 256)
        probs[base + i] = f32_to_bf16_rne_device(scores[base + i] / sum);
}
__global__ void value_tiles(const uint16_t *probs, const uint16_t *kv, const uint16_t *gate, uint16_t *out,
                            int rows, int heads, int slots, int pitch, int first, int start, bool sliding) {
    int lane = threadIdx.x % 32, lo = lane % 16, hi = lane / 16, h = blockIdx.z;
    int cb = blockIdx.x * 64 + (threadIdx.x / 32) * 16, row = blockIdx.y * 16 + lo;
    F8 acc{};
    int rb = blockIdx.y * 16;
    int from = sliding ? max(0, start + rb - 511 - first) / 16 * 16 : 0;
    int end = min(pitch, (start + rb + 16 - first + 15) / 16 * 16);
    for (int base = from; base < end; base += 16) {
        B8 a{}, b{};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int k = base + hi * 4 + j % 4 + 8 * (j / 4);
            if (k < slots)
                a[j] = kv[(size_t(k) * heads + h) * 448 + 192 + cb + lo];
        }
        if (row < rows)
            b = fragment(probs + (size_t(row) * heads + h) * pitch + base + hi * 4);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, acc);
    }
    if (row >= rows)
        return;
    for (int j = 0; j < 8; ++j) {
        size_t i = (size_t(row) * heads + h) * 256 + cb + hi * 8 + j;
        out[i] = f32_to_bf16_rne_device(round_bf16(acc[j]) *
                                        round_bf16(1.f / (1.f + expf(-bf16_bits_to_f32(gate[i])))));
    }
}
}
hipError_t shelf_hy4_mla_prefill(const uint16_t *q, const uint16_t *kv, const uint16_t *rotary,
                                 const uint16_t *gate, const void *sink, bool sink_fp32, uint16_t *out,
                                 float *scores, uint16_t *probs, int rows, int heads, int slots, int first_key,
                                 int first_query, bool sliding) {
    int pitch = (slots + 15) / 16 * 16, rb = (rows + 15) / 16;
    score_tiles<<<dim3((pitch + 63) / 64, rb, heads), 128>>>(q, kv, rotary, scores, rows, heads, slots, pitch,
                                                             first_key, first_query, sliding);
    auto status = hipGetLastError();
    if (status != hipSuccess)
        return status;
    normalize<<<dim3(heads, rows), 256>>>(scores, probs, sink, sink_fp32, pitch);
    status = hipGetLastError();
    if (status != hipSuccess)
        return status;
    value_tiles<<<dim3(4, rb, heads), 128>>>(probs, kv, gate, out, rows, heads, slots, pitch, first_key,
                                             first_query, sliding);
    return hipGetLastError();
}

namespace {
using H8 = _Float16 __attribute__((ext_vector_type(8)));
__device__ __forceinline__ H8 half_frag(const _Float16 *p) {
    union {
        uint64_t u[2];
        H8 f;
    } vec;
    __builtin_memcpy(&vec.u[0], p, 8);
    __builtin_memcpy(&vec.u[1], p + 8, 8);
    return vec.f;
}
__device__ __forceinline__ H8 half_from_bf(const uint16_t *ptr) {
    B8 b = fragment(ptr);
    H8 f{};
#pragma unroll
    for (int j = 0; j < 8; ++j)
        f[j] = static_cast<_Float16>(bf16_bits_to_f32(static_cast<uint16_t>(b[j])));
    return f;
}
__global__ void project_query(const uint16_t *q, const _Float16 *w, _Float16 *query, int rows, int heads) {
    int lane = threadIdx.x % 32, lo = lane % 16, hi = lane / 16, h = blockIdx.z;
    int col = blockIdx.x * 64 + (threadIdx.x / 32) * 16, r = blockIdx.y * 16 + lo;
    F8 acc{};
    for (int k = 0; k < 192; k += 16) {
        H8 a{}, b{};
        for (int j = 0; j < 8; ++j)
            a[j] = w[(size_t(h) * 448 + k + hi * 4 + j % 4 + 8 * (j / 4)) * 512 + col + lo];
        if (r < rows)
            b = half_from_bf(q + (size_t(r) * heads + h) * 256 + k + hi * 4);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
    }
    if (r < rows)
        for (int j = 0; j < 8; ++j)
            query[(size_t(r) * heads + h) * 512 + col + hi * 8 + j] = static_cast<_Float16>(acc[j]);
}
__global__ void recall_scores(const _Float16 *input, const _Float16 *query, const uint16_t *rotary,
                              const uint16_t *q, const int *ids, int heads, int stride, int pitch,
                              float *scores) {
    int lane = threadIdx.x % 32, lo = lane % 16, hi = lane / 16;
    int kb = blockIdx.x * 64 + (threadIdx.x / 32) * 16, h = blockIdx.y * 16 + lo, r = blockIdx.z;
    if (kb >= pitch)
        return;
    int token = kb + lo < stride ? ids[r * stride + kb + lo] : -1;
    F8 acc{};
    for (int k = 0; k < 512; k += 16) {
        H8 a{};
        if (token >= 0)
            a = half_frag(input + size_t(token) * 512 + k + hi * 4);
        H8 b = half_frag(query + (size_t(r) * heads + h) * 512 + k + hi * 4);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
    }
    for (int k = 0; k < 64; k += 16) {
        H8 a{};
        if (token >= 0)
            a = half_from_bf(rotary + size_t(token) * 64 + k + hi * 4);
        H8 b = half_from_bf(q + (size_t(r) * heads + h) * 256 + 192 + k + hi * 4);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
    }
    for (int j = 0; j < 8; ++j) {
        int col = kb + hi * 8 + j;
        scores[(size_t(r) * heads + h) * pitch + col] =
            col < stride && ids[r * stride + col] >= 0 ? round_bf16(acc[j]) * 0.0625f : -INFINITY;
    }
}
__global__ void recall_values(const _Float16 *input, const uint16_t *probs, const int *ids, int heads,
                              int stride, int pitch, _Float16 *values) {
    int lane = threadIdx.x % 32, lo = lane % 16, hi = lane / 16;
    int cb = blockIdx.x * 64 + (threadIdx.x / 32) * 16, h = blockIdx.y * 16 + lo, r = blockIdx.z;
    F8 acc{};
    for (int k = 0; k < pitch; k += 16) {
        H8 a{};
        for (int j = 0; j < 8; ++j) {
            int pos = k + hi * 4 + j % 4 + 8 * (j / 4);
            int token = pos < stride ? ids[r * stride + pos] : -1;
            if (token >= 0)
                a[j] = input[size_t(token) * 512 + cb + lo];
        }
        H8 b = half_from_bf(probs + (size_t(r) * heads + h) * pitch + k + hi * 4);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
    }
    for (int j = 0; j < 8; ++j)
        values[(size_t(r) * heads + h) * 512 + cb + hi * 8 + j] = static_cast<_Float16>(acc[j]);
}
__global__ void project_value(const _Float16 *values, const _Float16 *w, const uint16_t *gate, int heads,
                              int rows, uint16_t *out) {
    int lane = threadIdx.x % 32, lo = lane % 16, hi = lane / 16, h = blockIdx.z;
    int cb = blockIdx.x * 64 + (threadIdx.x / 32) * 16, r = blockIdx.y * 16 + lo;
    F8 acc{};
    for (int k = 0; k < 512; k += 16) {
        H8 a = half_frag(w + (size_t(h) * 448 + 192 + cb + lo) * 512 + k + hi * 4);
        H8 b{};
        if (r < rows)
            b = half_frag(values + (size_t(r) * heads + h) * 512 + k + hi * 4);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
    }
    if (r < rows)
        for (int j = 0; j < 8; ++j) {
            size_t i = (size_t(r) * heads + h) * 256 + cb + hi * 8 + j;
            out[i] = f32_to_bf16_rne_device(round_bf16(acc[j]) *
                                            round_bf16(1.f / (1.f + expf(-bf16_bits_to_f32(gate[i])))));
        }
}
}
hipError_t shelf_hy4_mla_absorbed(const uint16_t *q, const void *folded, const uint16_t *rotary,
                                  const uint16_t *gate, const void *sinks, bool sink_fp32, const int *indices,
                                  int stride, const void *input_buffer, void *query_buffer, void *value_buffer,
                                  float *scores, uint16_t *probs, uint16_t *out, int rows, int heads) {
    auto w = static_cast<const _Float16 *>(folded), input = static_cast<const _Float16 *>(input_buffer);
    auto query = static_cast<_Float16 *>(query_buffer), values = static_cast<_Float16 *>(value_buffer);
    int pitch = (stride + 15) / 16 * 16, rb = (rows + 15) / 16;
    project_query<<<dim3(8, rb, heads), 128>>>(q, w, query, rows, heads);
    auto rc = hipGetLastError();
    if (rc != hipSuccess)
        return rc;
    recall_scores<<<dim3((pitch + 63) / 64, heads / 16, rows), 128>>>(input, query, rotary, q, indices, heads,
                                                                      stride, pitch, scores);
    rc = hipGetLastError();
    if (rc != hipSuccess)
        return rc;
    normalize<<<dim3(heads, rows), 256>>>(scores, probs, sinks, sink_fp32, pitch);
    rc = hipGetLastError();
    if (rc != hipSuccess)
        return rc;
    recall_values<<<dim3(8, heads / 16, rows), 128>>>(input, probs, indices, heads, stride, pitch, values);
    rc = hipGetLastError();
    if (rc != hipSuccess)
        return rc;
    project_value<<<dim3(4, rb, heads), 128>>>(values, w, gate, heads, rows, out);
    return hipGetLastError();
}
