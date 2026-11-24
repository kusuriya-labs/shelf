#include <shelf/models/hy4_flash/mla_prefill.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>

namespace {
using namespace shelf::kernel::exl3_detail::mul24;
using shelf::hy4::Hy4StepParams;
using B8 = short __attribute__((ext_vector_type(8)));
using F8 = float __attribute__((ext_vector_type(8)));
__device__ int effective_stride(int stride, bool contiguous, bool sliding, const Hy4StepParams *params, int row_offset,
                                int chunk_rows) {
    if (!params)
        return stride;
    int position = params->position + row_offset;
    if (params->draft_last >= 0) {
        int history = contiguous ? (sliding ? min(512, position) : position) : 2045 + ((position - 1) & 3);
        return history + chunk_rows;
    }
    if (contiguous)
        return min(stride, sliding ? min(512, position + chunk_rows) : position + chunk_rows);
    int mod = 0;
    for (int k = 0; k < chunk_rows; ++k)
        mod = max(mod, (position + k) & 3);
    return min(stride, 2045 + mod);
}
__device__ __forceinline__ B8 fragment(const uint16_t *source) {
    union {
        uint64_t u[2];
        B8 b;
    } value;
    __builtin_memcpy(&value.u[0], source, 8);
    __builtin_memcpy(&value.u[1], source + 8, 8);
    return value.b;
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
__global__ void normalize(float *scores, uint16_t *probs, const void *sinks, bool fp32, int pitch, int slots,
                             bool sliding, const Hy4StepParams *params, int row_offset, bool contiguous,
                             int chunk_rows) {
    __shared__ float parts[8];
    int h = blockIdx.x, r = blockIdx.y, t = threadIdx.x;
    int limit = effective_stride(slots, contiguous, sliding, params, row_offset, chunk_rows);
    limit = (limit + 15) / 16 * 16;
    size_t base = (size_t(r) * gridDim.x + h) * pitch;
    float sink =
        fp32 ? static_cast<const float *>(sinks)[h] : bf16_bits_to_f32(static_cast<const uint16_t *>(sinks)[h]);
    float peak = sink;
    for (int i = t; i < limit; i += 256)
        peak = fmaxf(peak, scores[base + i]);
    peak = reduce<true>(peak, parts);
    float sum = t ? 0.f : expf(sink - peak);
    for (int i = t; i < limit; i += 256) {
        float e = expf(scores[base + i] - peak);
        scores[base + i] = e;
        sum += e;
    }
    sum = reduce<false>(sum, parts);
    for (int i = t; i < limit; i += 256)
        probs[base + i] = f32_to_bf16_rne_device(scores[base + i] / sum);
}
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
                               float *scores, const Hy4StepParams *params, bool contiguous, bool sliding,
                               int row_offset, int chunk_rows) {
    int eff_stride = effective_stride(stride, contiguous, sliding, params, row_offset, chunk_rows);
    int lane = threadIdx.x % 32, lo = lane % 16, hi = lane / 16;
    int kb = blockIdx.x * 64 + (threadIdx.x / 32) * 16, h = blockIdx.y * 16 + lo, r = blockIdx.z;
    if (kb >= eff_stride)
        return;
    int token = kb + lo < eff_stride ? ids[r * stride + kb + lo] : -1;
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
            col < eff_stride && ids[r * stride + col] >= 0 ? round_bf16(acc[j]) * 0.0625f : -INFINITY;
    }
}
__global__ void recall_values(const _Float16 *input, const uint16_t *probs, const int *ids, int heads,
                               int stride, int pitch, _Float16 *values, const Hy4StepParams *params,
                               bool contiguous, bool sliding, int row_offset, int chunk_rows) {
    int eff_stride = effective_stride(stride, contiguous, sliding, params, row_offset, chunk_rows);
    int limit = (eff_stride + 15) / 16 * 16;
    int lane = threadIdx.x % 32, lo = lane % 16, hi = lane / 16;
    int cb = blockIdx.x * 64 + (threadIdx.x / 32) * 16, h = blockIdx.y * 16 + lo, r = blockIdx.z;
    F8 acc{};
    for (int k = 0; k < limit; k += 16) {
        H8 a{};
        for (int j = 0; j < 8; ++j) {
            int pos = k + hi * 4 + j % 4 + 8 * (j / 4);
            int token = pos < eff_stride ? ids[r * stride + pos] : -1;
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
__global__ void recall_fused_batch(const _Float16 *input, const _Float16 *query, const uint16_t *rotary,
                                   const uint16_t *q, const int *ids, const int *row_positions, int heads,
                                   int stride, int pitch, float *scores, uint16_t *probs, _Float16 *values,
                                   const void *sinks, bool sink_fp32, bool sliding) {
    __shared__ float parts[8];
    int r = blockIdx.y, head_base = blockIdx.x * 16;
    int eff_stride = sliding ? min(row_positions[r] + 1, 512) : row_positions[r] + 1;
    int lane = threadIdx.x % 32, lo = lane % 16, hi = lane / 16;
    int h = head_base + lo;
    if (threadIdx.x < 128) {
        for (int tile = 0; tile < pitch; tile += 64) {
            int kb = tile + (threadIdx.x / 32) * 16;
            if (kb >= pitch) continue;
            int token = kb + lo < eff_stride ? ids[r * stride + kb + lo] : -1;
            F8 acc{};
            for (int k = 0; k < 512; k += 16) {
                H8 a{};
                if (token >= 0) a = half_frag(input + (size_t(r) * stride + token) * 512 + k + hi * 4);
                H8 b = half_frag(query + (size_t(r) * heads + h) * 512 + k + hi * 4);
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
            }
            for (int k = 0; k < 64; k += 16) {
                H8 a{};
                if (token >= 0) a = half_from_bf(rotary + (size_t(r) * stride + token) * 64 + k + hi * 4);
                H8 b = half_from_bf(q + (size_t(r) * heads + h) * 256 + 192 + k + hi * 4);
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
            }
            for (int j = 0; j < 8; ++j) {
                int col = kb + hi * 8 + j;
                scores[(size_t(r) * heads + h) * pitch + col] =
                    col < eff_stride && ids[r * stride + col] >= 0 ? round_bf16(acc[j]) * 0.0625f : -INFINITY;
            }
        }
    }
    __syncthreads();
    int limit = (eff_stride + 15) / 16 * 16;
    for (int head = head_base; head < head_base + 16; ++head) {
        size_t base = (size_t(r) * heads + head) * pitch;
        float sink = sink_fp32 ? static_cast<const float *>(sinks)[head]
                               : bf16_bits_to_f32(static_cast<const uint16_t *>(sinks)[head]);
        float peak = sink;
        for (int i = threadIdx.x; i < limit; i += 256) peak = fmaxf(peak, scores[base + i]);
        peak = reduce<true>(peak, parts);
        float sum = threadIdx.x ? 0.f : expf(sink - peak);
        for (int i = threadIdx.x; i < limit; i += 256) {
            float e = expf(scores[base + i] - peak);
            scores[base + i] = e;
            sum += e;
        }
        sum = reduce<false>(sum, parts);
        for (int i = threadIdx.x; i < limit; i += 256)
            probs[base + i] = f32_to_bf16_rne_device(scores[base + i] / sum);
        __syncthreads();
    }
    if (threadIdx.x < 128) {
        for (int tile = 0; tile < 512; tile += 64) {
            int cb = tile + (threadIdx.x / 32) * 16;
            F8 acc{};
            for (int k = 0; k < limit; k += 16) {
                H8 a{};
                for (int j = 0; j < 8; ++j) {
                    int pos = k + hi * 4 + j % 4 + 8 * (j / 4);
                    int token = pos < eff_stride ? ids[r * stride + pos] : -1;
                    if (token >= 0) a[j] = input[(size_t(r) * stride + token) * 512 + cb + lo];
                }
                H8 b = half_from_bf(probs + (size_t(r) * heads + h) * pitch + k + hi * 4);
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
            }
            for (int j = 0; j < 8; ++j)
                values[(size_t(r) * heads + h) * 512 + cb + hi * 8 + j] = static_cast<_Float16>(acc[j]);
        }
    }
}

}
hipError_t shelf_hy4_mla_absorbed(const uint16_t *q, const void *folded, const uint16_t *rotary,
                                   const uint16_t *gate, const void *sinks, bool sink_fp32, const int *indices,
                                   int stride, const void *input_buffer, void *query_buffer, void *value_buffer,
                                   float *scores, uint16_t *probs, uint16_t *out, int rows, int heads,
                                   hipStream_t stream, const Hy4StepParams *params, bool contiguous, bool sliding,
                                   int row_offset, int chunk_rows, bool projected) {
    auto w = static_cast<const _Float16 *>(folded), input = static_cast<const _Float16 *>(input_buffer);
    auto query = static_cast<_Float16 *>(query_buffer), values = static_cast<_Float16 *>(value_buffer);
    int pitch = (stride + 15) / 16 * 16, rb = (rows + 15) / 16;
    if (!projected)
        project_query<<<dim3(8, rb, heads), 128, 0, stream>>>(q, w, query, rows, heads);
    auto rc = hipGetLastError();
    if (rc != hipSuccess)
        return rc;
    recall_scores<<<dim3((pitch + 63) / 64, heads / 16, rows), 128, 0, stream>>>(input, query, rotary, q, indices,
                                                                       heads, stride, pitch, scores, params, contiguous,
                                                                       sliding, row_offset, chunk_rows);
    rc = hipGetLastError();
    if (rc != hipSuccess)
        return rc;
    normalize<<<dim3(heads, rows), 256, 0, stream>>>(scores, probs, sinks, sink_fp32, pitch, stride, sliding, params,
                                                       row_offset, contiguous, chunk_rows);
    rc = hipGetLastError();
    if (rc != hipSuccess)
        return rc;
    recall_values<<<dim3(8, heads / 16, rows), 128, 0, stream>>>(input, probs, indices, heads, stride, pitch, values,
                                                                    params, contiguous, sliding, row_offset, chunk_rows);
    rc = hipGetLastError();
    if (rc != hipSuccess)
        return rc;
    if (!projected)
        project_value<<<dim3(4, rb, heads), 128, 0, stream>>>(values, w, gate, heads, rows, out);
    return hipGetLastError();
}

hipError_t shelf_hy4_mla_absorbed_batch(const uint16_t *q, const void *folded, const uint16_t *rotary,
                                         const uint16_t *gate, const void *sinks, bool sink_fp32,
                                         const int *indices, int stride, const void *input_buffer,
                                         void *query_buffer, void *value_buffer, float *scores, uint16_t *probs,
                                         uint16_t *out, int rows, int heads, hipStream_t stream,
                                         const int *row_positions, bool sliding) {
    auto w = static_cast<const _Float16 *>(folded), input = static_cast<const _Float16 *>(input_buffer);
    auto query = static_cast<_Float16 *>(query_buffer), values = static_cast<_Float16 *>(value_buffer);
    int pitch = (stride + 15) / 16 * 16, rb = (rows + 15) / 16;
    project_query<<<dim3(8, rb, heads), 128, 0, stream>>>(q, w, query, rows, heads);
    auto rc = hipGetLastError();
    if (rc != hipSuccess) return rc;
    recall_fused_batch<<<dim3(heads / 16, rows), 256, 0, stream>>>(
        input, query, rotary, q, indices, row_positions, heads, stride, pitch, scores, probs, values,
        sinks, sink_fp32, sliding);
    rc = hipGetLastError();
    if (rc != hipSuccess) return rc;
    project_value<<<dim3(4, rb, heads), 128, 0, stream>>>(values, w, gate, heads, rows, out);
    return hipGetLastError();
}

hipError_t shelf_hy4_mla_project_query(const uint16_t *q, const void *folded, void *query, int rows, int heads, hipStream_t stream) {
    project_query<<<dim3(8, (rows + 15) / 16, heads), 128, 0, stream>>>(q, static_cast<const _Float16 *>(folded), static_cast<_Float16 *>(query), rows, heads);
    return hipGetLastError();
}
hipError_t shelf_hy4_mla_project_value(const void *values, const void *folded, const uint16_t *gate, uint16_t *out, int rows, int heads, hipStream_t stream) {
    project_value<<<dim3(4, (rows + 15) / 16, heads), 128, 0, stream>>>(static_cast<const _Float16 *>(values), static_cast<const _Float16 *>(folded), gate, heads, rows, out);
    return hipGetLastError();
}
