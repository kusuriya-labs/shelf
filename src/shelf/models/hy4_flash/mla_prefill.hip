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
    int lane = threadIdx.x & 31;
    for (int d = 16; d; d >>= 1) {
        float other = __shfl_down(x, d, 32);
        x = Max ? fmaxf(x, other) : x + other;
    }
    if (!lane)
        parts[threadIdx.x >> 5] = x;
    __syncthreads();
    if constexpr (Max) {
        float v = parts[lane & 7];
        for (int d = 4; d; d >>= 1)
            v = fmaxf(v, __shfl_xor(v, d, 32));
        return v;
    } else {
        float p0 = parts[0] + parts[4], p1 = parts[1] + parts[5];
        float p2 = parts[2] + parts[6], p3 = parts[3] + parts[7];
        return (p0 + p2) + (p1 + p3);
    }
}
__global__ void normalize(float *scores, uint16_t *probs, const void *sinks, bool fp32, int pitch, int slots,
                             bool sliding, const Hy4StepParams *params, int row_offset, bool contiguous,
                             int chunk_rows) {
    __shared__ float parts[2][8];
    int h = blockIdx.x, r = blockIdx.y, t = threadIdx.x;
    int limit = effective_stride(slots, contiguous, sliding, params, row_offset, chunk_rows);
    limit = (limit + 15) / 16 * 16;
    size_t base = (size_t(r) * gridDim.x + h) * pitch;
    float sink =
        fp32 ? static_cast<const float *>(sinks)[h] : bf16_bits_to_f32(static_cast<const uint16_t *>(sinks)[h]);
    if (limit <= 2048) {
        float val[8];
        float peak = sink;
        int n = 0;
        for (int i = t; i < limit; i += 256, ++n) {
            float v = scores[base + i];
            val[n] = v;
            peak = fmaxf(peak, v);
        }
        peak = reduce<true>(peak, parts[0]);
        float sum = t ? 0.f : expf(sink - peak);
        for (int k = 0; k < n; ++k) {
            float e = expf(val[k] - peak);
            val[k] = e;
            sum += e;
        }
        sum = reduce<false>(sum, parts[1]);
        for (int k = 0, i = t; k < n; ++k, i += 256)
            probs[base + i] = f32_to_bf16_rne_device(val[k] / sum);
        return;
    }
    float peak = sink;
    for (int i = t; i < limit; i += 256)
        peak = fmaxf(peak, scores[base + i]);
    peak = reduce<true>(peak, parts[0]);
    float sum = t ? 0.f : expf(sink - peak);
    for (int i = t; i < limit; i += 256) {
        float e = expf(scores[base + i] - peak);
        scores[base + i] = e;
        sum += e;
    }
    sum = reduce<false>(sum, parts[1]);
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
__device__ __forceinline__ void bf4_to_half4(uint2 raw, _Float16 *dst) {
    uint16_t s[4] = {
        uint16_t(raw.x & 0xffffu),
        uint16_t(raw.x >> 16),
        uint16_t(raw.y & 0xffffu),
        uint16_t(raw.y >> 16),
    };
#pragma unroll
    for (int i = 0; i < 4; ++i)
        dst[i] = static_cast<_Float16>(bf16_bits_to_f32(s[i]));
}
__device__ __forceinline__ void bf8_to_half8(uint4 raw, _Float16 *dst) {
    uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        dst[i * 2] = static_cast<_Float16>(bf16_bits_to_f32(uint16_t(w[i] & 0xffffu)));
        dst[i * 2 + 1] = static_cast<_Float16>(bf16_bits_to_f32(uint16_t(w[i] >> 16)));
    }
}
__global__ void project_query(const uint16_t *q, const _Float16 *w, _Float16 *query, int rows, int heads,
                              const int *block_map = nullptr, int block_rows = 1) {
    int lane = threadIdx.x % 32, lo = lane % 16, hi = lane / 16, h = blockIdx.z;
    int col = blockIdx.x * 64 + (threadIdx.x / 32) * 16, r = blockIdx.y * 16 + lo;
    int q_row = r < rows && block_map ? block_map[r / block_rows] * block_rows + r % block_rows : r;
    F8 acc{};
    for (int k = 0; k < 192; k += 16) {
        H8 a{}, b{};
        for (int j = 0; j < 8; ++j)
            a[j] = w[(size_t(h) * 448 + k + hi * 4 + j % 4 + 8 * (j / 4)) * 512 + col + lo];
        if (r < rows)
            b = half_from_bf(q + (size_t(q_row) * heads + h) * 256 + k + hi * 4);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
    }
    if (r < rows)
        for (int j = 0; j < 8; ++j)
            query[(size_t(r) * heads + h) * 512 + col + hi * 8 + j] = static_cast<_Float16>(acc[j]);
}
__global__ void __launch_bounds__(256) project_query_tiled(const uint16_t *q, const _Float16 *w, _Float16 *query,
                                                           int rows, int heads) {
    __shared__ _Float16 w_lds[16][256];
    __shared__ _Float16 q_lds[64][24];
    int t = threadIdx.x, wave = t >> 5, lane = t & 31, lo = lane & 15, hi = lane >> 4;
    int h = blockIdx.z;
    int col_base = blockIdx.x * 256;
    int r_base = blockIdx.y * 64;
    int w_n = wave >> 1, w_m = wave & 1;
    F8 acc[2][4]{};
    const _Float16 *w_head = w + size_t(h) * 448 * 512 + col_base;
    int w_r0 = t >> 5, w_c = (t & 31) * 8;
    int q_r = t >> 2, q_k = (t & 3) * 4;
    int r_load = r_base + q_r;
    const uint16_t *q_ptr = r_load < rows ? q + (size_t(r_load) * heads + h) * 256 + q_k : nullptr;
    for (int k = 0; k < 192; k += 16) {
        *reinterpret_cast<uint4 *>(&w_lds[w_r0][w_c]) =
            *reinterpret_cast<const uint4 *>(w_head + size_t(k + w_r0) * 512 + w_c);
        *reinterpret_cast<uint4 *>(&w_lds[w_r0 + 8][w_c]) =
            *reinterpret_cast<const uint4 *>(w_head + size_t(k + w_r0 + 8) * 512 + w_c);
        if (q_ptr)
            bf4_to_half4(*reinterpret_cast<const uint2 *>(q_ptr + k), &q_lds[q_r][q_k]);
        else
            *reinterpret_cast<uint2 *>(&q_lds[q_r][q_k]) = uint2{};
        __syncthreads();
        H8 b[2];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
            b[mt] = half_frag(&q_lds[w_m * 32 + mt * 16 + lo][hi * 4]);
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            H8 a{};
            int c_off = w_n * 64 + nt * 16 + lo;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                a[j] = w_lds[hi * 4 + (j & 3) + 8 * (j >> 2)][c_off];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt)
                acc[mt][nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b[mt], acc[mt][nt]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
        int r = r_base + w_m * 32 + mt * 16 + lo;
        if (r >= rows)
            continue;
        _Float16 *dst_row = query + (size_t(r) * heads + h) * 512 + col_base + w_n * 64 + hi * 8;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            H8 out_h;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                out_h[j] = static_cast<_Float16>(acc[mt][nt][j]);
            *reinterpret_cast<H8 *>(dst_row + nt * 16) = out_h;
        }
    }
}
template <int HeadGroups>
__global__ void __launch_bounds__(256) recall_scores_tiled(const _Float16 *input, const _Float16 *query,
                                                             const uint16_t *rotary, const uint16_t *q,
                                                             const int *ids, int heads, int stride, int pitch,
                                                             float *scores, const Hy4StepParams *params,
                                                             bool contiguous, bool sliding, int row_offset,
                                                             int chunk_rows) {
    int eff_stride = effective_stride(stride, contiguous, sliding, params, row_offset, chunk_rows);
    int kb_base = blockIdx.x * 128;
    if (kb_base >= eff_stride)
        return;
    constexpr int NT = HeadGroups / 2;
    __shared__ int s_ids[128];
    __shared__ _Float16 s_a[128][40];
    __shared__ _Float16 s_b[HeadGroups * 16][40];
    int t = threadIdx.x, wave = t >> 5, lane = t & 31, lo = lane & 15, hi = lane >> 4;
    int r = blockIdx.z;
    if (t < 128)
        s_ids[t] = kb_base + t < eff_stride ? ids[size_t(r) * stride + kb_base + t] : -1;
    __syncthreads();
    int w_k = wave & 3, w_h = wave >> 2;
    F8 acc[2][NT]{};
    const _Float16 *q_row = query + size_t(r) * heads * 512;
    int tok_lane = (t >> 2) & 63, k_lane = (t & 3) * 8;
    int h_lane = t >> 2;
    for (int k = 0; k < 512; k += 32) {
#pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
            int tok = pass * 64 + tok_lane;
            int id = s_ids[tok];
            *reinterpret_cast<uint4 *>(&s_a[tok][k_lane]) =
                id >= 0 ? *reinterpret_cast<const uint4 *>(input + size_t(id) * 512 + k + k_lane) : uint4{};
        }
        if (h_lane < HeadGroups * 16)
            *reinterpret_cast<uint4 *>(&s_b[h_lane][k_lane]) =
                *reinterpret_cast<const uint4 *>(q_row + size_t(h_lane) * 512 + k + k_lane);
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < 32; ks += 16) {
            H8 a[2];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt)
                a[mt] = half_frag(&s_a[(w_k + mt * 4) * 16 + lo][ks + hi * 4]);
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                H8 b = half_frag(&s_b[(w_h * NT + nt) * 16 + lo][ks + hi * 4]);
#pragma unroll
                for (int mt = 0; mt < 2; ++mt)
                    acc[mt][nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[mt], b, acc[mt][nt]);
            }
        }
        __syncthreads();
    }
    const uint16_t *qr_row = q + size_t(r) * heads * 256 + 192;
    for (int k = 0; k < 64; k += 32) {
#pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
            int tok = pass * 64 + tok_lane;
            int id = s_ids[tok];
            if (id >= 0)
                bf8_to_half8(*reinterpret_cast<const uint4 *>(rotary + size_t(id) * 64 + k + k_lane),
                             &s_a[tok][k_lane]);
            else
                *reinterpret_cast<uint4 *>(&s_a[tok][k_lane]) = uint4{};
        }
        if (h_lane < HeadGroups * 16)
            bf8_to_half8(*reinterpret_cast<const uint4 *>(qr_row + size_t(h_lane) * 256 + k + k_lane),
                         &s_b[h_lane][k_lane]);
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < 32; ks += 16) {
            H8 a[2];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt)
                a[mt] = half_frag(&s_a[(w_k + mt * 4) * 16 + lo][ks + hi * 4]);
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                H8 b = half_frag(&s_b[(w_h * NT + nt) * 16 + lo][ks + hi * 4]);
#pragma unroll
                for (int mt = 0; mt < 2; ++mt)
                    acc[mt][nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[mt], b, acc[mt][nt]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
        int kb_loc = (w_k + mt * 4) * 16;
        int kb = kb_base + kb_loc;
        if (kb >= eff_stride)
            continue;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            int h = (w_h * NT + nt) * 16 + lo;
            float out_s[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                int loc = kb_loc + hi * 8 + j;
                out_s[j] = s_ids[loc] >= 0 ? round_bf16(acc[mt][nt][j]) * 0.0625f : -INFINITY;
            }
            float *dst = scores + (size_t(r) * heads + h) * pitch + kb + hi * 8;
            *reinterpret_cast<float4 *>(dst) = float4{out_s[0], out_s[1], out_s[2], out_s[3]};
            *reinterpret_cast<float4 *>(dst + 4) = float4{out_s[4], out_s[5], out_s[6], out_s[7]};
        }
    }
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
template <int HeadGroups>
__global__ void __launch_bounds__(256) recall_values_tiled(const _Float16 *input, const uint16_t *probs,
                                                             const int *ids, int heads, int stride, int pitch,
                                                             _Float16 *values, const Hy4StepParams *params,
                                                             bool contiguous, bool sliding, int row_offset,
                                                             int chunk_rows) {
    int eff_stride = effective_stride(stride, contiguous, sliding, params, row_offset, chunk_rows);
    int limit = (eff_stride + 15) / 16 * 16;
    constexpr int HC = HeadGroups == 2 ? 2 : 4;
    constexpr int NC = 8 / HC;
    constexpr int NT = 16 / NC;
    __shared__ int s_ids[32];
    __shared__ _Float16 s_v[32][264];
    __shared__ _Float16 s_p[HeadGroups * 16][40];
    int t = threadIdx.x, wave = t >> 5, lane = t & 31, lo = lane & 15, hi = lane >> 4;
    int r = blockIdx.z;
    int cb_base = blockIdx.x * 256;
    int w_h = wave % HC, w_c = wave / HC;
    F8 acc[NT]{};
    const uint16_t *p_row = probs + size_t(r) * heads * pitch;
    int p_h = t >> 2, p_k = (t & 3) * 8;
    for (int k = 0; k < limit; k += 32) {
        if (t < 32)
            s_ids[t] = k + t < eff_stride ? ids[size_t(r) * stride + k + t] : -1;
        __syncthreads();
#pragma unroll
        for (int tok = wave; tok < 32; tok += 8) {
            int id = s_ids[tok];
            *reinterpret_cast<uint4 *>(&s_v[tok][lane * 8]) =
                id >= 0 ? *reinterpret_cast<const uint4 *>(input + size_t(id) * 512 + cb_base + lane * 8) : uint4{};
        }
        if (HeadGroups == 4) {
            if (k + p_k < pitch)
                bf8_to_half8(*reinterpret_cast<const uint4 *>(p_row + size_t(p_h) * pitch + k + p_k), &s_p[p_h][p_k]);
            else
                *reinterpret_cast<uint4 *>(&s_p[p_h][p_k]) = uint4{};
        } else if (p_h < 32) {
            if (k + p_k < pitch)
                bf8_to_half8(*reinterpret_cast<const uint4 *>(p_row + size_t(p_h) * pitch + k + p_k), &s_p[p_h][p_k]);
            else
                *reinterpret_cast<uint4 *>(&s_p[p_h][p_k]) = uint4{};
        }
        __syncthreads();
        for (int ks = 0; ks < 32 && k + ks < limit; ks += 16) {
            H8 b = half_frag(&s_p[w_h * 16 + lo][ks + hi * 4]);
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                H8 a{};
                int c_idx = (w_c * NT + nt) * 16 + lo;
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    a[j] = s_v[ks + hi * 4 + (j & 3) + 8 * (j >> 2)][c_idx];
                acc[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc[nt]);
            }
        }
        __syncthreads();
    }
    int h = w_h * 16 + lo;
    _Float16 *dst = values + (size_t(r) * heads + h) * 512 + cb_base + w_c * (NT * 16) + hi * 8;
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        H8 out_h;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            out_h[j] = static_cast<_Float16>(acc[nt][j]);
        *reinterpret_cast<H8 *>(dst + nt * 16) = out_h;
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
                              int rows, uint16_t *out, const int *block_map = nullptr, int block_rows = 1) {
    int lane = threadIdx.x % 32, lo = lane % 16, hi = lane / 16, h = blockIdx.z;
    int cb = blockIdx.x * 64 + (threadIdx.x / 32) * 16, r = blockIdx.y * 16 + lo;
    int out_row = r < rows && block_map ? block_map[r / block_rows] * block_rows + r % block_rows : r;
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
            size_t i = (size_t(out_row) * heads + h) * 256 + cb + hi * 8 + j;
            out[i] = f32_to_bf16_rne_device(round_bf16(acc[j]) *
                                            round_bf16(1.f / (1.f + expf(-bf16_bits_to_f32(gate[i])))));
        }
}
__global__ void __launch_bounds__(256) project_value_tiled(const _Float16 *values, const _Float16 *w,
                                                           const uint16_t *gate, int heads, int rows,
                                                           uint16_t *out) {
    __shared__ _Float16 w_lds[128][40];
    __shared__ _Float16 v_lds[128][40];
    int t = threadIdx.x, wave = t >> 5, lane = t & 31, lo = lane & 15, hi = lane >> 4;
    int h = blockIdx.z;
    int c_base = blockIdx.x * 128;
    int r_base = blockIdx.y * 128;
    int w_n = wave >> 1, w_m = wave & 1;
    F8 acc[4][2]{};
    const _Float16 *w_head = w + (size_t(h) * 448 + 192 + c_base) * 512;
    int w_row_base = t >> 2, w_k_off = (t & 3) * 8;
    int v_row_base = t >> 2, v_k_off = (t & 3) * 8;
    const _Float16 *v_base_ptr = values + size_t(h) * 512 + v_k_off;
    size_t v_row_stride = size_t(heads) * 512;
    for (int k = 0; k < 512; k += 32) {
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            int wr = s * 64 + w_row_base;
            *reinterpret_cast<uint4 *>(&w_lds[wr][w_k_off]) =
                *reinterpret_cast<const uint4 *>(w_head + size_t(wr) * 512 + k + w_k_off);
            int vr = s * 64 + v_row_base;
            int r_load = r_base + vr;
            *reinterpret_cast<uint4 *>(&v_lds[vr][v_k_off]) =
                r_load < rows ? *reinterpret_cast<const uint4 *>(v_base_ptr + size_t(r_load) * v_row_stride + k) : uint4{};
        }
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < 32; ks += 16) {
            H8 a[2];
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
                a[nt] = half_frag(&w_lds[(w_n * 2 + nt) * 16 + lo][ks + hi * 4]);
#pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                H8 b = half_frag(&v_lds[(w_m * 4 + mt) * 16 + lo][ks + hi * 4]);
#pragma unroll
                for (int nt = 0; nt < 2; ++nt)
                    acc[mt][nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[nt], b, acc[mt][nt]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
        int r = r_base + (w_m * 4 + mt) * 16 + lo;
        if (r >= rows)
            continue;
        size_t base_i = (size_t(r) * heads + h) * 256 + c_base + w_n * 32 + hi * 8;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            size_t idx = base_i + nt * 16;
            uint4 g_raw = *reinterpret_cast<const uint4 *>(gate + idx);
            uint32_t gw[4] = {g_raw.x, g_raw.y, g_raw.z, g_raw.w};
            uint32_t ow[4];
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                float g0 = bf16_bits_to_f32(uint16_t(gw[p] & 0xffffu));
                float g1 = bf16_bits_to_f32(uint16_t(gw[p] >> 16));
                uint16_t o0 = f32_to_bf16_rne_device(
                    round_bf16(acc[mt][nt][p * 2]) * round_bf16(1.f / (1.f + expf(-g0))));
                uint16_t o1 = f32_to_bf16_rne_device(
                    round_bf16(acc[mt][nt][p * 2 + 1]) * round_bf16(1.f / (1.f + expf(-g1))));
                ow[p] = uint32_t(o0) | (uint32_t(o1) << 16);
            }
            *reinterpret_cast<uint4 *>(out + idx) = uint4{ow[0], ow[1], ow[2], ow[3]};
        }
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

__device__ int sparse_eff_stride(int stride, int block_rows, const int *row_positions, int r, bool draft) {
    if (draft) {
        int pos0 = row_positions[(r / block_rows) * block_rows];
        return min(stride, 2045 + ((pos0 - 1) & 3) + block_rows);
    }
    int pos = row_positions[r];
    return min(stride, pos >= 2048 ? 2045 + (pos & 3) : pos + 1);
}
__global__ void recall_scores_sparse(const _Float16 *input, const _Float16 *query, const uint16_t *rotary,
                                     const uint16_t *q, const int *ids, int heads, int stride, int id_stride,
                                     int pitch, float *scores, const int *row_positions, const int *block_map,
                                     int block_rows, bool draft) {
    int r = blockIdx.z;
    int eff_stride = sparse_eff_stride(stride, block_rows, row_positions, r, draft);
    int lane = threadIdx.x % 32, lo = lane % 16, hi = lane / 16;
    int kb = blockIdx.x * 64 + (threadIdx.x / 32) * 16, h = blockIdx.y * 16 + lo;
    if (kb >= eff_stride)
        return;
    int k_lane = r / block_rows;
    int q_row = block_map ? block_map[k_lane] * block_rows + r % block_rows : r;
    const _Float16 *lane_input = input + size_t(k_lane) * id_stride * 512;
    const uint16_t *lane_rotary = rotary + size_t(k_lane) * id_stride * 64;
    int token = kb + lo < eff_stride ? ids[r * stride + kb + lo] : -1;
    F8 acc{};
    for (int k = 0; k < 512; k += 16) {
        H8 a{};
        if (token >= 0)
            a = half_frag(lane_input + size_t(token) * 512 + k + hi * 4);
        H8 b = half_frag(query + (size_t(r) * heads + h) * 512 + k + hi * 4);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
    }
    for (int k = 0; k < 64; k += 16) {
        H8 a{};
        if (token >= 0)
            a = half_from_bf(lane_rotary + size_t(token) * 64 + k + hi * 4);
        H8 b = half_from_bf(q + (size_t(q_row) * heads + h) * 256 + 192 + k + hi * 4);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
    }
    for (int j = 0; j < 8; ++j) {
        int col = kb + hi * 8 + j;
        scores[(size_t(r) * heads + h) * pitch + col] =
            col < eff_stride && ids[r * stride + col] >= 0 ? round_bf16(acc[j]) * 0.0625f : -INFINITY;
    }
}
__global__ void normalize_sparse(float *scores, uint16_t *probs, const void *sinks, bool fp32, int pitch, int stride,
                                 const int *row_positions, int block_rows, bool draft) {
    __shared__ float parts[8];
    int h = blockIdx.x, r = blockIdx.y, t = threadIdx.x;
    int limit = sparse_eff_stride(stride, block_rows, row_positions, r, draft);
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
__global__ void recall_values_sparse(const _Float16 *input, const uint16_t *probs, const int *ids, int heads,
                                     int stride, int id_stride, int pitch, _Float16 *values,
                                     const int *row_positions, int block_rows, bool draft) {
    int r = blockIdx.z;
    int eff_stride = sparse_eff_stride(stride, block_rows, row_positions, r, draft);
    int limit = (eff_stride + 15) / 16 * 16;
    int lane = threadIdx.x % 32, lo = lane % 16, hi = lane / 16;
    int cb = blockIdx.x * 64 + (threadIdx.x / 32) * 16, h = blockIdx.y * 16 + lo;
    const _Float16 *lane_input = input + size_t(r / block_rows) * id_stride * 512;
    F8 acc{};
    for (int k = 0; k < limit; k += 16) {
        H8 a{};
        for (int j = 0; j < 8; ++j) {
            int pos = k + hi * 4 + j % 4 + 8 * (j / 4);
            int token = pos < eff_stride ? ids[r * stride + pos] : -1;
            if (token >= 0)
                a[j] = lane_input[size_t(token) * 512 + cb + lo];
        }
        H8 b = half_from_bf(probs + (size_t(r) * heads + h) * pitch + k + hi * 4);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
    }
    for (int j = 0; j < 8; ++j)
        values[(size_t(r) * heads + h) * 512 + cb + hi * 8 + j] = static_cast<_Float16>(acc[j]);
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
    if (!projected) {
        if (rows > 16)
            project_query_tiled<<<dim3(2, (rows + 63) / 64, heads), 256, 0, stream>>>(q, w, query, rows, heads);
        else
            project_query<<<dim3(8, rb, heads), 128, 0, stream>>>(q, w, query, rows, heads);
    }
    auto rc = hipGetLastError();
    if (rc != hipSuccess)
        return rc;
    if (heads == 64)
        recall_scores_tiled<4><<<dim3((pitch + 127) / 128, 1, rows), 256, 0, stream>>>(
            input, query, rotary, q, indices, heads, stride, pitch, scores, params, contiguous, sliding, row_offset,
            chunk_rows);
    else if (heads == 32)
        recall_scores_tiled<2><<<dim3((pitch + 127) / 128, 1, rows), 256, 0, stream>>>(
            input, query, rotary, q, indices, heads, stride, pitch, scores, params, contiguous, sliding, row_offset,
            chunk_rows);
    else
        recall_scores<<<dim3((pitch + 63) / 64, heads / 16, rows), 128, 0, stream>>>(
            input, query, rotary, q, indices, heads, stride, pitch, scores, params, contiguous, sliding, row_offset,
            chunk_rows);
    rc = hipGetLastError();
    if (rc != hipSuccess)
        return rc;
    normalize<<<dim3(heads, rows), 256, 0, stream>>>(scores, probs, sinks, sink_fp32, pitch, stride, sliding, params,
                                                       row_offset, contiguous, chunk_rows);
    rc = hipGetLastError();
    if (rc != hipSuccess)
        return rc;
    if (heads == 64)
        recall_values_tiled<4><<<dim3(2, 1, rows), 256, 0, stream>>>(
            input, probs, indices, heads, stride, pitch, values, params, contiguous, sliding, row_offset, chunk_rows);
    else if (heads == 32)
        recall_values_tiled<2><<<dim3(2, 1, rows), 256, 0, stream>>>(
            input, probs, indices, heads, stride, pitch, values, params, contiguous, sliding, row_offset, chunk_rows);
    else
        recall_values<<<dim3(8, heads / 16, rows), 128, 0, stream>>>(
            input, probs, indices, heads, stride, pitch, values, params, contiguous, sliding, row_offset, chunk_rows);
    rc = hipGetLastError();
    if (rc != hipSuccess)
        return rc;
    if (!projected) {
        if (rows > 16)
            project_value_tiled<<<dim3(2, (rows + 127) / 128, heads), 256, 0, stream>>>(values, w, gate, heads, rows, out);
        else
            project_value<<<dim3(4, rb, heads), 128, 0, stream>>>(values, w, gate, heads, rows, out);
    }
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
    if (rows > 16)
        project_query_tiled<<<dim3(2, (rows + 63) / 64, heads), 256, 0, stream>>>(q, w, query, rows, heads);
    else
        project_query<<<dim3(8, rb, heads), 128, 0, stream>>>(q, w, query, rows, heads);
    auto rc = hipGetLastError();
    if (rc != hipSuccess) return rc;
    recall_fused_batch<<<dim3(heads / 16, rows), 256, 0, stream>>>(
        input, query, rotary, q, indices, row_positions, heads, stride, pitch, scores, probs, values,
        sinks, sink_fp32, sliding);
    rc = hipGetLastError();
    if (rc != hipSuccess) return rc;
    if (rows > 16)
        project_value_tiled<<<dim3(2, (rows + 127) / 128, heads), 256, 0, stream>>>(values, w, gate, heads, rows, out);
    else
        project_value<<<dim3(4, rb, heads), 128, 0, stream>>>(values, w, gate, heads, rows, out);
    return hipGetLastError();
}

hipError_t shelf_hy4_mla_absorbed_sparse_batch(const uint16_t *q, const void *folded, const uint16_t *rotary,
                                                const uint16_t *gate, const void *sinks, bool sink_fp32,
                                                const int *indices, int stride, int id_stride,
                                                const void *input_buffer, void *query_buffer, void *value_buffer,
                                                float *scores, uint16_t *probs, uint16_t *out, int sparse_blocks,
                                                int block_rows, int heads, hipStream_t stream,
                                                const int *row_positions, const int *block_map, bool draft) {
    auto w = static_cast<const _Float16 *>(folded), input = static_cast<const _Float16 *>(input_buffer);
    auto query = static_cast<_Float16 *>(query_buffer), values = static_cast<_Float16 *>(value_buffer);
    int rows = sparse_blocks * block_rows;
    int pitch = (stride + 15) / 16 * 16, rb = (rows + 15) / 16;
    project_query<<<dim3(8, rb, heads), 128, 0, stream>>>(q, w, query, rows, heads, block_map, block_rows);
    auto rc = hipGetLastError();
    if (rc != hipSuccess) return rc;
    recall_scores_sparse<<<dim3((pitch + 63) / 64, heads / 16, rows), 128, 0, stream>>>(
        input, query, rotary, q, indices, heads, stride, id_stride, pitch, scores, row_positions, block_map,
        block_rows, draft);
    rc = hipGetLastError();
    if (rc != hipSuccess) return rc;
    normalize_sparse<<<dim3(heads, rows), 256, 0, stream>>>(scores, probs, sinks, sink_fp32, pitch, stride,
                                                            row_positions, block_rows, draft);
    rc = hipGetLastError();
    if (rc != hipSuccess) return rc;
    recall_values_sparse<<<dim3(8, heads / 16, rows), 128, 0, stream>>>(
        input, probs, indices, heads, stride, id_stride, pitch, values, row_positions, block_rows, draft);
    rc = hipGetLastError();
    if (rc != hipSuccess) return rc;
    project_value<<<dim3(4, rb, heads), 128, 0, stream>>>(values, w, gate, heads, rows, out, block_map, block_rows);
    return hipGetLastError();
}

hipError_t shelf_hy4_mla_project_query(const uint16_t *q, const void *folded, void *query, int rows, int heads, hipStream_t stream) {
    if (rows > 16)
        project_query_tiled<<<dim3(2, (rows + 63) / 64, heads), 256, 0, stream>>>(q, static_cast<const _Float16 *>(folded), static_cast<_Float16 *>(query), rows, heads);
    else
        project_query<<<dim3(8, (rows + 15) / 16, heads), 128, 0, stream>>>(q, static_cast<const _Float16 *>(folded), static_cast<_Float16 *>(query), rows, heads);
    return hipGetLastError();
}
hipError_t shelf_hy4_mla_project_value(const void *values, const void *folded, const uint16_t *gate, uint16_t *out, int rows, int heads, hipStream_t stream) {
    if (rows > 16)
        project_value_tiled<<<dim3(2, (rows + 127) / 128, heads), 256, 0, stream>>>(static_cast<const _Float16 *>(values), static_cast<const _Float16 *>(folded), gate, heads, rows, out);
    else
        project_value<<<dim3(4, (rows + 15) / 16, heads), 128, 0, stream>>>(static_cast<const _Float16 *>(values), static_cast<const _Float16 *>(folded), gate, heads, rows, out);
    return hipGetLastError();
}
