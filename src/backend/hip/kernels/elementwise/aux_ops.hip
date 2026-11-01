// shelf backend/hip — small auxiliary bf16 ops for the GPU step executor.
// (vendored PhaseShift kernels cover the heavy ops; these glue the IR kinds)
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdlib>

#include <cstdint>

#include <shelf/foundation/numeric.hpp>

namespace shelf::hipk {

// out[r][d] = in[r][begin + d]; row-major bf16 with input row stride.
// (last-dim slice between two ops that need contiguous rows)
extern "C" __global__ __launch_bounds__(256)
void bf16_row_slice(const shelf::bf16_t* __restrict__ in, shelf::bf16_t* __restrict__ out,
                    uint32_t cols, uint32_t begin, uint32_t row_stride, uint32_t total) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const uint32_t r = i / cols;
    const uint32_t d = i - r * cols;
    out[i] = in[static_cast<size_t>(r) * row_stride + begin + d];
}

// 8 要素（bf16 × 8 = 16 B）ずつ運ぶ版。`cols` が 8 の倍数で、入力側の開始位置
// （row_stride・begin）も 8 要素境界にあるときだけ使う（呼び手が判定する）。
// 1 命令 16 B のロード/ストアになるので、prefill の大きいスライス（数百 KB）で効く。
extern "C" __global__ __launch_bounds__(256)
void bf16_row_slice_vec8(const shelf::bf16_t* __restrict__ in,
                         shelf::bf16_t* __restrict__ out, uint32_t cols, uint32_t begin,
                         uint32_t row_stride, uint32_t vec_per_row, uint32_t total_vec) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_vec) return;
    const uint32_t r = i / vec_per_row;
    const uint32_t d = i - r * vec_per_row;
    const uint4 v = *reinterpret_cast<const uint4*>(
        in + static_cast<size_t>(r) * row_stride + begin + static_cast<size_t>(d) * 8u);
    *reinterpret_cast<uint4*>(out + static_cast<size_t>(r) * cols + static_cast<size_t>(d) * 8u) = v;
}

// out = x * sigmoid(z), elementwise bf16.
extern "C" __global__ __launch_bounds__(256)
void bf16_sigmoid_gate(const shelf::bf16_t* __restrict__ x, const shelf::bf16_t* __restrict__ z,
                       shelf::bf16_t* __restrict__ out, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float xv = __bfloat162float(__ushort_as_bfloat16(x[i].data));
    const float zv = __bfloat162float(__ushort_as_bfloat16(z[i].data));
    const float s = 1.0f / (1.0f + expf(-zv));
    out[i] = shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(xv * s))};
}

// out = a + b (bf16 residual stream, matches production bf16 residuals)
extern "C" __global__ __launch_bounds__(256)
void bf16_add(const shelf::bf16_t* __restrict__ a, const shelf::bf16_t* __restrict__ b,
              shelf::bf16_t* __restrict__ out, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float av = __bfloat162float(__ushort_as_bfloat16(a[i].data));
    const float bv = __bfloat162float(__ushort_as_bfloat16(b[i].data));
    out[i] = shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(av + bv))};
}

// f32 logits argmax -> token (single block reduction, vocab <= 1<<20)
extern "C" __global__ __launch_bounds__(1024)
void f32_argmax(const float* __restrict__ logits, uint32_t n, uint32_t* __restrict__ out_tok) {
    __shared__ float best_v[1024];
    __shared__ uint32_t best_i[1024];
    uint32_t bv = 0;
    float bv0 = -__builtin_huge_valf();
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = logits[i];
        if (v > bv0) { bv0 = v; bv = i; }
    }
    best_v[threadIdx.x] = bv0;
    best_i[threadIdx.x] = bv;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (best_v[threadIdx.x + s] > best_v[threadIdx.x]) {
                best_v[threadIdx.x] = best_v[threadIdx.x + s];
                best_i[threadIdx.x] = best_i[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *out_tok = best_i[0];
}

hipError_t launch_bf16_row_slice(const shelf::bf16_t* in, shelf::bf16_t* out, uint32_t rows,
                                 uint32_t cols, uint32_t begin, uint32_t in_row_stride,
                                 hipStream_t stream) {
    // 8 要素（16 B）単位で運べる形ならそちらを使う（アラインメントは要素数で判定）。
    // A/B 用: `SHELF_FORCE_ELEM_V4=1`（全体）または
    // `SHELF_FORCE_SLICE_SCALAR=1`（行スライスだけ）でスカラー版に固定する。
    static const bool force_v4 = std::getenv("SHELF_FORCE_ELEM_V4") != nullptr ||
                                 std::getenv("SHELF_FORCE_SLICE_SCALAR") != nullptr;
    static const bool dump_elem = std::getenv("SHELF_DUMP_ELEM") != nullptr;
    if (dump_elem) {
        static int dumped = 0;
        if (dumped++ < 200)
            std::fprintf(stderr, "[elem] slice rows=%u cols=%u begin=%u in_rs=%u\n",
                         rows, cols, begin, in_row_stride);
    }
    if (!force_v4 && (cols % 8u) == 0u && (begin % 8u) == 0u && (in_row_stride % 8u) == 0u) {
        const uint32_t vpr = cols / 8u;
        const uint32_t total_vec = rows * vpr;
        bf16_row_slice_vec8<<<(total_vec + 255u) / 256u, 256, 0, stream>>>(
            in, out, cols, begin, in_row_stride, vpr, total_vec);
        return hipGetLastError();
    }
    const uint32_t total = rows * cols;
    bf16_row_slice<<<(total + 255u) / 256u, 256, 0, stream>>>(in, out, cols, begin,
                                                              in_row_stride, total);
    return hipGetLastError();
}

hipError_t launch_bf16_sigmoid_gate(const shelf::bf16_t* x, const shelf::bf16_t* z,
                                    shelf::bf16_t* out, uint32_t n, hipStream_t stream) {
    bf16_sigmoid_gate<<<(n + 255u) / 256u, 256, 0, stream>>>(x, z, out, n);
    return hipGetLastError();
}

hipError_t launch_bf16_add(const shelf::bf16_t* a, const shelf::bf16_t* b, shelf::bf16_t* out,
                           uint32_t n, hipStream_t stream) {
    bf16_add<<<(n + 255u) / 256u, 256, 0, stream>>>(a, b, out, n);
    return hipGetLastError();
}

hipError_t launch_f32_argmax(const float* logits, uint32_t n, uint32_t* out_tok,
                             hipStream_t stream) {
    f32_argmax<<<1, 1024, 0, stream>>>(logits, n, out_tok);
    return hipGetLastError();
}

// dtype conversions for activation streams
extern "C" __global__ __launch_bounds__(256)
void bf16_to_f32_k(const shelf::bf16_t* __restrict__ in, float* __restrict__ out, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __bfloat162float(__ushort_as_bfloat16(in[i].data));
}

extern "C" __global__ __launch_bounds__(256)
void f32_to_bf16_k(const float* __restrict__ in, shelf::bf16_t* __restrict__ out, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(in[i]))};
}

hipError_t launch_bf16_to_f32(const shelf::bf16_t* in, float* out, uint32_t n,
                              hipStream_t stream) {
    bf16_to_f32_k<<<(n + 255u) / 256u, 256, 0, stream>>>(in, out, n);
    return hipGetLastError();
}

hipError_t launch_f32_to_bf16(const float* in, shelf::bf16_t* out, uint32_t n,
                              hipStream_t stream) {
    f32_to_bf16_k<<<(n + 255u) / 256u, 256, 0, stream>>>(in, out, n);
    return hipGetLastError();
}

// ---- T17: MoE の接着剤 --------------------------------------------------
// ルーター logits: 1 ブロック = 1 エキスパート。H をスレッドで分担して
// 合計する（決定的な順序: スレッド i が h = i, i+256, ... を足す）。
extern "C" __global__ __launch_bounds__(256)
void moe_router_logit(const shelf::bf16_t* __restrict__ x,
                      const shelf::bf16_t* __restrict__ w, float* __restrict__ out,
                      uint32_t hidden) {
    __shared__ float red[256];
    const uint32_t e = blockIdx.x;
    const shelf::bf16_t* row = w + static_cast<size_t>(e) * hidden;
    float acc = 0.0f;
    for (uint32_t h = threadIdx.x; h < hidden; h += blockDim.x) {
        acc += __bfloat162float(__ushort_as_bfloat16(x[h].data)) *
               __bfloat162float(__ushort_as_bfloat16(row[h].data));
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2u; s > 0u; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0u) out[e] = red[0];
}

// acc[h] += scale * x[h]（f32 累積に bf16 のエキスパート出力を足す）。
extern "C" __global__ __launch_bounds__(256)
void moe_accumulate(float* __restrict__ acc, const shelf::bf16_t* __restrict__ x,
                    float scale, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    acc[i] += scale * __bfloat162float(__ushort_as_bfloat16(x[i].data));
}

// out[h] = scale * x[h]（f32 の共有エキスパート出力を bf16 にする経路）。
extern "C" __global__ __launch_bounds__(256)
void scaled_f32_to_bf16(const float* __restrict__ in, shelf::bf16_t* __restrict__ out,
                        float scale, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(in[i] * scale))};
}

hipError_t launch_moe_router_logit(const shelf::bf16_t* x, const shelf::bf16_t* w,
                                   float* out, uint32_t experts, uint32_t hidden,
                                   hipStream_t stream) {
    if (experts == 0u || hidden == 0u) return hipErrorInvalidValue;
    moe_router_logit<<<experts, 256, 0, stream>>>(x, w, out, hidden);
    return hipGetLastError();
}

hipError_t launch_moe_accumulate(float* acc, const shelf::bf16_t* x, float scale,
                                 uint32_t n, hipStream_t stream) {
    moe_accumulate<<<(n + 255u) / 256u, 256, 0, stream>>>(acc, x, scale, n);
    return hipGetLastError();
}

hipError_t launch_scaled_f32_to_bf16(const float* in, shelf::bf16_t* out, float scale,
                                     uint32_t n, hipStream_t stream) {
    scaled_f32_to_bf16<<<(n + 255u) / 256u, 256, 0, stream>>>(in, out, scale, n);
    return hipGetLastError();
}

// acc[h] += in[h]（f32。CPU で計算したエキスパートの寄与を足す）。
extern "C" __global__ __launch_bounds__(256)
void moe_add_f32(float* __restrict__ acc, const float* __restrict__ in, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    acc[i] += in[i];
}

hipError_t launch_moe_add_f32(float* acc, const float* in, uint32_t n, hipStream_t stream) {
    moe_add_f32<<<(n + 255u) / 256u, 256, 0, stream>>>(acc, in, n);
    return hipGetLastError();
}


} // namespace shelf::hipk
