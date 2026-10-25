// shelf backend/hip — small auxiliary bf16 ops for the GPU step executor.
// (vendored PhaseShift kernels cover the heavy ops; these glue the IR kinds)
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

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

} // namespace shelf::hipk
