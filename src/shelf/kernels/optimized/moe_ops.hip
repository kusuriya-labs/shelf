#include <shelf/kernels/optimized/moe_ops.h>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace ps::kernel {

namespace {

constexpr uint32_t kThreads = kMoeOpsBlock;
constexpr uint32_t kRouterMaxExperts = 512u;
constexpr uint32_t kExpertsMaxIntermediate = 1024u;

__device__ __forceinline__ float block_reduce_max_f32(float v, float* red) {
    red[threadIdx.x] = v;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2u; s != 0u; s >>= 1u) {
        if (threadIdx.x < s)
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    return red[0];
}

__device__ __forceinline__ float block_reduce_sum_f32(float v, float* red) {
    red[threadIdx.x] = v;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2u; s != 0u; s >>= 1u) {
        if (threadIdx.x < s)
            red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    return red[0];
}

__global__ void __launch_bounds__(kThreads)
moe_router_f32_impl(const float* __restrict__ x, uint32_t x_row_stride,
                    const __hip_bfloat16* __restrict__ w,
                    uint32_t experts, uint32_t hidden,
                    uint32_t top_k, bool renorm,
                    int32_t* __restrict__ top_idx, float* __restrict__ top_w) {
    __shared__ float probs[kRouterMaxExperts];
    __shared__ float red[kThreads];
    const uint32_t row = blockIdx.x;
    const float* xrow = x + static_cast<size_t>(row) * x_row_stride;
    for (uint32_t e = threadIdx.x; e < experts; e += blockDim.x) {
        const __hip_bfloat16* wrow = w + static_cast<size_t>(e) * hidden;
        float acc = 0.f;
        for (uint32_t h = 0; h < hidden; ++h)
            acc += xrow[h] * __bfloat162float(wrow[h]);
        probs[e] = acc;
    }
    __syncthreads();
    float mx = -INFINITY;
    for (uint32_t e = threadIdx.x; e < experts; e += blockDim.x)
        mx = fmaxf(mx, probs[e]);
    const float row_max = block_reduce_max_f32(mx, red);
    for (uint32_t e = threadIdx.x; e < experts; e += blockDim.x)
        probs[e] = __expf(probs[e] - row_max);
    __syncthreads();
    float partial = 0.f;
    for (uint32_t e = threadIdx.x; e < experts; e += blockDim.x)
        partial += probs[e];
    const float inv_sum = 1.0f / block_reduce_sum_f32(partial, red);
    for (uint32_t e = threadIdx.x; e < experts; e += blockDim.x)
        probs[e] *= inv_sum;
    __syncthreads();
    if (threadIdx.x == 0u) {
        int32_t* idx_row = top_idx + static_cast<size_t>(row) * top_k;
        float* wgt_row = top_w + static_cast<size_t>(row) * top_k;
        for (uint32_t s = 0; s < top_k; ++s) {
            uint32_t best = 0u;
            float best_v = -1.f;
            for (uint32_t e = 0; e < experts; ++e) {
                const float v = probs[e];
                if (v < 0.f) continue;
                if (v > best_v) {
                    best_v = v;
                    best = e;
                }
            }
            probs[best] = -1.f;
            idx_row[s] = static_cast<int32_t>(best);
            wgt_row[s] = best_v;
        }
        if (renorm) {
            float wsum = 0.f;
            for (uint32_t s = 0; s < top_k; ++s) wsum += wgt_row[s];
            for (uint32_t s = 0; s < top_k; ++s) wgt_row[s] /= wsum;
        }
    }
}

__global__ void __launch_bounds__(kThreads)
moe_experts_f32_impl(const float* __restrict__ x, uint32_t x_row_stride,
                     const __hip_bfloat16* __restrict__ gate_up,
                     const __hip_bfloat16* __restrict__ down,
                     uint32_t hidden, uint32_t inter, uint32_t experts,
                     const int32_t* __restrict__ top_idx,
                     const float* __restrict__ top_w, uint32_t top_k,
                     float* __restrict__ out, uint32_t out_row_stride) {
    __shared__ float gate[kExpertsMaxIntermediate];
    __shared__ float up[kExpertsMaxIntermediate];
    const uint32_t row = blockIdx.x;
    const float* xrow = x + static_cast<size_t>(row) * x_row_stride;
    const int32_t* idx_row = top_idx + static_cast<size_t>(row) * top_k;
    const float* wgt_row = top_w + static_cast<size_t>(row) * top_k;
    float* out_row = out + static_cast<size_t>(row) * out_row_stride;
    for (uint32_t h = threadIdx.x; h < hidden; h += blockDim.x) out_row[h] = 0.f;
    __syncthreads();
    const size_t expert_stride = static_cast<size_t>(2u * inter) * hidden;
    const size_t down_stride = static_cast<size_t>(hidden) * inter;
    for (uint32_t s = 0; s < top_k; ++s) {
        const int32_t e = idx_row[s];
        const float wgt = wgt_row[s];
        const __hip_bfloat16* gu = gate_up + static_cast<size_t>(e) * expert_stride;
        const __hip_bfloat16* dn = down + static_cast<size_t>(e) * down_stride;
        for (uint32_t i = threadIdx.x; i < inter; i += blockDim.x) {
            const __hip_bfloat16* grow = gu + static_cast<size_t>(i) * hidden;
            const __hip_bfloat16* urow =
                gu + static_cast<size_t>(inter + i) * hidden;
            float gacc = 0.f;
            float uacc = 0.f;
            for (uint32_t h = 0; h < hidden; ++h) {
                const float xv = xrow[h];
                gacc += xv * __bfloat162float(grow[h]);
                uacc += xv * __bfloat162float(urow[h]);
            }
            gate[i] = gacc;
            up[i] = uacc;
        }
        __syncthreads();
        for (uint32_t h = threadIdx.x; h < hidden; h += blockDim.x) {
            const __hip_bfloat16* drow = dn + static_cast<size_t>(h) * inter;
            float acc = 0.f;
            for (uint32_t i = 0; i < inter; ++i) {
                const float g = gate[i];
                acc += (g * (1.0f / (1.0f + __expf(-g)))) * up[i] *
                       __bfloat162float(drow[i]);
            }
            out_row[h] += wgt * acc;
        }
        __syncthreads();
    }
    (void)experts;
}

}  // namespace

hipError_t launch_moe_router_f32(
    const float* x, uint32_t x_row_stride,
    const void* router_w,
    uint32_t experts, uint32_t hidden,
    uint32_t top_k, bool renorm,
    int32_t* top_idx, float* top_w,
    uint32_t rows, hipStream_t stream) {
    if (x == nullptr || router_w == nullptr || top_idx == nullptr || top_w == nullptr)
        return hipErrorInvalidValue;
    if (rows == 0u || hidden == 0u) return hipErrorInvalidValue;
    if (x_row_stride < hidden) return hipErrorInvalidValue;
    if (experts == 0u || experts > kRouterMaxExperts) return hipErrorNotSupported;
    if (top_k == 0u || top_k > experts) return hipErrorNotSupported;
    moe_router_f32_impl<<<rows, kThreads, 0, stream>>>(
        x, x_row_stride, static_cast<const __hip_bfloat16*>(router_w),
        experts, hidden, top_k, renorm, top_idx, top_w);
    return hipGetLastError();
}

hipError_t launch_moe_experts_f32(
    const float* x, uint32_t x_row_stride,
    const void* gate_up,
    const void* down,
    uint32_t hidden, uint32_t inter, uint32_t experts,
    const int32_t* top_idx, const float* top_w, uint32_t top_k,
    float* out, uint32_t out_row_stride,
    uint32_t rows, hipStream_t stream) {
    if (x == nullptr || gate_up == nullptr || down == nullptr ||
        top_idx == nullptr || top_w == nullptr || out == nullptr)
        return hipErrorInvalidValue;
    if (rows == 0u || hidden == 0u || inter == 0u) return hipErrorInvalidValue;
    if (x_row_stride < hidden || out_row_stride < hidden) return hipErrorInvalidValue;
    if (experts == 0u || top_k == 0u || top_k > experts) return hipErrorNotSupported;
    if (inter > kExpertsMaxIntermediate) return hipErrorNotSupported;
    moe_experts_f32_impl<<<rows, kThreads, 0, stream>>>(
        x, x_row_stride, static_cast<const __hip_bfloat16*>(gate_up),
        static_cast<const __hip_bfloat16*>(down), hidden, inter, experts,
        top_idx, top_w, top_k, out, out_row_stride);
    return hipGetLastError();
}

}
