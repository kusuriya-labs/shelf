#include <shelf/kernels/optimized/sampling.h>
#include <shelf/kernels/optimized/sampling_common.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cmath>

namespace shelf::kernel {
namespace {

constexpr uint32_t kThreads = 256u;
constexpr float kNegInf = -3.4e38f;
constexpr uint32_t kNoToken = 0xFFFFFFFFu;

__device__ __forceinline__
bool argmax_better(float cv, uint32_t ct, float bv, uint32_t bt) {
    return cv > bv || (cv == bv && ct < bt);
}

__device__ __forceinline__
void argmax_update(float cv, uint32_t ct, float& bv, uint32_t& bt) {
    if (argmax_better(cv, ct, bv, bt)) {
        bv = cv;
        bt = ct;
    }
}

__device__ __forceinline__
void argmax_reduce_block(float& best_value, uint32_t& best_token) {
    for (uint32_t off = 16u; off > 0u; off >>= 1u) {
        const float ov = __shfl_down(best_value, off, 32);
        const uint32_t ot = __shfl_down(best_token, off, 32);
        argmax_update(ov, ot, best_value, best_token);
    }
    __shared__ float s_vals[8];
    __shared__ uint32_t s_toks[8];
    const uint32_t wave = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    if (lane == 0u) {
        s_vals[wave] = best_value;
        s_toks[wave] = best_token;
    }
    __syncthreads();
    if (threadIdx.x < 32u) {
        const uint32_t i = threadIdx.x;
        best_value = i < 8u ? s_vals[i] : kNegInf;
        best_token = i < 8u ? s_toks[i] : kNoToken;
        for (uint32_t off = 16u; off > 0u; off >>= 1u) {
            const float ov = __shfl_down(best_value, off, 32);
            const uint32_t ot = __shfl_down(best_token, off, 32);
            argmax_update(ov, ot, best_value, best_token);
        }
    }
}

__device__ __forceinline__ void argmax_single_body(
                                   const float* __restrict__ logits,
                                   const ::shelf::runtime::DeviceSamplingParams* __restrict__ sampling,
                                   int32_t* __restrict__ sampled_tokens,
                                   uint32_t vocab_size,
                                   uint32_t logits_row_stride,
                                   const uint32_t* __restrict__ output_rows) {
    const uint32_t slot = blockIdx.x;
    const uint32_t row = output_rows != nullptr ? output_rows[slot] : slot;
    __shared__ uint32_t do_sample;
    if (threadIdx.x == 0u) {
        do_sample = (sampling[slot].mode != 0u) ? 1u : 0u;
        if (do_sample == 0u) sampled_tokens[row] = -1;
    }
    __syncthreads();
    if (do_sample == 0u) return;
    float best_value = kNegInf;
    uint32_t best_token = kNoToken;
    const float* row_logits = logits + static_cast<size_t>(slot) * logits_row_stride;
    for (uint32_t v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        argmax_update(row_logits[v], v, best_value, best_token);
    }
    argmax_reduce_block(best_value, best_token);
    if (threadIdx.x == 0u)
        sampled_tokens[row] =
            best_token == kNoToken ? -1 : static_cast<int32_t>(best_token);
}

__global__ void argmax_single_impl(const float* __restrict__ logits,
                                   const ::shelf::runtime::DeviceSamplingParams* __restrict__ sampling,
                                   int32_t* __restrict__ sampled_tokens,
                                   uint32_t vocab_size,
                                   uint32_t logits_row_stride) {
    argmax_single_body(logits, sampling, sampled_tokens, vocab_size,
                       logits_row_stride, nullptr);
}

__global__ void argmax_stage1_impl(const float* __restrict__ logits,
                                   const ::shelf::runtime::DeviceSamplingParams* __restrict__ sampling,
                                   ArgMaxPair* __restrict__ partials,
                                   uint32_t vocab_size, uint32_t partitions,
                                   uint32_t logits_row_stride) {
    const uint32_t task = blockIdx.x;
    const uint32_t row = task / partitions;
    const uint32_t partition = task % partitions;
    if (sampling[row].mode == 0u) return;
    const uint32_t begin = static_cast<uint32_t>(
        (static_cast<uint64_t>(vocab_size) * partition) / partitions);
    const uint32_t end = static_cast<uint32_t>(
        (static_cast<uint64_t>(vocab_size) * (partition + 1u)) / partitions);
    float best_value = kNegInf;
    uint32_t best_token = kNoToken;
    const float* row_logits = logits + static_cast<size_t>(row) * logits_row_stride;
    for (uint32_t v = begin + threadIdx.x; v < end; v += blockDim.x) {
        argmax_update(row_logits[v], v, best_value, best_token);
    }
    argmax_reduce_block(best_value, best_token);
    if (threadIdx.x == 0u)
        partials[static_cast<size_t>(row) * partitions + partition] =
            ArgMaxPair{best_value, best_token};
}

__global__ void argmax_stage2_impl(const ::shelf::runtime::DeviceSamplingParams* __restrict__ sampling,
                                   int32_t* __restrict__ sampled_tokens,
                                   const ArgMaxPair* __restrict__ partials,
                                   uint32_t partitions) {
    const uint32_t row = blockIdx.x;
    __shared__ uint32_t do_sample;
    if (threadIdx.x == 0u) {
        do_sample = (sampling[row].mode != 0u) ? 1u : 0u;
        if (do_sample == 0u) sampled_tokens[row] = -1;
    }
    __syncthreads();
    if (do_sample == 0u) return;
    float best_value = kNegInf;
    uint32_t best_token = kNoToken;
    const ArgMaxPair* row_partials = partials + static_cast<size_t>(row) * partitions;
    for (uint32_t p = threadIdx.x; p < partitions; p += blockDim.x)
        argmax_update(row_partials[p].value, row_partials[p].token, best_value,
                      best_token);
    argmax_reduce_block(best_value, best_token);
    if (threadIdx.x == 0u) sampled_tokens[row] = static_cast<int32_t>(best_token);
}

__device__ __forceinline__
void top2_update(float c, uint32_t t, float& v1, uint32_t& t1, float& v2, uint32_t& t2) {
    if (argmax_better(c, t, v1, t1)) {
        v2 = v1;
        t2 = t1;
        v1 = c;
        t1 = t;
    } else if (t != t1 && argmax_better(c, t, v2, t2)) {
        v2 = c;
        t2 = t;
    }
}

__device__ __forceinline__
void top2_merge(float& v1, uint32_t& t1, float& v2, uint32_t& t2,
                float ov1, uint32_t ot1, float ov2, uint32_t ot2) {
    float wv1, wv2, lv1, lv2;
    uint32_t wt1, wt2, lt1, lt2;
    if (argmax_better(ov1, ot1, v1, t1)) {
        wv1 = ov1; wt1 = ot1; wv2 = ov2; wt2 = ot2;
        lv1 = v1; lt1 = t1; lv2 = v2; lt2 = t2;
    } else {
        wv1 = v1; wt1 = t1; wv2 = v2; wt2 = t2;
        lv1 = ov1; lt1 = ot1; lv2 = ov2; lt2 = ot2;
    }
    v1 = wv1;
    t1 = wt1;
    if (lt1 != wt1 && argmax_better(lv1, lt1, wv2, wt2)) {
        v2 = lv1;
        t2 = lt1;
    } else {
        v2 = wv2;
        t2 = wt2;
    }
    (void)lv2;
    (void)lt2;
}

__global__ void argmax_top2_impl(const float* __restrict__ logits,
                                 const ::shelf::runtime::DeviceSamplingParams* __restrict__ sampling,
                                 ArgMaxTop2* __restrict__ out,
                                 uint32_t vocab_size,
                                 uint32_t logits_row_stride) {
    const uint32_t row = blockIdx.x;
    __shared__ uint32_t do_sample;
    if (threadIdx.x == 0u) {
        do_sample = (sampling == nullptr || sampling[row].mode != 0u) ? 1u : 0u;
        if (do_sample == 0u) out[row] = ArgMaxTop2{};
    }
    __syncthreads();
    if (do_sample == 0u) return;
    float v1 = kNegInf, v2 = kNegInf;
    uint32_t t1 = kNoToken, t2 = kNoToken;
    const float* row_logits = logits + static_cast<size_t>(row) * logits_row_stride;
    for (uint32_t v = threadIdx.x; v < vocab_size; v += blockDim.x)
        top2_update(row_logits[v], v, v1, t1, v2, t2);
    for (uint32_t off = 16u; off > 0u; off >>= 1u) {
        const float ov1 = __shfl_down(v1, off, 32);
        const uint32_t ot1 = __shfl_down(t1, off, 32);
        const float ov2 = __shfl_down(v2, off, 32);
        const uint32_t ot2 = __shfl_down(t2, off, 32);
        top2_merge(v1, t1, v2, t2, ov1, ot1, ov2, ot2);
    }
    __shared__ ArgMaxTop2 s[8];
    const uint32_t wave = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    if (lane == 0u) {
        s[wave] = ArgMaxTop2{v1, t1, v2, t2};
    }
    __syncthreads();
    if (threadIdx.x == 0u) {
        float rv1 = kNegInf, rv2 = kNegInf;
        uint32_t rt1 = kNoToken, rt2 = kNoToken;
        for (uint32_t w = 0u; w < 8u; ++w)
            top2_merge(rv1, rt1, rv2, rt2, s[w].v1, s[w].t1, s[w].v2, s[w].t2);
        out[row] = ArgMaxTop2{rv1, rt1, rv2, rt2};
    }
}

__global__ void apply_token_bitmask_impl(
    float* logits, const ::shelf::runtime::DeviceSamplingParams* sampling,
    uint32_t vocab, uint32_t stride) {
    const uint32_t row = blockIdx.y;
    const uint32_t token = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t* mask = sampling[row].token_bitmask;
    if (token < vocab && mask != nullptr && (mask[token >> 5] & (1u << (token & 31u))) == 0u) {
        logits[static_cast<size_t>(row) * stride + token] = -INFINITY;
    }
}

}  // namespace

hipError_t launch_apply_token_bitmask_f32(
    float* logits, const ::shelf::runtime::DeviceSamplingParams* sampling,
    uint32_t outputs, uint32_t vocab_size, uint32_t row_stride, hipStream_t stream) {
    if (outputs == 0u) return hipSuccess;
    if (logits == nullptr || sampling == nullptr || vocab_size == 0u || row_stride < vocab_size)
        return hipErrorInvalidValue;
    hipLaunchKernelGGL(apply_token_bitmask_impl,
        dim3((vocab_size + kThreads - 1u) / kThreads, outputs), dim3(kThreads),
        0, stream, logits, sampling, vocab_size, row_stride);
    return hipGetLastError();
}

hipError_t launch_argmax_f32_top2(
    const float* logits,
    const ::shelf::runtime::DeviceSamplingParams* sampling,
    ArgMaxTop2* out,
    uint32_t outputs,
    uint32_t vocab_size,
    uint32_t logits_row_stride,
    hipStream_t stream) {
    if (outputs == 0u) return hipSuccess;
    if (vocab_size == 0u) return hipErrorInvalidValue;
    if (logits == nullptr || out == nullptr) return hipErrorInvalidValue;
    argmax_top2_impl<<<outputs, kThreads, 0, stream>>>(
        logits, sampling, out, vocab_size, logits_row_stride);
    return hipGetLastError();
}

hipError_t launch_argmax_f32_single(const ArgmaxF32Args& args,
                                            hipStream_t stream) {
    if (args.outputs == 0u) return hipSuccess;
    if (args.vocab_size == 0u) return hipErrorInvalidValue;
    if (args.logits == nullptr || args.sampling == nullptr ||
        args.sampled_tokens == nullptr)
        return hipErrorInvalidValue;
    argmax_single_impl<<<args.outputs, kThreads, 0, stream>>>(
        args.logits, args.sampling, args.sampled_tokens, args.vocab_size,
        args.logits_row_stride);
    return hipGetLastError();
}

hipError_t launch_argmax_f32_partitioned(
    const ArgmaxF32Args& args, ArgMaxPair* partials, uint32_t partial_capacity,
    uint32_t partitions, hipStream_t stream) {
    if (args.outputs == 0u) return hipSuccess;
    if (args.vocab_size == 0u || partitions == 0u) return hipErrorInvalidValue;
    if (args.logits == nullptr || args.sampling == nullptr ||
        args.sampled_tokens == nullptr || partials == nullptr)
        return hipErrorInvalidValue;
    if (static_cast<uint64_t>(args.outputs) * partitions > partial_capacity)
        return hipErrorInvalidValue;
    const uint32_t stage1_grid = args.outputs * partitions;
    argmax_stage1_impl<<<stage1_grid, kThreads, 0, stream>>>(
        args.logits, args.sampling, partials, args.vocab_size, partitions,
        args.logits_row_stride);
    argmax_stage2_impl<<<args.outputs, kThreads, 0, stream>>>(
        args.sampling, args.sampled_tokens, partials, partitions);
    return hipGetLastError();
}

}  // namespace shelf::kernel
