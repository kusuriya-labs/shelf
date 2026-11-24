#include <shelf/runtime/tp/tp_argmax.h>
#include <shelf/core/memory/types.h>

#include "tp_transport_internal.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace shelf::runtime {

namespace {

constexpr uint32_t kPackThreads = 256u;

__global__ void tp_argmax_pack_kernel(const float* logits, TpArgmaxPair* pairs, uint32_t rows,
                                      uint32_t row_stride, uint32_t local_cols,
                                      uint32_t local_offset) {
    __shared__ float shared_value[kPackThreads];
    __shared__ int shared_id[kPackThreads];
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const uint32_t tid = threadIdx.x;
    const float* src = logits + static_cast<uint64_t>(row) * row_stride;
    float best = -3.4e38f;
    int best_id = INT_MAX;
    for (uint32_t c = blockIdx.y * blockDim.x + tid; c < local_cols; c += gridDim.y * blockDim.x) {
        const float value = src[c];
        const int id = static_cast<int>(local_offset + c);
        if (value > best || (value == best && (best_id < 0 || id < best_id))) {
            best = value;
            best_id = id;
        }
    }
    shared_value[tid] = best;
    shared_id[tid] = best_id;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
        if (tid < stride) {
            const float other = shared_value[tid + stride];
            const int other_id = shared_id[tid + stride];
            if (other > shared_value[tid] ||
                (other == shared_value[tid] && other_id < shared_id[tid])) {
                shared_value[tid] = other;
                shared_id[tid] = other_id;
            }
        }
        __syncthreads();
    }
    if (tid == 0u) {
        pairs[static_cast<size_t>(blockIdx.y) * rows + row] =
            {shared_value[0], shared_id[0] == INT_MAX ? -1 : shared_id[0]};
    }
}

__global__ void tp_argmax_finish_pack(const TpArgmaxPair* partials, TpArgmaxPair* pairs, uint32_t rows) {
    const auto row = blockIdx.x;
    const auto lane = threadIdx.x;
    float best = -3.4e38f;
    int id = INT_MAX;
    for (uint32_t part = lane; part < kTpArgmaxPartitions; part += 32u) {
        const auto p = partials[static_cast<size_t>(part) * rows + row];
        if (p.id >= 0 && (p.value > best || (p.value == best && p.id < id))) {
            best = p.value;
            id = p.id;
        }
    }
    for (unsigned offset = 16; offset != 0; offset >>= 1) {
        const float other = __shfl_down(best, offset, 32);
        const int other_id = __shfl_down(id, offset, 32);
        if (other > best || (other == best && other_id < id)) {
            best = other;
            id = other_id;
        }
    }
    if (lane == 0u) pairs[row] = {best, id == INT_MAX ? -1 : id};
}

__global__ void tp_argmax_reduce_kernel(const TpArgmaxPair* pairs, uint32_t pair_stride,
                                        uint32_t ranks, uint32_t rows, int32_t* out_ids) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float best = -3.4e38f;
    int best_id = INT_MAX;
    for (uint32_t r = 0u; r < ranks; ++r) {
        const TpArgmaxPair pair = pairs[static_cast<uint64_t>(r) * pair_stride + row];
        if (pair.id >= 0 && (pair.value > best || (pair.value == best && pair.id < best_id))) {
            best = pair.value;
            best_id = pair.id;
        }
    }
    out_ids[row] = best_id == INT_MAX ? -1 : best_id;
}

__global__ void tp_sum_f32_to_bf16_4_kernel(uint16_t* dst, const float* s0, const float* s1,
                                            const float* s2, const float* s3, uint64_t count) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += stride) {
        dst[i] = __bfloat16_as_ushort(__float2bfloat16(s0[i] + s1[i] + s2[i] + s3[i]));
    }
}

uint32_t grid_for(std::size_t count, uint32_t threads) {
    const uint64_t blocks = (static_cast<uint64_t>(count) + threads - 1u) / threads;
    return static_cast<uint32_t>(std::min<uint64_t>(blocks, 65535u));
}

}

bool tp_argmax_enabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("SHELF_TP_ARGMAX");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

bool tp_greedy_argmax_eligible(const TpGreedyProbe& probe) noexcept {
    return probe.temperature == 0.0f && probe.top_p == 1.0f && probe.min_p == 0.0f &&
           probe.top_k == 0u && !probe.has_token_bitmask;
}

void tp_argmax_pack_host(const float* logits, uint32_t rows, uint32_t row_stride,
                         uint32_t local_cols, uint32_t local_offset, TpArgmaxPair* pairs) {
    for (uint32_t row = 0u; row < rows; ++row) {
        const float* src = logits + static_cast<uint64_t>(row) * row_stride;
        float best = -3.4e38f;
        int best_id = -1;
        for (uint32_t c = 0u; c < local_cols; ++c) {
            const int id = static_cast<int>(local_offset + c);
            if (src[c] > best || (src[c] == best && (best_id < 0 || id < best_id))) {
                best = src[c];
                best_id = id;
            }
        }
        pairs[row].value = best;
        pairs[row].id = best_id;
    }
}

void tp_argmax_reduce_host(const TpArgmaxPair* pairs, uint32_t pair_stride, uint32_t ranks,
                           uint32_t rows, int32_t* out_ids) {
    for (uint32_t row = 0u; row < rows; ++row) {
        float best = -3.4e38f;
        int best_id = -1;
        for (uint32_t r = 0u; r < ranks; ++r) {
            const TpArgmaxPair pair = pairs[static_cast<uint64_t>(r) * pair_stride + row];
            if (pair.id < 0) continue;
            if (pair.value > best || (pair.value == best && (best_id < 0 || pair.id < best_id))) {
                best = pair.value;
                best_id = pair.id;
            }
        }
        out_ids[row] = best_id;
    }
}

void tp_sum_bf16_4_host(uint16_t* dst, const uint16_t* s0, const uint16_t* s1,
                        const uint16_t* s2, const uint16_t* s3, std::size_t count) {
    for (std::size_t i = 0u; i < count; ++i) {
        const float a = tp_bf16_bits_to_f32(s0[i]);
        const float b = tp_bf16_bits_to_f32(s1[i]);
        const float c = tp_bf16_bits_to_f32(s2[i]);
        const float d = tp_bf16_bits_to_f32(s3[i]);
        dst[i] = f32_to_bf16_rne(a + b + c + d);
    }
}

void tp_sum_f32_4_host(float* dst, const float* s0, const float* s1, const float* s2,
                       const float* s3, std::size_t count) {
    for (std::size_t i = 0u; i < count; ++i) dst[i] = s0[i] + s1[i] + s2[i] + s3[i];
}

hipError_t launch_tp_argmax_pack(const float* logits, TpArgmaxPair* pairs, uint32_t rows,
                                 uint32_t row_stride, uint32_t local_cols,
                                 uint32_t local_offset, hipStream_t stream, TpArgmaxPair* partials) {
    if (rows == 0u) return hipSuccess;
    if (logits == nullptr || pairs == nullptr || local_cols == 0u || row_stride < local_cols ||
        static_cast<uint64_t>(local_offset) + local_cols > static_cast<uint64_t>(INT_MAX)) return hipErrorInvalidValue;
    const bool split = partials != nullptr && local_cols >= kTpArgmaxPartitions * kPackThreads;
    const uint32_t parts = split ? kTpArgmaxPartitions : 1u;
    tp_argmax_pack_kernel<<<dim3(rows, parts), kPackThreads, 0, stream>>>(
        logits, split ? partials : pairs, rows, row_stride, local_cols, local_offset);
    if (split) {
        const auto err = hipGetLastError();
        if (err != hipSuccess) return err;
        tp_argmax_finish_pack<<<rows, 32, 0, stream>>>(partials, pairs, rows);
    }
    return hipGetLastError();
}

hipError_t launch_tp_argmax_reduce(const TpArgmaxPair* pairs, uint32_t pair_stride,
                                   uint32_t ranks, uint32_t rows, int32_t* out_ids,
                                   hipStream_t stream) {
    if (rows == 0u) return hipSuccess;
    if (pairs == nullptr || out_ids == nullptr || ranks == 0u || pair_stride < rows) return hipErrorInvalidValue;
    const uint32_t threads = 256u;
    const uint32_t blocks = grid_for(rows, threads);
    tp_argmax_reduce_kernel<<<blocks, threads, 0, stream>>>(pairs, pair_stride, ranks, rows,
                                                            out_ids);
    return hipGetLastError();
}

hipError_t launch_tp_sum_f32_to_bf16_4(uint16_t* dst, const float* s0, const float* s1,
                                       const float* s2, const float* s3, std::size_t count,
                                       hipStream_t stream) {
    if (count == 0u) return hipSuccess;
    if (dst == nullptr || s0 == nullptr || s1 == nullptr || s2 == nullptr || s3 == nullptr)
        return hipErrorInvalidValue;
    const uint32_t threads = 256u;
    tp_sum_f32_to_bf16_4_kernel<<<grid_for(count, threads), threads, 0, stream>>>(
        dst, s0, s1, s2, s3, static_cast<uint64_t>(count));
    return hipGetLastError();
}

}
