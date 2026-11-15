#include <shelf/kernels/fused/qsa_key_cache.h>
#include <shelf/kernels/fused/detail/qsa_common.h>
#include <shelf/kernels/optimized/qsa.h>

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace shelf::kernel {
namespace {
using namespace detail::qsa;

constexpr uint32_t kThreads = kQsaThreads;

__global__ __launch_bounds__(kQsaHeadDim) void qsa_key_cache_batch_kernel(
    QsaKeyCacheBatchArgs args) {
    const uint32_t d = threadIdx.x;
    __shared__ float scratch[kQsaHeadDim];
    __shared__ float normed[kQsaHeadDim];
    const ::shelf::runtime::DeviceRequestDescriptor request = args.requests[blockIdx.x];
    const uint32_t slot = request.request_handle.slot;
    if (slot >= args.max_sequences) return;
    bf16_t* pooled = args.pooled_base +
                     static_cast<std::uint64_t>(slot) * args.pooled_slot_stride +
                     static_cast<std::uint64_t>(args.state_index) *
                         args.pooled_layer_stride;
    bf16_t* partial = args.partial_base +
                      static_cast<std::uint64_t>(slot) * args.partial_slot_stride +
                      static_cast<std::uint64_t>(args.state_index) *
                          args.partial_layer_stride;
    uint32_t* pool_count_ptr =
        args.pool_count_base +
        static_cast<std::uint64_t>(slot) * args.count_slot_stride +
        static_cast<std::uint64_t>(args.state_index) * args.count_layer_stride;
    uint32_t* partial_count_ptr =
        args.partial_count_base +
        static_cast<std::uint64_t>(slot) * args.count_slot_stride +
        static_cast<std::uint64_t>(args.state_index) * args.count_layer_stride;

    uint32_t partial_count = *partial_count_ptr;
    uint32_t pool_count = *pool_count_ptr;
    for (uint32_t t = 0; t < request.row_count; ++t) {
        const uint32_t row = request.row_begin + t;
        if (row >= args.actual_rows) break;
        if (args.partial_history != nullptr && row < args.history_rows) {
            bf16_t* hp = args.partial_history +
                static_cast<std::uint64_t>(args.state_index) *
                    args.history_partial_state_stride +
                static_cast<std::uint64_t>(row) * args.compress_ratio * args.head_dim;
            for (uint32_t i = 0; i < args.compress_ratio; ++i) {
                hp[static_cast<std::uint64_t>(i) * args.head_dim + d] =
                    partial[static_cast<std::uint64_t>(i) * args.head_dim + d];
            }
            if (d == 0u && args.count_history != nullptr) {
                uint32_t* hc = args.count_history +
                    static_cast<std::uint64_t>(args.state_index) *
                        args.history_count_state_stride +
                    static_cast<std::uint64_t>(row) * 2u;
                hc[0] = pool_count;
                hc[1] = partial_count;
            }
        }
        const bf16_t* raw =
            args.raw_keys + static_cast<std::uint64_t>(row) * args.raw_row_stride;
        partial[static_cast<std::uint64_t>(partial_count) * args.head_dim + d] = raw[d];
        ++partial_count;
        __syncthreads();
        if (partial_count == args.compress_ratio) {
            float mean = 0.0f;
            for (uint32_t i = 0; i < args.compress_ratio; ++i) {
                mean += bf16_value(partial +
                                   static_cast<std::uint64_t>(i) * args.head_dim + d);
            }
            mean /= static_cast<float>(args.compress_ratio);
            const float square = block_sum(scratch, mean * mean);
            __syncthreads();
            const float inv =
                rsqrtf(square / static_cast<float>(args.head_dim) + args.eps);
            normed[d] = mean * inv * (1.0f + bf16_value(args.norm_weight + d));
            __syncthreads();
            if (pool_count < args.max_blocks) {
                const uint32_t position = args.positions[row];
                const uint32_t start = position >= args.compress_ratio - 1u
                                           ? position - (args.compress_ratio - 1u)
                                           : 0u;
                bf16_t* dst =
                    pooled + static_cast<std::uint64_t>(pool_count) * args.head_dim;
                const uint32_t half = args.rotary_dim / 2u;
                if (d < half) {
                    const float exponent = -2.0f * static_cast<float>(d) /
                                           static_cast<float>(args.rotary_dim);
                    const float frequency =
                        args.inv_freq != nullptr
                            ? static_cast<float>(args.inv_freq[d])
                            : powf(args.theta, exponent);
                    const float angle =
                        (static_cast<float>(start) / args.position_scale) * frequency;
                    const float cosine = cosf(angle);
                    const float sine = sinf(angle);
                    const float x0 = normed[d];
                    const float x1 = normed[d + half];
                    dst[d] = bf16_from(x0 * cosine - x1 * sine);
                } else if (d < args.rotary_dim) {
                    const uint32_t i = d - half;
                    const float exponent = -2.0f * static_cast<float>(i) /
                                           static_cast<float>(args.rotary_dim);
                    const float frequency =
                        args.inv_freq != nullptr
                            ? static_cast<float>(args.inv_freq[i])
                            : powf(args.theta, exponent);
                    const float angle =
                        (static_cast<float>(start) / args.position_scale) * frequency;
                    const float cosine = cosf(angle);
                    const float sine = sinf(angle);
                    const float x0 = normed[i];
                    const float x1 = normed[d];
                    dst[d] = bf16_from(x1 * cosine + x0 * sine);
                } else {
                    dst[d] = bf16_from(normed[d]);
                }
                ++pool_count;
            }
            partial_count = 0;
            __syncthreads();
        }
    }
    if (d == 0u) {
        *partial_count_ptr = partial_count;
        *pool_count_ptr = pool_count;
    }
}

__device__ __forceinline__ void qsa_rope_store(
    bf16_t* dst, const float* normed, uint32_t d, uint32_t rotary_dim,
    float start, float position_scale, const double* inv_freq, float theta) {
    const uint32_t half = rotary_dim / 2u;
    if (d < half) {
        const float exponent = -2.0f * static_cast<float>(d) /
                               static_cast<float>(rotary_dim);
        const float frequency = inv_freq != nullptr
                                    ? static_cast<float>(inv_freq[d])
                                    : powf(theta, exponent);
        const float angle = (start / position_scale) * frequency;
        const float cosine = cosf(angle);
        const float sine = sinf(angle);
        const float x0 = normed[d];
        const float x1 = normed[d + half];
        dst[d] = bf16_from(x0 * cosine - x1 * sine);
    } else if (d < rotary_dim) {
        const uint32_t i = d - half;
        const float exponent = -2.0f * static_cast<float>(i) /
                               static_cast<float>(rotary_dim);
        const float frequency = inv_freq != nullptr
                                    ? static_cast<float>(inv_freq[i])
                                    : powf(theta, exponent);
        const float angle = (start / position_scale) * frequency;
        const float cosine = cosf(angle);
        const float sine = sinf(angle);
        const float x0 = normed[i];
        const float x1 = normed[d];
        dst[d] = bf16_from(x1 * cosine + x0 * sine);
    } else {
        dst[d] = bf16_from(normed[d]);
    }
}

// Prefill form of the batch key cache: one CTA per completed compressed block.
// Each CTA reads its own rows from raw_keys, so no cross-block ordering is needed.
// The caller uses this only when partial_history is null (no speculative snapshots).
__global__ __launch_bounds__(kThreads) void qsa_key_cache_blocks_kernel(
    QsaKeyCacheBatchArgs args) {
    const uint32_t d = threadIdx.x;
    const ::shelf::runtime::DeviceRequestDescriptor request = args.requests[blockIdx.y];
    const uint32_t slot = request.request_handle.slot;
    if (slot >= args.max_sequences) return;
    const uint32_t row_begin = request.row_begin;
    const uint32_t row_end = row_begin + request.row_count < args.actual_rows
                                 ? row_begin + request.row_count
                                 : args.actual_rows;
    if (row_end <= row_begin) return;
    const uint32_t rows_here = row_end - row_begin;
    const uint32_t cr = args.compress_ratio;
    const uint32_t unit = blockIdx.x;
    if (cr < 2u || unit > (rows_here + cr - 1u) / cr + 1u) return;

    bf16_t* pooled = args.pooled_base +
        static_cast<std::uint64_t>(slot) * args.pooled_slot_stride +
        static_cast<std::uint64_t>(args.state_index) * args.pooled_layer_stride;
    bf16_t* partial = args.partial_base +
        static_cast<std::uint64_t>(slot) * args.partial_slot_stride +
        static_cast<std::uint64_t>(args.state_index) * args.partial_layer_stride;
    uint32_t* pool_count_ptr = args.pool_count_base +
        static_cast<std::uint64_t>(slot) * args.count_slot_stride +
        static_cast<std::uint64_t>(args.state_index) * args.count_layer_stride;
    uint32_t* partial_count_ptr = args.partial_count_base +
        static_cast<std::uint64_t>(slot) * args.count_slot_stride +
        static_cast<std::uint64_t>(args.state_index) * args.count_layer_stride;

    __shared__ float scratch[kThreads];
    __shared__ float normed[kQsaHeadDim];

    const uint32_t pending = *partial_count_ptr;
    const uint32_t pool_start = *pool_count_ptr;

    uint32_t head = 0u;
    uint32_t body = 0u;
    uint32_t tail = 0u;
    bool partial_only = false;
    if (pending != 0u && rows_here < cr - pending) {
        partial_only = true;
    } else {
        head = pending == 0u ? 0u : cr - pending;
        body = (rows_here - head) / cr;
        tail = rows_here - head - body * cr;
    }
    const uint32_t head_units = head != 0u ? 1u : 0u;
    const uint32_t body_end = head_units + body;
    const uint32_t tail_unit = head != 0u ? 0u : body_end;

    if (partial_only) {
        if (unit == 0u) {
            for (uint32_t i = 0u; i < rows_here; ++i)
                partial[(pending + i) * args.head_dim + d] =
                    args.raw_keys[static_cast<std::uint64_t>(row_begin + i) *
                                      args.raw_row_stride + d];
        }
        return;
    }

    if (head != 0u && unit == 0u) {
        float sum = 0.0f;
        for (uint32_t i = 0u; i < pending; ++i)
            sum += bf16_value(partial + static_cast<std::uint64_t>(i) * args.head_dim + d);
        for (uint32_t i = 0u; i < head; ++i)
            sum += bf16_value(args.raw_keys +
                              static_cast<std::uint64_t>(row_begin + i) *
                                  args.raw_row_stride + d);
        const float mean = sum / static_cast<float>(cr);
        const float square = block_sum(scratch, mean * mean);
        __syncthreads();
        const float inv = rsqrtf(square / static_cast<float>(args.head_dim) + args.eps);
        normed[d] = mean * inv * (1.0f + bf16_value(args.norm_weight + d));
        __syncthreads();
        if (pool_start < args.max_blocks) {
            const uint32_t position = args.positions[row_begin + head - 1u];
            const float start = position >= cr - 1u
                                    ? static_cast<float>(position - (cr - 1u))
                                    : 0.0f;
            qsa_rope_store(pooled + static_cast<std::uint64_t>(pool_start) * args.head_dim,
                           normed, d, args.rotary_dim, start, args.position_scale,
                           args.inv_freq, args.theta);
        }
    }

    if (unit >= head_units && unit < body_end) {
        const uint32_t k = unit - head_units;
        const uint32_t block_row = row_begin + head + k * cr;
        float sum = 0.0f;
        for (uint32_t i = 0u; i < cr; ++i)
            sum += bf16_value(args.raw_keys +
                              static_cast<std::uint64_t>(block_row + i) *
                                  args.raw_row_stride + d);
        const float mean = sum / static_cast<float>(cr);
        const float square = block_sum(scratch, mean * mean);
        __syncthreads();
        const float inv = rsqrtf(square / static_cast<float>(args.head_dim) + args.eps);
        normed[d] = mean * inv * (1.0f + bf16_value(args.norm_weight + d));
        __syncthreads();
        const uint32_t pool_index = pool_start + head_units + k;
        if (pool_index < args.max_blocks) {
            const uint32_t position = args.positions[block_row + cr - 1u];
            const float start = position >= cr - 1u
                                    ? static_cast<float>(position - (cr - 1u))
                                    : 0.0f;
            qsa_rope_store(pooled + static_cast<std::uint64_t>(pool_index) * args.head_dim,
                           normed, d, args.rotary_dim, start, args.position_scale,
                           args.inv_freq, args.theta);
        }
    }

    if (unit == tail_unit && (tail != 0u || (head_units + body) != 0u)) {
        const uint32_t tail_row = row_begin + head + body * cr;
        for (uint32_t i = 0u; i < tail; ++i)
            partial[i * args.head_dim + d] =
                args.raw_keys[static_cast<std::uint64_t>(tail_row + i) *
                                  args.raw_row_stride + d];
        if (body != 0u) {
            const uint32_t last_block_row = row_begin + head + (body - 1u) * cr;
            for (uint32_t i = tail; i < cr; ++i)
                partial[i * args.head_dim + d] =
                    args.raw_keys[static_cast<std::uint64_t>(last_block_row + i) *
                                      args.raw_row_stride + d];
        } else if (head != 0u) {
            for (uint32_t i = tail; i < cr; ++i) {
                if (i < pending) continue;
                partial[i * args.head_dim + d] =
                    args.raw_keys[static_cast<std::uint64_t>(row_begin + i - pending) *
                                      args.raw_row_stride + d];
            }
        }
    }
}

// Written after the block kernel: the counters are shared by every block CTA, so they
// cannot change while the block kernel is running.
__global__ void qsa_key_cache_counts_kernel(QsaKeyCacheBatchArgs args) {
    if (threadIdx.x != 0u) return;
    const ::shelf::runtime::DeviceRequestDescriptor request = args.requests[blockIdx.x];
    const uint32_t slot = request.request_handle.slot;
    if (slot >= args.max_sequences) return;
    const uint32_t row_begin = request.row_begin;
    const uint32_t row_end = row_begin + request.row_count < args.actual_rows
                                 ? row_begin + request.row_count
                                 : args.actual_rows;
    if (row_end <= row_begin) return;
    const uint32_t rows_here = row_end - row_begin;
    const uint32_t cr = args.compress_ratio;
    if (cr < 2u) return;
    uint32_t* pool_count_ptr = args.pool_count_base +
        static_cast<std::uint64_t>(slot) * args.count_slot_stride +
        static_cast<std::uint64_t>(args.state_index) * args.count_layer_stride;
    uint32_t* partial_count_ptr = args.partial_count_base +
        static_cast<std::uint64_t>(slot) * args.count_slot_stride +
        static_cast<std::uint64_t>(args.state_index) * args.count_layer_stride;
    const uint32_t pending = *partial_count_ptr;
    const uint32_t pool_start = *pool_count_ptr;
    if (pending != 0u && rows_here < cr - pending) {
        *partial_count_ptr = pending + rows_here;
        return;
    }
    const uint32_t head = pending == 0u ? 0u : cr - pending;
    const uint32_t body = (rows_here - head) / cr;
    const uint32_t tail = rows_here - head - body * cr;
    uint32_t next_pool = pool_start + (head != 0u ? 1u : 0u) + body;
    if (next_pool > args.max_blocks) next_pool = args.max_blocks;
    *pool_count_ptr = next_pool;
    *partial_count_ptr = tail;
}


__global__ __launch_bounds__(kQsaHeadDim) void qsa_key_cache_kernel(
    QsaKeyCacheArgs args) {
    const uint32_t d = threadIdx.x;
    __shared__ float scratch[kQsaHeadDim];
    __shared__ float normed[kQsaHeadDim];
    uint32_t partial_count = *args.partial_count;
    uint32_t pool_count = *args.pool_count;
    for (uint32_t row = 0; row < args.rows; ++row) {
        const shelf::bf16_t* raw =
            args.raw_keys + static_cast<std::uint64_t>(row) * args.head_dim;
        args.partial[static_cast<std::uint64_t>(partial_count) * args.head_dim + d] =
            raw[d];
        ++partial_count;
        __syncthreads();
        if (partial_count == args.compress_ratio) {
            float mean = 0.0f;
            for (uint32_t t = 0; t < args.compress_ratio; ++t) {
                mean += bf16_value(args.partial +
                                   static_cast<std::uint64_t>(t) * args.head_dim + d);
            }
            mean /= static_cast<float>(args.compress_ratio);
            const float square = block_sum(scratch, mean * mean);
            __syncthreads();
            const float inv = rsqrtf(square / static_cast<float>(args.head_dim) + args.eps);
            normed[d] = mean * inv * (1.0f + bf16_value(args.norm_weight + d));
            __syncthreads();
            if (pool_count < args.max_blocks) {
                const uint32_t start =
                    args.positions[row] >= args.compress_ratio - 1u
                        ? args.positions[row] - (args.compress_ratio - 1u)
                        : 0u;
                shelf::bf16_t* dst =
                    args.pooled + static_cast<std::uint64_t>(pool_count) * args.head_dim;
                const uint32_t half = args.rotary_dim / 2u;
                if (d < half) {
                    const float exponent = -2.0f * static_cast<float>(d) /
                                           static_cast<float>(args.rotary_dim);
                    const float frequency = args.inv_freq != nullptr
                                                ? static_cast<float>(args.inv_freq[d])
                                                : powf(args.theta, exponent);
                    const float angle = (static_cast<float>(start) /
                                         args.position_scale) * frequency;
                    const float cosine = cosf(angle);
                    const float sine = sinf(angle);
                    const float x0 = normed[d];
                    const float x1 = normed[d + half];
                    dst[d] = bf16_from(x0 * cosine - x1 * sine);
                } else if (d < args.rotary_dim) {
                    const uint32_t i = d - half;
                    const float exponent = -2.0f * static_cast<float>(i) /
                                           static_cast<float>(args.rotary_dim);
                    const float frequency = args.inv_freq != nullptr
                                                ? static_cast<float>(args.inv_freq[i])
                                                : powf(args.theta, exponent);
                    const float angle = (static_cast<float>(start) /
                                         args.position_scale) * frequency;
                    const float cosine = cosf(angle);
                    const float sine = sinf(angle);
                    const float x0 = normed[i];
                    const float x1 = normed[d];
                    dst[d] = bf16_from(x1 * cosine + x0 * sine);
                } else {
                    dst[d] = bf16_from(normed[d]);
                }
                ++pool_count;
            }
            partial_count = 0;
            __syncthreads();
        }
    }
    if (d == 0u) {
        *args.partial_count = partial_count;
        *args.pool_count = pool_count;
    }
}

}  // namespace

hipError_t launch_qsa_key_cache(const QsaKeyCacheArgs& args, hipStream_t stream) {
    if (args.raw_keys == nullptr || args.norm_weight == nullptr ||
        args.positions == nullptr || args.pooled == nullptr || args.partial == nullptr ||
        args.pool_count == nullptr || args.partial_count == nullptr) {
        return hipErrorInvalidValue;
    }
    if (args.rows == 0u || args.max_blocks == 0u || args.head_dim != kQsaHeadDim ||
        args.compress_ratio == 0u || args.compress_ratio > 4u ||
        args.rotary_dim == 0u || (args.rotary_dim & 1u) != 0u ||
        args.rotary_dim > args.head_dim || args.position_scale < 1.0f ||
        args.position_scale > 2.0f) {
        return hipErrorInvalidValue;
    }
    qsa_key_cache_kernel<<<1, kQsaHeadDim, 0, stream>>>(args);
    return hipGetLastError();
}


namespace {

bool key_cache_batch_valid(const QsaKeyCacheBatchArgs& args) {
    if (args.raw_keys == nullptr || args.norm_weight == nullptr ||
        args.positions == nullptr || args.requests == nullptr ||
        args.pooled_base == nullptr || args.partial_base == nullptr ||
        args.pool_count_base == nullptr || args.partial_count_base == nullptr) {
        return false;
    }
    if (args.num_requests == 0u || args.actual_rows == 0u || args.max_sequences == 0u ||
        args.max_blocks == 0u || args.head_dim != kQsaHeadDim ||
        args.compress_ratio == 0u || args.compress_ratio > 4u ||
        args.rotary_dim == 0u || (args.rotary_dim & 1u) != 0u ||
        args.rotary_dim > args.head_dim || args.position_scale < 1.0f ||
        args.position_scale > 2.0f || args.raw_row_stride < args.head_dim ||
        args.pooled_slot_stride < args.pooled_layer_stride ||
        args.pooled_layer_stride < static_cast<std::uint64_t>(args.max_blocks) * args.head_dim ||
        args.partial_slot_stride < args.partial_layer_stride ||
        args.partial_layer_stride <
            static_cast<std::uint64_t>(args.compress_ratio) * args.head_dim ||
        args.count_slot_stride < args.count_layer_stride) {
        return false;
    }
    return true;
}

}  // namespace

hipError_t launch_qsa_key_cache_batch(const QsaKeyCacheBatchArgs& args,
                                      hipStream_t stream) {
    if (!key_cache_batch_valid(args)) return hipErrorInvalidValue;
    qsa_key_cache_batch_kernel<<<args.num_requests, kQsaHeadDim, 0, stream>>>(args);
    return hipGetLastError();
}

hipError_t launch_qsa_key_cache_blocks(const QsaKeyCacheBatchArgs& args,
                                       hipStream_t stream) {
    if (!key_cache_batch_valid(args) || args.partial_history != nullptr) {
        return hipErrorInvalidValue;
    }
    const uint32_t blocks = (args.actual_rows + args.compress_ratio - 1u) /
                                args.compress_ratio +
                            1u;
    const dim3 grid(blocks, args.num_requests);
    qsa_key_cache_blocks_kernel<<<grid, kQsaHeadDim, 0, stream>>>(args);
    qsa_key_cache_counts_kernel<<<args.num_requests, 1, 0, stream>>>(args);
    return hipGetLastError();
}

}  // namespace shelf::kernel
