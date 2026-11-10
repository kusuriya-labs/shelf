#include <shelf/kernels/optimized/qsa.h>

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstddef>

namespace shelf::kernel {
namespace {

constexpr uint32_t kThreads = kQsaThreads;

__device__ __forceinline__ float bf16_value(const shelf::bf16_t* p) {
    return __uint_as_float(static_cast<std::uint32_t>(p->data) << 16);
}

__device__ __forceinline__ shelf::bf16_t bf16_from(float value) {
    return shelf::bf16_t{
        static_cast<std::uint16_t>(__bfloat16_as_ushort(__float2bfloat16(value)))};
}

__device__ __forceinline__ float block_sum(float* scratch, float value) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (uint32_t stride = kThreads / 2u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    return scratch[0];
}

__global__ __launch_bounds__(kThreads) void qsa_pool_kernel(QsaPoolArgs args) {
    const uint32_t block = blockIdx.x;
    const uint32_t d = threadIdx.x;
    __shared__ float scratch[kThreads];
    float mean = 0.0f;
    for (uint32_t t = 0; t < args.compress_ratio; ++t) {
        const uint32_t row = block * args.compress_ratio + t;
        mean += bf16_value(args.raw_keys + static_cast<std::uint64_t>(row) * args.head_dim + d);
    }
    mean /= static_cast<float>(args.compress_ratio);
    const float square = block_sum(scratch, mean * mean);
    __syncthreads();
    const float inv = rsqrtf(square / static_cast<float>(args.head_dim) + args.eps);
    const float weight = 1.0f + bf16_value(args.norm_weight + d);
    args.pooled[static_cast<std::uint64_t>(block) * args.head_dim + d] =
        bf16_from(mean * inv * weight);
}

__device__ __forceinline__ bool qsa_worse(float a, uint32_t ia, float b, uint32_t ib) noexcept {
    return a < b || (a == b && ia > ib);
}

__device__ __forceinline__ void qsa_swap_pair(float* scores, int32_t* indices,
                                              uint32_t a, uint32_t b) {
    const float score = scores[a];
    const int32_t index = indices[a];
    scores[a] = scores[b];
    indices[a] = indices[b];
    scores[b] = score;
    indices[b] = index;
}

__device__ __forceinline__ void qsa_sort_candidates_ascending(
    float* scores, int32_t* indices, uint32_t base, uint32_t count) {
    for (uint32_t size = 2u; size <= count; size <<= 1u) {
        for (uint32_t stride = size >> 1u; stride > 0u; stride >>= 1u) {
            for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
                const uint32_t j = i ^ stride;
                if (j <= i) continue;
                const bool ascending = (i & size) == 0u;
                const bool i_worse =
                    qsa_worse(scores[base + i], static_cast<uint32_t>(indices[base + i]),
                              scores[base + j], static_cast<uint32_t>(indices[base + j]));
                if (ascending ? !i_worse : i_worse)
                    qsa_swap_pair(scores, indices, base + i, base + j);
            }
            __syncthreads();
        }
    }
}

__device__ __forceinline__ void qsa_merge_descending(float* scores, int32_t* indices,
                                                     uint32_t count) {
    for (uint32_t stride = count >> 1u; stride > 0u; stride >>= 1u) {
        for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
            const uint32_t j = i ^ stride;
            if (j <= i) continue;
            const bool i_worse =
                qsa_worse(scores[i], static_cast<uint32_t>(indices[i]),
                          scores[j], static_cast<uint32_t>(indices[j]));
            if (i_worse) qsa_swap_pair(scores, indices, i, j);
        }
        __syncthreads();
    }
}

__global__ __launch_bounds__(kQsaThreads) void qsa_select_batch_kernel(
    QsaSelectBatchArgs args) {
    constexpr uint32_t kWarps = kQsaThreads / 32u;
    constexpr uint32_t kBlocksPerWarp = kQsaSelectChunk / kWarps;
    constexpr uint32_t kDimsPerLane = kQsaHeadDim / 32u;

    __shared__ bf16_t q_stage[kQsaSelectMaxHeads][kQsaHeadDim];
    __shared__ float block_score[kQsaSelectChunk];
    __shared__ uint32_t candidate_count;

    extern __shared__ uint8_t qsa_select_dynamic[];
    const uint32_t row = blockIdx.x;
    const uint32_t t = threadIdx.x;
    const uint32_t heads = args.n_heads;
    const bf16_t* q_row = args.q + static_cast<uint64_t>(row) * args.q_row_stride;
    for (uint32_t i = t; i < heads * kQsaHeadDim; i += kQsaThreads) {
        q_stage[i / kQsaHeadDim][i % kQsaHeadDim] = q_row[i];
    }

    uint32_t* mask_words_ptr = nullptr;
    const uint32_t mask_words = (args.blocks + 31u) / 32u;
    if (args.block_mask != nullptr) {
        mask_words_ptr = reinterpret_cast<uint32_t*>(
            args.block_mask + static_cast<uint64_t>(row) * args.mask_row_bytes);
        for (uint32_t i = t; i < mask_words; i += kQsaThreads) mask_words_ptr[i] = 0u;
    }
    int32_t* out = args.selected != nullptr
                       ? args.selected + static_cast<uint64_t>(row) * args.selected_row_stride
                       : nullptr;

    const uint32_t slot = args.row_sequence_slots[row];
    const uint32_t complete = (args.positions[row] + 1u) / args.compress_ratio;
    const uint32_t visible = complete < args.blocks ? complete : args.blocks;
    const uint32_t top_k = args.top_k < kQsaSelectMaxTopK ? args.top_k : kQsaSelectMaxTopK;
    uint32_t half = kQsaSelectChunk;
    while (half < top_k) half <<= 1u;
    const uint32_t count = half * 2u;
    float* scores = reinterpret_cast<float*>(qsa_select_dynamic);
    int32_t* indices =
        reinterpret_cast<int32_t*>(qsa_select_dynamic + count * sizeof(float));
    __syncthreads();
    if (slot >= args.max_sequences) {
        if (out != nullptr) {
            for (uint32_t i = t; i < args.top_k; i += kQsaThreads) out[i] = -1;
        }
        return;
    }

    const bf16_t* pooled = args.pooled_base +
                           static_cast<uint64_t>(slot) * args.pooled_slot_stride +
                           static_cast<uint64_t>(args.state_index) *
                               args.pooled_layer_stride;

    const uint32_t warp = t >> 5u;
    const uint32_t lane = t & 31u;
    const uint32_t lane_dim = lane * kDimsPerLane;
    for (uint32_t i = t; i < count; i += kQsaThreads) {
        scores[i] = -INFINITY;
        indices[i] = -1;
    }
    if (t == 0u) candidate_count = 0u;
    __syncthreads();

    for (uint32_t base = 0u; base < visible; base += kQsaSelectChunk) {
        const uint32_t n =
            visible - base < kQsaSelectChunk ? visible - base : kQsaSelectChunk;
        for (uint32_t slot_in_warp = 0u; slot_in_warp < kBlocksPerWarp; ++slot_in_warp) {
            const uint32_t b = warp * kBlocksPerWarp + slot_in_warp;
            if (b >= n) continue;
            const bf16_t* k_row =
                pooled + static_cast<uint64_t>(base + b) * kQsaHeadDim + lane_dim;
            const float k0 = bf16_value(k_row + 0u);
            const float k1 = bf16_value(k_row + 1u);
            const float k2 = bf16_value(k_row + 2u);
            const float k3 = bf16_value(k_row + 3u);
            float total = 0.0f;
            for (uint32_t head = 0u; head < heads; ++head) {
                const bf16_t* qp = &q_stage[head][lane_dim];
                float dot = bf16_value(qp + 0u) * k0 + bf16_value(qp + 1u) * k1 +
                            bf16_value(qp + 2u) * k2 + bf16_value(qp + 3u) * k3;
                dot += __shfl_xor(dot, 16u);
                dot += __shfl_xor(dot, 8u);
                dot += __shfl_xor(dot, 4u);
                dot += __shfl_xor(dot, 2u);
                dot += __shfl_xor(dot, 1u);
                total += fmaxf(dot, 0.0f);
            }
            if (lane == 0u) block_score[b] = total * args.scale;
        }
        __syncthreads();
        if (t < n) {
            scores[half + candidate_count + t] = block_score[t];
            indices[half + candidate_count + t] = static_cast<int32_t>(base + t);
        }
        __syncthreads();
        if (t == 0u) candidate_count += n;
        __syncthreads();
        if (candidate_count == half) {
            qsa_sort_candidates_ascending(scores, indices, half, half);
            qsa_merge_descending(scores, indices, count);
            if (t == 0u) candidate_count = 0u;
            for (uint32_t i = t; i < half; i += kQsaThreads) {
                scores[half + i] = -INFINITY;
                indices[half + i] = -1;
            }
            __syncthreads();
        }
    }

    if (candidate_count != 0u) {
        qsa_sort_candidates_ascending(scores, indices, half, half);
        qsa_merge_descending(scores, indices, count);
    }

    const uint32_t selected = visible < top_k ? visible : top_k;
    if (out != nullptr) {
        if (t == 0u) {
            for (uint32_t i = 1u; i < selected; ++i) {
                const int32_t value = indices[i];
                uint32_t j = i;
                while (j > 0u && indices[j - 1u] > value) {
                    indices[j] = indices[j - 1u];
                    --j;
                }
                indices[j] = value;
            }
        }
        __syncthreads();
        for (uint32_t i = t; i < args.top_k; i += kQsaThreads)
            out[i] = i < selected ? indices[i] : -1;
    }
    if (mask_words_ptr != nullptr) {
        for (uint32_t i = t; i < selected; i += kQsaThreads) {
            const uint32_t block = static_cast<uint32_t>(indices[i]);
            atomicOr(&mask_words_ptr[block / 32u], 1u << (block % 32u));
        }
        if (t == 0u && complete < args.blocks) {
            atomicOr(&mask_words_ptr[complete / 32u], 1u << (complete % 32u));
        }
    }
}

using qsa_wbf8_t = short __attribute__((ext_vector_type(8)));
using qsa_wf8_t = float __attribute__((ext_vector_type(8)));

__device__ __forceinline__ qsa_wbf8_t qsa_pack_frag(uint2 a, uint2 b) {
    union { unsigned w[4]; qsa_wbf8_t f; } u;
    u.w[0] = a.x; u.w[1] = a.y; u.w[2] = b.x; u.w[3] = b.y;
    return u.f;
}

__device__ __forceinline__ qsa_wbf8_t qsa_frag_a(
    const bf16_t* src, uint32_t pitch, uint32_t kbase, uint32_t lane) {
    const bf16_t* p = src + static_cast<size_t>(lane & 15u) * pitch
        + 4u * (lane >> 4u) + kbase;
    const uint2 a = *reinterpret_cast<const uint2*>(p);
    const uint2 b = *reinterpret_cast<const uint2*>(p + 8);
    return qsa_pack_frag(a, b);
}

__device__ __forceinline__ qsa_wbf8_t qsa_frag_b_kn(
    const bf16_t* src, uint32_t pitch, uint32_t nbase, uint32_t kbase,
    uint32_t lane) {
    const uint32_t n = nbase + (lane & 15u);
    const uint32_t k0 = kbase + 4u * (lane >> 4u);
    union { unsigned short h[8]; qsa_wbf8_t f; } u;
    #pragma unroll
    for (uint32_t e = 0u; e < 4u; ++e) {
        u.h[e] = src[static_cast<size_t>(k0 + e) * pitch + n].data;
        u.h[4u + e] = src[static_cast<size_t>(k0 + 8u + e) * pitch + n].data;
    }
    return u.f;
}

__device__ __forceinline__ qsa_wf8_t qsa_wmma(qsa_wbf8_t a, qsa_wbf8_t b,
                                              qsa_wf8_t c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}

constexpr uint32_t kQsaWmmaPitch = kQsaHeadDim + 8u;
constexpr uint32_t kQsaWmmaKPitch = 40u;
constexpr uint32_t kQsaScoreChunk = 32u;

__global__ __launch_bounds__(kQsaThreads, 2) void qsa_score_wmma_kernel(
    QsaScoreArgs args) {
    const uint32_t row_group = blockIdx.x * kQsaScoreRowsPerCta;
    const uint32_t block_tile = args.block_tile == 0u ? kQsaScoreBlocksPerCta : args.block_tile;
    const uint32_t block_group = blockIdx.y * block_tile;
    if (row_group >= args.rows || block_group >= args.blocks) return;
    const uint32_t valid_rows = args.rows - row_group < kQsaScoreRowsPerCta
        ? args.rows - row_group
        : kQsaScoreRowsPerCta;
    const uint32_t block_here = args.blocks - block_group < block_tile
        ? args.blocks - block_group
        : block_tile;
    const uint32_t t = threadIdx.x;
    const uint32_t lane = t & 31u;
    const uint32_t hi = lane >> 4u;
    const uint32_t lo = lane & 15u;

    __shared__ bf16_t q_tile[64][kQsaWmmaPitch];
    __shared__ bf16_t k_tile[kQsaHeadDim][kQsaWmmaKPitch];
    __shared__ float score_tile[16][kQsaScoreChunk];

    const uint32_t slot = args.row_sequence_slots[row_group];
    const bf16_t* pooled = args.pooled_base +
        static_cast<uint64_t>(slot) * args.pooled_slot_stride +
        static_cast<uint64_t>(args.state_index) * args.pooled_layer_stride;

    for (uint32_t i = t; i < 64u * kQsaHeadDim; i += kQsaThreads) {
        const uint32_t m = i / kQsaHeadDim;
        const uint32_t d = i % kQsaHeadDim;
        const uint32_t r = m / 4u;
        const uint32_t h = m % 4u;
        q_tile[m][d] = r < valid_rows
            ? args.q[static_cast<uint64_t>(row_group + r) * args.q_row_stride +
                     static_cast<uint64_t>(h) * kQsaHeadDim + d]
            : bf16_t{0};
    }
    __syncthreads();

    for (uint32_t chunk = 0u; chunk < block_here; chunk += kQsaScoreChunk) {
        const uint32_t n = block_here - chunk < kQsaScoreChunk
            ? block_here - chunk
            : kQsaScoreChunk;
        for (uint32_t i = t; i < kQsaScoreChunk * kQsaHeadDim; i += kQsaThreads) {
            const uint32_t b = i / kQsaHeadDim;
            const uint32_t d = i % kQsaHeadDim;
            k_tile[d][b] = b < n
                ? pooled[static_cast<uint64_t>(args.block_begin + block_group + chunk + b) *
                             kQsaHeadDim + d]
                : bf16_t{0};
        }
        __syncthreads();

        const uint32_t mt = t >> 5u;
        if (mt * 4u < valid_rows) {
            qsa_wbf8_t qfrag[8];
            #pragma unroll
            for (uint32_t ks = 0u; ks < 8u; ++ks)
                qfrag[ks] = qsa_frag_a(&q_tile[mt * 16u][0], kQsaWmmaPitch,
                                       ks * 16u, lane);
            for (uint32_t nt = 0u; nt < 2u; ++nt) {
                qsa_wf8_t c = {};
                #pragma unroll
                for (uint32_t ks = 0u; ks < 8u; ++ks) {
                    const qsa_wbf8_t fb = qsa_frag_b_kn(
                        &k_tile[0][0], kQsaWmmaKPitch, nt * 16u, ks * 16u, lane);
                    c = qsa_wmma(qfrag[ks], fb, c);
                }
                const uint32_t col = nt * 16u + lo;
                const uint32_t ra = mt * 4u + 2u * hi;
                const float sa = (fmaxf(c[0], 0.0f) + fmaxf(c[1], 0.0f) +
                                  fmaxf(c[2], 0.0f) + fmaxf(c[3], 0.0f)) * args.scale;
                const float sb = (fmaxf(c[4], 0.0f) + fmaxf(c[5], 0.0f) +
                                  fmaxf(c[6], 0.0f) + fmaxf(c[7], 0.0f)) * args.scale;
                if (ra < valid_rows) score_tile[ra][col] = sa;
                if (ra + 1u < valid_rows) score_tile[ra + 1u][col] = sb;
            }
        }
        __syncthreads();

        for (uint32_t i = t; i < valid_rows * kQsaScoreChunk; i += kQsaThreads) {
            const uint32_t r = i / kQsaScoreChunk;
            const uint32_t b = i % kQsaScoreChunk;
            if (b < n) {
                args.scores[static_cast<uint64_t>(args.row_begin + row_group + r) *
                                args.score_row_stride +
                            block_group + chunk + b] = score_tile[r][b];
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__ uint32_t qsa_radix_key(float value) {
    const uint32_t bits = __float_as_uint(value);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ float qsa_radix_value(uint32_t key) {
    const uint32_t bits = (key & 0x80000000u) ? (key & 0x7FFFFFFFu) : ~key;
    return __uint_as_float(bits);
}

// Exact top-k over scores[0..usable). Writes the selected indices (unsorted) to out.
// Ties on the score keep the smaller index, matching the sort/merge path.
__device__ void qsa_radix_select(const float* scores, uint32_t usable, uint32_t k,
                                 uint32_t* hist, uint32_t* state, std::int32_t* out,
                                 float* out_scores, uint32_t index_base) {
    const uint32_t t = threadIdx.x;
    const uint32_t stride = blockDim.x;
    uint32_t rank = k;
    uint32_t prefix = 0u;
    uint32_t above = 0u;
    for (uint32_t pass = 0u; pass < 4u; ++pass) {
        const uint32_t shift = 24u - 8u * pass;
        for (uint32_t i = t; i < 256u; i += stride) hist[i] = 0u;
        __syncthreads();
        for (uint32_t i = t; i < usable; i += stride) {
            const uint32_t key = qsa_radix_key(scores[i]);
            if (pass == 0u || (key >> (shift + 8u)) == (prefix >> (shift + 8u)))
                atomicAdd(&hist[(key >> shift) & 0xFFu], 1u);
        }
        __syncthreads();
        if (t == 0u) {
            uint32_t acc = 0u;
            uint32_t bucket = 0u;
            for (int b = 255; b >= 0; --b) {
                const uint32_t count = hist[static_cast<uint32_t>(b)];
                if (acc + count >= rank) {
                    bucket = static_cast<uint32_t>(b);
                    break;
                }
                acc += count;
            }
            state[0] = bucket;
            state[1] = acc;
            state[2] = rank - acc;
        }
        __syncthreads();
        above += state[1];
        rank = state[2];
        prefix |= state[0] << shift;
        __syncthreads();
    }
    const float threshold = qsa_radix_value(prefix);
    uint32_t need = k - above;
    uint32_t index_prefix = 0u;
    for (uint32_t pass = 0u; pass < 2u && need != 0u; ++pass) {
        const uint32_t shift = 8u - 8u * pass;
        for (uint32_t i = t; i < 256u; i += stride) hist[i] = 0u;
        __syncthreads();
        for (uint32_t i = t; i < usable; i += stride) {
            if (scores[i] != threshold) continue;
            if (pass != 0u && (i >> (shift + 8u)) != (index_prefix >> (shift + 8u)))
                continue;
            atomicAdd(&hist[(i >> shift) & 0xFFu], 1u);
        }
        __syncthreads();
        if (t == 0u) {
            uint32_t acc = 0u;
            uint32_t bucket = 0u;
            for (uint32_t b = 0u; b < 256u; ++b) {
                const uint32_t count = hist[b];
                if (acc + count >= need) {
                    bucket = b;
                    break;
                }
                acc += count;
            }
            state[0] = bucket;
            state[1] = need - acc;
        }
        __syncthreads();
        index_prefix |= state[0] << shift;
        need = state[1];
        __syncthreads();
    }
    if (t == 0u) state[3] = 0u;
    __syncthreads();
    for (uint32_t i = t; i < usable; i += stride) {
        const float value = scores[i];
        if (value > threshold || (value == threshold && i <= index_prefix)) {
            const uint32_t pos = atomicAdd(&state[3], 1u);
            out[pos] = static_cast<std::int32_t>(index_base + i);
            if (out_scores != nullptr) out_scores[pos] = value;
        }
    }
    __syncthreads();
}

__global__ __launch_bounds__(kQsaThreads) void qsa_topk_kernel(QsaTopkArgs args) {
    const uint32_t row = blockIdx.x;
    if (row >= args.rows) return;
    const uint32_t t = threadIdx.x;

    __shared__ uint32_t hist[256];
    __shared__ uint32_t radix_state[4];
    extern __shared__ uint8_t qsa_topk_dynamic[];

    const uint32_t top_k =
        args.top_k < kQsaSelectMaxTopK ? args.top_k : kQsaSelectMaxTopK;
    uint32_t half = kQsaSelectChunk;
    while (half < top_k) half <<= 1u;
    const uint32_t count = half * 2u;
    float* scores = reinterpret_cast<float*>(qsa_topk_dynamic);
    int32_t* indices =
        reinterpret_cast<int32_t*>(qsa_topk_dynamic + count * sizeof(float));
    const uint32_t visible = [] (uint32_t position, uint32_t compress, uint32_t blocks) {
        const uint32_t complete = (position + 1u) / compress;
        return complete < blocks ? complete : blocks;
    }(args.positions[row], args.compress_ratio, args.blocks);
    const uint32_t usable = visible > args.block_begin
        ? (visible - args.block_begin < args.block_count
               ? visible - args.block_begin
               : args.block_count)
        : 0u;

    const uint32_t block_max = visible < top_k ? visible : top_k;
    const bool select_once = args.first != 0u && args.last != 0u &&
                             args.block_begin == 0u && usable == visible &&
                             usable <= 65536u;
    if (select_once) {
        if (block_max != 0u)
            qsa_radix_select(
                args.scores + static_cast<uint64_t>(row) * args.score_row_stride,
                usable, block_max, hist, radix_state, indices, nullptr, 0u);
    } else {
        for (uint32_t i = t; i < count; i += kQsaThreads) {
            if (i >= half || args.first != 0u) {
                scores[i] = -INFINITY;
                indices[i] = -1;
            } else {
                scores[i] = args.state_scores[static_cast<uint64_t>(row) * count + i];
                indices[i] = args.state_indices[static_cast<uint64_t>(row) * count + i];
            }
        }
        __syncthreads();

        if (usable != 0u) {
            const uint32_t want = usable < top_k ? usable : top_k;
            qsa_radix_select(
                args.scores + static_cast<uint64_t>(row) * args.score_row_stride,
                usable, want, hist, radix_state, indices + half, scores + half,
                args.block_begin);
        }

        qsa_sort_candidates_ascending(scores, indices, half, half);
        qsa_merge_descending(scores, indices, count);

        if (args.last == 0u) {
            for (uint32_t i = t; i < half; i += kQsaThreads) {
                args.state_scores[static_cast<uint64_t>(row) * count + i] = scores[i];
                args.state_indices[static_cast<uint64_t>(row) * count + i] = indices[i];
            }
            return;
        }
    }

    int32_t* out = args.selected != nullptr
        ? args.selected + static_cast<uint64_t>(row) * args.selected_row_stride
        : nullptr;
    if (out != nullptr) {
        if (t == 0u) {
            for (uint32_t i = 1u; i < block_max; ++i) {
                const int32_t value = indices[i];
                uint32_t j = i;
                while (j > 0u && indices[j - 1u] > value) {
                    indices[j] = indices[j - 1u];
                    --j;
                }
                indices[j] = value;
            }
        }
        __syncthreads();
        for (uint32_t i = t; i < args.top_k; i += kQsaThreads)
            out[i] = i < block_max ? indices[i] : -1;
    }
    uint32_t* words = reinterpret_cast<uint32_t*>(
        args.block_mask + static_cast<uint64_t>(row) * args.mask_row_bytes);
    const uint32_t mask_words = (args.blocks + 31u) / 32u;
    for (uint32_t i = t; i < mask_words; i += kQsaThreads) words[i] = 0u;
    __syncthreads();
    for (uint32_t i = t; i < block_max; i += kQsaThreads) {
        const uint32_t block = static_cast<uint32_t>(indices[i]);
        atomicOr(&words[block / 32u], 1u << (block % 32u));
    }
    const uint32_t complete = (args.positions[row] + 1u) / args.compress_ratio;
    if (t == 0u && complete < args.blocks)
        atomicOr(&words[complete / 32u], 1u << (complete % 32u));
}

__global__ __launch_bounds__(kQsaHeadDim) void qsa_rope_kernel(QsaRopeArgs args) {
    const uint32_t row = blockIdx.x;
    const uint32_t half = args.rotary_dim / 2u;
    const float position = static_cast<float>(args.positions[row]);
    for (uint32_t head = 0; head < args.heads; ++head) {
        shelf::bf16_t* base =
            args.x + (static_cast<std::uint64_t>(row) * args.heads + head) * args.head_dim;
        for (uint32_t i = threadIdx.x; i < half; i += blockDim.x) {
            const float exponent = -2.0f * static_cast<float>(i) /
                                   static_cast<float>(args.rotary_dim);
            const float frequency = args.inv_freq != nullptr
                                        ? static_cast<float>(args.inv_freq[i])
                                        : powf(args.theta, exponent);
            const float angle = (position / args.position_scale) * frequency;
            const float cosine = cosf(angle) * args.attention_factor;
            const float sine = sinf(angle) * args.attention_factor;
            const float x0 = bf16_value(base + i);
            const float x1 = bf16_value(base + i + half);
            base[i] = bf16_from(x0 * cosine - x1 * sine);
            base[i + half] = bf16_from(x1 * cosine + x0 * sine);
        }
    }
}

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

hipError_t launch_qsa_pool(const QsaPoolArgs& args, hipStream_t stream) {    if (args.raw_keys == nullptr || args.norm_weight == nullptr || args.pooled == nullptr) {
        return hipErrorInvalidValue;
    }
    if (args.rows == 0u || args.blocks == 0u || args.head_dim != kQsaHeadDim ||
        args.compress_ratio == 0u || args.blocks * args.compress_ratio > args.rows) {
        return hipErrorInvalidValue;
    }
    qsa_pool_kernel<<<args.blocks, kThreads, 0, stream>>>(args);
    return hipGetLastError();
}

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


hipError_t launch_qsa_score_wmma(const QsaScoreArgs& args, hipStream_t stream) {
    if (args.q == nullptr || args.pooled_base == nullptr || args.scores == nullptr ||
        args.row_sequence_slots == nullptr || args.rows == 0u || args.blocks == 0u ||
        args.n_heads != 4u || args.head_dim != kQsaHeadDim ||
        args.compress_ratio == 0u || args.max_sequences == 0u ||
        args.score_row_stride < args.blocks ||
        args.q_row_stride < 4u * kQsaHeadDim ||
        args.pooled_slot_stride < args.pooled_layer_stride ||
        args.pooled_layer_stride <
            static_cast<uint64_t>(args.blocks) * kQsaHeadDim ||
        args.scale == 0.0f) {
        return hipErrorInvalidValue;
    }
    const uint32_t row_groups =
        (args.rows + kQsaScoreRowsPerCta - 1u) / kQsaScoreRowsPerCta;
    const uint32_t tile = args.block_tile == 0u ? kQsaScoreBlocksPerCta : args.block_tile;
    if (tile > kQsaScoreBlockTile) return hipErrorInvalidValue;
    const uint32_t block_tiles = (args.blocks + tile - 1u) / tile;
    const dim3 grid(row_groups, block_tiles);
    qsa_score_wmma_kernel<<<grid, kQsaThreads, 0, stream>>>(args);
    return hipGetLastError();
}

hipError_t launch_qsa_topk(const QsaTopkArgs& args, hipStream_t stream) {
    if (args.scores == nullptr || args.state_scores == nullptr ||
        args.state_indices == nullptr || args.block_mask == nullptr ||
        args.positions == nullptr || args.rows == 0u ||
        args.top_k == 0u || args.compress_ratio == 0u ||
        args.score_row_stride < args.block_count ||
        args.mask_row_bytes < ((args.blocks + 31u) / 32u) * 4u) {
        return hipErrorInvalidValue;
    }
    if (args.selected != nullptr && args.selected_row_stride < args.top_k) {
        return hipErrorInvalidValue;
    }
    uint32_t half = kQsaSelectChunk;
    while (half < args.top_k) half <<= 1u;
    const std::size_t dynamic_bytes =
        2u * static_cast<std::size_t>(half) *
        (sizeof(float) + sizeof(std::int32_t));
    qsa_topk_kernel<<<args.rows, kQsaThreads, dynamic_bytes, stream>>>(args);
    return hipGetLastError();
}

hipError_t launch_qsa_select_batch(const QsaSelectBatchArgs& args, hipStream_t stream) {
    if (args.q == nullptr || args.pooled_base == nullptr || args.block_mask == nullptr ||
        args.row_sequence_slots == nullptr || args.positions == nullptr) {
        return hipErrorInvalidValue;
    }
    if (args.rows == 0u || args.blocks == 0u || args.top_k == 0u ||
        args.n_heads == 0u || args.n_heads > kQsaSelectMaxHeads ||
        args.top_k > kQsaSelectMaxTopK || args.head_dim != kQsaHeadDim ||
        args.compress_ratio == 0u || args.max_sequences == 0u ||
        args.q_row_stride < args.n_heads * args.head_dim ||
        args.pooled_slot_stride < args.pooled_layer_stride ||
        args.pooled_layer_stride <
            static_cast<uint64_t>(args.blocks) * args.head_dim ||
        args.scale == 0.0f) {
        return hipErrorInvalidValue;
    }
    const uint32_t mask_words = (args.blocks + 31u) / 32u;
    if (args.mask_row_bytes < mask_words * 4u || (args.mask_row_bytes % 4u) != 0u ||
        (reinterpret_cast<uintptr_t>(args.block_mask) % 4u) != 0u) {
        return hipErrorInvalidValue;
    }
    if (args.selected != nullptr && args.selected_row_stride < args.top_k) {
        return hipErrorInvalidValue;
    }
    uint32_t half = kQsaSelectChunk;
    while (half < args.top_k) half <<= 1u;
    const std::size_t dynamic_bytes =
        static_cast<std::size_t>(half * 2u) * (sizeof(float) + sizeof(std::int32_t));
    qsa_select_batch_kernel<<<args.rows, kQsaThreads, dynamic_bytes, stream>>>(args);
    return hipGetLastError();
}

hipError_t launch_qsa_rope(const QsaRopeArgs& args, hipStream_t stream) {
    if (args.x == nullptr || args.positions == nullptr) return hipErrorInvalidValue;
    if (args.rows == 0u || args.heads == 0u || args.head_dim == 0u ||
        args.rotary_dim == 0u || (args.rotary_dim & 1u) != 0u ||
        args.rotary_dim > args.head_dim || args.head_dim != kQsaHeadDim ||
        args.position_scale < 1.0f || args.position_scale > 2.0f) {
        return hipErrorInvalidValue;
    }
    qsa_rope_kernel<<<args.rows, kQsaHeadDim, 0, stream>>>(args);
    return hipGetLastError();
}

}  // namespace shelf::kernel
