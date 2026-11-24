#include <shelf/kernels/optimized/qsa.h>
#include <shelf/kernels/fused/detail/qsa_common.h>

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstddef>

namespace shelf::kernel {
namespace {

constexpr uint32_t kThreads = kQsaThreads;

using namespace detail::qsa;

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

__device__ __forceinline__ void qsa_score_cta_body(
    const bf16_t* q_rows, uint32_t q_row_stride,
    const bf16_t* pooled,
    float* score_rows, uint32_t score_row_stride,
    uint32_t block_begin, uint32_t block_group, uint32_t block_here,
    uint32_t valid_rows, float scale,
    bf16_t (*q_tile)[kQsaWmmaPitch], bf16_t (*k_tile)[kQsaWmmaKPitch],
    float (*score_tile)[kQsaScoreChunk]) {
    const uint32_t t = threadIdx.x;
    const uint32_t lane = t & 31u;
    const uint32_t hi = lane >> 4u;
    const uint32_t lo = lane & 15u;

    for (uint32_t i = t; i < 64u * kQsaHeadDim; i += kQsaThreads) {
        const uint32_t m = i / kQsaHeadDim;
        const uint32_t d = i % kQsaHeadDim;
        const uint32_t r = m / 4u;
        const uint32_t h = m % 4u;
        q_tile[m][d] = r < valid_rows
            ? q_rows[static_cast<uint64_t>(r) * q_row_stride +
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
                ? pooled[static_cast<uint64_t>(block_begin + block_group + chunk + b) *
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
                                  fmaxf(c[2], 0.0f) + fmaxf(c[3], 0.0f)) * scale;
                const float sb = (fmaxf(c[4], 0.0f) + fmaxf(c[5], 0.0f) +
                                  fmaxf(c[6], 0.0f) + fmaxf(c[7], 0.0f)) * scale;
                if (ra < valid_rows) score_tile[ra][col] = sa;
                if (ra + 1u < valid_rows) score_tile[ra + 1u][col] = sb;
            }
        }
        __syncthreads();

        for (uint32_t i = t; i < valid_rows * kQsaScoreChunk; i += kQsaThreads) {
            const uint32_t r = i / kQsaScoreChunk;
            const uint32_t b = i % kQsaScoreChunk;
            if (b < n) {
                score_rows[static_cast<uint64_t>(r) * score_row_stride +
                           block_group + chunk + b] = score_tile[r][b];
            }
        }
        __syncthreads();
    }
}

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

    __shared__ bf16_t q_tile[64][kQsaWmmaPitch];
    __shared__ bf16_t k_tile[kQsaHeadDim][kQsaWmmaKPitch];
    __shared__ float score_tile[16][kQsaScoreChunk];

    const uint32_t slot = args.row_sequence_slots[row_group];
    const bf16_t* pooled = args.pooled_base +
        static_cast<uint64_t>(slot) * args.pooled_slot_stride +
        static_cast<uint64_t>(args.state_index) * args.pooled_layer_stride;

    qsa_score_cta_body(
        args.q + static_cast<uint64_t>(row_group) * args.q_row_stride,
        args.q_row_stride, pooled,
        args.scores + static_cast<uint64_t>(args.row_begin + row_group) *
                          args.score_row_stride,
        args.score_row_stride, args.block_begin, block_group, block_here,
        valid_rows, args.scale, q_tile, k_tile, score_tile);
}

__global__ __launch_bounds__(kQsaThreads, 2) void qsa_score_slices_kernel(
    QsaScoreBatchedArgs args) {
    const ::shelf::runtime::DeviceRequestDescriptor req = args.requests[blockIdx.z];
    const uint32_t tile_end = args.row_tile_begin + args.row_tile_rows;
    const uint32_t request_end = req.row_begin + req.row_count;
    const uint32_t begin = req.row_begin > args.row_tile_begin
        ? req.row_begin
        : args.row_tile_begin;
    const uint32_t end = request_end < tile_end ? request_end : tile_end;
    if (end <= begin) return;
    const uint32_t row_group = blockIdx.x * kQsaScoreRowsPerCta;
    const uint32_t window_rows = end - begin;
    if (row_group >= window_rows) return;
    const uint32_t valid_rows = window_rows - row_group < kQsaScoreRowsPerCta
        ? window_rows - row_group
        : kQsaScoreRowsPerCta;

    const uint32_t request_blocks = req.sequence_length / args.compress_ratio;
    const uint32_t slice_blocks = request_blocks < args.max_blocks
        ? request_blocks
        : args.max_blocks;
    if (slice_blocks <= args.block_begin) return;
    const uint32_t blocks_here = slice_blocks - args.block_begin < args.block_count
        ? slice_blocks - args.block_begin
        : args.block_count;
    const uint32_t block_tile = args.block_tile == 0u ? kQsaScoreBlocksPerCta : args.block_tile;
    const uint32_t block_group = blockIdx.y * block_tile;
    if (block_group >= blocks_here) return;
    const uint32_t block_here = blocks_here - block_group < block_tile
        ? blocks_here - block_group
        : block_tile;

    __shared__ bf16_t q_tile[64][kQsaWmmaPitch];
    __shared__ bf16_t k_tile[kQsaHeadDim][kQsaWmmaKPitch];
    __shared__ float score_tile[16][kQsaScoreChunk];

    const uint32_t absolute_row = begin + row_group;
    const uint32_t slot = args.row_sequence_slots[absolute_row];
    const bf16_t* pooled = args.pooled_base +
        static_cast<uint64_t>(slot) * args.pooled_slot_stride +
        static_cast<uint64_t>(args.state_index) * args.pooled_layer_stride;

    qsa_score_cta_body(
        args.q + static_cast<uint64_t>(absolute_row) * args.q_row_stride,
        args.q_row_stride, pooled,
        args.scores + static_cast<uint64_t>(absolute_row - args.row_tile_begin) *
                          args.score_row_stride,
        args.score_row_stride, args.block_begin, block_group, block_here,
        valid_rows, args.scale, q_tile, k_tile, score_tile);
}

__device__ __forceinline__ uint32_t qsa_radix_key(float value) {
    const uint32_t bits = value == 0.0f ? 0u : __float_as_uint(value);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

constexpr uint32_t kQsaRadixWaves = 8u;
constexpr uint32_t kQsaRadixCandidates = 2048u;

struct QsaRadixScratch {
    uint32_t hist[kQsaRadixWaves][256];
    uint32_t wave_sums[kQsaRadixWaves];
    uint32_t state[8];
    uint32_t cand_keys[kQsaRadixCandidates];
    uint32_t cand_idx[kQsaRadixCandidates];
};

__device__ __forceinline__ void qsa_radix_scan(uint32_t value, uint32_t* wave_sums,
                                               uint32_t& exclusive, uint32_t& total) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t waves = blockDim.x >> 5u;
    uint32_t x = value;
    #pragma unroll
    for (uint32_t d = 1u; d < 32u; d <<= 1u) {
        const uint32_t other = __shfl_up(x, d, 32);
        if (lane >= d) x += other;
    }
    if (lane == 31u) wave_sums[wave] = x;
    __syncthreads();
    if (wave == 0u) {
        uint32_t w = threadIdx.x < waves ? wave_sums[threadIdx.x] : 0u;
        #pragma unroll
        for (uint32_t d = 1u; d < 32u; d <<= 1u) {
            const uint32_t other = __shfl_up(w, d, 32);
            if (lane >= d) w += other;
        }
        if (threadIdx.x < waves) wave_sums[threadIdx.x] = w;
    }
    __syncthreads();
    exclusive = (wave > 0u ? wave_sums[wave - 1u] : 0u) + x - value;
    total = wave_sums[waves - 1u];
}

__device__ void qsa_radix_select(const float* scores, uint32_t usable, uint32_t k,
                                 QsaRadixScratch* s, std::int32_t* out,
                                 float* out_scores, uint32_t index_base) {
    const uint32_t t = threadIdx.x;
    const uint32_t stride = blockDim.x;
    const uint32_t waves = stride >> 5u;
    uint32_t rank = k;
    uint32_t prefix = 0u;
    uint32_t out_pos = 0u;
    uint32_t cand_cnt = 0u;
    uint32_t index_prefix = 0xFFFFFFFFu;
    bool in_lds = false;
    bool select_all_ties = false;

    for (uint32_t pass = 0u; pass < 4u; ++pass) {
        const uint32_t shift = 24u - 8u * pass;
        if (t == 0u) {
            s->state[0] = 0u;
            s->state[1] = 0u;
            s->state[2] = 0u;
        }
        for (uint32_t i = t; i < 256u; i += stride) {
            for (uint32_t w = 0u; w < waves; ++w) s->hist[w][i] = 0u;
        }
        __syncthreads();
        if (!in_lds) {
            for (uint32_t i = t; i < usable; i += stride) {
                const uint32_t key = qsa_radix_key(scores[i]);
                if (pass == 0u || (key >> (shift + 8u)) == (prefix >> (shift + 8u)))
                    atomicAdd(&s->hist[t >> 5u][(key >> shift) & 0xFFu], 1u);
            }
        } else {
            for (uint32_t j = t; j < cand_cnt; j += stride) {
                const uint32_t key = s->cand_keys[j];
                if ((key >> (shift + 8u)) == (prefix >> (shift + 8u)))
                    atomicAdd(&s->hist[t >> 5u][(key >> shift) & 0xFFu], 1u);
            }
        }
        __syncthreads();
        uint32_t bucket_count = 0u;
        for (uint32_t w = 0u; w < waves; ++w) bucket_count += s->hist[w][255u - t];
        uint32_t exc = 0u;
        uint32_t total = 0u;
        qsa_radix_scan(bucket_count, s->wave_sums, exc, total);
        if (exc < rank && exc + bucket_count >= rank) {
            s->state[0] = 255u - t;
            s->state[1] = rank - exc;
            s->state[2] = bucket_count;
        }
        __syncthreads();
        prefix |= s->state[0] << shift;
        rank = s->state[1];
        const uint32_t count_here = s->state[2];
        if (rank == count_here) select_all_ties = true;
        if (!in_lds && count_here != 0u && count_here <= kQsaRadixCandidates) {
            if (t == 0u) {
                s->state[3] = out_pos;
                s->state[4] = 0u;
            }
            __syncthreads();
            const uint32_t ptop = prefix >> shift;
            for (uint32_t i = t; i < usable; i += stride) {
                const uint32_t key = qsa_radix_key(scores[i]);
                const uint32_t top = key >> shift;
                if (top > ptop) {
                    const uint32_t pos = atomicAdd(&s->state[3], 1u);
                    out[pos] = static_cast<std::int32_t>(index_base + i);
                    if (out_scores != nullptr) out_scores[pos] = scores[i];
                } else if (top == ptop) {
                    const uint32_t cpos = atomicAdd(&s->state[4], 1u);
                    if (cpos < kQsaRadixCandidates) {
                        s->cand_keys[cpos] = key;
                        s->cand_idx[cpos] = i;
                    }
                }
            }
            __syncthreads();
            out_pos = s->state[3];
            cand_cnt = s->state[4];
            in_lds = true;
        } else {
            __syncthreads();
        }
    }

    if (!select_all_ties) {
        uint32_t need = rank;
        for (uint32_t pass = 0u; pass < 4u && need != 0u; ++pass) {
            const uint32_t shift = 24u - 8u * pass;
            if (t == 0u) {
                s->state[0] = 0u;
                s->state[1] = 0u;
            }
            for (uint32_t i = t; i < 256u; i += stride) {
                for (uint32_t w = 0u; w < waves; ++w) s->hist[w][i] = 0u;
            }
            __syncthreads();
            if (in_lds) {
                for (uint32_t j = t; j < cand_cnt; j += stride) {
                    if (s->cand_keys[j] != prefix) continue;
                    const uint32_t idx = s->cand_idx[j];
                    if (pass != 0u &&
                        (idx >> (shift + 8u)) != (index_prefix >> (shift + 8u)))
                        continue;
                    atomicAdd(&s->hist[t >> 5u][(idx >> shift) & 0xFFu], 1u);
                }
            } else {
                for (uint32_t i = t; i < usable; i += stride) {
                    if (qsa_radix_key(scores[i]) != prefix) continue;
                    if (pass != 0u &&
                        (i >> (shift + 8u)) != (index_prefix >> (shift + 8u)))
                        continue;
                    atomicAdd(&s->hist[t >> 5u][(i >> shift) & 0xFFu], 1u);
                }
            }
            __syncthreads();
            uint32_t bucket_count = 0u;
            for (uint32_t w = 0u; w < waves; ++w) bucket_count += s->hist[w][t];
            uint32_t exc = 0u;
            uint32_t total = 0u;
            qsa_radix_scan(bucket_count, s->wave_sums, exc, total);
            if (exc < need && exc + bucket_count >= need) {
                s->state[0] = t;
                s->state[1] = need - exc;
            }
            __syncthreads();
            index_prefix =
                (index_prefix & ~(0xFFu << shift)) | (s->state[0] << shift);
            need = s->state[1];
            __syncthreads();
        }
    }

    if (t == 0u) s->state[3] = out_pos;
    __syncthreads();
    if (in_lds) {
        for (uint32_t j = t; j < cand_cnt; j += stride) {
            const uint32_t key = s->cand_keys[j];
            const uint32_t i = s->cand_idx[j];
            if (key > prefix || (key == prefix && i <= index_prefix)) {
                const uint32_t pos = atomicAdd(&s->state[3], 1u);
                out[pos] = static_cast<std::int32_t>(index_base + i);
                if (out_scores != nullptr) out_scores[pos] = scores[i];
            }
        }
    } else {
        for (uint32_t i = t; i < usable; i += stride) {
            const float value = scores[i];
            const uint32_t key = qsa_radix_key(value);
            if (key > prefix || (key == prefix && i <= index_prefix)) {
                const uint32_t pos = atomicAdd(&s->state[3], 1u);
                out[pos] = static_cast<std::int32_t>(index_base + i);
                if (out_scores != nullptr) out_scores[pos] = value;
            }
        }
    }
    __syncthreads();
}

__device__ void qsa_union_select(const float* values, const std::int32_t* block_index,
                                  uint32_t count, uint32_t k, QsaRadixScratch* s,
                                  float* out_values, std::int32_t* out_indices) {
    const uint32_t t = threadIdx.x;
    const uint32_t stride = blockDim.x;
    const uint32_t waves = stride >> 5u;
    if (t == 0u) s->state[3] = 0u;
    __syncthreads();
    for (uint32_t i = t; i < count; i += stride) {
        if (block_index[i] >= 0) atomicAdd(&s->state[3], 1u);
    }
    __syncthreads();
    const uint32_t valid = s->state[3];
    __syncthreads();
    if (k >= valid) {
        if (t == 0u) s->state[3] = 0u;
        __syncthreads();
        for (uint32_t i = t; i < count; i += stride) {
            if (block_index[i] < 0) continue;
            const uint32_t pos = atomicAdd(&s->state[3], 1u);
            out_values[pos] = values[i];
            out_indices[pos] = block_index[i];
        }
        __syncthreads();
        return;
    }
    uint32_t rank = k;
    uint32_t prefix = 0u;
    uint32_t above = 0u;
    for (uint32_t pass = 0u; pass < 4u; ++pass) {
        const uint32_t shift = 24u - 8u * pass;
        if (t == 0u) {
            s->state[0] = 0u;
            s->state[1] = 0u;
            s->state[2] = 0u;
        }
        for (uint32_t i = t; i < 256u; i += stride) {
            for (uint32_t w = 0u; w < waves; ++w) s->hist[w][i] = 0u;
        }
        __syncthreads();
        for (uint32_t i = t; i < count; i += stride) {
            if (block_index[i] < 0) continue;
            const uint32_t key = qsa_radix_key(values[i]);
            if (pass == 0u || (key >> (shift + 8u)) == (prefix >> (shift + 8u)))
                atomicAdd(&s->hist[t >> 5u][(key >> shift) & 0xFFu], 1u);
        }
        __syncthreads();
        uint32_t bucket_count = 0u;
        for (uint32_t w = 0u; w < waves; ++w) bucket_count += s->hist[w][255u - t];
        uint32_t exc = 0u;
        uint32_t total = 0u;
        qsa_radix_scan(bucket_count, s->wave_sums, exc, total);
        if (exc < rank && exc + bucket_count >= rank) {
            s->state[0] = 255u - t;
            s->state[1] = exc;
            s->state[2] = rank - exc;
        }
        __syncthreads();
        prefix |= s->state[0] << shift;
        above += s->state[1];
        rank = s->state[2];
        __syncthreads();
    }

    uint32_t need = k - above;
    uint32_t index_prefix = 0u;
    for (uint32_t pass = 0u; pass < 4u && need != 0u; ++pass) {
        const uint32_t shift = 24u - 8u * pass;
        if (t == 0u) {
            s->state[0] = 0u;
            s->state[1] = 0u;
        }
        for (uint32_t i = t; i < 256u; i += stride) {
            for (uint32_t w = 0u; w < waves; ++w) s->hist[w][i] = 0u;
        }
        __syncthreads();
        for (uint32_t i = t; i < count; i += stride) {
            if (block_index[i] < 0 || qsa_radix_key(values[i]) != prefix) continue;
            const uint32_t index = static_cast<uint32_t>(block_index[i]);
            if (pass != 0u && (index >> (shift + 8u)) != (index_prefix >> (shift + 8u)))
                continue;
            atomicAdd(&s->hist[t >> 5u][(index >> shift) & 0xFFu], 1u);
        }
        __syncthreads();
        uint32_t bucket_count = 0u;
        for (uint32_t w = 0u; w < waves; ++w) bucket_count += s->hist[w][t];
        uint32_t exc = 0u;
        uint32_t total = 0u;
        qsa_radix_scan(bucket_count, s->wave_sums, exc, total);
        if (exc < need && exc + bucket_count >= need) {
            s->state[0] = t;
            s->state[1] = need - exc;
        }
        __syncthreads();
        index_prefix |= s->state[0] << shift;
        need = s->state[1];
        __syncthreads();
    }
    if (t == 0u) s->state[3] = 0u;
    __syncthreads();
    for (uint32_t i = t; i < count; i += stride) {
        const std::int32_t index = block_index[i];
        if (index < 0) continue;
        const float value = values[i];
        if (qsa_radix_key(value) > prefix ||
            (qsa_radix_key(value) == prefix && static_cast<uint32_t>(index) <= index_prefix)) {
            const uint32_t pos = atomicAdd(&s->state[3], 1u);
            if (pos < k) {
                out_values[pos] = value;
                out_indices[pos] = index;
            }
        }
    }
    __syncthreads();
}

__global__ __launch_bounds__(kQsaTopkMaxThreads) void qsa_topk_kernel(QsaTopkArgs args) {
    const uint32_t row = blockIdx.x;
    if (row >= args.rows) return;
    const uint32_t t = threadIdx.x;

    __shared__ QsaRadixScratch radix_scratch;
    extern __shared__ uint8_t qsa_topk_dynamic[];

    const uint32_t top_k =
        args.top_k < kQsaSelectMaxTopK ? args.top_k : kQsaSelectMaxTopK;
    uint32_t half = kQsaSelectChunk;
    while (half < top_k) half <<= 1u;
    const uint32_t count = half * 2u;
    float* scores = reinterpret_cast<float*>(qsa_topk_dynamic);
    int32_t* indices =
        reinterpret_cast<int32_t*>(qsa_topk_dynamic + count * sizeof(float));
    float* out_scores = reinterpret_cast<float*>(
        qsa_topk_dynamic + count * (sizeof(float) + sizeof(int32_t)));
    int32_t* out_indices = reinterpret_cast<int32_t*>(
        qsa_topk_dynamic + count * (sizeof(float) + sizeof(int32_t)) +
        half * sizeof(float));
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
                usable, block_max, &radix_scratch, indices, nullptr, 0u);
    } else {
        for (uint32_t i = t; i < count; i += blockDim.x) {
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
                usable, want, &radix_scratch, indices + half, scores + half,
                args.block_begin);
        }

        if (args.legacy_merge != 0u || top_k < kQsaTopkUnionSelectMinTopK) {
            qsa_sort_candidates_ascending(scores, indices, half, half);
            qsa_merge_descending(scores, indices, count);
        } else {
            const uint32_t keep = args.last != 0u ? block_max : half;
            for (uint32_t i = t; i < half; i += blockDim.x) {
                out_scores[i] = -INFINITY;
                out_indices[i] = -1;
            }
            __syncthreads();
            qsa_union_select(scores, indices, count, keep, &radix_scratch,
                             out_scores, out_indices);
            for (uint32_t i = t; i < half; i += blockDim.x) {
                scores[i] = out_scores[i];
                indices[i] = out_indices[i];
            }
            __syncthreads();
        }

        if (args.last == 0u) {
            for (uint32_t i = t; i < half; i += blockDim.x) {
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
        uint32_t width = 1u;
        while (width < block_max) width <<= 1u;
        for (uint32_t i = block_max + t; i < width; i += blockDim.x) indices[i] = INT32_MAX;
        __syncthreads();
        for (uint32_t size = 2u; size <= width; size <<= 1u) {
            for (uint32_t distance = size >> 1u; distance; distance >>= 1u) {
                for (uint32_t pair = t; pair < width / 2u; pair += blockDim.x) {
                    const uint32_t i = (pair / distance) * (distance * 2u) + pair % distance;
                    const int32_t x = indices[i], y = indices[i + distance];
                    const bool ascending = (i & size) == 0u;
                    indices[i] = ascending ? min(x, y) : max(x, y);
                    indices[i + distance] = ascending ? max(x, y) : min(x, y);
                }
                __syncthreads();
            }
        }
        for (uint32_t i = t; i < args.top_k; i += blockDim.x)
            out[i] = i < block_max ? indices[i] : -1;
    }
    uint32_t* words = reinterpret_cast<uint32_t*>(
        args.block_mask + static_cast<uint64_t>(row) * args.mask_row_bytes);
    const uint32_t mask_words = (args.blocks + 31u) / 32u;
    for (uint32_t i = t; i < mask_words; i += blockDim.x) words[i] = 0u;
    __syncthreads();
    for (uint32_t i = t; i < block_max; i += blockDim.x) {
        const uint32_t block = static_cast<uint32_t>(indices[i]);
        if (block < args.blocks) atomicOr(&words[block / 32u], 1u << (block % 32u));
    }
    const uint32_t complete = (args.positions[row] + 1u) / args.compress_ratio;
    if (t == 0u && complete < args.blocks)
        atomicOr(&words[complete / 32u], 1u << (complete % 32u));
}

__device__ __forceinline__ float qsa_bf16_bits_value(uint32_t bits) {
    return __uint_as_float(bits << 16u);
}

__device__ __forceinline__ uint32_t qsa_bf16_bits_from(float value) {
    return static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16(value)));
}

__global__ __launch_bounds__(kQsaHeadDim) void qsa_rope_kernel(QsaRopeArgs args) {
    const uint32_t row = blockIdx.x;
    const uint32_t half = args.rotary_dim / 2u;
    const float position = static_cast<float>(args.positions[row]);
    __shared__ float s_cos[64];
    __shared__ float s_sin[64];
    for (uint32_t j = threadIdx.x; j < half; j += blockDim.x) {
        const float exponent = -2.0f * static_cast<float>(j) /
                               static_cast<float>(args.rotary_dim);
        const float frequency = args.inv_freq != nullptr
                                    ? static_cast<float>(args.inv_freq[j])
                                    : powf(args.theta, exponent);
        const float angle = (position / args.position_scale) * frequency;
        s_cos[j] = cosf(angle) * args.attention_factor;
        s_sin[j] = sinf(angle) * args.attention_factor;
    }
    __syncthreads();
    const bool aligned = (reinterpret_cast<uintptr_t>(args.x) & 15u) == 0u && half % 8u == 0u;
    const uint32_t chunks = aligned ? half / 8u : 0u;
    for (uint32_t head = 0; head < args.heads; ++head) {
        shelf::bf16_t* base =
            args.x + (static_cast<std::uint64_t>(row) * args.heads + head) * args.head_dim;
        for (uint32_t chunk = threadIdx.x; chunk < chunks; chunk += blockDim.x) {
            const uint32_t offset = chunk * 8u;
            const uint4 low = *reinterpret_cast<const uint4*>(base + offset);
            const uint4 high = *reinterpret_cast<const uint4*>(base + half + offset);
            const uint32_t lw[4] = {low.x, low.y, low.z, low.w};
            const uint32_t hw[4] = {high.x, high.y, high.z, high.w};
            uint32_t lo[4];
            uint32_t hi[4];
            #pragma unroll
            for (uint32_t p = 0u; p < 4u; ++p) {
                const uint32_t j = offset + 2u * p;
                const float c0 = s_cos[j];
                const float s0 = s_sin[j];
                const float c1 = s_cos[j + 1u];
                const float s1 = s_sin[j + 1u];
                const float a0 = qsa_bf16_bits_value(lw[p] & 0xFFFFu);
                const float a1 = qsa_bf16_bits_value(lw[p] >> 16u);
                const float b0 = qsa_bf16_bits_value(hw[p] & 0xFFFFu);
                const float b1 = qsa_bf16_bits_value(hw[p] >> 16u);
                lo[p] = qsa_bf16_bits_from(a0 * c0 - b0 * s0) |
                        (qsa_bf16_bits_from(a1 * c1 - b1 * s1) << 16u);
                hi[p] = qsa_bf16_bits_from(b0 * c0 + a0 * s0) |
                        (qsa_bf16_bits_from(b1 * c1 + a1 * s1) << 16u);
            }
            *reinterpret_cast<uint4*>(base + offset) =
                uint4{lo[0], lo[1], lo[2], lo[3]};
            *reinterpret_cast<uint4*>(base + half + offset) =
                uint4{hi[0], hi[1], hi[2], hi[3]};
        }
        for (uint32_t j = chunks * 8u + threadIdx.x; j < half; j += blockDim.x) {
            const float x0 = bf16_value(base + j);
            const float x1 = bf16_value(base + j + half);
            base[j] = bf16_from(x0 * s_cos[j] - x1 * s_sin[j]);
            base[j + half] = bf16_from(x1 * s_cos[j] + x0 * s_sin[j]);
        }
    }
}

}  // namespace

hipError_t launch_qsa_pool(const QsaPoolArgs& args, hipStream_t stream) {
    if (args.raw_keys == nullptr || args.norm_weight == nullptr || args.pooled == nullptr) {
        return hipErrorInvalidValue;
    }
    if (args.rows == 0u || args.blocks == 0u || args.head_dim != kQsaHeadDim ||
        args.compress_ratio == 0u || args.blocks * args.compress_ratio > args.rows) {
        return hipErrorInvalidValue;
    }
    qsa_pool_kernel<<<args.blocks, kThreads, 0, stream>>>(args);
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

hipError_t launch_qsa_score_wmma_batched(const QsaScoreBatchedArgs& args,
                                         hipStream_t stream) {
    if (args.q == nullptr || args.pooled_base == nullptr || args.scores == nullptr ||
        args.row_sequence_slots == nullptr || args.requests == nullptr ||
        args.request_count == 0u || args.row_tile_rows == 0u || args.block_count == 0u ||
        args.max_row_groups == 0u || args.max_block_tiles == 0u || args.max_blocks == 0u ||
        args.n_heads != 4u || args.head_dim != kQsaHeadDim ||
        args.compress_ratio == 0u || args.max_sequences == 0u ||
        args.score_row_stride < args.block_count ||
        args.q_row_stride < 4u * kQsaHeadDim ||
        args.pooled_slot_stride < args.pooled_layer_stride ||
        args.pooled_layer_stride <
            static_cast<uint64_t>(args.max_blocks) * kQsaHeadDim ||
        args.scale == 0.0f) {
        return hipErrorInvalidValue;
    }
    const uint32_t tile = args.block_tile == 0u ? kQsaScoreBlocksPerCta : args.block_tile;
    if (tile > kQsaScoreBlockTile || args.max_block_tiles > kQsaScoreBlockTile) {
        return hipErrorInvalidValue;
    }
    const dim3 grid(args.max_row_groups, args.max_block_tiles, args.request_count);
    qsa_score_slices_kernel<<<grid, kQsaThreads, 0, stream>>>(args);
    return hipGetLastError();
}

hipError_t launch_qsa_topk(const QsaTopkArgs& args, hipStream_t stream) {
    if (args.scores == nullptr || args.state_scores == nullptr ||
        args.state_indices == nullptr || args.block_mask == nullptr ||
        args.positions == nullptr || args.rows == 0u ||
        args.top_k == 0u || args.top_k > kQsaSelectMaxTopK ||
        args.block_begin > args.blocks || args.block_count > args.blocks - args.block_begin ||
        args.compress_ratio == 0u || (args.mask_row_bytes & 3u) != 0u ||
        (reinterpret_cast<uintptr_t>(args.block_mask) & 3u) != 0u ||
        args.score_row_stride < args.block_count ||
        args.mask_row_bytes < ((static_cast<uint64_t>(args.blocks) + 31u) / 32u) * 4u) {
        return hipErrorInvalidValue;
    }
    if (args.selected != nullptr && args.selected_row_stride < args.top_k) {
        return hipErrorInvalidValue;
    }
    uint32_t half = kQsaSelectChunk;
    while (half < args.top_k) half <<= 1u;
    const std::size_t dynamic_bytes =
        3u * static_cast<std::size_t>(half) *
        (sizeof(float) + sizeof(std::int32_t));
    qsa_topk_kernel<<<args.rows, kQsaRadixWaves * 32u, dynamic_bytes, stream>>>(args);
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
