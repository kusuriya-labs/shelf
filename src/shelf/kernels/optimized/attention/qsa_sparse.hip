#include <shelf/kernels/optimized/attention/qsa_sparse.h>
#include <shelf/kernels/optimized/detail/lds_barrier_device.h>
#include <shelf/kernels/fused/detail/octave_common.h>
#include <shelf/quantization/codec/e4m3_detail.h>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace shelf::kernel {
namespace {

using bf16 = __hip_bfloat16;
namespace oct = ::shelf::kernel::detail::octave;

constexpr uint32_t kD = kQsaSparseHeadDim;
constexpr uint32_t kR = kQsaSparseBlockTokens;
constexpr uint32_t kHeadsPerWarp = 3u;
constexpr uint32_t kWarps = 2u;
constexpr uint32_t kThreads = kWarps * 32u;
constexpr uint32_t kDimsPerLane = kD / 32u;
constexpr uint32_t kScorerLanes = kHeadsPerWarp * kR;
constexpr uint32_t kPartialStride = kQsaSparseQPerKv * (2u + kD);

template <KVCacheDType DType>
struct QsaKVReader;

template <>
struct QsaKVReader<KVCacheDType::BF16> {
    struct Block {
        uint4 cell[kR];
    };

    __device__ __forceinline__ static void load_k(const uint8_t* base, uint32_t stride,
                                                  uint32_t lane, uint32_t cells,
                                                  Block& raw) {
#pragma unroll
        for (uint32_t c = 0; c < kR; ++c) {
            if (c < cells)
                raw.cell[c] = *reinterpret_cast<const uint4*>(
                    base + static_cast<std::size_t>(c) * stride +
                    lane * kDimsPerLane * sizeof(bf16));
            else
                raw.cell[c] = uint4{0u, 0u, 0u, 0u};
        }
    }

    __device__ __forceinline__ static void load_v(const uint8_t* base, uint32_t stride,
                                                  uint32_t lane, uint32_t cells,
                                                  Block& raw) {
        load_k(base, stride, lane, cells, raw);
    }

    __device__ __forceinline__ static void values(const Block& raw, uint32_t c,
                                                  float (&x)[kDimsPerLane]) {
        const bf16* v = reinterpret_cast<const bf16*>(&raw.cell[c]);
#pragma unroll
        for (uint32_t d = 0; d < kDimsPerLane; ++d) x[d] = __bfloat162float(v[d]);
    }
};

template <>
struct QsaKVReader<KVCacheDType::FP8_E4M3> {
    struct Block {
        float cell[kR][kDimsPerLane];
    };

    __device__ __forceinline__ static void load_k(const uint8_t* base, uint32_t stride,
                                                  uint32_t lane, uint32_t cells,
                                                  uint32_t scale_stride,
                                                  const float* scales, Block& raw) {
#pragma unroll
        for (uint32_t c = 0; c < kR; ++c) {
            if (c >= cells) {
#pragma unroll
                for (uint32_t d = 0; d < kDimsPerLane; ++d) raw.cell[c][d] = 0.0f;
                continue;
            }
            const uint8_t* codes =
                base + static_cast<std::size_t>(c) * stride + lane * kDimsPerLane;
            const float scale = scales[c * scale_stride];
            const uint32_t p0 = *reinterpret_cast<const uint32_t*>(codes);
            const uint32_t p1 = *reinterpret_cast<const uint32_t*>(codes + 4);
            raw.cell[c][0] = __builtin_amdgcn_cvt_f32_fp8(static_cast<int>(p0), 0) * scale;
            raw.cell[c][1] = __builtin_amdgcn_cvt_f32_fp8(static_cast<int>(p0), 1) * scale;
            raw.cell[c][2] = __builtin_amdgcn_cvt_f32_fp8(static_cast<int>(p0), 2) * scale;
            raw.cell[c][3] = __builtin_amdgcn_cvt_f32_fp8(static_cast<int>(p0), 3) * scale;
            raw.cell[c][4] = __builtin_amdgcn_cvt_f32_fp8(static_cast<int>(p1), 0) * scale;
            raw.cell[c][5] = __builtin_amdgcn_cvt_f32_fp8(static_cast<int>(p1), 1) * scale;
            raw.cell[c][6] = __builtin_amdgcn_cvt_f32_fp8(static_cast<int>(p1), 2) * scale;
            raw.cell[c][7] = __builtin_amdgcn_cvt_f32_fp8(static_cast<int>(p1), 3) * scale;
        }
    }

    __device__ __forceinline__ static void load_v(const uint8_t* base, uint32_t stride,
                                                  uint32_t lane, uint32_t cells,
                                                  uint32_t scale_stride,
                                                  const float* scales, Block& raw) {
        load_k(base, stride, lane, cells, scale_stride, scales, raw);
    }

    __device__ __forceinline__ static void values(const Block& raw, uint32_t c,
                                                  float (&x)[kDimsPerLane]) {
#pragma unroll
        for (uint32_t d = 0; d < kDimsPerLane; ++d) x[d] = raw.cell[c][d];
    }
};

template <>
struct QsaKVReader<KVCacheDType::OCTAVE> {
    struct Block {
        float cell[kR][kDimsPerLane];
    };

    __device__ __forceinline__ static void load_k(const uint8_t* base, uint32_t stride,
                                                  uint32_t lane, uint32_t cells,
                                                  OctaveKVCacheFormat format,
                                                  const float* levels, Block& raw) {
        const uint32_t slot_bytes = octave_k_slot_bytes(format);
        const uint32_t segment =
            format.k_mode == OctaveKMode::Compact ? 256u : 64u;
        const uint32_t dim = lane * kDimsPerLane;
        const uint32_t plane = dim / segment;
        const uint32_t local = dim & (segment - 1u);
        const bool compact = format.k_mode == OctaveKMode::Compact;
        const uint32_t bits = (compact || plane == 0u) ? format.k_nope_bits
                                                       : format.k_rope_bits;
        const uint32_t code_offset = (compact || plane == 0u)
            ? 0u
            : octave_plane_bytes(64u, format.k_rope_bits) +
                  (plane - 1u) * octave_plane_bytes(64u, format.k_nope_bits);
        const uint32_t scale_count = compact ? 1u : 4u;
#pragma unroll
        for (uint32_t c = 0; c < kR; ++c) {
            if (c >= cells) {
#pragma unroll
                for (uint32_t d = 0; d < kDimsPerLane; ++d) raw.cell[c][d] = 0.0f;
                continue;
            }
            const uint8_t* token = base + static_cast<std::size_t>(c) * stride;
            const float scale = oct::half_at(
                token + slot_bytes - scale_count * 2u + plane * 2u);
            uint32_t codes[kDimsPerLane];
            oct::unpack8(token + code_offset, local, segment, bits, codes);
            float values[kDimsPerLane];
#pragma unroll
            for (uint32_t d = 0; d < kDimsPerLane; ++d)
                values[d] = levels[bits * 64u + codes[d]] * scale;
            oct::inverse_wave(values, lane, compact);
#pragma unroll
            for (uint32_t d = 0; d < kDimsPerLane; ++d)
                raw.cell[c][d] = values[d] * oct::sign_of(dim + d, 11u);

        }
    }

    __device__ __forceinline__ static void load_v(const uint8_t* base, uint32_t stride,
                                                  uint32_t lane, uint32_t cells,
                                                  OctaveKVCacheFormat format,
                                                  const float* levels, Block& raw) {
        const uint32_t slot_bytes = octave_v_slot_bytes(format);
        const uint32_t dim = lane * kDimsPerLane;
#pragma unroll
        for (uint32_t c = 0; c < kR; ++c) {
            if (c >= cells) {
#pragma unroll
                for (uint32_t d = 0; d < kDimsPerLane; ++d) raw.cell[c][d] = 0.0f;
                continue;
            }
            const uint8_t* token = base + static_cast<std::size_t>(c) * stride;
            const float scale = oct::half_at(token + slot_bytes - 2u);
            uint32_t codes[kDimsPerLane];
            oct::unpack8(token, dim, 256u, format.v_bits, codes);
            float values[kDimsPerLane];
#pragma unroll
            for (uint32_t d = 0; d < kDimsPerLane; ++d)
                values[d] = levels[format.v_bits * 64u + codes[d]] * scale;
            oct::inverse_wave(values, lane, true);
#pragma unroll
            for (uint32_t d = 0; d < kDimsPerLane; ++d)
                raw.cell[c][d] = values[d] * oct::sign_of(dim + d, 29u);
        }
    }

    __device__ __forceinline__ static void values(const Block& raw, uint32_t c,
                                                  float (&x)[kDimsPerLane]) {
#pragma unroll
        for (uint32_t d = 0; d < kDimsPerLane; ++d) x[d] = raw.cell[c][d];
    }
};

__device__ __forceinline__ void load_q(const bf16* src, uint32_t lane,
                                       float (&q)[kDimsPerLane], float scale) {
    const uint4 w = *reinterpret_cast<const uint4*>(src + lane * kDimsPerLane);
    const bf16* values = reinterpret_cast<const bf16*>(&w);
#pragma unroll
    for (uint32_t d = 0; d < kDimsPerLane; ++d)
        q[d] = __bfloat162float(values[d]) * scale;
}

template <KVCacheDType DType, bool kSplit>
__global__ __launch_bounds__(kThreads) void qsa_sparse_attention_kernel(
    const PagedAttentionCommonArgs a, const uint8_t* __restrict__ k_pool,
    const uint8_t* __restrict__ v_pool, const float* __restrict__ k_scales,
    const float* __restrict__ v_scales, const float* __restrict__ octave_levels,
    OctaveKVCacheFormat octave_format, float* __restrict__ partials, uint32_t splits) {
    static_assert(DType == KVCacheDType::BF16 || DType == KVCacheDType::FP8_E4M3 ||
                      DType == KVCacheDType::OCTAVE,
                  "Unsupported QSA Sparse KV dtype");
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t split = kSplit ? blockIdx.x : 0u;
    const uint32_t h_kv = blockIdx.y;
    const uint32_t row = blockIdx.z;
    const uint32_t h0 = h_kv * kQsaSparseQPerKv + warp * kHeadsPerWarp;

    __shared__ float red[kWarps][32][kScorerLanes + 1u];

    if (row >= a.rows) return;
    const uint32_t slot = a.row_sequence_slots[row];
    const uint32_t visible = a.row_positions[row] + 1u;

    float q[kHeadsPerWarp][kDimsPerLane];
#pragma unroll
    for (uint32_t h = 0; h < kHeadsPerWarp; ++h) {
        const bf16* q_head = reinterpret_cast<const bf16*>(a.q) +
                             static_cast<std::size_t>(row) * a.q_row_stride +
                             static_cast<std::size_t>(h0 + h) * kD;
        load_q(q_head, lane, q[h], a.scale);
    }

    float acc[kHeadsPerWarp][kDimsPerLane];
#pragma unroll
    for (uint32_t h = 0; h < kHeadsPerWarp; ++h) {
#pragma unroll
        for (uint32_t d = 0; d < kDimsPerLane; ++d) acc[h][d] = 0.0f;
    }
    float m_run = -INFINITY;
    float l_run = 0.0f;

    const uint32_t my_c = lane % kR;
    const bool scorer = lane < kScorerLanes;

    const uint32_t visible_blocks = (visible + kR - 1u) / kR;
    const uint32_t nblk =
        visible_blocks < a.mask_blocks ? visible_blocks : a.mask_blocks;
    uint32_t j0 = 0u;
    uint32_t j1 = nblk;
    if (kSplit) {
        const uint32_t per = (nblk + splits - 1u) / splits;
        j0 = split * per;
        j1 = j0 + per < nblk ? j0 + per : nblk;
    }

    if (slot < a.max_sequences && j0 < j1) {
        const uint32_t* block_table =
            a.block_tables + static_cast<std::size_t>(slot) * a.block_table_stride;
        const uint32_t* words = reinterpret_cast<const uint32_t*>(
            a.block_mask + static_cast<std::size_t>(row) * a.mask_stride);
        const uint32_t k_slot_bytes = octave_k_slot_bytes(octave_format);
        const uint32_t v_slot_bytes = octave_v_slot_bytes(octave_format);
        const uint32_t elem_bytes =
            DType == KVCacheDType::FP8_E4M3
                ? 1u
                : (DType == KVCacheDType::OCTAVE ? k_slot_bytes
                                                 : static_cast<uint32_t>(sizeof(bf16)));
        const uint8_t* k_layer = DType == KVCacheDType::OCTAVE
            ? k_pool + static_cast<std::size_t>(a.layer) * a.elems_per_layer /
                           a.head_dim * k_slot_bytes
            : k_pool + static_cast<std::size_t>(a.layer) * a.elems_per_layer * elem_bytes;
        const uint8_t* v_layer = DType == KVCacheDType::OCTAVE
            ? v_pool + static_cast<std::size_t>(a.layer) * a.elems_per_layer /
                           a.head_dim * v_slot_bytes
            : v_pool + static_cast<std::size_t>(a.layer) * a.elems_per_layer * elem_bytes;
        const float* k_scale_layer = k_scales != nullptr
            ? k_scales + static_cast<std::size_t>(a.layer) * a.num_pages *
                             a.page_tokens * a.kv_heads
            : nullptr;
        const float* v_scale_layer = v_scales != nullptr
            ? v_scales + static_cast<std::size_t>(a.layer) * a.num_pages *
                             a.page_tokens * a.kv_heads
            : nullptr;
        const uint32_t blocks_per_page = a.page_tokens / kR;
        const uint32_t last_word = (j1 - 1u) >> 5u;

        for (uint32_t wi = j0 >> 5u; wi <= last_word; ++wi) {
            uint32_t bits = words[wi];
            if (wi == (j0 >> 5u)) bits &= 0xFFFFFFFFu << (j0 & 31u);
            if (wi == last_word) {
                const uint32_t high = (j1 - 1u) & 31u;
                if (high != 31u) bits &= (1u << (high + 1u)) - 1u;
            }
            while (bits != 0u) {
                const uint32_t block = wi * 32u + static_cast<uint32_t>(__ffs(static_cast<int>(bits)) - 1);
                bits &= bits - 1u;
                const uint32_t token0 = block * kR;
                const uint32_t cells = visible - token0 < kR ? visible - token0 : kR;
                const uint32_t page = block / blocks_per_page;
                const uint32_t off = token0 - page * a.page_tokens;
                const uint32_t phys = block_table[page];
                if (phys >= a.num_pages) continue;
                const std::size_t elem = static_cast<std::size_t>(phys) * a.elems_per_page +
                                         static_cast<std::size_t>(off) * a.elems_per_token +
                                         static_cast<std::size_t>(h_kv) * kD;
                const std::size_t scale_index =
                    (static_cast<std::size_t>(phys) * a.page_tokens + off) *
                        a.kv_heads + h_kv;
                typename QsaKVReader<DType>::Block kr;
                typename QsaKVReader<DType>::Block vr;
                if constexpr (DType == KVCacheDType::OCTAVE) {
                    const std::size_t token_head = elem / a.head_dim;
                    QsaKVReader<DType>::load_k(
                        k_layer + token_head * k_slot_bytes, k_slot_bytes, lane, cells,
                        octave_format, octave_levels, kr);
                    QsaKVReader<DType>::load_v(
                        v_layer + token_head * v_slot_bytes, v_slot_bytes, lane, cells,
                        octave_format, octave_levels, vr);
                } else if constexpr (DType == KVCacheDType::FP8_E4M3) {
                    const uint32_t byte_stride = a.elems_per_token * elem_bytes;
                    const float* k_ptr = k_scale_layer + scale_index;
                    const float* v_ptr = v_scale_layer + scale_index;
                    QsaKVReader<DType>::load_k(k_layer + elem * elem_bytes, byte_stride,
                                               lane, cells, a.kv_heads, k_ptr, kr);
                    QsaKVReader<DType>::load_v(v_layer + elem * elem_bytes, byte_stride,
                                               lane, cells, a.kv_heads, v_ptr, vr);
                } else {
                    const uint32_t byte_stride = a.elems_per_token * elem_bytes;
                    QsaKVReader<DType>::load_k(k_layer + elem * elem_bytes, byte_stride,
                                               lane, cells, kr);
                    QsaKVReader<DType>::load_v(v_layer + elem * elem_bytes, byte_stride,
                                               lane, cells, vr);
                }


                float part[kHeadsPerWarp][kR];
#pragma unroll
                for (uint32_t c = 0; c < kR; ++c) {
                    float kx[kDimsPerLane];
                    QsaKVReader<DType>::values(kr, c, kx);
#pragma unroll
                    for (uint32_t h = 0; h < kHeadsPerWarp; ++h) {
                        float sum = 0.0f;
#pragma unroll
                        for (uint32_t d = 0; d < kDimsPerLane; ++d) sum += q[h][d] * kx[d];
                        part[h][c] = sum;
                    }
                }
#pragma unroll
                for (uint32_t h = 0; h < kHeadsPerWarp; ++h) {
#pragma unroll
                    for (uint32_t c = 0; c < kR; ++c) red[warp][lane][h * kR + c] = part[h][c];
                }
                detail::lds_barrier();
                float s = -INFINITY;
                if (scorer) {
                    float sum = 0.0f;
#pragma unroll
                    for (uint32_t i = 0; i < 32u; ++i) sum += red[warp][i][lane];
                    s = my_c < cells ? sum : -INFINITY;
                }
                detail::lds_barrier();

                float bmax = s;
#pragma unroll
                for (uint32_t offset = 1u; offset < kR; offset <<= 1u)
                    bmax = fmaxf(bmax, __shfl_xor(bmax, offset, 32));
                const float m_new = fmaxf(m_run, bmax);
                const float alpha = m_new == -INFINITY ? 1.0f : __expf(m_run - m_new);
                const float p = s == -INFINITY ? 0.0f : __expf(s - m_new);
                float psum = p;
#pragma unroll
                for (uint32_t offset = 1u; offset < kR; offset <<= 1u)
                    psum += __shfl_xor(psum, offset, 32);
                m_run = m_new;
                l_run = l_run * alpha + psum;

#pragma unroll
                for (uint32_t h = 0; h < kHeadsPerWarp; ++h) {
                    const float al = __shfl_sync(0xFFFFFFFFULL, alpha, h * kR);
#pragma unroll
                    for (uint32_t d = 0; d < kDimsPerLane; ++d) acc[h][d] *= al;
                }
#pragma unroll
                for (uint32_t c = 0; c < kR; ++c) {
                    if (c >= cells) continue;
                    float vx[kDimsPerLane];
                    QsaKVReader<DType>::values(vr, c, vx);
#pragma unroll
                    for (uint32_t h = 0; h < kHeadsPerWarp; ++h) {
                        const float pc = __shfl_sync(0xFFFFFFFFULL, p, h * kR + c);
#pragma unroll
                        for (uint32_t d = 0; d < kDimsPerLane; ++d) acc[h][d] += pc * vx[d];
                    }
                }
            }
        }
    }

    float mh[kHeadsPerWarp];
    float lh[kHeadsPerWarp];
#pragma unroll
    for (uint32_t h = 0; h < kHeadsPerWarp; ++h) {
        mh[h] = __shfl_sync(0xFFFFFFFFULL, m_run, h * kR);
        lh[h] = __shfl_sync(0xFFFFFFFFULL, l_run, h * kR);
    }

    if (!kSplit) {
        float* out = static_cast<float*>(a.output) +
                     static_cast<std::size_t>(row) * a.output_row_stride +
                     static_cast<std::size_t>(h0) * kD + lane * kDimsPerLane;
#pragma unroll
        for (uint32_t h = 0; h < kHeadsPerWarp; ++h) {
            const float inv = lh[h] > 0.0f ? 1.0f / lh[h] : 0.0f;
            float* o = out + static_cast<std::size_t>(h) * kD;
#pragma unroll
            for (uint32_t d = 0; d < kDimsPerLane; ++d) o[d] = acc[h][d] * inv;
        }
        return;
    }

    float* base = partials +
                  (static_cast<std::size_t>(row) * a.kv_heads + h_kv) * splits *
                      kPartialStride +
                  static_cast<std::size_t>(split) * kPartialStride;
#pragma unroll
    for (uint32_t h = 0; h < kHeadsPerWarp; ++h) {
        const uint32_t local = warp * kHeadsPerWarp + h;
        float* o = base + 2u * kQsaSparseQPerKv + local * kD + lane * kDimsPerLane;
#pragma unroll
        for (uint32_t d = 0; d < kDimsPerLane; ++d) o[d] = acc[h][d];
    }
    if (lane == 0u) {
#pragma unroll
        for (uint32_t h = 0; h < kHeadsPerWarp; ++h) {
            const uint32_t local = warp * kHeadsPerWarp + h;
            base[local] = mh[h];
            base[kQsaSparseQPerKv + local] = lh[h];
        }
    }
}

__global__ __launch_bounds__(kD) void qsa_sparse_reduce_kernel(
    const PagedAttentionCommonArgs a, const float* __restrict__ partials, uint32_t splits) {
    const uint32_t task = blockIdx.x;
    const uint32_t row = task / a.q_heads;
    const uint32_t q_head = task % a.q_heads;
    const uint32_t kv_head = q_head / kQsaSparseQPerKv;
    const uint32_t local = q_head % kQsaSparseQPerKv;
    const uint32_t d = threadIdx.x;
    if (row >= a.rows) return;

    const float* base = partials +
                       (static_cast<std::size_t>(row) * a.kv_heads + kv_head) * splits *
                           kPartialStride;
    float m = -INFINITY;
    for (uint32_t s = 0; s < splits; ++s)
        m = fmaxf(m, base[static_cast<std::size_t>(s) * kPartialStride + local]);

    float l = 0.0f;
    float value = 0.0f;
    if (m != -INFINITY) {
        for (uint32_t s = 0; s < splits; ++s) {
            const std::size_t offset = static_cast<std::size_t>(s) * kPartialStride;
            const float w = __expf(base[offset + local] - m);
            l += w * base[offset + kQsaSparseQPerKv + local];
            value += w * base[offset + 2u * kQsaSparseQPerKv + local * kD + d];
        }
    }
    float* out = static_cast<float*>(a.output) +
                 static_cast<std::size_t>(row) * a.output_row_stride +
                 static_cast<std::size_t>(q_head) * kD + d;
    *out = l > 0.0f ? value / l : 0.0f;
}

}  // namespace

uint64_t qsa_sparse_partial_bytes(uint32_t rows, uint32_t kv_heads, uint32_t splits) {
    return static_cast<uint64_t>(rows) * kv_heads * splits * kPartialStride * sizeof(float);
}

uint32_t qsa_sparse_splits(uint32_t rows, uint32_t kv_heads, uint64_t workspace_bytes) {
    if (rows == 0u || kv_heads == 0u) return 1u;
    constexpr uint32_t kTargetCtas = 128u;
    constexpr uint32_t kMaxSplits = 32u;
    uint32_t splits = 1u;
    while (splits < kMaxSplits &&
           static_cast<uint64_t>(rows) * kv_heads * (splits << 1u) <= kTargetCtas)
        splits <<= 1u;
    while (splits > 1u && qsa_sparse_partial_bytes(rows, kv_heads, splits) > workspace_bytes)
        splits >>= 1u;
    return splits;
}

template <KVCacheDType DType>
hipError_t launch_qsa_sparse_attention_typed(
    const PagedAttentionCommonArgs& common,
    const uint8_t* k_pool, const uint8_t* v_pool,
    const float* k_scales, const float* v_scales,
    const float* octave_levels, OctaveKVCacheFormat octave_format,
    float* partials, uint32_t splits, hipStream_t stream) {
    if (common.q == nullptr || common.output == nullptr || common.block_mask == nullptr ||
        common.row_positions == nullptr || common.row_sequence_slots == nullptr ||
        common.block_tables == nullptr || k_pool == nullptr || v_pool == nullptr)
        return hipErrorInvalidValue;
    if (DType == KVCacheDType::FP8_E4M3 &&
        (k_scales == nullptr || v_scales == nullptr))
        return hipErrorInvalidValue;
    if (DType == KVCacheDType::OCTAVE &&
        (octave_levels == nullptr || !octave_format_valid(octave_format)))
        return hipErrorInvalidValue;
    if (common.rows == 0u || common.kv_heads == 0u || splits == 0u) return hipErrorInvalidValue;
    if (common.head_dim != kD || common.q_heads != common.kv_heads * kQsaSparseQPerKv ||
        common.output_dtype != PagedAttentionOutputDType::F32 ||
        common.mask_block_shift != 2u || common.page_tokens == 0u ||
        (common.page_tokens % kR) != 0u ||
        (common.elems_per_token % kDimsPerLane) != 0u ||
        (common.q_row_stride % kDimsPerLane) != 0u ||
        (common.output_row_stride % kDimsPerLane) != 0u)
        return hipErrorInvalidValue;
    if (splits > 1u && partials == nullptr) return hipErrorInvalidValue;
    if (common.tree_anc_offsets != nullptr) return hipErrorInvalidValue;

    const dim3 grid(splits > 1u ? splits : 1u, common.kv_heads, common.rows);
    const dim3 block(32u, kWarps);
    if (splits > 1u) {
        qsa_sparse_attention_kernel<DType, true><<<grid, block, 0, stream>>>(
            common, k_pool, v_pool, k_scales, v_scales, octave_levels,
            octave_format, partials, splits);
    } else {
        qsa_sparse_attention_kernel<DType, false><<<grid, block, 0, stream>>>(
            common, k_pool, v_pool, k_scales, v_scales, octave_levels,
            octave_format, partials, splits);
    }
    return hipGetLastError();
}

hipError_t launch_qsa_sparse_attention(const PagedAttentionCommonArgs& common,
                                       const bf16_t* k_pool, const bf16_t* v_pool,
                                       float* partials, uint32_t splits,
                                       hipStream_t stream) {
    return launch_qsa_sparse_attention_typed<KVCacheDType::BF16>(
        common, reinterpret_cast<const uint8_t*>(k_pool),
        reinterpret_cast<const uint8_t*>(v_pool), nullptr, nullptr, nullptr,
        OctaveKVCacheFormat{}, partials, splits, stream);
}

hipError_t launch_qsa_sparse_attention_fp8(
    const PagedAttentionCommonArgs& common,
    const uint8_t* k_pool, const uint8_t* v_pool,
    const float* k_scales, const float* v_scales,
    float* partials, uint32_t splits, hipStream_t stream) {
    return launch_qsa_sparse_attention_typed<KVCacheDType::FP8_E4M3>(
        common, k_pool, v_pool, k_scales, v_scales, nullptr,
        OctaveKVCacheFormat{}, partials, splits, stream);
}

hipError_t launch_qsa_sparse_attention_octave(
    const PagedAttentionCommonArgs& common,
    const uint8_t* k_pool, const uint8_t* v_pool,
    const float* levels, OctaveKVCacheFormat format,
    float* partials, uint32_t splits, hipStream_t stream) {
    return launch_qsa_sparse_attention_typed<KVCacheDType::OCTAVE>(
        common, k_pool, v_pool, nullptr, nullptr, levels, format,
        partials, splits, stream);
}

hipError_t launch_qsa_sparse_reduce(const PagedAttentionCommonArgs& common,
                                    const float* partials, uint32_t splits,
                                    hipStream_t stream) {
    if (common.output == nullptr || partials == nullptr || common.rows == 0u ||
        common.q_heads == 0u || splits == 0u)
        return hipErrorInvalidValue;
    const uint32_t grid = common.rows * common.q_heads;
    if (grid == 0u) return hipErrorInvalidValue;
    qsa_sparse_reduce_kernel<<<grid, kD, 0, stream>>>(common, partials, splits);
    return hipGetLastError();
}

}  // namespace shelf::kernel
