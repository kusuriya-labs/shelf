// shelf GEMM BF16 device layer (SYCL port of backend/hip
// kernels/gemm/detail/linear_bf16_device.h).
//
// Mapping contract from the HIP original:
//   * ExactRows: same block/wave/lane map, same 256-element main-chunk and
//     strided-tail loop order; short8_t vector loads become 16-byte memcpy
//     loads; the shuffle-down reduce tree becomes sycl's sub_group
//     collective (lane0-consumed semantics unchanged).
//   * Wmma / WmmaWide / WmmaKPartition: the gfx12 wave32 WMMA builtin
//     (__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12) has no SYCL
//     equivalent, so the 16x16x16 tile is computed with per-lane scalar
//     FMA dots. The output store map is preserved exactly (r =
//     row_start + rt*16 + lane_low4, cols = out_start + 8*k_half + j), as
//     are the Split-K wave partition boundaries and the ascending-wave
//     reduce order; only the intra-tile K summation order differs (fp32,
//     ascending k).
#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>

#include <sycl/sycl.hpp>

#include <shelf/backend/sycl/kernels/detail/sycl_util.h>
#include <shelf/backend/sycl/kernels/gemm/linear_bf16.h>
#include <shelf/backend/sycl/kernels/gemm/linear_bf16_config.h>
#include <shelf/foundation/numeric.hpp>

namespace shelf::syclk {
namespace gemm_detail {

using bf16 = shelf::bf16_t;

constexpr uint32_t kTileOut = 16u;
constexpr uint32_t kTileRow = 16u;
constexpr uint32_t kWideRowTiles = 4u;
constexpr uint32_t kThreadsPerBlock = 32u;

constexpr uint32_t kSplitWarps = 8u;
constexpr uint32_t kSplitThreadsPerBlock = 32u * kSplitWarps;

inline float bf16_f32(bf16 b) { return ::shelf::bf16_to_f32(b); }

// ---------------------------------------------------------------------------
// Static lane maps (identical shapes to the HIP originals).
// ---------------------------------------------------------------------------

struct WmmaLinearMap {
    uint32_t lane;       // local id, 0..31
    uint32_t lane_low4;  // lane & 15 -> tile row
    uint32_t k_half;     // lane >> 4  -> which 8 of the 16 out columns

    template <class IT>
    static WmmaLinearMap current(IT& it) {
        const uint32_t lane = static_cast<uint32_t>(it.get_local_linear_id());
        return {lane, lane & 15u, lane >> 4};
    }
};

struct SplitKMap {
    uint32_t lane;
    uint32_t wave;       // K partition index 0..kSplitWarps-1
    uint32_t lane_low4;
    uint32_t k_half;

    template <class IT>
    static SplitKMap current(IT& it) {
        const uint32_t tid = static_cast<uint32_t>(it.get_local_linear_id());
        return {tid & 31u, tid >> 5u, tid & 15u, (tid & 31u) >> 4};
    }
};

// ---------------------------------------------------------------------------
// ExactRows (contract §3): scalar partial-dot stage, wave reduce, lane0
// transposed store. Loop orders preserved; kDot2 selects the same scalar
// order (see header note — fdot2 has no portable equivalent).
// ---------------------------------------------------------------------------

template <uint32_t kRows, bool kDot2, class IT>
inline void gemm_exact_rows_stage(
    IT& it, const bf16* wrow, const bf16* input,
    uint32_t k, uint32_t input_row_stride, float (&acc)[kRows]) {
    (void)kDot2;
    const uint32_t lane = static_cast<uint32_t>(it.get_local_linear_id()) & 31u;
    const uint32_t iters = k >> 8u;
    for (uint32_t i = 0; i < iters; ++i) {
        const uint32_t off = (i << 8) + (lane << 3);
        std::uint16_t wb[8];
        std::memcpy(wb, wrow + off, 16u);
        float wf[8];
#pragma unroll
        for (uint32_t j = 0; j < 8u; ++j)
            wf[j] = bf16_f32(bf16_t{wb[j]});
#pragma unroll
        for (uint32_t m = 0; m < kRows; ++m) {
            const bf16* xrow = input + static_cast<size_t>(m) * input_row_stride;
            std::uint16_t xb[8];
            std::memcpy(xb, xrow + off, 16u);
#pragma unroll
            for (uint32_t j = 0; j < 8u; ++j)
                acc[m] += wf[j] * bf16_f32(bf16_t{xb[j]});
        }
    }
    const uint32_t tail = k & 255u;
    if (tail != 0u) {
        const uint32_t base = iters << 8;
#pragma unroll
        for (uint32_t m = 0; m < kRows; ++m) {
            const bf16* xrow = input + static_cast<size_t>(m) * input_row_stride;
            for (uint32_t j = lane; j < tail; j += 32u)
                acc[m] += bf16_f32(wrow[base + j]) * bf16_f32(xrow[base + j]);
        }
    }
}

// Wave sum then lane0 transposed store (contract §3).
template <uint32_t kRows, class IT>
inline void gemm_exact_rows_reduce_store(
    IT& it, const float (&acc)[kRows], uint32_t lane, void* __restrict__ output,
    uint32_t output_dtype, uint32_t r, uint32_t output_row_stride) {
    const auto sg = it.get_sub_group();
#pragma unroll
    for (uint32_t m = 0; m < kRows; ++m) {
        const float a = wave_sum_all(sg, acc[m]);
        if (lane == 0u) {
            const size_t oidx = static_cast<size_t>(m) * output_row_stride + r;
            if (output_dtype == static_cast<uint32_t>(Bf16GemmOutputDType::BF16))
                static_cast<bf16*>(output)[oidx] =
                    bf16_t{f32_to_bf16_bits(a)};
            else
                static_cast<float*>(output)[oidx] = a;
        }
    }
}

template <uint32_t kRows, bool kDot2 = false, class IT>
inline void gemm_bf16_exact_rows_body(
    IT& it,
    const bf16* __restrict__ weight,
    const bf16* __restrict__ input,
    void* __restrict__ output,
    const uint32_t output_dtype,
    const uint32_t n,
    const uint32_t k,
    const uint32_t input_row_stride,
    const uint32_t output_row_stride) {
    const uint32_t tid = static_cast<uint32_t>(it.get_local_linear_id());
    const uint32_t wave = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t r = static_cast<uint32_t>(it.get_group(0)) * 8u + wave;
    if (r >= n)
        return;
    const bf16* wrow = weight + static_cast<size_t>(r) * k;
    float acc[kRows];
#pragma unroll
    for (uint32_t m = 0; m < kRows; ++m)
        acc[m] = 0.0f;
    gemm_exact_rows_stage<kRows, kDot2>(it, wrow, input, k, input_row_stride, acc);
    gemm_exact_rows_reduce_store<kRows>(it, acc, lane, output, output_dtype, r,
                                        output_row_stride);
}

// ---------------------------------------------------------------------------
// Wmma tile (scalar emulation, store map preserved). Block: 32 lanes;
// grid.x: 16 out-features per block; grid.y: kRowTiles*16 rows per block.
// ---------------------------------------------------------------------------

template <uint32_t kRowTiles, class IT>
inline void bf16_wmma_body(
    IT& it,
    const bf16* __restrict__ weight,
    const bf16* __restrict__ input,
    void* __restrict__ output,
    const uint32_t output_dtype,
    const uint32_t rows,
    const uint32_t out_features,
    const uint32_t k,
    const uint32_t input_row_stride,
    const uint32_t output_row_stride) {
    static_assert(kRowTiles >= 1u, "bf16 wmma row tiles must be positive");

    const WmmaLinearMap map = WmmaLinearMap::current(it);
    const uint32_t out_start = static_cast<uint32_t>(it.get_group(0)) * kTileOut;
    const uint32_t row_start = static_cast<uint32_t>(it.get_group(1)) * (kRowTiles * kTileRow);
    if (out_start >= out_features || row_start >= rows)
        return;

    // Per-lane outputs: tile row r = row_start + rt*16 + lane_low4;
    // columns o = out_start + 8*k_half + j, j in 0..7.
    float acc[kRowTiles][8];
#pragma unroll
    for (uint32_t rt = 0; rt < kRowTiles; ++rt)
#pragma unroll
        for (int j = 0; j < 8; ++j)
            acc[rt][j] = 0.0f;

    const uint32_t r0 = map.lane_low4;  // tile row (input row) held by this lane
    const uint32_t o0 = 8u * map.k_half;

    for (uint32_t kb = 0; kb < ((k + 15u) >> 4); ++kb) {
        const uint32_t ks = kb * 16u;
        const uint32_t kend = (k - ks < 16u) ? k - ks : 16u;
#pragma unroll
        for (uint32_t rt = 0; rt < kRowTiles; ++rt) {
            const uint32_t b_col = row_start + rt * kTileRow + r0;
            if (b_col >= rows) continue;
            const bf16* xrow = input + static_cast<size_t>(b_col) * input_row_stride;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t o = out_start + o0 + static_cast<uint32_t>(j);
                if (o >= out_features) continue;
                const bf16* wrow = weight + static_cast<size_t>(o) * k;
                for (uint32_t kk = 0; kk < kend; ++kk)
                    acc[rt][j] += bf16_f32(wrow[ks + kk]) * bf16_f32(xrow[ks + kk]);
            }
        }
    }

#pragma unroll
    for (uint32_t rt = 0; rt < kRowTiles; ++rt) {
        const uint32_t r = row_start + rt * kTileRow + r0;
        if (r >= rows) continue;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint32_t o = out_start + o0 + static_cast<uint32_t>(j);
            if (o >= out_features)
                continue;
            const size_t out_idx = static_cast<size_t>(r) * output_row_stride + o;
            if (output_dtype == static_cast<uint32_t>(Bf16GemmOutputDType::BF16)) {
                static_cast<bf16*>(output)[out_idx] =
                    bf16_t{f32_to_bf16_bits(acc[rt][j])};
            } else {
                static_cast<float*>(output)[out_idx] = acc[rt][j];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Split-K (contract §2): 8 waves own contiguous K ranges; partials staged
// through local memory; wave 0 reduces in ascending wave order and stores
// with the same map as the Wmma path.
// ---------------------------------------------------------------------------

template <class IT>
inline void bf16_splitk_wmma_body(
    IT& it, float* __restrict__ red /*local [kSplitWarps][8][32]*/,
    const bf16* __restrict__ weight,
    const bf16* __restrict__ input,
    void* __restrict__ output,
    const uint32_t output_dtype,
    const uint32_t rows,
    const uint32_t out_features,
    const uint32_t k,
    const uint32_t input_row_stride,
    const uint32_t output_row_stride) {
    const SplitKMap map = SplitKMap::current(it);
    const uint32_t out_start = static_cast<uint32_t>(it.get_group(0)) * kTileOut;
    const uint32_t row_start = static_cast<uint32_t>(it.get_group(1)) * kTileRow;
    if (out_start >= out_features || row_start >= rows)
        return;

    const uint32_t r = row_start + map.lane_low4;
    const bool row_valid = r < rows;
    const uint32_t o0 = 8u * map.k_half;

    const uint32_t k_tiles = (k + 15u) >> 4;
    const uint32_t kb_begin = (k_tiles * map.wave) / kSplitWarps;
    const uint32_t kb_end = (k_tiles * (map.wave + 1u)) / kSplitWarps;

    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j)
        acc[j] = 0.0f;

    for (uint32_t kb = kb_begin; kb < kb_end; ++kb) {
        const uint32_t ks = kb * 16u;
        const uint32_t kend = (k - ks < 16u) ? k - ks : 16u;
        if (row_valid) {
            const bf16* xrow = input + static_cast<size_t>(r) * input_row_stride;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t o = out_start + o0 + static_cast<uint32_t>(j);
                if (o >= out_features) continue;
                const bf16* wrow = weight + static_cast<size_t>(o) * k;
                for (uint32_t kk = 0; kk < kend; ++kk)
                    acc[j] += bf16_f32(wrow[ks + kk]) * bf16_f32(xrow[ks + kk]);
            }
        }
    }

    // Stage wave partials to LDS. One barrier between stage and reduce
    // (contract §2); the group is barrier-uniform here (the only early exit
    // is block-uniform).
#pragma unroll
    for (int j = 0; j < 8; ++j)
        red[(map.wave * 8u + static_cast<uint32_t>(j)) * 32u + map.lane] = acc[j];
    it.barrier(::sycl::access::fence_space::local_space);

    if (map.wave == 0u) {
        float s[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (uint32_t w = 0; w < kSplitWarps; ++w) {
#pragma unroll
            for (int j = 0; j < 8; ++j)
                s[j] += red[(w * 8u + static_cast<uint32_t>(j)) * 32u + map.lane];
        }
        if (r < rows) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t o = out_start + o0 + static_cast<uint32_t>(j);
                if (o >= out_features)
                    continue;
                const size_t out_idx = static_cast<size_t>(r) * output_row_stride + o;
                if (output_dtype == static_cast<uint32_t>(Bf16GemmOutputDType::BF16)) {
                    static_cast<bf16*>(output)[out_idx] =
                        bf16_t{f32_to_bf16_bits(s[j])};
                } else {
                    static_cast<float*>(output)[out_idx] = s[j];
                }
            }
        }
    }
}

} // namespace gemm_detail
} // namespace shelf::syclk
