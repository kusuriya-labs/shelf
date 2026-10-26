// shelf GEMM BF16 device layer (tile-refactor Phase 2/3/4). Private header,
// used only by src/backend/hip/kernels/gemm/linear_bf16.hip. Frozen layout /
// ordering contracts: docs/architecture/hip-tile-refactor-procedure.md §3
// (Phase 1 contract table: WmmaLinearMap, SplitKMap, ExactRows).
//
// Layering (procedure §1.1): these are Algorithm-Layer / Tile-Primitive helpers;
// kernel entries, host launchers and the public ABI stay in the .hip TU.
#pragma once

#include <cstdint>
#include <cstddef>

#include <hip/hip_bf16.h>

#include <shelf/backend/hip/kernels/detail/tile/bf16_wmma_gfx12.h>
#include <shelf/backend/hip/kernels/gemm/linear_bf16.h>
#include <shelf/backend/hip/kernels/gemm/linear_bf16_config.h>
#include <shelf/foundation/numeric.hpp>

namespace shelf::hipk {
namespace gemm_detail {

using tile::Bf16x8;
using tile::F32x8;

using short8_t = short __attribute__((ext_vector_type(8)));
using float8_t = float __attribute__((ext_vector_type(8)));
using short2_t = short __attribute__((ext_vector_type(2)));

constexpr uint32_t kTileOut = 16u;
constexpr uint32_t kTileRow = 16u;
constexpr uint32_t kWideRowTiles = 4u;
constexpr uint32_t kThreadsPerBlock = 32u;

constexpr uint32_t kSplitWarps = 8u;
constexpr uint32_t kSplitThreadsPerBlock = 32u * kSplitWarps;


// ---------------------------------------------------------------------------
// Tile primitives (tile-refactor Phase 2). Private to this HIP TU until the
// responsibilities stabilize (procedure docs/architecture/hip-tile-refactor-procedure.md §4).
// These wrap hardware behavior — layout contract is documented at each use.
// ---------------------------------------------------------------------------

// Static lane layout for the single-wave WMMA kernels (see Phase 1 contract §1).
// A/C fragment row = lane_low4; the lane owns K halves selected by k_half.
struct WmmaLinearMap {
    uint32_t lane;       // threadIdx.x, 0..31
    uint32_t lane_low4;  // lane & 15  -> A row / C row
    uint32_t k_half;     // lane >> 4  -> which 8 of the 16 K columns

    __device__ __forceinline__ static WmmaLinearMap current() {
        const uint32_t lane = threadIdx.x;
        return {lane, lane & 15u, lane >> 4};
    }
};

// Static lane/wave layout for the Split-K WMMA kernel (contract §2): 8 waves
// each own a contiguous K-tile range; lanes carry the same WMMA fragment map.
struct SplitKMap {
    uint32_t lane;
    uint32_t wave;       // K partition index 0..kSplitWarps-1
    uint32_t lane_low4;
    uint32_t k_half;

    __device__ __forceinline__ static SplitKMap current() {
        const uint32_t lane = threadIdx.x & 31u;
        return {lane, threadIdx.x >> 5u, lane & 15u, lane >> 4};
    }
};

__device__ __forceinline__ uint64_t load_bf16x4_masked(
    const shelf::bf16_t* base, uint32_t start_k, uint32_t logical_k) {
    const shelf::bf16_t* p = base + start_k;
    if (start_k + 3u < logical_k && (reinterpret_cast<uintptr_t>(p) & 7u) == 0u) {
        return *reinterpret_cast<const uint64_t*>(p);
    }
    uint64_t out = 0;
    #pragma unroll
    for (uint32_t q = 0; q < 4u; ++q) {
        const uint32_t kk = start_k + q;
        const uint16_t bits = (kk < logical_k) ? p[q].data : 0u;
        out |= static_cast<uint64_t>(bits) << (16u * q);
    }
    return out;
}

__device__ __forceinline__ short8_t assemble_short8(uint64_t lo, uint64_t hi) {
    short8_t v = {};
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        v[q] = static_cast<short>((lo >> (16u * q)) & 0xFFFFu);
        v[q + 4] = static_cast<short>((hi >> (16u * q)) & 0xFFFFu);
    }
    return v;
}

__device__ __forceinline__ float bf16_f32(short s) {
    const uint32_t ui = static_cast<uint32_t>(static_cast<unsigned short>(s)) << 16u;
    union { uint32_t u; float f; } b;
    b.u = ui;
    return b.f;
}

__device__ __forceinline__ short8_t wide_load_a(
    const shelf::bf16_t* wrow, uint32_t ks, bool out_valid, uint32_t k, uint32_t k_half) {
    uint64_t a0 = 0, a1 = 0;
    if (out_valid) {
        a0 = load_bf16x4_masked(wrow, ks + k_half * 4u, k);
        a1 = load_bf16x4_masked(wrow, ks + 8u + k_half * 4u, k);
    }
    return assemble_short8(a0, a1);
}

__device__ __forceinline__ void wide_tile(
    uint32_t ks, const short8_t& a, const shelf::bf16_t* xrow, bool row_valid,
    float8_t& acc, uint32_t k, uint32_t k_half) {
    uint64_t b0 = 0, b1 = 0;
    if (row_valid) {
        b0 = load_bf16x4_masked(xrow, ks + k_half * 4u, k);
        b1 = load_bf16x4_masked(xrow, ks + 8u + k_half * 4u, k);
    }
    acc = tile::dot_bf16_16x16x16(a, assemble_short8(b0, b1), acc);
}

template <uint32_t kRowTiles>
__device__ __forceinline__ void gemm_bf16_wmma_ktiles(
    const shelf::bf16_t* wrow,
    const shelf::bf16_t* const* xrow,
    const bool* row_valid,
    float8_t* acc,
    uint32_t k_tiles,
    uint32_t k,
    uint32_t k_half,
    bool out_valid) {
    for (uint32_t kb = 0; kb < k_tiles; ++kb) {
        const uint32_t ks = kb * 16u;
        const short8_t a = wide_load_a(wrow, ks, out_valid, k, k_half);
        #pragma unroll
        for (uint32_t rt = 0; rt < kRowTiles; ++rt)
            wide_tile(ks, a, xrow[rt], row_valid[rt], acc[rt], k, k_half);
    }
}

// Algorithm layer (Phase 3): tile context — block tiling, A-side view, K tile
// count, uniform activation flag (contract §1: same expressions, relocated).
struct GemmTileContext {
    WmmaLinearMap map;
    uint32_t out_start;   // N index of tile origin
    uint32_t row_start;   // row index of tile origin
    const shelf::bf16_t* wrow;
    bool out_valid;
    uint32_t k_tiles;
    bool block_active;    // block-uniform return condition
};

template <uint32_t kRowTiles>
__device__ __forceinline__ GemmTileContext gemm_make_tile_context(
    const shelf::bf16_t* weight, uint32_t rows, uint32_t out_features, uint32_t k) {
    GemmTileContext ctx{};
    ctx.map = WmmaLinearMap::current();
    ctx.out_start = blockIdx.x * kTileOut;
    ctx.row_start = blockIdx.y * (kRowTiles * kTileRow);
    ctx.block_active = ctx.out_start < out_features && ctx.row_start < rows;
    if (!ctx.block_active)
        return ctx;
    const uint32_t a_row = ctx.out_start + ctx.map.lane_low4;
    ctx.out_valid = a_row < out_features;
    ctx.wrow = weight + static_cast<size_t>(a_row) * k;
    ctx.k_tiles = (k + 15u) >> 4;
    return ctx;
}

// B-side row views for each row tile (contract §1: b_col / row_valid / xrow).
template <uint32_t kRowTiles>
struct GemmRowViews {
    const shelf::bf16_t* xrow[kRowTiles];
    bool row_valid[kRowTiles];
};

template <uint32_t kRowTiles>
__device__ __forceinline__ GemmRowViews<kRowTiles> gemm_prepare_row_views(
    const GemmTileContext& ctx, const shelf::bf16_t* input, uint32_t rows,
    uint32_t input_row_stride) {
    GemmRowViews<kRowTiles> v{};
    #pragma unroll
    for (uint32_t rt = 0; rt < kRowTiles; ++rt) {
        const uint32_t b_col = ctx.row_start + rt * kTileRow + ctx.map.lane_low4;
        v.row_valid[rt] = b_col < rows;
        v.xrow[rt] = input + static_cast<size_t>(b_col) * input_row_stride;
    }
    return v;
}

// Algorithm layer (Phase 3, contract §1 store mapping): one output tile row
// `r = row_start + rt*16 + lane_low4`, columns `out_start + 8*k_half + j`.
// Body extracted verbatim; no rounding or tail-mask changes.
template <uint32_t kRowTiles>
__device__ __forceinline__ void bf16_wmma_store(
    const WmmaLinearMap& map, uint32_t out_start, uint32_t row_start,
    uint32_t rows, uint32_t out_features, uint32_t output_row_stride,
    void* __restrict__ output, uint32_t output_dtype, const float8_t (&acc)[kRowTiles]) {
    #pragma unroll
    for (uint32_t rt = 0; rt < kRowTiles; ++rt) {
        if (row_start + rt * kTileRow < rows) {
            const uint32_t r = row_start + rt * kTileRow + map.lane_low4;
            if (r < rows) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const uint32_t o = out_start + 8u * (map.lane >> 4) + static_cast<uint32_t>(j);
                    if (o >= out_features)
                        continue;
                    const size_t out_idx = static_cast<size_t>(r) * output_row_stride + o;
                    if (output_dtype == static_cast<uint32_t>(Bf16GemmOutputDType::BF16)) {
                        static_cast<shelf::bf16_t*>(output)[out_idx] =
                            shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(acc[rt][j]))};
                    } else {
                        static_cast<float*>(output)[out_idx] = acc[rt][j];
                    }
                }
            }
        }
    }
}

// Accumulator fragments start zeroed (contract §1: all elements, register
// usage unchanged).
template <uint32_t kRowTiles>
__device__ __forceinline__ void gemm_init_accumulators(float8_t (&acc)[kRowTiles]) {
    #pragma unroll
    for (uint32_t rt = 0; rt < kRowTiles; ++rt)
        acc[rt] = float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
}

// K-tile accumulation dispatch (contract §1: `active` switch over 1..4 and
// the ascending K-tile order are preserved).
template <uint32_t kRowTiles>
__device__ __forceinline__ void gemm_accumulate_wmma(
    const GemmTileContext& ctx, const GemmRowViews<kRowTiles>& views,
    float8_t (&acc)[kRowTiles], uint32_t rows, uint32_t k) {
    const uint32_t k_half = ctx.map.k_half;
    if constexpr (kRowTiles == 1u) {
        gemm_bf16_wmma_ktiles<1u>(ctx.wrow, views.xrow, views.row_valid, acc, ctx.k_tiles,
                                  k, k_half, ctx.out_valid);
    } else {
        const uint32_t active = (rows - ctx.row_start + kTileRow - 1u) / kTileRow;
        switch (active) {
        case 1u:
            gemm_bf16_wmma_ktiles<1u>(ctx.wrow, views.xrow, views.row_valid, acc, ctx.k_tiles,
                                      k, k_half, ctx.out_valid);
            break;
        case 2u:
            if constexpr (kRowTiles >= 2u)
                gemm_bf16_wmma_ktiles<2u>(ctx.wrow, views.xrow, views.row_valid, acc,
                                          ctx.k_tiles, k, k_half, ctx.out_valid);
            break;
        case 3u:
            if constexpr (kRowTiles >= 3u)
                gemm_bf16_wmma_ktiles<3u>(ctx.wrow, views.xrow, views.row_valid, acc,
                                          ctx.k_tiles, k, k_half, ctx.out_valid);
            break;
        default:
            gemm_bf16_wmma_ktiles<kRowTiles>(ctx.wrow, views.xrow, views.row_valid, acc,
                                             ctx.k_tiles, k, k_half, ctx.out_valid);
            break;
        }
    }
}

template <uint32_t kRowTiles>
__device__ __forceinline__ void bf16_wmma_body(
    const shelf::bf16_t* __restrict__ weight,
    const shelf::bf16_t* __restrict__ input,
    void* __restrict__ output,
    const uint32_t output_dtype,
    const uint32_t rows,
    const uint32_t out_features,
    const uint32_t k,
    const uint32_t input_row_stride,
    const uint32_t output_row_stride) {
    static_assert(kRowTiles >= 1u, "bf16 wmma row tiles must be positive");

    const GemmTileContext ctx =
        gemm_make_tile_context<kRowTiles>(weight, rows, out_features, k);
    if (!ctx.block_active)
        return;

    float8_t acc[kRowTiles];
    gemm_init_accumulators<kRowTiles>(acc);

    const GemmRowViews<kRowTiles> views =
        gemm_prepare_row_views<kRowTiles>(ctx, input, rows, input_row_stride);

    gemm_accumulate_wmma<kRowTiles>(ctx, views, acc, rows, k);

    bf16_wmma_store<kRowTiles>(ctx.map, ctx.out_start, ctx.row_start, rows, out_features,
                               output_row_stride, output, output_dtype, acc);
}


// Algorithm layer (Phase 4, contract §2): split-K context — 8 waves each own
// a contiguous K-tile range; frozen floor-division split boundaries.
struct SplitKContext {
    SplitKMap map;
    uint32_t out_start;
    uint32_t row_start;
    const shelf::bf16_t* wrow;
    const shelf::bf16_t* xrow;
    bool out_valid;
    bool row_valid;
    uint32_t kb_begin;
    uint32_t kb_end;
    bool block_active;   // block-uniform return condition
};

__device__ __forceinline__ SplitKContext gemm_make_splitk_context(
    const shelf::bf16_t* weight, const shelf::bf16_t* input,
    uint32_t rows, uint32_t out_features, uint32_t k, uint32_t input_row_stride) {
    SplitKContext ctx{};
    ctx.map = SplitKMap::current();
    ctx.out_start = blockIdx.x * kTileOut;
    ctx.row_start = blockIdx.y * kTileRow;
    ctx.block_active = ctx.out_start < out_features && ctx.row_start < rows;
    if (!ctx.block_active)
        return ctx;
    const uint32_t a_row = ctx.out_start + ctx.map.lane_low4;
    const uint32_t b_col = ctx.row_start + ctx.map.lane_low4;
    ctx.out_valid = a_row < out_features;
    ctx.row_valid = b_col < rows;
    ctx.wrow = weight + static_cast<size_t>(a_row) * k;
    ctx.xrow = input + static_cast<size_t>(b_col) * input_row_stride;
    const uint32_t k_tiles = (k + 15u) >> 4;
    ctx.kb_begin = (k_tiles * ctx.map.wave) / kSplitWarps;
    ctx.kb_end = (k_tiles * (ctx.map.wave + 1u)) / kSplitWarps;
    return ctx;
}

// Partial dot over this wave's K range (contract §2: load pairs and order fixed).
__device__ __forceinline__ float8_t gemm_accumulate_splitk(const SplitKContext& ctx,
                                                           uint32_t k) {
    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const uint32_t k_half = ctx.map.k_half;
    for (uint32_t kb = ctx.kb_begin; kb < ctx.kb_end; ++kb) {
        const uint32_t ks = kb * 16u;
        uint64_t a0 = 0, a1 = 0, b0 = 0, b1 = 0;
        if (ctx.out_valid) {
            a0 = load_bf16x4_masked(ctx.wrow, ks + k_half * 4u, k);
            a1 = load_bf16x4_masked(ctx.wrow, ks + 8u + k_half * 4u, k);
        }
        if (ctx.row_valid) {
            b0 = load_bf16x4_masked(ctx.xrow, ks + k_half * 4u, k);
            b1 = load_bf16x4_masked(ctx.xrow, ks + 8u + k_half * 4u, k);
        }
        const short8_t a = assemble_short8(a0, a1);
        const short8_t b = assemble_short8(b0, b1);
        acc = tile::dot_bf16_16x16x16(a, b, acc);
    }
    return acc;
}

// Stage wave partials to LDS. Barrier stays at the call site (contract §2:
// single unconditional __syncthreads between stage and reduce).
__device__ __forceinline__ void gemm_write_splitk_partials(
    const SplitKMap& map, const float8_t& acc, float (&red)[kSplitWarps][8][32]) {
    #pragma unroll
    for (int j = 0; j < 8; ++j)
        red[map.wave][j][map.lane] = acc[j];
}

// Wave-0-only reduction in ascending wave order, then store (contract §2:
// s[j] += red[w][j][lane] sequential; no pairwise reassociation).
__device__ __forceinline__ void gemm_reduce_store_splitk(
    const SplitKContext& ctx, const float (&red)[kSplitWarps][8][32],
    void* __restrict__ output, uint32_t output_dtype, uint32_t rows,
    uint32_t out_features, uint32_t output_row_stride) {
    float s[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (uint32_t w = 0; w < kSplitWarps; ++w) {
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            s[j] += red[w][j][ctx.map.lane];
    }
    const uint32_t r = ctx.row_start + ctx.map.lane_low4;
    if (r < rows) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint32_t o = ctx.out_start + 8u * (ctx.map.lane >> 4) + static_cast<uint32_t>(j);
            if (o >= out_features)
                continue;
            const size_t out_idx = static_cast<size_t>(r) * output_row_stride + o;
            if (output_dtype == static_cast<uint32_t>(Bf16GemmOutputDType::BF16)) {
                static_cast<shelf::bf16_t*>(output)[out_idx] =
                    shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(s[j]))};
            } else {
                static_cast<float*>(output)[out_idx] = s[j];
            }
        }
    }
}

__device__ __forceinline__ void bf16_splitk_wmma_body(
    const shelf::bf16_t* __restrict__ weight,
    const shelf::bf16_t* __restrict__ input,
    void* __restrict__ output,
    const uint32_t output_dtype,
    const uint32_t rows,
    const uint32_t out_features,
    const uint32_t k,
    const uint32_t input_row_stride,
    const uint32_t output_row_stride) {
    const SplitKContext ctx = gemm_make_splitk_context(
        weight, input, rows, out_features, k, input_row_stride);
    if (!ctx.block_active)
        return;

    const float8_t acc = gemm_accumulate_splitk(ctx, k);

    __shared__ float red[kSplitWarps][8][32];
    gemm_write_splitk_partials(ctx.map, acc, red);
    __syncthreads();
    if (ctx.map.wave == 0u)
        gemm_reduce_store_splitk(ctx, red, output, output_dtype, rows,
                                 out_features, output_row_stride);
}

// Algorithm layer (Phase 4, contract §3): ExactRows is NOT a WMMA path — it
// keeps its own scalar / fdot2 partial-dot stage, shuffle-down reduce tree,
// and lane0 store. All loop orders and builtin call sequences preserved.
template <uint32_t kRows, bool kDot2>
__device__ __forceinline__ void gemm_exact_rows_stage(
    const shelf::bf16_t* wrow, const shelf::bf16_t* input,
    uint32_t k, uint32_t input_row_stride, float (&acc)[kRows]) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t iters = k >> 8u;
    for (uint32_t i = 0; i < iters; ++i) {
        const uint32_t off = (i << 8) + (lane << 3);
        const short8_t w = *reinterpret_cast<const short8_t*>(wrow + off);
        if constexpr (kDot2) {
            #pragma unroll
            for (uint32_t m = 0; m < kRows; ++m) {
                const short8_t x = *reinterpret_cast<const short8_t*>(
                    input + static_cast<size_t>(m) * input_row_stride + off);
                #pragma unroll
                for (uint32_t j = 0; j < 8u; j += 2u) {
                    const short2_t w2 = short2_t{w[j], w[j + 1u]};
                    const short2_t x2 = short2_t{x[j], x[j + 1u]};
                    acc[m] = __builtin_amdgcn_fdot2_f32_bf16(w2, x2, acc[m], false);
                }
            }
        } else {
            float wf[8];
            #pragma unroll
            for (uint32_t j = 0; j < 8u; ++j)
                wf[j] = bf16_f32(w[j]);
            #pragma unroll
            for (uint32_t m = 0; m < kRows; ++m) {
                const short8_t x = *reinterpret_cast<const short8_t*>(
                    input + static_cast<size_t>(m) * input_row_stride + off);
                #pragma unroll
                for (uint32_t j = 0; j < 8u; ++j)
                    acc[m] += wf[j] * bf16_f32(x[j]);
            }
        }
    }
    const uint32_t tail = k & 255u;
    if (tail != 0u) {
        const uint32_t base = iters << 8;
        #pragma unroll
        for (uint32_t m = 0; m < kRows; ++m) {
            const shelf::bf16_t* xrow = input + static_cast<size_t>(m) * input_row_stride;
            for (uint32_t j = lane; j < tail; j += 32u)
                acc[m] += bf16_f32(static_cast<short>(wrow[base + j].data))
                        * bf16_f32(static_cast<short>(xrow[base + j].data));
        }
    }
}

// Shuffle-down reduce (s=16..1) then lane0 transposed store (contract §3).
template <uint32_t kRows>
__device__ __forceinline__ void gemm_exact_rows_reduce_store(
    const float (&acc)[kRows], uint32_t lane, void* __restrict__ output,
    uint32_t output_dtype, uint32_t r, uint32_t output_row_stride) {
    #pragma unroll
    for (uint32_t m = 0; m < kRows; ++m) {
        float a = acc[m];
        #pragma unroll
        for (uint32_t s = 16u; s > 0u; s >>= 1)
            a += __shfl_down(a, s, 32);
        if (lane == 0u) {
            const size_t oidx = static_cast<size_t>(m) * output_row_stride + r;
            if (output_dtype == static_cast<uint32_t>(Bf16GemmOutputDType::BF16))
                static_cast<shelf::bf16_t*>(output)[oidx] =
                    shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(a))};
            else
                static_cast<float*>(output)[oidx] = a;
        }
    }
}

template <uint32_t kRows, bool kDot2 = false>
__device__ __forceinline__ void gemm_bf16_exact_rows_body(
    const shelf::bf16_t* __restrict__ weight,
    const shelf::bf16_t* __restrict__ input,
    void* __restrict__ output,
    const uint32_t output_dtype,
    const uint32_t n,
    const uint32_t k,
    const uint32_t input_row_stride,
    const uint32_t output_row_stride) {
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t r = blockIdx.x * 8u + wave;
    if (r >= n)
        return;
    const shelf::bf16_t* wrow = weight + static_cast<size_t>(r) * k;
    float acc[kRows];
    #pragma unroll
    for (uint32_t m = 0; m < kRows; ++m)
        acc[m] = 0.0f;
    gemm_exact_rows_stage<kRows, kDot2>(wrow, input, k, input_row_stride, acc);
    gemm_exact_rows_reduce_store<kRows>(acc, lane, output, output_dtype, r,
                                        output_row_stride);
}
}  // namespace gemm_detail
}  // namespace shelf::hipk
