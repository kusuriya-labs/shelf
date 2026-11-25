#include <shelf/kernels/optimized/linear/exl3_mul24.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <shelf/quantization/exl3/exl3_int8.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

namespace shelf::kernel {
namespace {
using namespace ::shelf::kernel::exl3_detail;
using namespace ::shelf::kernel::exl3_detail::mul24;
using int32x2_t = int32_t __attribute__((ext_vector_type(2)));
using int32x8_t = int32_t __attribute__((ext_vector_type(8)));

// Prefill INT8 WMMA tile for the rowscale contract. The weights (B) live in a
// per (n tile, 64-K supertile) fragment plane, so a lane's 32 bytes are its four
// k16 fragments and a wave reads a contiguous KiB straight into registers,
// bypassing LDS. The activations (A) are staged in LDS one 64-K tile at a time
// and shared by the four waves that sweep N. One barrier a staged tile, the next
// tile's loads are issued unconditionally, and each weight half is refilled after
// its fragments are consumed.
//
// FUSE adds the rotate-out epilogue (rowscale, H128, svh, BF16 store) so the
// int32 accumulator never reaches memory. Rows with a flag slot keep the
// separate per-group result and are written by the flagged-only fixup.
constexpr uint32_t kSuperK = 64u;
constexpr uint32_t kBlockM = 128u;
constexpr uint32_t kBlockN = 256u;
constexpr uint32_t kTilesM = 4u;
constexpr uint32_t kTilesN = 4u;
constexpr uint32_t kWavesM = 2u;
constexpr uint32_t kWavesN = 4u;
constexpr uint32_t kWaves = kWavesM * kWavesN;
constexpr uint32_t kThreads = kWaves * 32u;
constexpr uint32_t kSteps = kSuperK / 16u;
constexpr uint32_t kStride = kSuperK + 8u;
constexpr uint32_t kASize = kBlockM * kStride;
constexpr uint32_t kChunks = kSuperK / 16u;
constexpr uint32_t kPassA = (kBlockM * kChunks + kThreads - 1u) / kThreads;
constexpr uint32_t kGroupM = 4u;
constexpr uint32_t kGroups = kBlockM / 16u;
constexpr float kInvSqrt128 = 0.08838834764831845f;

template <bool FULL, bool FUSE>
__global__ void __launch_bounds__(kThreads) exl3_mul24_prefill_super64_kernel(
    const int8_t* __restrict__ a, const int8_t* __restrict__ wshuf,
    int32_t* __restrict__ c, uint32_t rows, uint32_t n, uint32_t n_base, uint32_t k,
    const float* __restrict__ rowscale, const int32_t* __restrict__ slots,
    const uint16_t* __restrict__ svh, void* __restrict__ out,
    uint32_t out_row_stride) {
    __shared__ int8_t smem[2u * kASize];
    __shared__ int32_t scratch[16u * kBlockN];

    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t tid = threadIdx.x;
    const uint32_t row = lane & 15u;
    const uint32_t half = lane >> 4u;

    uint32_t bm = blockIdx.x, bn = blockIdx.y;
    if (gridDim.x > kGroupM) {
        const uint32_t per = kGroupM * gridDim.y;
        const uint32_t lin = blockIdx.y * gridDim.x + blockIdx.x;
        const uint32_t first = (lin / per) * kGroupM, r = lin % per;
        const uint32_t gm = min(gridDim.x - first, kGroupM);
        bm = first + r % gm;
        bn = r / gm;
    }
    const uint32_t m0 = bm * kBlockM;
    const uint32_t n0 = n_base + bn * kBlockN;
    const uint32_t wm = wave % kWavesM;
    const uint32_t wn = wave / kWavesM;
    const uint32_t ntile = n0 / 16u + wn * kTilesN;
    const uint32_t ksuper = k / kSuperK;

    int32x8_t acc[kTilesM][kTilesN];
    #pragma unroll
    for (uint32_t mt = 0; mt < kTilesM; ++mt)
        #pragma unroll
        for (uint32_t nt = 0; nt < kTilesN; ++nt)
            #pragma unroll
            for (uint32_t i = 0; i < 8u; ++i) acc[mt][nt][i] = 0;

    uint4 ar[kPassA];
    const uint32_t ar0 = tid / kChunks, ac0 = (tid % kChunks) * 16u;
    auto load_a = [&](uint32_t s) {
        const uint32_t k0 = s * kSuperK;
        #pragma unroll
        for (uint32_t p = 0; p < kPassA; ++p) {
            const uint32_t r = ar0 + p * (kThreads / kChunks);
            const uint32_t m = m0 + r;
            ar[p] = (FULL || m < rows)
                ? *reinterpret_cast<const uint4*>(a + static_cast<size_t>(m) * k + k0 + ac0)
                : make_uint4(0, 0, 0, 0);
        }
    };
    auto store_a = [&](uint32_t buf) {
        int8_t* da = smem + buf * kASize;
        #pragma unroll
        for (uint32_t p = 0; p < kPassA; ++p) {
            const uint32_t r = ar0 + p * (kThreads / kChunks);
            uint2* d = reinterpret_cast<uint2*>(da + r * kStride + ac0);
            const uint2* v = reinterpret_cast<const uint2*>(&ar[p]);
            d[0] = v[0];
            d[1] = v[1];
        }
    };

    int32x2_t w[kTilesN][4];
    auto load_w_half = [&](uint32_t s, uint32_t h) {
        #pragma unroll
        for (uint32_t nt = 0; nt < kTilesN; ++nt) {
            const size_t base = ((static_cast<size_t>(ntile + nt) * ksuper + s) * 1024u) +
                                static_cast<size_t>(lane) * 32u + h * 16u;
            const uint4 lo = *reinterpret_cast<const uint4*>(wshuf + base);
            w[nt][h * 2u + 0u][0] = static_cast<int32_t>(lo.x);
            w[nt][h * 2u + 0u][1] = static_cast<int32_t>(lo.y);
            w[nt][h * 2u + 1u][0] = static_cast<int32_t>(lo.z);
            w[nt][h * 2u + 1u][1] = static_cast<int32_t>(lo.w);
        }
    };

    load_a(0u);
    load_w_half(0u, 0u);
    load_w_half(0u, 1u);
    store_a(0u);
    __syncthreads();

    for (uint32_t s = 0; s < ksuper; ++s) {
        const uint32_t sn = (s + 1u < ksuper) ? (s + 1u) : s;
        load_a(sn);

        const int8_t* pa = smem + (s & 1u) * kASize;
        #pragma unroll
        for (uint32_t ks = 0; ks < kSteps; ++ks) {
            int32x2_t af[kTilesM];
            #pragma unroll
            for (uint32_t mt = 0; mt < kTilesM; ++mt)
                af[mt] = *reinterpret_cast<const int32x2_t*>(
                    pa + (wm * (kTilesM * 16u) + mt * 16u + row) * kStride + ks * 16u + half * 8u);
            #pragma unroll
            for (uint32_t mt = 0; mt < kTilesM; ++mt)
                #pragma unroll
                for (uint32_t nt = 0; nt < kTilesN; ++nt)
                    acc[mt][nt] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                        true, af[mt], true, w[nt][ks], acc[mt][nt], false);
            if (ks == 1u) load_w_half(sn, 0u);
            if (ks + 1u == kSteps) load_w_half(sn, 1u);
        }

        store_a((s + 1u) & 1u);
        __syncthreads();
    }

    if constexpr (!FUSE) {
        #pragma unroll
        for (uint32_t mt = 0; mt < kTilesM; ++mt)
            #pragma unroll
            for (uint32_t i = 0; i < 8u; ++i) {
                const uint32_t m = m0 + wm * (kTilesM * 16u) + mt * 16u + half * 8u + i;
                if (m >= rows) continue;
                #pragma unroll
                for (uint32_t nt = 0; nt < kTilesN; ++nt)
                    c[static_cast<size_t>(m) * n + (n0 + wn * (kTilesN * 16u) + nt * 16u) + row] =
                        acc[mt][nt][i];
            }
    } else {
        uint32_t* const ob = reinterpret_cast<uint32_t*>(out);
        const uint32_t orow = out_row_stride / 2u;
        #pragma unroll 1
        for (uint32_t g = 0; g < kGroups; ++g) {
            if (wm == g / kTilesM) {
                const uint32_t mt = g % kTilesM;
                #pragma unroll
                for (uint32_t nt = 0; nt < kTilesN; ++nt)
                    #pragma unroll
                    for (uint32_t i = 0; i < 8u; ++i)
                        scratch[(half * 8u + i) * kBlockN + wn * (kTilesN * 16u) + nt * 16u + row] =
                            acc[mt][nt][i];
            }
            __syncthreads();
            #pragma unroll
            for (uint32_t rr = 0; rr < 2u; ++rr) {
                const uint32_t rl = wave * 2u + rr;
                const uint32_t m = m0 + g * 16u + rl;
                if (m >= rows || slots[m] >= 0) continue;
                const float rs = rowscale[m];
                #pragma unroll
                for (uint32_t b = 0; b < 2u; ++b) {
                    const uint32_t col = b * 128u + lane * 4u;
                    const int4 a4 = *reinterpret_cast<const int4*>(&scratch[rl * kBlockN + col]);
                    const uint2 packed = rotate_i32_block_bf16(a4, rs, svh, n0 + col, lane);
                    *reinterpret_cast<uint2*>(ob + static_cast<size_t>(m) * orow +
                                              ((n0 + col) >> 1u)) = packed;
                }
            }
            __syncthreads();
        }
    }
}

// A launch whose grid runs for more than about a millisecond is followed by a ~260 us stall
// before the next kernel starts (measured by kernel trace), so the tile is cut into column
// bands that each finish well under that. A column band keeps the weight plane read per
// (row block, n tile) unchanged; row bands would re-read it per band. The duration is
// proportional to row blocks x n tiles x k.
constexpr float kMsPerCtaK = 3.0e-7f;
constexpr float kMaxLaunchMs = 0.84f;

inline uint32_t super64_launch_parts(uint32_t row_blocks, uint32_t n_blocks, uint32_t k) {
    const float ms = static_cast<float>(row_blocks) * static_cast<float>(n_blocks) *
                     static_cast<float>(k) * kMsPerCtaK;
    uint32_t parts = static_cast<uint32_t>(ms / kMaxLaunchMs) + 1u;
    if (parts > row_blocks) parts = row_blocks;
    return parts == 0u ? 1u : parts;
}
}  // namespace

hipError_t launch_exl3_mul24_prefill_super64(
    const int8_t* codes, const int8_t* weights_super64, int32_t* yprime,
    uint32_t rows, uint32_t n, uint32_t k, hipStream_t stream) {
    if (rows == 0u) return hipSuccess;
    if (!codes || !weights_super64 || !yprime) return hipErrorInvalidValue;
    if ((n % kBlockN) != 0u || (k % kSuperK) != 0u) return hipErrorInvalidValue;
    const uint32_t n_blocks = n / kBlockN;
    const uint32_t row_blocks = (rows + kBlockM - 1u) / kBlockM;
    const uint32_t parts = super64_launch_parts(row_blocks, n_blocks, k);
    const uint32_t nb_part = (n_blocks + parts - 1u) / parts;
    const bool full = (rows % kBlockM) == 0u;
    for (uint32_t b0 = 0u; b0 < n_blocks; b0 += nb_part) {
        const uint32_t part_blocks = min(n_blocks - b0, nb_part);
        const dim3 grid(row_blocks, part_blocks);
        const uint32_t n_base = b0 * kBlockN;
        if (full) {
            exl3_mul24_prefill_super64_kernel<true, false><<<grid, kThreads, 0, stream>>>(
                codes, weights_super64, yprime, rows, n, n_base, k, nullptr, nullptr, nullptr,
                nullptr, 0u);
        } else {
            exl3_mul24_prefill_super64_kernel<false, false><<<grid, kThreads, 0, stream>>>(
                codes, weights_super64, yprime, rows, n, n_base, k, nullptr, nullptr, nullptr,
                nullptr, 0u);
        }
        if (hipGetLastError() != hipSuccess) return hipGetLastError();
    }
    return hipGetLastError();
}

hipError_t launch_exl3_mul24_prefill_super64_fused(
    const int8_t* codes, const int8_t* weights_super64, const float* rowscale,
    const int32_t* slots, const uint16_t* svh, void* out, uint32_t out_row_stride,
    uint32_t rows, uint32_t n, uint32_t k, hipStream_t stream) {
    if (rows == 0u) return hipSuccess;
    if (!codes || !weights_super64 || !rowscale || !slots || !svh || !out)
        return hipErrorInvalidValue;
    if ((n % kBlockN) != 0u || (k % kSuperK) != 0u || (out_row_stride % 2u) != 0u)
        return hipErrorInvalidValue;
    const uint32_t n_blocks = n / kBlockN;
    const uint32_t row_blocks = (rows + kBlockM - 1u) / kBlockM;
    const uint32_t parts = super64_launch_parts(row_blocks, n_blocks, k);
    const uint32_t nb_part = (n_blocks + parts - 1u) / parts;
    const bool full = (rows % kBlockM) == 0u;
    for (uint32_t b0 = 0u; b0 < n_blocks; b0 += nb_part) {
        const uint32_t part_blocks = min(n_blocks - b0, nb_part);
        const dim3 grid(row_blocks, part_blocks);
        const uint32_t n_base = b0 * kBlockN;
        if (full) {
            exl3_mul24_prefill_super64_kernel<true, true><<<grid, kThreads, 0, stream>>>(
                codes, weights_super64, nullptr, rows, n, n_base, k, rowscale, slots, svh, out,
                out_row_stride);
        } else {
            exl3_mul24_prefill_super64_kernel<false, true><<<grid, kThreads, 0, stream>>>(
                codes, weights_super64, nullptr, rows, n, n_base, k, rowscale, slots, svh, out,
                out_row_stride);
        }
        if (hipGetLastError() != hipSuccess) return hipGetLastError();
    }
    return hipGetLastError();
}

}  // namespace shelf::kernel
