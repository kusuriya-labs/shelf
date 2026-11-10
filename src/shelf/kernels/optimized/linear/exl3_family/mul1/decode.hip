#include <shelf/kernels/optimized/linear/exl3_mul1.h>
#include <shelf/kernels/optimized/linear/exl3_mul24.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul1/decode.h>
#include <shelf/kernels/optimized/linear/exl3_family/group_plan.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul1/launch_plan.h>
#include <shelf/core/memory/types.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>

namespace shelf::kernel {
namespace {
using namespace exl3_detail::mul1;

using f16x8_t = _Float16 __attribute__((ext_vector_type(8)));
using f32x8_t = float __attribute__((ext_vector_type(8)));

constexpr uint32_t kOutBlockBase = kExl3W16A16OutBlock;
constexpr uint32_t kRowTile = kExl3W16A16RowTile;
constexpr uint32_t kThreads = kExl3W16A16Threads;
constexpr uint32_t kWarps = kThreads / 32u;
constexpr uint32_t kXrotPitch = 136u;
constexpr float kInvSqrt128 = 0.08838834764831844055f;

template <uint32_t kNTiles>
struct WorkAreaT {
    uint16_t tile[kWarps][2][kNTiles * 16u * 16u];
    uint32_t ring[kWarps][64];
};

template <uint32_t kNTiles>
union SharedAreaT {
    WorkAreaT<kNTiles> work;
    float accum[kRowTile * kOutBlockBase * kNTiles];
};

__device__ __forceinline__ f16x8_t load_f16x8(const void* p) {
    return *reinterpret_cast<const f16x8_t*>(p);
}

template <uint32_t kGroups, bool kPartial, bool kPre, uint32_t kNTiles, uint32_t kDecode>
__global__ __launch_bounds__(kThreads) void gemm_exl3_w16a16_kernel(Exl3GemmArgs args,
                                                                    uint32_t splits) {
    constexpr uint32_t kOutBlock = kOutBlockBase * kNTiles;
    constexpr uint32_t kNTilesPerBlock = kOutBlock / 16u;
    constexpr uint32_t kRows = kRowTile * kGroups;
    using SharedArea = SharedAreaT<kNTiles>;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t lane16 = lane & 15u;
    const uint32_t lane_high = lane >> 4;

    const uint32_t n_tile_base = kPartial ? blockIdx.x / splits : blockIdx.x;
    const uint32_t split = kPartial ? blockIdx.x % splits : 0u;
    const uint32_t n0 = n_tile_base * kOutBlock;
    uint32_t r0 = blockIdx.y * kRows;
    uint32_t active_rows = args.rows - r0 < kRows ? args.rows - r0 : kRows;
    const uint32_t group_index = blockIdx.z;
    if (args.group_expert != nullptr &&
        !exl3_detail::resolve_group_plan<kGroups>(args, group_index, r0, active_rows))
        return;
    const __half* xrot_global = static_cast<const __half*>(args.xrot);

    __shared__ __align__(16) __half xrot[kPre ? 1u : kRows * kXrotPitch];
    __shared__ __align__(16) SharedArea area;
    __shared__ __align__(16) float rowsum[kRows];

    f32x8_t acc[kGroups][kNTiles];
#pragma unroll
    for (uint32_t g = 0u; g < kGroups; ++g) {
#pragma unroll
        for (uint32_t t = 0u; t < kNTiles; ++t) {
            acc[g][t] = f32x8_t{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        }
    }
    float row_acc[kGroups][2];
#pragma unroll
    for (uint32_t g = 0u; g < kGroups; ++g) {
        row_acc[g][0] = 0.f;
        row_acc[g][1] = 0.f;
    }

    const uint16_t* suh = static_cast<const uint16_t*>(args.suh);
    const uint16_t* svh = static_cast<const uint16_t*>(args.svh);
    const shelf::bf16_t* input = static_cast<const shelf::bf16_t*>(args.input);

    const uint32_t k_chunks = args.k / 128u;
    for (uint32_t kc = split; kc < k_chunks; kc += (kPartial ? splits : 1u)) {
        const uint32_t k_base = kc * 128u;
        __syncthreads();
        if constexpr (!kPre) {
            float suh_v[4];
#pragma unroll
            for (uint32_t i = 0u; i < 4u; ++i) {
                suh_v[i] = __half2float(__ushort_as_half(suh[k_base + lane * 4u + i]));
            }

#pragma unroll
            for (uint32_t g = 0u; g < kGroups; ++g) {
#pragma unroll
                for (uint32_t ri = 0u; ri < 2u; ++ri) {
                    const uint32_t r = g * kRowTile + warp * 2u + ri;
                    if (r >= active_rows) continue;
                    uint32_t input_row = r0 + r;
                    if (args.row_token != nullptr) {
                        const int32_t token =
                            args.row_token[group_index * kRowTile + r];
                        if (token < 0) continue;
                        input_row = static_cast<uint32_t>(token);
                    }
                    const shelf::bf16_t* src =
                        input + static_cast<uint64_t>(input_row) * args.input_row_stride +
                        k_base + lane * 4u;
                    float v0 = bf16_value(src + 0) * suh_v[0];
                    float v1 = bf16_value(src + 1) * suh_v[1];
                    float v2 = bf16_value(src + 2) * suh_v[2];
                    float v3 = bf16_value(src + 3) * suh_v[3];
                    exl3_detail::fwht128(v0, v1, v2, v3, lane);
                    const __half hv0 = __float2half(v0 * kInvSqrt128);
                    const __half hv1 = __float2half(v1 * kInvSqrt128);
                    const __half hv2 = __float2half(v2 * kInvSqrt128);
                    const __half hv3 = __float2half(v3 * kInvSqrt128);
                    row_acc[g][ri] += __half2float(hv0) + __half2float(hv1) +
                                      __half2float(hv2) + __half2float(hv3);
                    __half* dst = xrot + r * kXrotPitch + lane * 4u;
                    dst[0] = hv0;
                    dst[1] = hv1;
                    dst[2] = hv2;
                    dst[3] = hv3;
                }
            }
        }
        __syncthreads();
        if constexpr (kPre) {
#pragma unroll
            for (uint32_t g = 0u; g < kGroups; ++g) {
#pragma unroll
                for (uint32_t ri = 0u; ri < 2u; ++ri) {
                    const uint32_t r = g * kRowTile + warp * 2u + ri;
                    if (r >= active_rows) continue;
                    const __half* p = static_cast<const __half*>(args.xrot) +
                                      static_cast<uint64_t>(r0 + r) * args.k + k_base +
                                      lane * 4u;
                    row_acc[g][ri] += __half2float(p[0]) + __half2float(p[1]) +
                                      __half2float(p[2]) + __half2float(p[3]);
                }
            }
        }

        const uint32_t n_tile = n_tile_base * kNTilesPerBlock + warp * kNTiles;
#pragma unroll
        for (uint32_t ks = 0u; ks < 8u; ++ks) {
            const uint32_t buf = ks & 1u;
            uint16_t* tile = area.work.tile[warp][buf];
            f16x8_t a[kNTiles];
#pragma unroll
            for (uint32_t t = 0u; t < kNTiles; ++t) {
                decode_tile<kDecode>(args, kc * 8u + ks, n_tile + t, lane,
                                  area.work.ring[warp], tile + t * 256u);
                a[t] = load_f16x8(tile + t * 256u + lane16 * 16u + lane_high * 8u);
            }
#pragma unroll
            for (uint32_t g = 0u; g < kGroups; ++g) {
                if constexpr (kGroups > 1u) {
                    if (g * kRowTile >= active_rows) continue;
                }
                f16x8_t b;
                if constexpr (kPre) {
                    b = load_f16x8(static_cast<const __half*>(args.xrot) +
                                   static_cast<uint64_t>(r0 + g * kRowTile + lane16) *
                                       args.k +
                                   k_base + ks * 16u + lane_high * 8u);
                } else {
                    b = load_f16x8(xrot + (g * kRowTile + lane16) * kXrotPitch +
                                   ks * 16u + lane_high * 8u);
                }
#pragma unroll
                for (uint32_t t = 0u; t < kNTiles; ++t) {
                    acc[g][t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                        a[t], b, acc[g][t]);
                }
            }
        }
    }

    if constexpr (kPartial) {
        const uint64_t partial_plane =
            static_cast<uint64_t>(args.rows) * args.out_features;
        float* rowsum_base =
            args.splitk_scratch + static_cast<uint64_t>(splits) * partial_plane;
#pragma unroll
        for (uint32_t g = 0u; g < kGroups; ++g) {
#pragma unroll
            for (uint32_t ri = 0u; ri < 2u; ++ri) {
                float sum = row_acc[g][ri];
                sum += __shfl_down(sum, 16, 32);
                sum += __shfl_down(sum, 8, 32);
                sum += __shfl_down(sum, 4, 32);
                sum += __shfl_down(sum, 2, 32);
                sum += __shfl_down(sum, 1, 32);
                const uint32_t r = g * kRowTile + warp * 2u + ri;
                if (lane == 0u && r < active_rows) {
                    rowsum_base[static_cast<uint64_t>(split) * args.rows + (r0 + r)] = sum;
                }
            }
        }
#pragma unroll
        for (uint32_t g = 0u; g < kGroups; ++g) {
            const uint32_t r = g * kRowTile + lane16;
            if (r < active_rows) {
                const uint64_t row_index =
                    static_cast<uint64_t>(split) * args.rows + (r0 + r);
#pragma unroll
                for (uint32_t t = 0u; t < kNTiles; ++t) {
                    const uint32_t out_col =
                        n0 + (warp * kNTiles + t) * 16u + lane_high * 8u;
                    float* dst =
                        args.splitk_scratch + row_index * args.out_features + out_col;
#pragma unroll
                    for (uint32_t j = 0u; j < 8u; ++j) dst[j] = acc[g][t][j];
                }
            }
        }
    } else {
#pragma unroll
        for (uint32_t g = 0u; g < kGroups; ++g) {
#pragma unroll
            for (uint32_t ri = 0u; ri < 2u; ++ri) {
                float sum = row_acc[g][ri];
                sum += __shfl_down(sum, 16, 32);
                sum += __shfl_down(sum, 8, 32);
                sum += __shfl_down(sum, 4, 32);
                sum += __shfl_down(sum, 2, 32);
                sum += __shfl_down(sum, 1, 32);
                if (lane == 0u && g * kRowTile + warp * 2u + ri < active_rows) {
                    rowsum[g * kRowTile + warp * 2u + ri] = sum;
                }
            }
        }
        __syncthreads();

#pragma unroll
        for (uint32_t g = 0u; g < kGroups; ++g) {
            if constexpr (kGroups > 1u) {
                if (g * kRowTile >= active_rows) continue;
            }
#pragma unroll
            for (uint32_t t = 0u; t < kNTiles; ++t) {
#pragma unroll
                for (uint32_t j = 0u; j < 8u; ++j) {
                    const uint32_t out_col =
                        (warp * kNTiles + t) * 16u + lane_high * 8u + j;
                    area.accum[lane16 * kOutBlock + out_col] = acc[g][t][j];
                }
            }
            __syncthreads();

#pragma unroll
            for (uint32_t ri = 0u; ri < 2u; ++ri) {
                const uint32_t r = g * kRowTile + warp * 2u + ri;
                if (r >= active_rows) continue;
                const uint32_t row = r0 + r;
                const float bias = exl3_detail::mul1::kMul1BiasOverInv * rowsum[r];
#pragma unroll
                for (uint32_t cb = 0u; cb < kNTiles; ++cb) {
                    const float* src = area.accum + (r - g * kRowTile) * kOutBlock +
                                       cb * 128u + lane * 4u;
                    float v0 = src[0] + bias;
                    float v1 = src[1] + bias;
                    float v2 = src[2] + bias;
                    float v3 = src[3] + bias;
                    exl3_detail::fwht128(v0, v1, v2, v3, lane);
                    const uint32_t col = n0 + cb * 128u + lane * 4u;
                    const float s0 = __half2float(__ushort_as_half(svh[col + 0]));
                    const float s1 = __half2float(__ushort_as_half(svh[col + 1]));
                    const float s2 = __half2float(__ushort_as_half(svh[col + 2]));
                    const float s3 = __half2float(__ushort_as_half(svh[col + 3]));
                    const uint64_t out_base =
                        static_cast<uint64_t>(row) * args.output_row_stride + col;
                    const float scale = kInvSqrt128 * exl3_detail::mul1::kMul1Inv;
                    const float y0 = v0 * scale * s0;
                    const float y1 = v1 * scale * s1;
                    const float y2 = v2 * scale * s2;
                    const float y3 = v3 * scale * s3;
                    if (args.output_dtype == 0u) {
                        shelf::bf16_t* dst =
                            static_cast<shelf::bf16_t*>(args.output) + out_base;
                        dst[0] = shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(y0))};
                        dst[1] = shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(y1))};
                        dst[2] = shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(y2))};
                        dst[3] = shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(y3))};
                    } else {
                        float* dst = static_cast<float*>(args.output) + out_base;
                        dst[0] = y0;
                        dst[1] = y1;
                        dst[2] = y2;
                        dst[3] = y3;
                    }
                }
            }
            if constexpr (kGroups > 1u) {
                __syncthreads();
            }
        }
    }
}

template <uint32_t kNTiles>
__global__ __launch_bounds__(kThreads) void exl3_splitk_reduce_kernel(Exl3GemmArgs args,
                                                                     uint32_t splits) {
    constexpr uint32_t kOutBlock = kOutBlockBase * kNTiles;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t n0 = blockIdx.x * kOutBlock;
    const uint32_t r0 = blockIdx.y * kRowTile;
    const uint32_t active_rows =
        args.rows - r0 < kRowTile ? args.rows - r0 : kRowTile;
    const uint16_t* svh = static_cast<const uint16_t*>(args.svh);
    const uint64_t partial_plane = static_cast<uint64_t>(args.rows) * args.out_features;
    const float* partials = args.splitk_scratch;
    const float* rowsums = partials + static_cast<uint64_t>(splits) * partial_plane;
    const float scale = kInvSqrt128 * exl3_detail::mul1::kMul1Inv;

#pragma unroll
    for (uint32_t ri = 0u; ri < 2u; ++ri) {
        const uint32_t r = warp * 2u + ri;
        if (r >= active_rows) break;
        const uint32_t row = r0 + r;
        float row_sum = 0.f;
        for (uint32_t s = 0u; s < splits; ++s) {
            row_sum += rowsums[static_cast<uint64_t>(s) * args.rows + row];
        }
        const float bias = exl3_detail::mul1::kMul1BiasOverInv * row_sum;
#pragma unroll
        for (uint32_t cb = 0u; cb < kNTiles; ++cb) {
            const uint32_t col = n0 + cb * 128u + lane * 4u;
            float v0 = bias;
            float v1 = bias;
            float v2 = bias;
            float v3 = bias;
            for (uint32_t s = 0u; s < splits; ++s) {
                const float* src = partials +
                                   (static_cast<uint64_t>(s) * args.rows + row) *
                                       args.out_features +
                                   col;
                v0 += src[0];
                v1 += src[1];
                v2 += src[2];
                v3 += src[3];
            }
            exl3_detail::fwht128(v0, v1, v2, v3, lane);
            const float s0 = __half2float(__ushort_as_half(svh[col + 0]));
            const float s1 = __half2float(__ushort_as_half(svh[col + 1]));
            const float s2 = __half2float(__ushort_as_half(svh[col + 2]));
            const float s3 = __half2float(__ushort_as_half(svh[col + 3]));
            const uint64_t out_base =
                static_cast<uint64_t>(row) * args.output_row_stride + col;
            const float y0 = v0 * scale * s0;
            const float y1 = v1 * scale * s1;
            const float y2 = v2 * scale * s2;
            const float y3 = v3 * scale * s3;
            if (args.output_dtype == 0u) {
                shelf::bf16_t* dst = static_cast<shelf::bf16_t*>(args.output) + out_base;
                dst[0] = shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(y0))};
                dst[1] = shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(y1))};
                dst[2] = shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(y2))};
                dst[3] = shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(y3))};
            } else {
                float* dst = static_cast<float*>(args.output) + out_base;
                dst[0] = y0;
                dst[1] = y1;
                dst[2] = y2;
                dst[3] = y3;
            }
        }
    }
}

using LaunchPlan = exl3_detail::mul1::LaunchPlan;

template <uint32_t kGroups, bool kPre, uint32_t kNTiles, uint32_t kDecode>
hipError_t launch_kernel(const Exl3GemmArgs& args, const LaunchPlan& plan,
                         hipStream_t stream) {
    if (plan.splits > 1u) {
        gemm_exl3_w16a16_kernel<kGroups, true, kPre, kNTiles, kDecode>
            <<<dim3(plan.grid_x * plan.splits, plan.grid_y, plan.grid_z), kThreads, 0,
               stream>>>(args, plan.splits);
    } else {
        gemm_exl3_w16a16_kernel<kGroups, false, kPre, kNTiles, kDecode>
            <<<dim3(plan.grid_x, plan.grid_y, plan.grid_z), kThreads, 0, stream>>>(args,
                                                                                  1u);
    }
    return hipGetLastError();
}

template <bool kPre, uint32_t kDecode>
hipError_t launch_plan_typed(const Exl3GemmArgs& args, const LaunchPlan& plan,
                             hipStream_t stream) {
    const uint32_t n_tiles = plan.n_tiles;
    hipError_t err = hipSuccess;
    if (n_tiles == 2u) {
        err = plan.groups == 4u   ? launch_kernel<4u, kPre, 2u, kDecode>(args, plan, stream)
              : plan.groups == 2u ? launch_kernel<2u, kPre, 2u, kDecode>(args, plan, stream)
                                  : launch_kernel<1u, kPre, 2u, kDecode>(args, plan, stream);
    } else {
        err = plan.groups == 4u   ? launch_kernel<4u, kPre, 1u, kDecode>(args, plan, stream)
              : plan.groups == 2u ? launch_kernel<2u, kPre, 1u, kDecode>(args, plan, stream)
                                  : launch_kernel<1u, kPre, 1u, kDecode>(args, plan, stream);
    }
    if (err != hipSuccess) return err;
    if (plan.splits > 1u) {
        const uint32_t reduce_grid_y = (args.rows + kRowTile - 1u) / kRowTile;
        if (n_tiles == 2u) {
            exl3_splitk_reduce_kernel<2u><<<dim3(plan.grid_x, reduce_grid_y), kThreads,
                                            0, stream>>>(args, plan.splits);
        } else {
            exl3_splitk_reduce_kernel<1u><<<dim3(plan.grid_x, reduce_grid_y), kThreads,
                                            0, stream>>>(args, plan.splits);
        }
    }
    return hipGetLastError();
}

template <bool kPre>
hipError_t launch_plan(const Exl3GemmArgs& args, const LaunchPlan& plan,
                       hipStream_t stream) {
    if (args.fast_decode != 0u) {
        if (args.ka == 5u && args.half == 0u) {
            return launch_plan_typed<kPre, 1u>(args, plan, stream);
        }
        if (args.ka == 3u && args.half == 0u) {
            return launch_plan_typed<kPre, 2u>(args, plan, stream);
        }
        if (args.ka == 3u && args.half == 1u) {
            return launch_plan_typed<kPre, 3u>(args, plan, stream);
        }
    }
    return launch_plan_typed<kPre, 0u>(args, plan, stream);
}

}

hipError_t launch_gemm_exl3_w16a16(const Exl3GemmArgs& args, hipStream_t stream) {
    if (args.trellis == nullptr || args.suh == nullptr || args.svh == nullptr ||
        args.input == nullptr || args.output == nullptr) {
        return hipErrorInvalidValue;
    }
    if (args.rows == 0u || args.out_features == 0u || args.k == 0u) return hipErrorInvalidValue;
    if ((args.out_features % kOutBlockBase) != 0u || (args.k % 128u) != 0u) {
        return hipErrorInvalidValue;
    }
    if (args.ka < 1u || args.ka > 8u || (args.half != 0u && args.ka > 3u)) {
        return hipErrorInvalidValue;
    }
    return launch_plan<false>(args, exl3_detail::mul1::make_launch_plan(args), stream);
}

hipError_t launch_gemm_exl3_w16a16_grouped(const Exl3GroupedArgs& args,
                                           hipStream_t stream) {
    if (args.trellis == nullptr || args.suh == nullptr || args.svh == nullptr ||
        args.input == nullptr || args.output == nullptr || args.group_expert == nullptr ||
        args.group_rows == nullptr || args.group_count == nullptr) {
        return hipErrorInvalidValue;
    }
    if (args.out_features == 0u || (args.out_features % kOutBlockBase) != 0u ||
        args.k == 0u || (args.k % 128u) != 0u) {
        return hipErrorInvalidValue;
    }
    if (args.ka < 1u || args.ka > 8u || (args.half != 0u && args.ka > 3u)) {
        return hipErrorInvalidValue;
    }
    if (args.input_row_stride < args.k || args.output_row_stride < args.out_features ||
        args.group_capacity == 0u || args.group_capacity > 65535u) {
        return hipErrorInvalidValue;
    }
    if (args.mul24 != 0u && args.half != 0u) return hipErrorInvalidValue;  // mul24 has integer bit rates only
    if (args.int8_act != 0u && args.mul24 == 0u) return hipErrorInvalidValue;  // INT8 activations exist for mul24 only

    Exl3GemmArgs dense;
    dense.trellis = args.trellis;
    dense.suh = args.suh;
    dense.svh = args.svh;
    dense.input = args.input;
    dense.output = args.output;
    dense.rows = kRowTile;
    dense.out_features = args.out_features;
    dense.k = args.k;
    dense.input_row_stride = args.input_row_stride;
    dense.output_row_stride = args.output_row_stride;
    dense.output_dtype = args.output_dtype;
    dense.ka = args.ka;
    dense.half = args.half;
    static const uint32_t fast_decode = [] {
        const char* e = std::getenv("SHELF_EXL3_FAST_DECODE");
        return e != nullptr ? static_cast<uint32_t>(std::strtoul(e, nullptr, 10)) : 1u;
    }();
    dense.fast_decode = fast_decode;
    static const uint32_t fuse_override = [] {
        const char* e = std::getenv("SHELF_EXL3_GROUPED_FUSE");
        if (e == nullptr) return kExl3GroupedFuse;
        return static_cast<uint32_t>(std::strtoul(e, nullptr, 10));
    }();
    static const uint32_t ntiles_override = [] {
        const char* e = std::getenv("SHELF_EXL3_GROUPED_NTILES");
        return e != nullptr ? static_cast<uint32_t>(std::strtoul(e, nullptr, 10)) : 0u;
    }();
    dense.n_tiles = ntiles_override;
    dense.groups = (args.expert_group_begin != nullptr && fuse_override > 1u) ? fuse_override
                                                                            : 1u;
    dense.group_expert = args.group_expert;
    dense.group_rows = args.group_rows;
    dense.group_count = args.group_count;
    dense.expert_group_begin = args.expert_group_begin;
    dense.row_token = args.row_token;
    dense.trellis_batch_stride = args.trellis_batch_stride;
    dense.suh_batch_stride = args.suh_batch_stride;
    dense.svh_batch_stride = args.svh_batch_stride;

    // An n-tile override (SHELF_EXL3_GROUPED_NTILES) that does not divide the output would silently drop columns.
    if ((args.out_features % (kOutBlockBase * exl3_detail::mul1::pick_n_tiles(dense))) != 0u) return hipErrorInvalidValue;
    LaunchPlan plan;
    plan.n_tiles = exl3_detail::mul1::pick_n_tiles(dense);
    plan.grid_x = args.out_features / (kOutBlockBase * plan.n_tiles);
    plan.grid_y = 1u;
    plan.grid_z = args.group_capacity;
    plan.splits = 1u;
    plan.groups = dense.groups;
    // mul24 has its own kernel (exl3_mul24_grouped.hip): same group plan, mul24 decode, FP16 or INT8 activations.
    if (args.mul24 != 0u)
        return launch_exl3_mul24_grouped(dense, args.int8_act != 0u, exl3_detail::mul1::pick_n_tiles(dense), plan.grid_x,
                                         plan.grid_z, stream);
    return launch_plan<false>(dense, plan, stream);
}

hipError_t launch_gemm_exl3_w16a16_pre(const Exl3GemmArgs& args, const void* xrot,
                                       hipStream_t stream) {
    if (args.trellis == nullptr || args.suh == nullptr || args.svh == nullptr ||
        args.output == nullptr) {
        return hipErrorInvalidValue;
    }
    if (args.rows == 0u || args.out_features == 0u || args.k == 0u) return hipErrorInvalidValue;
    if ((args.out_features % kOutBlockBase) != 0u || (args.k % 128u) != 0u) {
        return hipErrorInvalidValue;
    }
    if (args.ka < 1u || args.ka > 8u || (args.half != 0u && args.ka > 3u)) {
        return hipErrorInvalidValue;
    }
    Exl3GemmArgs pre_args = args;
    pre_args.xrot = const_cast<void*>(xrot);
    return launch_plan<true>(pre_args, exl3_detail::mul1::make_launch_plan(pre_args), stream);
}


}  // namespace shelf::kernel
