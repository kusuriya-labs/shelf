// vendored from PhaseShift (own work); namespace/type adapted only.
#include <shelf/backend/hip/kernels/attention/paged_attention.h>
#include <shelf/backend/hip/kernels/e4m3_codec.h>
#include <shelf/backend/hip/kernels/attention/kv_types.h>
#include <shelf/backend/hip/kernels/attention/detail/paged_decode_device.h>
#include <shelf/backend/hip/kernels/detail/tile/wave32.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <cfloat>

namespace shelf::hipk {

// Staged device helpers live in the private header (attn_detail::); keep the
// historical in-TU spelling for entries and launchers via aliases.
namespace v_decode_bf16 = attn_detail::v_decode_bf16;
namespace v_decode_bf16_split = attn_detail::v_decode_bf16_split;
namespace v_decode_bf16_split_qh = attn_detail::v_decode_bf16_split_qh;
namespace v_decode_fp8 = attn_detail::v_decode_fp8;
namespace v_decode_fp8_split = attn_detail::v_decode_fp8_split;

// in PhaseShift this alias came from detail/attention_paged_common.h
// (included but otherwise unused there; not vendored — alias kept verbatim).
using bf16 = __hip_bfloat16;

extern "C" __global__ __launch_bounds__(kPagedAttentionBlock)
void phaseshift_attention_paged_bf16(
    ::shelf::hipk::PagedAttentionBf16Args args) {
    v_decode_bf16::attention_paged_bf16_body<true>(
        args.common, reinterpret_cast<const bf16*>(args.k_pool),
        reinterpret_cast<const bf16*>(args.v_pool), args.common.output);
}

hipError_t launch_attention_paged_bf16(
    const PagedAttentionCommonArgs& common,
    const bf16_t* k_pool,
    const bf16_t* v_pool,
    hipStream_t stream) {
    using namespace v_decode_bf16;
    if (common.rows == 0u || common.head_dim == 0u || common.head_dim > kHeadLimit)
        return hipErrorInvalidValue;
    if (common.q_heads == 0u || common.kv_heads == 0u || common.q_heads % common.kv_heads != 0u)
        return hipErrorInvalidValue;
    if (common.page_tokens == 0u || common.block_table_stride == 0u || common.num_pages == 0u)
        return hipErrorInvalidValue;
    if (common.q == nullptr || common.output == nullptr ||
        common.row_sequence_slots == nullptr || common.row_positions == nullptr ||
        common.block_tables == nullptr || k_pool == nullptr || v_pool == nullptr)
        return hipErrorInvalidValue;
    const uint64_t grid_x = static_cast<uint64_t>(common.rows) * common.q_heads;
    if (grid_x == 0u) return hipErrorInvalidValue;
    const PagedAttentionCommonArgs a = common;
    const dim3 grid(static_cast<uint32_t>(grid_x));
    if (common.head_dim == kHeadLimit) {
        PagedAttentionBf16Args args{};
        args.common = a;
        args.k_pool = k_pool;
        args.v_pool = v_pool;
        phaseshift_attention_paged_bf16<<<grid, kThreads, 0, stream>>>(args);
    } else {
        attention_paged_bf16_impl<false><<<grid, kThreads, 0, stream>>>(
            a, reinterpret_cast<const bf16*>(k_pool), reinterpret_cast<const bf16*>(v_pool), common.output);
    }
    return hipGetLastError();
}

namespace {

uint32_t attention_split_reduce_qh_per_wg() {
    const char* s = std::getenv("PHASESHIFT_PAGED_ATTENTION_SPLIT_REDUCE_QH");
    if (s == nullptr || s[0] == '\0') return 1u;
    const long n = std::strtol(s, nullptr, 10);
    if (n == 1 || n == 2 || n == 3 || n == 4 || n == 6) return static_cast<uint32_t>(n);
    return 1u;
}

}  // namespace

uint32_t paged_attention_split_reduce_qh_per_wg() {
    return attention_split_reduce_qh_per_wg();
}

extern "C" __global__ __launch_bounds__(kPagedAttentionSplitBlock)
void phaseshift_attention_paged_bf16_split_q4_full(
    ::shelf::hipk::PagedAttentionSplitArgs args) {
    v_decode_bf16_split::attention_paged_split_bf16_body<4u, true>(
        args.common, reinterpret_cast<const bf16*>(args.k_pool),
        reinterpret_cast<const bf16*>(args.v_pool), args.partials, args.splits);
}

extern "C" __global__ __launch_bounds__(kPagedAttentionSplitBlock)
void phaseshift_attention_paged_bf16_reduce_q4_full_qh1(
    ::shelf::hipk::PagedAttentionSplitReduceArgs args) {
    v_decode_bf16_split_qh::attention_paged_split_reduce_qh_bf16_body<4u, true, 1u>(
        args.common, args.partials, args.splits, args.common.output);
}

extern "C" __global__ __launch_bounds__(kPagedAttentionSplitBlock)
void phaseshift_attention_paged_bf16_reduce_q4_full_qh2(
    ::shelf::hipk::PagedAttentionSplitReduceArgs args) {
    v_decode_bf16_split_qh::attention_paged_split_reduce_qh_bf16_body<4u, true, 2u>(
        args.common, args.partials, args.splits, args.common.output);
}

extern "C" __global__ __launch_bounds__(kPagedAttentionSplitBlock)
void phaseshift_attention_paged_bf16_reduce_q4_full_base(
    ::shelf::hipk::PagedAttentionSplitReduceArgs args) {
    v_decode_bf16_split::attention_paged_split_reduce_bf16_body<4u, true>(
        args.common, args.partials, args.splits, args.common.output);
}

extern "C" __global__ __launch_bounds__(kPagedAttentionSplitBlock)
void phaseshift_attention_paged_bf16_split_other(
    ::shelf::hipk::PagedAttentionSplitArgs args) {
    v_decode_bf16_split::attention_paged_split_bf16_body<6u, true>(
        args.common, reinterpret_cast<const bf16*>(args.k_pool),
        reinterpret_cast<const bf16*>(args.v_pool), args.partials, args.splits);
}

extern "C" __global__ __launch_bounds__(kPagedAttentionSplitBlock)
void phaseshift_attention_paged_bf16_reduce_other(
    ::shelf::hipk::PagedAttentionSplitReduceArgs args) {
    v_decode_bf16_split::attention_paged_split_reduce_bf16_body<6u, true>(
        args.common, args.partials, args.splits, args.common.output);
}

hipError_t launch_attention_paged_bf16_split(
    const PagedAttentionCommonArgs& common,
    const bf16_t* k_pool,
    const bf16_t* v_pool,
    float* partials,
    uint32_t splits,
    hipStream_t stream) {
    using namespace v_decode_bf16_split;
    if (common.rows == 0u || common.head_dim == 0u || common.head_dim > kHeadLimit)
        return hipErrorInvalidValue;
    if (common.q_heads == 0u || common.kv_heads == 0u || common.q_heads % common.kv_heads != 0u)
        return hipErrorInvalidValue;
    const uint32_t q_per_kv = common.q_heads / common.kv_heads;
    if (q_per_kv != 4u && q_per_kv != 6u)
        return hipErrorInvalidValue;
    if (common.page_tokens == 0u || common.block_table_stride == 0u || common.num_pages == 0u)
        return hipErrorInvalidValue;
    if (partials == nullptr || (splits != 2u && splits != 4u && splits != 8u && splits != 16u))
        return hipErrorInvalidValue;
    if (common.q == nullptr || common.output == nullptr ||
        common.row_sequence_slots == nullptr || common.row_positions == nullptr ||
        common.block_tables == nullptr || k_pool == nullptr || v_pool == nullptr)
        return hipErrorInvalidValue;
    const uint64_t grid_x =
        static_cast<uint64_t>(common.rows) * common.kv_heads * splits;
    if (grid_x == 0u) return hipErrorInvalidValue;
    const PagedAttentionCommonArgs a = common;
    const dim3 grid(static_cast<uint32_t>(grid_x));
    const bool full_head = common.head_dim == kHeadLimit;
    if (q_per_kv == 4u && full_head) {
        PagedAttentionSplitArgs args{};
        args.common = a;
        args.k_pool = k_pool;
        args.v_pool = v_pool;
        args.partials = partials;
        args.splits = splits;
        phaseshift_attention_paged_bf16_split_q4_full
            <<<grid, kSplitThreads, 0, stream>>>(args);
        return hipGetLastError();
    }
    if (q_per_kv == 4u) {
        if (full_head) {
            attention_paged_split_bf16_impl<4u, false><<<grid, kSplitThreads, 0, stream>>>(
                a, reinterpret_cast<const bf16*>(k_pool),
                reinterpret_cast<const bf16*>(v_pool), partials, splits);
        }
    } else {
        if (full_head) {
            PagedAttentionSplitArgs stable{};
            stable.common = a;
            stable.k_pool = k_pool;
            stable.v_pool = v_pool;
            stable.partials = partials;
            stable.splits = splits;
            phaseshift_attention_paged_bf16_split_other
                <<<grid, kSplitThreads, 0, stream>>>(stable);
        } else {
            attention_paged_split_bf16_impl<6u, false><<<grid, kSplitThreads, 0, stream>>>(
                a, reinterpret_cast<const bf16*>(k_pool),
                reinterpret_cast<const bf16*>(v_pool), partials, splits);
        }
    }
    return hipGetLastError();
}

hipError_t launch_attention_paged_bf16_split_reduce(
    const PagedAttentionCommonArgs& common,
    const float* partials,
    uint32_t splits,
    hipStream_t stream) {
    using namespace v_decode_bf16_split;
    if (common.rows == 0u || common.head_dim == 0u || common.head_dim > kHeadLimit)
        return hipErrorInvalidValue;
    if (common.q_heads == 0u || common.kv_heads == 0u || common.q_heads % common.kv_heads != 0u)
        return hipErrorInvalidValue;
    const uint32_t q_per_kv = common.q_heads / common.kv_heads;
    if (q_per_kv != 4u && q_per_kv != 6u)
        return hipErrorInvalidValue;
    if (partials == nullptr || (splits != 2u && splits != 4u && splits != 8u && splits != 16u))
        return hipErrorInvalidValue;
    if (common.output == nullptr || common.row_sequence_slots == nullptr ||
        common.row_positions == nullptr)
        return hipErrorInvalidValue;
    const PagedAttentionCommonArgs a = common;
    const bool full_head = common.head_dim == kHeadLimit;
    {
        using namespace v_decode_bf16_split_qh;
        const uint32_t qh_per_wg = attention_split_reduce_qh_per_wg();
        if (qh_per_wg < q_per_kv && q_per_kv % qh_per_wg == 0u) {
            if (q_per_kv == 4u) {
                if (full_head && (qh_per_wg == 1u || qh_per_wg == 2u)) {
                    PagedAttentionSplitReduceArgs stable{};
                    stable.common = a;
                    stable.partials = partials;
                    stable.splits = splits;
                    const uint64_t qh_grid = static_cast<uint64_t>(a.rows) *
                                             a.kv_heads *
                                             (q_per_kv / qh_per_wg);
                    const dim3 qhg(static_cast<uint32_t>(qh_grid));
                    if (qh_per_wg == 1u) {
                        phaseshift_attention_paged_bf16_reduce_q4_full_qh1
                            <<<qhg, kPagedAttentionSplitBlock, 0, stream>>>(stable);
                    } else {
                        phaseshift_attention_paged_bf16_reduce_q4_full_qh2
                            <<<qhg, kPagedAttentionSplitBlock, 0, stream>>>(stable);
                    }
                    return hipGetLastError();
                }
                return full_head
                    ? launch_split_reduce_qh<4u, true>(a, partials, splits, common.output,
                                                       qh_per_wg, stream)
                    : launch_split_reduce_qh<4u, false>(a, partials, splits, common.output,
                                                        qh_per_wg, stream);
            }
            return full_head
                ? launch_split_reduce_qh<6u, true>(a, partials, splits, common.output,
                                                   qh_per_wg, stream)
                : launch_split_reduce_qh<6u, false>(a, partials, splits, common.output,
                                                    qh_per_wg, stream);
        }
    }
    const uint64_t grid_x = static_cast<uint64_t>(common.rows) * common.kv_heads;
    if (grid_x == 0u) return hipErrorInvalidValue;
    const dim3 grid(static_cast<uint32_t>(grid_x));
    if (q_per_kv == 4u && full_head) {
        PagedAttentionSplitReduceArgs stable{};
        stable.common = a;
        stable.partials = partials;
        stable.splits = splits;
        phaseshift_attention_paged_bf16_reduce_q4_full_base
            <<<grid, kSplitThreads, 0, stream>>>(stable);
        return hipGetLastError();
    }
    if (q_per_kv == 4u) {
        if (full_head) {
            attention_paged_split_reduce_bf16_impl<4u, false><<<grid, kSplitThreads, 0, stream>>>(
                a, partials, splits, common.output);
        }
    } else {
        if (full_head) {
            PagedAttentionSplitReduceArgs stable{};
            stable.common = a;
            stable.partials = partials;
            stable.splits = splits;
            phaseshift_attention_paged_bf16_reduce_other
                <<<grid, kSplitThreads, 0, stream>>>(stable);
        } else {
            attention_paged_split_reduce_bf16_impl<6u, false><<<grid, kSplitThreads, 0, stream>>>(
                a, partials, splits, common.output);
        }
    }
    return hipGetLastError();
}

hipError_t launch_attention_paged_fp8_e4m3(
    const PagedAttentionCommonArgs& common,
    const ::shelf::hipk::fp8e4m3_storage_t* k_pool,
    const ::shelf::hipk::fp8e4m3_storage_t* v_pool,
    const float* k_scale_pool,
    const float* v_scale_pool,
    hipStream_t stream) {
    using namespace v_decode_fp8;
    if (common.rows == 0u || common.head_dim == 0u || common.head_dim > kHeadLimit)
        return hipErrorInvalidValue;
    if (common.q_heads == 0u || common.kv_heads == 0u || common.q_heads % common.kv_heads != 0u)
        return hipErrorInvalidValue;
    if (common.page_tokens == 0u || common.block_table_stride == 0u || common.num_pages == 0u)
        return hipErrorInvalidValue;
    if (common.q == nullptr || common.output == nullptr ||
        common.row_sequence_slots == nullptr || common.row_positions == nullptr ||
        common.block_tables == nullptr || k_pool == nullptr || v_pool == nullptr ||
        k_scale_pool == nullptr || v_scale_pool == nullptr)
        return hipErrorInvalidValue;
    const uint64_t grid_x = static_cast<uint64_t>(common.rows) * common.q_heads;
    if (grid_x == 0u) return hipErrorInvalidValue;
    const PagedAttentionCommonArgs a = common;
    const dim3 grid(static_cast<uint32_t>(grid_x));
    if (common.head_dim == kHeadLimit) {
        attention_paged_fp8_impl<true><<<grid, kThreads, 0, stream>>>(
            a, reinterpret_cast<const fp8*>(k_pool), reinterpret_cast<const fp8*>(v_pool),
            k_scale_pool, v_scale_pool, common.output);
    } else {
        attention_paged_fp8_impl<false><<<grid, kThreads, 0, stream>>>(
            a, reinterpret_cast<const fp8*>(k_pool), reinterpret_cast<const fp8*>(v_pool),
            k_scale_pool, v_scale_pool, common.output);
    }
    return hipGetLastError();
}


extern "C" __global__ __launch_bounds__(kPagedAttentionSplitBlock)
void phaseshift_attention_paged_fp8_split_q4_full(
    ::shelf::hipk::PagedAttentionFp8SplitArgs args) {
    v_decode_fp8_split::attention_paged_split_fp8_body<4u, true>(
        args.common,
        reinterpret_cast<const ::shelf::hipk::fp8e4m3_storage_t*>(args.k_pool),
        reinterpret_cast<const ::shelf::hipk::fp8e4m3_storage_t*>(args.v_pool),
        args.k_scale_pool, args.v_scale_pool, args.partials, args.splits);
}

extern "C" __global__ __launch_bounds__(kPagedAttentionSplitBlock)
void phaseshift_attention_paged_fp8_split_other(
    ::shelf::hipk::PagedAttentionFp8SplitArgs args) {
    v_decode_fp8_split::attention_paged_split_fp8_body<6u, true>(
        args.common,
        reinterpret_cast<const ::shelf::hipk::fp8e4m3_storage_t*>(args.k_pool),
        reinterpret_cast<const ::shelf::hipk::fp8e4m3_storage_t*>(args.v_pool),
        args.k_scale_pool, args.v_scale_pool, args.partials, args.splits);
}

hipError_t launch_attention_paged_fp8_split(
    const PagedAttentionCommonArgs& common,
    const ::shelf::hipk::fp8e4m3_storage_t* k_pool,
    const ::shelf::hipk::fp8e4m3_storage_t* v_pool,
    const float* k_scale_pool,
    const float* v_scale_pool,
    float* partials,
    uint32_t splits,
    hipStream_t stream) {
    using namespace v_decode_fp8_split;
    if (common.rows == 0u || common.head_dim == 0u || common.head_dim > kHeadLimit)
        return hipErrorInvalidValue;
    if (common.q_heads == 0u || common.kv_heads == 0u || common.q_heads % common.kv_heads != 0u)
        return hipErrorInvalidValue;
    const uint32_t q_per_kv = common.q_heads / common.kv_heads;
    if (q_per_kv != 4u && q_per_kv != 6u)
        return hipErrorInvalidValue;
    if (common.page_tokens == 0u || common.block_table_stride == 0u || common.num_pages == 0u)
        return hipErrorInvalidValue;
    if (partials == nullptr || (splits != 2u && splits != 4u && splits != 8u && splits != 16u))
        return hipErrorInvalidValue;
    if (common.q == nullptr || common.output == nullptr ||
        common.row_sequence_slots == nullptr || common.row_positions == nullptr ||
        common.block_tables == nullptr || k_pool == nullptr || v_pool == nullptr ||
        k_scale_pool == nullptr || v_scale_pool == nullptr)
        return hipErrorInvalidValue;
    const uint64_t grid_x =
        static_cast<uint64_t>(common.rows) * common.kv_heads * splits;
    if (grid_x == 0u) return hipErrorInvalidValue;
    const PagedAttentionCommonArgs a = common;
    const dim3 grid(static_cast<uint32_t>(grid_x));
    const bool full_head = common.head_dim == kHeadLimit;
    if (q_per_kv == 4u && full_head) {
        PagedAttentionFp8SplitArgs args{};
        args.common = a;
        args.k_pool = k_pool;
        args.v_pool = v_pool;
        args.k_scale_pool = k_scale_pool;
        args.v_scale_pool = v_scale_pool;
        args.partials = partials;
        args.splits = splits;
        phaseshift_attention_paged_fp8_split_q4_full
            <<<grid, kSplitThreads, 0, stream>>>(args);
        return hipGetLastError();
    }
    if (q_per_kv == 4u) {
        attention_paged_split_fp8_impl<4u, false><<<grid, kSplitThreads, 0, stream>>>(
            a, reinterpret_cast<const fp8*>(k_pool), reinterpret_cast<const fp8*>(v_pool),
            k_scale_pool, v_scale_pool, partials, splits);
        return hipGetLastError();
    }
    if (full_head) {
        PagedAttentionFp8SplitArgs args{};
        args.common = a;
        args.k_pool = k_pool;
        args.v_pool = v_pool;
        args.k_scale_pool = k_scale_pool;
        args.v_scale_pool = v_scale_pool;
        args.partials = partials;
        args.splits = splits;
        phaseshift_attention_paged_fp8_split_other
            <<<grid, kSplitThreads, 0, stream>>>(args);
        return hipGetLastError();
    }
    attention_paged_split_fp8_impl<6u, false><<<grid, kSplitThreads, 0, stream>>>(
        a, reinterpret_cast<const fp8*>(k_pool), reinterpret_cast<const fp8*>(v_pool),
        k_scale_pool, v_scale_pool, partials, splits);
    return hipGetLastError();
}


}  // namespace shelf::hipk
