// vendored from PhaseShift (own work); namespace/type adapted only.
// (SYCL port of backend/hip kernels/attention/attention_paged_decode.hip:
// kernel entries + launchers. Device bodies live in the private header.
// The extern "C" phaseshift_* snapshot symbols are dropped: no SYCL-side
// tool reads kernel symbols by name.)
#include <shelf/backend/sycl/kernels/attention/paged_attention.h>
#include <shelf/backend/sycl/kernels/attention/detail/paged_decode_device.h>
#include <shelf/backend/sycl/kernels/attention/kv_types.h>
#include <shelf/backend/sycl/kernels/e4m3_codec.h>
#include <shelf/backend/sycl/kernels/detail/sycl_util.h>

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <cfloat>

namespace shelf::syclk {

namespace v_decode_bf16 = attn_detail::v_decode_bf16;
namespace v_decode_bf16_split = attn_detail::v_decode_bf16_split;
namespace v_decode_bf16_split_qh = attn_detail::v_decode_bf16_split_qh;
namespace v_decode_fp8 = attn_detail::v_decode_fp8;
namespace v_decode_fp8_split = attn_detail::v_decode_fp8_split;

using bf16 = shelf::bf16_t;

namespace {

// local accessor bundle for the decode kernels (kWaves=8, kHeadLimit=256)
struct DecodeLocalBundle {
    ::sycl::local_accessor<float, 1> merge_max;
    ::sycl::local_accessor<float, 1> merge_sum;
    ::sycl::local_accessor<float, 1> acc;
    ::sycl::local_accessor<uint32_t, 1> row_valid;
    ::sycl::local_accessor<uint32_t, 1> visible;
    ::sycl::local_accessor<uint32_t, 1> invalid;
    ::sycl::local_accessor<const uint32_t*, 1> block_table;
    ::sycl::local_accessor<uint32_t, 1> anc_base;
    ::sycl::local_accessor<uint32_t, 1> anc_count;
    explicit DecodeLocalBundle(::sycl::handler& cgh)
        : merge_max(::sycl::range<1>{v_decode_bf16::kWaves}, cgh),
          merge_sum(::sycl::range<1>{v_decode_bf16::kWaves}, cgh),
          acc(::sycl::range<1>{v_decode_bf16::kWaves * v_decode_bf16::kHeadLimit}, cgh),
          row_valid(::sycl::range<1>{1}, cgh),
          visible(::sycl::range<1>{1}, cgh),
          invalid(::sycl::range<1>{1}, cgh),
          block_table(::sycl::range<1>{1}, cgh),
          anc_base(::sycl::range<1>{1}, cgh),
          anc_count(::sycl::range<1>{1}, cgh) {}

    v_decode_bf16::Locals locals() const {
        return v_decode_bf16::Locals{
            merge_max.get_pointer().get(), merge_sum.get_pointer().get(),
            acc.get_pointer().get(), row_valid.get_pointer().get(),
            visible.get_pointer().get(), invalid.get_pointer().get(),
            block_table.get_pointer().get(), anc_base.get_pointer().get(),
            anc_count.get_pointer().get()};
    }
};

} // namespace

::shelf::Status launch_attention_paged_bf16(
    const PagedAttentionCommonArgs& common,
    const bf16_t* k_pool,
    const bf16_t* v_pool,
    const ::shelf::sycl::Stream& stream) {
    using namespace v_decode_bf16;
    if (common.rows == 0u || common.head_dim == 0u || common.head_dim > kHeadLimit)
        return ::shelf::Status::invalid_argument("bad paged attention shape", __FILE__, __LINE__);
    if (common.q_heads == 0u || common.kv_heads == 0u || common.q_heads % common.kv_heads != 0u)
        return ::shelf::Status::invalid_argument("bad head counts", __FILE__, __LINE__);
    if (common.page_tokens == 0u || common.block_table_stride == 0u || common.num_pages == 0u)
        return ::shelf::Status::invalid_argument("bad paging", __FILE__, __LINE__);
    if (common.q == nullptr || common.output == nullptr ||
        common.row_sequence_slots == nullptr || common.row_positions == nullptr ||
        common.block_tables == nullptr || k_pool == nullptr || v_pool == nullptr)
        return ::shelf::Status::invalid_argument("null pointer argument", __FILE__, __LINE__);
    if (!::shelf::syclk::wave32_supported(stream.queue()))
        return ::shelf::Status::unsupported(
            "paged attention decode requires wave32 sub-groups", __FILE__, __LINE__);
    const uint32_t grid_x = common.rows * common.q_heads;
    if (grid_x == 0u) return ::shelf::Status::invalid_argument("empty grid", __FILE__, __LINE__);
    const PagedAttentionCommonArgs a = common;
    const bf16* kp = reinterpret_cast<const bf16*>(k_pool);
    const bf16* vp = reinterpret_cast<const bf16*>(v_pool);
    const bool full_head = (common.head_dim == kHeadLimit);
    ::sycl::queue& q = stream.queue();
    return ::shelf::syclk::sycl_submit(q, "sycl-submit", __FILE__, __LINE__, [&](::sycl::handler& cgh) {
        DecodeLocalBundle B(cgh);
        cgh.parallel_for(
            ::sycl::nd_range<1>{::sycl::range<1>{grid_x * kThreads},
                                ::sycl::range<1>{kThreads}},
            [=](::sycl::nd_item<1> it) {
                Locals L = B.locals();
                if (full_head)
                    attention_paged_bf16_body<true>(it, L, a, kp, vp, a.output);
                else
                    attention_paged_bf16_body<false>(it, L, a, kp, vp, a.output);
            });
    });
}

namespace {

uint32_t attention_split_reduce_qh_per_wg() {
    const char* s = std::getenv("PHASESHIFT_PAGED_ATTENTION_SPLIT_REDUCE_QH");
    if (s == nullptr || s[0] == '\0') return 1u;
    const long n = std::strtol(s, nullptr, 10);
    if (n == 1 || n == 2 || n == 3 || n == 4 || n == 6) return static_cast<uint32_t>(n);
    return 1u;
}

} // namespace

uint32_t paged_attention_split_reduce_qh_per_wg() {
    return attention_split_reduce_qh_per_wg();
}

::shelf::Status launch_attention_paged_bf16_split(
    const PagedAttentionCommonArgs& common,
    const bf16_t* k_pool,
    const bf16_t* v_pool,
    float* partials,
    uint32_t splits,
    const ::shelf::sycl::Stream& stream) {
    using namespace v_decode_bf16_split;
    if (common.rows == 0u || common.head_dim == 0u || common.head_dim > kHeadLimit)
        return ::shelf::Status::invalid_argument("bad paged attention shape", __FILE__, __LINE__);
    if (common.q_heads == 0u || common.kv_heads == 0u || common.q_heads % common.kv_heads != 0u)
        return ::shelf::Status::invalid_argument("bad head counts", __FILE__, __LINE__);
    const uint32_t q_per_kv = common.q_heads / common.kv_heads;
    if (q_per_kv != 4u && q_per_kv != 6u)
        return ::shelf::Status::unsupported("q_per_kv not 4 or 6", __FILE__, __LINE__);
    if (common.page_tokens == 0u || common.block_table_stride == 0u || common.num_pages == 0u)
        return ::shelf::Status::invalid_argument("bad paging", __FILE__, __LINE__);
    if (partials == nullptr || (splits != 2u && splits != 4u && splits != 8u && splits != 16u))
        return ::shelf::Status::invalid_argument("bad splits", __FILE__, __LINE__);
    if (common.q == nullptr || common.output == nullptr ||
        common.row_sequence_slots == nullptr || common.row_positions == nullptr ||
        common.block_tables == nullptr || k_pool == nullptr || v_pool == nullptr)
        return ::shelf::Status::invalid_argument("null pointer argument", __FILE__, __LINE__);
    if (!::shelf::syclk::wave32_supported(stream.queue()))
        return ::shelf::Status::unsupported(
            "paged attention split requires wave32 sub-groups", __FILE__, __LINE__);
    const uint32_t grid_x = common.rows * common.kv_heads * splits;
    if (grid_x == 0u) return ::shelf::Status::invalid_argument("empty grid", __FILE__, __LINE__);
    const PagedAttentionCommonArgs a = common;
    const bf16* kp = reinterpret_cast<const bf16*>(k_pool);
    const bf16* vp = reinterpret_cast<const bf16*>(v_pool);
    const bool full_head = (common.head_dim == kHeadLimit);

    const auto launch = [&](auto kq_tag) -> ::shelf::Status {
        constexpr uint32_t kQ = decltype(kq_tag)::value;
        ::sycl::queue& q = stream.queue();
        return ::shelf::syclk::sycl_submit(q, "sycl-submit", __FILE__, __LINE__, [&](::sycl::handler& cgh) {
            ::sycl::local_accessor<float, 1> q_lds(::sycl::range<1>{kQ * kHeadLimit}, cgh);
            ::sycl::local_accessor<float, 1> acc_lds(
                ::sycl::range<1>{kSplitWaves * kQ * kHeadLimit}, cgh);
            ::sycl::local_accessor<float, 1> m_lds(::sycl::range<1>{kSplitWaves * kQ}, cgh);
            ::sycl::local_accessor<float, 1> s_lds(::sycl::range<1>{kSplitWaves * kQ}, cgh);
            ::sycl::local_accessor<uint32_t, 1> row_valid(::sycl::range<1>{1}, cgh);
            ::sycl::local_accessor<uint32_t, 1> visible(::sycl::range<1>{1}, cgh);
            ::sycl::local_accessor<uint32_t, 1> page_begin(::sycl::range<1>{1}, cgh);
            ::sycl::local_accessor<uint32_t, 1> page_end(::sycl::range<1>{1}, cgh);
            ::sycl::local_accessor<const uint32_t*, 1> block_table(::sycl::range<1>{1}, cgh);
            cgh.parallel_for(
                ::sycl::nd_range<1>{::sycl::range<1>{grid_x * kSplitThreads},
                                    ::sycl::range<1>{kSplitThreads}},
                [=](::sycl::nd_item<1> it) {
                    SplitLocals<kQ> L{
                        q_lds.get_pointer().get(), acc_lds.get_pointer().get(),
                        m_lds.get_pointer().get(), s_lds.get_pointer().get(),
                        row_valid.get_pointer().get(), visible.get_pointer().get(),
                        page_begin.get_pointer().get(), page_end.get_pointer().get(),
                        block_table.get_pointer().get()};
                    if (full_head)
                        attention_paged_split_bf16_body<kQ, true>(it, L, a, kp, vp,
                                                                  partials, splits);
                    else
                        attention_paged_split_bf16_body<kQ, false>(it, L, a, kp, vp,
                                                                   partials, splits);
                });
        });
    };
    if (q_per_kv == 4u) return launch(std::integral_constant<uint32_t, 4u>{});
    return launch(std::integral_constant<uint32_t, 6u>{});
}

::shelf::Status launch_attention_paged_bf16_split_reduce(
    const PagedAttentionCommonArgs& common,
    const float* partials,
    uint32_t splits,
    const ::shelf::sycl::Stream& stream) {
    using namespace v_decode_bf16_split;
    if (common.rows == 0u || common.head_dim == 0u || common.head_dim > kHeadLimit)
        return ::shelf::Status::invalid_argument("bad paged attention shape", __FILE__, __LINE__);
    if (common.q_heads == 0u || common.kv_heads == 0u || common.q_heads % common.kv_heads != 0u)
        return ::shelf::Status::invalid_argument("bad head counts", __FILE__, __LINE__);
    const uint32_t q_per_kv = common.q_heads / common.kv_heads;
    if (q_per_kv != 4u && q_per_kv != 6u)
        return ::shelf::Status::unsupported("q_per_kv not 4 or 6", __FILE__, __LINE__);
    if (partials == nullptr || (splits != 2u && splits != 4u && splits != 8u && splits != 16u))
        return ::shelf::Status::invalid_argument("bad splits", __FILE__, __LINE__);
    if (common.output == nullptr || common.row_sequence_slots == nullptr ||
        common.row_positions == nullptr)
        return ::shelf::Status::invalid_argument("null pointer argument", __FILE__, __LINE__);
    const PagedAttentionCommonArgs a = common;
    const bool full_head = common.head_dim == kHeadLimit;

    {
        using namespace v_decode_bf16_split_qh;
        const uint32_t qh_per_wg = attention_split_reduce_qh_per_wg();
        if (qh_per_wg < q_per_kv && q_per_kv % qh_per_wg == 0u) {
            const uint32_t tiles = q_per_kv / qh_per_wg;
            const uint64_t grid64 = static_cast<uint64_t>(a.rows) * a.kv_heads * tiles;
            if (grid64 == 0u) return ::shelf::Status::invalid_argument("empty grid", __FILE__, __LINE__);
            const uint32_t grid_x = static_cast<uint32_t>(grid64);
            const auto launch_qh = [&](auto kq_tag, auto qh_tag) -> ::shelf::Status {
                constexpr uint32_t kQ = decltype(kq_tag)::value;
                constexpr uint32_t kQh = decltype(qh_tag)::value;
                ::sycl::queue& q = stream.queue();
                return ::shelf::syclk::sycl_submit(q, "sycl-submit", __FILE__, __LINE__, [&](::sycl::handler& cgh) {
                    ::sycl::local_accessor<float, 1> g(::sycl::range<1>{kQh}, cgh);
                    ::sycl::local_accessor<float, 1> sum(::sycl::range<1>{kQh}, cgh);
                    ::sycl::local_accessor<float, 1> wgt(
                        ::sycl::range<1>{kQh * v_decode_bf16_split_qh::kSplitMaxSplits}, cgh);
                    ::sycl::local_accessor<uint32_t, 1> row_valid(::sycl::range<1>{1}, cgh);
                    cgh.parallel_for(
                        ::sycl::nd_range<1>{
                            ::sycl::range<1>{grid_x * kPagedAttentionSplitBlock},
                            ::sycl::range<1>{kPagedAttentionSplitBlock}},
                        [=](::sycl::nd_item<1> it) {
                            if (full_head)
                                attention_paged_split_reduce_qh_bf16_body<kQ, true, kQh>(
                                    it, g.get_pointer().get(), sum.get_pointer().get(),
                                    wgt.get_pointer().get(), row_valid.get_pointer().get(),
                                    a, partials, splits, a.output);
                            else
                                attention_paged_split_reduce_qh_bf16_body<kQ, false, kQh>(
                                    it, g.get_pointer().get(), sum.get_pointer().get(),
                                    wgt.get_pointer().get(), row_valid.get_pointer().get(),
                                    a, partials, splits, a.output);
                        });
                });
            };
            if (q_per_kv == 4u) {
                switch (qh_per_wg) {
                    case 1u: return launch_qh(std::integral_constant<uint32_t, 4u>{}, std::integral_constant<uint32_t, 1u>{});
                    case 2u: return launch_qh(std::integral_constant<uint32_t, 4u>{}, std::integral_constant<uint32_t, 2u>{});
                    case 3u: return launch_qh(std::integral_constant<uint32_t, 4u>{}, std::integral_constant<uint32_t, 3u>{});
                    case 4u: return launch_qh(std::integral_constant<uint32_t, 4u>{}, std::integral_constant<uint32_t, 4u>{});
                    default: return ::shelf::Status::invalid_argument("bad qh_per_wg", __FILE__, __LINE__);
                }
            }
            switch (qh_per_wg) {
                case 1u: return launch_qh(std::integral_constant<uint32_t, 6u>{}, std::integral_constant<uint32_t, 1u>{});
                case 2u: return launch_qh(std::integral_constant<uint32_t, 6u>{}, std::integral_constant<uint32_t, 2u>{});
                case 3u: return launch_qh(std::integral_constant<uint32_t, 6u>{}, std::integral_constant<uint32_t, 3u>{});
                default: return ::shelf::Status::invalid_argument("bad qh_per_wg", __FILE__, __LINE__);
            }
        }
    }
    const uint32_t grid_x = common.rows * common.kv_heads;
    if (grid_x == 0u) return ::shelf::Status::invalid_argument("empty grid", __FILE__, __LINE__);
    const auto launch_base = [&](auto kq_tag) -> ::shelf::Status {
        constexpr uint32_t kQ = decltype(kq_tag)::value;
        ::sycl::queue& q = stream.queue();
        return ::shelf::syclk::sycl_submit(q, "sycl-submit", __FILE__, __LINE__, [&](::sycl::handler& cgh) {
            ::sycl::local_accessor<float, 1> g(::sycl::range<1>{kQ}, cgh);
            ::sycl::local_accessor<float, 1> sum(::sycl::range<1>{kQ}, cgh);
            ::sycl::local_accessor<float, 1> wgt(
                ::sycl::range<1>{kQ * v_decode_bf16_split::kSplitMaxSplits}, cgh);
            ::sycl::local_accessor<uint32_t, 1> row_valid(::sycl::range<1>{1}, cgh);
            cgh.parallel_for(
                ::sycl::nd_range<1>{::sycl::range<1>{grid_x * kSplitThreads},
                                    ::sycl::range<1>{kSplitThreads}},
                [=](::sycl::nd_item<1> it) {
                    if (full_head)
                        attention_paged_split_reduce_bf16_body<kQ, true>(
                            it, g.get_pointer().get(), sum.get_pointer().get(),
                            wgt.get_pointer().get(), row_valid.get_pointer().get(),
                            a, partials, splits, a.output);
                    else
                        attention_paged_split_reduce_bf16_body<kQ, false>(
                            it, g.get_pointer().get(), sum.get_pointer().get(),
                            wgt.get_pointer().get(), row_valid.get_pointer().get(),
                            a, partials, splits, a.output);
                });
        });
    };
    if (q_per_kv == 4u) return launch_base(std::integral_constant<uint32_t, 4u>{});
    return launch_base(std::integral_constant<uint32_t, 6u>{});
}

::shelf::Status launch_attention_paged_fp8_e4m3(
    const PagedAttentionCommonArgs& common,
    const ::shelf::syclk::fp8e4m3_storage_t* k_pool,
    const ::shelf::syclk::fp8e4m3_storage_t* v_pool,
    const float* k_scale_pool,
    const float* v_scale_pool,
    const ::shelf::sycl::Stream& stream) {
    using namespace v_decode_fp8;
    if (common.rows == 0u || common.head_dim == 0u || common.head_dim > kHeadLimit)
        return ::shelf::Status::invalid_argument("bad paged attention shape", __FILE__, __LINE__);
    if (common.q_heads == 0u || common.kv_heads == 0u || common.q_heads % common.kv_heads != 0u)
        return ::shelf::Status::invalid_argument("bad head counts", __FILE__, __LINE__);
    if (common.page_tokens == 0u || common.block_table_stride == 0u || common.num_pages == 0u)
        return ::shelf::Status::invalid_argument("bad paging", __FILE__, __LINE__);
    if (common.q == nullptr || common.output == nullptr ||
        common.row_sequence_slots == nullptr || common.row_positions == nullptr ||
        common.block_tables == nullptr || k_pool == nullptr || v_pool == nullptr ||
        k_scale_pool == nullptr || v_scale_pool == nullptr)
        return ::shelf::Status::invalid_argument("null pointer argument", __FILE__, __LINE__);
    if (!::shelf::syclk::wave32_supported(stream.queue()))
        return ::shelf::Status::unsupported(
            "paged attention fp8 decode requires wave32 sub-groups", __FILE__, __LINE__);
    const uint32_t grid_x = common.rows * common.q_heads;
    if (grid_x == 0u) return ::shelf::Status::invalid_argument("empty grid", __FILE__, __LINE__);
    const PagedAttentionCommonArgs a = common;
    const bool full_head = (common.head_dim == kHeadLimit);
    ::sycl::queue& q = stream.queue();
    return ::shelf::syclk::sycl_submit(q, "sycl-submit", __FILE__, __LINE__, [&](::sycl::handler& cgh) {
        DecodeLocalBundle B(cgh);
        ::sycl::local_accessor<uint16_t, 1> lut(::sycl::range<1>{256}, cgh);
        cgh.parallel_for(
            ::sycl::nd_range<1>{::sycl::range<1>{grid_x * kThreads},
                                ::sycl::range<1>{kThreads}},
            [=](::sycl::nd_item<1> it) {
                auto L = B.locals();
                uint16_t* decode_lut = lut.get_pointer().get();
                if (full_head)
                    attention_paged_fp8_body<true>(it, L, decode_lut, a, k_pool, v_pool,
                                                   k_scale_pool, v_scale_pool, a.output);
                else
                    attention_paged_fp8_body<false>(it, L, decode_lut, a, k_pool, v_pool,
                                                    k_scale_pool, v_scale_pool, a.output);
            });
    });
}

::shelf::Status launch_attention_paged_fp8_split(
    const PagedAttentionCommonArgs& common,
    const ::shelf::syclk::fp8e4m3_storage_t* k_pool,
    const ::shelf::syclk::fp8e4m3_storage_t* v_pool,
    const float* k_scale_pool,
    const float* v_scale_pool,
    float* partials,
    uint32_t splits,
    const ::shelf::sycl::Stream& stream) {
    using namespace v_decode_fp8_split;
    if (common.rows == 0u || common.head_dim == 0u ||
        common.head_dim > v_decode_bf16::kHeadLimit)
        return ::shelf::Status::invalid_argument("bad paged attention shape", __FILE__, __LINE__);
    if (common.q_heads == 0u || common.kv_heads == 0u || common.q_heads % common.kv_heads != 0u)
        return ::shelf::Status::invalid_argument("bad head counts", __FILE__, __LINE__);
    const uint32_t q_per_kv = common.q_heads / common.kv_heads;
    if (q_per_kv != 4u && q_per_kv != 6u)
        return ::shelf::Status::unsupported("q_per_kv not 4 or 6", __FILE__, __LINE__);
    if (common.page_tokens == 0u || common.block_table_stride == 0u || common.num_pages == 0u)
        return ::shelf::Status::invalid_argument("bad paging", __FILE__, __LINE__);
    if (partials == nullptr || (splits != 2u && splits != 4u && splits != 8u && splits != 16u))
        return ::shelf::Status::invalid_argument("bad splits", __FILE__, __LINE__);
    if (common.q == nullptr || common.output == nullptr ||
        common.row_sequence_slots == nullptr || common.row_positions == nullptr ||
        common.block_tables == nullptr || k_pool == nullptr || v_pool == nullptr ||
        k_scale_pool == nullptr || v_scale_pool == nullptr)
        return ::shelf::Status::invalid_argument("null pointer argument", __FILE__, __LINE__);
    if (!::shelf::syclk::wave32_supported(stream.queue()))
        return ::shelf::Status::unsupported(
            "paged attention fp8 split requires wave32 sub-groups", __FILE__, __LINE__);
    const uint32_t grid_x = common.rows * common.kv_heads * splits;
    if (grid_x == 0u) return ::shelf::Status::invalid_argument("empty grid", __FILE__, __LINE__);
    const PagedAttentionCommonArgs a = common;
    const bool full_head = (common.head_dim == v_decode_bf16::kHeadLimit);

    const auto launch = [&](auto kq_tag) -> ::shelf::Status {
        constexpr uint32_t kQ = decltype(kq_tag)::value;
        ::sycl::queue& q = stream.queue();
        return ::shelf::syclk::sycl_submit(q, "sycl-submit", __FILE__, __LINE__, [&](::sycl::handler& cgh) {
            ::sycl::local_accessor<float, 1> q_lds(
                ::sycl::range<1>{kQ * v_decode_bf16::kHeadLimit}, cgh);
            ::sycl::local_accessor<float, 1> acc_lds(
                ::sycl::range<1>{kSplitWaves * kQ * v_decode_bf16::kHeadLimit}, cgh);
            ::sycl::local_accessor<float, 1> m_lds(::sycl::range<1>{kSplitWaves * kQ}, cgh);
            ::sycl::local_accessor<float, 1> s_lds(::sycl::range<1>{kSplitWaves * kQ}, cgh);
            ::sycl::local_accessor<uint32_t, 1> row_valid(::sycl::range<1>{1}, cgh);
            ::sycl::local_accessor<uint32_t, 1> visible(::sycl::range<1>{1}, cgh);
            ::sycl::local_accessor<uint32_t, 1> page_begin(::sycl::range<1>{1}, cgh);
            ::sycl::local_accessor<uint32_t, 1> page_end(::sycl::range<1>{1}, cgh);
            ::sycl::local_accessor<const uint32_t*, 1> block_table(::sycl::range<1>{1}, cgh);
            ::sycl::local_accessor<uint16_t, 1> lut(::sycl::range<1>{256}, cgh);
            cgh.parallel_for(
                ::sycl::nd_range<1>{::sycl::range<1>{grid_x * kSplitThreads},
                                    ::sycl::range<1>{kSplitThreads}},
                [=](::sycl::nd_item<1> it) {
                    using SL = v_decode_bf16_split::SplitLocals<kQ>;
                    constexpr uint32_t kHL = v_decode_bf16::kHeadLimit;
                    SL L{
                        q_lds.get_pointer().get(), acc_lds.get_pointer().get(),
                        m_lds.get_pointer().get(), s_lds.get_pointer().get(),
                        row_valid.get_pointer().get(), visible.get_pointer().get(),
                        page_begin.get_pointer().get(), page_end.get_pointer().get(),
                        block_table.get_pointer().get()};
                    if (full_head)
                        attention_paged_split_fp8_body<kQ, true>(
                            it, L, lut.get_pointer().get(), a, k_pool, v_pool,
                            k_scale_pool, v_scale_pool, partials, splits);
                    else
                        attention_paged_split_fp8_body<kQ, false>(
                            it, L, lut.get_pointer().get(), a, k_pool, v_pool,
                            k_scale_pool, v_scale_pool, partials, splits);
                    (void)kHL;
                });
        });
    };
    if (q_per_kv == 4u) return launch(std::integral_constant<uint32_t, 4u>{});
    return launch(std::integral_constant<uint32_t, 6u>{});
}

} // namespace shelf::syclk
