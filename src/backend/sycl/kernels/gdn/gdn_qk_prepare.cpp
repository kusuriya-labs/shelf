// vendored from PhaseShift (own work); namespace/type adapted only.
// (SYCL port of backend/hip kernels/gdn/gdn_qk_prepare.hip.)
#include <shelf/backend/sycl/kernels/gdn/gdn_qk_prepare.h>
#include <shelf/backend/sycl/kernels/detail/gdn_prepare_device.h>
#include <shelf/backend/sycl/kernels/detail/sycl_util.h>

#include <cstddef>
#include <cstdint>

namespace shelf::syclk {

namespace {

using bf16 = shelf::bf16_t;

constexpr uint32_t kQkThreads = detail::kGdnL2Threads;
static_assert(kQkThreads == 32u);
static_assert(kQkThreads == kFusedGdnQkPrepareThreads);

} // namespace

::shelf::Status launch_fused_gdn_qk_prepare(
    const FusedGdnQkPrepareArgs& args,
    const ::shelf::sycl::Stream& stream) {
    if (args.q == nullptr || args.k == nullptr || args.nq_out == nullptr ||
        args.nk_out == nullptr || args.sq_out == nullptr)
        return ::shelf::Status::invalid_argument("null pointer argument", __FILE__, __LINE__);
    if (args.rows == 0u || args.features == 0u || args.group_size == 0u)
        return ::shelf::Status::invalid_argument("empty launch rejected", __FILE__, __LINE__);
    if (args.features % args.group_size != 0u)
        return ::shelf::Status::invalid_argument("features % group_size != 0", __FILE__, __LINE__);
    if (args.q_row_stride < args.features || args.k_row_stride < args.features ||
        args.nq_row_stride < args.features || args.nk_row_stride < args.features ||
        args.sq_row_stride < args.features)
        return ::shelf::Status::invalid_argument("row stride too small", __FILE__, __LINE__);
    if (!::shelf::syclk::wave32_supported(stream.queue()))
        return ::shelf::Status::unsupported(
            "qk_prepare requires wave32 sub-groups", __FILE__, __LINE__);
    const uint32_t groups = args.features / args.group_size;
    const bf16* q = args.q;
    const bf16* k = args.k;
    ::sycl::queue& sq = stream.queue();
    return ::shelf::syclk::sycl_submit(sq, "sycl-submit", __FILE__, __LINE__, [&](::sycl::handler& cgh) {
        cgh.parallel_for(
            ::sycl::nd_range<2>{::sycl::range<2>{2u * groups * kQkThreads, args.rows},
                                ::sycl::range<2>{kQkThreads, 1}},
            [=](::sycl::nd_item<2> it) {
                const uint32_t row = static_cast<uint32_t>(it.get_group(1));
                const uint32_t lane = static_cast<uint32_t>(it.get_local_linear_id());
                const uint32_t half = static_cast<uint32_t>(it.get_group(0));
                if (half < groups) {
                    const size_t in_off = static_cast<size_t>(row) * args.q_row_stride +
                                          static_cast<size_t>(half) * args.group_size;
                    const size_t out_off = static_cast<size_t>(row) * args.nq_row_stride +
                                           static_cast<size_t>(half) * args.group_size;
                    const size_t scaled_off = static_cast<size_t>(row) * args.sq_row_stride +
                                              static_cast<size_t>(half) * args.group_size;
                    const bool aligned = (in_off & 3u) == 0u;
                    detail::gdn_l2_normalize_head_body<true>(
                        q + in_off,
                        args.nq_out + out_off, args.sq_out + scaled_off, args.scale,
                        it.get_sub_group(), lane, args.group_size, args.eps, aligned);
                } else {
                    const uint32_t g = half - groups;
                    const size_t in_off = static_cast<size_t>(row) * args.k_row_stride +
                                          static_cast<size_t>(g) * args.group_size;
                    const size_t out_off = static_cast<size_t>(row) * args.nk_row_stride +
                                           static_cast<size_t>(g) * args.group_size;
                    const bool aligned = (in_off & 3u) == 0u;
                    detail::gdn_l2_normalize_head_body<false>(
                        k + in_off,
                        args.nk_out + out_off, nullptr, 0.0f, it.get_sub_group(),
                        lane, args.group_size, args.eps, aligned);
                }
            });
    });
}

} // namespace shelf::syclk
