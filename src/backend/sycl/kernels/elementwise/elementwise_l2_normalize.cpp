// vendored from PhaseShift (own work); namespace/type adapted only.
// (SYCL port of backend/hip kernels/elementwise/elementwise_l2_normalize.hip.)
#include <shelf/backend/sycl/kernels/elementwise/l2_normalize.h>
#include <shelf/backend/sycl/kernels/detail/l2_normalize_device.h>
#include <shelf/backend/sycl/kernels/detail/sycl_util.h>
#include <shelf/foundation/numeric.hpp>

#include <cstddef>
#include <cstdint>

namespace shelf::syclk {

namespace {

using bf16 = detail::bf16;

constexpr uint32_t kL2Threads = detail::kL2Threads;
static_assert(kL2Threads == 32u);
static_assert(kL2Threads == kL2NormalizeThreads);
constexpr uint32_t kL2ElemsPerLane = detail::kL2ElemsPerLane;

} // namespace

::shelf::Status launch_l2_normalize_bf16_f32(
    const bf16_t* input, uint32_t input_row_stride,
    float* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, uint32_t group_size,
    float eps, const ::shelf::sycl::Stream& stream) {
    if (input == nullptr || output == nullptr)
        return ::shelf::Status::invalid_argument("null pointer argument", __FILE__, __LINE__);
    if (rows == 0u || features == 0u || group_size == 0u)
        return ::shelf::Status::invalid_argument("empty launch rejected", __FILE__, __LINE__);
    if (features % group_size != 0u)
        return ::shelf::Status::invalid_argument("features % group_size != 0", __FILE__, __LINE__);
    if (!::shelf::syclk::wave32_supported(stream.queue()))
        return ::shelf::Status::unsupported(
            "l2_normalize requires wave32 sub-groups", __FILE__, __LINE__);
    const uint32_t groups = features / group_size;
    const bf16* in = reinterpret_cast<const bf16*>(input);
    ::sycl::queue& q = stream.queue();
    return ::shelf::syclk::sycl_submit(q, "sycl-submit", __FILE__, __LINE__, [&](::sycl::handler& cgh) {
        cgh.parallel_for(
            ::sycl::nd_range<2>{::sycl::range<2>{groups * kL2Threads, rows},
                                ::sycl::range<2>{kL2Threads, 1}},
            [=](::sycl::nd_item<2> it) {
                const uint32_t row = static_cast<uint32_t>(it.get_group(1));
                const uint32_t g = static_cast<uint32_t>(it.get_group(0));
                const uint32_t lane = static_cast<uint32_t>(it.get_local_linear_id());
                const size_t in_off = static_cast<size_t>(row) * input_row_stride +
                                      static_cast<size_t>(g) * group_size;
                const size_t out_off = static_cast<size_t>(row) * output_row_stride +
                                       static_cast<size_t>(g) * group_size;
                const bool aligned = (in_off & 3u) == 0u;
                detail::l2_normalize_head_body<false>(
                    in + in_off, output + out_off, nullptr, 0.0f, it.get_sub_group(),
                    lane, group_size, eps, aligned);
            });
    });
}

} // namespace shelf::syclk
