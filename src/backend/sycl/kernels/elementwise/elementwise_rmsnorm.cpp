// vendored from PhaseShift (own work); namespace/type adapted only.
// (SYCL port of backend/hip kernels/elementwise/elementwise_rmsnorm.hip.)
#include <shelf/backend/sycl/kernels/elementwise/rmsnorm.h>
#include <shelf/backend/sycl/kernels/detail/rmsnorm_device.h>
#include <shelf/backend/sycl/kernels/detail/sycl_util.h>

#include <cstddef>
#include <cstdint>

namespace shelf::syclk {

namespace {

constexpr uint32_t kMaxBlock = 256u;
constexpr uint32_t kMaxWaves = kMaxBlock / 32u;

template <RmsNormDataType InT, RmsNormDataType OutT, RmsNormDataType Wt,
          RmsNormWeightLayout WLayout, RmsNormWeightMode WMode>
::shelf::Status launch_one(
    const void* input, const void* weight, void* output,
    uint32_t rows, uint32_t features, uint32_t group_size,
    uint32_t input_row_stride, uint32_t output_row_stride,
    float eps, const ::shelf::sycl::Stream& stream) {
    const uint32_t grid = rmsnorm_bf16_pf_oneplus_grid(rows, features, group_size);
    if (grid == 0u) return ::shelf::Status::make_ok();
    const uint32_t block = rmsnorm_bf16_pf_oneplus_block_size(group_size);
    if (!::shelf::syclk::wave32_supported(stream.queue()))
        return ::shelf::Status::unsupported(
            "rmsnorm requires wave32 sub-groups", __FILE__, __LINE__);
    ::sycl::queue& q = stream.queue();
    return ::shelf::syclk::sycl_submit(q, "sycl-submit", __FILE__, __LINE__, [&](::sycl::handler& cgh) {
        ::sycl::local_accessor<float, 1> wave_sums(::sycl::range<1>{kMaxWaves}, cgh);
        cgh.parallel_for(
            ::sycl::nd_range<1>{::sycl::range<1>{grid * block},
                                ::sycl::range<1>{block}},
            [=](::sycl::nd_item<1> it) {
                const uint32_t groups = features / group_size;
                const uint32_t work = static_cast<uint32_t>(it.get_group(0));
                const uint32_t row = work / groups;
                const uint32_t gi = work % groups;
                detail::rmsnorm_row_body<InT, OutT, Wt, WLayout, WMode>(
                    it, wave_sums.get_pointer().get(), input, weight, output,
                    row, gi, group_size, input_row_stride, output_row_stride, eps);
            });
    });
}

} // namespace

uint32_t rmsnorm_bf16_pf_oneplus_block_size(uint32_t group_size) noexcept {
    const uint32_t chunks = (group_size + 3u) / 4u;
    return chunks <= 32u ? 32u : chunks <= 64u ? 64u : chunks <= 128u ? 128u
                                                                      : 256u;
}

uint32_t rmsnorm_bf16_pf_oneplus_grid(uint32_t rows, uint32_t features,
                                      uint32_t group_size) noexcept {
    if (group_size == 0u || features == 0u || features % group_size != 0u) {
        return 0u;
    }
    return rows * (features / group_size);
}

::shelf::Status launch_rmsnorm(
    const void* input,
    RmsNormDataType input_dtype,
    const void* weight,
    RmsNormDataType weight_dtype,
    RmsNormWeightLayout weight_layout,
    RmsNormWeightMode weight_mode,
    void* output,
    RmsNormDataType output_dtype,
    uint32_t rows,
    uint32_t features,
    uint32_t group_size,
    uint32_t input_row_stride,
    uint32_t output_row_stride,
    float eps,
    const ::shelf::sycl::Stream& stream) {
    using T = RmsNormDataType;
    using L = RmsNormWeightLayout;
    using M = RmsNormWeightMode;
    if (group_size == 0u || features == 0u || features % group_size != 0u) {
        return ::shelf::Status::invalid_argument("bad group layout", __FILE__, __LINE__);
    }
    if (input_dtype != T::BF16 && input_dtype != T::F32) return ::shelf::Status::invalid_argument("bad input dtype", __FILE__, __LINE__);
    if (output_dtype != T::BF16 && output_dtype != T::F32) return ::shelf::Status::invalid_argument("bad output dtype", __FILE__, __LINE__);
    if (weight_dtype != T::BF16) return ::shelf::Status::invalid_argument("bad weight dtype", __FILE__, __LINE__);
    const bool in_b = input_dtype == T::BF16;
    const bool out_b = output_dtype == T::BF16;
    const bool pf = weight_layout == L::PER_FEATURE;
    const bool pg = weight_layout == L::PER_GROUP;
    const bool none = weight_layout == L::NONE;
    const bool op = weight_mode == M::ONE_PLUS;
#define EW_LAUNCH(I, O, W, WL, WM) \
    launch_one<I, O, W, WL, WM>(input, weight, output, rows, features, group_size, \
                                input_row_stride, output_row_stride, eps, stream)
    if (in_b && out_b) {
        if (pf)
            return op ? EW_LAUNCH(T::BF16, T::BF16, T::BF16, L::PER_FEATURE, M::ONE_PLUS)
                      : EW_LAUNCH(T::BF16, T::BF16, T::BF16, L::PER_FEATURE, M::DIRECT);
        if (pg)
            return op ? EW_LAUNCH(T::BF16, T::BF16, T::BF16, L::PER_GROUP, M::ONE_PLUS)
                      : EW_LAUNCH(T::BF16, T::BF16, T::BF16, L::PER_GROUP, M::DIRECT);
        if (none) return EW_LAUNCH(T::BF16, T::BF16, T::BF16, L::NONE, M::ONE_PLUS);
        return ::shelf::Status::invalid_argument("unsupported rmsnorm combo", __FILE__, __LINE__);
    }
    if (in_b && !out_b) {
        if (pf)
            return op ? EW_LAUNCH(T::BF16, T::F32, T::BF16, L::PER_FEATURE, M::ONE_PLUS)
                      : EW_LAUNCH(T::BF16, T::F32, T::BF16, L::PER_FEATURE, M::DIRECT);
        if (pg)
            return op ? EW_LAUNCH(T::BF16, T::F32, T::BF16, L::PER_GROUP, M::ONE_PLUS)
                      : EW_LAUNCH(T::BF16, T::F32, T::BF16, L::PER_GROUP, M::DIRECT);
        return ::shelf::Status::invalid_argument("unsupported rmsnorm combo", __FILE__, __LINE__);
    }
    if (!in_b && out_b) {
        return ::shelf::Status::invalid_argument("unsupported rmsnorm combo", __FILE__, __LINE__);
    }
    if (pf)
        return op ? EW_LAUNCH(T::F32, T::F32, T::BF16, L::PER_FEATURE, M::ONE_PLUS)
                  : EW_LAUNCH(T::F32, T::F32, T::BF16, L::PER_FEATURE, M::DIRECT);
    if (pg)
        return op ? EW_LAUNCH(T::F32, T::F32, T::BF16, L::PER_GROUP, M::ONE_PLUS)
                  : EW_LAUNCH(T::F32, T::F32, T::BF16, L::PER_GROUP, M::DIRECT);
    if (none) return EW_LAUNCH(T::F32, T::F32, T::BF16, L::NONE, M::DIRECT);
    return ::shelf::Status::invalid_argument("unsupported rmsnorm combo", __FILE__, __LINE__);
#undef EW_LAUNCH
}

} // namespace shelf::syclk
