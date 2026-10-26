// vendored from PhaseShift (own work, CopenDeCamp); namespace/type adapted only.
// (SYCL port of backend/hip kernels/gemm/linear_bf16.hip.)
#include <shelf/backend/sycl/kernels/gemm/linear_bf16.h>
#include <shelf/backend/sycl/kernels/gemm/linear_bf16_config.h>
#include <shelf/backend/sycl/kernels/gemm/detail/linear_bf16_device.h>
#include <shelf/backend/sycl/kernels/detail/sycl_util.h>
#include <shelf/foundation/numeric.hpp>

#include <cstdint>
#include <cstddef>

namespace shelf::syclk {

// Device helpers live in the private header above; launchers stay here.
using namespace gemm_detail;

namespace {

constexpr uint32_t kExactRowsThreads = 256u;
static_assert(kExactRowsThreads == kBf16GemmExactRowsThreads);
static_assert(kTileOut == kBf16WmmaOutTile);
static_assert(kTileRow == kBf16WmmaRowTile);
static_assert(kWideRowTiles == kBf16WmmaWideRowTiles);
static_assert(kThreadsPerBlock == kBf16WmmaThreads);
static_assert(kSplitThreadsPerBlock == kBf16SplitKWmmaThreads);

// one launch of the templated ExactRows kernel (rows <= 16)
template <uint32_t kRows, bool kDot2>
::shelf::Status launch_exact_rows_impl(
    const bf16_t* weight, const bf16_t* input, void* output,
    Bf16GemmOutputDType output_dtype, uint32_t out_features, uint32_t k,
    uint32_t input_row_stride, uint32_t output_row_stride,
    const ::shelf::sycl::Stream& stream) {
    if (!::shelf::syclk::wave32_supported(stream.queue()))
        return ::shelf::Status::unsupported(
            "gemm exact-rows requires wave32 sub-groups", __FILE__, __LINE__);
    const uint32_t gx = bf16_exact_rows_grid_x(out_features);
    const uint32_t od = static_cast<uint32_t>(output_dtype);
    const bf16* w = reinterpret_cast<const bf16*>(weight);
    const bf16* x = reinterpret_cast<const bf16*>(input);
    ::sycl::queue& q = stream.queue();
    return ::shelf::syclk::sycl_submit(q, "sycl-submit", __FILE__, __LINE__, [&](::sycl::handler& cgh) {
        cgh.parallel_for(
            ::sycl::nd_range<1>{::sycl::range<1>{gx * kExactRowsThreads},
                                ::sycl::range<1>{kExactRowsThreads}},
            [=](::sycl::nd_item<1> it) {
                gemm_bf16_exact_rows_body<kRows, kDot2>(
                    it, w, x, output, od, out_features, k, input_row_stride,
                    output_row_stride);
            });
    });
}

::shelf::Status launch_exact_rows(
    const bf16_t* weight, const bf16_t* input, void* output,
    Bf16GemmOutputDType output_dtype, uint32_t rows, uint32_t out_features,
    uint32_t k, uint32_t input_row_stride, uint32_t output_row_stride,
    const ::shelf::sycl::Stream& stream) {
    if (rows > kBf16GemmExactRowsKernelMax) {
        const size_t output_elem = output_dtype == Bf16GemmOutputDType::BF16
                                       ? sizeof(shelf::bf16_t) : sizeof(float);
        for (uint32_t base = 0u; base < rows; base += kBf16GemmExactRowsKernelMax) {
            const uint32_t chunk =
                (rows - base) < kBf16GemmExactRowsKernelMax ? (rows - base)
                                                            : kBf16GemmExactRowsKernelMax;
            SHELF_RETURN_IF_ERROR(launch_exact_rows(
                weight, input + static_cast<size_t>(base) * input_row_stride,
                static_cast<char*>(output) +
                    static_cast<size_t>(base) * output_row_stride * output_elem,
                output_dtype, chunk, out_features, k, input_row_stride,
                output_row_stride, stream));
        }
        return ::shelf::Status::make_ok();
    }
    switch (rows) {
        case 1u:  return launch_exact_rows_impl<1u, false>(weight, input, output, output_dtype, out_features, k, input_row_stride, output_row_stride, stream);
        case 2u:  return launch_exact_rows_impl<2u, false>(weight, input, output, output_dtype, out_features, k, input_row_stride, output_row_stride, stream);
        case 3u:  return launch_exact_rows_impl<3u, false>(weight, input, output, output_dtype, out_features, k, input_row_stride, output_row_stride, stream);
        case 4u:  return launch_exact_rows_impl<4u, false>(weight, input, output, output_dtype, out_features, k, input_row_stride, output_row_stride, stream);
        case 5u:  return launch_exact_rows_impl<5u, false>(weight, input, output, output_dtype, out_features, k, input_row_stride, output_row_stride, stream);
        case 6u:  return launch_exact_rows_impl<6u, false>(weight, input, output, output_dtype, out_features, k, input_row_stride, output_row_stride, stream);
        case 7u:  return launch_exact_rows_impl<7u, false>(weight, input, output, output_dtype, out_features, k, input_row_stride, output_row_stride, stream);
        case 8u:  return launch_exact_rows_impl<8u, false>(weight, input, output, output_dtype, out_features, k, input_row_stride, output_row_stride, stream);
        case 9u:  return launch_exact_rows_impl<9u, false>(weight, input, output, output_dtype, out_features, k, input_row_stride, output_row_stride, stream);
        case 10u: return launch_exact_rows_impl<10u, false>(weight, input, output, output_dtype, out_features, k, input_row_stride, output_row_stride, stream);
        case 11u: return launch_exact_rows_impl<11u, false>(weight, input, output, output_dtype, out_features, k, input_row_stride, output_row_stride, stream);
        case 12u: return launch_exact_rows_impl<12u, false>(weight, input, output, output_dtype, out_features, k, input_row_stride, output_row_stride, stream);
        case 13u: return launch_exact_rows_impl<13u, false>(weight, input, output, output_dtype, out_features, k, input_row_stride, output_row_stride, stream);
        case 14u: return launch_exact_rows_impl<14u, false>(weight, input, output, output_dtype, out_features, k, input_row_stride, output_row_stride, stream);
        case 15u: return launch_exact_rows_impl<15u, false>(weight, input, output, output_dtype, out_features, k, input_row_stride, output_row_stride, stream);
        case 16u: return launch_exact_rows_impl<16u, false>(weight, input, output, output_dtype, out_features, k, input_row_stride, output_row_stride, stream);
        default:  return ::shelf::Status::unsupported("bad exact-rows chunk", __FILE__, __LINE__);
    }
}

::shelf::Status launch_exact_rows_dot2(
    const bf16_t* weight, const bf16_t* input, void* output,
    Bf16GemmOutputDType output_dtype, uint32_t rows, uint32_t out_features,
    uint32_t k, uint32_t input_row_stride, uint32_t output_row_stride,
    const ::shelf::sycl::Stream& stream) {
    if (rows > kBf16GemmExactRowsKernelMax) {
        const size_t output_elem = output_dtype == Bf16GemmOutputDType::BF16
                                       ? sizeof(shelf::bf16_t) : sizeof(float);
        for (uint32_t base = 0u; base < rows; base += kBf16GemmExactRowsKernelMax) {
            const uint32_t chunk =
                (rows - base) < kBf16GemmExactRowsKernelMax ? (rows - base)
                                                            : kBf16GemmExactRowsKernelMax;
            const bf16_t* chunk_input = input + static_cast<size_t>(base) * input_row_stride;
            char* chunk_output = static_cast<char*>(output) +
                static_cast<size_t>(base) * output_row_stride * output_elem;
            if (chunk == 1u) {
                SHELF_RETURN_IF_ERROR(launch_exact_rows(
                    weight, chunk_input, chunk_output, output_dtype, chunk,
                    out_features, k, input_row_stride, output_row_stride, stream));
                continue;
            }
            SHELF_RETURN_IF_ERROR(launch_exact_rows_dot2(
                weight, chunk_input, chunk_output, output_dtype, chunk,
                out_features, k, input_row_stride, output_row_stride, stream));
        }
        return ::shelf::Status::make_ok();
    }
    // The amdgcn fdot2 builtin has no portable equivalent (see header note);
    // the scalar body accumulates in the identical ascending order, so rows
    // dispatch to the same template instantiations with kDot2=true (a no-op
    // selector kept for config parity).
    switch (rows) {
        case 1u:  return launch_exact_rows_impl<1u, true>(weight, input, output, output_dtype, out_features, k, input_row_stride, output_row_stride, stream);
        case 2u:  return launch_exact_rows_impl<2u, true>(weight, input, output, output_dtype, out_features, k, input_row_stride, output_row_stride, stream);
        case 3u:  return launch_exact_rows_impl<3u, true>(weight, input, output, output_dtype, out_features, k, input_row_stride, output_row_stride, stream);
        case 4u:  return launch_exact_rows_impl<4u, true>(weight, input, output, output_dtype, out_features, k, input_row_stride, output_row_stride, stream);
        case 5u:  return launch_exact_rows_impl<5u, true>(weight, input, output, output_dtype, out_features, k, input_row_stride, output_row_stride, stream);
        case 6u:  return launch_exact_rows_impl<6u, true>(weight, input, output, output_dtype, out_features, k, input_row_stride, output_row_stride, stream);
        case 7u:  return launch_exact_rows_impl<7u, true>(weight, input, output, output_dtype, out_features, k, input_row_stride, output_row_stride, stream);
        case 8u:  return launch_exact_rows_impl<8u, true>(weight, input, output, output_dtype, out_features, k, input_row_stride, output_row_stride, stream);
        case 9u:  return launch_exact_rows_impl<9u, true>(weight, input, output, output_dtype, out_features, k, input_row_stride, output_row_stride, stream);
        case 10u: return launch_exact_rows_impl<10u, true>(weight, input, output, output_dtype, out_features, k, input_row_stride, output_row_stride, stream);
        case 11u: return launch_exact_rows_impl<11u, true>(weight, input, output, output_dtype, out_features, k, input_row_stride, output_row_stride, stream);
        case 12u: return launch_exact_rows_impl<12u, true>(weight, input, output, output_dtype, out_features, k, input_row_stride, output_row_stride, stream);
        case 13u: return launch_exact_rows_impl<13u, true>(weight, input, output, output_dtype, out_features, k, input_row_stride, output_row_stride, stream);
        case 14u: return launch_exact_rows_impl<14u, true>(weight, input, output, output_dtype, out_features, k, input_row_stride, output_row_stride, stream);
        case 15u: return launch_exact_rows_impl<15u, true>(weight, input, output, output_dtype, out_features, k, input_row_stride, output_row_stride, stream);
        case 16u: return launch_exact_rows_impl<16u, true>(weight, input, output, output_dtype, out_features, k, input_row_stride, output_row_stride, stream);
        default:  return ::shelf::Status::unsupported("bad exact-rows chunk", __FILE__, __LINE__);
    }
}

template <uint32_t kRowTiles>
::shelf::Status launch_wmma(
    const bf16_t* weight, const bf16_t* input, void* output,
    Bf16GemmOutputDType output_dtype, uint32_t rows, uint32_t out_features,
    uint32_t k, uint32_t input_row_stride, uint32_t output_row_stride,
    const ::shelf::sycl::Stream& stream) {
    const uint32_t gx = bf16_wmma_grid_x(out_features);
    const uint32_t gy = bf16_wmma_grid_y(rows, kRowTiles);
    const uint32_t od = static_cast<uint32_t>(output_dtype);
    const bf16* w = reinterpret_cast<const bf16*>(weight);
    const bf16* x = reinterpret_cast<const bf16*>(input);
    ::sycl::queue& q = stream.queue();
    return ::shelf::syclk::sycl_submit(q, "sycl-submit", __FILE__, __LINE__, [&](::sycl::handler& cgh) {
        cgh.parallel_for(
            ::sycl::nd_range<2>{::sycl::range<2>{gx * kThreadsPerBlock, gy},
                                ::sycl::range<2>{kThreadsPerBlock, 1}},
            [=](::sycl::nd_item<2> it) {
                bf16_wmma_body<kRowTiles>(it, w, x, output, od, rows,
                                          out_features, k, input_row_stride,
                                          output_row_stride);
            });
    });
}

::shelf::Status launch_splitk_wmma(
    const bf16_t* weight, const bf16_t* input, void* output,
    Bf16GemmOutputDType output_dtype, uint32_t rows, uint32_t out_features,
    uint32_t k, uint32_t input_row_stride, uint32_t output_row_stride,
    const ::shelf::sycl::Stream& stream) {
    if (!::shelf::syclk::wave32_supported(stream.queue()))
        return ::shelf::Status::unsupported(
            "gemm split-k requires wave32 sub-groups", __FILE__, __LINE__);
    const uint32_t gx = bf16_wmma_grid_x(out_features);
    const uint32_t gy = bf16_wmma_grid_y(rows, 1u);
    const uint32_t od = static_cast<uint32_t>(output_dtype);
    const bf16* w = reinterpret_cast<const bf16*>(weight);
    const bf16* x = reinterpret_cast<const bf16*>(input);
    ::sycl::queue& q = stream.queue();
    return ::shelf::syclk::sycl_submit(q, "sycl-submit", __FILE__, __LINE__, [&](::sycl::handler& cgh) {
        ::sycl::local_accessor<float, 1> red(
            ::sycl::range<1>{kSplitWarps * 8u * 32u}, cgh);
        cgh.parallel_for(
            ::sycl::nd_range<2>{::sycl::range<2>{gx, gy},
                                ::sycl::range<2>{kSplitThreadsPerBlock, 1}},
            [=](::sycl::nd_item<2> it) {
                bf16_splitk_wmma_body(it, red.get_pointer().get(), w, x, output,
                                      od, rows, out_features, k, input_row_stride,
                                      output_row_stride);
            });
    });
}

} // namespace

::shelf::Status launch_gemm_bf16(
    const Bf16GemmConfig& config,
    const bf16_t* weight,
    const bf16_t* input,
    void* output,
    Bf16GemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k,
    uint32_t input_row_stride,
    uint32_t output_row_stride,
    const ::shelf::sycl::Stream& stream) {
    if (rows == 0 || out_features == 0 || k == 0)
        return ::shelf::Status::make_ok();

    switch (config.id) {
    case Bf16GemmConfigId::ExactRows: {
        if (config.exact_rows == 0u || config.exact_rows > kBf16GemmExactRowsMax ||
            config.exact_rows != rows)
            return ::shelf::Status::unsupported("exact-rows config mismatch",
                                                __FILE__, __LINE__);
        return launch_exact_rows(weight, input, output, output_dtype,
                                 config.exact_rows, out_features, k,
                                 input_row_stride, output_row_stride, stream);
    }
    case Bf16GemmConfigId::ExactRowsDot2: {
        if (config.exact_rows == 0u || config.exact_rows > kBf16GemmExactRowsMax ||
            config.exact_rows != rows)
            return ::shelf::Status::unsupported("exact-rows config mismatch",
                                                __FILE__, __LINE__);
        return launch_exact_rows_dot2(weight, input, output, output_dtype,
                                      config.exact_rows, out_features, k,
                                      input_row_stride, output_row_stride, stream);
    }
    case Bf16GemmConfigId::WmmaKPartition:
        return launch_splitk_wmma(weight, input, output, output_dtype, rows,
                                  out_features, k, input_row_stride,
                                  output_row_stride, stream);
    case Bf16GemmConfigId::WmmaWide:
        return launch_wmma<kWideRowTiles>(weight, input, output, output_dtype,
                                          rows, out_features, k, input_row_stride,
                                          output_row_stride, stream);
    case Bf16GemmConfigId::Wmma:
    default:
        return launch_wmma<1u>(weight, input, output, output_dtype, rows,
                               out_features, k, input_row_stride,
                               output_row_stride, stream);
    }
}

} // namespace shelf::syclk
