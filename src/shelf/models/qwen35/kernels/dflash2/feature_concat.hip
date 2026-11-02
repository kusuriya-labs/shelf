#include <shelf/models/qwen35/kernels/dflash2/feature_concat.h>
#include <shelf/kernels/optimized/detail/vector_io.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {

using namespace ::shelf::kernel::detail;
constexpr uint32_t kConcatThreads = 256u;

namespace {

__global__ void feature_concat_kernel(
    const bf16* tap0,
    const bf16* tap1,
    const bf16* tap2,
    const bf16* tap3,
    const bf16* tap4,
    uint32_t tap_features,
    uint32_t rows,
    bf16* output,
    uint32_t output_row_stride) {
    const uint32_t pair = blockIdx.y;
    const uint32_t row = pair / kDFlash2TargetTaps;
    const uint32_t tap = pair % kDFlash2TargetTaps;
    if (row >= rows) return;

    const bf16* src = (tap == 0u) ? tap0
                     : (tap == 1u) ? tap1
                     : (tap == 2u) ? tap2
                     : (tap == 3u) ? tap3
                                   : tap4;
    const bf16* src_row = src + static_cast<std::size_t>(row) * tap_features;
    bf16* dst_row = output + static_cast<std::size_t>(row) * output_row_stride +
                    static_cast<std::size_t>(tap) * tap_features;

    const uint32_t col0 = (blockIdx.x * blockDim.x + threadIdx.x) * kVec;
    if (col0 >= tap_features) return;
    if (col0 + kVec <= tap_features && aligned8(src_row + col0) && aligned8(dst_row + col0)) {
        store_bf16x4(dst_row + col0, load_bf16x4(src_row + col0));
        return;
    }
    for (uint32_t i = 0u; i < kVec; ++i) {
        const uint32_t c = col0 + i;
        if (c >= tap_features) break;
        dst_row[c] = src_row[c];
    }
}

}  // namespace

hipError_t launch_dflash2_feature_concat(
    const bf16_t* tap0,
    const bf16_t* tap1,
    const bf16_t* tap2,
    const bf16_t* tap3,
    const bf16_t* tap4,
    uint32_t tap_features,
    uint32_t rows,
    bf16_t* output,
    uint32_t output_row_stride,
    hipStream_t stream) {
    if (tap0 == nullptr || tap1 == nullptr || tap2 == nullptr || tap3 == nullptr ||
        tap4 == nullptr || output == nullptr) {
        return hipErrorInvalidValue;
    }
    if (tap_features == 0u || rows == 0u) {
        return hipSuccess;
    }
    if (output_row_stride < kDFlash2TargetTaps * tap_features) {
        return hipErrorInvalidValue;
    }
    const uint32_t blocks_x =
        (tap_features + kConcatThreads * kVec - 1u) / (kConcatThreads * kVec);
    dim3 grid(blocks_x, rows * kDFlash2TargetTaps);
    feature_concat_kernel<<<grid, kConcatThreads, 0, stream>>>(
        reinterpret_cast<const bf16*>(tap0),
        reinterpret_cast<const bf16*>(tap1),
        reinterpret_cast<const bf16*>(tap2),
        reinterpret_cast<const bf16*>(tap3),
        reinterpret_cast<const bf16*>(tap4),
        tap_features,
        rows,
        reinterpret_cast<bf16*>(output),
        output_row_stride);
    return hipGetLastError();
}

}
