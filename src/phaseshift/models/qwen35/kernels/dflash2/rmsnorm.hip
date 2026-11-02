#include <phaseshift/models/qwen35/kernels/dflash2/rmsnorm.h>
#include "detail/rmsnorm_device.h"
#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::kernel {

using namespace ::ps::kernel::detail;

namespace {

__global__ void dflash2_rmsnorm_kernel(
    const bf16* input,
    const bf16* weight,
    bf16* output,
    uint32_t rows,
    uint32_t features,
    uint32_t group_size,
    float eps) {
    __shared__ float scratch[kDflash2NormThreads];
    const uint32_t row = blockIdx.y;
    const uint32_t group = blockIdx.x;
    if (row >= rows || group >= features / group_size) return;

    const uint32_t base = group * group_size;
    const bf16* x = input + static_cast<std::size_t>(row) * features + base;
    bf16* out = output + static_cast<std::size_t>(row) * features + base;
    const uint32_t weight_index_base = (group_size == features) ? base : 0u;
    dflash2_rmsnorm_group_body(x, weight, out, group_size, weight_index_base, eps, scratch);
}

}  // namespace

hipError_t launch_dflash2_rmsnorm_direct_bf16(
    const bf16_t* input,
    const bf16_t* weight,
    bf16_t* output,
    uint32_t rows,
    uint32_t features,
    uint32_t group_size,
    float eps,
    hipStream_t stream) {
    if (input == nullptr || weight == nullptr || output == nullptr) {
        return hipErrorInvalidValue;
    }
    if (rows == 0u || features == 0u || group_size == 0u || features % group_size != 0u) {
        return hipErrorInvalidValue;
    }
    dim3 grid(features / group_size, rows);
    dflash2_rmsnorm_kernel<<<grid, kDflash2NormThreads, 0, stream>>>(
        reinterpret_cast<const bf16*>(input), reinterpret_cast<const bf16*>(weight),
        reinterpret_cast<bf16*>(output), rows, features, group_size, eps);
    return hipGetLastError();
}

}  // namespace ps::kernel
