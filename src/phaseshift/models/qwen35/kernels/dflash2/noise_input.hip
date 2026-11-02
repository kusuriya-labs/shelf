#include <phaseshift/models/qwen35/kernels/dflash2/noise_input.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::kernel {

constexpr uint32_t kNoiseInputThreads = 256u;

namespace {

__global__ void dflash2_noise_token_ids_kernel(
    int32_t anchor_token,
    int32_t mask_token,
    int32_t* output,
    uint32_t block_rows) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= block_rows) return;
    output[row] = row == 0u ? anchor_token : mask_token;
}

__global__ void dflash2_noise_token_ids_batch_kernel(
    const DFlash2RequestDescriptor* requests,
    int32_t mask_token,
    int32_t* token_ids,
    uint32_t* row_has_prev) {
    const DFlash2RequestDescriptor& request = requests[blockIdx.y];
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= request.block_rows) return;
    const uint32_t index = request.row_begin + row;
    token_ids[index] = row == 0u ? request.anchor_token : mask_token;
    row_has_prev[index] = row == 0u ? 0u : 1u;
}

}  // namespace

hipError_t launch_dflash2_noise_token_ids(
    int32_t anchor_token,
    int32_t mask_token,
    int32_t* output,
    uint32_t block_rows,
    hipStream_t stream) {
    if (output == nullptr) {
        return hipErrorInvalidValue;
    }
    if (block_rows == 0u) {
        return hipSuccess;
    }
    const uint32_t blocks = (block_rows + kNoiseInputThreads - 1u) / kNoiseInputThreads;
    dflash2_noise_token_ids_kernel<<<blocks, kNoiseInputThreads, 0, stream>>>(
        anchor_token, mask_token, output, block_rows);
    return hipGetLastError();
}

hipError_t launch_dflash2_noise_token_ids_batch(
    const DFlash2RequestDescriptor* requests,
    uint32_t request_count,
    uint32_t max_block_rows,
    int32_t mask_token,
    int32_t* token_ids,
    uint32_t* row_has_prev,
    hipStream_t stream) {
    if (requests == nullptr || token_ids == nullptr || row_has_prev == nullptr) {
        return hipErrorInvalidValue;
    }
    if (request_count == 0u || max_block_rows == 0u) {
        return hipErrorInvalidValue;
    }
    const uint32_t blocks = (max_block_rows + kNoiseInputThreads - 1u) / kNoiseInputThreads;
    dim3 grid(blocks, request_count);
    dflash2_noise_token_ids_batch_kernel<<<grid, kNoiseInputThreads, 0, stream>>>(
        requests, mask_token, token_ids, row_has_prev);
    return hipGetLastError();
}

}  // namespace ps::kernel
