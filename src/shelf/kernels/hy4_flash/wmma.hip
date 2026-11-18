#include <shelf/kernels/fused/exl3/group_body.h>
#include <shelf/kernels/hy4_flash/linear.h>

namespace {
template <int Bits, int Tiles>
__global__ void batch(const int8_t *codes, const float *scales, const uint32_t *weight, float *partial,
                      int rows, int k, int n, int blocks) {
    shelf::kernel::exl3_detail::mul24::group_body<Bits, Tiles, false, 4, false, true>(
        codes, scales, k, k / 128, weight, partial, rows, nullptr, n, k, k / 128, blocks, blockIdx.x, 0);
}
template <int Bits>
void launch(const int8_t *codes, const float *scales, const uint32_t *weight, float *partial, int rows, int k,
            int n, hipStream_t stream) {
    if (rows >= 32) {
        int blocks = (rows + 31) / 32;
        batch<Bits, 2>
            <<<blocks *(n / 64), 128, 0, stream>>>(codes, scales, weight, partial, rows, k, n, blocks);
    } else {
        int blocks = (rows + 15) / 16;
        batch<Bits, 1>
            <<<blocks *(n / 64), 128, 0, stream>>>(codes, scales, weight, partial, rows, k, n, blocks);
    }
}
}
hipError_t shelf_hy4_wmma(const int8_t *codes, const float *scales, const uint32_t *weight, float *partial,
                          int rows, int k, int n, int bits, hipStream_t stream) {
    if (bits == 4)
        launch<4>(codes, scales, weight, partial, rows, k, n, stream);
    else if (bits == 6)
        launch<6>(codes, scales, weight, partial, rows, k, n, stream);
    else
        return hipErrorInvalidValue;
    return hipGetLastError();
}

namespace {
__global__ void expert_batch(const int8_t *codes, const float *scales, const Hy4ExpertProjection *matrices,
                             const Hy4Route *routes, const Hy4ExpertBatch *batches, float *partial) {
    auto batch = batches[blockIdx.y];
    size_t first = batch.first;
    auto weight = matrices[3 * routes[first].expert].trellis;
    shelf::kernel::exl3_detail::mul24::group_body<4, 1, false, 4, false, true>(
        codes + first * 1536, scales + first * 12, 1536, 12, weight, partial + first * 1536, batch.count,
        nullptr, 1536, 1536, 12, 1, blockIdx.x, 0);
}
}
hipError_t shelf_hy4_wmma_experts(const int8_t *codes, const float *scales, const Hy4ExpertProjection *matrices,
                                  const Hy4Route *routes, const Hy4ExpertBatch *batches, int count,
                                  float *partial) {
    expert_batch<<<dim3(24, count), 128>>>(codes, scales, matrices, routes, batches, partial);
    return hipGetLastError();
}
