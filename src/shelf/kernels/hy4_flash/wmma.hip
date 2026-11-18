#include <shelf/kernels/fused/exl3/group_body.h>

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
