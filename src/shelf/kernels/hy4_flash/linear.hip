#include <shelf/kernels/hy4_flash/linear.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <shelf/quantization/exl3/exl3_int8.h>

hipError_t shelf_hy4_wmma(const int8_t *, const float *, const uint32_t *, float *, int, int, int, int,
                          hipStream_t, int);

hipError_t shelf_hy4_wmma_experts(const int8_t *, const float *, const Hy4ExpertProjection *, const Hy4Route *,
                                  const Hy4ExpertBatch *, int, float *, bool);
namespace {
using namespace shelf::kernel::exl3_detail;
using namespace shelf::kernel::exl3_detail::mul24;

__global__ void rotate_input(const uint16_t *input, const uint16_t *suh, int8_t *codes, float *scales,
                             int width, int padded, const Hy4ExpertProjection *group = nullptr, int stride = 0,
                             const Hy4Route *routes = nullptr) {
    if (group)
        suh = group[(routes ? routes[blockIdx.y].expert : blockIdx.y) * stride].suh;
    const unsigned lane = threadIdx.x;
    const int base = blockIdx.x * 128 + lane * 4;
    float v[4];
    for (int i = 0; i < 4; ++i)
        v[i] = base + i < width ? bf16_bits_to_f32(input[size_t(blockIdx.y) * width + base + i]) *
                                      f16_bits_to_f32(suh[base + i])
                                : 0.0f;
    fwht128(v[0], v[1], v[2], v[3], lane);
    float peak = 0;
    for (float &x : v) {
        x *= kInvSqrt128;
        peak = fmaxf(peak, fabsf(x));
    }
    for (int mask = 16; mask; mask /= 2)
        peak = fmaxf(peak, __shfl_xor(peak, mask, 32));
    const float scale = peak / 127.0f;
    if (lane == 0)
        scales[size_t(blockIdx.y) * (padded / 128) + blockIdx.x] = scale;
    for (int i = 0; i < 4; ++i)
        codes[size_t(blockIdx.y) * padded + base + i] = scale ? int8_t(rintf(v[i] / scale)) : 0;
}

__global__ void rotate_output(const float *partial, const uint16_t *svh, uint16_t *output, int width,
                              int padded, const Hy4ExpertProjection *group = nullptr, int stride = 0,
                              float *logits = nullptr, const Hy4Route *routes = nullptr, int splits = 1) {
    if (group)
        svh = group[(routes ? routes[blockIdx.y].expert : blockIdx.y) * stride].svh;
    const int base = blockIdx.x * 128 + threadIdx.x * 4;
    float v[4];
    for (int i = 0; i < 4; ++i)
        v[i] = partial[size_t(blockIdx.y) * padded + base + i];
    for (int s = 1; s < splits; ++s)
        for (int i = 0; i < 4; ++i)
            v[i] += partial[(size_t(s) * gridDim.y + blockIdx.y) * padded + base + i];
    fwht128(v[0], v[1], v[2], v[3], threadIdx.x);
    for (int i = 0; i < 4 && base + i < width; ++i) {
        float value = v[i] * kInvSqrt128 * f16_bits_to_f32(svh[base + i]);
        size_t offset = size_t(blockIdx.y) * width + base + i;
        if (logits)
            logits[offset] = value;
        else
            output[offset] = f32_to_bf16_rne_device(value);
    }
}
}

extern "C" hipError_t shelf_hy4_linear(const uint16_t *input, const uint16_t *suh, const uint16_t *svh,
                                       const uint32_t *trellis, int8_t *codes, float *scales, float *partial,
                                       uint16_t *output, int rows, int input_width, int output_width,
                                       int padded_input, int padded_output, int bits, hipStream_t stream,
                                       int splits) {
    if (!input || !suh || !svh || !trellis || !codes || !scales || !partial || !output || rows < 1 ||
        rows > 65535 || input_width < 1 || output_width < 1 || padded_input < input_width ||
        padded_output < output_width || padded_input % 128 || padded_output % 128 || (bits != 4 && bits != 6))
        return hipErrorInvalidValue;
    if (splits < 1 || splits > padded_input / 128)
        return hipErrorInvalidValue;
    rotate_input<<<dim3(padded_input / 128, rows), 32, 0, stream>>>(input, suh, codes, scales, input_width,
                                                                    padded_input);
    auto status = hipGetLastError();
    if (status != hipSuccess)
        return status;
    status = shelf_hy4_wmma(codes, scales, trellis, partial, rows, padded_input, padded_output, bits, stream,
                            splits);
    if (status != hipSuccess)
        return status;
    rotate_output<<<dim3(padded_output / 128, rows), 32, 0, stream>>>(
        partial, svh, output, output_width, padded_output, nullptr, 0, nullptr, nullptr, splits);
    return hipGetLastError();
}

extern "C" hipError_t shelf_hy4_expert_linear(const uint16_t *x, const Hy4ExpertProjection *matrices,
                                              int stride, int8_t *codes, float *scales, float *partial,
                                              uint16_t *y, int rows, const Hy4Route *routes,
                                              const Hy4ExpertBatch *batches, int num_batches, int mixed) {
    if (!x || !matrices || !codes || !scales || !partial || !y || rows < 1 || rows > 65535 || stride < 1)
        return hipErrorInvalidValue;
    if (!routes || !batches || num_batches < 1 || num_batches > rows || stride != 3)
        return hipErrorInvalidValue;
    rotate_input<<<dim3(12, rows), 32>>>(x, nullptr, codes, scales, 1536, 1536, matrices, stride, routes);
    auto status = hipGetLastError();
    if (status != hipSuccess)
        return status;
    status = shelf_hy4_wmma_experts(codes, scales, matrices, routes, batches, num_batches, partial, mixed);
    if (status != hipSuccess)
        return status;
    status = hipGetLastError();
    if (status != hipSuccess)
        return status;
    rotate_output<<<dim3(12, rows), 32>>>(partial, nullptr, y, 1536, 1536, matrices, stride, nullptr, routes);
    return hipGetLastError();
}

hipError_t shelf_hy4_head(const uint16_t *x, const uint16_t *suh, const uint16_t *svh, const uint32_t *trellis,
                          int8_t *codes, float *scales, float *partial, float *logits, int rows) {
    if (!x || !suh || !svh || !trellis || !codes || !scales || !partial || !logits || rows < 1 || rows > 6)
        return hipErrorInvalidValue;
    rotate_input<<<dim3(24, rows), 32>>>(x, suh, codes, scales, 3072, 3072);
    auto status = hipGetLastError();
    if (status != hipSuccess)
        return status;
    status = shelf_hy4_wmma(codes, scales, trellis, partial, rows, 3072, 120832, 6, nullptr, 1);
    if (status != hipSuccess)
        return status;
    rotate_output<<<dim3(944, rows), 32>>>(partial, svh, nullptr, 120832, 120832, nullptr, 0, logits);
    return hipGetLastError();
}
