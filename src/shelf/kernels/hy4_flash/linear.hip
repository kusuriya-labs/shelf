#include <shelf/kernels/hy4_flash/linear.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <shelf/quantization/exl3/exl3_int8.h>

namespace {
using namespace shelf::kernel::exl3_detail;
using namespace shelf::kernel::exl3_detail::mul24;

__global__ void rotate_input(const uint16_t* input, const uint16_t* suh, int8_t* codes,
                             float* scales, int width, int padded) {
    const unsigned lane = threadIdx.x;
    const int base = blockIdx.x * 128 + lane * 4;
    float v[4];
    for (int i = 0; i < 4; ++i)
        v[i] = base + i < width ? bf16_bits_to_f32(input[size_t(blockIdx.y) * width + base + i])
            * f16_bits_to_f32(suh[base + i]) : 0.0f;
    fwht128(v[0], v[1], v[2], v[3], lane);
    float peak = 0;
    for (float& x : v) {
        x *= kInvSqrt128;
        peak = fmaxf(peak, fabsf(x));
    }
    for (int mask = 16; mask; mask /= 2)
        peak = fmaxf(peak, __shfl_xor(peak, mask, 32));
    const float scale = peak / 127.0f;
    if (lane == 0) scales[size_t(blockIdx.y) * (padded / 128) + blockIdx.x] = scale;
    for (int i = 0; i < 4; ++i)
        codes[size_t(blockIdx.y) * padded + base + i] = scale ? int8_t(rintf(v[i] / scale)) : 0;
}

template<int Bits>
__global__ void multiply(const int8_t* codes, const float* scales, const uint32_t* trellis,
                         float* partial, int k, int n) {
    const unsigned col = threadIdx.x % 16;
    const unsigned worker = threadIdx.x / 16;
    const int row = blockIdx.y;
    __shared__ float sums[8][16];
    float acc = 0;
    constexpr unsigned words = Bits * 8;
    for (unsigned kt = worker; kt < unsigned(k / 16); kt += 8) {
        const uint32_t* tile = trellis + (size_t(kt) * (n / 16) + blockIdx.x) * words;
        int dot = 0;
        for (unsigned j = 0; j < 16; ++j) {
            const unsigned t = j / 8 * 128 + col * 8 + j % 8;
            const unsigned bit = ((t + 1) * Bits + 256 * Bits - 16) % (256 * Bits);
            const uint64_t pair = uint64_t(tile[bit / 32]) << 32 | tile[(bit / 32 + 1) % words];
            const uint32_t state = (pair >> (48 - bit % 32)) & 65535;
            const uint32_t x = state * shelf::quantization::exl3::kExl3Mul24Multiplier;
            const int bytes = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24);
            dot += ((bytes - 509) >> 2) * codes[size_t(row) * k + kt * 16 + j];
        }
        acc += dot * scales[size_t(row) * (k / 128) + kt / 8];
    }
    sums[worker][col] = acc;
    __syncthreads();
    if (worker == 0) {
        for (unsigned i = 1; i < 8; ++i) acc += sums[i][col];
        partial[size_t(row) * n + blockIdx.x * 16 + col] = acc * shelf::quantization::exl3::kExl3Int8GridStep;
    }
}

__global__ void rotate_output(const float* partial, const uint16_t* svh, uint16_t* output,
                              int width, int padded) {
    const int base = blockIdx.x * 128 + threadIdx.x * 4;
    float v[4];
    for (int i = 0; i < 4; ++i) v[i] = partial[size_t(blockIdx.y) * padded + base + i];
    fwht128(v[0], v[1], v[2], v[3], threadIdx.x);
    for (int i = 0; i < 4 && base + i < width; ++i)
        output[size_t(blockIdx.y) * width + base + i] =
            f32_to_bf16_rne_device(v[i] * kInvSqrt128 * f16_bits_to_f32(svh[base + i]));
}
}

extern "C" hipError_t shelf_hy4_linear(
    const uint16_t* input, const uint16_t* suh, const uint16_t* svh,
    const uint32_t* trellis, int8_t* codes, float* scales, float* partial,
    uint16_t* output, int rows, int input_width, int output_width,
    int padded_input, int padded_output, int bits, hipStream_t stream) {
    if (!input || !suh || !svh || !trellis || !codes || !scales || !partial || !output ||
        rows < 1 || rows > 65535 || input_width < 1 || output_width < 1 ||
        padded_input < input_width || padded_output < output_width || padded_input % 128 ||
        padded_output % 128 || (bits != 4 && bits != 6))
        return hipErrorInvalidValue;
    rotate_input<<<dim3(padded_input / 128, rows), 32, 0, stream>>>(
        input, suh, codes, scales, input_width, padded_input);
    auto status = hipGetLastError();
    if (status != hipSuccess) return status;
    const dim3 grid(padded_output / 16, rows);
    if (bits == 4)
        multiply<4><<<grid, 128, 0, stream>>>(codes, scales, trellis, partial, padded_input, padded_output);
    else
        multiply<6><<<grid, 128, 0, stream>>>(codes, scales, trellis, partial, padded_input, padded_output);
    status = hipGetLastError();
    if (status != hipSuccess) return status;
    rotate_output<<<dim3(padded_output / 128, rows), 32, 0, stream>>>(
        partial, svh, output, output_width, padded_output);
    return hipGetLastError();
}
