#include <shelf/kernels/hy4_flash/linear.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <shelf/quantization/exl3/exl3_int8.h>

hipError_t shelf_hy4_wmma(const int8_t *, const float *, const uint32_t *, float *, int, int, int, int,
                          hipStream_t);

namespace {
using namespace shelf::kernel::exl3_detail;
using namespace shelf::kernel::exl3_detail::mul24;

__global__ void rotate_input(const uint16_t *input, const uint16_t *suh, int8_t *codes, float *scales,
                             int width, int padded, const Hy4ExpertProjection *group = nullptr,
                             int stride = 0) {
    if (group)
        suh = group[blockIdx.y * stride].suh;
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

template <int Bits> __device__ __forceinline__ uint32_t unpack_four(const uint32_t *tile, unsigned t) {
    constexpr int words = Bits * 8;
    unsigned bit = ((t + 1) * Bits + 256 * Bits - 16) % (256 * Bits);
    unsigned hi = tile[bit / 32], lo = tile[(bit / 32 + 1) % words];
    unsigned packed = 0;
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
        unsigned end = (bit % 32) + j * Bits + 16;
        unsigned state = end <= 32 ? (hi >> (32 - end)) : __builtin_amdgcn_alignbit(hi, lo, 64 - end);
        unsigned value = __umul24(state & 65535, 0xdcd12du);
        unsigned sum = __builtin_amdgcn_udot4(value, 0x01010101u, 3u, false);
        packed |= ((sum >> 2) & 255u) << (8 * j);
    }
    return packed ^ 0x80808080u;
}
__device__ __forceinline__ int dot4(uint32_t a, uint32_t b, int acc) {
#if defined(__gfx1030__)
    return __builtin_amdgcn_sdot4(int32_t(a), int32_t(b), acc, false);
#else
    return __builtin_amdgcn_sudot4(true, a, true, b, acc, false);
#endif
}

template <int Bits, bool Packed = false>
__global__ void multiply(const int8_t *codes, const float *scales, const uint32_t *trellis, float *partial,
                         int k, int n, const Hy4ExpertProjection *group = nullptr, int stride = 0) {
    if (group)
        trellis = group[blockIdx.y * stride].trellis;
    const unsigned col = threadIdx.x % 16;
    const unsigned worker = threadIdx.x / 16;
    const int row = blockIdx.y;
    __shared__ float sums[8][16];
    float acc = 0;
    constexpr unsigned words = Bits * 8;
    for (unsigned kt = worker; kt < unsigned(k / 16); kt += 8) {
        const uint32_t *tile = trellis + (size_t(kt) * (n / 16) + blockIdx.x) * words;
        int dot = 0;
        if constexpr (Packed) {
#pragma unroll
            for (unsigned g = 0; g < 4; ++g) {
                uint32_t a;
                __builtin_memcpy(&a, codes + size_t(row) * k + kt * 16 + g * 4, 4);
                unsigned t = g / 2 * 128 + col * 8 + g % 2 * 4;
                dot = dot4(a, unpack_four<Bits>(tile, t), dot);
            }
        } else
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
        for (unsigned i = 1; i < 8; ++i)
            acc += sums[i][col];
        partial[size_t(row) * n + blockIdx.x * 16 + col] = acc * shelf::quantization::exl3::kExl3Int8GridStep;
    }
}

__global__ void rotate_output(const float *partial, const uint16_t *svh, uint16_t *output, int width,
                              int padded, const Hy4ExpertProjection *group = nullptr, int stride = 0,
                              float *logits = nullptr) {
    if (group)
        svh = group[blockIdx.y * stride].svh;
    const int base = blockIdx.x * 128 + threadIdx.x * 4;
    float v[4];
    for (int i = 0; i < 4; ++i)
        v[i] = partial[size_t(blockIdx.y) * padded + base + i];
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
                                       int wmma) {
    if (!input || !suh || !svh || !trellis || !codes || !scales || !partial || !output || rows < 1 ||
        rows > 65535 || input_width < 1 || output_width < 1 || padded_input < input_width ||
        padded_output < output_width || padded_input % 128 || padded_output % 128 || (bits != 4 && bits != 6))
        return hipErrorInvalidValue;
    rotate_input<<<dim3(padded_input / 128, rows), 32, 0, stream>>>(input, suh, codes, scales, input_width,
                                                                    padded_input);
    auto status = hipGetLastError();
    if (status != hipSuccess)
        return status;
    const dim3 grid(padded_output / 16, rows);
    if (wmma) {
        status =
            shelf_hy4_wmma(codes, scales, trellis, partial, rows, padded_input, padded_output, bits, stream);
        if (status != hipSuccess)
            return status;
    } else if (bits == 4)
        multiply<4><<<grid, 128, 0, stream>>>(codes, scales, trellis, partial, padded_input, padded_output);
    else
        multiply<6><<<grid, 128, 0, stream>>>(codes, scales, trellis, partial, padded_input, padded_output);
    status = hipGetLastError();
    if (status != hipSuccess)
        return status;
    rotate_output<<<dim3(padded_output / 128, rows), 32, 0, stream>>>(partial, svh, output, output_width,
                                                                      padded_output);
    return hipGetLastError();
}

extern "C" hipError_t shelf_hy4_expert_linear(const uint16_t *x, const Hy4ExpertProjection *matrices,
                                              int stride, int8_t *codes, float *scales, float *partial,
                                              uint16_t *y, int rows) {
    if (!x || !matrices || !codes || !scales || !partial || !y || rows < 1 || rows > 65535 || stride < 1)
        return hipErrorInvalidValue;
    rotate_input<<<dim3(12, rows), 32>>>(x, nullptr, codes, scales, 1536, 1536, matrices, stride);
    auto status = hipGetLastError();
    if (status != hipSuccess)
        return status;
    static const bool use_dot = [] {
        auto v = std::getenv("SHELF_HY4_DOT4");
        return v && v[0] == '1' && !v[1];
    }();
    if (use_dot)
        multiply<4, true>
            <<<dim3(96, rows), 128>>>(codes, scales, nullptr, partial, 1536, 1536, matrices, stride);
    else
        multiply<4><<<dim3(96, rows), 128>>>(codes, scales, nullptr, partial, 1536, 1536, matrices, stride);
    status = hipGetLastError();
    if (status != hipSuccess)
        return status;
    rotate_output<<<dim3(12, rows), 32>>>(partial, nullptr, y, 1536, 1536, matrices, stride);
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
    multiply<6><<<dim3(7552, rows), 128>>>(codes, scales, trellis, partial, 3072, 120832);
    status = hipGetLastError();
    if (status != hipSuccess)
        return status;
    rotate_output<<<dim3(944, rows), 32>>>(partial, svh, nullptr, 120832, 120832, nullptr, 0, logits);
    return hipGetLastError();
}
