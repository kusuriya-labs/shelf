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

template <bool Mixed>
__global__ void multiply(const int8_t *codes, const float *scales, const uint32_t *trellis, float *partial,
                         int k, int n, const Hy4ExpertProjection *group = nullptr, int stride = 0,
                         const Hy4Route *routes = nullptr) {
    const auto &matrix = group[(routes ? routes[blockIdx.y].expert : blockIdx.y) * stride];
    trellis = matrix.trellis;
    const unsigned bits = Mixed ? matrix.bits : 4;
    const unsigned col = threadIdx.x % 16;
    const unsigned worker = threadIdx.x / 16;
    const int row = blockIdx.y;
    __shared__ float sums[8][16];
    float acc = 0;
    const unsigned words = bits * 8;
    for (unsigned kt = worker; kt < unsigned(k / 16); kt += 8) {
        const uint32_t *tile = trellis + (size_t(kt) * (n / 16) + blockIdx.x) * words;
        int dot = 0;
#pragma unroll
        for (unsigned g = 0; g < 4; ++g) {
            uint32_t a;
            __builtin_memcpy(&a, codes + size_t(row) * k + kt * 16 + g * 4, 4);
            unsigned t = g / 2 * 128 + col * 8 + g % 2 * 4;
            dot = dot4(a, (bits == 3 ? unpack_four<3>(tile, t) : unpack_four<4>(tile, t)), dot);
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

template <bool Mixed>
__global__ void expert_batch16(const int8_t *codes, const float *scales, const Hy4ExpertProjection *matrices,
                               const Hy4Route *routes, const Hy4ExpertBatch *batches, float *partial) {
    __shared__ uint32_t b[32][16], a[16][33];
    const auto batch = batches[blockIdx.y];
    unsigned tid = threadIdx.x, col = tid % 16, row = tid / 16;
    auto matrix = matrices[3 * routes[batch.first].expert];
    auto weight = matrix.trellis;
    const unsigned bits = Mixed ? matrix.bits : 4;
    float acc = 0;
    for (int g = 0; g < 12; ++g) {
        if (tid < 128) {
            int seg = tid / 16;
            auto tile = weight + ((g * 8 + seg) * 96 + blockIdx.x) * (bits * 8);
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                unsigned t = j / 2 * 128 + col * 8 + j % 2 * 4;
                b[seg * 4 + j][col] = bits == 3 ? unpack_four<3>(tile, t) : unpack_four<4>(tile, t);
            }
        }
        for (unsigned j = tid; j < unsigned(batch.count * 32); j += 256)
            __builtin_memcpy(&a[j / 32][j % 32],
                             codes + size_t(batch.first + j / 32) * 1536 + g * 128 + (j % 32) * 4, 4);
        __syncthreads();
        if (row < unsigned(batch.count)) {
            int sum = 0;
#pragma unroll
            for (int j = 0; j < 32; ++j)
                sum = dot4(a[row][j], b[j][col], sum);
            acc += sum * scales[(batch.first + row) * 12 + g];
        }
        __syncthreads();
    }
    if (row < unsigned(batch.count))
        partial[(batch.first + row) * 1536 + blockIdx.x * 16 + col] =
            acc * shelf::quantization::exl3::kExl3Int8GridStep;
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
                                              const Hy4ExpertBatch *batches, int num_batches, int wmma,
                                              int mixed) {
    if (!x || !matrices || !codes || !scales || !partial || !y || rows < 1 || rows > 65535 || stride < 1)
        return hipErrorInvalidValue;
    if ((batches && (!routes || num_batches < 1 || num_batches > rows || stride != 3)) ||
        (!batches && num_batches))
        return hipErrorInvalidValue;
    rotate_input<<<dim3(12, rows), 32>>>(x, nullptr, codes, scales, 1536, 1536, matrices, stride, routes);
    auto status = hipGetLastError();
    if (status != hipSuccess)
        return status;
    if (batches && wmma) {
        status = shelf_hy4_wmma_experts(codes, scales, matrices, routes, batches, num_batches, partial, mixed);
        if (status != hipSuccess)
            return status;
    } else if (batches) {
        if (mixed)
            expert_batch16<true>
                <<<dim3(96, num_batches), 256>>>(codes, scales, matrices, routes, batches, partial);
        else
            expert_batch16<false>
                <<<dim3(96, num_batches), 256>>>(codes, scales, matrices, routes, batches, partial);
    } else {
        if (mixed)
            multiply<true><<<dim3(96, rows), 128>>>(codes, scales, nullptr, partial, 1536, 1536, matrices,
                                                    stride, routes);
        else
            multiply<false><<<dim3(96, rows), 128>>>(codes, scales, nullptr, partial, 1536, 1536, matrices,
                                                     stride, routes);
    }
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
