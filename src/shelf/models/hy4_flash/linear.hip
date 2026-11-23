#include <shelf/models/hy4_flash/moe_graph_plan.h>
#include <shelf/kernels/optimized/linear/exl3_mul24.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <shelf/quantization/exl3/exl3_int8.h>

hipError_t shelf_hy4_wmma(const int8_t *, const float *, const uint32_t *, float *, int, int, int, int,
                          hipStream_t, int);

hipError_t shelf_hy4_wmma_experts(const int8_t *, const float *, const Hy4ExpertProjection *, const Hy4Route *,
                                  const Hy4ExpertBatch *, int, float *, bool, int, int, hipStream_t);
hipError_t shelf_hy4_wmma_decode(const int8_t *codes, const float *scales, const Hy4ExpertProjection *matrices,
                                 const Hy4Route *routes, const int *count, float *partial, int capacity,
                                 int k, int n, int projections, hipStream_t stream, const Hy4ExpertBatch *batches,
                                 const int *num_batches, bool prefill);
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
                              float *logits = nullptr, const Hy4Route *routes = nullptr, int splits = 1,
                              const int *count = nullptr) {
    if (count && int(blockIdx.y) >= *count)
        return;
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
__global__ void gather_rotate_pair(const uint16_t *input, const Hy4Route *routes, const int *d_count,
                                   const Hy4ExpertProjection *group, int stride, int8_t *codes, float *scales,
                                   int8_t *codes1, float *scales1, int width, int padded) {
    if (int(blockIdx.y) >= *d_count)
        return;
    const int src = routes[blockIdx.y].row;
    const int expert = routes[blockIdx.y].expert;
    const uint16_t *suh0 = group[size_t(expert) * stride].suh;
    const uint16_t *suh1 = group[size_t(expert) * stride + 1].suh;
    const unsigned lane = threadIdx.x;
    const int base = blockIdx.x * 128 + lane * 4;
    float v0[4], v1[4];
    for (int i = 0; i < 4; ++i) {
        float value = base + i < width ? bf16_bits_to_f32(input[size_t(src) * width + base + i]) : 0.0f;
        v0[i] = value * f16_bits_to_f32(suh0[base + i]);
        v1[i] = value * f16_bits_to_f32(suh1[base + i]);
    }
    fwht128(v0[0], v0[1], v0[2], v0[3], lane);
    fwht128(v1[0], v1[1], v1[2], v1[3], lane);
    float peak0 = 0, peak1 = 0;
    for (int i = 0; i < 4; ++i) {
        v0[i] *= kInvSqrt128;
        v1[i] *= kInvSqrt128;
        peak0 = fmaxf(peak0, fabsf(v0[i]));
        peak1 = fmaxf(peak1, fabsf(v1[i]));
    }
    for (int mask = 16; mask; mask /= 2) {
        peak0 = fmaxf(peak0, __shfl_xor(peak0, mask, 32));
        peak1 = fmaxf(peak1, __shfl_xor(peak1, mask, 32));
    }
    const float scale0 = peak0 / 127.0f;
    const float scale1 = peak1 / 127.0f;
    if (!lane) {
        scales[size_t(blockIdx.y) * (padded / 128) + blockIdx.x] = scale0;
        scales1[size_t(blockIdx.y) * (padded / 128) + blockIdx.x] = scale1;
    }
    for (int i = 0; i < 4; ++i) {
        codes[size_t(blockIdx.y) * padded + base + i] = scale0 ? int8_t(rintf(v0[i] / scale0)) : 0;
        codes1[size_t(blockIdx.y) * padded + base + i] = scale1 ? int8_t(rintf(v1[i] / scale1)) : 0;
    }
}

__global__ void rotate_output_swiglu_quant(const float *partial0, const float *partial1,
                                           const Hy4ExpertProjection *matrices, int stride, int8_t *codes,
                                           float *scales, int padded, const Hy4Route *routes, const int *d_count) {
    if (int(blockIdx.y) >= *d_count)
        return;
    const auto &mat = matrices[size_t(routes[blockIdx.y].expert) * stride];
    const uint16_t *svh0 = mat.svh, *svh1 = (&mat)[1].svh, *suh2 = (&mat)[2].suh;
    const int base = blockIdx.x * 128 + threadIdx.x * 4;
    float v0[4], v1[4];
    for (int i = 0; i < 4; ++i) {
        size_t off = size_t(blockIdx.y) * padded + base + i;
        v0[i] = partial0[off];
        v1[i] = partial1[off];
    }
    fwht128(v0[0], v0[1], v0[2], v0[3], threadIdx.x);
    fwht128(v1[0], v1[1], v1[2], v1[3], threadIdx.x);
    float v[4];
    for (int i = 0; i < 4; ++i) {
        float g = round_bf16(v0[i] * kInvSqrt128 * f16_bits_to_f32(svh0[base + i]));
        float u = round_bf16(v1[i] * kInvSqrt128 * f16_bits_to_f32(svh1[base + i]));
        g = fminf(g, 10.f);
        u = fminf(10.f, fmaxf(-10.f, u));
        uint16_t bits = f32_to_bf16_rne_device(round_bf16(g / (1.f + expf(-g))) * u);
        v[i] = bf16_bits_to_f32(bits) * f16_bits_to_f32(suh2[base + i]);
    }
    fwht128(v[0], v[1], v[2], v[3], threadIdx.x);
    float peak = 0;
    for (float &x : v) {
        x *= kInvSqrt128;
        peak = fmaxf(peak, fabsf(x));
    }
    for (int mask = 16; mask; mask /= 2)
        peak = fmaxf(peak, __shfl_xor(peak, mask, 32));
    const float scale = peak / 127.0f;
    if (!threadIdx.x)
        scales[size_t(blockIdx.y) * (padded / 128) + blockIdx.x] = scale;
    for (int i = 0; i < 4; ++i)
        codes[size_t(blockIdx.y) * padded + base + i] = scale ? int8_t(rintf(v[i] / scale)) : 0;
}
}

hipError_t shelf_hy4_linear_workspace(const uint16_t *input, const uint16_t *suh, const uint16_t *svh,
                                       const uint32_t *trellis, int8_t *codes, float *scales, float *partial,
                                       uint16_t *output, int rows, int input_width, int output_width,
                                       int padded_input, int padded_output, int bits, hipStream_t stream,
                                       int splits, int8_t *expanded) {
    if (!input || !suh || !svh || !trellis || !codes || !scales || !partial || !output || rows < 1 ||
        rows > 65535 || input_width < 1 || output_width < 1 || padded_input < input_width ||
        padded_output < output_width || padded_input % 128 || padded_output % 128 || (bits != 4 && bits != 6))
        return hipErrorInvalidValue;
    if (splits < 1 || splits > padded_input / 128)
        return hipErrorInvalidValue;
    auto status = hipSuccess;
    if (rows <= 16 && (bits == 4 || bits == 6)) {
        const bool out_fused = splits == 1 && output_width == padded_output;
        status = shelf_hy4_linear_fused_gemm(input, suh, out_fused ? svh : nullptr, trellis, partial,
                                             out_fused ? output : nullptr, rows, input_width, output_width,
                                             padded_input, padded_output, bits, splits, stream);
        if (status != hipSuccess)
            return status;
        if (out_fused)
            return hipSuccess;
    } else {
        rotate_input<<<dim3(padded_input / 128, rows), 32, 0, stream>>>(input, suh, codes, scales, input_width,
                                                                        padded_input);
        status = hipGetLastError();
        if (status != hipSuccess)
            return status;
        if (expanded) {
            status = shelf::kernel::launch_exl3_mul24_recon_wmma(reinterpret_cast<const uint16_t *>(trellis),
                                                                 bits, expanded, padded_output, padded_input, stream, true);
            if (status == hipSuccess)
                status = shelf::kernel::launch_exl3_mul24_prefill_wmma(codes, scales, padded_input, padded_input / 32,
                                                                       expanded, partial, rows, padded_output,
                                                                       padded_input, stream, splits);
        } else {
            status = shelf_hy4_wmma(codes, scales, trellis, partial, rows, padded_input, padded_output, bits, stream, splits);
        }
        if (status != hipSuccess)
            return status;
    }
    rotate_output<<<dim3(padded_output / 128, rows), 32, 0, stream>>>(
        partial, svh, output, output_width, padded_output, nullptr, 0, nullptr, nullptr, splits);
    return hipGetLastError();
}

extern "C" hipError_t shelf_hy4_linear(const uint16_t *input, const uint16_t *suh, const uint16_t *svh,
                                       const uint32_t *trellis, int8_t *codes, float *scales, float *partial,
                                       uint16_t *output, int rows, int input_width, int output_width,
                                       int padded_input, int padded_output, int bits, hipStream_t stream,
                                       int splits) {
    return shelf_hy4_linear_workspace(input, suh, svh, trellis, codes, scales, partial, output, rows, input_width,
                                      output_width, padded_input, padded_output, bits, stream, splits, nullptr);
}

extern "C" hipError_t shelf_hy4_expert_linear(const uint16_t *x, const Hy4ExpertProjection *matrices,
                                              int stride, int8_t *codes, float *scales, float *partial,
                                              uint16_t *y, int rows, const Hy4Route *routes,
                                              const Hy4ExpertBatch *batches, int num_batches, int mixed, int k,
                                              int n, hipStream_t stream) {
    if (!x || !matrices || !codes || !scales || !partial || !y || rows < 1 || rows > 65535 || stride < 1)
        return hipErrorInvalidValue;
    if (!routes || !batches || num_batches < 1 || num_batches > rows || stride != 3)
        return hipErrorInvalidValue;
    if (k < 1 || k % 128 || n < 1 || n % 64)
        return hipErrorInvalidValue;
    rotate_input<<<dim3(k / 128, rows), 32, 0, stream>>>(x, nullptr, codes, scales, k, k, matrices, stride, routes);
    auto status = hipGetLastError();
    if (status != hipSuccess)
        return status;
    status = shelf_hy4_wmma_experts(codes, scales, matrices, routes, batches, num_batches, partial, mixed, k, n, stream);
    if (status != hipSuccess)
        return status;
    status = hipGetLastError();
    if (status != hipSuccess)
        return status;
    rotate_output<<<dim3(n / 128, rows), 32, 0, stream>>>(partial, nullptr, y, n, n, matrices, stride, nullptr, routes);
    return hipGetLastError();
}

hipError_t shelf_hy4_head(const uint16_t *x, const uint16_t *suh, const uint16_t *svh, const uint32_t *trellis,
                          int8_t *codes, float *scales, float *partial, float *logits, int rows, hipStream_t stream) {
    if (!x || !suh || !svh || !trellis || !codes || !scales || !partial || !logits || rows < 1 || rows > 96)
        return hipErrorInvalidValue;
    rotate_input<<<dim3(24, rows), 32, 0, stream>>>(x, suh, codes, scales, 3072, 3072);
    auto status = hipGetLastError();
    if (status != hipSuccess)
        return status;
    status = shelf_hy4_wmma(codes, scales, trellis, partial, rows, 3072, 120832, 6, stream, 1);
    if (status != hipSuccess)
        return status;
    rotate_output<<<dim3(944, rows), 32, 0, stream>>>(partial, svh, nullptr, 120832, 120832, nullptr, 0, logits);
    return hipGetLastError();
}

hipError_t shelf_hy4_expert_decode(const uint16_t *latent, const Hy4ExpertProjection *matrices,
                                   const Hy4Route *routes, const int *count, int capacity, int intermediate,
                                   int8_t *codes, float *scales, float *partial, uint16_t *output,
                                   hipStream_t stream, const Hy4ExpertBatch *batches, const int *num_batches, bool prefill) {
    if (!latent || !matrices || !routes || !count || !codes || !scales || !partial || !output ||
        (prefill && !batches) || bool(batches) != bool(num_batches) ||
        capacity < 1 || capacity > 16 * (prefill ? shelf::hy4::expert_tile_rows : shelf::hy4::Hy4GroupedPlan::max_rows) ||
        (intermediate != 768 && intermediate != 1536))
        return hipErrorInvalidValue;
    constexpr int width = 1536;
    auto codes1 = codes + capacity * width;
    auto scales1 = scales + capacity * (width / 128);
    auto partial1 = partial + capacity * intermediate;
    gather_rotate_pair<<<dim3(width / 128, capacity), 32, 0, stream>>>(
        latent, routes, count, matrices, 3, codes, scales, codes1, scales1, width, width);
    auto status = hipGetLastError();
    if (status != hipSuccess)
        return status;
    status = shelf_hy4_wmma_decode(codes, scales, matrices, routes, count, partial, capacity,
                                   width, intermediate, 2, stream, batches, num_batches, prefill);
    if (status != hipSuccess)
        return status;
    rotate_output_swiglu_quant<<<dim3(intermediate / 128, capacity), 32, 0, stream>>>(
        partial, partial1, matrices, 3, codes, scales, intermediate, routes, count);
    status = hipGetLastError();
    if (status != hipSuccess)
        return status;
    status = shelf_hy4_wmma_decode(codes, scales, matrices + 2, routes, count, partial, capacity,
                                   intermediate, width, 1, stream, batches, num_batches, prefill);
    if (status != hipSuccess)
        return status;
    rotate_output<<<dim3(width / 128, capacity), 32, 0, stream>>>(
        partial, nullptr, output, width, width, matrices + 2, 3, nullptr, routes, 1, count);
    return hipGetLastError();
}

hipError_t shelf_hy4_expert_decode(const uint16_t *latent, const Hy4ExpertProjection *matrices,
                                   const Hy4Route *routes, const int *count, int capacity, int intermediate,
                                   int8_t *codes, float *scales, float *partial, uint16_t *output,
                                   hipStream_t stream, const Hy4ExpertBatch *batches, const int *num_batches) {
    return shelf_hy4_expert_decode(latent, matrices, routes, count, capacity, intermediate, codes,
                                        scales, partial, output, stream, batches, num_batches, false);
}
