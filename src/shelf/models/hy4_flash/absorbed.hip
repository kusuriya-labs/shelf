#include <shelf/models/hy4_flash/absorbed.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <shelf/quantization/exl3/exl3_int8.h>

namespace shelf::hy4 {
namespace {
using namespace shelf::kernel::exl3_detail;
using namespace shelf::kernel::exl3_detail::mul24;

__global__ void fold(const uint32_t *trellis, const uint16_t *svh, _Float16 *result, int n) {
    __shared__ float shared[128];
    int out = blockIdx.x * 128 + threadIdx.x, k = blockIdx.y;
    unsigned t = (k % 16) / 8 * 128 + (out % 16) * 8 + k % 8;
    unsigned bit = ((t + 1) * 4 + 1024 - 16) % 1024;
    auto tile = trellis + size_t((k / 16) * (n / 16) + out / 16) * 32;
    uint64_t pair = uint64_t(tile[bit / 32]) << 32 | tile[(bit / 32 + 1) % 32];
    unsigned value = __umul24((pair >> (48 - bit % 32)) & 65535, 0xdcd12du);
    int bytes = __builtin_amdgcn_udot4(value, 0x01010101, 0, false);
    float weight = float((bytes - 509) >> 2) * quantization::exl3::kExl3Int8GridStep;
    weight = fwht128_smem(weight, shared, threadIdx.x) * kInvSqrt128 * f16_bits_to_f32(svh[out]);
    result[size_t(out) * 512 + k] = static_cast<_Float16>(weight);
}
__global__ void quant_latent(Mat latent, const uint16_t *suh, float *data) {
    int col = blockIdx.x * 128 + threadIdx.x * 4;
    float v[4], peak = 0;
    for (int i = 0; i < 4; ++i)
        v[i] = bf16_bits_to_f32(latent.p[blockIdx.y * 512 + col + i]) * f16_bits_to_f32(suh[col + i]);
    fwht128(v[0], v[1], v[2], v[3], threadIdx.x);
    for (float &f : v) {
        f *= kInvSqrt128;
        peak = fmaxf(peak, fabsf(f));
    }
    for (int mask = 16; mask; mask /= 2)
        peak = fmaxf(peak, __shfl_xor(peak, mask, 32));
    float scale = peak / 127;
    for (int i = 0; i < 4; ++i)
        data[blockIdx.y * 512 + col + i] = scale ? rintf(v[i] / scale) * scale : 0;
}
__global__ void absorb_query(Mat q, const _Float16 *folded, float *query) {
    int k = blockIdx.x * 256 + threadIdx.x, head = blockIdx.y, row = blockIdx.z;
    const auto a = q.p + size_t(row) * q.cols + head * 256;
    const auto w = folded + size_t(head) * 448 * 512;
    float dot = 0;
    for (int j = 0; j < 192; ++j)
        dot += bf16_bits_to_f32(a[j]) * float(w[j * 512 + k]);
    query[(size_t(row) * gridDim.y + head) * 512 + k] = dot;
}
__global__ void score(Mat q, Mat rotary, const float *query, const float *input, const int *indices, int stride,
                      float *probs) {
    int slot = blockIdx.x * 4 + threadIdx.x / 32, head = blockIdx.y, row = blockIdx.z, lane = threadIdx.x % 32;
    if (slot >= stride)
        return;
    int token = indices[row * stride + slot];
    float dot = 0;
    if (token >= 0) {
        for (int k = lane; k < 512; k += 32)
            dot += query[(size_t(row) * gridDim.y + head) * 512 + k] * input[size_t(token) * 512 + k];
        for (int j = lane; j < 64; j += 32)
            dot += bf16_bits_to_f32(q.p[size_t(row) * q.cols + head * 256 + 192 + j]) *
                   bf16_bits_to_f32(rotary.p[size_t(token) * 64 + j]);
        for (int d = 16; d; d /= 2)
            dot += __shfl_down(dot, d, 32);
    }
    if (!lane)
        probs[(size_t(row) * gridDim.y + head) * stride + slot] = token < 0 ? -INFINITY : dot * 0.0625f;
}
__global__ void softmax(float *p, int stride, Param sinks) {
    __shared__ float scratch[256];
    int c = threadIdx.x, head = blockIdx.x;
    auto probs = p + (size_t(blockIdx.y) * gridDim.x + head) * stride;
    float sink = sinks.type == io::SType::F32 ? static_cast<const float *>(sinks.p)[head]
                                              : bf16_bits_to_f32(static_cast<const uint16_t *>(sinks.p)[head]);
    float vmax = sink;
    for (int i = c; i < stride; i += 256)
        vmax = fmaxf(vmax, probs[i]);
    scratch[c] = vmax;
    __syncthreads();
    for (int d = 128; d; d /= 2) {
        if (c < d)
            scratch[c] = fmaxf(scratch[c], scratch[c + d]);
        __syncthreads();
    }
    vmax = scratch[0];
    __syncthreads();
    float sum = c == 0 ? expf(sink - vmax) : 0;
    for (int i = c; i < stride; i += 256) {
        probs[i] = expf(probs[i] - vmax);
        sum += probs[i];
    }
    scratch[c] = sum;
    __syncthreads();
    for (int d = 128; d; d /= 2) {
        if (c < d)
            scratch[c] += scratch[c + d];
        __syncthreads();
    }
    for (int i = c; i < stride; i += 256)
        probs[i] /= scratch[0];
}
__global__ void weighted_latent(const float *p, const int *ids, int stride, const float *input, float *v) {
    int k = blockIdx.x * 256 + threadIdx.x, head = blockIdx.y, row = blockIdx.z;
    auto probs = p + (size_t(row) * gridDim.y + head) * stride;
    float sum = 0;
    for (int t = 0; t < stride; ++t) {
        int id = ids[row * stride + t];
        if (id >= 0)
            sum += probs[t] * input[size_t(id) * 512 + k];
    }
    v[(size_t(row) * gridDim.y + head) * 512 + k] = sum;
}
__global__ void expand_value(const _Float16 *folded, const float *v, Mat gate, Mat out) {
    int c = blockIdx.x * 4 + threadIdx.x / 32, h = blockIdx.y, r = blockIdx.z, lane = threadIdx.x % 32;
    auto w = folded + (size_t(h) * 448 + 192 + c) * 512;
    const auto input = v + (size_t(r) * gridDim.y + h) * 512;
    float sum = 0;
    for (int k = lane; k < 512; k += 32)
        sum += float(w[k]) * input[k];
    for (int d = 16; d; d /= 2)
        sum += __shfl_down(sum, d, 32);
    size_t i = size_t(r) * out.cols + h * 256 + c;
    if (!lane)
        out.p[i] = f32_to_bf16_rne_device(sum / (1.f + expf(-bf16_bits_to_f32(gate.p[i]))));
}
}
void fold_kv_weights(Residency &gpu) {
    for (const auto &[key, dest] : gpu.folded) {
        auto n = gpu.model.shape.at(key)[1];
        fold<<<dim3(n / 128, 512), 128>>>(
            static_cast<const uint32_t *>(gpu.device(gpu.model.at(key + ".trellis"))),
            static_cast<const uint16_t *>(gpu.device(gpu.model.at(key + ".svh"))),
            static_cast<_Float16 *>(dest), n);
        checked(hipGetLastError());
    }
    checked(hipDeviceSynchronize());
}
void attend_absorbed(Ops &op, const std::string &projection, Mat latent, Mat rotary, Mat q, Mat gate,
                     const int *positions, int stride, Param sinks, Mat out) {
    auto folded = static_cast<const _Float16 *>(op.gpu.folded.at(projection));
    int heads = q.cols / 256, rows = q.rows;
    auto input = static_cast<float *>(op.arena.next(size_t(latent.rows) * 512 * 4));
    auto query = static_cast<float *>(op.arena.next(size_t(rows) * heads * 512 * 4));
    auto probs = static_cast<float *>(op.arena.next(size_t(rows) * heads * stride * 4));
    auto values = static_cast<float *>(op.arena.next(size_t(rows) * heads * 512 * 4));
    quant_latent<<<dim3(4, latent.rows), 32>>>(
        latent, static_cast<const uint16_t *>(op.param(projection + ".suh").p), input);
    checked(hipGetLastError());
    absorb_query<<<dim3(2, heads, rows), 256>>>(q, folded, query);
    checked(hipGetLastError());
    score<<<dim3((stride + 3) / 4, heads, rows), 128>>>(q, rotary, query, input, positions, stride, probs);
    checked(hipGetLastError());
    softmax<<<dim3(heads, rows), 256>>>(probs, stride, sinks);
    checked(hipGetLastError());
    weighted_latent<<<dim3(2, heads, rows), 256>>>(probs, positions, stride, input, values);
    checked(hipGetLastError());
    expand_value<<<dim3(64, heads, rows), 128>>>(folded, values, gate, out);
    checked(hipGetLastError());
}
}
