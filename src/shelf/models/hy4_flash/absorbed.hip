#include <shelf/models/hy4_flash/absorbed.h>
#include <shelf/models/hy4_flash/mla_prefill.h>
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
__global__ void quant_latent(Mat latent, const uint16_t *suh, _Float16 *data, const int *meta) {
    if (meta && int(blockIdx.y) >= meta[512] + meta[513])
        return;
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
}
void fold_kv_weights(Residency &gpu, int rank) {
    for (const auto &[key, dest] : gpu.folded[rank]) {
        if (!dest)
            continue;
        auto n = gpu.model.shape_of(key, rank)->at(1);
        fold<<<dim3(n / 128, 512), 128>>>(
            static_cast<const uint32_t *>(gpu.device(gpu.model.at(key + ".trellis", rank))),
            static_cast<const uint16_t *>(gpu.device(gpu.model.at(key + ".svh", rank))),
            static_cast<_Float16 *>(dest), n);
        checked(hipGetLastError());
    }
    checked(hipDeviceSynchronize());
}
void attend_absorbed(Ops &op, const std::string &projection, Mat latent, Mat rotary, Mat q, Mat gate,
                     const int *positions, int stride, Param sinks, Mat out, bool contiguous, bool sliding,
                     int row_offset, int chunk_rows, const int *meta) {
    auto folded = static_cast<const _Float16 *>(op.gpu.folded[op.rank].at(projection));
    int heads = q.cols / 256, rows = q.rows;
    auto input = static_cast<_Float16 *>(op.arena.next(size_t(latent.rows) * 512 * 2));
    auto query = op.arena.next(size_t(rows) * heads * 512 * 2);
    auto values = op.arena.next(size_t(rows) * heads * 512 * 2);
    size_t length = size_t(rows) * heads * ((stride + 15) / 16 * 16);
    auto scores = static_cast<float *>(op.arena.next(length * 4));
    auto probs = static_cast<uint16_t *>(op.arena.next(length * 2));
    quant_latent<<<dim3(4, latent.rows), 32, 0, op.stream>>>(
        latent, static_cast<const uint16_t *>(op.param(projection + ".suh").p), input, meta);
    checked(hipGetLastError());
    checked(shelf_hy4_mla_absorbed(q.p, folded, rotary.p, gate.p, sinks.p, sinks.type == io::SType::F32,
                                   positions, stride, input, query, values, scores, probs, out.p, rows, heads,
                                   op.stream, op.params, contiguous, sliding, row_offset, chunk_rows));
}
void attend_absorbed_batch(Ops &op, const std::string &projection, Mat latent, Mat rotary, Mat q, Mat gate,
                           const int *indices, int stride, const int *row_positions, Param sinks, Mat out,
                           bool sliding) {
    auto folded = static_cast<const _Float16 *>(op.gpu.folded[op.rank].at(projection));
    int heads = q.cols / 256, rows = q.rows;
    auto input = static_cast<_Float16 *>(op.arena.next(size_t(latent.rows) * 512 * 2));
    auto query = op.arena.next(size_t(rows) * heads * 512 * 2);
    auto values = op.arena.next(size_t(rows) * heads * 512 * 2);
    size_t length = size_t(rows) * heads * ((stride + 15) / 16 * 16);
    auto scores = static_cast<float *>(op.arena.next(length * 4));
    auto probs = static_cast<uint16_t *>(op.arena.next(length * 2));
    quant_latent<<<dim3(4, latent.rows), 32, 0, op.stream>>>(
        latent, static_cast<const uint16_t *>(op.param(projection + ".suh").p), input, nullptr);
    checked(hipGetLastError());
    checked(shelf_hy4_mla_absorbed_batch(q.p, folded, rotary.p, gate.p, sinks.p, sinks.type == io::SType::F32,
                                         indices, stride, input, query, values, scores, probs, out.p, rows, heads,
                                         op.stream, row_positions, sliding));
}

}
