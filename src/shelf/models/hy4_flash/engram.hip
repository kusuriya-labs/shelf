#include <shelf/models/hy4_flash/decoder.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <hip/hip_fp16.h>
#include <chrono>
#include <sys/resource.h>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <span>

namespace shelf::hy4 {
namespace {
using namespace shelf::kernel::exl3_detail::mul24;
constexpr int64_t multipliers[4][3] = {{81853575799317, 39690085502883, 24229915018165},
                                       {19457948045719, 44729556510419, 84779352922315},
                                       {72974230057307, 21530602121221, 55998057243097},
                                       {29128509562787, 16745155128523, 22474588302733}};
__device__ float sum(float v, float *scratch) {
    int t = threadIdx.x;
    scratch[t] = v;
    __syncthreads();
    for (int d = blockDim.x / 2; d; d /= 2) {
        if (t < d)
            scratch[t] += scratch[t + d];
        __syncthreads();
    }
    return scratch[0];
}
__global__ void decode_heads(const uint8_t *packed, EngramNorms norms, Mat embed, NgramLayout layout, bool ngram) {
    __shared__ float scratch[128];
    int c = threadIdx.x, head = blockIdx.x;
    const uint8_t *src = ngram ? packed + blockIdx.y * layout.stride + layout.offsets[head]
                               : packed + (blockIdx.y * 16 + head) * 98;
    float v = 0;
    if (c < 96) {
        if (ngram) {
            unsigned bits = layout.bits[head], pos = c * bits, shift = pos % 8;
            unsigned code = src[pos / 8] >> shift;
            if (shift + bits > 8)
                code |= unsigned(src[pos / 8 + 1]) << (8 - shift);
            code &= (1u << bits) - 1;
            unsigned width = 12 * bits;
            uint16_t scale = uint16_t(src[width] | (src[width + 1] << 8));
            v = scale ? __half2float(*reinterpret_cast<const __half *>(&scale)) *
                            __half2float(*reinterpret_cast<const __half *>(layout.books[head] + code)) : 0.f;
        } else {
            int b = src[c], exp = (b >> 3) & 15, frac = b & 7;
            float fp = exp ? ldexpf(1.f + frac / 8.f, exp - 7) : ldexpf(float(frac), -9);
            v = (b & 128 ? -fp : fp) * bf16_bits_to_f32(uint16_t(src[96] | (src[97] << 8)));
        }
        v = bf16_bits_to_f32(f32_to_bf16_rne_device(v));
    }
    float inv = rsqrtf(sum(v * v, scratch) / 96 + 1.1920928955e-7f);
    if (c < 96)
        embed.p[blockIdx.y * 1536 + head * 96 + c] =
            f32_to_bf16_rne_device(round_bf16(v * inv) * bf16_bits_to_f32(norms.head[head][c]));
}
__global__ void bias(Mat x, const uint16_t *b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < x.rows * x.cols)
        x.p[i] = f32_to_bf16_rne_device(bf16_bits_to_f32(x.p[i]) + bf16_bits_to_f32(b[i % x.cols]));
}
__global__ void add_memory(Mat x, Mat k, Mat q, Mat value, const uint16_t *scale, const int *ids, int stream) {
    __shared__ float scratch[256];
    int row = blockIdx.x, tid = threadIdx.x;
    float dot = 0;
    for (int i = tid; i < 3072; i += 256)
        dot += round_bf16(bf16_bits_to_f32(k.p[row * 3072 + i]) * bf16_bits_to_f32(q.p[row * 3072 + i]));
    dot = round_bf16(round_bf16(sum(dot, scratch)) * 0.01804219591f);
    float signed_root = dot == 0 ? 0 : copysignf(round_bf16(sqrtf(fmaxf(fabsf(dot), 1e-6f))), dot);
    float gate = round_bf16(1.f / (1.f + expf(-signed_root)));
    if (ids[row] == 120002 || ids[row] == 120120 || ids[row] == 120124)
        return;
    for (int i = tid; i < 3072; i += 256) {
        size_t index = (size_t(row) * 4 + stream) * 3072 + i;
        float v = bf16_bits_to_f32(value.p[row * 3072 + i]);
        x.p[index] = f32_to_bf16_rne_device(
            bf16_bits_to_f32(x.p[index]) +
            round_bf16(round_bf16(v * gate) + round_bf16(v * bf16_bits_to_f32(scale[i]))));
    }
}
}
void launch_decode_heads(const uint8_t *packed, EngramNorms norms, Mat embed, NgramLayout layout, bool ngram) {
    decode_heads<<<dim3(16, embed.rows), 128>>>(packed, norms, embed, layout, ngram);
    checked(hipGetLastError());
}
std::vector<uint64_t> engram_indices(const Checkpoint &model, int g, const std::vector<int> &prior,
                                     const std::vector<int> &ids) {
    auto p = "model.layers." + std::to_string(1 + 10 * g) + ".engram_layer.";
    const auto &lookup = model.at(p + "lookup_table");
    if (lookup.spec.dtype != io::SType::I32 || lookup.bytes != 120818 * 4)
        throw std::runtime_error("Engram の語彙写像が異なります");
    auto lut = static_cast<const int32_t *>(lookup.host);
    std::vector<uint64_t> indices;
    indices.reserve(ids.size() * 16);
    for (size_t r = 0; r < ids.size(); ++r) {
        int64_t mix[3]{};
        for (int j = 0; j < 3; ++j) {
            int index = int(r) - j, id = 120002;
            if (index >= 0)
                id = ids[index];
            else if (int(prior.size()) + index >= 0)
                id = prior[int(prior.size()) + index];
            int normalized = lut[id >= 0 && id < 120818 ? id : 120002];
            if (normalized < 0 || normalized > 92619)
                throw std::runtime_error("Engram の正規化 ID が不正です");
            mix[j] = normalized * multipliers[g][j];
        }
        size_t offset = 0;
        for (int h = 0; h < 16; ++h) {
            int size = model.moduli[g][h];
            indices.push_back(offset + ((mix[0] ^ mix[1] ^ (h < 8 ? 0 : mix[2])) % size));
            offset += size;
        }
    }
    return indices;
}
struct EngramSource {
    Ops &op;
    void prefetch(const std::vector<int> &prior, const std::vector<int> &ids);
    std::span<const uint8_t> rows(int g, const std::vector<int> &prior, const std::vector<int> &ids,
                                  std::vector<uint8_t> &storage);
};
void EngramSource::prefetch(const std::vector<int> &prior, const std::vector<int> &ids) {
#ifdef SHELF_HY4_RDMA
    if (!op.gpu.model.rdma)
        return;
    rdma::Indices indices;
    for (int g = 0; g < 4; ++g)
        indices[g] = engram_indices(op.gpu.model, g, prior, ids);
    auto client = op.gpu.engram_client.get();
    op.gpu.engram_pending = std::async(std::launch::async, [client, indices = std::move(indices)] {
        return client->read(indices);
    });
#else
    (void)prior;
    (void)ids;
#endif
}
std::span<const uint8_t> EngramSource::rows(int g, const std::vector<int> &prior, const std::vector<int> &ids,
                                           std::vector<uint8_t> &storage) {
    size_t bytes = ids.size() * 16 * 98;
#ifdef SHELF_HY4_RDMA
    if (op.gpu.model.rdma) {
        if (g == 0)
            op.gpu.engram_rows = op.gpu.engram_pending.get();
        auto result = op.gpu.engram_rows[g];
        if (result.size() != bytes)
            throw std::runtime_error("RDMA Engram の取得長が不正です");
        return result;
    }
#endif
    if (op.gpu.model.ngram) {
        const auto &heads = op.gpu.model.ngram_heads[g];
        size_t stride = 0;
        for (const auto &head : heads)
            stride += 12 * head.bits + 2;
        storage.resize(ids.size() * stride);
        auto indices = engram_indices(op.gpu.model, g, prior, ids);
        for (size_t r = 0; r < ids.size(); ++r) {
            size_t dest_offset = r * stride, base = 0;
            for (size_t h = 0; h < 16; ++h) {
                const auto &head = heads[h];
                size_t row = indices[r * 16 + h] - base, width = 12 * head.bits;
                std::memcpy(storage.data() + dest_offset, head.codes + row * width, width);
                std::memcpy(storage.data() + dest_offset + width, head.scales + row * 2, 2);
                dest_offset += width + 2;
                base += op.gpu.model.moduli[g][h];
            }
        }
        return storage;
    }
    auto table = "model.layers." + std::to_string(1 + 10 * g) +
                 ".engram_layer.multi_head_embedding.embedding.weight";
    auto bits = static_cast<const uint8_t *>(op.gpu.model.at(table).host);
    auto scales = static_cast<const uint8_t *>(op.gpu.model.at(table + ".scale").host);
    auto indices = engram_indices(op.gpu.model, g, prior, ids);
    storage.resize(bytes);
    for (size_t i = 0; i < indices.size(); ++i) {
        auto dest = storage.data() + i * 98;
        std::memcpy(dest, bits + indices[i] * 96, 96);
        std::memcpy(dest + 96, scales + indices[i] * 2, 2);
    }
    return storage;
}
void prefetch_engram(Ops &op, const std::vector<int> &prior, const std::vector<int> &ids) {
    EngramSource{op}.prefetch(prior, ids);
}
Mat engram(Ops &op, int layer, Mat x, const std::vector<int> &prior, const std::vector<int> &ids) {
    int g = (layer - 1) / 10;
    auto p = "model.layers." + std::to_string(layer) + ".engram_layer.";
    auto bf = [&](const std::string &name) {
        auto v = op.param(p + name);
        if (v.type != io::SType::BF16)
            throw std::runtime_error("Engram パラメーターの型が異なります");
        return static_cast<const uint16_t *>(v.p);
    };
    std::chrono::steady_clock::time_point lookup_begin;
    rusage before{};
    if (op.gpu.expert_begin[0]) {
        getrusage(RUSAGE_THREAD, &before);
        lookup_begin = std::chrono::steady_clock::now();
    }
    std::vector<uint8_t> storage;
    auto packed = EngramSource{op}.rows(g, prior, ids, storage);
    if (op.gpu.expert_begin[0]) {
        auto end = std::chrono::steady_clock::now();
        rusage after{};
        getrusage(RUSAGE_THREAD, &after);
        op.gpu.engram_lookup_ms += std::chrono::duration<double, std::milli>(end - lookup_begin).count();
        op.gpu.engram_major_faults += after.ru_majflt - before.ru_majflt;
    }
    auto memory = static_cast<uint8_t *>(op.arena.next(packed.size()));
    op.copy(memory, packed.data(), packed.size(), hipMemcpyHostToDevice);
    auto embed = op.mat(x.rows, 1536);
    EngramNorms norms{};
    for (int i = 0; i < 16; ++i)
        norms.head[i] = bf("lookup_per_head_norms." + std::to_string(i) + ".weight");
    NgramLayout layout{};
    if (op.gpu.model.ngram) {
        for (int h = 0; h < 16; ++h) {
            const auto &head = op.gpu.model.ngram_heads[g][h];
            layout.books[h] = static_cast<const uint16_t *>(op.gpu.device(op.gpu.model.at(head.codebook)));
            layout.bits[h] = head.bits;
            layout.offsets[h] = layout.stride;
            layout.stride += 12 * head.bits + 2;
        }
    }
    launch_decode_heads(memory, norms, embed, layout, op.gpu.model.ngram);
    auto project = [&](const std::string &key) {
        Mat y = op.linear(p + key, embed);
        bias<<<(y.rows * y.cols + 255) / 256, 256>>>(y, bf(key + ".bias"));
        checked(hipGetLastError());
        return y;
    };
    auto values = op.norm(project("value_proj"), p + "unified_pre_gamma", 1.1920928955e-7f);
    auto token_ids = op.indices(ids);
    for (int i = 0; i < 4; ++i) {
        auto s = std::to_string(i);
        auto k = op.norm(project("key_projs." + s), p + "norm_key." + s + ".weight", 1.1920928955e-7f);
        auto q = op.norm(op.slice(x, i * 3072, 3072), p + "norm_query." + s + ".weight", 1.1920928955e-7f);
        add_memory<<<x.rows, 256>>>(x, k, q, values, bf("unified_per_hc_scales." + s), token_ids, i);
        checked(hipGetLastError());
    }
    return x;
}
}
