#include <shelf/models/hy4_flash/decoder.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace shelf::hy4 {
namespace {
using namespace shelf::kernel::exl3_detail::mul24;
constexpr int64_t multipliers[4][3] = {{81853575799317, 39690085502883, 24229915018165},
                                       {19457948045719, 44729556510419, 84779352922315},
                                       {72974230057307, 21530602121221, 55998057243097},
                                       {29128509562787, 16745155128523, 22474588302733}};
struct Norms {
    const uint16_t *head[16];
};
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
__global__ void decode_heads(const uint8_t *packed, Norms norms, Mat embed) {
    __shared__ float scratch[128];
    int c = threadIdx.x, head = blockIdx.x;
    const uint8_t *src = packed + (blockIdx.y * 16 + head) * 98;
    float v = 0;
    if (c < 96) {
        int b = src[c], exp = (b >> 3) & 15, frac = b & 7;
        float fp = exp ? ldexpf(1.f + frac / 8.f, exp - 7) : ldexpf(float(frac), -9);
        v = (b & 128 ? -fp : fp) * bf16_bits_to_f32(uint16_t(src[96] | (src[97] << 8)));
        v = bf16_bits_to_f32(f32_to_bf16_rne_device(v));
    }
    float inv = rsqrtf(sum(v * v, scratch) / 96 + 1.1920928955e-7f);
    if (c < 96)
        embed.p[blockIdx.y * 1536 + head * 96 + c] =
            f32_to_bf16_rne_device(round_bf16(v * inv) * bf16_bits_to_f32(norms.head[head][c]));
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
std::vector<uint8_t> gather_engram(const Checkpoint &model, int layer, int rows, const std::vector<int> &ids,
                                   const std::vector<int> &prior) {
    int g = (layer - 1) / 10;
    auto p = "model.layers." + std::to_string(layer) + ".engram_layer.";
    auto table = p + "multi_head_embedding.embedding.weight";
    const auto &lookup = model.at(p + "lookup_table", -1);
    if (lookup.spec.dtype != io::SType::I32 || lookup.bytes != 120818 * 4)
        throw std::runtime_error("Engram の語彙写像が異なります");
    auto lut = static_cast<const int32_t *>(lookup.host);
    auto bits = static_cast<const uint8_t *>(model.at(table, -1).host);
    auto scales = static_cast<const uint8_t *>(model.at(table + ".scale", -1).host);
    std::vector<uint8_t> packed(size_t(rows) * 16 * 98);
    for (int r = 0; r < rows; ++r) {
        int64_t mix[3]{};
        for (int j = 0; j < 3; ++j) {
            int index = r - j, id = 120002;
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
            size_t idx = offset + ((mix[0] ^ mix[1] ^ (h < 8 ? 0 : mix[2])) % size);
            auto dest = packed.data() + (r * 16 + h) * 98;
            std::memcpy(dest, bits + idx * 96, 96);
            std::memcpy(dest + 96, scales + idx * 2, 2);
            offset += size;
        }
    }
    return packed;
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
    uint8_t *memory = nullptr;
    if (op.capture)
        memory = op.gpu.graph_input->engram[g];
    else {
        auto packed = op.take_engram(g);
        if (packed.empty())
            packed = gather_engram(op.gpu.model, layer, x.rows, ids, prior);
        memory = static_cast<uint8_t *>(op.arena.next(packed.size()));
        op.copy(memory, packed.data(), packed.size(), hipMemcpyHostToDevice);
    }
    auto embed = op.mat(x.rows, 1536);
    Norms norms{};
    for (int i = 0; i < 16; ++i)
        norms.head[i] = bf("lookup_per_head_norms." + std::to_string(i) + ".weight");
    decode_heads<<<dim3(16, x.rows), 128, 0, op.stream>>>(memory, norms, embed);
    checked(hipGetLastError());
    auto project = [&](const std::string &key) { return op.linear(p + key, embed); };
    constexpr float eps = 1.1920928955e-7f;
    auto values = op.norm_bias(project("value_proj"), bf("value_proj.bias"), p + "unified_pre_gamma", eps);
    auto token_ids = op.capture ? op.params->ids : op.indices(ids);
    for (int i = 0; i < 4; ++i) {
        auto s = std::to_string(i);
        auto k = op.norm_bias(project("key_projs." + s), bf("key_projs." + s + ".bias"),
                              p + "norm_key." + s + ".weight", eps);
        auto q = op.norm_slice(x, i * 3072, 3072, p + "norm_query." + s + ".weight", eps);
        add_memory<<<x.rows, 256, 0, op.stream>>>(x, k, q, values, bf("unified_per_hc_scales." + s), token_ids, i);
        checked(hipGetLastError());
    }
    return x;
}

void Ops::prefetch_engram(const std::vector<int> &ids, const std::vector<int> &prior) {
    if (engram_future.valid())
        engram_future.wait();
    engram_data.reset();
    engram_future = std::async(std::launch::async, [this, ids, prior] {
        std::array<std::vector<uint8_t>, 4> result;
        for (int g = 0; g < 4; ++g)
            result[g] = gather_engram(gpu.model, 1 + 10 * g, ids.size(), ids, prior);
        return result;
    });
}
std::vector<uint8_t> Ops::take_engram(int group) {
    if (!engram_data) {
        if (!engram_future.valid())
            return {};
        engram_data = engram_future.get();
    }
    return std::move((*engram_data)[group]);
}
}
