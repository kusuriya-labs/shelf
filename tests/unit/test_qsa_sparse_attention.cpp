#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/types.h>
#include <shelf/kernels/optimized/attention/qsa_sparse.h>

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    } else {
        std::fprintf(stderr, "ok: %s\n", what);
    }
}

float bf16_to_float(std::uint16_t bits) {
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16u;
    float value = 0.0f;
    std::memcpy(&value, &wide, sizeof(value));
    return value;
}

std::vector<shelf::bf16_t> random_bf16(std::size_t count, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<shelf::bf16_t> out(count);
    for (auto& value : out) value.data = shelf::f32_to_bf16_rne(dist(rng));
    return out;
}

constexpr std::uint32_t kKvHeads = 4u;
constexpr std::uint32_t kQHeads = 24u;
constexpr std::uint32_t kHeadDim = 256u;
constexpr std::uint32_t kPageTokens = 16u;
constexpr std::uint32_t kNumPages = 8u;
constexpr std::uint32_t kRatio = 4u;
constexpr std::uint32_t kMaskShift = 2u;
constexpr std::uint32_t kMaskBlocks = 32u;
constexpr std::uint32_t kMaskStride = kMaskBlocks / 8u;
constexpr std::uint32_t kRows = 4u;
constexpr std::uint32_t kMaxSequences = 2u;
constexpr std::uint32_t kBlockTableStride = 8u;
constexpr std::uint32_t kLayers = 2u;

const std::uint32_t kPositions[kRows] = {63u, 10u, 41u, 20u};
const std::uint32_t kSlots[kRows] = {0u, 1u, 0u, 7u};
const std::uint32_t kBlockTables[kMaxSequences * kBlockTableStride] = {
    0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u,
    7u, 6u, 5u, 4u, 3u, 2u, 1u, 0u,
};

std::uint8_t mask_byte(std::uint32_t row, std::uint32_t block) {
    const std::uint32_t row_blocks[kRows][8] = {
        {0u, 1u, 5u, 8u, 15u, 0xFFu, 0xFFu, 0xFFu},
        {0u, 2u, 3u, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu},
        {1u, 9u, 10u, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu},
        {4u, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu},
    };
    std::uint8_t bits = 0u;
    for (std::uint32_t i = 0; i < 8u; ++i) {
        const std::uint32_t b = row_blocks[row][i];
        if (b >= 32u || (b >> 3u) != block) continue;
        bits = static_cast<std::uint8_t>(bits | (1u << (b & 7u)));
    }
    return bits;
}

std::uint32_t mask_bit(const std::vector<std::uint8_t>& mask, std::uint32_t row,
                       std::uint32_t block) {
    const std::size_t byte = static_cast<std::size_t>(row) * kMaskStride + (block >> 3u);
    return (mask[byte] >> (block & 7u)) & 1u;
}

std::vector<float> reference_attention(const std::vector<shelf::bf16_t>& q,
                                       const std::vector<shelf::bf16_t>& k_pool,
                                       const std::vector<shelf::bf16_t>& v_pool,
                                       const std::vector<std::uint8_t>& mask) {
    const std::uint32_t elems_per_token = kKvHeads * kHeadDim;
    const std::uint32_t elems_per_page = kPageTokens * elems_per_token;
    const std::uint32_t elems_per_layer = kNumPages * elems_per_page;
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    std::vector<float> out(static_cast<std::size_t>(kRows) * kQHeads * kHeadDim, 0.0f);

    for (std::uint32_t row = 0; row < kRows; ++row) {
        if (kSlots[row] >= kMaxSequences) continue;
        std::vector<std::uint32_t> keys;
        for (std::uint32_t t = 0; t <= kPositions[row]; ++t) {
            if (mask_bit(mask, row, t >> kMaskShift)) keys.push_back(t);
        }
        for (std::uint32_t head = 0; head < kQHeads; ++head) {
            const std::uint32_t kv_head = head / (kQHeads / kKvHeads);
            std::vector<float> scores(keys.size());
            float max_score = -INFINITY;
            for (std::size_t i = 0; i < keys.size(); ++i) {
                const std::uint32_t token = keys[i];
                const std::uint32_t phys =
                    kBlockTables[kSlots[row] * kBlockTableStride + token / kPageTokens];
                const std::uint64_t base =
                    static_cast<std::uint64_t>(kLayers - 1u) * elems_per_layer +
                    static_cast<std::uint64_t>(phys) * elems_per_page +
                    static_cast<std::uint64_t>(token % kPageTokens) * elems_per_token +
                    static_cast<std::uint64_t>(kv_head) * kHeadDim;
                float dot = 0.0f;
                for (std::uint32_t d = 0; d < kHeadDim; ++d) {
                    dot += bf16_to_float(q[(static_cast<std::size_t>(row) * kQHeads + head) *
                                               kHeadDim + d].data) *
                           bf16_to_float(k_pool[base + d].data);
                }
                scores[i] = dot * scale;
                max_score = std::fmax(max_score, scores[i]);
            }
            float sum = 0.0f;
            for (std::size_t i = 0; i < keys.size(); ++i)
                sum += std::exp(scores[i] - max_score);
            for (std::uint32_t d = 0; d < kHeadDim; ++d) {
                float acc = 0.0f;
                for (std::size_t i = 0; i < keys.size(); ++i) {
                    const std::uint32_t token = keys[i];
                    const std::uint32_t phys =
                        kBlockTables[kSlots[row] * kBlockTableStride + token / kPageTokens];
                    const std::uint64_t base =
                        static_cast<std::uint64_t>(kLayers - 1u) * elems_per_layer +
                        static_cast<std::uint64_t>(phys) * elems_per_page +
                        static_cast<std::uint64_t>(token % kPageTokens) * elems_per_token +
                        static_cast<std::uint64_t>(kv_head) * kHeadDim;
                    acc += std::exp(scores[i] - max_score) * bf16_to_float(v_pool[base + d].data);
                }
                out[(static_cast<std::size_t>(row) * kQHeads + head) * kHeadDim + d] =
                    sum > 0.0f ? acc / sum : 0.0f;
            }
        }
    }
    return out;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float worst = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) worst = std::fmax(worst, std::fabs(a[i] - b[i]));
    return worst;
}

}  // namespace

int main() {
    const std::uint32_t elems_per_token = kKvHeads * kHeadDim;
    const std::uint32_t elems_per_page = kPageTokens * elems_per_token;
    const std::uint32_t elems_per_layer = kNumPages * elems_per_page;

    std::mt19937 rng(20261116u);
    const std::vector<shelf::bf16_t> q = random_bf16(
        static_cast<std::size_t>(kRows) * kQHeads * kHeadDim, rng);
    const std::vector<shelf::bf16_t> k_pool =
        random_bf16(static_cast<std::size_t>(kLayers) * elems_per_layer, rng);
    const std::vector<shelf::bf16_t> v_pool =
        random_bf16(static_cast<std::size_t>(kLayers) * elems_per_layer, rng);

    std::vector<std::uint8_t> mask(static_cast<std::size_t>(kRows) * kMaskStride, 0u);
    for (std::uint32_t row = 0; row < kRows; ++row)
        for (std::uint32_t b = 0; b < kMaskStride; ++b)
            mask[static_cast<std::size_t>(row) * kMaskStride + b] = mask_byte(row, b);

    const std::vector<float> want = reference_attention(q, k_pool, v_pool, mask);

    shelf::bf16_t* d_q = nullptr;
    shelf::bf16_t* d_k = nullptr;
    shelf::bf16_t* d_v = nullptr;
    std::uint8_t* d_mask = nullptr;
    std::uint32_t* d_positions = nullptr;
    std::uint32_t* d_slots = nullptr;
    std::uint32_t* d_tables = nullptr;
    float* d_out = nullptr;
    float* d_partials = nullptr;
    hipStream_t stream = nullptr;

    check(hipMalloc(&d_q, q.size() * sizeof(shelf::bf16_t)) == hipSuccess, "malloc q");
    check(hipMalloc(&d_k, k_pool.size() * sizeof(shelf::bf16_t)) == hipSuccess, "malloc k");
    check(hipMalloc(&d_v, v_pool.size() * sizeof(shelf::bf16_t)) == hipSuccess, "malloc v");
    check(hipMalloc(&d_mask, mask.size()) == hipSuccess, "malloc mask");
    check(hipMalloc(&d_positions, sizeof(kPositions)) == hipSuccess, "malloc positions");
    check(hipMalloc(&d_slots, sizeof(kSlots)) == hipSuccess, "malloc slots");
    check(hipMalloc(&d_tables, sizeof(kBlockTables)) == hipSuccess, "malloc tables");
    check(hipMalloc(&d_out, want.size() * sizeof(float)) == hipSuccess, "malloc out");
    check(hipMalloc(&d_partials, 4u * 1024u * 1024u) == hipSuccess, "malloc partials");
    check(hipStreamCreate(&stream) == hipSuccess, "stream");

    check(hipMemcpyAsync(d_q, q.data(), q.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice,
                         stream) == hipSuccess, "copy q");
    check(hipMemcpyAsync(d_k, k_pool.data(), k_pool.size() * sizeof(shelf::bf16_t),
                         hipMemcpyHostToDevice, stream) == hipSuccess, "copy k");
    check(hipMemcpyAsync(d_v, v_pool.data(), v_pool.size() * sizeof(shelf::bf16_t),
                         hipMemcpyHostToDevice, stream) == hipSuccess, "copy v");
    check(hipMemcpyAsync(d_mask, mask.data(), mask.size(), hipMemcpyHostToDevice, stream) ==
              hipSuccess, "copy mask");
    check(hipMemcpyAsync(d_positions, kPositions, sizeof(kPositions), hipMemcpyHostToDevice,
                         stream) == hipSuccess, "copy positions");
    check(hipMemcpyAsync(d_slots, kSlots, sizeof(kSlots), hipMemcpyHostToDevice, stream) ==
              hipSuccess, "copy slots");
    check(hipMemcpyAsync(d_tables, kBlockTables, sizeof(kBlockTables), hipMemcpyHostToDevice,
                         stream) == hipSuccess, "copy tables");

    shelf::kernel::PagedAttentionCommonArgs args{};
    args.q = d_q;
    args.output = d_out;
    args.output_dtype = shelf::kernel::PagedAttentionOutputDType::F32;
    args.q_row_stride = kQHeads * kHeadDim;
    args.output_row_stride = kQHeads * kHeadDim;
    args.row_sequence_slots = d_slots;
    args.row_positions = d_positions;
    args.block_tables = d_tables;
    args.block_table_stride = kBlockTableStride;
    args.max_sequences = kMaxSequences;
    args.rows = kRows;
    args.layer = kLayers - 1u;
    args.page_tokens = kPageTokens;
    args.num_pages = kNumPages;
    args.num_attention_layers = kLayers;
    args.q_heads = kQHeads;
    args.kv_heads = kKvHeads;
    args.head_dim = kHeadDim;
    args.elems_per_token = elems_per_token;
    args.elems_per_page = elems_per_page;
    args.elems_per_layer = elems_per_layer;
    args.scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    args.block_mask = d_mask;
    args.mask_stride = kMaskStride;
    args.mask_blocks = kMaskBlocks;
    args.mask_block_shift = kMaskShift;

    std::vector<float> got(want.size(), 0.0f);

    {
        check(hipMemsetAsync(d_out, 0, got.size() * sizeof(float), stream) == hipSuccess,
              "clear out");
        check(shelf::kernel::launch_qsa_sparse_attention(args, d_k, d_v, nullptr, 1u, stream) ==
                  hipSuccess,
              "launch single");
        check(hipMemcpyAsync(got.data(), d_out, got.size() * sizeof(float),
                             hipMemcpyDeviceToHost, stream) == hipSuccess, "copy out");
        check(hipStreamSynchronize(stream) == hipSuccess, "sync single");
        const float diff = max_abs_diff(got, want);
        check(diff < 2.0e-2f, "single split matches the reference");
        std::fprintf(stderr, "single split max abs diff %.6f\n", diff);
        check(got[(static_cast<std::size_t>(3u) * kQHeads) * kHeadDim] == 0.0f,
              "invalid slot row stays zero");
    }

    {
        const std::uint32_t splits = 8u;
        check(shelf::kernel::qsa_sparse_partial_bytes(kRows, kKvHeads, splits) <=
                  4u * 1024u * 1024u,
              "partial workspace fits the test buffer");
        check(hipMemsetAsync(d_out, 0, got.size() * sizeof(float), stream) == hipSuccess,
              "clear out split");
        check(hipMemsetAsync(d_partials, 0, 4u * 1024u * 1024u, stream) == hipSuccess,
              "clear partials");
        check(shelf::kernel::launch_qsa_sparse_attention(args, d_k, d_v, d_partials, splits,
                                                         stream) == hipSuccess,
              "launch split");
        check(shelf::kernel::launch_qsa_sparse_reduce(args, d_partials, splits, stream) ==
                  hipSuccess,
              "launch reduce");
        check(hipMemcpyAsync(got.data(), d_out, got.size() * sizeof(float),
                             hipMemcpyDeviceToHost, stream) == hipSuccess, "copy out split");
        check(hipStreamSynchronize(stream) == hipSuccess, "sync split");
        const float diff = max_abs_diff(got, want);
        check(diff < 3.0e-2f, "split attention matches the reference");
        std::fprintf(stderr, "split attention max abs diff %.6f\n", diff);
    }

    // 最終ブロックに無効トークンがあるケース (visible = 1, 2, 3)。
    // 未使用スロットの K/V を NaN で埋めても、出力が NaN に汚染されないことを確認する。
    {
        const std::uint32_t short_positions[kRows] = {1u, 2u, 3u, 0u};
        check(hipMemcpyAsync(d_positions, short_positions, sizeof(short_positions),
                             hipMemcpyHostToDevice, stream) == hipSuccess,
              "copy short positions");
        std::vector<shelf::bf16_t> nan_k = k_pool;
        std::vector<shelf::bf16_t> nan_v = v_pool;
        const std::uint32_t nan_bits = 0x7FC0u;
        for (std::size_t i = 0; i < nan_k.size(); ++i) {
            nan_k[i].data = static_cast<unsigned short>(nan_bits);
            nan_v[i].data = static_cast<unsigned short>(nan_bits);
        }
        // visible トークンだけ元の値に戻す (行 0..2、block 0、各ページ先頭)。
        for (std::uint32_t row = 0; row < 3u; ++row) {
            const std::uint32_t phys = kBlockTables[kSlots[row] * kBlockTableStride];
            for (std::uint32_t t = 0; t <= short_positions[row]; ++t) {
                const std::size_t base =
                    static_cast<std::size_t>(kLayers - 1u) * elems_per_layer +
                    static_cast<std::size_t>(phys) * elems_per_page + t * elems_per_token;
                for (std::size_t e = 0; e < elems_per_token; ++e) {
                    nan_k[base + e] = k_pool[base + e];
                    nan_v[base + e] = v_pool[base + e];
                }
            }
        }
        check(hipMemcpyAsync(d_k, nan_k.data(), nan_k.size() * sizeof(shelf::bf16_t),
                             hipMemcpyHostToDevice, stream) == hipSuccess, "copy nan k");
        check(hipMemcpyAsync(d_v, nan_v.data(), nan_v.size() * sizeof(shelf::bf16_t),
                             hipMemcpyHostToDevice, stream) == hipSuccess, "copy nan v");
        check(hipMemsetAsync(d_out, 0, got.size() * sizeof(float), stream) == hipSuccess,
              "clear out short");
        check(shelf::kernel::launch_qsa_sparse_attention(args, d_k, d_v, nullptr, 1u,
                                                         stream) == hipSuccess,
              "launch short block");
        check(hipMemcpyAsync(got.data(), d_out, got.size() * sizeof(float),
                             hipMemcpyDeviceToHost, stream) == hipSuccess, "copy short out");
        check(hipStreamSynchronize(stream) == hipSuccess, "sync short");
        bool finite = true;
        for (std::uint32_t row = 0; row < 3u; ++row) {
            for (std::uint32_t h = 0; h < kQHeads * kHeadDim; ++h) {
                const float value = got[static_cast<std::size_t>(row) * kQHeads * kHeadDim + h];
                if (!std::isfinite(value)) finite = false;
            }
        }
        check(finite, "partial final block does not leak NaN");
        // 元のデータに戻す。
        check(hipMemcpyAsync(d_positions, kPositions, sizeof(kPositions),
                             hipMemcpyHostToDevice, stream) == hipSuccess, "restore positions");
        check(hipMemcpyAsync(d_k, k_pool.data(), k_pool.size() * sizeof(shelf::bf16_t),
                             hipMemcpyHostToDevice, stream) == hipSuccess, "restore k");
        check(hipMemcpyAsync(d_v, v_pool.data(), v_pool.size() * sizeof(shelf::bf16_t),
                             hipMemcpyHostToDevice, stream) == hipSuccess, "restore v");
    }

    check(shelf::kernel::launch_qsa_sparse_attention_fp8(
              args, reinterpret_cast<const std::uint8_t*>(d_k),
              reinterpret_cast<const std::uint8_t*>(d_v), nullptr, nullptr,
              nullptr, 1u, stream) == hipErrorInvalidValue,
          "fp8 sparse rejects null scales");

    check(shelf::kernel::qsa_sparse_splits(1u, kKvHeads, 4u * 1024u * 1024u) > 1u,
          "decode-sized rows split the block list");
    check(shelf::kernel::qsa_sparse_splits(256u, kKvHeads, 4u * 1024u * 1024u) == 1u,
          "prefill-sized rows do not split");
    check(shelf::kernel::qsa_sparse_splits(1u, kKvHeads, 1u) == 1u,
          "workspace limit reduces splits");

    shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    shelf::gpu::discard_cleanup_result(hipFree(d_q));
    shelf::gpu::discard_cleanup_result(hipFree(d_k));
    shelf::gpu::discard_cleanup_result(hipFree(d_v));
    shelf::gpu::discard_cleanup_result(hipFree(d_mask));
    shelf::gpu::discard_cleanup_result(hipFree(d_positions));
    shelf::gpu::discard_cleanup_result(hipFree(d_slots));
    shelf::gpu::discard_cleanup_result(hipFree(d_tables));
    shelf::gpu::discard_cleanup_result(hipFree(d_out));
    shelf::gpu::discard_cleanup_result(hipFree(d_partials));

    std::printf("qsa sparse attention: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
