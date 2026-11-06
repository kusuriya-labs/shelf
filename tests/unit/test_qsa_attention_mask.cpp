#include <shelf/kernels/optimized/attention/paged_attention.h>
#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/types.h>

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

std::uint16_t to_bf16(float value) {
    return shelf::f32_to_bf16_rne(value);
}

}  // namespace

int main() {
    constexpr std::uint32_t kQHeads = 2u;
    constexpr std::uint32_t kKvHeads = 1u;
    constexpr std::uint32_t kHeadDim = 16u;
    constexpr std::uint32_t kPageTokens = 4u;
    constexpr std::uint32_t kNumPages = 4u;
    constexpr std::uint32_t kRows = 3u;
    constexpr std::uint32_t kMaxSequences = 2u;
    constexpr std::uint32_t kBlockTableStride = 4u;
    constexpr std::uint32_t kMaskBlocks = 8u;
    constexpr std::uint32_t kMaskStride = 4u;
    constexpr std::uint32_t kBlockShift = 1u;
    const float kScale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

    const std::uint32_t q_features = kQHeads * kHeadDim;
    const std::uint32_t elems_per_token = kKvHeads * kHeadDim;
    const std::uint32_t elems_per_page = kPageTokens * elems_per_token;
    const std::uint32_t elems_per_layer = kNumPages * elems_per_page;

    const std::uint32_t positions[kRows] = {5u, 2u, 7u};
    const std::uint32_t slots[kRows] = {0u, 1u, 0u};
    const std::uint32_t block_tables[kMaxSequences * kBlockTableStride] = {
        0u, 1u, 2u, 3u,
        3u, 2u, 1u, 0u,
    };
    const std::uint8_t mask[kRows * kMaskStride] = {
        0x05u, 0u, 0u, 0u,
        0x09u, 0u, 0u, 0u,
        0x0au, 0u, 0u, 0u,
    };

    std::mt19937 rng(424242u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<std::uint16_t> q(static_cast<std::size_t>(kRows) * q_features);
    for (auto& v : q) v = to_bf16(dist(rng));
    std::vector<std::uint16_t> k_pool(elems_per_layer);
    std::vector<std::uint16_t> v_pool(elems_per_layer);
    for (auto& v : k_pool) v = to_bf16(dist(rng));
    for (auto& v : v_pool) v = to_bf16(dist(rng));

    std::vector<float> want(static_cast<std::size_t>(kRows) * q_features, 0.0f);
    for (std::uint32_t row = 0; row < kRows; ++row) {
        std::vector<std::uint32_t> keys;
        for (std::uint32_t t = 0; t < (kMaskBlocks << kBlockShift); ++t) {
            const std::uint32_t block = t >> kBlockShift;
            const bool marked =
                ((mask[row * kMaskStride + (block >> 3u)] >> (block & 7u)) & 1u) != 0u;
            if (t <= positions[row] && marked) keys.push_back(t);
        }
        check(!keys.empty(), "block mask leaves at least one visible key");
        for (std::uint32_t head = 0; head < kQHeads; ++head) {
            const std::uint32_t kv_head = 0u;
            std::vector<float> scores(keys.size());
            float max_score = -INFINITY;
            for (std::size_t i = 0; i < keys.size(); ++i) {
                const std::uint32_t token = keys[i];
                const std::uint32_t page =
                    block_tables[slots[row] * kBlockTableStride + token / kPageTokens];
                const std::uint32_t off = token % kPageTokens;
                const std::uint64_t base =
                    static_cast<std::uint64_t>(page) * elems_per_page +
                    static_cast<std::uint64_t>(off) * elems_per_token +
                    kv_head * kHeadDim;
                float dot = 0.0f;
                for (std::uint32_t d = 0; d < kHeadDim; ++d) {
                    dot += bf16_to_float(q[static_cast<std::size_t>(row) * q_features +
                                           head * kHeadDim + d]) *
                           bf16_to_float(k_pool[base + d]);
                }
                scores[i] = dot * kScale;
                max_score = std::fmax(max_score, scores[i]);
            }
            float sum = 0.0f;
            std::vector<float> acc(kHeadDim, 0.0f);
            for (std::size_t i = 0; i < keys.size(); ++i) {
                const float weight = std::exp(scores[i] - max_score);
                sum += weight;
                const std::uint32_t token = keys[i];
                const std::uint32_t page =
                    block_tables[slots[row] * kBlockTableStride + token / kPageTokens];
                const std::uint32_t off = token % kPageTokens;
                const std::uint64_t base =
                    static_cast<std::uint64_t>(page) * elems_per_page +
                    static_cast<std::uint64_t>(off) * elems_per_token +
                    kv_head * kHeadDim;
                for (std::uint32_t d = 0; d < kHeadDim; ++d)
                    acc[d] += weight * bf16_to_float(v_pool[base + d]);
            }
            for (std::uint32_t d = 0; d < kHeadDim; ++d)
                want[static_cast<std::size_t>(row) * q_features + head * kHeadDim + d] =
                    acc[d] / sum;
        }
    }

    void* d_q = nullptr;
    void* d_out = nullptr;
    void* d_k = nullptr;
    void* d_v = nullptr;
    void* d_slots = nullptr;
    void* d_positions = nullptr;
    void* d_blocks = nullptr;
    void* d_mask = nullptr;
    bool ok = true;
    ok = ok && hipMalloc(&d_q, q.size() * 2) == hipSuccess;
    ok = ok && hipMalloc(&d_out, want.size() * sizeof(float)) == hipSuccess;
    ok = ok && hipMalloc(&d_k, k_pool.size() * 2) == hipSuccess;
    ok = ok && hipMalloc(&d_v, v_pool.size() * 2) == hipSuccess;
    ok = ok && hipMalloc(&d_slots, kRows * sizeof(std::uint32_t)) == hipSuccess;
    ok = ok && hipMalloc(&d_positions, kRows * sizeof(std::uint32_t)) == hipSuccess;
    ok = ok && hipMalloc(&d_blocks, sizeof(block_tables)) == hipSuccess;
    ok = ok && hipMalloc(&d_mask, sizeof(mask)) == hipSuccess;
    check(ok, "device allocations");
    if (!ok) return 1;

    check(hipMemcpy(d_q, q.data(), q.size() * 2, hipMemcpyHostToDevice) == hipSuccess,
          "copy q");
    check(hipMemcpy(d_k, k_pool.data(), k_pool.size() * 2, hipMemcpyHostToDevice) ==
              hipSuccess,
          "copy k");
    check(hipMemcpy(d_v, v_pool.data(), v_pool.size() * 2, hipMemcpyHostToDevice) ==
              hipSuccess,
          "copy v");
    check(hipMemcpy(d_slots, slots, sizeof(slots), hipMemcpyHostToDevice) == hipSuccess,
          "copy slots");
    check(hipMemcpy(d_positions, positions, sizeof(positions), hipMemcpyHostToDevice) ==
              hipSuccess,
          "copy positions");
    check(hipMemcpy(d_blocks, block_tables, sizeof(block_tables),
                    hipMemcpyHostToDevice) == hipSuccess,
          "copy block table");
    check(hipMemcpy(d_mask, mask, sizeof(mask), hipMemcpyHostToDevice) == hipSuccess,
          "copy mask");
    check(hipMemset(d_out, 0, want.size() * sizeof(float)) == hipSuccess, "zero out");

    shelf::kernel::PagedAttentionCommonArgs args;
    args.q = static_cast<const shelf::bf16_t*>(d_q);
    args.output = d_out;
    args.output_dtype = shelf::kernel::PagedAttentionOutputDType::F32;
    args.q_row_stride = q_features;
    args.output_row_stride = q_features;
    args.row_sequence_slots = static_cast<const std::uint32_t*>(d_slots);
    args.row_positions = static_cast<const std::uint32_t*>(d_positions);
    args.block_tables = static_cast<const std::uint32_t*>(d_blocks);
    args.block_table_stride = kBlockTableStride;
    args.max_sequences = kMaxSequences;
    args.rows = kRows;
    args.layer = 0u;
    args.page_tokens = kPageTokens;
    args.num_pages = kNumPages;
    args.num_attention_layers = 1u;
    args.q_heads = kQHeads;
    args.kv_heads = kKvHeads;
    args.head_dim = kHeadDim;
    args.elems_per_token = elems_per_token;
    args.elems_per_page = elems_per_page;
    args.elems_per_layer = elems_per_layer;
    args.scale = kScale;
    args.block_mask = static_cast<const std::uint8_t*>(d_mask);
    args.mask_stride = kMaskStride;
    args.mask_blocks = kMaskBlocks;
    args.mask_block_shift = kBlockShift;

    const hipError_t error = shelf::kernel::launch_attention_paged_bf16(
        args, static_cast<const shelf::bf16_t*>(d_k),
        static_cast<const shelf::bf16_t*>(d_v), nullptr);
    check(error == hipSuccess, "masked attention launch");
    check(hipStreamSynchronize(nullptr) == hipSuccess, "masked attention sync");

    std::vector<float> got(want.size());
    check(hipMemcpy(got.data(), d_out, got.size() * sizeof(float),
                    hipMemcpyDeviceToHost) == hipSuccess,
          "read output");
    double max_error = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i)
        max_error = std::max<double>(max_error,
                                     std::fabs(static_cast<double>(got[i]) - want[i]));
    std::fprintf(stderr, "masked attention max_abs=%g\n", max_error);
    check(max_error <= 3.0e-3, "masked attention matches CPU reference");

    (void)hipFree(d_q);
    (void)hipFree(d_out);
    (void)hipFree(d_k);
    (void)hipFree(d_v);
    (void)hipFree(d_slots);
    (void)hipFree(d_positions);
    (void)hipFree(d_blocks);
    (void)hipFree(d_mask);

    std::fprintf(stderr, failures == 0 ? "PASS\n" : "FAILURES=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
