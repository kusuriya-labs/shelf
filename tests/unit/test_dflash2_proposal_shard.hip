#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/tensor.h>
#include <shelf/core/memory/types.h>
#include <shelf/kernels/dflash2/candidate_selector.h>
#include <shelf/kernels/optimized/linear/lm_head_int8.h>
#include <shelf/runtime/dflash2/proposal_shard.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

using shelf::runtime::dflash2::ProposalShardBuffers;

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

template <typename T>
static bool alloc(
    shelf::gpu::GpuArena& arena,
    const std::vector<std::size_t>& shape,
    shelf::gpu::Tensor& out) {
    std::size_t count = 1;
    for (std::size_t d : shape) count *= d;
    auto view = arena.allocate_aligned(count * sizeof(T), 128);
    if (!view.ok()) return false;
    auto backing = view.release();
    std::vector<std::size_t> strides(shape.size());
    std::size_t stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }
    auto t = shelf::gpu::Tensor::create<T>(backing, shape, strides);
    if (!t.ok()) return false;
    out = t.release();
    return true;
}

struct Outputs {
    std::vector<int32_t> ids;
    std::vector<float> logits;
    std::vector<float> stats;
};

static Outputs read(const ProposalShardBuffers& buffers, uint32_t rows, uint32_t top_k) {
    Outputs out;
    out.ids.resize(static_cast<std::size_t>(rows) * top_k);
    out.logits.resize(static_cast<std::size_t>(rows) * top_k);
    out.stats.resize(static_cast<std::size_t>(rows) * 2u);
    (void)hipMemcpy(out.ids.data(), buffers.topk_ids.data<int32_t>(),
                    out.ids.size() * sizeof(int32_t), hipMemcpyDeviceToHost);
    (void)hipMemcpy(out.logits.data(), buffers.topk_logits.data<float>(),
                    out.logits.size() * sizeof(float), hipMemcpyDeviceToHost);
    (void)hipMemcpy(out.stats.data(), buffers.stats.data<float>(),
                    out.stats.size() * sizeof(float), hipMemcpyDeviceToHost);
    return out;
}

int main() {
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    auto arena_result = shelf::gpu::GpuArena::create(0, 512ull * 1024ull * 1024ull);
    if (!arena_result.ok()) {
        std::printf("SKIP: arena (%s)\n", arena_result.status().message().c_str());
        return 77;
    }
    shelf::gpu::GpuArena arena = arena_result.release();
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        std::printf("FAIL hipStreamCreate\n");
        return 1;
    }

    const uint32_t vocab = 4096u;
    const uint32_t k = 128u;
    const uint32_t hidden = k;
    const uint32_t rows = 4u;
    const uint32_t top_k = 16u;
    const uint32_t shard_vocab = vocab / 2u;

    std::mt19937 rng(11u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<shelf::bf16_t> host_weight(static_cast<std::size_t>(vocab) * k);
    for (auto& w : host_weight) w.data = shelf::f32_to_bf16_rne(dist(rng));
    std::vector<shelf::bf16_t> host_hidden(static_cast<std::size_t>(rows) * hidden);
    for (auto& h : host_hidden) h.data = shelf::f32_to_bf16_rne(dist(rng));

    shelf::gpu::Tensor device_weight;
    shelf::gpu::Tensor device_hidden;
    shelf::gpu::Tensor codes;
    shelf::gpu::Tensor scales;
    if (!alloc<shelf::bf16_t>(arena, {host_weight.size()}, device_weight) ||
        !alloc<shelf::bf16_t>(arena, {host_hidden.size()}, device_hidden) ||
        !alloc<uint8_t>(arena, {vocab, k}, codes) ||
        !alloc<shelf::fp16_t>(arena, {vocab, k / 32u}, scales)) {
        std::printf("FAIL alloc weight\n");
        return 1;
    }
    (void)hipMemcpy(device_weight.data<void>(), host_weight.data(),
                    host_weight.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice);
    (void)hipMemcpy(device_hidden.data<void>(), host_hidden.data(),
                    host_hidden.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice);
    check(shelf::kernel::launch_quantize_bf16_to_int8_w32(
              device_weight.data<shelf::bf16_t>(), k, codes.data<uint8_t>(),
              scales.data<shelf::fp16_t>(), vocab, k, stream) == hipSuccess,
          "quantize head");

    shelf::weights::MatrixWeight full_head;
    full_head.encoding = shelf::weights::MatrixEncoding::Int8W32;
    full_head.rows = vocab;
    full_head.cols = k;
    full_head.k_padded = k;
    full_head.codes = codes;
    full_head.scales = scales;

    auto full_buffers_result =
        shelf::runtime::dflash2::create_proposal_shard_buffers(arena, rows, vocab, top_k);
    check(full_buffers_result.ok(), "full buffers");
    if (!full_buffers_result.ok()) return 1;
    ProposalShardBuffers full = full_buffers_result.release();
    check(shelf::runtime::dflash2::proposal_shard_compute(
              full_head, full, device_hidden.data<shelf::bf16_t>(), hidden, rows, 0u,
              stream).ok(),
          "full shard compute");
    const Outputs reference = read(full, rows, top_k);

    auto shard_a_result = shelf::runtime::dflash2::create_proposal_shard_buffers(
        arena, rows, shard_vocab, top_k);
    auto shard_b_result = shelf::runtime::dflash2::create_proposal_shard_buffers(
        arena, rows, shard_vocab, top_k);
    check(shard_a_result.ok() && shard_b_result.ok(), "shard buffers");
    if (!shard_a_result.ok() || !shard_b_result.ok()) return 1;
    ProposalShardBuffers shard_a = shard_a_result.release();
    ProposalShardBuffers shard_b = shard_b_result.release();

    auto codes_a = codes.slice(0, 0u, shard_vocab);
    auto codes_b = codes.slice(0, shard_vocab, shard_vocab);
    auto scales_a = scales.slice(0, 0u, shard_vocab);
    auto scales_b = scales.slice(0, shard_vocab, shard_vocab);
    check(codes_a.ok() && codes_b.ok() && scales_a.ok() && scales_b.ok(), "slice head");
    if (!codes_a.ok() || !codes_b.ok() || !scales_a.ok() || !scales_b.ok()) return 1;

    shelf::weights::MatrixWeight head_a = full_head;
    head_a.rows = shard_vocab;
    head_a.codes = codes_a.value();
    head_a.scales = scales_a.value();
    shelf::weights::MatrixWeight head_b = full_head;
    head_b.rows = shard_vocab;
    head_b.codes = codes_b.value();
    head_b.scales = scales_b.value();

    check(shelf::runtime::dflash2::proposal_shard_compute(
              head_a, shard_a, device_hidden.data<shelf::bf16_t>(), hidden, rows, 0u,
              stream).ok(),
          "shard a compute");
    check(shelf::runtime::dflash2::proposal_shard_compute(
              head_b, shard_b, device_hidden.data<shelf::bf16_t>(), hidden, rows,
              shard_vocab, stream).ok(),
          "shard b compute");
    const Outputs part_a = read(shard_a, rows, top_k);
    const Outputs part_b = read(shard_b, rows, top_k);

    const std::size_t rank_stride = static_cast<std::size_t>(rows) * top_k;
    std::vector<int32_t> host_ids(rank_stride * 2u);
    std::vector<float> host_logits(rank_stride * 2u);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t j = 0; j < top_k; ++j) {
            const std::size_t slot = static_cast<std::size_t>(r) * top_k + j;
            host_ids[slot] = part_a.ids[slot];
            host_logits[slot] = part_a.logits[slot];
            host_ids[rank_stride + slot] = part_b.ids[slot];
            host_logits[rank_stride + slot] = part_b.logits[slot];
        }
    }
    shelf::gpu::Tensor candidate_ids;
    shelf::gpu::Tensor candidate_logits;
    shelf::gpu::Tensor merged_ids;
    shelf::gpu::Tensor merged_logits;
    if (!alloc<int32_t>(arena, {rows, top_k * 2u}, candidate_ids) ||
        !alloc<float>(arena, {rows, top_k * 2u}, candidate_logits) ||
        !alloc<int32_t>(arena, {rows, top_k}, merged_ids) ||
        !alloc<float>(arena, {rows, top_k}, merged_logits)) {
        std::printf("FAIL alloc merge\n");
        return 1;
    }
    (void)hipMemcpy(candidate_ids.data<void>(), host_ids.data(),
                    host_ids.size() * sizeof(int32_t), hipMemcpyHostToDevice);
    (void)hipMemcpy(candidate_logits.data<void>(), host_logits.data(),
                    host_logits.size() * sizeof(float), hipMemcpyHostToDevice);
    check(shelf::runtime::dflash2::proposal_merge_candidates(
              candidate_ids.data<int32_t>(), candidate_logits.data<float>(), 2u,
              rank_stride, rows, top_k, merged_ids.data<int32_t>(),
              merged_logits.data<float>(), stream).ok(),
          "merge candidates");
    std::vector<int32_t> got_ids(static_cast<std::size_t>(rows) * top_k);
    std::vector<float> got_logits(static_cast<std::size_t>(rows) * top_k);
    (void)hipMemcpy(got_ids.data(), merged_ids.data<int32_t>(),
                    got_ids.size() * sizeof(int32_t), hipMemcpyDeviceToHost);
    (void)hipMemcpy(got_logits.data(), merged_logits.data<float>(),
                    got_logits.size() * sizeof(float), hipMemcpyDeviceToHost);

    std::size_t id_mismatch = 0;
    std::size_t logit_mismatch = 0;
    for (std::size_t i = 0; i < got_ids.size(); ++i) {
        if (got_ids[i] != reference.ids[i]) ++id_mismatch;
        if (got_logits[i] != reference.logits[i]) ++logit_mismatch;
    }
    check(id_mismatch == 0, "merged ids match the full head (" +
                                std::to_string(id_mismatch) + ")");
    check(logit_mismatch == 0, "merged logits match the full head (" +
                                   std::to_string(logit_mismatch) + ")");

    std::vector<float> host_stats(static_cast<std::size_t>(rows) * 2u * 2u);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t k2 = 0; k2 < 2u; ++k2) {
            host_stats[static_cast<std::size_t>(r) * 2u + k2] =
                part_a.stats[static_cast<std::size_t>(r) * 2u + k2];
            host_stats[static_cast<std::size_t>(rows) * 2u +
                       static_cast<std::size_t>(r) * 2u + k2] =
                part_b.stats[static_cast<std::size_t>(r) * 2u + k2];
        }
    }
    shelf::gpu::Tensor device_stats;
    shelf::gpu::Tensor device_combined;
    if (!alloc<float>(arena, {rows * 2u * 2u}, device_stats) ||
        !alloc<float>(arena, {rows, 2u}, device_combined)) {
        std::printf("FAIL alloc stats\n");
        return 1;
    }
    (void)hipMemcpy(device_stats.data<void>(), host_stats.data(),
                    host_stats.size() * sizeof(float), hipMemcpyHostToDevice);
    check(shelf::runtime::dflash2::proposal_combine_denominators(
              device_stats.data<float>(), 2u, static_cast<uint32_t>(rows * 2u), rows,
              device_combined.data<float>(), stream).ok(),
          "combine denominators");
    std::vector<float> combined(static_cast<std::size_t>(rows) * 2u);
    (void)hipMemcpy(combined.data(), device_combined.data<void>(),
                    combined.size() * sizeof(float), hipMemcpyDeviceToHost);
    std::vector<float> merged_max(rows);
    std::vector<float> merged_denominator(rows);
    for (uint32_t r = 0; r < rows; ++r) {
        merged_max[r] = combined[static_cast<std::size_t>(r) * 2u];
        merged_denominator[r] = combined[static_cast<std::size_t>(r) * 2u + 1u];
    }
    bool stats_ok = true;
    for (uint32_t r = 0; r < rows; ++r) {
        if (merged_max[r] != reference.stats[static_cast<std::size_t>(r) * 2u])
            stats_ok = false;
        const float want = reference.stats[static_cast<std::size_t>(r) * 2u + 1u];
        const float got = merged_denominator[r];
        if (std::fabs(got - want) > 1e-4f * want) stats_ok = false;
    }
    check(stats_ok, "sharded denominator matches the full vocabulary");

    std::vector<float> normalizers(static_cast<std::size_t>(rows) * 2u);
    for (uint32_t r = 0; r < rows; ++r) {
        normalizers[static_cast<std::size_t>(r) * 2u] = merged_max[r];
        normalizers[static_cast<std::size_t>(r) * 2u + 1u] = merged_denominator[r];
    }
    shelf::gpu::Tensor device_normalizers;
    shelf::gpu::Tensor probabilities;
    if (!alloc<float>(arena, {rows, 2u}, device_normalizers) ||
        !alloc<float>(arena,
                      {rows, sizeof(shelf::kernel::DFlash2TreeProbabilities) / sizeof(float)},
                      probabilities)) {
        std::printf("FAIL alloc probabilities\n");
        return 1;
    }
    (void)hipMemcpy(device_normalizers.data<void>(), normalizers.data(),
                    normalizers.size() * sizeof(float), hipMemcpyHostToDevice);
    check(shelf::kernel::launch_dflash2_topk_probabilities(
              merged_ids.data<int32_t>(), merged_logits.data<float>(),
              device_normalizers.data<float>(),
              reinterpret_cast<shelf::kernel::DFlash2TreeProbabilities*>(
                  probabilities.data<float>()),
              rows, top_k, stream) == hipSuccess,
          "probability launch");
    std::vector<shelf::kernel::DFlash2TreeProbabilities> host_probabilities(rows);
    (void)hipMemcpy(host_probabilities.data(), probabilities.data<void>(),
                    host_probabilities.size() * sizeof(shelf::kernel::DFlash2TreeProbabilities),
                    hipMemcpyDeviceToHost);
    bool probability_ok = true;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t j = 0; j < shelf::kernel::kDflash2TreeRetained; ++j) {
            const float want = std::exp(
                got_logits[static_cast<std::size_t>(r) * top_k + j] - merged_max[r]) /
                merged_denominator[r];
            const float got = host_probabilities[r].probabilities[j];
            if (std::fabs(got - want) > 1e-5f * want) probability_ok = false;
        }
    }
    check(probability_ok, "top-k probabilities use the full-vocabulary denominator");

    shelf::gpu::Tensor full_probabilities;
    if (!alloc<float>(
            arena,
            {rows, sizeof(shelf::kernel::DFlash2TreeProbabilities) / sizeof(float)},
            full_probabilities)) {
        std::printf("FAIL alloc full probabilities\n");
        return 1;
    }
    check(shelf::kernel::launch_dflash2_tree_probabilities(
              full.logits.data<float>(), full.topk_ids.data<int32_t>(),
              full.topk_logits.data<float>(),
              reinterpret_cast<shelf::kernel::DFlash2TreeProbabilities*>(
                  full_probabilities.data<float>()),
              rows, vocab, top_k, stream) == hipSuccess,
          "full probability launch");
    std::vector<shelf::kernel::DFlash2TreeProbabilities> host_full(rows);
    (void)hipMemcpy(host_full.data(), full_probabilities.data<void>(),
                    host_full.size() * sizeof(shelf::kernel::DFlash2TreeProbabilities),
                    hipMemcpyDeviceToHost);
    bool sharded_matches_full = true;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t j = 0; j < shelf::kernel::kDflash2TreeRetained; ++j) {
            if (host_probabilities[r].ids[j] != host_full[r].ids[j]) sharded_matches_full = false;
            const float want = host_full[r].probabilities[j];
            const float got = host_probabilities[r].probabilities[j];
            if (std::fabs(got - want) > 1e-6f) sharded_matches_full = false;
        }
        if (std::fabs(host_probabilities[r].residual - host_full[r].residual) > 1e-6f)
            sharded_matches_full = false;
    }
    check(sharded_matches_full,
          "sharded probabilities match the full-vocabulary tree probabilities");

    (void)hipStreamSynchronize(stream);
    (void)hipStreamDestroy(stream);
    (void)arena.shutdown();
    if (g_fail != 0) {
        std::printf("dflash2 proposal shard: failed=%d\n", g_fail);
        return 1;
    }
    std::printf("dflash2 proposal shard: ok\n");
    return 0;
}
