#include <shelf/core/memory/types.h>
#include <shelf/kernels/optimized/qsa.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    } else {
        std::fprintf(stderr, "ok: %s\n", what.c_str());
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

std::vector<std::int32_t> reference_select(const std::vector<float>& scores,
                                           std::uint32_t visible, std::uint32_t top_k) {
    std::vector<std::uint32_t> order(visible);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) {
        if (scores[a] != scores[b]) return scores[a] > scores[b];
        return a < b;
    });
    const std::uint32_t take = std::min<std::uint32_t>(visible, top_k);
    std::vector<std::int32_t> selected(take);
    for (std::uint32_t i = 0; i < take; ++i) selected[i] = static_cast<std::int32_t>(order[i]);
    return selected;
}

std::vector<std::int32_t> selected_row(const std::vector<std::int32_t>& values,
                                       std::uint32_t row, std::uint32_t top_k) {
    std::vector<std::int32_t> out;
    for (std::uint32_t i = 0; i < top_k; ++i) {
        const std::int32_t value = values[static_cast<std::size_t>(row) * top_k + i];
        if (value >= 0) out.push_back(value);
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::uint32_t mask_bit(const std::vector<std::uint8_t>& mask, std::uint32_t row_stride_bytes,
                       std::uint32_t row, std::uint32_t block) {
    const std::size_t byte = static_cast<std::size_t>(row) * row_stride_bytes + (block >> 3u);
    return (mask[byte] >> (block & 7u)) & 1u;
}

struct Scenario {
    const char* label;
    std::uint32_t blocks;
    std::uint32_t compress;
    std::uint32_t heads;
    std::uint32_t top_k;
    std::vector<std::uint32_t> positions;
    std::vector<std::uint32_t> row_slots;
    std::uint32_t max_sequences;
};

std::vector<float> reference_scores(const std::vector<shelf::bf16_t>& q,
                                    const std::vector<shelf::bf16_t>& pooled,
                                    const Scenario& s, std::uint32_t row,
                                    std::uint32_t visible) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(shelf::kernel::kQsaHeadDim));
    std::vector<float> scores(visible, 0.0f);
    for (std::uint32_t block = 0; block < visible; ++block) {
        const shelf::bf16_t* key =
            &pooled[static_cast<std::size_t>(s.row_slots[row]) * 2u * s.blocks *
                        shelf::kernel::kQsaHeadDim +
                    static_cast<std::size_t>(1u) * s.blocks * shelf::kernel::kQsaHeadDim +
                    static_cast<std::size_t>(block) * shelf::kernel::kQsaHeadDim];
        float total = 0.0f;
        for (std::uint32_t head = 0; head < s.heads; ++head) {
            const shelf::bf16_t* query =
                &q[static_cast<std::size_t>(row) * s.heads * shelf::kernel::kQsaHeadDim +
                   static_cast<std::size_t>(head) * shelf::kernel::kQsaHeadDim];
            float dot = 0.0f;
            for (std::uint32_t d = 0; d < shelf::kernel::kQsaHeadDim; ++d) {
                dot += bf16_to_float(query[d].data) * bf16_to_float(key[d].data);
            }
            total += std::max(dot, 0.0f);
        }
        scores[block] = total * scale;
    }
    return scores;
}

void run_scenario(const Scenario& s, bool split = false, std::uint32_t tiles = 1u) {
    constexpr std::uint32_t kHeadDim = shelf::kernel::kQsaHeadDim;
    constexpr std::uint32_t kStates = 2u;
    constexpr std::uint32_t kPooledState = 1u;
    const std::uint32_t rows = static_cast<std::uint32_t>(s.positions.size());
    const std::uint32_t mask_row_bytes = ((s.blocks + 31u) / 32u) * 4u;
    std::uint32_t half = shelf::kernel::kQsaSelectChunk;
    while (half < s.top_k) half <<= 1u;
    const std::uint32_t state_count = half * 2u;
    const std::string label = std::string(s.label) + (split ? " [split]" : "");

    std::mt19937 rng(20261106u);
    const auto q = random_bf16(rows * s.heads * kHeadDim, rng);
    const auto pooled = random_bf16(s.max_sequences * kStates * s.blocks * kHeadDim, rng);

    void* d_q = nullptr;
    void* d_pooled = nullptr;
    void* d_positions = nullptr;
    void* d_slots = nullptr;
    void* d_selected = nullptr;
    void* d_mask = nullptr;
    void* d_scores = nullptr;
    void* d_state_scores = nullptr;
    void* d_state_indices = nullptr;
    bool ok = hipMalloc(&d_q, q.size() * sizeof(shelf::bf16_t)) == hipSuccess;
    ok = ok && hipMalloc(&d_pooled, pooled.size() * sizeof(shelf::bf16_t)) == hipSuccess;
    ok = ok && hipMalloc(&d_positions, s.positions.size() * sizeof(std::uint32_t)) == hipSuccess;
    ok = ok && hipMalloc(&d_slots, s.row_slots.size() * sizeof(std::uint32_t)) == hipSuccess;
    ok = ok && hipMalloc(&d_selected, rows * s.top_k * sizeof(std::int32_t)) == hipSuccess;
    ok = ok && hipMalloc(&d_mask, rows * mask_row_bytes) == hipSuccess;
    if (split) {
        ok = ok && hipMalloc(&d_scores,
                             static_cast<std::size_t>(rows) *
                                 shelf::kernel::kQsaScoreBlockTile * sizeof(float)) == hipSuccess;
        ok = ok && hipMalloc(&d_state_scores,
                             static_cast<std::size_t>(rows) * state_count * sizeof(float)) == hipSuccess;
        ok = ok && hipMalloc(&d_state_indices,
                             static_cast<std::size_t>(rows) * state_count * sizeof(std::int32_t)) == hipSuccess;
    }
    ok = ok && hipMemcpy(d_q, q.data(), q.size() * sizeof(shelf::bf16_t),
                         hipMemcpyHostToDevice) == hipSuccess;
    ok = ok && hipMemcpy(d_pooled, pooled.data(), pooled.size() * sizeof(shelf::bf16_t),
                         hipMemcpyHostToDevice) == hipSuccess;
    ok = ok && hipMemcpy(d_positions, s.positions.data(), s.positions.size() * sizeof(std::uint32_t),
                         hipMemcpyHostToDevice) == hipSuccess;
    ok = ok && hipMemcpy(d_slots, s.row_slots.data(),
                         s.row_slots.size() * sizeof(std::uint32_t),
                         hipMemcpyHostToDevice) == hipSuccess;
    ok = ok && hipMemset(d_mask, 0, rows * mask_row_bytes) == hipSuccess;
    check(ok, label + ": device setup");
    if (!ok) {
        hipFree(d_q);
        hipFree(d_pooled);
        hipFree(d_positions);
        hipFree(d_slots);
        hipFree(d_selected);
        hipFree(d_mask);
        hipFree(d_scores);
        hipFree(d_state_scores);
        hipFree(d_state_indices);
        return;
    }

    shelf::kernel::QsaSelectBatchArgs args;
    args.q = static_cast<const shelf::bf16_t*>(d_q);
    args.pooled_base = static_cast<const shelf::bf16_t*>(d_pooled);
    args.selected = static_cast<std::int32_t*>(d_selected);
    args.block_mask = static_cast<std::uint8_t*>(d_mask);
    args.row_sequence_slots = static_cast<const std::uint32_t*>(d_slots);
    args.positions = static_cast<const std::uint32_t*>(d_positions);
    args.rows = rows;
    args.blocks = s.blocks;
    args.n_heads = s.heads;
    args.head_dim = kHeadDim;
    args.compress_ratio = s.compress;
    args.state_index = kPooledState;
    args.max_sequences = s.max_sequences;
    args.top_k = s.top_k;
    args.q_row_stride = s.heads * kHeadDim;
    args.selected_row_stride = s.top_k;
    args.mask_row_bytes = mask_row_bytes;
    args.pooled_slot_stride = static_cast<std::uint64_t>(kStates) * s.blocks * kHeadDim;
    args.pooled_layer_stride = static_cast<std::uint64_t>(s.blocks) * kHeadDim;
    args.scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

    auto run_kernel = [&]() -> hipError_t {
        if (!split) return shelf::kernel::launch_qsa_select_batch(args, nullptr);
        if (s.heads != 4u) return hipErrorInvalidValue;
        shelf::kernel::QsaScoreArgs sa;
        sa.q = static_cast<const shelf::bf16_t*>(d_q);
        sa.pooled_base = static_cast<const shelf::bf16_t*>(d_pooled);
        sa.scores = static_cast<float*>(d_scores);
        sa.row_sequence_slots = static_cast<const std::uint32_t*>(d_slots);
        sa.rows = rows;
        sa.row_begin = 0u;
        sa.block_begin = 0u;
        sa.blocks = s.blocks;
        sa.score_row_stride = shelf::kernel::kQsaScoreBlockTile;
        sa.n_heads = s.heads;
        sa.head_dim = kHeadDim;
        sa.compress_ratio = s.compress;
        sa.state_index = kPooledState;
        sa.max_sequences = s.max_sequences;
        sa.q_row_stride = s.heads * kHeadDim;
        sa.pooled_slot_stride = static_cast<std::uint64_t>(kStates) * s.blocks * kHeadDim;
        sa.pooled_layer_stride = static_cast<std::uint64_t>(s.blocks) * kHeadDim;
        sa.scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
        shelf::kernel::QsaTopkArgs ta;
        ta.scores = static_cast<const float*>(d_scores);
        ta.state_scores = static_cast<float*>(d_state_scores);
        ta.state_indices = static_cast<std::int32_t*>(d_state_indices);
        ta.block_mask = static_cast<std::uint8_t*>(d_mask);
        ta.selected = static_cast<std::int32_t*>(d_selected);
        ta.positions = static_cast<const std::uint32_t*>(d_positions);
        ta.rows = rows;
        ta.block_begin = 0u;
        ta.block_count = s.blocks;
        ta.score_row_stride = shelf::kernel::kQsaScoreBlockTile;
        ta.blocks = s.blocks;
        ta.compress_ratio = s.compress;
        ta.top_k = s.top_k;
        ta.mask_row_bytes = mask_row_bytes;
        ta.selected_row_stride = s.top_k;
        ta.first = 1u;
        ta.last = 1u;
        const std::uint32_t tile_blocks = (s.blocks + tiles - 1u) / tiles;
        for (std::uint32_t begin = 0u; begin < s.blocks; begin += tile_blocks) {
            const std::uint32_t count = std::min(tile_blocks, s.blocks - begin);
            sa.block_begin = begin;
            sa.blocks = count;
            hipError_t error = shelf::kernel::launch_qsa_score_wmma(sa, nullptr);
            if (error != hipSuccess) return error;
            ta.block_begin = begin;
            ta.block_count = count;
            ta.first = begin == 0u ? 1u : 0u;
            ta.last = begin + count >= s.blocks ? 1u : 0u;
            error = shelf::kernel::launch_qsa_topk(ta, nullptr);
            if (error != hipSuccess) return error;
        }
        return hipSuccess;
    };
    check(run_kernel() == hipSuccess, label + ": select launch");
    check(hipDeviceSynchronize() == hipSuccess, label + ": select sync");

    std::vector<std::int32_t> got_selected(rows * s.top_k, -2);
    std::vector<std::uint8_t> got_mask(rows * mask_row_bytes, 0u);
    ok = hipMemcpy(got_selected.data(), d_selected, got_selected.size() * sizeof(std::int32_t),
                   hipMemcpyDeviceToHost) == hipSuccess;
    ok = ok && hipMemcpy(got_mask.data(), d_mask, got_mask.size(),
                         hipMemcpyDeviceToHost) == hipSuccess;
    check(ok, label + ": read back");

    for (std::uint32_t row = 0; row < rows; ++row) {
        const std::uint32_t visible =
            std::min<std::uint32_t>(s.blocks, (s.positions[row] + 1u) / s.compress);
        if (s.row_slots[row] >= s.max_sequences) {
            check(std::all_of(got_selected.begin() + row * s.top_k,
                              got_selected.begin() + (row + 1u) * s.top_k,
                              [](std::int32_t v) { return v == -1; }),
                  label + ": row " + std::to_string(row) +
                      " with an invalid slot selects nothing");
            bool clear = true;
            for (std::uint32_t b = 0; b < mask_row_bytes * 8u; ++b) {
                if (mask_bit(got_mask, mask_row_bytes, row, b) != 0u) clear = false;
            }
            check(clear, label + ": row " + std::to_string(row) +
                             " with an invalid slot leaves the bitmap clear");
            continue;
        }
        const auto scores = reference_scores(q, pooled, s, row, visible);
        auto want = reference_select(scores, visible, s.top_k);
        std::sort(want.begin(), want.end());
        const auto got = selected_row(got_selected, row, s.top_k);
        check(got == want, label + ": row " + std::to_string(row) +
                               " visible " + std::to_string(visible) +
                               " matches the CPU reference top-k");

        const std::uint32_t take = std::min<std::uint32_t>(visible, s.top_k);
        std::uint32_t padding = 0u;
        for (std::uint32_t i = 0; i < s.top_k; ++i) {
            if (got_selected[row * s.top_k + i] == -1) ++padding;
        }
        check(padding == s.top_k - take,
              label + ": row " + std::to_string(row) +
                  " pads unused entries with -1");

        const std::uint32_t tail = (s.positions[row] + 1u) / s.compress;
        std::vector<std::int32_t> visible_blocks = want;
        if (tail < s.blocks) visible_blocks.push_back(static_cast<std::int32_t>(tail));
        std::sort(visible_blocks.begin(), visible_blocks.end());
        bool mask_matches = true;
        std::uint32_t set_bits = 0u;
        for (std::uint32_t b = 0; b < mask_row_bytes * 8u; ++b) {
            const bool in_range = b < s.blocks;
            const bool bit = mask_bit(got_mask, mask_row_bytes, row, b) != 0u;
            const bool expected =
                in_range && std::binary_search(visible_blocks.begin(), visible_blocks.end(),
                                               static_cast<std::int32_t>(b));
            if (bit != expected) mask_matches = false;
            if (bit) ++set_bits;
        }
        check(mask_matches, label + ": row " + std::to_string(row) +
                                " bitmap marks the selection, the partial tail block and "
                                "nothing past the last block");
        check(set_bits == static_cast<std::uint32_t>(visible_blocks.size()),
              label + ": row " + std::to_string(row) +
                  " bitmap bit count matches the selection plus the tail block");
    }

    {
        std::vector<shelf::bf16_t> zero_q(q.size(), shelf::bf16_t{0});
        ok = hipMemcpy(d_q, zero_q.data(), zero_q.size() * sizeof(shelf::bf16_t),
                       hipMemcpyHostToDevice) == hipSuccess;
        ok = ok && hipMemset(d_mask, 0, rows * mask_row_bytes) == hipSuccess;
        ok = ok && run_kernel() == hipSuccess;
        ok = ok && hipDeviceSynchronize() == hipSuccess;
        ok = ok && hipMemcpy(got_selected.data(), d_selected,
                             got_selected.size() * sizeof(std::int32_t),
                             hipMemcpyDeviceToHost) == hipSuccess;
        ok = ok && hipMemcpy(got_mask.data(), d_mask, got_mask.size(),
                             hipMemcpyDeviceToHost) == hipSuccess;
        bool tie_ok = ok;
        bool tie_mask_ok = ok;
        for (std::uint32_t row = 0; row < rows; ++row) {
            if (s.row_slots[row] >= s.max_sequences) continue;
            const std::uint32_t visible =
                std::min<std::uint32_t>(s.blocks, (s.positions[row] + 1u) / s.compress);
            const std::uint32_t take = std::min<std::uint32_t>(visible, s.top_k);
            const std::uint32_t tail = (s.positions[row] + 1u) / s.compress;
            for (std::uint32_t i = 0; i < s.top_k; ++i) {
                const std::int32_t value = got_selected[row * s.top_k + i];
                const bool expected = i < take && value == static_cast<std::int32_t>(i);
                const bool past = i >= take && value == -1;
                if (!expected && !past) tie_ok = false;
            }
            for (std::uint32_t b = 0; b < s.blocks; ++b) {
                const bool expected = b < take || (tail < s.blocks && b == tail);
                if ((mask_bit(got_mask, mask_row_bytes, row, b) != 0u) != expected)
                    tie_mask_ok = false;
            }
        }
        check(tie_ok, label + ": equal scores keep the smaller block indices");
        check(tie_mask_ok, label + ": equal scores mark the leading blocks");
    }

    (void)hipFree(d_q);
    (void)hipFree(d_pooled);
    (void)hipFree(d_positions);
    (void)hipFree(d_slots);
    (void)hipFree(d_selected);
    (void)hipFree(d_mask);
    (void)hipFree(d_scores);
    (void)hipFree(d_state_scores);
    (void)hipFree(d_state_indices);
}

}  // namespace

int main() {
    constexpr std::uint32_t kChunk = shelf::kernel::kQsaSelectChunk;

    run_scenario({.label = "blocks 200 / compress 4 / top-k 131",
                  .blocks = 200u,
                  .compress = 4u,
                  .heads = 4u,
                  .top_k = kChunk * 2u + 3u,
                  .positions = {0u, 3u, 7u, 23u, 799u, 799u},
                  .row_slots = {0u, 1u, 2u, 0u, 1u, 3u},
                  .max_sequences = 3u});

    run_scenario({.label = "blocks 600 / compress 4 / production top-k 512",
                  .blocks = 600u,
                  .compress = 4u,
                  .heads = 4u,
                  .top_k = 512u,
                  .positions = {251u, 255u, 259u, 511u, 515u, 519u, 2043u, 2047u, 2051u, 2399u,
                                3u, 7u},
                  .row_slots = {0u, 1u, 2u, 0u, 1u, 2u, 0u, 1u, 2u, 0u, 1u, 12u},
                  .max_sequences = 3u});

    run_scenario({.label = "blocks 96 / compress 1",
                  .blocks = 96u,
                  .compress = 1u,
                  .heads = 4u,
                  .top_k = 40u,
                  .positions = {0u, 38u, 39u, 40u, 62u, 63u, 64u, 95u, 200u},
                  .row_slots = {0u, 1u, 2u, 0u, 1u, 2u, 0u, 1u, 2u},
                  .max_sequences = 3u});

    run_scenario({.label = "blocks 130 / compress 2 / 8 heads",
                  .blocks = 130u,
                  .compress = 2u,
                  .heads = 8u,
                  .top_k = 64u,
                  .positions = {1u, 77u, 79u, 81u, 127u, 129u, 131u, 259u},
                  .row_slots = {0u, 1u, 2u, 0u, 1u, 2u, 0u, 1u},
                  .max_sequences = 3u});

    run_scenario({.label = "blocks 600 / compress 4 / split score+topk",
                  .blocks = 600u,
                  .compress = 4u,
                  .heads = 4u,
                  .top_k = 131u,
                  .positions = {251u, 255u, 259u, 511u, 515u, 519u, 2043u, 2047u,
                                2051u},
                  .row_slots = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
                  .max_sequences = 1u},
                 true);

    run_scenario({.label = "blocks 8 / compress 4 / split no visible block",
                  .blocks = 8u,
                  .compress = 4u,
                  .heads = 4u,
                  .top_k = 40u,
                  .positions = {0u, 1u, 2u, 2u},
                  .row_slots = {0u, 0u, 0u, 0u},
                  .max_sequences = 1u},
                 true);

    run_scenario({.label = "blocks 8192 / compress 4 / split full block tile",
                  .blocks = 8192u,
                  .compress = 4u,
                  .heads = 4u,
                  .top_k = 512u,
                  .positions = {32767u, 32768u, 4u, 40u},
                  .row_slots = {0u, 0u, 0u, 0u},
                  .max_sequences = 1u},
                 true);

    run_scenario({.label = "blocks 12000 / compress 4 / split 2 tiles",
                  .blocks = 12000u,
                  .compress = 4u,
                  .heads = 4u,
                  .top_k = 512u,
                  .positions = {47996u, 47999u, 47991u, 47987u},
                  .row_slots = {0u, 0u, 0u, 0u},
                  .max_sequences = 1u},
                 true, 2u);

    run_scenario({.label = "blocks 12000 / compress 4 / split 3 tiles",
                  .blocks = 12000u,
                  .compress = 4u,
                  .heads = 4u,
                  .top_k = 131u,
                  .positions = {47996u, 47999u, 47991u, 47987u},
                  .row_slots = {0u, 0u, 0u, 0u},
                  .max_sequences = 1u},
                 true, 3u);

    std::printf("%s (%d failures)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
