#include <shelf/core/memory/arena.h>
#include <shelf/inference/state/qsa_state_pool.h>
#include <shelf/kernels/optimized/qsa.h>
#include <shelf/runtime/batch/device_batch_context.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
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

shelf::bf16_t to_bf16(float value) {
    return shelf::bf16_t{shelf::f32_to_bf16_rne(value)};
}

std::vector<shelf::bf16_t> random_bf16(std::size_t count, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<shelf::bf16_t> out(count);
    for (auto& value : out) value.data = shelf::f32_to_bf16_rne(dist(rng));
    return out;
}

}  // namespace

int main() {
    constexpr std::uint32_t kSequences = 3;
    constexpr std::uint32_t kStates = 1;
    constexpr std::uint32_t kHeadDim = 128;
    constexpr std::uint32_t kCompress = 4;
    constexpr std::uint32_t kBlocks = 4;
    constexpr std::uint32_t kRows = 16;
    constexpr std::uint32_t kIndexHeads = 4;
    constexpr std::uint32_t kSelected = 2;

    auto arena = shelf::gpu::GpuArena::create(0, 32ull * 1024ull * 1024ull);
    check(arena.ok(), "arena create");
    if (!arena.ok()) return 1;

    auto pool_result = shelf::state::QsaKeyStatePool::create(
        arena.value(), kSequences,
        shelf::state::QsaKeyStatePoolLayout::make(kStates, kHeadDim, kCompress, kBlocks));
    check(pool_result.ok(), "qsa state pool create");
    if (!pool_result.ok()) {
        std::fprintf(stderr, "%s\n", pool_result.status().message().c_str());
        return 1;
    }
    auto pool = std::move(pool_result.value());
    check(pool.initialize(0, nullptr).ok(), "initialize slot 0");
    check(pool.initialize(1, nullptr).ok(), "initialize slot 1");
    check(pool.initialize(2, nullptr).ok(), "initialize slot 2");
    check(!pool.initialize(1, nullptr).ok(), "double initialize rejected");

    std::mt19937 rng(20261105u);
    const auto raw = random_bf16(kRows * kHeadDim, rng);
    const auto norm_weight = random_bf16(kHeadDim, rng);

    std::vector<std::uint32_t> positions(kRows);
    for (std::uint32_t i = 0; i < 10u; ++i) positions[i] = i;
    for (std::uint32_t i = 0; i < 6u; ++i) positions[10u + i] = i;

    std::vector<shelf::runtime::DeviceRequestDescriptor> requests(2);
    requests[0].request_handle.slot = 0u;
    requests[0].row_begin = 0u;
    requests[0].row_count = 10u;
    requests[1].request_handle.slot = 1u;
    requests[1].row_begin = 10u;
    requests[1].row_count = 6u;

    void* d_raw = nullptr;
    void* d_weight = nullptr;
    void* d_positions = nullptr;
    void* d_requests = nullptr;
    check(hipMalloc(&d_raw, raw.size() * sizeof(shelf::bf16_t)) == hipSuccess,
          "malloc raw");
    check(hipMalloc(&d_weight, norm_weight.size() * sizeof(shelf::bf16_t)) == hipSuccess,
          "malloc weight");
    check(hipMalloc(&d_positions, positions.size() * sizeof(std::uint32_t)) == hipSuccess,
          "malloc positions");
    check(hipMalloc(&d_requests, requests.size() * sizeof(requests[0])) == hipSuccess,
          "malloc requests");
    check(hipMemcpy(d_raw, raw.data(), raw.size() * sizeof(shelf::bf16_t),
                    hipMemcpyHostToDevice) == hipSuccess,
          "copy raw");
    check(hipMemcpy(d_weight, norm_weight.data(),
                    norm_weight.size() * sizeof(shelf::bf16_t),
                    hipMemcpyHostToDevice) == hipSuccess,
          "copy weight");
    check(hipMemcpy(d_positions, positions.data(),
                    positions.size() * sizeof(std::uint32_t),
                    hipMemcpyHostToDevice) == hipSuccess,
          "copy positions");
    check(hipMemcpy(d_requests, requests.data(),
                    requests.size() * sizeof(requests[0]),
                    hipMemcpyHostToDevice) == hipSuccess,
          "copy requests");

    const auto view = pool.device_view();
    shelf::kernel::QsaKeyCacheBatchArgs cache_args;
    cache_args.raw_keys = static_cast<const shelf::bf16_t*>(d_raw);
    cache_args.norm_weight = static_cast<const shelf::bf16_t*>(d_weight);
    cache_args.positions = static_cast<const std::uint32_t*>(d_positions);
    cache_args.requests =
        static_cast<const shelf::runtime::DeviceRequestDescriptor*>(d_requests);
    cache_args.pooled_base = view.pooled_base;
    cache_args.partial_base = view.partial_base;
    cache_args.pool_count_base = view.pool_count_base;
    cache_args.partial_count_base = view.partial_count_base;
    cache_args.num_requests = 2u;
    cache_args.actual_rows = kRows;
    cache_args.max_sequences = view.max_sequences;
    cache_args.state_index = 0u;
    cache_args.max_blocks = view.max_blocks;
    cache_args.head_dim = view.head_dim;
    cache_args.compress_ratio = view.compress_ratio;
    cache_args.rotary_dim = 64u;
    cache_args.raw_row_stride = kHeadDim;
    cache_args.pooled_slot_stride = view.pooled_slot_stride;
    cache_args.pooled_layer_stride = view.pooled_layer_stride;
    cache_args.partial_slot_stride = view.partial_slot_stride;
    cache_args.partial_layer_stride = view.partial_layer_stride;
    cache_args.count_slot_stride = view.count_slot_stride;
    cache_args.count_layer_stride = view.count_layer_stride;
    cache_args.theta = 10000000.0f;
    cache_args.position_scale = 2.0f;
    check(shelf::kernel::launch_qsa_key_cache_batch(cache_args, nullptr) == hipSuccess,
          "key cache batch launch");
    check(hipDeviceSynchronize() == hipSuccess, "key cache sync");

    std::vector<shelf::bf16_t> want_pooled(kSequences * kBlocks * kHeadDim,
                                           shelf::bf16_t{0});
    std::vector<shelf::bf16_t> want_partial(kSequences * kCompress * kHeadDim,
                                            shelf::bf16_t{0});
    std::vector<std::uint32_t> want_pool_count(kSequences, 0u);
    std::vector<std::uint32_t> want_partial_count(kSequences, 0u);
    for (const auto& request : requests) {
        const std::uint32_t slot = request.request_handle.slot;
        for (std::uint32_t t = 0; t < request.row_count; ++t) {
            const std::uint32_t row = request.row_begin + t;
            for (std::uint32_t d = 0; d < kHeadDim; ++d) {
                want_partial[(slot * kCompress + want_partial_count[slot]) * kHeadDim + d] =
                    raw[row * kHeadDim + d];
            }
            ++want_partial_count[slot];
            if (want_partial_count[slot] == kCompress) {
                float mean[kHeadDim];
                float square = 0.0f;
                for (std::uint32_t d = 0; d < kHeadDim; ++d) {
                    float sum = 0.0f;
                    for (std::uint32_t i = 0; i < kCompress; ++i) {
                        sum += bf16_to_float(
                            want_partial[(slot * kCompress + i) * kHeadDim + d].data);
                    }
                    mean[d] = sum / static_cast<float>(kCompress);
                    square += mean[d] * mean[d];
                }
                const float inv = 1.0f / std::sqrt(square / kHeadDim + 1e-6f);
                float normed[kHeadDim];
                for (std::uint32_t d = 0; d < kHeadDim; ++d) {
                    normed[d] = mean[d] * inv *
                                (1.0f + bf16_to_float(norm_weight[d].data));
                }
                const std::uint32_t position = positions[row];
                const std::uint32_t start = position - (kCompress - 1u);
                for (std::uint32_t d = 0; d < kHeadDim; ++d) {
                    float value = normed[d];
                    if (d < 32u) {
                        const float angle =
                            (static_cast<float>(start) / 2.0f) *
                            std::pow(10000000.0f, -2.0f * d / 64.0f);
                        value = normed[d] * std::cos(angle) -
                                normed[d + 32u] * std::sin(angle);
                    } else if (d < 64u) {
                        const std::uint32_t i = d - 32u;
                        const float angle =
                            (static_cast<float>(start) / 2.0f) *
                            std::pow(10000000.0f, -2.0f * i / 64.0f);
                        value = normed[d] * std::cos(angle) +
                                normed[i] * std::sin(angle);
                    }
                    want_pooled[(slot * kBlocks + want_pool_count[slot]) * kHeadDim + d] =
                        to_bf16(value);
                }
                ++want_pool_count[slot];
                want_partial_count[slot] = 0;
            }
        }
    }

    auto read_slot = [&](std::uint32_t slot, std::vector<shelf::bf16_t>& pooled,
                         std::vector<shelf::bf16_t>& partial,
                         std::uint32_t& pool_count, std::uint32_t& partial_count) {
        auto state = pool.qsa_state(slot, 0u);
        pooled.resize(kBlocks * kHeadDim);
        partial.resize(kCompress * kHeadDim);
        (void)hipMemcpy(pooled.data(), state.pooled.data<shelf::bf16_t>(),
                        pooled.size() * sizeof(shelf::bf16_t), hipMemcpyDeviceToHost);
        (void)hipMemcpy(partial.data(), state.partial.data<shelf::bf16_t>(),
                        partial.size() * sizeof(shelf::bf16_t), hipMemcpyDeviceToHost);
        (void)hipMemcpy(&pool_count, state.pool_count.data<std::uint32_t>(),
                        sizeof(pool_count), hipMemcpyDeviceToHost);
        (void)hipMemcpy(&partial_count, state.partial_count.data<std::uint32_t>(),
                        sizeof(partial_count), hipMemcpyDeviceToHost);
    };

    for (std::uint32_t slot = 0; slot < 2u; ++slot) {
        std::vector<shelf::bf16_t> got_pooled;
        std::vector<shelf::bf16_t> got_partial;
        std::uint32_t got_pool_count = 0;
        std::uint32_t got_partial_count = 0;
        read_slot(slot, got_pooled, got_partial, got_pool_count, got_partial_count);
        check(got_pool_count == want_pool_count[slot], "pooled block count");
        check(got_partial_count == want_partial_count[slot], "partial key count");
        double max_pooled = 0.0;
        for (std::uint32_t b = 0; b < got_pool_count; ++b) {
            for (std::uint32_t d = 0; d < kHeadDim; ++d) {
                const std::size_t i = (slot * kBlocks + b) * kHeadDim + d;
                max_pooled = std::max<double>(
                    max_pooled,
                    std::fabs(static_cast<double>(bf16_to_float(got_pooled[b * kHeadDim + d].data)) -
                              bf16_to_float(want_pooled[i].data)));
            }
        }
        double max_partial = 0.0;
        for (std::uint32_t t = 0; t < got_partial_count; ++t) {
            for (std::uint32_t d = 0; d < kHeadDim; ++d) {
                const std::size_t i = (slot * kCompress + t) * kHeadDim + d;
                max_partial = std::max<double>(
                    max_partial,
                    std::fabs(static_cast<double>(bf16_to_float(got_partial[t * kHeadDim + d].data)) -
                              bf16_to_float(want_partial[i].data)));
            }
        }
        std::fprintf(stderr, "slot %u pooled max_abs=%g partial max_abs=%g\n", slot,
                     max_pooled, max_partial);
        check(max_pooled <= 3.0e-2, "pooled key matches CPU reference");
        check(max_partial == 0.0, "partial key matches CPU reference");
    }

    check(pool.copy_slot_from(2u, pool, 0u, nullptr).ok(), "copy slot 0 to slot 2");
    check(hipDeviceSynchronize() == hipSuccess, "copy sync");
    {
        std::vector<shelf::bf16_t> a_pooled;
        std::vector<shelf::bf16_t> a_partial;
        std::vector<shelf::bf16_t> b_pooled;
        std::vector<shelf::bf16_t> b_partial;
        std::uint32_t a_pc = 0, a_qc = 0, b_pc = 0, b_qc = 0;
        read_slot(0u, a_pooled, a_partial, a_pc, a_qc);
        read_slot(2u, b_pooled, b_partial, b_pc, b_qc);
        check(a_pc == b_pc && a_qc == b_qc &&
                  std::memcmp(a_pooled.data(), b_pooled.data(),
                              a_pooled.size() * sizeof(shelf::bf16_t)) == 0 &&
                  std::memcmp(a_partial.data(), b_partial.data(),
                              a_partial.size() * sizeof(shelf::bf16_t)) == 0,
              "slot copy reproduces pooled, partial and counts");
    }

    const auto q = random_bf16(kRows * kIndexHeads * kHeadDim, rng);
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    const std::uint32_t mask_row_bytes = ((kBlocks + 31u) / 32u) * 4u;

    void* d_q = nullptr;
    void* d_selected = nullptr;
    void* d_mask = nullptr;
    check(hipMalloc(&d_q, q.size() * sizeof(shelf::bf16_t)) == hipSuccess, "malloc q");
    check(hipMalloc(&d_selected, kRows * kSelected * sizeof(std::int32_t)) == hipSuccess,
          "malloc selected");
    check(hipMalloc(&d_mask, kRows * mask_row_bytes) == hipSuccess, "malloc block mask");
    check(hipMemcpy(d_q, q.data(), q.size() * sizeof(shelf::bf16_t),
                    hipMemcpyHostToDevice) == hipSuccess,
          "copy q");

    std::vector<std::uint32_t> row_slots(kRows);
    for (std::uint32_t i = 0; i < 10u; ++i) row_slots[i] = 0u;
    for (std::uint32_t i = 10u; i < kRows; ++i) row_slots[i] = 1u;
    void* d_slots = nullptr;
    check(hipMalloc(&d_slots, row_slots.size() * sizeof(std::uint32_t)) == hipSuccess,
          "malloc row slots");
    check(hipMemcpy(d_slots, row_slots.data(), row_slots.size() * sizeof(std::uint32_t),
                    hipMemcpyHostToDevice) == hipSuccess,
          "copy row slots");

    shelf::kernel::QsaSelectBatchArgs select_args;
    select_args.q = static_cast<const shelf::bf16_t*>(d_q);
    select_args.pooled_base = view.pooled_base;
    select_args.selected = static_cast<std::int32_t*>(d_selected);
    select_args.block_mask = static_cast<std::uint8_t*>(d_mask);
    select_args.row_sequence_slots = static_cast<const std::uint32_t*>(d_slots);
    select_args.positions = static_cast<const std::uint32_t*>(d_positions);
    select_args.rows = kRows;
    select_args.blocks = kBlocks;
    select_args.n_heads = kIndexHeads;
    select_args.head_dim = kHeadDim;
    select_args.compress_ratio = kCompress;
    select_args.state_index = 0u;
    select_args.max_sequences = view.max_sequences;
    select_args.top_k = kSelected;
    select_args.q_row_stride = kIndexHeads * kHeadDim;
    select_args.selected_row_stride = kSelected;
    select_args.mask_row_bytes = mask_row_bytes;
    select_args.pooled_slot_stride = view.pooled_slot_stride;
    select_args.pooled_layer_stride = view.pooled_layer_stride;
    select_args.scale = scale;
    check(shelf::kernel::launch_qsa_select_batch(select_args, nullptr) == hipSuccess,
          "select batch launch");
    check(hipDeviceSynchronize() == hipSuccess, "select sync");

    std::vector<std::int32_t> got_selected(kRows * kSelected, -2);
    (void)hipMemcpy(got_selected.data(), d_selected,
                    got_selected.size() * sizeof(std::int32_t), hipMemcpyDeviceToHost);
    std::vector<std::uint8_t> got_mask(kRows * mask_row_bytes, 0u);
    (void)hipMemcpy(got_mask.data(), d_mask, got_mask.size(), hipMemcpyDeviceToHost);

    bool selected_ok = true;
    bool mask_ok = true;
    for (std::uint32_t row = 0; row < kRows; ++row) {
        const std::uint32_t slot = row_slots[row];
        const std::uint32_t visible = std::min((positions[row] + 1u) / kCompress, kBlocks);
        std::vector<float> scores(visible, 0.0f);
        for (std::uint32_t b = 0; b < visible; ++b) {
            float total = 0.0f;
            for (std::uint32_t h = 0; h < kIndexHeads; ++h) {
                float dot = 0.0f;
                for (std::uint32_t d = 0; d < kHeadDim; ++d) {
                    dot += bf16_to_float(q[(row * kIndexHeads + h) * kHeadDim + d].data) *
                           bf16_to_float(
                               want_pooled[(slot * kBlocks + b) * kHeadDim + d].data);
                }
                total += std::max(dot, 0.0f);
            }
            scores[b] = total * scale;
        }
        std::vector<std::uint32_t> order(visible);
        for (std::uint32_t b = 0; b < visible; ++b) order[b] = b;
        std::stable_sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) {
            if (scores[a] != scores[b]) return scores[a] > scores[b];
            return a < b;
        });
        const std::uint32_t take = std::min(visible, kSelected);
        std::vector<std::int32_t> want;
        for (std::uint32_t j = 0; j < take; ++j) want.push_back(static_cast<std::int32_t>(order[j]));
        std::vector<std::int32_t> got;
        for (std::uint32_t j = 0; j < kSelected; ++j) {
            const std::int32_t value = got_selected[row * kSelected + j];
            if (j >= take) {
                if (value != -1) selected_ok = false;
            } else if (value >= 0) {
                got.push_back(value);
            }
        }
        std::sort(got.begin(), got.end());
        std::sort(want.begin(), want.end());
        if (got != want) selected_ok = false;

        const std::uint32_t tail = (positions[row] + 1u) / kCompress;
        std::vector<std::int32_t> marked = want;
        if (tail < kBlocks) marked.push_back(static_cast<std::int32_t>(tail));
        std::sort(marked.begin(), marked.end());
        for (std::uint32_t b = 0; b < kBlocks; ++b) {
            const bool expected =
                std::binary_search(marked.begin(), marked.end(), static_cast<std::int32_t>(b));
            const std::size_t byte = static_cast<std::size_t>(row) * mask_row_bytes +
                                     (b >> 3u);
            const bool set = ((got_mask[byte] >> (b & 7u)) & 1u) != 0u;
            if (set != expected) mask_ok = false;
        }
    }
    check(selected_ok, "select matches the pooled keys of each sequence slot");
    check(mask_ok, "block mask marks the selection and the partial tail block");


    check(pool.release(0u).ok(), "release slot 0");
    check(pool.release(1u).ok(), "release slot 1");
    check(pool.release(2u).ok(), "release slot 2");

    (void)hipFree(d_raw);
    (void)hipFree(d_weight);
    (void)hipFree(d_positions);
    (void)hipFree(d_requests);
    (void)hipFree(d_q);
    (void)hipFree(d_slots);
    (void)hipFree(d_selected);
    (void)hipFree(d_mask);

    std::fprintf(stderr, failures == 0 ? "PASS\n" : "FAILURES=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
