#include <shelf/core/memory/arena.h>
#include <shelf/inference/state/qsa_state_pool.h>
#include <shelf/inference/state/spec_state_history.h>
#include <shelf/kernels/fused/qsa_key_cache.h>
#include <shelf/kernels/optimized/qsa.h>
#include <shelf/runtime/batch/device_batch_context.h>

#include <hip/hip_runtime.h>

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

std::vector<shelf::bf16_t> random_bf16(std::size_t count, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<shelf::bf16_t> out(count);
    for (auto& value : out) value.data = shelf::f32_to_bf16_rne(dist(rng));
    return out;
}

}  // namespace

int main() {
    constexpr std::uint32_t kSequences = 2;
    constexpr std::uint32_t kStates = 1;
    constexpr std::uint32_t kHeadDim = 128;
    constexpr std::uint32_t kCompress = 4;
    constexpr std::uint32_t kBlocks = 8;
    constexpr std::uint32_t kRows = 8;

    auto arena = shelf::gpu::GpuArena::create(0, 32ull * 1024ull * 1024ull);
    check(arena.ok(), "arena create");
    if (!arena.ok()) return 1;

    auto pool_result = shelf::state::QsaKeyStatePool::create(
        arena.value(), kSequences,
        shelf::state::QsaKeyStatePoolLayout::make(kStates, kHeadDim, kCompress, kBlocks));
    check(pool_result.ok(), "pool create");
    if (!pool_result.ok()) return 1;
    auto pool = std::move(pool_result.value());
    check(pool.initialize(0u, nullptr).ok(), "init slot 0");
    check(pool.initialize(1u, nullptr).ok(), "init slot 1");

    std::mt19937 rng(20261116u);
    const auto raw = random_bf16(kRows * kHeadDim, rng);
    const auto weight = random_bf16(kHeadDim, rng);
    std::vector<std::uint32_t> positions(kRows);
    for (std::uint32_t i = 0; i < kRows; ++i) positions[i] = i;

    shelf::bf16_t* d_raw = nullptr;
    shelf::bf16_t* d_weight = nullptr;
    std::uint32_t* d_positions = nullptr;
    shelf::runtime::DeviceRequestDescriptor* d_request = nullptr;
    shelf::bf16_t* d_journal = nullptr;
    std::int32_t* d_row_map = nullptr;
    std::uint32_t* d_selected_positions = nullptr;
    std::vector<shelf::bf16_t> selected_raw;
    std::vector<std::uint32_t> selected_positions;
    const std::uint32_t selected_rows[] = {0u, 2u, 5u};
    for (std::uint32_t row : selected_rows) {
        for (std::uint32_t d = 0; d < kHeadDim; ++d)
            selected_raw.push_back(raw[static_cast<std::size_t>(row) * kHeadDim + d]);
        selected_positions.push_back(positions[row]);
    }

    check(hipMalloc(&d_raw, raw.size() * sizeof(shelf::bf16_t)) == hipSuccess, "malloc raw");
    check(hipMalloc(&d_weight, weight.size() * sizeof(shelf::bf16_t)) == hipSuccess,
          "malloc weight");
    check(hipMalloc(&d_positions, positions.size() * sizeof(std::uint32_t)) == hipSuccess,
          "malloc positions");
    check(hipMalloc(&d_request, sizeof(*d_request)) == hipSuccess, "malloc request");
    check(hipMalloc(&d_journal, raw.size() * sizeof(shelf::bf16_t)) == hipSuccess,
          "malloc journal");
    check(hipMalloc(&d_row_map, 3u * sizeof(std::int32_t)) == hipSuccess, "malloc row map");
    check(hipMalloc(&d_selected_positions, 3u * sizeof(std::uint32_t)) == hipSuccess,
          "malloc selected positions");
    check(hipMemcpy(d_raw, raw.data(), raw.size() * sizeof(shelf::bf16_t),
                    hipMemcpyHostToDevice) == hipSuccess, "copy raw");
    check(hipMemcpy(d_weight, weight.data(), weight.size() * sizeof(shelf::bf16_t),
                    hipMemcpyHostToDevice) == hipSuccess, "copy weight");
    check(hipMemcpy(d_positions, positions.data(), positions.size() * sizeof(std::uint32_t),
                    hipMemcpyHostToDevice) == hipSuccess, "copy positions");
    check(hipMemcpy(d_row_map, selected_rows, sizeof(selected_rows),
                    hipMemcpyHostToDevice) == hipSuccess, "copy row map");
    check(hipMemcpy(d_selected_positions, selected_positions.data(),
                    selected_positions.size() * sizeof(std::uint32_t),
                    hipMemcpyHostToDevice) == hipSuccess, "copy selected positions");
    const auto view = pool.device_view();

    auto base_args = [&]() {
        shelf::kernel::QsaKeyCacheBatchArgs args;
        args.norm_weight = static_cast<const shelf::bf16_t*>(d_weight);
        args.pooled_base = view.pooled_base;
        args.partial_base = view.partial_base;
        args.pool_count_base = view.pool_count_base;
        args.partial_count_base = view.partial_count_base;
        args.max_sequences = view.max_sequences;
        args.state_index = 0u;
        args.max_blocks = view.max_blocks;
        args.head_dim = view.head_dim;
        args.compress_ratio = view.compress_ratio;
        args.rotary_dim = 64u;
        args.raw_row_stride = kHeadDim;
        args.pooled_slot_stride = view.pooled_slot_stride;
        args.pooled_layer_stride = view.pooled_layer_stride;
        args.partial_slot_stride = view.partial_slot_stride;
        args.partial_layer_stride = view.partial_layer_stride;
        args.count_slot_stride = view.count_slot_stride;
        args.count_layer_stride = view.count_layer_stride;
        args.theta = 10000000.0f;
        args.position_scale = 2.0f;
        return args;
    };

    {
        shelf::runtime::DeviceRequestDescriptor descriptor{};
        descriptor.request_handle.slot = 0u;
        descriptor.row_begin = 0u;
        descriptor.row_count = kRows;
        check(hipMemcpy(d_request, &descriptor, sizeof(descriptor),
                        hipMemcpyHostToDevice) == hipSuccess, "copy request");
        auto args = base_args();
        args.raw_keys = static_cast<const shelf::bf16_t*>(d_raw);
        args.positions = static_cast<const std::uint32_t*>(d_positions);
        args.requests = d_request;
        args.key_journal = d_journal;
        args.journal_rows = kRows;
        args.num_requests = 1u;
        args.actual_rows = kRows;
        check(shelf::kernel::launch_qsa_key_cache_batch(args, nullptr) == hipSuccess,
              "journal run launch");
        check(hipDeviceSynchronize() == hipSuccess, "journal run sync");
        std::vector<shelf::bf16_t> journal(raw.size());
        check(hipMemcpy(journal.data(), d_journal, journal.size() * sizeof(shelf::bf16_t),
                        hipMemcpyDeviceToHost) == hipSuccess, "copy journal");
        check(std::memcmp(journal.data(), raw.data(), raw.size() * sizeof(shelf::bf16_t)) == 0,
              "journal records the normed key of every row");
    }

    auto read_state = [&](shelf::state::QsaKeyStatePool& p, std::uint32_t slot,
                          std::vector<shelf::bf16_t>& pooled,
                          std::vector<shelf::bf16_t>& partial, std::uint32_t& pool_count,
                          std::uint32_t& partial_count) {
        auto state = p.qsa_state(slot, 0u);
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

    check(pool.reset(0u, nullptr).ok() && pool.reset(1u, nullptr).ok(), "reset slots");
    {
        auto args = base_args();
        args.raw_keys = d_journal;
        args.positions = d_selected_positions;
        args.raw_row_map = d_row_map;
        args.requests = nullptr;
        args.num_requests = 1u;
        args.replay_slot = 0u;
        args.replay_rows = 3u;
        args.actual_rows = 3u;
        check(shelf::kernel::launch_qsa_key_cache_batch(args, nullptr) == hipSuccess,
              "replay launch");
    }
    {
        auto args = base_args();
        args.raw_keys = static_cast<const shelf::bf16_t*>(d_raw) + 0u;
        std::vector<shelf::bf16_t> compact(selected_raw);
        shelf::bf16_t* d_compact = nullptr;
        std::uint32_t* d_compact_positions = nullptr;
        check(hipMalloc(&d_compact, compact.size() * sizeof(shelf::bf16_t)) == hipSuccess,
              "malloc compact raw");
        check(hipMalloc(&d_compact_positions, selected_positions.size() * sizeof(std::uint32_t)) ==
                  hipSuccess, "malloc compact positions");
        check(hipMemcpy(d_compact, compact.data(), compact.size() * sizeof(shelf::bf16_t),
                        hipMemcpyHostToDevice) == hipSuccess, "copy compact raw");
        check(hipMemcpy(d_compact_positions, selected_positions.data(),
                        selected_positions.size() * sizeof(std::uint32_t),
                        hipMemcpyHostToDevice) == hipSuccess, "copy compact positions");
        shelf::runtime::DeviceRequestDescriptor descriptor{};
        descriptor.request_handle.slot = 1u;
        descriptor.row_begin = 0u;
        descriptor.row_count = 3u;
        check(hipMemcpy(d_request, &descriptor, sizeof(descriptor),
                        hipMemcpyHostToDevice) == hipSuccess, "copy direct request");
        args.raw_keys = d_compact;
        args.positions = d_compact_positions;
        args.requests = d_request;
        args.num_requests = 1u;
        args.actual_rows = 3u;
        check(shelf::kernel::launch_qsa_key_cache_batch(args, nullptr) == hipSuccess,
              "direct launch");
        check(hipDeviceSynchronize() == hipSuccess, "direct sync");
        (void)hipFree(d_compact);
        (void)hipFree(d_compact_positions);
    }

    std::vector<shelf::bf16_t> replay_pooled, replay_partial, direct_pooled, direct_partial;
    std::uint32_t replay_pool = 0u, replay_partial_count = 0u;
    std::uint32_t direct_pool = 0u, direct_partial_count = 0u;
    read_state(pool, 0u, replay_pooled, replay_partial, replay_pool, replay_partial_count);
    read_state(pool, 1u, direct_pooled, direct_partial, direct_pool, direct_partial_count);
    check(replay_pool == direct_pool, "replay block count matches the direct run");
    check(replay_partial_count == direct_partial_count,
          "replay partial count matches the direct run");
    check(std::memcmp(replay_pooled.data(), direct_pooled.data(),
                      replay_pooled.size() * sizeof(shelf::bf16_t)) == 0,
          "replay pooled keys match the direct run");
    check(std::memcmp(replay_partial.data(), direct_partial.data(),
                      replay_partial.size() * sizeof(shelf::bf16_t)) == 0,
          "replay partial keys match the direct run");

    {
        constexpr std::uint32_t kHistoryRows = 4u;
        auto history_result = shelf::state::create_spec_state_history(
            arena.value(),
            shelf::state::SpecStatePools{&pool, nullptr}, kHistoryRows);
        check(history_result.ok(), "spec history create");
        if (history_result.ok()) {
            auto history = history_result.release();
            check(history.qsa.rows == kHistoryRows, "history rows");
            check(history.qsa.partial_state_stride ==
                      static_cast<std::uint64_t>(kHistoryRows) * kCompress * kHeadDim,
                  "history partial stride");
            check(history.qsa.journal_state_stride ==
                      static_cast<std::uint64_t>(kHistoryRows) * kHeadDim,
                  "history journal stride");
            check(history.ple.store == nullptr, "history has no PLE without a PLE pool");
            std::vector<shelf::bf16_t> pattern(static_cast<std::size_t>(kHistoryRows) *
                                               kCompress * kHeadDim);
            for (std::size_t i = 0; i < pattern.size(); ++i)
                pattern[i].data = static_cast<std::uint16_t>(i & 0x3FFFu);
            const std::uint32_t counts[2] = {3u, 2u};
            const std::uint32_t row = 2u;
            check(hipMemcpy(history.qsa.partial + static_cast<std::size_t>(row) * kCompress *
                                                     kHeadDim,
                            pattern.data() + static_cast<std::size_t>(row) * kCompress *
                                                 kHeadDim,
                            static_cast<std::size_t>(kCompress) * kHeadDim *
                                sizeof(shelf::bf16_t),
                            hipMemcpyHostToDevice) == hipSuccess, "fill history partial");
            check(hipMemcpy(history.qsa.counts + static_cast<std::size_t>(row) * 2u, counts,
                            sizeof(counts), hipMemcpyHostToDevice) == hipSuccess,
                  "fill history counts");
            check(shelf::state::restore_spec_state(
                      shelf::state::SpecStatePools{&pool, nullptr}, 0u, history, row,
                      nullptr).ok(), "restore row");
            check(hipDeviceSynchronize() == hipSuccess, "restore sync");
            std::vector<shelf::bf16_t> restored_pooled, restored_partial;
            std::uint32_t restored_pool = 0u, restored_partial_count = 0u;
            read_state(pool, 0u, restored_pooled, restored_partial, restored_pool,
                       restored_partial_count);
            check(restored_pool == 3u && restored_partial_count == 2u,
                  "restore reproduces the counts");
            check(std::memcmp(restored_partial.data(),
                              pattern.data() + static_cast<std::size_t>(row) * kCompress *
                                                   kHeadDim,
                              static_cast<std::size_t>(kCompress) * kHeadDim *
                                  sizeof(shelf::bf16_t)) == 0,
                  "restore reproduces the partial block");
            check(!shelf::state::restore_spec_state(
                       shelf::state::SpecStatePools{&pool, nullptr}, 0u, history,
                       kHistoryRows, nullptr).ok(),
                  "restore rejects an out-of-range row");
        }
    }

    (void)hipFree(d_raw);
    (void)hipFree(d_weight);
    (void)hipFree(d_positions);
    (void)hipFree(d_request);
    (void)hipFree(d_journal);
    (void)hipFree(d_row_map);
    (void)hipFree(d_selected_positions);

    std::fprintf(stderr, failures == 0 ? "PASS\n" : "FAILURES=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
