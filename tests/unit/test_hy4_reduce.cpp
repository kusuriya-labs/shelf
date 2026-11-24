#include <shelf/models/hy4_flash/ops.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <thread>
#include <bit>
#include <algorithm>

namespace {
void check_peer_collectives() {
    using namespace shelf::hy4;
    constexpr size_t elements = 65539, bytes = elements * sizeof(uint16_t);
    Checkpoint model;
    Residency gpu(model, 64);
    gpu.ranks = 4;
    gpu.tp_size = gpu.ep_size = 2;
    gpu.physical.fill(0);
    for (int group = 0; group < 2; ++group) {
        gpu.tp_barrier[group] = std::make_unique<SyncBarrier>(2);
        gpu.ep_barrier[group] = std::make_unique<SyncBarrier>(2);
    }
    std::array<uint16_t *, 4> data{};
    for (int r = 0; r < gpu.ranks; ++r) {
        gpu.peer[r].fill(true);
        auto arena = shelf::gpu::GpuArena::create(0, 3 * (bytes + 256));
        if (!arena.ok())
            throw std::runtime_error(arena.status().message());
        gpu.arenas.push_back(arena.release());
        auto allocate = [&] {
            auto memory = gpu.arenas.back().allocate_aligned(bytes, 256);
            if (!memory.ok())
                throw std::runtime_error(memory.status().message());
            return memory.value().data();
        };
        data[r] = static_cast<uint16_t *>(allocate());
        gpu.tp_recv[r] = allocate();
        gpu.ep_recv[r] = allocate();
        checked(hipStreamCreateWithFlags(&gpu.compute[r], hipStreamNonBlocking));
        for (auto events : {&gpu.tp_ready, &gpu.tp_done, &gpu.tp_arrived, &gpu.tp_consumed,
                            &gpu.ep_ready, &gpu.ep_done, &gpu.ep_arrived, &gpu.ep_consumed})
            for (auto &event : (*events)[r])
                gpu.create_event(event, r);
    }
    bool rejected = false;
    try {
        gpu.record_event(gpu.tp_ready[1][0], 0);
    } catch (const std::logic_error &) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("別の論理 rank での event の記録を検出できませんでした");
    std::array<std::exception_ptr, 4> failures{};
    std::array<std::jthread, 4> workers;
    try {
        for (int r = 0; r < gpu.ranks; ++r)
            workers[r] = std::jthread([&, r] {
                try {
                    for (int pass = 0; pass < 6; ++pass) {
                        uint16_t bits = std::bit_cast<uint32_t>(float(pass + r + 1)) >> 16;
                        std::vector<uint16_t> values(elements, bits);
                        checked(hipMemcpyAsync(data[r], values.data(), bytes, hipMemcpyHostToDevice, gpu.compute[r]));
                        gpu.tp_sum(data[r], r, bytes);
                        gpu.ep_reduce(data[r], r, bytes);
                        checked(hipMemcpyAsync(values.data(), data[r], bytes, hipMemcpyDeviceToHost, gpu.compute[r]));
                        checked(hipStreamSynchronize(gpu.compute[r]));
                        uint16_t expected = std::bit_cast<uint32_t>(float(4 * pass + 10)) >> 16;
                        if (!std::all_of(values.begin(), values.end(), [&](auto value) { return value == expected; }))
                            throw std::runtime_error("peer 転送を使った TP・EP の合算結果が不正です");
                    }
                } catch (const std::exception &e) {
                    std::fprintf(stderr, "rank=%d err=%s\n", r, e.what());
                    failures[r] = std::current_exception();
                    gpu.abort_collectives();
                } catch (...) {
                    failures[r] = std::current_exception();
                    gpu.abort_collectives();
                }
            });
    } catch (...) {
        gpu.abort_collectives();
        throw;
    }
    for (auto &worker : workers)
        worker.join();
    gpu.shutdown();
    for (auto &failure : failures)
        if (failure)
            std::rethrow_exception(failure);
}

void check_lm_head_collectives() {
    using namespace shelf::hy4;
    for (const char *transport : {"push", "peer", "host"}) {
        setenv("SHELF_HY4_EP_TRANSPORT", transport, 1);
        {
            Checkpoint model;
            Residency gpu(model, 64);
            gpu.ranks = gpu.ep_size = 4;
            gpu.tp_size = 1;
            gpu.lm_head_tp = true;
            gpu.physical.fill(0);
            const bool host = std::string_view(transport) == "host";
            gpu.push = std::string_view(transport) != "peer";
            for (int r = 0; r < 4; ++r) {
                if (!host)
                    for (int other = 0; other < 4; ++other)
                        if (other != r)
                            gpu.peer[r][other] = true;
                if (host)
                    checked(hipHostMalloc(&gpu.lm_staging[r], Residency::lm_staging_bytes(), hipHostMallocPortable));
                size_t total_buf = 2 * lm_hidden_chunk_bytes + 2 * 4 * lm_argmax_stride +
                                   (r == 0 ? 2 * lm_logits_chunk_bytes : 0) + 4096 * 128;
                auto arena = shelf::gpu::GpuArena::create(0, total_buf + 4096);
                if (!arena.ok())
                    throw std::runtime_error(arena.status().message());
                gpu.arenas.push_back(arena.release());
                auto alloc = [&](size_t sz) {
                    auto mem = gpu.arenas.back().allocate_aligned(sz, 256);
                    if (!mem.ok())
                        throw std::runtime_error(mem.status().message());
                    return mem.value().data();
                };
                gpu.lm_hidden_buf[r] = alloc(2 * lm_hidden_chunk_bytes);
                gpu.lm_head_recv[r] = alloc(2 * 4 * lm_argmax_stride);
                if (r == 0)
                    gpu.lm_logits_recv[0] = alloc(2 * lm_logits_chunk_bytes);
                checked(hipStreamCreateWithFlags(&gpu.compute[r], hipStreamNonBlocking));
                for (int p = 0; p < 2; ++p)
                    for (auto events : {&gpu.lm_ready, &gpu.lm_done, &gpu.lm_arrived, &gpu.lm_consumed})
                        gpu.create_event((*events)[r][p], r);
            }
            constexpr int rows = 37;
            auto arena_extra = shelf::gpu::GpuArena::create(0, 64ULL << 20);
            if (!arena_extra.ok())
                throw std::runtime_error(arena_extra.status().message());
            auto alloc_e = [&](size_t sz) {
                auto mem = arena_extra.value().allocate_aligned(sz, 256);
                if (!mem.ok())
                    throw std::runtime_error(mem.status().message());
                return mem.value().data();
            };
            auto *src_hidden = static_cast<uint16_t *>(alloc_e(rows * lm_hidden_row_bytes));
            std::vector<uint16_t> h_hidden(rows * 3072);
            for (size_t i = 0; i < h_hidden.size(); ++i)
                h_hidden[i] = uint16_t((i * 131 + 7) & 0xffff);
            checked(hipMemcpyAsync(src_hidden, h_hidden.data(), h_hidden.size() * 2, hipMemcpyHostToDevice,
                                   gpu.compute[0]));
            std::array<void *, 4> hidden_dst{};
            for (int pass = 0; pass < 4; ++pass) {
                gpu.lm_scatter_hidden(src_hidden, rows * lm_hidden_row_bytes, hidden_dst, false);
                for (int r = 1; r < 4; ++r) {
                    std::vector<uint16_t> got(h_hidden.size());
                    checked(hipMemcpyAsync(got.data(), hidden_dst[r], got.size() * 2, hipMemcpyDeviceToHost,
                                           gpu.compute[r]));
                    checked(hipStreamSynchronize(gpu.compute[r]));
                    if (got != h_hidden)
                        throw std::runtime_error("lm_scatter_hidden の結果が一致しません");
                }
                std::array<const float *, 4> d_vals{};
                std::array<const int *, 4> d_ids{};
                std::array<std::vector<float>, 4> h_vals;
                std::array<std::vector<int>, 4> h_ids;
                std::vector<int> expected_ids(rows);
                for (int r = 0; r < 4; ++r) {
                    auto *vp = static_cast<float *>(alloc_e(rows * sizeof(float)));
                    auto *ip = static_cast<int *>(alloc_e(rows * sizeof(int)));
                    h_vals[r].resize(rows);
                    h_ids[r].resize(rows);
                    for (int row = 0; row < rows; ++row) {
                        h_vals[r][row] = float(((row + pass + r) % 7 == 0) ? 10.0f : (row + pass + r) * 0.5f);
                        h_ids[r][row] = r * 30208 + row * 17 + pass;
                    }
                    checked(hipMemcpyAsync(vp, h_vals[r].data(), rows * 4, hipMemcpyHostToDevice, gpu.compute[r]));
                    checked(hipMemcpyAsync(ip, h_ids[r].data(), rows * 4, hipMemcpyHostToDevice, gpu.compute[r]));
                    d_vals[r] = vp;
                    d_ids[r] = ip;
                }
                for (int row = 0; row < rows; ++row) {
                    float best_v = h_vals[0][row];
                    int best_i = h_ids[0][row];
                    for (int r = 1; r < 4; ++r) {
                        if (h_vals[r][row] > best_v || (h_vals[r][row] == best_v && h_ids[r][row] < best_i)) {
                            best_v = h_vals[r][row];
                            best_i = h_ids[r][row];
                        }
                    }
                    expected_ids[row] = best_i;
                }
                auto *out_ids = static_cast<int *>(alloc_e(rows * sizeof(int)));
                gpu.lm_gather_argmax(d_vals, d_ids, rows, out_ids);
                std::vector<int> got_ids(rows);
                checked(hipMemcpyAsync(got_ids.data(), out_ids, rows * 4, hipMemcpyDeviceToHost, gpu.compute[0]));
                checked(hipStreamSynchronize(gpu.compute[0]));
                if (got_ids != expected_ids)
                    throw std::runtime_error("lm_gather_argmax の結果が一致しません");
            }
            constexpr int full_v = hy4_vocab_size;
            constexpr int local_v = full_v / 4;
            constexpr int lrows = 24;
            std::array<const float *, 4> d_logits{};
            std::array<std::vector<float>, 4> h_logits{};
            std::vector<float> expected_logits(size_t(lrows) * full_v);
            for (int r = 0; r < 4; ++r) {
                h_logits[r].resize(size_t(lrows) * local_v);
                for (int row = 0; row < lrows; ++row)
                    for (int c = 0; c < local_v; ++c) {
                        float v = float((r * 1000 + row * 13 + c) % 97) * 0.25f;
                        h_logits[r][size_t(row) * local_v + c] = v;
                        expected_logits[size_t(row) * full_v + r * local_v + c] = v;
                    }
                auto *dp = static_cast<float *>(alloc_e(h_logits[r].size() * sizeof(float)));
                checked(hipMemcpyAsync(dp, h_logits[r].data(), h_logits[r].size() * 4, hipMemcpyHostToDevice,
                                       gpu.compute[r]));
                d_logits[r] = dp;
            }
            auto *out_logits = static_cast<float *>(alloc_e(expected_logits.size() * sizeof(float)));
            gpu.lm_gather_logits(d_logits, lrows, local_v, full_v, out_logits);
            std::vector<float> got_logits(expected_logits.size());
            checked(hipMemcpyAsync(got_logits.data(), out_logits, got_logits.size() * 4, hipMemcpyDeviceToHost,
                                   gpu.compute[0]));
            checked(hipStreamSynchronize(gpu.compute[0]));
            if (got_logits != expected_logits)
                throw std::runtime_error("lm_gather_logits の結果が一致しません");
            (void)arena_extra.value().shutdown();
            gpu.shutdown();
        }
        {
            constexpr size_t elements = 65539, bytes = elements * sizeof(uint16_t);
            Checkpoint model;
            Residency gpu(model, 64);
            gpu.ranks = 4;
            gpu.tp_size = gpu.ep_size = 2;
            gpu.lm_head_tp = true;
            gpu.physical.fill(0);
            gpu.push = std::string_view(transport) != "peer";
            const bool host = std::string_view(transport) == "host";
            for (int group = 0; group < 2; ++group) {
                gpu.tp_barrier[group] = std::make_unique<SyncBarrier>(2);
                gpu.ep_barrier[group] = std::make_unique<SyncBarrier>(2);
            }
            gpu.all_barrier = std::make_unique<SyncBarrier>(4);
            std::array<uint16_t *, 4> data{};
            std::array<float *, 4> vals{};
            std::array<int *, 4> ids{};
            std::array<int *, 4> out_ids{};
            for (int r = 0; r < gpu.ranks; ++r) {
                if (!host)
                    gpu.peer[r].fill(true);
                if (host) {
                    checked(hipHostMalloc(&gpu.reduce_staging[r], tp_prefill_rows * 3072u * 2u, hipHostMallocPortable));
                    checked(hipHostMalloc(&gpu.lm_staging[r], Residency::lm_staging_bytes(), hipHostMallocPortable));
                }
                size_t total = 3 * (bytes + 256) + 2 * lm_hidden_chunk_bytes + 2 * 4 * lm_argmax_stride +
                               2 * lm_logits_chunk_bytes + 65536;
                auto arena = shelf::gpu::GpuArena::create(0, total);
                if (!arena.ok())
                    throw std::runtime_error(arena.status().message());
                gpu.arenas.push_back(arena.release());
                auto alloc = [&](size_t sz) {
                    auto mem = gpu.arenas.back().allocate_aligned(sz, 256);
                    if (!mem.ok())
                        throw std::runtime_error(mem.status().message());
                    return mem.value().data();
                };
                data[r] = static_cast<uint16_t *>(alloc(bytes));
                gpu.tp_recv[r] = alloc(bytes);
                gpu.ep_recv[r] = alloc(bytes);
                gpu.lm_hidden_buf[r] = alloc(2 * lm_hidden_chunk_bytes);
                gpu.lm_head_recv[r] = alloc(2 * 4 * lm_argmax_stride);
                gpu.lm_logits_recv[r] = alloc(2 * lm_logits_chunk_bytes);
                vals[r] = static_cast<float *>(alloc(64 * sizeof(float)));
                ids[r] = static_cast<int *>(alloc(64 * sizeof(int)));
                out_ids[r] = static_cast<int *>(alloc(64 * sizeof(int)));
                checked(hipStreamCreateWithFlags(&gpu.compute[r], hipStreamNonBlocking));
                for (int p = 0; p < 2; ++p)
                    for (auto events : {&gpu.tp_ready, &gpu.tp_done, &gpu.tp_arrived, &gpu.tp_consumed,
                                        &gpu.ep_ready, &gpu.ep_done, &gpu.ep_arrived, &gpu.ep_consumed,
                                        &gpu.lm_ready, &gpu.lm_done, &gpu.lm_arrived, &gpu.lm_consumed})
                        gpu.create_event((*events)[r][p], r);
            }
            std::array<std::exception_ptr, 4> failures{};
            std::array<std::jthread, 4> workers;
            for (int r = 0; r < 4; ++r)
                workers[r] = std::jthread([&, r] {
                    try {
                        for (int pass = 0; pass < 8; ++pass) {
                            uint16_t bits = std::bit_cast<uint32_t>(float(pass + r + 1)) >> 16;
                            std::vector<uint16_t> values(elements, bits);
                            checked(hipMemcpyAsync(data[r], values.data(), bytes, hipMemcpyHostToDevice,
                                                   gpu.compute[r]));
                            gpu.tp_sum(data[r], r, bytes);
                            gpu.ep_reduce(data[r], r, bytes);
                            std::vector<float> hv(16);
                            std::vector<int> hi(16), expected_ids(16);
                            for (int row = 0; row < 16; ++row) {
                                hv[row] = float((row + pass + r) % 4 == 0 ? 50.0f : (r + 1) * 1.0f);
                                hi[row] = r * 30208 + row;
                                float best_v = -1e9f;
                                int best_i = 0;
                                for (int t = 0; t < 4; ++t) {
                                    float v = float((row + pass + t) % 4 == 0 ? 50.0f : (t + 1) * 1.0f);
                                    int id = t * 30208 + row;
                                    if (v > best_v || (v == best_v && id < best_i)) {
                                        best_v = v;
                                        best_i = id;
                                    }
                                }
                                expected_ids[row] = best_i;
                            }
                            checked(hipMemcpyAsync(vals[r], hv.data(), 16 * 4, hipMemcpyHostToDevice, gpu.compute[r]));
                            checked(hipMemcpyAsync(ids[r], hi.data(), 16 * 4, hipMemcpyHostToDevice, gpu.compute[r]));
                            gpu.tp_argmax(vals[r], ids[r], out_ids[r], r, 16);
                            std::vector<int> got_ids(16);
                            checked(hipMemcpyAsync(got_ids.data(), out_ids[r], 16 * 4, hipMemcpyDeviceToHost,
                                                   gpu.compute[r]));
                            checked(hipMemcpyAsync(values.data(), data[r], bytes, hipMemcpyDeviceToHost,
                                                   gpu.compute[r]));
                            checked(hipStreamSynchronize(gpu.compute[r]));
                            uint16_t expected = std::bit_cast<uint32_t>(float(4 * pass + 10)) >> 16;
                            if (!std::all_of(values.begin(), values.end(), [&](auto value) { return value == expected; }))
                                throw std::runtime_error("分離バッファ併用時の TP/EP 合算値が不正です");
                            if (got_ids != expected_ids)
                                throw std::runtime_error("tp_argmax の結果が一致しません");
                            gpu.tp_broadcast_ids(ids[r], r, 16);
                            checked(hipMemcpyAsync(got_ids.data(), ids[r], 16 * sizeof(int), hipMemcpyDeviceToHost,
                                                   gpu.compute[r]));
                            checked(hipStreamSynchronize(gpu.compute[r]));
                            for (int row = 0; row < 16; ++row)
                                if (got_ids[row] != row)
                                    throw std::runtime_error("tp_broadcast_ids の結果が一致しません");
                        }
                    } catch (...) {
                        failures[r] = std::current_exception();
                        gpu.abort_collectives();
                    }
                });
            for (auto &worker : workers)
                worker.join();
            gpu.shutdown();
            for (auto &failure : failures)
                if (failure)
                    std::rethrow_exception(failure);
        }
    }
    unsetenv("SHELF_HY4_EP_TRANSPORT");
}

void check_tp4_collectives() {
    using namespace shelf::hy4;
    constexpr size_t small_elements = 65539;
    constexpr size_t large_elements = 1200000;
    constexpr size_t max_bytes = large_elements * sizeof(uint16_t);
    constexpr size_t max_f32_bytes = large_elements * sizeof(float);
    constexpr size_t recv_bytes = std::max<size_t>(6u << 20, 3u * ((tp_prefill_rows * 3072u * 2u) / 4u));
    constexpr int logit_rows = 4;
    constexpr int local_vocab = 64;
    constexpr size_t full_logits_bytes = size_t(logit_rows) * 4u * size_t(local_vocab) * sizeof(float);
    Checkpoint model;
    Residency gpu(model, 64);
    gpu.ranks = 4;
    gpu.tp_size = 4;
    gpu.ep_size = 1;
    gpu.lm_head_tp = true;
    gpu.physical.fill(0);
    gpu.tp_barrier[0] = std::make_unique<SyncBarrier>(4);
    gpu.all_barrier = std::make_unique<SyncBarrier>(4);
    std::array<uint16_t *, 4> data{};
    std::array<float *, 4> f32_data{}, local_vals{}, local_logits{}, full_logits{};
    std::array<int *, 4> local_ids{}, tokens_out{};
    for (int r = 0; r < gpu.ranks; ++r) {
        size_t extra_lm = 2 * 4 * lm_argmax_stride + (r == 0 ? 2 * lm_logits_chunk_bytes : 0);
        auto arena = shelf::gpu::GpuArena::create(0, max_bytes + max_f32_bytes + recv_bytes + full_logits_bytes + extra_lm + 65536);
        if (!arena.ok())
            throw std::runtime_error(arena.status().message());
        gpu.arenas.push_back(arena.release());
        auto allocate = [&](size_t size) {
            auto memory = gpu.arenas.back().allocate_aligned(size, 256);
            if (!memory.ok())
                throw std::runtime_error(memory.status().message());
            return memory.value().data();
        };
        data[r] = static_cast<uint16_t *>(allocate(max_bytes));
        f32_data[r] = static_cast<float *>(allocate(max_f32_bytes));
        gpu.tp_recv[r] = allocate(recv_bytes);
        gpu.lm_head_recv[r] = allocate(2 * 4 * lm_argmax_stride);
        if (r == 0)
            gpu.lm_logits_recv[0] = allocate(2 * lm_logits_chunk_bytes);
        local_vals[r] = static_cast<float *>(allocate(logit_rows * sizeof(float)));
        local_ids[r] = static_cast<int *>(allocate(logit_rows * sizeof(int)));
        tokens_out[r] = static_cast<int *>(allocate(logit_rows * sizeof(int)));
        local_logits[r] = static_cast<float *>(allocate(logit_rows * local_vocab * sizeof(float)));
        full_logits[r] = static_cast<float *>(allocate(full_logits_bytes));
        checked(hipHostMalloc(&gpu.reduce_staging[r], tp_prefill_rows * 3072u * 2u));
        checked(hipHostMalloc(&gpu.lm_staging[r], Residency::lm_staging_bytes(), hipHostMallocPortable));
        checked(hipStreamCreateWithFlags(&gpu.compute[r], hipStreamNonBlocking));
        for (auto events : {&gpu.tp_ready, &gpu.tp_done, &gpu.tp_arrived, &gpu.tp_consumed, &gpu.tp_ag_done,
                            &gpu.lm_ready, &gpu.lm_done, &gpu.lm_arrived, &gpu.lm_consumed})
            for (auto &event : (*events)[r])
                gpu.create_event(event, r);
    }
    struct Mode {
        bool push;
        bool all_peer;
        bool mixed;
    };
    const std::array<Mode, 4> modes{{
        {true, true, false},
        {false, true, false},
        {true, false, false},
        {false, false, true},
    }};
    for (const auto &mode : modes) {
        gpu.push = mode.push;
        for (int r = 0; r < 4; ++r) {
            gpu.peer[r].fill(mode.all_peer || mode.mixed);
            if (mode.mixed) {
                gpu.peer[0][1] = gpu.peer[1][0] = false;
            }
            gpu.peer[r][r] = true;
        }
        std::array<std::exception_ptr, 4> failures{};
        std::array<std::jthread, 4> workers;
        for (int r = 0; r < 4; ++r)
            workers[r] = std::jthread([&, r] {
                try {
                    for (size_t elements : {small_elements, large_elements}) {
                        const size_t bytes = elements * sizeof(uint16_t);
                        for (int pass = 0; pass < 3; ++pass) {
                            std::vector<uint16_t> values(elements);
                            for (size_t idx = 0; idx < elements; ++idx) {
                                float v = float((pass + r + 1) + int(idx % 5));
                                values[idx] = std::bit_cast<uint32_t>(v) >> 16;
                            }
                            checked(hipMemcpyAsync(data[r], values.data(), bytes, hipMemcpyHostToDevice,
                                                   gpu.compute[r]));
                            gpu.tp_sum(data[r], r, bytes);
                            checked(hipMemcpyAsync(values.data(), data[r], bytes, hipMemcpyDeviceToHost,
                                                   gpu.compute[r]));
                            checked(hipStreamSynchronize(gpu.compute[r]));
                            for (size_t idx = 0; idx < elements; ++idx) {
                                float expected_sum = 0.0f;
                                for (int k = 0; k < 4; ++k) {
                                    uint16_t bf = std::bit_cast<uint32_t>(float((pass + k + 1) + int(idx % 5))) >> 16;
                                    expected_sum += std::bit_cast<float>(uint32_t(bf) << 16);
                                }
                                uint16_t expected = std::bit_cast<uint32_t>(expected_sum) >> 16;
                                if (values[idx] != expected) {
                                    char buf[160];
                                    std::snprintf(buf, sizeof(buf),
                                                  "TP=4 不一致 push=%d peer=%d mixed=%d elems=%zu pass=%d idx=%zu got=0x%04x exp=0x%04x",
                                                  int(mode.push), int(mode.all_peer), int(mode.mixed), elements, pass,
                                                  idx, values[idx], expected);
                                    throw std::runtime_error(buf);
                                }
                            }
                            std::vector<float> f32_in(elements);
                            for (size_t idx = 0; idx < elements; ++idx) {
                                float sign = (r & 1) ? -1.0f : 1.0f;
                                f32_in[idx] = sign * 1024.0f + float((pass + r + 1) + int(idx % 5)) * 0.25f;
                            }
                            checked(hipMemcpyAsync(f32_data[r], f32_in.data(), elements * sizeof(float),
                                                   hipMemcpyHostToDevice, gpu.compute[r]));
                            gpu.tp_sum_f32(data[r], f32_data[r], r, bytes);
                            checked(hipMemcpyAsync(values.data(), data[r], bytes, hipMemcpyDeviceToHost,
                                                   gpu.compute[r]));
                            checked(hipStreamSynchronize(gpu.compute[r]));
                            for (size_t idx = 0; idx < elements; ++idx) {
                                float expected_sum = 0.0f;
                                for (int k = 0; k < 4; ++k) {
                                    float sign = (k & 1) ? -1.0f : 1.0f;
                                    expected_sum += sign * 1024.0f + float((pass + k + 1) + int(idx % 5)) * 0.25f;
                                }
                                uint16_t expected = std::bit_cast<uint32_t>(expected_sum) >> 16;
                                if (values[idx] != expected) {
                                    char buf[160];
                                    std::snprintf(buf, sizeof(buf),
                                                  "TP=4 F32 不一致 push=%d peer=%d mixed=%d elems=%zu pass=%d idx=%zu got=0x%04x exp=0x%04x",
                                                  int(mode.push), int(mode.all_peer), int(mode.mixed), elements, pass,
                                                  idx, values[idx], expected);
                                    throw std::runtime_error(buf);
                                }
                            }
                        }
                    }
                    std::array<float, logit_rows> vals{};
                    std::array<int, logit_rows> ids{};
                    for (int row = 0; row < logit_rows; ++row) {
                        int winner = (row + 2) % 4;
                        vals[row] = (r == winner) ? 10.0f + float(row) : float(r + row);
                        ids[row] = r * 100 + row;
                    }
                    vals[0] = (r == 1 || r == 3) ? 25.0f : 1.0f;
                    ids[0] = (r == 1) ? 150 : (r == 3 ? 350 : r * 100);
                    checked(hipMemcpyAsync(local_vals[r], vals.data(), logit_rows * sizeof(float),
                                           hipMemcpyHostToDevice, gpu.compute[r]));
                    checked(hipMemcpyAsync(local_ids[r], ids.data(), logit_rows * sizeof(int),
                                           hipMemcpyHostToDevice, gpu.compute[r]));
                    gpu.tp_argmax(local_vals[r], local_ids[r], tokens_out[r], r, logit_rows);
                    std::array<int, logit_rows> out_ids{};
                    checked(hipMemcpyAsync(out_ids.data(), tokens_out[r], logit_rows * sizeof(int),
                                           hipMemcpyDeviceToHost, gpu.compute[r]));
                    std::vector<float> slice(size_t(logit_rows) * local_vocab);
                    for (int row = 0; row < logit_rows; ++row)
                        for (int j = 0; j < local_vocab; ++j)
                            slice[size_t(row) * local_vocab + j] = float(row * 1000 + r * local_vocab + j);
                    checked(hipMemcpyAsync(local_logits[r], slice.data(), slice.size() * sizeof(float),
                                           hipMemcpyHostToDevice, gpu.compute[r]));
                    gpu.tp_gather_logits(local_logits[r], full_logits[r], r, logit_rows, local_vocab, 4 * local_vocab);
                    std::vector<float> gathered(size_t(logit_rows) * 4u * local_vocab);
                    if (r == 0)
                        checked(hipMemcpyAsync(gathered.data(), full_logits[0], gathered.size() * sizeof(float),
                                               hipMemcpyDeviceToHost, gpu.compute[0]));
                    checked(hipStreamSynchronize(gpu.compute[r]));
                    if (out_ids[0] != 150)
                        throw std::runtime_error("TP=4 argmax の同値判定が不正です");
                    for (int row = 1; row < logit_rows; ++row)
                        if (out_ids[row] != ((row + 2) % 4) * 100 + row)
                            throw std::runtime_error("TP=4 argmax の結果が不正です");
                    if (r == 0)
                        for (int row = 0; row < logit_rows; ++row)
                            for (int col = 0; col < 4 * local_vocab; ++col)
                                if (gathered[size_t(row) * 4u * local_vocab + col] != float(row * 1000 + col))
                                    throw std::runtime_error("TP=4 logits gather の結果が不正です");
                } catch (const std::exception &e) {
                    std::fprintf(stderr, "rank=%d err=%s\n", r, e.what());
                    failures[r] = std::current_exception();
                    gpu.abort_collectives();
                } catch (...) {
                    failures[r] = std::current_exception();
                    gpu.abort_collectives();
                }
            });
        for (auto &worker : workers)
            worker.join();
        for (auto &failure : failures)
            if (failure)
                std::rethrow_exception(failure);
    }
    gpu.shutdown();
}
}

int main() {
    constexpr size_t count = 65539;
    std::vector<uint16_t> a(count, 0x3f80), b(count, 0x4000), output(count);
    uint16_t *device_a = nullptr, *device_b = nullptr;
    auto check = [](hipError_t status) {
        if (status != hipSuccess) {
            std::fprintf(stderr, "%s\n", hipGetErrorString(status));
            return false;
        }
        return true;
    };
    bool ok = check(hipMalloc(&device_a, count * sizeof(uint16_t))) &&
              check(hipMalloc(&device_b, count * sizeof(uint16_t)));
    if (ok)
        ok = check(hipMemcpy(device_a, a.data(), count * sizeof(uint16_t), hipMemcpyHostToDevice)) &&
             check(hipMemcpy(device_b, b.data(), count * sizeof(uint16_t), hipMemcpyHostToDevice));
    if (ok) {
        shelf::hy4::tp_add_bf16(device_a, device_b, count * sizeof(uint16_t));
        ok = check(hipGetLastError()) &&
             check(hipMemcpy(output.data(), device_a, count * sizeof(uint16_t), hipMemcpyDeviceToHost));
    }
    for (auto value : output)
        if (value != 0x4040)
            ok = false;
    if (device_b)
        ok = check(hipFree(device_b)) && ok;
    if (device_a)
        ok = check(hipFree(device_a)) && ok;
    if (ok) {
        try {
            check_peer_collectives();
            check_lm_head_collectives();
            check_tp4_collectives();
        } catch (const std::exception &e) {
            std::fprintf(stderr, "%s\n", e.what());
            return 1;
        }
    }
    return ok ? 0 : 1;
}
