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
                checked(hipEventCreateWithFlags(&event, hipEventDisableTiming | hipEventReleaseToSystem));
    }
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
                        checked(hipEventCreateWithFlags(&(*events)[r][p],
                                                        hipEventDisableTiming | hipEventReleaseToSystem));
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
                        checked(hipEventCreateWithFlags(&(*events)[r][p],
                                                        hipEventDisableTiming | hipEventReleaseToSystem));
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
        } catch (const std::exception &e) {
            std::fprintf(stderr, "%s\n", e.what());
            return 1;
        }
    }
    return ok ? 0 : 1;
}
