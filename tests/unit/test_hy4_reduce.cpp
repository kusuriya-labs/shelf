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
        } catch (const std::exception &e) {
            std::fprintf(stderr, "%s\n", e.what());
            return 1;
        }
    }
    return ok ? 0 : 1;
}
