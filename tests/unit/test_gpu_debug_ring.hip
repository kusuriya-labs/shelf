#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/gpu/debug_ring.h>
#include <shelf/core/gpu/scoped_device.h>

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <thread>

using shelf::gpu::DebugRing;
using shelf::gpu::DebugRingDevice;
using shelf::gpu::DebugRingRecord;

static int g_fail = 0;

static void fail(const char* msg) {
    ++g_fail;
    std::printf("FAIL %s\n", msg);
}

static void check(bool cond, const char* msg) {
    if (!cond) fail(msg);
}

__global__ void push_kernel(DebugRingDevice ring, uint32_t base, uint32_t count) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    for (uint32_t i = 0u; i < count; ++i) {
        DebugRingRecord record{};
        record.kind = 1u;
        record.rank = 0u;
        record.stage = i;
        record.value0 = base + i;
        record.value1 = 0x1234u + i;
        record.value2 = 7u;
        record.data[0] = 0x1111222233334444ull + i;
        (void)shelf::gpu::shelf_debug_ring_push(ring, record);
    }
}

__global__ void spin_kernel(DebugRingDevice ring, int* release) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    DebugRingRecord record{};
    record.kind = 9u;
    record.rank = 0u;
    record.stage = 5u;
    record.value0 = 4242u;
    (void)shelf::gpu::shelf_debug_ring_push(ring, record);
    constexpr int kSpinLimit = 1 << 30;
    for (int spins = 0; spins < kSpinLimit; ++spins) {
        if (__hip_atomic_load(release, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != 0) {
            return;
        }
        __builtin_amdgcn_s_sleep(64);
    }
}

static void test_basic() {
    auto ring_result = DebugRing::create(0, 64);
    check(ring_result.ok(), "create ring");
    if (!ring_result.ok()) return;
    auto ring = ring_result.release();
    auto scope = shelf::gpu::ScopedDevice::create(0);
    check(scope.ok(), "scope device 0");
    if (!scope.ok()) return;
    hipStream_t stream = nullptr;
    check(hipStreamCreate(&stream) == hipSuccess, "create stream");

    const DebugRingDevice view = ring->device_view();
    check(view.slots != nullptr, "device view slots");
    push_kernel<<<1, 1, 0, stream>>>(view, 100u, 8u);
    check(hipStreamSynchronize(stream) == hipSuccess, "sync basic");
    check(ring->pending() == 8u, "basic pending");

    DebugRingRecord records[8]{};
    const std::size_t got = ring->drain(records, 8u);
    check(got == 8u, "basic drain count");
    for (std::size_t i = 0; i < got; ++i) {
        check(records[i].sequence == i, "basic sequence");
        check(records[i].kind == 1u, "basic kind");
        check(records[i].stage == i, "basic stage");
        check(records[i].value0 == 100u + i, "basic value0");
        check(records[i].value1 == 0x1234u + i, "basic value1");
        check(records[i].data[0] == 0x1111222233334444ull + i, "basic data");
    }
    check(ring->pending() == 0u, "basic empty");
    check(!ring->try_pop(records[0]), "basic pop empty");

    shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));
}

static void test_drop() {
    auto ring_result = DebugRing::create(0, 8);
    check(ring_result.ok(), "create drop ring");
    if (!ring_result.ok()) return;
    auto ring = ring_result.release();
    auto scope = shelf::gpu::ScopedDevice::create(0);
    check(scope.ok(), "scope drop device");
    if (!scope.ok()) return;
    hipStream_t stream = nullptr;
    check(hipStreamCreate(&stream) == hipSuccess, "create drop stream");

    push_kernel<<<1, 1, 0, stream>>>(ring->device_view(), 0u, 20u);
    check(hipStreamSynchronize(stream) == hipSuccess, "sync drop");
    check(ring->pending() == 8u, "drop pending capped");
    check(ring->lost() == 12u, "drop lost count");

    DebugRingRecord records[8]{};
    const std::size_t got = ring->drain(records, 8u);
    check(got == 8u, "drop drain count");
    for (std::size_t i = 0; i < got; ++i) check(records[i].value0 == i, "drop keeps oldest");

    shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));
}

static void test_read_while_running() {
    auto ring_result = DebugRing::create(0, 16);
    check(ring_result.ok(), "create spin ring");
    if (!ring_result.ok()) return;
    auto ring = ring_result.release();
    auto scope = shelf::gpu::ScopedDevice::create(0);
    check(scope.ok(), "scope spin device");
    if (!scope.ok()) return;
    hipStream_t stream = nullptr;
    check(hipStreamCreate(&stream) == hipSuccess, "create spin stream");

    int* release = nullptr;
    const hipError_t alloc = hipHostMalloc(reinterpret_cast<void**>(&release), sizeof(int),
                                           hipHostMallocMapped);
    check(alloc == hipSuccess && release != nullptr, "alloc release flag");
    if (release == nullptr) return;
    *release = 0;

    spin_kernel<<<1, 1, 0, stream>>>(ring->device_view(), release);

    bool observed = false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        if (ring->pending() != 0u) {
            observed = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    check(observed, "record visible while kernel spins");

    DebugRingRecord record{};
    check(ring->try_pop(record), "pop while running");
    check(record.kind == 9u, "spin kind");
    check(record.value0 == 4242u, "spin value0");

    *release = 1;
    check(hipStreamSynchronize(stream) == hipSuccess, "sync spin");
    check(ring->pending() == 0u, "spin drained");

    shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    shelf::gpu::discard_cleanup_result(hipHostFree(release));
}

static void test_second_device() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 2) return;
    auto ring_result = DebugRing::create(1, 16);
    check(ring_result.ok(), "create ring device 1");
    if (!ring_result.ok()) return;
    auto ring = ring_result.release();
    auto scope = shelf::gpu::ScopedDevice::create(1);
    check(scope.ok(), "scope device 1");
    if (!scope.ok()) return;
    hipStream_t stream = nullptr;
    check(hipStreamCreate(&stream) == hipSuccess, "create stream device 1");
    push_kernel<<<1, 1, 0, stream>>>(ring->device_view(), 500u, 3u);
    check(hipStreamSynchronize(stream) == hipSuccess, "sync device 1");
    DebugRingRecord records[3]{};
    check(ring->drain(records, 3u) == 3u, "drain device 1");
    check(records[2].value0 == 502u, "device 1 payload");
    shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));
}

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 1) {
        std::printf("SKIP: debug ring requires a GPU\n");
        return 77;
    }
    test_basic();
    test_drop();
    test_read_while_running();
    test_second_device();
    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
