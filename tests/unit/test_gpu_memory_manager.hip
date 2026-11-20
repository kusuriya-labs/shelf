#include <shelf/core/memory/gpu_memory_manager.h>

#include <hip/hip_runtime.h>

#include <cstdio>

namespace {

int passed = 0;
int failed = 0;

void check(bool cond, const char* name) {
    if (cond) {
        passed++;
        std::printf("PASS: %s\n", name);
    } else {
        failed++;
        std::printf("FAIL: %s\n", name);
    }
}

}  // namespace

int main() {
    std::printf("=== GpuMemoryManager Tests ===\n");

    int device_count = 0;
    const hipError_t count_err = hipGetDeviceCount(&device_count);
    if (count_err != hipSuccess) {
        std::printf("HIP error: %d\n", static_cast<int>(count_err));
        return 1;
    }
    if (device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }

    const std::size_t capacity = 256ull * 1024ull * 1024ull;

    {
        auto empty = shelf::gpu::GpuMemoryManager::create({}, capacity);
        check(!empty.ok(), "empty device list rejected");
    }
    {
        auto duplicated = shelf::gpu::GpuMemoryManager::create({0, 0}, capacity);
        check(!duplicated.ok(), "duplicate device rejected");
    }

    auto manager_result = shelf::gpu::GpuMemoryManager::create({0}, capacity);
    check(manager_result.ok(), "manager create ok");
    if (!manager_result.ok()) {
        std::printf("create failed: %s\n", manager_result.status().message().c_str());
        std::printf("\nResults: %d passed, %d failed\n", passed, failed);
        return 1;
    }
    shelf::gpu::GpuMemoryManager manager = manager_result.release();

    check(manager.device_count() == 1u, "device_count == 1");
    check(!manager.is_shutdown(), "manager is live after create");

    auto arena_result = manager.arena_for(0);
    check(arena_result.ok(), "arena_for(0) ok");
    if (arena_result.ok()) {
        shelf::gpu::GpuArena& arena = *arena_result.value();
        check(arena.capacity() == capacity, "borrowed arena capacity matches");
        check(arena.device() == 0, "borrowed arena device matches");
        auto allocation = arena.allocate_aligned(4096u, 256u);
        check(allocation.ok(), "borrowed arena allocates");
        check(arena.used() == 4096u, "borrowed arena tracks the allocation");
    }

    auto missing = manager.arena_for(device_count + 1);
    check(!missing.ok(), "unknown device rejected");

    const shelf::Status shutdown_status = manager.shutdown();
    check(shutdown_status.ok(), "shutdown ok");
    check(manager.is_shutdown(), "manager reports shutdown");

    std::printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
