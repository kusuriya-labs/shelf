#include <phaseshift/core/memory/arena.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define HIP_ASSERT(x) do { \
    hipError_t err = x; \
    if (err != hipSuccess) { \
        printf("HIP error at %s:%d: %d\n", __FILE__, __LINE__, err); \
        return 1; \
    } \
} while(0)

namespace {

int passed = 0;
int failed = 0;

void check(bool cond, const char* name) {
    if (cond) {
        passed++;
        printf("PASS: %s\n", name);
    } else {
        failed++;
        printf("FAIL: %s\n", name);
    }
}

ps::gpu::DeviceAllocationView must_alloc(ps::gpu::GpuArena& arena, std::size_t bytes, std::size_t align, const char* name) {
    auto r = arena.allocate_aligned(bytes, align);
    if (!r.ok()) {
        check(false, name);
        printf("alloc failed: %s\n", r.status().message().c_str());
        std::exit(1);
    }
    return r.release();
}

}

int main() {
    printf("=== GpuArena VMM Tests ===\n");

    setenv("PHASESHIFT_ARENA_VMM", "1", 1);

    int device_count = 0;
    HIP_ASSERT(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("SKIP: no GPU available\n");
        return 77;
    }
    HIP_ASSERT(hipSetDevice(0));

    const std::size_t KiB = 1024ull;
    const std::size_t MiB = 1024ull * 1024ull;
    const std::size_t GiB = 1024ull * MiB;

    int supported = 0;
    HIP_ASSERT(hipDeviceGetAttribute(
        &supported, hipDeviceAttributeVirtualMemoryManagementSupported, 0));
    check(supported == 1, "VMM supported");
    if (supported != 1) {
        printf("\nResults: %d passed, %d failed\n", passed, failed);
        return 1;
    }

    {
        auto arena_result = ps::gpu::GpuArena::create(0, GiB);
        check(arena_result.ok(), "arena create ok");
        if (!arena_result.ok()) {
            printf("create failed: %s\n", arena_result.status().message().c_str());
            printf("\nResults: %d passed, %d failed\n", passed, failed);
            return 1;
        }
        auto arena = arena_result.release();

        check(arena.capacity() == GiB, "capacity == 1 GiB");
        check(arena.used() == 0, "used == 0 after create");
        check(arena.committed() == 0, "committed == 0 after create");
        check(arena.is_vmm(), "is_vmm true");

        auto a = must_alloc(arena, MiB, 256, "first alloc ok");
        check(a.data() != nullptr, "first alloc data non-null");
        check(a.bytes() == MiB, "first alloc bytes == 1 MiB");
        check(reinterpret_cast<std::uintptr_t>(a.data()) % 256 == 0, "first alloc ptr aligned 256");
        check(arena.used() >= MiB, "used >= 1 MiB");
        check(arena.committed() >= arena.used(), "committed >= used");
        check(arena.committed() < arena.capacity(), "committed < capacity");
        const std::size_t quantum = arena.committed();
        check(quantum >= 256 * MiB, "commit quantum >= 256 MiB");

        const std::size_t committed0 = arena.committed();
        auto a2 = must_alloc(arena, 4096, 256, "small alloc ok");
        (void)a2;
        check(arena.committed() == committed0, "no extra commit within chunk");

        HIP_ASSERT(hipMemset(a.data(), 0x5a, a.bytes()));
        uint8_t head = 0, tail = 0;
        HIP_ASSERT(hipMemcpy(&head, a.data(), 1, hipMemcpyDeviceToHost));
        HIP_ASSERT(hipMemcpy(&tail, reinterpret_cast<uint8_t*>(a.data()) + a.bytes() - 1, 1, hipMemcpyDeviceToHost));
        check(head == 0x5a, "gpu access head 0x5a");
        check(tail == 0x5a, "gpu access tail 0x5a");

        const std::size_t used_before = arena.used();
        const std::size_t committed_before = arena.committed();
        const std::size_t dummy_bytes = (quantum - 4 * KiB) - used_before;
        auto dummy = must_alloc(arena, dummy_bytes, 256, "boundary dummy alloc ok");
        (void)dummy;
        auto straddle = must_alloc(arena, 8 * KiB, 256, "straddle alloc ok");
        check(arena.committed() > committed_before, "committed grew crossing boundary");
        check(arena.committed() >= arena.used(), "committed >= used after boundary");
        check(arena.committed() <= GiB, "committed <= capacity after boundary");

        HIP_ASSERT(hipMemset(straddle.data(), 0x3c, straddle.bytes()));
        uint8_t s_head = 0, s_tail = 0;
        HIP_ASSERT(hipMemcpy(&s_head, straddle.data(), 1, hipMemcpyDeviceToHost));
        HIP_ASSERT(hipMemcpy(&s_tail, reinterpret_cast<uint8_t*>(straddle.data()) + straddle.bytes() - 1, 1, hipMemcpyDeviceToHost));
        check(s_head == 0x3c, "mapping boundary head 0x3c");
        check(s_tail == 0x3c, "mapping boundary tail 0x3c");

        auto st = arena.shutdown();
        check(st.ok(), "arena A shutdown ok");
    }

    {
        auto c_result = ps::gpu::GpuArena::create(0, GiB);
        check(c_result.ok(), "move ctor arena create ok");
        if (!c_result.ok()) {
            printf("\nResults: %d passed, %d failed\n", passed, failed);
            return 1;
        }
        auto c = c_result.release();
        auto a = must_alloc(c, MiB, 256, "move ctor alloc ok");

        ps::gpu::GpuArena moved = std::move(c);
        check(moved.used() >= MiB, "moved has allocation");
        HIP_ASSERT(hipMemset(a.data(), 0x77, a.bytes()));
        uint8_t mv = 0;
        HIP_ASSERT(hipMemcpy(&mv, a.data(), 1, hipMemcpyDeviceToHost));
        check(mv == 0x77, "moved existing pointer valid");
        auto a2 = moved.allocate_aligned(4096, 256);
        check(a2.ok(), "moved can allocate more");
        auto mst = moved.shutdown();
        check(mst.ok() && moved.is_shutdown(), "moved shutdown ok");
    }

    {
        auto old_result = ps::gpu::GpuArena::create(0, GiB);
        check(old_result.ok(), "move assign old create ok");
        if (!old_result.ok()) {
            printf("\nResults: %d passed, %d failed\n", passed, failed);
            return 1;
        }
        auto old = old_result.release();
        must_alloc(old, MiB, 256, "move assign old alloc ok");

        auto src_result = ps::gpu::GpuArena::create(0, GiB);
        check(src_result.ok(), "move assign src create ok");
        if (!src_result.ok()) {
            printf("\nResults: %d passed, %d failed\n", passed, failed);
            return 1;
        }
        auto src = src_result.release();
        auto sa = must_alloc(src, MiB, 256, "move assign src alloc ok");

        old = std::move(src);
        check(old.used() >= MiB, "old now owns src allocation");
        HIP_ASSERT(hipMemset(sa.data(), 0x88, sa.bytes()));
        uint8_t av = 0;
        HIP_ASSERT(hipMemcpy(&av, sa.data(), 1, hipMemcpyDeviceToHost));
        check(av == 0x88, "move assign new mapping valid");
        auto ast = old.shutdown();
        check(ast.ok(), "move assign shutdown ok");
    }

    {
        auto d_result = ps::gpu::GpuArena::create(0, GiB);
        check(d_result.ok(), "shutdown arena create ok");
        if (!d_result.ok()) {
            printf("\nResults: %d passed, %d failed\n", passed, failed);
            return 1;
        }
        auto d = d_result.release();
        must_alloc(d, MiB, 256, "shutdown arena alloc ok");

        auto st = d.shutdown();
        check(st.ok(), "shutdown ok");
        check(d.is_shutdown(), "is_shutdown true");
        check(d.capacity() == 0, "capacity 0 after shutdown");
        check(d.used() == 0, "used 0 after shutdown");
        check(d.committed() == 0, "committed 0 after shutdown");

        auto st2 = d.shutdown();
        check(st2.ok(), "double shutdown ok");
    }

    printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
