#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/kernels/dflash2/kv_ring.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <algorithm>

using ps::bf16_t;
using ps::f32_to_bf16_rne;

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

static bool alloc_bf16(
    ps::gpu::GpuArena& arena, const std::vector<std::size_t>& shape, ps::gpu::Tensor& out) {
    std::size_t count = 1;
    for (std::size_t d : shape) count *= d;
    auto view = arena.allocate_aligned(count * sizeof(bf16_t), 128);
    if (!view.ok()) return false;
    auto backing = view.release();
    std::vector<std::size_t> strides(shape.size());
    std::size_t stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }
    auto t = ps::gpu::Tensor::create<bf16_t>(backing, shape, strides);
    if (!t.ok()) return false;
    out = t.release();
    return true;
}

static bf16_t value_k(uint32_t position, uint32_t feature) {
    const uint32_t raw = (position * 131u + feature * 17u) % 4096u;
    return bf16_t{f32_to_bf16_rne(static_cast<float>(raw) * 0.001f - 2.0f)};
}

static bf16_t value_v(uint32_t position, uint32_t feature) {
    const uint32_t raw = (position * 29u + feature * 97u + 13u) % 4096u;
    return bf16_t{f32_to_bf16_rne(static_cast<float>(raw) * 0.0005f - 1.0f)};
}

static std::vector<bf16_t> make_src(
    uint32_t rows, uint32_t position_start, uint32_t kv_features, bool is_k) {
    std::vector<bf16_t> out(static_cast<std::size_t>(rows) * kv_features);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t d = 0; d < kv_features; ++d) {
            out[static_cast<std::size_t>(r) * kv_features + d] =
                is_k ? value_k(position_start + r, d) : value_v(position_start + r, d);
        }
    }
    return out;
}

struct Ring {
    ps::gpu::Tensor k;
    ps::gpu::Tensor v;
    std::vector<bf16_t> host_k;
    std::vector<bf16_t> host_v;
    uint32_t capacity = 0;
    uint32_t kv_features = 0;
    uint32_t length = 0;
    uint32_t next_position = 0;

    uint32_t start() const { return next_position - length; }
};

static void append(
    Ring& ring,
    ps::gpu::GpuArena& arena,
    hipStream_t stream,
    const std::vector<bf16_t>& k_src,
    const std::vector<bf16_t>& v_src,
    uint32_t rows,
    uint32_t position_start) {
    const uint32_t capacity = ring.capacity;
    const uint32_t skip = rows > capacity ? rows - capacity : 0u;
    const uint32_t effective_rows = rows > capacity ? capacity : rows;
    const uint32_t effective_position = position_start + skip;

    ps::gpu::Tensor dk;
    ps::gpu::Tensor dv;
    check(alloc_bf16(arena, {rows, ring.kv_features}, dk), "src k alloc");
    check(alloc_bf16(arena, {rows, ring.kv_features}, dv), "src v alloc");
    check(hipMemcpy(dk.data<bf16_t>(), k_src.data(), k_src.size() * sizeof(bf16_t),
                    hipMemcpyHostToDevice) == hipSuccess,
          "k upload");
    check(hipMemcpy(dv.data<bf16_t>(), v_src.data(), v_src.size() * sizeof(bf16_t),
                    hipMemcpyHostToDevice) == hipSuccess,
          "v upload");

    const hipError_t err = ps::kernel::launch_dflash2_kv_ring_append_bf16(
        dk.data<bf16_t>() + static_cast<std::size_t>(skip) * ring.kv_features,
        dv.data<bf16_t>() + static_cast<std::size_t>(skip) * ring.kv_features,
        ring.k.data<bf16_t>(), ring.v.data<bf16_t>(), effective_rows, ring.kv_features,
        capacity, effective_position, stream);
    check(err == hipSuccess, "append launch");
    check(hipStreamSynchronize(stream) == hipSuccess, "append sync");

    const std::size_t total = static_cast<std::size_t>(capacity) * ring.kv_features;
    ring.host_k.resize(total);
    ring.host_v.resize(total);
    check(hipMemcpy(ring.host_k.data(), ring.k.data<bf16_t>(), total * sizeof(bf16_t),
                    hipMemcpyDeviceToHost) == hipSuccess,
          "k read");
    check(hipMemcpy(ring.host_v.data(), ring.v.data<bf16_t>(), total * sizeof(bf16_t),
                    hipMemcpyDeviceToHost) == hipSuccess,
          "v read");

    const uint64_t new_length = static_cast<uint64_t>(ring.length) + rows;
    ring.length = new_length > capacity ? capacity : static_cast<uint32_t>(new_length);
    ring.next_position = position_start + rows;
}

static void verify_retained(const Ring& ring, const std::string& tag) {
    for (uint32_t i = 0; i < ring.length; ++i) {
        const uint32_t position = ring.start() + i;
        const uint32_t slot = position % ring.capacity;
        for (uint32_t d = 0; d < ring.kv_features; ++d) {
            const std::size_t idx = static_cast<std::size_t>(slot) * ring.kv_features + d;
            if (ring.host_k[idx].data != value_k(position, d).data ||
                ring.host_v[idx].data != value_v(position, d).data) {
                check(false, tag + " mismatch at logical " + std::to_string(i) +
                                 " position " + std::to_string(position));
                return;
            }
        }
    }
}

static void run_suite(
    ps::gpu::GpuArena& arena,
    hipStream_t stream,
    uint32_t capacity,
    uint32_t kv_features) {
    const std::string base =
        "cap" + std::to_string(capacity) + "kv" + std::to_string(kv_features);

    Ring ring;
    ring.capacity = capacity;
    ring.kv_features = kv_features;
    check(alloc_bf16(arena, {capacity, kv_features}, ring.k), base + " k alloc");
    check(alloc_bf16(arena, {capacity, kv_features}, ring.v), base + " v alloc");

    append(ring, arena, stream, make_src(4u, 0u, kv_features, true),
           make_src(4u, 0u, kv_features, false), 4u, 0u);
    check(ring.length == std::min<uint32_t>(capacity, 4u) && ring.next_position == 4u,
          base + " A metadata");
    verify_retained(ring, base + " A");

    const uint32_t after_a = ring.length;
    append(ring, arena, stream, make_src(4u, 4u, kv_features, true),
           make_src(4u, 4u, kv_features, false), 4u, 4u);
    check(ring.length == std::min<uint32_t>(capacity, after_a + 4u) &&
              ring.next_position == 8u,
          base + " B metadata");
    verify_retained(ring, base + " B");

    const uint32_t after_b = ring.length;
    append(ring, arena, stream, make_src(1u, 8u, kv_features, true),
           make_src(1u, 8u, kv_features, false), 1u, 8u);
    check(ring.length == std::min<uint32_t>(capacity, after_b + 1u) &&
              ring.next_position == 9u && ring.start() == 9u - ring.length,
          base + " C metadata");
    verify_retained(ring, base + " C");

    const uint32_t after_c = ring.length;
    append(ring, arena, stream, make_src(4u, 9u, kv_features, true),
           make_src(4u, 9u, kv_features, false), 4u, 9u);
    check(ring.length == std::min<uint32_t>(capacity, after_c + 4u) &&
              ring.next_position == 13u && ring.start() == 13u - ring.length,
          base + " D metadata");
    verify_retained(ring, base + " D");

    ring.length = 0;
    ring.next_position = 100u;
    append(ring, arena, stream, make_src(3u, 100u, kv_features, true),
           make_src(3u, 100u, kv_features, false), 3u, 100u);
    check(ring.length == 3u && ring.next_position == 103u, base + " E metadata");
    verify_retained(ring, base + " E");

    Ring big;
    big.capacity = capacity;
    big.kv_features = kv_features;
    check(alloc_bf16(arena, {capacity, kv_features}, big.k), base + " big k alloc");
    check(alloc_bf16(arena, {capacity, kv_features}, big.v), base + " big v alloc");
    append(big, arena, stream, make_src(20u, 0u, kv_features, true),
           make_src(20u, 0u, kv_features, false), 20u, 0u);
    check(big.length == std::min<uint32_t>(capacity, 20u) && big.next_position == 20u &&
              big.start() == 20u - big.length,
          base + " overflow metadata");
    verify_retained(big, base + " overflow");
}

int main() {
    auto arena_result = ps::gpu::GpuArena::create(0, 256ull * 1024ull * 1024ull);
    if (!arena_result.ok()) {
        std::printf("SKIP: arena (%s)\n", arena_result.status().message().c_str());
        return 77;
    }
    ps::gpu::GpuArena arena = arena_result.release();
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        std::printf("FAIL hipStreamCreate\n");
        return 1;
    }

    run_suite(arena, stream, 8u, 16u);
    run_suite(arena, stream, 3u, 4u);
    run_suite(arena, stream, 2048u, 8u);

    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    const ps::Status shutdown = arena.shutdown();
    check(shutdown.ok(), "arena shutdown: " + shutdown.message());

    std::printf("test_dflash2_kv_ring: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
