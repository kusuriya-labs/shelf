#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/kernels/dflash2/attention.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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

static uint32_t lcg(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return s;
}

static bf16_t random_bf16(uint32_t& s) {
    const float u = static_cast<float>(lcg(s) >> 8) / static_cast<float>(1u << 24);
    return bf16_t{f32_to_bf16_rne((u * 2.0f - 1.0f) * 2.0f)};
}

struct RingAttentionCase {
    uint32_t q_heads = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t context_rows = 0;
    uint32_t block_rows = 0;
    uint32_t context_start = 0;
    uint32_t block_start = 0;
    uint32_t window = 0;
    uint32_t capacity = 0;
    uint32_t seed = 1;
    float scale = 1.0f;
    bool random = false;
    std::vector<bf16_t> q;
    std::vector<bf16_t> k_ctx;
    std::vector<bf16_t> v_ctx;
    std::vector<bf16_t> k_noise;
    std::vector<bf16_t> v_noise;
};

static void fill_value(std::vector<bf16_t>& data, float value) {
    for (bf16_t& v : data) v = bf16_t{f32_to_bf16_rne(value)};
}

static void build_case(RingAttentionCase& c) {
    const uint32_t kv_features = c.kv_heads * c.head_dim;
    const uint32_t q_features = c.q_heads * c.head_dim;
    c.scale = 1.0f / std::sqrt(static_cast<float>(c.head_dim));
    c.q.assign(static_cast<std::size_t>(c.block_rows) * q_features, bf16_t{0});
    c.k_ctx.assign(static_cast<std::size_t>(c.context_rows) * kv_features, bf16_t{0});
    c.v_ctx.assign(static_cast<std::size_t>(c.context_rows) * kv_features, bf16_t{0});
    c.k_noise.assign(static_cast<std::size_t>(c.block_rows) * kv_features, bf16_t{0});
    c.v_noise.assign(static_cast<std::size_t>(c.block_rows) * kv_features, bf16_t{0});
    if (!c.random) {
        fill_value(c.q, 1.0f);
        fill_value(c.k_ctx, 1.0f);
        fill_value(c.k_noise, 1.0f);
        fill_value(c.v_ctx, 0.0f);
        fill_value(c.v_noise, 0.0f);
        return;
    }
    uint32_t s = c.seed;
    for (bf16_t& v : c.q) v = random_bf16(s);
    for (bf16_t& v : c.k_ctx) v = random_bf16(s);
    for (bf16_t& v : c.v_ctx) v = random_bf16(s);
    for (bf16_t& v : c.k_noise) v = random_bf16(s);
    for (bf16_t& v : c.v_noise) v = random_bf16(s);
}

static bool run_case(
    ps::gpu::GpuArena& arena,
    hipStream_t stream,
    const RingAttentionCase& c,
    std::vector<bf16_t>& contiguous_out) {
    const uint32_t kv_features = c.kv_heads * c.head_dim;
    const uint32_t q_features = c.q_heads * c.head_dim;

    ps::gpu::Tensor dq;
    ps::gpu::Tensor dkc;
    ps::gpu::Tensor dvc;
    ps::gpu::Tensor dkn;
    ps::gpu::Tensor dvn;
    ps::gpu::Tensor dout;
    if (!alloc_bf16(arena, {c.block_rows, q_features}, dq) ||
        !alloc_bf16(arena, {c.context_rows, kv_features}, dkc) ||
        !alloc_bf16(arena, {c.context_rows, kv_features}, dvc) ||
        !alloc_bf16(arena, {c.block_rows, kv_features}, dkn) ||
        !alloc_bf16(arena, {c.block_rows, kv_features}, dvn) ||
        !alloc_bf16(arena, {c.block_rows, q_features}, dout)) {
        check(false, "attention ring alloc");
        return false;
    }
    const auto upload = [](ps::gpu::Tensor& dst, const std::vector<bf16_t>& src) {
        return hipMemcpy(dst.data<bf16_t>(), src.data(), src.size() * sizeof(bf16_t),
                         hipMemcpyHostToDevice) == hipSuccess;
    };
    check(upload(dq, c.q), "q upload");
    check(upload(dkc, c.k_ctx), "k upload");
    check(upload(dvc, c.v_ctx), "v upload");
    check(upload(dkn, c.k_noise), "kn upload");
    check(upload(dvn, c.v_noise), "vn upload");

    hipError_t err = ps::kernel::launch_dflash2_attention_bf16(
        dq.data<bf16_t>(), q_features, dkc.data<bf16_t>(), dvc.data<bf16_t>(), c.context_rows,
        kv_features, dkn.data<bf16_t>(), dvn.data<bf16_t>(), c.block_rows, kv_features,
        dout.data<bf16_t>(), q_features, c.q_heads, c.kv_heads, c.head_dim, c.context_start,
        c.block_start, c.window, c.scale, stream);
    check(err == hipSuccess, "contiguous attention launch");
    check(hipStreamSynchronize(stream) == hipSuccess, "contiguous sync");
    contiguous_out.resize(static_cast<std::size_t>(c.block_rows) * q_features);
    check(hipMemcpy(contiguous_out.data(), dout.data<bf16_t>(),
                    contiguous_out.size() * sizeof(bf16_t), hipMemcpyDeviceToHost) == hipSuccess,
          "contiguous read");
    return true;
}

static void compare_ring(
    ps::gpu::GpuArena& arena,
    hipStream_t stream,
    const RingAttentionCase& c,
    const std::vector<bf16_t>& want,
    const std::string& tag) {
    const uint32_t kv_features = c.kv_heads * c.head_dim;
    const uint32_t q_features = c.q_heads * c.head_dim;

    std::vector<bf16_t> k_ring(static_cast<std::size_t>(c.capacity) * kv_features,
                               bf16_t{f32_to_bf16_rne(123.0f)});
    std::vector<bf16_t> v_ring(static_cast<std::size_t>(c.capacity) * kv_features,
                               bf16_t{f32_to_bf16_rne(-77.0f)});
    for (uint32_t i = 0; i < c.context_rows; ++i) {
        const uint32_t slot = (c.context_start + i) % c.capacity;
        for (uint32_t d = 0; d < kv_features; ++d) {
            k_ring[static_cast<std::size_t>(slot) * kv_features + d] =
                c.k_ctx[static_cast<std::size_t>(i) * kv_features + d];
            v_ring[static_cast<std::size_t>(slot) * kv_features + d] =
                c.v_ctx[static_cast<std::size_t>(i) * kv_features + d];
        }
    }

    ps::gpu::Tensor dq;
    ps::gpu::Tensor dkr;
    ps::gpu::Tensor dvr;
    ps::gpu::Tensor dkn;
    ps::gpu::Tensor dvn;
    ps::gpu::Tensor dout;
    check(alloc_bf16(arena, {c.block_rows, q_features}, dq), tag + " q alloc");
    check(alloc_bf16(arena, {c.capacity, kv_features}, dkr), tag + " kr alloc");
    check(alloc_bf16(arena, {c.capacity, kv_features}, dvr), tag + " vr alloc");
    check(alloc_bf16(arena, {c.block_rows, kv_features}, dkn), tag + " kn alloc");
    check(alloc_bf16(arena, {c.block_rows, kv_features}, dvn), tag + " vn alloc");
    check(alloc_bf16(arena, {c.block_rows, q_features}, dout), tag + " out alloc");
    check(hipMemcpy(dq.data<bf16_t>(), c.q.data(), c.q.size() * sizeof(bf16_t),
                    hipMemcpyHostToDevice) == hipSuccess,
          tag + " q upload");
    check(hipMemcpy(dkr.data<bf16_t>(), k_ring.data(), k_ring.size() * sizeof(bf16_t),
                    hipMemcpyHostToDevice) == hipSuccess,
          tag + " kr upload");
    check(hipMemcpy(dvr.data<bf16_t>(), v_ring.data(), v_ring.size() * sizeof(bf16_t),
                    hipMemcpyHostToDevice) == hipSuccess,
          tag + " vr upload");
    check(hipMemcpy(dkn.data<bf16_t>(), c.k_noise.data(),
                    c.k_noise.size() * sizeof(bf16_t), hipMemcpyHostToDevice) == hipSuccess,
          tag + " kn upload");
    check(hipMemcpy(dvn.data<bf16_t>(), c.v_noise.data(),
                    c.v_noise.size() * sizeof(bf16_t), hipMemcpyHostToDevice) == hipSuccess,
          tag + " vn upload");

    const hipError_t err = ps::kernel::launch_dflash2_attention_ring_bf16(
        dq.data<bf16_t>(), q_features, dkr.data<bf16_t>(), dvr.data<bf16_t>(), c.capacity,
        c.context_rows, c.context_start, kv_features, dkn.data<bf16_t>(), dvn.data<bf16_t>(),
        c.block_rows, kv_features, dout.data<bf16_t>(), q_features, c.q_heads, c.kv_heads,
        c.head_dim, c.block_start, c.window, c.scale, stream);
    check(err == hipSuccess, tag + " ring launch");
    check(hipStreamSynchronize(stream) == hipSuccess, tag + " ring sync");
    std::vector<bf16_t> got(want.size());
    check(hipMemcpy(got.data(), dout.data<bf16_t>(), got.size() * sizeof(bf16_t),
                    hipMemcpyDeviceToHost) == hipSuccess,
          tag + " ring read");

    std::size_t mismatch = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (got[i].data != want[i].data) ++mismatch;
    }
    std::printf("  %-34s bit-exact=%s (%zu/%zu)\n", tag.c_str(),
                mismatch == 0 ? "yes" : "no", got.size() - mismatch, got.size());
    check(mismatch == 0, tag + " ring vs contiguous bit exact");

    if (c.q_heads == 4u * c.kv_heads && c.head_dim == 128u) {
        const hipError_t gerr = ps::kernel::launch_dflash2_attention_ring_gqa_bf16(
            dq.data<bf16_t>(), q_features, dkr.data<bf16_t>(), dvr.data<bf16_t>(), c.capacity,
            c.context_rows, c.context_start, kv_features, dkn.data<bf16_t>(), dvn.data<bf16_t>(),
            c.block_rows, kv_features, dout.data<bf16_t>(), q_features, c.q_heads, c.kv_heads,
            c.head_dim, c.block_start, c.window, c.scale, stream);
        check(gerr == hipSuccess, tag + " gqa ring launch");
        check(hipStreamSynchronize(stream) == hipSuccess, tag + " gqa ring sync");
        std::vector<bf16_t> got_g(want.size());
        check(hipMemcpy(got_g.data(), dout.data<bf16_t>(), got_g.size() * sizeof(bf16_t),
                        hipMemcpyDeviceToHost) == hipSuccess,
              tag + " gqa ring read");
        std::size_t gm = 0;
        for (std::size_t i = 0; i < got_g.size(); ++i) {
            if (got_g[i].data != want[i].data) ++gm;
        }
        std::printf("  %-34s gqa bit-exact=%s (%zu/%zu)\n", tag.c_str(),
                    gm == 0u ? "yes" : "no", got_g.size() - gm, got_g.size());
        check(gm == 0, tag + " gqa ring vs contiguous bit exact");
    }
}

static void run_compare(
    ps::gpu::GpuArena& arena,
    hipStream_t stream,
    RingAttentionCase c,
    const std::string& tag) {
    build_case(c);
    std::vector<bf16_t> want;
    if (!run_case(arena, stream, c, want)) return;
    compare_ring(arena, stream, c, want, tag);
}

static void run_boundary_marker(
    ps::gpu::GpuArena& arena,
    hipStream_t stream,
    uint32_t context_rows,
    uint32_t context_start,
    uint32_t block_start,
    uint32_t window,
    uint32_t marker_context_index,
    bool marker_is_context,
    bool expect_visible,
    const std::string& tag) {
    RingAttentionCase c;
    c.q_heads = 2u;
    c.kv_heads = 1u;
    c.head_dim = 4u;
    c.context_rows = context_rows;
    c.block_rows = 4u;
    c.context_start = context_start;
    c.block_start = block_start;
    c.window = window;
    c.capacity = context_rows < 8u ? 8u : context_rows;
    build_case(c);
    const uint32_t kv_features = c.kv_heads * c.head_dim;
    std::vector<bf16_t>& target = marker_is_context ? c.v_ctx : c.v_noise;
    const std::size_t base = static_cast<std::size_t>(marker_context_index) * kv_features;
    for (uint32_t d = 0; d < kv_features; ++d) {
        target[base + d] = bf16_t{f32_to_bf16_rne(1.0f)};
    }
    std::vector<bf16_t> want;
    if (!run_case(arena, stream, c, want)) return;
    bool any_nonzero = false;
    for (const bf16_t v : want) {
        if (v.data != 0u) any_nonzero = true;
    }
    check(any_nonzero == expect_visible,
          tag + (expect_visible ? " must be visible" : " must be masked"));
    compare_ring(arena, stream, c, want, tag);
}

int main() {
    auto arena_result = ps::gpu::GpuArena::create(0, 512ull * 1024ull * 1024ull);
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

    {
        RingAttentionCase c;
        c.q_heads = 4u;
        c.kv_heads = 2u;
        c.head_dim = 8u;
        c.context_rows = 8u;
        c.block_rows = 8u;
        c.context_start = 0u;
        c.block_start = 8u;
        c.window = 32u;
        c.capacity = 16u;
        c.random = true;
        c.seed = 1234u;
        run_compare(arena, stream, c, "no wrap ctx8 blk8 cap16");
    }
    {
        RingAttentionCase c;
        c.q_heads = 4u;
        c.kv_heads = 2u;
        c.head_dim = 8u;
        c.context_rows = 8u;
        c.block_rows = 4u;
        c.context_start = 3u;
        c.block_start = 11u;
        c.window = 32u;
        c.capacity = 8u;
        c.random = true;
        c.seed = 77u;
        run_compare(arena, stream, c, "one wrap ctx8 start3 cap8");
    }
    {
        RingAttentionCase c;
        c.q_heads = 4u;
        c.kv_heads = 2u;
        c.head_dim = 8u;
        c.context_rows = 5u;
        c.block_rows = 4u;
        c.context_start = 1003u;
        c.block_start = 1008u;
        c.window = 32u;
        c.capacity = 5u;
        c.random = true;
        c.seed = 9u;
        run_compare(arena, stream, c, "multi wrap ctx5 start1003");
    }
    {
        RingAttentionCase c;
        c.q_heads = 4u;
        c.kv_heads = 2u;
        c.head_dim = 8u;
        c.context_rows = 1u;
        c.block_rows = 2u;
        c.context_start = 7u;
        c.block_start = 8u;
        c.window = 32u;
        c.capacity = 8u;
        c.random = true;
        c.seed = 31u;
        run_compare(arena, stream, c, "context length 1");
    }
    {
        RingAttentionCase c;
        c.q_heads = 4u;
        c.kv_heads = 2u;
        c.head_dim = 8u;
        c.context_rows = 8u;
        c.block_rows = 2u;
        c.context_start = 0u;
        c.block_start = 8u;
        c.window = 32u;
        c.capacity = 8u;
        c.random = true;
        c.seed = 55u;
        run_compare(arena, stream, c, "context length capacity blk2");
    }
    {
        RingAttentionCase c;
        c.q_heads = 4u;
        c.kv_heads = 2u;
        c.head_dim = 8u;
        c.context_rows = 8u;
        c.block_rows = 8u;
        c.context_start = 0u;
        c.block_start = 8u;
        c.window = 32u;
        c.capacity = 8u;
        c.random = true;
        c.seed = 88u;
        run_compare(arena, stream, c, "context length capacity blk8");
    }
    {
        RingAttentionCase c;
        c.q_heads = 32u;
        c.kv_heads = 8u;
        c.head_dim = 128u;
        c.context_rows = 16u;
        c.block_rows = 8u;
        c.context_start = 5u;
        c.block_start = 21u;
        c.window = 2048u;
        c.capacity = 16u;
        c.random = true;
        c.seed = 4321u;
        run_compare(arena, stream, c, "gqa 32/8 head128");
    }
    {
        RingAttentionCase c;
        c.q_heads = 8u;
        c.kv_heads = 2u;
        c.head_dim = 128u;
        c.context_rows = 8u;
        c.block_rows = 4u;
        c.context_start = 3u;
        c.block_start = 11u;
        c.window = 32u;
        c.capacity = 8u;
        c.random = true;
        c.seed = 77u;
        run_compare(arena, stream, c, "gqa one wrap ctx8 start3 cap8");
    }
    {
        RingAttentionCase c;
        c.q_heads = 32u;
        c.kv_heads = 8u;
        c.head_dim = 128u;
        c.context_rows = 2041u;
        c.block_rows = 8u;
        c.context_start = 2044u;
        c.block_start = 4085u;
        c.window = 2048u;
        c.capacity = 2048u;
        c.random = true;
        c.seed = 555u;
        run_compare(arena, stream, c, "gqa total 2049 one wrap cap2048");
    }
    {
        RingAttentionCase c;
        c.q_heads = 32u;
        c.kv_heads = 8u;
        c.head_dim = 128u;
        c.context_rows = 2048u;
        c.block_rows = 8u;
        c.context_start = 0u;
        c.block_start = 2048u;
        c.window = 2048u;
        c.capacity = 2048u;
        c.random = true;
        c.seed = 777u;
        run_compare(arena, stream, c, "gqa total 2056 cap2048");
    }

    run_boundary_marker(arena, stream, 4u, 0u, 4u, 4u, 0u, true, false,
                        "distance 4 (window 4) masked");
    run_boundary_marker(arena, stream, 4u, 0u, 4u, 4u, 1u, true, true,
                        "distance 3 (window 4) visible");
    run_boundary_marker(arena, stream, 4u, 0u, 4u, 8u, 3u, false, true,
                        "future noise row +3 (window 8)");

    check(hipStreamSynchronize(stream) == hipSuccess, "final sync");
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    const ps::Status shutdown = arena.shutdown();
    check(shutdown.ok(), "arena shutdown: " + shutdown.message());

    std::printf("test_dflash2_attention_ring: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
