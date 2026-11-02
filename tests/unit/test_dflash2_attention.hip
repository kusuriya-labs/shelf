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
#include <cstdlib>
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

static float bf16_to_f32(bf16_t x) {
    const uint32_t u = static_cast<uint32_t>(x.data) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

static bool alloc_bf16(
    ps::gpu::GpuArena& arena,
    const std::vector<std::size_t>& shape,
    ps::gpu::Tensor& out) {
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

struct AttentionCase {
    uint32_t q_heads = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t context_rows = 0;
    uint32_t block_rows = 0;
    uint32_t context_start = 0;
    uint32_t block_start = 0;
    uint32_t window = 0;
    float scale = 1.0f;
    std::vector<bf16_t> q;
    std::vector<bf16_t> k_ctx;
    std::vector<bf16_t> v_ctx;
    std::vector<bf16_t> k_noise;
    std::vector<bf16_t> v_noise;
};

static void reference_attention(const AttentionCase& c, std::vector<bf16_t>& out) {
    const uint32_t group = c.q_heads / c.kv_heads;
    const uint32_t kv_features = c.kv_heads * c.head_dim;
    const uint32_t q_features = c.q_heads * c.head_dim;
    const uint32_t keys = c.context_rows + c.block_rows;
    out.assign(static_cast<std::size_t>(c.block_rows) * q_features, bf16_t{0});
    std::vector<float> scores(keys);
    for (uint32_t r = 0; r < c.block_rows; ++r) {
        const int64_t qp = static_cast<int64_t>(c.block_start) + r;
        for (uint32_t h = 0; h < c.q_heads; ++h) {
            const uint32_t kvh = h / group;
            float max_score = -1.0e30f;
            for (uint32_t k = 0; k < keys; ++k) {
                const bool is_ctx = k < c.context_rows;
                const uint32_t idx = is_ctx ? k : k - c.context_rows;
                const int64_t kp = is_ctx ? static_cast<int64_t>(c.context_start) + idx
                                          : static_cast<int64_t>(c.block_start) + idx;
                const int64_t distance = kp - qp;
                const int64_t abs_distance = distance < 0 ? -distance : distance;
                float score = -1.0e30f;
                if (abs_distance < static_cast<int64_t>(c.window)) {
                    const bf16_t* k_row = is_ctx
                                              ? c.k_ctx.data() + static_cast<std::size_t>(idx) * kv_features
                                              : c.k_noise.data() + static_cast<std::size_t>(idx) * kv_features;
                    float acc = 0.0f;
                    for (uint32_t d = 0; d < c.head_dim; ++d) {
                        acc += bf16_to_f32(
                                   c.q[static_cast<std::size_t>(r) * q_features +
                                       static_cast<std::size_t>(h) * c.head_dim + d]) *
                               bf16_to_f32(k_row[static_cast<std::size_t>(kvh) * c.head_dim + d]);
                    }
                    score = acc * c.scale;
                }
                scores[k] = score;
                max_score = score > max_score ? score : max_score;
            }
            float sum = 0.0f;
            for (uint32_t k = 0; k < keys; ++k) {
                const float e = std::exp(scores[k] - max_score);
                scores[k] = e;
                sum += e;
            }
            const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (uint32_t d = 0; d < c.head_dim; ++d) {
                float acc = 0.0f;
                for (uint32_t k = 0; k < keys; ++k) {
                    if (scores[k] == 0.0f) continue;
                    const bool is_ctx = k < c.context_rows;
                    const uint32_t idx = is_ctx ? k : k - c.context_rows;
                    const bf16_t* v_row = is_ctx
                                              ? c.v_ctx.data() + static_cast<std::size_t>(idx) * kv_features
                                              : c.v_noise.data() + static_cast<std::size_t>(idx) * kv_features;
                    acc += scores[k] *
                           bf16_to_f32(v_row[static_cast<std::size_t>(kvh) * c.head_dim + d]);
                }
                out[static_cast<std::size_t>(r) * q_features +
                    static_cast<std::size_t>(h) * c.head_dim + d] =
                    bf16_t{f32_to_bf16_rne(acc * inv_sum)};
            }
        }
    }
}

static bool run_kernel(
    ps::gpu::GpuArena& arena,
    hipStream_t stream,
    const AttentionCase& c,
    std::vector<bf16_t>& got) {
    const std::size_t kv_elems =
        static_cast<std::size_t>(c.context_rows) * c.kv_heads * c.head_dim;
    const std::size_t noise_elems =
        static_cast<std::size_t>(c.block_rows) * c.kv_heads * c.head_dim;
    const std::size_t q_elems = static_cast<std::size_t>(c.block_rows) * c.q_heads * c.head_dim;

    ps::gpu::Tensor dev_q;
    ps::gpu::Tensor dev_k_ctx;
    ps::gpu::Tensor dev_v_ctx;
    ps::gpu::Tensor dev_k_noise;
    ps::gpu::Tensor dev_v_noise;
    ps::gpu::Tensor dev_out;
    if (!alloc_bf16(arena, {c.block_rows, c.q_heads * c.head_dim}, dev_q) ||
        !alloc_bf16(arena, {c.context_rows, c.kv_heads * c.head_dim}, dev_k_ctx) ||
        !alloc_bf16(arena, {c.context_rows, c.kv_heads * c.head_dim}, dev_v_ctx) ||
        !alloc_bf16(arena, {c.block_rows, c.kv_heads * c.head_dim}, dev_k_noise) ||
        !alloc_bf16(arena, {c.block_rows, c.kv_heads * c.head_dim}, dev_v_noise) ||
        !alloc_bf16(arena, {c.block_rows, c.q_heads * c.head_dim}, dev_out)) {
        check(false, "attention alloc");
        return false;
    }
    const auto upload = [&](ps::gpu::Tensor& dst, const std::vector<bf16_t>& src,
                            std::size_t elems, const char* tag) {
        return hipMemcpy(dst.data<bf16_t>(), src.data(), elems * sizeof(bf16_t),
                         hipMemcpyHostToDevice) == hipSuccess;
    };
    check(upload(dev_q, c.q, q_elems, "q"), "q upload");
    check(upload(dev_k_ctx, c.k_ctx, kv_elems, "k_ctx"), "k_ctx upload");
    check(upload(dev_v_ctx, c.v_ctx, kv_elems, "v_ctx"), "v_ctx upload");
    check(upload(dev_k_noise, c.k_noise, noise_elems, "k_noise"), "k_noise upload");
    check(upload(dev_v_noise, c.v_noise, noise_elems, "v_noise"), "v_noise upload");

    const hipError_t err = ps::kernel::launch_dflash2_attention_bf16(
        dev_q.data<bf16_t>(), c.q_heads * c.head_dim,
        dev_k_ctx.data<bf16_t>(), dev_v_ctx.data<bf16_t>(), c.context_rows,
        c.kv_heads * c.head_dim,
        dev_k_noise.data<bf16_t>(), dev_v_noise.data<bf16_t>(), c.block_rows,
        c.kv_heads * c.head_dim,
        dev_out.data<bf16_t>(), c.q_heads * c.head_dim,
        c.q_heads, c.kv_heads, c.head_dim, c.context_start, c.block_start, c.window, c.scale,
        stream);
    check(err == hipSuccess, "attention launch");
    if (err != hipSuccess) return false;
    check(hipStreamSynchronize(stream) == hipSuccess, "attention sync");
    got.resize(q_elems);
    check(hipMemcpy(got.data(), dev_out.data<bf16_t>(), got.size() * sizeof(bf16_t),
                    hipMemcpyDeviceToHost) == hipSuccess,
          "attention readback");
    return true;
}

static AttentionCase base_case(uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
                               uint32_t context_rows, uint32_t block_rows,
                               uint32_t context_start, uint32_t block_start, uint32_t window) {
    AttentionCase c;
    c.q_heads = q_heads;
    c.kv_heads = kv_heads;
    c.head_dim = head_dim;
    c.context_rows = context_rows;
    c.block_rows = block_rows;
    c.context_start = context_start;
    c.block_start = block_start;
    c.window = window;
    c.scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const std::size_t kv_elems =
        static_cast<std::size_t>(context_rows) * kv_heads * head_dim;
    const std::size_t noise_elems = static_cast<std::size_t>(block_rows) * kv_heads * head_dim;
    const std::size_t q_elems = static_cast<std::size_t>(block_rows) * q_heads * head_dim;
    c.q.assign(q_elems, bf16_t{f32_to_bf16_rne(1.0f)});
    c.k_ctx.assign(kv_elems, bf16_t{f32_to_bf16_rne(1.0f)});
    c.v_ctx.assign(kv_elems, bf16_t{0});
    c.k_noise.assign(noise_elems, bf16_t{f32_to_bf16_rne(1.0f)});
    c.v_noise.assign(noise_elems, bf16_t{0});
    return c;
}

static void run_marker_case(
    ps::gpu::GpuArena& arena,
    hipStream_t stream,
    const AttentionCase& c,
    bool marker_is_context,
    uint32_t marker_index,
    bool expect_visible,
    const std::string& tag) {
    AttentionCase m = c;
    const uint32_t kv_features = c.kv_heads * c.head_dim;
    std::vector<bf16_t>& target = marker_is_context ? m.v_ctx : m.v_noise;
    const std::size_t base = static_cast<std::size_t>(marker_index) * kv_features;
    for (uint32_t d = 0; d < kv_features; ++d) {
        target[base + d] = bf16_t{f32_to_bf16_rne(1.0f)};
    }
    std::vector<bf16_t> got;
    if (!run_kernel(arena, stream, m, got)) return;
    bool any_nonzero = false;
    for (const bf16_t v : got) {
        if (v.data != 0u) any_nonzero = true;
    }
    check(any_nonzero == expect_visible,
          tag + (expect_visible ? " must be visible" : " must be masked"));
}

static void run_random_case(ps::gpu::GpuArena& arena, hipStream_t stream, uint32_t seed) {
    AttentionCase c = base_case(4u, 2u, 8u, 8u, 4u, 100u, 108u, 16u);
    uint32_t s = seed;
    for (bf16_t& v : c.q) {
        const float u = static_cast<float>(lcg(s) >> 8) / static_cast<float>(1u << 24);
        v = bf16_t{f32_to_bf16_rne((u * 2.0f - 1.0f) * 2.0f)};
    }
    for (bf16_t& v : c.k_ctx) {
        const float u = static_cast<float>(lcg(s) >> 8) / static_cast<float>(1u << 24);
        v = bf16_t{f32_to_bf16_rne((u * 2.0f - 1.0f) * 2.0f)};
    }
    for (bf16_t& v : c.k_noise) {
        const float u = static_cast<float>(lcg(s) >> 8) / static_cast<float>(1u << 24);
        v = bf16_t{f32_to_bf16_rne((u * 2.0f - 1.0f) * 2.0f)};
    }
    for (bf16_t& v : c.v_ctx) {
        const float u = static_cast<float>(lcg(s) >> 8) / static_cast<float>(1u << 24);
        v = bf16_t{f32_to_bf16_rne((u * 2.0f - 1.0f) * 2.0f)};
    }
    for (bf16_t& v : c.v_noise) {
        const float u = static_cast<float>(lcg(s) >> 8) / static_cast<float>(1u << 24);
        v = bf16_t{f32_to_bf16_rne((u * 2.0f - 1.0f) * 2.0f)};
    }

    std::vector<bf16_t> got;
    if (!run_kernel(arena, stream, c, got)) return;
    std::vector<bf16_t> want;
    reference_attention(c, want);
    check(got.size() == want.size(), "random size");
    if (got.size() != want.size()) return;
    double diff = 0.0;
    double ref_norm = 0.0;
    double dot = 0.0;
    double got_norm = 0.0;
    double max_abs = 0.0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        const double a = bf16_to_f32(want[i]);
        const double b = bf16_to_f32(got[i]);
        check(std::isfinite(a) && std::isfinite(b), "random finite");
        const double d = a - b;
        max_abs = std::fabs(d) > max_abs ? std::fabs(d) : max_abs;
        diff += d * d;
        ref_norm += a * a;
        dot += a * b;
        got_norm += b * b;
    }
    const double rel_l2 = ref_norm > 0.0 ? std::sqrt(diff / ref_norm) : 0.0;
    const double cosine = (ref_norm > 0.0 && got_norm > 0.0)
                              ? dot / (std::sqrt(ref_norm) * std::sqrt(got_norm))
                              : 1.0;
    std::printf("  %-28s rel_l2=%.3e cosine=%.6f max_abs=%.3e\n", "random ctx8 blk4", rel_l2,
                cosine, max_abs);
    check(rel_l2 <= 2.0e-3, "random rel_l2 " + std::to_string(rel_l2));
    check(cosine >= 0.9999, "random cosine " + std::to_string(cosine));
}

static void run_gqa_check(ps::gpu::GpuArena& arena, hipStream_t stream) {
    AttentionCase c = base_case(4u, 2u, 8u, 2u, 2u, 0u, 2u, 16u);
    const uint32_t kv_features = c.kv_heads * c.head_dim;
    for (uint32_t row = 0; row < c.context_rows; ++row) {
        for (uint32_t d = 0; d < c.head_dim; ++d) {
            c.v_ctx[static_cast<std::size_t>(row) * kv_features + d] =
                bf16_t{f32_to_bf16_rne(1.0f)};
            c.v_ctx[static_cast<std::size_t>(row) * kv_features + c.head_dim + d] =
                bf16_t{f32_to_bf16_rne(2.0f)};
        }
    }
    for (uint32_t row = 0; row < c.block_rows; ++row) {
        for (uint32_t d = 0; d < c.head_dim; ++d) {
            c.v_noise[static_cast<std::size_t>(row) * kv_features + d] =
                bf16_t{f32_to_bf16_rne(1.0f)};
            c.v_noise[static_cast<std::size_t>(row) * kv_features + c.head_dim + d] =
                bf16_t{f32_to_bf16_rne(2.0f)};
        }
    }
    std::vector<bf16_t> got;
    if (!run_kernel(arena, stream, c, got)) return;
    const uint32_t q_features = c.q_heads * c.head_dim;
    for (uint32_t r = 0; r < c.block_rows; ++r) {
        for (uint32_t h = 0; h < c.q_heads; ++h) {
            const float want = (h / 2u == 0u) ? 1.0f : 2.0f;
            for (uint32_t d = 0; d < c.head_dim; ++d) {
                const float v = bf16_to_f32(
                    got[static_cast<std::size_t>(r) * q_features +
                        static_cast<std::size_t>(h) * c.head_dim + d]);
                check(std::fabs(v - want) < 1.0e-3f,
                      "GQA head " + std::to_string(h) + " must map to kv " +
                          std::to_string(h / 2u));
            }
        }
    }
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
        AttentionCase c = base_case(2u, 1u, 4u, 4u, 4u, 0u, 4u, 8u);
        run_marker_case(arena, stream, c, false, 3u, true, "future noise row +3 (window 8)");
        run_marker_case(arena, stream, c, false, 0u, true, "own noise row 0 (window 8)");
        run_marker_case(arena, stream, c, true, 0u, true, "context row -4 (window 8)");
    }
    {
        AttentionCase c = base_case(2u, 1u, 4u, 4u, 4u, 0u, 4u, 4u);
        run_marker_case(arena, stream, c, false, 3u, true, "distance 3 (window 4)");
        run_marker_case(arena, stream, c, true, 0u, false, "distance 4 (window 4) masked");
        run_marker_case(arena, stream, c, false, 0u, true, "distance 0 (window 4)");
    }
    {
        AttentionCase c = base_case(2u, 1u, 4u, 2u, 2u, 0u, 2u, 2u);
        run_marker_case(arena, stream, c, true, 1u, true, "context idx 1 at distance 1");
        run_marker_case(arena, stream, c, true, 0u, false, "context idx 0 at distance 2 masked");
        run_marker_case(arena, stream, c, false, 0u, true, "noise idx 0 at distance 0");
        run_marker_case(arena, stream, c, false, 1u, true, "noise idx 1 at distance 1");
    }
    run_gqa_check(arena, stream);
    run_random_case(arena, stream, 1234u);
    run_random_case(arena, stream, 4321u);

    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    const ps::Status shutdown = arena.shutdown();
    check(shutdown.ok(), "arena shutdown: " + shutdown.message());

    std::printf("test_dflash2_attention: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
