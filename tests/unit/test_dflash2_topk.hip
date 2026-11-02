#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/kernels/dflash2/topk.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

using ps::bf16_t;

static bool alloc_f32(
    ps::gpu::GpuArena& arena, const std::vector<std::size_t>& shape, ps::gpu::Tensor& out) {
    std::size_t count = 1;
    for (std::size_t d : shape) count *= d;
    auto view = arena.allocate_aligned(count * sizeof(float), 128);
    if (!view.ok()) return false;
    auto backing = view.release();
    std::vector<std::size_t> strides(shape.size());
    std::size_t stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }
    auto t = ps::gpu::Tensor::create<float>(backing, shape, strides);
    if (!t.ok()) return false;
    out = t.release();
    return true;
}

static bool alloc_i32(
    ps::gpu::GpuArena& arena, const std::vector<std::size_t>& shape, ps::gpu::Tensor& out) {
    std::size_t count = 1;
    for (std::size_t d : shape) count *= d;
    auto view = arena.allocate_aligned(count * sizeof(int32_t), 128);
    if (!view.ok()) return false;
    auto backing = view.release();
    std::vector<std::size_t> strides(shape.size());
    std::size_t stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }
    auto t = ps::gpu::Tensor::create<int32_t>(backing, shape, strides);
    if (!t.ok()) return false;
    out = t.release();
    return true;
}

struct Reference {
    std::vector<int32_t> ids;
    std::vector<float> logits;
};

static Reference reference_topk(
    const std::vector<float>& logits, uint32_t rows, uint32_t vocab, uint32_t k) {
    Reference out;
    out.ids.resize(static_cast<std::size_t>(rows) * k);
    out.logits.resize(static_cast<std::size_t>(rows) * k);
    std::vector<std::pair<float, int32_t>> items(vocab);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t v = 0; v < vocab; ++v) {
            items[v] = {logits[static_cast<std::size_t>(r) * vocab + v],
                        static_cast<int32_t>(v)};
        }
        std::stable_sort(items.begin(), items.end(),
                         [](const std::pair<float, int32_t>& a,
                            const std::pair<float, int32_t>& b) {
                             if (a.first != b.first) return a.first > b.first;
                             return a.second < b.second;
                         });
        for (uint32_t j = 0; j < k; ++j) {
            out.logits[static_cast<std::size_t>(r) * k + j] = items[j].first;
            out.ids[static_cast<std::size_t>(r) * k + j] = items[j].second;
        }
    }
    return out;
}

static void run_case(
    ps::gpu::GpuArena& arena,
    hipStream_t stream,
    const std::string& name,
    const std::vector<float>& host_logits,
    uint32_t rows,
    uint32_t vocab,
    uint32_t k) {
    ps::gpu::Tensor logits;
    ps::gpu::Tensor out_ids;
    ps::gpu::Tensor out_logits;
    ps::gpu::Tensor scratch_ids;
    ps::gpu::Tensor scratch_logits;
    const bool alloc_ok =
        alloc_f32(arena, {rows, vocab}, logits) &&
        alloc_i32(arena, {rows, k}, out_ids) &&
        alloc_f32(arena, {rows, k}, out_logits) &&
        alloc_i32(arena, {rows, ps::kernel::kDflash2TopKScratchPerRow}, scratch_ids) &&
        alloc_f32(arena, {rows, ps::kernel::kDflash2TopKScratchPerRow}, scratch_logits);
    check(alloc_ok, name + " alloc");
    if (!alloc_ok) return;
    check(hipMemcpy(logits.data<float>(), host_logits.data(),
                    host_logits.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess,
          name + " upload");

    const hipError_t err = ps::kernel::launch_dflash2_topk_f32(
        logits.data<float>(), rows, vocab, vocab, k, out_ids.data<int32_t>(),
        out_logits.data<float>(), scratch_ids.data<int32_t>(),
        scratch_logits.data<float>(), stream);
    check(err == hipSuccess, name + " launch");
    if (err != hipSuccess) return;
    check(hipStreamSynchronize(stream) == hipSuccess, name + " sync");

    std::vector<int32_t> got_ids(static_cast<std::size_t>(rows) * k);
    std::vector<float> got_logits(static_cast<std::size_t>(rows) * k);
    check(hipMemcpy(got_ids.data(), out_ids.data<int32_t>(),
                    got_ids.size() * sizeof(int32_t), hipMemcpyDeviceToHost) == hipSuccess,
          name + " read ids");
    check(hipMemcpy(got_logits.data(), out_logits.data<float>(),
                    got_logits.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess,
          name + " read logits");

    const Reference ref = reference_topk(host_logits, rows, vocab, k);
    std::size_t id_mismatch = 0;
    std::size_t logit_mismatch = 0;
    for (std::size_t i = 0; i < ref.ids.size(); ++i) {
        if (got_ids[i] != ref.ids[i]) ++id_mismatch;
        if (got_logits[i] != ref.logits[i]) ++logit_mismatch;
    }
    check(id_mismatch == 0, name + " ids mismatch " + std::to_string(id_mismatch));
    check(logit_mismatch == 0,
          name + " logits mismatch " + std::to_string(logit_mismatch));
    std::printf("  %-28s rows=%u vocab=%u k=%u mismatches=%zu/%zu\n", name.c_str(), rows,
                vocab, k, id_mismatch, logit_mismatch);
}

int main() {
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    auto arena_result = ps::gpu::GpuArena::create(0, 1ull * 1024ull * 1024ull * 1024ull);
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

    const uint32_t k = 16u;

    {
        const uint32_t rows = 3u;
        const uint32_t vocab = 5000u;
        std::vector<float> logits(static_cast<std::size_t>(rows) * vocab);
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t v = 0; v < vocab; ++v) {
                logits[static_cast<std::size_t>(r) * vocab + v] =
                    static_cast<float>(vocab) - static_cast<float>(v) -
                    static_cast<float>(r) * 0.25f;
            }
        }
        run_case(arena, stream, "strictly_descending", logits, rows, vocab, k);
    }

    {
        const uint32_t rows = 2u;
        const uint32_t vocab = 4096u;
        std::vector<float> logits(static_cast<std::size_t>(rows) * vocab, 1.5f);
        run_case(arena, stream, "all_equal", logits, rows, vocab, k);
    }

    {
        const uint32_t rows = 2u;
        const uint32_t vocab = 16u;
        std::vector<float> logits(static_cast<std::size_t>(rows) * vocab, 0.0f);
        for (std::size_t i = 0; i < logits.size(); ++i) {
            logits[i] = static_cast<float>(i);
        }
        run_case(arena, stream, "k_equals_vocab", logits, rows, vocab, k);
    }

    {
        const uint32_t rows = 4u;
        const uint32_t vocab = 257u;
        std::vector<float> logits(static_cast<std::size_t>(rows) * vocab);
        for (std::size_t i = 0; i < logits.size(); ++i) {
            logits[i] = static_cast<float>((i * 7919u) % 251u);
        }
        run_case(arena, stream, "vocab_lt_partition", logits, rows, vocab, k);
    }

    {
        const uint32_t rows = 2u;
        const uint32_t vocab = ps::kernel::kDflash2TopKThreads * 16u * 4u + 33u;
        std::vector<float> logits(static_cast<std::size_t>(rows) * vocab);
        for (std::size_t i = 0; i < logits.size(); ++i) {
            logits[i] = static_cast<float>((i * 2654435761u) % 4093u);
        }
        run_case(arena, stream, "cross_partition", logits, rows, vocab, k);
    }

    {
        const uint32_t rows = 3u;
        const uint32_t vocab = 2049u;
        std::vector<float> logits(static_cast<std::size_t>(rows) * vocab);
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t v = 0; v < vocab; ++v) {
                logits[static_cast<std::size_t>(r) * vocab + v] =
                    (v % 37u == 0u) ? 2.0f
                                    : static_cast<float>((v * 31u + r) % 11u) * 0.0f +
                                          static_cast<float>((v % 13u) - 6) * 1.0f;
            }
        }
        run_case(arena, stream, "ties_mixed", logits, rows, vocab, k);
    }

    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    const ps::Status shutdown = arena.shutdown();
    check(shutdown.ok(), "arena shutdown");
    std::printf("test_dflash2_topk: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
