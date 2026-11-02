#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/kernels/dflash2/candidate_selector.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

using ps::bf16_t;

static float f32_from_bf16(bf16_t x) {
    const uint32_t u = static_cast<uint32_t>(x.data) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

static bf16_t bf16_from_f32(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    u += 0x7FFFu + ((u >> 16u) & 1u);
    bf16_t out;
    out.data = static_cast<uint16_t>(u >> 16);
    return out;
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

struct SelectorFixture {
    uint32_t draft_rows = 0;
    uint32_t rank = 0;
    uint32_t top_k = 0;
    uint32_t vocab = 0;
    int32_t anchor = 0;
    bool dependency_sensitive = false;
    std::vector<int32_t> ids;
    std::vector<float> logits;
    std::vector<bf16_t> hidden;
    std::vector<bf16_t> predecessor;
    std::vector<bf16_t> successor;
};

static std::vector<int32_t> reference_selector(
    const SelectorFixture& fx, bool use_predecessor_updates) {
    std::vector<int32_t> out(fx.draft_rows, 0);
    int32_t pred = fx.anchor;
    for (uint32_t r = 0; r < fx.draft_rows; ++r) {
        int32_t best_token = 0;
        float best_score = -std::numeric_limits<float>::infinity();
        bool have = false;
        for (uint32_t j = 0; j < fx.top_k; ++j) {
            const int32_t token = fx.ids[static_cast<std::size_t>(r) * fx.top_k + j];
            double acc = 0.0;
            for (uint32_t d = 0; d < fx.rank; ++d) {
                acc += static_cast<double>(
                           f32_from_bf16(fx.predecessor[static_cast<std::size_t>(pred) *
                                                             fx.rank + d])) *
                       static_cast<double>(
                           f32_from_bf16(fx.hidden[static_cast<std::size_t>(r) * fx.rank + d])) *
                       static_cast<double>(
                           f32_from_bf16(fx.successor[static_cast<std::size_t>(token) *
                                                          fx.rank + d]));
            }
            const float score =
                fx.logits[static_cast<std::size_t>(r) * fx.top_k + j] +
                static_cast<float>(acc);
            const bool better = !have || score > best_score ||
                                (score == best_score && token < best_token);
            if (better) {
                best_score = score;
                best_token = token;
                have = true;
            }
        }
        out[r] = best_token;
        if (use_predecessor_updates) pred = best_token;
    }
    return out;
}

static void run_case(
    ps::gpu::GpuArena& arena,
    hipStream_t stream,
    const std::string& name,
    const SelectorFixture& fx) {
    ps::gpu::Tensor hidden;
    ps::gpu::Tensor ids;
    ps::gpu::Tensor logits;
    ps::gpu::Tensor predecessor;
    ps::gpu::Tensor successor;
    ps::gpu::Tensor out;
    const bool alloc_ok =
        alloc_bf16(arena, {fx.draft_rows, fx.rank}, hidden) &&
        alloc_i32(arena, {fx.draft_rows, fx.top_k}, ids) &&
        alloc_f32(arena, {fx.draft_rows, fx.top_k}, logits) &&
        alloc_bf16(arena, {fx.vocab, fx.rank}, predecessor) &&
        alloc_bf16(arena, {fx.vocab, fx.rank}, successor) &&
        alloc_i32(arena, {fx.draft_rows}, out);
    check(alloc_ok, name + " alloc");
    if (!alloc_ok) return;
    const auto upload = [&](ps::gpu::Tensor& dst, const auto& src) {
        return hipMemcpy(dst.template data<typename std::decay_t<decltype(src)>::value_type>(),
                         src.data(), src.size() * sizeof(typename std::decay_t<decltype(src)>::value_type),
                         hipMemcpyHostToDevice) == hipSuccess;
    };
    check(upload(hidden, fx.hidden) && upload(ids, fx.ids) && upload(logits, fx.logits) &&
              upload(predecessor, fx.predecessor) && upload(successor, fx.successor),
          name + " upload");

    ps::kernel::DFlash2CandidateSelectorArgs args;
    args.hidden_proj = hidden.data<bf16_t>();
    args.topk_ids = ids.data<int32_t>();
    args.topk_logits = logits.data<float>();
    args.predecessor_codebook = predecessor.data<bf16_t>();
    args.successor_codebook = successor.data<bf16_t>();
    args.anchor_token = fx.anchor;
    args.output_tokens = out.data<int32_t>();
    args.draft_rows = fx.draft_rows;
    args.rank = fx.rank;
    args.top_k = fx.top_k;
    args.vocab_size = fx.vocab;

    const hipError_t err = ps::kernel::launch_dflash2_candidate_selector(args, stream);
    check(err == hipSuccess, name + " launch");
    if (err != hipSuccess) return;
    check(hipStreamSynchronize(stream) == hipSuccess, name + " sync");

    std::vector<int32_t> got(fx.draft_rows, 0);
    check(hipMemcpy(got.data(), out.data<int32_t>(), got.size() * sizeof(int32_t),
                    hipMemcpyDeviceToHost) == hipSuccess,
          name + " read");

    const std::vector<int32_t> ref = reference_selector(fx, true);
    check(got == ref, name + " token mismatch");
    std::printf("  %-28s got:", name.c_str());
    for (int32_t t : got) std::printf(" %d", t);
    std::printf("  ref:");
    for (int32_t t : ref) std::printf(" %d", t);
    std::printf("\n");

    if (fx.dependency_sensitive) {
        const std::vector<int32_t> independent = reference_selector(fx, false);
        check(independent != ref,
              name + " fixture does not exercise predecessor dependency");
    }
}

int main() {
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
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
        SelectorFixture fx;
        fx.draft_rows = 3u;
        fx.rank = 8u;
        fx.top_k = 4u;
        fx.vocab = 64u;
        fx.anchor = 7;
        fx.ids.resize(static_cast<std::size_t>(fx.draft_rows) * fx.top_k);
        fx.logits.resize(static_cast<std::size_t>(fx.draft_rows) * fx.top_k);
        fx.hidden.resize(static_cast<std::size_t>(fx.draft_rows) * fx.rank);
        fx.predecessor.resize(static_cast<std::size_t>(fx.vocab) * fx.rank);
        fx.successor.resize(static_cast<std::size_t>(fx.vocab) * fx.rank);
        uint32_t state = 12345u;
        const auto next = [&state]() {
            state = state * 1664525u + 1013904223u;
            return state >> 8;
        };
        for (auto& v : fx.ids) v = static_cast<int32_t>(next() % fx.vocab);
        for (auto& v : fx.logits) v = static_cast<float>(next() % 1000u) * 0.01f;
        for (auto& v : fx.hidden) v = bf16_from_f32(static_cast<float>(next() % 400u) * 0.01f - 2.0f);
        for (auto& v : fx.predecessor) v = bf16_from_f32(static_cast<float>(next() % 400u) * 0.01f - 2.0f);
        for (auto& v : fx.successor) v = bf16_from_f32(static_cast<float>(next() % 400u) * 0.01f - 2.0f);
        run_case(arena, stream, "tiny_rank8_topk4", fx);
    }

    {
        SelectorFixture fx;
        fx.draft_rows = 2u;
        fx.rank = 8u;
        fx.top_k = 4u;
        fx.vocab = 32u;
        fx.anchor = 1;
        fx.dependency_sensitive = true;
        fx.ids = {3, 5, 9, 11, 2, 4, 6, 7};
        fx.logits = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        fx.hidden.assign(static_cast<std::size_t>(fx.draft_rows) * fx.rank, bf16_from_f32(1.0f));
        fx.predecessor.assign(static_cast<std::size_t>(fx.vocab) * fx.rank, bf16_from_f32(0.0f));
        fx.successor.assign(static_cast<std::size_t>(fx.vocab) * fx.rank, bf16_from_f32(0.0f));
        const auto set_pred = [&](int32_t token, std::size_t lane, float value) {
            fx.predecessor[static_cast<std::size_t>(token) * fx.rank + lane] =
                bf16_from_f32(value);
        };
        const auto set_succ = [&](int32_t token, std::size_t lane, float value) {
            fx.successor[static_cast<std::size_t>(token) * fx.rank + lane] =
                bf16_from_f32(value);
        };
        set_pred(1, 0, 1.0f);
        set_pred(3, 1, 1.0f);
        set_pred(5, 1, -1.0f);
        set_succ(3, 0, 1.0f);
        set_succ(5, 1, 1.0f);
        set_succ(2, 0, 10.0f);
        set_succ(4, 1, 10.0f);
        run_case(arena, stream, "predecessor_dependency", fx);
    }

    {
        SelectorFixture fx;
        fx.draft_rows = 7u;
        fx.rank = 256u;
        fx.top_k = 16u;
        fx.vocab = 256u;
        fx.anchor = 0;
        fx.ids.resize(static_cast<std::size_t>(fx.draft_rows) * fx.top_k);
        fx.logits.resize(static_cast<std::size_t>(fx.draft_rows) * fx.top_k);
        fx.hidden.resize(static_cast<std::size_t>(fx.draft_rows) * fx.rank);
        fx.predecessor.resize(static_cast<std::size_t>(fx.vocab) * fx.rank);
        fx.successor.resize(static_cast<std::size_t>(fx.vocab) * fx.rank);
        uint32_t state = 999u;
        const auto next = [&state]() {
            state = state * 1103515245u + 12345u;
            return state >> 9;
        };
        for (auto& v : fx.ids) v = static_cast<int32_t>(next() % fx.vocab);
        for (auto& v : fx.logits) v = static_cast<float>(next() % 500u) * 0.001f;
        for (auto& v : fx.hidden) v = bf16_from_f32(static_cast<float>((next() % 200u)) * 0.01f - 1.0f);
        for (auto& v : fx.predecessor) v = bf16_from_f32(static_cast<float>((next() % 200u)) * 0.01f - 1.0f);
        for (auto& v : fx.successor) v = bf16_from_f32(static_cast<float>((next() % 200u)) * 0.01f - 1.0f);
        run_case(arena, stream, "full_rank256_topk16", fx);
    }

    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    const ps::Status shutdown = arena.shutdown();
    check(shutdown.ok(), "arena shutdown");
    std::printf("test_dflash2_candidate_selector: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
