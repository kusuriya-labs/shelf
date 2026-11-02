#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/tensor.h>
#include <shelf/core/memory/types.h>
#include <shelf/models/qwen35/kernels/eagle4/attention.h>
#include <shelf/models/qwen35/kernels/eagle4/candidate_ops.h>
#include <shelf/models/qwen35/kernels/eagle4/ops.h>
#include <shelf/kernels/optimized/radix_topn.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using ps::bf16_t;
using ps::f32_to_bf16_rne;

static float bf16_to_f32(bf16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value.data) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

static bool alloc_bytes(ps::gpu::GpuArena& arena, std::size_t bytes, ps::gpu::Tensor& out) {
    auto view = arena.allocate_aligned(bytes, 256);
    if (!view.ok()) return false;
    auto backing = view.release();
    auto t = ps::gpu::Tensor::create<uint8_t>(backing, {bytes}, {1});
    if (!t.ok()) return false;
    out = t.release();
    return true;
}

static bool alloc_bf16(
    ps::gpu::GpuArena& arena, const std::vector<std::size_t>& shape, ps::gpu::Tensor& out) {
    std::size_t count = 1;
    for (std::size_t d : shape) count *= d;
    auto view = arena.allocate_aligned(count * sizeof(bf16_t), 256);
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
    auto view = arena.allocate_aligned(count * sizeof(float), 256);
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
    auto view = arena.allocate_aligned(count * sizeof(int32_t), 256);
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

static uint32_t lcg(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return s;
}

static bf16_t random_bf16(uint32_t& s) {
    const float u = static_cast<float>(lcg(s) >> 8) / static_cast<float>(1u << 24);
    return bf16_t{f32_to_bf16_rne((u * 2.0f - 1.0f) * 2.0f)};
}

template <typename T>
static bool upload(ps::gpu::Tensor& dst, const T* src, std::size_t bytes) {
    return hipMemcpy(dst.data<T>(), src, bytes, hipMemcpyHostToDevice) == hipSuccess;
}

template <typename T>
static bool download(const ps::gpu::Tensor& src, T* dst, std::size_t bytes) {
    return hipMemcpy(dst, src.data<T>(), bytes, hipMemcpyDeviceToHost) == hipSuccess;
}

static double rel_error(float got, float want) {
    const double scale = std::max(std::fabs(static_cast<double>(want)), 1.0e-3);
    return std::fabs(static_cast<double>(got) - static_cast<double>(want)) / scale;
}

static void cpu_rmsnorm(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    std::vector<float>& out,
    uint32_t rows,
    uint32_t features,
    float eps,
    uint32_t offset) {
    for (uint32_t r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < features; ++i) {
            const float v = input[static_cast<std::size_t>(r) * features + i];
            sum += v * v;
        }
        const float inv = 1.0f / std::sqrt(sum / static_cast<float>(features) + eps);
        for (uint32_t i = 0; i < features; ++i) {
            const float norm = input[static_cast<std::size_t>(r) * features + i] * inv;
            const float norm_round = bf16_to_f32(
                bf16_t{f32_to_bf16_rne(norm)});
            out[static_cast<std::size_t>(r) * features * 3u + offset + i] =
                bf16_to_f32(bf16_t{f32_to_bf16_rne(weight[i] * norm_round)});
        }
    }
}

static void test_tap_norm_concat(ps::gpu::GpuArena& arena, hipStream_t stream) {
    constexpr uint32_t kRows = 3;
    constexpr uint32_t kFeatures = 256;
    constexpr float kEps = 1.0e-6f;
    std::vector<bf16_t> tap0(static_cast<std::size_t>(kRows) * kFeatures);
    std::vector<bf16_t> tap1(tap0.size());
    std::vector<bf16_t> tap2(tap0.size());
    std::vector<bf16_t> weight0(kFeatures), weight1(kFeatures), weight2(kFeatures);
    uint32_t seed = 7u;
    for (std::size_t i = 0; i < tap0.size(); ++i) {
        tap0[i] = random_bf16(seed);
        tap1[i] = random_bf16(seed);
        tap2[i] = random_bf16(seed);
    }
    for (std::size_t i = 0; i < kFeatures; ++i) {
        weight0[i] = bf16_t{f32_to_bf16_rne(1.0f + 0.25f * static_cast<float>(i % 3u))};
        weight1[i] = bf16_t{f32_to_bf16_rne(0.5f + 0.1f * static_cast<float>(i % 5u))};
        weight2[i] = bf16_t{f32_to_bf16_rne(1.25f - 0.05f * static_cast<float>(i % 4u))};
    }

    ps::gpu::Tensor dtap0, dtap1, dtap2, dw0, dw1, dw2, dout;
    check(alloc_bf16(arena, {kRows, kFeatures}, dtap0), "tap0 alloc");
    check(alloc_bf16(arena, {kRows, kFeatures}, dtap1), "tap1 alloc");
    check(alloc_bf16(arena, {kRows, kFeatures}, dtap2), "tap2 alloc");
    check(alloc_bf16(arena, {kFeatures}, dw0), "w0 alloc");
    check(alloc_bf16(arena, {kFeatures}, dw1), "w1 alloc");
    check(alloc_bf16(arena, {kFeatures}, dw2), "w2 alloc");
    check(alloc_bf16(arena, {kRows, kFeatures * 3u}, dout), "out alloc");
    check(upload(dtap0, tap0.data(), tap0.size() * sizeof(bf16_t)), "tap0 upload");
    check(upload(dtap1, tap1.data(), tap1.size() * sizeof(bf16_t)), "tap1 upload");
    check(upload(dtap2, tap2.data(), tap2.size() * sizeof(bf16_t)), "tap2 upload");
    check(upload(dw0, weight0.data(), weight0.size() * sizeof(bf16_t)), "w0 upload");
    check(upload(dw1, weight1.data(), weight1.size() * sizeof(bf16_t)), "w1 upload");
    check(upload(dw2, weight2.data(), weight2.size() * sizeof(bf16_t)), "w2 upload");

    const hipError_t err = ps::kernel::launch_eagle4_tap_norm_concat_bf16(
        dtap0.data<bf16_t>(), dtap1.data<bf16_t>(), dtap2.data<bf16_t>(),
        dw0.data<bf16_t>(), dw1.data<bf16_t>(), dw2.data<bf16_t>(), kFeatures, kRows,
        dout.data<bf16_t>(), kEps, stream);
    check(err == hipSuccess, "tap norm concat launch");
    hipStreamSynchronize(stream);

    std::vector<bf16_t> host_out(static_cast<std::size_t>(kRows) * kFeatures * 3u);
    check(download(dout, host_out.data(), host_out.size() * sizeof(bf16_t)), "out download");

    std::vector<float> f(tap0.size());
    for (std::size_t i = 0; i < tap0.size(); ++i) f[i] = bf16_to_f32(tap0[i]);
    std::vector<float> w(kFeatures);
    for (std::size_t i = 0; i < kFeatures; ++i) w[i] = bf16_to_f32(weight0[i]);
    std::vector<float> want(static_cast<std::size_t>(kRows) * kFeatures * 3u);
    cpu_rmsnorm(f, w, want, kRows, kFeatures, kEps, 0u);
    for (std::size_t i = 0; i < tap1.size(); ++i) f[i] = bf16_to_f32(tap1[i]);
    for (std::size_t i = 0; i < kFeatures; ++i) w[i] = bf16_to_f32(weight1[i]);
    cpu_rmsnorm(f, w, want, kRows, kFeatures, kEps, kFeatures);
    for (std::size_t i = 0; i < tap2.size(); ++i) f[i] = bf16_to_f32(tap2[i]);
    for (std::size_t i = 0; i < kFeatures; ++i) w[i] = bf16_to_f32(weight2[i]);
    cpu_rmsnorm(f, w, want, kRows, kFeatures, kEps, 2u * kFeatures);

    double worst = 0.0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        worst = std::max(worst, rel_error(bf16_to_f32(host_out[i]), want[i]));
    }
    check(worst < 2.0e-2, "tap norm concat matches the CPU reference (worst=" +
                              std::to_string(worst) + ")");
}

static void test_ring_attention(ps::gpu::GpuArena& arena, hipStream_t stream) {
    constexpr uint32_t kHeads = 2u;
    constexpr uint32_t kHeadDim = 128u;
    constexpr uint32_t kCapacity = 32u;
    constexpr uint32_t kWindow = 16u;
    constexpr uint32_t kRows = 4u;
    constexpr uint32_t kPositionStart = 20u;
    const uint32_t kv_features = kHeads * kHeadDim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

    std::vector<bf16_t> host_q(static_cast<std::size_t>(kRows) * kv_features);
    std::vector<bf16_t> host_k(static_cast<std::size_t>(kCapacity) * kv_features);
    std::vector<bf16_t> host_v(host_k.size());
    uint32_t seed = 11u;
    for (bf16_t& v : host_q) v = random_bf16(seed);
    for (bf16_t& v : host_k) v = random_bf16(seed);
    for (bf16_t& v : host_v) v = random_bf16(seed);

    for (uint32_t position = 0; position < kPositionStart + kRows; ++position) {
        const uint32_t slot = position % kCapacity;
        for (uint32_t f = 0; f < kv_features; ++f) {
            const std::size_t index = static_cast<std::size_t>(slot) * kv_features + f;
            host_k[index] = random_bf16(seed);
            host_v[index] = random_bf16(seed);
        }
    }

    ps::gpu::Tensor dq, dk, dv, dout;
    check(alloc_bf16(arena, {kRows, kv_features}, dq), "attn q alloc");
    check(alloc_bf16(arena, {kCapacity, kv_features}, dk), "attn k alloc");
    check(alloc_bf16(arena, {kCapacity, kv_features}, dv), "attn v alloc");
    check(alloc_bf16(arena, {kRows, kv_features}, dout), "attn out alloc");
    check(upload(dq, host_q.data(), host_q.size() * sizeof(bf16_t)), "attn q upload");
    check(upload(dk, host_k.data(), host_k.size() * sizeof(bf16_t)), "attn k upload");
    check(upload(dv, host_v.data(), host_v.size() * sizeof(bf16_t)), "attn v upload");

    const hipError_t err = ps::kernel::launch_eagle4_ring_attention_bf16(
        dq.data<bf16_t>(), kv_features, dk.data<bf16_t>(), dv.data<bf16_t>(), kCapacity,
        kv_features, kPositionStart, kRows, kHeads, kHeadDim, kWindow, scale,
        dout.data<bf16_t>(), kv_features, stream);
    check(err == hipSuccess, "ring attention launch");
    hipStreamSynchronize(stream);
    std::vector<bf16_t> host_out(static_cast<std::size_t>(kRows) * kv_features);
    check(download(dout, host_out.data(), host_out.size() * sizeof(bf16_t)), "attn download");

    double worst = 0.0;
    for (uint32_t row = 0; row < kRows; ++row) {
        const uint32_t qpos = kPositionStart + row;
        const uint32_t lo = qpos + 1u > kWindow ? qpos + 1u - kWindow : 0u;
        for (uint32_t head = 0; head < kHeads; ++head) {
            std::vector<double> scores(qpos - lo + 1u);
            double max_score = -1.0e30;
            for (uint32_t k = lo; k <= qpos; ++k) {
                const uint32_t slot = k % kCapacity;
                double acc = 0.0;
                for (uint32_t d = 0; d < kHeadDim; ++d) {
                    acc += bf16_to_f32(
                               host_q[static_cast<std::size_t>(row) * kv_features +
                                      head * kHeadDim + d]) *
                           bf16_to_f32(
                               host_k[static_cast<std::size_t>(slot) * kv_features +
                                      head * kHeadDim + d]);
                }
                scores[k - lo] = acc * scale;
                max_score = std::max(max_score, scores[k - lo]);
            }
            double sum = 0.0;
            for (double& s : scores) {
                s = std::exp(s - max_score);
                sum += s;
            }
            for (uint32_t d = 0; d < kHeadDim; ++d) {
                double acc = 0.0;
                for (uint32_t k = lo; k <= qpos; ++k) {
                    const uint32_t slot = k % kCapacity;
                    acc += scores[k - lo] / sum *
                           bf16_to_f32(
                               host_v[static_cast<std::size_t>(slot) * kv_features +
                                      head * kHeadDim + d]);
                }
                const float got = bf16_to_f32(
                    host_out[static_cast<std::size_t>(row) * kv_features + head * kHeadDim + d]);
                worst = std::max(worst, rel_error(got, static_cast<float>(acc)));
            }
        }
    }
    check(worst < 3.0e-2, "ring attention matches the CPU reference (worst=" +
                              std::to_string(worst) + ")");
}

static void test_candidate_ops(ps::gpu::GpuArena& arena, hipStream_t stream) {
    constexpr uint32_t kRows = 2u;
    constexpr uint32_t kHidden = 256u;
    constexpr uint32_t kRank = 16u;
    constexpr uint32_t kVocab = 512u;
    constexpr uint32_t kCandidates = 32u;
    constexpr uint32_t kTopK = 3u;

    std::vector<bf16_t> host_hidden(static_cast<std::size_t>(kRows) * kHidden);
    std::vector<bf16_t> host_down(static_cast<std::size_t>(kRank) * kHidden);
    std::vector<bf16_t> host_head(static_cast<std::size_t>(kVocab) * kRank);
    std::vector<bf16_t> host_lm(static_cast<std::size_t>(kVocab) * kHidden);
    uint32_t seed = 23u;
    for (bf16_t& v : host_hidden) v = random_bf16(seed);
    for (bf16_t& v : host_down) v = random_bf16(seed);
    for (bf16_t& v : host_head) v = random_bf16(seed);
    for (bf16_t& v : host_lm) v = random_bf16(seed);

    ps::gpu::Tensor dhidden, ddown, dhead, dlm, dproj, drough, dcand, dlogits, dtop_ids,
        dtop_logits, dtop_probs, dradix, dradix_logits, dargmax, dgemv;
    check(alloc_bf16(arena, {kRows, kHidden}, dhidden), "hidden alloc");
    check(alloc_bf16(arena, {kRank, kHidden}, ddown), "down alloc");
    check(alloc_bf16(arena, {kVocab, kRank}, dhead), "head alloc");
    check(alloc_bf16(arena, {kVocab, kHidden}, dlm), "lm alloc");
    check(alloc_bf16(arena, {kRows, kRank}, dproj), "proj alloc");
    check(alloc_f32(arena, {kRows, kVocab}, drough), "rough alloc");
    check(alloc_i32(arena, {kRows, kCandidates}, dcand), "cand alloc");
    check(alloc_f32(arena, {kRows, kCandidates}, dlogits), "logits alloc");
    check(alloc_i32(arena, {kRows, kTopK}, dtop_ids), "top ids alloc");
    check(alloc_f32(arena, {kRows, kTopK}, dtop_logits), "top logits alloc");
    check(alloc_f32(arena, {kRows, kTopK}, dtop_probs), "top probs alloc");
    check(alloc_bf16(arena, {kRows, kVocab}, dgemv), "gemv alloc");
    check(alloc_f32(arena, {kRows, kCandidates}, dradix_logits), "radix logits alloc");
    check(alloc_i32(arena, {kRows, 1}, dargmax), "argmax alloc");
    const std::size_t partitions = ps::kernel::dflash2_radix_default_partitions(kVocab);
    std::size_t scratch_bytes =
        ps::kernel::dflash2_radix_topn_scratch_bytes(kRows, partitions, kVocab);
    check(alloc_bytes(arena, scratch_bytes, dradix), "radix alloc");

    check(upload(dhidden, host_hidden.data(), host_hidden.size() * sizeof(bf16_t)), "hidden up");
    check(upload(ddown, host_down.data(), host_down.size() * sizeof(bf16_t)), "down up");
    check(upload(dhead, host_head.data(), host_head.size() * sizeof(bf16_t)), "head up");
    check(upload(dlm, host_lm.data(), host_lm.size() * sizeof(bf16_t)), "lm up");

    check(ps::kernel::launch_eagle4_rank_project_bf16(
              dhidden.data<bf16_t>(), kHidden, ddown.data<bf16_t>(),
              dproj.data<bf16_t>(), kRows, kRank, kHidden, stream) == hipSuccess,
          "rank project launch");
    check(ps::kernel::launch_eagle4_rough_scores_bf16(
              dproj.data<bf16_t>(), kRank, dhead.data<bf16_t>(), drough.data<float>(),
              kRows, kRank, kVocab, stream) == hipSuccess,
          "rough scores launch");
    hipStreamSynchronize(stream);

    std::vector<float> host_rough(static_cast<std::size_t>(kRows) * kVocab);
    check(download(drough, host_rough.data(), host_rough.size() * sizeof(float)),
          "rough download");
    double worst = 0.0;
    std::vector<float> want_rough(static_cast<std::size_t>(kRows) * kVocab);
    for (uint32_t r = 0; r < kRows; ++r) {
        std::vector<float> u(kRank, 0.0f);
        for (uint32_t j = 0; j < kRank; ++j) {
            double acc = 0.0;
            for (uint32_t d = 0; d < kHidden; ++d) {
                acc += bf16_to_f32(host_down[static_cast<std::size_t>(j) * kHidden + d]) *
                       bf16_to_f32(
                           host_hidden[static_cast<std::size_t>(r) * kHidden + d]);
            }
            u[j] = bf16_to_f32(bf16_t{f32_to_bf16_rne(static_cast<float>(acc))});
        }
        for (uint32_t v = 0; v < kVocab; ++v) {
            double acc = 0.0;
            for (uint32_t j = 0; j < kRank; ++j) {
                acc += bf16_to_f32(host_head[static_cast<std::size_t>(v) * kRank + j]) *
                       u[j];
            }
            want_rough[static_cast<std::size_t>(r) * kVocab + v] =
                bf16_to_f32(bf16_t{f32_to_bf16_rne(static_cast<float>(acc))});
            worst = std::max(
                worst, rel_error(host_rough[static_cast<std::size_t>(r) * kVocab + v],
                                 want_rough[static_cast<std::size_t>(r) * kVocab + v]));
        }
    }
    check(worst < 5.0e-2, "rough scores match the CPU reference (worst=" +
                              std::to_string(worst) + ")");

    check(ps::kernel::launch_dflash2_radix_topn(
              drough.data<float>(), kRows, kVocab, kVocab, kCandidates,
              dcand.data<int32_t>(), dradix_logits.data<float>(), partitions,
              dradix.data<uint8_t>(), scratch_bytes, stream) == hipSuccess,
          "radix launch");
    hipStreamSynchronize(stream);
    std::vector<int32_t> host_cand(static_cast<std::size_t>(kRows) * kCandidates);
    check(download(dcand, host_cand.data(), host_cand.size() * sizeof(int32_t)),
          "cand download");

    check(ps::kernel::launch_eagle4_candidate_logits(
              dlm.data<bf16_t>(), kHidden, dhidden.data<bf16_t>(), kHidden,
              dcand.data<int32_t>(), kRows, kCandidates, kHidden, dlogits.data<float>(),
              stream) == hipSuccess,
          "candidate logits launch");
    check(ps::kernel::launch_eagle4_candidate_topk(
              dlogits.data<float>(), dcand.data<int32_t>(), kRows, kCandidates, kTopK,
              dtop_ids.data<int32_t>(), dtop_logits.data<float>(), dtop_probs.data<float>(),
              stream) == hipSuccess,
          "candidate topk launch");
    hipStreamSynchronize(stream);

    std::vector<float> host_logits(static_cast<std::size_t>(kRows) * kCandidates);
    check(download(dlogits, host_logits.data(), host_logits.size() * sizeof(float)),
          "logits download");
    std::vector<int32_t> host_top(static_cast<std::size_t>(kRows) * kTopK);
    check(download(dtop_ids, host_top.data(), host_top.size() * sizeof(int32_t)),
          "top download");

    for (uint32_t r = 0; r < kRows; ++r) {
        std::vector<std::pair<float, int32_t>> entries;
        for (uint32_t c = 0; c < kCandidates; ++c) {
            double acc = 0.0;
            for (uint32_t d = 0; d < kHidden; ++d) {
                acc += bf16_to_f32(
                           host_lm[static_cast<std::size_t>(host_cand[
                                       static_cast<std::size_t>(r) * kCandidates + c]) *
                                       kHidden + d]) *
                       bf16_to_f32(
                           host_hidden[static_cast<std::size_t>(r) * kHidden + d]);
            }
            const float got =
                host_logits[static_cast<std::size_t>(r) * kCandidates + c];
            worst = std::max(worst, rel_error(got, static_cast<float>(acc)));
            entries.emplace_back(static_cast<float>(acc),
                                 host_cand[static_cast<std::size_t>(r) * kCandidates + c]);
        }
        std::sort(entries.begin(), entries.end(),
                  [](const std::pair<float, int32_t>& a,
                     const std::pair<float, int32_t>& b) {
                      if (a.first != b.first) return a.first > b.first;
                      return a.second < b.second;
                  });
        for (uint32_t i = 0; i < kTopK; ++i) {
            check(host_top[static_cast<std::size_t>(r) * kTopK + i] == entries[i].second,
                  "candidate topk id order row " + std::to_string(r) + " rank " +
                      std::to_string(i));
        }
        check(host_cand[static_cast<std::size_t>(r) * kCandidates] >= 0,
              "candidate id non-negative");
    }
    check(worst < 5.0e-2, "candidate logits match the CPU reference (worst=" +
                              std::to_string(worst) + ")");
}

static bool run(ps::gpu::GpuArena& arena, hipStream_t stream) {
    test_tap_norm_concat(arena, stream);
    test_ring_attention(arena, stream);
    test_candidate_ops(arena, stream);
    return g_fail == 0;
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
    const bool ok = run(arena, stream);
    (void)hipStreamDestroy(stream);
    std::printf("%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
