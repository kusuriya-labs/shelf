#include <shelf/core/memory/types.h>
#include <shelf/kernels/optimized/qsa.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    } else {
        std::fprintf(stderr, "ok: %s\n", what);
    }
}

float bf16_to_float(std::uint16_t bits) {
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16u;
    float value = 0.0f;
    std::memcpy(&value, &wide, sizeof(value));
    return value;
}

std::vector<shelf::bf16_t> random_bf16(std::size_t count, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<shelf::bf16_t> out(count);
    for (auto& value : out) value.data = shelf::f32_to_bf16_rne(dist(rng));
    return out;
}

}  // namespace

int main() {
    constexpr std::uint32_t kHeadDim = shelf::kernel::kQsaHeadDim;
    std::mt19937 rng(777);

    {
        constexpr std::uint32_t kRows = 8;
        constexpr std::uint32_t kBlocks = 2;
        constexpr std::uint32_t kCr = 4;
        const auto raw = random_bf16(kRows * kHeadDim, rng);
        const auto weight = random_bf16(kHeadDim, rng);
        std::vector<float> want(kBlocks * kHeadDim);
        for (std::uint32_t b = 0; b < kBlocks; ++b) {
            float mean[kHeadDim] = {};
            for (std::uint32_t t = 0; t < kCr; ++t) {
                for (std::uint32_t d = 0; d < kHeadDim; ++d) {
                    mean[d] += bf16_to_float(raw[(b * kCr + t) * kHeadDim + d].data);
                }
            }
            float square = 0.0f;
            for (std::uint32_t d = 0; d < kHeadDim; ++d) {
                mean[d] /= static_cast<float>(kCr);
                square += mean[d] * mean[d];
            }
            const float inv = 1.0f / std::sqrt(square / kHeadDim + 1e-6f);
            for (std::uint32_t d = 0; d < kHeadDim; ++d) {
                want[b * kHeadDim + d] =
                    mean[d] * inv * (1.0f + bf16_to_float(weight[d].data));
            }
        }

        void* d_raw = nullptr;
        void* d_weight = nullptr;
        void* d_pooled = nullptr;
        check(hipMalloc(&d_raw, raw.size() * sizeof(shelf::bf16_t)) == hipSuccess,
              "hipMalloc raw");
        check(hipMalloc(&d_weight, weight.size() * sizeof(shelf::bf16_t)) == hipSuccess,
              "hipMalloc weight");
        check(hipMalloc(&d_pooled, want.size() * sizeof(shelf::bf16_t)) == hipSuccess,
              "hipMalloc pooled");
        hipMemcpy(d_raw, raw.data(), raw.size() * sizeof(shelf::bf16_t),
                  hipMemcpyHostToDevice);
        hipMemcpy(d_weight, weight.data(), weight.size() * sizeof(shelf::bf16_t),
                  hipMemcpyHostToDevice);
        shelf::kernel::QsaPoolArgs args;
        args.raw_keys = static_cast<const shelf::bf16_t*>(d_raw);
        args.norm_weight = static_cast<const shelf::bf16_t*>(d_weight);
        args.pooled = static_cast<shelf::bf16_t*>(d_pooled);
        args.rows = kRows;
        args.blocks = kBlocks;
        args.head_dim = kHeadDim;
        args.compress_ratio = kCr;
        args.eps = 1e-6f;
        check(shelf::kernel::launch_qsa_pool(args, nullptr) == hipSuccess,
              "qsa pool launch");
        check(hipDeviceSynchronize() == hipSuccess, "qsa pool sync");
        std::vector<shelf::bf16_t> got(want.size());
        hipMemcpy(got.data(), d_pooled, got.size() * sizeof(shelf::bf16_t),
                  hipMemcpyDeviceToHost);
        double max_error = 0.0;
        for (std::size_t i = 0; i < got.size(); ++i) {
            max_error = std::max<double>(
                max_error, std::fabs(static_cast<double>(bf16_to_float(got[i].data)) -
                                     want[i]));
        }
        std::fprintf(stderr, "qsa pool max_abs=%g\n", max_error);
        check(max_error <= 2.0e-2, "qsa pool matches the CPU reference");
        (void)hipFree(d_raw);
        (void)hipFree(d_weight);
        (void)hipFree(d_pooled);
    }

    {
        constexpr std::uint32_t kRows = 3;
        constexpr std::uint32_t kBlocks = 5;
        constexpr std::uint32_t kHeads = 4;
        const auto q = random_bf16(kRows * kHeads * kHeadDim, rng);
        const auto pooled = random_bf16(kBlocks * kHeadDim, rng);
        const std::uint32_t visible[kRows] = {5u, 3u, 1u};
        const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
        std::vector<float> want(kRows * kBlocks);
        for (std::uint32_t r = 0; r < kRows; ++r) {
            for (std::uint32_t b = 0; b < kBlocks; ++b) {
                if (b >= visible[r]) {
                    want[r * kBlocks + b] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                float total = 0.0f;
                for (std::uint32_t h = 0; h < kHeads; ++h) {
                    float dot = 0.0f;
                    for (std::uint32_t d = 0; d < kHeadDim; ++d) {
                        dot += bf16_to_float(q[(r * kHeads + h) * kHeadDim + d].data) *
                               bf16_to_float(pooled[b * kHeadDim + d].data);
                    }
                    total += std::max(dot, 0.0f);
                }
                want[r * kBlocks + b] = total * scale;
            }
        }

        void* d_q = nullptr;
        void* d_pooled = nullptr;
        void* d_scores = nullptr;
        void* d_visible = nullptr;
        check(hipMalloc(&d_q, q.size() * sizeof(shelf::bf16_t)) == hipSuccess,
              "hipMalloc q");
        check(hipMalloc(&d_pooled, pooled.size() * sizeof(shelf::bf16_t)) == hipSuccess,
              "hipMalloc pooled");
        check(hipMalloc(&d_scores, want.size() * sizeof(float)) == hipSuccess,
              "hipMalloc scores");
        check(hipMalloc(&d_visible, sizeof(visible)) == hipSuccess, "hipMalloc visible");
        hipMemcpy(d_q, q.data(), q.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice);
        hipMemcpy(d_pooled, pooled.data(), pooled.size() * sizeof(shelf::bf16_t),
                  hipMemcpyHostToDevice);
        hipMemcpy(d_visible, visible, sizeof(visible), hipMemcpyHostToDevice);
        shelf::kernel::QsaScoresArgs args;
        args.q = static_cast<const shelf::bf16_t*>(d_q);
        args.pooled = static_cast<const shelf::bf16_t*>(d_pooled);
        args.scores = static_cast<float*>(d_scores);
        args.visible = static_cast<const std::uint32_t*>(d_visible);
        args.rows = kRows;
        args.blocks = kBlocks;
        args.n_heads = kHeads;
        args.head_dim = kHeadDim;
        args.scale = scale;
        check(shelf::kernel::launch_qsa_scores(args, nullptr) == hipSuccess,
              "qsa scores launch");
        check(hipDeviceSynchronize() == hipSuccess, "qsa scores sync");
        std::vector<float> got(want.size());
        hipMemcpy(got.data(), d_scores, got.size() * sizeof(float),
                  hipMemcpyDeviceToHost);
        double max_error = 0.0;
        for (std::size_t i = 0; i < got.size(); ++i) {
            if (!std::isfinite(want[i])) {
                check(!std::isfinite(got[i]), "qsa score beyond visible is -inf");
                continue;
            }
            max_error = std::max<double>(max_error, std::fabs(got[i] - want[i]));
        }
        std::fprintf(stderr, "qsa scores max_abs=%g\n", max_error);
        check(max_error <= 2.0e-2, "qsa scores match the CPU reference");
        (void)hipFree(d_q);
        (void)hipFree(d_pooled);
        (void)hipFree(d_scores);
        (void)hipFree(d_visible);
    }

    std::fprintf(stderr, failures == 0 ? "PASS\n" : "FAILURES=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
