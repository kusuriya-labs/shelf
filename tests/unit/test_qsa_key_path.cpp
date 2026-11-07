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
        constexpr std::uint32_t kHeads = 4;
        constexpr std::uint32_t kRotary = 32;
        constexpr float kTheta = 10000000.0f;
        const std::uint32_t positions[kRows] = {0u, 5u, 100u};
        auto x = random_bf16(kRows * kHeads * kHeadDim, rng);
        auto want = x;
        const std::uint32_t half = kRotary / 2u;
        for (std::uint32_t r = 0; r < kRows; ++r) {
            for (std::uint32_t h = 0; h < kHeads; ++h) {
                for (std::uint32_t i = 0; i < half; ++i) {
                    const float exponent = -2.0f * static_cast<float>(i) / kRotary;
                    const float angle = (static_cast<float>(positions[r]) / 2.0f) *
                                        std::pow(kTheta, exponent);
                    const float cosine = std::cos(angle);
                    const float sine = std::sin(angle);
                    const std::size_t base = (r * kHeads + h) * kHeadDim;
                    const float x0 = bf16_to_float(x[base + i].data);
                    const float x1 = bf16_to_float(x[base + i + half].data);
                    want[base + i].data =
                        shelf::f32_to_bf16_rne(x0 * cosine - x1 * sine);
                    want[base + i + half].data =
                        shelf::f32_to_bf16_rne(x1 * cosine + x0 * sine);
                }
            }
        }
        void* d_x = nullptr;
        void* d_positions = nullptr;
        check(hipMalloc(&d_x, x.size() * sizeof(shelf::bf16_t)) == hipSuccess,
              "hipMalloc rope x");
        check(hipMalloc(&d_positions, sizeof(positions)) == hipSuccess,
              "hipMalloc rope positions");
        hipMemcpy(d_x, x.data(), x.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice);
        hipMemcpy(d_positions, positions, sizeof(positions), hipMemcpyHostToDevice);
        shelf::kernel::QsaRopeArgs args;
        args.x = static_cast<shelf::bf16_t*>(d_x);
        args.positions = static_cast<const std::uint32_t*>(d_positions);
        args.rows = kRows;
        args.heads = kHeads;
        args.head_dim = kHeadDim;
        args.rotary_dim = kRotary;
        args.theta = kTheta;
        args.position_scale = 2.0f;
        check(shelf::kernel::launch_qsa_rope(args, nullptr) == hipSuccess,
              "qsa rope launch");
        check(hipDeviceSynchronize() == hipSuccess, "qsa rope sync");
        std::vector<shelf::bf16_t> got(x.size());
        hipMemcpy(got.data(), d_x, got.size() * sizeof(shelf::bf16_t),
                  hipMemcpyDeviceToHost);
        double max_error = 0.0;
        for (std::size_t i = 0; i < got.size(); ++i) {
            max_error = std::max<double>(
                max_error, std::fabs(static_cast<double>(bf16_to_float(got[i].data)) -
                                     bf16_to_float(want[i].data)));
        }
        std::fprintf(stderr, "qsa rope max_abs=%g\n", max_error);
        check(max_error <= 2.0e-2, "qsa rope matches the CPU reference");
        (void)hipFree(d_x);
        (void)hipFree(d_positions);
    }

    {
        constexpr std::uint32_t kRows = 10;
        constexpr std::uint32_t kCr = 4;
        constexpr std::uint32_t kRotary = 32;
        constexpr std::uint32_t kMaxBlocks = 4;
        constexpr float kTheta = 10000000.0f;
        const auto raw = random_bf16(kRows * kHeadDim, rng);
        const auto norm_weight = random_bf16(kHeadDim, rng);
        std::vector<std::uint32_t> positions(kRows);
        for (std::uint32_t i = 0; i < kRows; ++i) positions[i] = i;

        std::vector<shelf::bf16_t> want_pooled(kMaxBlocks * kHeadDim, shelf::bf16_t{0});
        std::vector<shelf::bf16_t> want_partial(kCr * kHeadDim, shelf::bf16_t{0});
        std::uint32_t want_pool_count = 0;
        std::uint32_t want_partial_count = 0;
        for (std::uint32_t row = 0; row < kRows; ++row) {
            for (std::uint32_t d = 0; d < kHeadDim; ++d) {
                want_partial[want_partial_count * kHeadDim + d] =
                    raw[row * kHeadDim + d];
            }
            ++want_partial_count;
            if (want_partial_count == kCr) {
                float mean[kHeadDim];
                float square = 0.0f;
                for (std::uint32_t d = 0; d < kHeadDim; ++d) {
                    float sum = 0.0f;
                    for (std::uint32_t t = 0; t < kCr; ++t) {
                        sum += bf16_to_float(
                            want_partial[t * kHeadDim + d].data);
                    }
                    mean[d] = sum / static_cast<float>(kCr);
                    square += mean[d] * mean[d];
                }
                const float inv = 1.0f / std::sqrt(square / kHeadDim + 1e-6f);
                float normed[kHeadDim];
                for (std::uint32_t d = 0; d < kHeadDim; ++d) {
                    normed[d] = mean[d] * inv *
                                (1.0f + bf16_to_float(norm_weight[d].data));
                }
                const std::uint32_t start = positions[row] - (kCr - 1u);
                for (std::uint32_t d = 0; d < kHeadDim; ++d) {
                    float value = normed[d];
                    if (d < kRotary / 2u) {
                        const float exponent = -2.0f * d / kRotary;
                        const float angle =
                            (static_cast<float>(start) / 2.0f) *
                            std::pow(kTheta, exponent);
                        const float c = std::cos(angle);
                        const float sn = std::sin(angle);
                        value = normed[d] * c - normed[d + kRotary / 2u] * sn;
                    } else if (d < kRotary) {
                        const std::uint32_t i = d - kRotary / 2u;
                        const float exponent = -2.0f * i / kRotary;
                        const float angle =
                            (static_cast<float>(start) / 2.0f) *
                            std::pow(kTheta, exponent);
                        const float c = std::cos(angle);
                        const float sn = std::sin(angle);
                        value = normed[d] * c + normed[i] * sn;
                    }
                    want_pooled[want_pool_count * kHeadDim + d].data =
                        shelf::f32_to_bf16_rne(value);
                }
                ++want_pool_count;
                want_partial_count = 0;
            }
        }

        void* d_raw = nullptr;
        void* d_weight = nullptr;
        void* d_positions = nullptr;
        void* d_pooled = nullptr;
        void* d_partial = nullptr;
        void* d_pool_count = nullptr;
        void* d_partial_count = nullptr;
        check(hipMalloc(&d_raw, raw.size() * sizeof(shelf::bf16_t)) == hipSuccess,
              "hipMalloc cache raw");
        check(hipMalloc(&d_weight, norm_weight.size() * sizeof(shelf::bf16_t)) ==
                  hipSuccess,
              "hipMalloc cache weight");
        check(hipMalloc(&d_positions, positions.size() * sizeof(std::uint32_t)) ==
                  hipSuccess,
              "hipMalloc cache positions");
        check(hipMalloc(&d_pooled, want_pooled.size() * sizeof(shelf::bf16_t)) ==
                  hipSuccess,
              "hipMalloc cache pooled");
        check(hipMalloc(&d_partial, want_partial.size() * sizeof(shelf::bf16_t)) ==
                  hipSuccess,
              "hipMalloc cache partial");
        check(hipMalloc(&d_pool_count, sizeof(std::uint32_t)) == hipSuccess,
              "hipMalloc cache pool count");
        check(hipMalloc(&d_partial_count, sizeof(std::uint32_t)) == hipSuccess,
              "hipMalloc cache partial count");
        hipMemcpy(d_raw, raw.data(), raw.size() * sizeof(shelf::bf16_t),
                  hipMemcpyHostToDevice);
        hipMemcpy(d_weight, norm_weight.data(),
                  norm_weight.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice);
        hipMemcpy(d_positions, positions.data(), positions.size() * sizeof(std::uint32_t),
                  hipMemcpyHostToDevice);
        std::uint32_t zero = 0;
        hipMemcpy(d_pool_count, &zero, sizeof(zero), hipMemcpyHostToDevice);
        hipMemcpy(d_partial_count, &zero, sizeof(zero), hipMemcpyHostToDevice);

        shelf::kernel::QsaKeyCacheArgs args;
        args.raw_keys = static_cast<const shelf::bf16_t*>(d_raw);
        args.norm_weight = static_cast<const shelf::bf16_t*>(d_weight);
        args.positions = static_cast<const std::uint32_t*>(d_positions);
        args.pooled = static_cast<shelf::bf16_t*>(d_pooled);
        args.partial = static_cast<shelf::bf16_t*>(d_partial);
        args.pool_count = static_cast<std::uint32_t*>(d_pool_count);
        args.partial_count = static_cast<std::uint32_t*>(d_partial_count);
        args.rows = kRows;
        args.max_blocks = kMaxBlocks;
        args.head_dim = kHeadDim;
        args.compress_ratio = kCr;
        args.rotary_dim = kRotary;
        args.theta = kTheta;
        args.position_scale = 2.0f;
        check(shelf::kernel::launch_qsa_key_cache(args, nullptr) == hipSuccess,
              "qsa key cache launch");
        check(hipDeviceSynchronize() == hipSuccess, "qsa key cache sync");
        std::vector<shelf::bf16_t> got_pooled(want_pooled.size());
        std::vector<shelf::bf16_t> got_partial(want_partial.size());
        std::uint32_t got_pool_count = 0;
        std::uint32_t got_partial_count = 0;
        hipMemcpy(got_pooled.data(), d_pooled, got_pooled.size() * sizeof(shelf::bf16_t),
                  hipMemcpyDeviceToHost);
        hipMemcpy(got_partial.data(), d_partial,
                  got_partial.size() * sizeof(shelf::bf16_t), hipMemcpyDeviceToHost);
        hipMemcpy(&got_pool_count, d_pool_count, sizeof(got_pool_count),
                  hipMemcpyDeviceToHost);
        hipMemcpy(&got_partial_count, d_partial_count, sizeof(got_partial_count),
                  hipMemcpyDeviceToHost);
        double max_pooled = 0.0;
        for (std::uint32_t b = 0; b < want_pool_count; ++b) {
            for (std::uint32_t d = 0; d < kHeadDim; ++d) {
                const std::uint32_t i = b * kHeadDim + d;
                max_pooled = std::max<double>(
                    max_pooled,
                    std::fabs(static_cast<double>(bf16_to_float(got_pooled[i].data)) -
                              bf16_to_float(want_pooled[i].data)));
            }
        }
        double max_partial = 0.0;
        for (std::uint32_t t = 0; t < want_partial_count; ++t) {
            for (std::uint32_t d = 0; d < kHeadDim; ++d) {
                const std::uint32_t i = t * kHeadDim + d;
                max_partial = std::max<double>(
                    max_partial,
                    std::fabs(static_cast<double>(bf16_to_float(got_partial[i].data)) -
                              bf16_to_float(want_partial[i].data)));
            }
        }
        std::fprintf(stderr, "qsa key cache pooled max_abs=%g partial max_abs=%g\n",
                     max_pooled, max_partial);
        check(got_pool_count == want_pool_count && got_partial_count == want_partial_count,
              "qsa key cache counts match");
        check(max_pooled <= 2.0e-2 && max_partial <= 2.0e-2,
              "qsa key cache matches the CPU reference");
        (void)hipFree(d_raw);
        (void)hipFree(d_weight);
        (void)hipFree(d_positions);
        (void)hipFree(d_pooled);
        (void)hipFree(d_partial);
        (void)hipFree(d_pool_count);
        (void)hipFree(d_partial_count);
    }


    std::fprintf(stderr, failures == 0 ? "PASS\n" : "FAILURES=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
