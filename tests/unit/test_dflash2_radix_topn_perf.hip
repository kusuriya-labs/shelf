#include "support/dflash2_topn_test_common.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

using dflash2_topn_test::Rng;

namespace {

constexpr uint32_t kWarmup = 100u;
constexpr uint32_t kBatch = 200u;
constexpr uint32_t kBatches = 24u;
constexpr uint32_t kRadixLaunches = 10u;

struct Sample {
    float median_us = 0.0f;
    float p10_us = 0.0f;
    float p90_us = 0.0f;
};

float percentile(std::vector<float>& values, double q) {
    const std::size_t n = values.size();
    std::size_t idx = static_cast<std::size_t>(q * static_cast<double>(n - 1u));
    return values[idx];
}

Sample summarize(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    Sample s;
    s.median_us = percentile(values, 0.5);
    s.p10_us = percentile(values, 0.10);
    s.p90_us = percentile(values, 0.90);
    return s;
}

void fill_logits(std::vector<float>& logits, bool tie_heavy) {
    Rng rng;
    rng.state = 0x5eed1234u;
    for (float& value : logits) {
        if (tie_heavy) {
            value = (rng.below(2u) == 0u) ? -0.25f : static_cast<float>(rng.below(97u)) * 0.001f;
        } else {
            value = rng.sym();
        }
    }
}

struct Runners {
    float* logits = nullptr;
    int32_t* radix_ids = nullptr;
    float* radix_values = nullptr;
    void* radix_scratch = nullptr;
};

template <typename Launch>
Sample measure(const std::vector<hipEvent_t>& events, Launch&& launch) {
    for (uint32_t i = 0u; i < kWarmup; ++i)
        launch();
    (void)hipDeviceSynchronize();
    std::vector<float> samples;
    samples.reserve(kBatches);
    for (uint32_t b = 0u; b < kBatches; ++b) {
        (void)hipEventRecord(events[0], nullptr);
        for (uint32_t i = 0u; i < kBatch; ++i)
            launch();
        (void)hipEventRecord(events[1], nullptr);
        (void)hipEventSynchronize(events[1]);
        float ms = 0.0f;
        (void)hipEventElapsedTime(&ms, events[0], events[1]);
        samples.push_back(ms * 1000.0f / static_cast<float>(kBatch));
    }
    return summarize(std::move(samples));
}

void run_matrix(const std::vector<hipEvent_t>& events, bool tie_heavy) {
    const uint32_t vocab = 248320u;
    const uint32_t radix_partitions = ps::kernel::dflash2_radix_default_partitions(vocab);
    const std::size_t radix_scratch_bytes =
        ps::kernel::dflash2_radix_topn_scratch_bytes(8u, radix_partitions, vocab);

    std::printf("input=%s vocab=%u radix_partitions=%u "
                "radix_scratch=%zuB radix_launches=%u warmup=%u "
                "batch=%u batches=%u\n",
                tie_heavy ? "tie" : "random", vocab, radix_partitions,
                radix_scratch_bytes, kRadixLaunches, kWarmup, kBatch, kBatches);
    std::printf("%4s %5s | %10s %10s %10s\n", "rows", "pool",
                "radix_med", "radix_p10", "radix_p90");

    for (uint32_t rows : {1u, 2u, 4u, 8u}) {
        Runners r;
        const std::size_t logit_elems = static_cast<std::size_t>(rows) * vocab;
        bool ok = hipMalloc(&r.logits, logit_elems * sizeof(float)) == hipSuccess &&
                  hipMalloc(&r.radix_ids, static_cast<std::size_t>(rows) * 128u * sizeof(int32_t)) ==
                      hipSuccess &&
                  hipMalloc(&r.radix_values, static_cast<std::size_t>(rows) * 128u * sizeof(float)) ==
                      hipSuccess &&
                  hipMalloc(&r.radix_scratch, radix_scratch_bytes) == hipSuccess;
        if (!ok) {
            std::printf("rows=%u allocation failed\n", rows);
            return;
        }
        std::vector<float> host_logits(logit_elems, 0.0f);
        fill_logits(host_logits, tie_heavy);
        (void)hipMemcpy(r.logits, host_logits.data(), logit_elems * sizeof(float),
                        hipMemcpyHostToDevice);

        for (uint32_t pool : {16u, 24u, 32u, 48u, 64u, 96u, 128u}) {
            const Sample radix = measure(events, [&] {
                (void)ps::kernel::launch_dflash2_radix_topn(
                    r.logits, rows, vocab, vocab, pool, r.radix_ids, r.radix_values,
                    radix_partitions, r.radix_scratch, radix_scratch_bytes, nullptr);
            });
            std::printf("%4u %5u | %10.3f %10.3f %10.3f\n",
                        rows, pool, radix.median_us, radix.p10_us, radix.p90_us);
        }

        (void)hipFree(r.logits);
        (void)hipFree(r.radix_ids);
        (void)hipFree(r.radix_values);
        (void)hipFree(r.radix_scratch);
    }
}

}  // namespace

int main() {
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess)
        return 1;

    std::vector<hipEvent_t> events(2u);
    for (auto& e : events) {
        if (hipEventCreate(&e) != hipSuccess) {
            std::printf("FAIL: hipEventCreate\n");
            return 1;
        }
    }

    run_matrix(events, false);
    run_matrix(events, true);

    for (auto& e : events)
        (void)hipEventDestroy(e);
    std::printf("test_dflash2_radix_topn_perf: done\n");
    return 0;
}
