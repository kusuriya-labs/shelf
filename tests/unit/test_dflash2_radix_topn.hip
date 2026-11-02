#include "support/dflash2_topn_test_common.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using dflash2_topn_test::Rng;

namespace {

int passed = 0;
int failed = 0;

void check(bool cond, const char* label) {
    if (cond) {
        ++passed;
    } else {
        ++failed;
        std::printf("FAIL: %s\n", label);
    }
}

uint32_t float_bits(float v) {
    uint32_t u = 0u;
    std::memcpy(&u, &v, sizeof(u));
    return u;
}

uint32_t ordered_bits(float v) {
    const uint32_t bits = float_bits(v);
    if (v != v)
        return 0u;
    if (v == 0.0f)
        return 0x80000000u;
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

uint64_t host_topn_key(float v, int32_t id) {
    return (static_cast<uint64_t>(ordered_bits(v)) << 32) | static_cast<uint32_t>(~id);
}

enum class Mode {
    Random,
    Tie,
    AllEqual,
    ZeroMix,
    DupMax,
    RepeatedThreshold,
    HasInfinity,
    HasNaN,
};

struct Oracle {
    std::vector<int32_t> ids;
    std::vector<float> values;
};

Oracle cpu_topn_key(const std::vector<float>& row, uint32_t pool) {
    std::vector<uint32_t> live;
    live.reserve(row.size());
    for (uint32_t i = 0u; i < row.size(); ++i) {
        if (row[i] != -INFINITY)
            live.push_back(i);
    }
    std::sort(live.begin(), live.end(), [&row](uint32_t a, uint32_t b) {
        return host_topn_key(row[a], static_cast<int32_t>(a)) >
               host_topn_key(row[b], static_cast<int32_t>(b));
    });
    Oracle out;
    const uint32_t n = static_cast<uint32_t>(std::min<std::size_t>(pool, live.size()));
    out.ids.resize(pool, 0);
    out.values.resize(pool, -INFINITY);
    for (uint32_t i = 0u; i < n; ++i) {
        out.ids[i] = static_cast<int32_t>(live[i]);
        out.values[i] = row[live[i]];
    }
    return out;
}

void fill_row(std::vector<float>& logits, uint32_t row_index, uint32_t vocab, Mode mode,
              Rng& rng) {
    const std::size_t beg = static_cast<std::size_t>(row_index) * vocab;
    for (uint32_t v = 0u; v < vocab; ++v) {
        float value = 0.0f;
        switch (mode) {
            case Mode::Random:
                value = rng.sym();
                break;
            case Mode::Tie:
                value = 1.0f - static_cast<float>(v % 3u) * 0.25f;
                break;
            case Mode::AllEqual:
                value = 0.375f;
                break;
            case Mode::ZeroMix:
                value = (v % 5u == 0u)   ? 0.0f
                        : (v % 5u == 1u) ? -0.0f
                        : (v % 5u == 2u) ? -0.0f
                                         : rng.sym() * 0.5f;
                break;
            case Mode::DupMax:
                value = (v % 2u == 0u) ? -1.5f + static_cast<float>(v % 7u) * 0.125f
                                       : -1.5f + static_cast<float>((v + 1u) % 7u) * 0.125f;
                break;
            case Mode::RepeatedThreshold:
                value = (v < vocab / 2u) ? 0.25f : rng.sym() * 0.1f;
                break;
            case Mode::HasInfinity:
                value = (rng.below(16u) == 0u) ? -INFINITY : rng.sym();
                break;
            case Mode::HasNaN:
                value = (rng.below(8u) == 0u) ? std::nanf("") : rng.sym();
                break;
        }
        logits[beg + v] = value;
    }
}

struct DeviceBuffers {
    float* logits = nullptr;
    int32_t* ids = nullptr;
    float* values = nullptr;
    void* radix_scratch = nullptr;

    void release() {
        (void)hipFree(logits);
        (void)hipFree(ids);
        (void)hipFree(values);
        (void)hipFree(radix_scratch);
        logits = nullptr;
        ids = nullptr;
        values = nullptr;
        radix_scratch = nullptr;
    }
};

bool run_case(uint32_t rows, uint32_t vocab, uint32_t pool, Mode mode, uint32_t seed,
              uint32_t partitions_override, const char* tag) {
    const uint32_t radix_partitions = partitions_override != 0u
                                          ? partitions_override
                                          : ps::kernel::dflash2_radix_default_partitions(vocab);
    const std::size_t out_elems = static_cast<std::size_t>(rows) * pool;

    const std::size_t radix_scratch_bytes =
        ps::kernel::dflash2_radix_topn_scratch_bytes(rows, radix_partitions, vocab);

    Rng rng;
    rng.state = seed;
    std::vector<float> logits(static_cast<std::size_t>(rows) * vocab, 0.0f);
    for (uint32_t r = 0u; r < rows; ++r)
        fill_row(logits, r, vocab, mode, rng);

    DeviceBuffers dev;
    bool alloc_ok =
        hipMalloc(&dev.logits, logits.size() * sizeof(float)) == hipSuccess &&
        hipMalloc(&dev.ids, out_elems * sizeof(int32_t)) == hipSuccess &&
        hipMalloc(&dev.values, out_elems * sizeof(float)) == hipSuccess &&
        hipMalloc(&dev.radix_scratch, radix_scratch_bytes) == hipSuccess;
    if (!alloc_ok) {
        dev.release();
        std::printf("FAIL: %s allocation\n", tag);
        return false;
    }
    (void)hipMemcpy(dev.logits, logits.data(), logits.size() * sizeof(float),
                    hipMemcpyHostToDevice);
    (void)hipMemset(dev.ids, 0, out_elems * sizeof(int32_t));
    (void)hipMemset(dev.values, 0, out_elems * sizeof(float));
    (void)hipMemset(dev.radix_scratch, 0, radix_scratch_bytes);

    hipError_t radix_err = ps::kernel::launch_dflash2_radix_topn(
        dev.logits, rows, vocab, vocab, pool, dev.ids, dev.values, radix_partitions,
        dev.radix_scratch, radix_scratch_bytes, nullptr);
    hipError_t sync_err = hipDeviceSynchronize();
    if (radix_err != hipSuccess || sync_err != hipSuccess) {
        std::printf("FAIL: %s launch radix=%s sync=%s\n", tag,
                    hipGetErrorString(radix_err), hipGetErrorString(sync_err));
        dev.release();
        return false;
    }

    std::vector<int32_t> ids(out_elems, 0);
    std::vector<float> values(out_elems, 0.0f);
    (void)hipMemcpy(ids.data(), dev.ids, out_elems * sizeof(int32_t), hipMemcpyDeviceToHost);
    (void)hipMemcpy(values.data(), dev.values, out_elems * sizeof(float), hipMemcpyDeviceToHost);

    bool radix_ids_exact = true;
    bool radix_bits_exact = true;
    bool unique = true;
    bool full_rows = true;
    for (uint32_t r = 0u; r < rows; ++r) {
        std::vector<float> row(logits.begin() + static_cast<std::size_t>(r) * vocab,
                               logits.begin() + static_cast<std::size_t>(r + 1u) * vocab);
        const Oracle oracle = cpu_topn_key(row, pool);
        uint32_t live = 0u;
        for (float value : row) {
            if (value != -INFINITY)
                ++live;
        }
        if (live < pool)
            full_rows = false;
        for (uint32_t i = 0u; i < pool; ++i) {
            const std::size_t idx = static_cast<std::size_t>(r) * pool + i;
            if (ids[idx] != oracle.ids[i])
                radix_ids_exact = false;
            if (std::memcmp(&values[idx], &oracle.values[i], sizeof(float)) != 0)
                radix_bits_exact = false;
            for (uint32_t j = i + 1u; j < pool; ++j) {
                if (ids[idx] != 0 && ids[idx] == ids[static_cast<std::size_t>(r) * pool + j])
                    unique = false;
            }
        }
    }
    if (!full_rows)
        unique = true;

    char label[192];
    std::snprintf(label, sizeof(label), "%s radix ids exact", tag);
    check(radix_ids_exact, label);
    std::snprintf(label, sizeof(label), "%s radix logits bit exact", tag);
    check(radix_bits_exact, label);
    std::snprintf(label, sizeof(label), "%s radix ids unique", tag);
    check(unique, label);

    (void)hipMemset(dev.ids, 0, out_elems * sizeof(int32_t));
    (void)hipMemset(dev.values, 0, out_elems * sizeof(float));
    hipError_t rerun = ps::kernel::launch_dflash2_radix_topn(
        dev.logits, rows, vocab, vocab, pool, dev.ids, dev.values, radix_partitions,
        dev.radix_scratch, radix_scratch_bytes, nullptr);
    (void)hipDeviceSynchronize();
    std::vector<int32_t> ids2(out_elems, 0);
    std::vector<float> values2(out_elems, 0.0f);
    if (rerun == hipSuccess) {
        (void)hipMemcpy(ids2.data(), dev.ids, out_elems * sizeof(int32_t),
                        hipMemcpyDeviceToHost);
        (void)hipMemcpy(values2.data(), dev.values, out_elems * sizeof(float),
                        hipMemcpyDeviceToHost);
    }
    const bool deterministic = rerun == hipSuccess &&
                               std::memcmp(ids.data(), ids2.data(),
                                           ids.size() * sizeof(int32_t)) == 0 &&
                               std::memcmp(values.data(), values2.data(),
                                           values.size() * sizeof(float)) == 0;
    std::snprintf(label, sizeof(label), "%s radix deterministic", tag);
    check(deterministic, label);

    dev.release();
    return true;
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

    uint32_t seed = 0x9e3779b9u;
    const uint32_t pools[] = {16u, 24u, 32u, 48u, 64u, 80u, 128u};

    for (uint32_t pool : pools) {
        for (Mode mode : {Mode::Random, Mode::Tie}) {
            char tag[128];
            std::snprintf(tag, sizeof(tag), "vocab=1024 rows=7 pool=%u mode=%d", pool,
                          static_cast<int>(mode));
            if (!run_case(7u, 1024u, pool, mode, seed, 0u, tag))
                ++failed;
            seed = seed * 1664525u + 1013904223u;
        }
    }
    for (uint32_t pool : pools) {
        char tag[128];
        std::snprintf(tag, sizeof(tag), "vocab=4096 rows=7 pool=%u", pool);
        if (!run_case(7u, 4096u, pool, Mode::Random, seed, 0u, tag))
            ++failed;
        seed = seed * 1664525u + 1013904223u;
    }
    for (uint32_t pool : pools) {
        char tag[128];
        std::snprintf(tag, sizeof(tag), "vocab=4096 rows=7 pool=%u tie", pool);
        if (!run_case(7u, 4096u, pool, Mode::Tie, seed, 0u, tag))
            ++failed;
        seed = seed * 1664525u + 1013904223u;
    }
    for (uint32_t pool : {16u, 32u, 64u, 80u, 128u}) {
        for (Mode mode : {Mode::Random, Mode::Tie}) {
            char tag[128];
            std::snprintf(tag, sizeof(tag), "vocab=248320 rows=7 pool=%u mode=%d", pool,
                          static_cast<int>(mode));
            if (!run_case(7u, 248320u, pool, mode, seed, 0u, tag))
                ++failed;
            seed = seed * 1664525u + 1013904223u;
        }
    }
    for (uint32_t rows : {1u, 3u}) {
        char tag[128];
        std::snprintf(tag, sizeof(tag), "rows=%u vocab=4096 pool=32", rows);
        if (!run_case(rows, 4096u, 32u, Mode::Random, seed, 0u, tag))
            ++failed;
        seed = seed * 1664525u + 1013904223u;
    }

    const struct {
        Mode mode;
        const char* name;
    } specials[] = {
        {Mode::AllEqual, "all equal"},
        {Mode::ZeroMix, "zero mix"},
        {Mode::DupMax, "duplicate max"},
        {Mode::RepeatedThreshold, "repeated threshold"},
        {Mode::HasInfinity, "has -INFINITY"},
        {Mode::HasNaN, "has NaN"},
    };
    for (const auto& s : specials) {
        for (uint32_t pool : {16u, 32u, 64u, 128u}) {
            char tag[128];
            std::snprintf(tag, sizeof(tag), "special vocab=4096 rows=3 pool=%u %s", pool,
                          s.name);
            if (!run_case(3u, 4096u, pool, s.mode, seed, 0u, tag))
                ++failed;
            seed = seed * 1664525u + 1013904223u;
        }
    }
    for (uint32_t pool : {16u, 64u}) {
        char tag[128];
        std::snprintf(tag, sizeof(tag), "special vocab=248320 rows=1 pool=%u all equal",
                      pool);
        if (!run_case(1u, 248320u, pool, Mode::AllEqual, seed, 0u, tag))
            ++failed;
        seed = seed * 1664525u + 1013904223u;
        std::snprintf(tag, sizeof(tag), "special vocab=248320 rows=1 pool=%u has NaN", pool);
        if (!run_case(1u, 248320u, pool, Mode::HasNaN, seed, 0u, tag))
            ++failed;
        seed = seed * 1664525u + 1013904223u;
    }
    for (uint32_t pool : {16u, 48u}) {
        char tag[128];
        std::snprintf(tag, sizeof(tag), "small vocab=512 rows=3 pool=%u parts=8", pool);
        if (!run_case(3u, 512u, pool, Mode::Random, seed, 8u, tag))
            ++failed;
        seed = seed * 1664525u + 1013904223u;
        std::snprintf(tag, sizeof(tag), "small vocab=257 rows=3 pool=%u parts=4", pool);
        if (!run_case(3u, 257u, pool, Mode::Tie, seed, 4u, tag))
            ++failed;
        seed = seed * 1664525u + 1013904223u;
    }

    std::printf("test_dflash2_radix_topn: passed=%d failed=%d\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
