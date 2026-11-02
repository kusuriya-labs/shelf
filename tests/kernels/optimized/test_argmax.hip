#include <phaseshift/kernels/optimized/sampling.h>
#include <phaseshift/core/gpu/cleanup.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <vector>

namespace {

using ps::kernel::ArgMaxPair;
using ps::kernel::ArgmaxF32Args;

int32_t cpu_argmax(const float* logits, uint32_t vocab) {
    float bv = -3.4e38f;
    uint32_t bi = 0xFFFFFFFFu;
    for (uint32_t v = 0; v < vocab; ++v) {
        const float val = logits[v];
        if (val > bv || (val == bv && v < bi)) {
            bv = val;
            bi = v;
        }
    }
    return static_cast<int32_t>(bi);
}

uint32_t h32(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return static_cast<uint32_t>(x);
}

float rand_f(uint64_t s, int lo = -10, int hi = 10) {
    const float r = static_cast<float>(h32(s) % 10000u) / 10000.0f;
    return static_cast<float>(lo) + r * static_cast<float>(hi - lo);
}

struct Buf {
    void* dev = nullptr;
    size_t bytes = 0;
    bool alloc(size_t b) {
        bytes = b;
        return hipMalloc(&dev, b) == hipSuccess;
    }
    ~Buf() {
        if (dev) ps::gpu::discard_cleanup_result(hipFree(dev));
    }
};

struct Cfg {
    const char* name;
    uint32_t vocab;
    uint32_t outputs;
    uint32_t row_stride;  // 0 => vocab
    std::function<void(std::vector<float>&)> fill;  // size outputs*rs
    std::vector<uint8_t> flags;
};

int g_fail = 0;

void run_case(const Cfg& c, uint32_t partitions) {
    const uint32_t rs = c.row_stride ? c.row_stride : c.vocab;
    const size_t n = static_cast<size_t>(c.outputs) * rs;
    std::vector<float> logits(n, 0.0f);
    if (c.fill) c.fill(logits);
    std::vector<uint8_t> flags = c.flags;
    if (flags.empty()) flags.assign(c.outputs, 1);
    std::vector<::ps::runtime::DeviceSamplingParams> sampling(c.outputs);
    for (uint32_t m = 0; m < c.outputs; ++m) {
        sampling[m] = ::ps::runtime::DeviceSamplingParams{};
        sampling[m].mode = flags[m] ? 1u : 0u;
    }

    std::vector<int32_t> expected(c.outputs, -1);
    for (uint32_t m = 0; m < c.outputs; ++m)
        expected[m] = flags[m]
                          ? cpu_argmax(logits.data() + static_cast<size_t>(m) * rs, c.vocab)
                          : -1;

    Buf bl, bf, bo, bp;
    if (!bl.alloc(n * 4) ||
        !bf.alloc(c.outputs * sizeof(::ps::runtime::DeviceSamplingParams)) ||
        !bo.alloc(c.outputs * 4)) {
        std::printf("  [%s] ALLOC FAIL\n", c.name);
        ++g_fail;
        return;
    }
    if (partitions)
        if (!bp.alloc(static_cast<size_t>(c.outputs) * partitions * sizeof(ArgMaxPair))) {
            std::printf("  [%s] PARTIAL ALLOC FAIL\n", c.name);
            ++g_fail;
            return;
        }
    if (hipMemcpy(bl.dev, logits.data(), n * 4, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(bf.dev, sampling.data(),
                  c.outputs * sizeof(::ps::runtime::DeviceSamplingParams),
                  hipMemcpyHostToDevice) != hipSuccess) {
        std::printf("  [%s] COPY FAIL\n", c.name);
        ++g_fail;
        return;
    }

    size_t mm = 0;
    for (int impl = 0; impl < 2; ++impl) {
        const bool part = (impl == 1);
        if (part && !partitions) continue;
        if (c.outputs == 0) continue;
        if (hipMemset(bo.dev, 0x77, c.outputs * 4) != hipSuccess) {
            mm += 1000;
            continue;
        }
        ArgmaxF32Args args;
        args.logits = static_cast<const float*>(bl.dev);
        args.sampling = static_cast<const ::ps::runtime::DeviceSamplingParams*>(bf.dev);
        args.sampled_tokens = static_cast<int32_t*>(bo.dev);
        args.outputs = c.outputs;
        args.vocab_size = c.vocab;
        args.logits_row_stride = rs;
        hipError_t st;
        if (part)
            st = ps::kernel::launch_argmax_f32_partitioned(
                args, static_cast<ArgMaxPair*>(bp.dev),
                static_cast<uint32_t>(c.outputs) * partitions, partitions, nullptr);
        else
            st = ps::kernel::launch_argmax_f32_single(args, nullptr);
        if (st != hipSuccess || hipDeviceSynchronize() != hipSuccess) {
            mm += 1000;
            continue;
        }
        std::vector<int32_t> got(c.outputs);
        if (hipMemcpy(got.data(), bo.dev, c.outputs * 4, hipMemcpyDeviceToHost) != hipSuccess) {
            mm += 1000;
            continue;
        }
        for (uint32_t m = 0; m < c.outputs; ++m)
            if (got[m] != expected[m]) ++mm;
    }
    const bool pass = (mm == 0);
    if (!pass) ++g_fail;
    std::printf("  [%s] v=%u out=%u rs=%u P=%u | mm=%zu %s\n", c.name, c.vocab,
                c.outputs, rs, partitions, mm, pass ? "PASS" : "FAIL");
}

}  // namespace

int main(int, char**) {
    if (hipSetDevice(0) != hipSuccess) return 1;

    const uint32_t P = 8;
    Cfg cases[] = {
        {"unique128", 128, 4, 0,
         [](std::vector<float>& l) {
             for (size_t i = 0; i < l.size(); ++i) l[i] = rand_f(0xB0B + i);
             l[100] = 1e5f;
         },
         {}},
        {"unique248k", 248320, 2, 0,
         [](std::vector<float>& l) {
             for (size_t i = 0; i < l.size(); ++i) l[i] = rand_f(0xC0DE + i);
         },
         {}},
        {"tie-lane", 256, 1, 0,
         [](std::vector<float>& l) {
             std::fill(l.begin(), l.end(), 0.0f);
             l[3] = 100.0f;
             l[20] = 100.0f;
         },
         {}},
        {"tie-crosswave", 1024, 1, 0,
         [](std::vector<float>& l) {
             std::fill(l.begin(), l.end(), 0.0f);
             l[500] = 50.0f;
             l[100] = 50.0f;
         },
         {}},
        {"tie-partition", 248320, 1, 0,
         [](std::vector<float>& l) {
             std::fill(l.begin(), l.end(), 0.0f);
             l[100000] = 42.0f;
             l[5000] = 42.0f;
         },
         {}},
        {"all-equal", 1024, 3, 0,
         [](std::vector<float>& l) { std::fill(l.begin(), l.end(), 1.0f); },
         {}},
        {"all-negative", 512, 2, 0,
         [](std::vector<float>& l) {
             std::fill(l.begin(), l.end(), -5.0f);
             l[7] = -1.0f;
             l[3] = -1.0f;
         },
         {}},
        {"flags-none", 128, 4, 0,
         [](std::vector<float>& l) {
             for (size_t i = 0; i < l.size(); ++i) l[i] = rand_f(0xF + i);
         },
         {0, 0, 0, 0}},
        {"flags-mixed", 128, 4, 0,
         [](std::vector<float>& l) {
             for (size_t i = 0; i < l.size(); ++i) l[i] = rand_f(0xF + i);
         },
         {1, 0, 1, 0}},
        {"row-stride-pad", 128, 2, 136,
         [](std::vector<float>& l) {
             for (size_t i = 0; i < l.size(); ++i) l[i] = rand_f(0x55 + i);
         },
         {}},
        {"row-stride-tie", 64, 1, 80,
         [](std::vector<float>& l) {
             std::fill(l.begin(), l.end(), 0.0f);
             l[10] = 9.0f;
             l[50] = 9.0f;
         },
         {}},
        {"actual-qwen", 248320, 1, 0,
         [](std::vector<float>& l) {
             for (size_t i = 0; i < l.size(); ++i) l[i] = rand_f(0x710 + i);
             l[h32(0x1234) % 248320u] = 3.14159f;
         },
         {}},
        {"many-outputs", 248320, 8, 0,
         [](std::vector<float>& l) {
             for (size_t i = 0; i < l.size(); ++i) l[i] = rand_f(0x744 + i);
         },
         {}},
        {"plus-zero-minus-zero", 256, 1, 0,
         [](std::vector<float>& l) {
             std::fill(l.begin(), l.end(), 0.0f);
             l[5] = -0.0f;
             l[2] = 0.0f;
         },
         {}},
    };

    for (const auto& c : cases) {
        run_case(c, P);
        run_case(c, 0);
    }

    // outputs=0 no-op: pre-fill output, verify unchanged
    {
        Buf bo;
        std::vector<int32_t> four = {111, 222, 333, 444};
        ArgmaxF32Args args;
        args.outputs = 0;
        bool ok = bo.alloc(4 * 4);
        if (ok)
            ok = hipMemcpy(bo.dev, four.data(), 16, hipMemcpyHostToDevice) == hipSuccess;
        args.sampled_tokens = static_cast<int32_t*>(bo.dev);
        const hipError_t st = ps::kernel::launch_argmax_f32_single(args, nullptr);
        std::vector<int32_t> got(4);
        if (ok && st == hipSuccess)
            ok = hipMemcpy(got.data(), bo.dev, 16, hipMemcpyDeviceToHost) == hipSuccess;
        ok = ok && (st == hipSuccess);
        for (int i = 0; i < 4; ++i)
            if (got[i] != four[i]) ok = false;
        if (!ok) ++g_fail;
        std::printf("  [%s] | %s\n", "outputs-0", ok ? "PASS" : "FAIL");
    }

    std::printf("%s: %d failures\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
