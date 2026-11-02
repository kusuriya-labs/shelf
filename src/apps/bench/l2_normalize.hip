#include <shelf/kernels/optimized/l2_normalize.h>
#include <shelf/kernels/correctness/model_dispatch_correctness.h>
#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/types.h>
#include <shelf/runtime/program/device_program.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using V = ::ps::runtime::DeviceValueBinding;
using P = ::ps::runtime::DeviceProgramView;
using B = ::ps::runtime::DispatchBinding;
using DT = ::ps::runtime::ValueDType;

struct Options {
    std::string variant = "optimized";
    std::vector<uint32_t> rows = {1, 2, 4, 16, 128, 512, 2048};
    uint32_t features = 2048;
    uint32_t group_size = 128;
    float eps = 1e-6f;
    int warmup = 50;
    int samples = 100;
    int launches = 20;
    uint32_t seed = 1;
    int device = 0;
    bool check = false;
    std::string out;
};

void usage(FILE* f) {
    std::fprintf(f,
                 "usage: shelf-bench l2-normalize [options]\n"
                 "  --variant correctness|optimized\n"
                 "  --rows 1,2,...\n"
                 "  --features N\n"
                 "  --group-size N\n"
                 "  --eps FLOAT\n"
                 "  --warmup N\n"
                 "  --samples N\n"
                 "  --launches N\n"
                 "  --seed N\n"
                 "  --device N\n"
                 "  --check\n"
                 "  --out PATH\n");
}

bool parse_list(const std::string& s, std::vector<uint32_t>& out) {
    out.clear();
    size_t i = 0;
    while (i < s.size()) {
        size_t j = i;
        while (j < s.size() && std::isdigit((unsigned char)s[j])) ++j;
        if (j == i) return false;
        out.push_back((uint32_t)std::stoul(s.substr(i, j - i)));
        i = j;
        if (i < s.size() && s[i] == ',') ++i;
    }
    return !out.empty();
}

uint32_t h32(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return (uint32_t)x;
}

int32_t rng1k(uint64_t x) { return (int32_t)(h32(x) % 2001u) - 1000; }

uint16_t bf16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return (uint16_t)(u >> 16);
}

double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    if (v.empty()) return 0.0;
    const double idx = p * (double)(v.size() - 1);
    const size_t lo = (size_t)idx;
    const size_t hi = std::min((size_t)std::ceil(idx), (size_t)v.size() - 1);
    const double frac = idx - (double)lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

void set_val(V& v, uint32_t id, uint32_t feat, uint32_t stride, void* ptr, DT dt) {
    v.logical_value = id;
    v.storage = ::ps::runtime::ValueStorage::WORKSPACE;
    v.dtype = dt;
    v.row_domain = ::ps::runtime::ValueRowDomain::TOKEN_ROWS;
    v.slot = 0;
    v.feature_count = feat;
    v.row_stride = stride;
    v.offset = (uint64_t)(uintptr_t)ptr;
    v.bytes = 0;
}

struct Env {
    uint32_t rows = 0;
    uint32_t features = 0;
    uint32_t group = 0;
    void* in = nullptr;
    void* out = nullptr;
    void* ctx_dev = nullptr;
    void* scratch = nullptr;
    void* err = nullptr;
    void* vals_dev = nullptr;
    V vals[2] = {};
    P prog = {};
    B binding = {};
    ~Env() {
        void* bufs[] = {in, out, ctx_dev, scratch, err, vals_dev};
        for (auto* q : bufs)
            if (q) ps::gpu::discard_cleanup_result(hipFree(q));
    }
};

bool build(Env& e, uint32_t seed, float eps) {
    const size_t elems = (size_t)e.rows * e.features;
    if (hipMalloc(&e.in, elems * 2) != hipSuccess) return false;
    if (hipMalloc(&e.out, elems * 4) != hipSuccess) return false;
    if (hipMalloc(&e.scratch, (size_t)256 * 4096 * sizeof(float)) != hipSuccess) return false;
    if (hipMalloc(&e.err, sizeof(uint32_t)) != hipSuccess) return false;

    std::vector<uint16_t> h(elems);
    for (size_t i = 0; i < elems; ++i)
        h[i] = bf16_bits(rng1k(seed * 13u + 1u + i) / 1000.0f);
    if (hipMemcpy(e.in, h.data(), elems * 2, hipMemcpyHostToDevice) != hipSuccess) return false;

    ::ps::runtime::DeviceBatchContext ctx{};
    ctx.actual_rows = e.rows;
    ctx.num_requests = 0;
    if (hipMalloc(&e.ctx_dev, sizeof(ctx)) != hipSuccess) return false;
    if (hipMemcpy(e.ctx_dev, &ctx, sizeof(ctx), hipMemcpyHostToDevice) != hipSuccess) return false;

    set_val(e.vals[0], 0, e.features, e.features, e.in, DT::BF16);
    set_val(e.vals[1], 1, e.features, e.features, e.out, DT::F32);
    e.binding = {};
    e.binding.kernel_id = ::ps::runtime::KernelId::L2_NORMALIZE;
    e.binding.input_slots[0] = 0;
    e.binding.output_slots[0] = 1;
    e.binding.input_count = 1;
    e.binding.output_count = 1;
    e.binding.group_size = e.group;
    e.binding.scalar_a = eps;
    e.prog = {};
    e.prog.workspace_base = 0;
    if (hipMalloc(&e.vals_dev, 2 * sizeof(V)) != hipSuccess) return false;
    if (hipMemcpy(e.vals_dev, e.vals, 2 * sizeof(V), hipMemcpyHostToDevice) != hipSuccess)
        return false;
    e.prog.values = (const V*)e.vals_dev;
    e.prog.value_count = 2;
    return true;
}

hipError_t launch_env(const Env& e, bool optimized, hipStream_t s) {
    if (optimized)
        return ::ps::kernel::launch_l2_normalize_bf16_f32(
            (const ::ps::bf16_t*)e.in, e.features, (float*)e.out, e.features,
            e.rows, e.features, e.group, e.binding.scalar_a, s);
    ps::gpu::discard_cleanup_result(hipMemsetAsync(e.err, 0, sizeof(uint32_t), s));
    ::ps::kernel::ModelDispatchStateView sv{};
    auto st = ::ps::kernel::launch_model_dispatch_correctness(
        e.prog, e.binding, (::ps::runtime::DeviceBatchContext*)e.ctx_dev, sv,
        (float*)e.scratch, (uint32_t*)e.err, s);
    if (!st.ok()) return hipErrorUnknown;
    return hipGetLastError();
}

}  // namespace

int run_l2_normalize(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing arg for %s\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--help" || a == "-h") { usage(stdout); return 0; }
        else if (a == "--variant") o.variant = next();
        else if (a == "--rows") { if (!parse_list(next(), o.rows)) std::exit(2); }
        else if (a == "--features") o.features = (uint32_t)std::stoul(next());
        else if (a == "--group-size") o.group_size = (uint32_t)std::stoul(next());
        else if (a == "--eps") o.eps = std::strtof(next(), nullptr);
        else if (a == "--warmup") o.warmup = std::atoi(next());
        else if (a == "--samples") o.samples = std::atoi(next());
        else if (a == "--launches") o.launches = std::atoi(next());
        else if (a == "--seed") o.seed = (uint32_t)std::stoul(next());
        else if (a == "--device") o.device = std::atoi(next());
        else if (a == "--check") o.check = true;
        else if (a == "--out") o.out = next();
        else { usage(stderr); return 2; }
    }
    if (o.variant != "correctness" && o.variant != "optimized") { usage(stderr); return 2; }
    if (o.samples <= 0 || o.launches <= 0) return 2;
    if (hipSetDevice(o.device) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    if (!o.out.empty()) {
        FILE* f = std::fopen(o.out.c_str(), "a");
        if (f) {
            if (std::ftell(f) == 0)
                std::fprintf(f,
                             "variant,features,group,rows,warmup,samples,"
                             "launches_per_sample,min_us,p50_us,p95_us,mean_us\n");
            std::fclose(f);
        }
    }

    int rc = 0;
    for (uint32_t rows : o.rows) {
        Env e;
        e.rows = rows;
        e.features = o.features;
        e.group = o.group_size;
        if (!build(e, o.seed, o.eps)) { std::printf("build fail rows=%u\n", rows); rc = 1; continue; }
        if (hipDeviceSynchronize() != hipSuccess) { rc = 1; continue; }

        if (o.check) {
            const size_t n = (size_t)rows * o.features;
            std::vector<float> ref(n, 0.0f), got(n, 0.0f);
            if (launch_env(e, false, stream) != hipSuccess) { rc = 1; continue; }
            if (hipMemcpy(ref.data(), e.out, n * 4, hipMemcpyDeviceToHost) != hipSuccess) { rc = 1; continue; }
            if (launch_env(e, true, stream) != hipSuccess) { rc = 1; continue; }
            if (hipMemcpy(got.data(), e.out, n * 4, hipMemcpyDeviceToHost) != hipSuccess) { rc = 1; continue; }
            double max_abs = 0.0, max_rel = 0.0;
            bool bad = false;
            for (size_t i = 0; i < n; ++i) {
                const double g = got[i], r = ref[i];
                if (std::isnan(g) || std::isinf(g)) { bad = true; break; }
                const double da = std::fabs(g - r);
                const double dr = da / std::max(1.0, std::fabs(r));
                max_abs = std::max(max_abs, da);
                max_rel = std::max(max_rel, dr);
                if (da > 2e-4 && dr > 5e-4) { bad = true; break; }
            }
            std::printf("CHECK %s rows=%-5u max_abs=%.3g max_rel=%.3g %s\n",
                        o.variant.c_str(), rows, max_abs, max_rel, bad ? "FAIL" : "PASS");
            if (bad) rc = 1;
        }

        auto launch = [&](hipStream_t s) -> hipError_t {
            return launch_env(e, o.variant == "optimized", s);
        };
        for (int i = 0; i < o.warmup; ++i)
            if (launch(stream) != hipSuccess) rc = 1;
        if (hipStreamSynchronize(stream) != hipSuccess) { rc = 1; continue; }
        hipEvent_t ev0 = nullptr, ev1 = nullptr;
        if (hipEventCreate(&ev0) != hipSuccess || hipEventCreate(&ev1) != hipSuccess) { rc = 1; continue; }
        std::vector<double> per;
        per.reserve(o.samples);
        for (int s = 0; s < o.samples; ++s) {
            if (hipEventRecord(ev0, stream) != hipSuccess) break;
            for (int l = 0; l < o.launches; ++l)
                if (launch(stream) != hipSuccess) break;
            if (hipEventRecord(ev1, stream) != hipSuccess) break;
            if (hipEventSynchronize(ev1) != hipSuccess) break;
            float ms = 0.0f;
            if (hipEventElapsedTime(&ms, ev0, ev1) != hipSuccess) break;
            per.push_back(ms * 1000.0 / (double)o.launches);
        }
        ps::gpu::discard_cleanup_result(hipEventDestroy(ev0));
        ps::gpu::discard_cleanup_result(hipEventDestroy(ev1));
        if (per.empty()) { rc = 1; continue; }
        const double mn = percentile(per, 0.0);
        const double p50 = percentile(per, 0.5);
        const double p95 = percentile(per, 0.95);
        double mean = 0.0;
        for (double x : per) mean += x;
        mean /= (double)per.size();
        if (!o.out.empty()) {
            FILE* f = std::fopen(o.out.c_str(), "a");
            if (f) {
                std::fprintf(f, "%s,%u,%u,%u,%d,%d,%d,%.3f,%.3f,%.3f,%.3f\n",
                             o.variant.c_str(), e.features, e.group, rows, o.warmup, o.samples,
                             o.launches, mn, p50, p95, mean);
                std::fclose(f);
            }
        }
        std::printf("%-11s l2 f=%-5u g=%-4u rows=%-5u p50=%.2fus mean=%.2fus\n",
                    o.variant.c_str(), e.features, e.group, rows, p50, mean);
    }
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return rc;
}
