#include <shelf/kernels/optimized/elementwise.h>
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

using V = ::shelf::runtime::DeviceValueBinding;
using P = ::shelf::runtime::DeviceProgramView;
using B = ::shelf::runtime::DispatchBinding;
using DT = ::shelf::runtime::ValueDType;

struct Options {
    std::string variant = "optimized";
    std::string profile = "residual";
    std::vector<uint32_t> rows = {1, 4, 16, 64, 256, 1024, 2048};
    uint32_t features = 2560;
    uint32_t aux = 0;
    int warmup = 20;
    int samples = 30;
    int launches = 10;
    uint32_t seed = 1;
    int device = 0;
    std::string out;
};

void usage(FILE* f) {
    std::fprintf(f,
                 "usage: shelf-bench elementwise [options]\n"
                 "  --variant correctness|optimized\n"
                 "  --profile residual|swiglu|sigmoid|mul|scale|silu-b2f|silu-f2b|split-halves|split-int\n"
                 "  --features N\n"
                 "  --aux N\n"
                 "  --rows 1,4,...\n"
                 "  --warmup N\n"
                 "  --samples N\n"
                 "  --launches N\n"
                 "  --seed N\n"
                 "  --device N\n"
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
    v.storage = ::shelf::runtime::ValueStorage::WORKSPACE;
    v.dtype = dt;
    v.row_domain = ::shelf::runtime::ValueRowDomain::TOKEN_ROWS;
    v.slot = 0;
    v.feature_count = feat;
    v.row_stride = stride;
    v.offset = (uint64_t)(uintptr_t)ptr;
    v.bytes = 0;
}

struct Env {
    std::string profile;
    uint32_t rows = 0;
    uint32_t features = 0;
    uint32_t aux = 0;
    void* a = nullptr;
    void* b = nullptr;
    void* o0 = nullptr;
    void* o1 = nullptr;
    void* sig = nullptr;
    void* vals_dev[2] = {nullptr, nullptr};
    void* ctx_dev = nullptr;
    void* scratch = nullptr;
    void* err = nullptr;
    P prog[2] = {};
    B binding[2] = {};
    uint32_t n_corr = 0;
    ~Env() {
        void* bufs[] = {a, b, o0, o1, sig};
        for (auto* q : bufs)
            if (q) shelf::gpu::discard_cleanup_result(hipFree(q));
        for (auto* q : vals_dev)
            if (q) shelf::gpu::discard_cleanup_result(hipFree(q));
        void* misc[] = {ctx_dev, scratch, err};
        for (auto* q : misc)
            if (q) shelf::gpu::discard_cleanup_result(hipFree(q));
    }
};

bool build(Env& e, uint32_t seed) {
    const bool split = (e.profile == "split-halves" || e.profile == "split-int");
    const uint32_t f = e.features;
    const uint32_t f_in = split ? 2u * f : f;
    const size_t in_elems = (size_t)e.rows * f_in;
    const size_t out_elems = (size_t)e.rows * f;
    if (hipMalloc(&e.a, in_elems * 4) != hipSuccess) return false;
    if (hipMalloc(&e.b, in_elems * 4) != hipSuccess) return false;
    if (hipMalloc(&e.o0, out_elems * 4) != hipSuccess) return false;
    if (hipMalloc(&e.o1, out_elems * 4) != hipSuccess) return false;
    if (hipMalloc(&e.sig, out_elems * 4) != hipSuccess) return false;
    if (hipMalloc(&e.scratch, (size_t)256 * 4096 * sizeof(float)) != hipSuccess) return false;
    if (hipMalloc(&e.err, sizeof(uint32_t)) != hipSuccess) return false;

    auto fill_f = [&](void* p, size_t n, uint32_t tag) {
        std::vector<float> v(n);
        for (size_t i = 0; i < n; ++i) v[i] = rng1k(seed * 13u + tag * 100000u + i) / 1000.0f;
        return hipMemcpy(p, v.data(), n * 4, hipMemcpyHostToDevice) == hipSuccess;
    };
    auto fill_bf16 = [&](void* p, size_t n, uint32_t tag) {
        std::vector<uint16_t> v(n);
        for (size_t i = 0; i < n; ++i) v[i] = bf16_bits(rng1k(seed * 13u + tag * 100000u + i) / 1000.0f);
        return hipMemcpy(p, v.data(), n * 2, hipMemcpyHostToDevice) == hipSuccess;
    };
    const bool f32_a = (e.profile == "silu-f2b" || e.profile == "mul" || e.profile == "scale");
    if (!(f32_a ? fill_f(e.a, in_elems, 1) : fill_bf16(e.a, in_elems, 1))) return false;
    if (e.profile == "residual" || e.profile == "swiglu") {
        if (!fill_bf16(e.b, in_elems, 2)) return false;
    } else if (e.profile == "mul") {
        if (!fill_f(e.b, in_elems, 2)) return false;
    }

    ::shelf::runtime::DeviceBatchContext ctx{};
    ctx.actual_rows = e.rows;
    ctx.num_requests = 0;
    if (hipMalloc(&e.ctx_dev, sizeof(ctx)) != hipSuccess) return false;
    if (hipMemcpy(e.ctx_dev, &ctx, sizeof(ctx), hipMemcpyHostToDevice) != hipSuccess) return false;

    V vals[4];
    e.n_corr = 1;
    if (split) {
        set_val(vals[0], 0, f_in, f_in, e.a, DT::BF16);
        set_val(vals[1], 1, f, f, e.o0, DT::BF16);
        set_val(vals[2], 2, f, f, e.o1, DT::BF16);
        B& binding = e.binding[0];
        binding.kernel_id = ::shelf::runtime::KernelId::SPLIT;
        binding.input_slots[0] = 0;
        binding.output_slots[0] = 1;
        binding.output_slots[1] = 2;
        binding.input_count = 1;
        binding.output_count = 2;
        binding.flags = (e.profile == "split-int")
            ? static_cast<uint32_t>(::shelf::runtime::SplitLayout::INTERLEAVED_HEADS) : 0u;
        binding.group_size = e.aux;
    } else if (e.profile == "residual" || e.profile == "swiglu" || e.profile == "mul") {
        DT a_dt = f32_a ? DT::F32 : DT::BF16;
        DT b_dt = f32_a ? DT::F32 : DT::BF16;
        DT o_dt = (e.profile == "mul") ? DT::BF16 : DT::BF16;
        set_val(vals[0], 0, f, f, e.a, a_dt);
        set_val(vals[1], 1, f, f, e.b, b_dt);
        set_val(vals[2], 2, f, f, e.o0, o_dt);
        B& binding = e.binding[0];
        binding.input_slots[0] = 0;
        binding.input_slots[1] = 1;
        binding.output_slots[0] = 2;
        binding.input_count = 2;
        binding.output_count = 1;
        binding.kernel_id = (e.profile == "residual") ? ::shelf::runtime::KernelId::RESIDUAL_ADD
                            : (e.profile == "swiglu") ? ::shelf::runtime::KernelId::SWIGLU
                                                      : ::shelf::runtime::KernelId::MUL;
    } else {
        DT a_dt = f32_a ? DT::F32 : DT::BF16;
        DT o_dt = (e.profile == "sigmoid" || e.profile == "scale" ||
                   e.profile == "silu-b2f")
                      ? DT::F32
                      : DT::BF16;
        set_val(vals[0], 0, f, f, e.a, a_dt);
        set_val(vals[1], 1, f, f, e.o0, o_dt);
        B& binding = e.binding[0];
        binding.input_slots[0] = 0;
        binding.output_slots[0] = 1;
        binding.input_count = 1;
        binding.output_count = 1;
        if (e.profile == "sigmoid") binding.kernel_id = ::shelf::runtime::KernelId::SIGMOID;
        else if (e.profile == "silu-b2f" || e.profile == "silu-f2b")
            binding.kernel_id = ::shelf::runtime::KernelId::SILU;
        else {
            binding.kernel_id = ::shelf::runtime::KernelId::SCALE;
            binding.scalar_a = 0.5f;
        }
    }
    const uint32_t vcount = (split || e.profile == "residual" || e.profile == "swiglu" ||
                             e.profile == "mul") ? 3u
                                                : 2u;
    for (int i = 0; i < (int)e.n_corr; ++i) {
        if (hipMalloc(&e.vals_dev[i], vcount * sizeof(V)) != hipSuccess) return false;
        std::memcpy(e.vals_dev[i], vals, vcount * sizeof(V));
        e.prog[i] = {};
        e.prog[i].values = (const V*)e.vals_dev[i];
        e.prog[i].value_count = vcount;
        e.prog[i].workspace_base = 0;
    }
    return true;
}

hipError_t launch_correctness(const Env& e, hipStream_t s) {
    shelf::gpu::discard_cleanup_result(hipMemsetAsync(e.err, 0, sizeof(uint32_t), s));
    ::shelf::kernel::ModelDispatchStateView sv{};
    for (uint32_t i = 0; i < e.n_corr; ++i) {
        auto st = shelf::kernel::launch_model_dispatch_correctness(
            e.prog[i], e.binding[i], (::shelf::runtime::DeviceBatchContext*)e.ctx_dev, sv,
            (float*)e.scratch, (uint32_t*)e.err, s);
        if (!st.ok()) return hipErrorUnknown;
    }
    return hipGetLastError();
}

hipError_t launch_optimized(const Env& e, hipStream_t s) {
    const uint32_t f = e.features;
    const uint32_t rows = e.rows;
    if (e.profile == "residual")
        return ::shelf::kernel::launch_residual_add_bf16((const ::shelf::bf16_t*)e.a, f,
            (const ::shelf::bf16_t*)e.b, f, (::shelf::bf16_t*)e.o0, f, rows, f, s);
    if (e.profile == "swiglu")
        return ::shelf::kernel::launch_swiglu_bf16((const ::shelf::bf16_t*)e.a, f,
            (const ::shelf::bf16_t*)e.b, f, (::shelf::bf16_t*)e.o0, f, rows, f, s);
    if (e.profile == "sigmoid")
        return ::shelf::kernel::launch_sigmoid_bf16_f32((const ::shelf::bf16_t*)e.a, f,
            (float*)e.o0, f, rows, f, s);
    if (e.profile == "silu-b2f")
        return ::shelf::kernel::launch_silu_bf16_f32((const ::shelf::bf16_t*)e.a, f,
            (float*)e.o0, f, rows, f, s);
    if (e.profile == "silu-f2b")
        return ::shelf::kernel::launch_silu_f32_bf16((const float*)e.a, f,
            (::shelf::bf16_t*)e.o0, f, rows, f, s);
    if (e.profile == "mul")
        return ::shelf::kernel::launch_mul_f32_f32_bf16((const float*)e.a, f,
            (const float*)e.b, f, (::shelf::bf16_t*)e.o0, f, rows, f, s);
    if (e.profile == "scale")
        return ::shelf::kernel::launch_scale_f32((const float*)e.a, f, (float*)e.o0, f,
            rows, f, 0.5f, s);
    if (e.profile == "split-halves")
        return ::shelf::kernel::launch_split_bf16((const ::shelf::bf16_t*)e.a, 2u * f,
            (::shelf::bf16_t*)e.o0, f, (::shelf::bf16_t*)e.o1, f, rows, f,
            ::shelf::kernel::ElementwiseSplitLayout::Halves, 0u, s);
    if (e.profile == "split-int")
        return ::shelf::kernel::launch_split_bf16((const ::shelf::bf16_t*)e.a, 2u * f,
            (::shelf::bf16_t*)e.o0, f, (::shelf::bf16_t*)e.o1, f, rows, f,
            ::shelf::kernel::ElementwiseSplitLayout::InterleavedHeads, e.aux, s);
    return hipErrorInvalidValue;
}

}  // namespace

int run_elementwise(int argc, char** argv) {
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
        else if (a == "--profile") o.profile = next();
        else if (a == "--rows") { if (!parse_list(next(), o.rows)) std::exit(2); }
        else if (a == "--features") o.features = (uint32_t)std::stoul(next());
        else if (a == "--aux") o.aux = (uint32_t)std::stoul(next());
        else if (a == "--warmup") o.warmup = std::atoi(next());
        else if (a == "--samples") o.samples = std::atoi(next());
        else if (a == "--launches") o.launches = std::atoi(next());
        else if (a == "--seed") o.seed = (uint32_t)std::stoul(next());
        else if (a == "--device") o.device = std::atoi(next());
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
                             "variant,profile,features,aux,rows,warmup,samples,"
                             "launches_per_sample,min_us,p50_us,p95_us,mean_us\n");
            std::fclose(f);
        }
    }

    int rc = 0;
    for (uint32_t rows : o.rows) {
        Env e;
        e.profile = o.profile;
        e.rows = rows;
        e.features = o.features;
        e.aux = o.aux;
        if (!build(e, o.seed)) { std::printf("build fail rows=%u\n", rows); rc = 1; continue; }
        if (hipDeviceSynchronize() != hipSuccess) { rc = 1; continue; }
        auto launch = [&](hipStream_t s) -> hipError_t {
            if (o.variant == "correctness") return launch_correctness(e, s);
            return launch_optimized(e, s);
        };
        for (int i = 0; i < o.warmup; ++i)
            if (launch(stream) != hipSuccess) rc = 1;
        if (hipStreamSynchronize(stream) != hipSuccess) { rc = 1; continue; }
        hipEvent_t ev0 = nullptr, ev1 = nullptr;
        if (hipEventCreate(&ev0) != hipSuccess || hipEventCreate(&ev1) != hipSuccess) {
            rc = 1;
            continue;
        }
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
        shelf::gpu::discard_cleanup_result(hipEventDestroy(ev0));
        shelf::gpu::discard_cleanup_result(hipEventDestroy(ev1));
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
                std::fprintf(f, "%s,%s,%u,%u,%u,%d,%d,%d,%.3f,%.3f,%.3f,%.3f\n", o.variant.c_str(),
                             o.profile.c_str(), e.features, e.aux, rows, o.warmup, o.samples,
                             o.launches, mn, p50, p95, mean);
                std::fclose(f);
            }
        }
        std::printf("%-11s %-12s f=%-5u aux=%-4u rows=%-5u p50=%.2fus mean=%.2fus\n",
                    o.variant.c_str(), o.profile.c_str(), e.features, e.aux, rows, p50, mean);
    }
    shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return rc;
}
