#include <shelf/kernels/optimized/attention/rope.h>
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

using bf16t = shelf::bf16_t;
using V = ::shelf::runtime::DeviceValueBinding;
using P = ::shelf::runtime::DeviceProgramView;
using B = ::shelf::runtime::DispatchBinding;

struct Options {
    std::string variant = "optimized";
    std::vector<uint32_t> rows = {1, 2, 4, 16, 128, 512, 2048};
    uint32_t features = 6144;
    uint32_t head_dim = 256;
    uint32_t rotary = 64;
    float theta = 1.0e7f;
    std::string pos_mode = "row";
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
                 "usage: shelf-bench rope [options]\n"
                 "  --variant correctness|optimized|optimized-pair\n"
                 "    (optimized = generic optimized, optimized-pair = head_dim256/rotary64 specialized)\n"
                 "  --features N --head-dim N --rotary N --theta F\n"
                 "  --rows 1,2,...\n"
                 "  --pos-mode none|row|rope\n"
                 "  --warmup N --samples N --launches N\n"
                 "  --seed N --device N --check --out PATH\n");
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

float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
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

struct Setup {
    void* in = nullptr;
    void* out_c = nullptr;
    void* out_o = nullptr;
    void* pos_row = nullptr;
    void* pos_rope = nullptr;
    void* ctx = nullptr;
    void* vals = nullptr;
    void* scratch = nullptr;
    void* err = nullptr;
    void* inv_freq = nullptr;
    const uint32_t* positions = nullptr;
    uint32_t rows = 0;
    uint32_t features = 0;
    ~Setup() {
        void* all[] = {in, out_c, out_o, pos_row, pos_rope, ctx, vals, scratch, err,
                       inv_freq};
        for (auto* p : all)
            if (p) shelf::gpu::discard_cleanup_result(hipFree(p));
    }
};

bool build(Setup& s, const Options& o, uint32_t rows, B& binding) {
    s.rows = rows;
    s.features = o.features;
    const uint32_t in_stride = o.features;
    const uint32_t out_stride = o.features;
    auto alloc = [&](void** dst, size_t bytes) { return hipMalloc(dst, bytes) == hipSuccess; };
    if (!alloc(&s.in, (size_t)rows * in_stride * 4)) return false;
    if (!alloc(&s.out_c, (size_t)rows * out_stride * 2)) return false;
    if (!alloc(&s.out_o, (size_t)rows * out_stride * 2)) return false;
    if (!alloc(&s.ctx, sizeof(::shelf::runtime::DeviceBatchContext))) return false;
    if (!alloc(&s.vals, 2 * sizeof(V))) return false;
    if (!alloc(&s.scratch, (size_t)256 * 4096 * sizeof(float))) return false;
    if (!alloc(&s.err, sizeof(uint32_t))) return false;
    if (!alloc(&s.inv_freq, 32 * sizeof(double))) return false;
    if (::shelf::kernel::launch_rope_inv_freq_init((double*)s.inv_freq, 32, o.theta, nullptr)
        != hipSuccess)
        return false;

    std::vector<float> h_in((size_t)rows * in_stride);
    for (size_t i = 0; i < h_in.size(); ++i) h_in[i] = rng1k(o.seed * 13u + i) / 1000.0f;
    if (hipMemcpy(s.in, h_in.data(), h_in.size() * 4, hipMemcpyHostToDevice) != hipSuccess) return false;
    if (hipMemset(s.out_c, 0, (size_t)rows * out_stride * 2) != hipSuccess) return false;
    if (hipMemset(s.out_o, 0, (size_t)rows * out_stride * 2) != hipSuccess) return false;

    std::vector<uint32_t> h_pos(rows);
    for (uint32_t r = 0; r < rows; ++r) h_pos[r] = (r * 37u + 5u) % 4096u;
    if (o.pos_mode != "none") {
        if (!alloc(&s.pos_row, rows * 4)) return false;
        if (hipMemcpy(s.pos_row, h_pos.data(), rows * 4, hipMemcpyHostToDevice) != hipSuccess) return false;
        s.positions = (const uint32_t*)s.pos_row;
    }
    if (o.pos_mode == "rope") {
        if (!alloc(&s.pos_rope, rows * 4)) return false;
        if (hipMemcpy(s.pos_rope, h_pos.data(), rows * 4, hipMemcpyHostToDevice) != hipSuccess) return false;
        s.positions = (const uint32_t*)s.pos_rope;
    }

    ::shelf::runtime::DeviceBatchContext ctx{};
    ctx.actual_rows = rows;
    ctx.num_requests = 0;
    ctx.rope_positions = (o.pos_mode == "rope") ? (const uint32_t*)s.pos_rope : nullptr;
    ctx.row_positions = (o.pos_mode == "row") ? (uint32_t*)s.pos_row : nullptr;
    if (hipMemcpy(s.ctx, &ctx, sizeof(ctx), hipMemcpyHostToDevice) != hipSuccess) return false;

    V vals[2];
    auto setv = [&](V& v, uint32_t id, uint32_t feat, uint32_t stride, void* ptr,
                    ::shelf::runtime::ValueDType dt) {
        v.logical_value = id;
        v.storage = ::shelf::runtime::ValueStorage::WORKSPACE;
        v.dtype = dt;
        v.row_domain = ::shelf::runtime::ValueRowDomain::TOKEN_ROWS;
        v.slot = 0;
        v.feature_count = feat;
        v.row_stride = stride;
        v.offset = (uint64_t)(uintptr_t)ptr;
        v.bytes = 0;
    };
    setv(vals[0], 0, o.features, in_stride, s.in, ::shelf::runtime::ValueDType::F32);
    setv(vals[1], 1, o.features, out_stride, s.out_c, ::shelf::runtime::ValueDType::BF16);
    if (hipMemcpy(s.vals, vals, 2 * sizeof(V), hipMemcpyHostToDevice) != hipSuccess) return false;

    binding = {};
    binding.kernel_id = ::shelf::runtime::KernelId::ROPE;
    binding.input_slots[0] = 0;
    binding.output_slots[0] = 1;
    binding.input_count = 1;
    binding.output_count = 1;
    binding.group_size = o.head_dim;
    binding.rotary_dim = o.rotary;
    binding.scalar_b = o.theta;
    return true;
}

hipError_t launch_correctness(const Setup& s, const B& binding, hipStream_t stream) {
    P prog{};
    prog.values = (const V*)s.vals;
    prog.value_count = 2;
    prog.workspace_base = 0;
    shelf::gpu::discard_cleanup_result(hipMemsetAsync(s.err, 0, sizeof(uint32_t), stream));
    ::shelf::kernel::ModelDispatchStateView sv{};
    auto st = ::shelf::kernel::launch_model_dispatch_correctness(
        prog, binding, (::shelf::runtime::DeviceBatchContext*)s.ctx, sv, (float*)s.scratch,
        (uint32_t*)s.err, stream);
    if (!st.ok()) return hipErrorUnknown;
    return hipGetLastError();
}

hipError_t launch_optimized(const Setup& s, const Options& o, hipStream_t stream) {
    if (o.variant == "optimized-pair") {
        return ::shelf::kernel::launch_rope_f32_bf16_pair(
            (const float*)s.in, s.features, (bf16t*)s.out_o, s.features, s.positions,
            s.rows, o.features, o.head_dim, o.rotary, o.theta,
            (const double*)s.inv_freq, 1.0f, stream);
    }
    return ::shelf::kernel::launch_rope_f32_bf16_generic(
        (const float*)s.in, s.features, (bf16t*)s.out_o, s.features, s.positions,
        s.rows, o.features, o.head_dim, o.rotary, o.theta,
        (const double*)s.inv_freq, 1.0f, stream);
}

}  // namespace

int run_rope(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing arg for %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--help" || a == "-h") { usage(stdout); return 0; }
        else if (a == "--variant") o.variant = next();
        else if (a == "--features") o.features = (uint32_t)std::stoul(next());
        else if (a == "--head-dim") o.head_dim = (uint32_t)std::stoul(next());
        else if (a == "--rotary") o.rotary = (uint32_t)std::stoul(next());
        else if (a == "--theta") o.theta = std::strtof(next(), nullptr);
        else if (a == "--rows") { if (!parse_list(next(), o.rows)) std::exit(2); }
        else if (a == "--pos-mode") o.pos_mode = next();
        else if (a == "--warmup") o.warmup = std::atoi(next());
        else if (a == "--samples") o.samples = std::atoi(next());
        else if (a == "--launches") o.launches = std::atoi(next());
        else if (a == "--seed") o.seed = (uint32_t)std::stoul(next());
        else if (a == "--device") o.device = std::atoi(next());
        else if (a == "--check") o.check = true;
        else if (a == "--out") o.out = next();
        else { usage(stderr); return 2; }
    }
    if (o.variant != "correctness" && o.variant != "optimized" &&
        o.variant != "optimized-pair") { usage(stderr); return 2; }
    if (o.samples <= 0 || o.launches <= 0) return 2;
    if (o.pos_mode != "none" && o.pos_mode != "row" && o.pos_mode != "rope") { usage(stderr); return 2; }
    if (hipSetDevice(o.device) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    if (!o.out.empty()) {
        FILE* f = std::fopen(o.out.c_str(), "a");
        if (f) {
            if (std::ftell(f) == 0)
                std::fprintf(f, "variant,features,head_dim,rotary,rows,warmup,samples,"
                                "launches_per_sample,min_us,p10_us,p50_us,p90_us,p95_us,mean_us\n");
            std::fclose(f);
        }
    }

    int rc = 0;
    for (uint32_t rows : o.rows) {
        Setup s;
        B binding{};
        if (!build(s, o, rows, binding)) { std::printf("build fail rows=%u\n", rows); rc = 1; continue; }
        if (hipDeviceSynchronize() != hipSuccess) { rc = 1; continue; }

        if (o.check) {
            if (launch_correctness(s, binding, stream) != hipSuccess) { rc = 1; continue; }
            if (launch_optimized(s, o, stream) != hipSuccess) { rc = 1; continue; }
            if (hipStreamSynchronize(stream) != hipSuccess) { rc = 1; continue; }
            std::vector<uint16_t> rc16((size_t)rows * o.features), ro16((size_t)rows * o.features);
            if (hipMemcpy(rc16.data(), s.out_c, rc16.size() * 2, hipMemcpyDeviceToHost) != hipSuccess) { rc = 1; continue; }
            if (hipMemcpy(ro16.data(), s.out_o, ro16.size() * 2, hipMemcpyDeviceToHost) != hipSuccess) { rc = 1; continue; }
            size_t mismatch = 0, bad = 0;
            double max_abs = 0.0, max_rel = 0.0;
            for (size_t i = 0; i < rc16.size(); ++i) {
                const double ref = (double)bf16_to_f32(rc16[i]);
                const double got = (double)bf16_to_f32(ro16[i]);
                const double diff = std::fabs(got - ref);
                const double rel = diff / std::fmax(1.0, std::fabs(ref));
                max_abs = std::max(max_abs, diff);
                max_rel = std::max(max_rel, rel);
                if (rc16[i] != ro16[i]) ++mismatch;
                if (rel > 2e-2) ++bad;
            }
            std::printf("CHECK %s rows=%-5u mismatch=%zu max_abs=%.3g max_rel=%.3g %s\n",
                        o.variant.c_str(), rows, mismatch, max_abs, max_rel,
                        bad == 0 ? "PASS" : "FAIL");
            if (bad != 0) rc = 1;
        }

        auto launch = [&](hipStream_t st) -> hipError_t {
            if (o.variant == "correctness") return launch_correctness(s, binding, st);
            return launch_optimized(s, o, st);
        };
        for (int i = 0; i < o.warmup; ++i)
            if (launch(stream) != hipSuccess) rc = 1;
        if (hipStreamSynchronize(stream) != hipSuccess) { rc = 1; continue; }
        hipEvent_t ev0 = nullptr, ev1 = nullptr;
        if (hipEventCreate(&ev0) != hipSuccess || hipEventCreate(&ev1) != hipSuccess) { rc = 1; continue; }
        std::vector<double> per;
        per.reserve(o.samples);
        for (int sm = 0; sm < o.samples; ++sm) {
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
        const double p10 = percentile(per, 0.1);
        const double p50 = percentile(per, 0.5);
        const double p90 = percentile(per, 0.9);
        const double p95 = percentile(per, 0.95);
        double mean = 0.0;
        for (double x : per) mean += x;
        mean /= (double)per.size();
        if (!o.out.empty()) {
            FILE* f = std::fopen(o.out.c_str(), "a");
            if (f) {
                std::fprintf(f, "%s,%u,%u,%u,%u,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                             o.variant.c_str(), o.features, o.head_dim, o.rotary, rows, o.warmup,
                             o.samples, o.launches, mn, p10, p50, p90, p95, mean);
                std::fclose(f);
            }
        }
        std::printf("%-15s rope f=%-5u hd=%u rot=%u rows=%-5u p10=%.2fus p50=%.2fus p90=%.2fus mean=%.2fus\n",
                    o.variant.c_str(), o.features, o.head_dim, o.rotary, rows, p10, p50, p90, mean);
    }
    shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return rc;
}
