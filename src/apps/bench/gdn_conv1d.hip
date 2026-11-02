#include <shelf/kernels/optimized/gdn/conv1d.h>
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
using S = ::shelf::runtime::DeviceStateBinding;
using P = ::shelf::runtime::DeviceProgramView;
using B = ::shelf::runtime::DispatchBinding;

struct Options {
    std::string variant = "optimized";
    uint32_t conv_dim = 10240;
    std::vector<uint32_t> rows = {1, 2, 4, 16, 64, 128, 512, 2048};
    uint32_t requests = 1;
    uint32_t state_index = 0;
    uint32_t num_gdn_states = 1;
    uint32_t max_sequences = 8;
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
                 "usage: shelf-bench gdn-conv1d [options]\n"
                 "  --variant correctness|optimized\n"
                 "  --conv-dim N\n"
                 "  --rows 1,2,...\n"
                 "  --requests N\n"
                 "  --state-index N\n"
                 "  --num-gdn-states N\n"
                 "  --max-sequences N\n"
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

struct Bufs {
    void* weight = nullptr;
    void* input = nullptr;
    void* out_correct = nullptr;
    void* out_opt = nullptr;
    void* state_correct = nullptr;
    void* state_opt = nullptr;
    void* requests = nullptr;
    void* vals_dev = nullptr;
    void* state_dev = nullptr;
    void* param_dev = nullptr;
    void* ctx_dev = nullptr;
    void* scratch = nullptr;
    void* error_word = nullptr;
    ~Bufs() {
        void* all[] = {weight, input, out_correct, out_opt, state_correct, state_opt, requests,
                       vals_dev, state_dev, param_dev, ctx_dev, scratch, error_word};
        for (auto* p : all)
            if (p) shelf::gpu::discard_cleanup_result(hipFree(p));
    }
};

struct Setup {
    Bufs buf;
    uint32_t conv_dim = 0;
    uint32_t rows = 0;
    uint32_t nreq = 0;
    uint32_t in_stride = 0;
    uint32_t out_stride = 0;
    uint64_t hist_stride = 0;
    uint64_t layer_stride = 0;
    uint64_t slot_stride = 0;
    uint64_t state_total = 0;
};

bool build(Setup& s, const Options& o, uint32_t rows) {
    const uint32_t conv_dim = o.conv_dim;
    const uint32_t history = 3u;
    const uint32_t hist_stride = (conv_dim + 7u) & ~7u;
    s.conv_dim = conv_dim;
    s.rows = rows;
    s.nreq = std::min(o.requests, rows);
    if (s.nreq == 0u) s.nreq = 1u;
    s.in_stride = conv_dim;
    s.out_stride = conv_dim;
    s.hist_stride = hist_stride;
    s.layer_stride = (uint64_t)history * hist_stride;
    s.slot_stride = (uint64_t)o.num_gdn_states * s.layer_stride;
    s.state_total = (uint64_t)o.max_sequences * s.slot_stride;

    auto m = [&](void** dst, size_t bytes) { return hipMalloc(dst, bytes) == hipSuccess; };
    if (!m(&s.buf.weight, (size_t)conv_dim * 4u * 2)) return false;
    if (!m(&s.buf.input, (size_t)rows * s.in_stride * 2)) return false;
    if (!m(&s.buf.out_correct, (size_t)rows * s.out_stride * 4)) return false;
    if (!m(&s.buf.out_opt, (size_t)rows * s.out_stride * 4)) return false;
    if (!m(&s.buf.state_correct, s.state_total * 2)) return false;
    if (!m(&s.buf.state_opt, s.state_total * 2)) return false;
    if (!m(&s.buf.requests, (size_t)s.nreq * sizeof(::shelf::runtime::DeviceRequestDescriptor))) return false;

    std::vector<uint16_t> h_w((size_t)conv_dim * 4u);
    std::vector<uint16_t> h_in((size_t)rows * s.in_stride);
    std::vector<uint16_t> h_state(s.state_total);
    for (size_t i = 0; i < h_w.size(); ++i) h_w[i] = bf16_bits(rng1k(o.seed * 3u + i) / 1000.0f);
    for (size_t i = 0; i < h_in.size(); ++i) h_in[i] = bf16_bits(rng1k(o.seed * 7u + i) / 1000.0f);
    for (size_t i = 0; i < h_state.size(); ++i) h_state[i] = bf16_bits(rng1k(o.seed * 11u + i) / 512.0f);

    if (hipMemcpy(s.buf.weight, h_w.data(), h_w.size() * 2, hipMemcpyHostToDevice) != hipSuccess) return false;
    if (hipMemcpy(s.buf.input, h_in.data(), h_in.size() * 2, hipMemcpyHostToDevice) != hipSuccess) return false;
    if (hipMemcpy(s.buf.state_correct, h_state.data(), h_state.size() * 2, hipMemcpyHostToDevice) != hipSuccess) return false;
    if (hipMemcpy(s.buf.state_opt, h_state.data(), h_state.size() * 2, hipMemcpyHostToDevice) != hipSuccess) return false;
    if (hipMemset(s.buf.out_correct, 0, (size_t)rows * s.out_stride * 4) != hipSuccess) return false;
    if (hipMemset(s.buf.out_opt, 0, (size_t)rows * s.out_stride * 4) != hipSuccess) return false;

    std::vector<::shelf::runtime::DeviceRequestDescriptor> h_req(s.nreq);
    uint32_t begin = 0;
    for (uint32_t r = 0; r < s.nreq; ++r) {
        const uint32_t remaining = rows - begin;
        const uint32_t count = (r + 1u == s.nreq) ? remaining : (rows / s.nreq);
        std::memset(&h_req[r], 0, sizeof(h_req[r]));
        h_req[r].request_handle.slot = r % o.max_sequences;
        h_req[r].row_begin = begin;
        h_req[r].row_count = count;
        begin += count;
    }
    if (hipMemcpy(s.buf.requests, h_req.data(), h_req.size() * sizeof(h_req[0]), hipMemcpyHostToDevice) != hipSuccess) return false;

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
    setv(vals[0], 0, conv_dim, s.in_stride, s.buf.input, ::shelf::runtime::ValueDType::BF16);
    setv(vals[1], 1, conv_dim, s.out_stride, s.buf.out_correct, ::shelf::runtime::ValueDType::F32);
    if (hipMalloc(&s.buf.vals_dev, 2 * sizeof(V)) != hipSuccess) return false;
    std::memcpy(s.buf.vals_dev, vals, 2 * sizeof(V));

    S st;
    st.slot = 0;
    st.kind = (uint8_t)::shelf::runtime::PrimitiveStateKind::GDN_CONV_STATE;
    st.state_index = o.state_index;
    if (hipMalloc(&s.buf.state_dev, sizeof(S)) != hipSuccess) return false;
    std::memcpy(s.buf.state_dev, &st, sizeof(S));

    ::shelf::runtime::StaticParameterSlot param;
    param.device_ptr = s.buf.weight;
    param.elements = conv_dim * 4u;
    param.dtype = ::shelf::runtime::ValueDType::BF16;
    if (hipMalloc(&s.buf.param_dev, sizeof(param)) != hipSuccess) return false;
    std::memcpy(s.buf.param_dev, &param, sizeof(param));

    ::shelf::runtime::DeviceBatchContext ctx{};
    ctx.actual_rows = rows;
    ctx.num_requests = s.nreq;
    ctx.requests = (const ::shelf::runtime::DeviceRequestDescriptor*)s.buf.requests;
    if (hipMalloc(&s.buf.ctx_dev, sizeof(ctx)) != hipSuccess) return false;
    if (hipMemcpy(s.buf.ctx_dev, &ctx, sizeof(ctx), hipMemcpyHostToDevice) != hipSuccess) return false;
    if (hipMalloc(&s.buf.scratch, (size_t)1024 * 4096 * sizeof(float)) != hipSuccess) return false;
    if (hipMalloc(&s.buf.error_word, sizeof(uint32_t)) != hipSuccess) return false;
    return true;
}

hipError_t launch_correctness(const Setup& s, const Options& o, hipStream_t stream) {
    P prog{};
    prog.values = (const V*)s.buf.vals_dev;
    prog.value_count = 2;
    prog.states = (const S*)s.buf.state_dev;
    prog.state_count = 1;
    prog.workspace_base = 0;
    prog.parameter_table = (const ::shelf::runtime::StaticParameterSlot*)s.buf.param_dev;
    prog.parameter_slot_count = 1;

    B binding{};
    binding.kernel_id = ::shelf::runtime::KernelId::STATEFUL_CAUSAL_CONV1D;
    binding.descriptor_slot = 1;
    binding.input_slots[0] = 0;
    binding.output_slots[0] = 1;
    binding.input_count = 1;
    binding.output_count = 1;
    binding.parameter_index0 = 0;

    ::shelf::kernel::ModelDispatchStateView state_view{};
    state_view.gdn.conv_base = (bf16t*)s.buf.state_correct;
    state_view.gdn.num_gdn_states = o.num_gdn_states;
    state_view.gdn.max_sequences = o.max_sequences;
    state_view.gdn.conv_slot_stride = s.slot_stride;
    state_view.gdn.conv_layer_stride = s.layer_stride;
    state_view.gdn.conv_history_stride = s.hist_stride;
    state_view.gdn.conv_dim = s.conv_dim;
    state_view.gdn.conv_history = 3u;

    shelf::gpu::discard_cleanup_result(hipMemsetAsync(s.buf.error_word, 0, sizeof(uint32_t), stream));
    auto st = shelf::kernel::launch_model_dispatch_correctness(
        prog, binding, (::shelf::runtime::DeviceBatchContext*)s.buf.ctx_dev, state_view,
        (float*)s.buf.scratch, (uint32_t*)s.buf.error_word, stream);
    if (!st.ok()) return hipErrorUnknown;
    return hipGetLastError();
}

hipError_t launch_optimized(const Setup& s, const Options& o, hipStream_t stream) {
    ::shelf::kernel::GdnConv1dArgs ka{};
    ka.input = (const bf16t*)s.buf.input;
    ka.output = (float*)s.buf.out_opt;
    ka.conv_weight = (const bf16t*)s.buf.weight;
    ka.conv_state = (bf16t*)s.buf.state_opt;
    ka.requests = (const ::shelf::runtime::DeviceRequestDescriptor*)s.buf.requests;
    ka.num_requests = s.nreq;
    ka.actual_rows = s.rows;
    ka.max_request_rows = (s.rows + s.nreq - 1u) / s.nreq;
    ka.input_row_stride = s.in_stride;
    ka.output_row_stride = s.out_stride;
    ka.conv_dim = s.conv_dim;
    ka.conv_history = 3u;
    ka.state_index = o.state_index;
    ka.max_sequences = o.max_sequences;
    ka.conv_slot_stride = s.slot_stride;
    ka.conv_layer_stride = s.layer_stride;
    ka.conv_history_stride = s.hist_stride;
    return ::shelf::kernel::launch_gdn_conv1d_bf16_f32(ka, stream);
}

}  // namespace

int run_gdn_conv1d(int argc, char** argv) {
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
        else if (a == "--conv-dim") o.conv_dim = (uint32_t)std::stoul(next());
        else if (a == "--rows") { if (!parse_list(next(), o.rows)) std::exit(2); }
        else if (a == "--requests") o.requests = (uint32_t)std::stoul(next());
        else if (a == "--state-index") o.state_index = (uint32_t)std::stoul(next());
        else if (a == "--num-gdn-states") o.num_gdn_states = (uint32_t)std::stoul(next());
        else if (a == "--max-sequences") o.max_sequences = (uint32_t)std::stoul(next());
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
    if (o.max_sequences == 0u || o.num_gdn_states == 0u) return 2;
    if (o.state_index >= o.num_gdn_states) return 2;
    if (hipSetDevice(o.device) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    if (!o.out.empty()) {
        FILE* f = std::fopen(o.out.c_str(), "a");
        if (f) {
            if (std::ftell(f) == 0)
                std::fprintf(f, "variant,conv_dim,rows,requests,state_index,warmup,samples,"
                                "launches_per_sample,min_us,p50_us,p95_us,mean_us\n");
            std::fclose(f);
        }
    }

    int rc = 0;
    for (uint32_t rows : o.rows) {
        Setup s;
        if (!build(s, o, rows)) { std::printf("build fail rows=%u\n", rows); rc = 1; continue; }
        if (hipDeviceSynchronize() != hipSuccess) { rc = 1; continue; }

        if (o.check) {
            if (launch_correctness(s, o, stream) != hipSuccess) { rc = 1; continue; }
            if (launch_optimized(s, o, stream) != hipSuccess) { rc = 1; continue; }
            if (hipStreamSynchronize(stream) != hipSuccess) { rc = 1; continue; }
            std::vector<float> oc((size_t)rows * s.out_stride), oo((size_t)rows * s.out_stride);
            std::vector<uint16_t> sc(s.state_total), so(s.state_total);
            if (hipMemcpy(oc.data(), s.buf.out_correct, oc.size() * 4, hipMemcpyDeviceToHost) != hipSuccess) { rc = 1; continue; }
            if (hipMemcpy(oo.data(), s.buf.out_opt, oo.size() * 4, hipMemcpyDeviceToHost) != hipSuccess) { rc = 1; continue; }
            if (hipMemcpy(sc.data(), s.buf.state_correct, sc.size() * 2, hipMemcpyDeviceToHost) != hipSuccess) { rc = 1; continue; }
            if (hipMemcpy(so.data(), s.buf.state_opt, so.size() * 2, hipMemcpyDeviceToHost) != hipSuccess) { rc = 1; continue; }
            double max_abs = 0.0, max_rel = 0.0;
            size_t bad = 0, state_diff = 0;
            for (size_t i = 0; i < oc.size(); ++i) {
                const double diff = std::fabs((double)oo[i] - (double)oc[i]);
                const double rel = diff / std::fmax(1.0, std::fabs((double)oc[i]));
                max_abs = std::max(max_abs, diff);
                max_rel = std::max(max_rel, rel);
                if (rel > 2e-4) ++bad;
            }
            for (size_t i = 0; i < sc.size(); ++i)
                if (sc[i] != so[i]) ++state_diff;
            std::printf("CHECK %s rows=%-5u max_abs=%.3g max_rel=%.3g state_diff=%zu %s\n",
                        o.variant.c_str(), rows, max_abs, max_rel, state_diff,
                        (bad == 0 && state_diff == 0) ? "PASS" : "FAIL");
            if (bad != 0 || state_diff != 0) rc = 1;
        }

        auto launch = [&](hipStream_t st) -> hipError_t {
            if (o.variant == "correctness") return launch_correctness(s, o, st);
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
        const double p50 = percentile(per, 0.5);
        const double p95 = percentile(per, 0.95);
        double mean = 0.0;
        for (double x : per) mean += x;
        mean /= (double)per.size();
        if (!o.out.empty()) {
            FILE* f = std::fopen(o.out.c_str(), "a");
            if (f) {
                std::fprintf(f, "%s,%u,%u,%u,%u,%d,%d,%d,%.3f,%.3f,%.3f,%.3f\n",
                             o.variant.c_str(), s.conv_dim, rows, s.nreq, o.state_index, o.warmup,
                             o.samples, o.launches, mn, p50, p95, mean);
                std::fclose(f);
            }
        }
        std::printf("%-11s conv f=%-5u rows=%-5u nreq=%-3u p50=%.2fus mean=%.2fus\n",
                    o.variant.c_str(), s.conv_dim, rows, s.nreq, p50, mean);
    }
    shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return rc;
}
