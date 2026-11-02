#include <shelf/kernels/optimized/gdn/recurrence.h>
#include <shelf/kernels/correctness/model_dispatch_correctness.h>
#include <shelf/models/qwen35/state/state_types.h>
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

using bf16t = ps::bf16_t;
using V = ::ps::runtime::DeviceValueBinding;
using S = ::ps::runtime::DeviceStateBinding;
using P = ::ps::runtime::DeviceProgramView;
using B = ::ps::runtime::DispatchBinding;

struct Profile {
    uint32_t key_heads = 16;
    uint32_t num_v_heads = 32;
    uint32_t head_k = 128;
    uint32_t head_v = 128;
};

struct Options {
    std::string variant = "optimized";
    std::vector<uint32_t> rows = {1, 4, 16, 64, 256, 1024, 2048};
    std::vector<uint32_t> nreq = {1, 4, 16};
    Profile profile;
    int warmup = 20;
    int samples = 30;
    int launches = 10;
    uint32_t seed = 1;
    int device = 0;
    bool check = false;
    bool history = false;
    std::string out;
};

void usage(FILE* f) {
    std::fprintf(f,
                 "usage: shelf-bench gdn-recurrence [options]\n"
                 "  --variant correctness|optimized|optimized-multirow\n"
                 "  --rows 1,4,...\n"
                 "  --nreq 1,4,16\n"
                 "  --key-heads N\n"
                 "  --value-heads N\n"
                 "  --head-k N\n"
                 "  --head-v N\n"
                 "  --warmup N\n"
                 "  --samples N\n"
                 "  --launches N\n"
                 "  --seed N\n"
                 "  --device N\n"
                 "  --check\n"
                 "  --history\n"
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

bool upload_state(void* dst, const std::vector<float>& src, size_t n) {
    std::vector<ps::fp16_t> tmp(n);
    for (size_t i = 0; i < n; ++i) tmp[i] = ps::f32_to_fp16(src[i]);
    return hipMemcpy(dst, tmp.data(), n * sizeof(ps::fp16_t), hipMemcpyHostToDevice) == hipSuccess;
}

hipError_t upload_state_cleanup(void* dst, const std::vector<float>& src, size_t n) {
    std::vector<ps::fp16_t> tmp(n);
    for (size_t i = 0; i < n; ++i) tmp[i] = ps::f32_to_fp16(src[i]);
    return hipMemcpy(dst, tmp.data(), n * sizeof(ps::fp16_t), hipMemcpyHostToDevice);
}

bool download_state(const void* src, std::vector<float>& dst, size_t n) {
    std::vector<ps::fp16_t> tmp(n);
    if (hipMemcpy(tmp.data(), src, n * sizeof(ps::fp16_t), hipMemcpyDeviceToHost) != hipSuccess)
        return false;
    dst.resize(n);
    for (size_t i = 0; i < n; ++i) dst[i] = ps::fp16_to_f32(tmp[i]);
    return true;
}

struct Env {
    Profile p;
    uint32_t rows = 0;
    uint32_t num_requests = 0;
    uint32_t max_request_rows = 0;
    uint32_t qk = 0;
    uint32_t vdim = 0;
    void* q = nullptr;
    void* k = nullptr;
    void* v = nullptr;
    void* a = nullptr;
    void* b = nullptr;
    void* dt_bias = nullptr;
    void* a_log = nullptr;
    void* state = nullptr;
    void* requests = nullptr;
    void* out = nullptr;
    void* vals_dev = nullptr;
    void* state_dev = nullptr;
    void* param_dev = nullptr;
    void* ctx_dev = nullptr;
    void* scratch = nullptr;
    void* error_word = nullptr;
    void* hist = nullptr;
    P prog{};
    B binding{};
    ::ps::kernel::ModelDispatchStateView state_view{};
    ::ps::kernel::GdnRecurrenceArgs ka{};
    std::vector<::ps::runtime::DeviceRequestDescriptor> h_req;
    ~Env() {
        void** all = nullptr;
        (void)all;
        void* bufs[] = {q, k, v, a, b, dt_bias, a_log, state, requests, out, vals_dev, state_dev,
                        param_dev, ctx_dev, scratch, error_word, hist};
        for (auto* p : bufs)
            if (p) ps::gpu::discard_cleanup_result(hipFree(p));
    }
};

bool build(Env& e, uint32_t seed) {
    e.qk = e.p.key_heads * e.p.head_k;
    e.vdim = e.p.num_v_heads * e.p.head_v;
    const uint32_t nvh = e.p.num_v_heads;
    const uint32_t rows = e.rows;
    const uint32_t nr = e.num_requests;
    const uint32_t max_seq = nr;
    e.max_request_rows = (rows + nr - 1u) / nr;
    const uint64_t layer_stride = (uint64_t)nvh * e.p.head_k * e.p.head_v;
    const uint64_t slot_stride = layer_stride;
    const uint64_t state_total = (uint64_t)max_seq * slot_stride;

    void** bufs[] = {&e.q, &e.k, &e.v, &e.a, &e.b, &e.dt_bias, &e.a_log, &e.state, &e.requests,
                     &e.out, &e.vals_dev, &e.state_dev, &e.param_dev, &e.ctx_dev};
    size_t szs[] = {(size_t)rows * e.qk * 4, (size_t)rows * e.qk * 4, (size_t)rows * e.vdim * 2,
                    (size_t)rows * nvh * 2, (size_t)rows * nvh * 2, (size_t)nvh * 2, (size_t)nvh * 2,
                    state_total * sizeof(ps::fp16_t), (size_t)nr * sizeof(::ps::runtime::DeviceRequestDescriptor),
                    (size_t)rows * e.vdim * 4, 6 * sizeof(V), sizeof(S),
                    2 * sizeof(::ps::runtime::StaticParameterSlot), sizeof(::ps::runtime::DeviceBatchContext)};
    for (size_t i = 0; i < 14; ++i)
        if (hipMalloc(bufs[i], szs[i]) != hipSuccess) return false;
    if (hipMalloc(&e.scratch, (size_t)256 * 4096 * sizeof(float)) != hipSuccess) return false;
    if (hipMalloc(&e.error_word, sizeof(uint32_t)) != hipSuccess) return false;

    std::vector<float> h_q((size_t)rows * e.qk), h_k((size_t)rows * e.qk);
    std::vector<uint16_t> h_v((size_t)rows * e.vdim), h_a((size_t)rows * nvh), h_b((size_t)rows * nvh);
    std::vector<uint16_t> h_dt(nvh), h_al(nvh);
    std::vector<float> h_state(state_total);
    for (size_t i = 0; i < h_q.size(); ++i) h_q[i] = rng1k(seed * 3u + i) / 1000.0f;
    for (size_t i = 0; i < h_k.size(); ++i) h_k[i] = rng1k(seed * 5u + i) / 1000.0f;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t kh = 0; kh < e.p.key_heads; ++kh) {
            const size_t off = (size_t)r * e.qk + (size_t)kh * e.p.head_k;
            double sq_q = 0.0, sq_k = 0.0;
            for (uint32_t d = 0; d < e.p.head_k; ++d) {
                sq_q += (double)h_q[off + d] * (double)h_q[off + d];
                sq_k += (double)h_k[off + d] * (double)h_k[off + d];
            }
            const float nq = sq_q > 0.0 ? (float)(1.0 / std::sqrt(sq_q)) : 0.0f;
            const float nk = sq_k > 0.0 ? (float)(1.0 / std::sqrt(sq_k)) : 0.0f;
            for (uint32_t d = 0; d < e.p.head_k; ++d) {
                h_q[off + d] *= nq;
                h_k[off + d] *= nk;
            }
        }
    }
    for (size_t i = 0; i < h_v.size(); ++i) h_v[i] = bf16_bits(rng1k(seed * 7u + i) / 1000.0f);
    for (size_t i = 0; i < h_a.size(); ++i) h_a[i] = bf16_bits(rng1k(seed * 11u + i) / 1000.0f);
    for (size_t i = 0; i < h_b.size(); ++i) h_b[i] = bf16_bits(rng1k(seed * 13u + i) / 1000.0f);
    for (size_t i = 0; i < h_dt.size(); ++i) h_dt[i] = bf16_bits(rng1k(seed * 17u + i) / 1000.0f);
    for (size_t i = 0; i < h_al.size(); ++i) h_al[i] = bf16_bits(rng1k(seed * 19u + i) / 1000.0f);
    for (size_t i = 0; i < h_state.size(); ++i) h_state[i] = rng1k(seed * 23u + i) / 512.0f;

    if (hipMemcpy(e.q, h_q.data(), h_q.size() * 4, hipMemcpyHostToDevice) != hipSuccess) return false;
    if (hipMemcpy(e.k, h_k.data(), h_k.size() * 4, hipMemcpyHostToDevice) != hipSuccess) return false;
    if (hipMemcpy(e.v, h_v.data(), h_v.size() * 2, hipMemcpyHostToDevice) != hipSuccess) return false;
    if (hipMemcpy(e.a, h_a.data(), h_a.size() * 2, hipMemcpyHostToDevice) != hipSuccess) return false;
    if (hipMemcpy(e.b, h_b.data(), h_b.size() * 2, hipMemcpyHostToDevice) != hipSuccess) return false;
    if (hipMemcpy(e.dt_bias, h_dt.data(), h_dt.size() * 2, hipMemcpyHostToDevice) != hipSuccess) return false;
    if (hipMemcpy(e.a_log, h_al.data(), h_al.size() * 2, hipMemcpyHostToDevice) != hipSuccess) return false;
    if (!upload_state(e.state, h_state, h_state.size())) return false;
    if (hipMemset(e.out, 0, (size_t)rows * e.vdim * 4) != hipSuccess) return false;

    e.h_req.resize(nr);
    uint32_t rb = 0;
    for (uint32_t r = 0; r < nr; ++r) {
        std::memset(&e.h_req[r], 0, sizeof(e.h_req[r]));
        e.h_req[r].request_handle.slot = r;
        e.h_req[r].row_begin = rb;
        const uint32_t rc = (r == nr - 1) ? (rows - rb) : std::min(e.max_request_rows, rows - rb);
        e.h_req[r].row_count = rc;
        rb += rc;
    }
    if (hipMemcpy(e.requests, e.h_req.data(), e.h_req.size() * sizeof(e.h_req[0]), hipMemcpyHostToDevice) != hipSuccess) return false;

    V vals[6];
    auto setv = [&](V& v, uint32_t id, uint32_t feat, uint32_t stride, void* ptr,
                    ::ps::runtime::ValueDType dt) {
        v.logical_value = id;
        v.storage = ::ps::runtime::ValueStorage::WORKSPACE;
        v.dtype = dt;
        v.row_domain = ::ps::runtime::ValueRowDomain::TOKEN_ROWS;
        v.slot = 0;
        v.feature_count = feat;
        v.row_stride = stride;
        v.offset = (uint64_t)(uintptr_t)ptr;
        v.bytes = 0;
    };
    setv(vals[0], 0, e.qk, e.qk, e.q, ::ps::runtime::ValueDType::F32);
    setv(vals[1], 1, e.qk, e.qk, e.k, ::ps::runtime::ValueDType::F32);
    setv(vals[2], 2, e.vdim, e.vdim, e.v, ::ps::runtime::ValueDType::BF16);
    setv(vals[3], 3, nvh, nvh, e.a, ::ps::runtime::ValueDType::BF16);
    setv(vals[4], 4, nvh, nvh, e.b, ::ps::runtime::ValueDType::BF16);
    setv(vals[5], 5, e.vdim, e.vdim, e.out, ::ps::runtime::ValueDType::F32);
    std::memcpy(e.vals_dev, vals, 6 * sizeof(V));

    S st;
    st.slot = 0;
    st.kind = (uint8_t)::ps::runtime::PrimitiveStateKind::GDN_RECURRENCE_STATE;
    st.state_index = 0;
    std::memcpy(e.state_dev, &st, sizeof(S));

    ::ps::runtime::StaticParameterSlot params[2];
    params[0].device_ptr = e.dt_bias;
    params[0].elements = nvh;
    params[0].dtype = ::ps::runtime::ValueDType::BF16;
    params[1].device_ptr = e.a_log;
    params[1].elements = nvh;
    params[1].dtype = ::ps::runtime::ValueDType::BF16;
    std::memcpy(e.param_dev, params, 2 * sizeof(::ps::runtime::StaticParameterSlot));

    ::ps::runtime::DeviceBatchContext ctx{};
    ctx.actual_rows = rows;
    ctx.num_requests = nr;
    ctx.requests = (const ::ps::runtime::DeviceRequestDescriptor*)e.requests;
    std::memcpy(e.ctx_dev, &ctx, sizeof(ctx));

    e.prog = {};
    e.prog.values = (const V*)e.vals_dev;
    e.prog.value_count = 6;
    e.prog.states = (const S*)e.state_dev;
    e.prog.state_count = 1;
    e.prog.workspace_base = 0;
    e.prog.parameter_table = (const ::ps::runtime::StaticParameterSlot*)e.param_dev;
    e.prog.parameter_slot_count = 2;

    e.binding = {};
    e.binding.kernel_id = ::ps::runtime::KernelId::GDN_RECURRENCE;
    e.binding.descriptor_slot = 1;
    e.binding.input_slots[0] = 0;
    e.binding.input_slots[1] = 1;
    e.binding.input_slots[2] = 2;
    e.binding.input_slots[3] = 3;
    e.binding.input_slots[4] = 4;
    e.binding.output_slots[0] = 5;
    e.binding.input_count = 5;
    e.binding.output_count = 1;
    e.binding.parameter_index0 = 0;
    e.binding.parameter_index1 = 1;

    e.state_view = {};
    e.state_view.gdn.recurrent_base = (ps::fp16_t*)e.state;
    e.state_view.gdn.num_gdn_states = 1;
    e.state_view.gdn.max_sequences = max_seq;
    e.state_view.gdn.recurrent_slot_stride = slot_stride;
    e.state_view.gdn.recurrent_layer_stride = layer_stride;
    e.state_view.gdn.num_v_heads = nvh;
    e.state_view.gdn.head_k = e.p.head_k;
    e.state_view.gdn.head_v = e.p.head_v;

    e.ka = {};
    e.ka.q = (const float*)e.q;
    e.ka.k = (const float*)e.k;
    e.ka.v = (const bf16t*)e.v;
    e.ka.a = (const bf16t*)e.a;
    e.ka.b = (const bf16t*)e.b;
    e.ka.dt_bias = (const bf16t*)e.dt_bias;
    e.ka.a_log = (const bf16t*)e.a_log;
    e.ka.recurrent_state = (ps::fp16_t*)e.state;
    e.ka.output = (float*)e.out;
    e.ka.requests = (const ::ps::runtime::DeviceRequestDescriptor*)e.requests;
    e.ka.num_requests = nr;
    e.ka.actual_rows = rows;
    e.ka.q_row_stride = e.qk;
    e.ka.k_row_stride = e.qk;
    e.ka.v_row_stride = e.vdim;
    e.ka.a_row_stride = nvh;
    e.ka.b_row_stride = nvh;
    e.ka.output_row_stride = e.vdim;
    e.ka.state_index = 0;
    e.ka.max_sequences = max_seq;
    e.ka.num_gdn_states = 1;
    e.ka.key_heads = e.p.key_heads;
    e.ka.num_v_heads = nvh;
    e.ka.head_k = e.p.head_k;
    e.ka.head_v = e.p.head_v;
    e.ka.recurrent_slot_stride = slot_stride;
    e.ka.recurrent_layer_stride = layer_stride;
    return true;
}

void reset_state(const Env& e, uint32_t seed) {
    const uint32_t nvh = e.p.num_v_heads;
    const uint64_t layer_stride = (uint64_t)nvh * e.p.head_k * e.p.head_v;
    const uint64_t state_total = (uint64_t)e.num_requests * layer_stride;
    std::vector<float> h_state(state_total);
    for (size_t i = 0; i < h_state.size(); ++i) h_state[i] = rng1k(seed * 23u + i) / 512.0f;
    ps::gpu::discard_cleanup_result(
        upload_state_cleanup(e.state, h_state, h_state.size()));
    ps::gpu::discard_cleanup_result(hipMemset(e.out, 0, (size_t)e.rows * e.vdim * 4));
}

hipError_t launch_correctness(const Env& e, hipStream_t s) {
    ps::gpu::discard_cleanup_result(hipMemsetAsync(e.error_word, 0, sizeof(uint32_t), s));
    auto st = ps::kernel::launch_model_dispatch_correctness(
        e.prog, e.binding, (::ps::runtime::DeviceBatchContext*)e.ctx_dev, e.state_view,
        (float*)e.scratch, (uint32_t*)e.error_word, s);
    if (!st.ok()) return hipErrorUnknown;
    return hipGetLastError();
}

hipError_t launch_optimized(const Env& e, hipStream_t s) {
    if (e.ka.head_k == 128u && e.ka.head_v % 64u == 0u)
        return ::ps::kernel::launch_gdn_recurrence_f32_wmma(e.ka, s);
    return ::ps::kernel::launch_gdn_recurrence_f32(e.ka, s);
}

hipError_t launch_optimized_multirow(const Env& e, hipStream_t s) {
    if (::ps::kernel::gdn_recurrence_decode_rows_supported(e.ka))
        return ::ps::kernel::launch_gdn_recurrence_f32_wmma_decode_rows_exact(e.ka, s);
    return launch_optimized(e, s);
}

}  // namespace

int run_gdn_recurrence(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing arg for %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--help" || a == "-h") { usage(stdout); return 0; }
        else if (a == "--variant") o.variant = next();
        else if (a == "--rows") { if (!parse_list(next(), o.rows)) std::exit(2); }
        else if (a == "--nreq") { if (!parse_list(next(), o.nreq)) std::exit(2); }
        else if (a == "--key-heads") o.profile.key_heads = (uint32_t)std::stoul(next());
        else if (a == "--value-heads") o.profile.num_v_heads = (uint32_t)std::stoul(next());
        else if (a == "--head-k") o.profile.head_k = (uint32_t)std::stoul(next());
        else if (a == "--head-v") o.profile.head_v = (uint32_t)std::stoul(next());
        else if (a == "--warmup") o.warmup = std::atoi(next());
        else if (a == "--samples") o.samples = std::atoi(next());
        else if (a == "--launches") o.launches = std::atoi(next());
        else if (a == "--seed") o.seed = (uint32_t)std::stoul(next());
        else if (a == "--device") o.device = std::atoi(next());
        else if (a == "--check") o.check = true;
        else if (a == "--history") o.history = true;
        else if (a == "--out") o.out = next();
        else { usage(stderr); return 2; }
    }
    if (o.variant != "correctness" && o.variant != "optimized" &&
        o.variant != "optimized-multirow") { usage(stderr); return 2; }
    if (o.samples <= 0 || o.launches <= 0) return 2;
    if (hipSetDevice(o.device) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    if (!o.out.empty()) {
        FILE* f = std::fopen(o.out.c_str(), "a");
        if (f) {
            if (std::ftell(f) == 0)
                std::fprintf(f, "variant,key_heads,num_v_heads,head_k,head_v,rows,"
                                "num_requests,max_request_rows,warmup,samples,launches_per_sample,"
                                "min_us,p50_us,p95_us,mean_us\n");
            std::fclose(f);
        }
    }

    int rc = 0;
    for (uint32_t rows : o.rows) {
        for (uint32_t nr : o.nreq) {
            if (nr > rows) continue;
            Env e;
            e.p = o.profile;
            e.rows = rows;
            e.num_requests = nr;
            if (!build(e, o.seed)) { std::printf("build fail rows=%u nr=%u\n", rows, nr); rc = 1; continue; }
            if (hipDeviceSynchronize() != hipSuccess) { rc = 1; continue; }
            if (o.history && nr == 1u && rows > 1u) {
                const uint64_t slot_stride =
                    (uint64_t)e.p.num_v_heads * e.p.head_k * e.p.head_v;
                const uint32_t capture = rows - 1u;
                if (hipMalloc(&e.hist, (size_t)capture * slot_stride * sizeof(ps::fp16_t)) !=
                    hipSuccess) {
                    rc = 1;
                    continue;
                }
                e.ka.recurrent_history = (ps::fp16_t*)e.hist;
                e.ka.recurrent_history_row_stride = slot_stride;
                e.ka.capture_rows = capture;
            }
            auto launch = [&](hipStream_t s) -> hipError_t {
                if (o.variant == "correctness") return launch_correctness(e, s);
                if (o.variant == "optimized-multirow") return launch_optimized_multirow(e, s);
                return launch_optimized(e, s);
            };
            if (o.check) {
                reset_state(e, o.seed);
                if (launch_correctness(e, stream) != hipSuccess) { rc = 1; continue; }
                if (hipStreamSynchronize(stream) != hipSuccess) { rc = 1; continue; }
                std::vector<float> ref((size_t)rows * e.vdim);
                if (hipMemcpy(ref.data(), e.out, ref.size() * 4, hipMemcpyDeviceToHost) != hipSuccess) { rc = 1; continue; }
                const uint64_t st_total = (uint64_t)nr * (uint64_t)e.p.num_v_heads * e.p.head_k * e.p.head_v;
                std::vector<float> ref_state(st_total);
                if (!download_state(e.state, ref_state, st_total)) { rc = 1; continue; }
                reset_state(e, o.seed);
                if (launch(stream) != hipSuccess) { rc = 1; continue; }
                if (hipStreamSynchronize(stream) != hipSuccess) { rc = 1; continue; }
                std::vector<float> got(ref.size());
                if (hipMemcpy(got.data(), e.out, got.size() * 4, hipMemcpyDeviceToHost) != hipSuccess) { rc = 1; continue; }
                std::vector<float> got_state(st_total);
                if (!download_state(e.state, got_state, st_total)) { rc = 1; continue; }
                const bool bf16_path = (o.variant == "optimized") && e.ka.head_k == 128u && e.ka.head_v % 64u == 0u;
                double sref_pre = 0.0;
                if (bf16_path)
                    for (size_t i = 0; i < ref_state.size(); ++i)
                        if (std::fabs((double)ref_state[i]) > sref_pre) sref_pre = std::fabs((double)ref_state[i]);
                double ref_pre = 0.0;
                if (bf16_path)
                    for (size_t i = 0; i < ref.size(); ++i)
                        if (std::fabs((double)ref[i]) > ref_pre) ref_pre = std::fabs((double)ref[i]);
                const double sthr = bf16_path ? 1.0 : 1e-3;
                size_t sbad = 0; double smax = 0.0, srefmax = 0.0;
                for (size_t i = 0; i < ref_state.size(); ++i) {
                    const double diff = std::fabs((double)got_state[i] - (double)ref_state[i]);
                    const double den = bf16_path ? std::max(0.05 * sref_pre, 1e-6)
                                                 : std::max(1.0, std::fabs((double)ref_state[i]));
                    if (diff / den > sthr) ++sbad;
                    if (diff / den > smax) smax = diff / den;
                    if (std::fabs((double)ref_state[i]) > srefmax) srefmax = std::fabs((double)ref_state[i]);
                }
                const double othr = bf16_path ? 1.0 : 1e-3;
                double max_rel = 0.0, refmax = 0.0, relmax = 0.0, absmax = 0.0; double sumabs = 0.0; size_t wrel = 0, wabs = 0;
                size_t bad = 0;
                for (size_t i = 0; i < ref.size(); ++i) {
                    const double diff = std::fabs((double)got[i] - (double)ref[i]);
                    const double den = bf16_path ? std::max(0.05 * ref_pre, 1e-6)
                                                 : std::max(1.0, std::fabs((double)ref[i]));
                    if (diff / den > othr) ++bad;
                    if (diff / den > max_rel) max_rel = diff / den;
                    if (std::fabs((double)ref[i]) > refmax) refmax = std::fabs((double)ref[i]);
                    sumabs += diff;
                    if (!bf16_path && std::fabs((double)ref[i]) > 0.01) {
                        const double r = diff / std::fabs((double)ref[i]);
                        if (r > relmax) { relmax = r; wrel = i; }
                    }
                    if (diff > absmax) { absmax = diff; wabs = i; }
                }
                std::printf("CHECK %-11s rows=%-5u nr=%-3u mismatches=%zu max_rel=%.3g relmax=%.3g meanabs=%.3g refmax=%.3g state_bad=%zu state_max=%.3g srefmax=%.3g wabs=(%zu,%zu) %s\n",
                            o.variant.c_str(), rows, nr, bad, max_rel, relmax, sumabs / (double)ref.size(), refmax, sbad, smax, srefmax,
                            wabs / e.vdim, wabs % e.vdim,
                            (bad == 0 && sbad == 0) ? "OK" : "FAIL");
                if (bad != 0 || sbad != 0) rc = 1;
            }
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
                    std::fprintf(f, "%s,%u,%u,%u,%u,%u,%u,%u,%d,%d,%d,%.3f,%.3f,%.3f,%.3f\n",
                                  o.variant.c_str(), o.profile.key_heads, o.profile.num_v_heads,
                                  o.profile.head_k, o.profile.head_v, rows, nr,
                                  e.max_request_rows, o.warmup, o.samples, o.launches, mn, p50, p95, mean);
                    std::fclose(f);
                }
            }
            std::printf("%-11s kh=%u nvh=%u hk=%u hv=%u rows=%-5u nr=%-3u mr=%-4u "
                        "p10=%.2fus p50=%.2fus p90=%.2fus mean=%.2fus\n",
                        o.variant.c_str(), o.profile.key_heads, o.profile.num_v_heads, o.profile.head_k,
                        o.profile.head_v, rows, nr, e.max_request_rows, p10, p50, p90, mean);
        }
    }
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return rc;
}
