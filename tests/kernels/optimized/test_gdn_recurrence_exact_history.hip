#include <shelf/kernels/optimized/gdn/recurrence.h>
#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

int passed = 0;
int failed = 0;

void check(bool cond, const char* name) {
    if (cond) {
        passed++;
        std::printf("PASS: %s\n", name);
    } else {
        failed++;
        std::printf("FAIL: %s\n", name);
    }
}

uint32_t xr(uint32_t& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

float rndf(uint32_t& s, float scale) {
    return ((float)(int32_t)(xr(s) % 2001u) - 1000.0f) / 1000.0f * scale;
}

uint16_t bf16_of(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return (uint16_t)(u >> 16);
}

struct Shapes {
    uint32_t key_heads = 16;
    uint32_t num_v_heads = 48;
    uint32_t head_k = 128;
    uint32_t head_v = 128;
};

struct Buf {
    float* q = nullptr;
    float* k = nullptr;
    uint16_t* v = nullptr;
    uint16_t* a = nullptr;
    uint16_t* b = nullptr;
    uint16_t* dt_bias = nullptr;
    uint16_t* a_log = nullptr;
    shelf::fp16_t* state_ref = nullptr;
    shelf::fp16_t* state_cand = nullptr;
    float* out_ref = nullptr;
    float* out_cand = nullptr;
    shelf::fp16_t* hist_ref = nullptr;
    shelf::fp16_t* hist_cand = nullptr;
    void* requests = nullptr;
    uint32_t rows = 0;
    uint32_t qk = 0;
    uint32_t vdim = 0;
    uint64_t slot_stride = 0;
    uint32_t seed = 0;

    void free_all() {
        void* bufs[] = {q,      k,        v,       a,        b,        dt_bias,   a_log,
                        state_ref, state_cand, out_ref, out_cand, hist_ref, hist_cand,
                        requests};
        for (void* p : bufs)
            if (p) shelf::gpu::discard_cleanup_result(hipFree(p));
    }
};

bool setup(Buf& d, const Shapes& sh, uint32_t rows, uint32_t capture_rows, uint32_t seed) {
    d.rows = rows;
    d.seed = seed;
    d.qk = sh.key_heads * sh.head_k;
    d.vdim = sh.num_v_heads * sh.head_v;
    d.slot_stride = (uint64_t)sh.num_v_heads * sh.head_k * sh.head_v;

    const size_t qk_bytes = (size_t)rows * d.qk * 4u;
    const size_t vdim_bytes = (size_t)rows * d.vdim * 2u;
    const size_t gate_bytes = (size_t)rows * sh.num_v_heads * 2u;
    const size_t st_bytes = (size_t)rows * d.slot_stride * 4u;
    const size_t hist_bytes =
        (size_t)capture_rows * d.slot_stride * sizeof(shelf::fp16_t);

    if (hipMalloc(&d.q, qk_bytes) != hipSuccess) return false;
    if (hipMalloc(&d.k, qk_bytes) != hipSuccess) return false;
    if (hipMalloc(&d.v, vdim_bytes) != hipSuccess) return false;
    if (hipMalloc(&d.a, gate_bytes) != hipSuccess) return false;
    if (hipMalloc(&d.b, gate_bytes) != hipSuccess) return false;
    if (hipMalloc(&d.dt_bias, (size_t)sh.num_v_heads * 2u) != hipSuccess) return false;
    if (hipMalloc(&d.a_log, (size_t)sh.num_v_heads * 2u) != hipSuccess) return false;
    if (hipMalloc(&d.state_ref, (size_t)d.slot_stride * sizeof(shelf::fp16_t)) != hipSuccess)
        return false;
    if (hipMalloc(&d.state_cand, (size_t)d.slot_stride * sizeof(shelf::fp16_t)) != hipSuccess)
        return false;
    if (hipMalloc(&d.out_ref, (size_t)rows * d.vdim * 4u) != hipSuccess) return false;
    if (hipMalloc(&d.out_cand, (size_t)rows * d.vdim * 4u) != hipSuccess) return false;
    if (hipMalloc(&d.hist_ref, hist_bytes) != hipSuccess) return false;
    if (hipMalloc(&d.hist_cand, hist_bytes) != hipSuccess) return false;
    if (hipMalloc(&d.requests, sizeof(::shelf::runtime::DeviceRequestDescriptor)) != hipSuccess)
        return false;

    std::vector<uint16_t> h_dt(sh.num_v_heads), h_al(sh.num_v_heads);
    uint32_t s = seed * 2654435761u + 17u;
    for (uint32_t i = 0; i < sh.num_v_heads; ++i) {
        h_dt[i] = bf16_of(rndf(s, 0.5f));
        h_al[i] = bf16_of(rndf(s, 0.5f));
    }
    if (hipMemcpy(d.dt_bias, h_dt.data(), h_dt.size() * 2u, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d.a_log, h_al.data(), h_al.size() * 2u, hipMemcpyHostToDevice) != hipSuccess)
        return false;

    std::vector<float> hq((size_t)rows * d.qk), hk((size_t)rows * d.qk);
    std::vector<uint16_t> hv((size_t)rows * d.vdim), ha((size_t)rows * sh.num_v_heads),
        hb((size_t)rows * sh.num_v_heads);
    uint32_t si = seed * 7919u + 3u;
    for (size_t i = 0; i < hq.size(); ++i) hq[i] = rndf(si, 1.0f);
    for (size_t i = 0; i < hk.size(); ++i) hk[i] = rndf(si, 1.0f);
    for (size_t i = 0; i < hv.size(); ++i) hv[i] = bf16_of(rndf(si, 1.0f));
    for (size_t i = 0; i < ha.size(); ++i) ha[i] = bf16_of(rndf(si, 0.5f));
    for (size_t i = 0; i < hb.size(); ++i) hb[i] = bf16_of(rndf(si, 0.5f));
    if (hipMemcpy(d.q, hq.data(), hq.size() * 4u, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d.k, hk.data(), hk.size() * 4u, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d.v, hv.data(), hv.size() * 2u, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d.a, ha.data(), ha.size() * 2u, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d.b, hb.data(), hb.size() * 2u, hipMemcpyHostToDevice) != hipSuccess)
        return false;

    std::vector<float> hs((size_t)d.slot_stride);
    uint32_t ss = seed * 31u + 5u;
    for (size_t i = 0; i < hs.size(); ++i) hs[i] = rndf(ss, 0.25f);
    std::vector<shelf::fp16_t> hs16(hs.size());
    for (size_t i = 0; i < hs.size(); ++i) hs16[i] = shelf::f32_to_fp16(hs[i]);
    if (hipMemcpy(d.state_ref, hs16.data(), hs16.size() * sizeof(shelf::fp16_t),
                  hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d.state_cand, hs16.data(), hs16.size() * sizeof(shelf::fp16_t),
                  hipMemcpyHostToDevice) != hipSuccess)
        return false;

    std::vector<shelf::fp16_t> hh(hist_bytes / sizeof(shelf::fp16_t),
                               shelf::f32_to_fp16(-12345.0f));
    if (hipMemcpy(d.hist_ref, hh.data(), hist_bytes, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d.hist_cand, hh.data(), hist_bytes, hipMemcpyHostToDevice) != hipSuccess)
        return false;
    return true;
}

::shelf::kernel::GdnRecurrenceArgs make_args(const Buf& d, const Shapes& sh) {
    ::shelf::kernel::GdnRecurrenceArgs a;
    a.q = d.q;
    a.k = d.k;
    a.v = reinterpret_cast<const shelf::bf16_t*>(d.v);
    a.a = reinterpret_cast<const shelf::bf16_t*>(d.a);
    a.b = reinterpret_cast<const shelf::bf16_t*>(d.b);
    a.dt_bias = reinterpret_cast<const shelf::bf16_t*>(d.dt_bias);
    a.a_log = reinterpret_cast<const shelf::bf16_t*>(d.a_log);
    a.recurrent_state = d.state_ref;
    a.output = d.out_ref;
    a.requests = reinterpret_cast<const ::shelf::runtime::DeviceRequestDescriptor*>(d.requests);
    a.num_requests = 1u;
    a.actual_rows = d.rows;
    a.q_row_stride = d.qk;
    a.k_row_stride = d.qk;
    a.v_row_stride = d.vdim;
    a.a_row_stride = sh.num_v_heads;
    a.b_row_stride = sh.num_v_heads;
    a.output_row_stride = d.vdim;
    a.state_index = 0;
    a.max_sequences = 1u;
    a.num_gdn_states = 1u;
    a.key_heads = sh.key_heads;
    a.num_v_heads = sh.num_v_heads;
    a.head_k = sh.head_k;
    a.head_v = sh.head_v;
    a.recurrent_slot_stride = d.slot_stride;
    a.recurrent_layer_stride = d.slot_stride;
    return a;
}

bool upload_req(const Buf& d, uint32_t row_begin, uint32_t row_count) {
    ::shelf::runtime::DeviceRequestDescriptor req{};
    req.request_handle.slot = 0u;
    req.row_begin = row_begin;
    req.row_count = row_count;
    return hipMemcpy(d.requests, &req, sizeof(req), hipMemcpyHostToDevice) == hipSuccess;
}

bool run_case(const Shapes& sh, uint32_t rows, uint32_t capture_rows, hipStream_t stream) {
    Buf d;
    if (!setup(d, sh, rows, capture_rows, 7000u + rows)) {
        std::printf("FAIL: setup rows=%u\n", rows);
        d.free_all();
        return false;
    }

    bool ok = true;
    for (uint32_t r = 0u; r < rows && ok; ++r) {
        if (!upload_req(d, r, 1u)) ok = false;
        ::shelf::kernel::GdnRecurrenceArgs a = make_args(d, sh);
        a.actual_rows = rows;
        if (ok && ::shelf::kernel::launch_gdn_recurrence_f32_wmma(a, stream) != hipSuccess) ok = false;
        if (ok && hipStreamSynchronize(stream) != hipSuccess) ok = false;
        if (ok && r < capture_rows)
            ok = hipMemcpy(d.hist_ref + (size_t)r * d.slot_stride, d.state_ref,
                           (size_t)d.slot_stride * sizeof(shelf::fp16_t),
                           hipMemcpyDeviceToDevice) == hipSuccess;
    }

    if (ok && !upload_req(d, 0u, rows)) ok = false;
    ::shelf::kernel::GdnRecurrenceArgs cand = make_args(d, sh);
    cand.recurrent_state = d.state_cand;
    cand.output = d.out_cand;
    cand.recurrent_history = d.hist_cand;
    cand.recurrent_history_row_stride = d.slot_stride;
    cand.capture_rows = capture_rows;
    if (ok && ::shelf::kernel::launch_gdn_recurrence_f32_wmma_serial(cand, stream) != hipSuccess)
        ok = false;
    if (ok && hipStreamSynchronize(stream) != hipSuccess) ok = false;

    bool state_ok = ok;
    bool out_ok = ok;
    bool hist_ok = ok;
    if (ok) {
        std::vector<shelf::fp16_t> s_ref(d.slot_stride), s_cand(d.slot_stride);
        shelf::gpu::discard_cleanup_result(
            hipMemcpy(s_ref.data(), d.state_ref, s_ref.size() * sizeof(shelf::fp16_t),
                      hipMemcpyDeviceToHost));
        shelf::gpu::discard_cleanup_result(
            hipMemcpy(s_cand.data(), d.state_cand, s_cand.size() * sizeof(shelf::fp16_t),
                      hipMemcpyDeviceToHost));
        state_ok = std::memcmp(s_ref.data(), s_cand.data(),
                               s_ref.size() * sizeof(shelf::fp16_t)) == 0;

        std::vector<float> o_ref((size_t)rows * d.vdim), o_cand((size_t)rows * d.vdim);
        shelf::gpu::discard_cleanup_result(
            hipMemcpy(o_ref.data(), d.out_ref, o_ref.size() * 4u, hipMemcpyDeviceToHost));
        shelf::gpu::discard_cleanup_result(
            hipMemcpy(o_cand.data(), d.out_cand, o_cand.size() * 4u, hipMemcpyDeviceToHost));
        out_ok = std::memcmp(o_ref.data(), o_cand.data(), o_ref.size() * 4u) == 0;

        const size_t hn = (size_t)capture_rows * d.slot_stride;
        std::vector<shelf::fp16_t> h_ref(hn), h_cand(hn);
        shelf::gpu::discard_cleanup_result(
            hipMemcpy(h_ref.data(), d.hist_ref, hn * sizeof(shelf::fp16_t),
                      hipMemcpyDeviceToHost));
        shelf::gpu::discard_cleanup_result(
            hipMemcpy(h_cand.data(), d.hist_cand, hn * sizeof(shelf::fp16_t),
                      hipMemcpyDeviceToHost));
        hist_ok = std::memcmp(h_ref.data(), h_cand.data(), hn * sizeof(shelf::fp16_t)) == 0;
        if (!hist_ok) {
            size_t hi = 0;
            while (hi < hn && std::memcmp(&h_ref[hi], &h_cand[hi], sizeof(shelf::fp16_t)) == 0)
                ++hi;
            std::printf("  [hist] rows=%u cap=%u first_bad=%zu (row=%zu off=%zu) ref=%.6g cand=%.6g\n",
                        rows, capture_rows, hi, hi / d.slot_stride, hi % d.slot_stride,
                        (double)shelf::fp16_to_f32(h_ref[hi]),
                        (double)shelf::fp16_to_f32(h_cand[hi]));
        }
    }

    char label[192];
    std::snprintf(label, sizeof(label),
                  "exact serial rows=%u capture=%u state/output/history bit-exact",
                  rows, capture_rows);
    check(ok && state_ok && out_ok && hist_ok, label);
    d.free_all();
    return ok && state_ok && out_ok && hist_ok;
}

}  // namespace

int main() {
    std::printf("=== GDN recurrence exact serial history capture ===\n");
    unsetenv("SHELF_GDN_RECURRENCE_LOSSY");
    setenv("SHELF_GDN_RECURRENCE_EXACT", "1", 1);

    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    if (::shelf::kernel::gdn_recurrence_lossy_enabled()) {
        std::printf("FAIL: exact mode not active\n");
        shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));
        return 1;
    }

    Shapes sh{};
    const uint32_t row_counts[] = {1u, 2u, 4u, 8u, 12u, 17u};
    for (uint32_t rows : row_counts) run_case(sh, rows, rows, stream);
    run_case(sh, 8u, 3u, stream);

    shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    std::printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
