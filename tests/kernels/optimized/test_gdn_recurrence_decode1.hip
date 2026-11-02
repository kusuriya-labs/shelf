#include <shelf/kernels/optimized/gdn/recurrence.h>
#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
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
    ps::fp16_t* state_ref = nullptr;
    ps::fp16_t* state_dec = nullptr;
    float* out_ref = nullptr;
    float* out_dec = nullptr;
    void* requests = nullptr;
    uint32_t nreq = 0;
    uint32_t qk = 0;
    uint32_t vdim = 0;
    uint64_t slot_stride = 0;
    uint32_t seed = 0;

    void free_all() {
        void* bufs[] = {q, k, v, a, b, dt_bias, a_log, state_ref, state_dec,
                        out_ref, out_dec, requests};
        for (void* p : bufs)
            if (p) ps::gpu::discard_cleanup_result(hipFree(p));
    }
};

bool setup(Buf& d, const Shapes& sh, uint32_t nreq, uint32_t seed) {
    d.nreq = nreq;
    d.seed = seed;
    d.qk = sh.key_heads * sh.head_k;
    d.vdim = sh.num_v_heads * sh.head_v;
    d.slot_stride = (uint64_t)sh.num_v_heads * sh.head_k * sh.head_v;

    const uint32_t rows = nreq;
    const size_t qk_bytes = (size_t)rows * d.qk * 4u;
    const size_t vdim_bytes = (size_t)rows * d.vdim * 2u;
    const size_t gate_bytes = (size_t)rows * sh.num_v_heads * 2u;

    if (hipMalloc(&d.q, qk_bytes) != hipSuccess) return false;
    if (hipMalloc(&d.k, qk_bytes) != hipSuccess) return false;
    if (hipMalloc(&d.v, vdim_bytes) != hipSuccess) return false;
    if (hipMalloc(&d.a, gate_bytes) != hipSuccess) return false;
    if (hipMalloc(&d.b, gate_bytes) != hipSuccess) return false;
    if (hipMalloc(&d.dt_bias, (size_t)sh.num_v_heads * 2u) != hipSuccess) return false;
    if (hipMalloc(&d.a_log, (size_t)sh.num_v_heads * 2u) != hipSuccess) return false;
    const size_t st_bytes = (size_t)nreq * d.slot_stride * sizeof(ps::fp16_t);
    if (hipMalloc(&d.state_ref, st_bytes) != hipSuccess) return false;
    if (hipMalloc(&d.state_dec, st_bytes) != hipSuccess) return false;
    if (hipMalloc(&d.out_ref, (size_t)rows * d.vdim * 4u) != hipSuccess) return false;
    if (hipMalloc(&d.out_dec, (size_t)rows * d.vdim * 4u) != hipSuccess) return false;
    if (hipMalloc(&d.requests, (size_t)nreq * sizeof(::ps::runtime::DeviceRequestDescriptor)) !=
        hipSuccess)
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

    std::vector<::ps::runtime::DeviceRequestDescriptor> req(nreq);
    for (uint32_t r = 0; r < nreq; ++r) {
        std::memset(&req[r], 0, sizeof(req[r]));
        req[r].request_handle.slot = r;
        req[r].row_begin = r;
        req[r].row_count = 1u;
    }
    return hipMemcpy(d.requests, req.data(), req.size() * sizeof(req[0]),
                     hipMemcpyHostToDevice) == hipSuccess;
}

void write_step_inputs(const Buf& d, const Shapes& sh, uint32_t step, bool zero,
                       float in_scale) {
    const uint32_t rows = d.nreq;
    uint32_t s = d.seed * 7919u + step * 104729u + 3u;
    if (zero) s = 0u;
    std::vector<float> hq((size_t)rows * d.qk), hk((size_t)rows * d.qk);
    std::vector<uint16_t> hv((size_t)rows * d.vdim), ha((size_t)rows * sh.num_v_heads),
        hb((size_t)rows * sh.num_v_heads);
    const float sc = zero ? 0.0f : in_scale;
    for (size_t i = 0; i < hq.size(); ++i) hq[i] = rndf(s, sc);
    for (size_t i = 0; i < hk.size(); ++i) hk[i] = rndf(s, sc);
    for (size_t i = 0; i < hv.size(); ++i) hv[i] = bf16_of(rndf(s, zero ? 0.0f : in_scale));
    for (size_t i = 0; i < ha.size(); ++i)
        ha[i] = bf16_of(rndf(s, zero ? 0.0f : 0.5f * in_scale));
    for (size_t i = 0; i < hb.size(); ++i)
        hb[i] = bf16_of(rndf(s, zero ? 0.0f : 0.5f * in_scale));
    ps::gpu::discard_cleanup_result(hipMemcpyAsync(d.q, hq.data(), hq.size() * 4u,
                                                   hipMemcpyHostToDevice));
    ps::gpu::discard_cleanup_result(hipMemcpyAsync(d.k, hk.data(), hk.size() * 4u,
                                                   hipMemcpyHostToDevice));
    ps::gpu::discard_cleanup_result(hipMemcpyAsync(d.v, hv.data(), hv.size() * 2u,
                                                   hipMemcpyHostToDevice));
    ps::gpu::discard_cleanup_result(hipMemcpyAsync(d.a, ha.data(), ha.size() * 2u,
                                                   hipMemcpyHostToDevice));
    ps::gpu::discard_cleanup_result(hipMemcpyAsync(d.b, hb.data(), hb.size() * 2u,
                                                   hipMemcpyHostToDevice));
}

void reset_states(const Buf& d, uint32_t seed, bool zero) {
    const size_t st_n = (size_t)d.nreq * d.slot_stride;
    uint32_t s = seed * 31u + 5u;
    std::vector<float> hs(st_n);
    for (size_t i = 0; i < hs.size(); ++i) hs[i] = rndf(s, zero ? 0.0f : 0.25f);
    std::vector<ps::fp16_t> hs16(hs.size());
    for (size_t i = 0; i < hs.size(); ++i) hs16[i] = ps::f32_to_fp16(hs[i]);
    ps::gpu::discard_cleanup_result(hipMemcpy(d.state_ref, hs16.data(),
                                              hs16.size() * sizeof(ps::fp16_t),
                                              hipMemcpyHostToDevice));
    ps::gpu::discard_cleanup_result(hipMemcpy(d.state_dec, hs16.data(),
                                              hs16.size() * sizeof(ps::fp16_t),
                                              hipMemcpyHostToDevice));
}

::ps::kernel::GdnRecurrenceArgs make_args(const Buf& d, const Shapes& sh) {
    ::ps::kernel::GdnRecurrenceArgs a;
    a.q = d.q;
    a.k = d.k;
    a.v = reinterpret_cast<const ps::bf16_t*>(d.v);
    a.a = reinterpret_cast<const ps::bf16_t*>(d.a);
    a.b = reinterpret_cast<const ps::bf16_t*>(d.b);
    a.dt_bias = reinterpret_cast<const ps::bf16_t*>(d.dt_bias);
    a.a_log = reinterpret_cast<const ps::bf16_t*>(d.a_log);
    a.recurrent_state = d.state_ref;
    a.output = d.out_ref;
    a.requests = reinterpret_cast<const ::ps::runtime::DeviceRequestDescriptor*>(d.requests);
    a.num_requests = d.nreq;
    a.actual_rows = d.nreq;
    a.q_row_stride = d.qk;
    a.k_row_stride = d.qk;
    a.v_row_stride = d.vdim;
    a.a_row_stride = sh.num_v_heads;
    a.b_row_stride = sh.num_v_heads;
    a.output_row_stride = d.vdim;
    a.state_index = 0;
    a.max_sequences = d.nreq;
    a.num_gdn_states = 1;
    a.key_heads = sh.key_heads;
    a.num_v_heads = sh.num_v_heads;
    a.head_k = sh.head_k;
    a.head_v = sh.head_v;
    a.recurrent_slot_stride = d.slot_stride;
    a.recurrent_layer_stride = d.slot_stride;
    return a;
}

bool run_sequence(Buf& d, const Shapes& sh, uint32_t nreq, uint32_t steps, bool zero,
                  float in_scale, hipStream_t stream) {
    reset_states(d, d.seed, zero);
    const size_t out_n = (size_t)d.nreq * d.vdim;
    const size_t st_n = (size_t)d.nreq * d.slot_stride;
    std::vector<float> o_ref(out_n), o_dec(out_n);
    std::vector<ps::fp16_t> st_ref(st_n), st_dec(st_n);

    bool ok = true;
    size_t bad_out = 0, bad_st = 0;
    uint32_t bad_step = 0;
    for (uint32_t step = 0; step < steps; ++step) {
        write_step_inputs(d, sh, step, zero, in_scale);
        const ::ps::kernel::GdnRecurrenceArgs a = make_args(d, sh);
        if (::ps::kernel::launch_gdn_recurrence_f32_wmma(a, stream) != hipSuccess) {
            std::printf("FAIL: ref launch nreq=%u step=%u\n", nreq, step);
            return false;
        }
        ::ps::kernel::GdnRecurrenceArgs ad = a;
        ad.recurrent_state = d.state_dec;
        ad.output = d.out_dec;
        const hipError_t hd =
            ::ps::kernel::launch_gdn_recurrence_f32_wmma_decode1(ad, stream);
        if (hd != hipSuccess) {
            std::printf("FAIL: decode launch nreq=%u step=%u\n", nreq, step);
            return false;
        }
        if (hipStreamSynchronize(stream) != hipSuccess) return false;
        ps::gpu::discard_cleanup_result(hipMemcpy(o_ref.data(), d.out_ref, out_n * 4u,
                                                  hipMemcpyDeviceToHost));
        ps::gpu::discard_cleanup_result(hipMemcpy(o_dec.data(), d.out_dec, out_n * 4u,
                                                  hipMemcpyDeviceToHost));
        ps::gpu::discard_cleanup_result(hipMemcpy(st_ref.data(), d.state_ref,
                                                  st_n * sizeof(ps::fp16_t),
                                                  hipMemcpyDeviceToHost));
        ps::gpu::discard_cleanup_result(hipMemcpy(st_dec.data(), d.state_dec,
                                                  st_n * sizeof(ps::fp16_t),
                                                  hipMemcpyDeviceToHost));
        bool out_ok = true;
        for (size_t i = 0; i < out_n; ++i) {
            if (std::memcmp(&o_ref[i], &o_dec[i], 4) != 0) { ok = false; out_ok = false; bad_out = i; bad_step = step; break; }
        }
        if (ok) {
            for (size_t i = 0; i < st_n; ++i) {
                if (std::memcmp(&st_ref[i], &st_dec[i], sizeof(ps::fp16_t)) != 0) { ok = false; bad_st = i; bad_step = step; break; }
            }
        }
        if (!ok) {
            const uint32_t ro = bad_out ? (uint32_t)(bad_out / d.vdim) : 0;
            std::printf("  [diag] step=%u nreq=%u out_bad=%zu state_bad=%zu "
                        "ref=0x%08x dec=0x%08x row=%u\n",
                        bad_step, nreq, bad_out, bad_st,
                        out_ok ? (uint32_t)(*(const uint16_t*)&st_ref[bad_st]) : *(const uint32_t*)&o_ref[bad_out],
                        out_ok ? (uint32_t)(*(const uint16_t*)&st_dec[bad_st]) : *(const uint32_t*)&o_dec[bad_out],
                        ro);
            break;
        }
    }
    char label[192];
    std::snprintf(label, sizeof(label),
                  "decode1 vs wmma lossy nreq=%u steps=%u%s scale=%.2f",
                  nreq, steps, zero ? " zero-inputs" : "", (double)in_scale);
    check(ok, label);
    return ok;
}

bool run_serial_equiv(Buf& d, const Shapes& sh, uint32_t rows, hipStream_t stream) {
    using ::ps::runtime::DeviceRequestDescriptor;
    DeviceRequestDescriptor req{};
    req.request_handle.slot = 0u;
    req.row_begin = 0u;
    req.row_count = rows;
    if (hipMemcpy(d.requests, &req, sizeof(req), hipMemcpyHostToDevice) != hipSuccess)
        return false;

    reset_states(d, d.seed, false);
    write_step_inputs(d, sh, 0u, false, 1.0f);

    ::ps::kernel::GdnRecurrenceArgs a = make_args(d, sh);
    a.num_requests = 1u;
    a.actual_rows = rows;
    a.max_sequences = 1u;

    a.recurrent_state = d.state_ref;
    a.output = d.out_ref;
    for (uint32_t r = 0u; r < rows; ++r) {
        a.row_override = static_cast<int32_t>(r);
        if (::ps::kernel::launch_gdn_recurrence_f32_wmma_decode1(a, stream) != hipSuccess)
            return false;
    }

    ::ps::kernel::GdnRecurrenceArgs b = a;
    b.row_override = -1;
    b.recurrent_state = d.state_dec;
    b.output = d.out_dec;
    if (::ps::kernel::launch_gdn_recurrence_f32_wmma_decode1_serial(b, stream) != hipSuccess)
        return false;
    if (hipStreamSynchronize(stream) != hipSuccess) return false;

    const size_t st_n = static_cast<size_t>(d.slot_stride);
    std::vector<ps::fp16_t> st_ref(st_n), st_dec(st_n);
    ps::gpu::discard_cleanup_result(
        hipMemcpy(st_ref.data(), d.state_ref, st_n * sizeof(ps::fp16_t), hipMemcpyDeviceToHost));
    ps::gpu::discard_cleanup_result(
        hipMemcpy(st_dec.data(), d.state_dec, st_n * sizeof(ps::fp16_t), hipMemcpyDeviceToHost));
    bool ok = std::memcmp(st_ref.data(), st_dec.data(), st_n * sizeof(ps::fp16_t)) == 0;
    const size_t out_n = static_cast<size_t>(rows) * d.vdim;
    std::vector<float> o_ref(out_n), o_dec(out_n);
    ps::gpu::discard_cleanup_result(
        hipMemcpy(o_ref.data(), d.out_ref, out_n * 4u, hipMemcpyDeviceToHost));
    ps::gpu::discard_cleanup_result(
        hipMemcpy(o_dec.data(), d.out_dec, out_n * 4u, hipMemcpyDeviceToHost));
    if (ok) ok = std::memcmp(o_ref.data(), o_dec.data(), out_n * 4u) == 0;
    char label[160];
    std::snprintf(label, sizeof(label),
                  "decode1 serial rows=%u bit-exact vs sequential decode1", rows);
    check(ok, label);
    return ok;
}

bool run_multirow_equiv(Buf& d, const Shapes& sh, uint32_t rows, bool with_history,
                        uint32_t capture_rows, hipStream_t stream) {
    using ::ps::runtime::DeviceRequestDescriptor;
    DeviceRequestDescriptor req{};
    req.request_handle.slot = 0u;
    req.row_begin = 0u;
    req.row_count = rows;
    if (hipMemcpy(d.requests, &req, sizeof(req), hipMemcpyHostToDevice) != hipSuccess)
        return false;

    const uint64_t hist_stride = d.slot_stride;
    ps::fp16_t* hist_ref = nullptr;
    ps::fp16_t* hist_dec = nullptr;
    if (with_history) {
        if (hipMalloc(&hist_ref, (size_t)capture_rows * hist_stride * sizeof(ps::fp16_t)) !=
            hipSuccess)
            return false;
        if (hipMalloc(&hist_dec, (size_t)capture_rows * hist_stride * sizeof(ps::fp16_t)) !=
            hipSuccess) {
            ps::gpu::discard_cleanup_result(hipFree(hist_ref));
            return false;
        }
    }

    reset_states(d, d.seed, false);
    write_step_inputs(d, sh, 0u, false, 1.0f);

    ::ps::kernel::GdnRecurrenceArgs a = make_args(d, sh);
    a.num_requests = 1u;
    a.actual_rows = rows;
    a.max_sequences = 1u;
    if (with_history) {
        a.recurrent_history = hist_ref;
        a.recurrent_history_row_stride = hist_stride;
        a.capture_rows = capture_rows;
    }

    ::ps::kernel::GdnRecurrenceArgs a_ref = a;
    a_ref.recurrent_state = d.state_ref;
    a_ref.output = d.out_ref;
    for (uint32_t r = 0u; r < rows; ++r) {
        a_ref.row_override = static_cast<int32_t>(r);
        if (::ps::kernel::launch_gdn_recurrence_f32_wmma_decode1(a_ref, stream) != hipSuccess)
            return false;
    }

    ::ps::kernel::GdnRecurrenceArgs a_new = a;
    a_new.recurrent_state = d.state_dec;
    a_new.output = d.out_dec;
    if (with_history) a_new.recurrent_history = hist_dec;
    if (::ps::kernel::launch_gdn_recurrence_f32_wmma_decode_rows_exact(a_new, stream) !=
        hipSuccess)
        return false;
    if (hipStreamSynchronize(stream) != hipSuccess) return false;

    const size_t st_n = static_cast<size_t>(d.slot_stride);
    const size_t out_n = static_cast<size_t>(rows) * d.vdim;
    std::vector<ps::fp16_t> st_ref(st_n), st_dec(st_n);
    std::vector<float> o_ref(out_n), o_dec(out_n);
    ps::gpu::discard_cleanup_result(
        hipMemcpy(st_ref.data(), d.state_ref, st_n * sizeof(ps::fp16_t), hipMemcpyDeviceToHost));
    ps::gpu::discard_cleanup_result(
        hipMemcpy(st_dec.data(), d.state_dec, st_n * sizeof(ps::fp16_t), hipMemcpyDeviceToHost));
    ps::gpu::discard_cleanup_result(
        hipMemcpy(o_ref.data(), d.out_ref, out_n * 4u, hipMemcpyDeviceToHost));
    ps::gpu::discard_cleanup_result(
        hipMemcpy(o_dec.data(), d.out_dec, out_n * 4u, hipMemcpyDeviceToHost));
    bool ok = std::memcmp(st_ref.data(), st_dec.data(), st_n * sizeof(ps::fp16_t)) == 0;
    if (ok) ok = std::memcmp(o_ref.data(), o_dec.data(), out_n * 4u) == 0;

    bool hist_ok = true;
    if (with_history) {
        const size_t hn = static_cast<size_t>(capture_rows) * hist_stride;
        std::vector<ps::fp16_t> h_ref(hn), h_dec(hn);
        ps::gpu::discard_cleanup_result(
            hipMemcpy(h_ref.data(), hist_ref, hn * sizeof(ps::fp16_t), hipMemcpyDeviceToHost));
        ps::gpu::discard_cleanup_result(
            hipMemcpy(h_dec.data(), hist_dec, hn * sizeof(ps::fp16_t), hipMemcpyDeviceToHost));
        hist_ok = std::memcmp(h_ref.data(), h_dec.data(), hn * sizeof(ps::fp16_t)) == 0;
        if (!hist_ok) {
            size_t hi = 0;
            while (hi < hn &&
                   std::memcmp(&h_ref[hi], &h_dec[hi], sizeof(ps::fp16_t)) == 0)
                ++hi;
            std::printf("  [hist] rows=%u cap=%u first_bad=%zu (row=%zu off=%zu) ref=%.6g dec=%.6g\n",
                        rows, capture_rows, hi, hi / hist_stride, hi % hist_stride,
                        (double)ps::fp16_to_f32(h_ref[hi]), (double)ps::fp16_to_f32(h_dec[hi]));
        }
        ps::gpu::discard_cleanup_result(hipFree(hist_ref));
        ps::gpu::discard_cleanup_result(hipFree(hist_dec));
    }
    if (!ok) std::printf("  [out/state] rows=%u cap=%u ok=0\n", rows, capture_rows);

    char label[192];
    if (with_history)
        std::snprintf(label, sizeof(label),
                      "multirow rows=%u capture=%u output/state/history bit-exact", rows,
                      capture_rows);
    else
        std::snprintf(label, sizeof(label),
                      "multirow rows=%u output/state bit-exact vs sequential decode1", rows);
    check(ok && hist_ok, label);
    return ok && hist_ok;
}

}  // namespace

int main() {
    std::printf("=== GDN recurrence decode1 bit-exact vs production lossy wmma ===\n");
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    Shapes sh{};
    const uint32_t nreqs[] = {1u, 2u, 4u, 8u};
    const uint32_t step_counts[] = {1u, 2u, 16u, 128u};
    for (uint32_t nreq : nreqs) {
        Buf d;
        if (!setup(d, sh, nreq, 1234u + nreq)) {
            std::printf("FAIL: setup nreq=%u\n", nreq);
            d.free_all();
            failed++;
            continue;
        }
        for (uint32_t steps : step_counts) {
            const float in_scale = steps >= 128u ? 0.25f : 1.0f;
            run_sequence(d, sh, nreq, steps, false, in_scale, stream);
        }
        run_sequence(d, sh, nreq, 16u, true, 1.0f, stream);
        d.free_all();
    }

    const uint32_t serial_rows[] = {2u, 4u, 8u};
    for (uint32_t rows : serial_rows) {
        Buf d;
        if (!setup(d, sh, rows, 999u + rows)) {
            std::printf("FAIL: setup serial rows=%u\n", rows);
            d.free_all();
            failed++;
            continue;
        }
        run_serial_equiv(d, sh, rows, stream);
        d.free_all();
    }

    for (uint32_t rows : serial_rows) {
        Buf d;
        if (!setup(d, sh, rows, 4242u + rows)) {
            std::printf("FAIL: setup multirow rows=%u\n", rows);
            d.free_all();
            failed++;
            continue;
        }
        run_multirow_equiv(d, sh, rows, false, 0u, stream);
        const uint32_t captures[] = {1u, rows - 1u, rows};
        for (uint32_t cap : captures) {
            if (cap == 0u) continue;
            run_multirow_equiv(d, sh, rows, true, cap, stream);
        }
        d.free_all();
    }

    Shapes sh_tp2{};
    sh_tp2.key_heads = 8u;
    sh_tp2.num_v_heads = 24u;
    for (uint32_t nreq : {1u, 4u}) {
        Buf d;
        if (!setup(d, sh_tp2, nreq, 3100u + nreq)) {
            std::printf("FAIL: setup tp2 nreq=%u\n", nreq);
            d.free_all();
            failed++;
            continue;
        }
        run_sequence(d, sh_tp2, nreq, 1u, false, 1.0f, stream);
        run_sequence(d, sh_tp2, nreq, 16u, false, 1.0f, stream);
        run_sequence(d, sh_tp2, nreq, 16u, true, 1.0f, stream);
        d.free_all();
    }
    for (uint32_t rows : {2u, 4u, 8u}) {
        Buf d;
        if (!setup(d, sh_tp2, rows, 3200u + rows)) {
            std::printf("FAIL: setup tp2 serial rows=%u\n", rows);
            d.free_all();
            failed++;
            continue;
        }
        run_serial_equiv(d, sh_tp2, rows, stream);
        d.free_all();
    }
    for (uint32_t rows : {2u, 8u}) {
        Buf d;
        if (!setup(d, sh_tp2, rows, 3300u + rows)) {
            std::printf("FAIL: setup tp2 multirow rows=%u\n", rows);
            d.free_all();
            failed++;
            continue;
        }
        run_multirow_equiv(d, sh_tp2, rows, false, 0u, stream);
        d.free_all();
    }

    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    std::printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
