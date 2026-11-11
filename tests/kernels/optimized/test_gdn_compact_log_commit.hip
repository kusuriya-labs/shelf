#include <shelf/kernels/optimized/gdn/recurrence.h>
#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>

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

float rndf(uint32_t& s) {
    return ((float)(int32_t)(xr(s) % 2001u) - 1000.0f) / 1000.0f;
}

uint16_t bf16_of(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return (uint16_t)(u >> 16);
}

constexpr uint32_t kKeyHeads = 16u;
constexpr uint32_t kHeads = 48u;
constexpr uint32_t kHeadK = 128u;
constexpr uint32_t kHeadV = 128u;
constexpr uint32_t kRows = 16u;
constexpr uint32_t kStates = 2u;
constexpr uint32_t kQk = kKeyHeads * kHeadK;
constexpr uint32_t kVdim = kHeads * kHeadV;
constexpr uint64_t kSlotStride = (uint64_t)kHeads * kHeadK * kHeadV;

struct Buf {
    float* q = nullptr;
    float* k = nullptr;
    uint16_t* v = nullptr;
    uint16_t* a = nullptr;
    uint16_t* b = nullptr;
    uint16_t* dt_bias = nullptr;
    uint16_t* a_log = nullptr;
    shelf::fp16_t* state = nullptr;
    float* out = nullptr;
    shelf::fp16_t* hist = nullptr;
    float* compact_delta = nullptr;
    float* compact_k = nullptr;
    float* compact_a = nullptr;
    void* requests = nullptr;
    void free_all() {
        void* bufs[] = {q, k, v, a, b, dt_bias, a_log, state, out, hist,
                        compact_delta, compact_k, compact_a, requests};
        for (void* p : bufs)
            if (p) shelf::gpu::discard_cleanup_result(hipFree(p));
    }
};

bool read_state_f16(const shelf::fp16_t* src, std::vector<float>& dst, size_t n) {
    std::vector<shelf::fp16_t> tmp(n);
    if (hipMemcpy(tmp.data(), src, n * sizeof(shelf::fp16_t), hipMemcpyDeviceToHost) != hipSuccess)
        return false;
    dst.resize(n);
    for (size_t i = 0; i < n; ++i) dst[i] = shelf::fp16_to_f32(tmp[i]);
    return true;
}

bool write_state_f16(shelf::fp16_t* dst, const std::vector<float>& src, size_t n) {
    std::vector<shelf::fp16_t> tmp(n);
    for (size_t i = 0; i < n; ++i) tmp[i] = shelf::f32_to_fp16(src[i]);
    return hipMemcpy(dst, tmp.data(), n * sizeof(shelf::fp16_t), hipMemcpyHostToDevice) == hipSuccess;
}

bool setup(Buf& d, uint32_t seed) {
    const size_t qk_bytes = (size_t)kRows * kQk * 4u;
    const size_t vd_bytes = (size_t)kRows * kVdim * 2u;
    const size_t g_bytes = (size_t)kRows * kHeads * 2u;
    if (hipMalloc(&d.q, qk_bytes) != hipSuccess) return false;
    if (hipMalloc(&d.k, qk_bytes) != hipSuccess) return false;
    if (hipMalloc(&d.v, vd_bytes) != hipSuccess) return false;
    if (hipMalloc(&d.a, g_bytes) != hipSuccess) return false;
    if (hipMalloc(&d.b, g_bytes) != hipSuccess) return false;
    if (hipMalloc(&d.dt_bias, kHeads * 2u) != hipSuccess) return false;
    if (hipMalloc(&d.a_log, kHeads * 2u) != hipSuccess) return false;
    if (hipMalloc(&d.state, kStates * kSlotStride * sizeof(shelf::fp16_t)) != hipSuccess)
        return false;
    if (hipMalloc(&d.out, (size_t)kRows * kVdim * 4u) != hipSuccess) return false;
    if (hipMalloc(&d.compact_delta, (size_t)kStates * kRows * kVdim * 4u) != hipSuccess)
        return false;
    if (hipMalloc(&d.compact_k, (size_t)kStates * kRows * kHeads * kHeadK * 4u) != hipSuccess)
        return false;
    if (hipMalloc(&d.compact_a, (size_t)kStates * kRows * kHeads * 4u) != hipSuccess)
        return false;
    if (hipMalloc(&d.hist, (size_t)kRows * kStates * kSlotStride * sizeof(shelf::fp16_t)) !=
            hipSuccess)
        return false;
    shelf::gpu::discard_cleanup_result(
        hipMemset(d.hist, 0, (size_t)kRows * kStates * kSlotStride * sizeof(shelf::fp16_t)));
    if (hipMalloc(&d.requests, sizeof(::shelf::runtime::DeviceRequestDescriptor)) != hipSuccess)
        return false;

    uint32_t s = seed;
    std::vector<float> hq((size_t)kRows * kQk), hk((size_t)kRows * kQk);
    std::vector<uint16_t> hv((size_t)kRows * kVdim), ha((size_t)kRows * kHeads),
        hb((size_t)kRows * kHeads), hdt(kHeads), hal(kHeads);
    for (float& x : hq) x = rndf(s);
    for (float& x : hk) x = rndf(s);
    for (uint16_t& x : hv) x = bf16_of(rndf(s));
    for (uint16_t& x : ha) x = bf16_of(rndf(s) * 0.5f);
    for (uint16_t& x : hb) x = bf16_of(rndf(s) * 0.5f);
    for (uint16_t& x : hdt) x = bf16_of(rndf(s) * 0.5f);
    for (uint16_t& x : hal) x = bf16_of(rndf(s) * 0.5f);
    std::vector<float> hs((size_t)kStates * kSlotStride);
    for (float& x : hs) x = rndf(s) * 0.25f;

    if (hipMemcpy(d.q, hq.data(), hq.size() * 4u, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d.k, hk.data(), hk.size() * 4u, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d.v, hv.data(), hv.size() * 2u, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d.a, ha.data(), ha.size() * 2u, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d.b, hb.data(), hb.size() * 2u, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d.dt_bias, hdt.data(), hdt.size() * 2u, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d.a_log, hal.data(), hal.size() * 2u, hipMemcpyHostToDevice) != hipSuccess ||
        !write_state_f16(d.state, hs, hs.size()))
        return false;

    ::shelf::runtime::DeviceRequestDescriptor req{};
    req.request_handle.slot = 0u;
    req.row_begin = 0u;
    req.row_count = kRows;
    return hipMemcpy(d.requests, &req, sizeof(req), hipMemcpyHostToDevice) == hipSuccess;
}

::shelf::kernel::GdnRecurrenceArgs make_args(const Buf& d) {
    ::shelf::kernel::GdnRecurrenceArgs a;
    a.q = d.q;
    a.k = d.k;
    a.v = reinterpret_cast<const shelf::bf16_t*>(d.v);
    a.a = reinterpret_cast<const shelf::bf16_t*>(d.a);
    a.b = reinterpret_cast<const shelf::bf16_t*>(d.b);
    a.dt_bias = reinterpret_cast<const shelf::bf16_t*>(d.dt_bias);
    a.a_log = reinterpret_cast<const shelf::bf16_t*>(d.a_log);
    a.recurrent_state = d.state;
    a.output = d.out;
    a.requests = reinterpret_cast<const ::shelf::runtime::DeviceRequestDescriptor*>(d.requests);
    a.num_requests = 1u;
    a.actual_rows = kRows;
    a.q_row_stride = kQk;
    a.k_row_stride = kQk;
    a.v_row_stride = kVdim;
    a.a_row_stride = kHeads;
    a.b_row_stride = kHeads;
    a.output_row_stride = kVdim;
    a.state_index = 0u;
    a.max_sequences = 1u;
    a.num_gdn_states = kStates;
    a.key_heads = kKeyHeads;
    a.num_v_heads = kHeads;
    a.head_k = kHeadK;
    a.head_v = kHeadV;
    a.recurrent_slot_stride = (uint64_t)kStates * kSlotStride;
    a.recurrent_layer_stride = kSlotStride;
    a.compact_delta = d.compact_delta;
    a.compact_k = d.compact_k;
    a.compact_a = d.compact_a;
    a.compact_delta_layer_stride = (uint64_t)kRows * kVdim;
    a.compact_k_layer_stride = (uint64_t)kRows * kHeads * kHeadK;
    a.compact_a_layer_stride = (uint64_t)kRows * kHeads;
    a.recurrent_history = d.hist;
    a.recurrent_history_row_stride = (uint64_t)kStates * kSlotStride;
    a.capture_rows = kRows;
    a.capture_verify_only = 0u;
    return a;
}

void test_chunk_verify(hipStream_t stream) {
    Buf d;
    if (!setup(d, 17001u)) { check(false, "chunk setup"); d.free_all(); return; }
    std::vector<float> q(kRows * kQk), k(kRows * kQk);
    bool ok = hipMemcpy(q.data(), d.q, q.size() * 4u, hipMemcpyDeviceToHost) == hipSuccess &&
              hipMemcpy(k.data(), d.k, k.size() * 4u, hipMemcpyDeviceToHost) == hipSuccess;
    for (auto& x : q) x *= 0.1f;
    for (auto& x : k) x *= 0.1f;
    ok = ok && hipMemcpy(d.q, q.data(), q.size() * 4u, hipMemcpyHostToDevice) == hipSuccess &&
         hipMemcpy(d.k, k.data(), k.size() * 4u, hipMemcpyHostToDevice) == hipSuccess;
    const size_t state_n = (size_t)kStates * kSlotStride;
    std::vector<float> base, ref_state, chunk_state;
    ok = ok && read_state_f16(d.state, base, state_n);
    auto a = make_args(d);
    a.recurrent_history = nullptr;
    a.capture_rows = 0u;
    a.skip_state_store = 1u;
    a.num_gdn_states = 1u;
    setenv("SHELF_GDN_VERIFY_CHUNKED", "0", 1);
    ok = ok && shelf::kernel::launch_gdn_recurrence_f32_wmma_verify(a, stream) == hipSuccess;
    std::vector<float> ref(kRows * kVdim), chunk(ref.size());
    ok = ok && hipStreamSynchronize(stream) == hipSuccess &&
         hipMemcpy(ref.data(), d.out, ref.size() * 4u, hipMemcpyDeviceToHost) == hipSuccess;
    setenv("SHELF_GDN_VERIFY_CHUNKED", "1", 1);
    a.recurrent_history = d.hist;
    a.capture_rows = kRows;
    ok = ok && shelf::kernel::launch_gdn_recurrence_f32_wmma_verify(a, stream) == hipSuccess &&
         hipStreamSynchronize(stream) == hipSuccess &&
         hipMemcpy(chunk.data(), d.out, chunk.size() * 4u, hipMemcpyDeviceToHost) == hipSuccess;
    check(ok && std::memcmp(ref.data(), chunk.data(), ref.size() * 4u) == 0,
          "chunk opt-in falls back when history is requested");
    a.recurrent_history = nullptr;
    a.capture_rows = 0u;
    unsetenv("SHELF_GDN_VERIFY_CHUNKED");
    a.compact_rows = kRows;
    ok = ok && shelf::kernel::launch_gdn_recurrence_f32_wmma_commit(a, stream) == hipSuccess &&
         hipStreamSynchronize(stream) == hipSuccess && read_state_f16(d.state, ref_state, state_n) &&
         write_state_f16(d.state, base, state_n);
    setenv("SHELF_GDN_VERIFY_CHUNKED", "1", 1);
    ok = ok && shelf::kernel::launch_gdn_recurrence_f32_wmma_verify(a, stream) == hipSuccess &&
         hipStreamSynchronize(stream) == hipSuccess &&
         hipMemcpy(chunk.data(), d.out, chunk.size() * 4u, hipMemcpyDeviceToHost) == hipSuccess;
    std::vector<float> explicit_chunk = chunk;
    unsetenv("SHELF_GDN_VERIFY_CHUNKED");
    ok = ok && shelf::kernel::launch_gdn_recurrence_f32_wmma_verify(a, stream) == hipSuccess &&
         hipStreamSynchronize(stream) == hipSuccess &&
         hipMemcpy(chunk.data(), d.out, chunk.size() * 4u, hipMemcpyDeviceToHost) == hipSuccess;
    check(ok && std::memcmp(explicit_chunk.data(), chunk.data(), chunk.size() * 4u) == 0,
          "single-request default selects chunk verify");
    setenv("SHELF_GDN_VERIFY_CHUNKED", "1", 1);
    if (std::getenv("SHELF_GDN_CHUNK_NUMERIC_EXACT") != nullptr) {
        std::vector<float> exact_output = chunk;
        unsetenv("SHELF_GDN_VERIFY_CHUNKED_EXACT");
        ok = ok && shelf::kernel::launch_gdn_recurrence_f32_wmma_verify(a, stream) == hipSuccess &&
             hipStreamSynchronize(stream) == hipSuccess &&
             hipMemcpy(chunk.data(), d.out, chunk.size() * 4u, hipMemcpyDeviceToHost) == hipSuccess;
        check(ok && std::memcmp(exact_output.data(), chunk.data(), chunk.size() * 4u) != 0,
              "chunk exact opt-in selects a different numeric result");
        setenv("SHELF_GDN_VERIFY_CHUNKED_EXACT", "1", 1);
        ok = ok && shelf::kernel::launch_gdn_recurrence_f32_wmma_verify(a, stream) == hipSuccess &&
             hipStreamSynchronize(stream) == hipSuccess;
        chunk = std::move(exact_output);
    }
    std::vector<float> deferred_state;
    ok = ok && read_state_f16(d.state, deferred_state, state_n);
    check(ok && deferred_state == base, "chunk verify leaves live state unchanged");
    ok = ok && shelf::kernel::launch_gdn_recurrence_f32_wmma_commit(a, stream) == hipSuccess &&
         hipStreamSynchronize(stream) == hipSuccess && read_state_f16(d.state, chunk_state, state_n);
    double maxdiff = 0, mean = 0, meanref = 0, maxref = 0, statemax = 0, stateref = 0;
    if (ok) {
        for (size_t i = 0; i < ref.size(); ++i) {
            const double diff = std::abs((double)ref[i] - chunk[i]);
            ok = ok && std::isfinite(diff);
            maxdiff = std::max(maxdiff, diff); mean += diff; meanref += std::abs((double)ref[i]);
            maxref = std::max(maxref, std::abs((double)ref[i]));
        }
        for (size_t i = 0; i < state_n; ++i) {
            statemax = std::max(statemax, std::abs((double)ref_state[i] - chunk_state[i]));
            stateref = std::max(stateref, std::abs((double)ref_state[i]));
        }
    }
    std::printf("chunk output max_abs=%.9g mean_abs=%.9g mean_relative_to_mean=%.9g max_relative_to_peak=%.9g replay_max_abs=%.9g replay_relative_to_peak=%.9g\n",
                maxdiff, mean / ref.size(), mean / meanref, maxdiff / maxref, statemax, statemax / stateref);
    check(ok && maxref > 0 && stateref > 0 && maxdiff / maxref < 0.05 && statemax / stateref < 0.05,
          "chunk verify output and replay close to serial");
    unsetenv("SHELF_GDN_VERIFY_CHUNKED");
    d.free_all();
}

void test_tree_chunk_batch(hipStream_t stream) {
    constexpr uint32_t count = 4u, rows = 9u, total = count * rows;
    Buf src, d;
    if (!setup(src, 17003u)) { check(false, "batch tree setup"); src.free_all(); return; }
    bool ok = true;
    auto duplicate = [&](auto*& dst, const auto* input, size_t row_bytes) {
        ok = ok && hipMalloc(&dst, total * row_bytes) == hipSuccess;
        for (uint32_t i = 0; ok && i < count; ++i)
            ok = hipMemcpy(reinterpret_cast<char*>(dst) + i * rows * row_bytes, input,
                           rows * row_bytes, hipMemcpyDeviceToDevice) == hipSuccess;
    };
    duplicate(d.q, src.q, kQk * sizeof(float));
    duplicate(d.k, src.k, kQk * sizeof(float));
    duplicate(d.v, src.v, kVdim * sizeof(uint16_t));
    duplicate(d.a, src.a, kHeads * sizeof(uint16_t));
    duplicate(d.b, src.b, kHeads * sizeof(uint16_t));
    ok = ok && hipMalloc(&d.state, count * kStates * kSlotStride * sizeof(shelf::fp16_t)) == hipSuccess;
    for (uint32_t i = 0; ok && i < count; ++i)
        ok = hipMemcpy(d.state + i * kStates * kSlotStride, src.state,
                       kStates * kSlotStride * sizeof(shelf::fp16_t), hipMemcpyDeviceToDevice) == hipSuccess;
    ok = ok && hipMalloc(&d.out, total * kVdim * sizeof(float)) == hipSuccess &&
         hipMalloc(&d.compact_delta, total * kVdim * sizeof(float)) == hipSuccess &&
         hipMalloc(&d.compact_k, total * kHeads * kHeadK * sizeof(float)) == hipSuccess &&
         hipMalloc(&d.compact_a, total * kHeads * sizeof(float)) == hipSuccess &&
         hipMalloc(&d.requests, count * sizeof(::shelf::runtime::DeviceRequestDescriptor)) == hipSuccess;
    int32_t* parents = nullptr;
    ok = ok && hipMalloc(&parents, total * sizeof(int32_t)) == hipSuccess;
    std::vector<int32_t> parent(total);
    std::vector<::shelf::runtime::DeviceRequestDescriptor> req(count);
    for (uint32_t i = 0; i < count; ++i) {
        req[i].request_handle.slot = i;
        req[i].row_begin = i * rows;
        req[i].row_count = rows;
        parent[i * rows] = -1;
        for (uint32_t j = 1; j < rows; ++j)
            parent[i * rows + j] = static_cast<int32_t>(i * rows + (j - 1u) / 2u);
    }
    ok = ok && hipMemcpy(d.requests, req.data(), req.size() * sizeof(req[0]), hipMemcpyHostToDevice) == hipSuccess &&
         hipMemcpy(parents, parent.data(), parent.size() * sizeof(parent[0]), hipMemcpyHostToDevice) == hipSuccess;
    auto a = make_args(d);
    a.dt_bias = reinterpret_cast<const shelf::bf16_t*>(src.dt_bias);
    a.a_log = reinterpret_cast<const shelf::bf16_t*>(src.a_log);
    a.num_requests = count;
    a.actual_rows = total;
    a.max_request_rows = rows;
    a.max_sequences = count;
    a.num_gdn_states = 1u;
    a.row_parent = parents;
    a.recurrent_history = nullptr;
    a.capture_rows = 0u;
    a.skip_state_store = 1u;
    a.compact_delta_layer_stride = total * kVdim;
    a.compact_k_layer_stride = total * kHeads * kHeadK;
    a.compact_a_layer_stride = total * kHeads;
    setenv("SHELF_GDN_VERIFY_CHUNKED", "1", 1);
    setenv("SHELF_GDN_TREE_CHUNKED", "1", 1);
    ok = ok && shelf::kernel::launch_gdn_recurrence_f32_wmma_verify(a, stream) == hipSuccess &&
         hipStreamSynchronize(stream) == hipSuccess;
    std::vector<float> batch(total * kVdim), separate(batch.size());
    ok = ok && hipMemcpy(batch.data(), d.out, batch.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess;
    for (uint32_t i = 0; ok && i < count; ++i) {
        a.num_requests = 1u;
        a.requests = static_cast<const ::shelf::runtime::DeviceRequestDescriptor*>(d.requests) + i;
        ok = shelf::kernel::launch_gdn_recurrence_f32_wmma_verify(a, stream) == hipSuccess;
    }
    ok = ok && hipStreamSynchronize(stream) == hipSuccess &&
         hipMemcpy(separate.data(), d.out, separate.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess;
    check(ok && batch == separate, "tree chunk batch over 32 rows matches separate requests");
    unsetenv("SHELF_GDN_TREE_CHUNKED");
    unsetenv("SHELF_GDN_VERIFY_CHUNKED");
    if (parents) shelf::gpu::discard_cleanup_result(hipFree(parents));
    d.free_all();
    src.free_all();
}

void test_tree_chunk(hipStream_t stream) {
    Buf d;
    if (!setup(d, 17002u)) { check(false, "tree chunk setup"); d.free_all(); return; }
    std::vector<float> q(kRows * kQk), k(kRows * kQk);
    bool ok = hipMemcpy(q.data(), d.q, q.size() * 4u, hipMemcpyDeviceToHost) == hipSuccess &&
              hipMemcpy(k.data(), d.k, k.size() * 4u, hipMemcpyDeviceToHost) == hipSuccess;
    for (auto& x : q) x *= 0.1f;
    for (auto& x : k) x *= 0.1f;
    ok = ok && hipMemcpy(d.q, q.data(), q.size() * 4u, hipMemcpyHostToDevice) == hipSuccess &&
         hipMemcpy(d.k, k.data(), k.size() * 4u, hipMemcpyHostToDevice) == hipSuccess;
    std::vector<uint16_t> hv(kRows * kVdim), ha(kRows * kHeads), hb(kRows * kHeads);
    std::vector<uint16_t> hdt(kHeads), hal(kHeads);
    ok = ok && hipMemcpy(hv.data(), d.v, hv.size() * 2u, hipMemcpyDeviceToHost) == hipSuccess &&
         hipMemcpy(ha.data(), d.a, ha.size() * 2u, hipMemcpyDeviceToHost) == hipSuccess &&
         hipMemcpy(hb.data(), d.b, hb.size() * 2u, hipMemcpyDeviceToHost) == hipSuccess &&
         hipMemcpy(hdt.data(), d.dt_bias, hdt.size() * 2u, hipMemcpyDeviceToHost) == hipSuccess &&
         hipMemcpy(hal.data(), d.a_log, hal.size() * 2u, hipMemcpyDeviceToHost) == hipSuccess;
    auto bf = [](uint16_t v) {
        uint32_t bits = static_cast<uint32_t>(v) << 16u;
        float x;
        std::memcpy(&x, &bits, sizeof(x));
        return static_cast<double>(x);
    };
    int32_t* parents_dev = nullptr;
    int32_t* path_dev = nullptr;
    ok = ok && hipMalloc(&parents_dev, kRows * sizeof(int32_t)) == hipSuccess &&
         hipMalloc(&path_dev, kRows * sizeof(int32_t)) == hipSuccess;
    const size_t state_n = (size_t)kStates * kSlotStride;
    std::vector<float> base;
    ok = ok && read_state_f16(d.state, base, state_n);
    setenv("SHELF_GDN_VERIFY_CHUNKED", "1", 1);
    setenv("SHELF_GDN_TREE_CHUNKED", "1", 1);
    for (uint32_t n : {8u, 10u, 16u}) {
        for (uint32_t shape = 0; shape < 3; ++shape) {
            std::vector<int32_t> parent(n, -1);
            for (uint32_t i = 1; i < n; ++i)
                parent[i] = shape == 0 ? static_cast<int32_t>(i - 1u) :
                            shape == 1 ? 0 : static_cast<int32_t>((i - 1u) / 2u);
            ::shelf::runtime::DeviceRequestDescriptor req{};
            req.request_handle.slot = 0u;
            req.row_count = n;
            ok = ok && hipMemcpy(d.requests, &req, sizeof(req), hipMemcpyHostToDevice) == hipSuccess &&
                 hipMemcpy(parents_dev, parent.data(), n * sizeof(int32_t), hipMemcpyHostToDevice) == hipSuccess &&
                 write_state_f16(d.state, base, state_n);
            auto a = make_args(d);
            a.actual_rows = n;
            a.row_parent = parents_dev;
            a.skip_state_store = 1u;
            a.num_gdn_states = 1u;
            unsetenv("SHELF_GDN_TREE_CHUNKED");
            ok = ok && shelf::kernel::launch_gdn_recurrence_f32_wmma_verify(a, stream) == hipSuccess &&
                 hipStreamSynchronize(stream) == hipSuccess;
            std::vector<float> ref(n * kVdim), got(ref.size());
            ok = ok && hipMemcpy(ref.data(), d.out, ref.size() * 4u, hipMemcpyDeviceToHost) == hipSuccess;
            std::vector<float> ref_state;
            ok = ok && read_state_f16(d.hist + (size_t)(n - 1u) * kStates * kSlotStride, ref_state, kSlotStride);
            a.recurrent_history = nullptr;
            a.capture_rows = 0;
            setenv("SHELF_GDN_TREE_CHUNKED", "1", 1);
            ok = ok && shelf::kernel::launch_gdn_recurrence_f32_wmma_verify(a, stream) == hipSuccess &&
                 hipStreamSynchronize(stream) == hipSuccess &&
                 hipMemcpy(got.data(), d.out, got.size() * 4u, hipMemcpyDeviceToHost) == hipSuccess;
            std::vector<int32_t> path;
            for (int32_t i = static_cast<int32_t>(n) - 1; i >= 0; i = parent[i]) path.push_back(i);
            std::reverse(path.begin(), path.end());
            ok = ok && hipMemcpy(path_dev, path.data(), path.size() * sizeof(int32_t), hipMemcpyHostToDevice) == hipSuccess;
            a.compact_rows = static_cast<uint32_t>(path.size());
            a.compact_row_indices = path_dev;
            a.compact_row_capacity = n;
            ok = ok && shelf::kernel::launch_gdn_recurrence_f32_wmma_commit(a, stream) == hipSuccess &&
                 hipStreamSynchronize(stream) == hipSuccess;
            std::vector<float> state;
            ok = ok && read_state_f16(d.state, state, kSlotStride);
            if (shape == 0u) {
                ok = ok && write_state_f16(d.state, base, state_n);
                a.row_parent = nullptr;
                a.compact_row_indices = nullptr;
                a.compact_rows = n;
                ok = ok && shelf::kernel::launch_gdn_recurrence_f32_wmma_verify(a, stream) == hipSuccess &&
                     hipStreamSynchronize(stream) == hipSuccess;
                std::vector<float> chain(got.size()), chain_state;
                ok = ok && hipMemcpy(chain.data(), d.out, chain.size() * 4u, hipMemcpyDeviceToHost) == hipSuccess &&
                     shelf::kernel::launch_gdn_recurrence_f32_wmma_commit(a, stream) == hipSuccess &&
                     hipStreamSynchronize(stream) == hipSuccess &&
                     read_state_f16(d.state, chain_state, kSlotStride);
                check(ok && std::memcmp(chain.data(), got.data(), got.size() * sizeof(float)) == 0 &&
                      chain_state == state, "tree chain matches chain chunk bitwise");
            }
            double diff = 0, peak = 0, sdiff = 0, speak = 0;
            if (ok) {
                for (size_t i = 0; i < ref.size(); ++i) {
                    diff = std::max(diff, std::abs(double(ref[i]) - got[i]));
                    peak = std::max(peak, std::abs(double(ref[i])));
                }
                for (size_t i = 0; i < state.size(); ++i) {
                    sdiff = std::max(sdiff, std::abs(double(ref_state[i]) - state[i]));
                    speak = std::max(speak, std::abs(double(ref_state[i])));
                }
            }
            std::vector<std::vector<double>> cpu_states(n, std::vector<double>(kHeadK * kHeadV));
            std::vector<double> cpu_output(n * kHeadV);
            for (uint32_t i = 0; i < n; ++i) {
                auto& dst = cpu_states[i];
                for (uint32_t d0 = 0; d0 < kHeadK; ++d0)
                    for (uint32_t v0 = 0; v0 < kHeadV; ++v0)
                        dst[d0 * kHeadV + v0] = parent[i] < 0
                            ? base[d0 * kHeadV + v0]
                            : cpu_states[parent[i]][d0 * kHeadV + v0];
                const double decay = std::exp(-std::exp(bf(hal[0])) *
                    std::log1p(std::exp(bf(ha[i * kHeads]) + bf(hdt[0]))));
                const double beta = 1.0 / (1.0 + std::exp(-bf(hb[i * kHeads])));
                for (uint32_t v0 = 0; v0 < kHeadV; ++v0) {
                    double product = 0.0;
                    for (uint32_t d0 = 0; d0 < kHeadK; ++d0)
                        product += k[i * kQk + d0] * dst[d0 * kHeadV + v0];
                    const double delta = beta * (bf(hv[i * kVdim + v0]) - decay * product);
                    double output = 0.0;
                    for (uint32_t d0 = 0; d0 < kHeadK; ++d0) {
                        auto& x = dst[d0 * kHeadV + v0];
                        x = decay * x + k[i * kQk + d0] * delta;
                        output += q[i * kQk + d0] * x;
                    }
                    cpu_output[i * kHeadV + v0] = output;
                }
            }
            double cdiff = 0, cpeak = 0, csdiff = 0, cspeak = 0;
            if (ok) {
                for (uint32_t i = 0; i < n; ++i)
                    for (uint32_t v0 = 0; v0 < kHeadV; ++v0) {
                        const double x = cpu_output[i * kHeadV + v0];
                        cdiff = std::max(cdiff, std::abs(x - got[i * kVdim + v0]));
                        cpeak = std::max(cpeak, std::abs(x));
                    }
                for (uint32_t j = 0; j < kHeadK * kHeadV; ++j) {
                    const double x = cpu_states[n - 1u][j];
                    csdiff = std::max(csdiff, std::abs(x - state[j]));
                    cspeak = std::max(cspeak, std::abs(x));
                }
            }
            std::printf("tree CPU n=%u shape=%u output=%.6g state=%.6g\n", n, shape, cdiff / cpeak, csdiff / cspeak);
            check(ok && cpeak > 0 && cspeak > 0 && cdiff / cpeak < 0.001 && csdiff / cspeak < 0.001,
                  "tree chunk output and replay match CPU");
            std::printf("tree n=%u shape=%u output=%.6g state=%.6g\n", n, shape, diff / peak, sdiff / speak);
            check(ok && peak > 0 && speak > 0 && diff / peak < 0.001 && sdiff / speak < 0.001,
                  "tree chunk output and replay match serial");
        }
    }
    unsetenv("SHELF_GDN_TREE_CHUNKED");
    unsetenv("SHELF_GDN_VERIFY_CHUNKED");
    if (parents_dev) shelf::gpu::discard_cleanup_result(hipFree(parents_dev));
    if (path_dev) shelf::gpu::discard_cleanup_result(hipFree(path_dev));
    d.free_all();
}

bool all_bits_equal(const std::vector<float>& x, const std::vector<float>& y, size_t& first) {
    if (x.size() != y.size()) return false;
    for (size_t i = 0; i < x.size(); ++i) {
        if (std::memcmp(&x[i], &y[i], 4) != 0) {
            first = i;
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    std::printf("=== GDN compact-log commit bit-exactness ===\n");
    const bool chunk_exact = std::getenv("SHELF_GDN_CHUNK_NUMERIC_EXACT") != nullptr;
    unsetenv("SHELF_GDN_RECURRENCE_EXACT");
    if (chunk_exact) setenv("SHELF_GDN_VERIFY_CHUNKED_EXACT", "1", 1);
    else unsetenv("SHELF_GDN_VERIFY_CHUNKED_EXACT");
    unsetenv("SHELF_GDN_RECURRENCE_LOSSY");

    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    if (chunk_exact) {
        test_chunk_verify(stream);
        shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));
        return failed > 0 ? 1 : 0;
    }

    if (!::shelf::kernel::gdn_recurrence_lossy_enabled()) {
        std::printf("FAIL: lossy mode expected for wmma commit path\n");
        return 1;
    }

    test_chunk_verify(stream);
    test_tree_chunk(stream);
    test_tree_chunk_batch(stream);

    Buf d;
    if (!setup(d, 20260930u)) {
        std::printf("FAIL: setup\n");
        d.free_all();
        return 1;
    }

    bool ok = true;
    const size_t state_n = (size_t)kStates * kSlotStride;
    std::vector<float> s0(state_n);
    ok = read_state_f16(d.state, s0, state_n);

    std::vector<std::vector<float>> ref_traj(kRows + 1u);
    for (auto& v : ref_traj) v.resize(state_n);
    std::vector<std::vector<float>> chain_hist(kRows);
    for (auto& v : chain_hist) v.resize(state_n);

    for (uint32_t r = 0u; r < kRows && ok; ++r) {
        ::shelf::runtime::DeviceRequestDescriptor req{};
        req.request_handle.slot = 0u;
        req.row_begin = r;
        req.row_count = 1u;
        if (hipMemcpy(d.requests, &req, sizeof(req), hipMemcpyHostToDevice) != hipSuccess)
            ok = false;
        for (uint32_t li = 0u; li < kStates && ok; ++li) {
            ::shelf::kernel::GdnRecurrenceArgs a = make_args(d);
            a.state_index = li;
            if (::shelf::kernel::launch_gdn_recurrence_f32_wmma_verify(a, stream) !=
                hipSuccess)
                ok = false;
        }
        if (ok && hipStreamSynchronize(stream) != hipSuccess) ok = false;
        if (ok)
            ok = read_state_f16(d.state, ref_traj[r + 1u], state_n);
        if (ok)
            ok = read_state_f16(d.hist + (size_t)r * kStates * kSlotStride,
                                chain_hist[r], state_n);
    }
    {
        ::shelf::runtime::DeviceRequestDescriptor req{};
        req.request_handle.slot = 0u;
        req.row_begin = 0u;
        req.row_count = kRows;
        if (hipMemcpy(d.requests, &req, sizeof(req), hipMemcpyHostToDevice) != hipSuccess)
            ok = false;
    }
    check(ok, "spill chain runs");

    {
        bool hok = true;
        size_t hfirst = 0;
        for (uint32_t r = 0u; r < kRows; ++r) {
            if (!all_bits_equal(ref_traj[r + 1u], chain_hist[r], hfirst)) {
                hok = false;
                std::printf("  [hist] r=%u first_bad=%zu ref=%.6g hist=%.6g\n", r, hfirst,
                            (double)ref_traj[r + 1u][hfirst], (double)chain_hist[r][hfirst]);
                break;
            }
        }
        check(hok, "history capture matches chain state (multi-layer)");
    }

    bool all_hist_ok = true;
    bool all_commit_ok = true;
    for (uint32_t m = 1u; m <= kRows; ++m) {
        write_state_f16(d.state, s0, state_n);
        ::shelf::kernel::GdnRecurrenceArgs c = make_args(d);
        c.compact_rows = m;
        if (::shelf::kernel::launch_gdn_recurrence_f32_wmma_commit(c, stream) != hipSuccess) {
            all_commit_ok = false;
            break;
        }
        if (hipStreamSynchronize(stream) != hipSuccess) {
            all_commit_ok = false;
            break;
        }
        std::vector<float> got(state_n);
        read_state_f16(d.state, got, state_n);
        size_t first = 0;
        if (!all_bits_equal(ref_traj[m], got, first)) {
            all_commit_ok = false;
            std::printf("  [commit] m=%u first_bad=%zu ref=%.6g got=%.6g\n", m, first,
                        (double)ref_traj[m][first], (double)got[first]);
            break;
        }
        if (m == kRows) all_hist_ok = true;
    }
    check(all_commit_ok, "commit from compact log is bit-exact vs sequential wmma");
    (void)all_hist_ok;

    std::vector<float> serial16(state_n);
    {
        write_state_f16(d.state, s0, state_n);
        bool sok = true;
        for (uint32_t li = 0u; li < kStates && sok; ++li) {
            ::shelf::kernel::GdnRecurrenceArgs a = make_args(d);
            a.state_index = li;
            if (::shelf::kernel::launch_gdn_recurrence_f32_wmma_verify(a, stream) !=
                hipSuccess)
                sok = false;
        }
        if (sok && hipStreamSynchronize(stream) != hipSuccess) sok = false;
        if (sok)
            sok = read_state_f16(d.state, serial16, state_n);
        check(sok, "wmma_verify chain runs (multi-layer)");
        size_t first = 0;
        check(all_bits_equal(ref_traj[kRows], serial16, first),
              "wmma_verify matches per-row wmma chain (multi-layer)");
    }

    {
        write_state_f16(d.state, s0, state_n);
        ::shelf::kernel::GdnRecurrenceArgs c = make_args(d);
        c.compact_rows = kRows;
        bool cok =
            ::shelf::kernel::launch_gdn_recurrence_f32_wmma_commit(c, stream) == hipSuccess;
        if (cok && hipStreamSynchronize(stream) != hipSuccess) cok = false;
        std::vector<float> got(state_n);
        if (cok)
            cok = read_state_f16(d.state, got, state_n);
        size_t first = 0;
        if (cok && !all_bits_equal(serial16, got, first)) {
            cok = false;
            std::printf("  [serial-commit] first_bad=%zu ref=%.6g got=%.6g\n", first,
                        (double)serial16[first], (double)got[first]);
        }
        check(cok, "commit reproduces wmma_verify state (multi-layer, rows=16)");
    }

    {
        int32_t idx_host[kRows];
        for (uint32_t i = 0; i < kRows; ++i) idx_host[i] = static_cast<int32_t>(i);
        int32_t* idx_dev = nullptr;
        bool p_ok = hipMalloc(&idx_dev, sizeof(idx_host)) == hipSuccess;
        if (p_ok) {
            p_ok = hipMemcpy(idx_dev, idx_host, sizeof(idx_host),
                             hipMemcpyHostToDevice) == hipSuccess;
        }
        for (uint32_t m = 1u; p_ok && m <= kRows; ++m) {
            write_state_f16(d.state, s0, state_n);
            ::shelf::kernel::GdnRecurrenceArgs c = make_args(d);
            c.compact_rows = m;
            c.compact_row_indices = idx_dev;
            c.compact_row_capacity = kRows;
            if (::shelf::kernel::launch_gdn_recurrence_f32_wmma_commit(c, stream) !=
                hipSuccess) {
                p_ok = false;
                break;
            }
            if (hipStreamSynchronize(stream) != hipSuccess) {
                p_ok = false;
                break;
            }
            std::vector<float> got(state_n);
            if (!read_state_f16(d.state, got, state_n)) {
                p_ok = false;
                break;
            }
            size_t first = 0;
            if (!all_bits_equal(ref_traj[m], got, first)) {
                p_ok = false;
                std::printf("  [prefix-idx] m=%u first_bad=%zu ref=%.6g got=%.6g\n", m,
                            first, (double)ref_traj[m][first], (double)got[first]);
                break;
            }
        }
        check(p_ok, "commit with prefix row indices matches sequential wmma");
        if (idx_dev != nullptr) shelf::gpu::discard_cleanup_result(hipFree(idx_dev));
    }

    {
        const size_t delta_n = (size_t)kStates * kRows * kVdim;
        const size_t kn_n = (size_t)kStates * kRows * kHeads * kHeadK;
        const size_t an_n = (size_t)kStates * kRows * kHeads;
        bool g_ok = true;
        std::vector<float> hd(delta_n), hkk(kn_n), haa(an_n);
        g_ok = g_ok &&
               hipMemcpy(hd.data(), d.compact_delta, delta_n * 4u,
                         hipMemcpyDeviceToHost) == hipSuccess;
        g_ok = g_ok &&
               hipMemcpy(hkk.data(), d.compact_k, kn_n * 4u,
                         hipMemcpyDeviceToHost) == hipSuccess;
        g_ok = g_ok &&
               hipMemcpy(haa.data(), d.compact_a, an_n * 4u,
                         hipMemcpyDeviceToHost) == hipSuccess;

        std::vector<float> td = hd, tk = hkk, ta = haa;
        for (uint32_t li = 0; li < kStates; ++li) {
            std::memcpy(td.data() + (size_t)li * kRows * kVdim + (size_t)kVdim,
                        hd.data() + (size_t)li * kRows * kVdim + (size_t)2 * kVdim,
                        (size_t)kVdim * 4u);
            std::memcpy(tk.data() + (size_t)li * kRows * kHeads * kHeadK +
                                (size_t)kHeads * kHeadK,
                        hkk.data() + (size_t)li * kRows * kHeads * kHeadK +
                            (size_t)2 * kHeads * kHeadK,
                        (size_t)kHeads * kHeadK * 4u);
            std::memcpy(ta.data() + (size_t)li * kRows * kHeads + (size_t)kHeads,
                        haa.data() + (size_t)li * kRows * kHeads + (size_t)2 * kHeads,
                        (size_t)kHeads * 4u);
        }

        float* td_dev = nullptr;
        float* tk_dev = nullptr;
        float* ta_dev = nullptr;
        int32_t* gap_dev = nullptr;
        g_ok = g_ok && hipMalloc(&td_dev, delta_n * 4u) == hipSuccess;
        g_ok = g_ok && hipMalloc(&tk_dev, kn_n * 4u) == hipSuccess;
        g_ok = g_ok && hipMalloc(&ta_dev, an_n * 4u) == hipSuccess;
        g_ok = g_ok && hipMalloc(&gap_dev, 2 * sizeof(int32_t)) == hipSuccess;
        if (g_ok) {
            g_ok = hipMemcpy(td_dev, td.data(), delta_n * 4u, hipMemcpyHostToDevice) ==
                       hipSuccess &&
                   hipMemcpy(tk_dev, tk.data(), kn_n * 4u, hipMemcpyHostToDevice) ==
                       hipSuccess &&
                   hipMemcpy(ta_dev, ta.data(), an_n * 4u, hipMemcpyHostToDevice) ==
                       hipSuccess;
            const int32_t gap_host[2] = {0, 2};
            g_ok = g_ok &&
                   hipMemcpy(gap_dev, gap_host, sizeof(gap_host),
                             hipMemcpyHostToDevice) == hipSuccess;
        }

        std::vector<float> got_gap(state_n);
        std::vector<float> got_temp(state_n);
        if (g_ok) {
            write_state_f16(d.state, s0, state_n);
            ::shelf::kernel::GdnRecurrenceArgs c = make_args(d);
            c.compact_rows = 2u;
            c.compact_row_indices = gap_dev;
            c.compact_row_capacity = kRows;
            g_ok = ::shelf::kernel::launch_gdn_recurrence_f32_wmma_commit(c, stream) ==
                   hipSuccess;
            g_ok = g_ok && hipStreamSynchronize(stream) == hipSuccess;
            g_ok = g_ok && read_state_f16(d.state, got_gap, state_n);
        }
        if (g_ok) {
            write_state_f16(d.state, s0, state_n);
            ::shelf::kernel::GdnRecurrenceArgs c = make_args(d);
            c.compact_rows = 2u;
            c.compact_delta = td_dev;
            c.compact_k = tk_dev;
            c.compact_a = ta_dev;
            g_ok = ::shelf::kernel::launch_gdn_recurrence_f32_wmma_commit(c, stream) ==
                   hipSuccess;
            g_ok = g_ok && hipStreamSynchronize(stream) == hipSuccess;
            g_ok = g_ok && read_state_f16(d.state, got_temp, state_n);
        }
        size_t first = 0;
        const bool eq = g_ok && all_bits_equal(got_gap, got_temp, first);
        if (g_ok && !eq) {
            std::printf("  [gap-idx] first_bad=%zu gap=%.6g temp=%.6g\n", first,
                        (double)got_gap[first], (double)got_temp[first]);
        }
        check(eq, "commit with gap row indices matches reordered log prefix");
        if (td_dev != nullptr) shelf::gpu::discard_cleanup_result(hipFree(td_dev));
        if (tk_dev != nullptr) shelf::gpu::discard_cleanup_result(hipFree(tk_dev));
        if (ta_dev != nullptr) shelf::gpu::discard_cleanup_result(hipFree(ta_dev));
        if (gap_dev != nullptr) shelf::gpu::discard_cleanup_result(hipFree(gap_dev));
    }

    constexpr uint32_t kRowsExact = 8u;
    std::vector<float> rows8(state_n);
    {
        write_state_f16(d.state, s0, state_n);
        bool sok = true;
        for (uint32_t li = 0u; li < kStates && sok; ++li) {
            ::shelf::kernel::GdnRecurrenceArgs a = make_args(d);
            a.state_index = li;
            a.actual_rows = kRowsExact;
            if (::shelf::kernel::launch_gdn_recurrence_f32_wmma_verify(a, stream) !=
                hipSuccess)
                sok = false;
        }
        if (sok && hipStreamSynchronize(stream) != hipSuccess) sok = false;
        if (sok)
            sok = read_state_f16(d.state, rows8, state_n);
        check(sok, "wmma_verify chain runs (multi-layer)");
        size_t first = 0;
        check(all_bits_equal(ref_traj[kRowsExact], rows8, first),
              "wmma_verify matches per-row wmma chain (multi-layer)");

        std::vector<std::vector<float>> rows8_hist(kRowsExact);
        for (auto& v : rows8_hist) v.resize(state_n);
        bool hok = true;
        for (uint32_t r = 0u; r < kRowsExact && hok; ++r)
            hok = read_state_f16(d.hist + (size_t)r * kStates * kSlotStride,
                                 rows8_hist[r], state_n);
        size_t hfirst = 0;
        for (uint32_t r = 0u; hok && r < kRowsExact; ++r) {
            if (!all_bits_equal(chain_hist[r], rows8_hist[r], hfirst)) {
                hok = false;
                std::printf("  [hist-exact] r=%u first_bad=%zu chain=%.6g hist=%.6g\n", r,
                            hfirst, (double)chain_hist[r][hfirst],
                            (double)rows8_hist[r][hfirst]);
            }
        }
        check(hok, "wmma_verify history capture matches chain history (multi-layer)");
    }

    {
        write_state_f16(d.state, s0, state_n);
        ::shelf::kernel::GdnRecurrenceArgs c = make_args(d);
        c.compact_rows = kRowsExact;
        bool cok =
            ::shelf::kernel::launch_gdn_recurrence_f32_wmma_commit(c, stream) == hipSuccess;
        if (cok && hipStreamSynchronize(stream) != hipSuccess) cok = false;
        std::vector<float> got(state_n);
        if (cok)
            cok = read_state_f16(d.state, got, state_n);
        size_t first = 0;
        if (cok && !all_bits_equal(rows8, got, first)) {
            cok = false;
            std::printf("  [rows-commit] first_bad=%zu ref=%.6g got=%.6g\n", first,
                        (double)rows8[first], (double)got[first]);
        }
        check(cok, "commit reproduces wmma_verify state (multi-layer, rows=8)");
    }

    d.free_all();
    shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    std::printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
