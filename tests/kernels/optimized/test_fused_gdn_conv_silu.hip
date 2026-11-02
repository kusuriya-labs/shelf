#include <phaseshift/kernels/optimized/gdn/conv1d.h>
#include <phaseshift/kernels/optimized/gdn/fused_gdn_conv_silu.h>
#include <phaseshift/kernels/optimized/elementwise.h>
#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using bf16t = ps::bf16_t;

int g_passed = 0;
int g_failed = 0;

uint32_t h32(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return static_cast<uint32_t>(x);
}

int32_t rng1k(uint64_t x) { return static_cast<int32_t>(h32(x) % 2001u) - 1000; }

uint16_t bf16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return static_cast<uint16_t>(u >> 16);
}

struct Req {
    uint32_t slot;
    uint32_t row_begin;
    uint32_t row_count;
};

struct Case {
    const char* name;
    uint32_t conv_dim;
    uint32_t state_index;
    uint32_t num_gdn_states;
    uint32_t max_sequences;
    std::vector<Req> reqs;
    uint32_t rows;
    uint32_t in_stride;
    uint32_t out_stride;
    uint32_t act_stride;
    uint32_t seed;
};

struct Bufs {
    void* weight = nullptr;
    void* input = nullptr;
    void* conv_split = nullptr;
    void* conv_fused = nullptr;
    void* act_split = nullptr;
    void* act_fused = nullptr;
    void* state_split = nullptr;
    void* state_fused = nullptr;
    void* requests = nullptr;
    ~Bufs() {
        void* all[] = {weight, input, conv_split, conv_fused, act_split,
                       act_fused, state_split, state_fused, requests};
        for (auto* p : all)
            if (p) ps::gpu::discard_cleanup_result(hipFree(p));
    }
};

bool run_case(const Case& c) {
    const uint32_t conv_dim = c.conv_dim;
    const uint32_t history = 3u;
    const uint32_t hist_stride = (conv_dim + 7u) & ~7u;
    const uint64_t layer_stride = static_cast<uint64_t>(history) * hist_stride;
    const uint64_t slot_stride = static_cast<uint64_t>(c.num_gdn_states) * layer_stride;
    const uint64_t state_total = static_cast<uint64_t>(c.max_sequences) * slot_stride;
    const uint32_t in_stride = c.in_stride != 0u ? c.in_stride : conv_dim;
    const uint32_t out_stride = c.out_stride != 0u ? c.out_stride : conv_dim;
    const uint32_t act_stride = c.act_stride != 0u ? c.act_stride : conv_dim;
    const uint32_t rows = c.rows;
    const uint32_t nreq = static_cast<uint32_t>(c.reqs.size());

    Bufs buf;
    auto m = [&](void** dst, size_t bytes) { return hipMalloc(dst, bytes) == hipSuccess; };
    if (!m(&buf.weight, static_cast<size_t>(conv_dim) * 4u * 2)) return false;
    if (!m(&buf.input, static_cast<size_t>(rows) * in_stride * 2)) return false;
    if (!m(&buf.conv_split, static_cast<size_t>(rows) * out_stride * 4)) return false;
    if (!m(&buf.conv_fused, static_cast<size_t>(rows) * out_stride * 4)) return false;
    if (!m(&buf.act_split, static_cast<size_t>(rows) * act_stride * 2)) return false;
    if (!m(&buf.act_fused, static_cast<size_t>(rows) * act_stride * 2)) return false;
    if (!m(&buf.state_split, state_total * 2)) return false;
    if (!m(&buf.state_fused, state_total * 2)) return false;
    if (!m(&buf.requests,
           static_cast<size_t>(nreq) * sizeof(::ps::runtime::DeviceRequestDescriptor)))
        return false;

    std::vector<uint16_t> h_w(static_cast<size_t>(conv_dim) * 4u);
    std::vector<uint16_t> h_in(static_cast<size_t>(rows) * in_stride);
    std::vector<uint16_t> h_state(state_total);
    for (size_t i = 0; i < h_w.size(); ++i)
        h_w[i] = bf16_bits(static_cast<float>(rng1k(c.seed * 3u + i)) / 1000.0f);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t ch = 0; ch < conv_dim; ++ch) {
            float x = static_cast<float>(rng1k(c.seed * 7u + static_cast<uint64_t>(r) * 131u + ch)) / 1000.0f;
            if ((r * 131u + ch) % 509u == 0u) x = 0.0f;
            if ((r * 131u + ch) % 509u == 1u) x = 8.0f;
            if ((r * 131u + ch) % 509u == 2u) x = -8.0f;
            h_in[static_cast<size_t>(r) * in_stride + ch] = bf16_bits(x);
        }
    }
    for (size_t i = 0; i < h_state.size(); ++i)
        h_state[i] = bf16_bits(static_cast<float>(rng1k(c.seed * 11u + i)) / 512.0f);

    if (hipMemcpy(buf.weight, h_w.data(), h_w.size() * 2, hipMemcpyHostToDevice) != hipSuccess)
        return false;
    if (hipMemcpy(buf.input, h_in.data(), h_in.size() * 2, hipMemcpyHostToDevice) != hipSuccess)
        return false;
    if (hipMemcpy(buf.state_split, h_state.data(), h_state.size() * 2, hipMemcpyHostToDevice) != hipSuccess)
        return false;
    if (hipMemcpy(buf.state_fused, h_state.data(), h_state.size() * 2, hipMemcpyHostToDevice) != hipSuccess)
        return false;
    if (hipMemset(buf.conv_split, 0, static_cast<size_t>(rows) * out_stride * 4) != hipSuccess)
        return false;
    if (hipMemset(buf.conv_fused, 0, static_cast<size_t>(rows) * out_stride * 4) != hipSuccess)
        return false;
    if (hipMemset(buf.act_split, 0, static_cast<size_t>(rows) * act_stride * 2) != hipSuccess)
        return false;
    if (hipMemset(buf.act_fused, 0, static_cast<size_t>(rows) * act_stride * 2) != hipSuccess)
        return false;

    std::vector<::ps::runtime::DeviceRequestDescriptor> h_req(nreq);
    for (uint32_t r = 0; r < nreq; ++r) {
        std::memset(&h_req[r], 0, sizeof(h_req[r]));
        h_req[r].request_handle.slot = c.reqs[r].slot;
        h_req[r].row_begin = c.reqs[r].row_begin;
        h_req[r].row_count = c.reqs[r].row_count;
    }
    if (hipMemcpy(buf.requests, h_req.data(), h_req.size() * sizeof(h_req[0]),
                  hipMemcpyHostToDevice) != hipSuccess)
        return false;

    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return false;

    ::ps::kernel::GdnConv1dArgs ks{};
    ks.input = static_cast<const bf16t*>(buf.input);
    ks.output = static_cast<float*>(buf.conv_split);
    ks.conv_weight = static_cast<const bf16t*>(buf.weight);
    ks.conv_state = static_cast<bf16t*>(buf.state_split);
    ks.requests = static_cast<const ::ps::runtime::DeviceRequestDescriptor*>(buf.requests);
    ks.num_requests = nreq;
    ks.actual_rows = rows;
    ks.input_row_stride = in_stride;
    ks.output_row_stride = out_stride;
    ks.conv_dim = conv_dim;
    ks.conv_history = history;
    ks.state_index = c.state_index;
    ks.max_sequences = c.max_sequences;
    ks.conv_slot_stride = slot_stride;
    ks.conv_layer_stride = layer_stride;
    ks.conv_history_stride = hist_stride;
    hipError_t e0 = ::ps::kernel::launch_gdn_conv1d_bf16_f32(ks, stream);
    hipError_t e1 = ::ps::kernel::launch_silu_f32_bf16(
        static_cast<const float*>(buf.conv_split), out_stride,
        static_cast<bf16t*>(buf.act_split), act_stride, rows, conv_dim, stream);

    ::ps::kernel::FusedGdnConvSiluArgs kf{};
    kf.conv.input = static_cast<const bf16t*>(buf.input);
    kf.conv.output = static_cast<float*>(buf.conv_fused);
    kf.conv.conv_weight = static_cast<const bf16t*>(buf.weight);
    kf.conv.conv_state = static_cast<bf16t*>(buf.state_fused);
    kf.conv.requests = static_cast<const ::ps::runtime::DeviceRequestDescriptor*>(buf.requests);
    kf.conv.num_requests = nreq;
    kf.conv.actual_rows = rows;
    kf.conv.input_row_stride = in_stride;
    kf.conv.output_row_stride = out_stride;
    kf.conv.conv_dim = conv_dim;
    kf.conv.conv_history = history;
    kf.conv.state_index = c.state_index;
    kf.conv.max_sequences = c.max_sequences;
    kf.conv.conv_slot_stride = slot_stride;
    kf.conv.conv_layer_stride = layer_stride;
    kf.conv.conv_history_stride = hist_stride;
    kf.act_out = static_cast<bf16t*>(buf.act_fused);
    kf.act_row_stride = act_stride;
    hipError_t e2 = ::ps::kernel::launch_fused_gdn_conv_silu(kf, stream);

    const bool launch_ok =
        e0 == hipSuccess && e1 == hipSuccess && e2 == hipSuccess;
    const bool sync_ok = hipStreamSynchronize(stream) == hipSuccess;
    (void)hipStreamDestroy(stream);
    if (!launch_ok || !sync_ok) {
        std::printf("  [%s] launch fail e=%d/%d/%d\n", c.name, static_cast<int>(e0),
                    static_cast<int>(e1), static_cast<int>(e2));
        return false;
    }

    std::vector<uint8_t> conv_s(static_cast<size_t>(rows) * out_stride * 4);
    std::vector<uint8_t> conv_f(static_cast<size_t>(rows) * out_stride * 4);
    std::vector<uint8_t> act_s(static_cast<size_t>(rows) * act_stride * 2);
    std::vector<uint8_t> act_f(static_cast<size_t>(rows) * act_stride * 2);
    std::vector<uint16_t> st_s(state_total), st_f(state_total);
    if (hipMemcpy(conv_s.data(), buf.conv_split, conv_s.size(), hipMemcpyDeviceToHost) != hipSuccess)
        return false;
    if (hipMemcpy(conv_f.data(), buf.conv_fused, conv_f.size(), hipMemcpyDeviceToHost) != hipSuccess)
        return false;
    if (hipMemcpy(act_s.data(), buf.act_split, act_s.size(), hipMemcpyDeviceToHost) != hipSuccess)
        return false;
    if (hipMemcpy(act_f.data(), buf.act_fused, act_f.size(), hipMemcpyDeviceToHost) != hipSuccess)
        return false;
    if (hipMemcpy(st_s.data(), buf.state_split, st_s.size() * 2, hipMemcpyDeviceToHost) != hipSuccess)
        return false;
    if (hipMemcpy(st_f.data(), buf.state_fused, st_f.size() * 2, hipMemcpyDeviceToHost) != hipSuccess)
        return false;

    const bool conv_same = std::memcmp(conv_s.data(), conv_f.data(), conv_s.size()) == 0;
    const bool act_same = std::memcmp(act_s.data(), act_f.data(), act_s.size()) == 0;
    size_t state_diff = 0;
    for (size_t i = 0; i < st_s.size(); ++i)
        if (st_s[i] != st_f[i]) ++state_diff;
    const bool pass = conv_same && act_same && state_diff == 0u;
    size_t first = conv_s.size();
    if (!conv_same) {
        for (size_t i = 0; i < conv_s.size(); ++i) {
            if (conv_s[i] != conv_f[i]) {
                first = i;
                break;
            }
        }
    } else if (!act_same) {
        first = act_s.size();
        for (size_t i = 0; i < act_s.size(); ++i) {
            if (act_s[i] != act_f[i]) {
                first = i;
                break;
            }
        }
    }
    std::printf("  [%s] dim=%u rows=%u nreq=%u | conv=%d act=%d state_diff=%zu first=%zu %s\n",
                c.name, conv_dim, rows, nreq, static_cast<int>(conv_same),
                static_cast<int>(act_same), state_diff, first, pass ? "PASS" : "FAIL");
    return pass;
}

}

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;

    int failures = 0;
    auto run = [&](const Case& c) {
        if (!run_case(c)) ++failures;
    };

    run({"27b-decode", 10240, 0, 1, 4, {{0, 0, 1}}, 1, 0, 0, 0, 101});
    run({"27b-decode-multi", 10240, 0, 2, 4,
         {{0, 0, 1}, {1, 1, 1}, {2, 2, 1}, {3, 3, 1}}, 4, 0, 0, 0, 102});
    run({"27b-prefill", 10240, 0, 1, 1, {{0, 0, 128}}, 128, 0, 0, 0, 103});
    run({"4b-prefill", 8192, 0, 1, 1, {{0, 0, 64}}, 64, 0, 0, 0, 104});
    run({"4b-decode-multi", 8192, 0, 2, 4,
         {{0, 0, 1}, {1, 1, 1}, {2, 2, 1}, {3, 3, 1}}, 4, 0, 0, 0, 105});
    run({"state-index", 8192, 1, 3, 2, {{1, 0, 2}}, 2, 0, 0, 0, 106});
    run({"mixed-requests", 8192, 0, 1, 3,
         {{0, 0, 1}, {1, 1, 5}, {2, 6, 1}}, 7, 0, 0, 0, 108});
    run({"padded-stride", 8192, 0, 1, 2, {{0, 0, 4}}, 4, 8192 + 16, 8192 + 32, 8192 + 64, 109});
    run({"27b-tp2-decode", 5120, 0, 1, 4, {{0, 0, 1}}, 1, 0, 0, 0, 201});
    run({"27b-tp2-prefill", 5120, 0, 1, 1, {{0, 0, 128}}, 128, 0, 0, 0, 202});
    run({"27b-tp2-decode-multi", 5120, 0, 2, 4,
         {{0, 0, 1}, {1, 1, 1}, {2, 2, 1}, {3, 3, 1}}, 4, 0, 0, 0, 203});

    if (failures == 0) {
        ++g_passed;
        std::printf("test_fused_gdn_conv_silu: PASS (split == fused byte-exact)\n");
    } else {
        ++g_failed;
        std::printf("test_fused_gdn_conv_silu: FAIL\n");
    }
    std::printf("test_fused_gdn_conv_silu: passed=%d failed=%d\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
