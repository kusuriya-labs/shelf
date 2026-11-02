#include <phaseshift/kernels/optimized/l2_normalize.h>
#include <phaseshift/kernels/optimized/elementwise.h>
#include <phaseshift/kernels/optimized/gdn/fused_gdn_qk_prepare.h>
#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr float kEps = 1e-6f;

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

uint16_t bf16_bits_of(float f) {
    if (std::isnan(f)) return 0x7fc0u;
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return static_cast<uint16_t>(u >> 16);
}

float elem_value(uint64_t i) {
    const uint32_t r = h32(i * 2654435761ull + 77ull);
    const int32_t v = static_cast<int32_t>(r % 4001u) - 2000;
    float f = static_cast<float>(v) / 128.0f;
    if (i % 1543u == 5u) f = 0.0f;
    if (i % 1543u == 17u) f = -0.0f;
    if (i % 6091u == 9u) f = 1.0e-28f;
    return f;
}

float nasty_value(uint64_t i) {
    const uint32_t r = h32(i * 40503ull + 99991ull);
    switch (r % 16u) {
        case 0:
            return std::numeric_limits<float>::quiet_NaN();
        case 1:
            return std::numeric_limits<float>::infinity();
        case 2:
            return -std::numeric_limits<float>::infinity();
        case 3:
            return (r & 32u) ? 234.5f : -187.25f;
        case 4:
            return 1.0e-30f;
        case 5:
            return 6.55e4f;
        default:
            break;
    }
    const int32_t v = static_cast<int32_t>(r % 4001u) - 2000;
    return static_cast<float>(v) / 128.0f;
}

struct Buf {
    void* dev = nullptr;
    size_t bytes = 0;
    ~Buf() {
        if (dev) ps::gpu::discard_cleanup_result(hipFree(dev));
    }
    bool alloc(size_t b) {
        bytes = b;
        if (b == 0u) return true;
        return hipMalloc(&dev, b) == hipSuccess;
    }
    bool fill(const void* src, size_t b) {
        if (b == 0u) return true;
        return hipMemcpy(dev, src, b, hipMemcpyHostToDevice) == hipSuccess;
    }
    bool zero(size_t b) {
        if (b == 0u) return true;
        return hipMemset(dev, 0, b) == hipSuccess;
    }
};

struct Layout {
    uint32_t rows;
    uint32_t qk_features;
    uint32_t group_size;
    uint32_t stride_pad;
    uint32_t base_offset_elems;
    bool nasty_inputs;
};

void check_case(const Layout& lay, bool& ok) {
    const uint32_t rows = lay.rows;
    const uint32_t qk = lay.qk_features;
    const uint32_t group = lay.group_size;
    const uint32_t in_stride = 2u * qk + lay.stride_pad;
    const uint32_t out_stride = qk + lay.stride_pad;
    const size_t off_e = lay.base_offset_elems;
    const size_t in_elems = off_e + static_cast<size_t>(rows) * in_stride;
    const size_t out_elems = off_e + static_cast<size_t>(rows) * out_stride;
    const float scale = 1.0f / sqrtf(static_cast<float>(group));

    std::vector<uint16_t> h_act(in_elems);
    for (size_t i = 0; i < in_elems; ++i) {
        const float f = lay.nasty_inputs ? nasty_value(i) : elem_value(i);
        h_act[i] = bf16_bits_of(f);
    }

    Buf act, nq_s, nk_s, sq_s, nq_f, nk_f, sq_f;
    if (!act.alloc(in_elems * 2) || !nq_s.alloc(out_elems * 4) ||
        !nk_s.alloc(out_elems * 4) || !sq_s.alloc(out_elems * 4) ||
        !nq_f.alloc(out_elems * 4) || !nk_f.alloc(out_elems * 4) ||
        !sq_f.alloc(out_elems * 4)) {
        ok = false;
        return;
    }
    if (!act.fill(h_act.data(), in_elems * 2) || !nq_s.zero(out_elems * 4) ||
        !nk_s.zero(out_elems * 4) || !sq_s.zero(out_elems * 4) ||
        !nq_f.zero(out_elems * 4) || !nk_f.zero(out_elems * 4) ||
        !sq_f.zero(out_elems * 4)) {
        ok = false;
        return;
    }

    const size_t off2 = off_e * 2u;
    const size_t off4 = off_e * 4u;
    auto* q_p = reinterpret_cast<::ps::bf16_t*>(static_cast<uint8_t*>(act.dev) + off2);
    auto* k_p = reinterpret_cast<::ps::bf16_t*>(static_cast<uint8_t*>(act.dev) + off2 +
                                                static_cast<size_t>(qk) * 2u);
    auto* nq_s_p = reinterpret_cast<float*>(static_cast<uint8_t*>(nq_s.dev) + off4);
    auto* nk_s_p = reinterpret_cast<float*>(static_cast<uint8_t*>(nk_s.dev) + off4);
    auto* sq_s_p = reinterpret_cast<float*>(static_cast<uint8_t*>(sq_s.dev) + off4);
    auto* nq_f_p = reinterpret_cast<float*>(static_cast<uint8_t*>(nq_f.dev) + off4);
    auto* nk_f_p = reinterpret_cast<float*>(static_cast<uint8_t*>(nk_f.dev) + off4);
    auto* sq_f_p = reinterpret_cast<float*>(static_cast<uint8_t*>(sq_f.dev) + off4);

    hipStream_t s = nullptr;
    if (hipStreamCreate(&s) != hipSuccess) {
        ok = false;
        return;
    }

    const hipError_t e0 = ::ps::kernel::launch_l2_normalize_bf16_f32(
        q_p, in_stride, nq_s_p, out_stride, rows, qk, group, kEps, s);
    const hipError_t e1 = ::ps::kernel::launch_l2_normalize_bf16_f32(
        k_p, in_stride, nk_s_p, out_stride, rows, qk, group, kEps, s);
    const hipError_t e2 = ::ps::kernel::launch_scale_f32(
        nq_s_p, out_stride, sq_s_p, out_stride, rows, qk, scale, s);

    ::ps::kernel::FusedGdnQkPrepareArgs args{};
    args.q = q_p;
    args.q_row_stride = in_stride;
    args.k = k_p;
    args.k_row_stride = in_stride;
    args.nq_out = nq_f_p;
    args.nq_row_stride = out_stride;
    args.nk_out = nk_f_p;
    args.nk_row_stride = out_stride;
    args.sq_out = sq_f_p;
    args.sq_row_stride = out_stride;
    args.eps = kEps;
    args.scale = scale;
    args.rows = rows;
    args.features = qk;
    args.group_size = group;
    const hipError_t e3 = ::ps::kernel::launch_fused_gdn_qk_prepare(args, s);

    const bool launch_ok = e0 == hipSuccess && e1 == hipSuccess &&
                           e2 == hipSuccess && e3 == hipSuccess;
    const bool sync_ok = hipStreamSynchronize(s) == hipSuccess;
    (void)hipStreamDestroy(s);
    if (!launch_ok || !sync_ok) {
        std::printf("test_fused_gdn_qk_prepare: launch fail rows=%u qk=%u\n", rows, qk);
        ok = false;
        return;
    }

    std::vector<uint8_t> nq_sb(out_elems * 4), nq_fb(out_elems * 4);
    std::vector<uint8_t> nk_sb(out_elems * 4), nk_fb(out_elems * 4);
    std::vector<uint8_t> sq_sb(out_elems * 4), sq_fb(out_elems * 4);
    if (hipMemcpy(nq_sb.data(), nq_s.dev, out_elems * 4, hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(nq_fb.data(), nq_f.dev, out_elems * 4, hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(nk_sb.data(), nk_s.dev, out_elems * 4, hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(nk_fb.data(), nk_f.dev, out_elems * 4, hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(sq_sb.data(), sq_s.dev, out_elems * 4, hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(sq_fb.data(), sq_f.dev, out_elems * 4, hipMemcpyDeviceToHost) != hipSuccess) {
        ok = false;
        return;
    }
    const bool nq_same = std::memcmp(nq_sb.data(), nq_fb.data(), out_elems * 4) == 0;
    const bool nk_same = std::memcmp(nk_sb.data(), nk_fb.data(), out_elems * 4) == 0;
    const bool sq_same = std::memcmp(sq_sb.data(), sq_fb.data(), out_elems * 4) == 0;
    if (!nq_same || !nk_same || !sq_same) {
        size_t first = out_elems * 4;
        const std::vector<uint8_t>* gp = nullptr;
        const std::vector<uint8_t>* wp = nullptr;
        if (!nq_same) {
            gp = &nq_fb;
            wp = &nq_sb;
        } else if (!nk_same) {
            gp = &nk_fb;
            wp = &nk_sb;
        } else {
            gp = &sq_fb;
            wp = &sq_sb;
        }
        for (size_t i = 0; i < out_elems * 4; ++i) {
            if ((*gp)[i] != (*wp)[i]) {
                first = i;
                break;
            }
        }
        std::printf("test_fused_gdn_qk_prepare: MISMATCH rows=%u qk=%u group=%u pad=%u off=%u nasty=%d nq=%d nk=%d sq=%d first=%zu got=%02x want=%02x\n",
                    rows, qk, group, lay.stride_pad,
                    static_cast<uint32_t>(off_e), static_cast<int>(lay.nasty_inputs),
                    static_cast<int>(nq_same), static_cast<int>(nk_same),
                    static_cast<int>(sq_same), first,
                    first < out_elems * 4 ? (*gp)[first] : 0,
                    first < out_elems * 4 ? (*wp)[first] : 0);
        ok = false;
    }
}

}

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;
    bool ok = true;
    const Layout lays[] = {
        {1u, 2048u, 128u, 0u, 0u, false},
        {1u, 2048u, 128u, 0u, 8u, false},
        {2u, 2048u, 128u, 0u, 0u, false},
        {3u, 2048u, 128u, 16u, 4u, false},
        {5u, 2048u, 128u, 0u, 8u, false},
        {16u, 2048u, 128u, 0u, 0u, false},
        {129u, 2048u, 128u, 0u, 4u, false},
        {512u, 2048u, 128u, 0u, 0u, false},
        {2048u, 2048u, 128u, 0u, 0u, false},
        {1u, 1024u, 128u, 0u, 0u, false},
        {64u, 1024u, 128u, 0u, 0u, false},
        {2048u, 1024u, 128u, 0u, 0u, false},
        {1u, 2048u, 128u, 0u, 0u, true},
        {2u, 2048u, 128u, 0u, 0u, true},
        {5u, 2048u, 128u, 8u, 0u, true},
        {16u, 2048u, 128u, 0u, 8u, true},
        {129u, 2048u, 128u, 0u, 0u, true},
        {1u, 1024u, 128u, 0u, 0u, true},
        {64u, 1024u, 128u, 0u, 0u, true},
    };
    for (const Layout& lay : lays) check_case(lay, ok);
    if (ok) {
        ++g_passed;
        std::printf("test_fused_gdn_qk_prepare: PASS (split == fused byte-exact)\n");
    } else {
        ++g_failed;
        std::printf("test_fused_gdn_qk_prepare: FAIL\n");
    }
    std::printf("test_fused_gdn_qk_prepare: passed=%d failed=%d\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
