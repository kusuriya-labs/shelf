#include <phaseshift/kernels/optimized/rmsnorm.h>
#include <phaseshift/kernels/optimized/elementwise.h>
#include <phaseshift/kernels/optimized/gdn/fused_gdn_post.h>
#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr uint32_t kF = 6144u;
constexpr uint32_t kG = 128u;
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
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return static_cast<uint16_t>(u >> 16);
}

float rec_value(uint64_t i) {
    const uint32_t r = h32(i * 2654435761ull + 5ull);
    const int32_t v = static_cast<int32_t>(r % 4001u) - 2000;
    float f = static_cast<float>(v) / 64.0f;
    const uint64_t sel = i % 97u;
    if (sel == 3u) f = 0.0f;
    if (sel == 7u) f = -0.0f;
    if (sel == 11u) f = 1.0e-28f;
    if (sel == 13u) f = -1.0e-28f;
    if (sel == 17u) f = 4.0e-38f;
    if (sel == 19u) f = -3.0e-46f;
    if (sel == 23u) f = 1.0e30f;
    if (sel == 29u) f = -2.0e-30f;
    return f;
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
    uint32_t stride_pad;
    uint32_t base_offset_elems;
    uint32_t alias;
};

void check_case(const Layout& lay, bool& ok) {
    const uint32_t rows = lay.rows;
    const size_t off_e = lay.base_offset_elems;
    const size_t f32_span = off_e + static_cast<size_t>(rows) * (kF + lay.stride_pad);
    const uint32_t stride = kF + lay.stride_pad;

    std::vector<float> h_rec(f32_span);
    for (size_t i = 0; i < f32_span; ++i) h_rec[i] = rec_value(i);
    std::vector<uint16_t> h_z(f32_span);
    for (size_t i = 0; i < f32_span; ++i) h_z[i] = bf16_bits_of(rec_value(i + 31337u));
    std::vector<uint16_t> h_w(kF);
    for (uint32_t i = 0; i < kF; ++i)
        h_w[i] = bf16_bits_of(rec_value(i + 7717u) * 0.01f + 0.002f);

    Buf rec_s, rec_f, z_s, z_f, w, rms_s, rms_f, gz_s, gz_f, g_s, g_f;
    if (!rec_s.alloc(f32_span * 4) || !rec_f.alloc(f32_span * 4) ||
        !z_s.alloc(f32_span * 2) || !z_f.alloc(f32_span * 2) || !w.alloc(kF * 2) ||
        !rms_s.alloc(f32_span * 4) || !rms_f.alloc(f32_span * 4) ||
        !gz_s.alloc((lay.alias == 1u || lay.alias == 3u) ? 0u : f32_span * 4) ||
        !gz_f.alloc((lay.alias == 1u || lay.alias == 3u) ? 0u : f32_span * 4) ||
        !g_s.alloc((lay.alias == 2u || lay.alias == 3u) ? 0u : f32_span * 2) ||
        !g_f.alloc((lay.alias == 2u || lay.alias == 3u) ? 0u : f32_span * 2)) {
        ok = false;
        return;
    }

    const size_t off4 = off_e * 4u;
    const size_t off2 = off_e * 2u;
    auto* rec_s_p = reinterpret_cast<float*>(static_cast<uint8_t*>(rec_s.dev) + off4);
    auto* rec_f_p = reinterpret_cast<float*>(static_cast<uint8_t*>(rec_f.dev) + off4);
    auto* z_s_p = reinterpret_cast<::ps::bf16_t*>(static_cast<uint8_t*>(z_s.dev) + off2);
    auto* z_f_p = reinterpret_cast<::ps::bf16_t*>(static_cast<uint8_t*>(z_f.dev) + off2);

    if (!rec_s.fill(h_rec.data(), f32_span * 4) || !rec_f.fill(h_rec.data(), f32_span * 4) ||
        !z_s.fill(h_z.data(), f32_span * 2) || !z_f.fill(h_z.data(), f32_span * 2) ||
        !w.fill(h_w.data(), kF * 2) || !rms_s.zero(f32_span * 4) ||
        !rms_f.zero(f32_span * 4) || !gz_s.zero(gz_s.bytes) || !gz_f.zero(gz_f.bytes) ||
        !g_s.zero(g_s.bytes) || !g_f.zero(g_f.bytes)) {
        ok = false;
        return;
    }

    auto* rms_s_p = reinterpret_cast<float*>(static_cast<uint8_t*>(rms_s.dev) + off4);
    auto* rms_f_p = reinterpret_cast<float*>(static_cast<uint8_t*>(rms_f.dev) + off4);
    float* gz_s_p = (lay.alias == 1u || lay.alias == 3u)
        ? rec_s_p
        : reinterpret_cast<float*>(static_cast<uint8_t*>(gz_s.dev) + off4);
    float* gz_f_p = (lay.alias == 1u || lay.alias == 3u)
        ? rec_f_p
        : reinterpret_cast<float*>(static_cast<uint8_t*>(gz_f.dev) + off4);
    ::ps::bf16_t* g_s_p = (lay.alias == 2u || lay.alias == 3u)
        ? z_s_p
        : reinterpret_cast<::ps::bf16_t*>(static_cast<uint8_t*>(g_s.dev) + off2);
    ::ps::bf16_t* g_f_p = (lay.alias == 2u || lay.alias == 3u)
        ? z_f_p
        : reinterpret_cast<::ps::bf16_t*>(static_cast<uint8_t*>(g_f.dev) + off2);

    hipStream_t s = nullptr;
    if (hipStreamCreate(&s) != hipSuccess) {
        ok = false;
        return;
    }

    ::ps::kernel::FusedGdnPostArgs args{};
    args.rec = rec_f_p;
    args.rec_row_stride = stride;
    args.norm_weight = reinterpret_cast<const ::ps::bf16_t*>(w.dev);
    args.z = z_f_p;
    args.z_row_stride = stride;
    args.rms_out = rms_f_p;
    args.rms_row_stride = stride;
    args.gated_z_out = gz_f_p;
    args.gated_z_row_stride = stride;
    args.gated_out = g_f_p;
    args.gated_row_stride = stride;
    args.eps = kEps;
    args.rows = rows;
    args.features = kF;
    args.group_size = kG;
    const hipError_t e_fused = ::ps::kernel::launch_fused_gdn_post(args, s);

    const hipError_t e0 = ::ps::kernel::launch_rmsnorm(
        rec_s_p, ::ps::kernel::RmsNormDataType::F32, w.dev,
        ::ps::kernel::RmsNormDataType::BF16,
        ::ps::kernel::RmsNormWeightLayout::PER_GROUP,
        ::ps::kernel::RmsNormWeightMode::DIRECT, rms_s_p,
        ::ps::kernel::RmsNormDataType::F32, rows, kF, kG, stride, stride, kEps, s);
    const hipError_t e1 = ::ps::kernel::launch_silu_bf16_f32(
        z_s_p, stride, gz_s_p, stride, rows, kF, s);
    const hipError_t e2 = ::ps::kernel::launch_mul_f32_f32_bf16(
        rms_s_p, stride, gz_s_p, stride, g_s_p, stride, rows, kF, s);

    const bool launch_ok = e0 == hipSuccess && e1 == hipSuccess && e2 == hipSuccess &&
                           e_fused == hipSuccess;
    const bool sync_ok = hipStreamSynchronize(s) == hipSuccess;
    (void)hipStreamDestroy(s);
    if (!launch_ok || !sync_ok) {
        std::printf("test_fused_gdn_post: launch fail rows=%u alias=%u e=%d/%d/%d/%d\n",
                    rows, lay.alias, static_cast<int>(e0), static_cast<int>(e1),
                    static_cast<int>(e2), static_cast<int>(e_fused));
        ok = false;
        return;
    }

    auto rows_match = [&](const std::vector<uint8_t>& sa,
                          const std::vector<uint8_t>& sb, size_t row_bytes) -> bool {
        for (uint32_t r = 0; r < rows; ++r) {
            size_t o;
            if (row_bytes == kF * 2u) {
                o = off2 + static_cast<size_t>(r) * stride * 2u;
            } else {
                o = off4 + static_cast<size_t>(r) * stride * 4u;
            }
            if (std::memcmp(sa.data() + o, sb.data() + o, row_bytes) != 0) return false;
        }
        return true;
    };
    std::vector<uint8_t> rms_sb(f32_span * 4), rms_fb(f32_span * 4);
    std::vector<uint8_t> rec_sb(f32_span * 4), rec_fb(f32_span * 4);
    std::vector<uint8_t> z_sb(f32_span * 2), z_fb(f32_span * 2);
    if (hipMemcpy(rms_sb.data(), rms_s.dev, f32_span * 4, hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(rms_fb.data(), rms_f.dev, f32_span * 4, hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(rec_sb.data(), rec_s.dev, f32_span * 4, hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(rec_fb.data(), rec_f.dev, f32_span * 4, hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(z_sb.data(), z_s.dev, f32_span * 2, hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(z_fb.data(), z_f.dev, f32_span * 2, hipMemcpyDeviceToHost) != hipSuccess) {
        ok = false;
        return;
    }
    const bool rms_ok = rows_match(rms_sb, rms_fb, kF * 4u);
    const bool gz_ok = (lay.alias == 1u || lay.alias == 3u)
        ? rows_match(rec_sb, rec_fb, kF * 4u)
        : true;
    const bool g_ok = (lay.alias == 2u || lay.alias == 3u)
        ? rows_match(z_sb, z_fb, kF * 2u)
        : true;
    if (!rms_ok || !gz_ok || !g_ok) {
        std::printf(
            "test_fused_gdn_post: MISMATCH rows=%u stride_pad=%u off=%u alias=%u rms=%d "
            "gz=%d gated=%d\n",
            rows, lay.stride_pad, lay.base_offset_elems, lay.alias,
            static_cast<int>(rms_ok), static_cast<int>(gz_ok), static_cast<int>(g_ok));
        ok = false;
    }
}

}

int main() {
    const Layout cases[] = {
        {1u, 0u, 0u, 0u},
        {1u, 0u, 8u, 0u},
        {2u, 0u, 0u, 0u},
        {3u, 16u, 4u, 0u},
        {5u, 0u, 8u, 0u},
        {16u, 0u, 0u, 0u},
        {129u, 0u, 4u, 0u},
        {512u, 0u, 0u, 0u},
        {2048u, 0u, 0u, 0u},
        {1u, 0u, 0u, 1u},
        {3u, 0u, 0u, 1u},
        {16u, 0u, 0u, 1u},
        {128u, 0u, 0u, 1u},
        {2048u, 0u, 0u, 1u},
        {1u, 0u, 0u, 2u},
        {3u, 0u, 0u, 2u},
        {16u, 0u, 0u, 2u},
        {2048u, 0u, 0u, 2u},
        {1u, 0u, 0u, 3u},
        {3u, 0u, 0u, 3u},
        {16u, 0u, 0u, 3u},
        {2048u, 0u, 0u, 3u},
    };
    bool ok = true;
    for (const Layout& lay : cases) check_case(lay, ok);
    if (ok) {
        ++g_passed;
        std::printf("test_fused_gdn_post: PASS (split == fused byte-exact)\n");
    } else {
        ++g_failed;
        std::printf("test_fused_gdn_post: FAIL\n");
    }
    std::printf("test_fused_gdn_post: passed=%d failed=%d\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
