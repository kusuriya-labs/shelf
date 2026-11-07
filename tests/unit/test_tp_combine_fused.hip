#include <shelf/core/memory/types.h>
#include <shelf/kernels/fused/tp_combine.h>

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

int g_fail = 0;

void check(bool cond, const char* what) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", what);
    }
}

shelf::bf16_t to_bf16(float value) {
    return shelf::bf16_t{
        static_cast<std::uint16_t>(__bfloat16_as_ushort(__float2bfloat16(value)))};
}

float from_bf16(shelf::bf16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value.data) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

template <typename T>
T* alloc(std::size_t count) {
    T* p = nullptr;
    if (hipMalloc(reinterpret_cast<void**>(&p), count * sizeof(T)) != hipSuccess) {
        std::printf("FAIL hipMalloc %zu\n", count * sizeof(T));
        std::exit(1);
    }
    return p;
}

template <typename T>
void upload(T* dst, const std::vector<T>& src) {
    if (hipMemcpy(dst, src.data(), src.size() * sizeof(T), hipMemcpyHostToDevice) !=
        hipSuccess) {
        std::printf("FAIL upload %zu\n", src.size());
        std::exit(1);
    }
}

template <typename T>
void download(T* dst, const T* src, std::size_t count) {
    if (hipMemcpy(dst, src, count * sizeof(T), hipMemcpyDeviceToHost) != hipSuccess) {
        std::printf("FAIL download %zu\n", count);
        std::exit(1);
    }
}

void test_residual() {
    constexpr std::uint32_t kRows = 4;
    constexpr std::uint32_t kFeatures = 16;
    std::vector<shelf::bf16_t> own(kRows * kFeatures);
    std::vector<shelf::bf16_t> peer(kRows * kFeatures);
    std::vector<shelf::bf16_t> residual(kRows * kFeatures);
    for (std::uint32_t i = 0; i < kRows * kFeatures; ++i) {
        own[i] = to_bf16(static_cast<float>(i) * 0.25f);
        peer[i] = to_bf16(static_cast<float>(i) * 0.5f);
        residual[i] = to_bf16(static_cast<float>(i) * 0.125f);
    }

    shelf::bf16_t* d_own = alloc<shelf::bf16_t>(own.size());
    shelf::bf16_t* d_peer = alloc<shelf::bf16_t>(peer.size());
    shelf::bf16_t* d_residual = alloc<shelf::bf16_t>(residual.size());
    shelf::bf16_t* d_out = alloc<shelf::bf16_t>(own.size());
    void** d_peers = alloc<void*>(1);
    upload(d_own, own);
    upload(d_peer, peer);
    upload(d_residual, residual);
    std::vector<void*> peers{static_cast<void*>(d_peer)};
    upload(d_peers, peers);

    shelf::kernel::TpSumResidualArgs args;
    args.out = d_out;
    args.own = d_own;
    args.peers = d_peers;
    args.peer_count = 1;
    args.residual = d_residual;
    args.rows = kRows;
    args.features = kFeatures;
    args.combine_stride = kFeatures;
    args.residual_stride = kFeatures;
    args.output_stride = kFeatures;
    check(shelf::kernel::launch_tp_sum_residual_bf16(args, nullptr) == hipSuccess,
          "residual launch");
    check(hipDeviceSynchronize() == hipSuccess, "residual sync");

    std::vector<shelf::bf16_t> out(own.size());
    download(out.data(), d_out, out.size());
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (std::uint32_t i = 0; i < kRows * kFeatures; ++i) {
        const float want = from_bf16(own[i]) + from_bf16(peer[i]) + from_bf16(residual[i]);
        const float error = std::fabs(from_bf16(out[i]) - want);
        max_abs = std::fmax(max_abs, error);
        max_rel = std::fmax(max_rel, error / std::fmax(1.0f, std::fabs(want)));
    }
    check(max_abs <= 0.02f || max_rel <= 0.02f, "residual matches the CPU reference");
    std::printf("residual max_abs=%g max_rel=%g\n", max_abs, max_rel);

    (void)hipFree(d_own);
    (void)hipFree(d_peer);
    (void)hipFree(d_residual);
    (void)hipFree(d_out);
    (void)hipFree(d_peers);
}

void test_hyper_apply() {
    constexpr std::uint32_t kRows = 4;
    constexpr std::uint32_t kHidden = 16;
    constexpr std::uint32_t kHc = 2;
    constexpr std::uint32_t kWidth = kHc * kHidden;
    std::vector<shelf::bf16_t> own(kRows * kHidden);
    std::vector<shelf::bf16_t> peer(kRows * kHidden);
    std::vector<shelf::bf16_t> raw(kRows * kWidth);
    std::vector<shelf::bf16_t> injection(kRows * kHc);
    for (std::uint32_t i = 0; i < kRows * kHidden; ++i) {
        own[i] = to_bf16(static_cast<float>(i) * 0.25f);
        peer[i] = to_bf16(static_cast<float>(i) * 0.5f);
    }
    for (std::uint32_t i = 0; i < kRows * kWidth; ++i) {
        raw[i] = to_bf16(static_cast<float>(i) * 0.125f);
    }
    for (std::uint32_t i = 0; i < kRows * kHc; ++i) {
        injection[i] = to_bf16(0.5f + 0.25f * static_cast<float>(i % kHc));
    }

    shelf::bf16_t* d_own = alloc<shelf::bf16_t>(own.size());
    shelf::bf16_t* d_peer = alloc<shelf::bf16_t>(peer.size());
    shelf::bf16_t* d_raw = alloc<shelf::bf16_t>(raw.size());
    shelf::bf16_t* d_injection = alloc<shelf::bf16_t>(injection.size());
    shelf::bf16_t* d_out = alloc<shelf::bf16_t>(raw.size());
    void** d_peers = alloc<void*>(1);
    upload(d_own, own);
    upload(d_peer, peer);
    upload(d_raw, raw);
    upload(d_injection, injection);
    std::vector<void*> peers{static_cast<void*>(d_peer)};
    upload(d_peers, peers);

    shelf::kernel::TpSumHyperApplyArgs args;
    args.out = d_out;
    args.own = d_own;
    args.peers = d_peers;
    args.peer_count = 1;
    args.raw = d_raw;
    args.injection = d_injection;
    args.rows = kRows;
    args.hidden = kHidden;
    args.hc = kHc;
    args.combine_stride = kHidden;
    args.raw_stride = kWidth;
    args.injection_stride = kHc;
    args.output_stride = kWidth;
    check(shelf::kernel::launch_tp_sum_hyper_apply_bf16(args, nullptr) == hipSuccess,
          "hyper apply launch");
    check(hipDeviceSynchronize() == hipSuccess, "hyper apply sync");

    std::vector<shelf::bf16_t> out(raw.size());
    download(out.data(), d_out, out.size());
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (std::uint32_t r = 0; r < kRows; ++r) {
        for (std::uint32_t h = 0; h < kHc; ++h) {
            for (std::uint32_t i = 0; i < kHidden; ++i) {
                const float y = from_bf16(own[r * kHidden + i]) +
                                from_bf16(peer[r * kHidden + i]);
                const float want = from_bf16(raw[r * kWidth + h * kHidden + i]) +
                                   y * from_bf16(injection[r * kHc + h]);
                const float error =
                    std::fabs(from_bf16(out[r * kWidth + h * kHidden + i]) - want);
                max_abs = std::fmax(max_abs, error);
                max_rel = std::fmax(max_rel, error / std::fmax(1.0f, std::fabs(want)));
            }
        }
    }
    check(max_abs <= 0.02f || max_rel <= 0.02f, "hyper apply matches the CPU reference");
    std::printf("hyper apply max_abs=%g max_rel=%g\n", max_abs, max_rel);

    (void)hipFree(d_own);
    (void)hipFree(d_peer);
    (void)hipFree(d_raw);
    (void)hipFree(d_injection);
    (void)hipFree(d_out);
    (void)hipFree(d_peers);
}

}  // namespace

int main() {
    test_residual();
    test_hyper_apply();
    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
