#include <shelf/core/gpu/cleanup.h>
#include <shelf/kernels/optimized/sampling.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

int passed = 0;
int failed = 0;

void check(bool cond, const char* name) {
    std::printf("%s: %s\n", cond ? "PASS" : "FAIL", name);
    if (cond) ++passed; else ++failed;
}

bool one_case(uint32_t outputs, uint32_t V, uint32_t seed, const char* label) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-4.0f, 4.0f);
    std::vector<float> logits(static_cast<std::size_t>(outputs) * V);
    for (auto& v : logits) v = d(rng);

    float* d_logits = nullptr;
    int32_t* d_tokens = nullptr;
    ps::kernel::ArgMaxTop2* d_top2 = nullptr;
    if (hipMalloc(&d_logits, logits.size() * sizeof(float)) != hipSuccess) return false;
    if (hipMalloc(&d_tokens, outputs * sizeof(int32_t)) != hipSuccess) return false;
    if (hipMalloc(&d_top2, outputs * sizeof(ps::kernel::ArgMaxTop2)) != hipSuccess) return false;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return false;
    if (hipMemcpy(d_logits, logits.data(), logits.size() * sizeof(float),
                  hipMemcpyHostToDevice) != hipSuccess) {
        return false;
    }
    std::vector<::ps::runtime::DeviceSamplingParams> sampling(outputs);
    for (auto& sp : sampling) {
        sp = ::ps::runtime::DeviceSamplingParams{};
        sp.mode = 1u;
    }
    ::ps::runtime::DeviceSamplingParams* d_flags = nullptr;
    if (hipMalloc(&d_flags, sizeof(::ps::runtime::DeviceSamplingParams) * outputs) != hipSuccess)
        return false;
    if (hipMemcpy(d_flags, sampling.data(),
                  sizeof(::ps::runtime::DeviceSamplingParams) * outputs,
                  hipMemcpyHostToDevice) != hipSuccess)
        return false;

    ps::kernel::ArgmaxF32Args args{};
    args.logits = d_logits;
    args.sampling = d_flags;
    args.sampled_tokens = d_tokens;
    args.outputs = outputs;
    args.vocab_size = V;
    args.logits_row_stride = V;
    if (ps::kernel::launch_argmax_f32_single(args, stream) != hipSuccess) return false;
    if (ps::kernel::launch_argmax_f32_top2(d_logits, d_flags, d_top2, outputs, V, V, stream) !=
        hipSuccess) {
        return false;
    }
    if (hipStreamSynchronize(stream) != hipSuccess) return false;

    std::vector<int32_t> got_tokens(outputs);
    std::vector<ps::kernel::ArgMaxTop2> got_top2(outputs);
    if (hipMemcpy(got_tokens.data(), d_tokens, outputs * sizeof(int32_t),
                  hipMemcpyDeviceToHost) != hipSuccess) return false;
    if (hipMemcpy(got_top2.data(), d_top2, outputs * sizeof(ps::kernel::ArgMaxTop2),
                  hipMemcpyDeviceToHost) != hipSuccess) return false;

    bool ok = true;
    for (uint32_t r = 0u; r < outputs && ok; ++r) {
        const float* row = logits.data() + static_cast<std::size_t>(r) * V;
        uint32_t ref_t1 = 0u;
        float ref_v1 = row[0];
        for (uint32_t v = 1u; v < V; ++v) {
            if (row[v] > ref_v1 || (row[v] == ref_v1 && v < ref_t1)) {
                ref_v1 = row[v];
                ref_t1 = v;
            }
        }
        float ref_v2 = -3.4e38f;
        bool have2 = false;
        for (uint32_t v = 0u; v < V; ++v) {
            if (v == ref_t1) continue;
            if (!have2 || row[v] > ref_v2) {
                ref_v2 = row[v];
                have2 = true;
            }
        }
        if (static_cast<uint32_t>(got_tokens[r]) != ref_t1) {
            std::printf("  row %u: token mismatch got=%d ref=%u\n", r, got_tokens[r], ref_t1);
            ok = false;
        }
        if (got_top2[r].t1 != ref_t1) {
            std::printf("  row %u: top2.t1 mismatch got=%u ref=%u\n", r, got_top2[r].t1, ref_t1);
            ok = false;
        }
        if (got_top2[r].v1 != ref_v1) {
            std::printf("  row %u: top2.v1 mismatch\n", r);
            ok = false;
        }
        if (have2 && got_top2[r].v2 != ref_v2) {
            std::printf("  row %u: top2.v2 mismatch got=%.8g ref=%.8g\n", r, got_top2[r].v2, ref_v2);
            ok = false;
        }
    }
    (void)hipFree(d_logits);
    (void)hipFree(d_tokens);
    (void)hipFree(d_top2);
    (void)hipFree(d_flags);
    (void)hipStreamDestroy(stream);
    check(ok, label);
    return ok;
}

bool near_tie_case(const char* label) {
    const uint32_t V = 1024u;
    std::vector<float> logits(V, -10.0f);
    logits[500] = 3.0f;
    logits[700] = 3.0f;  // exact tie with token 700 > 500 -> top1 = 500
    logits[123] = 2.9999998f;
    float* d_logits = nullptr;
    ps::kernel::ArgMaxTop2* d_top2 = nullptr;
    if (hipMalloc(&d_logits, V * sizeof(float)) != hipSuccess) return false;
    if (hipMalloc(&d_top2, sizeof(ps::kernel::ArgMaxTop2)) != hipSuccess) return false;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return false;
    if (hipMemcpy(d_logits, logits.data(), V * sizeof(float), hipMemcpyHostToDevice) !=
        hipSuccess) {
        return false;
    }
    if (ps::kernel::launch_argmax_f32_top2(d_logits, nullptr, d_top2, 1u, V, V, stream) !=
        hipSuccess) {
        return false;
    }
    if (hipStreamSynchronize(stream) != hipSuccess) return false;
    ps::kernel::ArgMaxTop2 got{};
    (void)hipMemcpy(&got, d_top2, sizeof(got), hipMemcpyDeviceToHost);
    bool ok = got.t1 == 500u && got.v1 == 3.0f && got.v2 == 3.0f;
    if (!ok) std::printf("  tie: t1=%u v1=%.8g v2=%.8g\n", got.t1, got.v1, got.v2);
    (void)hipFree(d_logits);
    (void)hipFree(d_top2);
    (void)hipStreamDestroy(stream);
    check(ok, label);
    return ok;
}

}  // namespace

int main() {
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess || n == 0) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;

    one_case(1u, 248320u, 1u, "top2 vs argmax V=248320 random");
    one_case(4u, 512u, 2u, "top2 vs argmax outputs=4 V=512");
    one_case(1u, 37u, 3u, "top2 vs argmax V=37 (small)");
    one_case(1u, 256u, 4u, "top2 vs argmax V=256");
    near_tie_case("top2 exact tie -> margin 0, lower token top1");

    std::printf("MTP_ARGMAX_TOP2: %s (%d passed, %d failed)\n",
                failed == 0 ? "PASS" : "FAIL", passed, failed);
    return failed == 0 ? 0 : 1;
}
