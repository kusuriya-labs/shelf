#include <shelf/kernels/optimized/sampling_common.h>
#include <shelf/core/gpu/cleanup.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

struct RngKey {
    uint64_t seed;
    uint64_t sample_index;
    uint64_t attempt;
    uint64_t token;
};

struct RngOut {
    uint64_t bits;
    float uniform;
};

__global__ void rng_kernel(const RngKey* __restrict__ keys, RngOut* __restrict__ out,
                           uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const RngKey k = keys[i];
    out[i].bits =
        shelf::kernel::sampling_rng_bits(k.seed, k.sample_index, k.attempt, k.token);
    out[i].uniform =
        shelf::kernel::sampling_uniform(k.seed, k.sample_index, k.attempt, k.token);
}

int g_fail = 0;

void check(bool cond, const char* msg) {
    if (!cond) ++g_fail;
    std::printf("  [%-34s] %s\n", msg, cond ? "PASS" : "FAIL");
}

}  // namespace

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;

    std::vector<RngKey> keys;
    for (uint64_t seed : {0ull, 1ull, 42ull, 0xDEADBEEFull}) {
        for (uint64_t index : {0ull, 1ull, 7ull, 1000000ull}) {
            for (uint64_t attempt : {0ull, 1ull, 5ull}) {
                for (uint64_t token : {0ull, 1ull, 3ull, 17ull, 248319ull}) {
                    keys.push_back(RngKey{seed, index, attempt, token});
                }
            }
        }
    }
    const uint32_t n = static_cast<uint32_t>(keys.size());

    RngKey* d_keys = nullptr;
    RngOut* d_out = nullptr;
    if (hipMalloc(&d_keys, sizeof(RngKey) * n) != hipSuccess ||
        hipMalloc(&d_out, sizeof(RngOut) * n) != hipSuccess) {
        std::printf("FAIL: hipMalloc\n");
        return 1;
    }
    if (hipMemcpy(d_keys, keys.data(), sizeof(RngKey) * n, hipMemcpyHostToDevice) !=
        hipSuccess) {
        std::printf("FAIL: hipMemcpy\n");
        return 1;
    }
    rng_kernel<<<(n + 255u) / 256u, 256u>>>(d_keys, d_out, n);
    if (hipDeviceSynchronize() != hipSuccess) {
        std::printf("FAIL: kernel\n");
        return 1;
    }
    std::vector<RngOut> got(n);
    if (hipMemcpy(got.data(), d_out, sizeof(RngOut) * n, hipMemcpyDeviceToHost) !=
        hipSuccess) {
        std::printf("FAIL: readback\n");
        return 1;
    }

    std::printf("host/device hash contract (%u keys)\n", n);
    uint32_t bits_mismatch = 0;
    uint32_t uniform_mismatch = 0;
    uint32_t range_bad = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const RngKey& k = keys[i];
        const uint64_t host_bits =
            shelf::kernel::sampling_rng_bits(k.seed, k.sample_index, k.attempt, k.token);
        const float host_u =
            shelf::kernel::sampling_uniform(k.seed, k.sample_index, k.attempt, k.token);
        if (host_bits != got[i].bits) ++bits_mismatch;
        if (std::memcmp(&host_u, &got[i].uniform, sizeof(float)) != 0) ++uniform_mismatch;
        if (!(got[i].uniform > 0.0f && got[i].uniform < 1.0f)) ++range_bad;
    }
    check(bits_mismatch == 0, "integer hash bit exact host/device");
    check(uniform_mismatch == 0, "uniform bit exact host/device");
    check(range_bad == 0, "uniform in (0,1)");

    const uint64_t base =
        shelf::kernel::sampling_rng_bits(7ull, 11ull, 3ull, 1234ull);
    check(shelf::kernel::sampling_rng_bits(8ull, 11ull, 3ull, 1234ull) != base,
          "seed changes stream");
    check(shelf::kernel::sampling_rng_bits(7ull, 12ull, 3ull, 1234ull) != base,
          "sample_index changes stream");
    check(shelf::kernel::sampling_rng_bits(7ull, 11ull, 4ull, 1234ull) != base,
          "attempt changes stream");
    check(shelf::kernel::sampling_rng_bits(7ull, 11ull, 3ull, 1235ull) != base,
          "token changes stream");

    check(shelf::kernel::sampling_rng_bits(7ull, 11ull, 3ull, 1234ull) == base,
          "same key reproduces bits");

    shelf::gpu::discard_cleanup_result(hipFree(d_keys));
    shelf::gpu::discard_cleanup_result(hipFree(d_out));

    std::printf("%s: %d failures\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
