#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <chrono>

namespace shelf::runtime {
hipError_t tp_flag_test_baseline(void*, const void*, std::size_t, bool, hipStream_t);
hipError_t tp_flag_test_send(const void*, void*, void*, std::uint64_t, std::size_t, hipStream_t);
hipError_t tp_flag_test_receive(void*, const void*, void*, std::uint64_t, std::size_t, bool, std::uint64_t, hipStream_t);
}

static void check(bool value, const char* msg) {
    if (!value) { std::fprintf(stderr, "%s\n", msg); std::exit(1); }
}
#define HIP(x) check((x) == hipSuccess, #x)

int main() {
    constexpr std::size_t n = 257;
    void *data[2], *inbox[2], *flags[2], *baseline, *baseline_peer, *peer_ptrs;
    HIP(hipMalloc(&baseline, n * 4));
    HIP(hipMalloc(&baseline_peer, n * 4));
    HIP(hipMalloc(&peer_ptrs, sizeof(void*)));
    HIP(hipMemcpy(peer_ptrs, &baseline_peer, sizeof(void*), hipMemcpyHostToDevice));
    hipStream_t stream[2];
    hipDeviceProp_t props{};
    HIP(hipGetDeviceProperties(&props, 0));
    for (int i = 0; i < 2; ++i) {
        HIP(hipMalloc(&data[i], n * 4));
        HIP(hipMalloc(&inbox[i], n * 4));
        HIP(hipExtMallocWithFlags(&flags[i], 16, hipDeviceMallocFinegrained));
        HIP(hipMemset(flags[i], 0, 16));
        HIP(hipStreamCreateWithFlags(&stream[i], hipStreamNonBlocking));
    }
    for (int bf = 0; bf < 2; ++bf) {
        const std::size_t bytes = n * (bf ? 2 : 4);
        for (std::uint64_t generation = 1; generation <= 1001; ++generation) {
            const std::uint64_t tag = generation + bf * 1001;
            for (int r = 0; r < 2; ++r) {
                if (bf) {
                    std::uint16_t a[n]; for (auto& x : a) x = r ? 0x4000 : 0x3f80;
                    HIP(hipMemcpyAsync(data[r], a, bytes, hipMemcpyHostToDevice, stream[r]));
                    HIP(hipStreamSynchronize(stream[r]));
                } else {
                    float a[n]; for (auto& x : a) x = r ? 2.0f : 1.0f;
                    HIP(hipMemcpyAsync(data[r], a, bytes, hipMemcpyHostToDevice, stream[r]));
                    HIP(hipStreamSynchronize(stream[r]));
                }
            }
            if (generation == 1) {
                HIP(hipMemcpy(baseline, data[0], bytes, hipMemcpyDeviceToDevice));
                HIP(hipMemcpy(baseline_peer, data[1], bytes, hipMemcpyDeviceToDevice));
                HIP(shelf::runtime::tp_flag_test_baseline(baseline, peer_ptrs, n, bf, stream[0]));
                HIP(hipStreamSynchronize(stream[0]));
            }
            HIP(shelf::runtime::tp_flag_test_send(data[0], inbox[1], flags[1], tag, bytes, stream[0]));
            HIP(shelf::runtime::tp_flag_test_receive(data[0], inbox[0], flags[0], tag, n, bf, static_cast<std::uint64_t>(props.clockRate) * 2000, stream[0]));
            if (generation == 1) std::this_thread::sleep_for(std::chrono::milliseconds(20));
            HIP(shelf::runtime::tp_flag_test_send(data[1], inbox[0], flags[0], tag, bytes, stream[1]));
            HIP(shelf::runtime::tp_flag_test_receive(data[1], inbox[1], flags[1], tag, n, bf, static_cast<std::uint64_t>(props.clockRate) * 2000, stream[1]));
            for (auto s : stream) HIP(hipStreamSynchronize(s));
            if (generation == 1) {
                unsigned char expected[n * 4], actual[n * 4];
                HIP(hipMemcpy(expected, baseline, bytes, hipMemcpyDeviceToHost));
                HIP(hipMemcpy(actual, data[0], bytes, hipMemcpyDeviceToHost));
                check(std::memcmp(expected, actual, bytes) == 0, "baseline mismatch");
            }
            for (int r = 0; r < 2; ++r) {
                if (bf) {
                    std::uint16_t a[n]; HIP(hipMemcpy(a, data[r], bytes, hipMemcpyDeviceToHost));
                    for (auto x : a) check(x == 0x4040, "BF16 sum mismatch");
                } else {
                    float a[n]; HIP(hipMemcpy(a, data[r], bytes, hipMemcpyDeviceToHost));
                    for (auto x : a) check(x == 3.0f, "F32 sum mismatch");
                }
            }
        }
    }
    HIP(shelf::runtime::tp_flag_test_receive(data[0], inbox[0], flags[0], 9000, n, false,
        static_cast<std::uint64_t>(props.clockRate) * 100, stream[0]));
    HIP(hipStreamSynchronize(stream[0]));
    unsigned int error = 0;
    HIP(hipMemcpy(&error, static_cast<char*>(flags[0]) + 8, sizeof(error), hipMemcpyDeviceToHost));
    check(error == 1, "timeout was not reported");
    for (int i = 0; i < 2; ++i) {
        HIP(hipStreamDestroy(stream[i])); HIP(hipFree(flags[i])); HIP(hipFree(inbox[i])); HIP(hipFree(data[i]));
    }
    HIP(hipFree(peer_ptrs)); HIP(hipFree(baseline_peer)); HIP(hipFree(baseline));
    std::puts("tp peer flag tests passed");
}
