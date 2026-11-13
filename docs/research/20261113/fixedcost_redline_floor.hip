#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

__global__ void floor_kernel(unsigned int* value) {
    *value = 1;
}

static void check(hipError_t status) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "HIP error: %s\n", hipGetErrorString(status));
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    const int count = argc > 1 ? std::atoi(argv[1]) : 547;
    if (count <= 0) return 1;
    unsigned int* value = nullptr;
    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr, stop = nullptr;
    check(hipMalloc(&value, sizeof(*value)));
    check(hipStreamCreate(&stream));
    check(hipEventCreate(&start));
    check(hipEventCreate(&stop));
    for (int run = 0; run < 4; ++run) {
        check(hipEventRecord(start, stream));
        auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < count; ++i)
            hipLaunchKernelGGL(floor_kernel, dim3(1), dim3(1), 0, stream, value);
        check(hipGetLastError());
        auto end = std::chrono::steady_clock::now();
        check(hipEventRecord(stop, stream));
        check(hipEventSynchronize(stop));
        float gpu_ms = 0;
        check(hipEventElapsedTime(&gpu_ms, start, stop));
        double host_ms = std::chrono::duration<double, std::milli>(end - begin).count();
        std::printf("{\"run\":%d,\"count\":%d,\"host_ms\":%.6f,\"gpu_ms\":%.6f}\n", run, count, host_ms, gpu_ms);
    }
    check(hipEventDestroy(start));
    check(hipEventDestroy(stop));
    check(hipStreamDestroy(stream));
    check(hipFree(value));
}
