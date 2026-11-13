#include <hip/hip_runtime_api.h>
#include <chrono>
#include <cstdio>
#include <dlfcn.h>

namespace {
thread_local bool active = false;
thread_local bool batch_active = false;
thread_local unsigned launches = 0;
thread_local std::chrono::steady_clock::time_point started;
thread_local std::chrono::steady_clock::time_point batch_started;
}

extern "C" hipError_t hipMemsetAsync(void* ptr, int value, size_t bytes, hipStream_t stream) {
    using Fn = hipError_t (*)(void*, int, size_t, hipStream_t);
    static auto real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "hipMemsetAsync"));
    if (bytes == sizeof(unsigned) && value == 0) {
        active = true;
        launches = 0;
        started = std::chrono::steady_clock::now();
    }
    return real(ptr, value, bytes, stream);
}

extern "C" hipError_t hipLaunchKernel(const void* function, dim3 grid, dim3 block,
                                      void** args, size_t shared, hipStream_t stream) {
    using Fn = hipError_t (*)(const void*, dim3, dim3, void**, size_t, hipStream_t);
    static auto real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "hipLaunchKernel"));
    if (active) ++launches;
    return real(function, grid, block, args, shared, stream);
}

extern "C" hipError_t hipMemcpyAsync(void* dst, const void* src, size_t bytes,
                                     hipMemcpyKind kind, hipStream_t stream) {
    using Fn = hipError_t (*)(void*, const void*, size_t, hipMemcpyKind, hipStream_t);
    static auto real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "hipMemcpyAsync"));
    if (!active && !batch_active && bytes == sizeof(unsigned) &&
        kind == hipMemcpyHostToDevice) {
        batch_active = true;
        batch_started = std::chrono::steady_clock::now();
    }
    auto status = real(dst, src, bytes, kind, stream);
    if (active && bytes == sizeof(unsigned) && kind == hipMemcpyDeviceToHost) {
        const auto end = std::chrono::steady_clock::now();
        if (launches > 100) {
            const double ms = std::chrono::duration<double, std::milli>(end - started).count();
            const double batch_ms = batch_active
                ? std::chrono::duration<double, std::milli>(end - batch_started).count() : -1;
            std::fprintf(stderr,
                         "[fixedcost_trace] launches=%u host_program_enqueue_ms=%.6f "
                         "host_batch_enqueue_ms=%.6f\n", launches, ms, batch_ms);
        }
        active = false;
        batch_active = false;
    }
    return status;
}
