#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

struct Options {
    int warmup = 3;
    int iterations = 20;
    int iters = 1000;
    int device = 0;
};

void usage(FILE* f) {
    std::fprintf(f,
                 "usage: phaseshift-bench gpu-memory [options]\n"
                 "  --warmup N\n"
                 "  --iterations N\n"
                 "  --iters N        inner loop iterations (LDS probes)\n"
                 "  --device N\n");
}

__global__ void global_copy4(float4* dst, const float4* src, std::size_t n4) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        dst[i] = src[i];
    }
}

__global__ void global_copy1(float* dst, const float* src, std::size_t n) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        dst[i] = src[i];
    }
}

__global__ void lds_scalar(float* out, int iters, int stride) {
    __shared__ float buf[4096];
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
        buf[i] = static_cast<float>(i);
    }
    __syncthreads();
    float acc = 0.0f;
    for (int it = 0; it < iters; ++it) {
        acc += buf[(threadIdx.x * stride + it) & 4095];
    }
    if (acc == 1234.5f) out[0] = acc;
}

__global__ void lds_vector(float* out, int iters, int stride) {
    __shared__ float4 buf[1024];
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
        buf[i] = make_float4(static_cast<float>(i), 1.0f, 2.0f, 3.0f);
    }
    __syncthreads();
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int it = 0; it < iters; ++it) {
        const float4 v = buf[(threadIdx.x * stride + it) & 1023];
        acc.x += v.x;
        acc.y += v.y;
        acc.z += v.z;
        acc.w += v.w;
    }
    if (acc.x == 1234.5f) out[0] = acc.x;
}

double time_kernel(hipStream_t stream, void (*launch)(void*), void* arg,
                   const Options& opt) {
    std::vector<double> samples;
    for (int it = 0; it < opt.warmup + opt.iterations; ++it) {
        hipEvent_t start = nullptr;
        hipEvent_t stop = nullptr;
        (void)hipEventCreate(&start);
        (void)hipEventCreate(&stop);
        (void)hipEventRecord(start, stream);
        launch(arg);
        (void)hipEventRecord(stop, stream);
        if (hipEventSynchronize(stop) != hipSuccess) {
            (void)hipEventDestroy(start);
            (void)hipEventDestroy(stop);
            return -1.0;
        }
        float ms = 0.0f;
        (void)hipEventElapsedTime(&ms, start, stop);
        (void)hipEventDestroy(start);
        (void)hipEventDestroy(stop);
        if (it < opt.warmup) continue;
        samples.push_back(static_cast<double>(ms) * 1000.0);
    }
    if (samples.empty()) return -1.0;
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

struct CopyArg {
    float* dst;
    const float* src;
    std::size_t n;
    std::size_t n4;
    int vec;
    hipStream_t stream;
};

void launch_copy(void* p) {
    auto* a = static_cast<CopyArg*>(p);
    if (a->vec == 4) {
        global_copy4<<<1024, 256, 0, a->stream>>>(
            reinterpret_cast<float4*>(a->dst), reinterpret_cast<const float4*>(a->src),
            a->n4);
    } else {
        global_copy1<<<1024, 256, 0, a->stream>>>(a->dst, a->src, a->n);
    }
}

}  // namespace

int run_gpu_memory(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](int& out) {
            if (i + 1 < argc) out = std::atoi(argv[++i]);
        };
        if (a == "--help" || a == "-h") {
            usage(stdout);
            return 0;
        }
        if (a == "--warmup") next(opt.warmup);
        else if (a == "--iterations") next(opt.iterations);
        else if (a == "--iters") next(opt.iters);
        else if (a == "--device") next(opt.device);
        else {
            usage(stderr);
            return 2;
        }
    }
    if (hipSetDevice(opt.device) != hipSuccess) {
        std::printf("FAIL: hipSetDevice\n");
        return 1;
    }
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    std::printf("gpu-memory global working-set sweep (copy, read+write bytes counted)\n");
    std::printf("  %-12s %-6s %12s %10s\n", "bytes", "vec", "median_us", "GB/s");
    const std::size_t sizes[] = {4096ull, 262144ull, 4194304ull, 33554432ull,
                                 134217728ull, 536870912ull};
    for (std::size_t bytes : sizes) {
        float* src = nullptr;
        float* dst = nullptr;
        if (hipMalloc(&src, bytes) != hipSuccess || hipMalloc(&dst, bytes) != hipSuccess) {
            std::printf("  %-12zu allocation failed\n", bytes);
            continue;
        }
        (void)hipMemset(src, 1, bytes);
        for (int vec : {1, 4}) {
            CopyArg arg{};
            arg.dst = dst;
            arg.src = src;
            arg.n = bytes / sizeof(float);
            arg.n4 = bytes / sizeof(float4);
            arg.vec = vec;
            arg.stream = stream;
            const double us = time_kernel(stream, launch_copy, &arg, opt);
            if (us > 0.0) {
                std::printf("  %-12zu %-6d %12.3f %10.1f\n", bytes, vec, us,
                            2.0 * static_cast<double>(bytes) / us / 1000.0);
            }
        }
        (void)hipFree(src);
        (void)hipFree(dst);
    }

    std::printf("gpu-memory LDS probes (one shared access per thread per iteration)\n");
    std::printf("  %-8s %-6s %-8s %12s %10s\n", "kernel", "stride", "threads", "median_us",
                "GB/s");
    float* out = nullptr;
    (void)hipMalloc(&out, sizeof(float));
    const int threads = 256;
    const int blocks = 32;
    for (int stride : {1, 2, 4, 8, 16, 32}) {
        hipEvent_t start = nullptr;
        hipEvent_t stop = nullptr;
        (void)hipEventCreate(&start);
        (void)hipEventCreate(&stop);
        std::vector<double> samples;
        for (int it = 0; it < opt.warmup + opt.iterations; ++it) {
            (void)hipEventRecord(start, stream);
            lds_scalar<<<blocks, threads, 0, stream>>>(out, opt.iters, stride);
            (void)hipEventRecord(stop, stream);
            (void)hipEventSynchronize(stop);
            float ms = 0.0f;
            (void)hipEventElapsedTime(&ms, start, stop);
            if (it < opt.warmup) continue;
            samples.push_back(static_cast<double>(ms) * 1000.0);
        }
        std::sort(samples.begin(), samples.end());
        const double us = samples[samples.size() / 2];
        const double bytes =
            static_cast<double>(threads) * blocks * opt.iters * sizeof(float);
        std::printf("  %-8s %-6d %-8d %12.3f %10.1f\n", "lds32", stride, threads, us,
                    bytes / us / 1000.0);
        (void)hipEventDestroy(start);
        (void)hipEventDestroy(stop);
    }
    for (int stride : {1, 2, 4, 8, 16, 32}) {
        hipEvent_t start = nullptr;
        hipEvent_t stop = nullptr;
        (void)hipEventCreate(&start);
        (void)hipEventCreate(&stop);
        std::vector<double> samples;
        for (int it = 0; it < opt.warmup + opt.iterations; ++it) {
            (void)hipEventRecord(start, stream);
            lds_vector<<<blocks, threads, 0, stream>>>(out, opt.iters, stride);
            (void)hipEventRecord(stop, stream);
            (void)hipEventSynchronize(stop);
            float ms = 0.0f;
            (void)hipEventElapsedTime(&ms, start, stop);
            if (it < opt.warmup) continue;
            samples.push_back(static_cast<double>(ms) * 1000.0);
        }
        std::sort(samples.begin(), samples.end());
        const double us = samples[samples.size() / 2];
        const double bytes =
            static_cast<double>(threads) * blocks * opt.iters * sizeof(float4);
        std::printf("  %-8s %-6d %-8d %12.3f %10.1f\n", "lds128", stride, threads, us,
                    bytes / us / 1000.0);
        (void)hipEventDestroy(start);
        (void)hipEventDestroy(stop);
    }
    (void)hipFree(out);
    (void)hipStreamDestroy(stream);
    return 0;
}
