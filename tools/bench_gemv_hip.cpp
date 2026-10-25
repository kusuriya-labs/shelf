// bench: bf16 GEMM config selection on the real Qwen3.5-4B decode shapes.
// Usage: bench_gemv [gpu] [rows]   -> prints us + GB/s per config per shape
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <hip/hip_runtime.h>
#include <shelf/backend/hip/kernels/linear_bf16.h>

using namespace shelf;
using namespace shelf::hipk;

struct Shape { const char* name; uint32_t N, K; };

static double time_launch(const Bf16GemmConfig& cfg, const bf16_t* w, const bf16_t* x,
                          float* of, bf16_t* ob, uint32_t rows, uint32_t N, uint32_t K,
                          hipStream_t s) {
    void* out = (cfg.id == Bf16GemmConfigId::WmmaKPartition || rows > 1) ? (void*)ob : (void*)ob;
    (void)out;
    auto call = [&](int) -> bool {
        hipError_t e = launch_gemm_bf16(cfg, w, x, ob, Bf16GemmOutputDType::BF16, rows, N, K, K,
                                        N, s);
        return e == hipSuccess;
    };
    if (!call(0)) return -1.0;
    hipStreamSynchronize(s);
    const int iters = 50;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) call(i);
    hipStreamSynchronize(s);
    auto us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
    return us;
}

int main(int argc, char** argv) {
    int gpu = argc > 1 ? std::atoi(argv[1]) : 0;
    uint32_t rows = argc > 2 ? static_cast<uint32_t>(std::atoi(argv[2])) : 1u;
    hipSetDevice(gpu);
    hipStream_t s; hipStreamCreateWithFlags(&s, hipStreamNonBlocking);

    // Qwen3.5-4B real (N,K) per step. down/o/out_proj have K != 2560.
    static const Shape shapes[] = {
        {"q_gate  8192x2560", 8192, 2560}, {"k_proj  1024x2560", 1024, 2560},
        {"v_proj  1024x2560", 1024, 2560}, {"o_proj  2560x4096", 2560, 4096},
        {"gdn_qkv 6144x2560", 6144, 2560}, {"gdn_z   4096x2560", 4096, 2560},
        {"gdn_b     32x2560", 32, 2560},   {"gdn_a     32x2560", 32, 2560},
        {"gdn_out 2560x4096", 2560, 4096}, {"mlp_gu  9216x2560", 9216, 2560},
        {"mlp_d   2560x9216", 2560, 9216}, {"lm_head 248320x2560", 248320, 2560},
    };
    static const struct { const char* n; Bf16GemmConfig c; } cfgs[] = {
        {"ExactRows",        {Bf16GemmConfigId::ExactRows, 0}},
        {"ExactRowsDot2",    {Bf16GemmConfigId::ExactRowsDot2, 0}},
        {"Wmma",             {Bf16GemmConfigId::Wmma, 0}},
        {"WmmaWide",         {Bf16GemmConfigId::WmmaWide, 0}},
        {"WmmaKPartition",   {Bf16GemmConfigId::WmmaKPartition, 0}},
    };

    std::printf("rows=%u\n%-22s", rows, "shape");
    for (auto& c : cfgs) std::printf("%16s", c.n);
    std::printf("\n");
    for (auto& sh : shapes) {
        std::vector<bf16_t> w((size_t)sh.N * sh.K, bf16_t{0x3c00});  // ~1.0
        std::vector<bf16_t> x((size_t)rows * sh.K, bf16_t{0x3c00});
        bf16_t *dw, *dx, *dob;
        hipMalloc(&dw, w.size() * 2); hipMalloc(&dx, x.size() * 2);
        hipMalloc(&dob, (size_t)rows * sh.N * 2);
        hipMemcpy(dw, w.data(), w.size() * 2, hipMemcpyHostToDevice);
        hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice);
        std::printf("%-22s", sh.name);
        const double bytes = (double)sh.N * sh.K * 2.0;
        for (auto& cc : cfgs) {
            Bf16GemmConfig cfg = cc.c;
            cfg.exact_rows = (uint8_t)rows;
            double us = time_launch(cfg, dw, dx, nullptr, dob, rows, sh.N, sh.K, s);
            if (us < 0) { std::printf("%16s", "n/s"); continue; }
            std::printf("%8.1fus/%5.0fGB", us, bytes / us / 1e3);
        }
        std::printf("\n");
        hipFree(dw); hipFree(dx); hipFree(dob);
    }
    hipStreamDestroy(s);
    return 0;
}
