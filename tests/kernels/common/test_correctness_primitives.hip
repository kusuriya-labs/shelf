#include <phaseshift/kernels/correctness/standalone/primitives_correctness.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/core/memory/tensor.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>
#include <cstring>
#include <algorithm>
#include <cstdint>

#define HIP_ASSERT(x) do { \
    hipError_t err = x; \
    if (err != hipSuccess) { \
        printf("HIP error at %s:%d: %d\n", __FILE__, __LINE__, err); \
        return 1; \
    } \
} while(0)

namespace {

int passed = 0;
int failed = 0;

void check(bool cond, const char* name) {
    if (cond) {
        passed++;
        printf("PASS: %s\n", name);
    } else {
        failed++;
        printf("FAIL: %s\n", name);
    }
}

ps::bf16_t f32_to_bf16(float f) {
    uint32_t ui;
    std::memcpy(&ui, &f, sizeof(float));
    uint16_t bf = static_cast<uint16_t>((ui + 0x8000U) >> 16);
    return ps::bf16_t{bf};
}

float bf16_to_f32(ps::bf16_t bf) {
    uint32_t ui = static_cast<uint32_t>(bf.data) << 16;
    float f;
    std::memcpy(&f, &ui, sizeof(float));
    return f;
}

std::vector<ps::bf16_t> rand_bf16(std::size_t n, std::mt19937& rng, float scale = 1.0f) {
    std::vector<ps::bf16_t> out(n);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = f32_to_bf16(dist(rng));
    }
    return out;
}

std::vector<ps::bf16_t> rand_pos_bf16(std::size_t n, std::mt19937& rng, float scale = 1.0f) {
    std::vector<ps::bf16_t> out(n);
    std::uniform_real_distribution<float> dist(0.01f, scale);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = f32_to_bf16(dist(rng));
    }
    return out;
}

float cpu_silu(float x) {
    return x / (1.0f + std::exp(-x));
}

std::vector<ps::bf16_t> cpu_rmsnorm_row(const ps::bf16_t* input, const ps::bf16_t* weight,
                                          std::size_t dim, double eps) {
    std::vector<ps::bf16_t> output(dim);

    float sum_sq = 0.0f;
    for (std::size_t j = 0; j < dim; ++j) {
        float v = bf16_to_f32(input[j]);
        sum_sq += v * v;
    }
    sum_sq /= (float)dim;
    float rs = 1.0f / std::sqrt(sum_sq + eps);

    for (std::size_t j = 0; j < dim; ++j) {
        float v = bf16_to_f32(input[j]);
        float w = bf16_to_f32(weight[j]);
        output[j] = f32_to_bf16(v * rs * w);
    }
    return output;
}

std::vector<ps::bf16_t> cpu_silu(const ps::bf16_t* input, std::size_t n) {
    std::vector<ps::bf16_t> output(n);
    for (std::size_t i = 0; i < n; ++i) {
        output[i] = f32_to_bf16(cpu_silu(bf16_to_f32(input[i])));
    }
    return output;
}

std::vector<ps::bf16_t> cpu_swiglu(const ps::bf16_t* gate, const ps::bf16_t* up, std::size_t n) {
    std::vector<ps::bf16_t> output(n);
    for (std::size_t i = 0; i < n; ++i) {
        output[i] = f32_to_bf16(cpu_silu(bf16_to_f32(gate[i])) * bf16_to_f32(up[i]));
    }
    return output;
}

float max_abs_diff_bf16(const ps::bf16_t* a, const ps::bf16_t* b, std::size_t n) {
    float max_diff = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        float diff = std::abs(bf16_to_f32(a[i]) - bf16_to_f32(b[i]));
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

}

int test_rmsnorm(ps::gpu::GpuArena& arena, std::mt19937& rng) {
    printf("\n--- RMSNorm Tests ---\n");

    const std::size_t seq_len = 1;
    const std::size_t hidden = 5120;
    const std::size_t total = seq_len * hidden;
    const double eps = 1e-6;

    auto inp_alloc = arena.allocate_aligned(total * sizeof(ps::bf16_t), 16);
    if (!inp_alloc.ok()) { printf("FAIL: rmsnorm input alloc\n"); return -1; }
    auto wgt_alloc = arena.allocate_aligned(hidden * sizeof(ps::bf16_t), 16);
    if (!wgt_alloc.ok()) { printf("FAIL: rmsnorm weight alloc\n"); return -1; }
    auto out_alloc = arena.allocate_aligned(total * sizeof(ps::bf16_t), 16);
    if (!out_alloc.ok()) { printf("FAIL: rmsnorm output alloc\n"); return -1; }

    auto inp_result = ps::gpu::Tensor::create<ps::bf16_t>(inp_alloc.release(), {seq_len, hidden}, {hidden, 1});
    if (!inp_result.ok()) { printf("FAIL: rmsnorm input tensor create\n"); return -1; }
    auto wgt_result = ps::gpu::Tensor::create<ps::bf16_t>(wgt_alloc.release(), {hidden}, {1});
    if (!wgt_result.ok()) { printf("FAIL: rmsnorm weight tensor create\n"); return -1; }
    auto out_result = ps::gpu::Tensor::create<ps::bf16_t>(out_alloc.release(), {seq_len, hidden}, {hidden, 1});
    if (!out_result.ok()) { printf("FAIL: rmsnorm output tensor create\n"); return -1; }

    auto inp = inp_result.release();
    auto wgt = wgt_result.release();
    auto out = out_result.release();

    auto h_input = rand_bf16(total, rng, 0.5f);
    auto h_weight = rand_pos_bf16(hidden, rng, 2.0f);

    HIP_ASSERT(hipMemcpy(inp.data<ps::bf16_t>(), h_input.data(),
                         total * sizeof(ps::bf16_t), hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(wgt.data<ps::bf16_t>(), h_weight.data(),
                         hidden * sizeof(ps::bf16_t), hipMemcpyHostToDevice));

    auto status = ps::kernel::rmsnorm(inp, wgt, out, eps, 0);
    if (!status.ok()) {
        printf("FAIL: rmsnorm kernel error: %s\n", status.message().c_str());
        return -1;
    }

    std::vector<ps::bf16_t> h_output(total);
    HIP_ASSERT(hipMemcpy(h_output.data(), out.data<ps::bf16_t>(),
                         total * sizeof(ps::bf16_t), hipMemcpyDeviceToHost));

    std::vector<ps::bf16_t> h_ref(total);
    for (std::size_t i = 0; i < seq_len; ++i) {
        auto row_ref = cpu_rmsnorm_row(h_input.data() + i * hidden, h_weight.data(), hidden, eps);
        std::copy(row_ref.begin(), row_ref.end(), h_ref.begin() + i * hidden);
    }

    float err = max_abs_diff_bf16(h_output.data(), h_ref.data(), total);
    printf("RMSNorm L-inf error: %.6f\n", err);
    check(err < 5e-2f, "rmsnorm correctness");

    return 0;
}

int test_silu(ps::gpu::GpuArena& arena, std::mt19937& rng) {
    printf("\n--- SiLU Tests ---\n");

    const std::size_t n = 17408;

    auto in_alloc = arena.allocate_aligned(n * sizeof(ps::bf16_t), 16);
    if (!in_alloc.ok()) { printf("FAIL: silu input alloc\n"); return -1; }
    auto out_alloc = arena.allocate_aligned(n * sizeof(ps::bf16_t), 16);
    if (!out_alloc.ok()) { printf("FAIL: silu output alloc\n"); return -1; }

    auto in_result = ps::gpu::Tensor::create<ps::bf16_t>(in_alloc.release(), {n}, {1});
    if (!in_result.ok()) { printf("FAIL: silu input tensor\n"); return -1; }
    auto out_result = ps::gpu::Tensor::create<ps::bf16_t>(out_alloc.release(), {n}, {1});
    if (!out_result.ok()) { printf("FAIL: silu output tensor\n"); return -1; }

    auto in_t = in_result.release();
    auto out = out_result.release();

    auto h_input = rand_bf16(n, rng, 5.0f);

    HIP_ASSERT(hipMemcpy(in_t.data<ps::bf16_t>(), h_input.data(),
                         n * sizeof(ps::bf16_t), hipMemcpyHostToDevice));

    auto status = ps::kernel::silu(in_t, out, 0);
    if (!status.ok()) {
        printf("FAIL: silu kernel error: %s\n", status.message().c_str());
        return -1;
    }

    std::vector<ps::bf16_t> h_output(n);
    HIP_ASSERT(hipMemcpy(h_output.data(), out.data<ps::bf16_t>(),
                         n * sizeof(ps::bf16_t), hipMemcpyDeviceToHost));

    auto h_ref = cpu_silu(h_input.data(), n);

    float err = max_abs_diff_bf16(h_output.data(), h_ref.data(), n);
    printf("SiLU L-inf error: %.6f\n", err);
    check(err < 1e-2f, "silu correctness");

    return 0;
}

int test_swiglu(ps::gpu::GpuArena& arena, std::mt19937& rng) {
    printf("\n--- SwiGLU Tests ---\n");

    const std::size_t n = 17408;

    auto gate_alloc = arena.allocate_aligned(n * sizeof(ps::bf16_t), 16);
    if (!gate_alloc.ok()) { printf("FAIL: swiglu gate alloc\n"); return -1; }
    auto up_alloc = arena.allocate_aligned(n * sizeof(ps::bf16_t), 16);
    if (!up_alloc.ok()) { printf("FAIL: swiglu up alloc\n"); return -1; }
    auto out_alloc = arena.allocate_aligned(n * sizeof(ps::bf16_t), 16);
    if (!out_alloc.ok()) { printf("FAIL: swiglu output alloc\n"); return -1; }

    auto gate_result = ps::gpu::Tensor::create<ps::bf16_t>(gate_alloc.release(), {n}, {1});
    if (!gate_result.ok()) { printf("FAIL: swiglu gate tensor\n"); return -1; }
    auto up_result = ps::gpu::Tensor::create<ps::bf16_t>(up_alloc.release(), {n}, {1});
    if (!up_result.ok()) { printf("FAIL: swiglu up tensor\n"); return -1; }
    auto out_result = ps::gpu::Tensor::create<ps::bf16_t>(out_alloc.release(), {n}, {1});
    if (!out_result.ok()) { printf("FAIL: swiglu output tensor\n"); return -1; }

    auto gate = gate_result.release();
    auto up = up_result.release();
    auto out = out_result.release();

    auto h_gate = rand_bf16(n, rng, 5.0f);
    auto h_up = rand_bf16(n, rng, 5.0f);

    HIP_ASSERT(hipMemcpy(gate.data<ps::bf16_t>(), h_gate.data(),
                         n * sizeof(ps::bf16_t), hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(up.data<ps::bf16_t>(), h_up.data(),
                         n * sizeof(ps::bf16_t), hipMemcpyHostToDevice));

    auto status = ps::kernel::swiglu(gate, up, out, 0);
    if (!status.ok()) {
        printf("FAIL: swiglu kernel error: %s\n", status.message().c_str());
        return -1;
    }

    std::vector<ps::bf16_t> h_output(n);
    HIP_ASSERT(hipMemcpy(h_output.data(), out.data<ps::bf16_t>(),
                         n * sizeof(ps::bf16_t), hipMemcpyDeviceToHost));

    auto h_ref = cpu_swiglu(h_gate.data(), h_up.data(), n);

    float err = max_abs_diff_bf16(h_output.data(), h_ref.data(), n);
    printf("SwiGLU L-inf error: %.6f\n", err);
    check(err < 1e-2f, "swiglu correctness");

    return 0;
}

int main() {
    printf("=== Kernel Primitives Tests ===\n");

    int device_count = 0;
    HIP_ASSERT(hipGetDeviceCount(&device_count));

    if (device_count == 0) {
        printf("SKIP: no GPU available\n");
        return 77;
    }

    HIP_ASSERT(hipSetDevice(0));

    auto arena_result = ps::gpu::GpuArena::create(0, 64 * 1024 * 1024);
    if (!arena_result.ok()) {
        printf("FAIL: arena create: %s\n", arena_result.status().message().c_str());
        return 1;
    }
    auto arena = arena_result.release();

    std::mt19937 rng(42);

    if (test_rmsnorm(arena, rng) != 0) {
        printf("\nResults: %d passed, %d failed\n", passed, failed);
        return 1;
    }
    if (test_silu(arena, rng) != 0) {
        printf("\nResults: %d passed, %d failed\n", passed, failed);
        return 1;
    }
    if (test_swiglu(arena, rng) != 0) {
        printf("\nResults: %d passed, %d failed\n", passed, failed);
        return 1;
    }

    printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
