#include <shelf/kernels/optimized/output_gather.h>
#include <shelf/kernels/correctness/model_dispatch_correctness.h>
#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/types.h>
#include <shelf/runtime/program/device_program.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using V = ::shelf::runtime::DeviceValueBinding;
using P = ::shelf::runtime::DeviceProgramView;
using B = ::shelf::runtime::DispatchBinding;

int passed = 0;
int failed = 0;

uint32_t h32(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return (uint32_t)x;
}

int32_t rng1k(uint64_t x) { return (int32_t)(h32(x) % 2001u) - 1000; }

struct Case {
    const char* name;
    uint32_t rows;
    uint32_t outputs;
    uint32_t features;
    uint32_t in_stride;
    uint32_t out_stride;
    bool f32;
    uint32_t seed;
};

bool run_case(const Case& c) {
    const uint32_t in_stride = c.in_stride != 0u ? c.in_stride : c.features;
    const uint32_t out_stride = c.out_stride != 0u ? c.out_stride : c.features;
    void* din = nullptr;
    void* dout_c = nullptr;
    void* dout_o = nullptr;
    void* drow = nullptr;
    void* dctx = nullptr;
    void* dvals = nullptr;
    void* dscratch = nullptr;
    void* derr = nullptr;
    auto cleanup = [&]() {
        void* all[] = {din, dout_c, dout_o, drow, dctx, dvals, dscratch, derr};
        for (auto* p : all)
            if (p) shelf::gpu::discard_cleanup_result(hipFree(p));
    };
    auto alloc = [&](void** dst, size_t bytes) { return hipMalloc(dst, bytes) == hipSuccess; };
    const uint32_t in_elem = c.f32 ? 4u : 2u;
    const uint32_t out_elem = in_elem;
    if (!alloc(&din, (size_t)c.rows * in_stride * in_elem) ||
        !alloc(&dout_c, (size_t)c.outputs * out_stride * out_elem) ||
        !alloc(&dout_o, (size_t)c.outputs * out_stride * out_elem) ||
        !alloc(&drow, c.outputs * 4) ||
        !alloc(&dctx, sizeof(::shelf::runtime::DeviceBatchContext)) ||
        !alloc(&dvals, 2 * sizeof(V)) ||
        !alloc(&dscratch, (size_t)256 * 4096 * sizeof(float)) ||
        !alloc(&derr, sizeof(uint32_t))) {
        cleanup();
        return false;
    }

    std::vector<uint8_t> h_in((size_t)c.rows * in_stride * in_elem);
    for (size_t i = 0; i < (size_t)c.rows * in_stride; ++i) {
        const int32_t v = rng1k(c.seed * 13u + i);
        if (c.f32) {
            const float f = v / 1000.0f;
            std::memcpy(&h_in[i * 4], &f, 4);
        } else {
            const uint16_t b = (uint16_t)((uint32_t)(v + 1000) * 13u);
            h_in[i * 2] = (uint8_t)(b & 0xFF);
            h_in[i * 2 + 1] = (uint8_t)(b >> 8);
        }
    }
    if (hipMemcpy(din, h_in.data(), h_in.size(), hipMemcpyHostToDevice) != hipSuccess) { cleanup(); return false; }
    if (hipMemset(dout_c, 0, (size_t)c.outputs * out_stride * out_elem) != hipSuccess) { cleanup(); return false; }
    if (hipMemset(dout_o, 0, (size_t)c.outputs * out_stride * out_elem) != hipSuccess) { cleanup(); return false; }

    std::vector<uint32_t> h_row(c.outputs);
    for (uint32_t i = 0; i < c.outputs; ++i)
        h_row[i] = (c.rows == 0u) ? 0u : ((i * 7u + 3u) % c.rows);
    if (hipMemcpy(drow, h_row.data(), c.outputs * 4, hipMemcpyHostToDevice) != hipSuccess) { cleanup(); return false; }

    ::shelf::runtime::DeviceBatchContext ctx{};
    ctx.actual_rows = c.rows;
    ctx.num_outputs = c.outputs;
    ctx.output_rows = (uint32_t*)drow;
    if (hipMemcpy(dctx, &ctx, sizeof(ctx), hipMemcpyHostToDevice) != hipSuccess) { cleanup(); return false; }

    const auto dt = c.f32 ? ::shelf::runtime::ValueDType::F32 : ::shelf::runtime::ValueDType::BF16;
    V vals[2];
    auto setv = [&](V& v, uint32_t id, uint32_t feat, uint32_t stride, void* ptr,
                    ::shelf::runtime::ValueDType d, ::shelf::runtime::ValueRowDomain dom) {
        v.logical_value = id;
        v.storage = ::shelf::runtime::ValueStorage::WORKSPACE;
        v.dtype = d;
        v.row_domain = dom;
        v.slot = 0;
        v.feature_count = feat;
        v.row_stride = stride;
        v.offset = (uint64_t)(uintptr_t)ptr;
        v.bytes = 0;
    };
    setv(vals[0], 0, c.features, in_stride, din, dt, ::shelf::runtime::ValueRowDomain::TOKEN_ROWS);
    setv(vals[1], 1, c.features, out_stride, dout_c, dt, ::shelf::runtime::ValueRowDomain::OUTPUT_ROWS);
    if (hipMemcpy(dvals, vals, 2 * sizeof(V), hipMemcpyHostToDevice) != hipSuccess) { cleanup(); return false; }

    P prog{};
    prog.values = (const V*)dvals;
    prog.value_count = 2;
    prog.workspace_base = 0;

    B binding{};
    binding.kernel_id = ::shelf::runtime::KernelId::OUTPUT_GATHER;
    binding.input_slots[0] = 0;
    binding.output_slots[0] = 1;
    binding.input_count = 1;
    binding.output_count = 1;

    shelf::gpu::discard_cleanup_result(hipMemset(derr, 0, sizeof(uint32_t)));
    ::shelf::kernel::ModelDispatchStateView sv{};
    auto stc = ::shelf::kernel::launch_model_dispatch_correctness(
        prog, binding, (::shelf::runtime::DeviceBatchContext*)dctx, sv, (float*)dscratch,
        (uint32_t*)derr, nullptr);
    if (!stc.ok()) { std::printf("  [%s] correctness launch failed\n", c.name); cleanup(); return false; }
    shelf::gpu::discard_cleanup_result(hipGetLastError());

    setv(vals[1], 1, c.features, out_stride, dout_o, dt, ::shelf::runtime::ValueRowDomain::OUTPUT_ROWS);
    hipError_t he = ::shelf::kernel::launch_output_gather(
        din, in_stride, dout_o, out_stride, (const ::shelf::runtime::DeviceBatchContext*)dctx,
        c.outputs, c.features, dt, nullptr);
    if (he != hipSuccess) { std::printf("  [%s] opt launch failed: %s\n", c.name, hipGetErrorString(he)); cleanup(); return false; }
    shelf::gpu::discard_cleanup_result(hipGetLastError());

    std::vector<uint8_t> rr((size_t)c.outputs * out_stride * out_elem);
    std::vector<uint8_t> ro(rr.size());
    if (hipMemcpy(rr.data(), dout_c, rr.size(), hipMemcpyDeviceToHost) != hipSuccess) { cleanup(); return false; }
    if (hipMemcpy(ro.data(), dout_o, ro.size(), hipMemcpyDeviceToHost) != hipSuccess) { cleanup(); return false; }
    size_t diff = 0;
    for (size_t i = 0; i < rr.size(); ++i)
        if (rr[i] != ro[i]) ++diff;
    const bool pass = (diff == 0u);
    std::printf("  [%s] rows=%u outputs=%u f=%u elem=%u | diff=%zu %s\n",
                c.name, c.rows, c.outputs, c.features, in_elem, diff, pass ? "PASS" : "FAIL");
    cleanup();
    return pass;
}

}  // namespace

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;

    const Case cases[] = {
        {"bf16-1", 32, 1, 5120, 0, 0, false, 1},
        {"bf16-2", 32, 2, 5120, 0, 0, false, 2},
        {"bf16-4", 64, 4, 5120, 0, 0, false, 3},
        {"bf16-pad", 32, 2, 5120, 5200, 5136, false, 4},
        {"f32-1", 32, 1, 5120, 0, 0, true, 5},
        {"f32-3", 32, 3, 2560, 0, 0, true, 6},
        {"small", 4, 2, 33, 0, 0, false, 7},
    };
    for (const auto& c : cases) {
        if (run_case(c)) ++passed;
        else ++failed;
    }
    std::printf("\n=== %d passed, %d failed ===\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
