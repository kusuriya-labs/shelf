#include <phaseshift/kernels/optimized/l2_normalize.h>
#include <phaseshift/kernels/correctness/model_dispatch_correctness.h>
#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/runtime/program/device_program.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using V = ::ps::runtime::DeviceValueBinding;
using P = ::ps::runtime::DeviceProgramView;
using B = ::ps::runtime::DispatchBinding;
using DT = ::ps::runtime::ValueDType;

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

uint16_t bf16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return (uint16_t)(u >> 16);
}

struct Buf {
    void* dev = nullptr;
    size_t bytes = 0;
    ~Buf() { if (dev) ps::gpu::discard_cleanup_result(hipFree(dev)); }
    bool alloc(size_t b) { bytes = b; return hipMalloc(&dev, b) == hipSuccess; }
    bool fill_u16(const std::vector<uint16_t>& h) {
        return hipMemcpy(dev, h.data(), h.size() * 2, hipMemcpyHostToDevice) == hipSuccess;
    }
    bool read_f(std::vector<float>& o) const {
        o.resize(bytes / 4);
        return hipMemcpy(o.data(), dev, bytes, hipMemcpyDeviceToHost) == hipSuccess;
    }
};

void set_val(V& v, uint32_t id, uint32_t feat, uint32_t stride, void* ptr, DT dt) {
    v.logical_value = id;
    v.storage = ::ps::runtime::ValueStorage::WORKSPACE;
    v.dtype = dt;
    v.row_domain = ::ps::runtime::ValueRowDomain::TOKEN_ROWS;
    v.slot = 0;
    v.feature_count = feat;
    v.row_stride = stride;
    v.offset = (uint64_t)(uintptr_t)ptr;
    v.bytes = 0;
}

struct Ctx {
    ::ps::runtime::DeviceBatchContext h{};
    void* dev = nullptr;
    void* scratch = nullptr;
    void* err = nullptr;
    void* vals_dev = nullptr;
    ~Ctx() {
        void* b[] = {dev, scratch, err, vals_dev};
        for (auto* q : b)
            if (q) ps::gpu::discard_cleanup_result(hipFree(q));
    }
    bool init(uint32_t rows) {
        h = {};
        h.actual_rows = rows;
        h.num_requests = 0;
        if (hipMalloc(&dev, sizeof(h)) != hipSuccess) return false;
        if (hipMemcpy(dev, &h, sizeof(h), hipMemcpyHostToDevice) != hipSuccess) return false;
        if (hipMalloc(&scratch, (size_t)256 * 4096 * sizeof(float)) != hipSuccess) return false;
        if (hipMalloc(&err, sizeof(uint32_t)) != hipSuccess) return false;
        if (hipMalloc(&vals_dev, 2 * sizeof(V)) != hipSuccess) return false;
        return true;
    }
    bool run(const V* host_vals, const B& b, hipStream_t s) {
        if (hipMemcpy(vals_dev, host_vals, 2 * sizeof(V), hipMemcpyHostToDevice) != hipSuccess)
            return false;
        P p{};
        p.values = (const V*)vals_dev;
        p.value_count = 2;
        p.workspace_base = 0;
        ps::gpu::discard_cleanup_result(hipMemsetAsync(err, 0, sizeof(uint32_t), s));
        ::ps::kernel::ModelDispatchStateView sv{};
        auto st = ::ps::kernel::launch_model_dispatch_correctness(
            p, b, (::ps::runtime::DeviceBatchContext*)dev, sv, (float*)scratch,
            (uint32_t*)err, s);
        if (!st.ok()) return false;
        if (hipStreamSynchronize(s) != hipSuccess) return false;
        uint32_t e = 0;
        if (hipMemcpy(&e, err, 4, hipMemcpyDeviceToHost) != hipSuccess) return false;
        return e == 0u;
    }
};

enum class Fill { Normal, Zero, Small, Mixed, Large };

struct Case {
    const char* name;
    uint32_t rows;
    uint32_t f;
    uint32_t g;
    uint32_t in_stride;
    uint32_t out_stride;
    float eps;
    uint32_t seed;
    Fill fill;
};

bool run_case(const Case& c, hipStream_t stream) {
    Ctx ctx;
    if (!ctx.init(c.rows)) { std::printf("FAIL [%s] ctx\n", c.name); return false; }
    const size_t in_elems = (size_t)c.rows * c.in_stride;
    const size_t out_elems = (size_t)c.rows * c.out_stride;

    Buf in, ref, opt;
    if (!in.alloc(in_elems * 2) || !ref.alloc(out_elems * 4) || !opt.alloc(out_elems * 4)) {
        std::printf("FAIL [%s] alloc\n", c.name);
        return false;
    }

    std::vector<uint16_t> h(in_elems);
    for (uint32_t r = 0; r < c.rows; ++r) {
        for (uint32_t j = 0; j < c.f; ++j) {
            float x;
            switch (c.fill) {
                case Fill::Zero: x = 0.0f; break;
                case Fill::Small: x = (float)rng1k(c.seed + (uint64_t)r * 131u + j) / 100000.0f; break;
                case Fill::Mixed: x = (float)rng1k(c.seed + (uint64_t)r * 131u + j) / 500.0f; break;
                case Fill::Large: x = (float)rng1k(c.seed + (uint64_t)r * 131u + j) * 10.0f; break;
                default: x = (float)rng1k(c.seed + (uint64_t)r * 131u + j) / 1000.0f; break;
            }
            h[(size_t)r * c.in_stride + j] = bf16_bits(x);
        }
    }
    if (!in.fill_u16(h)) { std::printf("FAIL [%s] fill\n", c.name); return false; }

    V vals[2];
    set_val(vals[0], 0, c.f, c.in_stride, in.dev, DT::BF16);
    set_val(vals[1], 1, c.f, c.out_stride, opt.dev, DT::F32);
    B binding{};
    binding.kernel_id = ::ps::runtime::KernelId::L2_NORMALIZE;
    binding.input_slots[0] = 0;
    binding.output_slots[0] = 1;
    binding.input_count = 1;
    binding.output_count = 1;
    binding.group_size = c.g;
    binding.scalar_a = c.eps;

    set_val(vals[1], 1, c.f, c.out_stride, ref.dev, DT::F32);
    if (!ctx.run(vals, binding, stream)) { std::printf("FAIL [%s] correctness\n", c.name); return false; }

    set_val(vals[1], 1, c.f, c.out_stride, opt.dev, DT::F32);
    hipError_t h_err = ::ps::kernel::launch_l2_normalize_bf16_f32(
        (const ::ps::bf16_t*)in.dev, c.in_stride, (float*)opt.dev, c.out_stride,
        c.rows, c.f, c.g, c.eps, stream);
    if (h_err != hipSuccess) { std::printf("FAIL [%s] opt launch\n", c.name); return false; }
    if (hipStreamSynchronize(stream) != hipSuccess) { std::printf("FAIL [%s] sync\n", c.name); return false; }

    std::vector<float> rf, of;
    if (!ref.read_f(rf) || !opt.read_f(of)) { std::printf("FAIL [%s] read\n", c.name); return false; }

    double max_abs = 0.0, max_rel = 0.0;
    bool bad = false;
    for (uint32_t r = 0; r < c.rows && !bad; ++r) {
        for (uint32_t j = 0; j < c.f; ++j) {
            const size_t i = (size_t)r * c.out_stride + j;
            const double g = of[i], rr = rf[i];
            if (std::isnan((float)g) || std::isinf((float)g)) { bad = true; break; }
            const double da = std::fabs(g - rr);
            const double dr = da / std::max(1e-6, std::fabs(rr));
            max_abs = std::max(max_abs, da);
            max_rel = std::max(max_rel, dr);
            if (da > 2e-4 && dr > 5e-4) { bad = true; break; }
        }
    }
    std::printf("  [%s] rows=%u f=%u g=%u in_rs=%u out_rs=%u eps=%.1e | max_abs=%.3g max_rel=%.3g %s\n",
                c.name, c.rows, c.f, c.g, c.in_stride, c.out_stride, c.eps, max_abs, max_rel,
                bad ? "FAIL" : "PASS");
    return !bad;
}

}  // namespace

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    const Case cases[] = {
        {"r1", 1, 2048, 128, 2048, 2048, 1e-6f, 1, Fill::Normal},
        {"r2", 2, 2048, 128, 2048, 2048, 1e-6f, 2, Fill::Normal},
        {"r16", 16, 2048, 128, 2048, 2048, 1e-6f, 3, Fill::Normal},
        {"r128", 128, 2048, 128, 2048, 2048, 1e-6f, 4, Fill::Normal},
        {"r2048", 2048, 2048, 128, 2048, 2048, 1e-6f, 5, Fill::Normal},
        {"pad", 8, 2048, 128, 2064, 2064, 1e-6f, 6, Fill::Normal},
        {"zero", 8, 2048, 128, 2048, 2048, 1e-6f, 7, Fill::Zero},
        {"small", 8, 2048, 128, 2048, 2048, 1e-6f, 8, Fill::Small},
        {"mixed", 8, 2048, 128, 2048, 2048, 1e-6f, 9, Fill::Mixed},
        {"large", 8, 2048, 128, 2048, 2048, 1e-6f, 10, Fill::Large},
        {"group-tail", 4, 2048, 64, 2048, 2048, 1e-6f, 11, Fill::Normal},
        {"tp2-r1", 1, 1024, 128, 1024, 1024, 1e-6f, 12, Fill::Normal},
        {"tp2-r64", 64, 1024, 128, 1024, 1024, 1e-6f, 13, Fill::Normal},
        {"tp2-r2048", 2048, 1024, 128, 1024, 1024, 1e-6f, 14, Fill::Normal},
    };

    for (const auto& c : cases) {
        if (run_case(c, stream)) passed++;
        else failed++;
    }

    std::printf("\n=== %d passed, %d failed ===\n", passed, failed);
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return failed == 0 ? 0 : 1;
}
