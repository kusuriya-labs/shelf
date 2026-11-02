#include <phaseshift/kernels/optimized/group_ops.h>
#include <phaseshift/kernels/correctness/model_dispatch_correctness.h>
#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/runtime/program/device_program.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
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

float uniform01(uint64_t x) {
    return static_cast<float>(static_cast<int32_t>(h32(x) % 2001u) - 1000) / 1000.0f;
}

struct Buf {
    void* dev = nullptr;
    size_t bytes = 0;
    ~Buf() { if (dev) ps::gpu::discard_cleanup_result(hipFree(dev)); }
    bool alloc(size_t b) { bytes = b; return hipMalloc(&dev, b) == hipSuccess; }
    bool fill_f(const std::vector<float>& h) {
        return hipMemcpy(dev, h.data(), h.size() * 4, hipMemcpyHostToDevice) == hipSuccess;
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
        if (hipMalloc(&vals_dev, 3 * sizeof(V)) != hipSuccess) return false;
        return true;
    }
    bool run(const V* host_vals, const B& b, hipStream_t s) {
        if (hipMemcpy(vals_dev, host_vals, 3 * sizeof(V), hipMemcpyHostToDevice) != hipSuccess)
            return false;
        P p{};
        p.values = (const V*)vals_dev;
        p.value_count = 3;
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

struct Metrics {
    double max_abs = 0.0;
    double max_rel = 0.0;
    double rmse = 0.0;
    bool non_finite = false;
};

Metrics compare(const std::vector<float>& got, const std::vector<double>& ref,
                uint32_t rows, uint32_t stride, uint32_t features) {
    Metrics m;
    double sum2 = 0.0;
    size_t count = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < features; ++c) {
            const size_t i = static_cast<size_t>(r) * stride + c;
            const double g = got[i];
            if (!std::isfinite(g)) {
                m.non_finite = true;
                return m;
            }
            const double rr = ref[static_cast<size_t>(r) * features + c];
            const double da = std::fabs(g - rr);
            const double dr = da / std::max(1e-6, std::fabs(rr));
            m.max_abs = std::max(m.max_abs, da);
            m.max_rel = std::max(m.max_rel, dr);
            sum2 += da * da;
            ++count;
        }
    }
    m.rmse = count != 0 ? std::sqrt(sum2 / static_cast<double>(count)) : 0.0;
    return m;
}

bool within(const Metrics& m, double abs_tol, double rel_tol) {
    if (m.non_finite) return false;
    return !(m.max_abs > abs_tol && m.max_rel > rel_tol);
}

struct MeanCase {
    const char* name;
    uint32_t rows;
    uint32_t features;
    uint32_t group;
    uint32_t in_stride;
    uint32_t out_stride;
    uint64_t seed;
};

bool run_mean_case(const MeanCase& c, hipStream_t stream) {
    Ctx ctx;
    if (!ctx.init(c.rows)) { std::printf("FAIL [%s] ctx\n", c.name); return false; }
    const uint32_t in_features = c.features * c.group;
    const size_t in_elems = static_cast<size_t>(c.rows) * c.in_stride;
    const size_t out_elems = static_cast<size_t>(c.rows) * c.out_stride;

    Buf in, ref, opt;
    if (!in.alloc(in_elems * 4) || !ref.alloc(out_elems * 4) || !opt.alloc(out_elems * 4)) {
        std::printf("FAIL [%s] alloc\n", c.name);
        return false;
    }

    std::vector<float> h(in_elems, 0.0f);
    for (uint32_t r = 0; r < c.rows; ++r)
        for (uint32_t j = 0; j < in_features; ++j)
            h[static_cast<size_t>(r) * c.in_stride + j] =
                uniform01(c.seed + static_cast<uint64_t>(r) * 131u + j);
    if (!in.fill_f(h)) { std::printf("FAIL [%s] fill\n", c.name); return false; }

    std::vector<double> cpu(static_cast<size_t>(c.rows) * c.features, 0.0);
    for (uint32_t r = 0; r < c.rows; ++r) {
        for (uint32_t col = 0; col < c.features; ++col) {
            double acc = 0.0;
            for (uint32_t g = 0; g < c.group; ++g) {
                acc += h[static_cast<size_t>(r) * c.in_stride + g * c.features + col];
            }
            cpu[static_cast<size_t>(r) * c.features + col] =
                acc / static_cast<double>(c.group);
        }
    }

    V vals[3];
    set_val(vals[0], 0, in_features, c.in_stride, in.dev, DT::F32);
    set_val(vals[1], 1, c.features, c.out_stride, ref.dev, DT::F32);
    B binding{};
    binding.kernel_id = ::ps::runtime::KernelId::GROUP_MEAN;
    binding.input_slots[0] = 0;
    binding.output_slots[0] = 1;
    binding.input_count = 1;
    binding.output_count = 1;
    binding.group_size = c.group;

    if (!ctx.run(vals, binding, stream)) { std::printf("FAIL [%s] correctness\n", c.name); return false; }

    set_val(vals[1], 1, c.features, c.out_stride, opt.dev, DT::F32);
    hipError_t h_err = ::ps::kernel::launch_group_mean_f32(
        (const float*)in.dev, c.in_stride, (float*)opt.dev, c.out_stride,
        c.rows, c.features, c.group, stream);
    if (h_err != hipSuccess) { std::printf("FAIL [%s] opt launch\n", c.name); return false; }
    if (hipStreamSynchronize(stream) != hipSuccess) { std::printf("FAIL [%s] sync\n", c.name); return false; }

    std::vector<float> rf, of;
    if (!ref.read_f(rf) || !opt.read_f(of)) { std::printf("FAIL [%s] read\n", c.name); return false; }

    const Metrics mr = compare(rf, cpu, c.rows, c.out_stride, c.features);
    const Metrics mo = compare(of, cpu, c.rows, c.out_stride, c.features);
    const bool ok = within(mr, 1e-5, 1e-4) && within(mo, 1e-5, 1e-4);
    std::printf(
        "  mean [%s] rows=%u f=%u g=%u | correctness max_abs=%.3g max_rel=%.3g rmse=%.3g | "
        "optimized max_abs=%.3g max_rel=%.3g rmse=%.3g %s\n",
        c.name, c.rows, c.features, c.group, mr.max_abs, mr.max_rel, mr.rmse,
        mo.max_abs, mo.max_rel, mo.rmse, ok ? "PASS" : "FAIL");
    return ok;
}

struct BroadcastCase {
    const char* name;
    uint32_t rows;
    uint32_t features;
    uint32_t group;
    uint32_t a_stride;
    uint32_t b_stride;
    uint32_t out_stride;
    uint64_t seed;
    float scale;
};

bool run_broadcast_case(const BroadcastCase& c, hipStream_t stream) {
    Ctx ctx;
    if (!ctx.init(c.rows)) { std::printf("FAIL [%s] ctx\n", c.name); return false; }
    const uint32_t out_features = c.features * c.group;
    const size_t a_elems = static_cast<size_t>(c.rows) * c.a_stride;
    const size_t b_elems = static_cast<size_t>(c.rows) * c.b_stride;
    const size_t out_elems = static_cast<size_t>(c.rows) * c.out_stride;

    Buf a, b, ref, opt;
    if (!a.alloc(a_elems * 4) || !b.alloc(b_elems * 4) ||
        !ref.alloc(out_elems * 4) || !opt.alloc(out_elems * 4)) {
        std::printf("FAIL [%s] alloc\n", c.name);
        return false;
    }

    std::vector<float> ha(a_elems, 0.0f);
    std::vector<float> hb(b_elems, 0.0f);
    for (uint32_t r = 0; r < c.rows; ++r) {
        for (uint32_t j = 0; j < c.features; ++j)
            ha[static_cast<size_t>(r) * c.a_stride + j] =
                c.scale * uniform01(c.seed + static_cast<uint64_t>(r) * 17u + j);
        for (uint32_t g = 0; g < c.group; ++g)
            hb[static_cast<size_t>(r) * c.b_stride + g] =
                c.scale * uniform01(c.seed + 100000u + static_cast<uint64_t>(r) * 31u + g);
    }
    if (!a.fill_f(ha) || !b.fill_f(hb)) { std::printf("FAIL [%s] fill\n", c.name); return false; }

    std::vector<double> cpu(static_cast<size_t>(c.rows) * out_features, 0.0);
    for (uint32_t r = 0; r < c.rows; ++r) {
        for (uint32_t g = 0; g < c.group; ++g) {
            for (uint32_t h = 0; h < c.features; ++h) {
                cpu[static_cast<size_t>(r) * out_features + g * c.features + h] =
                    static_cast<double>(ha[static_cast<size_t>(r) * c.a_stride + h]) *
                    static_cast<double>(hb[static_cast<size_t>(r) * c.b_stride + g]);
            }
        }
    }

    V vals[3];
    set_val(vals[0], 0, c.features, c.a_stride, a.dev, DT::F32);
    set_val(vals[1], 1, c.group, c.b_stride, b.dev, DT::F32);
    set_val(vals[2], 2, out_features, c.out_stride, ref.dev, DT::F32);
    B binding{};
    binding.kernel_id = ::ps::runtime::KernelId::GROUP_BROADCAST_MUL;
    binding.input_slots[0] = 0;
    binding.input_slots[1] = 1;
    binding.output_slots[0] = 2;
    binding.input_count = 2;
    binding.output_count = 1;
    binding.group_size = c.group;

    if (!ctx.run(vals, binding, stream)) { std::printf("FAIL [%s] correctness\n", c.name); return false; }

    set_val(vals[2], 2, out_features, c.out_stride, opt.dev, DT::F32);
    hipError_t h_err = ::ps::kernel::launch_group_broadcast_mul_f32(
        (const float*)a.dev, c.a_stride, (const float*)b.dev, c.b_stride,
        (float*)opt.dev, c.out_stride, c.rows, c.features, c.group, stream);
    if (h_err != hipSuccess) { std::printf("FAIL [%s] opt launch\n", c.name); return false; }
    if (hipStreamSynchronize(stream) != hipSuccess) { std::printf("FAIL [%s] sync\n", c.name); return false; }

    std::vector<float> rf, of;
    if (!ref.read_f(rf) || !opt.read_f(of)) { std::printf("FAIL [%s] read\n", c.name); return false; }

    const Metrics mr = compare(rf, cpu, c.rows, c.out_stride, out_features);
    const Metrics mo = compare(of, cpu, c.rows, c.out_stride, out_features);
    const bool ok = within(mr, 1e-4, 1e-4) && within(mo, 1e-4, 1e-4);
    std::printf(
        "  broadcast [%s] rows=%u f=%u g=%u scale=%.3g | correctness max_abs=%.3g max_rel=%.3g "
        "rmse=%.3g | optimized max_abs=%.3g max_rel=%.3g rmse=%.3g %s\n",
        c.name, c.rows, c.features, c.group, c.scale, mr.max_abs, mr.max_rel, mr.rmse,
        mo.max_abs, mo.max_rel, mo.rmse, ok ? "PASS" : "FAIL");
    return ok;
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

    const MeanCase mean_cases[] = {
        {"r1-f2560-g4", 1, 2560, 4, 10240, 2560, 1},
        {"r16-f2560-g4", 16, 2560, 4, 10240, 2560, 2},
        {"r128-f2560-g4", 128, 2560, 4, 10240, 2560, 3},
        {"r5-f96-g3", 5, 96, 3, 288, 96, 4},
        {"r8-f128-g1", 8, 128, 1, 128, 128, 5},
        {"r8-f256-g4-pad", 8, 256, 4, 1040, 272, 6},
        {"r3-f8-g4", 3, 8, 4, 32, 8, 7},
    };
    for (const auto& c : mean_cases) {
        if (run_mean_case(c, stream)) passed++;
        else failed++;
    }

    const BroadcastCase broadcast_cases[] = {
        {"r1-f2560-g4", 1, 2560, 4, 2560, 4, 10240, 11, 1.0f},
        {"r16-f2560-g4", 16, 2560, 4, 2560, 4, 10240, 12, 1.0f},
        {"r64-f2560-g4", 64, 2560, 4, 2560, 4, 10240, 13, 1.0f},
        {"r5-f96-g3", 5, 96, 3, 96, 3, 288, 14, 1.0f},
        {"r8-f256-g4-pad", 8, 256, 4, 272, 8, 1056, 15, 1.0f},
        {"r2-f8-g4", 2, 8, 4, 8, 4, 32, 16, 1.0f},
        {"r16-f2560-g4-large", 16, 2560, 4, 2560, 4, 10240, 17, 100.0f},
    };
    for (const auto& c : broadcast_cases) {
        if (run_broadcast_case(c, stream)) passed++;
        else failed++;
    }

    std::printf("\n=== %d passed, %d failed ===\n", passed, failed);
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return failed == 0 ? 0 : 1;
}
