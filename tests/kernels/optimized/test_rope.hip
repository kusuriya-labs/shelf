#include <shelf/kernels/optimized/attention/rope.h>
#include <shelf/kernels/correctness/model_dispatch_correctness.h>
#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/types.h>
#include <shelf/runtime/program/device_program.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using bf16t = shelf::bf16_t;
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

float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

enum class PosMode { None, Row, Rope };

struct Case {
    const char* name;
    uint32_t rows;
    uint32_t features;
    uint32_t head_dim;
    uint32_t rotary;
    uint32_t in_stride;
    uint32_t out_stride;
    float theta;
    uint32_t seed;
    PosMode mode;
};

bool run_case(const Case& c) {
    const uint32_t rows = c.rows;
    const uint32_t in_stride = c.in_stride != 0u ? c.in_stride : c.features;
    const uint32_t out_stride = c.out_stride != 0u ? c.out_stride : c.features;

    void* din = nullptr;
    void* dout_c = nullptr;
    void* dout_o = nullptr;
    void* dpos_rope = nullptr;
    void* dpos_row = nullptr;
    void* dctx = nullptr;
    void* dvals = nullptr;
    void* dscratch = nullptr;
    void* derr = nullptr;
    void* dinv = nullptr;
    auto cleanup = [&]() {
        void* all[] = {din, dout_c, dout_o, dpos_rope, dpos_row, dctx, dvals, dscratch,
                       derr, dinv};
        for (auto* p : all)
            if (p) shelf::gpu::discard_cleanup_result(hipFree(p));
    };
    auto alloc = [&](void** dst, size_t bytes) { return hipMalloc(dst, bytes) == hipSuccess; };
    if (!alloc(&din, (size_t)rows * in_stride * 4) ||
        !alloc(&dout_c, (size_t)rows * out_stride * 2) ||
        !alloc(&dout_o, (size_t)rows * out_stride * 2) ||
        !alloc(&dctx, sizeof(::shelf::runtime::DeviceBatchContext)) ||
        !alloc(&dvals, 2 * sizeof(V)) ||
        !alloc(&dscratch, (size_t)256 * 4096 * sizeof(float)) ||
        !alloc(&derr, sizeof(uint32_t)) ||
        !alloc(&dinv, 32 * sizeof(double))) {
        cleanup();
        return false;
    }
    if (::shelf::kernel::launch_rope_inv_freq_init((double*)dinv, 32, c.theta, nullptr)
        != hipSuccess) {
        cleanup();
        return false;
    }

    std::vector<float> h_in((size_t)rows * in_stride);
    for (size_t i = 0; i < h_in.size(); ++i)
        h_in[i] = rng1k(c.seed * 13u + i) / 1000.0f;
    if (hipMemcpy(din, h_in.data(), h_in.size() * 4, hipMemcpyHostToDevice) != hipSuccess) { cleanup(); return false; }
    if (hipMemset(dout_c, 0, (size_t)rows * out_stride * 2) != hipSuccess) { cleanup(); return false; }
    if (hipMemset(dout_o, 0, (size_t)rows * out_stride * 2) != hipSuccess) { cleanup(); return false; }

    std::vector<uint32_t> h_pos(rows);
    for (uint32_t r = 0; r < rows; ++r) {
        const uint32_t special[] = {0u, 1u, 127u, 2047u};
        h_pos[r] = (r < 4u) ? special[r] : (r * 37u + 5u) % 4096u;
    }
    const uint32_t* positions = nullptr;
    if (c.mode != PosMode::None) {
        if (hipMalloc(&dpos_row, rows * 4) != hipSuccess) { cleanup(); return false; }
        if (hipMemcpy(dpos_row, h_pos.data(), rows * 4, hipMemcpyHostToDevice) != hipSuccess) { cleanup(); return false; }
        positions = (const uint32_t*)dpos_row;
    }
    if (c.mode == PosMode::Rope) {
        if (hipMalloc(&dpos_rope, rows * 4) != hipSuccess) { cleanup(); return false; }
        if (hipMemcpy(dpos_rope, h_pos.data(), rows * 4, hipMemcpyHostToDevice) != hipSuccess) { cleanup(); return false; }
        positions = (const uint32_t*)dpos_rope;
    }

    ::shelf::runtime::DeviceBatchContext ctx{};
    ctx.actual_rows = rows;
    ctx.num_requests = 0;
    ctx.rope_positions = (c.mode == PosMode::Rope) ? (const uint32_t*)dpos_rope : nullptr;
    ctx.row_positions = (c.mode == PosMode::Row) ? (uint32_t*)dpos_row : nullptr;
    if (c.rotary / 2u <= 32u) ctx.rope_inv_freq = (const double*)dinv;
    if (hipMemcpy(dctx, &ctx, sizeof(ctx), hipMemcpyHostToDevice) != hipSuccess) { cleanup(); return false; }

    V vals[2];
    auto setv = [&](V& v, uint32_t id, uint32_t feat, uint32_t stride, void* ptr,
                    ::shelf::runtime::ValueDType dt) {
        v.logical_value = id;
        v.storage = ::shelf::runtime::ValueStorage::WORKSPACE;
        v.dtype = dt;
        v.row_domain = ::shelf::runtime::ValueRowDomain::TOKEN_ROWS;
        v.slot = 0;
        v.feature_count = feat;
        v.row_stride = stride;
        v.offset = (uint64_t)(uintptr_t)ptr;
        v.bytes = 0;
    };
    setv(vals[0], 0, c.features, in_stride, din, ::shelf::runtime::ValueDType::F32);
    setv(vals[1], 1, c.features, out_stride, dout_c, ::shelf::runtime::ValueDType::BF16);
    if (hipMemcpy(dvals, vals, 2 * sizeof(V), hipMemcpyHostToDevice) != hipSuccess) { cleanup(); return false; }

    P prog{};
    prog.values = (const V*)dvals;
    prog.value_count = 2;
    prog.workspace_base = 0;

    B binding{};
    binding.kernel_id = ::shelf::runtime::KernelId::ROPE;
    binding.input_slots[0] = 0;
    binding.output_slots[0] = 1;
    binding.input_count = 1;
    binding.output_count = 1;
    binding.group_size = c.head_dim;
    binding.rotary_dim = c.rotary;
    binding.scalar_b = c.theta;

    shelf::gpu::discard_cleanup_result(hipMemset(derr, 0, sizeof(uint32_t)));
    ::shelf::kernel::ModelDispatchStateView sv{};
    auto stc = ::shelf::kernel::launch_model_dispatch_correctness(
        prog, binding, (::shelf::runtime::DeviceBatchContext*)dctx, sv, (float*)dscratch,
        (uint32_t*)derr, nullptr);
    if (!stc.ok()) { std::printf("  [%s] correctness launch failed\n", c.name); cleanup(); return false; }
    shelf::gpu::discard_cleanup_result(hipGetLastError());

    hipError_t he = ::shelf::kernel::launch_rope_f32_bf16(
        (const float*)din, in_stride, (bf16t*)dout_o, out_stride, positions,
        rows, c.features, c.head_dim, c.rotary, c.theta, (const double*)dinv, 1.0f,
        nullptr);
    if (he != hipSuccess) { std::printf("  [%s] opt launch failed: %s\n", c.name, hipGetErrorString(he)); cleanup(); return false; }
    shelf::gpu::discard_cleanup_result(hipGetLastError());

    std::vector<uint16_t> rc((size_t)rows * out_stride), ro((size_t)rows * out_stride);
    if (hipMemcpy(rc.data(), dout_c, rc.size() * 2, hipMemcpyDeviceToHost) != hipSuccess) { cleanup(); return false; }
    if (hipMemcpy(ro.data(), dout_o, ro.size() * 2, hipMemcpyDeviceToHost) != hipSuccess) { cleanup(); return false; }

    double max_abs = 0.0, max_rel = 0.0;
    size_t mismatch = 0, bad = 0;
    for (size_t i = 0; i < rc.size(); ++i) {
        const double ref = (double)bf16_to_f32(rc[i]);
        const double got = (double)bf16_to_f32(ro[i]);
        if (!std::isfinite(got)) { ++bad; continue; }
        const double diff = std::fabs(got - ref);
        const double rel = diff / std::fmax(1.0, std::fabs(ref));
        if (diff > max_abs) max_abs = diff;
        if (rel > max_rel) max_rel = rel;
        if (rc[i] != ro[i]) ++mismatch;
        if (rel > 2e-2) { ++bad; }
    }
    const bool pass = (bad == 0u);
    std::printf("  [%s] rows=%u f=%u hd=%u rot=%u mode=%d theta=%.0f | mismatch=%zu max_abs=%.3e max_rel=%.3e %s\n",
                c.name, rows, c.features, c.head_dim, c.rotary, (int)c.mode, c.theta,
                mismatch, max_abs, max_rel, pass ? "PASS" : "FAIL");
    cleanup();
    return pass;
}

}  // namespace

bool run_bitexact_case(uint32_t rows, uint32_t features,
                       const std::vector<uint32_t>& h_pos, uint32_t seed) {
    const uint32_t head_dim = 256;
    const uint32_t rotary = 64;
    const float theta = 1.0e7f;

    void* din = nullptr;
    void* dgen = nullptr;
    void* dpair = nullptr;
    void* dpos = nullptr;
    void* dinv = nullptr;
    auto cleanup = [&]() {
        void* all[] = {din, dgen, dpair, dpos, dinv};
        for (auto* p : all)
            if (p) shelf::gpu::discard_cleanup_result(hipFree(p));
    };
    if (hipMalloc(&din, (size_t)rows * features * 4) != hipSuccess ||
        hipMalloc(&dgen, (size_t)rows * features * 2) != hipSuccess ||
        hipMalloc(&dpair, (size_t)rows * features * 2) != hipSuccess ||
        hipMalloc(&dpos, rows * 4) != hipSuccess ||
        hipMalloc(&dinv, 32 * sizeof(double)) != hipSuccess) {
        cleanup();
        return false;
    }
    if (::shelf::kernel::launch_rope_inv_freq_init((double*)dinv, 32, theta, nullptr)
        != hipSuccess) {
        cleanup();
        return false;
    }

    std::vector<float> h_in((size_t)rows * features);
    for (size_t i = 0; i < h_in.size(); ++i)
        h_in[i] = rng1k(seed * 131u + i) / 1000.0f;
    if (hipMemcpy(din, h_in.data(), h_in.size() * 4, hipMemcpyHostToDevice) != hipSuccess) { cleanup(); return false; }
    if (hipMemcpy(dpos, h_pos.data(), h_pos.size() * 4, hipMemcpyHostToDevice) != hipSuccess) { cleanup(); return false; }

    hipError_t e1 = ::shelf::kernel::launch_rope_f32_bf16_generic(
        (const float*)din, features, (bf16t*)dgen, features, (const uint32_t*)dpos,
        rows, features, head_dim, rotary, theta, nullptr, 1.0f, nullptr);
    hipError_t e2 = ::shelf::kernel::launch_rope_f32_bf16_pair(
        (const float*)din, features, (bf16t*)dpair, features, (const uint32_t*)dpos,
        rows, features, head_dim, rotary, theta, (const double*)dinv, 1.0f, nullptr);
    if (e1 != hipSuccess || e2 != hipSuccess) {
        std::printf("  [bitexact r=%u f=%u] launch failed: %s / %s\n", rows, features,
                    hipGetErrorString(e1), hipGetErrorString(e2));
        cleanup();
        return false;
    }
    if (hipDeviceSynchronize() != hipSuccess) { cleanup(); return false; }

    std::vector<uint16_t> rg((size_t)rows * features), rp((size_t)rows * features);
    if (hipMemcpy(rg.data(), dgen, rg.size() * 2, hipMemcpyDeviceToHost) != hipSuccess) { cleanup(); return false; }
    if (hipMemcpy(rp.data(), dpair, rp.size() * 2, hipMemcpyDeviceToHost) != hipSuccess) { cleanup(); return false; }

    size_t mismatch = 0;
    size_t first = 0;
    for (size_t i = 0; i < rg.size(); ++i) {
        if (rg[i] != rp[i]) {
            if (mismatch == 0) first = i;
            ++mismatch;
        }
    }
    const bool pass = (mismatch == 0u);
    if (pass)
        std::printf("  [bitexact r=%-3u f=%-5u] mismatch=0 PASS\n", rows, features);
    else
        std::printf("  [bitexact r=%-3u f=%-5u] mismatch=%zu first=%zu FAIL\n",
                    rows, features, mismatch, first);
    cleanup();
    return pass;
}

__global__ void verify_inv_freq_kernel(
    const double* __restrict__ table, uint32_t half, float theta,
    uint32_t* __restrict__ mismatch) {
    const uint32_t i = threadIdx.x;
    if (i >= half)
        return;
    const double ref = 1.0 / pow(static_cast<double>(theta),
                                 static_cast<double>(i) / static_cast<double>(half));
    const uint64_t rb = *reinterpret_cast<const uint64_t*>(&ref);
    const uint64_t tb = reinterpret_cast<const uint64_t*>(table)[i];
    if (rb != tb)
        atomicAdd(mismatch, 1u);
}

bool run_freq_identity_case(float theta) {
    void* dtable = nullptr;
    void* dmm = nullptr;
    auto cleanup = [&]() {
        if (dtable) shelf::gpu::discard_cleanup_result(hipFree(dtable));
        if (dmm) shelf::gpu::discard_cleanup_result(hipFree(dmm));
    };
    if (hipMalloc(&dtable, 32 * sizeof(double)) != hipSuccess ||
        hipMalloc(&dmm, sizeof(uint32_t)) != hipSuccess) {
        cleanup();
        return false;
    }
    if (::shelf::kernel::launch_rope_inv_freq_init((double*)dtable, 32, theta, nullptr)
        != hipSuccess) {
        cleanup();
        return false;
    }
    if (hipMemset(dmm, 0, sizeof(uint32_t)) != hipSuccess) {
        cleanup();
        return false;
    }
    verify_inv_freq_kernel<<<1, 32, 0, nullptr>>>(
        (const double*)dtable, 32, theta, (uint32_t*)dmm);
    if (hipDeviceSynchronize() != hipSuccess) {
        cleanup();
        return false;
    }
    uint32_t mm = 0;
    if (hipMemcpy(&mm, dmm, sizeof(uint32_t), hipMemcpyDeviceToHost) != hipSuccess) {
        cleanup();
        return false;
    }
    const bool pass = (mm == 0u);
    std::printf("  [freq-identity theta=%.3g] mismatch=%u %s\n", theta, mm,
                pass ? "PASS" : "FAIL");
    cleanup();
    return pass;
}

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;

    const Case cases[] = {
        {"q-27b-r1-row", 1, 6144, 256, 64, 0, 0, 1.0e7f, 1, PosMode::Row},
        {"k-27b-r1-row", 1, 1024, 256, 64, 0, 0, 1.0e7f, 2, PosMode::Row},
        {"q-27b-r1-rope", 1, 6144, 256, 64, 0, 0, 1.0e7f, 3, PosMode::Rope},
        {"q-27b-r1-none", 1, 6144, 256, 64, 0, 0, 1.0e7f, 4, PosMode::None},
        {"q-27b-r2", 2, 6144, 256, 64, 0, 0, 1.0e7f, 5, PosMode::Row},
        {"q-27b-r16", 16, 6144, 256, 64, 0, 0, 1.0e7f, 6, PosMode::Row},
        {"q-27b-r128", 128, 6144, 256, 64, 0, 0, 1.0e7f, 7, PosMode::Row},
        {"q-27b-pad", 4, 6144, 256, 64, 6160, 6160, 1.0e7f, 8, PosMode::Row},
        {"q-4b-r1", 1, 4096, 256, 64, 0, 0, 1.0e7f, 9, PosMode::Row},
        {"q-theta-small", 4, 6144, 256, 64, 0, 0, 5.0e5f, 10, PosMode::Row},
        {"full-rotary", 2, 512, 256, 256, 0, 0, 1.0e7f, 11, PosMode::Row},
        {"q-tp2-r1", 1, 3072, 256, 64, 0, 0, 1.0e7f, 20, PosMode::Row},
        {"k-tp2-r1", 1, 512, 256, 64, 0, 0, 1.0e7f, 21, PosMode::Row},
        {"q-tp2-r64", 64, 3072, 256, 64, 0, 0, 1.0e7f, 22, PosMode::Row},
        {"q-tp2-r2048", 2048, 3072, 256, 64, 0, 0, 1.0e7f, 23, PosMode::Row},
        {"k-tp2-r64", 64, 512, 256, 64, 0, 0, 1.0e7f, 24, PosMode::Row},
    };
    for (const auto& c : cases) {
        if (run_case(c)) ++passed;
        else ++failed;
    }

    std::printf("\n== inv_freq table: GPU precompute vs inline expression ==\n");
    if (run_freq_identity_case(1.0e7f)) ++passed;
    else ++failed;
    if (run_freq_identity_case(5.0e5f)) ++passed;
    else ++failed;

    std::printf("\n== bit-exact: specialized pair vs generic optimized ==\n");
    const uint32_t kPosList[] = {0u, 1u, 2u, 31u, 128u, 1000u, 8191u, 32767u};
    for (uint32_t features : {1024u, 6144u}) {
        for (uint32_t p : kPosList) {
            std::vector<uint32_t> pos1{p};
            if (run_bitexact_case(1, features, pos1, 100 + features / 1024u + p)) ++passed;
            else ++failed;
        }
        for (uint32_t rows : {2u, 16u, 128u}) {
            std::vector<uint32_t> pos(rows);
            for (uint32_t r = 0; r < rows; ++r) pos[r] = kPosList[r % 8u];
            if (run_bitexact_case(rows, features, pos, 200 + rows + features / 1024u)) ++passed;
            else ++failed;
        }
    }

    std::printf("\n=== %d passed, %d failed ===\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
