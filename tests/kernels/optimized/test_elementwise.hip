#include <phaseshift/kernels/optimized/elementwise.h>
#include <phaseshift/kernels/correctness/model_dispatch_correctness.h>
#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/runtime/program/device_program.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

using V = ::ps::runtime::DeviceValueBinding;
using P = ::ps::runtime::DeviceProgramView;
using B = ::ps::runtime::DispatchBinding;
using DT = ::ps::runtime::ValueDType;

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
    ~Buf() {
        if (dev) ps::gpu::discard_cleanup_result(hipFree(dev));
    }
    bool alloc(size_t b) {
        bytes = b;
        return hipMalloc(&dev, b) == hipSuccess;
    }
    bool fill_f(const std::vector<float>& h) {
        return hipMemcpy(dev, h.data(), h.size() * 4, hipMemcpyHostToDevice) == hipSuccess;
    }
    bool fill_u16(const std::vector<uint16_t>& h) {
        return hipMemcpy(dev, h.data(), h.size() * 2, hipMemcpyHostToDevice) == hipSuccess;
    }
    bool read_f(std::vector<float>& o) const {
        o.resize(bytes / 4);
        return hipMemcpy(o.data(), dev, bytes, hipMemcpyDeviceToHost) == hipSuccess;
    }
    bool read_u16(std::vector<uint16_t>& o) const {
        o.resize(bytes / 2);
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
    static constexpr uint32_t kMaxVals = 4;
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
        if (hipMalloc(&vals_dev, kMaxVals * sizeof(V)) != hipSuccess) return false;
        return true;
    }
    bool run(const V* host_vals, uint32_t vcount, const B& b) {
        if (vcount > kMaxVals) return false;
        if (hipMemcpy(vals_dev, host_vals, (size_t)vcount * sizeof(V),
                      hipMemcpyHostToDevice) != hipSuccess) {
            return false;
        }
        P p{};
        p.values = (const V*)vals_dev;
        p.value_count = vcount;
        p.workspace_base = 0;
        ps::gpu::discard_cleanup_result(hipMemsetAsync(err, 0, sizeof(uint32_t), nullptr));
        ::ps::kernel::ModelDispatchStateView sv{};
        auto st = ps::kernel::launch_model_dispatch_correctness(
            p, b, (::ps::runtime::DeviceBatchContext*)dev, sv, (float*)scratch,
            (uint32_t*)err, nullptr);
        if (!st.ok()) return false;
        uint32_t e = 0;
        if (hipMemcpy(&e, err, 4, hipMemcpyDeviceToHost) != hipSuccess) return false;
        return e == 0u;
    }
};

void gather_u16(const Buf& buf, uint32_t rows, uint32_t f, uint32_t out_stride,
                std::vector<uint8_t>& bytes) {
    std::vector<uint16_t> t;
    buf.read_u16(t);
    bytes.clear();
    bytes.reserve((size_t)rows * f * 2);
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t j = 0; j < f; ++j) {
            const uint16_t v = t[(size_t)r * out_stride + j];
            bytes.push_back(v & 0xFFu);
            bytes.push_back((v >> 8) & 0xFFu);
        }
}

void gather_f(const Buf& buf, uint32_t rows, uint32_t f, uint32_t out_stride,
              std::vector<uint8_t>& bytes) {
    std::vector<float> t;
    buf.read_f(t);
    bytes.clear();
    bytes.reserve((size_t)rows * f * 4);
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t j = 0; j < f; ++j) {
            const float v = t[(size_t)r * out_stride + j];
            uint32_t u;
            std::memcpy(&u, &v, 4);
            bytes.push_back(u & 0xFFu);
            bytes.push_back((u >> 8) & 0xFFu);
            bytes.push_back((u >> 16) & 0xFFu);
            bytes.push_back((u >> 24) & 0xFFu);
        }
}

size_t cmp_bits(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    size_t mm = 0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i)
        if (a[i] != b[i]) ++mm;
    return (a.size() != b.size()) ? (mm + 1) : mm;
}

enum class Op {
    Residual,
    Swiglu,
    Sigmoid,
    SiluB2F,
    SiluF2B,
    Mul,
    Scale,
    SplitHalves,
    SplitInterleaved,
};

struct Case {
    const char* name;
    Op op;
    uint32_t f;
    uint32_t rows;
    uint32_t in_stride;
    uint32_t out_stride;
    uint32_t head_dim;
    float scale;
    uint32_t seed;
};

bool run_case(const Case& c, size_t* mm_out) {
    Ctx ctx;
    if (!ctx.init(c.rows)) { std::printf("  [%s] CTX FAIL\n", c.name); *mm_out = 1; return false; }
    const uint32_t rows = c.rows;
    const uint32_t f = c.f;
    const uint32_t seed = c.seed;
    const bool split = (c.op == Op::SplitHalves || c.op == Op::SplitInterleaved);
    const uint32_t f_in = split ? 2u * f : f;
    const bool f32_out = (c.op == Op::Sigmoid || c.op == Op::SiluB2F || c.op == Op::Scale);
    const bool f32_a = (c.op == Op::SiluF2B || c.op == Op::Mul || c.op == Op::Scale);
    const bool b_is_bf16 = (c.op == Op::Residual || c.op == Op::Swiglu);

    auto gen_f = [&](uint32_t n, uint32_t tag) {
        std::vector<float> v(n);
        for (uint32_t i = 0; i < n; ++i) v[i] = rng1k(seed * 13u + tag * 100000u + i) / 1000.0f;
        return v;
    };
    auto gen_bf16 = [&](uint32_t n, uint32_t tag) {
        std::vector<uint16_t> v(n);
        for (uint32_t i = 0; i < n; ++i) v[i] = bf16_bits(rng1k(seed * 13u + tag * 100000u + i) / 1000.0f);
        return v;
    };

    Buf a, b, o0, o1, sig;
    const size_t n_in = (size_t)rows * f_in * (c.in_stride > 0 ? (c.in_stride / f_in) : 1);
    // input buffer size: rows * in_stride elements (in_stride in ELEMENTS)
    const size_t in_elems = (size_t)rows * c.in_stride;
    const size_t out_elems = (size_t)rows * c.out_stride;
    if (!a.alloc(in_elems * 4) || !b.alloc(in_elems * 4) || !o0.alloc(out_elems * 4) ||
        !o1.alloc(out_elems * 4) || !sig.alloc(out_elems * 4)) {
        std::printf("  [%s] ALLOC FAIL\n", c.name);
        *mm_out = 1;
        return false;
    }
    (void)n_in;

    // fill a
    if (f32_a) {
        auto ha = gen_f(in_elems, 1);
        if (!a.fill_f(ha)) { *mm_out = 1; return false; }
    } else {
        auto ha = gen_bf16(in_elems, 1);
        if (!a.fill_u16(ha)) { *mm_out = 1; return false; }
    }
    // fill b
    if (c.op == Op::Residual || c.op == Op::Swiglu) {
        auto hb = gen_bf16(in_elems, 2);
        if (!b.fill_u16(hb)) { *mm_out = 1; return false; }
    } else if (c.op == Op::Mul) {
        auto hb = gen_f(in_elems, 2);
        if (!b.fill_f(hb)) { *mm_out = 1; return false; }
        }

    // ---- correctness ----
    {
        V vals[4];
        if (split) {
            set_val(vals[0], 0, f_in, c.in_stride, a.dev, DT::BF16);
            set_val(vals[1], 1, f, c.out_stride, o0.dev, DT::BF16);
            set_val(vals[2], 2, f, c.out_stride, o1.dev, DT::BF16);
            B binding{};
            binding.kernel_id = ::ps::runtime::KernelId::SPLIT;
            binding.input_slots[0] = 0;
            binding.output_slots[0] = 1;
            binding.output_slots[1] = 2;
            binding.input_count = 1;
            binding.output_count = 2;
            binding.flags = (c.op == Op::SplitInterleaved)
                ? static_cast<uint32_t>(::ps::runtime::SplitLayout::INTERLEAVED_HEADS) : 0u;
            binding.group_size = c.head_dim;
            if (!ctx.run(vals, 3, binding)) { std::printf("  [%s] CORR-SPLIT FAIL\n", c.name); *mm_out = 1; return false; }
        } else if (c.op == Op::Residual || c.op == Op::Swiglu || c.op == Op::Mul) {
            DT a_dt = f32_a ? DT::F32 : DT::BF16;
            DT b_dt = b_is_bf16 ? DT::BF16 : DT::F32;
            DT o_dt = f32_out ? DT::F32 : DT::BF16;
            set_val(vals[0], 0, f, c.in_stride, a.dev, a_dt);
            set_val(vals[1], 1, f, c.in_stride, b.dev, b_dt);
            set_val(vals[2], 2, f, c.out_stride, o0.dev, o_dt);
            B binding{};
            binding.input_slots[0] = 0;
            binding.input_slots[1] = 1;
            binding.output_slots[0] = 2;
            binding.input_count = 2;
            binding.output_count = 1;
            binding.kernel_id = (c.op == Op::Residual) ? ::ps::runtime::KernelId::RESIDUAL_ADD
                               : (c.op == Op::Swiglu) ? ::ps::runtime::KernelId::SWIGLU
                                                      : ::ps::runtime::KernelId::MUL;
            if (!ctx.run(vals, 3, binding)) { std::printf("  [%s] CORR-BIN FAIL\n", c.name); *mm_out = 1; return false; }
        } else {
            DT a_dt = f32_a ? DT::F32 : DT::BF16;
            DT o_dt = f32_out ? DT::F32 : DT::BF16;
            set_val(vals[0], 0, f, c.in_stride, a.dev, a_dt);
            set_val(vals[1], 1, f, c.out_stride, o0.dev, o_dt);
            B binding{};
            binding.input_slots[0] = 0;
            binding.output_slots[0] = 1;
            binding.input_count = 1;
            binding.output_count = 1;
            if (c.op == Op::Sigmoid) binding.kernel_id = ::ps::runtime::KernelId::SIGMOID;
            else if (c.op == Op::SiluB2F || c.op == Op::SiluF2B)
                binding.kernel_id = ::ps::runtime::KernelId::SILU;
            else {
                binding.kernel_id = ::ps::runtime::KernelId::SCALE;
                binding.scalar_a = c.scale;
            }
            if (!ctx.run(vals, 2, binding)) { std::printf("  [%s] CORR-UNARY FAIL\n", c.name); *mm_out = 1; return false; }
        }
    }

    // ---- read reference ----
    std::vector<uint8_t> ref0, ref1, got0, got1;
    if (split) {
        gather_u16(o0, rows, f, c.out_stride, ref0);
        gather_u16(o1, rows, f, c.out_stride, ref1);
    } else if (f32_out) {
        gather_f(o0, rows, f, c.out_stride, ref0);
    } else {
        gather_u16(o0, rows, f, c.out_stride, ref0);
    }

    // ---- optimized ----
    {
        bool ok = false;
        switch (c.op) {
            case Op::Residual:
                ok = ::ps::kernel::launch_residual_add_bf16((const ::ps::bf16_t*)a.dev,
                    c.in_stride, (const ::ps::bf16_t*)b.dev, c.in_stride, (::ps::bf16_t*)o0.dev,
                    c.out_stride, rows, f, nullptr) == hipSuccess;
                break;
            case Op::Swiglu:
                ok = ::ps::kernel::launch_swiglu_bf16((const ::ps::bf16_t*)a.dev,
                    c.in_stride, (const ::ps::bf16_t*)b.dev, c.in_stride, (::ps::bf16_t*)o0.dev,
                    c.out_stride, rows, f, nullptr) == hipSuccess;
                break;
            case Op::Sigmoid:
                ok = ::ps::kernel::launch_sigmoid_bf16_f32((const ::ps::bf16_t*)a.dev,
                    c.in_stride, (float*)o0.dev, c.out_stride, rows, f, nullptr) == hipSuccess;
                break;
            case Op::SiluB2F:
                ok = ::ps::kernel::launch_silu_bf16_f32((const ::ps::bf16_t*)a.dev,
                    c.in_stride, (float*)o0.dev, c.out_stride, rows, f, nullptr) == hipSuccess;
                break;
            case Op::SiluF2B:
                ok = ::ps::kernel::launch_silu_f32_bf16((const float*)a.dev, c.in_stride,
                    (::ps::bf16_t*)o0.dev, c.out_stride, rows, f, nullptr) == hipSuccess;
                break;
            case Op::Mul:
                ok = ::ps::kernel::launch_mul_f32_f32_bf16((const float*)a.dev,
                    c.in_stride, (const float*)b.dev, c.in_stride, (::ps::bf16_t*)o0.dev,
                    c.out_stride, rows, f, nullptr) == hipSuccess;
                break;
            case Op::Scale:
                ok = ::ps::kernel::launch_scale_f32((const float*)a.dev, c.in_stride,
                    (float*)o0.dev, c.out_stride, rows, f, c.scale, nullptr) == hipSuccess;
                break;
            case Op::SplitHalves:
            case Op::SplitInterleaved:
                ok = ::ps::kernel::launch_split_bf16((const ::ps::bf16_t*)a.dev,
                    c.in_stride, (::ps::bf16_t*)o0.dev, c.out_stride, (::ps::bf16_t*)o1.dev,
                    c.out_stride, rows, f,
                    c.op == Op::SplitInterleaved
                        ? ::ps::kernel::ElementwiseSplitLayout::InterleavedHeads
                        : ::ps::kernel::ElementwiseSplitLayout::Halves,
                    c.head_dim, nullptr) == hipSuccess;
                break;
        }
        if (!ok) { std::printf("  [%s] OPT LAUNCH FAIL\n", c.name); *mm_out = 1; return false; }
    }
    if (hipDeviceSynchronize() != hipSuccess) { *mm_out = 1; return false; }

    // ---- read optimized + compare ----
    size_t mm = 0;
    if (split) {
        gather_u16(o0, rows, f, c.out_stride, got0);
        gather_u16(o1, rows, f, c.out_stride, got1);
        mm += cmp_bits(ref0, got0);
        mm += cmp_bits(ref1, got1);
    } else if (f32_out) {
        gather_f(o0, rows, f, c.out_stride, got0);
        mm += cmp_bits(ref0, got0);
    } else {
        gather_u16(o0, rows, f, c.out_stride, got0);
        mm += cmp_bits(ref0, got0);
    }
    *mm_out = mm;
    const bool pass = (mm == 0);
    std::printf("  [%s] op=%d f=%u rows=%u in_rs=%u out_rs=%u hd=%u | mm=%zu %s\n", c.name,
                (int)c.op, f, rows, c.in_stride, c.out_stride, c.head_dim, mm,
                pass ? "PASS" : "FAIL");
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (hipSetDevice(0) != hipSuccess) return 1;
    const Case cases[] = {
        {"residual", Op::Residual, 2560, 16, 2560, 2560, 0, 0.0f, 101},
        {"residual-tail", Op::Residual, 33, 4, 33, 33, 0, 0.0f, 102},
        {"residual-pad", Op::Residual, 2560, 8, 2576, 2576, 0, 0.0f, 103},
        {"swiglu", Op::Swiglu, 4096, 16, 4096, 4096, 0, 0.0f, 104},
        {"swiglu-tail", Op::Swiglu, 31, 4, 31, 31, 0, 0.0f, 105},
        {"sigmoid", Op::Sigmoid, 2048, 16, 2048, 2048, 0, 0.0f, 106},
        {"sigmoid-tail", Op::Sigmoid, 5, 4, 5, 5, 0, 0.0f, 107},
        {"silu-b2f", Op::SiluB2F, 2048, 16, 2048, 2048, 0, 0.0f, 108},
        {"silu-b2f-pad", Op::SiluB2F, 2048, 8, 2064, 2064, 0, 0.0f, 109},
        {"silu-f2b", Op::SiluF2B, 2048, 16, 2048, 2048, 0, 0.0f, 110},
        {"silu-f2b-tail", Op::SiluF2B, 15, 4, 15, 15, 0, 0.0f, 111},
        {"mul", Op::Mul, 2048, 16, 2048, 2048, 0, 0.0f, 112},
        {"mul-tail", Op::Mul, 7, 4, 7, 7, 0, 0.0f, 113},
        {"scale", Op::Scale, 2048, 16, 2048, 2048, 0, 0.5f, 114},
        {"scale-tail", Op::Scale, 3, 4, 3, 3, 0, 2.0f, 115},
        {"split-halves", Op::SplitHalves, 1024, 16, 2048, 2048, 2048, 0.0f, 116},
        {"split-halves-tail", Op::SplitHalves, 100, 4, 200, 200, 200, 0.0f, 117},
        {"split-int", Op::SplitInterleaved, 2048, 8, 4096, 4096, 128, 0.0f, 118},
        {"split-int-tail", Op::SplitInterleaved, 64, 4, 128, 128, 32, 0.0f, 119},
        {"residual-27b", Op::Residual, 5120, 8, 5120, 5120, 0, 0.0f, 130},
        {"swiglu-actual-4b", Op::Swiglu, 9216, 8, 9216, 9216, 0, 0.0f, 131},
        {"swiglu-actual-27b", Op::Swiglu, 17408, 8, 17408, 17408, 0, 0.0f, 132},
        {"mul-actual-4b", Op::Mul, 9216, 8, 9216, 9216, 0, 0.0f, 133},
        {"sigmoid-actual-4b", Op::Sigmoid, 4096, 8, 4096, 4096, 0, 0.0f, 134},
        {"silu-b2f-actual-4b", Op::SiluB2F, 4096, 8, 4096, 4096, 0, 0.0f, 135},
        {"silu-f2b-actual-27b", Op::SiluF2B, 6144, 8, 6144, 6144, 0, 0.0f, 136},
        {"scale-actual-4b", Op::Scale, 2560, 8, 2560, 2560, 0, 0.5f, 137},
        {"split-halves-actual-4b", Op::SplitHalves, 4608, 8, 9216, 9216, 9216, 0.0f, 138},
        {"split-int-actual-4b", Op::SplitInterleaved, 2048, 8, 4096, 4096, 128, 0.0f, 139},
        {"silu-f2b-27b-r1", Op::SiluF2B, 10240, 1, 10240, 10240, 0, 0.0f, 140},
        {"silu-f2b-27b-r2", Op::SiluF2B, 10240, 2, 10240, 10240, 0, 0.0f, 141},
        {"silu-f2b-27b-r16", Op::SiluF2B, 10240, 16, 10240, 10240, 0, 0.0f, 142},
        {"silu-f2b-27b-r128", Op::SiluF2B, 10240, 128, 10240, 10240, 0, 0.0f, 143},
        {"silu-f2b-27b-r2048", Op::SiluF2B, 10240, 2048, 10240, 10240, 0, 0.0f, 144},
        {"silu-f2b-27b-pad", Op::SiluF2B, 10240, 4, 10256, 10256, 0, 0.0f, 145},
        {"silu-b2f-27b-r1", Op::SiluB2F, 6144, 1, 6144, 6144, 0, 0.0f, 146},
        {"silu-b2f-27b-r2", Op::SiluB2F, 6144, 2, 6144, 6144, 0, 0.0f, 147},
        {"silu-b2f-27b-r16", Op::SiluB2F, 6144, 16, 6144, 6144, 0, 0.0f, 148},
        {"silu-b2f-27b-r128", Op::SiluB2F, 6144, 128, 6144, 6144, 0, 0.0f, 149},
        {"silu-b2f-27b-r2048", Op::SiluB2F, 6144, 2048, 6144, 6144, 0, 0.0f, 150},
        {"silu-b2f-27b-pad", Op::SiluB2F, 6144, 4, 6160, 6160, 0, 0.0f, 151},
        {"scale-27b-r1", Op::Scale, 2048, 1, 2048, 2048, 0, 0.5f, 152},
        {"scale-27b-r2", Op::Scale, 2048, 2, 2048, 2048, 0, 0.5f, 153},
        {"scale-27b-r16", Op::Scale, 2048, 16, 2048, 2048, 0, 0.5f, 154},
        {"scale-27b-r128", Op::Scale, 2048, 128, 2048, 2048, 0, 0.5f, 155},
        {"scale-27b-r2048", Op::Scale, 2048, 2048, 2048, 2048, 0, 0.5f, 156},
        {"scale-27b-pad", Op::Scale, 2048, 4, 2064, 2064, 0, 0.5f, 157},
        {"silu-f2b-actual-4b-gdn", Op::SiluF2B, 8192, 1, 8192, 8192, 0, 0.0f, 160},
        {"silu-b2f-actual-4b-gdn", Op::SiluB2F, 4096, 1, 4096, 4096, 0, 0.0f, 161},
        {"tp2-silu-b2f-3072-r1", Op::SiluB2F, 3072, 1, 3072, 3072, 0, 0.0f, 170},
        {"tp2-silu-b2f-3072-r64", Op::SiluB2F, 3072, 64, 3072, 3072, 0, 0.0f, 171},
        {"tp2-silu-f2b-5120-r1", Op::SiluF2B, 5120, 1, 5120, 5120, 0, 0.0f, 172},
        {"tp2-silu-f2b-5120-r64", Op::SiluF2B, 5120, 64, 5120, 5120, 0, 0.0f, 173},
        {"tp2-sigmoid-3072-r1", Op::Sigmoid, 3072, 1, 3072, 3072, 0, 0.0f, 174},
        {"tp2-mul-3072-r1", Op::Mul, 3072, 1, 3072, 3072, 0, 0.0f, 175},
        {"tp2-swiglu-8704-r1", Op::Swiglu, 8704, 1, 8704, 8704, 0, 0.0f, 176},
        {"tp2-swiglu-8704-r64", Op::Swiglu, 8704, 64, 8704, 8704, 0, 0.0f, 177},
        {"tp2-scale-1024-r1", Op::Scale, 1024, 1, 1024, 1024, 0, 0.5f, 178},
        {"tp2-split-int-1536-r8", Op::SplitInterleaved, 1536, 8, 3072, 3072, 256, 0.0f, 179},
    };
    int fail = 0;
    for (const auto& c : cases) {
        size_t mm = 0;
        run_case(c, &mm);
        if (mm != 0) ++fail;
    }
    std::printf("%s: %d failures\n", fail == 0 ? "PASS" : "FAIL", fail);
    return fail == 0 ? 0 : 1;
}
