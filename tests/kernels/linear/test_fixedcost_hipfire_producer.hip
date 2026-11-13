#include <shelf/kernels/optimized/linear/exl3_family/mul24/projection.h>
#include <shelf/kernels/optimized/linear/exl3_mul24.h>
#include <shelf/kernels/optimized/elementwise.h>
#include <shelf/kernels/optimized/rmsnorm.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {
void check(hipError_t error) {
    if (error != hipSuccess) {
        std::fprintf(stderr, "%s\n", hipGetErrorString(error));
        std::exit(1);
    }
}
template <typename T> struct Buffer {
    T* ptr = nullptr;
    explicit Buffer(size_t count) { check(hipMalloc(&ptr, count * sizeof(T))); }
    ~Buffer() { if (ptr) check(hipFree(ptr)); }
};

bool run(bool norm, bool zero) {
    constexpr uint32_t k = 2560u;
    constexpr uint32_t groups = k / 128u;
    const uint32_t count = norm ? 3u : 1u;
    Buffer<shelf::bf16_t> a(k), b(k), out(k), reference(k);
    Buffer<uint16_t> signs(k * count);
    Buffer<int8_t> codes(k * count), expected_codes(k * count);
    Buffer<float> scales(groups * count), expected_scales(groups * count);
    std::mt19937 rng(91u + norm + zero);
    std::vector<shelf::bf16_t> ha(k), hb(k);
    std::vector<uint16_t> hs(k * count);
    for (uint32_t i = 0; i < k; ++i) {
        const float x = zero ? 0.0f : float(int(rng() % 2001u) - 1000) / 500.0f;
        const float y = norm ? float(int(rng() % 501u) - 250) / 1000.0f :
                               float(int(rng() % 2001u) - 1000) / 500.0f;
        ha[i] = shelf::bf16_t{shelf::f32_to_bf16_rne(x)};
        hb[i] = shelf::bf16_t{shelf::f32_to_bf16_rne(y)};
    }
    for (auto& v : hs) v = rng() & 1u ? 0x3c00u : 0xbc00u;
    check(hipMemcpy(a.ptr, ha.data(), k * 2u, hipMemcpyHostToDevice));
    check(hipMemcpy(b.ptr, hb.data(), k * 2u, hipMemcpyHostToDevice));
    check(hipMemcpy(signs.ptr, hs.data(), hs.size() * 2u, hipMemcpyHostToDevice));
    shelf::kernel::PlamoExl3ProjectionGroup args{};
    args.input = out.ptr; args.rows = 1; args.k = k; args.input_stride = k;
    args.scale_stride_bytes = groups * 4u; args.count = count;
    for (uint32_t i = 0; i < count; ++i) {
        args.p[i].suh = signs.ptr + i * k;
        args.p[i].codes = codes.ptr + i * k;
        args.p[i].scales = scales.ptr + i * groups;
    }
    if (norm) {
        check(shelf::kernel::launch_rmsnorm_offset_bf16(a.ptr, b.ptr, reference.ptr, 1, k, k,
                                                         k, k, true, 1e-5f, 1.0f, nullptr));
        check(shelf::kernel::launch_plamo_exl3_norm_rotate(args, a.ptr, b.ptr, out.ptr, 1e-5f, 1.0f, nullptr));
    } else {
        check(shelf::kernel::launch_swiglu_bf16(a.ptr, k, b.ptr, k, reference.ptr, k, 1, k, nullptr, true));
        check(shelf::kernel::launch_plamo_exl3_swiglu_rotate(args, a.ptr, b.ptr, out.ptr, true, nullptr));
    }
    for (uint32_t i = 0; i < count; ++i)
        check(shelf::kernel::launch_exl3_mul24_rotate_quantize(out.ptr, k, signs.ptr + i * k,
            expected_codes.ptr + i * k, expected_scales.ptr + i * groups, 1, k, k, groups * 4u, nullptr));
    std::vector<shelf::bf16_t> got(k), ref(k);
    std::vector<int8_t> c(k * count), ec(k * count);
    std::vector<float> sc(groups * count), es(groups * count);
    check(hipMemcpy(got.data(), out.ptr, k * 2u, hipMemcpyDeviceToHost));
    check(hipMemcpy(ref.data(), reference.ptr, k * 2u, hipMemcpyDeviceToHost));
    check(hipMemcpy(c.data(), codes.ptr, c.size(), hipMemcpyDeviceToHost));
    check(hipMemcpy(ec.data(), expected_codes.ptr, ec.size(), hipMemcpyDeviceToHost));
    check(hipMemcpy(sc.data(), scales.ptr, sc.size() * 4u, hipMemcpyDeviceToHost));
    check(hipMemcpy(es.data(), expected_scales.ptr, es.size() * 4u, hipMemcpyDeviceToHost));
    for (uint32_t i = 0; i < k; ++i) {
        const float actual = __builtin_bit_cast(float, uint32_t(got[i].data) << 16);
        const float expected = __builtin_bit_cast(float, uint32_t(ref[i].data) << 16);
        if (std::fabs(actual - expected) > (norm ? 0.02f : 0.0f)) return false;
    }
    return c == ec && sc == es;
}
}
int main() {
    for (bool norm : {false, true}) for (bool zero : {false, true})
        if (!run(norm, zero)) {
            std::fprintf(stderr, "producer mismatch norm=%d zero=%d\n", norm, zero);
            return 1;
        }
    std::puts("producer rotation matched");
}
