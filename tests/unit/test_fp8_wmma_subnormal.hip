#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using int32x2_t = int32_t __attribute__((ext_vector_type(2)));
using float8_t = float __attribute__((ext_vector_type(8)));

int g_fail = 0;

void fail(const char* msg) {
    ++g_fail;
    std::printf("FAIL %s\n", msg);
}

bool hip_ok(hipError_t e, const char* what) {
    if (e == hipSuccess)
        return true;
    std::printf("FAIL %s: %s\n", what, hipGetErrorString(e));
    return false;
}

// One FP8 (E4M3) WMMA 16x16x16 op per wave with every operand byte set to `a_byte`
// / `b_byte`. With uniform operands each of the 16 products of a dot is identical,
// so the expected accumulator value is independent of the fragment lane mapping,
// which is what makes this a clean probe of the operand format itself.
__global__ void wmma_uniform_e4m3_kernel(uint32_t a_byte, uint32_t b_byte, float* out) {
    const int32_t av = static_cast<int32_t>(a_byte * 0x01010101u);
    const int32_t bv = static_cast<int32_t>(b_byte * 0x01010101u);
    const int32x2_t a0 = int32x2_t{av, av};
    const int32x2_t b0 = int32x2_t{bv, bv};
    float8_t c = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, b0, c);
    #pragma unroll
    for (int j = 0; j < 8; ++j)
        out[threadIdx.x * 8u + static_cast<uint32_t>(j)] = c[j];
}

double e4m3_decode(uint32_t byte) {
    const uint32_t exp = (byte >> 3u) & 0x1Fu;
    const uint32_t man = byte & 0x7u;
    if (exp == 0u)
        return static_cast<double>(man) * (1.0 / 512.0);          // subnormal: M * 2^-9
    return std::ldexp(static_cast<double>(8u + man), static_cast<int>(exp) - 10);
}

// (byte_a, byte_b, label). The first three cases are E4M3 subnormals: if the WMMA
// unit flushed subnormal inputs, every product would collapse to zero and the
// MXFP4 A-side operand fold (which moves codes into this range) could not be used.
struct Case {
    uint32_t a;
    uint32_t b;
    const char* label;
};

int run_case(hipStream_t stream, const Case& c, float* d_out) {
    const double want = 16.0 * e4m3_decode(c.a) * e4m3_decode(c.b);
    wmma_uniform_e4m3_kernel<<<1, 32, 0, stream>>>(c.a, c.b, d_out);
    if (!hip_ok(hipGetLastError(), "wmma subnormal launch"))
        return 1;
    if (!hip_ok(hipStreamSynchronize(stream), "wmma subnormal sync"))
        return 1;
    std::vector<float> got(256u, 0.0f);
    if (!hip_ok(hipMemcpy(got.data(), d_out, got.size() * sizeof(float), hipMemcpyDeviceToHost),
                "read accumulator"))
        return 1;
    std::size_t bad = 0;
    for (float v : got) {
        if (static_cast<double>(v) != want)
            ++bad;
    }
    std::printf("e4m3 wmma %-22s a=%02X b=%02X want=%.9g mismatch=%zu/%zu\n",
                c.label, c.a, c.b, want, bad, got.size());
    if (bad != 0u)
        fail(c.label);
    return 0;
}

}  // namespace

int main() {
    if (!hip_ok(hipSetDevice(0), "hipSetDevice"))
        return 1;
    hipStream_t stream = nullptr;
    if (!hip_ok(hipStreamCreate(&stream), "stream create"))
        return 1;
    float* d_out = nullptr;
    if (!hip_ok(hipMalloc((void**)&d_out, 256u * sizeof(float)), "malloc accumulator")) {
        (void)hipStreamDestroy(stream);
        return 1;
    }

    const Case cases[] = {
        {0x01u, 0x01u, "subnormal 2^-9"},
        {0x07u, 0x07u, "subnormal 7*2^-9"},
        {0x01u, 0x07u, "subnormal mixed"},
        {0x08u, 0x07u, "min normal x subnormal"},
        {0x38u, 0x38u, "normal 1.0 control"},
        {0x7Eu, 0x01u, "max normal x subnormal"},
    };
    for (const auto& c : cases)
        run_case(stream, c, d_out);

    (void)hipFree(d_out);
    (void)hipStreamDestroy(stream);
    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
