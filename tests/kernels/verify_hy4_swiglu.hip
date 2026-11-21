// Compare a fused result with the original GPU SwiGLU applied to separate BF16 projections.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <vector>
using namespace shelf::kernel::exl3_detail;
using namespace shelf::kernel::exl3_detail::mul24;
__global__ void swiglu(const uint16_t *gate, const uint16_t *up, uint16_t *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    float g = fminf(bf16_bits_to_f32(gate[i]), 10.f), u = fminf(10.f, fmaxf(-10.f, bf16_bits_to_f32(up[i])));
    out[i] = f32_to_bf16_rne_device(round_bf16(g / (1.f + expf(-g))) * u);
}
#define CHECK(x)                                                                                                       \
    do {                                                                                                               \
        auto e = (x);                                                                                                  \
        if (e != hipSuccess) {                                                                                         \
            fprintf(stderr, "%s\n", hipGetErrorString(e));                                                             \
            return 2;                                                                                                  \
        }                                                                                                              \
    } while (0)
int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s gate.bf16 up.bf16 fused.bf16\n", argv[0]);
        return 2;
    }
    std::vector<uint16_t> a, b, c;
    auto read = [](char *p, std::vector<uint16_t> &v) {
        FILE *f = fopen(p, "rb");
        if (!f)
            exit(2);
        fseek(f, 0, SEEK_END);
        long n = ftell(f);
        rewind(f);
        v.resize(n / 2);
        if (fread(v.data(), 1, n, f) != size_t(n))
            exit(2);
        fclose(f);
    };
    read(argv[1], a);
    read(argv[2], b);
    read(argv[3], c);
    if (a.size() != b.size() || a.size() != c.size())
        return 2;
    uint16_t *da, *db, *dc;
    size_t bytes = a.size() * 2;
    CHECK(hipMalloc(&da, bytes));
    CHECK(hipMalloc(&db, bytes));
    CHECK(hipMalloc(&dc, bytes));
    CHECK(hipMemcpy(da, a.data(), bytes, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(db, b.data(), bytes, hipMemcpyHostToDevice));
    swiglu<<<(a.size() + 255) / 256, 256>>>(da, db, dc, a.size());
    CHECK(hipGetLastError());
    CHECK(hipMemcpy(a.data(), dc, bytes, hipMemcpyDeviceToHost));
    size_t diff = 0;
    for (size_t i = 0; i < a.size(); i++)
        diff += a[i] != c[i];
    printf("swiglu bit differences=%zu/%zu\n", diff, a.size());
    return diff != 0;
}
