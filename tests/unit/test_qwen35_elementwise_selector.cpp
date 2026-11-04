#include <shelf/inference/execution/selectors/elementwise_selector.h>
#include <shelf/kernels/optimized/elementwise.h>

#include <cstdio>
#include <cstdint>

using ::shelf::execution::ElementwiseImplementation;
using ::shelf::execution::ElementwiseProfile;
using ::shelf::execution::ElementwiseSelectorInput;
using ::shelf::execution::select_elementwise_implementation;

static int g_fail = 0;

static ElementwiseSelectorInput mk(ElementwiseProfile p, uint32_t rows, uint32_t features,
                                   uint32_t aux) {
    ElementwiseSelectorInput in;
    in.profile = p;
    in.rows = rows;
    in.features = features;
    in.aux = aux;
    return in;
}

static void check(bool ok, const char* msg) {
    if (!ok) ++g_fail;
    std::printf("  [%s] %s\n", msg, ok ? "PASS" : "FAIL");
}

static void expect(ElementwiseImplementation got, ElementwiseImplementation want, const char* msg) {
    const bool ok = got == want;
    if (!ok) ++g_fail;
    std::printf("  [%s] got=%d want=%d %s\n", msg, (int)got, (int)want, ok ? "PASS" : "FAIL");
}

int main() {
    const ElementwiseImplementation C = ElementwiseImplementation::Correctness;
    const ElementwiseImplementation O = ElementwiseImplementation::Optimized;
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::ResidualBf16, 4, 123456789u, 0)),
           C, "residual unknown features");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::SwigluBf16, 4, 123456789u, 0)),
           C, "swiglu unknown features");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::SplitBf16InterleavedHeads, 4, 2048u, 999u)),
           C, "split-interleaved unknown aux");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::ResidualBf16, 0, 2560u, 0)),
           C, "residual rows=0");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::ResidualBf16, 4, 2560u, 0)),
           O, "residual 2560 rows=4 in range");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::ResidualBf16, 8193, 2560u, 0)),
           C, "residual 2560 rows=8193 out of range");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::SiluF32ToBf16, 1, 10240u, 0)),
           O, "silu-f2b 10240 rows=1");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::SiluF32ToBf16, 2048, 10240u, 0)),
           O, "silu-f2b 10240 rows=2048");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::SiluBf16ToF32, 1, 6144u, 0)),
           O, "silu-b2f 6144 rows=1");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::SiluBf16ToF32, 2048, 6144u, 0)),
           O, "silu-b2f 6144 rows=2048");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::SiluF32ToBf16, 1, 8192u, 0)),
           O, "silu-f2b 8192 rows=1");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::SiluF32ToBf16, 2048, 8192u, 0)),
           O, "silu-f2b 8192 rows=2048");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::SiluF32ToBf16, 8193, 8192u, 0)),
           C, "silu-f2b 8192 rows=8193 out of range");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::SiluF32ToBf16, 1, 4096u, 0)),
           C, "silu-f2b 4096 wrong direction");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::SiluBf16ToF32, 1, 4096u, 0)),
           O, "silu-b2f 4096 rows=1");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::SiluBf16ToF32, 2048, 4096u, 0)),
           O, "silu-b2f 4096 rows=2048");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::SiluBf16ToF32, 8193, 4096u, 0)),
           C, "silu-b2f 4096 rows=8193 out of range");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::SiluBf16ToF32, 1, 8192u, 0)),
           C, "silu-b2f 8192 wrong direction");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::ScaleF32, 1, 2048u, 0)),
           O, "scale 2048 rows=1");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::ScaleF32, 2048, 2048u, 0)),
           O, "scale 2048 rows=2048");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::SiluF32ToBf16, 0, 10240u, 0)),
           C, "silu-f2b 10240 rows=0");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::SiluF32ToBf16, 8193, 10240u, 0)),
           C, "silu-f2b 10240 rows=8193 out of range");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::SiluF32ToBf16, 1, 2048u, 0)),
           C, "silu-f2b wrong features");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::SiluBf16ToF32, 8193, 6144u, 0)),
           C, "silu-b2f 6144 rows=8193 out of range");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::SiluBf16ToF32, 1, 10240u, 0)),
           C, "silu-b2f wrong features");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::ScaleF32, 0, 2048u, 0)),
           C, "scale 2048 rows=0");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::ScaleF32, 8193, 2048u, 0)),
           C, "scale 2048 rows=8193 out of range");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::ScaleF32, 1, 4096u, 0)),
           C, "scale wrong features");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::SiluBf16ToF32, 1, 3072u, 0)),
           O, "tp2 silu-b2f 3072 rows=1");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::SiluF32ToBf16, 1, 5120u, 0)),
           O, "tp2 silu-f2b 5120 rows=1");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::SigmoidBf16ToF32, 1, 3072u, 0)),
           O, "tp2 sigmoid 3072 rows=1");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::MulF32F32ToBf16, 1, 3072u, 0)),
           O, "tp2 mul 3072 rows=1");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::SwigluBf16, 1, 8704u, 0)),
           O, "tp2 swiglu 8704 rows=1");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::ScaleF32, 1, 1024u, 0)),
           O, "tp2 scale 1024 rows=1");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::SplitBf16InterleavedHeads, 1, 3072u, 256u)),
           O, "tp2 split-int 3072 aux=256 rows=1");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::SwigluBf16, 8193, 8704u, 0)),
           C, "tp2 swiglu 8704 rows=8193 out of range");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::SwigluBf16, 12288u, 512u, 0)),
           O, "moe group rows 12288 (bucket 1024 capacity) -> Optimized");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::SwigluBf16, 36864u, 512u, 0)),
           O, "moe group rows 36864 (bucket 4096 capacity) -> Optimized");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::SwigluBf16, 69632u, 512u, 0)),
           O, "moe group rows 69632 (bucket 8192 capacity) -> Optimized");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::SwigluBf16, 69633u, 512u, 0)),
           C, "moe group rows above the measured bound -> Correctness");
    expect(select_elementwise_implementation(
               mk(ElementwiseProfile::SwigluBf16, 5120u, 128u, 0)),
           O, "moe group rows at width 128 -> Optimized");

    // The row domain is split across grid.y (hardware bound 65535) and grid.z. grid.y must
    // stay within the bound and the product must cover every row, at the split point and at
    // the multiples of it.
    for (const uint32_t rows : {1u, 2u, 65534u, 65535u, 65536u, 69632u, 131070u, 131071u,
                                196605u, 196606u, 1000000u}) {
        const uint32_t gy = ::shelf::kernel::elementwise_grid_y(rows);
        const uint32_t gz = ::shelf::kernel::elementwise_grid_z(rows);
        const bool covered = gz >= 1u && gy <= ::shelf::kernel::kElementwiseGridYMax &&
                             static_cast<uint64_t>(gy) * gz >= rows;
        char msg[96];
        std::snprintf(msg, sizeof(msg), "grid.y / grid.z cover rows=%u (y=%u z=%u)", rows, gy, gz);
        check(covered, msg);
    }
    std::printf("%s: %d failures\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
