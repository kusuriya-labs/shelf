#include <shelf/inference/execution/selectors/elementwise_selector.h>

#include <cstdio>

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
    std::printf("%s: %d failures\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
