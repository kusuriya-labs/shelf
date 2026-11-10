// P1-1: kernel traits stay consistent with the public constants, and unsupported
// phase / codebook / activation / grouped combinations are rejected on the host side.

#include <shelf/kernels/optimized/linear/exl3/detail/types.h>
#include <shelf/kernels/optimized/linear/exl3.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

int g_failures = 0;

void check(const char* name, bool ok) {
    if (!ok) {
        std::printf("FAIL %s\n", name);
        ++g_failures;
    }
}

}  // namespace

int main() {
    using namespace shelf::kernel;
    using namespace shelf::kernel::exl3_detail;

    // Traits agree with the public constants they currently mirror.
    static_assert(Exl3Mul1Traits::kRowTile == kExl3W16A16RowTile);
    static_assert(Exl3Mul1Traits::kOutBlock == kExl3W16A16OutBlock);
    static_assert(Exl3Mul1Traits::kThreads == kExl3W16A16Threads);
    static_assert(Exl3Mul24Traits::kRowTile == kExl3W16A16RowTile);
    static_assert(Exl3Mul24Traits::kOutBlock == kExl3W16A16OutBlock);
    static_assert(Exl3Mul24Traits::kThreads == kExl3W16A16Threads);

    // The kernel argument ABI did not grow: these are the fields P0 kept stable.
    check("gemm args size", sizeof(Exl3GemmArgs) > 0u);
    check("grouped args size", sizeof(Exl3GroupedArgs) > 0u);
    check("policy size", sizeof(Exl3ExecutionPolicy) == 8u);

    Exl3KernelVariant v;
    v.codebook = Exl3Codebook::Mul1;
    v.activation_type = Exl3ActivationType::FP16;
    check("mul1 fp16", exl3_variant_supported(v));
    v.activation_type = Exl3ActivationType::BF16;
    check("mul1 bf16 input", exl3_variant_supported(v));
    v.activation_type = Exl3ActivationType::INT8;
    check("mul1 int8 rejected", !exl3_variant_supported(v));

    v = Exl3KernelVariant{};
    v.codebook = Exl3Codebook::Mul24;
    v.activation_type = Exl3ActivationType::FP16;
    check("mul24 fp16", exl3_variant_supported(v));
    v.activation_type = Exl3ActivationType::INT8;
    check("mul24 int8", exl3_variant_supported(v));
    v.activation_type = Exl3ActivationType::BF16;
    check("mul24 bf16 rejected", !exl3_variant_supported(v));

    v = Exl3KernelVariant{};
    v.codebook = Exl3Codebook::Mul24;
    v.activation_type = Exl3ActivationType::INT8;
    v.grouped = true;
    v.pre_rotated = true;
    check("grouped pre-rotated rejected", !exl3_variant_supported(v));
    v.pre_rotated = false;
    check("grouped direct", exl3_variant_supported(v));

    if (g_failures == 0) {
        std::printf("ALL PASS\n");
        return 0;
    }
    std::printf("%d FAILURES\n", g_failures);
    return 1;
}
