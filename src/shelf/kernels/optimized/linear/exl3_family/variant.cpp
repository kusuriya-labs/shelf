#include <shelf/kernels/optimized/linear/exl3_family/types.h>

namespace shelf::kernel::exl3_detail {

bool exl3_variant_supported(const Exl3KernelVariant& v) {
    if (v.codebook == Exl3Codebook::Mul1) {
        // mul1 runs FP16 WMMA; INT8 activations exist for mul24 only.
        if (v.activation_type == Exl3ActivationType::INT8) return false;
    } else {
        // mul24 has FP16 and INT8 activation kernels; BF16 is not a kernel mode.
        if (v.activation_type == Exl3ActivationType::BF16) return false;
    }
    // The grouped path rotates internally, so a pre-rotated input is not accepted.
    if (v.grouped && v.pre_rotated) return false;
    return true;
}

}  // namespace shelf::kernel::exl3_detail
