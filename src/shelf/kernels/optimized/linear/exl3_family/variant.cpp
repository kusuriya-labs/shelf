#include <shelf/kernels/optimized/linear/exl3_family/types.h>

namespace shelf::kernel::exl3_detail {

bool exl3_variant_supported(const Exl3KernelVariant& v) {
    return v.codebook == Exl3Codebook::Mul24 &&
           v.activation_type == Exl3ActivationType::INT8 &&
           !(v.grouped && v.pre_rotated);
}

}
