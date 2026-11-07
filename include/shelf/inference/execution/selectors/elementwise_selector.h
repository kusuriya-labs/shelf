#pragma once
#include <cstdint>

namespace shelf::execution {

enum class ElementwiseProfile : uint8_t {
    ResidualBf16 = 0,
    SiluBf16ToF32 = 1,
    SiluF32ToBf16 = 2,
    SigmoidBf16ToF32 = 3,
    MulF32F32ToBf16 = 4,
    SwigluBf16 = 5,
    ScaleF32 = 6,
    SplitBf16Halves = 7,
    SplitBf16InterleavedHeads = 8,
    ScaleBf16 = 9,
    SigmoidBf16 = 10,
    SiluBf16 = 11,
    SigmoidF32 = 12,
    Count = 13,
};

enum class ElementwiseImplementation : uint8_t {
    Correctness = 0,
    Optimized = 1,
};

struct ElementwiseSelectorInput {
    ElementwiseProfile profile = ElementwiseProfile::ResidualBf16;
    uint32_t rows = 0;
    uint32_t features = 0;
    uint32_t aux = 0;
};

ElementwiseImplementation select_elementwise_implementation(
    const ElementwiseSelectorInput& in);

}  // namespace shelf::execution
