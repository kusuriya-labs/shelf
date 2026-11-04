#include <shelf/inference/execution/selectors/elementwise_selector.h>
#include <shelf/inference/execution/selectors/selector_limits.h>

namespace shelf::execution {
namespace {

struct Rule {
    ElementwiseProfile profile;
    uint32_t features;
    uint32_t aux;
    uint32_t min_rows;
    uint32_t max_rows;
};

// The elementwise kernels address rows with blockIdx.y, so a launch is bounded by the
// grid.y limit. Group rows grow with tokens * top_k, so the MoE FFN needs a bound above
// the token row cap; 36864 is the capacity of a 4096-token bucket (20480 for 2048 and
// 12288 for 1024) and is measured.
constexpr uint32_t kMeasuredMaxGroupRows = 36864u;

constexpr Rule kRules[] = {
    {ElementwiseProfile::ResidualBf16, 2560u, 0u, 1u, kOptimizedMaxRows},
    {ElementwiseProfile::SwigluBf16, 9216u, 0u, 1u, kOptimizedMaxRows},
    {ElementwiseProfile::SigmoidBf16ToF32, 4096u, 0u, 1u, kOptimizedMaxRows},
    {ElementwiseProfile::MulF32F32ToBf16, 4096u, 0u, 1u, kOptimizedMaxRows},
    {ElementwiseProfile::SplitBf16InterleavedHeads, 4096u, 256u, 1u, kOptimizedMaxRows},
    {ElementwiseProfile::ResidualBf16, 5120u, 0u, 1u, kOptimizedMaxRows},
    {ElementwiseProfile::SwigluBf16, 17408u, 0u, 1u, kOptimizedMaxRows},
    {ElementwiseProfile::SigmoidBf16ToF32, 6144u, 0u, 1u, kOptimizedMaxRows},
    {ElementwiseProfile::MulF32F32ToBf16, 6144u, 0u, 1u, kOptimizedMaxRows},
    {ElementwiseProfile::SplitBf16InterleavedHeads, 6144u, 256u, 1u, kOptimizedMaxRows},
    {ElementwiseProfile::SiluF32ToBf16, 10240u, 0u, 1u, kOptimizedMaxRows},
    {ElementwiseProfile::SiluF32ToBf16, 8192u, 0u, 1u, kOptimizedMaxRows},
    {ElementwiseProfile::SiluBf16ToF32, 6144u, 0u, 1u, kOptimizedMaxRows},
    {ElementwiseProfile::SiluBf16ToF32, 4096u, 0u, 1u, kOptimizedMaxRows},
    {ElementwiseProfile::ScaleF32, 2048u, 0u, 1u, kOptimizedMaxRows},
    {ElementwiseProfile::SiluBf16ToF32, 3072u, 0u, 1u, kOptimizedMaxRows},
    {ElementwiseProfile::SiluF32ToBf16, 5120u, 0u, 1u, kOptimizedMaxRows},
    {ElementwiseProfile::SigmoidBf16ToF32, 3072u, 0u, 1u, kOptimizedMaxRows},
    {ElementwiseProfile::MulF32F32ToBf16, 3072u, 0u, 1u, kOptimizedMaxRows},
    {ElementwiseProfile::SwigluBf16, 8704u, 0u, 1u, kOptimizedMaxRows},
    {ElementwiseProfile::ScaleF32, 1024u, 0u, 1u, kOptimizedMaxRows},
    {ElementwiseProfile::SplitBf16InterleavedHeads, 3072u, 256u, 1u, kOptimizedMaxRows},
    // Qwen3.6-35B-A3B MoE FFN: the routed gate / up fusion and the shared expert work on
    // group rows, whose count grows with tokens * top_k.
    {ElementwiseProfile::SwigluBf16, 512u, 0u, 1u, kMeasuredMaxGroupRows},
    {ElementwiseProfile::SwigluBf16, 128u, 0u, 1u, kMeasuredMaxGroupRows},
    {ElementwiseProfile::SigmoidBf16ToF32, 1u, 0u, 1u, kOptimizedMaxRows},
    {ElementwiseProfile::ResidualBf16, 2048u, 0u, 1u, kOptimizedMaxRows},
};

}  // namespace

ElementwiseImplementation select_elementwise_implementation(
    const ElementwiseSelectorInput& in) {
    for (const auto& r : kRules) {
        if (r.profile != in.profile) continue;
        if (r.features != in.features) continue;
        if (r.aux != in.aux) continue;
        if (in.rows < r.min_rows || in.rows > r.max_rows) continue;
        return ElementwiseImplementation::Optimized;
    }
    return ElementwiseImplementation::Correctness;
}

}  // namespace shelf::execution
