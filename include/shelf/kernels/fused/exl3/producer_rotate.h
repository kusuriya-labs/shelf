#pragma once

#include <shelf/core/memory/types.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/projection.h>

#include <hip/hip_runtime.h>

namespace shelf::kernel {

hipError_t launch_plamo_exl3_swiglu_rotate(const PlamoExl3ProjectionGroup &args,
                                            const bf16_t *gate, const bf16_t *up,
                                            bf16_t *product, bool intermediate, hipStream_t stream);

hipError_t launch_plamo_exl3_norm_rotate(const PlamoExl3ProjectionGroup &args,
                                          const bf16_t *input, const bf16_t *weight,
                                          bf16_t *output, float eps, float offset, hipStream_t stream);

}  // namespace shelf::kernel
