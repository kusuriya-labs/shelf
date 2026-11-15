#pragma once

#include <shelf/core/memory/types.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/projection.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {

hipError_t launch_fused_input(const PlamoExl3ProjectionGroup &args, hipStream_t stream);
hipError_t launch_fused_output(const PlamoExl3ProjectionGroup &args, uint32_t bits, hipStream_t stream);
hipError_t launch_group_out(const PlamoExl3ProjectionGroup &args, hipStream_t stream);

}  // namespace shelf::kernel
