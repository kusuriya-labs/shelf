#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>

namespace ps::gpu {

// PhaseShift targets gfx1201 (AMD Radeon AI PRO R9700) exclusively. The
// architecture is a build/startup contract: it is validated once when the
// device is acquired and never queried on the hot path.
inline bool is_gfx1201(uint32_t device = 0) {
    hipDeviceProp_t p{};
    if (hipGetDeviceProperties(&p, device) != hipSuccess) return false;
    return std::strcmp(p.gcnArchName, "gfx1201") == 0;
}

}  // namespace ps::gpu
