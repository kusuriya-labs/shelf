#pragma once

#include <hip/hip_runtime.h>

namespace shelf::gpu {

inline void discard_cleanup_result(hipError_t result) {
    (void)result;
}

}  // namespace shelf::gpu
