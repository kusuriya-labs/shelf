// vendored from PhaseShift (own work); namespace/type adapted only.
#pragma once
#include <hip/hip_fp8.h>
#include <cstdint>

namespace shelf::hipk {

using fp8e4m3_storage_t = __hip_fp8_storage_t;
static_assert(sizeof(fp8e4m3_storage_t) == 1, "FP8 E4M3 storage must be exactly 1 byte");

enum class KVCacheDType : uint8_t {
    BF16 = 0,
    FP8_E4M3 = 1,
};

}  // namespace shelf::hipk
