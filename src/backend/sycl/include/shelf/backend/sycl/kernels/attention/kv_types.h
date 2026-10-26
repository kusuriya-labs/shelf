// vendored from PhaseShift (own work); namespace/type adapted only.
// (SYCL port: __hip_fp8_storage_t -> plain uint8 storage; the e4m3 codec in
// kernels/e4m3_codec.h is the software fallback for the ROCm intrinsics.)
#pragma once

#include <cstdint>

namespace shelf::syclk {

using fp8e4m3_storage_t = uint8_t;
static_assert(sizeof(fp8e4m3_storage_t) == 1, "FP8 E4M3 storage must be exactly 1 byte");

enum class KVCacheDType : uint8_t {
    BF16 = 0,
    FP8_E4M3 = 1,
};

} // namespace shelf::syclk
