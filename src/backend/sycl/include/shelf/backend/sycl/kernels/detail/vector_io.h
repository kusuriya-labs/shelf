// vendored from PhaseShift (own work); namespace/type adapted only.
// (SYCL port: __device__ builtins replaced by the bit-level helpers in
// sycl_util.h; semantics identical.)
#pragma once

#include <cstdint>

#include <shelf/backend/sycl/kernels/detail/sycl_util.h>

namespace shelf::syclk::detail {

using bf16 = shelf::bf16_t;  // 16-bit storage; conversions via helpers

constexpr uint32_t kVec = 4u;

// load_bf16x4 / store_bf16x4 / load_f32x4 / store_f32x4 / aligned8/16 all
// come from sycl_util.h (namespace shelf::syclk).
using ::shelf::syclk::aligned16;
using ::shelf::syclk::aligned8;
using ::shelf::syclk::F4;
using ::shelf::syclk::load_bf16x4;
using ::shelf::syclk::load_f32x4;
using ::shelf::syclk::store_bf16x4;
using ::shelf::syclk::store_f32x4;

} // namespace shelf::syclk::detail
