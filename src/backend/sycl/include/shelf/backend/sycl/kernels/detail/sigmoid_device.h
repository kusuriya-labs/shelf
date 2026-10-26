// vendored from PhaseShift (own work); namespace/type adapted only.
// (SYCL port: expf -> ::sycl::exp.)
#pragma once

namespace shelf::syclk::detail {

inline float sigmoid_f32(float x) { return 1.0f / (1.0f + ::sycl::exp(-x)); }

} // namespace shelf::syclk::detail
