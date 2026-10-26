// vendored from PhaseShift (own work); namespace/type adapted only.
// (SYCL port: expf -> ::sycl::exp.)
#pragma once

#include <sycl/sycl.hpp>

namespace shelf::syclk::detail {

inline float target_swiglu_f32(float gate, float up) {
    return up * (gate / (1.0f + ::sycl::exp(-gate)));
}

} // namespace shelf::syclk::detail
