// shelf SYCL Tile primitives — wave32 lane view and wave reductions
// (SYCL port of backend/hip kernels/detail/tile/wave32.h).
//
// The HIP original froze the xor order 16,8,4,2,1 as a bitwise gate.
// SYCL 2020 exposes no lane-level shuffle for the ported kernels, so the
// collectives are delegated to sycl sub_group reductions; wave/lane
// identities remain the static 32-thread mapping (frozen). Demonstrated
// consumers: attention paged decode (BF16), FP8 decode, BF16 split,
// gemm ExactRows. See kernels/detail/sycl_util.h for the implementations
// (Wave32 / wave_sum_all / wave_max_all / wave32_supported).
//
// NOTE: backend/hip also carries detail/tile/bf16_wmma_gfx12.h (the
// __builtin_amdgcn_wmma_* wrapper). It has no SYCL equivalent; the Wmma
// configs run through the scalar tile body in
// kernels/gemm/detail/linear_bf16_device.h instead.
#pragma once

#include <shelf/backend/sycl/kernels/detail/sycl_util.h>

namespace shelf::syclk {
namespace tile {

using ::shelf::syclk::Wave32;
using ::shelf::syclk::wave_max_all;
using ::shelf::syclk::wave_sum_all;

} // namespace tile
} // namespace shelf::syclk
