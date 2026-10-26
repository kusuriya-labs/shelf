// shelf backend/sycl — device-side shared helpers (SYCL port of
// backend/hip kernels/detail/vector_io.h + tile/wave32.h primitives).
//
// Conversion table from the HIP originals:
//   threadIdx.x            -> it.get_local_linear_id()
//   blockIdx.x/y/z         -> it.get_group(0/1/2)
//   blockDim.x             -> it.get_local_range(0)
//   __syncthreads()        -> it.barrier(access::fence_space::local_space)
//   __shfl_xor/_down trees -> wave_sum_all / wave_max_all (sycl sub_group
//                             collectives; see numeric note below)
//   __bfloat162float etc.  -> bf16_bits_to_f32 / f32_to_bf16_bits (bit ops,
//                             same RNE semantics)
//   __expf / rsqrtf / fmaxf -> ::sycl::exp / ::sycl::rsqrt / ::sycl::fmax
//   reinterpret uint64/128 -> memcpy-based load helpers (alignment-safe)
//
// Numeric note: the HIP kernels use frozen shuffle trees (xor 16..1 /
// down 16..1) whose reassociation order is part of the *HIP* bitwise gate
// (docs/architecture/hip-tile-refactor-procedure.md §7.2/§8). SYCL exposes
// no lane-level shuffle in 2020, so the port uses sycl sub_group
// collectives. The arithmetic content (operands, order of token visits,
// online-softmax structure) is unchanged, but floating-point summation
// order inside a wave may differ from the HIP kernels: SYCL and HIP
// backends are NOT bitwise twins; both are validated against the CPU
// reference within tolerance.
//
// The wave helpers require a hardware sub_group size of exactly 32 (AMD
// wave32 / NVIDIA half-warp semantics as used by the ported kernels).
// Launchers check this against the queue device (wave32_supported) before
// submitting such kernels.
#pragma once

#include <cstdint>
#include <cstring>

#include <sycl/sycl.hpp>

#include <shelf/foundation/numeric.hpp>
#include <shelf/status.hpp>

namespace shelf::syclk {

// SYCL submissions throw asynchronously-mapped errors at submit time only;
// convert anything thrown into a shelf::Status (mirrors the hipError_t
// return of the HIP launchers).
inline ::shelf::Status status_from_sycl(const char* expr, const char* file,
                                        int line) noexcept {
    try {
        throw;
    } catch (const ::sycl::exception& e) {
        return ::shelf::Status::device_error(expr, e.what(), file, line);
    } catch (const std::exception& e) {
        return ::shelf::Status::device_error(expr, e.what(), file, line);
    } catch (...) {
        return ::shelf::Status::device_error(expr, "unknown SYCL exception", file, line);
    }
}

// Submit a command group through the stream's queue, mapping host-side
// submission errors to Status (mirrors hipGetLastError() in the HIP
// launchers). Function form (not a macro) so lambdas with commas/pragma
// survive.
template <class CGF>
inline ::shelf::Status sycl_submit(::sycl::queue& q, const char* what,
                                   const char* file, int line, CGF&& cgf) {
    try {
        q.submit(std::forward<CGF>(cgf));
    } catch (...) {
        return status_from_sycl(what, file, line);
    }
    return ::shelf::Status::make_ok();
}

using shelf::bf16_t;

inline constexpr uint32_t kWaveSize = 32u;

// ---- bf16 scalar conversion (mirrors __bfloat162float/__float2bfloat16) ----
inline float bf16_bits_to_f32(std::uint16_t b) {
    const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}
inline std::uint16_t f32_to_bf16_bits(float f) {
    return shelf::f32_to_bf16_rne(f);  // RNE, same rounding as the HIP path
}

// ---- alignment helpers ------------------------------------------------------
inline bool aligned8(const void* p) {
    return (reinterpret_cast<std::uintptr_t>(p) & 7u) == 0u;
}
inline bool aligned16(const void* p) {
    return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0u;
}

// ---- packed IO (memcpy keeps strict aliasing + alignment legal) -------------
struct F4 {
    float v[4];
};
struct U4 {
    std::uint32_t v[4];
};

inline std::uint64_t load_u64(const void* p) {
    std::uint64_t x;
    std::memcpy(&x, p, sizeof(x));
    return x;
}
inline void store_u64(void* p, std::uint64_t x) { std::memcpy(p, &x, sizeof(x)); }

inline F4 load_bf16x4(const bf16_t* p) {
    const std::uint64_t bits = load_u64(p);
    F4 r;
    r.v[0] = bf16_bits_to_f32(static_cast<std::uint16_t>(bits & 0xFFFFu));
    r.v[1] = bf16_bits_to_f32(static_cast<std::uint16_t>((bits >> 16u) & 0xFFFFu));
    r.v[2] = bf16_bits_to_f32(static_cast<std::uint16_t>((bits >> 32u) & 0xFFFFu));
    r.v[3] = bf16_bits_to_f32(static_cast<std::uint16_t>((bits >> 48u) & 0xFFFFu));
    return r;
}
inline void store_bf16x4(bf16_t* p, const F4& r) {
    std::uint64_t bits = 0u;
    bits |= static_cast<std::uint64_t>(f32_to_bf16_bits(r.v[0]));
    bits |= static_cast<std::uint64_t>(f32_to_bf16_bits(r.v[1])) << 16u;
    bits |= static_cast<std::uint64_t>(f32_to_bf16_bits(r.v[2])) << 32u;
    bits |= static_cast<std::uint64_t>(f32_to_bf16_bits(r.v[3])) << 48u;
    store_u64(p, bits);
}
inline F4 load_f32x4(const float* p) {
    F4 r;
    std::memcpy(r.v, p, sizeof(r));
    return r;
}
inline void store_f32x4(float* p, const F4& r) { std::memcpy(p, r.v, sizeof(r)); }

// 8 bf16 (16 bytes) as raw bits, like the HIP short8_t stage in ExactRows.
inline void load_bf16x8_bits(const bf16_t* p, std::uint16_t (&out)[8]) {
    std::memcpy(out, p, 16u);
}

// ---- wave32 subgroup collectives -------------------------------------------
inline float wave_sum_all(const ::sycl::sub_group& sg, float x) {
    return ::sycl::reduce_over_group(sg, x, ::sycl::plus<float>());
}
inline float wave_max_all(const ::sycl::sub_group& sg, float x) {
    return ::sycl::reduce_over_group(sg, x, ::sycl::maximum<float>());
}

// Static lane/wave view of a block that is a multiple of 32 threads
// (mirror of hip tile::Wave32; RDNA wavefront index = tid >> 5).
struct Wave32 {
    std::uint32_t tid;
    std::uint32_t lane;   // tid & 31
    std::uint32_t wave;   // tid >> 5

    template <int D>
    static Wave32 current(const ::sycl::nd_item<D>& it) {
        const std::uint32_t tid = static_cast<std::uint32_t>(it.get_local_linear_id());
        return {tid, tid & 31u, tid >> 5u};
    }
};

// Host-side pre-dispatch check: the ported wave32 kernels require hardware
// subgroups of exactly 32 (true for AMD targets; not for the OpenMP device).
inline bool wave32_supported(const ::sycl::device& d) {
    for (const auto s : d.get_info<::sycl::info::device::sub_group_sizes>())
        if (s == kWaveSize) return true;
    return false;
}
inline bool wave32_supported(const ::sycl::queue& q) {
    return wave32_supported(q.get_device());
}

} // namespace shelf::syclk
