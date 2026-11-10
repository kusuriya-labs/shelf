#include <shelf/kernels/optimized/linear/exl3_mul24.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {

uint32_t exl3_mul24_gemm_splits(uint32_t rows, uint32_t n, uint32_t k, uint32_t scratch_rows,
                                uint32_t bits) {
    if (rows > 16u || rows == 0u || scratch_rows < rows) return 1u;
    if (bits == 4u && n == 17408u && k == 5120u && scratch_rows >= 64u) return 4u;
    // Wave slots of the device (6 resident waves per SIMD at ~215 VGPRs), queried once.
    static const uint32_t slots = [] {
        int dev = 0, wgp = 0;
        if (hipGetDevice(&dev) != hipSuccess ||
            hipDeviceGetAttribute(&wgp, hipDeviceAttributeMultiprocessorCount, dev) != hipSuccess ||
            wgp <= 0)
            return 768u;
        return static_cast<uint32_t>(wgp) * 4u * 6u;
    }();
    const uint32_t groups = k / 128u, tiles = n / 16u;
    uint32_t best = 1u;
    double best_fill = 0.0;
    // Fill = waves / (slots * rounds): the last round of a grid that is not a multiple of the slots runs partly empty.
    // Keep >= 8 groups per split so the weight prefetch pipeline still pays off.
    for (uint32_t s = 1u; s <= 16u && s * 16u <= scratch_rows && groups / s >= 8u; ++s) {
        const uint64_t waves = static_cast<uint64_t>(tiles) * s;
        const uint64_t rounds = (waves + slots - 1u) / slots;
        const double fill = static_cast<double>(waves) / static_cast<double>(rounds * slots);
        if (fill > best_fill + 0.02) { best_fill = fill; best = s; }
    }
    return best;
}

}  // namespace shelf::kernel
