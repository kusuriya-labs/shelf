// Golden test: make_workspace_layout must reproduce the previous hand calculation
// exactly (same offsets, same chunk / scratch rows, same fit decision).

#include <shelf/kernels/optimized/linear/exl3_family/mul24/workspace.h>
#include <shelf/kernels/optimized/linear/exl3_mul1.h>
#include <shelf/kernels/optimized/linear/exl3_mul24.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>

namespace {

inline uint64_t align256(uint64_t x) {
    return (x + 255ull) & ~255ull;
}

struct Ref {
    uint64_t codes_offset = 0, scales_offset = 0, slots_offset = 0, count_offset = 0;
    uint64_t flagged_codes_offset = 0, flagged_scales_offset = 0, xrot_bytes = 0;
    uint64_t yprime_offset = 0, flagged_yprime_offset = 0, accum_bytes = 0, dense_bytes = 0;
    uint32_t chunk = 0, scratch_rows = 0, cap = 0, code_stride = 0, scale_stride_bytes = 0;
    bool fits = false;
};

Ref reference(uint32_t rows, uint32_t n, uint32_t k, uint64_t xrot_cap, uint64_t accum_cap,
              uint64_t dense_cap) {
    Ref r;
    const uint32_t groups = k / 128u;
    r.code_stride = k;
    r.scale_stride_bytes = (groups * 4u + 15u) & ~15u;
    const uint64_t per_row_in = static_cast<uint64_t>(r.code_stride) + r.scale_stride_bytes;
    const uint64_t rows_in = per_row_in == 0u ? 0u : xrot_cap / per_row_in;
    const uint64_t rows_out = n == 0u ? 0u : accum_cap / (static_cast<uint64_t>(n) * 4u);
    r.scratch_rows = static_cast<uint32_t>(std::min<uint64_t>(rows_out, 0xFFFFFFFFull));
    r.chunk = static_cast<uint32_t>(std::min<uint64_t>(std::min<uint64_t>(rows_in, rows_out), rows));
    r.dense_bytes = static_cast<uint64_t>(k) * n;
    if (r.chunk == 0u) return r;
    const uint64_t chunk = r.chunk;
    r.cap = static_cast<uint32_t>(shelf::kernel::exl3_mul24_flag_cap(chunk));
    r.codes_offset = 0u;
    r.scales_offset = align256(chunk * r.code_stride);
    r.slots_offset = align256(r.scales_offset + chunk * r.scale_stride_bytes);
    r.count_offset = r.slots_offset + chunk * 4u;
    r.flagged_codes_offset = align256(r.count_offset + 4u);
    r.flagged_scales_offset = align256(r.flagged_codes_offset + static_cast<uint64_t>(r.cap) * r.code_stride);
    r.xrot_bytes = r.flagged_scales_offset + static_cast<uint64_t>(r.cap) * r.scale_stride_bytes;
    r.yprime_offset = 0u;
    r.flagged_yprime_offset = align256(chunk * n * 4u);
    r.accum_bytes = r.flagged_yprime_offset + static_cast<uint64_t>(r.cap) * n * 4u;
    r.fits = r.xrot_bytes <= xrot_cap && r.accum_bytes <= accum_cap && r.dense_bytes <= dense_cap;
    return r;
}

int g_failures = 0;

void check(const char* name, uint32_t rows, uint32_t n, uint32_t k, uint64_t xrot_cap,
           uint64_t accum_cap, uint64_t dense_cap) {
    const Ref r = reference(rows, n, k, xrot_cap, accum_cap, dense_cap);
    const auto l = shelf::kernel::exl3_detail::mul24::make_workspace_layout(rows, n, k, xrot_cap,
                                                                          accum_cap, dense_cap);
    const bool same =
        l.codes_offset == r.codes_offset && l.scales_offset == r.scales_offset &&
        l.slots_offset == r.slots_offset && l.count_offset == r.count_offset &&
        l.flagged_codes_offset == r.flagged_codes_offset &&
        l.flagged_scales_offset == r.flagged_scales_offset && l.xrot_bytes == r.xrot_bytes &&
        l.yprime_offset == r.yprime_offset && l.flagged_yprime_offset == r.flagged_yprime_offset &&
        l.accum_bytes == r.accum_bytes && l.dense_bytes == r.dense_bytes && l.chunk_rows == r.chunk &&
        l.scratch_rows == r.scratch_rows && l.flag_capacity == r.cap && l.code_stride == r.code_stride &&
        l.scale_stride_bytes == r.scale_stride_bytes && l.fits == r.fits;
    if (!same) {
        std::printf("FAIL %s: layout differs\n", name);
        ++g_failures;
    }
}

}  // namespace

int main() {
    constexpr uint64_t kGiB = 1ull << 30;
    // Typical prefill / decode shapes with plenty of workspace.
    check("prefill 2048x17408x5120", 2048u, 17408u, 5120u, 8ull * kGiB, 16ull * kGiB, 4ull * kGiB);
    check("prefill 2048x5120x17408", 2048u, 5120u, 17408u, 8ull * kGiB, 16ull * kGiB, 4ull * kGiB);
    check("decode 1x5120x5120", 1u, 5120u, 5120u, 8ull * kGiB, 16ull * kGiB, 4ull * kGiB);
    check("verify 8x248320x5120", 8u, 248320u, 5120u, 8ull * kGiB, 16ull * kGiB, 4ull * kGiB);
    // Small accum capacity forces chunk < rows.
    check("chunk by accum", 2048u, 17408u, 5120u, 8ull * kGiB, 64ull * 17408ull * 4ull, 4ull * kGiB);
    // Small xrot capacity forces chunk even smaller.
    check("chunk by xrot", 2048u, 17408u, 5120u, 64ull * 5120ull, 16ull * kGiB, 4ull * kGiB);
    // Dense region too small: layout fits is false but chunk is still valid.
    check("dense too small", 2048u, 5120u, 5120u, 8ull * kGiB, 16ull * kGiB, 1024ull);
    // Accum smaller than one row: chunk 0.
    check("chunk 0", 16u, 5120u, 5120u, 8ull * kGiB, 4096ull, 4ull * kGiB);
    // Boundary: exactly one row of accum capacity.
    check("one row accum", 16u, 5120u, 5120u, 8ull * kGiB, 5120ull * 4ull, 4ull * kGiB);
    // Scale stride rounding at k not a multiple of 32 groups.
    check("k 640", 40u, 384u, 640u, 8ull * kGiB, 16ull * kGiB, 4ull * kGiB);
    check("k 896", 5u, 128u, 896u, 8ull * kGiB, 16ull * kGiB, 4ull * kGiB);

    if (g_failures == 0) {
        std::printf("ALL PASS\n");
        return 0;
    }
    std::printf("%d FAILURES\n", g_failures);
    return 1;
}
