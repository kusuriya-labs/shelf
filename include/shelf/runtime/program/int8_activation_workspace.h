#pragma once
#include <shelf/core/memory/alignment.h>
#include <cstdint>

namespace ps::runtime {

struct Int8ActivationWorkspaceLayout {
    uint64_t codes_offset = 0;
    uint64_t scales_offset = 0;
    uint64_t total_bytes = 0;
    uint32_t k_padded = 0;
    uint32_t max_rows = 0;
    uint32_t code_row_stride_bytes = 0;
    uint32_t scale_row_stride_bytes = 0;

    static constexpr uint32_t scale_group() noexcept {
        return 128u;
    }

    static constexpr uint32_t scale_element_bytes() noexcept {
        return 4u;
    }

    static uint32_t k_padded_for(uint32_t k) noexcept {
        return (k + 31u) & ~31u;
    }

    static Int8ActivationWorkspaceLayout make(uint32_t k, uint32_t rows) noexcept {
        Int8ActivationWorkspaceLayout l;
        l.k_padded = k_padded_for(k);
        l.max_rows = rows;
        l.code_row_stride_bytes = static_cast<uint32_t>(
            ps::gpu::align_up_16(l.k_padded));
        const uint32_t groups = (l.k_padded + scale_group() - 1u) / scale_group();
        l.scale_row_stride_bytes = static_cast<uint32_t>(
            ps::gpu::align_up_16(
                static_cast<uint64_t>(groups) * scale_element_bytes()));
        l.codes_offset = 0;
        const uint64_t rows_padded =
            (static_cast<uint64_t>(l.max_rows) + 15ull) & ~15ull;
        const uint64_t codes_bytes = rows_padded * l.code_row_stride_bytes;
        l.scales_offset = ps::gpu::align_up_16(codes_bytes);
        l.total_bytes = l.scales_offset +
            static_cast<uint64_t>(l.max_rows) * l.scale_row_stride_bytes;
        return l;
    }

    uint64_t codes_bytes() const noexcept {
        const uint64_t rows_padded =
            (static_cast<uint64_t>(max_rows) + 15ull) & ~15ull;
        return rows_padded * code_row_stride_bytes;
    }

    uint64_t scales_bytes() const noexcept {
        return total_bytes - scales_offset;
    }
};

}
