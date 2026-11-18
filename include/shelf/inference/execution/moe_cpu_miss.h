#pragma once

#include <shelf/core/status.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::execution {

// One grouped W4A8 MXFP4 linear whose host-resident experts are computed on the CPU instead
// of being read over PCIe by the GPU. The plan arrays use the same layout as the grouped
// kernel: group g names the expert slot, the activation/output row base (in tiles of 16) and
// the row count.
struct MoeCpuMissDesc {
    const std::uint8_t* host_codes = nullptr;
    const std::uint8_t* host_scales = nullptr;
    std::uint32_t rows = 0;
    std::uint32_t k_padded = 0;
    std::uint32_t codes_batch_stride = 0;
    std::uint32_t scales_batch_stride = 0;
    std::uint32_t scale_row_stride = 0;

    const std::uint8_t* activation_codes = nullptr;
    const float* activation_scales = nullptr;
    std::uint32_t activation_code_stride = 0;
    std::uint32_t activation_scale_stride = 0;

    void* output = nullptr;
    std::uint32_t output_row_stride = 0;
    bool output_bf16 = false;

    const std::uint32_t* miss_slot = nullptr;
    const std::uint32_t* miss_rows = nullptr;
    const std::uint32_t* miss_base = nullptr;
    const std::uint32_t* miss_count = nullptr;
    std::uint32_t group_capacity = 0;
    std::uint32_t row_capacity = 0;
};

// Enabled by SHELF_MOE_CPU_MISS=1. `SHELF_MOE_CPU_MISS_ROWS` caps the rows per batch that go
// to the CPU (default 64); a larger miss set stays on the GPU.
bool moe_cpu_miss_enabled();

// Runs the miss groups on the CPU team and writes the rows back into `output`. The stream is
// synchronised before the plan is read, so the caller must not depend on queued GPU work for
// the same rows. `handled` reports whether every miss row was computed; when it is false the
// caller must launch the GPU miss plane instead.
Status run_moe_cpu_miss(const MoeCpuMissDesc& desc, hipStream_t stream, bool* handled);

}  // namespace shelf::execution
