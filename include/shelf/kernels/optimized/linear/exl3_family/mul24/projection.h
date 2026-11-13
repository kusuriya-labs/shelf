#pragma once

#include <shelf/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace shelf::kernel {

struct PlamoExl3Projection {
    const uint16_t *trellis = nullptr;
    const uint16_t *suh = nullptr;
    const uint16_t *svh = nullptr;
    int8_t *codes = nullptr;
    float *scales = nullptr;
    float *yprime = nullptr;
    bf16_t *out = nullptr;
    uint32_t n = 0;
    uint32_t splits = 0;
    uint32_t out_stride = 0;
};

struct PlamoExl3ProjectionGroup {
    const bf16_t *input = nullptr;
    PlamoExl3Projection p[3];
    uint32_t rows = 0;
    uint32_t k = 0;
    uint32_t input_stride = 0;
    uint32_t scale_stride_bytes = 0;
    uint32_t count = 0;
};

hipError_t launch_plamo_exl3_projection(const PlamoExl3ProjectionGroup &args, uint32_t bits,
                                        hipStream_t stream, uint32_t *launch_count = nullptr,
                                        bool allow_shape_fusion = true, bool input_rotated = false);

hipError_t launch_plamo_exl3_swiglu_rotate(const PlamoExl3ProjectionGroup &args,
                                            const bf16_t *gate, const bf16_t *up,
                                            bf16_t *product, bool intermediate, hipStream_t stream);

hipError_t launch_plamo_exl3_norm_rotate(const PlamoExl3ProjectionGroup &args,
                                          const bf16_t *input, const bf16_t *weight,
                                          bf16_t *output, float eps, float offset, hipStream_t stream);

hipError_t launch_plamo_exl3_flagged_gemm(const int8_t *codes, const float *scales, uint32_t code_stride,
                                          uint32_t scale_stride_bytes, const uint16_t *trellis,
                                          uint32_t bits, float *yprime, uint32_t rows,
                                          const uint32_t *active_rows, uint32_t n, uint32_t k,
                                          hipStream_t stream);

hipError_t launch_plamo_exl3_rotate_out_i32(const int32_t *acc, const float *rowscale, const int32_t *slots,
                                            const float *yprime, const uint16_t *svh, void *out,
                                            bool out_bf16, uint32_t rows, uint32_t n, uint32_t out_stride,
                                            hipStream_t stream);
struct PlamoExl3PrefillGroup {
    PlamoExl3ProjectionGroup projection;
    int8_t *packed[3]{};
    bool cached = false;
    bool rowscaled = false;
    float flag_share = 0.5f;
    float *rowscale[3]{};
    int32_t *row_flags[3]{};
};

hipError_t launch_plamo_exl3_projection_rotate(const PlamoExl3ProjectionGroup &args,
                                                hipStream_t stream);
hipError_t launch_plamo_exl3_prefill_recon(const PlamoExl3PrefillGroup &args, uint32_t bits,
                                           hipStream_t stream);
hipError_t launch_plamo_exl3_prefill_rowscale(const PlamoExl3PrefillGroup &args, hipStream_t stream);
hipError_t launch_plamo_exl3_prefill_gemm(const PlamoExl3PrefillGroup &args, uint32_t bits,
                                          hipStream_t stream);

}
