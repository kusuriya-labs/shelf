#pragma once

#include <phaseshift/core/status.h>

#include <cstdint>
#include <vector>

namespace ps::qwen4exp::hyper {

struct GatedResidualDims {
    std::uint32_t streams = 4;
    std::uint32_t hidden = 2560;
    std::uint32_t lowrank = 320;
    float eps = 1e-6f;
};

struct GatedResidualWeights {
    const float* hc_norm = nullptr;
    const float* mix_down = nullptr;
    const float* mix_up = nullptr;
    const float* block_inject = nullptr;
};

struct GatedResidualReadTrace {
    std::vector<float> normed;
    std::vector<float> down;
    std::vector<float> up;
    std::vector<float> gate;
    std::vector<float> mixed;
    std::vector<float> inject;
};

struct GatedResidualCollapseWeights {
    const float* hc_norm = nullptr;
    const float* mix_down = nullptr;
    const float* mix_up = nullptr;
};

Status gated_residual_read(const GatedResidualDims& dims,
                           const GatedResidualWeights& weights,
                           const float* x4, std::uint32_t tokens,
                           float* mixed, float* inject,
                           GatedResidualReadTrace* trace);

Status gated_residual_write(const GatedResidualDims& dims,
                            const float* x4, const float* sub_out,
                            const float* inject, std::uint32_t tokens,
                            float* out_x4);

Status gated_residual_collapse(const GatedResidualDims& dims,
                               const GatedResidualCollapseWeights& weights,
                               const float* x4, std::uint32_t tokens,
                               float* out);

}  // namespace ps::qwen4exp::hyper
