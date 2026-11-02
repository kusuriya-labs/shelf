#pragma once

#include <phaseshift/core/status.h>

#include <cstdint>
#include <vector>

namespace ps::qwen4exp::ple {

inline constexpr std::uint32_t kPleConvKernel = 4;
inline constexpr std::uint32_t kPleConvDilation = 3;
inline constexpr std::uint32_t kPleConvStateLen =
    (kPleConvKernel - 1u) * kPleConvDilation;

struct PleConvState {
    std::uint32_t conv_dim = 0;
    std::vector<float> buffer;

    void reset(std::uint32_t dim);
};

Status ple_dilated_conv1d(const float* weight, const float* input, std::uint32_t tokens,
                          std::uint32_t conv_dim, PleConvState& state, float* output);

}  // namespace ps::qwen4exp::ple
