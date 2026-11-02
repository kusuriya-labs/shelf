#pragma once

#include <shelf/core/status.h>

#include <cstdint>

namespace shelf::scheduling {

struct SamplingConfig {
    float temperature = 0.0f;
    float top_p = 1.0f;
    float min_p = 0.0f;
    uint32_t top_k = 0u;
    uint64_t seed = 0u;
};

enum class SamplingMode : uint8_t {
    None = 0,
    Greedy = 1,
    Stochastic = 2,
};

struct SamplingRequest {
    SamplingMode mode = SamplingMode::None;
    float temperature = 0.0f;
    float top_p = 1.0f;
    float min_p = 0.0f;
    uint32_t top_k = 0u;
    uint64_t seed = 0u;
    uint64_t sample_index = 0u;
};

Status validate_sampling_config(const SamplingConfig& config);

SamplingRequest make_sampling_request(
    const SamplingConfig& config,
    bool sample,
    uint64_t sample_index);

inline bool sampling_config_is_default(const SamplingConfig& config) noexcept {
    return config.temperature == 0.0f && config.top_p == 1.0f &&
           config.min_p == 0.0f && config.top_k == 0u && config.seed == 0u;
}

}  // namespace shelf::scheduling
