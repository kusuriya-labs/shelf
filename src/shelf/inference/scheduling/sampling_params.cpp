#include <shelf/inference/scheduling/sampling_params.h>

#include <cmath>

namespace shelf::scheduling {

Status validate_sampling_config(const SamplingConfig& config) {
    if (!std::isfinite(config.temperature)) {
        return Status::invalid_argument("sampling temperature must be finite", __FILE__,
                                        __LINE__);
    }
    if (config.temperature < 0.0f) {
        return Status::invalid_argument("sampling temperature must be >= 0", __FILE__,
                                        __LINE__);
    }
    if (!std::isfinite(config.top_p)) {
        return Status::invalid_argument("sampling top_p must be finite", __FILE__, __LINE__);
    }
    if (!(config.top_p > 0.0f) || config.top_p > 1.0f) {
        return Status::invalid_argument("sampling top_p must be in (0, 1]", __FILE__, __LINE__);
    }
    if (!std::isfinite(config.min_p)) {
        return Status::invalid_argument("sampling min_p must be finite", __FILE__, __LINE__);
    }
    if (config.min_p < 0.0f || config.min_p > 1.0f) {
        return Status::invalid_argument("sampling min_p must be in [0, 1]", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

SamplingRequest make_sampling_request(
    const SamplingConfig& config,
    bool sample,
    uint64_t sample_index) {
    SamplingRequest request;
    request.temperature = config.temperature;
    request.top_p = config.top_p;
    request.min_p = config.min_p;
    request.top_k = config.top_k;
    request.seed = config.seed;
    request.sample_index = sample_index;
    if (!sample) {
        request.mode = SamplingMode::None;
    } else if (config.temperature == 0.0f) {
        request.mode = SamplingMode::Greedy;
    } else {
        request.mode = SamplingMode::Stochastic;
    }
    return request;
}

}  // namespace shelf::scheduling
