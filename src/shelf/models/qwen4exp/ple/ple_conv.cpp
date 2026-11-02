#include <shelf/models/qwen4exp/ple/ple_conv.h>

#include <cstdint>

namespace ps::qwen4exp::ple {

void PleConvState::reset(std::uint32_t dim) {
    conv_dim = dim;
    buffer.assign(static_cast<std::size_t>(dim) * kPleConvStateLen, 0.0f);
}

Status ple_dilated_conv1d(const float* weight, const float* input, std::uint32_t tokens,
                          std::uint32_t conv_dim, PleConvState& state, float* output) {
    if (conv_dim == 0u) {
        return Status::invalid_argument("conv_dim must be positive", __FILE__, __LINE__);
    }
    if (state.conv_dim != conv_dim ||
        state.buffer.size() != static_cast<std::size_t>(conv_dim) * kPleConvStateLen) {
        return Status::invalid_state("conv state not initialized for conv_dim", __FILE__,
                                     __LINE__);
    }
    if (tokens > 0u && (weight == nullptr || input == nullptr || output == nullptr)) {
        return Status::invalid_argument("buffers must not be null", __FILE__, __LINE__);
    }

    for (std::uint32_t t = 0; t < tokens; ++t) {
        const float* x = input + static_cast<std::size_t>(t) * conv_dim;
        float* y = output + static_cast<std::size_t>(t) * conv_dim;
        for (std::uint32_t c = 0; c < conv_dim; ++c) {
            const float* w = weight + static_cast<std::size_t>(c) * kPleConvKernel;
            const float* hist = state.buffer.data() + c;
            const float acc = w[0] * hist[0ull * conv_dim] +
                              w[1] * hist[3ull * conv_dim] +
                              w[2] * hist[6ull * conv_dim] +
                              w[3] * x[c];
            y[c] = acc;
        }
        for (std::uint32_t c = 0; c < conv_dim; ++c) {
            float* hist = state.buffer.data() + c;
            for (std::uint32_t k = 0; k + 1u < kPleConvStateLen; ++k) {
                hist[static_cast<std::size_t>(k) * conv_dim] =
                    hist[static_cast<std::size_t>(k + 1u) * conv_dim];
            }
            hist[static_cast<std::size_t>(kPleConvStateLen - 1u) * conv_dim] = x[c];
        }
    }
    return Status::make_ok();
}

}  // namespace ps::qwen4exp::ple
