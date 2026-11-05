#pragma once

#include <shelf/core/status.h>
#include <shelf/models/common/qwen_hybrid/emitter.h>
#include <shelf/weights/matrix_weight.h>

#include <cstdint>

namespace shelf {
namespace qwen4exp {

struct Qwen4ExpPleLoweringGeometry {
    std::uint32_t hc = 0;
    std::uint32_t hidden = 0;
    std::uint32_t embed_dim = 0;
    float eps = 1e-6f;

    std::uint32_t norm_key_parameter_index = UINT32_MAX;
    std::uint32_t norm_query_parameter_index = UINT32_MAX;
    std::uint32_t norm_conv_parameter_index = UINT32_MAX;
    std::uint32_t conv_weight_parameter_index = UINT32_MAX;
    std::uint32_t state_index = 0;
};

struct Qwen4ExpPleLoweringWeights {
    const ::shelf::weights::MatrixWeight* key_proj = nullptr;
    const ::shelf::weights::MatrixWeight* value_proj = nullptr;
};

struct Qwen4ExpPleLoweringIndices {
    std::uint32_t w_key_proj = 0;
    std::uint32_t w_value_proj = 0;
};

// One PLE injection. `hyper` is the [rows, hc * hidden] stream stack and `embeddings` is
// the [rows, ple_embed_dim] n-gram embedding gathered on the host.
Result<::shelf::runtime::ValueId> emit_qwen4exp_ple(
    ::shelf::qwen_hybrid::Emitter& em,
    const Qwen4ExpPleLoweringGeometry& geometry,
    const Qwen4ExpPleLoweringWeights& weights,
    const Qwen4ExpPleLoweringIndices& indices,
    ::shelf::runtime::ValueId hyper,
    ::shelf::runtime::ValueId embeddings,
    const char* debug_prefix);

}  // namespace qwen4exp
}  // namespace shelf
