#pragma once

#include <shelf/core/status.h>
#include <shelf/models/common/qwen_hybrid/emitter.h>
#include <shelf/weights/matrix_weight.h>

#include <cstdint>

namespace shelf {
namespace qwen4exp {

namespace RT = ::shelf::runtime;

struct HyperConnectionSiteGeometry {
    std::uint32_t hc = 0;
    std::uint32_t hidden = 0;
    std::uint32_t lowrank = 0;
    float eps = 1e-6f;
    std::uint32_t norm_parameter_index = UINT32_MAX;
};

struct HyperConnectionSiteWeights {
    const ::shelf::weights::MatrixWeight* down = nullptr;
    const ::shelf::weights::MatrixWeight* up = nullptr;
    const ::shelf::weights::MatrixWeight* inject = nullptr;
};

struct HyperConnectionSiteIndices {
    std::uint32_t w_down = 0;
    std::uint32_t w_up = 0;
    std::uint32_t w_inject = 0;
};

struct HyperConnectionSiteOutput {
    // The gate-weighted mean that feeds the attention or MLP sublayer.
    RT::ValueId mixed;
    // The updated hyper stack; zero when the caller asked for the final mixer only.
    RT::ValueId hyper;
};

// One Qwen4Exp gated-residual site:
//
//   normed = grouped RMSNorm(hyper)
//   gate   = sigmoid(up(silu(down(normed) / hc)))
//   mixed  = mean_h(gate * normed)
//   inj    = 2 * sigmoid(inject(normed) / hc)
//   hyper' = hyper + sublayer * inj
//
// `with_inject=false` is the model-level hyper_connection_mixer: it returns only mixed.
Result<HyperConnectionSiteOutput> emit_hyper_connection_site(
    ::shelf::qwen_hybrid::Emitter& em,
    const HyperConnectionSiteGeometry& geometry,
    const HyperConnectionSiteWeights& weights,
    const HyperConnectionSiteIndices& indices,
    RT::ValueId hyper,
    RT::ValueId sublayer,
    bool with_inject,
    const char* debug_prefix);

}  // namespace qwen4exp
}  // namespace shelf
