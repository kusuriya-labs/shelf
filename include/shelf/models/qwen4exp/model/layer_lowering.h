#pragma once

#include <shelf/core/status.h>
#include <shelf/models/common/qwen_hybrid/emitter.h>
#include <shelf/models/common/qwen_hybrid/gdn_lowering.h>
#include <shelf/models/qwen4exp/model/hyper_connection_lowering.h>
#include <shelf/models/qwen4exp/model/moe_lowering.h>
#include <shelf/models/qwen4exp/model/ple_lowering.h>
#include <shelf/models/qwen4exp/model/qsa_lowering.h>

#include <cstdint>

namespace shelf {
namespace qwen4exp {

struct Qwen4ExpLayerLowering {
    bool is_gdn = false;

    HyperConnectionSiteGeometry hyper;
    // The MLP site has its own hc_norm. When unset, the attention norm is reused.
    std::uint32_t mlp_hyper_norm_parameter_index = UINT32_MAX;
    HyperConnectionSiteWeights attn_hyper_weights;
    HyperConnectionSiteIndices attn_hyper_indices;
    HyperConnectionSiteWeights mlp_hyper_weights;
    HyperConnectionSiteIndices mlp_hyper_indices;

    ::shelf::qwen_hybrid::GdnLoweringGeometry gdn_geometry;
    ::shelf::qwen_hybrid::GdnLoweringWeights gdn_weights;
    ::shelf::qwen_hybrid::GdnLoweringIndices gdn_indices;

    Qwen4ExpQsaLoweringGeometry qsa_geometry;
    Qwen4ExpQsaLoweringWeights qsa_weights;
    Qwen4ExpQsaLoweringIndices qsa_indices;

    Qwen4ExpMoeLoweringGeometry moe_geometry;
    Qwen4ExpMoeWeights moe_weights;
    Qwen4ExpMoeLoweringIndices moe_indices;

    // Layer 1 (zero-based) carries the PLE injection before the attention site.
    bool has_ple = false;
    Qwen4ExpPleLoweringGeometry ple_geometry;
    Qwen4ExpPleLoweringWeights ple_weights;
    Qwen4ExpPleLoweringIndices ple_indices;
    ::shelf::runtime::ValueId ple_embeddings{0};

    // Debug trace names for the intermediate values of this layer.
    std::vector<std::pair<std::string, ::shelf::runtime::ValueId>>* trace = nullptr;
};

// One Qwen4Exp decoder layer:
//   attn HyperConnection mix -> GDN or QSA -> apply
//   mlp  HyperConnection mix -> MoE FFN -> apply
//
// `hyper` is the [rows, hc * hidden] BF16 stream stack. A QSA layer runs the indexer and
// feeds its token mask to the gated attention.
Result<::shelf::runtime::ValueId> emit_qwen4exp_layer(
    ::shelf::qwen_hybrid::Emitter& em,
    const Qwen4ExpLayerLowering& layer,
    ::shelf::runtime::ValueId hyper,
    const char* debug_prefix);

}  // namespace qwen4exp
}  // namespace shelf
