#pragma once

#include <shelf/core/status.h>
#include <shelf/models/common/qwen_hybrid/emitter.h>
#include <shelf/models/common/qwen_hybrid/gdn_lowering.h>
#include <shelf/models/qwen4exp/model/hyper_connection_lowering.h>
#include <shelf/models/qwen4exp/model/moe_lowering.h>

#include <cstdint>

namespace shelf {
namespace qwen4exp {

struct Qwen4ExpLayerLowering {
    bool is_gdn = false;

    HyperConnectionSiteGeometry hyper;
    HyperConnectionSiteWeights attn_hyper_weights;
    HyperConnectionSiteIndices attn_hyper_indices;
    HyperConnectionSiteWeights mlp_hyper_weights;
    HyperConnectionSiteIndices mlp_hyper_indices;

    ::shelf::qwen_hybrid::GdnLoweringGeometry gdn_geometry;
    ::shelf::qwen_hybrid::GdnLoweringWeights gdn_weights;
    ::shelf::qwen_hybrid::GdnLoweringIndices gdn_indices;

    Qwen4ExpMoeLoweringGeometry moe_geometry;
    Qwen4ExpMoeWeights moe_weights;
    Qwen4ExpMoeLoweringIndices moe_indices;
};

// One Qwen4Exp decoder layer:
//   attn HyperConnection mix -> GDN -> apply
//   mlp  HyperConnection mix -> MoE FFN -> apply
//
// `hyper` is the [rows, hc * hidden] BF16 stream stack. QSA (full-attention) layers are
// rejected until the indexer primitive exists.
Result<::shelf::runtime::ValueId> emit_qwen4exp_layer(
    ::shelf::qwen_hybrid::Emitter& em,
    const Qwen4ExpLayerLowering& layer,
    ::shelf::runtime::ValueId hyper,
    const char* debug_prefix);

}  // namespace qwen4exp
}  // namespace shelf
