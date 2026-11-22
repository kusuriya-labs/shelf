#pragma once

#include <shelf/core/status.h>
#include <shelf/models/common/qwen_hybrid/attention_lowering.h>
#include <shelf/models/common/qwen_hybrid/emitter.h>
#include <shelf/models/common/qwen_hybrid/qsa_indexer_lowering.h>
#include <shelf/models/qwen4exp/model/qwen4exp_config.h>

#include <cstdint>

namespace shelf {
namespace qwen4exp {

using Qwen4ExpQsaIndexerGeometry = ::shelf::qwen_hybrid::QsaIndexerGeometry;
using Qwen4ExpQsaIndexerWeights = ::shelf::qwen_hybrid::QsaIndexerWeights;
using Qwen4ExpQsaIndexerIndices = ::shelf::qwen_hybrid::QsaIndexerIndices;
using Qwen4ExpQsaIndexerOutputs = ::shelf::qwen_hybrid::QsaIndexerOutputs;

Qwen4ExpQsaIndexerGeometry make_qwen4exp_qsa_indexer_geometry(
    const Qwen4ExpTextConfig& config, std::uint32_t blocks, std::uint32_t state_index);

Result<Qwen4ExpQsaIndexerOutputs> emit_qwen4exp_qsa_indexer(
    ::shelf::qwen_hybrid::Emitter& em,
    const Qwen4ExpQsaIndexerGeometry& geometry,
    const Qwen4ExpQsaIndexerWeights& weights,
    const Qwen4ExpQsaIndexerIndices& indices,
    ::shelf::runtime::ValueId normed,
    const char* debug_prefix);

struct Qwen4ExpQsaLoweringGeometry {
    ::shelf::qwen_hybrid::GatedAttentionLoweringGeometry attention;
    Qwen4ExpQsaIndexerGeometry indexer;
};

struct Qwen4ExpQsaLoweringWeights {
    ::shelf::qwen_hybrid::GatedAttentionLoweringWeights attention;
    Qwen4ExpQsaIndexerWeights indexer;
};

struct Qwen4ExpQsaLoweringIndices {
    ::shelf::qwen_hybrid::GatedAttentionLoweringIndices attention;
    Qwen4ExpQsaIndexerIndices indexer;
};

Result<::shelf::runtime::ValueId> emit_qwen4exp_qsa(
    ::shelf::qwen_hybrid::Emitter& em,
    const Qwen4ExpQsaLoweringGeometry& geometry,
    const Qwen4ExpQsaLoweringWeights& weights,
    const Qwen4ExpQsaLoweringIndices& indices,
    ::shelf::runtime::ValueId normed,
    const char* debug_prefix);

}
}
