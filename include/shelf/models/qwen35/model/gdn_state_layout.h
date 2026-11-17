#pragma once

#include <shelf/runtime/gdn_state_layout.h>

namespace shelf::qwen35 {

using ::shelf::runtime::PipelineStage;

inline std::uint32_t qwen35_stage_layer_count(
    const shelf::qwen_hybrid::QwenHybridConfig& tc, const PipelineStage& stage) {
    return ::shelf::runtime::stage_layer_count(tc, stage);
}

inline std::uint32_t qwen35_stage_gdn_layers(
    const shelf::qwen_hybrid::QwenHybridConfig& tc, const PipelineStage& stage) {
    return ::shelf::runtime::stage_gdn_layers(tc, stage);
}

inline std::uint32_t qwen35_stage_attention_layers(
    const shelf::qwen_hybrid::QwenHybridConfig& tc, const PipelineStage& stage) {
    return ::shelf::runtime::stage_attention_layers(tc, stage);
}

inline ::shelf::state::GdnStatePoolLayout make_qwen35_gdn_state_layout(
    const shelf::qwen_hybrid::QwenHybridConfig& tc) {
    return ::shelf::runtime::make_gdn_state_layout(tc);
}

inline ::shelf::state::GdnStatePoolLayout make_qwen35_gdn_state_layout_local(
    const shelf::qwen_hybrid::QwenHybridConfig& tc, uint32_t gdn_layers,
    uint32_t local_key_heads, uint32_t local_value_heads) {
    return ::shelf::runtime::make_gdn_state_layout_local(
        tc, gdn_layers, local_key_heads, local_value_heads);
}

inline Result<::shelf::state::GdnStatePoolLayout> make_qwen35_gdn_state_layout_stage(
    const shelf::qwen_hybrid::QwenHybridConfig& tc, const PipelineStage& stage,
    uint32_t tp_size) {
    return ::shelf::runtime::make_gdn_state_layout_stage(tc, stage, tp_size);
}

inline Result<::shelf::state::GdnStatePoolLayout> make_qwen35_gdn_state_layout_tp(
    const shelf::qwen_hybrid::QwenHybridConfig& tc, uint32_t tp_size) {
    return ::shelf::runtime::make_gdn_state_layout_tp(tc, tp_size);
}

}  // namespace shelf::qwen35
