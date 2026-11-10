#pragma once

#include <shelf/models/common/qwen_hybrid/config.h>
#include <shelf/models/qwen35/model/qwen35_config.h>
#include <shelf/models/common/qwen_hybrid/stage_range.h>
#include <shelf/models/qwen35/state/state_types.h>
#include <shelf/core/status.h>

namespace shelf {
namespace qwen35 {

using ::shelf::qwen_hybrid::PipelineStage;

inline std::uint32_t qwen35_stage_layer_count(const shelf::qwen_hybrid::QwenHybridConfig& tc,
                                              const PipelineStage& stage) {
    if (!stage.enabled) return static_cast<std::uint32_t>(tc.num_hidden_layers);
    return stage.layers.layer_count();
}

inline std::uint32_t qwen35_stage_gdn_layers(const shelf::qwen_hybrid::QwenHybridConfig& tc,
                                             const PipelineStage& stage) {
    if (!stage.enabled) {
        return static_cast<std::uint32_t>(
            shelf::qwen_hybrid::count_layer_topology(tc).gdn_layers);
    }
    std::uint32_t count = 0;
    for (std::uint32_t l = stage.layers.layer_lo; l < stage.layers.layer_hi; ++l) {
        if (shelf::qwen_hybrid::layer_is_gdn(tc, l)) ++count;
    }
    return count;
}

inline std::uint32_t qwen35_stage_attention_layers(const shelf::qwen_hybrid::QwenHybridConfig& tc,
                                                   const PipelineStage& stage) {
    if (!stage.enabled) {
        return static_cast<std::uint32_t>(
            shelf::qwen_hybrid::count_layer_topology(tc).attention_layers);
    }
    std::uint32_t count = 0;
    for (std::uint32_t l = stage.layers.layer_lo; l < stage.layers.layer_hi; ++l) {
        if (!shelf::qwen_hybrid::layer_is_gdn(tc, l)) ++count;
    }
    return count;
}

inline state::GdnStatePoolLayout make_qwen35_gdn_state_layout(
    const shelf::qwen_hybrid::QwenHybridConfig& tc) {
    uint32_t gdn_layers =
        static_cast<uint32_t>(shelf::qwen_hybrid::count_layer_topology(tc).gdn_layers);
    if (gdn_layers == 0 && tc.declared_model_type == "plamo3") return {};
    if (gdn_layers == 0) gdn_layers = 1;
    const uint32_t kh = tc.linear_num_key_heads != 0 ? tc.linear_num_key_heads : 16;
    const uint32_t vh = tc.linear_num_value_heads != 0 ? tc.linear_num_value_heads : 16;
    const uint32_t khd = tc.linear_key_head_dim != 0 ? tc.linear_key_head_dim : 128;
    const uint32_t vhd = tc.linear_value_head_dim != 0 ? tc.linear_value_head_dim : 128;
    uint32_t kern = tc.linear_conv_kernel_dim != 0 ? tc.linear_conv_kernel_dim : 4;
    if (kern == 0) kern = 4;
    return state::GdnStatePoolLayout::make(
        gdn_layers, 2 * kh * khd + vh * vhd, kern - 1, vh, khd, vhd);
}

inline state::GdnStatePoolLayout make_qwen35_gdn_state_layout_local(
    const shelf::qwen_hybrid::QwenHybridConfig& tc, uint32_t gdn_layers,
    uint32_t local_key_heads, uint32_t local_value_heads) {
    if (gdn_layers == 0) gdn_layers = 1;
    const uint32_t khd = tc.linear_key_head_dim != 0 ? tc.linear_key_head_dim : 128;
    const uint32_t vhd = tc.linear_value_head_dim != 0 ? tc.linear_value_head_dim : 128;
    uint32_t kern = tc.linear_conv_kernel_dim != 0 ? tc.linear_conv_kernel_dim : 4;
    if (kern == 0) kern = 4;
    return state::GdnStatePoolLayout::make(
        gdn_layers, 2u * local_key_heads * khd + local_value_heads * vhd, kern - 1,
        local_value_heads, khd, vhd);
}

inline Result<state::GdnStatePoolLayout> make_qwen35_gdn_state_layout_stage(
    const shelf::qwen_hybrid::QwenHybridConfig& tc, const PipelineStage& stage, uint32_t tp_size) {
    if (tp_size == 0) {
        return Status::invalid_argument("tp_size must be at least 1", __FILE__,
                                        __LINE__);
    }
    uint32_t gdn_layers = qwen35_stage_gdn_layers(tc, stage);
    if (gdn_layers == 0) gdn_layers = 1;
    const uint32_t kh = tc.linear_num_key_heads != 0 ? tc.linear_num_key_heads : 16;
    const uint32_t vh = tc.linear_num_value_heads != 0 ? tc.linear_num_value_heads : 16;
    const uint32_t khd = tc.linear_key_head_dim != 0 ? tc.linear_key_head_dim : 128;
    const uint32_t vhd = tc.linear_value_head_dim != 0 ? tc.linear_value_head_dim : 128;
    uint32_t kern = tc.linear_conv_kernel_dim != 0 ? tc.linear_conv_kernel_dim : 4;
    if (kern == 0) kern = 4;
    if ((kh % tp_size) != 0 || (vh % tp_size) != 0) {
        return Status::unsupported(
            "qwen35 tp requires GDN head counts divisible by tp_size",
            __FILE__, __LINE__);
    }
    const uint32_t local_kh = kh / tp_size;
    const uint32_t local_vh = vh / tp_size;
    return state::GdnStatePoolLayout::make(
        gdn_layers, 2u * local_kh * khd + local_vh * vhd, kern - 1,
        local_vh, khd, vhd);
}

inline Result<state::GdnStatePoolLayout> make_qwen35_gdn_state_layout_tp(
    const shelf::qwen_hybrid::QwenHybridConfig& tc, uint32_t tp_size) {
    if (tp_size == 0) {
        return Status::invalid_argument("tp_size must be at least 1", __FILE__,
                                        __LINE__);
    }
    const state::GdnStatePoolLayout global = make_qwen35_gdn_state_layout(tc);
    if (tp_size == 1) return global;
    const uint32_t kh = tc.linear_num_key_heads != 0 ? tc.linear_num_key_heads : 16;
    const uint32_t vh = tc.linear_num_value_heads != 0 ? tc.linear_num_value_heads : 16;
    if ((kh % tp_size) != 0 || (vh % tp_size) != 0) {
        return Status::unsupported(
            "qwen35 tp requires GDN head counts divisible by tp_size",
            __FILE__, __LINE__);
    }
    const uint32_t local_kh = kh / tp_size;
    const uint32_t local_vh = vh / tp_size;
    return state::GdnStatePoolLayout::make(
        global.num_gdn_states,
        2u * local_kh * global.head_k + local_vh * global.head_v,
        global.conv_history, local_vh, global.head_k, global.head_v);
}

}
}
