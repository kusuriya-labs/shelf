// shelf model/qwen35 — text-path configuration (plan Phase: model semantics).
//
// This module is the ONLY place model-family names may appear (enforced by
// tests/architecture). It interprets the checkpoint's config.json into a
// neutral per-layer description that model/qwen35 lowering consumes.
#pragma once

#include <string>
#include <vector>

#include <shelf/status.hpp>

namespace shelf::model::qwen35 {

enum class LayerKind : std::uint8_t { linear_attention, full_attention };

struct TextConfig {
    std::size_t hidden_size = 0;
    std::size_t num_layers = 0;
    std::size_t vocab_size = 0;

    // full attention
    std::size_t num_heads = 0;
    std::size_t num_kv_heads = 0;
    std::size_t head_dim = 0;
    std::size_t rotary_dim = 0;       // partial_rotary_factor * head_dim
    double rope_theta = 0.0;

    // gated deltanet (linear attention)
    std::size_t lin_key_heads = 0;
    std::size_t lin_value_heads = 0;
    std::size_t lin_key_head_dim = 0;
    std::size_t lin_value_head_dim = 0;
    std::size_t conv_kernel = 0;      // 4

    std::size_t intermediate_size = 0;   // dense MLP の幅（MoE 専用モデルでは 0）

    // ---- MoE（Qwen3.5-MoE 系）----
    // num_experts = 0 は dense MLP。num_experts > 0 のとき、`mlp_only_layers` に
    // 載っている層だけが dense MLP で、残りは MoE ブロックになる
    // （transformers の Qwen3_5MoeTextConfig と同じ判定）。
    std::size_t num_experts = 0;
    std::size_t num_experts_per_tok = 0;
    std::size_t moe_intermediate_size = 0;            // エキスパート 1 個の幅
    std::size_t shared_expert_intermediate_size = 0;  // 共有エキスパートの幅（0 = 無し）
    std::vector<std::size_t> mlp_only_layers;         // MoE モデル内の dense 層

    double rms_eps = 0.0;
    std::vector<LayerKind> layer_kinds;
    bool tie_word_embeddings = false;
    // 停止トークン。`eos_token_id` は先頭 ID（後方互換）、`eos_token_ids` は全要素。
    std::int64_t eos_token_id = -1;
    std::vector<std::int64_t> eos_token_ids;

    // q_proj width includes the per-element output gate: [2*nq*D, H].
    std::size_t q_gate_out_features() const noexcept { return 2 * num_heads * head_dim; }
    std::size_t kv_out_features() const noexcept { return num_kv_heads * head_dim; }
    std::size_t conv_channels() const noexcept {
        return 2 * lin_key_heads * lin_key_head_dim + lin_value_heads * lin_value_head_dim;
    }

    // 層 l の MLP が MoE か（dense MLP か）を返す。
    bool is_moe_layer(std::size_t l) const noexcept {
        if (num_experts == 0 || num_experts_per_tok == 0) return false;
        for (const auto only : mlp_only_layers)
            if (only == l) return false;
        return true;
    }
    bool has_moe() const noexcept { return num_experts > 0 && num_experts_per_tok > 0; }
    bool has_shared_expert() const noexcept { return shared_expert_intermediate_size > 0; }
};

// Parses config.json (the file may describe a multimodal checkpoint; only
// text_config is interpreted). Unknown rope types are rejected.
Result<TextConfig> load_text_config_json(const std::string& json_text);

} // namespace shelf::model::qwen35
