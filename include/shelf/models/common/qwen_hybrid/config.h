#pragma once
#include <shelf/core/status.h>
#include <shelf/models/common/qwen_hybrid/stop_tokens.h>

#include <cstddef>
#include <string>
#include <vector>

namespace shelf {
namespace qwen_hybrid {

// The model geometry that does not depend on how a layer's FFN is built: the hybrid
// backbone (GDN plus full attention), the norms, the rope setup and the embedding
// shape. A dense model and an MoE model share all of it; what differs is the FFN
// section that each model family adds next to it.
struct QwenHybridConfig {
    std::size_t vocab_size = 0;
    std::size_t hidden_size = 0;
    std::size_t num_hidden_layers = 0;

    std::size_t full_attention_interval = 0;

    std::size_t linear_num_key_heads = 0;
    std::size_t linear_num_value_heads = 0;
    std::size_t linear_key_head_dim = 0;
    std::size_t linear_value_head_dim = 0;
    std::size_t linear_conv_kernel_dim = 0;

    std::size_t num_attention_heads = 0;
    std::size_t num_key_value_heads = 0;
    std::size_t attention_head_dim = 0;

    StopTokens stop_tokens;

    float rms_norm_eps = 1.0e-6f;
    float rope_theta = 10000000.0f;
    float partial_rotary_factor = 0.25f;

    bool tie_word_embeddings = false;
    bool output_gate_silu = false;

    std::vector<int> layer_types;

    // What the checkpoint declares itself to be. Shelf reads it as a gate only: the
    // supported set is fixed at build time, so an unimplemented architecture is
    // rejected before any tensor is opened. It never selects code paths per model.
    std::string declared_model_type;
    std::string declared_text_model_type;
    std::vector<std::string> declared_architectures;
    // Read from the declared identity, not from the FFN keys. A model that declares a
    // routed-expert identity is not read by a dense reader even when the expert count is
    // spelled in a way this build does not know; the opposite direction stays decided by
    // the structure, because that is the part that can be checked against the weights.
    bool identity_declares_routed_experts = false;
};

// Reads the shared part of config.json. FFN keys (an intermediate size, an expert
// count) are left to the model family that owns them.
// config.json is rejected here when it does not declare one of the model identities
// this build implements (Status::unsupported), so an unimplemented architecture
// cannot reach tensor lookup.
Result<QwenHybridConfig> read_qwen_hybrid_config(const std::string& model_dir);

// The identities this build implements, for diagnostics and tests.
const std::vector<std::string>& supported_model_identities();

// Gates config.json on the identity it declares, without reading any other field.
// Every config entry point gates on this first, so an unimplemented architecture is
// named as such instead of failing later on a tensor name.
Status check_config_identity(const std::string& model_dir);

Result<bool> read_tie_word_embeddings(const std::string& model_dir);

// A model family owns its FFN keys (an intermediate size, an expert count), but
// config.json is parsed in one place. These accessors hand a family the text_config
// values it asked for: a missing key reads as 0, and has_text_config_key says whether
// the checkpoint wrote the key at all.
Result<std::vector<std::size_t>> read_text_config_sizes(
    const std::string& model_dir,
    const std::vector<std::string>& keys);

Result<bool> has_text_config_key(const std::string& model_dir, const char* key);

// Layer topology rule: layer_types when it covers every layer, otherwise derived
// from full_attention_interval.
bool layer_is_gdn(const QwenHybridConfig& config, std::size_t layer);

// How many layers of each kind the model has, from the same rule. State pools and
// the KV pool are sized from this, so counting layer_types directly (which a
// checkpoint may not declare) cannot disagree with what the loader builds.
struct HybridLayerCounts {
    std::size_t gdn_layers = 0;
    std::size_t attention_layers = 0;
};

HybridLayerCounts count_layer_topology(const QwenHybridConfig& config) noexcept;

}
}
