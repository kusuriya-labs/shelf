#pragma once
#include <shelf/core/status.h>
#include <cstdint>
#include <string>

namespace ps {
namespace qwen35 {

struct Qwen35RopeParameters {
    std::string rope_type = "default";
    double theta = 10000000.0;
    double partial_rotary_factor = 0.25;
    bool has_factor = false;
    double factor = 1.0;
    std::uint32_t original_max_position_embeddings = 0;
    std::uint32_t max_position_embeddings = 0;
    std::uint32_t head_dim = 0;
    double beta_fast = 32.0;
    double beta_slow = 1.0;
    bool truncate = true;
    bool has_attention_factor = false;
    double attention_factor = 1.0;
};

enum class RopeFactorSelection : std::uint8_t { Auto, One, Two, Four };

struct EffectiveRopeProfile {
    enum class Type : std::uint8_t { Default, Yarn };
    Type type = Type::Default;
    double theta = 10000000.0;
    std::uint32_t rotary_dim = 0;
    std::uint32_t original_max_position_embeddings = 0;
    double factor = 1.0;
    double attention_factor = 1.0;
    double beta_fast = 32.0;
    double beta_slow = 1.0;
    bool truncate = true;
    bool from_cli = false;
};

Result<Qwen35RopeParameters> read_qwen35_rope_parameters(const std::string& model_dir);

Result<RopeFactorSelection> parse_rope_factor_selection(const std::string& text);

Result<EffectiveRopeProfile> resolve_effective_rope_profile(
    const Qwen35RopeParameters& params, RopeFactorSelection selection);

std::uint32_t rope_max_seq_len_cap(const EffectiveRopeProfile& profile);

}
}
