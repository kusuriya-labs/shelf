#pragma once

#include <cstdint>

namespace shelf {
namespace runtime {

struct EffectiveRopeProfile {
    enum class Type : std::uint8_t { Default, Yarn, NtkAlpha };
    Type type = Type::Default;
    double theta = 10000000.0;
    std::uint32_t rotary_dim = 0;
    std::uint32_t original_max_position_embeddings = 0;
    double factor = 1.0;
    double attention_factor = 1.0;
    double beta_fast = 32.0;
    double beta_slow = 1.0;
    double alpha = 0.0;
    bool truncate = true;
    bool from_cli = false;
    bool float32_frequencies = false;
};

}  // namespace runtime
}  // namespace shelf
