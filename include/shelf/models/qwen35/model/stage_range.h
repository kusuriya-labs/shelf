#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace shelf::qwen35 {

struct StageRange {
    std::uint32_t layer_lo = 0;
    std::uint32_t layer_hi = 0;

    std::uint32_t layer_count() const noexcept {
        return layer_hi > layer_lo ? layer_hi - layer_lo : 0u;
    }

    bool operator==(const StageRange&) const = default;
};

struct PipelineStage {
    bool enabled = false;
    StageRange layers;

    bool first() const noexcept { return layer_lo() == 0u; }
    bool last(std::uint32_t num_layers) const noexcept {
        return enabled && layers.layer_hi == num_layers;
    }

    std::uint32_t layer_lo() const noexcept { return layers.layer_lo; }
    std::uint32_t layer_hi() const noexcept { return layers.layer_hi; }
};

inline std::vector<StageRange> even_stage_ranges(std::uint32_t num_layers,
                                                 std::uint32_t stages) {
    std::vector<StageRange> out;
    if (num_layers == 0u || stages == 0u || stages > num_layers) return out;
    const std::uint32_t per = num_layers / stages;
    std::uint32_t lo = 0;
    for (std::uint32_t s = 0; s < stages; ++s) {
        const std::uint32_t hi =
            (s + 1u == stages) ? num_layers : lo + per;
        out.push_back(StageRange{lo, hi});
        lo = hi;
    }
    return out;
}

}  // namespace shelf::qwen35
