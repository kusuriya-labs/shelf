#pragma once

#include <shelf/core/status.h>
#include <shelf/engine/compute_session.h>
#include <shelf/runtime/dflash2_config.h>
#include <shelf/runtime/scheduling_types.h>

#include <cstdint>
#include <string>
#include <vector>

namespace shelf::engine {

struct SpeculationMetric {
    std::string key;
    std::string value;
};

struct SpeculationResult {
    std::vector<int32_t> generated_ids;
    uint32_t decode_tokens = 0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    std::vector<SpeculationMetric> stats;
};

class SpeculationRunner {
 public:
    static Result<SpeculationResult> run_mtp(ComputeSession& session,
                                             const std::vector<int32_t>& prompt_ids,
                                             unsigned long max_new_tokens,
                                             uint32_t mtp_drafts,
                                             const runtime::SamplingConfig& sampling);

    static Result<SpeculationResult> run_dflash2(
        ComputeSession& session,
        const std::vector<int32_t>& prompt_ids,
        unsigned long max_new_tokens,
        const runtime::DFlash2Config& dflash_config,
        unsigned long dflash2_drafts,
        bool adaptive,
        bool dflash2_tree,
        unsigned long dflash2_tree_extra,
        unsigned long dflash2_tree_budget,
        const runtime::SamplingConfig& sampling);
};

}  // namespace shelf::engine
