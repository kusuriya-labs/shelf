#pragma once

#include "compute_runtime.h"

#include <phaseshift/models/qwen35/dflash2/config.h>
#include <phaseshift/models/qwen35/eagle4/config.h>
#include <phaseshift/models/qwen35/eagle4/executor.h>
#include <phaseshift/models/qwen35/eagle4/weights.h>
#include <phaseshift/models/qwen35/dflash2/executor.h>
#include <phaseshift/models/qwen35/dflash2/weights.h>
#include <phaseshift/models/qwen35/runtime/dflash2_session_pool.h>
#include <phaseshift/models/qwen35/runtime/dflash2_spec_decoder.h>
#include <phaseshift/resident/resident_client.h>

#include <cstdint>
#include <optional>
#include <vector>
#include <string>

namespace ps::app {

struct DFlashServeContext {
    bool enabled = false;
    ps::qwen35::dflash2::DFlash2Config config{};
    std::string model_dir;
    uint32_t drafts = 7u;
    ps::qwen35::dflash2::DFlash2Weights weights{};
    ps::qwen35::dflash2::DFlash2Executor executor{};
    bool executor_ready = false;
    ps::qwen35::runtime::DFlash2SpecBatchContext batch{};
    bool batch_ready = false;
    ps::qwen35::runtime::DFlash2SessionPool sessions{};
    bool sessions_ready = false;
    ps::qwen35::runtime::DFlash2BatcherStats stats{};

    bool eagle4_enabled = false;
    std::string eagle4_model_dir;
    uint32_t eagle4_drafts = 7u;
    std::string eagle4_candidate_head = "lowrank";
    std::string eagle4_lm_head = "target";
    ps::qwen35::eagle4::Eagle4Config eagle4_config{};

    struct Eagle4RankSlot {
        std::optional<ps::resident::ResidentModelAttachment> resident;
        ps::qwen35::eagle4::Eagle4Weights weights{};
        ps::qwen35::eagle4::Eagle4Executor executor{};
        bool ready = false;
    };
    std::vector<Eagle4RankSlot> eagle4_ranks;
};

int run_serve_stdio(Qwen35ComputeRuntime& runtime, DFlashServeContext& dflash);

}  // namespace ps::app
