#pragma once

#include "compute_runtime.h"

#include <shelf/runtime/dflash2_config.h>
#include <shelf/runtime/dflash2/executor.h>
#include <shelf/runtime/dflash2/tp_proposal.h>

#include <memory>
#include <shelf/runtime/dflash2/weights.h>
#include <shelf/models/qwen35/runtime/dflash2_session_pool.h>
#include <shelf/runtime/dflash2_spec_decoder.h>

#include <cstdint>
#include <optional>
#include <vector>
#include <string>

namespace shelf::app {

struct DFlashServeContext {
    bool enabled = false;
    bool adaptive = false;
    ::shelf::runtime::DFlash2Config config{};
    std::string model_dir;
    uint32_t drafts = 7u;
    ::shelf::runtime::dflash2::DFlash2Weights weights{};
    std::unique_ptr<::shelf::runtime::dflash2::DFlash2TpProposal> tp_proposal;
    ::shelf::runtime::dflash2::DFlash2Executor executor{};
    bool executor_ready = false;
    shelf::qwen35::runtime::DFlash2SpecBatchContext batch{};
    bool batch_ready = false;
    shelf::qwen35::runtime::DFlash2SessionPool sessions{};
    bool sessions_ready = false;
    shelf::qwen35::runtime::DFlash2BatcherStats stats{};
};

int run_serve_stdio(Qwen35ComputeRuntime& runtime, DFlashServeContext& dflash);

}  // namespace shelf::app
