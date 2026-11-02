#pragma once

#include "compute_runtime.h"

#include <shelf/models/qwen35/dflash2/config.h>
#include <shelf/models/qwen35/dflash2/executor.h>
#include <shelf/models/qwen35/dflash2/weights.h>
#include <shelf/models/qwen35/runtime/dflash2_session_pool.h>
#include <shelf/models/qwen35/runtime/dflash2_spec_decoder.h>
#include <shelf/resident/resident_client.h>

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
};

int run_serve_stdio(Qwen35ComputeRuntime& runtime, DFlashServeContext& dflash);

}  // namespace ps::app
