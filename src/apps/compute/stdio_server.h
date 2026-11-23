#pragma once

#include <shelf/engine/compute_session.h>
#include <shelf/runtime/dflash2_config.h>

#include <cstdint>
#include <string>

namespace shelf::app {

struct DFlashServeContext {
    bool enabled = false;
    bool adaptive = false;
    ::shelf::runtime::DFlash2Config config{};
    std::string model_dir;
    uint32_t drafts = 7u;
};

int run_serve_stdio(shelf::engine::ComputeSession& runtime, DFlashServeContext& dflash);

}  // namespace shelf::app
