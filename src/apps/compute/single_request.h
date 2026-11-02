#pragma once

#include "compute_runtime.h"

#include <shelf/models/qwen35/dflash2/config.h>
#include <shelf/models/qwen35/runtime/scheduling_types.h>

#include <cstdint>
#include <string>
#include <vector>

namespace shelf::app {

int run_single_shot(Qwen35ComputeRuntime& runtime,
                    const std::vector<int32_t>& prompt_ids,
                    unsigned long max_new_tokens,
                    const shelf::qwen35::runtime::SamplingConfig& sampling,
                    const std::string& kv_cache_dtype,
                    const std::string& dump_logits_path);

int run_dflash2_shot(Qwen35ComputeRuntime& runtime,
                     const std::vector<int32_t>& prompt_ids,
                     unsigned long max_new_tokens,
                     const shelf::qwen35::dflash2::DFlash2Config& dflash_config,
                     const std::string& dflash2_model_dir,
                     unsigned long dflash2_drafts,
                     int dflash2_stats,
                     const std::string& kv_cache_dtype,
                     const shelf::qwen35::runtime::SamplingConfig& sampling);

}  // namespace shelf::app
