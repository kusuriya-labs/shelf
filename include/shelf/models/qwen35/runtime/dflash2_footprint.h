#pragma once

#include <shelf/models/qwen35/dflash2/config.h>
#include <shelf/models/qwen35/runtime/dflash2_spec_decoder.h>
#include <shelf/inference/state/gdn_state_pool.h>

#include <cstddef>
#include <cstdint>

namespace shelf::qwen35 {
namespace runtime {

struct DFlash2Footprint {
    std::size_t draft_weights = 0u;
    std::size_t draft_scratch = 0u;
    std::size_t session_slot = 0u;
    std::size_t spec_batch = 0u;
    std::size_t sessions = 0u;

    std::size_t total() const noexcept {
        return draft_weights + draft_scratch + spec_batch + session_slot * sessions;
    }
};

bool dflash2_prefill_rows(const dflash2::DFlash2Config& config, uint32_t tp,
                          const char* value, uint32_t& rows) noexcept;

std::size_t dflash2_context_state_bytes(const dflash2::DFlash2Config& config) noexcept;

std::size_t dflash2_spec_workspace_bytes(const DFlash2SpecDecoderConfig& config,
                                         const ::shelf::state::GdnStatePool& pool,
                                         bool step_buffers = true,
                                         bool rec_snapshot = true) noexcept;

std::size_t dflash2_draft_scratch_bytes(const dflash2::DFlash2Config& config,
                                        std::size_t rows,
                                        std::size_t lm_head_k,
                                        std::size_t prefill_rows = 0u,
                                        bool exl3_mul24_head = false) noexcept;

std::size_t dflash2_batch_tree_metadata_bytes(std::size_t max_rows, uint32_t tree_budget) noexcept;

std::size_t dflash2_spec_batch_context_bytes(const ::shelf::state::GdnStatePool& pool,
                                             std::size_t max_rows,
                                             uint32_t tree_budget = 0u) noexcept;

DFlash2Footprint dflash2_footprint(const dflash2::DFlash2Config& config,
                                   const DFlash2SpecDecoderConfig& spec_config,
                                   const ::shelf::state::GdnStatePool& pool,
                                   std::size_t draft_weights_bytes,
                                   std::size_t lm_head_k,
                                   uint32_t sessions,
                                   std::size_t prefill_rows = 0u,
                                   bool exl3_mul24_head = false) noexcept;

}  // namespace runtime
}  // namespace shelf::qwen35
