#pragma once

#include <shelf/core/memory/types.h>
#include <shelf/core/status.h>
#include <shelf/models/qwen35/state/state_types.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace shelf::qwen35::runtime {

enum class SpecPhase : uint8_t {
    IDLE = 0u,
    DRAFTING = 1u,
    VERIFYING = 2u,
    COMMITTING = 3u,
    ROLLING_BACK = 4u,
};

const char* spec_phase_name(SpecPhase phase) noexcept;

struct SpecVerifyResult {
    uint32_t num_drafts = 0u;
    uint32_t num_accepted_drafts = 0u;
    uint32_t first_reject_index = 0xFFFFFFFFu;
    bool correction_valid = false;
    int32_t correction_token = -1;
    bool bonus_token_valid = false;
    int32_t bonus_token = -1;
    uint32_t num_emitted_tokens = 0u;
    std::vector<int32_t> emitted_tokens;
    uint32_t num_committed_target_tokens = 0u;
    uint32_t num_mtp_drafts = 0xFFFFFFFFu;
};

Status spec_greedy_accept(
    const int32_t* drafts,
    uint32_t num_drafts,
    const int32_t* target_tokens,
    bool bonus_token_enabled,
    SpecVerifyResult& out);

std::size_t spec_gdn_conv_bytes(const GdnStatePool& pool) noexcept;

std::size_t spec_gdn_recurrent_bytes(const GdnStatePool& pool) noexcept;

Status spec_gdn_snapshot(
    const GdnStatePool& pool,
    SequenceSlotId slot,
    void* conv_dst,
    void* recurrent_dst,
    std::size_t conv_bytes,
    std::size_t recurrent_bytes,
    hipStream_t stream);

Status spec_gdn_restore(
    const GdnStatePool& pool,
    SequenceSlotId slot,
    const void* conv_src,
    const void* recurrent_src,
    std::size_t conv_bytes,
    std::size_t recurrent_bytes,
    hipStream_t stream);

}  // namespace shelf::qwen35::runtime
