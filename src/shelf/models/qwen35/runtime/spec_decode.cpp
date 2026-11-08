#include <shelf/models/qwen35/runtime/spec_decode.h>

#include <algorithm>
#include <cstring>

namespace shelf::qwen35::runtime {

const char* spec_phase_name(SpecPhase phase) noexcept {
    switch (phase) {
        case SpecPhase::IDLE: return "IDLE";
        case SpecPhase::DRAFTING: return "DRAFTING";
        case SpecPhase::VERIFYING: return "VERIFYING";
        case SpecPhase::COMMITTING: return "COMMITTING";
        case SpecPhase::ROLLING_BACK: return "ROLLING_BACK";
    }
    return "UNKNOWN";
}

Status spec_greedy_accept(
    const int32_t* drafts,
    uint32_t num_drafts,
    const int32_t* target_tokens,
    bool bonus_token_enabled,
    SpecVerifyResult& out) {

    if (target_tokens == nullptr) {
        return Status::invalid_argument("spec_greedy_accept target_tokens is null", __FILE__, __LINE__);
    }
    if (num_drafts > 0u && drafts == nullptr) {
        return Status::invalid_argument("spec_greedy_accept drafts is null", __FILE__, __LINE__);
    }

    out = SpecVerifyResult{};
    out.num_drafts = num_drafts;

    if (num_drafts == 0u) {
        out.first_reject_index = 0u;
        out.num_committed_target_tokens = 0u;
        return Status::make_ok();
    }

    uint32_t accepted = 0u;
    while (accepted < num_drafts && drafts[accepted] == target_tokens[accepted]) {
        ++accepted;
    }
    out.num_accepted_drafts = accepted;

    if (accepted == num_drafts) {
        out.first_reject_index = 0xFFFFFFFFu;
        if (bonus_token_enabled) {
            out.bonus_token_valid = true;
            out.bonus_token = target_tokens[num_drafts];
        }
    } else {
        out.first_reject_index = accepted;
        out.correction_valid = true;
        out.correction_token = target_tokens[accepted];
    }

    out.emitted_tokens.reserve(static_cast<std::size_t>(accepted) + 1u);
    for (uint32_t i = 0u; i < accepted; ++i) {
        out.emitted_tokens.push_back(drafts[i]);
    }
    if (out.correction_valid) {
        out.emitted_tokens.push_back(out.correction_token);
    } else if (out.bonus_token_valid) {
        out.emitted_tokens.push_back(out.bonus_token);
    }
    out.num_emitted_tokens = static_cast<uint32_t>(out.emitted_tokens.size());
    out.num_committed_target_tokens = accepted + 1u;
    return Status::make_ok();
}

bool spec_bound_result(SpecVerifyResult& result, uint32_t remaining_tokens,
                       const qwen_hybrid::StopTokens& stop_tokens) {
    const uint32_t limit = std::min(remaining_tokens,
        static_cast<uint32_t>(result.emitted_tokens.size()));
    uint32_t count = limit;
    bool stopped = false;
    for (uint32_t i = 0u; i < limit; ++i) {
        if (qwen_hybrid::is_stop_token(stop_tokens, result.emitted_tokens[i])) {
            count = i + 1u;
            stopped = true;
            break;
        }
    }
    if (count > 0u && count < result.emitted_tokens.size()) {
        result.emitted_tokens.resize(count);
        result.num_accepted_drafts = std::min(result.num_accepted_drafts, count - 1u);
        result.num_committed_target_tokens = result.num_accepted_drafts + 1u;
        result.first_reject_index = result.num_accepted_drafts;
        result.correction_valid = true;
        result.correction_token = result.emitted_tokens.back();
        result.bonus_token_valid = false;
    }
    result.num_emitted_tokens = count;
    return stopped;
}

std::size_t spec_gdn_conv_bytes(const GdnStatePool& pool) noexcept {
    return static_cast<std::size_t>(pool.device_view().conv_slot_stride) * sizeof(bf16_t);
}

std::size_t spec_gdn_recurrent_bytes(const GdnStatePool& pool) noexcept {
    return static_cast<std::size_t>(pool.device_view().recurrent_slot_stride) * sizeof(fp16_t);
}

Status spec_gdn_snapshot(
    const GdnStatePool& pool,
    SequenceSlotId slot,
    void* conv_dst,
    void* recurrent_dst,
    std::size_t conv_bytes,
    std::size_t recurrent_bytes,
    hipStream_t stream) {

    if (conv_dst == nullptr || recurrent_dst == nullptr) {
        return Status::invalid_argument("spec_gdn_snapshot outputs are null", __FILE__, __LINE__);
    }
    if (slot >= pool.max_sequences()) {
        return Status::out_of_range("spec_gdn_snapshot slot out of range", __FILE__, __LINE__);
    }
    if (conv_bytes != spec_gdn_conv_bytes(pool) ||
        recurrent_bytes != spec_gdn_recurrent_bytes(pool)) {
        return Status::invalid_argument("spec_gdn_snapshot byte size mismatch", __FILE__, __LINE__);
    }
    const GdnStatePoolDeviceView view = pool.device_view();
    bf16_t* conv_src = view.conv_base + static_cast<std::size_t>(slot) * view.conv_slot_stride;
    const fp16_t* rec_src = view.recurrent_base + static_cast<std::size_t>(slot) * view.recurrent_slot_stride;
    hipError_t err = hipMemcpyAsync(conv_dst, conv_src, conv_bytes, hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
        return Status::hip_error("spec_gdn_snapshot conv copy", hipGetErrorString(err), __FILE__, __LINE__);
    }
    err = hipMemcpyAsync(recurrent_dst, rec_src, recurrent_bytes, hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
        return Status::hip_error("spec_gdn_snapshot recurrent copy", hipGetErrorString(err), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status spec_gdn_restore(
    const GdnStatePool& pool,
    SequenceSlotId slot,
    const void* conv_src,
    const void* recurrent_src,
    std::size_t conv_bytes,
    std::size_t recurrent_bytes,
    hipStream_t stream) {

    if (conv_src == nullptr || recurrent_src == nullptr) {
        return Status::invalid_argument("spec_gdn_restore inputs are null", __FILE__, __LINE__);
    }
    if (slot >= pool.max_sequences()) {
        return Status::out_of_range("spec_gdn_restore slot out of range", __FILE__, __LINE__);
    }
    if (conv_bytes != spec_gdn_conv_bytes(pool) ||
        recurrent_bytes != spec_gdn_recurrent_bytes(pool)) {
        return Status::invalid_argument("spec_gdn_restore byte size mismatch", __FILE__, __LINE__);
    }
    const GdnStatePoolDeviceView view = pool.device_view();
    bf16_t* conv_dst = view.conv_base + static_cast<std::size_t>(slot) * view.conv_slot_stride;
    fp16_t* rec_dst =
        view.recurrent_base + static_cast<std::size_t>(slot) * view.recurrent_slot_stride;
    hipError_t err = hipMemcpyAsync(conv_dst, conv_src, conv_bytes, hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
        return Status::hip_error("spec_gdn_restore conv copy", hipGetErrorString(err), __FILE__, __LINE__);
    }
    err = hipMemcpyAsync(rec_dst, recurrent_src, recurrent_bytes, hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
        return Status::hip_error("spec_gdn_restore recurrent copy", hipGetErrorString(err), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

}  // namespace shelf::qwen35::runtime
