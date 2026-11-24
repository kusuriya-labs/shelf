#include <shelf/models/qwen4exp/runtime/ple_input.h>

#include <algorithm>
#include <utility>

namespace shelf {
namespace qwen4exp {

std::shared_ptr<PleInputRuntime> PleInputRuntime::create(
    std::shared_ptr<const PleTable> table, std::uint32_t hidden) {
    if (table == nullptr || hidden == 0u) return nullptr;
    return std::shared_ptr<PleInputRuntime>(
        new PleInputRuntime(std::move(table), hidden));
}

void PleInputRuntime::reset_sequence(std::uint32_t slot) {
    history_.erase(slot);
}

Status PleInputRuntime::seed(std::uint32_t slot,
                             std::span<const std::int32_t> tokens) {
    if (table_ == nullptr) {
        return Status::invalid_state("PLE input runtime is empty", __FILE__, __LINE__);
    }
    std::vector<std::int32_t>& history = history_[slot];
    history.assign(tokens.begin(), tokens.end());
    return Status::make_ok();
}

Status PleInputRuntime::prepare(const ::shelf::scheduling::ScheduledBatch& batch,
                                std::vector<bf16_t>& out) {
    if (table_ == nullptr || batch.num_requests > batch.requests.size() || batch.token_ids == nullptr || batch.num_requests == 0u) {
        return Status::invalid_argument("PLE input batch is empty", __FILE__, __LINE__);
    }
    if (batch.token_ids_location != ::shelf::scheduling::TokenIdsLocation::Host) {
        return Status::unsupported(
            "the PLE input gather needs host token ids", __FILE__, __LINE__);
    }
    out.assign(static_cast<std::size_t>(batch.num_tokens) * hidden_, bf16_t{0});
    std::uint32_t row_offset = 0;
    for (std::uint32_t i = 0; i < batch.num_requests; ++i) {
        const auto& request = batch.requests[i];
        if (request.sequence == nullptr) {
            return Status::invalid_argument("PLE request has no sequence", __FILE__,
                                            __LINE__);
        }
        const std::uint32_t slot = request.sequence->slot;
        const std::uint32_t position = request.sequence->position;
        auto& history = history_[slot];
        if (position == 0u && !history.empty()) history.clear();
        if (history.size() > position) history.resize(position);
        if (history.size() < position) {
            return Status::invalid_state(
                "PLE token history does not cover the request prefix", __FILE__,
                __LINE__);
        }
        if (request.token_begin > batch.num_tokens || request.num_tokens > batch.num_tokens - request.token_begin ||
            request.num_tokens > batch.num_tokens - row_offset)
            return Status::invalid_argument("PLE token range exceeds the batch", __FILE__, __LINE__);
        const std::size_t count = request.num_tokens;
        const std::span<const std::int32_t> tokens(
            batch.token_ids + request.token_begin, count);
        auto rows = table_->gather_bf16_batch(history, tokens);
        if (!rows.ok()) return rows.status();
        if (rows.value().size() != count * hidden_) {
            return Status::invalid_state("PLE gather width mismatch", __FILE__,
                                         __LINE__);
        }
        std::copy(rows.value().begin(), rows.value().end(),
                  out.begin() + static_cast<std::size_t>(row_offset) * hidden_);
        history.insert(history.end(), tokens.begin(), tokens.end());
        row_offset += request.num_tokens;
    }
    if (row_offset != batch.num_tokens) {
        return Status::invalid_state("PLE input row count mismatch", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

}  // namespace qwen4exp
}  // namespace shelf
