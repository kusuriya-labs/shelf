// shelf runtime/generate — 連続バッチングのスケジューラ実装（`batch_scheduler.hpp`）。
#include <shelf/runtime/generate/batch_scheduler.hpp>

#include <algorithm>

namespace shelf::runtime::generate {

BatchScheduler::BatchScheduler(std::size_t max_slots, std::size_t prefill_chunk)
    : max_slots_(std::max<std::size_t>(max_slots, 1u)),
      prefill_chunk_(std::max<std::size_t>(prefill_chunk, 1u)) {
    slots_.resize(max_slots_);
}

bool BatchScheduler::submit(BatchRequest req) {
    if (req.prompt_ids.empty() || req.max_new_tokens == 0u) return false;
    if (free_slots() != 0u) return admit(std::move(req));
    queued_.push_back(std::move(req));
    return false;
}

bool BatchScheduler::admit(BatchRequest req) {
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        Slot& s = slots_[i];
        if (s.state != SlotState::free) continue;
        s.state = SlotState::prefill;
        s.req = std::move(req);
        s.prefilled = 0;
        s.generated.clear();
        s.position = 0;
        s.feed = 0;
        s.end = SlotEnd::none;
        return true;
    }
    return false;
}

void BatchScheduler::admit_queued() {
    while (!queued_.empty() && free_slots() != 0u) {
        BatchRequest req = std::move(queued_.front());
        queued_.pop_front();
        if (!admit(std::move(req))) break;
    }
}

bool BatchScheduler::cancel(std::string_view request_id) {
    for (auto& s : slots_) {
        if (s.state == SlotState::free || s.state == SlotState::done) continue;
        if (s.req.request_id != request_id) continue;
        s.state = SlotState::done;
        s.end = SlotEnd::cancelled;
        return true;
    }
    if (find(request_id) < max_slots_) return true;   // すでに done（take 待ち）
    for (auto it = queued_.begin(); it != queued_.end(); ++it) {
        if (it->request_id != request_id) continue;
        Finished f;
        f.request_id = it->request_id;
        f.prompt_tokens = it->prompt_ids.size();
        f.end = SlotEnd::cancelled;
        finished_.push_back(std::move(f));
        queued_.erase(it);
        return true;
    }
    return false;
}

bool BatchScheduler::next_prefill(PrefillAction* out) {
    if (out == nullptr) return false;
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        Slot& s = slots_[i];
        if (s.state != SlotState::prefill) continue;
        const std::size_t total = s.req.prompt_ids.size();
        if (s.prefilled >= total) {
            // 念のため（progress が来ていないだけ）。decode へ送る。
            s.state = SlotState::decode;
            s.position = static_cast<std::uint32_t>(total);
            s.feed = s.req.prompt_ids.back();
            continue;
        }
        const std::size_t rows = std::min(prefill_chunk_, total - s.prefilled);
        out->slot = i;
        out->begin = s.prefilled;
        out->rows = rows;
        out->base_pos = static_cast<std::uint32_t>(s.prefilled);
        out->tokens = s.req.prompt_ids.data() + s.prefilled;
        return true;
    }
    return false;
}

void BatchScheduler::prefill_progress(std::size_t slot, std::size_t rows) {
    if (slot >= slots_.size()) return;
    Slot& s = slots_[slot];
    if (s.state != SlotState::prefill) return;
    const std::size_t total = s.req.prompt_ids.size();
    s.prefilled = std::min(total, s.prefilled + rows);
    if (s.prefilled == total) {
        s.state = SlotState::decode;
        s.position = static_cast<std::uint32_t>(total);
        s.feed = s.req.prompt_ids.back();
    }
}

bool BatchScheduler::prime_prefill(std::size_t slot, std::size_t tokens) {
    if (slot >= slots_.size()) return false;
    Slot& s = slots_[slot];
    if (s.state != SlotState::prefill) return false;
    if (tokens == 0 || tokens >= s.req.prompt_ids.size()) return false;
    s.prefilled = tokens;
    return true;
}

bool BatchScheduler::next_decode(DecodeAction* out) {
    if (out == nullptr) return false;
    out->rows.clear();
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        const Slot& s = slots_[i];
        if (s.state != SlotState::decode) continue;
        out->rows.push_back(DecodeAction::Row{i, s.position, s.feed});
    }
    return !out->rows.empty();
}

SlotEnd BatchScheduler::decode_result(std::size_t slot, std::int32_t token, bool stop_hit) {
    if (slot >= slots_.size()) return SlotEnd::error;
    Slot& s = slots_[slot];
    if (s.state != SlotState::decode) {
        // 呼び手の誤り（終了済み・未 prefill）。終了理由は壊さず error を返す。
        return SlotEnd::error;
    }
    if (stop_hit) {
        // EOS は出力に載せない（`engine_contract.md` §2）。
        s.state = SlotState::done;
        s.end = SlotEnd::stop;
        return s.end;
    }
    s.generated.push_back(token);
    if (s.generated.size() >= s.req.max_new_tokens) {
        s.state = SlotState::done;
        s.end = SlotEnd::length;
        return s.end;
    }
    s.feed = token;
    ++s.position;
    return SlotEnd::none;
}

void BatchScheduler::fail_slot(std::size_t slot) {
    if (slot >= slots_.size()) return;
    Slot& s = slots_[slot];
    if (s.state == SlotState::free) return;
    s.state = SlotState::done;
    s.end = SlotEnd::error;
}

bool BatchScheduler::take_finished(Finished* out) {
    if (out == nullptr) return false;
    if (!finished_.empty()) {
        *out = std::move(finished_.front());
        finished_.pop_front();
        return true;
    }
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        Slot& s = slots_[i];
        if (s.state != SlotState::done || s.end == SlotEnd::none) continue;
        out->slot = i;
        out->request_id = s.req.request_id;
        out->prompt_tokens = s.req.prompt_ids.size();
        out->end = s.end;
        out->generated = std::move(s.generated);
        s.state = SlotState::free;
        s.req = BatchRequest{};
        s.prefilled = 0;
        s.generated.clear();
        s.position = 0;
        s.feed = 0;
        s.end = SlotEnd::none;
        admit_queued();
        return true;
    }
    return false;
}

std::size_t BatchScheduler::active() const noexcept {
    std::size_t n = 0;
    for (const auto& s : slots_)
        if (s.state == SlotState::prefill || s.state == SlotState::decode) ++n;
    return n;
}

std::size_t BatchScheduler::free_slots() const noexcept {
    std::size_t n = 0;
    for (const auto& s : slots_)
        if (s.state == SlotState::free) ++n;
    return n;
}

BatchScheduler::SlotInfo BatchScheduler::slot(std::size_t index) const {
    SlotInfo info;
    if (index >= slots_.size()) return info;
    const Slot& s = slots_[index];
    info.index = index;
    info.state = s.state;
    info.request_id = s.req.request_id;
    info.prompt_len = s.req.prompt_ids.size();
    info.prefilled = s.prefilled;
    info.generated = s.generated.size();
    info.position = s.position;
    info.feed = s.feed;
    info.end = s.end;
    return info;
}

const BatchRequest& BatchScheduler::request(std::size_t index) const {
    if (index >= slots_.size() || slots_[index].state == SlotState::free)
        return empty_request_;
    return slots_[index].req;
}

std::size_t BatchScheduler::find(std::string_view request_id) const {
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        if (slots_[i].state == SlotState::free) continue;
        if (slots_[i].req.request_id == request_id) return i;
    }
    return max_slots_;
}

}  // namespace shelf::runtime::generate
