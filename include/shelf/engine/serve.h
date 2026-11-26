#pragma once

#include <shelf/core/status.h>
#include <shelf/engine/compute_session.h>
#include <shelf/engine/speculation.h>
#include <shelf/runtime/dflash2_config.h>
#include <shelf/runtime/scheduling_types.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace shelf::engine {

struct ServeConfig {
    bool dflash_enabled = false;
    bool dflash_adaptive = false;
    runtime::DFlash2Config dflash_config{};
    std::string dflash_model_dir;
    uint32_t dflash_drafts = 7u;
};

struct ServeRequestSpec {
    int64_t request_id = 0;
    std::vector<int32_t> input_ids;
    uint32_t max_new_tokens = 0;
    runtime::SamplingConfig sampling{};
    runtime::ThinkingBudget thinking{};
    bool grammar = false;
    bool grammar_delayed = false;
    int32_t grammar_close_token = -1;
    bool has_token_bitmask = false;
    bool ignore_eos = false;
    std::vector<uint32_t> token_bitmask;
};

struct ServePrefillHooks {
    std::function<void(uint32_t processed_tokens, double elapsed_ms)> report;
    std::function<bool()> should_stop;
};

struct ServeEvent {
    enum class Kind { Token, Done, Draft, Error };
    Kind kind = Kind::Token;
    int64_t request_id = 0;
    int32_t token_id = 0;
    bool need_mask = true;
    std::string code;
    std::string message;
    std::string finish_reason;
    std::vector<int32_t> generated_ids;
    uint64_t prompt_tokens = 0;
    uint64_t restored_tokens = 0;
    uint64_t prefill_tokens = 0;
    uint64_t cache_checkpoint_tokens = 0;
    runtime::DFlash2RequestStats dflash_stats{};
    std::string error_message;
    uint64_t round_id = 0;
    std::vector<int32_t> draft_tokens;
};

struct ServeAdmitResult {
    bool accepted = false;
    bool cancelled = false;
    std::string code;
    std::string message;
};

struct ServeStats {
    bool prefix_enabled = false;
    uint64_t reuse_slots = 0;
    uint64_t entries = 0;
    uint64_t saves = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t restores = 0;
    uint64_t evictions = 0;
    uint64_t page_tokens = 0;
    uint64_t total_pages = 0;
    uint64_t used_pages = 0;
    uint64_t free_pages = 0;
    uint64_t active_requests = 0;
    uint64_t queued_requests = 0;
    bool banker_safe = false;
    uint64_t overcommit_pages = 0;
};

class ServeController {
 public:
    static Result<std::unique_ptr<ServeController>> create(ComputeSession& session,
                                                           const ServeConfig& config);

    ServeController(const ServeController&) = delete;
    ServeController& operator=(const ServeController&) = delete;
    ~ServeController();

    bool mtp_enabled() const noexcept;
    bool dflash_enabled() const noexcept;
    bool has_inflight() const noexcept;
    bool has_mtp_active() const noexcept;
    bool has_request(int64_t request_id) const noexcept;
    bool busy() const noexcept;
    bool all_waiting_for_mask() const noexcept;
    bool is_recent_terminal(int64_t request_id) const noexcept;
    bool exit_requested() const noexcept;
    uint32_t max_inflight() const noexcept;
    uint32_t vocab_size() const noexcept;

    Result<ServeAdmitResult> admit(const ServeRequestSpec& request,
                                   const ServePrefillHooks& hooks,
                                   std::vector<ServeEvent>& events);

    void step(std::vector<ServeEvent>& events);
    bool cancel(int64_t external_request_id, std::vector<ServeEvent>& events);
    void cancel_all_inflight(std::vector<ServeEvent>& events);
    void abandon_inflight(int64_t external_request_id, std::vector<ServeEvent>& events);
    void request_shutdown(std::vector<ServeEvent>& events);

    // Mask handling. Host words are uploaded here; the application never holds a device
    // pointer.
    Status apply_token_mask(int64_t external_request_id,
                            const std::vector<uint32_t>& words);
    Status draft_mask_request(int64_t external_request_id, uint64_t round,
                              bool& pending, std::size_t& draft_count,
                              std::size_t& words_per_row) const;
    Status apply_draft_masks(int64_t external_request_id, uint64_t round,
                             const uint32_t* masks, std::size_t words_per_row,
                             uint32_t valid_prefix);

    ServeStats stats() const;
    void write_final_stats();

 private:
    ServeController() = default;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace shelf::engine
