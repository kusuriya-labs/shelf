#pragma once
#include <shelf/models/hy4_flash/residency.h>
#include <functional>
#include <memory>
#include <string>

namespace shelf::hy4 {
class Ops;
struct Hy4BatchOptions {
    int chunk = 1024;
    float temperature = 0;
    uint64_t seed = 0;
    int top_k = 0;
    float top_p = 1;
    bool ignore_eos = false;
    int draft_tokens = 0;
    bool adaptive = false;
};
struct Hy4DflyStats {
    int draft_tokens = default_draft_tokens;
    size_t draft_rounds = 0;
    size_t accepted = 0;
    size_t k_switches = 0;
};
struct Hy4DflyAdaptiveController {
    int current_k = default_draft_tokens;
    int min_switch_interval = 8;
    float ema_alpha = 0.25f;
    double acc_ema = 0.0;
    bool has_sample = false;
    size_t rounds = 0;
    size_t rounds_since_switch = 0;
    size_t switches = 0;

    explicit Hy4DflyAdaptiveController(int initial_k = default_draft_tokens, int switch_interval = 8,
                                       float alpha = 0.25f);
    void reset(int initial_k);
    static int select_k(int current_k, double acc_ema, int batch_size);
    int observe(double accepted_per_lane, int batch_size);
};
struct Hy4PrefixStats {
    bool enabled = false;
    size_t reuse_slots = 0;
    size_t entries = 0;
    size_t saves = 0;
    size_t hits = 0;
    size_t misses = 0;
    size_t restores = 0;
    size_t evictions = 0;
};
struct Hy4PrefixSlot {
    bool valid = false;
    std::vector<int> tokens;
    size_t max_written = 0;
    uint64_t last_used = 0;
    bool snap_valid = false;
    size_t snap_pos = 0;
    size_t snap_max_written = 0;
    std::vector<std::vector<uint8_t>> swa_snap;
    std::vector<int32_t> pages;
    int live_swa_slot = -1;
    uint64_t live_swa_gen = 0;
};
struct Hy4Request {
    int id;
    std::vector<int> input_ids;
    int count;
    Hy4BatchOptions options;
};
class Hy4BatchSession {
public:
    using Emit = std::function<void(int, int)>;
    using Done = std::function<void(int, const std::vector<int> &, const char *)>;
    Hy4BatchSession(Residency &gpu, Emit emit, Done done, int draft_tokens = 0, bool adaptive = false);
    ~Hy4BatchSession();
    void admit(Hy4Request request);
    void cancel(int id);
    void step();
    void set_draft_tokens(int draft_tokens);
    int draft_tokens() const;
    void set_prefix_cache(bool enabled);
    void set_prefill_decode_interval(int interval);
    int prefill_decode_interval() const;
    Hy4PrefixStats prefix_stats() const;
    size_t last_prompt_tokens() const;
    size_t last_restored_tokens() const;
    bool adaptive() const;
    size_t active() const;
    size_t queued() const;
    size_t draft_rounds() const;
    size_t k_switches() const;
    double acceptance_ema() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
std::vector<std::vector<int>> generate_batch(Residency &gpu, const std::vector<std::vector<int>> &prompts,
                                              const std::vector<int> &counts, const Hy4BatchOptions &options,
                                              const std::function<void(int, int)> &emit = {});
int verify_speculation(Ops &op, const float *target_probs, const float *draft_probs, int position,
                       uint64_t seed, const std::vector<int> &draft, std::vector<int> &actual);
std::vector<int> generate(Residency &gpu, const std::vector<int> &ids, int count, int chunk, float temperature,
              uint64_t seed, const Path &logits_file, int top_k = 0, float top_p = 1,
              const std::function<void(int)> &emit = {}, bool ignore_eos = false,
              Hy4PrefixSlot *prefix_slot = nullptr, Hy4PrefixStats *prefix_stats = nullptr,
              size_t *restored_out = nullptr, int draft_tokens = 0, bool adaptive = false,
              Hy4DflyStats *stats_out = nullptr);
}
