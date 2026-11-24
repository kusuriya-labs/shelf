#pragma once

#include <shelf/core/memory/arena.h>
#include <shelf/models/hy4_flash/moe_graph_plan.h>
#include <shelf/models/hy4_flash/checkpoint.h>
#include <shelf/models/hy4_flash/sync.h>
#include <shelf/models/hy4_flash/graph.h>
#include <algorithm>
#include <atomic>
#include <memory>
#include <string_view>

namespace shelf::hy4 {
inline constexpr int max_prefill_chunk = 4096;
inline constexpr int tp_prefill_rows = max_prefill_chunk;
inline constexpr int min_draft_tokens = 2;
inline constexpr int max_draft_tokens = 5;
inline constexpr int max_verify_rows = max_draft_tokens + 1;
inline constexpr int default_draft_tokens = 4;
inline constexpr size_t kv_page_tokens = 256;
inline constexpr size_t kv_page_groups = kv_page_tokens / 4;
inline constexpr int hy4_vocab_size = 120832;
inline constexpr int hy4_lm_head_tile = 128;
inline constexpr int lm_hidden_max_rows = 512;
inline constexpr size_t lm_hidden_row_bytes = 3072 * 2;
inline constexpr size_t lm_hidden_chunk_bytes = size_t(lm_hidden_max_rows) * lm_hidden_row_bytes;
inline constexpr int lm_argmax_max_rows = 512;
inline constexpr size_t lm_argmax_stride = size_t(lm_argmax_max_rows) * 8;
inline constexpr size_t lm_logits_chunk_bytes = size_t(2) * 1024 * 1024;
inline bool lm_head_tp_supported(int ranks) {
    return ranks > 1 && hy4_vocab_size % (ranks * hy4_lm_head_tile) == 0;
}

inline bool multi_gpu_vmm_allowed(int ranks, std::string_view vmm) {
    return ranks <= 1 || vmm != "1";
}

struct alignas(16) PrefillRoutes {
    struct Choice { int expert; float weight; };
    Choice choices[16];
    unsigned masks[4];
};

void peer_prefill_packet(void *dst, const void *src, int rows, int group, hipStream_t stream);

struct Hy4StepParams {
    int32_t rows;
    int32_t position;
    uint64_t seed;
    int draft_last = -1;
    int ids[max_verify_rows];
};

struct GraphInput {
    Hy4StepParams step;
    uint8_t engram[4][max_verify_rows * 16 * 98];
};

struct BatchGraphParams {
    Hy4StepParams lanes[32];
    int ids[32 * max_verify_rows], anchors[32];
};

struct BatchGraphInput {
    BatchGraphParams step;
    int candidates[32 * max_draft_tokens];
    uint8_t engram[32 * max_verify_rows * 4 * 16 * 98];
};

struct Region {
    uint8_t *base = nullptr;
    size_t size = 0;
    size_t cursor = 0;
    void *next(size_t length);
};

struct Residency {
    Checkpoint &model;
    size_t tokens;
    int max_slots;
    int draft_tokens;
    size_t kv_pool_tokens = 0;
    size_t kv_pages_total = 0;
    size_t kv_page_stride = 0;
    bool batch_draft;
    bool index_wmma = true;
    std::array<std::vector<std::array<int, 42>>, 4> sparse_last;
    size_t prefill_rows() const { return std::min(tokens, size_t(tp_size > 1 ? tp_prefill_rows : max_prefill_chunk)); }
    int moe_group(int rank) const { return tp_size == 4 ? rank : rank / tp_size; }
    int expert_intermediate() const { return tp_size == 2 ? 768 : 1536; }
    int *sparse_top_slot(int layer, int slot, int rank = 0) const {
        return sparse_top[rank][layer] + (prefill_rows() + slot * max_verify_rows) * 511;
    }
    int ranks = 0;
    int ep_size = 1;
    int tp_size = 1;
    std::array<int, 4> physical{0, 1, 2, 3};
    std::vector<gpu::GpuArena> arenas;
    std::array<uint8_t *, 4> weight_base{};
    std::array<Region, 4> workspace{};
    std::array<hipStream_t, 4> compute{};
    std::array<std::array<int *, 42>, 4> sparse_top{};
    std::array<uint64_t, 4> routed_rows{};
    std::array<std::array<uint64_t, 4>, 4> transfer_bytes{};
    std::array<std::map<std::string, void *>, 4> folded;
    std::array<std::array<uint8_t *, 42>, 4> cache{};
    std::array<std::array<uint16_t *, 42>, 4> index_keys{};
    std::array<std::array<uint16_t *, 42>, 4> index_pending{};
    std::array<int32_t *, 4> page_table{};
    std::vector<int32_t> kv_free_pages;
    std::vector<uint32_t> kv_page_refs;
    std::vector<std::vector<int32_t>> slot_page_tables;
    static bool is_sparse_layer(int layer) { return layer == 0 || layer % 4 == 1; }
    size_t pages_for_tokens(size_t n) const { return n ? (n + kv_page_tokens - 1) / kv_page_tokens : 0; }
    size_t free_kv_pages() const { return kv_free_pages.size(); }
    size_t used_kv_pages() const { return kv_pages_total - kv_free_pages.size(); }
    void sync_slot_page_table(int slot);
    bool allocate_slot_pages(int slot, size_t needed_tokens);
    void release_slot_pages(int slot);
    void retain_pages(const std::vector<int32_t> &pages);
    void release_pages(std::vector<int32_t> &pages);
    int32_t alloc_single_page();
    explicit Residency(Checkpoint &model, size_t tokens, int slots = 1, size_t kv_pool_tokens = 0,
                       int draft_tokens = 0);
    static int parse_draft_tokens(std::string_view value);
    static int resolve_draft_tokens(int requested = 0);
    ~Residency() noexcept;
    Residency(const Residency &) = delete;
    Residency &operator=(const Residency &) = delete;
    std::array<hipEvent_t, 4> expert_begin{}, expert_end{};
    std::array<double, 4> expert_ms{};
    double attention_ms = 0, ffn_ms = 0, transfer_ms = 0;
    static Status check_batch_topology(int tp_size, int slots);
    Status reserve();
    using PeerProbe = bool (*)(void *dst, int dst_rank, const void *src, int src_rank, size_t bytes,
                               hipStream_t stream, bool push);
    bool configure_transports(PeerProbe probe = nullptr);
    void upload();
    void upload_experts(int rank, int layer);
    std::array<Hy4ExpertProjection *, 4> matrices{};
    std::array<int *, 4> moe_owners{};
    std::array<float *, 4> gate_expert_norms{};
    std::array<uint64_t *, 4> moe_counts{};
    std::array<hipStream_t, 4> local_expert{};
    std::array<RankEvent, 4> local_ready{}, local_done{};
    std::array<std::array<void *, 4>, 4> small_staging{};
    void reserve_experts(int rank, Region &buffer);
    void create_event(RankEvent &event, int rank, unsigned flags = hipEventDisableTiming | hipEventReleaseToSystem);
    void record_event(const RankEvent &event, int rank) const;
    void record_event(const RankEvent &event, int rank, hipStream_t stream) const;
    void sync_record(const RankEvent &event, bool &recorded, int rank) const;
    void collect_route_counts(int rank);
    static constexpr size_t matrix_bytes = 41 * 512 * 3 * sizeof(Hy4ExpertProjection);
    size_t transfer_packet_bytes() const {
        size_t rows = std::min(tokens, size_t(expert_tile_rows));
        size_t prefill = rows * (1536 * sizeof(uint16_t) + sizeof(PrefillRoutes));
        size_t decode = max_slots == 1 ? Hy4MoePlan::packet_bytes(Hy4MoePlan::max_rows)
                                     : Hy4GroupedPlan::packet_bytes(max_slots * (batch_draft ? max_verify_rows : 1));
        return std::max(prefill, decode);
    }
    void transfer(void *dst, int destination, const void *src, int source, size_t bytes, int prefill_rows = 0);
    std::array<std::array<bool, 4>, 4> peer{}, owns_peer{};
    bool push = false;
    bool graph_room = false;
    bool target_capture_failed = false, draft_capture_failed = false;
    std::array<GraphPlan, 4> target_graphs{};
    std::array<GraphPlan, 2> draft_graphs{};
    size_t target_victim = 0, draft_victim = 0;
    std::array<GraphInput *, 4> graph_input{};
    std::array<BatchGraphInput *, 4> batch_input{};
    uint64_t target_replays = 0, draft_replays = 0;
    std::array<std::array<std::array<RankEvent, 2>, 4>, 4> ready{}, ack{};
    std::array<std::array<int, 4>, 4> parity{};
    std::array<void *, 4> tp_recv{};
    std::array<void *, 4> ep_recv{};
    std::array<void *, 4> tp_ptr{}, ep_ptr{};
    std::array<const void *, 4> tp_src_ptr{}, ep_src_ptr{};
    std::array<size_t, 4> tp_bytes{}, ep_bytes{};
    bool lm_head_tp = false;
    bool lm_host_fallback = false;
    int lm_head_ranks() const { return lm_head_tp ? ranks : 1; }
    static size_t lm_staging_bytes() { return std::max(lm_hidden_chunk_bytes, lm_logits_chunk_bytes); }
    std::array<void *, 4> lm_hidden_buf{};
    std::array<void *, 4> lm_head_recv{};
    std::array<void *, 4> lm_logits_recv{};
    std::array<void *, 4> lm_staging{};
    std::array<const void *, 4> lm_src_ptr{};
    std::array<void *, 4> lm_dst_ptr{};
    std::array<size_t, 4> lm_xfer_bytes{};
    std::array<std::array<RankEvent, 2>, 4> lm_ready{}, lm_done{}, lm_arrived{}, lm_consumed{};
    std::array<std::array<bool, 2>, 4> lm_ready_rec{}, lm_done_rec{}, lm_arrived_rec{}, lm_consumed_rec{};
    int lm_parity = 0;
    std::array<std::array<RankEvent, 2>, 4> tp_ready{}, tp_done{}, tp_arrived{}, tp_consumed{}, tp_ag_done{};
    std::array<std::array<RankEvent, 2>, 4> ep_ready{}, ep_done{}, ep_arrived{}, ep_consumed{};
    std::array<std::array<bool, 2>, 4> tp_recorded{}, ep_recorded{};
    std::array<int, 4> tp_parity{}, ep_parity{};
    std::array<void *, 4> reduce_staging{};
    std::array<std::unique_ptr<SyncBarrier>, 4> tp_barrier{}, ep_barrier{};
    std::unique_ptr<SyncBarrier> all_barrier, capture_barrier;
    std::atomic<bool> capture_failed{false};
    void abort_collectives();
    void reset_collectives();
    void reduce_pair(void *data, int rank, size_t bytes, bool ep, GraphCapture *capture = nullptr,
                     const float *f32_src = nullptr);
    void reduce_tp4(void *data, int rank, size_t bytes, GraphCapture *capture = nullptr,
                    const float *f32_src = nullptr);
    void tp_sum(void *data, int rank, size_t bytes, GraphCapture *capture = nullptr);
    void tp_sum_f32(void *dst_bf16, const float *src_f32, int rank, size_t bf16_bytes,
                    GraphCapture *capture = nullptr);
    void ep_reduce(void *data, int rank, size_t bytes, GraphCapture *capture = nullptr);
    void lm_scatter_hidden(const void *src, size_t bytes, std::array<void *, 4> &dst_out, bool capture = false);
    void lm_gather_argmax(const std::array<const float *, 4> &vals, const std::array<const int *, 4> &ids, int rows,
                          int *out_ids);
    void lm_gather_logits(const std::array<const float *, 4> &local_logits, int rows, int local_vocab, int full_vocab,
                          float *out_logits);
    void tp_argmax(const float *vals, const int *ids, int *out_ids, int rank, int rows, GraphCapture *capture = nullptr);
    void tp_gather_logits(const float *local_logits, float *out_logits, int rank, int rows, int local_vocab,
                          int full_vocab, GraphCapture *capture = nullptr);
    void broadcast_lm_ids(size_t bytes, int parity);
    void tp_broadcast_ids(int *ids, int rank, int rows, GraphCapture *capture = nullptr);
    void shutdown();
    const void *device(const Weight &w) const;
};
void peer_copy(void *dst, const void *src, size_t bytes, hipStream_t stream = nullptr);
void tp_add_bf16(void *dst, const void *src, size_t bytes, hipStream_t stream = nullptr);
void tp_sum_bf16_4(void *dst, const void *s0, const void *s1, const void *s2, const void *s3, size_t bytes,
                   hipStream_t stream = nullptr);
void tp_add_f32_to_bf16(void *dst, const float *s0, const float *s1, size_t count, hipStream_t stream = nullptr);
void tp_sum_f32_to_bf16_4(void *dst, const float *s0, const float *s1, const float *s2, const float *s3,
                          size_t count, hipStream_t stream = nullptr);
void tp_argmax_pack(const float *vals, const int *ids, void *out_pairs, int rows, hipStream_t stream = nullptr);
void tp_argmax_reduce(const void *src_pairs, int stride_pairs, int tp_size, int rows, int *out_ids,
                      hipStream_t stream = nullptr);
void tp_scatter_logits(const float *src, float *dst, int rows, int local_vocab, int full_vocab, int offset,
                       hipStream_t stream = nullptr);
}
