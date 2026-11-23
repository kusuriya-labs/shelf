#pragma once

#include <shelf/core/memory/arena.h>
#include <shelf/models/hy4_flash/moe_graph_plan.h>
#include <shelf/models/hy4_flash/checkpoint.h>
#include <shelf/models/hy4_flash/sync.h>
#include <shelf/models/hy4_flash/graph.h>
#include <algorithm>
#include <memory>
#include <string_view>

namespace shelf::hy4 {
inline constexpr int max_prefill_chunk = 4096;
inline constexpr int tp_prefill_rows = 1024;

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
    int ids[6];
};

struct GraphInput {
    Hy4StepParams step;
    uint8_t engram[4][6 * 16 * 98];
};

struct BatchGraphParams {
    Hy4StepParams lanes[16];
    int ids[96], anchors[16];
};

struct BatchGraphInput {
    BatchGraphParams step;
    int candidates[80];
    uint8_t engram[96 * 4 * 16 * 98];
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
    bool batch_draft;
    bool index_wmma = true;
    std::vector<std::array<int, 42>> sparse_last;
    size_t prefill_rows() const { return std::min(tokens, size_t(tp_size > 1 ? tp_prefill_rows : max_prefill_chunk)); }
    int *sparse_top_slot(int layer, int slot) const {
        return sparse_top[0][layer] + (prefill_rows() + slot * 6) * 511;
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
    explicit Residency(Checkpoint &model, size_t tokens, int slots = 1);
    ~Residency() noexcept;
    Residency(const Residency &) = delete;
    Residency &operator=(const Residency &) = delete;
    std::array<hipEvent_t, 4> expert_begin{}, expert_end{};
    std::array<double, 4> expert_ms{};
    double attention_ms = 0, ffn_ms = 0, transfer_ms = 0;
    static Status check_batch_topology(int tp_size, int slots);
    Status reserve();
    void upload();
    void upload_experts(int rank, int layer);
    std::array<Hy4ExpertProjection *, 4> matrices{};
    std::array<int *, 4> moe_owners{};
    std::array<uint64_t *, 4> moe_counts{};
    std::array<hipStream_t, 4> local_expert{};
    std::array<hipEvent_t, 4> local_ready{}, local_done{};
    std::array<std::array<void *, 4>, 4> small_staging{};
    void reserve_experts(int rank, Region &buffer);
    void collect_route_counts(int rank);
    static constexpr size_t matrix_bytes = 41 * 512 * 3 * sizeof(Hy4ExpertProjection);
    size_t transfer_packet_bytes() const {
        size_t rows = std::min(tokens, size_t(expert_tile_rows));
        size_t prefill = rows * (1536 * sizeof(uint16_t) + sizeof(PrefillRoutes));
        size_t decode = max_slots == 1 ? Hy4MoePlan::packet_bytes(Hy4MoePlan::max_rows)
                                     : Hy4GroupedPlan::packet_bytes(max_slots * (batch_draft ? 6 : 1));
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
    GraphInput *graph_input = nullptr;
    BatchGraphInput *batch_input = nullptr;
    uint64_t target_replays = 0, draft_replays = 0;
    std::array<std::array<std::array<hipEvent_t, 2>, 4>, 4> ready{}, ack{};
    std::array<std::array<int, 4>, 4> parity{};
    std::array<void *, 4> tp_recv{};
    std::array<void *, 4> ep_recv{};
    std::array<void *, 4> tp_ptr{};
    std::array<size_t, 4> tp_bytes{};
    std::array<std::array<hipEvent_t, 2>, 4> tp_ready{}, tp_done{}, tp_arrived{}, tp_consumed{};
    std::array<std::array<hipEvent_t, 2>, 4> ep_ready{}, ep_done{}, ep_arrived{}, ep_consumed{};
    std::array<std::array<bool, 2>, 4> tp_recorded{}, ep_recorded{};
    std::array<int, 4> tp_parity{}, ep_parity{};
    std::array<void *, 4> reduce_staging{};
    std::array<std::unique_ptr<SyncBarrier>, 4> tp_barrier{}, ep_barrier{};
    std::unique_ptr<SyncBarrier> all_barrier;
    void abort_collectives();
    void reset_collectives();
    void reduce_pair(void *data, int rank, size_t bytes, bool ep);
    void tp_sum(void *data, int rank, size_t bytes);
    void ep_reduce(void *data, int rank, size_t bytes);
    void shutdown();
    const void *device(const Weight &w) const;
};
void peer_copy(void *dst, const void *src, size_t bytes, hipStream_t stream = nullptr);
void tp_add_bf16(void *dst, const void *src, size_t bytes, hipStream_t stream = nullptr);
}
