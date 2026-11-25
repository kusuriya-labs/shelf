#pragma once

#include <hip/hip_runtime_api.h>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace shelf::hy4 {
struct Residency;

inline constexpr size_t graph_headroom = 256ull << 20;

inline bool graph_topology(int ep, int tp, bool push, bool room, const std::array<std::array<bool, 4>, 4> &peer) {
    const int ranks = ep * tp;
    if (ep < 1 || tp < 1 || ranks < 1 || ranks > 4 || !push || !room)
        return false;
    if (tp != 1 && tp != 2 && !(tp == 4 && ep == 1))
        return false;
    for (int src = 0; src < ranks; ++src)
        for (int dst = 0; dst < ranks; ++dst)
            if (src != dst && !peer[src][dst])
                return false;
    return true;
}

struct BatchGraphLane {
    int sparse_slot = -1, dense_rows = 0;
    float temperature = 0, top_p = 0;
    int top_k = 0;
    bool operator==(const BatchGraphLane &) const = default;
};

struct GraphKey {
    int rows, dense_rows, bucket;
    bool verify;
    size_t workspace;
    float temperature, top_p;
    int top_k;
    int batch = 0;
    int draft_tokens = 0;
    const uint16_t *state = nullptr;
    std::array<BatchGraphLane, 32> lanes{};
    bool operator==(const GraphKey &) const = default;
};

struct GraphLayer {
    hipGraphExec_t scatter{}, local{};
    std::array<hipGraphExec_t, 4> expert{};
    std::array<hipEvent_t, 2> scatter_done{}, local_done{};
    std::array<std::array<hipEvent_t, 4>, 2> expert_done{};
};

struct GraphHeadStep {
    hipGraphExec_t scatter{};
    std::array<hipGraphExec_t, 4> rank_exec{};
    std::array<hipEvent_t, 2> scatter_done{};
    std::array<std::array<hipEvent_t, 4>, 2> rank_done{};
};

struct RankEvent {
    hipEvent_t handle{};
    int owner = -1;
    operator hipEvent_t() const { return handle; }
};

struct TpGraphStep {
    std::array<hipGraphExec_t, 4> exec{};
    std::array<std::array<RankEvent, 4>, 2> done{};
    uint8_t group_mode = 0;
};

struct GraphPlan {
    GraphKey key{};
    std::array<GraphLayer, 41> layers{};
    std::vector<GraphHeadStep> head_steps{};
    hipGraphExec_t entry{}, tail{};
    float *logits{};
    int *tokens{};
    uint16_t *hidden{};
    size_t cursor{};
    std::vector<TpGraphStep> tp_steps{};
    std::array<std::vector<hipGraphExec_t>, 4> tp_captured{};
    std::array<std::vector<uint8_t>, 4> tp_modes{};
    std::array<std::array<RankEvent, 4>, 2> tp_rank_done{};
    std::array<float *, 4> tp_logits{};
    std::array<int *, 4> tp_tokens{};
    std::array<uint16_t *, 4> tp_hidden{};
    std::array<size_t, 4> tp_cursor{};
    int tp_ranks = 0;
    std::array<RankEvent, 2> replay_done{};
    std::array<bool, 2> replay_recorded{};
    unsigned replay_slot{};
    bool events_ready{};
    void finalize_tp();
    void launch(Residency &gpu);
    void release(Residency &gpu, std::string &errors);
};

class GraphCapture {
  public:
    Residency &gpu;
    GraphPlan &plan;
    int rank = 0;
    GraphCapture(Residency &gpu, GraphPlan &plan, int rank = 0) : gpu(gpu), plan(plan), rank(rank) {}
    void begin(int r);
    void finish(hipGraphExec_t &exec);
    void step(uint8_t sync_mode);
    void cancel(std::string &errors);

  private:
    int active = -1;
};
}
