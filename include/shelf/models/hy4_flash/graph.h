#pragma once

#include <hip/hip_runtime_api.h>
#include <array>
#include <string>

namespace shelf::hy4 {
struct Residency;

inline constexpr size_t graph_headroom = 256ull << 20;

inline bool graph_topology(int ep, int tp, bool push, bool room, const std::array<std::array<bool, 4>, 4> &peer) {
    if (ep < 1 || ep > 4 || tp != 1 || !push || !room)
        return false;
    for (int src = 0; src < ep; ++src)
        for (int dst = 0; dst < ep; ++dst)
            if (src != dst && !peer[src][dst])
                return false;
    return true;
}

struct GraphKey {
    int rows, dense_rows, bucket;
    bool verify;
    size_t workspace;
    float temperature, top_p;
    int top_k;
    bool operator==(const GraphKey &) const = default;
};

struct GraphLayer {
    hipGraphExec_t scatter{};
    std::array<hipGraphExec_t, 4> expert{};
};

struct GraphPlan {
    GraphKey key{};
    std::array<GraphLayer, 41> layers{};
    hipGraphExec_t entry{}, tail{};
    float *logits{};
    size_t cursor{};
    void launch(Residency &gpu) const;
    void release(Residency &gpu, std::string &errors);
};

class GraphCapture {
  public:
    Residency &gpu;
    GraphPlan &plan;
    GraphCapture(Residency &gpu, GraphPlan &plan) : gpu(gpu), plan(plan) {}
    void begin(int rank);
    void finish(hipGraphExec_t &exec);
    void cancel(std::string &errors);

  private:
    int active = -1;
};
}
