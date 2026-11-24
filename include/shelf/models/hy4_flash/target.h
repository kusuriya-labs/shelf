#pragma once

#include <shelf/models/hy4_flash/decoder.h>

namespace shelf::hy4 {
struct Target {
    Ops op;
    std::vector<int> history;
    std::array<Selection, 5> tapped;
    Mat state{};
    size_t base = 0;
    double head_ms = 0;
    float *target_probs = nullptr;
    float *draft_probs = nullptr;
    float temperature;
    int top_k;
    float top_p;
    uint64_t seed;
    Target(Residency &gpu, float temp, uint64_t rng, int k, float p, int rank = 0);
    std::vector<int> forward(const std::vector<int> &ids, bool verify, bool need_head = true, const Path &dump = {});
    std::vector<int> propose(int anchor);
    void accept(int keep, int submitted);

    float *head_logits(Mat hidden);
    void head_argmax(Mat hidden, int *tokens);

    bool target_graph = false, draft_graph = false;

  private:
    void sample(float *logits, int rows, int position, uint64_t rng, int *ids, float *distribution,
                const Hy4StepParams *params = nullptr);
    std::vector<int> predict(Mat hidden, int position, bool draft = false, const Path &dump = {},
                             float *distribution = nullptr, float *logits = nullptr, int *tokens = nullptr);
    Mat layers(const std::vector<int> &ids, bool verify, bool need_head);
    std::vector<int> draft_steps(int anchor, int *device_ids);
};
}
