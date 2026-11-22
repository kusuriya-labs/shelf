#pragma once

#include <shelf/core/memory/arena.h>
#include <shelf/models/hy4_flash/linear.h>
#include <shelf/models/hy4_flash/checkpoint.h>
#include <shelf/models/hy4_flash/sync.h>
#include <memory>
#include <string_view>

namespace shelf::hy4 {
inline constexpr int max_prefill_chunk = 1024;

inline bool multi_gpu_vmm_allowed(int ranks, std::string_view vmm) {
    return ranks <= 1 || vmm != "1";
}

struct Hy4StepParams {
    int32_t rows;
    int32_t position;
    uint64_t seed;
    std::array<uint64_t, 4> engram;
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
    explicit Residency(Checkpoint &model, size_t tokens);
    ~Residency() noexcept;
    Residency(const Residency &) = delete;
    Residency &operator=(const Residency &) = delete;
    std::array<hipEvent_t, 4> expert_begin{}, expert_end{};
    std::array<double, 4> expert_ms{};
    double attention_ms = 0, ffn_ms = 0, transfer_ms = 0;
    Status reserve();
    void upload();
    void upload_experts(int rank, int layer);
    std::array<Hy4ExpertProjection *, 4> matrices{};
    static constexpr size_t matrix_bytes = 41 * 512 * 3 * sizeof(Hy4ExpertProjection);
    void transfer(void *dst, int destination, const void *src, int source, size_t bytes);
    std::array<std::array<bool, 4>, 4> peer{}, owns_peer{};
    bool push = false;
    void *staging = nullptr;
    std::array<std::array<std::array<hipEvent_t, 2>, 4>, 4> ready{}, ack{};
    std::array<std::array<int, 4>, 4> parity{};
    std::array<void *, 4> route_host{};
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
    void *route_buffer(int rank, size_t bytes);
    void shutdown();
    const void *device(const Weight &w) const;
};
void peer_copy(void *dst, const void *src, size_t bytes, hipStream_t stream = nullptr);
void tp_add_bf16(void *dst, const void *src, size_t bytes, hipStream_t stream = nullptr);
}
