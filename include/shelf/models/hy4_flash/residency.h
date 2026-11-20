#pragma once

#include <shelf/core/memory/arena.h>
#include <shelf/models/hy4_flash/linear.h>
#include <shelf/models/hy4_flash/checkpoint.h>

namespace shelf::hy4 {
inline constexpr int max_prefill_chunk = 1024;

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
    std::array<int, 4> physical{0, 1, 2, 3};
    std::vector<gpu::GpuArena> arenas;
    std::array<uint8_t *, 4> weight_base{};
    std::array<Region, 4> workspace{};
    std::array<uint64_t, 4> routed_rows{};
    std::array<std::array<uint64_t, 4>, 4> transfer_bytes{};
    std::map<std::string, void *> folded;
    std::array<uint8_t *, 42> cache{};
    std::array<uint16_t *, 42> index_keys{};
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
    std::array<std::array<hipEvent_t, 4>, 4> ready{}, ack{};
    std::array<void *, 4> route_host{};
    void *route_buffer(int rank, size_t bytes);
    void shutdown();
    const void *device(const Weight &w) const;
};
void peer_copy(void *dst, const void *src, size_t bytes);
}
