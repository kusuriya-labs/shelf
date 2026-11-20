#pragma once

#include <shelf/core/memory/arena.h>
#include <shelf/kernels/hy4_flash/linear.h>
#include <shelf/models/hy4_flash/checkpoint.h>
#ifdef SHELF_HY4_RDMA
#include <shelf/models/hy4_flash/rdma/client.h>
#include <future>
#endif

namespace shelf::hy4 {
inline constexpr int max_prefill_chunk = 1024;

struct Region {
    uint8_t *base = nullptr;
    size_t size = 0;
    size_t cursor = 0;
    void *next(size_t length);
};

struct Residency {
#ifdef SHELF_HY4_RDMA
    std::unique_ptr<rdma::Client> engram_client;
    std::future<rdma::Tables> engram_pending;
    rdma::Tables engram_rows;
    std::vector<uint8_t> snapshot_latest;
    std::array<uint8_t, 32> snapshot_upload_digest{};
    bool snapshot_upload_digest_ready = false;
    std::array<uint8_t, 32> snapshot_model_digest{};
    bool snapshot_model_digest_ready = false;
    std::array<uint8_t, 32> snapshot_engram_digest{};
    bool snapshot_engram_digest_ready = false;
#endif
    Checkpoint &model;
    size_t tokens;
    std::array<int, 3> physical{0, 1, 2};
    std::vector<gpu::GpuArena> arenas;
    std::array<uint8_t *, 3> weight_base{};
    std::array<Region, 3> workspace{};
    std::array<uint64_t, 3> routed_rows{};
    std::array<std::array<uint64_t, 3>, 3> transfer_bytes{};
    std::map<std::string, void *> folded;
    std::array<uint8_t *, 42> cache{};
    std::array<uint16_t *, 42> index_keys{};
    explicit Residency(Checkpoint &model, size_t tokens);
    ~Residency() noexcept;
    Residency(const Residency &) = delete;
    Residency &operator=(const Residency &) = delete;
    std::array<hipEvent_t, 3> expert_begin{}, expert_end{};
    std::array<double, 3> expert_ms{};
    double attention_ms = 0, ffn_ms = 0, transfer_ms = 0, engram_lookup_ms = 0;
    long engram_major_faults = 0;
    void reserve();
    void upload();
    void upload_experts(int rank, int layer);
    std::array<Hy4ExpertProjection *, 3> matrices{};
    static constexpr size_t matrix_bytes = 41 * 512 * 3 * sizeof(Hy4ExpertProjection);
    void transfer(void *dst, int destination, const void *src, int source, size_t bytes);
    std::array<std::array<bool, 3>, 3> peer{}, owns_peer{};
    void *staging = nullptr;
    std::array<std::array<hipEvent_t, 3>, 3> ready{};
    std::array<void *, 3> route_host{};
    void *route_buffer(int rank, size_t bytes);
    void shutdown();
    const void *device(const Weight &w) const;
};
}
