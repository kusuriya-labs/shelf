#pragma once

#include <shelf/core/memory/arena.h>
#include <shelf/kernels/hy4_flash/linear.h>
#include <shelf/models/hy4_flash/checkpoint.h>

namespace shelf::hy4 {
struct Region {
    uint8_t *base = nullptr;
    size_t size = 0;
    size_t cursor = 0;
    void *next(size_t length);
};

struct Residency {
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
    void reserve();
    void upload();
    void upload_experts(int rank, int layer);
    std::array<Hy4ExpertProjection *, 3> matrices{};
    static constexpr size_t matrix_bytes = 41 * 512 * 3 * sizeof(Hy4ExpertProjection);
    void transfer(void *dst, int destination, const void *src, int source, size_t bytes);
    std::array<std::array<bool, 3>, 3> peer{}, owns_peer{};
    void *staging = nullptr;
    void shutdown();
    const void *device(const Weight &w) const;
};
}
