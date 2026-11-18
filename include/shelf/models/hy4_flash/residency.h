#pragma once

#include <shelf/core/memory/arena.h>
#include <shelf/models/hy4_flash/checkpoint.h>

namespace shelf::hy4 {
struct Region {
    uint8_t* base=nullptr;
    size_t size=0;
    size_t cursor=0;
    void* next(size_t length);
};

struct Residency {
    const Checkpoint& model;
    size_t tokens;
    std::vector<gpu::GpuArena> arenas;
    std::array<uint8_t*,3> weight_base{};
    std::array<Region,3> workspace{};
    std::array<uint8_t*,42> cache{};
    std::array<uint16_t*,42> index_keys{};
    explicit Residency(const Checkpoint& model, size_t tokens);
    void reserve();
    void upload();
    void shutdown();
    const void* device(const Weight& w) const;
};
}
