#pragma once

#include <shelf/core/status.h>

#include <cstddef>
#include <memory>

namespace shelf::weights {

// Owns page-locked host buffers that device kernels read through the unified
// address space. Used for tables that are read once per token instead of once
// per weight: the embedding table is the only one today, so it stays out of
// VRAM and the kernel reads it over the bus.
//
// Copies share the buffers, so a weights struct that carries an embedding can
// still be copied. Buffers are freed when the last copy is destroyed or when
// release() runs.
class PinnedHostArena {
public:
    PinnedHostArena();
    ~PinnedHostArena() noexcept = default;

    PinnedHostArena(const PinnedHostArena&) noexcept = default;
    PinnedHostArena& operator=(const PinnedHostArena&) noexcept = default;
    PinnedHostArena(PinnedHostArena&&) noexcept = default;
    PinnedHostArena& operator=(PinnedHostArena&&) noexcept = default;

    Status allocate(std::size_t bytes, void** out);

    // Frees every buffer now, for every holder of this arena.
    void release() noexcept;

    std::size_t used() const noexcept;
    bool empty() const noexcept;

private:
    struct State;
    std::shared_ptr<State> state_;
};

}  // namespace shelf::weights
