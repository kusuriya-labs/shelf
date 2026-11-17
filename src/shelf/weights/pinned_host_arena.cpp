#include <shelf/weights/pinned_host_arena.h>

#include <shelf/core/gpu/cleanup.h>

#include <hip/hip_runtime.h>

#include <vector>

namespace shelf::weights {

struct PinnedHostArena::State {
    std::vector<void*> blocks;
    std::size_t used = 0u;

    ~State() noexcept {
        for (void* block : blocks) {
            if (block != nullptr) shelf::gpu::discard_cleanup_result(hipHostFree(block));
        }
    }
};

PinnedHostArena::PinnedHostArena() : state_(std::make_shared<State>()) {}

Status PinnedHostArena::allocate(std::size_t bytes, void** out) {
    if (out == nullptr) {
        return Status::invalid_argument("pinned host allocate needs an out pointer", __FILE__,
                                        __LINE__);
    }
    *out = nullptr;
    if (bytes == 0u) {
        return Status::invalid_argument("pinned host allocate needs bytes", __FILE__,
                                        __LINE__);
    }
    void* ptr = nullptr;
    const hipError_t err = hipHostMalloc(&ptr, bytes, hipHostMallocMapped);
    if (err != hipSuccess) {
        return Status::hip_error("hipHostMalloc", hipGetErrorString(err), __FILE__, __LINE__);
    }
    state_->blocks.push_back(ptr);
    state_->used += bytes;
    *out = ptr;
    return Status::make_ok();
}

void PinnedHostArena::release() noexcept {
    for (void*& block : state_->blocks) {
        if (block != nullptr) {
            shelf::gpu::discard_cleanup_result(hipHostFree(block));
            block = nullptr;
        }
    }
    state_->blocks.clear();
    state_->used = 0u;
}

std::size_t PinnedHostArena::used() const noexcept { return state_->used; }

bool PinnedHostArena::empty() const noexcept { return state_->blocks.empty(); }

}  // namespace shelf::weights
