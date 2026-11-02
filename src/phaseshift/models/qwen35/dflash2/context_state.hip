#include <phaseshift/models/qwen35/dflash2/context_state.h>

#include <limits>
#include <vector>

namespace ps::qwen35::dflash2 {

namespace {

Status allocate_ring_tensor(
    gpu::GpuArena& arena,
    const std::vector<std::size_t>& shape,
    gpu::Tensor& out) {
    std::size_t count = 1;
    for (std::size_t d : shape) {
        if (d == 0) {
            return Status::invalid_argument("ring shape has a zero dim", __FILE__, __LINE__);
        }
        if (std::numeric_limits<std::size_t>::max() / count < d) {
            return Status::invalid_argument("ring element count overflow", __FILE__, __LINE__);
        }
        count *= d;
    }
    auto view = arena.allocate_aligned(count * sizeof(bf16_t), 128);
    if (!view.ok()) return view.status();
    auto backing = view.release();
    std::vector<std::size_t> strides(shape.size());
    std::size_t stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }
    auto result = gpu::Tensor::create<bf16_t>(backing, shape, strides);
    if (!result.ok()) return result.status();
    out = result.release();
    return Status::make_ok();
}

}  // namespace

Result<DFlash2ContextState> create_dflash2_context_state(
    const DFlash2Config& config,
    gpu::GpuArena& arena,
    uint32_t initial_position) {
    if (config.num_hidden_layers == 0 || config.sliding_window == 0 ||
        config.attention_kv_rows() == 0) {
        return Status::invalid_argument("dflash2 context geometry is zero", __FILE__, __LINE__);
    }

    DFlash2ContextState state;
    state.num_layers = static_cast<uint32_t>(config.num_hidden_layers);
    state.capacity = static_cast<uint32_t>(config.sliding_window);
    state.kv_features = static_cast<uint32_t>(config.attention_kv_rows());
    state.next_position = initial_position;
    state.length = 0;

    const std::vector<std::size_t> shape = {state.num_layers, state.capacity,
                                            state.kv_features};
    Status st = allocate_ring_tensor(arena, shape, state.k_ring);
    if (!st.ok()) return st;
    st = allocate_ring_tensor(arena, shape, state.v_ring);
    if (!st.ok()) {
        state.k_ring = gpu::Tensor{};
        return st;
    }
    state.initialized = true;
    return state;
}

Status dflash2_context_reset(
    DFlash2ContextState& state,
    uint32_t next_position) {
    if (!state.initialized) {
        return Status::invalid_state("dflash2 context state not initialized", __FILE__,
                                     __LINE__);
    }
    state.length = 0;
    state.next_position = next_position;
    return Status::make_ok();
}

Status dflash2_context_shutdown(DFlash2ContextState& state) noexcept {
    state.k_ring = gpu::Tensor{};
    state.v_ring = gpu::Tensor{};
    state.num_layers = 0;
    state.capacity = 0;
    state.kv_features = 0;
    state.length = 0;
    state.next_position = 0;
    state.initialized = false;
    return Status::make_ok();
}

}  // namespace ps::qwen35::dflash2
