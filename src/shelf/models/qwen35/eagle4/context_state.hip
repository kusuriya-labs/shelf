#include <shelf/models/qwen35/eagle4/context_state.h>

#include <algorithm>
#include <limits>

namespace ps::qwen35::eagle4 {

namespace {

Result<gpu::Tensor> allocate_bf16(
    gpu::GpuArena& arena,
    const std::vector<std::size_t>& shape) {
    std::size_t count = 1;
    for (std::size_t d : shape) {
        if (d == 0) {
            return Status::invalid_argument("eagle4 ring shape has a zero dim", __FILE__,
                                            __LINE__);
        }
        if (std::numeric_limits<std::size_t>::max() / count < d) {
            return Status::invalid_argument("eagle4 ring element count overflow", __FILE__,
                                            __LINE__);
        }
        count *= d;
    }
    auto view = arena.allocate_aligned(count * sizeof(bf16_t), 256);
    if (!view.ok()) return view.status();
    auto backing = view.release();
    std::vector<std::size_t> strides(shape.size());
    std::size_t stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return gpu::Tensor::create<bf16_t>(backing, shape, strides);
}

}  // namespace

Result<Eagle4ContextState> create_eagle4_context_state(
    const Eagle4Config& config,
    gpu::GpuArena& arena,
    uint32_t initial_position,
    std::uint32_t tp_size) {
    if (config.sliding_window == 0) {
        return Status::invalid_argument("eagle4 context: sliding_window is zero", __FILE__,
                                        __LINE__);
    }
    if (config.num_hidden_layers != 1) {
        return Status::invalid_argument("eagle4 context: layer count mismatch", __FILE__,
                                        __LINE__);
    }
    if (tp_size == 0) {
        return Status::invalid_argument("eagle4 context: tp_size is zero", __FILE__, __LINE__);
    }
    if ((config.attention_kv_rows() % tp_size) != 0) {
        return Status::unsupported("eagle4 context: kv rows not divisible by tp_size",
                                   __FILE__, __LINE__);
    }
    Eagle4ContextState state;
    state.num_layers = static_cast<uint32_t>(config.num_hidden_layers);
    state.capacity = static_cast<uint32_t>(config.sliding_window);
    state.kv_features = static_cast<uint32_t>(config.attention_kv_rows() / tp_size);
    state.next_position = initial_position;

    const std::vector<std::size_t> shape{
        static_cast<std::size_t>(state.num_layers),
        static_cast<std::size_t>(state.capacity),
        static_cast<std::size_t>(state.kv_features)};
    auto k = allocate_bf16(arena, shape);
    if (!k.ok()) return k.status();
    state.k_ring = k.release();
    auto v = allocate_bf16(arena, shape);
    if (!v.ok()) return v.status();
    state.v_ring = v.release();

    state.initialized = true;
    return state;
}

Status eagle4_context_reset(Eagle4ContextState& state, uint32_t next_position) {
    if (!state.initialized) {
        return Status::invalid_state("eagle4 context reset: not initialized", __FILE__,
                                     __LINE__);
    }
    state.next_position = next_position;
    return Status::make_ok();
}

Status eagle4_context_shutdown(Eagle4ContextState& state) noexcept {
    state.k_ring = gpu::Tensor{};
    state.v_ring = gpu::Tensor{};
    state.num_layers = 0;
    state.capacity = 0;
    state.kv_features = 0;
    state.next_position = 0;
    state.initialized = false;
    return Status::make_ok();
}

}  // namespace ps::qwen35::eagle4
