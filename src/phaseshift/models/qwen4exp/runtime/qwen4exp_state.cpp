#include <phaseshift/models/qwen4exp/runtime/qwen4exp_state.h>

#include <cstdint>

namespace ps::qwen4exp::runtime {
namespace {

constexpr std::size_t kIndexerAlignment = 256;
constexpr std::size_t kContextAlignment = 256;

}

Result<IndexerKeyPool> IndexerKeyPool::create(
    gpu::GpuArena& arena,
    std::uint32_t max_sequences,
    std::uint32_t num_layers,
    std::uint32_t max_tokens)
{
    if (max_sequences == 0) {
        return Status::invalid_argument("max_sequences must be > 0", __FILE__, __LINE__);
    }
    if (num_layers == 0) {
        return Status::invalid_argument("num_layers must be > 0", __FILE__, __LINE__);
    }
    if (max_tokens == 0) {
        return Status::invalid_argument("max_tokens must be > 0", __FILE__, __LINE__);
    }

    const std::size_t slot_elems =
        static_cast<std::size_t>(num_layers) * max_tokens * kIndexerKeyDim;
    const std::size_t total_elems = slot_elems * static_cast<std::size_t>(max_sequences);
    const std::size_t total_bytes = total_elems * sizeof(bf16_t);

    auto allocation = arena.allocate_aligned(total_bytes, kIndexerAlignment);
    if (!allocation.ok()) {
        return allocation.status();
    }

    const std::size_t shape[4] = {
        static_cast<std::size_t>(max_sequences),
        static_cast<std::size_t>(num_layers),
        static_cast<std::size_t>(max_tokens),
        static_cast<std::size_t>(kIndexerKeyDim),
    };
    const std::size_t strides[4] = {
        slot_elems,
        static_cast<std::size_t>(max_tokens) * kIndexerKeyDim,
        static_cast<std::size_t>(kIndexerKeyDim),
        1,
    };

    auto tensor = gpu::Tensor::create<bf16_t>(allocation.value(), shape, 4, strides);
    if (!tensor.ok()) {
        return tensor.status();
    }

    IndexerKeyPool pool;
    pool.device_ = arena.device();
    pool.max_slots_ = max_sequences;
    pool.num_layers_ = num_layers;
    pool.max_tokens_ = max_tokens;
    pool.key_memory_ = tensor.value();
    pool.reserved_bytes_ = total_bytes;
    pool.slot_bytes_ = slot_elems * sizeof(bf16_t);
    pool.used_count_ = 0;
    pool.initialized_.assign(static_cast<std::size_t>(max_sequences), false);

    return pool;
}

bool IndexerKeyPool::valid_slot(SequenceSlotId slot) const noexcept {
    return !shutdown_ && slot < max_slots_;
}

std::size_t IndexerKeyPool::slot_elements() const noexcept {
    return static_cast<std::size_t>(num_layers_) * max_tokens_ * kIndexerKeyDim;
}

bool IndexerKeyPool::is_initialized(SequenceSlotId slot) const noexcept {
    return valid_slot(slot) && initialized_[slot];
}

Status IndexerKeyPool::initialize(SequenceSlotId slot, hipStream_t stream) {
    if (shutdown_) {
        return Status::invalid_state("pool is shut down", __FILE__, __LINE__);
    }
    if (slot >= max_slots_) {
        return Status::out_of_range("slot out of range", __FILE__, __LINE__);
    }
    if (initialized_[slot]) {
        return Status::invalid_state("slot already initialized", __FILE__, __LINE__);
    }

    bf16_t* base = key_memory_.data<bf16_t>();
    hipError_t err = hipMemsetAsync(
        base + static_cast<std::size_t>(slot) * slot_elements(),
        0,
        slot_bytes_,
        stream);
    if (err != hipSuccess) {
        return Status::hip_error("hipMemsetAsync", "initialize indexer keys", __FILE__, __LINE__);
    }

    initialized_[slot] = true;
    ++used_count_;
    return Status::make_ok();
}

Status IndexerKeyPool::reset(SequenceSlotId slot, hipStream_t stream) {
    if (shutdown_) {
        return Status::invalid_state("pool is shut down", __FILE__, __LINE__);
    }
    if (slot >= max_slots_) {
        return Status::out_of_range("slot out of range", __FILE__, __LINE__);
    }
    if (!initialized_[slot]) {
        return Status::invalid_state("slot not initialized", __FILE__, __LINE__);
    }

    bf16_t* base = key_memory_.data<bf16_t>();
    hipError_t err = hipMemsetAsync(
        base + static_cast<std::size_t>(slot) * slot_elements(),
        0,
        slot_bytes_,
        stream);
    if (err != hipSuccess) {
        return Status::hip_error("hipMemsetAsync", "reset indexer keys", __FILE__, __LINE__);
    }

    return Status::make_ok();
}

Status IndexerKeyPool::release(SequenceSlotId slot) {
    if (shutdown_) {
        return Status::invalid_state("pool is shut down", __FILE__, __LINE__);
    }
    if (slot >= max_slots_) {
        return Status::out_of_range("slot out of range", __FILE__, __LINE__);
    }
    if (!initialized_[slot]) {
        return Status::invalid_state("slot not initialized", __FILE__, __LINE__);
    }

    initialized_[slot] = false;
    --used_count_;
    return Status::make_ok();
}

Status IndexerKeyPool::shutdown() {
    if (shutdown_) {
        return Status::make_ok();
    }

    initialized_.clear();
    key_memory_ = gpu::Tensor{};
    used_count_ = 0;
    shutdown_ = true;
    return Status::make_ok();
}

bf16_t* IndexerKeyPool::key_row(SequenceSlotId slot, std::uint32_t layer, std::uint32_t token) {
    if (!is_initialized(slot) || layer >= num_layers_ || token >= max_tokens_) {
        return nullptr;
    }
    const std::size_t offset =
        static_cast<std::size_t>(slot) * slot_elements()
        + static_cast<std::size_t>(layer) * max_tokens_ * kIndexerKeyDim
        + static_cast<std::size_t>(token) * kIndexerKeyDim;
    return key_memory_.data<bf16_t>() + offset;
}

const bf16_t* IndexerKeyPool::key_cache(SequenceSlotId slot, std::uint32_t layer) const {
    if (!is_initialized(slot) || layer >= num_layers_) {
        return nullptr;
    }
    return key_memory_.data<bf16_t>() + key_elements_offset(slot, layer);
}

std::size_t IndexerKeyPool::key_elements_offset(SequenceSlotId slot,
                                                std::uint32_t layer) const {
    if (!valid_slot(slot) || layer >= num_layers_) {
        return 0;
    }
    return static_cast<std::size_t>(slot) * slot_elements()
        + static_cast<std::size_t>(layer) * max_tokens_ * kIndexerKeyDim;
}

Result<PleConvStatePool> PleConvStatePool::create(std::uint32_t max_sequences,
                                                  std::uint32_t conv_dim) {
    if (max_sequences == 0) {
        return Status::invalid_argument("max_sequences must be > 0", __FILE__, __LINE__);
    }
    if (conv_dim == 0) {
        return Status::invalid_argument("conv_dim must be > 0", __FILE__, __LINE__);
    }

    PleConvStatePool pool;
    pool.max_slots_ = max_sequences;
    pool.conv_dim_ = conv_dim;
    pool.used_count_ = 0;
    pool.states_.resize(static_cast<std::size_t>(max_sequences));
    pool.initialized_.assign(static_cast<std::size_t>(max_sequences), false);

    return pool;
}

bool PleConvStatePool::valid_slot(SequenceSlotId slot) const noexcept {
    return !shutdown_ && slot < max_slots_;
}

bool PleConvStatePool::is_initialized(SequenceSlotId slot) const noexcept {
    return valid_slot(slot) && initialized_[slot];
}

Status PleConvStatePool::initialize(SequenceSlotId slot) {
    if (shutdown_) {
        return Status::invalid_state("pool is shut down", __FILE__, __LINE__);
    }
    if (slot >= max_slots_) {
        return Status::out_of_range("slot out of range", __FILE__, __LINE__);
    }
    if (initialized_[slot]) {
        return Status::invalid_state("slot already initialized", __FILE__, __LINE__);
    }

    states_[slot].reset(conv_dim_);
    initialized_[slot] = true;
    ++used_count_;
    return Status::make_ok();
}

Status PleConvStatePool::reset(SequenceSlotId slot) {
    if (shutdown_) {
        return Status::invalid_state("pool is shut down", __FILE__, __LINE__);
    }
    if (slot >= max_slots_) {
        return Status::out_of_range("slot out of range", __FILE__, __LINE__);
    }
    if (!initialized_[slot]) {
        return Status::invalid_state("slot not initialized", __FILE__, __LINE__);
    }

    states_[slot].reset(conv_dim_);
    return Status::make_ok();
}

Status PleConvStatePool::release(SequenceSlotId slot) {
    if (shutdown_) {
        return Status::invalid_state("pool is shut down", __FILE__, __LINE__);
    }
    if (slot >= max_slots_) {
        return Status::out_of_range("slot out of range", __FILE__, __LINE__);
    }
    if (!initialized_[slot]) {
        return Status::invalid_state("slot not initialized", __FILE__, __LINE__);
    }

    states_[slot] = ple::PleConvState{};
    initialized_[slot] = false;
    --used_count_;
    return Status::make_ok();
}

Status PleConvStatePool::shutdown() {
    if (shutdown_) {
        return Status::make_ok();
    }

    states_.clear();
    initialized_.clear();
    used_count_ = 0;
    shutdown_ = true;
    return Status::make_ok();
}

Status PleConvStatePool::convolve(SequenceSlotId slot, const float* weight, const float* input,
                                  std::uint32_t tokens, float* output) {
    if (!is_initialized(slot)) {
        if (shutdown_) {
            return Status::invalid_state("pool is shut down", __FILE__, __LINE__);
        }
        if (slot >= max_slots_) {
            return Status::out_of_range("slot out of range", __FILE__, __LINE__);
        }
        return Status::invalid_state("slot not initialized", __FILE__, __LINE__);
    }

    return ple::ple_dilated_conv1d(
        weight, input, tokens, conv_dim_, states_[slot], output);
}

const ple::PleConvState* PleConvStatePool::state(SequenceSlotId slot) const noexcept {
    if (!is_initialized(slot)) {
        return nullptr;
    }
    return &states_[slot];
}

Result<PleNgramContext> PleNgramContext::create(
    gpu::GpuArena& arena,
    std::uint32_t max_sequences,
    std::uint32_t eos_token)
{
    if (max_sequences == 0) {
        return Status::invalid_argument("max_sequences must be > 0", __FILE__, __LINE__);
    }

    const std::size_t total_elems =
        static_cast<std::size_t>(max_sequences) * ple::kNgramContextTokens;
    const std::size_t total_bytes = total_elems * sizeof(std::uint32_t);

    auto allocation = arena.allocate_aligned(total_bytes, kContextAlignment);
    if (!allocation.ok()) {
        return allocation.status();
    }

    const std::size_t shape[2] = {
        static_cast<std::size_t>(max_sequences),
        static_cast<std::size_t>(ple::kNgramContextTokens),
    };
    const std::size_t strides[2] = {
        static_cast<std::size_t>(ple::kNgramContextTokens),
        1,
    };

    auto tensor = gpu::Tensor::create<std::uint32_t>(allocation.value(), shape, 2, strides);
    if (!tensor.ok()) {
        return tensor.status();
    }

    PleNgramContext ctx;
    ctx.device_ = arena.device();
    ctx.max_slots_ = max_sequences;
    ctx.eos_token_ = eos_token;
    ctx.device_memory_ = tensor.value();
    ctx.used_count_ = 0;
    ctx.contexts_.assign(total_elems, eos_token);
    ctx.initialized_.assign(static_cast<std::size_t>(max_sequences), false);

    return ctx;
}

bool PleNgramContext::valid_slot(SequenceSlotId slot) const noexcept {
    return !shutdown_ && slot < max_slots_;
}

bool PleNgramContext::is_initialized(SequenceSlotId slot) const noexcept {
    return valid_slot(slot) && initialized_[slot];
}

void PleNgramContext::fill_initial(SequenceSlotId slot) {
    std::uint32_t* row = contexts_.data()
        + static_cast<std::size_t>(slot) * ple::kNgramContextTokens;
    for (std::uint32_t i = 0; i < ple::kNgramContextTokens; ++i) {
        row[i] = eos_token_;
    }
}

Status PleNgramContext::initialize(SequenceSlotId slot) {
    if (shutdown_) {
        return Status::invalid_state("pool is shut down", __FILE__, __LINE__);
    }
    if (slot >= max_slots_) {
        return Status::out_of_range("slot out of range", __FILE__, __LINE__);
    }
    if (initialized_[slot]) {
        return Status::invalid_state("slot already initialized", __FILE__, __LINE__);
    }

    fill_initial(slot);
    initialized_[slot] = true;
    ++used_count_;
    return Status::make_ok();
}

Status PleNgramContext::advance(SequenceSlotId slot, const std::uint32_t* input_ids,
                                std::uint32_t count) {
    if (!is_initialized(slot)) {
        if (shutdown_) {
            return Status::invalid_state("pool is shut down", __FILE__, __LINE__);
        }
        if (slot >= max_slots_) {
            return Status::out_of_range("slot out of range", __FILE__, __LINE__);
        }
        return Status::invalid_state("slot not initialized", __FILE__, __LINE__);
    }
    if (count > 0u && input_ids == nullptr) {
        return Status::invalid_argument("input_ids must not be null", __FILE__, __LINE__);
    }

    const std::uint32_t* prev = context(slot);
    std::vector<std::uint32_t> history(
        static_cast<std::size_t>(count) + ple::kNgramContextTokens);
    for (std::uint32_t i = 0; i < ple::kNgramContextTokens; ++i) {
        history[i] = prev[i];
    }
    for (std::uint32_t i = 0; i < count; ++i) {
        history[ple::kNgramContextTokens + i] = input_ids[i];
    }

    std::uint32_t next[ple::kNgramContextTokens];
    ple::advance_ngram_context(history.data(), count + ple::kNgramContextTokens, next);

    std::uint32_t* row = contexts_.data()
        + static_cast<std::size_t>(slot) * ple::kNgramContextTokens;
    for (std::uint32_t i = 0; i < ple::kNgramContextTokens; ++i) {
        row[i] = next[i];
    }

    return Status::make_ok();
}

Status PleNgramContext::reset(SequenceSlotId slot) {
    if (shutdown_) {
        return Status::invalid_state("pool is shut down", __FILE__, __LINE__);
    }
    if (slot >= max_slots_) {
        return Status::out_of_range("slot out of range", __FILE__, __LINE__);
    }
    if (!initialized_[slot]) {
        return Status::invalid_state("slot not initialized", __FILE__, __LINE__);
    }

    fill_initial(slot);
    return Status::make_ok();
}

Status PleNgramContext::release(SequenceSlotId slot) {
    if (shutdown_) {
        return Status::invalid_state("pool is shut down", __FILE__, __LINE__);
    }
    if (slot >= max_slots_) {
        return Status::out_of_range("slot out of range", __FILE__, __LINE__);
    }
    if (!initialized_[slot]) {
        return Status::invalid_state("slot not initialized", __FILE__, __LINE__);
    }

    fill_initial(slot);
    initialized_[slot] = false;
    --used_count_;
    return Status::make_ok();
}

Status PleNgramContext::upload(SequenceSlotId slot, hipStream_t stream) {
    if (!is_initialized(slot)) {
        if (shutdown_) {
            return Status::invalid_state("pool is shut down", __FILE__, __LINE__);
        }
        if (slot >= max_slots_) {
            return Status::out_of_range("slot out of range", __FILE__, __LINE__);
        }
        return Status::invalid_state("slot not initialized", __FILE__, __LINE__);
    }

    const std::uint32_t* host = contexts_.data()
        + static_cast<std::size_t>(slot) * ple::kNgramContextTokens;
    std::uint32_t* device = device_memory_.data<std::uint32_t>()
        + static_cast<std::size_t>(slot) * ple::kNgramContextTokens;

    hipError_t err = hipMemcpyAsync(
        device,
        host,
        ple::kNgramContextTokens * sizeof(std::uint32_t),
        hipMemcpyHostToDevice,
        stream);
    if (err != hipSuccess) {
        return Status::hip_error("hipMemcpyAsync", "upload ngram context", __FILE__, __LINE__);
    }

    return Status::make_ok();
}

Status PleNgramContext::shutdown() {
    if (shutdown_) {
        return Status::make_ok();
    }

    contexts_.clear();
    initialized_.clear();
    device_memory_ = gpu::Tensor{};
    used_count_ = 0;
    shutdown_ = true;
    return Status::make_ok();
}

const std::uint32_t* PleNgramContext::context(SequenceSlotId slot) const noexcept {
    if (!is_initialized(slot)) {
        return nullptr;
    }
    return contexts_.data() + static_cast<std::size_t>(slot) * ple::kNgramContextTokens;
}

std::uint32_t* PleNgramContext::device_context(SequenceSlotId slot) noexcept {
    if (!is_initialized(slot)) {
        return nullptr;
    }
    return device_memory_.data<std::uint32_t>()
        + static_cast<std::size_t>(slot) * ple::kNgramContextTokens;
}

}
