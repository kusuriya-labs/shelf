#pragma once

#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/core/status.h>
#include <phaseshift/inference/state/paged_types.h>
#include <phaseshift/models/qwen4exp/ple/ngram_hash.h>
#include <phaseshift/models/qwen4exp/ple/ple_conv.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <vector>

namespace ps::qwen4exp::runtime {

using SequenceSlotId = ::ps::state::SequenceSlotId;

inline constexpr std::uint32_t kIndexerKeyDim = 128;

class IndexerKeyPool {
 public:
    static Result<IndexerKeyPool> create(
        gpu::GpuArena& arena,
        std::uint32_t max_sequences,
        std::uint32_t num_layers,
        std::uint32_t max_tokens);

    Status initialize(SequenceSlotId slot, hipStream_t stream);

    Status reset(SequenceSlotId slot, hipStream_t stream);

    Status release(SequenceSlotId slot);

    Status shutdown();

    bool is_shutdown() const noexcept { return shutdown_; }

    bf16_t* key_row(SequenceSlotId slot, std::uint32_t layer, std::uint32_t token);

    const bf16_t* key_cache(SequenceSlotId slot, std::uint32_t layer) const;

    std::size_t key_elements_offset(SequenceSlotId slot, std::uint32_t layer) const;

    std::uint32_t max_sequences() const noexcept { return max_slots_; }

    std::uint32_t num_layers() const noexcept { return num_layers_; }

    std::uint32_t max_tokens() const noexcept { return max_tokens_; }

    std::uint32_t key_dim() const noexcept { return kIndexerKeyDim; }

    int device() const noexcept { return device_; }

    std::size_t reserved_bytes() const noexcept { return reserved_bytes_; }

    std::size_t bytes_per_slot() const noexcept { return slot_bytes_; }

    std::uint32_t num_used_slots() const noexcept { return used_count_; }

    bool is_initialized(SequenceSlotId slot) const noexcept;

    const gpu::Tensor& key_memory() const noexcept { return key_memory_; }

 private:
    IndexerKeyPool() = default;

    bool valid_slot(SequenceSlotId slot) const noexcept;

    std::size_t slot_elements() const noexcept;

    gpu::Tensor key_memory_;
    int device_ = -1;
    std::uint32_t max_slots_ = 0;
    std::uint32_t num_layers_ = 0;
    std::uint32_t max_tokens_ = 0;
    std::vector<bool> initialized_;
    std::size_t reserved_bytes_ = 0;
    std::size_t slot_bytes_ = 0;
    std::uint32_t used_count_ = 0;
    bool shutdown_ = false;
};

class PleConvStatePool {
 public:
    static Result<PleConvStatePool> create(std::uint32_t max_sequences, std::uint32_t conv_dim);

    Status initialize(SequenceSlotId slot);

    Status reset(SequenceSlotId slot);

    Status release(SequenceSlotId slot);

    Status shutdown();

    bool is_shutdown() const noexcept { return shutdown_; }

    Status convolve(SequenceSlotId slot, const float* weight, const float* input,
                    std::uint32_t tokens, float* output);

    const ple::PleConvState* state(SequenceSlotId slot) const noexcept;

    std::uint32_t max_sequences() const noexcept { return max_slots_; }

    std::uint32_t conv_dim() const noexcept { return conv_dim_; }

    std::uint32_t conv_history() const noexcept { return ple::kPleConvStateLen; }

    std::uint32_t num_used_slots() const noexcept { return used_count_; }

    bool is_initialized(SequenceSlotId slot) const noexcept;

 private:
    PleConvStatePool() = default;

    bool valid_slot(SequenceSlotId slot) const noexcept;

    std::vector<ple::PleConvState> states_;
    std::vector<bool> initialized_;
    std::uint32_t max_slots_ = 0;
    std::uint32_t conv_dim_ = 0;
    std::uint32_t used_count_ = 0;
    bool shutdown_ = false;
};

class PleNgramContext {
 public:
    static Result<PleNgramContext> create(
        gpu::GpuArena& arena,
        std::uint32_t max_sequences,
        std::uint32_t eos_token);

    Status initialize(SequenceSlotId slot);

    Status advance(SequenceSlotId slot, const std::uint32_t* input_ids, std::uint32_t count);

    Status reset(SequenceSlotId slot);

    Status release(SequenceSlotId slot);

    Status upload(SequenceSlotId slot, hipStream_t stream);

    Status shutdown();

    bool is_shutdown() const noexcept { return shutdown_; }

    const std::uint32_t* context(SequenceSlotId slot) const noexcept;

    std::uint32_t* device_context(SequenceSlotId slot) noexcept;

    std::uint32_t max_sequences() const noexcept { return max_slots_; }

    std::uint32_t eos_token() const noexcept { return eos_token_; }

    int device() const noexcept { return device_; }

    std::uint32_t num_used_slots() const noexcept { return used_count_; }

    bool is_initialized(SequenceSlotId slot) const noexcept;

    const gpu::Tensor& device_memory() const noexcept { return device_memory_; }

 private:
    PleNgramContext() = default;

    bool valid_slot(SequenceSlotId slot) const noexcept;

    void fill_initial(SequenceSlotId slot);

    std::vector<std::uint32_t> contexts_;
    std::vector<bool> initialized_;
    gpu::Tensor device_memory_;
    int device_ = -1;
    std::uint32_t max_slots_ = 0;
    std::uint32_t eos_token_ = 0;
    std::uint32_t used_count_ = 0;
    bool shutdown_ = false;
};

}
