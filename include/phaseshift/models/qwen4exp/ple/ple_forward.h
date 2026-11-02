#pragma once

#include <phaseshift/core/status.h>
#include <phaseshift/inference/state/paged_types.h>
#include <phaseshift/models/qwen4exp/model/qwen4exp_config.h>
#include <phaseshift/models/qwen4exp/model/qwen4exp_weights.h>
#include <phaseshift/models/qwen4exp/ple/ngram_hash.h>
#include <phaseshift/models/qwen4exp/ple/ple_row_gather.h>
#include <phaseshift/models/qwen4exp/runtime/qwen4exp_state.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace ps::qwen4exp::ple {

struct DeviceBufferDeleter {
    void operator()(void* p) const noexcept {
        if (p != nullptr) (void)hipFree(p);
    }
};

using DeviceBufferPtr = std::unique_ptr<void, DeviceBufferDeleter>;

struct PleRowSource {
    enum class Kind { vram = 0, ram = 1, ssd = 2 };

    Kind kind = Kind::vram;
    TableLayout layout;
    TableLayout scale_layout;
    const void* device_table = nullptr;
    std::uint64_t table_rows = 0;
    const ::ps::weights::SafetensorsCollection* collection = nullptr;
    bool allow_hipfile = true;
};

class PleBranch {
 public:
    static Result<PleBranch> create(const Qwen4ExpTextConfig& config,
                                    const Qwen4ExpPleWeights& weights,
                                    std::uint32_t max_tokens,
                                    const PleRowSource& source, hipStream_t stream);

    Status run(const std::uint32_t* input_ids, std::uint32_t count,
               const float* hidden_states,
               ::ps::qwen4exp::runtime::PleNgramContext& ngram,
               ::ps::qwen4exp::runtime::PleConvStatePool& conv,
               ::ps::state::SequenceSlotId slot, float* out_host, hipStream_t stream);

    Status shutdown();

    bool is_shutdown() const noexcept { return shutdown_; }

    std::uint32_t embed_dim() const noexcept { return embed_dim_; }

    std::uint32_t hidden() const noexcept { return hidden_; }

    std::uint32_t streams() const noexcept { return streams_; }

    std::uint32_t max_tokens() const noexcept { return max_tokens_; }

    std::uint32_t eos_token() const noexcept { return eos_token_; }

    const NgramIdTables& ngram_tables() const noexcept { return tables_; }

    const TableLayout& row_layout() const noexcept { return source_.layout; }

    const float* key_proj_host() const noexcept { return key_proj_.data(); }

    const float* value_proj_host() const noexcept { return value_proj_.data(); }

    const float* conv_weight_host() const noexcept { return conv_weight_.data(); }

    const float* norm_key_host() const noexcept { return norm_key_.data(); }

    const float* norm_query_host() const noexcept { return norm_query_.data(); }

    const float* norm_conv_host() const noexcept { return norm_conv_.data(); }

 private:
    PleBranch() = default;

    Status release_device();

    std::uint32_t embed_dim_ = 0;
    std::uint32_t hidden_ = 0;
    std::uint32_t streams_ = 0;
    std::uint32_t max_tokens_ = 0;
    std::uint32_t eos_token_ = 0;
    float eps_ = 1.0e-6f;
    NgramIdTables tables_{};
    PleRowSource source_{};

    std::vector<float> key_proj_;
    std::vector<float> value_proj_;
    std::vector<float> conv_weight_;
    std::vector<float> norm_key_;
    std::vector<float> norm_query_;
    std::vector<float> norm_conv_;

    DeviceBufferPtr row_ids_dev_;
    DeviceBufferPtr rows_dev_;
    bool shutdown_ = false;
};

}  // namespace ps::qwen4exp::ple
