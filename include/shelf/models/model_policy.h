#pragma once

#include <shelf/core/status.h>
#include <shelf/inference/state/gdn_state_pool.h>
#include <shelf/models/common/qwen_hybrid/config.h>
#include <shelf/runtime/rope_profile.h>
#include <shelf/runtime/tp/tensor_parallel.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace shelf::models {

class TpRankModel;
class LoadedModel;

// The family classification the compute side may see without opening a family header.
enum class ModelClass : std::uint8_t {
    Dense,
    Moe,
    Qsa,
    Qwen4Exp,
    Plamo3,
};

struct ModelRuntimeRequest {
    std::string model_dir;
    std::string rope_factor = "auto";
    std::uint32_t max_seq_len = 512;
    std::uint32_t tensor_parallel = 1;
    std::uint32_t expert_parallel = 1;
    std::uint32_t pipeline_parallel = 1;
    bool mtp = false;
    bool dflash2 = false;
    std::uint32_t dflash2_drafts = 7;
    bool private_target_weights = false;
    bool private_target_lm_head = true;
};

// The normalized configuration a model family hands back to the execution layer. The
// application never decides a family, a RoPE profile or a family-specific limit.
struct ResolvedModelOptions {
    ModelClass model_class = ModelClass::Dense;
    std::string display_family;
    bool has_rope_profile = false;
    runtime::EffectiveRopeProfile rope_profile{};
    std::uint32_t max_seq_len = 512;
    bool dflash2_host_token_ids = false;
    bool private_target_weights = false;
    bool private_target_lm_head = true;
    std::uint32_t dflash2_drafts = 7;
    std::vector<std::string> notices;
};

Result<ResolvedModelOptions> resolve_model_options(const ModelRuntimeRequest& request);

Status validate_model_runtime_options(const ModelRuntimeRequest& request,
                                      const ResolvedModelOptions& resolved);

// Family-specific checkpoint reads the execution layer needs but must not name.
std::uint32_t pipeline_hidden_width(const std::string& model_dir, ModelClass model_class);

std::size_t mtp_batch_reserve_bytes(
    const TpRankModel& model,
    const qwen_hybrid::QwenHybridConfig& text_config,
    const runtime::MtpBatchReserveConfig& config,
    const state::GdnStatePoolLayout& layout,
    std::uint32_t tp_rank);

std::uint32_t max_mtp_drafts() noexcept;
std::uint32_t max_mtp_prefill_rows() noexcept;

struct MtpReserveRequest {
    std::uint32_t num_drafts = 7;
    std::uint32_t max_seq_len = 0;
    std::uint32_t page_tokens = 16;
    std::uint32_t max_batch_rows = 0;
    kernel::KVCacheDType kv_dtype = kernel::KVCacheDType::BF16;
    runtime::EffectiveRopeProfile rope_profile{};
};

std::size_t mtp_drafter_reserve_bytes(const LoadedModel& model,
                                      const MtpReserveRequest& request,
                                      bool batched, std::size_t max_requests);

}  // namespace shelf::models
