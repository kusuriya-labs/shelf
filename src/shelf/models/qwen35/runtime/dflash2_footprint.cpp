#include <shelf/models/qwen35/runtime/dflash2_footprint.h>

#include <shelf/models/qwen35/dflash2/config.h>
#include <shelf/models/qwen35/runtime/dflash2_spec_decoder.h>
#include <shelf/models/qwen35/runtime/spec_decode.h>
#include <shelf/inference/state/gdn_state_pool.h>
#include <shelf/runtime/batch/device_batch_context.h>
#include <shelf/runtime/program/int8_activation_workspace.h>
#include <shelf/models/qwen35/kernels/dflash2/batch.h>
#include <shelf/models/qwen35/kernels/dflash2/topk.h>

#include <algorithm>

namespace ps::qwen35 {
namespace runtime {

namespace {

constexpr std::size_t kAlignBytes = 256u;

std::size_t align_up(std::size_t bytes) noexcept {
    return ((bytes + kAlignBytes - 1u) / kAlignBytes) * kAlignBytes;
}

std::size_t rows_for(const dflash2::DFlash2Config& config, uint32_t sessions) noexcept {
    const std::size_t block_size = config.block_size == 0u ? 1u : config.block_size;
    const std::size_t batch_rows = block_size * (sessions == 0u ? 1u : sessions);
    return std::max(block_size,
                    std::min<std::size_t>(batch_rows, kDFlash2SpecMaxVerifyRows));
}

}  // namespace

std::size_t dflash2_context_state_bytes(const dflash2::DFlash2Config& config) noexcept {
    return align_up(config.context_ring_bytes());
}

std::size_t dflash2_spec_workspace_bytes(const DFlash2SpecDecoderConfig& config,
                                         const ::ps::state::GdnStatePool& pool) noexcept {
    const std::size_t verify_rows = kDFlash2SpecMaxVerifyRows;
    std::size_t total = align_up(verify_rows * sizeof(int32_t)) +
                        align_up(2u * verify_rows * sizeof(int32_t));

    const std::size_t conv_bytes = spec_gdn_conv_bytes(pool);
    const std::size_t rec_bytes = spec_gdn_recurrent_bytes(pool);
    total += align_up(conv_bytes) + align_up(rec_bytes);

    const ::ps::state::GdnStatePoolDeviceView view = pool.device_view();
    const uint32_t history_rows = config.num_drafts == 0u ? 1u : config.num_drafts;
    total += align_up(static_cast<std::size_t>(history_rows) * view.conv_slot_stride *
                      sizeof(bf16_t));
    total += align_up(static_cast<std::size_t>(history_rows) * view.recurrent_slot_stride *
                      sizeof(fp16_t));

    const std::size_t log_rows = static_cast<std::size_t>(config.num_drafts) + 1u;
    const std::size_t states = view.num_gdn_states;
    const std::size_t dper = log_rows * static_cast<std::size_t>(view.num_v_heads) * view.head_v;
    const std::size_t kper = log_rows * static_cast<std::size_t>(view.num_v_heads) * view.head_k;
    const std::size_t aper = log_rows * static_cast<std::size_t>(view.num_v_heads);
    total += align_up(states * (dper + kper + aper) * sizeof(float));
    total += align_up(sizeof(::ps::runtime::DeviceRequestDescriptor));
    return total;
}

std::size_t dflash2_spec_batch_context_bytes(const ::ps::state::GdnStatePool& pool,
                                             std::size_t max_rows) noexcept {
    const ::ps::state::GdnStatePoolDeviceView view = pool.device_view();
    std::size_t total = align_up(max_rows * sizeof(int32_t)) +
                        align_up(2u * max_rows * sizeof(int32_t));
    total += align_up(max_rows * view.conv_slot_stride * sizeof(bf16_t));

    const std::size_t states = view.num_gdn_states;
    const std::size_t dper = max_rows * static_cast<std::size_t>(view.num_v_heads) * view.head_v;
    const std::size_t kper = max_rows * static_cast<std::size_t>(view.num_v_heads) * view.head_k;
    const std::size_t aper = max_rows * static_cast<std::size_t>(view.num_v_heads);
    total += align_up(states * (dper + kper + aper) * sizeof(float));
    return total;
}

std::size_t dflash2_draft_scratch_bytes(const dflash2::DFlash2Config& config,
                                        std::size_t rows,
                                        std::size_t lm_head_k) noexcept {
    const std::size_t hidden = config.hidden_size;
    const std::size_t intermediate = config.intermediate_size;
    const std::size_t q_features = config.attention_q_rows();
    const std::size_t kv_features = config.attention_kv_rows();
    const std::size_t ctx_rows = config.sliding_window == 0u ? rows : config.sliding_window;
    const std::size_t draft = config.block_size != 0u ? config.block_size : rows;

    std::size_t bf16_rows = 0u;
    bf16_rows += 16u * rows * hidden;
    bf16_rows += 4u * rows * q_features;
    bf16_rows += 4u * rows * kv_features;
    bf16_rows += 4u * ctx_rows * kv_features;
    bf16_rows += 3u * rows * intermediate;
    bf16_rows += rows * config.tap_feature_size();
    bf16_rows += rows * config.conv_projection_rows();
    bf16_rows += draft * config.selector_rank;
    std::size_t total = align_up(bf16_rows * sizeof(bf16_t));

    total += align_up(draft * config.vocab_size * sizeof(float));
    total += align_up(draft * config.selector_top_k * (sizeof(int32_t) + sizeof(float)));
    total += align_up(draft * (1u + 2u * ::ps::kernel::kDflash2TopKScratchPerRow) *
                      sizeof(int32_t));

    const uint32_t act_k = static_cast<uint32_t>(
        std::max<std::size_t>(hidden, std::max<std::size_t>(lm_head_k, 1u)));
    const auto act_layout =
        ::ps::runtime::Int8ActivationWorkspaceLayout::make(act_k, static_cast<uint32_t>(draft));
    total += align_up(act_layout.codes_bytes()) + align_up(act_layout.scales_bytes());

    total += align_up(rows * (sizeof(uint32_t) + sizeof(int32_t) +
                              sizeof(::ps::kernel::DFlash2RequestDescriptor)));
    return total;
}

DFlash2Footprint dflash2_footprint(const dflash2::DFlash2Config& config,
                                   const DFlash2SpecDecoderConfig& spec_config,
                                   const ::ps::state::GdnStatePool& pool,
                                   std::size_t draft_weights_bytes,
                                   std::size_t lm_head_k,
                                   uint32_t sessions) noexcept {
    DFlash2Footprint footprint;
    footprint.draft_weights = draft_weights_bytes;
    footprint.draft_scratch =
        dflash2_draft_scratch_bytes(config, rows_for(config, sessions), lm_head_k);
    footprint.session_slot =
        dflash2_context_state_bytes(config) + dflash2_spec_workspace_bytes(spec_config, pool);
    const std::size_t batch_rows = std::min<std::size_t>(
        kDFlash2SpecMaxVerifyRows,
        (sessions == 0u ? 1u : sessions) * (config.block_size == 0u ? 1u : config.block_size));
    footprint.spec_batch = dflash2_spec_batch_context_bytes(pool, batch_rows);
    footprint.sessions = sessions == 0u ? 1u : sessions;
    return footprint;
}

}  // namespace runtime
}  // namespace ps::qwen35
