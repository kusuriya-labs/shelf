#include <shelf/engine/speculation.h>

#include "compute_session_impl.h"

#include <shelf/models/loaded_model.h>
#include <shelf/runtime/dflash2/dflash2_session_pool.h>
#include <shelf/runtime/mtp/mtp_drafter.h>
#include <shelf/runtime/dflash2/context_state.h>
#include <shelf/runtime/dflash2/executor.h>
#include <shelf/runtime/dflash2/tp_proposal.h>
#include <shelf/runtime/dflash2/weights.h>
#include <shelf/runtime/dflash2_spec_decoder.h>
#include <shelf/runtime/decode_backend.h>
#include <shelf/runtime/executor.h>
#include <shelf/runtime/runtime_request.h>
#include <shelf/runtime/state_types.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace shelf::engine {
namespace {

void add_u(SpeculationResult& result, const char* key, unsigned long long value,
           const char* format = "%llu") {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), format, value);
    result.stats.push_back(SpeculationMetric{key, buffer});
}

void add_d(SpeculationResult& result, const char* key, double value, int precision) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.*f", precision, value);
    result.stats.push_back(SpeculationMetric{key, buffer});
}

double elapsed_ms(std::chrono::steady_clock::time_point begin,
                  std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

}  // namespace

Result<SpeculationResult> SpeculationRunner::run_mtp(
    ComputeSession& session, const std::vector<int32_t>& prompt_ids,
    unsigned long max_new_tokens, uint32_t mtp_drafts,
    const runtime::SamplingConfig& sampling) {
    ComputeSession::Impl& impl = *session.impl_;
    runtime::Executor& executor = impl.executor();
    auto& slot_pool = impl.slot_pool();
    auto& gdn_pool = impl.gdn_pool();
    auto& kv_pool = impl.kv_pool();
    auto& arena = impl.arena();
    const hipStream_t stream = impl.stream();
    const auto& text_config = impl.model().hybrid_config();

    const uint32_t max_seq = impl.config_.max_seq_len;
    auto sequence_result = shelf::state::create_paged_sequence_state(
        slot_pool, gdn_pool, kv_pool, max_seq, stream);
    if (!sequence_result.ok()) return sequence_result.status();
    shelf::state::PagedSequenceState sequence = sequence_result.release();

    shelf::runtime::mtp::MtpDrafterConfig drafter_config;
    drafter_config.num_drafts = mtp_drafts;
    drafter_config.window_tokens = static_cast<uint32_t>(impl.config_.mtp_window);
    drafter_config.sink_tokens = static_cast<uint32_t>(impl.config_.mtp_sink);
    drafter_config.max_seq_len = max_seq;
    drafter_config.page_tokens = impl.config_.page_tokens;
    drafter_config.max_batch_rows = std::min(
        executor.config.max_batch_rows, shelf::runtime::mtp::kMtpMaxPrefillRows);
    drafter_config.kv_dtype =
        impl.config_.kv_cache_dtype == shelf::kernel::KVCacheDType::OCTAVE
            ? shelf::kernel::KVCacheDType::BF16 : impl.config_.kv_cache_dtype;
    if (impl.config_.rope_profile.has_value()) {
        drafter_config.rope_profile = *impl.config_.rope_profile;
    }
    const shelf::weights::MatrixWeight* lm_head_override = nullptr;
    if (auto* tp = impl.tp_coordinator_; tp != nullptr && tp->tp_size() > 0) {
        auto& private_lm_head = tp->rank(0).private_lm_head();
        if (private_lm_head.rows != 0u) lm_head_override = &private_lm_head;
    }
    std::vector<shelf::runtime::mtp::MtpDrafterInput> rank_inputs;
    shelf::runtime::mtp::MtpDrafterInput leader_input;
    auto* tp = impl.tp_coordinator_;
    const bool per_rank_mtp =
        tp != nullptr && tp->tp_size() > 0 && tp->rank(0).rank_model() != nullptr &&
        tp->rank(0).rank_model()->mtp_uses_per_rank_input();
    if (per_rank_mtp) {
        const int rank_count = tp->tp_size();
        rank_inputs.reserve(static_cast<std::size_t>(rank_count));
        for (int r = 0; r < rank_count; ++r) {
            auto& rank = tp->rank(static_cast<std::size_t>(r));
            auto per_rank = drafter_config;
            per_rank.local_kv_heads = rank.context().local_key_value_heads;
            auto input = rank.rank_model()->mtp_drafter_input(per_rank);
            if (!input.ok()) {
                (void)shelf::state::release_paged_sequence_state(sequence, stream);
                return input.status();
            }
            rank_inputs.push_back(std::move(input.value()));
        }
        drafter_config.local_kv_heads = tp->rank(0).context().local_key_value_heads;
    } else {
        auto input = impl.model().mtp_drafter_input(drafter_config, lm_head_override);
        if (!input.ok()) {
            (void)shelf::state::release_paged_sequence_state(sequence, stream);
            return input.status();
        }
        leader_input = std::move(input.value());
    }
    const shelf::runtime::mtp::MtpDrafterInput& drafter_input =
        rank_inputs.empty() ? leader_input : rank_inputs.front();
    std::vector<shelf::runtime::mtp::MtpGdnRankBinding> extra_ranks;
    if (tp != nullptr) {
        for (int r = 1; r < tp->tp_size(); ++r) {
            auto& rank = tp->rank(static_cast<std::size_t>(r));
            shelf::runtime::mtp::MtpGdnRankBinding binding;
            binding.pool = &rank.gdn_pool();
            binding.arena = &rank.arena();
            binding.stream = rank.stream();
            binding.device = rank.device_id();
            binding.executor = &rank.executor();
            binding.local_kv_heads = rank.context().local_key_value_heads;
            binding.tp_rank = static_cast<uint32_t>(r);
            if (r < static_cast<int>(rank_inputs.size()))
                binding.mtp_input = &rank_inputs[static_cast<std::size_t>(r)];
            extra_ranks.push_back(binding);
        }
    }
    auto drafter_result = shelf::runtime::mtp::create_mtp_drafter(
        drafter_input, &executor, gdn_pool, arena, stream, drafter_config,
        extra_ranks.empty() ? nullptr : &extra_ranks, tp);
    if (!drafter_result.ok()) {
        (void)shelf::state::release_paged_sequence_state(sequence, stream);
        return drafter_result.status();
    }
    std::unique_ptr<shelf::runtime::mtp::MtpDrafter> drafter =
        drafter_result.release();
    if (tp != nullptr) {
        Status mirrored = tp->on_sequence_created(sequence);
        if (!mirrored.ok()) {
            (void)shelf::runtime::mtp::mtp_drafter_shutdown(*drafter, stream);
            drafter.reset();
            (void)shelf::state::release_paged_sequence_state(sequence, stream);
            return mirrored;
        }
    }
    auto cleanup = [&]() {
        (void)shelf::runtime::mtp::mtp_drafter_shutdown(*drafter, stream);
        drafter.reset();
        if (tp != nullptr) (void)tp->on_sequence_released(sequence);
        (void)shelf::state::release_paged_sequence_state(sequence, stream);
    };

    const auto t_prefill_start = std::chrono::steady_clock::now();
    auto prefill_result = shelf::runtime::mtp::mtp_spec_prefill(
        *drafter, executor, sequence, prompt_ids.data(),
        static_cast<uint32_t>(prompt_ids.size()), sampling);
    if (!prefill_result.ok()) {
        cleanup();
        return prefill_result.status();
    }
    int32_t pending = prefill_result.value().pending_token;
    SpeculationResult result;
    result.generated_ids.push_back(pending);
    uint32_t decode_tokens = 0u;
    const auto t_prefill_end = std::chrono::steady_clock::now();

    const auto t_decode_start = std::chrono::steady_clock::now();
    bool finished = shelf::qwen_hybrid::is_stop_token(text_config.stop_tokens, pending);
    while (!finished && result.generated_ids.size() < max_new_tokens) {
        const uint32_t remaining =
            static_cast<uint32_t>(max_new_tokens - result.generated_ids.size());
        auto step_result = shelf::runtime::mtp::mtp_spec_step(
            *drafter, executor, sequence, pending, remaining);
        if (!step_result.ok()) {
            cleanup();
            return step_result.status();
        }
        auto step = step_result.release();
        for (uint32_t i = 0u; i < step.emitted_count; ++i) {
            result.generated_ids.push_back(step.emitted[i]);
            ++decode_tokens;
        }
        pending = step.pending_token;
        finished = step.finished;
    }
    const hipError_t sync_err = hipStreamSynchronize(stream);
    if (sync_err != hipSuccess) {
        cleanup();
        return Status::hip_error("hipStreamSynchronize", hipGetErrorString(sync_err),
                                 __FILE__, __LINE__);
    }
    const auto t_decode_end = std::chrono::steady_clock::now();

    const auto& timing = drafter->timing;
    const float mean_accepted = timing.rounds > 0u
        ? static_cast<float>(timing.accepted_drafts) / static_cast<float>(timing.rounds)
        : 0.0f;
    const float full_rate = timing.rounds > 0u
        ? static_cast<float>(timing.full_accepts) / static_cast<float>(timing.rounds)
        : 0.0f;
    const float emitted_per_round = timing.rounds > 0u
        ? static_cast<float>(timing.generated_tokens) / static_cast<float>(timing.rounds)
        : 0.0f;
    add_u(result, "MTP_DRAFTS", drafter_config.num_drafts);
    add_u(result, "MTP_ROUNDS", timing.rounds);
    add_u(result, "MTP_FULL_ACCEPTS", timing.full_accepts);
    add_u(result, "MTP_PARTIAL_ACCEPTS", timing.partial_accepts);
    add_u(result, "MTP_ACCEPTED_DRAFTS", timing.accepted_drafts);
    add_d(result, "MTP_MEAN_ACCEPTED", mean_accepted, 3);
    add_d(result, "MTP_FULL_ACCEPT_RATE", full_rate, 3);
    add_d(result, "MTP_EMITTED_PER_ROUND", emitted_per_round, 3);
    add_u(result, "MTP_VERIFY_ROWS", timing.verify_rows);
    add_u(result, "MTP_GDN_RESTORES", timing.gdn_restores);
    add_d(result, "MTP_DRAFT_MS", timing.draft_ms, 2);
    add_d(result, "MTP_VERIFY_MS", timing.verify_ms, 2);
    add_d(result, "MTP_GDN_RESTORE_MS", timing.gdn_restore_ms, 2);
    add_d(result, "MTP_COMPACT_REPLAY_MS", timing.compact_replay_ms, 2);
    add_d(result, "MTP_ROUND_MS", timing.round_ms, 2);

    cleanup();

    result.decode_tokens = decode_tokens;
    result.prefill_ms = elapsed_ms(t_prefill_start, t_prefill_end);
    result.decode_ms = elapsed_ms(t_decode_start, t_decode_end);
    return result;
}

Result<SpeculationResult> SpeculationRunner::run_dflash2(
    ComputeSession& session, const std::vector<int32_t>& prompt_ids,
    unsigned long max_new_tokens, const runtime::DFlash2Config& dflash_config,
    unsigned long dflash2_drafts, bool adaptive, bool dflash2_tree,
    unsigned long dflash2_tree_extra, unsigned long dflash2_tree_budget,
    const runtime::SamplingConfig& sampling) {
    ComputeSession::Impl& impl = *session.impl_;
    runtime::Executor& executor = impl.executor();
    const auto& model = impl.model();
    auto& slot_pool = impl.slot_pool();
    auto& gdn_pool = impl.gdn_pool();
    auto& kv_pool = impl.kv_pool();
    auto& arena = impl.arena();
    const hipStream_t stream = impl.stream();
    const auto& text_config = model.hybrid_config();
    runtime::TpCoordinator* tp_coord = impl.tp_coordinator_;
    std::vector<runtime::DFlash2GdnRankBinding> rank_bindings;
    if (tp_coord != nullptr) {
        rank_bindings.reserve(static_cast<std::size_t>(tp_coord->tp_size()));
        for (int r = 0; r < tp_coord->tp_size(); ++r) {
            auto& rank = tp_coord->rank(static_cast<std::size_t>(r));
            rank_bindings.push_back(runtime::DFlash2GdnRankBinding{
                &rank.gdn_pool(), &rank.arena(), rank.stream(), rank.device_id(),
                &rank.executor().host_model_state, &rank.executor()});
        }
    }

    if (impl.config_.dflash2_model_dir.empty()) {
        return Status::invalid_state("dflash2 model dir missing", __FILE__, __LINE__);
    }
    auto dflash_weights_result = runtime::dflash2::load_dflash2_weights(
        impl.config_.dflash2_model_dir, dflash_config, arena, stream, {});
    if (!dflash_weights_result.ok()) return dflash_weights_result.status();
    runtime::dflash2::DFlash2Weights dflash_weights = dflash_weights_result.release();

    const uint32_t max_seq = impl.config_.max_seq_len;
    const shelf::weights::MatrixWeight* private_lm_head = nullptr;
    const shelf::weights::MatrixWeight* private_embed = nullptr;
    auto* tp = impl.tp_coordinator_;
    const auto* tp_rank0_model =
        tp != nullptr && tp->tp_size() > 0 ? tp->rank(0).rank_model() : nullptr;
    const bool tp_sharded =
        impl.config_.tp > 1 ||
        (tp_rank0_model != nullptr && tp_rank0_model->lm_head() != nullptr &&
         tp_rank0_model->lm_head()->partition.has_value());
    if (tp_sharded) {
        if (tp == nullptr || tp->tp_size() == 0) {
            return Status::invalid_state("private target weights unavailable", __FILE__,
                                         __LINE__);
        }
        auto& rank0 = tp->rank(0);
        const auto* rank_model = rank0.rank_model();
        private_lm_head = rank_model != nullptr ? rank_model->replicated_lm_head() : nullptr;
        private_embed = rank_model != nullptr ? rank_model->replicated_embed_tokens() : nullptr;
        if (private_lm_head == nullptr) {
            private_lm_head = &rank0.private_lm_head();
            private_embed = &rank0.private_embed_tokens();
        }
    }
    std::unique_ptr<runtime::dflash2::DFlash2TpProposal> tp_proposal;
    shelf::kernel::EmbeddingShardTable embed_config_shards{};
    std::vector<const shelf::weights::MatrixWeight*> head_shards;
    uint32_t head_vocab_offset = 0;
    uint32_t head_local_vocab = 0;
    if (tp_sharded) {
        bool sharded = tp != nullptr;
        for (int r = 0; sharded && r < tp->tp_size(); ++r) {
            const auto* head = tp->rank(static_cast<std::size_t>(r)).rank_model()->lm_head();
            if (head == nullptr || !head->partition.has_value()) {
                sharded = false;
                break;
            }
            head_shards.push_back(head);
        }
        if (sharded) {
            std::vector<uint32_t> offsets;
            for (const auto* head : head_shards) {
                offsets.push_back(static_cast<uint32_t>(head->partition->index) *
                                  static_cast<uint32_t>(head->rows));
            }
            runtime::dflash2::DFlash2TpProposalConfig proposal_config;
            proposal_config.coordinator = tp;
            proposal_config.heads = head_shards;
            proposal_config.vocab_offsets = offsets;
            proposal_config.local_vocab = static_cast<uint32_t>(head_shards[0]->rows);
            proposal_config.top_k = static_cast<uint32_t>(dflash_config.selector_top_k);
            proposal_config.hidden_stride = static_cast<uint32_t>(dflash_config.hidden_size);
            proposal_config.max_rows = static_cast<uint32_t>(dflash_config.block_size);
            auto proposal_result =
                runtime::dflash2::DFlash2TpProposal::create(proposal_config);
            if (!proposal_result.ok()) return proposal_result.status();
            tp_proposal = std::make_unique<runtime::dflash2::DFlash2TpProposal>(
                proposal_result.release());
            head_vocab_offset = offsets[0];
            head_local_vocab = proposal_config.local_vocab;
            shelf::kernel::EmbeddingShardTable embed_shards{};
            bool embed_sharded = tp->tp_size() <=
                                 static_cast<int>(shelf::kernel::kEmbeddingMaxShards);
            for (int r = 0; embed_sharded && r < tp->tp_size(); ++r) {
                const auto* embed = tp->rank(static_cast<std::size_t>(r))
                                        .rank_model()
                                        ->embed_tokens();
                if (embed == nullptr || !embed->partition.has_value() ||
                    !embed->host_resident) {
                    embed_sharded = false;
                    break;
                }
                embed_shards.refs[static_cast<std::size_t>(r)] = shelf::kernel::EmbeddingShardRef{
                    static_cast<const shelf::bf16_t*>(embed->host_payload),
                    static_cast<uint32_t>(embed->partition->index) * embed->rows,
                    embed->rows};
            }
            if (embed_sharded && tp->tp_size() > 1) {
                embed_shards.count = static_cast<uint32_t>(tp->tp_size());
                embed_config_shards = embed_shards;
            }
        } else {
            return Status::unsupported(
                "dflash2 tp: target lm_head is not split across ranks, "
                "a full private copy is required",
                __FILE__, __LINE__);
        }
    }
    runtime::dflash2::DFlash2ExecutorConfig draft_config;
    draft_config.coarse_head_bytes = impl.config_.dflash2_coarse_bytes;
    draft_config.init_stream = stream;
    draft_config.max_prefill_feature_rows = impl.config_.dflash2_prefill_rows;
    draft_config.max_rows = static_cast<uint32_t>(dflash_config.block_size);
    draft_config.max_context_rows = static_cast<uint32_t>(dflash_config.sliding_window);
    draft_config.proposal_distributor = tp_proposal.get();
    draft_config.vocab_offset = head_vocab_offset;
    draft_config.local_vocab = head_local_vocab;
    draft_config.target_embed_shards = embed_config_shards;
    const shelf::weights::MatrixWeight* draft_embed = &model.embed_tokens();
    if (tp_sharded) {
        draft_embed = tp_proposal != nullptr
            ? tp->rank(0).rank_model()->embed_tokens()
            : private_embed;
    }
    draft_config.target_lm_head =
        tp_proposal != nullptr ? head_shards[0]
                               : (tp_sharded ? private_lm_head : &model.lm_head());
    draft_config.target_embed_tokens = draft_embed;
    auto draft_result = runtime::dflash2::create_dflash2_executor(
        dflash_config, dflash_weights, arena, draft_config);
    if (!draft_result.ok()) return draft_result.status();
    runtime::dflash2::DFlash2Executor draft = draft_result.release();

    auto sequence_result = shelf::state::create_paged_sequence_state(
        slot_pool, gdn_pool, kv_pool, max_seq, stream);
    if (!sequence_result.ok()) {
        (void)runtime::dflash2::dflash2_executor_shutdown(draft);
        return sequence_result.status();
    }
    shelf::state::PagedSequenceState sequence = sequence_result.release();
    auto release_sequence = [&]() {
        if (tp_coord != nullptr) {
            (void)tp_coord->on_sequence_released(sequence);
        }
        (void)shelf::state::release_paged_sequence_state(sequence, stream);
    };
    if (tp_coord != nullptr) {
        const shelf::Status mirror_st = tp_coord->on_sequence_created(sequence);
        if (!mirror_st.ok()) {
            release_sequence();
            (void)runtime::dflash2::dflash2_executor_shutdown(draft);
            return mirror_st;
        }
    }

    auto context_result =
        runtime::dflash2::create_dflash2_context_state(dflash_config, arena, 0u);
    if (!context_result.ok()) {
        release_sequence();
        (void)runtime::dflash2::dflash2_executor_shutdown(draft);
        return context_result.status();
    }
    runtime::dflash2::DFlash2ContextState context = context_result.release();

    runtime::DFlash2SpecDecoderConfig decoder_config;
    decoder_config.num_drafts = static_cast<uint32_t>(dflash2_drafts);
    decoder_config.tree = dflash2_tree;
    decoder_config.tree_extra = static_cast<uint32_t>(dflash2_tree_extra);
    decoder_config.tree_budget = static_cast<uint32_t>(dflash2_tree_budget);
    decoder_config.eos_tokens = text_config.stop_tokens;
    decoder_config.host_token_ids = impl.config_.dflash2_host_token_ids;
    auto decoder_result = runtime::create_dflash2_spec_decoder(
        executor, draft, context, sequence, gdn_pool, arena, decoder_config, stream,
        rank_bindings.empty() ? nullptr : &rank_bindings, tp_coord);
    if (!decoder_result.ok()) {
        (void)runtime::dflash2::dflash2_context_shutdown(context);
        release_sequence();
        (void)runtime::dflash2::dflash2_executor_shutdown(draft);
        return decoder_result.status();
    }
    runtime::DFlash2SpecDecoder decoder = decoder_result.release();
    runtime::DFlash2SpecTiming timing;
    decoder.timing = &timing;
    runtime::dflash2_spec_decoder_set_sampling(decoder, sampling);

    SpeculationResult result;
    uint32_t decode_tokens = 0u;

    const auto t_prefill_start = std::chrono::steady_clock::now();
    const uint32_t prompt_count = static_cast<uint32_t>(prompt_ids.size());
    const uint32_t tail_position =
        runtime::dflash2_prefill_tail_position(prompt_count, impl.config_.page_tokens);
    auto prefill_result = runtime::dflash2_spec_prefill(
        decoder, prompt_ids.data(), prompt_count, 0u, 0u, {}, tail_position);
    if (!prefill_result.ok()) {
        (void)runtime::dflash2_spec_decoder_shutdown(decoder);
        (void)runtime::dflash2::dflash2_context_shutdown(context);
        release_sequence();
        (void)runtime::dflash2::dflash2_executor_shutdown(draft);
        return prefill_result.status();
    }
    int32_t pending = prefill_result.value().pending_token;
    result.generated_ids.push_back(pending);
    const auto t_prefill_end = std::chrono::steady_clock::now();

    const auto t_decode_start = std::chrono::steady_clock::now();
    double accepted_ema = 2.0;
    uint32_t probe_age = 0;
    std::vector<uint64_t> adaptive_counts(decoder_config.num_drafts + 1u);
    uint64_t adaptive_rounds = 0;
    bool finished = shelf::qwen_hybrid::is_stop_token(decoder_config.eos_tokens, pending);
    while (!finished && result.generated_ids.size() < max_new_tokens) {
        const uint32_t remaining =
            static_cast<uint32_t>(max_new_tokens - result.generated_ids.size());
        if (adaptive) {
            ++adaptive_rounds;
            const bool enabled = accepted_ema >= 0.6;
            if (enabled) probe_age = 0;
            else if (probe_age < 64u) ++probe_age;
            const bool probe = !enabled && probe_age >= 64u;
            decoder.config.num_drafts = (enabled || probe) ? decoder_config.num_drafts : 0u;
            if (probe) probe_age = 0;
            ++adaptive_counts[decoder.config.num_drafts];
        }
        auto step_result =
            dflash2_tree
                ? runtime::dflash2_spec_step_tree(decoder, pending, remaining)
                : runtime::dflash2_spec_step(
                      decoder, pending, remaining, adaptive ? decoder_config.num_drafts : 0u);
        if (!step_result.ok()) {
            (void)runtime::dflash2_spec_decoder_shutdown(decoder);
            (void)runtime::dflash2::dflash2_context_shutdown(context);
            release_sequence();
            (void)runtime::dflash2::dflash2_executor_shutdown(draft);
            return step_result.status();
        }
        auto step = step_result.release();
        if (adaptive && step.num_drafts == decoder_config.num_drafts) {
            accepted_ema += 0.03 * (static_cast<double>(step.num_accepted) - accepted_ema);
        }
        for (uint32_t i = 0; i < step.emitted_count; ++i) {
            result.generated_ids.push_back(step.emitted[i]);
            ++decode_tokens;
        }
        pending = step.pending_token;
        finished = step.finished;
    }
    hipError_t sync_error = hipStreamSynchronize(stream);
    if (sync_error != hipSuccess) {
        (void)runtime::dflash2_spec_decoder_shutdown(decoder);
        (void)runtime::dflash2::dflash2_context_shutdown(context);
        release_sequence();
        (void)runtime::dflash2::dflash2_executor_shutdown(draft);
        return Status::hip_error("hipStreamSynchronize", hipGetErrorString(sync_error),
                                 __FILE__, __LINE__);
    }
    const auto t_decode_end = std::chrono::steady_clock::now();

    if (adaptive) {
        const double actual_ms = elapsed_ms(t_decode_start, t_decode_end);
        add_u(result, "DFLASH2_ADAPTIVE", adaptive_rounds);
        result.stats.pop_back();
        char adaptive[128];
        std::snprintf(adaptive, sizeof(adaptive),
                      "rounds=%llu actual_tps=%.3f",
                      static_cast<unsigned long long>(adaptive_rounds),
                      actual_ms > 0.0 ? 1000.0 * decode_tokens / actual_ms : 0.0);
        result.stats.push_back(SpeculationMetric{"DFLASH2_ADAPTIVE", adaptive});
        for (std::size_t k = 0; k < adaptive_counts.size(); ++k) {
            char value[64];
            std::snprintf(value, sizeof(value), "k=%zu count=%llu", k,
                          static_cast<unsigned long long>(adaptive_counts[k]));
            result.stats.push_back(SpeculationMetric{"DFLASH2_ADAPTIVE_K", value});
        }
    }
    const float mean_accepted = timing.rounds > 0u
        ? static_cast<float>(timing.accepted_drafts) / static_cast<float>(timing.rounds)
        : 0.0f;
    const float full_rate = timing.rounds > 0u
        ? static_cast<float>(timing.full_accepts) / static_cast<float>(timing.rounds)
        : 0.0f;
    const float emitted_per_round = timing.rounds > 0u
        ? static_cast<float>(timing.generated_tokens) / static_cast<float>(timing.rounds)
        : 0.0f;
    add_u(result, "DFLASH2_DRAFTS", decoder_config.num_drafts);
    add_u(result, "DFLASH2_BLOCK_SIZE", dflash_config.block_size);
    add_u(result, "DFLASH2_ROUNDS", timing.rounds);
    add_u(result, "DFLASH2_FULL_ACCEPTS", timing.full_accepts);
    add_u(result, "DFLASH2_PARTIAL_ACCEPTS", timing.partial_accepts);
    add_u(result, "DFLASH2_GDN_HISTORY_BYTES", timing.gdn_history_bytes);
    add_u(result, "DFLASH2_GDN_SNAPSHOT_BYTES", decoder.gdn_snapshot_bytes);
    add_u(result, "DFLASH2_GDN_COMPACT_BYTES", decoder.gdn_compact_bytes);
    result.stats.push_back(SpeculationMetric{
        "DFLASH2_GDN_MODE", dflash2_tree ? "tree-history" : "compact"});
    add_u(result, "DFLASH2_ACCEPTED_DRAFTS", timing.accepted_drafts);
    add_d(result, "DFLASH2_MEAN_ACCEPTED", mean_accepted, 3);
    add_d(result, "DFLASH2_FULL_ACCEPT_RATE", full_rate, 3);
    add_d(result, "DFLASH2_EMITTED_PER_ROUND", emitted_per_round, 3);
    add_d(result, "DFLASH2_DRAFT_MS", timing.draft_ms, 2);
    add_d(result, "DFLASH2_D2H_MS", timing.draft_d2h_ms, 2);
    add_d(result, "DFLASH2_DRAFT_GPU_MS", timing.draft_gpu_ms, 2);
    add_d(result, "DFLASH2_PROPOSAL_WAIT_MS", timing.proposal_wait_ms, 2);
    add_d(result, "DFLASH2_PROPOSAL_COPY_MS", timing.proposal_copy_ms, 2);
    add_d(result, "DFLASH2_VERIFY_MS", timing.verify_ms, 2);
    add_d(result, "DFLASH2_VERIFY_GPU_MS", timing.verify_gpu_ms, 2);
    add_d(result, "DFLASH2_DECISION_WAIT_MS", timing.decision_wait_ms, 2);
    add_d(result, "DFLASH2_DECISION_D2H_MS", timing.decision_copy_ms, 2);
    add_u(result, "DFLASH2_DEVICE_TOKEN_BRIDGE",
          decoder.device_token_bridge && !dflash2_tree ? 1 : 0);
    add_d(result, "DFLASH2_RERUN_MS", timing.rerun_ms, 2);
    add_d(result, "DFLASH2_TREE_SELECT_MS", timing.tree_select_ms, 2);
    if (timing.tree_probability_rows != 0u) {
        add_d(result, "DFLASH2_TREE_RESIDUAL_MASS",
              1.0 - timing.tree_retained_mass / timing.tree_probability_rows, 6);
    }
    add_d(result, "DFLASH2_TREE_WALK_MS", timing.tree_walk_ms, 2);
    add_d(result, "DFLASH2_TREE_COMMIT_MS", timing.tree_commit_ms, 2);
    add_d(result, "DFLASH2_TREE_RESTORE_MS", timing.tree_restore_ms, 2);
    add_d(result, "DFLASH2_TREE_GATHER_MS", timing.tree_gather_ms, 2);
    add_d(result, "DFLASH2_TREE_APPEND_MS", timing.tree_append_ms, 2);
    add_u(result, "DFLASH2_TREE_REPLAYS", timing.tree_replays);
    add_d(result, "DFLASH2_GDN_SNAPSHOT_MS", timing.gdn_snapshot_ms, 2);
    add_d(result, "DFLASH2_GDN_RESTORE_MS", timing.gdn_restore_ms, 2);
    add_d(result, "DFLASH2_COMMIT_MS", timing.dflash_commit_ms, 2);
    add_d(result, "DFLASH2_ROUND_MS", timing.round_ms, 2);

    (void)runtime::dflash2_spec_decoder_shutdown(decoder);
    (void)runtime::dflash2::dflash2_context_shutdown(context);
    release_sequence();
    (void)runtime::dflash2::dflash2_executor_shutdown(draft);

    result.decode_tokens = decode_tokens;
    result.prefill_ms = elapsed_ms(t_prefill_start, t_prefill_end);
    result.decode_ms = elapsed_ms(t_decode_start, t_decode_end);
    return result;
}

}  // namespace shelf::engine
