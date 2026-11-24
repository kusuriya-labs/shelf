#include "arena_budget.h"
#include "compute_cli.h"
#include "serve_host_memory.h"
#include "single_request.h"
#include "stdio_server.h"

#include <shelf/engine/compute_session.h>
#include <shelf/models/model_policy.h>
#include <shelf/runtime/decode_backend.h>
#include <shelf/runtime/dflash2/dflash2_footprint.h>
#include <shelf/runtime/dflash2_config.h>
#include <shelf/runtime/dflash2_spec_decoder.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    shelf::app::ComputeOptions opt;
    const int parse_rc = shelf::app::parse_compute_options(argc, argv, opt);
    if (parse_rc == shelf::app::kComputeParseHelp) {
        return 0;
    }
    if (parse_rc != shelf::app::kComputeParseOk) return parse_rc;
    const int validate_rc = shelf::app::validate_compute_options(opt);
    if (validate_rc != shelf::app::kComputeParseOk) return validate_rc;

    if (opt.serve_stdio && !shelf::app::configure_serve_host_memory()) {
        std::perror("ROCr host memory configuration");
        return 1;
    }

    shelf::models::ModelRuntimeRequest requested;
    requested.model_dir = opt.model_dir;
    requested.rope_factor = opt.rope_factor;
    requested.max_seq_len = static_cast<uint32_t>(opt.max_seq_len);
    requested.tensor_parallel = static_cast<uint32_t>(opt.tp);
    requested.expert_parallel = static_cast<uint32_t>(opt.expert_parallel);
    requested.pipeline_parallel = static_cast<uint32_t>(opt.pipeline_parallel);
    requested.mtp = opt.mtp;
    requested.dflash2 = shelf::app::compute_dflash_enabled(opt);
    requested.dflash2_drafts = static_cast<uint32_t>(opt.dflash2_drafts);
    if (opt.mtp && opt.tp > 1) {
        requested.private_target_weights = true;
        requested.private_target_lm_head = opt.mtp;
    }
    if (shelf::app::compute_dflash_enabled(opt) && opt.tp > 1) {
        requested.private_target_weights = opt.mtp;
        requested.private_target_lm_head = opt.mtp;
    }
    auto resolved_result = shelf::models::resolve_model_options(requested);
    if (!resolved_result.ok()) {
        fprintf(stderr, "%s\n", resolved_result.status().message().c_str());
        return 1;
    }
    const shelf::models::ResolvedModelOptions resolved = resolved_result.release();

    shelf::engine::ComputeConfig config;
    config.model_dir = opt.model_dir;
    config.requested = requested;
    config.resolved = resolved;
    config.resolved_ready = true;
    config.dflash2_model_dir = shelf::app::compute_dflash_enabled(opt)
        ? opt.dflash2_model_dir : std::string();
    config.mtp = opt.mtp;
    config.mtp_serve = opt.serve_stdio;
    config.mtp_drafts = static_cast<uint32_t>(opt.mtp_drafts);
    config.mtp_window = static_cast<uint32_t>(opt.mtp_window);
    config.mtp_sink = static_cast<uint32_t>(opt.mtp_sink);
    config.max_seq_len = static_cast<uint32_t>(opt.max_seq_len);
    config.max_concurrent_requests = static_cast<uint32_t>(opt.max_concurrent_requests);
    config.prefill_decode_interval = static_cast<uint32_t>(opt.prefill_decode_interval);
    if (opt.arena_gib_auto) {
        config.arena_mode = shelf::engine::ArenaBudgetMode::Auto;
        config.arena_bytes = 0u;
    } else {
        config.arena_mode = opt.arena_gib < 0.0
            ? shelf::engine::ArenaBudgetMode::TotalMinus
            : shelf::engine::ArenaBudgetMode::Fixed;
        if (!shelf::app::arena_budget_bytes(opt.arena_gib, config.arena_bytes)) {
            fprintf(stderr, "bad --arena-gib: adjusted capacity is invalid\n");
            return 1;
        }
    }
    config.page_tokens = static_cast<uint32_t>(opt.page_tokens);
    config.devices = opt.devices;
    config.tp = static_cast<uint32_t>(opt.tp);
    config.expert_parallel = static_cast<uint32_t>(opt.expert_parallel);
    config.private_target_weights = resolved.private_target_weights;
    config.private_target_lm_head = resolved.private_target_lm_head;
    config.pipeline_parallel = static_cast<uint32_t>(opt.pipeline_parallel);
    if (shelf::kernel::parse_octave_kv_cache_format(
            opt.kv_cache_dtype, config.octave_kv_format)) {
        config.kv_cache_dtype = shelf::kernel::KVCacheDType::OCTAVE;
    } else {
        config.kv_cache_dtype = opt.kv_cache_dtype == "fp8"
            ? shelf::kernel::KVCacheDType::FP8_E4M3
            : shelf::kernel::KVCacheDType::BF16;
    }
    config.verify_weights = opt.verify_weights != 0;
    config.tp_combine_f32 = static_cast<uint32_t>(opt.tp_combine_f32);
    config.decode_backend = opt.decode_backend;
    if (opt.pipeline_parallel > 1 && opt.prefix_cache_enabled != 0) {
        fprintf(stderr, "error: --pipeline-parallel does not support prefix cache\n");
        return 1;
    }
    config.prefix_cache = opt.prefix_cache_enabled != 0 && opt.pipeline_parallel == 1;
    config.prefix_cache_checkpoints =
        static_cast<uint32_t>(opt.prefix_cache_checkpoints);
    config.prefix_cache_strategy = opt.prefix_cache_strategy == 1
        ? shelf::runtime::PrefixCacheStrategy::Slot
        : shelf::runtime::PrefixCacheStrategy::Radix;
    config.prefix_cache_slot_count = config.prefix_cache_strategy ==
            shelf::runtime::PrefixCacheStrategy::Slot
        ? config.max_concurrent_requests
        : 0u;
    config.prefix_cache_save_interval =
        static_cast<uint32_t>(opt.prefix_cache_save_interval);
    config.working_set_reserve_bytes =
        static_cast<std::size_t>(opt.working_set_reserve_gib) * 1024ull * 1024ull * 1024ull;
    if (resolved.has_rope_profile) config.rope_profile = resolved.rope_profile;
    if (resolved.has_rope_profile) {
        const auto& rope_profile = resolved.rope_profile;
        fprintf(stderr, "ROPE profile: type=%s factor=%g theta=%g rotary_dim=%u "
                        "original_max_position_embeddings=%u attention_factor=%.9f "
                        "beta_fast=%g beta_slow=%g truncate=%d source=%s\n",
                rope_profile.type == shelf::runtime::EffectiveRopeProfile::Type::Yarn
                    ? "yarn" : "default",
                rope_profile.factor, rope_profile.theta, rope_profile.rotary_dim,
                rope_profile.original_max_position_embeddings,
                rope_profile.attention_factor, rope_profile.beta_fast,
                rope_profile.beta_slow, rope_profile.truncate ? 1 : 0,
                rope_profile.from_cli ? "cli" : "config");
    }
    ::shelf::runtime::DFlash2Config dflash_config;
    if (shelf::app::compute_dflash_enabled(opt)) {
        auto dflash_config_result =
            ::shelf::runtime::read_dflash2_config(opt.dflash2_model_dir);
        if (!dflash_config_result.ok()) {
            fprintf(stderr, "dflash2 config read failed: %s\n",
                    dflash_config_result.status().message().c_str());
            return 1;
        }
        dflash_config = dflash_config_result.release();
        const char* value = std::getenv("SHELF_DFLASH2_PREFILL_ROWS");
        if (!shelf::runtime::dflash2_prefill_rows(
                dflash_config, config.tp, value, config.dflash2_prefill_rows)) {
            fprintf(stderr, "invalid SHELF_DFLASH2_PREFILL_ROWS: expected 8, 256, or 16 for block_size=16\n");
            return 2;
        }
        if (config.dflash2_prefill_rows == 0u) {
            fprintf(stderr, "DFLASH2 prefill feature rows: legacy\n");
        } else {
            fprintf(stderr, "DFLASH2 prefill feature rows: %u\n",
                    config.dflash2_prefill_rows);
        }
        fprintf(stderr, "DFLASH2 rope: type=default theta=%g source=checkpoint\n",
                dflash_config.rope_theta);
        if (opt.dflash2_drafts > dflash_config.max_draft_tokens()) {
            fprintf(stderr, "--dflash2-drafts %lu exceeds block_size-1 %zu\n", opt.dflash2_drafts,
                    dflash_config.max_draft_tokens());
            return 2;
        }
        if (opt.serve_stdio && config.prefix_cache &&
            config.prefix_cache_strategy == shelf::runtime::PrefixCacheStrategy::Radix &&
            config.max_seq_len > dflash_config.sliding_window) {
            dflash_config.prefix_tail_rows = std::min<std::size_t>(
                config.page_tokens - 1u, config.max_seq_len - dflash_config.sliding_window);
        }
        if (dflash_config.context_ring_bytes() > UINT32_MAX) {
            fprintf(stderr, "dflash2 ring exceeds checkpoint size limit\n");
            return 2;
        }
        config.prefix_cache_spec_ring_bytes =
            static_cast<uint32_t>(dflash_config.context_ring_bytes());
        config.draft_bytes = shelf::app::safetensors_dir_bytes(opt.dflash2_model_dir);
        config.draft_config = dflash_config;
        config.draft_config_ready = true;
        config.dflash2_drafts = resolved.dflash2_drafts;
        config.dflash2_tree = opt.dflash2_tree && !opt.serve_stdio;
        config.dflash2_tree_budget = static_cast<uint32_t>(opt.dflash2_tree_budget);
        config.dflash2_tree_budget_auto = opt.dflash2_tree_budget_auto;
        config.dflash2_tree_extra = opt.serve_stdio ? 0u : static_cast<uint32_t>(opt.dflash2_tree_budget ? opt.dflash2_tree_budget - opt.dflash2_drafts - 1 : opt.dflash2_tree_extra);
        config.dflash2_max_sessions = static_cast<uint32_t>(opt.dflash2_max_sessions);
    }

    std::vector<int32_t> prompt_ids;
    if (opt.serve_stdio) {
        config.max_num_batched_tokens = static_cast<uint32_t>(opt.max_num_batched_tokens);
        if (opt.prefill_chunk_set) {
            const uint32_t chunk = static_cast<uint32_t>(opt.prefill_chunk);
            if (config.max_num_batched_tokens > chunk) {
                config.max_num_batched_tokens = chunk;
            }
        }
        if (config.max_num_batched_tokens < config.max_concurrent_requests) {
            config.max_num_batched_tokens = config.max_concurrent_requests;
        }
        if (config.mtp) {
            const uint32_t verify_rows = static_cast<uint32_t>(opt.mtp_drafts) + 1u;
            const uint32_t batch_rows = config.max_concurrent_requests * verify_rows;
            if (config.max_num_batched_tokens < batch_rows) {
                config.max_num_batched_tokens = batch_rows;
            }
            config.max_scheduled_output_rows = batch_rows;
        }
        if (shelf::app::compute_dflash_enabled(opt)) {
            const uint32_t batch_rows = config.dflash2_sessions() *
                shelf::runtime::dflash2_session_rows(
                    dflash_config.block_size, config.dflash2_drafts, config.dflash2_tree_budget);
            if (config.max_num_batched_tokens < batch_rows) {
                config.max_num_batched_tokens = batch_rows;
            }
            uint32_t output_rows =
                batch_rows < shelf::runtime::kDFlash2SpecMaxVerifyRows
                    ? batch_rows
                    : shelf::runtime::kDFlash2SpecMaxVerifyRows;
            // Non-speculative requests still need one output row each.
            if (output_rows < config.max_concurrent_requests) {
                output_rows = config.max_concurrent_requests;
            }
            config.max_scheduled_output_rows = output_rows;
            config.target_hidden_tap_count =
                static_cast<uint32_t>(dflash_config.num_target_layer_ids);
            for (std::size_t t = 0; t < dflash_config.num_target_layer_ids; ++t) {
                config.target_hidden_taps[t] =
                    static_cast<uint32_t>(dflash_config.target_layer_ids[t]);
            }
        }
    } else {
        if (!opt.input_ids_file.empty()) {
            prompt_ids = shelf::app::read_input_ids(opt.input_ids_file);
            if (prompt_ids.empty()) {
                fprintf(stderr, "no prompt ids read from %s\n", opt.input_ids_file.c_str());
                return 1;
            }
        } else {
            prompt_ids = {248041, 77091};
        }
        if (static_cast<uint64_t>(prompt_ids.size()) + opt.max_new_tokens > opt.max_seq_len) {
            fprintf(stderr, "prompt + generation exceeds max_seq_len %lu\n", opt.max_seq_len);
            return 2;
        }
        if (shelf::app::compute_dflash_enabled(opt)) {
            const unsigned long dflash_rows =
                std::max(static_cast<unsigned long>(dflash_config.block_size) + opt.dflash2_tree_extra,
                         opt.dflash2_tree_budget);
            unsigned long dflash_tokens =
                std::max<unsigned long>(prompt_ids.size(), dflash_rows);
            dflash_tokens = std::min<unsigned long>(dflash_tokens, 2048UL);
            config.max_num_batched_tokens = static_cast<uint32_t>(dflash_tokens);
            config.max_scheduled_output_rows = static_cast<uint32_t>(dflash_rows);
            config.target_hidden_tap_count =
                static_cast<uint32_t>(dflash_config.num_target_layer_ids);
            for (std::size_t t = 0; t < dflash_config.num_target_layer_ids; ++t) {
                config.target_hidden_taps[t] =
                    static_cast<uint32_t>(dflash_config.target_layer_ids[t]);
            }
        } else {
            const unsigned long chunk = std::max<unsigned long>(opt.prefill_chunk, 16UL);
            unsigned long rows =
                std::min<unsigned long>(prompt_ids.size() + 1UL, chunk);
            rows = std::max<unsigned long>(rows, 16UL);
            if (opt.mtp) {
                rows = std::max<unsigned long>(rows, opt.mtp_drafts + 1UL);
            }
            config.max_num_batched_tokens = static_cast<uint32_t>(rows);
            config.max_scheduled_output_rows = opt.mtp
                ? static_cast<uint32_t>(opt.mtp_drafts) + 1u
                : 0u;
        }
    }

    if (!opt.hidden_taps.empty()) {
        config.target_hidden_tap_count = 0;
        std::size_t pos = 0;
        while (pos <= opt.hidden_taps.size()) {
            const std::size_t comma = opt.hidden_taps.find(',', pos);
            const std::string tok = opt.hidden_taps.substr(
                pos, comma == std::string::npos ? std::string::npos : comma - pos);
            if (!tok.empty()) {
                if (config.target_hidden_tap_count >= shelf::runtime::kMaxTargetHiddenTaps) {
                    fprintf(stderr, "--hidden-taps exceeds %u entries\n",
                            shelf::runtime::kMaxTargetHiddenTaps);
                    return 2;
                }
                char* end = nullptr;
                const unsigned long v = std::strtoul(tok.c_str(), &end, 10);
                if (end == tok.c_str() || *end != '\0') {
                    fprintf(stderr, "bad --hidden-taps entry: %s\n", tok.c_str());
                    return 2;
                }
                config.target_hidden_taps[config.target_hidden_tap_count++] =
                    static_cast<uint32_t>(v);
            }
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
    }

    auto runtime_result = shelf::engine::ComputeSession::create(config);
    if (!runtime_result.ok()) {
        fprintf(stderr, "runtime create failed: %s\n",
                runtime_result.status().message().c_str());
        return 1;
    }
    std::unique_ptr<shelf::engine::ComputeSession> runtime = runtime_result.release();

    if (config.prefix_cache && !runtime->prefix_cache_active()) {
        fprintf(opt.serve_stdio ? stderr : stdout,
                "warning: prefix cache unavailable, continuing without prefix cache: %s\n",
                runtime->prefix_cache_error().c_str());
    }

    shelf::app::DFlashServeContext dflash_serve;
    if (shelf::app::compute_dflash_enabled(opt)) {
        dflash_serve.enabled = true;
        dflash_serve.config = dflash_config;
        dflash_serve.model_dir = opt.dflash2_model_dir;
        dflash_serve.drafts = config.dflash2_drafts;
        dflash_serve.adaptive = opt.dflash2_adaptive;
    }

    if (opt.serve_stdio) {
        return shelf::app::run_serve_stdio(*runtime, dflash_serve);
    }
    printf("DECODE_BACKEND=%s\n",
           shelf::runtime::decode_backend_name(
               runtime->config().decode_backend));
    if (opt.mtp) {
        return shelf::app::run_mtp_shot(*runtime, prompt_ids, opt.max_new_tokens,
                                        static_cast<uint32_t>(opt.mtp_drafts),
                                        opt.mtp_stats, opt.kv_cache_dtype, opt.sampling);
    }
    if (shelf::app::compute_dflash_enabled(opt)) {
        return shelf::app::run_dflash2_shot(*runtime, prompt_ids, opt.max_new_tokens, dflash_config,
                                opt.dflash2_model_dir, runtime->config().dflash2_drafts,
                                opt.dflash2_stats, opt.dflash2_adaptive,
                                opt.dflash2_tree && opt.dflash2_tree_budget != opt.dflash2_drafts + 1,
                                opt.dflash2_tree_budget ? opt.dflash2_tree_budget - opt.dflash2_drafts - 1 : opt.dflash2_tree_extra,
                                opt.dflash2_tree_budget, opt.kv_cache_dtype, opt.sampling);
    }
    return shelf::app::run_single_shot(*runtime, prompt_ids, opt.max_new_tokens,
                                    opt.sampling, opt.kv_cache_dtype,
                                    opt.dump_logits_path);
}
