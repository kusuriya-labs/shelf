#include "compute_cli.h"
#include "single_request.h"
#include "stdio_server.h"

#include <shelf/models/qwen35/dflash2/config.h>
#include <shelf/models/plamo3/model/config.h>
#include <shelf/models/qwen35/model/rope_profile.h>
#include <shelf/models/qwen35/runtime/decode_backend.h>
#include <shelf/models/qwen35/runtime/dflash2_footprint.h>

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

    auto plamo_result = shelf::plamo3::is_plamo3_config(opt.model_dir);
    if (!plamo_result.ok()) { fprintf(stderr, "%s\n", plamo_result.status().message().c_str()); return 1; }
    const bool is_plamo = plamo_result.value();
    if (is_plamo && opt.rope_factor != "auto") {
        fprintf(stderr, "plamo3: --rope-factor override is not supported\n");
        return 1;
    }
    if (is_plamo && opt.max_seq_len > 2048) {
        fprintf(stderr, "plamo3: --max-seq-len must be at most 2048 (sliding attention limit)\n");
        return 1;
    }
    shelf::qwen35::EffectiveRopeProfile rope_profile;
    if (!is_plamo) {
        auto selection_result = shelf::qwen35::parse_rope_factor_selection(opt.rope_factor);
        if (!selection_result.ok()) {
            fprintf(stderr, "rope factor resolve failed: %s\n",
                    selection_result.status().message().c_str());
            return 1;
        }
        auto params_result = shelf::qwen35::read_qwen35_rope_parameters(opt.model_dir);
        if (!params_result.ok()) {
            fprintf(stderr, "rope config read failed: %s\n",
                    params_result.status().message().c_str());
            return 1;
        }
        auto profile_result = shelf::qwen35::resolve_effective_rope_profile(
            params_result.value(), selection_result.value());
        if (!profile_result.ok()) {
            fprintf(stderr, "rope profile resolve failed: %s\n",
                    profile_result.status().message().c_str());
            return 1;
        }
        rope_profile = profile_result.release();
        const uint32_t rope_cap = shelf::qwen35::rope_max_seq_len_cap(rope_profile);
        if (rope_cap != 0 && opt.max_seq_len > rope_cap) {
            fprintf(stderr,
                    "--max-seq-len %lu exceeds the rope factor limit %u "
                    "(--rope-factor %s, model original length %u)\n",
                    opt.max_seq_len, rope_cap, opt.rope_factor.c_str(),
                    rope_profile.original_max_position_embeddings);
            return 1;
        }
    }

    shelf::app::Qwen35RuntimeConfig config;
    config.model_dir = opt.model_dir;
    config.dflash2_model_dir = shelf::app::compute_dflash_enabled(opt) ? opt.dflash2_model_dir : std::string();
    config.mtp = opt.mtp;
    config.mtp_serve = opt.serve_stdio;
    config.mtp_drafts = static_cast<uint32_t>(opt.mtp_drafts);
    config.max_seq_len = static_cast<uint32_t>(opt.max_seq_len);
    config.max_concurrent_requests = static_cast<uint32_t>(opt.max_concurrent_requests);
    if (opt.arena_gib_auto) {
        config.arena_mode = shelf::app::ArenaBudgetMode::Auto;
        config.arena_bytes = 0u;
    } else if (opt.arena_gib < 0.0) {
        config.arena_mode = shelf::app::ArenaBudgetMode::TotalMinus;
        config.arena_bytes = static_cast<std::size_t>(
            -opt.arena_gib * 1024.0 * 1024.0 * 1024.0);
    } else {
        config.arena_mode = shelf::app::ArenaBudgetMode::Fixed;
        config.arena_bytes = static_cast<std::size_t>(
            opt.arena_gib * 1024.0 * 1024.0 * 1024.0);
    }
    config.page_tokens = static_cast<uint32_t>(opt.page_tokens);
    config.devices = opt.devices;
    config.tp = static_cast<uint32_t>(opt.tp);
    if (opt.mtp && opt.tp > 1) {
        config.private_target_weights = true;
        config.private_target_lm_head = true;
    }
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
    config.prefix_cache = opt.prefix_cache_enabled != 0 && opt.pipeline_parallel == 1;
    if (opt.pipeline_parallel > 1 && opt.prefix_cache_enabled != 0) {
        fprintf(stderr, "WARN: --pipeline-parallel disables prefix cache\n");
    }
    config.prefix_cache_concurrency = static_cast<uint32_t>(opt.prefix_cache_concurrency);
    config.working_set_reserve_bytes =
        static_cast<std::size_t>(opt.working_set_reserve_gib) * 1024ull * 1024ull * 1024ull;
    if (!is_plamo) config.rope_profile = rope_profile;
    if (!is_plamo) fprintf(stderr, "ROPE profile: type=%s factor=%g theta=%g rotary_dim=%u "
                    "original_max_position_embeddings=%u attention_factor=%.9f "
                    "beta_fast=%g beta_slow=%g truncate=%d source=%s\n",
            rope_profile.type == shelf::qwen35::EffectiveRopeProfile::Type::Yarn ? "yarn" : "default",
            rope_profile.factor, rope_profile.theta, rope_profile.rotary_dim,
            rope_profile.original_max_position_embeddings, rope_profile.attention_factor,
            rope_profile.beta_fast, rope_profile.beta_slow, rope_profile.truncate ? 1 : 0,
            rope_profile.from_cli ? "cli" : "config");
    shelf::qwen35::dflash2::DFlash2Config dflash_config;
    if (shelf::app::compute_dflash_enabled(opt)) {
        auto dflash_config_result =
            shelf::qwen35::dflash2::read_dflash2_config(opt.dflash2_model_dir);
        if (!dflash_config_result.ok()) {
            fprintf(stderr, "dflash2 config read failed: %s\n",
                    dflash_config_result.status().message().c_str());
            return 1;
        }
        dflash_config = dflash_config_result.release();
        if (opt.serve_stdio) {
            const char* value = std::getenv("SHELF_DFLASH2_PREFILL_ROWS");
            if (!shelf::qwen35::runtime::dflash2_prefill_rows(
                    dflash_config, config.tp, value, config.dflash2_prefill_rows)) {
                fprintf(stderr, "invalid SHELF_DFLASH2_PREFILL_ROWS: expected 8 or 256\n");
                return 2;
            }
            if (config.dflash2_prefill_rows == 0u) {
                fprintf(stderr, "DFLASH2 prefill feature rows: legacy\n");
            } else {
                fprintf(stderr, "DFLASH2 prefill feature rows: %u\n",
                        config.dflash2_prefill_rows);
            }
        }
        fprintf(stderr, "DFLASH2 rope: type=default theta=%g source=checkpoint\n",
                dflash_config.rope_theta);
        if (opt.dflash2_drafts > dflash_config.max_draft_tokens()) {
            fprintf(stderr, "--dflash2-drafts %lu exceeds block_size-1 %zu\n", opt.dflash2_drafts,
                    dflash_config.max_draft_tokens());
            return 2;
        }
        config.prefix_cache_spec_ring_bytes =
            static_cast<uint32_t>(dflash_config.context_ring_bytes());
        config.draft_bytes = shelf::app::safetensors_dir_bytes(opt.dflash2_model_dir);
        config.draft_config = dflash_config;
        config.draft_config_ready = true;
        config.dflash2_drafts = static_cast<uint32_t>(opt.dflash2_drafts);
        config.dflash2_max_sessions = static_cast<uint32_t>(opt.dflash2_max_sessions);
        if (opt.tp > 1) {
            config.private_target_weights = true;
            config.private_target_lm_head = true;
        }
    }

    std::vector<int32_t> prompt_ids;
    if (opt.serve_stdio) {
        config.max_num_batched_tokens = static_cast<uint32_t>(opt.max_num_batched_tokens);
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
                shelf::qwen35::runtime::dflash2_session_rows(
                    dflash_config.block_size, config.dflash2_drafts);
            if (config.max_num_batched_tokens < batch_rows) {
                config.max_num_batched_tokens = batch_rows;
            }
            uint32_t output_rows =
                batch_rows < shelf::qwen35::runtime::kDFlash2SpecMaxVerifyRows
                    ? batch_rows
                    : shelf::qwen35::runtime::kDFlash2SpecMaxVerifyRows;
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
                static_cast<unsigned long>(dflash_config.block_size);
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
            unsigned long rows = std::max<unsigned long>(prompt_ids.size() + 1UL, 16UL);
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
                if (config.target_hidden_tap_count >= shelf::qwen35::kMaxTargetHiddenTaps) {
                    fprintf(stderr, "--hidden-taps exceeds %u entries\n",
                            shelf::qwen35::kMaxTargetHiddenTaps);
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

    auto runtime_result = shelf::app::Qwen35ComputeRuntime::create(config);
    if (!runtime_result.ok()) {
        fprintf(stderr, "runtime create failed: %s\n",
                runtime_result.status().message().c_str());
        return 1;
    }
    std::unique_ptr<shelf::app::Qwen35ComputeRuntime> runtime = runtime_result.release();

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
           shelf::qwen35::runtime::decode_backend_name(
               runtime->config().decode_backend));
    if (opt.mtp) {
        return shelf::app::run_mtp_shot(*runtime, prompt_ids, opt.max_new_tokens,
                                        static_cast<uint32_t>(opt.mtp_drafts),
                                        opt.mtp_stats, opt.kv_cache_dtype, opt.sampling);
    }
    if (shelf::app::compute_dflash_enabled(opt)) {
        return shelf::app::run_dflash2_shot(*runtime, prompt_ids, opt.max_new_tokens, dflash_config,
                                opt.dflash2_model_dir, runtime->config().dflash2_drafts,
                                opt.dflash2_stats, opt.dflash2_adaptive, opt.kv_cache_dtype, opt.sampling);
    }
    return shelf::app::run_single_shot(*runtime, prompt_ids, opt.max_new_tokens,
                                    opt.sampling, opt.kv_cache_dtype,
                                    opt.dump_logits_path);
}
