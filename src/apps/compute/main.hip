#include "compute_cli.h"
#include "single_request.h"
#include "stdio_server.h"

#include <phaseshift/models/model_source.h>
#include <phaseshift/models/qwen35/dflash2/config.h>
#include <phaseshift/models/qwen35/eagle4/config.h>
#include <phaseshift/models/qwen35/model/rope_profile.h>
#include <phaseshift/models/qwen35/runtime/decode_backend.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    ps::app::ComputeOptions opt;
    const int parse_rc = ps::app::parse_compute_options(argc, argv, opt);
    if (parse_rc == ps::app::kComputeParseHelp) {
        return 0;
    }
    if (parse_rc != ps::app::kComputeParseOk) return parse_rc;
    const int validate_rc = ps::app::validate_compute_options(opt);
    if (validate_rc != ps::app::kComputeParseOk) return validate_rc;

    ps::qwen35::EffectiveRopeProfile rope_profile;
    {
        auto selection_result = ps::qwen35::parse_rope_factor_selection(opt.rope_factor);
        if (!selection_result.ok()) {
            fprintf(stderr, "rope factor resolve failed: %s\n",
                    selection_result.status().message().c_str());
            return 1;
        }
        auto params_result = ps::qwen35::read_qwen35_rope_parameters(opt.model_dir);
        if (!params_result.ok()) {
            fprintf(stderr, "rope config read failed: %s\n",
                    params_result.status().message().c_str());
            return 1;
        }
        auto profile_result = ps::qwen35::resolve_effective_rope_profile(
            params_result.value(), selection_result.value());
        if (!profile_result.ok()) {
            fprintf(stderr, "rope profile resolve failed: %s\n",
                    profile_result.status().message().c_str());
            return 1;
        }
        rope_profile = profile_result.release();
        const uint32_t rope_cap = ps::qwen35::rope_max_seq_len_cap(rope_profile);
        if (rope_cap != 0 && opt.max_seq_len > rope_cap) {
            fprintf(stderr,
                    "--max-seq-len %lu exceeds the rope factor limit %u "
                    "(--rope-factor %s, model original length %u)\n",
                    opt.max_seq_len, rope_cap, opt.rope_factor.c_str(),
                    rope_profile.original_max_position_embeddings);
            return 1;
        }
        const bool yarn_active =
            rope_profile.type == ps::qwen35::EffectiveRopeProfile::Type::Yarn;
        if (yarn_active && (ps::app::compute_dflash_enabled(opt) ||
                            ps::app::compute_eagle4_enabled(opt))) {
            fprintf(stderr,
                    "static YaRN (--rope-factor %g) does not support speculative decoding "
                    "yet; drop --dflash2-model-dir / --eagle4-model-dir\n",
                    rope_profile.factor);
            return 1;
        }
    }

    ps::app::Qwen35RuntimeConfig config;
    config.model_dir = opt.model_dir;
    config.dflash2_model_dir = ps::app::compute_dflash_enabled(opt) ? opt.dflash2_model_dir : std::string();
    config.model_host_socket = opt.model_host.empty()
        ? ps::models::model_host_socket_from_env()
        : std::optional<std::string>(opt.model_host);
    config.max_seq_len = static_cast<uint32_t>(opt.max_seq_len);
    config.max_concurrent_requests = static_cast<uint32_t>(opt.max_concurrent_requests);
    config.kv_cache_capacity_tokens = static_cast<uint32_t>(opt.kv_cache_capacity_tokens);
    config.arena_bytes = opt.arena_gib_auto
        ? 0u
        : static_cast<std::size_t>(opt.arena_gib) * 1024ull * 1024ull * 1024ull;
    config.page_tokens = static_cast<uint32_t>(opt.page_tokens);
    config.devices = opt.devices;
    config.tp = static_cast<uint32_t>(opt.tp);
    config.kv_cache_dtype = opt.kv_cache_dtype == "fp8"
        ? ps::kernel::KVCacheDType::FP8_E4M3
        : ps::kernel::KVCacheDType::BF16;
    config.verify_weights = opt.verify_weights != 0;
    config.tp_combine_f32 = static_cast<uint32_t>(opt.tp_combine_f32);
    config.decode_backend = opt.decode_backend;
    config.prefix_cache = opt.prefix_cache_enabled != 0;
    config.prefix_cache_hot_tokens = static_cast<uint32_t>(opt.prefix_cache_hot_tokens);
    config.prefix_cache_cold_tokens = static_cast<uint32_t>(opt.prefix_cache_cold_tokens);
    config.prefix_cache_concurrency = static_cast<uint32_t>(opt.prefix_cache_concurrency);
    config.prefix_cache_cold_state_slots =
        static_cast<uint32_t>(opt.prefix_cache_cold_state_slots);
    config.working_set_reserve_bytes =
        static_cast<std::size_t>(opt.working_set_reserve_gib) * 1024ull * 1024ull * 1024ull;
    config.rope_profile = rope_profile;
    fprintf(stderr, "ROPE profile: type=%s factor=%g theta=%g rotary_dim=%u "
                    "original_max_position_embeddings=%u attention_factor=%.9f "
                    "beta_fast=%g beta_slow=%g truncate=%d source=%s\n",
            rope_profile.type == ps::qwen35::EffectiveRopeProfile::Type::Yarn ? "yarn" : "default",
            rope_profile.factor, rope_profile.theta, rope_profile.rotary_dim,
            rope_profile.original_max_position_embeddings, rope_profile.attention_factor,
            rope_profile.beta_fast, rope_profile.beta_slow, rope_profile.truncate ? 1 : 0,
            rope_profile.from_cli ? "cli" : "config");
    ps::qwen35::dflash2::DFlash2Config dflash_config;
    if (ps::app::compute_dflash_enabled(opt)) {
        auto dflash_config_result =
            ps::qwen35::dflash2::read_dflash2_config(opt.dflash2_model_dir);
        if (!dflash_config_result.ok()) {
            fprintf(stderr, "dflash2 config read failed: %s\n",
                    dflash_config_result.status().message().c_str());
            return 1;
        }
        dflash_config = dflash_config_result.release();
        if (opt.dflash2_drafts > dflash_config.max_draft_tokens()) {
            fprintf(stderr, "--dflash2-drafts %lu exceeds block_size-1 %zu\n", opt.dflash2_drafts,
                    dflash_config.max_draft_tokens());
            return 2;
        }
        config.prefix_cache_spec_ring_bytes =
            static_cast<uint32_t>(dflash_config.context_ring_bytes());
        config.draft_bytes = ps::app::safetensors_dir_bytes(opt.dflash2_model_dir);
        if (opt.tp > 1) {
            config.private_target_weights = true;
            config.private_target_lm_head = true;
        }
    }

    ps::qwen35::eagle4::Eagle4Config eagle_config;
    if (ps::app::compute_eagle4_enabled(opt)) {
        auto eagle_config_result =
            ps::qwen35::eagle4::read_eagle4_config(opt.eagle4_model_dir);
        if (!eagle_config_result.ok()) {
            fprintf(stderr, "eagle4 config read failed: %s\n",
                    eagle_config_result.status().message().c_str());
            return 1;
        }
        eagle_config = eagle_config_result.release();
        config.prefix_cache_spec_ring_bytes =
            static_cast<uint32_t>(eagle_config.context_ring_bytes());
        const std::size_t dir_bytes = ps::app::safetensors_dir_bytes(opt.eagle4_model_dir);
        std::size_t shared_bytes =
            eagle_config.vocab_size * eagle_config.hidden_size * sizeof(uint16_t);
        if (opt.eagle4_lm_head == "target") {
            shared_bytes +=
                eagle_config.vocab_size * eagle_config.hidden_size * sizeof(uint16_t);
        }
        config.draft_bytes = dir_bytes > shared_bytes ? dir_bytes - shared_bytes : 0u;
        if (opt.tp > 1) {
            config.private_target_weights = true;
            config.private_target_lm_head = opt.eagle4_lm_head == "target";
        }
    }

    std::vector<int32_t> prompt_ids;
    if (opt.serve_stdio) {
        config.max_scheduled_tokens = config.max_seq_len;
        if (ps::app::compute_dflash_enabled(opt)) {
            config.max_scheduled_tokens = config.max_seq_len;
            const uint32_t batch_rows = config.max_concurrent_requests *
                static_cast<uint32_t>(dflash_config.block_size);
            config.max_scheduled_output_rows =
                batch_rows < ps::qwen35::runtime::kDFlash2SpecMaxVerifyRows
                    ? batch_rows
                    : ps::qwen35::runtime::kDFlash2SpecMaxVerifyRows;
            config.target_hidden_tap_count =
                static_cast<uint32_t>(dflash_config.num_target_layer_ids);
            for (std::size_t t = 0; t < dflash_config.num_target_layer_ids; ++t) {
                config.target_hidden_taps[t] =
                    static_cast<uint32_t>(dflash_config.target_layer_ids[t]);
            }
        } else if (ps::app::compute_eagle4_enabled(opt)) {
            config.max_scheduled_tokens = config.max_seq_len;
            const unsigned long eagle4_block = std::max<unsigned long>(
                opt.eagle4_drafts + 1UL, opt.eagle4_tree_width);
            const uint32_t batch_rows =
                config.max_concurrent_requests *
                static_cast<uint32_t>(eagle4_block);
            config.max_scheduled_output_rows =
                batch_rows < ps::qwen35::runtime::kDFlash2SpecMaxVerifyRows
                    ? batch_rows
                    : ps::qwen35::runtime::kDFlash2SpecMaxVerifyRows;
            config.target_hidden_tap_count =
                static_cast<uint32_t>(eagle_config.num_aux_layers);
            for (std::size_t t = 0; t < eagle_config.num_aux_layers; ++t) {
                config.target_hidden_taps[t] =
                    static_cast<uint32_t>(eagle_config.aux_layer_ids[t]);
            }
        }
    } else {
        if (!opt.input_ids_file.empty()) {
            prompt_ids = ps::app::read_input_ids(opt.input_ids_file);
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
        if (ps::app::compute_eagle4_enabled(opt)) {
            config.max_scheduled_tokens = static_cast<uint32_t>(
                std::min<unsigned long>(prompt_ids.size(), 2048UL));
            if (config.max_scheduled_tokens < 16u) config.max_scheduled_tokens = 16u;
            const unsigned long eagle4_tree_rows =
                std::max<unsigned long>(opt.eagle4_drafts + 1UL, opt.eagle4_tree_width);
            config.max_scheduled_output_rows =
                static_cast<uint32_t>(std::min<unsigned long>(eagle4_tree_rows, 16UL));
            config.target_hidden_tap_count =
                static_cast<uint32_t>(eagle_config.num_aux_layers);
            for (std::size_t t = 0; t < eagle_config.num_aux_layers; ++t) {
                config.target_hidden_taps[t] =
                    static_cast<uint32_t>(eagle_config.aux_layer_ids[t]);
            }
        } else if (ps::app::compute_dflash_enabled(opt)) {
            const unsigned long dflash_rows =
                static_cast<unsigned long>(dflash_config.block_size);
            unsigned long dflash_tokens =
                std::max<unsigned long>(prompt_ids.size(), dflash_rows);
            dflash_tokens = std::min<unsigned long>(dflash_tokens, 2048UL);
            config.max_scheduled_tokens = static_cast<uint32_t>(dflash_tokens);
            config.max_scheduled_output_rows = static_cast<uint32_t>(dflash_rows);
            config.target_hidden_tap_count =
                static_cast<uint32_t>(dflash_config.num_target_layer_ids);
            for (std::size_t t = 0; t < dflash_config.num_target_layer_ids; ++t) {
                config.target_hidden_taps[t] =
                    static_cast<uint32_t>(dflash_config.target_layer_ids[t]);
            }
        } else {
            config.max_scheduled_tokens = static_cast<uint32_t>(
                std::max<unsigned long>(prompt_ids.size() + 1UL, 16UL));
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
                if (config.target_hidden_tap_count >= ps::qwen35::kMaxTargetHiddenTaps) {
                    fprintf(stderr, "--hidden-taps exceeds %u entries\n",
                            ps::qwen35::kMaxTargetHiddenTaps);
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

    auto runtime_result = ps::app::Qwen35ComputeRuntime::create(config);
    if (!runtime_result.ok()) {
        fprintf(stderr, "runtime create failed: %s\n",
                runtime_result.status().message().c_str());
        return 1;
    }
    std::unique_ptr<ps::app::Qwen35ComputeRuntime> runtime = runtime_result.release();

    if (config.prefix_cache && !runtime->prefix_cache_active()) {
        fprintf(opt.serve_stdio ? stderr : stdout,
                "warning: prefix cache unavailable, continuing without prefix cache: %s\n",
                runtime->prefix_cache_error().c_str());
    }

    ps::app::DFlashServeContext dflash_serve;
    if (ps::app::compute_dflash_enabled(opt)) {
        dflash_serve.enabled = true;
        dflash_serve.config = dflash_config;
        dflash_serve.model_dir = opt.dflash2_model_dir;
        dflash_serve.drafts = static_cast<uint32_t>(opt.dflash2_drafts);
    } else if (ps::app::compute_eagle4_enabled(opt)) {
        dflash_serve.eagle4_enabled = true;
        dflash_serve.eagle4_model_dir = opt.eagle4_model_dir;
        dflash_serve.eagle4_drafts = static_cast<uint32_t>(opt.eagle4_drafts);
        dflash_serve.eagle4_candidate_head = opt.eagle4_candidate_head;
        dflash_serve.eagle4_lm_head = opt.eagle4_lm_head;
        dflash_serve.eagle4_config = eagle_config;
    }

    if (opt.serve_stdio) {
        return ps::app::run_serve_stdio(*runtime, dflash_serve);
    }
    printf("DECODE_BACKEND=%s\n",
           ps::qwen35::runtime::decode_backend_name(
               runtime->config().decode_backend));
    if (ps::app::compute_eagle4_enabled(opt)) {
        return ps::app::run_eagle4_shot(*runtime, prompt_ids, opt.max_new_tokens,
                                        opt.eagle4_model_dir, opt.eagle4_drafts,
                                        opt.eagle4_tree_width, opt.eagle4_tree_dynamic,
                                        opt.eagle4_candidate_head, opt.eagle4_lm_head,
                                        opt.eagle4_stats, opt.kv_cache_dtype);
    }
    if (ps::app::compute_dflash_enabled(opt)) {
        return ps::app::run_dflash2_shot(*runtime, prompt_ids, opt.max_new_tokens, dflash_config,
                                opt.dflash2_model_dir, opt.dflash2_drafts, opt.dflash2_stats,
                                opt.kv_cache_dtype, opt.sampling);
    }
    return ps::app::run_single_shot(*runtime, prompt_ids, opt.max_new_tokens,
                                    opt.sampling, opt.kv_cache_dtype,
                                    opt.dump_logits_path);
}
