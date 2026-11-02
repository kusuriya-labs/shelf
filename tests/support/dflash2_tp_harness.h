#pragma once

#include "support/tp_run_harness.h"

#include <shelf/models/qwen35/dflash2/config.h>
#include <shelf/models/qwen35/dflash2/context_state.h>
#include <shelf/models/qwen35/dflash2/executor.h>
#include <shelf/models/qwen35/dflash2/weights.h>
#include <shelf/models/qwen35/runtime/dflash2_spec_decoder.h>
#include <shelf/models/qwen35/state/state_types.h>
#include <shelf/models/qwen35/model/gdn_state_layout.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>

namespace tp_harness {

struct DFlash2Options {
    RunOptions base;
    std::string dflash2_model_dir;
    std::uint32_t drafts = 7;
};

inline bool dflash2_step_loop(::ps::qwen35::runtime::DFlash2SpecDecoder& decoder,
                              const ::ps::qwen35::Qwen35TextConfig& text_config,
                              const RunOptions& opts, RunResult& result,
                              std::string& loop_report) {
    const auto t0 = std::chrono::steady_clock::now();
    auto prefill = ::ps::qwen35::runtime::dflash2_spec_prefill(
        decoder, opts.prompt.data(), static_cast<std::uint32_t>(opts.prompt.size()));
    const auto t_pre = std::chrono::steady_clock::now();
    if (!prefill.ok()) {
        result.error = "dflash2 prefill: " + prefill.status().message();
        return false;
    }
    const double prefill_ms =
        std::chrono::duration<double, std::milli>(t_pre - t0).count();
    const double prefill_tokens = static_cast<double>(opts.prompt.size());
    std::int32_t pending = prefill.value().pending_token;
    result.tokens.push_back(pending);
    bool finished = ::ps::qwen35::is_stop_token(text_config.stop_tokens, pending);
    while (!finished && result.tokens.size() < opts.max_new_tokens) {
        const std::uint32_t remaining =
            static_cast<std::uint32_t>(opts.max_new_tokens - result.tokens.size());
        auto step = ::ps::qwen35::runtime::dflash2_spec_step(decoder, pending, remaining);
        if (!step.ok()) {
            result.error = "dflash2 step: " + step.status().message();
            return false;
        }
        auto out = step.release();
        for (std::uint32_t i = 0; i < out.emitted_count; ++i) {
            result.tokens.push_back(out.emitted[i]);
        }
        pending = out.pending_token;
        finished = out.finished;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double total_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double decode_ms = total_ms - prefill_ms;
    const std::size_t n = result.tokens.size();
    const double generated = static_cast<double>(n > 0 ? n - 1 : 0);
    char buf[384];
    std::snprintf(buf, sizeof(buf),
                  "pp_prefill_ms=%.1f pp_tokens=%.0f pp_tokens_per_s=%.2f\n"
                  "tg_decode_ms=%.1f tg_tokens=%.0f tg_tokens_per_s=%.2f\n"
                  "dflash2_prefill_plus_decode_ms=%.1f dflash2_tokens=%zu "
                  "dflash2_tok_per_s=%.2f\n",
                  prefill_ms, prefill_tokens,
                  prefill_ms > 0.0 ? prefill_tokens * 1000.0 / prefill_ms : 0.0,
                  decode_ms, generated,
                  decode_ms > 0.0 ? generated * 1000.0 / decode_ms : 0.0,
                  total_ms, n, total_ms > 0.0 ? static_cast<double>(n) * 1000.0 / total_ms : 0.0);
    loop_report = buf;
    return true;
}

inline RunResult run_dflash2_tp1(const DFlash2Options& opts) {
    const RunOptions& o = opts.base;
    RunResult result;
    const int tp1_device = tp_harness::tp_devices_from_env()[0];
    if (hipSetDevice(tp1_device) != hipSuccess) {
        result.error = "dflash2 tp1 hipSetDevice";
        return result;
    }
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        result.error = "dflash2 tp1 stream create";
        return result;
    }
    Qwen35LoadOptions load_options;
    load_options.verify_quantized_payload_crc = o.verify_crc;

    auto arena_result = gpu::GpuArena::create(tp1_device, o.arena_bytes);
    if (!arena_result.ok()) {
        result.error = "dflash2 tp1 arena: " + arena_result.status().message();
        return result;
    }
    gpu::GpuArena arena = arena_result.release();

    auto model_result = Qwen35Model::load_from_safetensors(o.model_dir, arena, stream,
                                                           load_options);
    if (!model_result.ok()) {
        result.error = "dflash2 tp1 model: " + model_result.status().message();
        return result;
    }
    Qwen35Model model = model_result.release();

    const std::uint32_t max_blocks =
        ceil_div_u32(o.max_seq_len + 1u, o.page_tokens) + 1u;
    auto seq_pool_result =
        SequenceSlotPool::create(arena, o.max_scheduled_requests, max_blocks);
    if (!seq_pool_result.ok()) {
        result.error = "dflash2 tp1 seq pool: " + seq_pool_result.status().message();
        return result;
    }
    SequenceSlotPool seq_pool = seq_pool_result.release();

    auto gdn_layout = ps::qwen35::make_qwen35_gdn_state_layout(model.text_config());
    auto gdn_result = GdnStatePool::create(arena, o.max_scheduled_requests, gdn_layout);
    if (!gdn_result.ok()) {
        result.error = "dflash2 tp1 gdn pool: " + gdn_result.status().message();
        return result;
    }
    GdnStatePool gdn_pool = gdn_result.release();

    auto kv_result = PagedKVPool::create(
        arena, ceil_div_u32(o.max_seq_len + 1u, o.page_tokens) + 1u, o.page_tokens,
        count_attention_layers(model.text_config()),
        static_cast<std::uint32_t>(model.text_config().num_key_value_heads),
        static_cast<std::uint32_t>(model.text_config().attention_head_dim),
        KVCacheDType::BF16);
    if (!kv_result.ok()) {
        result.error = "dflash2 tp1 kv pool: " + kv_result.status().message();
        return result;
    }
    PagedKVPool kv_pool = kv_result.release();

    auto dflash_config_result = ::ps::qwen35::dflash2::read_dflash2_config(
        opts.dflash2_model_dir);
    if (!dflash_config_result.ok()) {
        result.error = "dflash2 config: " + dflash_config_result.status().message();
        return result;
    }
    const auto dflash_config = dflash_config_result.release();

    ExecutorConfig exec_config;
    exec_config.max_scheduled_tokens = std::max(o.max_scheduled_tokens,
                                                static_cast<std::uint32_t>(
                                                    dflash_config.block_size));
    exec_config.max_scheduled_requests = o.max_scheduled_requests;
    exec_config.max_scheduled_output_rows =
        static_cast<std::uint32_t>(dflash_config.block_size);
    for (std::size_t i = 0; i < o.taps.size() && i < kMaxTargetHiddenTaps; ++i) {
        exec_config.target_hidden_taps[i] = static_cast<std::uint32_t>(o.taps[i]);
    }
    exec_config.target_hidden_tap_count = static_cast<std::uint32_t>(
        std::min<std::size_t>(o.taps.size(), static_cast<std::size_t>(kMaxTargetHiddenTaps)));
    auto exec_result =
        create_model_executor(model, seq_pool, gdn_pool, kv_pool, arena, exec_config);
    if (!exec_result.ok()) {
        result.error = "dflash2 tp1 executor: " + exec_result.status().message();
        return result;
    }
    Executor executor = exec_result.release();

    auto weights_result = ::ps::qwen35::dflash2::load_dflash2_weights(
        opts.dflash2_model_dir, dflash_config, arena, stream);
    if (!weights_result.ok()) {
        result.error = "dflash2 weights: " + weights_result.status().message();
        return result;
    }
    auto dflash_weights = weights_result.release();

    ::ps::qwen35::dflash2::DFlash2ExecutorConfig draft_config;
    draft_config.max_rows = static_cast<std::uint32_t>(dflash_config.block_size);
    draft_config.max_context_rows = static_cast<std::uint32_t>(dflash_config.sliding_window);
    draft_config.target_lm_head = &model.weights().lm_head;
    draft_config.target_embed_tokens = &model.weights().embed_tokens;
    auto draft_result = ::ps::qwen35::dflash2::create_dflash2_executor(
        dflash_config, dflash_weights, arena, draft_config);
    if (!draft_result.ok()) {
        result.error = "dflash2 draft executor: " + draft_result.status().message();
        return result;
    }
    auto draft = draft_result.release();

    auto sequence_result = ::ps::state::create_paged_sequence_state(
        seq_pool, gdn_pool, kv_pool, o.max_seq_len, stream);
    if (!sequence_result.ok()) {
        result.error = "dflash2 sequence: " + sequence_result.status().message();
        return result;
    }
    auto sequence = sequence_result.release();

    auto context_result = ::ps::qwen35::dflash2::create_dflash2_context_state(
        dflash_config, arena, 0u);
    if (!context_result.ok()) {
        result.error = "dflash2 context: " + context_result.status().message();
        return result;
    }
    auto context = context_result.release();

    ::ps::qwen35::runtime::DFlash2SpecDecoderConfig decoder_config;
    decoder_config.num_drafts = opts.drafts;
    decoder_config.eos_tokens = model.text_config().stop_tokens;
    auto decoder_result = ::ps::qwen35::runtime::create_dflash2_spec_decoder(
        executor, draft, context, sequence, gdn_pool, arena, decoder_config, stream);
    if (!decoder_result.ok()) {
        result.error = "dflash2 decoder: " + decoder_result.status().message();
        return result;
    }
    auto decoder = decoder_result.release();
    ::ps::qwen35::runtime::DFlash2SpecTiming timing;
    decoder.timing = &timing;

    std::string loop_report;
    const bool ran =
        dflash2_step_loop(decoder, model.text_config(), o, result, loop_report);
    if (ran) {
        result.report = "dflash2_mode=tp1 rounds=" + std::to_string(timing.rounds) +
                        " accepted=" + std::to_string(timing.accepted_drafts) + "\n" +
                        loop_report;
    }

    (void)::ps::qwen35::runtime::dflash2_spec_decoder_shutdown(decoder);
    (void)::ps::qwen35::dflash2::dflash2_context_shutdown(context);
    (void)::ps::state::release_paged_sequence_state(sequence, stream);
    (void)::ps::qwen35::dflash2::dflash2_executor_shutdown(draft);
    executor_shutdown(executor);
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    arena.shutdown();
    result.ok = ran;
    return result;
}

inline RunResult run_dflash2_tp2(const DFlash2Options& opts) {
    const RunOptions& o = opts.base;
    RunResult result;

    auto dflash_config_result = ::ps::qwen35::dflash2::read_dflash2_config(
        opts.dflash2_model_dir);
    if (!dflash_config_result.ok()) {
        result.error = "dflash2 config: " + dflash_config_result.status().message();
        return result;
    }
    const auto dflash_config = dflash_config_result.release();

    TpCoordinatorConfig config;
    config.model_dir = o.model_dir;
    config.devices = tp_harness::tp_devices_from_env();
    config.max_scheduled_tokens =
        std::max(o.max_scheduled_tokens,
                 static_cast<std::uint32_t>(dflash_config.block_size));
    config.max_scheduled_requests = o.max_scheduled_requests;
    config.max_scheduled_output_rows =
        static_cast<std::uint32_t>(dflash_config.block_size);
    config.max_seq_len = o.max_seq_len;
    config.page_tokens = o.page_tokens;
    config.arena_bytes = o.arena_bytes;
    for (std::size_t i = 0; i < o.taps.size() && i < kMaxTargetHiddenTaps; ++i) {
        config.target_hidden_taps[i] = static_cast<std::uint32_t>(o.taps[i]);
    }
    config.target_hidden_tap_count = static_cast<std::uint32_t>(
        std::min<std::size_t>(o.taps.size(), static_cast<std::size_t>(kMaxTargetHiddenTaps)));
    config.load_options.verify_quantized_payload_crc = o.verify_crc;
    config.load_options.vocab_parallel_lm_head = false;

    auto coordinator_result = TpCoordinator::create(config);
    if (!coordinator_result.ok()) {
        result.error = "dflash2 tp2 coordinator: " + coordinator_result.status().message();
        return result;
    }
    std::unique_ptr<TpCoordinator> coordinator = coordinator_result.release();
    TpRankRuntime& rank0 = coordinator->rank(0);

    const auto& model = rank0.model();
    const hipStream_t stream = rank0.stream();
    gpu::GpuArena& arena = rank0.arena();

    auto weights_result = ::ps::qwen35::dflash2::load_dflash2_weights(
        opts.dflash2_model_dir, dflash_config, arena, stream);
    if (!weights_result.ok()) {
        result.error = "dflash2 weights: " + weights_result.status().message();
        return result;
    }
    auto dflash_weights = weights_result.release();

    ::ps::qwen35::dflash2::DFlash2ExecutorConfig draft_config;
    draft_config.max_rows = static_cast<std::uint32_t>(dflash_config.block_size);
    draft_config.max_context_rows = static_cast<std::uint32_t>(dflash_config.sliding_window);
    draft_config.target_lm_head = &model.weights().lm_head;
    draft_config.target_embed_tokens = &model.weights().embed_tokens;
    auto draft_result = ::ps::qwen35::dflash2::create_dflash2_executor(
        dflash_config, dflash_weights, arena, draft_config);
    if (!draft_result.ok()) {
        result.error = "dflash2 draft executor: " + draft_result.status().message();
        return result;
    }
    auto draft = draft_result.release();

    auto sequence_result = ::ps::state::create_paged_sequence_state(
        rank0.sequence_pool(), rank0.gdn_pool(), rank0.kv_pool(), o.max_seq_len, stream);
    if (!sequence_result.ok()) {
        result.error = "dflash2 sequence: " + sequence_result.status().message();
        return result;
    }
    auto sequence = sequence_result.release();
    const ::ps::Status mirror_st = coordinator->on_sequence_created(sequence);
    if (!mirror_st.ok()) {
        result.error = "dflash2 tp2 mirror: " + mirror_st.message();
        (void)::ps::state::release_paged_sequence_state(sequence, stream);
        return result;
    }

    auto context_result = ::ps::qwen35::dflash2::create_dflash2_context_state(
        dflash_config, arena, 0u);
    if (!context_result.ok()) {
        result.error = "dflash2 context: " + context_result.status().message();
        return result;
    }
    auto context = context_result.release();

    std::vector<::ps::qwen35::runtime::DFlash2GdnRankBinding> bindings;
    for (std::size_t r = 0; r < 2; ++r) {
        TpRankRuntime& rank = coordinator->rank(r);
        bindings.push_back(::ps::qwen35::runtime::DFlash2GdnRankBinding{
            &rank.gdn_pool(), &rank.arena(), rank.stream(), rank.device_id(), nullptr,
            &rank.executor()});
    }

    ::ps::qwen35::runtime::DFlash2SpecDecoderConfig decoder_config;
    decoder_config.num_drafts = opts.drafts;
    decoder_config.eos_tokens = model.text_config().stop_tokens;
    auto decoder_result = ::ps::qwen35::runtime::create_dflash2_spec_decoder(
        rank0.executor(), draft, context, sequence, rank0.gdn_pool(), arena,
        decoder_config, stream, &bindings, coordinator.get());
    if (!decoder_result.ok()) {
        result.error = "dflash2 decoder: " + decoder_result.status().message();
        return result;
    }
    auto decoder = decoder_result.release();
    ::ps::qwen35::runtime::DFlash2SpecTiming timing;
    decoder.timing = &timing;

    std::string loop_report;
    const bool ran =
        dflash2_step_loop(decoder, model.text_config(), o, result, loop_report);
    if (ran) {
        result.report =
            coordinator->debug_report() + "dflash2_mode=tp2 rounds=" +
            std::to_string(timing.rounds) +
            " accepted=" + std::to_string(timing.accepted_drafts) + "\n" +
            loop_report;
    }

    (void)::ps::qwen35::runtime::dflash2_spec_decoder_shutdown(decoder);
    (void)::ps::qwen35::dflash2::dflash2_context_shutdown(context);
    (void)coordinator->on_sequence_released(sequence);
    (void)::ps::state::release_paged_sequence_state(sequence, stream);
    (void)::ps::qwen35::dflash2::dflash2_executor_shutdown(draft);
    result.ok = ran;
    return result;
}

}
