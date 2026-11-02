#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/models/qwen35/eagle4/config.h>
#include <phaseshift/models/qwen35/eagle4/context_state.h>
#include <phaseshift/models/qwen35/eagle4/executor.h>
#include <phaseshift/models/qwen35/eagle4/weights.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/models/qwen35/runtime/continuous_batcher.h>
#include <phaseshift/models/qwen35/runtime/dflash2_spec_decoder.h>
#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/models/qwen35/runtime/prefix_cache.h>
#include <phaseshift/models/qwen35/state/state_types.h>
#include <phaseshift/models/qwen35/model/gdn_state_layout.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

#include "support/eagle4_checkpoint.h"
#include "support/qwen35_tp_fixture.h"

namespace fs = std::filesystem;
using namespace ps;
using namespace eagle4_ckpt;

using ps::qwen35::Executor;
using ps::qwen35::ExecutorConfig;
using ps::state::GdnStatePool;
using ps::state::GdnStatePoolLayout;
using ps::kernel::KVCacheDType;
using ps::state::PagedKVPool;
using ps::qwen35::Qwen35LoadOptions;
using ps::qwen35::Qwen35Model;
using ps::qwen35::Qwen35TextConfig;
using ps::state::SequenceSlotPool;
using ps::qwen35::create_model_executor;
using ps::state::create_paged_sequence_state;
using ps::state::release_paged_sequence_state;
using ps::qwen35::executor_shutdown;
using ps::qwen35::eagle4::Eagle4Config;
using ps::qwen35::eagle4::Eagle4Executor;
using ps::qwen35::eagle4::Eagle4ExecutorConfig;
using ps::qwen35::eagle4::Eagle4WeightLoadOptions;
using ps::qwen35::eagle4::Eagle4Weights;
using ps::qwen35::eagle4::create_eagle4_context_state;
using ps::qwen35::eagle4::create_eagle4_executor;
using ps::qwen35::eagle4::eagle4_context_shutdown;
using ps::qwen35::eagle4::eagle4_executor_shutdown;
using ps::qwen35::eagle4::load_eagle4_weights;
using ps::qwen35::runtime::ContinuousBatcher;
using ps::qwen35::runtime::ContinuousBatcherConfig;
using ps::qwen35::runtime::DFlash2BatcherBinding;
using ps::qwen35::runtime::DFlash2SessionPool;
using ps::qwen35::runtime::DFlash2SpecBatchContext;
using ps::qwen35::runtime::DFlash2SpecDecoder;
using ps::qwen35::runtime::DFlash2SpecDecoderConfig;
using ps::qwen35::runtime::PrefixCache;
using ps::qwen35::runtime::PrefixCacheConfig;
using ps::qwen35::runtime::PrefixCacheStats;
using ps::qwen35::runtime::RuntimeRequest;
using ps::qwen35::runtime::SamplingConfig;
using ps::qwen35::runtime::create_dflash2_spec_batch_context;
using ps::qwen35::runtime::dflash2_spec_decoder_shutdown;
using ps::qwen35::runtime::dflash2_spec_prefill;
using ps::qwen35::runtime::dflash2_spec_step;

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

static uint32_t ceil_div_u32(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}

static uint32_t count_attention_layers(const Qwen35TextConfig& tc) {
    uint32_t n = 0;
    for (int lt : tc.layer_types) {
        if (lt == 1) ++n;
    }
    return n;
}

static std::vector<int32_t> make_prompt(uint32_t seed, uint32_t len) {
    std::vector<int32_t> out;
    out.reserve(len);
    uint32_t s = seed;
    for (uint32_t i = 0; i < len; ++i) {
        s = s * 1664525u + 1013904223u;
        out.push_back(static_cast<int32_t>((s >> 16) % 100u));
    }
    return out;
}

static bool same_tokens(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

static void print_tokens(const char* tag, const std::vector<int32_t>& tokens) {
    std::printf("%s:", tag);
    for (int32_t t : tokens) std::printf(" %d", t);
    std::printf("\n");
}

struct RunOutcome {
    bool ok = false;
    std::string error;
    std::vector<int32_t> tokens;
};

static void run_concurrent(ContinuousBatcher& batcher,
                            const std::vector<int32_t>& prompt_a,
                            const std::vector<int32_t>& prompt_b,
                            uint32_t max_new_tokens,
                            RunOutcome& out_a,
                            RunOutcome& out_b) {
    out_a = RunOutcome{};
    out_b = RunOutcome{};
    SamplingConfig sampling;
    auto id_a = batcher.submit(prompt_a, max_new_tokens, sampling, true);
    if (!id_a.ok()) {
        out_a.error = "submit A: " + id_a.status().message();
        return;
    }
    auto id_b = batcher.submit(prompt_b, max_new_tokens, sampling, true);
    if (!id_b.ok()) {
        out_b.error = "submit B: " + id_b.status().message();
        return;
    }
    uint32_t guard = 0;
    while (batcher.has_pending()) {
        auto step = batcher.step();
        if (!step.ok()) {
            out_a.error = "step: " + step.status().message();
            out_b.error = out_a.error;
            return;
        }
        if (++guard > 4096u) {
            out_a.error = "step guard exceeded";
            out_b.error = out_a.error;
            return;
        }
    }
    const RuntimeRequest* ra = batcher.request(id_a.value());
    const RuntimeRequest* rb = batcher.request(id_b.value());
    if (ra == nullptr || rb == nullptr) {
        out_a.error = "finished request not found";
        out_b.error = out_a.error;
        return;
    }
    if (ra->generated.empty()) {
        out_a.error = "A generated empty; error_message=" + ra->error_message;
    }
    if (rb->generated.empty()) {
        out_b.error = "B generated empty; error_message=" + rb->error_message;
    }
    out_a.tokens = ra->generated;
    out_b.tokens = rb->generated;
    out_a.ok = out_a.error.empty();
    out_b.ok = out_b.error.empty();
}

static void run_request(ContinuousBatcher& batcher,
                        const std::vector<int32_t>& prompt,
                        uint32_t max_new_tokens,
                        RunOutcome& out) {
    out = RunOutcome{};
    SamplingConfig sampling;
    auto id_result = batcher.submit(prompt, max_new_tokens, sampling, true);
    if (!id_result.ok()) {
        out.error = "submit: " + id_result.status().message();
        return;
    }
    const uint64_t id = id_result.value();
    uint32_t guard = 0;
    while (batcher.has_pending()) {
        auto step = batcher.step();
        if (!step.ok()) {
            out.error = "step: " + step.status().message();
            return;
        }
        if (++guard > 4096u) {
            out.error = "step guard exceeded";
            return;
        }
    }
    const RuntimeRequest* request = batcher.request(id);
    if (request == nullptr) {
        out.error = "finished request not found";
        return;
    }
    if (request->generated.empty()) {
        out.error = "generated empty; error_message=" + request->error_message;
    }
    out.tokens = request->generated;
    out.ok = true;
}

static bool standalone_run(uint32_t tree_width,
                           Executor& executor,
                           Eagle4Executor& eagle_exec,
                           SequenceSlotPool& seq_pool,
                           GdnStatePool& gdn_pool,
                           PagedKVPool& kv_pool,
                           gpu::GpuArena& arena,
                           const Eagle4Config& eagle_config,
                           const std::vector<int32_t>& prompt,
                           uint32_t max_new_tokens,
                           hipStream_t stream,
                           RunOutcome& out) {
    out = RunOutcome{};
    auto sequence_result =
        create_paged_sequence_state(seq_pool, gdn_pool, kv_pool, 128u, stream);
    if (!sequence_result.ok()) {
        out.error = "sequence: " + sequence_result.status().message();
        return false;
    }
    ps::state::PagedSequenceState sequence = sequence_result.release();
    auto context_result = create_eagle4_context_state(eagle_config, arena, 0u);
    if (!context_result.ok()) {
        out.error = "eagle4 context: " + context_result.status().message();
        (void)release_paged_sequence_state(sequence, stream);
        return false;
    }
    ps::qwen35::eagle4::Eagle4ContextState context = context_result.release();
    DFlash2SpecDecoderConfig decoder_config;
    decoder_config.num_drafts = 3u;
    decoder_config.eagle4_tree_width = tree_width;
    auto decoder_result = create_eagle4_spec_decoder(
        executor, eagle_exec, context, sequence, gdn_pool, arena, decoder_config, stream);
    if (!decoder_result.ok()) {
        out.error = "decoder: " + decoder_result.status().message();
        (void)eagle4_context_shutdown(context);
        (void)release_paged_sequence_state(sequence, stream);
        return false;
    }
    DFlash2SpecDecoder decoder = decoder_result.release();
    auto prefill_result = dflash2_spec_prefill(
        decoder, prompt.data(), static_cast<uint32_t>(prompt.size()));
    if (!prefill_result.ok()) {
        out.error = "prefill: " + prefill_result.status().message();
        (void)dflash2_spec_decoder_shutdown(decoder);
        (void)eagle4_context_shutdown(context);
        (void)release_paged_sequence_state(sequence, stream);
        return false;
    }
    out.tokens.push_back(prefill_result.value().pending_token);
    int32_t pending = prefill_result.value().pending_token;
    bool finished = ps::qwen35::is_stop_token(decoder_config.eos_tokens, pending);
    while (!finished && out.tokens.size() < max_new_tokens) {
        const uint32_t remaining =
            static_cast<uint32_t>(max_new_tokens - out.tokens.size());
        auto step_result = dflash2_spec_step(decoder, pending, remaining);
        if (!step_result.ok()) {
            out.error = "step: " + step_result.status().message();
            (void)dflash2_spec_decoder_shutdown(decoder);
            (void)eagle4_context_shutdown(context);
            (void)release_paged_sequence_state(sequence, stream);
            return false;
        }
        auto step = step_result.release();
        for (uint32_t i = 0; i < step.emitted_count; ++i) {
            out.tokens.push_back(step.emitted[i]);
        }
        pending = step.pending_token;
        finished = step.finished;
    }
    (void)dflash2_spec_decoder_shutdown(decoder);
    (void)eagle4_context_shutdown(context);
    (void)release_paged_sequence_state(sequence, stream);
    out.ok = true;
    return true;
}

static int run_case(uint32_t tree_width) {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 1) {
        std::printf("SKIP: eagle4 spec batch test requires 1 GPU\n");
        return 77;
    }

    std::error_code ec;
    const fs::path dir =
        fs::temp_directory_path(ec) / ("ps_eagle4_spec_batch_" + std::to_string(::getpid()));
    fs::remove_all(dir, ec);
    const fs::path target_dir = dir / "target";
    const fs::path eagle_dir = dir / "eagle";
    fs::create_directories(target_dir, ec);
    fs::create_directories(eagle_dir, ec);

    const Status fixture = tp_fixture::build_fixture(target_dir);
    check(fixture.ok(), "target fixture: " + (fixture.ok() ? std::string() : fixture.message()));
    if (!fixture.ok()) {
        fs::remove_all(dir, ec);
        return 1;
    }

    check(write_eagle4_config_json(eagle_dir.string(), 128u, 128u, {0, 1, 2}),
          "eagle4 config json");
    auto config_result = ps::qwen35::eagle4::read_eagle4_config(eagle_dir.string());
    check(config_result.ok(),
          "eagle4 config reads: " + (config_result.ok() ? std::string()
                                                       : config_result.status().message()));
    if (!config_result.ok()) {
        fs::remove_all(dir, ec);
        return 1;
    }
    const Eagle4Config eagle_config = config_result.release();
    const HostModel host = make_host_model(384u, 128u);
    check(write_shard(eagle_dir.string(), eagle_config, host), "eagle4 checkpoint writes");

    if (hipSetDevice(0) != hipSuccess) {
        std::printf("FAIL hipSetDevice\n");
        fs::remove_all(dir, ec);
        return 1;
    }
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        std::printf("FAIL hipStreamCreate\n");
        fs::remove_all(dir, ec);
        return 1;
    }

    auto arena_result = gpu::GpuArena::create(0, 1024ull * 1024ull * 1024ull);
    if (!arena_result.ok()) {
        std::printf("SKIP: arena (%s)\n", arena_result.status().message().c_str());
        (void)hipStreamDestroy(stream);
        fs::remove_all(dir, ec);
        return 77;
    }
    gpu::GpuArena arena = arena_result.release();

    Qwen35LoadOptions load_options;
    auto model_result =
        Qwen35Model::load_from_safetensors(target_dir.string(), arena, stream, load_options);
    check(model_result.ok(),
          "target load: " + (model_result.ok() ? std::string() : model_result.status().message()));
    if (!model_result.ok()) {
        (void)arena.shutdown();
        fs::remove_all(dir, ec);
        return 1;
    }
    Qwen35Model model = model_result.release();

    const uint32_t page_tokens = 16;
    const uint32_t max_seq_len = 128;
    const uint32_t max_concurrent = 4;
    const uint32_t drafts = 3u;
    const uint32_t max_new_tokens = 16u;

    const uint32_t max_blocks = ceil_div_u32(max_seq_len + 1u, page_tokens) + 1u;
    auto seq_result =
        SequenceSlotPool::create(arena, max_concurrent + 2u, max_blocks);
    check(seq_result.ok(), "sequence pool");
    if (!seq_result.ok()) {
        (void)arena.shutdown();
        fs::remove_all(dir, ec);
        return 1;
    }
    SequenceSlotPool seq_pool = seq_result.release();

    auto gdn_layout = qwen35::make_qwen35_gdn_state_layout(model.text_config());
    auto gdn_result = GdnStatePool::create(arena, max_concurrent + 2u, gdn_layout);
    check(gdn_result.ok(), "gdn pool");
    if (!gdn_result.ok()) {
        (void)arena.shutdown();
        fs::remove_all(dir, ec);
        return 1;
    }
    GdnStatePool gdn_pool = gdn_result.release();

    auto kv_result = PagedKVPool::create(
        arena, ceil_div_u32(max_seq_len + 1u, page_tokens) + 1u, page_tokens,
        count_attention_layers(model.text_config()),
        static_cast<uint32_t>(model.text_config().num_key_value_heads),
        static_cast<uint32_t>(model.text_config().attention_head_dim),
        KVCacheDType::BF16);
    check(kv_result.ok(), "kv pool");
    if (!kv_result.ok()) {
        (void)arena.shutdown();
        fs::remove_all(dir, ec);
        return 1;
    }
    PagedKVPool kv_pool = kv_result.release();

    ExecutorConfig exec_config;
    exec_config.max_scheduled_tokens = 96;
    exec_config.max_scheduled_requests = max_concurrent;
    exec_config.max_scheduled_output_rows = 16;
    exec_config.target_hidden_tap_count = 3;
    exec_config.target_hidden_taps = {0u, 1u, 2u};
    auto exec_result =
        create_model_executor(model, seq_pool, gdn_pool, kv_pool, arena, exec_config);
    check(exec_result.ok(),
          "executor: " + (exec_result.ok() ? std::string() : exec_result.status().message()));
    if (!exec_result.ok()) {
        (void)arena.shutdown();
        fs::remove_all(dir, ec);
        return 1;
    }
    Executor executor = exec_result.release();

    Eagle4WeightLoadOptions eagle_load;
    eagle_load.load_embed_tokens = true;
    eagle_load.load_lm_head = true;
    auto eagle_weights_result =
        load_eagle4_weights(eagle_dir.string(), eagle_config, arena, stream, eagle_load);
    check(eagle_weights_result.ok(),
          "eagle4 weights: " +
              (eagle_weights_result.ok() ? std::string()
                                        : eagle_weights_result.status().message()));
    if (!eagle_weights_result.ok()) {
        executor_shutdown(executor);
        (void)arena.shutdown();
        fs::remove_all(dir, ec);
        return 1;
    }
    Eagle4Weights eagle_weights = eagle_weights_result.release();

    Eagle4ExecutorConfig draft_config;
    draft_config.max_rows = ps::qwen35::eagle4::kEagle4ExtendRowsMax;
    draft_config.num_candidates = static_cast<uint32_t>(eagle_config.num_candidates);
    draft_config.use_lowrank_candidate_head = true;
    auto eagle_exec_result =
        create_eagle4_executor(eagle_config, eagle_weights, arena, draft_config);
    check(eagle_exec_result.ok(),
          "eagle4 executor: " +
              (eagle_exec_result.ok() ? std::string()
                                     : eagle_exec_result.status().message()));
    if (!eagle_exec_result.ok()) {
        executor_shutdown(executor);
        (void)arena.shutdown();
        fs::remove_all(dir, ec);
        return 1;
    }
    Eagle4Executor eagle_exec = eagle_exec_result.release();

    const std::vector<int32_t> prompt_a = make_prompt(11u, 48u);

    RunOutcome oracle_a;
    RunOutcome oracle_b;
    const bool oa = standalone_run(tree_width, executor, eagle_exec, seq_pool, gdn_pool, kv_pool,
                                   arena, eagle_config, prompt_a, max_new_tokens, stream,
                                   oracle_a);
    std::vector<int32_t> prompt_b = prompt_a;
    if (oa && oracle_a.ok) {
        const uint32_t shared = std::min<uint32_t>(
            static_cast<uint32_t>(oracle_a.tokens.size()), 16u);
        prompt_b.insert(prompt_b.end(), oracle_a.tokens.begin(),
                        oracle_a.tokens.begin() + shared);
    }
    {
        const std::vector<int32_t> tail = make_prompt(977u, 8u);
        prompt_b.insert(prompt_b.end(), tail.begin(), tail.end());
    }
    const bool ob = standalone_run(tree_width, executor, eagle_exec, seq_pool, gdn_pool, kv_pool,
                                   arena, eagle_config, prompt_b, max_new_tokens, stream,
                                   oracle_b);
    check(oa && oracle_a.ok, "oracle A: " + oracle_a.error);
    check(ob && oracle_b.ok, "oracle B: " + oracle_b.error);
    if (!oa || !ob || !oracle_a.ok || !oracle_b.ok) {
        executor_shutdown(executor);
        (void)eagle4_executor_shutdown(eagle_exec);
        (void)arena.shutdown();
        fs::remove_all(dir, ec);
        return 1;
    }

    DFlash2SpecBatchContext batch;
    {
        std::vector<ps::qwen35::runtime::DFlash2GdnRankBinding> bindings;
        ps::qwen35::runtime::DFlash2GdnRankBinding binding;
        binding.pool = &gdn_pool;
        binding.arena = &arena;
        binding.stream = stream;
        binding.device = 0;
        bindings.push_back(binding);
        const uint32_t batch_max_rows = max_concurrent * (drafts + 1u);
        auto batch_result = create_dflash2_spec_batch_context(bindings, batch_max_rows);
        check(batch_result.ok(),
              "batch context: " +
                  (batch_result.ok() ? std::string() : batch_result.status().message()));
        if (!batch_result.ok()) {
            executor_shutdown(executor);
            (void)eagle4_executor_shutdown(eagle_exec);
            (void)arena.shutdown();
            fs::remove_all(dir, ec);
            return 1;
        }
        batch = batch_result.release();
    }

    DFlash2SpecDecoderConfig session_config;
    session_config.num_drafts = drafts;
    session_config.eagle4_tree_width = tree_width;
    session_config.eos_tokens = model.text_config().stop_tokens;
    auto sessions_result = DFlash2SessionPool::create(
        executor, nullptr, gdn_pool, arena, session_config, max_concurrent, stream,
        &eagle_exec, &eagle_config);
    check(sessions_result.ok(),
          "session pool: " +
              (sessions_result.ok() ? std::string() : sessions_result.status().message()));
    if (!sessions_result.ok()) {
        executor_shutdown(executor);
        (void)eagle4_executor_shutdown(eagle_exec);
        (void)arena.shutdown();
        fs::remove_all(dir, ec);
        return 1;
    }
    DFlash2SessionPool sessions = sessions_result.release();

    ContinuousBatcherConfig batcher_config;
    batcher_config.max_scheduled_tokens = 96;
    batcher_config.max_scheduled_requests = max_concurrent;
    batcher_config.max_seq_len = max_seq_len;
    batcher_config.page_tokens = page_tokens;
    batcher_config.eos_token_ids = model.text_config().stop_tokens;
    ContinuousBatcher batcher(executor, seq_pool, gdn_pool, kv_pool, batcher_config,
                              stream);

    PrefixCacheConfig cache_config;
    cache_config.spec_ring_bytes =
        static_cast<uint32_t>(eagle_config.context_ring_bytes());
    cache_config.min_prefix_tokens = 16;
    auto cache_result = PrefixCache::create(arena, kv_pool, gdn_pool, cache_config);
    check(cache_result.ok(),
          "prefix cache: " +
              (cache_result.ok() ? std::string() : cache_result.status().message()));
    if (!cache_result.ok()) {
        (void)sessions.shutdown();
        executor_shutdown(executor);
        (void)eagle4_executor_shutdown(eagle_exec);
        (void)arena.shutdown();
        fs::remove_all(dir, ec);
        return 1;
    }
    PrefixCache cache = cache_result.release();
    batcher.set_prefix_cache(&cache);

    DFlash2BatcherBinding binding;
    binding.batch = &batch;
    binding.arena = &arena;
    binding.sessions = &sessions;
    binding.num_drafts = drafts;
    binding.eagle4 = true;
    binding.eagle4_executor = &eagle_exec;
    binding.eagle4_config = &eagle_config;
    const Status bind_st = batcher.set_dflash2(binding);
    check(bind_st.ok(),
          "batcher binding: " + (bind_st.ok() ? std::string() : bind_st.message()));
    if (!bind_st.ok()) {
        executor_shutdown(executor);
        (void)eagle4_executor_shutdown(eagle_exec);
        (void)arena.shutdown();
        fs::remove_all(dir, ec);
        return 1;
    }

    RunOutcome batch_a;
    RunOutcome batch_b;
    run_concurrent(batcher, prompt_a, prompt_b, max_new_tokens, batch_a, batch_b);
    check(batch_a.ok, "batch A: " + batch_a.error);
    check(batch_b.ok, "batch B: " + batch_b.error);

    RunOutcome batch_a2;
    run_request(batcher, prompt_a, max_new_tokens, batch_a2);
    check(batch_a2.ok, "batch A2: " + batch_a2.error);

    RunOutcome batch_b2;
    run_request(batcher, prompt_b, max_new_tokens, batch_b2);
    check(batch_b2.ok, "batch B2: " + batch_b2.error);

    const PrefixCacheStats stats_before_hit = cache.stats();
    RunOutcome batch_b_hit;
    run_request(batcher, prompt_b, max_new_tokens, batch_b_hit);
    check(batch_b_hit.ok, "batch B hit: " + batch_b_hit.error);
    const PrefixCacheStats stats_after_hit = cache.stats();
    std::printf(
        "CACHE prompt_b=%zu saves=%llu input_boundary=%llu output_tail=%llu "
        "too_short=%llu too_large=%llu lookups=%llu hits=%llu misses=%llu "
        "restores=%llu\n",
        prompt_b.size(), (unsigned long long)stats_after_hit.saves,
        (unsigned long long)stats_after_hit.input_boundary_saves,
        (unsigned long long)stats_after_hit.output_tail_saves,
        (unsigned long long)stats_after_hit.skipped_too_short,
        (unsigned long long)stats_after_hit.skipped_too_large,
        (unsigned long long)stats_after_hit.lookups,
        (unsigned long long)stats_after_hit.hits,
        (unsigned long long)stats_after_hit.misses,
        (unsigned long long)stats_after_hit.restores);
    check(stats_after_hit.lookups > stats_before_hit.lookups,
          "B hit performs a lookup");
    check(stats_after_hit.restores > stats_before_hit.restores,
          "B hit restores the prefix ring");

    if (batch_a.ok && batch_b.ok && batch_a2.ok && batch_b2.ok) {
        print_tokens("oracle_a", oracle_a.tokens);
        print_tokens("batch_a", batch_a.tokens);
        print_tokens("oracle_b", oracle_b.tokens);
        print_tokens("batch_b", batch_b.tokens);
        check(batch_a.tokens.size() == max_new_tokens, "batch A length");
        check(batch_b.tokens.size() == max_new_tokens, "batch B length");
        check(same_tokens(batch_a.tokens, batch_a2.tokens), "batch A determinism");
        check(same_tokens(batch_b.tokens, batch_b2.tokens), "batch B determinism");
        check(same_tokens(oracle_a.tokens, batch_a.tokens),
              "batch A == standalone A");
        check(same_tokens(oracle_b.tokens, batch_b.tokens),
              "batch B == standalone B");
        check(same_tokens(oracle_b.tokens, batch_b_hit.tokens),
              "restored B == standalone B");
    }

    (void)sessions.shutdown();
    executor_shutdown(executor);
    (void)eagle4_executor_shutdown(eagle_exec);
    (void)arena.shutdown();
    fs::remove_all(dir, ec);
    if (g_fail == 0) {
        std::printf("OK tree_width=%u\n", tree_width);
    }
    return 0;
}

int main() {
    if (run_case(0u) != 0) return 1;
    if (g_fail != 0) return 1;
    if (run_case(4u) != 0) return 1;
    if (g_fail == 0) std::printf("OK\n");
    return g_fail == 0 ? 0 : 1;
}
