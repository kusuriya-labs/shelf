#include <shelf/models/qwen35/dflash2/config.h>
#include <shelf/models/qwen35/dflash2/context_state.h>
#include <shelf/models/qwen35/dflash2/executor.h>
#include <shelf/models/qwen35/dflash2/weights.h>
#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/models/qwen35/runtime/dflash2_spec_decoder.h>
#include <shelf/models/qwen35/runtime/executor.h>
#include <shelf/models/qwen35/state/state_types.h>
#include <shelf/models/qwen35/model/gdn_state_layout.h>
#include <hip/hip_runtime.h>


#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using shelf::qwen35::dflash2::DFlash2Config;
using shelf::qwen35::dflash2::DFlash2ContextState;
using shelf::qwen35::dflash2::DFlash2Executor;
using shelf::qwen35::dflash2::DFlash2ExecutorConfig;
using shelf::qwen35::dflash2::DFlash2Weights;

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

static uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

static uint32_t count_attention_layers(const shelf::qwen35::Qwen35TextConfig& tc) {
    uint32_t n = 0;
    for (int lt : tc.layer_types) {
        if (lt != 0) ++n;
    }
    return n;
}

struct Bundle {
    std::unique_ptr<shelf::state::PagedSequenceState> seq;
    std::unique_ptr<DFlash2ContextState> context;
    std::unique_ptr<shelf::qwen35::runtime::DFlash2SpecDecoder> decoder;
    shelf::qwen35::runtime::DFlash2SpecTiming timing{};
    bool ok = false;
};

struct BatchTimingAgg {
    uint32_t rounds = 0u;
    uint32_t accepted_drafts = 0u;
    uint32_t full_accepts = 0u;
    double round_ms = 0.0;
    double draft_gpu_ms = 0.0;
    double verify_gpu_ms = 0.0;
};

static bool make_bundle(
    shelf::qwen35::Executor& ex,
    DFlash2Executor& draft,
    const DFlash2Config& config,
    shelf::state::SequenceSlotPool& seq_pool,
    shelf::state::GdnStatePool& gdn_pool,
    shelf::state::PagedKVPool& kv_pool,
    shelf::gpu::GpuArena& arena,
    uint32_t seq_len,
    uint32_t num_drafts,
    hipStream_t stream,
    Bundle& out) {
    auto seq = shelf::state::create_paged_sequence_state(
        seq_pool, gdn_pool, kv_pool, seq_len, stream);
    if (!seq.ok()) return false;
    out.seq = std::make_unique<shelf::state::PagedSequenceState>(seq.release());
    auto ctx = shelf::qwen35::dflash2::create_dflash2_context_state(config, arena, 0u);
    if (!ctx.ok()) return false;
    out.context = std::make_unique<DFlash2ContextState>(ctx.release());
    shelf::qwen35::runtime::DFlash2SpecDecoderConfig dcfg;
    dcfg.num_drafts = num_drafts;
    auto dec = shelf::qwen35::runtime::create_dflash2_spec_decoder(
        ex, draft, *out.context, *out.seq, gdn_pool, arena, dcfg, stream);
    if (!dec.ok()) return false;
    out.decoder =
        std::make_unique<shelf::qwen35::runtime::DFlash2SpecDecoder>(dec.release());
    out.decoder->timing = &out.timing;
    out.ok = true;
    return true;
}

static bool run_batch_n(
    shelf::qwen35::runtime::DFlash2SpecBatchContext& batch_ctx,
    std::vector<Bundle>& bundles,
    const std::vector<int32_t>& prompt,
    uint32_t max_new,
    uint32_t num_drafts,
    std::vector<std::vector<int32_t>>& out,
    BatchTimingAgg* agg) {
    const std::size_t n = bundles.size();
    std::vector<int32_t> pending(n, -1);
    std::vector<bool> finished(n, false);
    out.assign(n, {});
    for (std::size_t i = 0; i < n; ++i) {
        auto prefill = shelf::qwen35::runtime::dflash2_spec_prefill(
            *bundles[i].decoder, prompt.data(), static_cast<uint32_t>(prompt.size()));
        if (!prefill.ok()) return false;
        pending[i] = prefill.value().pending_token;
        out[i].push_back(pending[i]);
    }
    while (true) {
        std::vector<shelf::qwen35::runtime::DFlash2SpecBatchSlot> slots;
        std::vector<std::size_t> active;
        for (std::size_t i = 0; i < n; ++i) {
            if (finished[i] || out[i].size() >= max_new) continue;
            shelf::qwen35::runtime::DFlash2SpecBatchSlot slot;
            slot.decoder = bundles[i].decoder.get();
            slot.anchor_token = pending[i];
            slot.num_drafts = num_drafts;
            slot.remaining_tokens = max_new - static_cast<uint32_t>(out[i].size());
            slots.push_back(slot);
            active.push_back(i);
        }
        if (slots.empty()) break;
        auto result = shelf::qwen35::runtime::dflash2_spec_step_batch(batch_ctx, slots);
        if (!result.ok()) {
            std::printf("FAIL batch step: %s\n", result.status().message().c_str());
            return false;
        }
        auto value = result.release();
        if (agg != nullptr) {
            agg->rounds += value.rounds;
            for (const auto& slot_out : value.slots) {
                agg->accepted_drafts += slot_out.num_accepted;
                if (slot_out.num_drafts != 0u &&
                    slot_out.num_accepted >= slot_out.num_drafts) {
                    ++agg->full_accepts;
                }
            }
            agg->round_ms += value.round_ms;
            agg->draft_gpu_ms += value.draft_gpu_ms;
            agg->verify_gpu_ms += value.verify_gpu_ms;
        }
        for (std::size_t k = 0; k < active.size(); ++k) {
            const std::size_t i = active[k];
            const auto& step = value.slots[k];
            for (uint32_t j = 0; j < step.emitted_count; ++j) out[i].push_back(step.emitted[j]);
            pending[i] = step.pending_token;
            if (step.finished) finished[i] = true;
        }
        bool all_done = true;
        for (std::size_t i = 0; i < n; ++i) {
            if (!finished[i] && out[i].size() < max_new) all_done = false;
        }
        if (all_done) break;
    }
    return true;
}

static bool measure_n(
    shelf::qwen35::runtime::DFlash2SpecBatchContext& batch_ctx,
    shelf::qwen35::Executor& ex,
    DFlash2Executor& draft,
    const DFlash2Config& config,
    shelf::state::SequenceSlotPool& seq_pool,
    shelf::state::GdnStatePool& gdn_pool,
    shelf::state::PagedKVPool& kv_pool,
    shelf::gpu::GpuArena& arena,
    const std::vector<int32_t>& prompt,
    uint32_t num_drafts,
    uint32_t max_new,
    uint32_t n,
    hipStream_t stream,
    std::size_t& tokens_out,
    double& ms_out,
    BatchTimingAgg& agg_out) {
    const uint32_t seq_len = static_cast<uint32_t>(prompt.size()) + max_new + num_drafts + 8u;
    std::vector<Bundle> bundles(n);
    for (uint32_t i = 0; i < n; ++i) {
        if (!make_bundle(ex, draft, config, seq_pool, gdn_pool, kv_pool, arena, seq_len,
                         num_drafts, stream, bundles[i])) {
            return false;
        }
    }
    std::vector<std::vector<int32_t>> out;
    agg_out = BatchTimingAgg{};
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = run_batch_n(batch_ctx, bundles, prompt, max_new, num_drafts, out, &agg_out);
    if (hipStreamSynchronize(stream) != hipSuccess) return false;
    const auto t1 = std::chrono::steady_clock::now();
    tokens_out = 0;
    for (const auto& o : out) tokens_out += o.size();
    ms_out = std::chrono::duration<double, std::milli>(t1 - t0).count();
    for (uint32_t i = 0; i < n; ++i) {
        check(shelf::qwen35::runtime::dflash2_spec_decoder_shutdown(*bundles[i].decoder).ok(),
              "perf decoder shutdown");
        (void)shelf::qwen35::dflash2::dflash2_context_shutdown(*bundles[i].context);
        (void)shelf::state::release_paged_sequence_state(*bundles[i].seq, stream);
    }
    return ok;
}

int main() {
    const char* model_dir = std::getenv("SHELF_MODEL_DIR_DFLASH2");
    const char* target_dir = std::getenv("SHELF_MODEL_DIR_DFLASH2_TARGET");
    if (model_dir == nullptr || model_dir[0] == '\0' || target_dir == nullptr ||
        target_dir[0] == '\0') {
        std::printf("SKIP: set SHELF_MODEL_DIR_DFLASH2 and _TARGET\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }

    auto config_result = shelf::qwen35::dflash2::read_dflash2_config(model_dir);
    check(config_result.ok(), "dflash config");
    if (!config_result.ok()) return 1;
    const DFlash2Config config = config_result.value();
    const uint32_t tap_count = static_cast<uint32_t>(config.num_target_layer_ids);
    const uint32_t block_rows = static_cast<uint32_t>(config.block_size);
    const uint32_t max_drafts = block_rows - 1u;
    const uint32_t k = max_drafts;

    auto arena_result = shelf::gpu::GpuArena::create(0, 30ull * 1024ull * 1024ull * 1024ull);
    if (!arena_result.ok()) {
        std::printf("SKIP: arena (%s)\n", arena_result.status().message().c_str());
        return 77;
    }
    shelf::gpu::GpuArena arena = arena_result.release();
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        std::printf("FAIL hipStreamCreate\n");
        return 1;
    }

    auto model_result = shelf::qwen35::Qwen35Model::load_from_safetensors(target_dir, arena,
                                                                       stream);
    check(model_result.ok(), "target model load");
    if (!model_result.ok()) return 1;
    shelf::qwen35::Qwen35Model model = model_result.release();
    const shelf::qwen35::Qwen35TextConfig& tc = model.text_config();

    auto weights_result = shelf::qwen35::dflash2::load_dflash2_weights(model_dir, config, arena,
                                                                    stream);
    check(weights_result.ok(), "dflash weight load");
    if (!weights_result.ok()) return 1;
    DFlash2Weights weights = weights_result.release();

    const uint32_t max_new = 128u;
    const uint32_t context = 64u;
    const uint32_t required_seq_len = context + max_new + max_drafts + 8u;
    const uint32_t page_tokens = 16u;
    const uint32_t kv_heads = static_cast<uint32_t>(tc.num_key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(tc.attention_head_dim);
    const uint32_t num_attn_layers = count_attention_layers(tc);
    const uint32_t num_pages = 8u * ceil_div(required_seq_len, page_tokens) + 24u;

    auto seq_pool_result = shelf::state::SequenceSlotPool::create(arena, 8u, num_pages + 8u);
    auto gdn_pool_result = shelf::state::GdnStatePool::create(
        arena, 8u, shelf::qwen35::make_qwen35_gdn_state_layout(tc));
    auto kv_pool_result = shelf::state::PagedKVPool::create(
        arena, num_pages, page_tokens, num_attn_layers, kv_heads, head_dim,
        shelf::kernel::KVCacheDType::BF16);
    check(seq_pool_result.ok() && gdn_pool_result.ok() && kv_pool_result.ok(), "pool create");
    if (!seq_pool_result.ok() || !gdn_pool_result.ok() || !kv_pool_result.ok()) return 1;
    shelf::state::SequenceSlotPool seq_pool = seq_pool_result.release();
    shelf::state::GdnStatePool gdn_pool = gdn_pool_result.release();
    shelf::state::PagedKVPool kv_pool = kv_pool_result.release();

    std::array<uint32_t, shelf::qwen35::kMaxTargetHiddenTaps> taps{};
    for (uint32_t t = 0; t < tap_count; ++t) {
        taps[t] = static_cast<uint32_t>(config.target_layer_ids[t]);
    }

    const uint32_t max_batch = 4u;
    shelf::qwen35::ExecutorConfig ec;
    ec.max_batch_rows = 2048u;
    ec.max_scheduled_requests = max_batch;
    ec.max_scheduled_output_rows = max_batch * block_rows;
    ec.target_hidden_taps = taps;
    ec.target_hidden_tap_count = tap_count;
    auto exec_result =
        shelf::qwen35::create_model_executor(model, seq_pool, gdn_pool, kv_pool, arena, ec);
    check(exec_result.ok(), "target executor create");
    if (!exec_result.ok()) return 1;
    shelf::qwen35::Executor ex = exec_result.release();

    DFlash2ExecutorConfig draft_cfg;
    draft_cfg.max_rows = block_rows;
    draft_cfg.max_batch_rows = max_batch * block_rows;
    draft_cfg.max_context_rows = static_cast<uint32_t>(config.sliding_window);
    draft_cfg.target_lm_head = &model.weights().lm_head;
    draft_cfg.target_embed_tokens = &model.weights().embed_tokens;
    auto draft_result =
        shelf::qwen35::dflash2::create_dflash2_executor(config, weights, arena, draft_cfg);
    check(draft_result.ok(), "draft executor create");
    if (!draft_result.ok()) return 1;
    DFlash2Executor draft = draft_result.release();

    std::vector<shelf::qwen35::runtime::DFlash2GdnRankBinding> bindings;
    shelf::qwen35::runtime::DFlash2GdnRankBinding binding;
    binding.pool = &gdn_pool;
    binding.arena = &arena;
    binding.stream = stream;
    binding.device = 0;
    bindings.push_back(binding);
    auto batch_result = shelf::qwen35::runtime::create_dflash2_spec_batch_context(
        bindings, max_batch * block_rows);
    check(batch_result.ok(), "batch context create");
    if (!batch_result.ok()) return 1;
    shelf::qwen35::runtime::DFlash2SpecBatchContext batch_ctx = batch_result.release();

    std::vector<int32_t> prompt(context);
    for (uint32_t i = 0; i < context; ++i) {
        prompt[i] = static_cast<int32_t>(100u + (i * 37u + 11u) % 2000u);
    }

    std::printf("k=%u ctx=%u max_new=%u\n", k, context, max_new);
    double base_ms = 0.0;
    for (uint32_t n = 1; n <= max_batch; ++n) {
        std::size_t tokens = 0;
        double ms = 0.0;
        BatchTimingAgg timing;
        const bool ok = measure_n(batch_ctx, ex, draft, config, seq_pool, gdn_pool, kv_pool,
                                  arena, prompt, k, max_new, n, stream, tokens, ms, timing);
        check(ok, "measure n=" + std::to_string(n));
        if (!ok) continue;
        if (n == 1u) base_ms = ms;
        const double tps = ms > 0.0 ? static_cast<double>(tokens) * 1000.0 / ms : 0.0;
        const double per_req = ms / static_cast<double>(n);
        std::printf("batch n=%u tokens=%zu ms=%.1f tok/s=%.1f ms/req=%.1f scale=%.2f\n", n,
                    tokens, ms, tps, per_req, base_ms > 0.0 ? per_req / base_ms : 0.0);
        const double draft_share =
            timing.round_ms > 0.0 ? 100.0 * timing.draft_gpu_ms / timing.round_ms : 0.0;
        const double verify_share =
            timing.round_ms > 0.0 ? 100.0 * timing.verify_gpu_ms / timing.round_ms : 0.0;
        const double mean_accepted =
            timing.rounds > 0u
                ? static_cast<double>(timing.accepted_drafts) / timing.rounds
                : 0.0;
        std::printf(
            "timing n=%u rounds=%u round_ms=%.1f draft_gpu_ms=%.1f draft_share=%.1f%% "
            "verify_gpu_ms=%.1f verify_share=%.1f%% mean_accepted=%.3f full_accepts=%u\n",
            n, timing.rounds, timing.round_ms, timing.draft_gpu_ms, draft_share,
            timing.verify_gpu_ms, verify_share, mean_accepted, timing.full_accepts);
    }

    check(shelf::qwen35::runtime::shutdown_dflash2_spec_batch_context(batch_ctx).ok(),
          "batch context shutdown");

    std::printf("test_dflash2_batch_perf: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
