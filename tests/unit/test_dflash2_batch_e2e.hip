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

#include "support/resident_model_fixture.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using ps::qwen35::dflash2::DFlash2Config;
using ps::qwen35::dflash2::DFlash2ContextState;
using ps::qwen35::dflash2::DFlash2Executor;
using ps::qwen35::dflash2::DFlash2ExecutorConfig;
using ps::qwen35::dflash2::DFlash2Weights;

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

static uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

static uint32_t count_attention_layers(const ps::qwen35::Qwen35TextConfig& tc) {
    uint32_t n = 0;
    for (int lt : tc.layer_types) {
        if (lt != 0) ++n;
    }
    return n;
}

struct Bundle {
    std::unique_ptr<ps::state::PagedSequenceState> seq;
    std::unique_ptr<DFlash2ContextState> context;
    std::unique_ptr<ps::qwen35::runtime::DFlash2SpecDecoder> decoder;
    bool ok = false;
};

static bool make_bundle(
    ps::qwen35::Executor& ex,
    DFlash2Executor& draft,
    const DFlash2Config& config,
    ps::state::SequenceSlotPool& seq_pool,
    ps::state::GdnStatePool& gdn_pool,
    ps::state::PagedKVPool& kv_pool,
    ps::gpu::GpuArena& arena,
    uint32_t seq_len,
    uint32_t num_drafts,
    hipStream_t stream,
    Bundle& out) {
    auto seq = ps::state::create_paged_sequence_state(
        seq_pool, gdn_pool, kv_pool, seq_len, stream);
    if (!seq.ok()) return false;
    out.seq = std::make_unique<ps::state::PagedSequenceState>(seq.release());
    auto ctx = ps::qwen35::dflash2::create_dflash2_context_state(config, arena, 0u);
    if (!ctx.ok()) return false;
    out.context = std::make_unique<DFlash2ContextState>(ctx.release());
    ps::qwen35::runtime::DFlash2SpecDecoderConfig dcfg;
    dcfg.num_drafts = num_drafts;
    auto dec = ps::qwen35::runtime::create_dflash2_spec_decoder(
        ex, draft, *out.context, *out.seq, gdn_pool, arena, dcfg, stream);
    if (!dec.ok()) return false;
    out.decoder =
        std::make_unique<ps::qwen35::runtime::DFlash2SpecDecoder>(dec.release());
    out.ok = true;
    return true;
}

static bool run_single(
    Bundle& b,
    const std::vector<int32_t>& prompt,
    uint32_t max_new,
    std::vector<int32_t>& out) {
    auto prefill = ps::qwen35::runtime::dflash2_spec_prefill(
        *b.decoder, prompt.data(), static_cast<uint32_t>(prompt.size()));
    if (!prefill.ok()) return false;
    out.clear();
    int32_t pending = prefill.value().pending_token;
    out.push_back(pending);
    while (out.size() < max_new && pending >= 0) {
        const uint32_t remaining = max_new - static_cast<uint32_t>(out.size());
        auto step = ps::qwen35::runtime::dflash2_spec_step(*b.decoder, pending, remaining);
        if (!step.ok()) return false;
        auto value = step.release();
        for (uint32_t i = 0; i < value.emitted_count; ++i) out.push_back(value.emitted[i]);
        pending = value.pending_token;
        if (value.finished) break;
    }
    if (out.size() > max_new) out.resize(max_new);
    return true;
}

static bool run_batch(
    ps::qwen35::runtime::DFlash2SpecBatchContext& batch_ctx,
    std::array<Bundle, 2>& b,
    const std::array<std::vector<int32_t>, 2>& prompt,
    uint32_t max_new,
    uint32_t num_drafts,
    std::array<std::vector<int32_t>, 2>& out) {
    std::array<int32_t, 2> pending{};
    std::array<bool, 2> finished{false, false};
    for (std::size_t i = 0; i < 2; ++i) {
        auto prefill = ps::qwen35::runtime::dflash2_spec_prefill(
            *b[i].decoder, prompt[i].data(), static_cast<uint32_t>(prompt[i].size()));
        if (!prefill.ok()) return false;
        pending[i] = prefill.value().pending_token;
        out[i].clear();
        out[i].push_back(pending[i]);
    }
    while (true) {
        std::vector<ps::qwen35::runtime::DFlash2SpecBatchSlot> slots;
        std::vector<std::size_t> active;
        for (std::size_t i = 0; i < 2; ++i) {
            if (finished[i] || out[i].size() >= max_new) continue;
            ps::qwen35::runtime::DFlash2SpecBatchSlot slot;
            slot.decoder = b[i].decoder.get();
            slot.anchor_token = pending[i];
            slot.num_drafts = num_drafts;
            slot.remaining_tokens = max_new - static_cast<uint32_t>(out[i].size());
            slots.push_back(slot);
            active.push_back(i);
        }
        if (slots.empty()) break;
        auto result = ps::qwen35::runtime::dflash2_spec_step_batch(batch_ctx, slots);
        if (!result.ok()) {
            std::printf("FAIL batch step: %s\n", result.status().message().c_str());
            return false;
        }
        auto value = result.release();
        for (std::size_t k = 0; k < active.size(); ++k) {
            const std::size_t i = active[k];
            const auto& step = value.slots[k];
            for (uint32_t j = 0; j < step.emitted_count; ++j) out[i].push_back(step.emitted[j]);
            pending[i] = step.pending_token;
            if (step.finished) finished[i] = true;
        }
        bool all_done = true;
        for (std::size_t i = 0; i < 2; ++i) {
            if (!finished[i] && out[i].size() < max_new) all_done = false;
        }
        if (all_done) break;
    }
    return true;
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

    auto config_result = ps::qwen35::dflash2::read_dflash2_config(model_dir);
    check(config_result.ok(), "dflash config");
    if (!config_result.ok()) return 1;
    const DFlash2Config config = config_result.value();
    const uint32_t tap_count = static_cast<uint32_t>(config.num_target_layer_ids);
    const uint32_t block_rows = static_cast<uint32_t>(config.block_size);
    const uint32_t max_drafts = block_rows - 1u;

    ps::resident::ResidentTestRequest resident_request;
    resident_request.qwen_model_dir = target_dir;
    resident_request.dflash2_model_dir = model_dir;
    resident_request.device = 0;
    resident_request.arena_bytes = 30ull * 1024ull * 1024ull * 1024ull;
    auto resident_result = ps::resident::ResidentModelFixture::acquire(resident_request);
    if (!resident_result.ok()) {
        std::printf("FAIL resident acquire: %s\n",
                    resident_result.status().message().c_str());
        return 1;
    }
    ps::resident::ResidentModelFixture resident = resident_result.release();

    auto arena_result = ps::gpu::GpuArena::create(0, resident.arena_bytes());
    if (!arena_result.ok()) {
        std::printf("SKIP: arena (%s)\n", arena_result.status().message().c_str());
        return 77;
    }
    ps::gpu::GpuArena arena = arena_result.release();
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        std::printf("FAIL hipStreamCreate\n");
        return 1;
    }

    auto model_result = resident.enabled()
        ? resident.take_qwen35()
        : ps::qwen35::Qwen35Model::load_from_safetensors(target_dir, arena, stream);
    check(model_result.ok(), "target model load");
    if (!model_result.ok()) return 1;
    ps::qwen35::Qwen35Model model = model_result.release();
    const ps::qwen35::Qwen35TextConfig& tc = model.text_config();

    auto weights_result = resident.enabled()
        ? resident.take_dflash2(nullptr)
        : ps::qwen35::dflash2::load_dflash2_weights(model_dir, config, arena, stream);
    check(weights_result.ok(), "dflash weight load");
    if (!weights_result.ok()) return 1;
    DFlash2Weights weights = weights_result.release();

    const uint32_t max_new = 64u;
    const uint32_t max_context = 32u;
    const uint32_t required_seq_len = max_context + max_new + max_drafts + 8u;
    const uint32_t page_tokens = 16u;
    const uint32_t kv_heads = static_cast<uint32_t>(tc.num_key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(tc.attention_head_dim);
    const uint32_t num_attn_layers = count_attention_layers(tc);
    const uint32_t num_pages = 4u * ceil_div(required_seq_len, page_tokens) + 24u;

    auto seq_pool_result = ps::state::SequenceSlotPool::create(arena, 4u, num_pages + 8u);
    auto gdn_pool_result = ps::state::GdnStatePool::create(
        arena, 4u, ps::qwen35::make_qwen35_gdn_state_layout(tc));
    auto kv_pool_result = ps::state::PagedKVPool::create(
        arena, num_pages, page_tokens, num_attn_layers, kv_heads, head_dim,
        ps::kernel::KVCacheDType::BF16);
    check(seq_pool_result.ok() && gdn_pool_result.ok() && kv_pool_result.ok(), "pool create");
    if (!seq_pool_result.ok() || !gdn_pool_result.ok() || !kv_pool_result.ok()) return 1;
    ps::state::SequenceSlotPool seq_pool = seq_pool_result.release();
    ps::state::GdnStatePool gdn_pool = gdn_pool_result.release();
    ps::state::PagedKVPool kv_pool = kv_pool_result.release();

    std::array<uint32_t, ps::qwen35::kMaxTargetHiddenTaps> taps{};
    for (uint32_t t = 0; t < tap_count; ++t) {
        taps[t] = static_cast<uint32_t>(config.target_layer_ids[t]);
    }

    ps::qwen35::ExecutorConfig ec;
    ec.max_scheduled_tokens = 2048u;
    ec.max_scheduled_requests = 2u;
    ec.max_scheduled_output_rows = 2u * block_rows;
    ec.target_hidden_taps = taps;
    ec.target_hidden_tap_count = tap_count;
    auto exec_result =
        ps::qwen35::create_model_executor(model, seq_pool, gdn_pool, kv_pool, arena, ec);
    check(exec_result.ok(), "target executor create");
    if (!exec_result.ok()) return 1;
    ps::qwen35::Executor ex = exec_result.release();

    DFlash2ExecutorConfig draft_cfg;
    draft_cfg.max_rows = block_rows;
    draft_cfg.max_batch_rows = 2u * block_rows;
    draft_cfg.max_context_rows = static_cast<uint32_t>(config.sliding_window);
    draft_cfg.target_lm_head = &model.weights().lm_head;
    draft_cfg.target_embed_tokens = &model.weights().embed_tokens;
    auto draft_result =
        ps::qwen35::dflash2::create_dflash2_executor(config, weights, arena, draft_cfg);
    check(draft_result.ok(), "draft executor create");
    if (!draft_result.ok())
        std::printf("draft create status: %s\n", draft_result.status().message().c_str());
    if (!draft_result.ok()) return 1;
    DFlash2Executor draft = draft_result.release();

    std::vector<ps::qwen35::runtime::DFlash2GdnRankBinding> bindings;
    ps::qwen35::runtime::DFlash2GdnRankBinding binding;
    binding.pool = &gdn_pool;
    binding.arena = &arena;
    binding.stream = stream;
    binding.device = 0;
    bindings.push_back(binding);
    const uint32_t batch_max_rows = 2u * block_rows;
    auto batch_result =
        ps::qwen35::runtime::create_dflash2_spec_batch_context(bindings, batch_max_rows);
    check(batch_result.ok(), "batch context create: " +
                                 (batch_result.ok() ? "" : batch_result.status().message()));
    if (!batch_result.ok()) return 1;
    ps::qwen35::runtime::DFlash2SpecBatchContext batch_ctx = batch_result.release();

    struct CaseSpec {
        std::string name;
        uint32_t seed_a;
        uint32_t seed_b;
        uint32_t k;
    };
    std::vector<CaseSpec> cases;
    cases.push_back({"pair_k7", 11u, 71u, 7u});
    cases.push_back({"pair_k3", 23u, 47u, 3u});
    cases.push_back({"pair_k1", 11u, 23u, 1u});

    for (const CaseSpec& cs : cases) {
        std::array<std::vector<int32_t>, 2> prompt;
        for (std::size_t s = 0; s < 2; ++s) {
            const uint32_t seed = s == 0 ? cs.seed_a : cs.seed_b;
            prompt[s].resize(max_context);
            for (uint32_t i = 0; i < max_context; ++i) {
                prompt[s][i] = static_cast<int32_t>(100u + (i * 37u + seed) % 2000u);
            }
        }
        const uint32_t seq_len = max_context + max_new + cs.k + 8u;

        std::array<Bundle, 2> single;
        bool singles_ok = true;
        for (std::size_t s = 0; s < 2; ++s) {
            singles_ok = singles_ok &&
                make_bundle(ex, draft, config, seq_pool, gdn_pool, kv_pool, arena, seq_len,
                            cs.k, stream, single[s]);
        }
        check(singles_ok, cs.name + " single bundles");
        if (!singles_ok) continue;

        std::array<std::vector<int32_t>, 2> single_tokens;
        bool single_ok = true;
        for (std::size_t s = 0; s < 2; ++s) {
            single_ok = single_ok && run_single(single[s], prompt[s], max_new, single_tokens[s]);
        }
        check(single_ok, cs.name + " single run");
        for (std::size_t s = 0; s < 2; ++s) {
            check(ps::qwen35::runtime::dflash2_spec_decoder_shutdown(*single[s].decoder).ok(),
                  cs.name + " single decoder shutdown");
            (void)ps::qwen35::dflash2::dflash2_context_shutdown(*single[s].context);
            (void)ps::state::release_paged_sequence_state(*single[s].seq, stream);
        }
        if (!single_ok) continue;

        std::array<Bundle, 2> batch;
        bool batch_bundles_ok = true;
        for (std::size_t s = 0; s < 2; ++s) {
            batch_bundles_ok = batch_bundles_ok &&
                make_bundle(ex, draft, config, seq_pool, gdn_pool, kv_pool, arena, seq_len,
                            cs.k, stream, batch[s]);
        }
        check(batch_bundles_ok, cs.name + " batch bundles");
        if (!batch_bundles_ok) continue;

        std::array<std::vector<int32_t>, 2> batch_tokens;
        const bool batch_ok = run_batch(batch_ctx, batch, prompt, max_new, cs.k, batch_tokens);
        check(batch_ok, cs.name + " batch run");

        bool exact = true;
        std::size_t mismatch = 0;
        if (batch_ok) {
            for (std::size_t s = 0; s < 2; ++s) {
                if (single_tokens[s] != batch_tokens[s]) {
                    exact = false;
                    while (mismatch < single_tokens[s].size() &&
                           mismatch < batch_tokens[s].size() &&
                           single_tokens[s][mismatch] == batch_tokens[s][mismatch]) {
                        ++mismatch;
                    }
                    break;
                }
            }
        }
        std::printf("%-10s k=%u gen=%zu exact=%s\n", cs.name.c_str(), cs.k,
                    batch_tokens[0].size(), (batch_ok && exact) ? "yes" : "no");
        if (batch_ok && !exact) {
            std::printf("  FIRST_MISMATCH=%zu\n", mismatch);
        }
        check(batch_ok && exact, cs.name + " batch == single");

        for (std::size_t s = 0; s < 2; ++s) {
            check(ps::qwen35::runtime::dflash2_spec_decoder_shutdown(*batch[s].decoder).ok(),
                  cs.name + " batch decoder shutdown");
            (void)ps::qwen35::dflash2::dflash2_context_shutdown(*batch[s].context);
            (void)ps::state::release_paged_sequence_state(*batch[s].seq, stream);
        }
    }

    check(ps::qwen35::runtime::shutdown_dflash2_spec_batch_context(batch_ctx).ok(),
          "batch context shutdown");

    std::printf("test_dflash2_batch_e2e: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
