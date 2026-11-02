#include <shelf/models/qwen35/dflash2/config.h>
#include <shelf/models/qwen35/dflash2/context_state.h>
#include <shelf/models/qwen35/dflash2/executor.h>
#include <shelf/models/qwen35/dflash2/weights.h>
#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/models/qwen35/runtime/dflash2_spec_decoder.h>
#include <shelf/models/qwen35/runtime/executor.h>
#include <shelf/models/qwen35/runtime/scheduling_types.h>
#include <shelf/models/qwen35/state/state_types.h>
#include <shelf/models/qwen35/model/gdn_state_layout.h>
#include <hip/hip_runtime.h>


#include <array>
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
using shelf::qwen35::SchedulePlan;
using shelf::qwen35::ScheduledBatch;
using shelf::qwen35::runtime::DFlash2SpecBatchContext;

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
    bool ok = false;
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
    out.ok = true;
    return true;
}

struct Prefilled {
    int32_t anchor = -1;
    std::array<int32_t, shelf::qwen35::dflash2::kMaxBlockSize - 1u> drafts{};
    uint32_t draft_count = 0;
};

static bool prefill_bundle(Bundle& b, const std::vector<int32_t>& prompt, uint32_t k,
                           hipStream_t stream, Prefilled& out) {
    auto prefill = shelf::qwen35::runtime::dflash2_spec_prefill(
        *b.decoder, prompt.data(), static_cast<uint32_t>(prompt.size()));
    if (!prefill.ok()) {
        std::printf("FAIL prefill: %s\n", prefill.status().message().c_str());
        return false;
    }
    out.anchor = prefill.value().pending_token;
    int32_t* proposal_device = b.decoder->draft->proposal_tokens.data<int32_t>();
    const shelf::Status st = shelf::qwen35::dflash2::dflash2_propose_cached(
        *b.decoder->draft, *b.decoder->context, out.anchor, k, proposal_device, stream);
    if (!st.ok()) {
        std::printf("FAIL propose: %s\n", st.message().c_str());
        return false;
    }
    const hipError_t err = hipMemcpy(out.drafts.data(), proposal_device,
                                     static_cast<std::size_t>(k) * sizeof(int32_t),
                                     hipMemcpyDeviceToHost);
    if (err != hipSuccess) return false;
    out.draft_count = k;
    return true;
}

static bool run_plan(shelf::qwen35::Executor& ex, SchedulePlan& plan,
                     const DFlash2SpecBatchContext& ctx, uint32_t capture_rows,
                     hipStream_t stream, std::vector<int32_t>& sampled,
                     uint32_t& num_outputs) {
    plan.finalize();
    const shelf::Status valid = shelf::qwen35::validate_scheduled_batch(plan.batch, 0u, 0u);
    if (!valid.ok()) {
        std::printf("FAIL validate: %s\n", valid.message().c_str());
        return false;
    }
    shelf::qwen35::ExecuteBatchOptions options;
    options.gdn_spec_history =
        shelf::qwen35::runtime::gdn_spec_history_view(ctx.history[0], capture_rows);
    options.gdn_compact.delta = ctx.compact[0].delta;
    options.gdn_compact.k = ctx.compact[0].k;
    options.gdn_compact.a = ctx.compact[0].a;
    options.gdn_compact.delta_layer_stride = ctx.compact[0].delta_layer_stride;
    options.gdn_compact.k_layer_stride = ctx.compact[0].k_layer_stride;
    options.gdn_compact.a_layer_stride = ctx.compact[0].a_layer_stride;
    auto executed = shelf::qwen35::execute_batch(ex, plan.batch, stream, options);
    if (!executed.ok()) {
        std::printf("FAIL execute: %s\n", executed.status().message().c_str());
        return false;
    }
    shelf::qwen35::BatchExecutionOutput output = executed.release();
    num_outputs = output.num_outputs;
    sampled.assign(output.num_outputs, -1);
    if (output.num_outputs == 0u) return true;
    const hipError_t err = hipMemcpy(
        sampled.data(), output.sampled_tokens.data<int32_t>(),
        static_cast<std::size_t>(output.num_outputs) * sizeof(int32_t),
        hipMemcpyDeviceToHost);
    return err == hipSuccess;
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
    const uint32_t k = block_rows - 1u;

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

    const uint32_t prompt_tokens = 32u;
    const uint32_t required_seq_len = prompt_tokens + k + 16u;
    const uint32_t page_tokens = 16u;
    const uint32_t kv_heads = static_cast<uint32_t>(tc.num_key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(tc.attention_head_dim);
    const uint32_t num_attn_layers = count_attention_layers(tc);
    const uint32_t num_pages = 8u * ceil_div(required_seq_len, page_tokens) + 24u;
    const uint32_t max_requests = 2u;
    const uint32_t max_rows = max_requests * block_rows;

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

    shelf::qwen35::ExecutorConfig ec;
    ec.max_batch_rows = 2048u;
    ec.max_scheduled_requests = max_requests;
    ec.max_scheduled_output_rows = max_rows;
    ec.target_hidden_taps = taps;
    ec.target_hidden_tap_count = tap_count;
    auto exec_result =
        shelf::qwen35::create_model_executor(model, seq_pool, gdn_pool, kv_pool, arena, ec);
    check(exec_result.ok(), "target executor create");
    if (!exec_result.ok()) return 1;
    shelf::qwen35::Executor ex = exec_result.release();

    DFlash2ExecutorConfig draft_cfg;
    draft_cfg.max_rows = block_rows;
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
        bindings, max_rows);
    check(batch_result.ok(), "batch context create");
    if (!batch_result.ok()) return 1;
    DFlash2SpecBatchContext batch_ctx = batch_result.release();

    std::vector<int32_t> prompt(prompt_tokens);
    for (uint32_t i = 0; i < prompt_tokens; ++i) {
        prompt[i] = static_cast<int32_t>(100u + (i * 37u + 11u) % 2000u);
    }

    std::array<Bundle, 4> bundles;
    const char* names[4] = {"mixed_decode", "mixed_verify", "alone_decode", "alone_verify"};
    bool bundles_ok = true;
    for (std::size_t i = 0; i < bundles.size(); ++i) {
        bundles_ok = bundles_ok &&
            make_bundle(ex, draft, config, seq_pool, gdn_pool, kv_pool, arena,
                        required_seq_len, k, stream, bundles[i]);
    }
    check(bundles_ok, "bundles");
    if (!bundles_ok) return 1;

    std::array<Prefilled, 4> pre;
    bool prefill_ok = true;
    for (std::size_t i = 0; i < bundles.size(); ++i) {
        prefill_ok = prefill_ok && prefill_bundle(bundles[i], prompt, k, stream, pre[i]);
    }
    check(prefill_ok, "prefill");
    if (!prefill_ok) return 1;

    for (std::size_t i = 1; i < bundles.size(); ++i) {
        bool same = pre[i].anchor == pre[0].anchor;
        for (uint32_t j = 0; j < k; ++j) {
            same = same && pre[i].drafts[j] == pre[0].drafts[j];
        }
        check(same, std::string("proposal deterministic ") + names[i]);
    }

    auto tokens_for = [&](std::size_t idx) {
        std::vector<int32_t> tokens;
        tokens.push_back(pre[idx].anchor);
        for (uint32_t j = 0; j < k; ++j) tokens.push_back(pre[idx].drafts[j]);
        return tokens;
    };

    std::vector<int32_t> mixed_tokens = tokens_for(0);
    std::vector<int32_t> mixed_decode_token{mixed_tokens[0]};
    std::vector<int32_t> decode_only_tokens{mixed_tokens[0]};

    SchedulePlan mixed;
    shelf::Status st = mixed.append_decode(bundles[0].seq.get(), mixed_decode_token[0],
                                        bundles[0].seq->position, true, true,
                                        shelf::qwen35::runtime::SamplingConfig{}, 0u);
    check(st.ok(), "mixed append decode");
    st = mixed.append_verify(bundles[1].seq.get(), tokens_for(1), bundles[1].seq->position,
                             k + 1u, shelf::qwen35::runtime::SamplingConfig{}, 0u);
    check(st.ok(), "mixed append verify");
    mixed.finalize();
    check(mixed.batch.num_decode_requests == 1u, "mixed decode count");
    check(mixed.batch.num_verify_requests == 1u, "mixed verify count");
    check(mixed.batch.num_prefill_requests == 0u, "mixed prefill count");
    check(mixed.batch.num_requests == 2u, "mixed request count");
    check(mixed.batch.speculative_verify, "mixed speculative verify flag");

    SchedulePlan ordering;
    st = ordering.append_verify(bundles[2].seq.get(), tokens_for(2), bundles[2].seq->position,
                                k + 1u, shelf::qwen35::runtime::SamplingConfig{}, 0u);
    check(st.ok(), "ordering append verify");
    st = ordering.append_decode(bundles[3].seq.get(), pre[3].anchor,
                                bundles[3].seq->position, true, true,
                                shelf::qwen35::runtime::SamplingConfig{}, 0u);
    check(!st.ok() && st.code() == shelf::Status::Code::invalid_state,
          "decode after verify rejected");

    std::vector<int32_t> mixed_sampled;
    uint32_t mixed_outputs = 0u;
    const bool mixed_ok = run_plan(ex, mixed, batch_ctx, 1u + k + 1u, stream, mixed_sampled,
                                   mixed_outputs);
    check(mixed_ok, "mixed execute");
    if (!mixed_ok) return 1;
    check(mixed_outputs == 1u + k + 1u, "mixed output rows");

    SchedulePlan verify_only;
    st = verify_only.append_verify(bundles[3].seq.get(), tokens_for(3),
                                   bundles[3].seq->position, k + 1u,
                                   shelf::qwen35::runtime::SamplingConfig{}, 0u);
    check(st.ok(), "verify only append");
    std::vector<int32_t> verify_only_sampled;
    uint32_t verify_only_outputs = 0u;
    const bool verify_only_ok = run_plan(ex, verify_only, batch_ctx, k + 1u, stream,
                                         verify_only_sampled, verify_only_outputs);
    check(verify_only_ok, "verify only execute");

    SchedulePlan decode_only;
    st = decode_only.append_decode(bundles[2].seq.get(), decode_only_tokens[0],
                                   bundles[2].seq->position, true, true,
                                   shelf::qwen35::runtime::SamplingConfig{}, 0u);
    check(st.ok(), "decode only append");
    std::vector<int32_t> decode_only_sampled;
    uint32_t decode_only_outputs = 0u;
    const bool decode_only_ok = run_plan(ex, decode_only, batch_ctx, 0u, stream,
                                         decode_only_sampled, decode_only_outputs);
    check(decode_only_ok, "decode only execute");

    if (mixed_ok && verify_only_ok && verify_only_outputs == k + 1u) {
        bool verify_exact = true;
        for (uint32_t j = 0; j < k + 1u; ++j) {
            if (mixed_sampled[1u + j] != verify_only_sampled[j]) {
                verify_exact = false;
                std::printf("  verify row %u mixed=%d alone=%d\n", j, mixed_sampled[1u + j],
                            verify_only_sampled[j]);
            }
        }
        check(verify_exact, "mixed verify rows == verify only rows");
    }
    if (mixed_ok && decode_only_ok && decode_only_outputs == 1u) {
        check(mixed_sampled[0] == decode_only_sampled[0],
              "mixed decode row == decode only row");
    }

    std::printf("mixed batch rows=%u verify_only_rows=%u decode_only_rows=%u\n",
                mixed_outputs, verify_only_outputs, decode_only_outputs);

    for (std::size_t i = 0; i < bundles.size(); ++i) {
        check(shelf::qwen35::runtime::dflash2_spec_decoder_shutdown(*bundles[i].decoder).ok(),
              std::string("decoder shutdown ") + names[i]);
        (void)shelf::qwen35::dflash2::dflash2_context_shutdown(*bundles[i].context);
        (void)shelf::state::release_paged_sequence_state(*bundles[i].seq, stream);
    }
    check(shelf::qwen35::runtime::shutdown_dflash2_spec_batch_context(batch_ctx).ok(),
          "batch context shutdown");

    std::printf("test_dflash2_mixed_batch: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
