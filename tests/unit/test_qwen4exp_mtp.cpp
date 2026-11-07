#include <shelf/core/memory/arena.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/inference/state/gdn_state_pool.h>
#include <shelf/inference/state/paged_kv_pool.h>
#include <shelf/inference/state/paged_sequence_state.h>
#include <shelf/inference/state/sequence_slot_pool.h>
#include <shelf/models/qwen35/model/gdn_state_layout.h>
#include <shelf/models/qwen35/runtime/executor.h>
#include <shelf/models/qwen35/runtime/scheduling_types.h>
#include <shelf/models/qwen4exp/model/lower_to_primitives.h>
#include <shelf/models/qwen4exp/model/qwen4exp_config.h>
#include <shelf/models/qwen4exp/weights/qwen4exp_weights.h>

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace q35 = shelf::qwen35;
namespace q4 = shelf::qwen4exp;
namespace rt = shelf::runtime;
namespace st = shelf::state;

namespace {

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    } else {
        std::fprintf(stderr, "ok: %s\n", what);
    }
}

q35::ScheduledBatch make_batch(st::PagedSequenceState& sequence,
                               std::vector<q35::ScheduledRequest>& requests,
                               const std::vector<int32_t>& tokens, uint32_t output_rows) {
    q35::ScheduledRequest req;
    req.sequence = &sequence;
    req.handle = sequence.request_handle();
    req.execution_class = tokens.size() == 1u ? rt::ExecutionClass::DECODE
                                              : rt::ExecutionClass::PREFILL;
    req.token_begin = 0u;
    req.num_tokens = static_cast<uint32_t>(tokens.size());
    req.prefix_tokens = 0u;
    req.compute_logits = true;
    req.sample = true;
    req.num_output_rows = output_rows;
    requests.clear();
    requests.push_back(req);

    q35::ScheduledBatch batch;
    batch.token_ids = tokens.data();
    batch.token_ids_location = q35::TokenIdsLocation::Host;
    batch.requests = requests;
    batch.num_tokens = static_cast<uint32_t>(tokens.size());
    batch.num_requests = 1u;
    batch.num_decode_requests = tokens.size() == 1u ? 1u : 0u;
    batch.num_verify_requests = 0u;
    batch.num_prefill_requests = tokens.size() == 1u ? 0u : 1u;
    return batch;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir;
    int device = 0;
    unsigned arena_gib = 8u;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            device = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--arena-gib") == 0 && i + 1 < argc) {
            arena_gib = static_cast<unsigned>(std::stoul(argv[++i]));
        }
    }
    if (model_dir.empty()) {
        model_dir = std::getenv("SHELF_MODEL_DIR_QWEN4EXP") != nullptr
            ? std::getenv("SHELF_MODEL_DIR_QWEN4EXP")
            : "";
    }
    if (model_dir.empty()) {
        std::fprintf(stderr, "SHELF_MODEL_DIR_QWEN4EXP is not set\n");
        return 77;
    }

    if (hipSetDevice(device) != hipSuccess) {
        std::fprintf(stderr, "hipSetDevice failed\n");
        return 1;
    }
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        std::fprintf(stderr, "hipStreamCreate failed\n");
        return 1;
    }

    auto config_result = q4::read_qwen4exp_text_config(model_dir);
    if (!config_result.ok()) {
        std::fprintf(stderr, "config: %s\n", config_result.status().message().c_str());
        return 1;
    }
    const q4::Qwen4ExpTextConfig config = config_result.value();
    const std::uint32_t hc = static_cast<std::uint32_t>(config.hc_count);
    const std::uint32_t hidden = static_cast<std::uint32_t>(config.hidden_size);
    const std::uint32_t wide = hc * hidden;
    std::fprintf(stderr, "mtp: hidden=%u hc=%u wide=%u\n", hidden, hc, wide);

    auto arena_result = shelf::gpu::GpuArena::create(
        device, static_cast<std::size_t>(arena_gib) * 1024ull * 1024ull * 1024ull);
    if (!arena_result.ok()) {
        std::fprintf(stderr, "arena: %s\n", arena_result.status().message().c_str());
        return 1;
    }
    shelf::gpu::GpuArena arena = arena_result.release();

    q4::Qwen4ExpLoadOptions load_options;
    load_options.tp_size = 1u;
    load_options.tp_rank = 0u;
    load_options.load_mtp = true;
    load_options.mtp_only = true;
    auto loaded = q4::load_qwen4exp_rank(model_dir, arena, stream, load_options);
    if (!loaded.ok()) {
        std::fprintf(stderr, "mtp load: %s\n", loaded.status().message().c_str());
        return 1;
    }
    q4::Qwen4ExpRankWeights weights = loaded.release();
    check(weights.mtp.present, "MTP block loaded");
    check(weights.mtp.layer.mlp.experts.local_experts == config.ffn.num_experts,
          "MTP block owns the unsharded expert range");
    check(weights.mtp.layer.attention.q_proj.rows == 2u * 24u * 256u,
          "MTP gated query width");

    q4::Qwen4ExpLowerOptions lower_options;
    lower_options.max_context_tokens = 64u;
    auto lowered = q4::lower_qwen4exp_mtp_to_primitives(config, weights, lower_options);
    if (!lowered.ok()) {
        std::fprintf(stderr, "mtp lower: %s\n", lowered.status().message().c_str());
        return 1;
    }
    check(lowered.value().graph.external_inputs.size() == 2u,
          "MTP graph takes tokens and the wide hidden");
    check(lowered.value().graph.external_outputs.size() == 5u,
          "MTP graph returns logits and the wide residual");

    shelf::qwen_hybrid::QwenHybridConfig mtp_config = config;
    mtp_config.num_hidden_layers = 1u;
    mtp_config.layer_types = {1u};
    mtp_config.full_attention_interval = 1u;

    constexpr std::uint32_t kPageTokens = 16u;
    constexpr std::uint32_t kMaxSeq = 64u;
    const std::uint32_t pages = (kMaxSeq + 1u + kPageTokens - 1u) / kPageTokens + 1u;
    auto slots = st::SequenceSlotPool::create(arena, 1u, pages);
    if (!slots.ok()) {
        std::fprintf(stderr, "slot pool: %s\n", slots.status().message().c_str());
        return 1;
    }
    const st::GdnStatePoolLayout gdn_layout = q35::make_qwen35_gdn_state_layout(mtp_config);
    auto gdn = st::GdnStatePool::create(arena, 1u, gdn_layout);
    if (!gdn.ok()) {
        std::fprintf(stderr, "gdn pool: %s\n", gdn.status().message().c_str());
        return 1;
    }
    auto kv = st::PagedKVPool::create(
        arena, pages, kPageTokens, 1u,
        static_cast<std::uint32_t>(mtp_config.num_key_value_heads),
        static_cast<std::uint32_t>(mtp_config.attention_head_dim),
        shelf::kernel::KVCacheDType::BF16);
    if (!kv.ok()) {
        std::fprintf(stderr, "kv pool: %s\n", kv.status().message().c_str());
        return 1;
    }
    auto sequence = st::create_paged_sequence_state(slots.value(), gdn.value(), kv.value(),
                                                    kMaxSeq, stream);
    if (!sequence.ok()) {
        std::fprintf(stderr, "sequence: %s\n", sequence.status().message().c_str());
        return 1;
    }

    q35::ExecutorModelInput input;
    input.model.config = &mtp_config;
    input.model.embed_tokens = &weights.embed_tokens;
    input.lowered = lowered.release();

    q35::ExecutorConfig exec_config;
    exec_config.max_batch_rows = 8u;
    exec_config.max_scheduled_requests = 1u;
    exec_config.max_scheduled_output_rows = 1u;
    exec_config.max_context_tokens = kMaxSeq;
    exec_config.external_hidden_input = true;
    exec_config.external_hidden_width = wide;
    exec_config.emit_normed_token_hidden = true;
    exec_config.normed_token_hidden_width = wide;
    auto executor = q35::create_executor(std::move(input), slots.value(), gdn.value(),
                                         kv.value(), arena, exec_config);
    if (!executor.ok()) {
        std::fprintf(stderr, "executor: %s\n", executor.status().message().c_str());
        return 1;
    }
    q35::Executor ex = executor.release();

    std::vector<uint16_t> hidden_host(static_cast<std::size_t>(wide) * 4u);
    for (std::size_t i = 0; i < hidden_host.size(); ++i) {
        const float value = 0.02f * static_cast<float>((i % 17u) + 1u) - 0.15f;
        hidden_host[i] = shelf::f32_to_bf16_rne(value);
    }
    if (hipMemcpy(ex.input_hidden.data<void>(), hidden_host.data(),
                  hidden_host.size() * sizeof(uint16_t), hipMemcpyHostToDevice) !=
        hipSuccess) {
        std::fprintf(stderr, "hidden upload failed\n");
        return 1;
    }

    std::vector<q35::ScheduledRequest> requests;
    const std::vector<int32_t> tokens{248041, 77091, 279, 2614};
    auto out = q35::execute_batch(
        ex, make_batch(sequence.value(), requests, tokens, 1u), stream);
    if (!out.ok()) {
        std::fprintf(stderr, "execute: %s\n", out.status().message().c_str());
        return 1;
    }
    check(out.value().logits.data() != nullptr, "MTP graph produced logits");
    check(ex.normed_token_hidden.data() != nullptr, "MTP graph produced the wide residual");

    std::vector<float> logits(static_cast<std::size_t>(static_cast<std::uint32_t>(
        config.vocab_size)));
    if (hipMemcpy(logits.data(), out.value().logits.data(), logits.size() * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess) {
        std::fprintf(stderr, "logits readback failed\n");
        return 1;
    }
    bool finite = true;
    bool nonzero = false;
    for (float value : logits) {
        if (!std::isfinite(value)) finite = false;
        if (value != 0.0f) nonzero = true;
    }
    check(finite, "MTP logits are finite");
    check(nonzero, "MTP logits are non-zero");

    std::vector<uint16_t> wide_host(static_cast<std::size_t>(wide));
    if (hipMemcpy(wide_host.data(), ex.normed_token_hidden.data(),
                  wide_host.size() * sizeof(uint16_t), hipMemcpyDeviceToHost) !=
        hipSuccess) {
        std::fprintf(stderr, "wide residual readback failed\n");
        return 1;
    }
    bool wide_nonzero = false;
    for (uint16_t value : wide_host) {
        if (value != 0u) wide_nonzero = true;
    }
    check(wide_nonzero, "MTP wide residual is non-zero");

    (void)q35::executor_shutdown(ex);
    (void)st::release_paged_sequence_state(sequence.value(), stream);
    (void)hipStreamSynchronize(stream);
    (void)hipStreamDestroy(stream);

    std::fprintf(stderr, failures == 0 ? "PASS\n" : "FAILURES=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
