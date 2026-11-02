#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/models/qwen35/dflash2/config.h>
#include <phaseshift/models/qwen35/model/lower_to_primitives.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/inference/execution/program_executor.h>
#include <phaseshift/models/qwen35/runtime/scheduling_types.h>
#include <phaseshift/models/qwen35/state/state_types.h>
#include <phaseshift/models/qwen35/model/gdn_state_layout.h>
#include <hip/hip_runtime.h>

#include "support/resident_model_fixture.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

static uint32_t ceil_div(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}

static uint32_t count_attention_layers(const ps::qwen35::Qwen35TextConfig& tc) {
    uint32_t n = 0;
    for (int lt : tc.layer_types) {
        if (lt != 0) ++n;
    }
    return n;
}

static ps::qwen35::ScheduledBatch make_batch(
    ps::state::PagedSequenceState& sequence,
    std::vector<ps::qwen35::ScheduledRequest>& requests,
    const int32_t* tokens,
    uint32_t num_tokens,
    uint32_t prefix_tokens,
    uint32_t output_rows) {
    ps::qwen35::ScheduledRequest req;
    req.sequence = &sequence;
    req.handle = sequence.request_handle();
    req.execution_class = num_tokens == 1u ? ps::runtime::ExecutionClass::DECODE
                                           : ps::runtime::ExecutionClass::PREFILL;
    req.token_begin = 0u;
    req.num_tokens = num_tokens;
    req.prefix_tokens = prefix_tokens;
    req.compute_logits = true;
    req.sample = true;
    req.num_output_rows = output_rows;
    requests.clear();
    requests.push_back(req);
    ps::qwen35::ScheduledBatch batch;
    batch.token_ids = tokens;
    batch.requests = requests;
    batch.num_tokens = num_tokens;
    batch.num_requests = 1u;
    batch.num_decode_requests = num_tokens == 1u ? 1u : 0u;
    batch.num_verify_requests = 0u;
    batch.num_prefill_requests = num_tokens == 1u ? 0u : 1u;
    return batch;
}

struct Captured {
    std::vector<uint8_t> bytes;
    uint32_t row_stride = 0;
    uint32_t feature_count = 0;
    uint32_t rows = 0;
};

int main() {
    const char* target_dir = std::getenv("PHASESHIFT_MODEL_DIR_DFLASH2_TARGET");
    if (target_dir == nullptr || target_dir[0] == '\0') {
        std::printf("SKIP: set PHASESHIFT_MODEL_DIR_DFLASH2_TARGET\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }

    std::array<uint32_t, ps::qwen35::kMaxTargetHiddenTaps> taps{};
    uint32_t tap_count = 0;
    {
        const char* dflash_dir = std::getenv("PHASESHIFT_MODEL_DIR_DFLASH2");
        if (dflash_dir != nullptr && dflash_dir[0] != '\0') {
            auto cfg = ps::qwen35::dflash2::read_dflash2_config(dflash_dir);
            if (cfg.ok() && cfg.value().num_target_layer_ids <= taps.size()) {
                for (std::size_t i = 0; i < cfg.value().num_target_layer_ids; ++i) {
                    taps[i] = static_cast<uint32_t>(cfg.value().target_layer_ids[i]);
                }
                tap_count = static_cast<uint32_t>(cfg.value().num_target_layer_ids);
            }
        }
        if (tap_count == 0u) {
            taps = {5u, 19u, 33u, 47u, 61u};
            tap_count = 5u;
        }
    }

    ps::resident::ResidentTestRequest resident_request;
    resident_request.qwen_model_dir = target_dir;
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
    check(model_result.ok(), "model load: " +
                                 (model_result.ok() ? ""
                                                    : model_result.status().message()));
    if (!model_result.ok()) {
        std::printf("test_dflash2_target_taps: failed=%d\n", g_fail);
        return 1;
    }
    ps::qwen35::Qwen35Model model = model_result.release();
    const ps::qwen35::Qwen35TextConfig& tc = model.text_config();

    uint32_t max_tap = 0;
    for (uint32_t t = 0; t < tap_count; ++t) max_tap = taps[t] > max_tap ? taps[t] : max_tap;
    if (tc.num_hidden_layers <= max_tap) {
        std::printf("SKIP: model has %zu layers, tap %u out of range\n",
                    tc.num_hidden_layers, max_tap);
        return 77;
    }

    {
        auto lower_off = ps::qwen35::lower_qwen35_to_primitives(tc, model.weights());
        check(lower_off.ok(), "lower (no taps)");
        if (lower_off.ok()) {
            check(lower_off.value().graph.external_outputs.size() == 4u,
                  "no-tap lowering keeps 4 external outputs");
        }
    }

    std::unordered_map<uint32_t, uint32_t> tap_value_id;
    {
        ps::qwen35::Qwen35LowerOptions options;
        options.hidden_taps = std::span<const uint32_t>(taps.data(), tap_count);
        auto lower_on = ps::qwen35::lower_qwen35_to_primitives(tc, model.weights(), options);
        check(lower_on.ok(), "lower (taps): " +
                                 (lower_on.ok() ? "" : lower_on.status().message()));
        if (!lower_on.ok()) {
            std::printf("test_dflash2_target_taps: failed=%d\n", g_fail);
            return 1;
        }
        const auto& graph = lower_on.value().graph;
        check(graph.external_outputs.size() == 4u + tap_count,
              "tap lowering adds one external output per tap");
        for (uint32_t t = 0; t < tap_count; ++t) {
            const std::string want = "L" + std::to_string(taps[t]) + ".output";
            for (const auto& node : graph.nodes) {
                if (node.debug_name == want && !node.outputs.empty()) {
                    tap_value_id[t] = node.outputs[0].id;
                }
            }
            check(tap_value_id.count(t) == 1u, "layer output value found: " + want);
        }
    }

    const uint32_t kv_heads = static_cast<uint32_t>(tc.num_key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(tc.attention_head_dim);
    const uint32_t num_attn_layers = count_attention_layers(tc);
    const uint32_t page_tokens = 16u;
    const uint32_t max_tokens = 32u;
    const uint32_t num_pages = ceil_div(max_tokens + 16u, page_tokens) + 8u;

    auto seq_pool_result = ps::state::SequenceSlotPool::create(arena, 2u, num_pages + 8u);
    auto gdn_pool_result = ps::state::GdnStatePool::create(
        arena, 2u, ps::qwen35::make_qwen35_gdn_state_layout(tc));
    auto kv_pool_result = ps::state::PagedKVPool::create(
        arena, num_pages, page_tokens, num_attn_layers, kv_heads, head_dim,
        ps::kernel::KVCacheDType::BF16);
    check(seq_pool_result.ok() && gdn_pool_result.ok() && kv_pool_result.ok(),
          "pool create");
    if (!seq_pool_result.ok() || !gdn_pool_result.ok() || !kv_pool_result.ok()) {
        std::printf("test_dflash2_target_taps: failed=%d\n", g_fail);
        return 1;
    }
    ps::state::SequenceSlotPool seq_pool = seq_pool_result.release();
    ps::state::GdnStatePool gdn_pool = gdn_pool_result.release();
    ps::state::PagedKVPool kv_pool = kv_pool_result.release();

    std::vector<int32_t> prompt(max_tokens);
    for (uint32_t i = 0; i < max_tokens; ++i) {
        prompt[i] = static_cast<int32_t>(100u + (i * 37u + 11u) % 2000u);
    }

    std::vector<int32_t> off_tokens;
    {
        ps::qwen35::ExecutorConfig ec;
        ec.max_scheduled_tokens = max_tokens;
        ec.max_scheduled_requests = 1u;
        ec.max_scheduled_output_rows = 1u;
        auto exec_result =
            ps::qwen35::create_model_executor(model, seq_pool, gdn_pool, kv_pool, arena, ec);
        check(exec_result.ok(), "create executor (no taps): " +
                                    (exec_result.ok() ? ""
                                                      : exec_result.status().message()));
        if (exec_result.ok()) {
            ps::qwen35::Executor ex = exec_result.release();
            for (const auto& tap : ex.dflash_target_hidden) {
                check(tap.ndim() == 0u, "no tap buffers without taps");
            }
            auto seq_result = ps::state::create_paged_sequence_state(
                seq_pool, gdn_pool, kv_pool, max_tokens + 16u, stream);
            check(seq_result.ok(), "sequence create (no taps)");
            if (seq_result.ok()) {
                ps::state::PagedSequenceState seq = seq_result.release();
                std::vector<ps::qwen35::ScheduledRequest> requests;
                ps::qwen35::ScheduledBatch batch = make_batch(
                    seq, requests, prompt.data(), max_tokens, 0u, 1u);
                auto executed = ps::qwen35::execute_batch(ex, batch, stream);
                check(executed.ok(), "execute (no taps): " +
                                         (executed.ok() ? ""
                                                        : executed.status().message()));
                if (executed.ok()) {
                    ps::qwen35::BatchExecutionOutput out = executed.release();
                    off_tokens.resize(out.sampled_tokens.dim(0));
                    if (!off_tokens.empty()) {
                        check(hipMemcpy(off_tokens.data(),
                                        out.sampled_tokens.data<int32_t>(),
                                        off_tokens.size() * sizeof(int32_t),
                                        hipMemcpyDeviceToHost) == hipSuccess,
                              "sampled tokens copy (no taps)");
                    }
                }
                (void)ps::state::release_paged_sequence_state(seq, stream);
            }
            const ps::Status st = ps::qwen35::executor_shutdown(ex);
            check(st.ok(), "executor shutdown (no taps): " + st.message());
        }
    }

    {
        ps::qwen35::ExecutorConfig ec;
        ec.max_scheduled_tokens = max_tokens;
        ec.max_scheduled_requests = 1u;
        ec.max_scheduled_output_rows = 1u;
        ec.target_hidden_taps = taps;
        ec.target_hidden_tap_count = tap_count;
        auto exec_result =
            ps::qwen35::create_model_executor(model, seq_pool, gdn_pool, kv_pool, arena, ec);
        check(exec_result.ok(), "create executor (taps): " +
                                    (exec_result.ok() ? ""
                                                      : exec_result.status().message()));
        if (!exec_result.ok()) {
            std::printf("test_dflash2_target_taps: failed=%d\n", g_fail);
            return 1;
        }
        ps::qwen35::Executor ex = exec_result.release();

        const uint32_t hidden = static_cast<uint32_t>(tc.hidden_size);
        for (uint32_t t = 0; t < tap_count; ++t) {
            const auto& tap = ex.dflash_target_hidden[t];
            check(tap.ndim() == 2u && tap.dim(0) >= max_tokens && tap.dim(1) == hidden,
                  "tap buffer shape " + std::to_string(t));
        }

        const uint32_t sizes[] = {32u, 8u, 1u};
        for (uint32_t si = 0; si < 3u; ++si) {
            const uint32_t n = sizes[si];
            auto seq_result = ps::state::create_paged_sequence_state(
                seq_pool, gdn_pool, kv_pool, max_tokens + 16u, stream);
            check(seq_result.ok(), "sequence create (taps)");
            if (!seq_result.ok()) continue;
            ps::state::PagedSequenceState seq = seq_result.release();

            std::unordered_map<uint32_t, Captured> captured;
            ps::qwen35::runtime::ValueTraceSink sink;
            sink.on_value = [&](uint32_t value_id, const void* device_ptr,
                                uint32_t row_stride, uint32_t feature_count,
                                ps::runtime::ValueDType dtype, uint32_t rows,
                                hipStream_t s) {
                if (dtype != ps::runtime::ValueDType::BF16) return;
                bool wanted = false;
                for (uint32_t t = 0; t < tap_count; ++t) {
                    auto it = tap_value_id.find(t);
                    if (it != tap_value_id.end() && it->second == value_id) wanted = true;
                }
                if (!wanted) return;
                const uint32_t stride = row_stride != 0u ? row_stride : feature_count;
                Captured c;
                c.row_stride = stride;
                c.feature_count = feature_count;
                c.rows = rows;
                c.bytes.resize(static_cast<std::size_t>(rows) * stride * 2u);
                if (!c.bytes.empty()) {
                    if (hipStreamSynchronize(s) != hipSuccess) return;
                    if (hipMemcpy(c.bytes.data(), device_ptr, c.bytes.size(),
                                  hipMemcpyDeviceToHost) != hipSuccess) {
                        return;
                    }
                }
                captured[value_id] = std::move(c);
            };
            ex.value_trace = &sink;

            std::vector<ps::qwen35::ScheduledRequest> requests;
            ps::qwen35::ScheduledBatch batch =
                make_batch(seq, requests, prompt.data(), n, 0u, 1u);
            auto executed = ps::qwen35::execute_batch(ex, batch, stream);
            check(executed.ok(), "execute (taps, n=" + std::to_string(n) +
                                     "): " +
                                     (executed.ok() ? "" : executed.status().message()));
            if (executed.ok()) {
                ps::qwen35::BatchExecutionOutput out = executed.release();
                if (n == 32u) {
                    std::vector<int32_t> on_tokens(out.sampled_tokens.dim(0));
                    if (!on_tokens.empty()) {
                        check(hipMemcpy(on_tokens.data(),
                                        out.sampled_tokens.data<int32_t>(),
                                        on_tokens.size() * sizeof(int32_t),
                                        hipMemcpyDeviceToHost) == hipSuccess,
                              "sampled tokens copy (taps)");
                    }
                    check(on_tokens.size() == off_tokens.size(),
                          "sampled token count matches taps OFF");
                    if (on_tokens.size() == off_tokens.size()) {
                        check(std::memcmp(on_tokens.data(), off_tokens.data(),
                                          on_tokens.size() * sizeof(int32_t)) == 0,
                              "sampled tokens match taps OFF");
                    }
                }
            }
            ex.value_trace = nullptr;

            for (uint32_t t = 0; t < tap_count; ++t) {
                auto vid_it = tap_value_id.find(t);
                if (vid_it == tap_value_id.end()) continue;
                auto cap_it = captured.find(vid_it->second);
                check(cap_it != captured.end(),
                      "layer output traced for tap " + std::to_string(t) +
                          " (n=" + std::to_string(n) + ")");
                if (cap_it == captured.end()) continue;
                const Captured& c = cap_it->second;
                check(c.rows == n, "traced rows for tap " + std::to_string(t) +
                                       " (n=" + std::to_string(n) + ")");
                const std::size_t row_bytes = static_cast<std::size_t>(c.feature_count) * 2u;
                const std::size_t tap_stride = static_cast<std::size_t>(hidden) * 2u;
                const auto* tap_base =
                    ex.dflash_target_hidden[t].data<ps::bf16_t>();
                const std::size_t rows = c.rows < n ? c.rows : n;
                std::vector<uint8_t> tap_host(rows * tap_stride);
                if (!tap_host.empty()) {
                    check(hipMemcpy(tap_host.data(), tap_base, tap_host.size(),
                                    hipMemcpyDeviceToHost) == hipSuccess,
                          "tap readback " + std::to_string(t));
                }
                for (std::size_t r = 0; r < rows; ++r) {
                    const uint8_t* want = c.bytes.data() + r * c.row_stride * 2u;
                    const uint8_t* got = tap_host.data() + r * tap_stride;
                    check(std::memcmp(want, got, row_bytes) == 0,
                          "tap " + std::to_string(t) + " row " + std::to_string(r) +
                              " matches layer output (n=" + std::to_string(n) + ")");
                }
            }

            (void)ps::state::release_paged_sequence_state(seq, stream);
        }

        const ps::Status st = ps::qwen35::executor_shutdown(ex);
        check(st.ok(), "executor shutdown (taps): " + st.message());
    }

    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    {
        const ps::Status shutdown = arena.shutdown();
        check(shutdown.ok(), "arena shutdown: " + shutdown.message());
    }

    std::printf("test_dflash2_target_taps: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
