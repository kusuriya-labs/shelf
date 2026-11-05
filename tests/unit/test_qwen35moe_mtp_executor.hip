#include <shelf/core/memory/arena.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/inference/state/paged_kv_pool.h>
#include <shelf/inference/state/paged_sequence_state.h>
#include <shelf/inference/state/sequence_slot_pool.h>
#include <shelf/models/qwen35/model/gdn_state_layout.h>
#include <shelf/models/qwen35/runtime/executor.h>
#include <shelf/models/qwen35/runtime/scheduling_types.h>
#include <shelf/models/qwen35moe/model/lower_to_primitives.h>
#include <shelf/models/qwen35moe/weights/qwen35moe_model_load.h>
#include <shelf/runtime/execution/row_bucket.h>

#include <hip/hip_runtime.h>

#include "moe_expert_fixture.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
namespace moe = shelf::qwen35moe;
namespace q35 = shelf::qwen35;
namespace rt = shelf::runtime;
namespace st = shelf::state;
using namespace moe_expert_fixture;

namespace {

int failed = 0;

void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++failed;
        std::printf("FAIL: %s\n", msg.c_str());
    }
}

uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

q35::ScheduledBatch make_batch(st::PagedSequenceState& sequence,
                               std::vector<q35::ScheduledRequest>& requests,
                               const int32_t* tokens, uint32_t num_tokens,
                               uint32_t output_rows) {
    q35::ScheduledRequest req;
    req.sequence = &sequence;
    req.handle = sequence.request_handle();
    req.execution_class = num_tokens == 1u ? rt::ExecutionClass::DECODE
                                           : rt::ExecutionClass::PREFILL;
    req.token_begin = 0u;
    req.num_tokens = num_tokens;
    req.prefix_tokens = sequence.position;
    req.compute_logits = true;
    req.sample = true;
    req.num_output_rows = output_rows;
    requests.clear();
    requests.push_back(req);

    q35::ScheduledBatch batch;
    batch.token_ids = tokens;
    batch.requests = requests;
    batch.num_tokens = num_tokens;
    batch.num_requests = 1u;
    batch.num_decode_requests = num_tokens == 1u ? 1u : 0u;
    batch.num_verify_requests = 0u;
    batch.num_prefill_requests = num_tokens == 1u ? 0u : 1u;
    return batch;
}

struct Round {
    std::vector<uint16_t> hidden;
    std::vector<int32_t> sampled;
    bool ok = false;
};

// One prefill and two decode rows through the drafter graph. The hidden input stands for
// the target's normed token hidden, and it is the same bytes every run, so two runs differ
// only where the graph's own weights reach the output.
Round run_rounds(q35::Executor& ex, st::PagedSequenceState& seq, hipStream_t stream,
                 const std::vector<int32_t>& prompt) {
    Round round;
    std::vector<q35::ScheduledRequest> requests;
    std::vector<int32_t> sampled;

    const auto call = [&](const std::vector<int32_t>& tokens, const char* tag) {
        auto out = q35::execute_batch(ex, make_batch(seq, requests, tokens.data(),
                                                     static_cast<uint32_t>(tokens.size()), 1u),
                                      stream);
        if (!out.ok()) {
            std::printf("FAIL: %s: %s\n", tag, out.status().message().c_str());
            ++failed;
            return q35::BatchExecutionOutput{};
        }
        return out.release();
    };

    {
        q35::BatchExecutionOutput pre = call(prompt, "mtp prefill");
        if (pre.token_hidden.data() == nullptr) return round;
        const std::size_t cols = static_cast<std::size_t>(pre.token_hidden.dim(1));
        round.hidden.assign(cols, 0u);
        if (hipMemcpy(round.hidden.data(), pre.token_hidden.data(), cols * sizeof(uint16_t),
                      hipMemcpyDeviceToHost) != hipSuccess) {
            std::printf("FAIL: mtp prefill hidden readback\n");
            ++failed;
            return round;
        }
        int32_t token = -1;
        if (pre.sampled_tokens.ndim() > 0 && pre.sampled_tokens.dim(0) > 0) {
            if (hipMemcpy(&token, pre.sampled_tokens.data<int32_t>(), sizeof(int32_t),
                          hipMemcpyDeviceToHost) != hipSuccess) {
                std::printf("FAIL: mtp prefill token readback\n");
                ++failed;
                return round;
            }
        }
        sampled.push_back(token);
    }
    for (int step = 0; step < 2; ++step) {
        const std::vector<int32_t> one{sampled.back()};
        q35::BatchExecutionOutput out = call(one, "mtp decode");
        if (out.token_hidden.data() == nullptr) return round;
        const std::size_t cols = static_cast<std::size_t>(out.token_hidden.dim(1));
        std::vector<uint16_t> h(cols);
        if (hipMemcpy(h.data(), out.token_hidden.data(), cols * sizeof(uint16_t),
                      hipMemcpyDeviceToHost) != hipSuccess) {
            std::printf("FAIL: mtp decode hidden readback\n");
            ++failed;
            return round;
        }
        round.hidden.insert(round.hidden.end(), h.begin(), h.end());
        int32_t token = -1;
        if (out.sampled_tokens.ndim() > 0 && out.sampled_tokens.dim(0) > 0) {
            if (hipMemcpy(&token, out.sampled_tokens.data<int32_t>(), sizeof(int32_t),
                          hipMemcpyDeviceToHost) != hipSuccess) {
                std::printf("FAIL: mtp decode token readback\n");
                ++failed;
                return round;
            }
        }
        sampled.push_back(token);
    }
    round.sampled = std::move(sampled);
    round.ok = true;
    return round;
}

void fill_hidden(q35::Executor& ex, uint32_t rows, uint32_t hidden) {
    std::vector<uint16_t> host(static_cast<std::size_t>(rows) * hidden);
    for (std::size_t i = 0; i < host.size(); ++i) {
        const float value = 0.02f * static_cast<float>((i % 17u) + 1u) - 0.15f;
        host[i] = shelf::f32_to_bf16_rne(value);
    }
    const hipError_t err = hipMemcpy(ex.input_hidden.data<void>(), host.data(),
                                     host.size() * sizeof(uint16_t), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::printf("FAIL: hidden input copy: %s\n", hipGetErrorString(err));
        ++failed;
    }
}

struct PlaneBytes {
    std::vector<uint8_t> codes;
    std::vector<uint8_t> scales;
    std::vector<uint8_t> row_ref;
};

PlaneBytes read_plane(const shelf::weights::MatrixWeight& w) {
    PlaneBytes p;
    p.codes.resize(static_cast<std::size_t>(w.codes_batch_stride_bytes) * w.experts);
    p.scales.resize(static_cast<std::size_t>(w.scales_batch_stride_bytes) * w.experts);
    p.row_ref.resize(static_cast<std::size_t>(w.experts) * w.rows);
    hipMemcpy(p.codes.data(), w.codes.data<void>(), p.codes.size(), hipMemcpyDeviceToHost);
    hipMemcpy(p.scales.data(), w.scales.data<void>(), p.scales.size(), hipMemcpyDeviceToHost);
    hipMemcpy(p.row_ref.data(), w.row_ref.data<void>(), p.row_ref.size(),
              hipMemcpyDeviceToHost);
    return p;
}

void zero_plane(const shelf::weights::MatrixWeight& w) {
    hipMemset(w.codes.data<void>(), 0, static_cast<std::size_t>(w.codes_batch_stride_bytes) * w.experts);
    hipMemset(w.scales.data<void>(), 0, static_cast<std::size_t>(w.scales_batch_stride_bytes) * w.experts);
    hipMemset(w.row_ref.data<void>(), 0, static_cast<std::size_t>(w.experts) * w.rows);
}

void restore_plane(const shelf::weights::MatrixWeight& w, const PlaneBytes& p) {
    hipMemcpy(w.codes.data<void>(), p.codes.data(), p.codes.size(), hipMemcpyHostToDevice);
    hipMemcpy(w.scales.data<void>(), p.scales.data(), p.scales.size(), hipMemcpyHostToDevice);
    hipMemcpy(w.row_ref.data<void>(), p.row_ref.data(), p.row_ref.size(), hipMemcpyHostToDevice);
}

void zero_bf16_matrix(const shelf::weights::MatrixWeight& w) {
    const shelf::gpu::Tensor& payload = w.data.ndim() > 0 ? w.data : w.bf16;
    if (payload.ndim() == 0) return;
    std::size_t bytes = 0;
    if (!payload.required_span_bytes(bytes).ok() || bytes == 0u) return;
    hipMemset(payload.data<void>(), 0, bytes);
}

bool hidden_differs(const Round& a, const Round& b, const std::string& tag) {
    if (a.hidden.size() != b.hidden.size() || a.hidden.empty()) return false;
    std::size_t changed = 0;
    for (std::size_t i = 0; i < a.hidden.size(); ++i)
        if (a.hidden[i] != b.hidden[i]) ++changed;
    if (changed == 0) {
        std::printf("FAIL: %s left every hidden byte the same\n", tag.c_str());
        return false;
    }
    return true;
}

bool hidden_usable(const Round& r, const std::string& tag) {
    if (!r.ok) return false;
    bool all_zero = true;
    for (const uint16_t v : r.hidden) {
        if (((v >> 7u) & 0xffu) == 0xffu) {
            std::printf("FAIL: %s hidden is not finite\n", tag.c_str());
            return false;
        }
        if (v != 0u) all_zero = false;
    }
    if (all_zero) {
        std::printf("FAIL: %s hidden is all zeros\n", tag.c_str());
        return false;
    }
    return true;
}

}  // namespace

int main() {
    std::printf("=== qwen35moe MTP graph through the executor (GPU) ===\n");
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    std::error_code ec;
    const fs::path root =
        fs::temp_directory_path(ec) / ("shelf_moe_mtp_executor_" + std::to_string(::getpid()));
    fs::remove_all(root, ec);

    Fixture f;
    if (!write_fixture(f, root / "moe_mtp", false, true, kInter, kInter, true)) {
        std::printf("FAIL: moe mtp fixture write\n");
        return 1;
    }

    auto arena = shelf::gpu::GpuArena::create(0, 2ull * 1024ull * 1024ull * 1024ull);
    if (!arena.ok()) {
        std::printf("FAIL: arena create\n");
        return 1;
    }
    shelf::gpu::GpuArena gpu_arena = arena.release();

    moe::Qwen35MoeLoadOptions load_options;
    load_options.load_mtp_layers = true;
    auto loaded = moe::load_qwen35moe_weights((root / "moe_mtp").string(), gpu_arena, stream,
                                              load_options);
    if (!loaded.ok()) {
        std::printf("FAIL: load: %s\n", loaded.status().message().c_str());
        return 1;
    }
    moe::Qwen35MoeModelWeights& weights = loaded.value();

    auto lowered = moe::lower_qwen35_moe_mtp_to_primitives(f.config, weights);
    if (!lowered.ok()) {
        std::printf("FAIL: lower: %s\n", lowered.status().message().c_str());
        return 1;
    }

    moe::Qwen35MoeTextConfig mtp_config = f.config;
    mtp_config.num_hidden_layers = 1;
    mtp_config.layer_types = {1};
    mtp_config.full_attention_interval = 1;

    const uint32_t hidden = static_cast<uint32_t>(f.config.hidden_size);
    const uint32_t kv_heads = static_cast<uint32_t>(f.config.num_key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(f.config.attention_head_dim);
    const uint32_t page_tokens = 16u;
    const uint32_t num_pages = ceil_div(128u + 16u, page_tokens) + 8u;

    const uint32_t rows = 16u;
    Round rounds[3];
    const char* tags[] = {"as loaded", "routed down plane zeroed",
                          "shared expert down plane zeroed"};
    PlaneBytes w2_saved = read_plane(weights.mtp.layer.ffn.expert_w2);

    for (uint32_t variant = 0; variant < 3u; ++variant) {
        auto seq_pool_result = st::SequenceSlotPool::create(gpu_arena, 1u, num_pages + 8u);
        auto gdn_pool_result =
            st::GdnStatePool::create(gpu_arena, 1u, q35::make_qwen35_gdn_state_layout(mtp_config));
        auto kv_pool_result = st::PagedKVPool::create(gpu_arena, num_pages, page_tokens, 1u,
                                                      kv_heads, head_dim,
                                                      shelf::kernel::KVCacheDType::BF16);
        if (!seq_pool_result.ok() || !gdn_pool_result.ok() || !kv_pool_result.ok()) {
            std::printf("FAIL: pool create\n");
            ++failed;
            break;
        }
        st::SequenceSlotPool seq_pool = seq_pool_result.release();
        st::GdnStatePool gdn_pool = gdn_pool_result.release();
        st::PagedKVPool kv_pool = kv_pool_result.release();

        auto seq_result = st::create_paged_sequence_state(seq_pool, gdn_pool, kv_pool, 128u,
                                                          stream);
        if (!seq_result.ok()) {
            std::printf("FAIL: sequence create: %s\n", seq_result.status().message().c_str());
            ++failed;
            break;
        }
        st::PagedSequenceState seq = seq_result.release();

        q35::ExecutorModelInput input;
        input.model.config = &mtp_config;
        input.model.embed_tokens = &weights.embed_tokens;
        input.lowered = lowered.value();

        q35::ExecutorConfig ec2;
        ec2.max_batch_rows = rows;
        ec2.max_scheduled_requests = 1u;
        ec2.max_scheduled_output_rows = 1u;
        ec2.external_hidden_input = true;
        auto exec_result = q35::create_executor(std::move(input), seq_pool, gdn_pool, kv_pool,
                                                gpu_arena, ec2);
        if (!exec_result.ok()) {
            std::printf("FAIL: create_executor: %s\n", exec_result.status().message().c_str());
            ++failed;
            break;
        }
        q35::Executor ex = exec_result.release();

        if (variant == 1u) zero_plane(weights.mtp.layer.ffn.expert_w2);
        if (variant == 2u) {
            restore_plane(weights.mtp.layer.ffn.expert_w2, w2_saved);
            zero_bf16_matrix(weights.mtp.layer.ffn.shared_down_proj);
        }

        std::vector<int32_t> prompt(1);
        for (uint32_t i = 0; i < prompt.size(); ++i)
            prompt[i] = static_cast<int32_t>((i * 91u + 7u) % kVocab);
        fill_hidden(ex, rows, hidden);
        rounds[variant] = run_rounds(ex, seq, stream, prompt);
        if (rounds[variant].ok) {
            check(hidden_usable(rounds[variant], tags[variant]),
                  std::string(tags[variant]) + " produces a usable hidden");
        }
        (void)q35::executor_shutdown(ex);
        (void)st::release_paged_sequence_state(seq, stream);
    }

    if (rounds[0].ok && rounds[1].ok)
        check(hidden_differs(rounds[0], rounds[1], "zeroing the routed expert down plane"),
              "the routed experts of the MTP block reach the block output");
    if (rounds[0].ok && rounds[2].ok)
        check(hidden_differs(rounds[0], rounds[2], "zeroing the shared expert down plane"),
              "the shared expert of the MTP block reaches the block output");

    fs::remove_all(root, ec);
    gpu_arena.shutdown();
    if (hipStreamDestroy(stream) != hipSuccess)
        std::printf("WARN: stream destroy\n");

    std::printf("test_qwen35moe_mtp_executor: %s (fail=%d)\n", failed == 0 ? "PASS" : "FAIL",
                failed);
    return failed == 0 ? 0 : 1;
}
