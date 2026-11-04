#include <shelf/core/memory/arena.h>
#include <shelf/models/loaded_model.h>
#include <shelf/models/qwen35/model/gdn_state_layout.h>
#include <shelf/models/qwen35/runtime/executor.h>
#include <shelf/models/qwen35/runtime/scheduling_types.h>
#include <shelf/models/qwen35/state/state_types.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/inference/execution/decode_perf_stats.h>
#include <shelf/runtime/execution/row_bucket.h>

#include <hip/hip_runtime.h>

#include "moe_expert_fixture.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
    // The batch has to continue where the sequence stands, which the prefill moved.
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

// A MoE path that loses its plan shows up as a non-finite hidden row, and a backbone that
// produced nothing shows up as an all-zero hidden. Both leave the sampler answering one fixed
// token, so a round is checked on its own before it is compared with another round.
bool hidden_finite(const q35::BatchExecutionOutput& out, const std::string& tag) {
    if (out.num_token_rows == 0u || out.token_hidden.data() == nullptr ||
        out.token_hidden.ndim() < 2)
        return true;
    const std::size_t cols = static_cast<std::size_t>(out.token_hidden.dim(1));
    std::vector<uint16_t> h(static_cast<std::size_t>(out.num_token_rows) * cols);
    if (hipMemcpy(h.data(), out.token_hidden.data(), h.size() * sizeof(uint16_t),
                  hipMemcpyDeviceToHost) != hipSuccess) {
        std::printf("FAIL: %s hidden readback\n", tag.c_str());
        ++failed;
        return false;
    }
    uint32_t bad = 0u;
    uint16_t max_abs = 0u;
    for (const uint16_t v : h) {
        if (((v >> 7u) & 0xffu) == 0xffu) ++bad;
        const uint16_t a = static_cast<uint16_t>(v & 0x7fffu);
        if (a > max_abs) max_abs = a;
    }
    if (bad != 0u) {
        std::printf("FAIL: %s hidden has %u non-finite rows\n", tag.c_str(), bad);
        ++failed;
        return false;
    }
    // A hidden that is all zeros means the backbone or the FFN produced nothing, and the
    // sampler turns that into one fixed token for every step.
    if (max_abs == 0u) {
        std::printf("FAIL: %s hidden is all zeros\n", tag.c_str());
        ++failed;
        return false;
    }
    return true;
}

// Runs one prefill and one decode step at a given row budget and returns the sampled
// tokens. `max_batch_rows` selects the row bucket, which is what sets the group row
// capacity of the plan, so the bucket is part of the case, not a detail.
bool run_rounds(q35::Executor& ex, st::PagedSequenceState& seq, hipStream_t stream,
                const std::vector<int32_t>& prompt, std::vector<int32_t>& sampled,
                const std::string& tag) {
    std::vector<q35::ScheduledRequest> requests;

    auto prefill = q35::execute_batch(ex, make_batch(seq, requests, prompt.data(),
                                                     static_cast<uint32_t>(prompt.size()), 1u),
                                      stream);
    if (!prefill.ok()) {
        std::printf("FAIL: %s prefill: %s\n", tag.c_str(), prefill.status().message().c_str());
        ++failed;
        return false;
    }
    q35::BatchExecutionOutput pre = prefill.release();
    if (!hidden_finite(pre, tag + " prefill")) return false;
    std::vector<int32_t> first(1, 0);
    if (pre.sampled_tokens.ndim() > 0 && pre.sampled_tokens.dim(0) > 0) {
        if (hipMemcpy(first.data(), pre.sampled_tokens.data<int32_t>(), sizeof(int32_t),
                      hipMemcpyDeviceToHost) != hipSuccess) {
            std::printf("FAIL: %s prefill token readback\n", tag.c_str());
            ++failed;
            return false;
        }
    }
    sampled.push_back(first[0]);

    for (int step = 0; step < 2; ++step) {
        std::vector<int32_t> one{sampled.back()};
        auto decode = q35::execute_batch(ex, make_batch(seq, requests, one.data(), 1u, 1u),
                                         stream);
        if (!decode.ok()) {
            std::printf("FAIL: %s decode %d: %s\n", tag.c_str(), step,
                        decode.status().message().c_str());
            ++failed;
            return false;
        }
        q35::BatchExecutionOutput out = decode.release();
        if (!hidden_finite(out, tag + " decode " + std::to_string(step))) return false;
        int32_t next = 0;
        if (out.sampled_tokens.ndim() > 0 && out.sampled_tokens.dim(0) > 0) {
            if (hipMemcpy(&next, out.sampled_tokens.data<int32_t>(), sizeof(int32_t),
                          hipMemcpyDeviceToHost) != hipSuccess) {
                std::printf("FAIL: %s decode token readback\n", tag.c_str());
                ++failed;
                return false;
            }
        }
        sampled.push_back(next);
    }
    return true;
}

}  // namespace

int main() {
    std::printf("=== qwen35moe through the executor (GPU) ===\n");
    ::setenv("SHELF_QWEN35_PERF_STATS", "1", 1);
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
        fs::temp_directory_path(ec) / ("shelf_moe_executor_" + std::to_string(::getpid()));
    fs::remove_all(root, ec);

    Fixture f;
    if (!write_fixture(f, root / "moe", false, true)) {
        std::printf("FAIL: moe model fixture write\n");
        return 1;
    }

    auto arena = shelf::gpu::GpuArena::create(0, 2ull * 1024ull * 1024ull * 1024ull);
    if (!arena.ok()) {
        std::printf("FAIL: arena create\n");
        return 1;
    }
    shelf::gpu::GpuArena gpu_arena = arena.release();

    auto loaded = shelf::models::LoadedModel::load((root / "moe").string(), gpu_arena, stream);
    if (!loaded.ok()) {
        std::printf("FAIL: LoadedModel::load: %s\n", loaded.status().message().c_str());
        return 1;
    }
    shelf::models::LoadedModel model = loaded.release();
    check(model.runs_grouped_experts(), "the fixture loads as a grouped-expert model");
    check(model.dense_model() == nullptr, "a grouped-expert model has no dense model to reach");

    const auto& tc = model.hybrid_config();
    const uint32_t kv_heads = static_cast<uint32_t>(tc.num_key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(tc.attention_head_dim);
    uint32_t attn_layers = 0;
    for (int lt : tc.layer_types)
        if (lt != 0) ++attn_layers;
    const uint32_t page_tokens = 16u;
    const uint32_t max_tokens = 128u;
    const uint32_t num_pages = ceil_div(max_tokens + 16u, page_tokens) + 8u;

    auto seq_pool_result = st::SequenceSlotPool::create(gpu_arena, 2u, num_pages + 8u);
    auto gdn_pool_result =
        st::GdnStatePool::create(gpu_arena, 2u, q35::make_qwen35_gdn_state_layout(tc));
    auto kv_pool_result = st::PagedKVPool::create(gpu_arena, num_pages, page_tokens,
                                                  attn_layers, kv_heads, head_dim,
                                                  shelf::kernel::KVCacheDType::BF16);
    check(seq_pool_result.ok() && gdn_pool_result.ok() && kv_pool_result.ok(), "pool create");
    if (!seq_pool_result.ok() || !gdn_pool_result.ok() || !kv_pool_result.ok()) return 1;
    st::SequenceSlotPool seq_pool = seq_pool_result.release();
    st::GdnStatePool gdn_pool = gdn_pool_result.release();
    st::PagedKVPool kv_pool = kv_pool_result.release();

    auto make_prompt = [](uint32_t count) {
        std::vector<int32_t> ids(count);
        for (uint32_t i = 0; i < count; ++i) ids[i] = static_cast<int32_t>((i * 37u + 11u) % kVocab);
        return ids;
    };

    // The plan capacity grows with tokens * top_k, so the bucket decides how many group
    // rows the program allocates. Both a small and a larger bucket have to run.
    const uint32_t buckets[] = {16u, 64u, 128u};
    for (uint32_t rows : buckets) {
        const std::string tag = "bucket " + std::to_string(rows);
        q35::ExecutorConfig ec2;
        ec2.max_batch_rows = rows;
        ec2.max_scheduled_requests = 1u;
        ec2.max_scheduled_output_rows = 1u;
        auto exec_result =
            model.create_executor(seq_pool, gdn_pool, kv_pool, gpu_arena, ec2);
        if (!exec_result.ok()) {
            std::printf("FAIL: %s create_executor: %s\n", tag.c_str(),
                        exec_result.status().message().c_str());
            ++failed;
            continue;
        }
        q35::Executor ex = exec_result.release();
        const auto bucket = rt::resolve_row_bucket(rows);
        const rt::Program* program = nullptr;
        for (const auto& p : ex.program_set.programs)
            if (bucket.ok() && p.row_bucket == bucket.value()) program = &p;
        check(program != nullptr && program->plan_scratch_slot != rt::kNoWorkspace,
              tag + ": the program for this bucket carries a plan");

        auto seq_result =
            st::create_paged_sequence_state(seq_pool, gdn_pool, kv_pool, max_tokens + 16u, stream);
        if (!seq_result.ok()) {
            std::printf("FAIL: %s sequence create: %s\n", tag.c_str(),
                        seq_result.status().message().c_str());
            ++failed;
            continue;
        }
        st::PagedSequenceState seq = seq_result.release();

        // A batch cannot exceed the executor's row budget, so the prompt is the bucket.
        const std::vector<int32_t> prompt = make_prompt(rows);
        std::vector<int32_t> sampled;
        const uint64_t swiglu_before =
            shelf::execution::perf_stats_instance()
                .kernels[static_cast<std::size_t>(rt::KernelId::SWIGLU)]
                .fallback;
        if (!run_rounds(ex, seq, stream, prompt, sampled, tag)) {
            (void)st::release_paged_sequence_state(seq, stream);
            continue;
        }
        check(sampled.size() == 3u, tag + ": prefill and two decode steps sampled");
        // The FFN elementwise runs on group rows, and the correctness interpreter counts
        // rows in tokens, so a grouped dispatch that fell back would process part of the
        // plane. None of them may fall back.
        check(swiglu_before ==
                  shelf::execution::perf_stats_instance()
                      .kernels[static_cast<std::size_t>(rt::KernelId::SWIGLU)]
                      .fallback,
              tag + ": no grouped swiglu dispatch took the correctness path");
        bool in_vocab = sampled.size() == 3u;
        for (int32_t t : sampled)
            if (t < 0 || static_cast<uint32_t>(t) >= kVocab) in_vocab = false;
        check(in_vocab, tag + ": sampled tokens are inside the vocab");
        std::printf("      %s sampled: %d %d %d\n", tag.c_str(), sampled[0], sampled[1],
                    sampled[2]);

        // The same rounds on a fresh sequence must give the same tokens: a routing round may
        // not leave counts for the next round, inside one run or across runs. The first
        // sequence is released first, so the pool has its pages back.
        (void)st::release_paged_sequence_state(seq, stream);
        std::vector<int32_t> again;
        auto seq2_result =
            st::create_paged_sequence_state(seq_pool, gdn_pool, kv_pool, max_tokens + 16u, stream);
        if (seq2_result.ok()) {
            st::PagedSequenceState seq2 = seq2_result.release();
            if (run_rounds(ex, seq2, stream, prompt, again, tag + " (second pass)"))
                check(again == sampled, tag + ": the same rounds give the same tokens");
            (void)st::release_paged_sequence_state(seq2, stream);
        } else {
            check(false, tag + ": second sequence create");
        }
    }

    // A MoE intermediate width outside the elementwise envelope cannot run through the
    // interpreter (it would process token rows over a group plane), so the executor has to
    // report it instead of producing values.
    {
        Fixture bad;
        if (!write_fixture(bad, root / "moe_unregistered", false, true, kInter, 384u)) {
            std::printf("FAIL: unregistered fixture write\n");
            ++failed;
        } else {
            auto bad_loaded = shelf::models::LoadedModel::load(
                (root / "moe_unregistered").string(), gpu_arena, stream);
            if (!bad_loaded.ok()) {
                std::printf("FAIL: unregistered fixture load: %s\n",
                            bad_loaded.status().message().c_str());
                ++failed;
            } else {
                shelf::models::LoadedModel bad_model = bad_loaded.release();
                q35::ExecutorConfig ec3;
                ec3.max_batch_rows = 16u;
                ec3.max_scheduled_requests = 1u;
                ec3.max_scheduled_output_rows = 1u;
                auto bad_exec = bad_model.create_executor(seq_pool, gdn_pool, kv_pool, gpu_arena, ec3);
                if (!bad_exec.ok()) {
                    std::printf("FAIL: unregistered create_executor: %s\n",
                                bad_exec.status().message().c_str());
                    ++failed;
                } else {
                    q35::Executor bad_ex = bad_exec.release();
                    auto bad_seq = st::create_paged_sequence_state(
                        seq_pool, gdn_pool, kv_pool, max_tokens + 16u, stream);
                    if (!bad_seq.ok()) {
                        std::printf("FAIL: unregistered sequence create\n");
                        ++failed;
                    } else {
                        st::PagedSequenceState bseq = bad_seq.release();
                        const std::vector<int32_t> prompt = make_prompt(16u);
                        std::vector<q35::ScheduledRequest> requests;
                        auto prefill = q35::execute_batch(
                            bad_ex, make_batch(bseq, requests, prompt.data(), 16u, 1u), stream);
                        check(!prefill.ok(),
                              "a group geometry outside the envelope is refused, not run");
                        if (!prefill.ok()) {
                            const std::string msg = prefill.status().message();
                            check(msg.find("grouped-row dispatch") != std::string::npos,
                                  "the refusal names the grouped row domain");
                        }
                        (void)st::release_paged_sequence_state(bseq, stream);
                    }
                }
            }
        }
    }

    fs::remove_all(root, ec);
    std::printf("%s (failed=%d)\n", failed == 0 ? "PASS" : "FAIL", failed);
    return failed == 0 ? 0 : 1;
}
