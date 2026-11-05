#include <shelf/core/memory/arena.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/inference/state/paged_kv_pool.h>
#include <shelf/inference/state/paged_sequence_state.h>
#include <shelf/inference/state/sequence_slot_pool.h>
#include <shelf/models/loaded_model.h>
#include <shelf/models/qwen35/model/gdn_state_layout.h>
#include <shelf/models/qwen35/runtime/executor.h>
#include <shelf/models/qwen35/runtime/scheduling_types.h>
#include <shelf/models/qwen35/state/state_types.h>
#include <shelf/runtime/execution/row_bucket.h>

#include <hip/hip_runtime.h>

#include "moe_expert_fixture.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
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

inline float bf16_bits_to_f32(uint16_t v) {
    const uint32_t bits = static_cast<uint32_t>(v) << 16;
    float f = 0.0f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

q35::ScheduledBatch make_batch(st::PagedSequenceState& sequence,
                               std::vector<q35::ScheduledRequest>& requests,
                               const int32_t* tokens, uint32_t num_tokens,
                               uint32_t output_rows, rt::ExecutionClass exec_class) {
    q35::ScheduledRequest req;
    req.sequence = &sequence;
    req.handle = sequence.request_handle();
    req.execution_class = exec_class;
    req.token_begin = 0u;
    req.num_tokens = num_tokens;
    req.prefix_tokens = sequence.position;
    req.compute_logits = true;
    req.sample = false;
    req.num_output_rows = output_rows;
    requests.clear();
    requests.push_back(req);

    q35::ScheduledBatch batch;
    batch.token_ids = tokens;
    batch.requests = requests;
    batch.num_tokens = num_tokens;
    batch.num_requests = 1u;
    batch.num_decode_requests = exec_class == rt::ExecutionClass::DECODE ? 1u : 0u;
    batch.num_verify_requests = 0u;
    batch.num_prefill_requests = exec_class == rt::ExecutionClass::PREFILL ? 1u : 0u;
    return batch;
}

// One walk of the same prompt: chunks are the token counts of consecutive batches, and the
// hidden of every token row is kept in row order. A walk with all chunks of 1 row is the
// reference a multi-row walk has to reproduce row by row.
std::vector<uint16_t> run_walk(q35::Executor& ex, st::SequenceSlotPool& seq_pool,
                               st::GdnStatePool& gdn_pool, st::PagedKVPool& kv_pool,
                               hipStream_t stream, const std::vector<int32_t>& prompt,
                               const std::vector<uint32_t>& chunks, uint32_t max_tokens) {
    std::vector<uint16_t> rows;
    auto seq_result =
        st::create_paged_sequence_state(seq_pool, gdn_pool, kv_pool, max_tokens + 16u, stream);
    if (!seq_result.ok()) {
        std::printf("FAIL: sequence create: %s\n", seq_result.status().message().c_str());
        ++failed;
        return rows;
    }
    st::PagedSequenceState seq = seq_result.release();
    std::vector<q35::ScheduledRequest> requests;
    uint32_t off = 0u;
    for (const uint32_t len : chunks) {
        const bool decode = off > 0u && len == 1u;
        auto out = q35::execute_batch(
            ex, make_batch(seq, requests, prompt.data() + off, len, len,
                           decode ? rt::ExecutionClass::DECODE : rt::ExecutionClass::PREFILL),
            stream);
        if (!out.ok()) {
            std::printf("FAIL: walk chunk %u at %u: %s\n", len, off,
                        out.status().message().c_str());
            ++failed;
            (void)st::release_paged_sequence_state(seq, stream);
            return rows;
        }
        q35::BatchExecutionOutput o = out.release();
        if (o.token_hidden.data() == nullptr || o.token_hidden.ndim() < 2) {
            std::printf("FAIL: walk chunk %u at %u has no hidden rows\n", len, off);
            ++failed;
            break;
        }
        const std::size_t cols = static_cast<std::size_t>(o.token_hidden.dim(1));
        std::vector<uint16_t> h(static_cast<std::size_t>(len) * cols);
        if (hipMemcpy(h.data(), o.token_hidden.data(), h.size() * sizeof(uint16_t),
                      hipMemcpyDeviceToHost) != hipSuccess) {
            std::printf("FAIL: walk hidden readback\n");
            ++failed;
            break;
        }
        rows.insert(rows.end(), h.begin(), h.end());
        off += len;
    }
    (void)st::release_paged_sequence_state(seq, stream);
    return rows;
}

// Scale-invariant difference of every row: the fixture is an untrained model whose hidden
// magnitude grows with context, so absolute differences say nothing. Per-row rel_l2 says
// which row of the walk first moved.
void rel_l2_rows(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b,
                 std::size_t cols) {
    if (a.size() < cols || b.size() < cols || a.size() != b.size()) return;
    const std::size_t rows = a.size() / cols;
    std::string line;
    for (std::size_t r = 0; r < rows; ++r) {
        double diff_sq = 0.0;
        double ref_sq = 0.0;
        for (std::size_t c = 0; c < cols; ++c) {
            const float fa = bf16_bits_to_f32(a[r * cols + c]);
            const float fb = bf16_bits_to_f32(b[r * cols + c]);
            const float d = fa - fb;
            diff_sq += static_cast<double>(d) * static_cast<double>(d);
            ref_sq += static_cast<double>(fa) * static_cast<double>(fa);
        }
        const double rel = ref_sq > 0.0 ? std::sqrt(diff_sq / ref_sq) : 0.0;
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%s%.2e", r ? " " : "", rel);
        line += buf;
    }
    std::printf("  rel_l2 per row: %s\n", line.c_str());
}

double rel_l2_last_row(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b,
                       std::size_t cols) {
    if (a.size() < cols || b.size() < cols) return -1.0;
    const std::size_t base = (a.size() / cols) - 1u;
    double diff_sq = 0.0;
    double ref_sq = 0.0;
    double max_abs = 0.0;
    for (std::size_t c = 0; c < cols; ++c) {
        const float fa = bf16_bits_to_f32(a[base * cols + c]);
        const float fb = bf16_bits_to_f32(b[base * cols + c]);
        const float d = fa - fb;
        diff_sq += static_cast<double>(d) * static_cast<double>(d);
        ref_sq += static_cast<double>(fa) * static_cast<double>(fa);
        max_abs = std::max(max_abs, static_cast<double>(std::fabs(d)));
    }
    std::printf("  last row: rel_l2 %.3e, max_abs %.3e, |ref| %.3e\n",
                ref_sq > 0.0 ? std::sqrt(diff_sq / ref_sq) : 0.0, max_abs,
                std::sqrt(ref_sq));
    return ref_sq > 0.0 ? std::sqrt(diff_sq / ref_sq) : 0.0;
}

}  // namespace

void run_case_set(const shelf::models::LoadedModel& model, shelf::gpu::GpuArena& gpu_arena,
                  hipStream_t stream, const fs::path& dir, bool all_attention,
                  const std::vector<int32_t>& prompt) {
    const auto& tc = model.hybrid_config();
    const uint32_t hidden = static_cast<uint32_t>(tc.hidden_size);
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
    auto kv_pool_result = st::PagedKVPool::create(gpu_arena, num_pages, page_tokens, attn_layers,
                                                  kv_heads, head_dim,
                                                  shelf::kernel::KVCacheDType::BF16);
    check(seq_pool_result.ok() && gdn_pool_result.ok() && kv_pool_result.ok(),
          std::string(dir.filename()) + ": pool create");
    if (!seq_pool_result.ok() || !gdn_pool_result.ok() || !kv_pool_result.ok()) return;
    st::SequenceSlotPool seq_pool = seq_pool_result.release();
    st::GdnStatePool gdn_pool = gdn_pool_result.release();
    st::PagedKVPool kv_pool = kv_pool_result.release();

    q35::ExecutorConfig ec2;
    ec2.max_batch_rows = 16u;
    ec2.max_scheduled_requests = 1u;
    ec2.max_scheduled_output_rows = 16u;
    auto exec_result = model.create_executor(seq_pool, gdn_pool, kv_pool, gpu_arena, ec2);
    if (!exec_result.ok()) {
        std::printf("FAIL: %s create_executor: %s\n", dir.filename().c_str(),
                    exec_result.status().message().c_str());
        ++failed;
        return;
    }
    q35::Executor ex = exec_result.release();

    const uint32_t tokens = static_cast<uint32_t>(prompt.size());
    std::printf("-- %s (layers: %s, %zu attn of %zu) --\n", dir.filename().c_str(),
                all_attention ? "all full attention" : "GDN + full attention",
                static_cast<std::size_t>(attn_layers), tc.num_hidden_layers);
    const std::vector<uint32_t> one_row(tokens, 1u);
    const std::vector<uint16_t> reference =
        run_walk(ex, seq_pool, gdn_pool, kv_pool, stream, prompt, one_row, max_tokens);
    check(reference.size() == static_cast<std::size_t>(tokens) * hidden,
          std::string(dir.filename()) + ": the 1-row walk produced one hidden per token row");

    struct Case {
        const char* tag;
        std::vector<uint32_t> chunks;
    };
    const Case cases[] = {
        {"4+4", {4u, 4u}},
        {"2x4", {2u, 2u, 2u, 2u}},
        {"8", {8u}},
        {"3+5", {3u, 5u}},
        {"1+7", {1u, 7u}},
        {"2+6", {2u, 6u}},
        {"5+3", {5u, 3u}},
        {"3+3+2", {3u, 3u, 2u}},
        {"2+2+2+2 dec tail", {2u, 2u, 2u, 2u}},
        {"1+1+6", {1u, 1u, 6u}},
        {"4 then 1x4", {4u, 1u, 1u, 1u, 1u}},
        {"1x4 then 4", {1u, 1u, 1u, 1u, 4u}},
    };
    for (const auto& c : cases) {
        std::printf("%s (chunks over %u tokens)\n", c.tag, tokens);
        const std::vector<uint16_t> got =
            run_walk(ex, seq_pool, gdn_pool, kv_pool, stream, prompt, c.chunks, max_tokens);
        if (got.size() != reference.size()) {
            std::printf("  rows: got %zu, reference %zu\n", got.size() / hidden,
                        reference.size() / hidden);
            check(false, std::string(c.tag) + ": walk produced a different row count");
            continue;
        }
        rel_l2_rows(reference, got, hidden);
        const double r = rel_l2_last_row(reference, got, hidden);
        check(r >= 0.0 && r < 5e-2, std::string(c.tag) + ": the last row moves with batching");
    }
    (void)q35::executor_shutdown(ex);
}

int main() {
    std::printf("=== qwen35moe row-count parity (GPU) ===\n");
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
        fs::temp_directory_path(ec) / ("shelf_moe_rows_" + std::to_string(::getpid()));
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

    Fixture a;
    if (!write_fixture(a, root / "moe_attn", false, true, kInter, kInter, false, true)) {
        std::printf("FAIL: all-attention moe fixture write\n");
        return 1;
    }

    const uint32_t tokens = 8u;
    std::vector<int32_t> prompt(tokens);
    for (uint32_t i = 0; i < tokens; ++i)
        prompt[i] = static_cast<int32_t>((i * 91u + 7u) % kVocab);

    for (const char* name : {"moe", "moe_attn"}) {
        auto loaded = shelf::models::LoadedModel::load((root / name).string(), gpu_arena, stream);
        if (!loaded.ok()) {
            std::printf("FAIL: LoadedModel::load %s: %s\n", name,
                        loaded.status().message().c_str());
            ++failed;
            continue;
        }
        shelf::models::LoadedModel model = loaded.release();
        run_case_set(model, gpu_arena, stream, root / name, std::string(name) == "moe_attn",
                     prompt);
    }

    fs::remove_all(root, ec);
    gpu_arena.shutdown();
    if (hipStreamDestroy(stream) != hipSuccess)
        std::printf("WARN: stream destroy\n");

    std::printf("test_qwen35moe_rows_parity: %s (fail=%d)\n", failed == 0 ? "PASS" : "FAIL",
                failed);
    return failed == 0 ? 0 : 1;
}
