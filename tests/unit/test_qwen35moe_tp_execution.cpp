#include <shelf/core/gpu/scoped_device.h>
#include <shelf/core/memory/arena.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/models/qwen35/model/gdn_state_layout.h>
#include <shelf/models/qwen35/runtime/executor.h>
#include <shelf/models/qwen35/runtime/scheduling_types.h>
#include <shelf/models/qwen35/state/state_types.h>
#include <shelf/models/qwen35moe/model/lower_to_primitives.h>
#include <shelf/models/qwen35moe/model/qwen35moe_model.h>
#include <shelf/models/qwen35moe/model/tensor_parallel_context.h>
#include <shelf/models/qwen35moe/weights/moe_tensor_parallel_plan.h>
#include <shelf/models/qwen35moe/weights/qwen35moe_model_load.h>
#include <shelf/runtime/tp/tp_barrier_group.h>
#include <shelf/runtime/tp/tp_transports.h>

#include <hip/hip_runtime.h>

#include "moe_expert_fixture.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
namespace moe = shelf::qwen35moe;
namespace q35 = shelf::qwen35;
namespace rt = shelf::runtime;
namespace st = shelf::state;
using namespace moe_expert_fixture;

namespace {

int g_fail = 0;

void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

std::vector<int32_t> make_prompt(uint32_t count) {
    std::vector<int32_t> ids(count);
    for (uint32_t i = 0; i < count; ++i)
        ids[i] = static_cast<int32_t>((i * 37u + 11u) % kVocab);
    return ids;
}

q35::ScheduledBatch make_batch(st::PagedSequenceState& sequence,
                               std::vector<q35::ScheduledRequest>& requests,
                               const int32_t* tokens, uint32_t num_tokens) {
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
    req.num_output_rows = 1u;
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

void download_row0(const shelf::gpu::Tensor& t, int device, std::vector<float>& out) {
    out.clear();
    if (t.ndim() < 2 || t.data() == nullptr) return;
    const std::size_t features = static_cast<std::size_t>(t.dim(1));
    const std::size_t elem = t.element_size();
    std::vector<std::uint8_t> host(features * elem);
    if (hipSetDevice(device) != hipSuccess) return;
    if (hipMemcpy(host.data(), t.data(), host.size(), hipMemcpyDeviceToHost) != hipSuccess)
        return;
    out.resize(features);
    if (elem == 2) {
        const auto* u = reinterpret_cast<const std::uint16_t*>(host.data());
        for (std::size_t i = 0; i < features; ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(u[i]) << 16;
            float f = 0.0f;
            std::memcpy(&f, &bits, sizeof(f));
            out[i] = f;
        }
    } else if (elem == 4) {
        const float* f = reinterpret_cast<const float*>(host.data());
        for (std::size_t i = 0; i < features; ++i) out[i] = f[i];
    }
}

struct RunResult {
    bool ok = false;
    std::string error;
    std::vector<int32_t> tokens;
    std::vector<float> hidden_row0;
    std::vector<float> logits_row0;
    std::uint32_t plane_experts = 0;
    std::uint32_t shared_rows = 0;
    std::uint32_t kv_heads = 0;
    std::uint32_t router_rows = 0;
};

// One device of a tp_size split, or the whole model when tp_size is 1. The rank loads its
// own weights through its partition plan, lowers with its own expert range, and runs one
// prefill and two decode steps. The barrier is shared, so a split run has to be driven on
// every rank at the same time.
RunResult run_rank(const std::string& model_dir, int device, std::uint32_t tp_size,
                   std::uint32_t tp_rank, rt::TpBarrierGroup* barrier) {
    RunResult result;
    auto scope = shelf::gpu::ScopedDevice::create(device);
    if (!scope.ok()) {
        result.error = "scoped device: " + scope.status().message();
        return result;
    }
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        result.error = "hipStreamCreate";
        return result;
    }
    auto arena_result = shelf::gpu::GpuArena::create(device, 2ull * 1024ull * 1024ull * 1024ull);
    if (!arena_result.ok()) {
        result.error = "arena create";
        return result;
    }
    shelf::gpu::GpuArena arena = arena_result.release();

    auto config_result = moe::read_qwen35_moe_text_config(model_dir);
    if (!config_result.ok()) {
        result.error = "config: " + config_result.status().message();
        return result;
    }
    const moe::Qwen35MoeTextConfig config = config_result.release();

    moe::Qwen35MoeLoadOptions load_options;
    shelf::weights::TensorPartitionPlan plan;
    if (tp_size > 1u) {
        auto plan_result = moe::build_qwen35moe_tensor_partition_plan(config, tp_size, tp_rank,
                                                                      false);
        if (!plan_result.ok()) {
            result.error = "partition plan: " + plan_result.status().message();
            return result;
        }
        plan = plan_result.release();
        load_options.weights.partition_plan = &plan;
    }

    auto model_result = moe::Qwen35MoeModel::load_from_safetensors(model_dir, arena, stream,
                                                                   load_options);
    if (!model_result.ok()) {
        result.error = "load: " + model_result.status().message();
        return result;
    }
    moe::Qwen35MoeModel model = model_result.release();

    q35::ExecutorConfig ec;
    ec.max_batch_rows = 16u;
    ec.max_scheduled_requests = 1u;
    ec.max_scheduled_output_rows = 1u;

    q35::Qwen35TensorParallelContext ctx;
    if (tp_size > 1u) {
        auto ctx_result = moe::make_qwen35moe_tensor_parallel_context(config, tp_size, tp_rank,
                                                                     false);
        if (!ctx_result.ok()) {
            result.error = "tp context: " + ctx_result.status().message();
            return result;
        }
        ctx = ctx_result.release();
        ec.tp = &ctx;
    }

    const auto& hybrid = model.text_config();
    const std::uint32_t local_kv_heads =
        tp_size > 1u ? ctx.local_key_value_heads
                     : static_cast<std::uint32_t>(hybrid.num_key_value_heads);
    const std::uint32_t head_dim = static_cast<std::uint32_t>(hybrid.attention_head_dim);
    std::uint32_t attn_layers = 0;
    for (int lt : hybrid.layer_types)
        if (lt != 0) ++attn_layers;

    const std::uint32_t page_tokens = 16u;
    const std::uint32_t max_tokens = 64u;
    const std::uint32_t num_pages = ceil_div(max_tokens + 16u, page_tokens) + 4u;

    auto seq_result = st::SequenceSlotPool::create(arena, 1u, num_pages + 4u);
    auto gdn_layout_result = q35::make_qwen35_gdn_state_layout_tp(hybrid, tp_size);
    if (!gdn_layout_result.ok()) {
        result.error = "gdn layout: " + gdn_layout_result.status().message();
        return result;
    }
    auto gdn_result =
        st::GdnStatePool::create(arena, 1u, gdn_layout_result.value());
    auto kv_result = st::PagedKVPool::create(arena, num_pages, page_tokens, attn_layers,
                                             local_kv_heads, head_dim,
                                             shelf::kernel::KVCacheDType::BF16);
    if (!seq_result.ok() || !gdn_result.ok() || !kv_result.ok()) {
        result.error = "pool create";
        return result;
    }
    st::SequenceSlotPool seq_pool = seq_result.release();
    st::GdnStatePool gdn_pool = gdn_result.release();
    st::PagedKVPool kv_pool = kv_result.release();

    moe::Qwen35MoeLowerOptions lower_options;
    lower_options.tp_size = tp_size;
    lower_options.tp_rank = tp_rank;
    auto lowered = moe::lower_qwen35_moe_to_primitives(model.text_config(), model.weights(),
                                                       lower_options);
    if (!lowered.ok()) {
        result.error = "lowering: " + lowered.status().message();
        return result;
    }

    q35::ExecutorModelInput input;
    input.model.config = &model.text_config();
    input.model.embed_tokens = &model.weights().embed_tokens;
    input.lowered = lowered.release();
    auto exec_result = q35::create_executor(std::move(input), seq_pool, gdn_pool, kv_pool, arena,
                                            ec);
    if (!exec_result.ok()) {
        result.error = "create_executor: " + exec_result.status().message();
        return result;
    }
    q35::Executor ex = exec_result.release();
    if (barrier != nullptr) ex.tp_barrier = barrier;

    result.plane_experts = model.weights().layers[0].ffn.expert_w13.experts;
    result.shared_rows = model.weights().layers[0].ffn.shared_gate_proj.rows;
    result.kv_heads = local_kv_heads;
    result.router_rows = model.weights().layers[0].ffn.router.rows;

    auto seq_state = st::create_paged_sequence_state(seq_pool, gdn_pool, kv_pool,
                                                     max_tokens + 16u, stream);
    if (!seq_state.ok()) {
        result.error = "sequence create: " + seq_state.status().message();
        return result;
    }
    st::PagedSequenceState sequence = seq_state.release();

    const std::vector<int32_t> prompt = make_prompt(16u);
    std::vector<q35::ScheduledRequest> requests;
    auto prefill = q35::execute_batch(ex, make_batch(sequence, requests, prompt.data(),
                                                     static_cast<std::uint32_t>(prompt.size())),
                                      stream);
    if (!prefill.ok()) {
        result.error = "prefill: " + prefill.status().message();
        (void)st::release_paged_sequence_state(sequence, stream);
        return result;
    }
    {
        q35::BatchExecutionOutput out = prefill.release();
        int32_t token = 0;
        if (out.sampled_tokens.ndim() > 0 && out.sampled_tokens.dim(0) > 0)
            (void)hipMemcpy(&token, out.sampled_tokens.data<int32_t>(), sizeof(int32_t),
                            hipMemcpyDeviceToHost);
        result.tokens.push_back(token);
        download_row0(out.token_hidden, device, result.hidden_row0);
        download_row0(ex.logits, device, result.logits_row0);
    }
    for (int step = 0; step < 2; ++step) {
        std::vector<int32_t> one{result.tokens.back()};
        auto decode = q35::execute_batch(ex, make_batch(sequence, requests, one.data(), 1u),
                                         stream);
        if (!decode.ok()) {
            result.error = "decode: " + decode.status().message();
            (void)st::release_paged_sequence_state(sequence, stream);
            return result;
        }
        q35::BatchExecutionOutput out = decode.release();
        int32_t token = 0;
        if (out.sampled_tokens.ndim() > 0 && out.sampled_tokens.dim(0) > 0)
            (void)hipMemcpy(&token, out.sampled_tokens.data<int32_t>(), sizeof(int32_t),
                            hipMemcpyDeviceToHost);
        result.tokens.push_back(token);
        download_row0(ex.logits, device, result.logits_row0);
    }
    (void)st::release_paged_sequence_state(sequence, stream);
    (void)q35::executor_shutdown(ex);
    (void)hipStreamSynchronize(stream);
    result.ok = true;
    return result;
}

double worst_relative(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 1.0e9;
    double worst = 0.0;
    double scale = 1.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - b[i]));
        scale = std::max(scale, std::fabs(static_cast<double>(a[i])));
    }
    return worst / scale;
}

void print_tokens(const char* tag, const std::vector<int32_t>& tokens) {
    std::printf("%s:", tag);
    for (int32_t t : tokens) std::printf(" %d", t);
    std::printf("\n");
}

}  // namespace

int main() {
    std::printf("=== qwen35moe tensor parallel execution (2 GPUs) ===\n");
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 2) {
        std::printf("SKIP: MoE tp execution requires 2 GPUs\n");
        return 77;
    }
    const int devices[2] = {0, 1};

    std::error_code ec;
    const fs::path root =
        fs::temp_directory_path(ec) / ("shelf_moe_tp_exec_" + std::to_string(::getpid()));
    fs::remove_all(root, ec);
    Fixture f;
    if (!write_fixture(f, root / "moe", false, true)) {
        std::printf("FAIL: moe fixture write\n");
        return 1;
    }
    const std::string dir = (root / "moe").string();

    const RunResult solo = run_rank(dir, devices[0], 1u, 0u, nullptr);
    if (!solo.ok) {
        std::printf("FAIL tp1 run: %s\n", solo.error.c_str());
        fs::remove_all(root, ec);
        return 1;
    }
    print_tokens("tp1 tokens", solo.tokens);
    std::printf("tp1: plane experts=%u shared rows=%u kv heads=%u router rows=%u\n",
                solo.plane_experts, solo.shared_rows, solo.kv_heads, solo.router_rows);
    check(solo.plane_experts == kExperts, "tp1 owns every expert");
    check(solo.shared_rows == kInter, "tp1 holds the whole shared expert");
    check(solo.router_rows == kExperts, "tp1 router logits are one column per expert");

    auto transport = rt::create_tp_transport({devices[0], devices[1]});
    if (!transport.ok()) {
        std::printf("FAIL transport: %s\n", transport.status().message().c_str());
        fs::remove_all(root, ec);
        return 1;
    }
    rt::TpTransportSelection selection = transport.release();
    std::unique_ptr<rt::TpTransport> tp_transport = std::move(selection.transport);
    std::printf("tp transport: %s (peer access %d)\n", tp_transport->name(),
                selection.peer_access_available ? 1 : 0);
    if (!tp_transport->verify().ok()) {
        std::printf("tp transport self-test failed: using host-mediated transport\n");
        tp_transport = std::make_unique<rt::HostMediatedTpTransport>(
            std::vector<int>{devices[0], devices[1]});
    }
    rt::TpBarrierGroup group(2, {devices[0], devices[1]}, tp_transport.get());
    if (!group.initialize().ok()) {
        std::printf("FAIL barrier initialize\n");
        fs::remove_all(root, ec);
        return 1;
    }

    RunResult rank0;
    RunResult rank1;
    std::thread t0([&] { rank0 = run_rank(dir, devices[0], 2u, 0u, &group); });
    std::thread t1([&] { rank1 = run_rank(dir, devices[1], 2u, 1u, &group); });
    t0.join();
    t1.join();
    group.shutdown();

    if (!rank0.ok || !rank1.ok) {
        std::printf("FAIL tp2 run: rank0=%s rank1=%s\n", rank0.error.c_str(),
                    rank1.error.c_str());
        fs::remove_all(root, ec);
        return 1;
    }
    print_tokens("tp2 rank0 tokens", rank0.tokens);
    print_tokens("tp2 rank1 tokens", rank1.tokens);
    std::printf("tp2: plane experts=%u shared rows=%u kv heads=%u router rows=%u\n",
                rank0.plane_experts, rank0.shared_rows, rank0.kv_heads, rank0.router_rows);

    check(rank0.plane_experts == kExperts / 2u && rank1.plane_experts == kExperts / 2u,
          "each rank holds half the expert plane");
    check(rank0.shared_rows == kInter / 2u, "each rank holds half the shared expert width");
    check(rank0.kv_heads == rank1.kv_heads && rank0.kv_heads == kKvHeads / 2u,
          "each rank's KV pool holds half the kv heads");
    check(rank0.router_rows == kExperts && rank1.router_rows == kExperts,
          "each rank's router keeps the whole expert count");

    check(rank0.tokens == solo.tokens, "tp2 rank 0 samples the same tokens as one device");
    check(rank1.tokens == solo.tokens, "tp2 rank 1 samples the same tokens as one device");

    const double logits_rel = worst_relative(solo.logits_row0, rank0.logits_row0);
    const double hidden_rel = worst_relative(solo.hidden_row0, rank0.hidden_row0);
    std::printf("logits row0 relative difference %g, hidden row0 %g\n",
                static_cast<double>(logits_rel), static_cast<double>(hidden_rel));
    check(logits_rel < 0.05, "tp2 logits stay inside 5 % of the single-device run");
    check(hidden_rel < 0.05, "tp2 hidden stays inside 5 % of the single-device run");
    check(worst_relative(rank0.logits_row0, rank1.logits_row0) == 0.0,
          "the two ranks hold the same logits after the sum");

    fs::remove_all(root, ec);
    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
