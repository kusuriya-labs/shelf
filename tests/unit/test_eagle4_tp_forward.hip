#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/gpu/scoped_device.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/kernels/optimized/embedding.h>
#include <phaseshift/models/qwen35/eagle4/config.h>
#include <phaseshift/models/qwen35/eagle4/context_state.h>
#include <phaseshift/models/qwen35/eagle4/executor.h>
#include <phaseshift/models/qwen35/eagle4/weights.h>
#include <phaseshift/models/qwen35/runtime/tp_batch_hook.h>
#include <phaseshift/runtime/tp/tp_barrier_group.h>
#include <phaseshift/runtime/tp/tp_execution.h>
#include <phaseshift/runtime/tp/tp_transports.h>
#include <hip/hip_runtime.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "support/eagle4_checkpoint.h"

namespace fs = std::filesystem;
using namespace eagle4_ckpt;

using ps::bf16_t;

namespace {

int g_fail = 0;

void check(bool cond, const std::string& msg) {
    if (cond) return;
    ++g_fail;
    std::printf("FAIL %s\n", msg.c_str());
}

std::string gpu_name(int device) {
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, device) != hipSuccess) return "unknown";
    return props.gcnArchName;
}

class TestDraftDispatcher final : public ps::qwen35::runtime::TpBatchHook {
public:
    ps::Status on_sequence_created(const ps::state::PagedSequenceState&) override {
        return ps::Status::make_ok();
    }

    ps::Status on_sequence_released(const ps::state::PagedSequenceState&) override {
        return ps::Status::make_ok();
    }

    ps::Result<ps::qwen35::BatchExecutionOutput> on_execute(
        const ps::qwen35::ScheduledBatch&) override {
        return ps::Status::unsupported("test dispatcher has no target execution", __FILE__,
                                       __LINE__);
    }

    ps::Result<ps::qwen35::BatchExecutionOutput> on_execute(
        const ps::qwen35::ScheduledBatch&,
        const ps::qwen35::ExecuteBatchOptions*) override {
        return ps::Status::unsupported("test dispatcher has no target execution", __FILE__,
                                       __LINE__);
    }

    ps::Status on_eagle4_draft(const ps::qwen35::eagle4::Eagle4DraftJob& job) override {
        const std::size_t count = job.ranks.size();
        if (count == 0u || count > 2u) {
            return ps::Status::invalid_argument("test dispatcher rank count", __FILE__,
                                                __LINE__);
        }
        ps::Status status[2] = {ps::Status::make_ok(), ps::Status::make_ok()};
        std::thread workers[2];
        for (std::size_t r = 0; r < count; ++r) {
            workers[r] = std::thread([&job, &status, r] {
                status[r] = ps::qwen35::eagle4::run_eagle4_rank_job(job, r);
            });
        }
        for (std::size_t r = 0; r < count; ++r) workers[r].join();
        for (std::size_t r = 0; r < count; ++r) {
            if (!status[r].ok()) return status[r];
        }
        return ps::Status::make_ok();
    }
};

struct StageSample {
    std::vector<float> embedding;
    std::vector<float> attention;
    std::vector<float> mlp;
    std::vector<float> normed;
    std::vector<std::int32_t> candidates;
    float best_logit = 0.0f;
};

struct RankFixture {
    int device = 0;
    std::unique_ptr<ps::gpu::GpuArena> arena;
    hipStream_t stream = nullptr;
    ps::qwen35::eagle4::Eagle4Weights weights;
    ps::qwen35::eagle4::Eagle4Executor executor;
    ps::qwen35::eagle4::Eagle4ContextState context;
    ps::gpu::Tensor taps[3];

    bool initialized() const { return executor.initialized && context.initialized; }
};

void shutdown_fixture(RankFixture& fixture) {
    auto scope = ps::gpu::ScopedDevice::create(fixture.device);
    if (scope.ok()) {
        if (fixture.executor.initialized) {
            (void)ps::qwen35::eagle4::eagle4_executor_shutdown(fixture.executor);
        }
        if (fixture.context.initialized) {
            (void)ps::qwen35::eagle4::eagle4_context_shutdown(fixture.context);
        }
        for (ps::gpu::Tensor& tap : fixture.taps) tap = ps::gpu::Tensor{};
        if (fixture.stream != nullptr) {
            ps::gpu::discard_cleanup_result(hipStreamDestroy(fixture.stream));
            fixture.stream = nullptr;
        }
    }
    fixture.arena.reset();
}

bool create_fixture(
    const std::string& model_dir,
    const ps::qwen35::eagle4::Eagle4Config& config,
    int device,
    std::uint32_t tp_size,
    std::uint32_t tp_rank,
    std::uint32_t num_candidates,
    const std::function<ps::Status(const ps::runtime::TpSumTarget&, hipStream_t,
                                   std::uint32_t)>& tp_sum,
    RankFixture& fixture) {
    fixture.device = device;
    auto scope = ps::gpu::ScopedDevice::create(device);
    if (!scope.ok()) {
        check(false, "device scope: " + scope.status().message());
        return false;
    }
    auto arena_result = ps::gpu::GpuArena::create(device, 512ull * 1024ull * 1024ull);
    if (!arena_result.ok()) {
        check(false, "arena create: " + arena_result.status().message());
        return false;
    }
    fixture.arena = std::make_unique<ps::gpu::GpuArena>(arena_result.release());
    if (hipStreamCreate(&fixture.stream) != hipSuccess) {
        check(false, "hipStreamCreate");
        return false;
    }
    ps::qwen35::eagle4::Eagle4WeightLoadOptions load_options;
    load_options.tp_size = tp_size;
    load_options.tp_rank = tp_rank;
    auto weights_result = ps::qwen35::eagle4::load_eagle4_weights(
        model_dir, config, *fixture.arena, fixture.stream, load_options);
    if (!weights_result.ok()) {
        check(false, "weights load: " + weights_result.status().message());
        return false;
    }
    fixture.weights = weights_result.release();

    ps::qwen35::eagle4::Eagle4ExecutorConfig exec_config;
    exec_config.num_candidates = num_candidates;
    exec_config.tp_size = tp_size;
    exec_config.tp_rank = tp_rank;
    exec_config.tp_sum = tp_sum;
    auto executor_result = ps::qwen35::eagle4::create_eagle4_executor(
        config, fixture.weights, *fixture.arena, exec_config);
    if (!executor_result.ok()) {
        check(false, "executor create: " + executor_result.status().message());
        return false;
    }
    fixture.executor = executor_result.release();

    auto context_result = ps::qwen35::eagle4::create_eagle4_context_state(
        config, *fixture.arena, 0u, tp_size);
    if (!context_result.ok()) {
        check(false, "context create: " + context_result.status().message());
        return false;
    }
    fixture.context = context_result.release();
    return true;
}

bool upload_taps(RankFixture& fixture, const std::vector<bf16_t>& host_taps,
                 std::uint32_t rows, std::uint32_t tap_features) {
    auto scope = ps::gpu::ScopedDevice::create(fixture.device);
    if (!scope.ok()) {
        check(false, "tap device scope: " + scope.status().message());
        return false;
    }
    const std::size_t per_tap = static_cast<std::size_t>(rows) * tap_features;
    for (std::uint32_t t = 0; t < kTaps; ++t) {
        auto view = fixture.arena->allocate_aligned(per_tap * sizeof(bf16_t), 256u);
        if (!view.ok()) {
            check(false, "tap alloc: " + view.status().message());
            return false;
        }
        auto created = ps::gpu::Tensor::create<bf16_t>(view.release(), {rows, tap_features},
                                                       {tap_features, 1u});
        if (!created.ok()) {
            check(false, "tap tensor: " + created.status().message());
            return false;
        }
        fixture.taps[t] = created.release();
        if (hipMemcpy(fixture.taps[t].data<bf16_t>(),
                      host_taps.data() + static_cast<std::size_t>(t) * per_tap,
                      per_tap * sizeof(bf16_t), hipMemcpyHostToDevice) != hipSuccess) {
            check(false, "tap upload");
            return false;
        }
    }
    return true;
}

bool sync_rank(RankFixture& fixture) {
    auto scope = ps::gpu::ScopedDevice::create(fixture.device);
    if (!scope.ok()) {
        check(false, "sync device scope: " + scope.status().message());
        return false;
    }
    if (hipStreamSynchronize(fixture.stream) != hipSuccess) {
        check(false, "stream sync: " + std::string(hipGetErrorString(hipGetLastError())));
        return false;
    }
    return true;
}

bool download_bf16(RankFixture& fixture, const ps::gpu::Tensor& tensor, std::uint32_t rows,
                   std::uint32_t features, std::vector<float>& out) {
    if (!sync_rank(fixture)) return false;
    auto scope = ps::gpu::ScopedDevice::create(fixture.device);
    if (!scope.ok()) {
        check(false, "download device scope: " + scope.status().message());
        return false;
    }
    std::vector<bf16_t> raw(static_cast<std::size_t>(rows) * features);
    if (hipMemcpy(raw.data(), tensor.data<bf16_t>(), raw.size() * sizeof(bf16_t),
                  hipMemcpyDeviceToHost) != hipSuccess) {
        check(false, "download copy");
        return false;
    }
    out.resize(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) out[i] = bf16_to_f32(raw[i]);
    return true;
}

bool download_i32(RankFixture& fixture, const ps::gpu::Tensor& tensor, std::uint32_t count,
                  std::vector<std::int32_t>& out) {
    if (!sync_rank(fixture)) return false;
    auto scope = ps::gpu::ScopedDevice::create(fixture.device);
    if (!scope.ok()) {
        check(false, "download i32 device scope: " + scope.status().message());
        return false;
    }
    out.resize(count);
    if (hipMemcpy(out.data(), tensor.data<std::int32_t>(), count * sizeof(std::int32_t),
                  hipMemcpyDeviceToHost) != hipSuccess) {
        check(false, "download i32 copy");
        return false;
    }
    return true;
}

bool download_f32(RankFixture& fixture, const ps::gpu::Tensor& tensor, std::uint32_t count,
                  std::vector<float>& out) {
    if (!sync_rank(fixture)) return false;
    auto scope = ps::gpu::ScopedDevice::create(fixture.device);
    if (!scope.ok()) {
        check(false, "download f32 device scope: " + scope.status().message());
        return false;
    }
    out.resize(count);
    if (hipMemcpy(out.data(), tensor.data<float>(), count * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess) {
        check(false, "download f32 copy");
        return false;
    }
    return true;
}

double relative_error(const std::vector<float>& got, const std::vector<float>& want,
                      double* max_abs_out) {
    double max_abs = 0.0;
    double rms = 0.0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        max_abs = std::max(max_abs, static_cast<double>(std::fabs(got[i] - want[i])));
        rms += static_cast<double>(want[i]) * static_cast<double>(want[i]);
    }
    rms = std::sqrt(rms / static_cast<double>(std::max<std::size_t>(1u, want.size())));
    if (max_abs_out != nullptr) *max_abs_out = max_abs;
    return rms > 0.0 ? max_abs / rms : max_abs;
}

bool embedding_partial_vocab_gpu1(const std::string& tag) {
    auto scope = ps::gpu::ScopedDevice::create(1);
    if (!scope.ok()) {
        check(false, tag + ": device scope: " + scope.status().message());
        return false;
    }
    constexpr std::uint32_t kRows = 256u;
    constexpr std::uint32_t kFeatures = 5120u;
    constexpr std::uint32_t kLocalVocab = 4096u;
    constexpr std::uint32_t kVocabOffset = kLocalVocab;

    bf16_t* table = nullptr;
    bf16_t* output = nullptr;
    std::int32_t* tokens = nullptr;
    std::uint32_t* error_word = nullptr;
    bool ok = true;
    if (hipMalloc(&table, static_cast<std::size_t>(kLocalVocab) * kFeatures * sizeof(bf16_t)) !=
            hipSuccess ||
        hipMalloc(&output, static_cast<std::size_t>(kRows) * kFeatures * sizeof(bf16_t)) !=
            hipSuccess ||
        hipMalloc(&tokens, kRows * sizeof(std::int32_t)) != hipSuccess ||
        hipMalloc(&error_word, sizeof(std::uint32_t)) != hipSuccess) {
        check(false, tag + ": hipMalloc");
        ok = false;
    }
    hipStream_t stream = nullptr;
    if (ok) {
        std::vector<bf16_t> host_table(static_cast<std::size_t>(kLocalVocab) * kFeatures);
        for (std::uint32_t r = 0; r < kLocalVocab; ++r) {
            const bf16_t value = to_bf16(static_cast<float>(r) * 0.25f);
            for (std::uint32_t c = 0; c < kFeatures; ++c) {
                host_table[static_cast<std::size_t>(r) * kFeatures + c] = value;
            }
        }
        std::vector<std::int32_t> host_tokens(kRows);
        for (std::uint32_t r = 0; r < kRows; ++r) {
            host_tokens[r] =
                static_cast<std::int32_t>(r % 3u == 0u ? (kLocalVocab + 7u) : r);
        }
        std::uint32_t zero_error = 0;
        if (hipMemcpy(table, host_table.data(), host_table.size() * sizeof(bf16_t),
                      hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(tokens, host_tokens.data(), host_tokens.size() * sizeof(std::int32_t),
                      hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(error_word, &zero_error, sizeof(zero_error), hipMemcpyHostToDevice) !=
                hipSuccess) {
            check(false, tag + ": hipMemcpy");
            ok = false;
        }
        if (ok && hipStreamCreate(&stream) != hipSuccess) {
            check(false, tag + ": hipStreamCreate");
            ok = false;
        }
        if (ok) {
            const hipError_t launched = ps::kernel::launch_embedding_bf16(
                table, tokens, output, kRows, kLocalVocab, kFeatures, kFeatures, error_word,
                stream, kVocabOffset, 1u);
            if (launched != hipSuccess) {
                check(false, tag + ": launch_embedding_bf16: " +
                                 std::string(hipGetErrorString(launched)));
                ok = false;
            }
        }
        if (ok && hipStreamSynchronize(stream) != hipSuccess) {
            check(false, tag + ": embedding sync: " +
                             std::string(hipGetErrorString(hipGetLastError())));
            ok = false;
        }
        if (ok) {
            std::vector<bf16_t> host_out(static_cast<std::size_t>(kRows) * kFeatures);
            if (hipMemcpy(host_out.data(), output, host_out.size() * sizeof(bf16_t),
                          hipMemcpyDeviceToHost) != hipSuccess) {
                check(false, tag + ": embedding output download");
                ok = false;
            }
            for (std::uint32_t r = 0; ok && r < kRows; ++r) {
                const bool foreign = host_tokens[r] < static_cast<std::int32_t>(kVocabOffset);
                const float want =
                    foreign ? 0.0f
                            : static_cast<float>(host_tokens[r] - kVocabOffset) * 0.25f;
                for (std::uint32_t c = 0; c < kFeatures; c += 257u) {
                    const float got =
                        bf16_to_f32(host_out[static_cast<std::size_t>(r) * kFeatures + c]);
                    if (std::fabs(got - want) > 1.0e-3f) {
                        check(false, tag + ": row " + std::to_string(r) + " col " +
                                         std::to_string(c) + " got " + std::to_string(got) +
                                         " want " + std::to_string(want));
                        ok = false;
                        break;
                    }
                }
            }
        }
    }
    if (stream != nullptr) (void)ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    (void)ps::gpu::discard_cleanup_result(hipFree(table));
    (void)ps::gpu::discard_cleanup_result(hipFree(output));
    (void)ps::gpu::discard_cleanup_result(hipFree(tokens));
    (void)ps::gpu::discard_cleanup_result(hipFree(error_word));
    return ok;
}

bool write_checkpoint(const fs::path& root, ps::qwen35::eagle4::Eagle4Config& config,
                      HostModel& host) {
    {
        std::ofstream f(root / "config.json", std::ios::binary | std::ios::trunc);
        f << R"json({
  "architectures": ["LlamaForCausalLMEagle4"],
  "eagle_config": {"eagle_aux_hidden_state_layer_ids": [1, 2, 3]},
  "efficient_lm_head_config": {"num_candidates": 16, "rank": 8, "type": "low_rank"},
  "fc_norm": true,
  "head_dim": 128,
  "hidden_size": 256,
  "intermediate_size": 64,
  "layer_types": ["sliding_attention"],
  "norm_output": true,
  "num_attention_heads": 2,
  "num_aux_hidden_states": 3,
  "num_hidden_layers": 1,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"},
  "sliding_window": 8,
  "target_hidden_size": 256,
  "tie_word_embeddings": false,
  "vocab_size": 64
}
)json";
    }
    auto config_result = ps::qwen35::eagle4::read_eagle4_config(root.string());
    check(config_result.ok(), "config reads");
    if (!config_result.ok()) return false;
    config = config_result.release();
    host = make_host_model();
    check(write_shard(root.string(), config, host), "checkpoint writes");
    return g_fail == 0;
}

}  // namespace

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 2) {
        std::printf("SKIP: TP2 eagle4 test needs two GPUs (have %d)\n", device_count);
        return 77;
    }
    std::printf("devices: 0=%s 1=%s\n", gpu_name(0).c_str(), gpu_name(1).c_str());

    const fs::path root =
        fs::temp_directory_path() /
        ("ps_eagle4_tp_forward_" + std::to_string(static_cast<long>(getpid())));
    fs::remove_all(root);
    fs::create_directories(root);

    ps::qwen35::eagle4::Eagle4Config config;
    HostModel host;
    if (!write_checkpoint(root, config, host)) {
        fs::remove_all(root);
        return 1;
    }

    check(embedding_partial_vocab_gpu1("embedding partial vocab on GPU1"),
          "embedding GPU1 ABI");

    constexpr std::uint32_t kRows = 4u;
    constexpr std::uint32_t kTopk = 3u;
    constexpr std::uint32_t kTapFeatures = 256u;
    constexpr std::uint32_t kFeatures = kHidden;
    const std::uint32_t tp1_candidates = kCandidates;
    const std::uint32_t tp2_candidates = kCandidates;
    const std::vector<std::int32_t> tokens = {3, 33, 5, 40};

    std::uint32_t seed = 4242u;
    std::vector<bf16_t> host_taps(static_cast<std::size_t>(kTaps) * kRows * kTapFeatures);
    for (bf16_t& value : host_taps) value = random_bf16(seed, 1.0f);

    StageSample tp1{};
    StageSample tp2{};
    std::vector<const bf16_t*> tp1_taps(kTaps, nullptr);
    std::vector<std::vector<const bf16_t*>> tp2_taps(
        2u, std::vector<const bf16_t*>(kTaps, nullptr));
    RankFixture tp1_fixture;
    RankFixture fixtures[2];
    std::vector<ps::qwen35::eagle4::Eagle4RankDraft> tp1_ranks(1u);
    std::vector<ps::qwen35::eagle4::Eagle4RankDraft> tp2_ranks(2u);
    ps::gpu::Tensor tp1_root;
    ps::gpu::Tensor tp2_root;
    TestDraftDispatcher dispatcher;
    std::unique_ptr<ps::runtime::TpTransportSelection> transport;
    std::unique_ptr<ps::runtime::TpBarrierGroup> barrier;
    bool ok = true;

    {
        RankFixture& rank0 = tp1_fixture;
        if (!create_fixture(root.string(), config, 0, 1u, 0u, tp1_candidates, nullptr,
                            rank0)) {
            ok = false;
        }

        if (ok && !upload_taps(rank0, host_taps, kRows, kTapFeatures)) ok = false;
        if (ok) {
            for (std::uint32_t t = 0; t < kTaps; ++t) {
                tp1_taps[t] = rank0.taps[t].data<bf16_t>();
            }
            auto view = rank0.arena->allocate_aligned(kTopk * sizeof(std::int32_t), 256u);
            if (!view.ok()) {
                check(false, "tp1 root alloc");
                ok = false;
            } else {
                auto created =
                    ps::gpu::Tensor::create<std::int32_t>(view.release(), {kTopk}, {1u});
                if (!created.ok()) {
                    check(false, "tp1 root tensor");
                    ok = false;
                } else {
                    tp1_root = created.release();
                }
            }
        }
        if (ok) {
            tp1_ranks[0].executor = &rank0.executor;
            tp1_ranks[0].context = &rank0.context;
            tp1_ranks[0].stream = rank0.stream;
            tp1_ranks[0].device = 0;
            tp1_ranks[0].taps = tp1_taps.data();
            tp1_ranks[0].tap_count = kTaps;
            const ps::Status st = ps::qwen35::eagle4::eagle4_extend_rows_tp(
                rank0.executor, rank0.context, &tp1_ranks, tp1_taps.data(), kTaps, 0u,
                tokens.data(), kRows, 0u, kTopk, tp1_root.data<std::int32_t>(), rank0.stream);
            check(st.ok(), "tp1 extend: " + st.message());
            if (!st.ok()) {
                ok = false;
            } else if (!download_bf16(rank0, rank0.executor.embed_out, kRows, kFeatures,
                                      tp1.embedding) ||
                       !download_bf16(rank0, rank0.executor.attn_o, kRows, kFeatures,
                                      tp1.attention) ||
                       !download_bf16(rank0, rank0.executor.mlp_down, kRows, kFeatures,
                                      tp1.mlp) ||
                       !download_bf16(rank0, rank0.executor.normed, kRows, kFeatures,
                                      tp1.normed) ||
                       !download_i32(rank0, tp1_root, kTopk, tp1.candidates)) {
                ok = false;
            } else {
                std::vector<float> logits;
                if (download_f32(rank0, rank0.executor.topk_logits, kTopk, logits)) {
                    tp1.best_logit = logits[0];
                } else {
                    ok = false;
                }
            }
        }
        shutdown_fixture(tp1_fixture);
    }

    bool tp2_ready = ok;
    if (tp2_ready) {
        auto selection = ps::runtime::create_tp_transport({0, 1});
        if (!selection.ok()) {
            check(false, "tp transport: " + selection.status().message());
            tp2_ready = false;
        } else {
            transport = std::make_unique<ps::runtime::TpTransportSelection>(
                selection.release());
            const ps::Status verify = transport->transport->verify();
            if (!verify.ok()) {
                std::printf("NOTE: %s transport failed verification (%s); host-mediated\n",
                            transport->transport->name(), verify.message().c_str());
                transport->transport =
                    std::make_unique<ps::runtime::HostMediatedTpTransport>(
                        std::vector<int>{0, 1});
            }
            barrier = std::make_unique<ps::runtime::TpBarrierGroup>(
                2, std::vector<int>{0, 1}, transport->transport.get());
            const ps::Status init = barrier->initialize();
            if (!init.ok()) {
                check(false, "tp barrier init: " + init.message());
                tp2_ready = false;
            }
        }
    }
    if (tp2_ready) {
        auto tp_sum = [&barrier](const ps::runtime::TpSumTarget& target, hipStream_t stream,
                                 std::uint32_t rank) {
            return barrier->arrive(static_cast<int>(rank), target, stream, false);
        };
        for (std::uint32_t r = 0; tp2_ready && r < 2u; ++r) {
            if (!create_fixture(root.string(), config, static_cast<int>(r), 2u, r,
                                tp2_candidates, tp_sum, fixtures[r])) {
                tp2_ready = false;
                break;
            }
            fixtures[r].executor.tp_hook = &dispatcher;
            if (!upload_taps(fixtures[r], host_taps, kRows, kTapFeatures)) {
                tp2_ready = false;
                break;
            }
            for (std::uint32_t t = 0; t < kTaps; ++t) {
                tp2_taps[r][t] = fixtures[r].taps[t].data<bf16_t>();
            }
            tp2_ranks[r].executor = &fixtures[r].executor;
            tp2_ranks[r].context = &fixtures[r].context;
            tp2_ranks[r].stream = fixtures[r].stream;
            tp2_ranks[r].device = static_cast<int>(r);
            tp2_ranks[r].taps = tp2_taps[r].data();
            tp2_ranks[r].tap_count = kTaps;
        }
        auto view = fixtures[0].arena->allocate_aligned(kTopk * sizeof(std::int32_t), 256u);
        if (!view.ok()) {
            check(false, "tp2 root alloc");
            tp2_ready = false;
        } else {
            auto created =
                ps::gpu::Tensor::create<std::int32_t>(view.release(), {kTopk}, {1u});
            if (!created.ok()) {
                check(false, "tp2 root tensor");
                tp2_ready = false;
            } else {
                tp2_root = created.release();
            }
        }
    }
    if (tp2_ready) {
        tp2_ranks[0].executor->tp_hook = &dispatcher;
        tp2_ranks[1].executor->tp_hook = &dispatcher;
        const ps::Status st = ps::qwen35::eagle4::eagle4_extend_rows_tp(
            *tp2_ranks[0].executor, *tp2_ranks[0].context, &tp2_ranks, tp2_taps[0].data(),
            kTaps, 0u, tokens.data(), kRows, 0u, kTopk, tp2_root.data<std::int32_t>(),
            tp2_ranks[0].stream);
        check(st.ok(), "tp2 extend: " + st.message());
        if (!st.ok()) {
            tp2_ready = false;
        } else if (!download_bf16(fixtures[0], fixtures[0].executor.embed_out, kRows,
                                  kFeatures, tp2.embedding) ||
                   !download_bf16(fixtures[0], fixtures[0].executor.attn_o, kRows, kFeatures,
                                  tp2.attention) ||
                   !download_bf16(fixtures[0], fixtures[0].executor.mlp_down, kRows,
                                  kFeatures, tp2.mlp) ||
                   !download_bf16(fixtures[0], fixtures[0].executor.normed, kRows, kFeatures,
                                  tp2.normed) ||
                   !download_i32(fixtures[0], tp2_root, kTopk, tp2.candidates)) {
            tp2_ready = false;
        }
    }

    if (tp2_ready) {
        struct StageEntry {
            const char* name;
            const std::vector<float>* got;
            const std::vector<float>* want;
        };
        const StageEntry stages[] = {
            {"A embedding partial + sum", &tp2.embedding, &tp1.embedding},
            {"B attention row-parallel", &tp2.attention, &tp1.attention},
            {"C mlp row-parallel", &tp2.mlp, &tp1.mlp},
            {"D final normed output", &tp2.normed, &tp1.normed},
        };
        bool diverged = false;
        for (const StageEntry& stage : stages) {
            double max_abs = 0.0;
            const double ratio = relative_error(*stage.got, *stage.want, &max_abs);
            std::printf("STAGE %s max_abs/rms=%.6f max_abs=%.6f\n", stage.name, ratio,
                        max_abs);
            if (!diverged && ratio > 3.0e-2) {
                std::printf("FIRST_DIVERGENCE %s (max_abs/rms=%.6f)\n", stage.name, ratio);
                check(false, std::string("tp2 matches tp1 at stage ") + stage.name);
                diverged = true;
            }
        }
        const std::int32_t local_vocab =
            static_cast<std::int32_t>(fixtures[0].executor.lm_head->rows);
        std::vector<std::int32_t> rank_ids[2];
        std::vector<float> rank_logits[2];
        bool merge_ok = true;
        for (std::uint32_t r = 0; r < 2u; ++r) {
            merge_ok = merge_ok && download_i32(fixtures[r], fixtures[r].executor.topk_ids,
                                                kTopk, rank_ids[r]);
            merge_ok = merge_ok && download_f32(fixtures[r], fixtures[r].executor.topk_logits,
                                                kTopk, rank_logits[r]);
        }
        if (merge_ok) {
            struct MergeCandidate {
                float logit;
                std::int32_t id;
            };
            std::vector<MergeCandidate> pool;
            pool.reserve(2u * kTopk);
            for (std::uint32_t i = 0; i < kTopk; ++i) {
                pool.push_back({rank_logits[0][i], rank_ids[0][i]});
                pool.push_back({rank_logits[1][i], rank_ids[1][i] + local_vocab});
            }
            std::sort(pool.begin(), pool.end(),
                      [](const MergeCandidate& a, const MergeCandidate& b) {
                          return a.logit > b.logit;
                      });
            std::vector<std::int32_t> expected(kTopk);
            for (std::uint32_t i = 0; i < kTopk; ++i) expected[i] = pool[i].id;
            std::printf("STAGE E candidate ids tp1=[%d,%d,%d] tp2=[%d,%d,%d] "
                        "merged_top1_logit=%.6f tp1_top1_logit=%.6f\n",
                        tp1.candidates[0], tp1.candidates[1], tp1.candidates[2],
                        tp2.candidates[0], tp2.candidates[1], tp2.candidates[2],
                        pool[0].logit, tp1.best_logit);
            bool in_range = true;
            for (std::int32_t id : tp2.candidates) {
                in_range = in_range && id >= 0 && id < static_cast<std::int32_t>(config.vocab_size);
            }
            check(in_range, "tp2 candidate ids are global vocabulary ids");
            check(expected == tp2.candidates, "tp2 merged candidate ids match the rank union");
            const float margin =
                1.0e-2f * std::fabs(tp1.best_logit) + 1.0e-3f;
            check(pool[0].logit + margin >= tp1.best_logit,
                  "tp2 best candidate is not worse than tp1 (tp2=" +
                      std::to_string(pool[0].logit) + ", tp1=" +
                      std::to_string(tp1.best_logit) + ")");
        } else {
            check(false, "tp2 candidate merge diagnostics");
        }
    }

    for (RankFixture& fixture : fixtures) {
        if (fixture.initialized()) shutdown_fixture(fixture);
    }
    if (barrier) barrier->shutdown();
    fs::remove_all(root);

    if (g_fail != 0) {
        std::printf("FAILED (%d)\n", g_fail);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
