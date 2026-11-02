#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/gpu/scoped_device.h>
#include <shelf/io/safetensors_writer.h>
#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/models/qwen35/runtime/continuous_batcher.h>
#include <shelf/models/qwen35/runtime/tensor_parallel.h>
#include <shelf/models/qwen35/state/state_types.h>
#include <shelf/quantization/manifest/crc32.h>
#include <shelf/quantization/manifest/quantized_manifest.h>
#include <shelf/inference/execution/program_executor.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <span>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <unistd.h>

#include "support/tp_run_harness.h"
#include "support/qwen35_tp_fixture.h"

namespace fs = std::filesystem;
using namespace ps;

using ps::state::GdnStatePool;
using ps::state::GdnStatePoolLayout;
using ps::state::PagedKVPool;
using ps::qwen35::Qwen35LayerWeights;
using ps::qwen35::Qwen35LoadOptions;
using ps::qwen35::Qwen35Model;
using ps::qwen35::Qwen35ModelWeights;
using ps::qwen35::Qwen35TextConfig;
using ps::state::SequenceSlotPool;
using ps::qwen35::Executor;
using ps::qwen35::ExecutorConfig;
using ps::qwen35::create_model_executor;
using ps::qwen35::executor_shutdown;
using ps::kernel::KVCacheDType;
using ps::qwen35::runtime::ContinuousBatcher;
using ps::qwen35::runtime::ContinuousBatcherConfig;
using ps::qwen35::runtime::RuntimeRequest;
using ps::qwen35::runtime::TpCoordinator;
using ps::qwen35::runtime::TpCoordinatorConfig;
using ps::qwen35::runtime::TpRankRuntime;
using ps::qwen35::runtime::ValueTraceSink;

static int g_fail = 0;

static void fail(const std::string& msg) {
    g_fail++;
    std::printf("FAIL %s\n", msg.c_str());
}

static void check(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}


static void print_tokens(const char* tag, const std::vector<int32_t>& tokens) {
    std::printf("%s:", tag);
    for (int32_t t : tokens) std::printf(" %d", t);
    std::printf("\n");
}

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 2) {
        std::printf("SKIP: qwen35 tp execution requires 2 GPUs\n");
        return 77;
    }

    std::error_code ec;
    const fs::path dir =
        fs::temp_directory_path(ec) / ("qwen35_tp_exec_" + std::to_string(::getpid()));
    fs::remove_all(dir, ec);
    Status fixture = tp_fixture::build_fixture(dir);
    if (tp_fixture::g_fail != 0) fail("fixture helpers reported failures");
    if (!fixture.ok()) {
        fail("fixture: " + fixture.message());
        fs::remove_all(dir, ec);
        std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
        return g_fail == 0 ? 0 : 1;
    }

    tp_harness::RunOptions opts;
    opts.model_dir = dir.string();
    opts.arena_bytes = 1024ull * 1024ull * 1024ull;
    opts.max_seq_len = 64;
    opts.max_batch_rows = 16;
    opts.max_scheduled_requests = 4;
    opts.page_tokens = 16;
    opts.taps = {0};
    opts.verify_crc = true;
    opts.prompt = {3, 11, 7, 42, 5};
    opts.max_new_tokens = 8;

    tp_harness::RunResult tp1 = tp_harness::run_tp1(opts);
    if (!tp1.ok) fail("tp1 run: " + tp1.error);

    tp_harness::RunResult tp2 = tp_harness::run_tp2(opts);
    if (!tp2.ok) fail("tp2 run: " + tp2.error);
    std::printf("%s", tp2.report.c_str());

    if (tp1.ok && tp2.ok) {
        print_tokens("tp1 tokens", tp1.tokens);
        print_tokens("tp2 tokens", tp2.tokens);

        check(tp1.gate_rows == 128, "tp1 global gate rows");
        check(tp1.down_cols == 128, "tp1 global down cols");
        check(tp1.q_rows == 128, "tp1 global q rows");
        check(tp1.kv_rows == 32, "tp1 global kv rows");
        check(tp1.gdn_qkv_rows == 2048, "tp1 global gdn qkv rows");
        check(tp1.kv_pool_heads == 2, "tp1 global kv heads");
        check(tp1.gdn_conv_dim == 2048, "tp1 global gdn conv dim");

        check(tp2.gate_rows == 64, "tp2 local gate rows");
        check(tp2.down_cols == 64, "tp2 local down cols");
        check(tp2.q_rows == 64, "tp2 local q rows");
        check(tp2.kv_rows == 16, "tp2 local kv rows");
        check(tp2.gdn_qkv_rows == 1024, "tp2 local gdn qkv rows");
        check(tp2.kv_pool_heads == 1, "tp2 local kv heads");
        check(tp2.gdn_conv_dim == 1024, "tp2 local gdn conv dim");
        check(tp2.weight_bytes > 0, "tp2 rank weight bytes reported");

        tp_harness::compare_traces(tp1.trace.rows0, tp2.trace.rows0,
                                   "tp1-vs-tp2rank0");
        tp_harness::compare_traces(tp2.trace.rows0, tp2.rank1_trace.rows0,
                                   "tp2rank0-vs-tp2rank1");

        check(!tp1.tokens.empty(), "tp1 produced tokens");
        check(tp1.tokens == tp2.tokens,
              "greedy token sequence matches between tp1 and tp2");
        check(!tp1.tokens.empty() && tp1.tokens[0] == tp2.tokens[0],
              "greedy first token matches between tp1 and tp2");

        check(!tp1.capture.logits_row0.empty() && !tp2.capture.logits_row0.empty(),
              "logits captured");
        if (!tp1.capture.logits_row0.empty() && !tp2.capture.logits_row0.empty()) {
            const int a1 = tp_harness::argmax(tp1.capture.logits_row0);
            const int a2 = tp_harness::argmax(tp2.capture.logits_row0);
            check(a1 == a2, "prefill logits argmax matches (tp1 vs tp2)");
            const float diff = tp_harness::max_abs_diff(tp1.capture.logits_row0,
                                                        tp2.capture.logits_row0);
            std::printf("logits max abs diff tp1 vs tp2: %g\n",
                        static_cast<double>(diff));
            check(diff <= 0.5f, "prefill logits within tolerance");
            const float hidden_diff = tp_harness::max_abs_diff(
                tp1.capture.hidden_row0, tp2.capture.hidden_row0);
            std::printf("final hidden max abs diff tp1 vs tp2: %g\n",
                        static_cast<double>(hidden_diff));
            check(hidden_diff <= 0.1f, "final hidden within tolerance");
        }

        if (!tp1.capture.tap0_row0.empty() && !tp2.capture.tap0_row0.empty()) {
            const float tap0_diff = tp_harness::max_abs_diff(tp1.capture.tap0_row0,
                                                             tp2.capture.tap0_row0);
            std::printf("layer0 output max abs diff tp1 vs tp2: %g\n",
                        static_cast<double>(tap0_diff));
            check(tap0_diff <= 0.1f, "layer0 output within tolerance");
        }

        if (!tp2.capture.hidden_row0.empty() && !tp2.rank1_capture.hidden_row0.empty()) {
            check(tp2.capture.hidden_row0 == tp2.rank1_capture.hidden_row0,
                  "tp2 rank0 and rank1 final hidden are identical");
            if (tp2.capture.hidden_row0 != tp2.rank1_capture.hidden_row0) {
                std::printf(
                    "rank0 vs rank1 hidden diff: %g\n",
                    static_cast<double>(tp_harness::max_abs_diff(
                        tp2.capture.hidden_row0, tp2.rank1_capture.hidden_row0)));
            }
        }
        if (!tp2.capture.logits_row0.empty() && !tp2.rank1_capture.logits_row0.empty()) {
            check(tp2.capture.logits_row0 == tp2.rank1_capture.logits_row0,
                  "tp2 rank0 and rank1 logits are identical");
        }
    }

    fs::remove_all(dir, ec);
    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
