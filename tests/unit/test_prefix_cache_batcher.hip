#include <shelf/core/gpu/cleanup.h>
#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/models/qwen35/runtime/continuous_batcher.h>
#include <shelf/models/qwen35/runtime/executor.h>
#include <shelf/models/qwen35/runtime/prefix_cache.h>
#include <shelf/models/qwen35/state/state_types.h>
#include <shelf/models/qwen35/model/gdn_state_layout.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

#include "support/qwen35_tp_fixture.h"

namespace fs = std::filesystem;
using namespace ps;

using ps::qwen35::Executor;
using ps::qwen35::ExecutorConfig;
using ps::state::GdnStatePool;
using ps::state::GdnStatePoolLayout;
using ps::kernel::KVCacheDType;
using ps::state::PagedKVPool;
using ps::qwen35::Qwen35LoadOptions;
using ps::qwen35::Qwen35Model;
using ps::qwen35::Qwen35TextConfig;
using ps::state::SequenceSlotPool;
using ps::qwen35::create_model_executor;
using ps::qwen35::executor_shutdown;
using ps::qwen35::runtime::ContinuousBatcher;
using ps::qwen35::runtime::ContinuousBatcherConfig;
using ps::qwen35::runtime::PrefixCache;
using ps::qwen35::runtime::PrefixCacheConfig;
using ps::qwen35::runtime::PrefixCacheStats;
using ps::qwen35::runtime::RuntimeRequest;

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

static uint32_t ceil_div_u32(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}

static uint32_t count_attention_layers(const Qwen35TextConfig& tc) {
    uint32_t n = 0;
    for (int lt : tc.layer_types) {
        if (lt == 1) ++n;
    }
    return n;
}

struct RunOutcome {
    bool ok = false;
    std::string error;
    std::vector<int32_t> tokens;
    uint32_t restored = 0;
};

static void run_request(ContinuousBatcher& batcher,
                        const std::vector<int32_t>& prompt,
                        uint32_t max_new_tokens,
                        RunOutcome& out) {
    out = RunOutcome{};
    auto id_result = batcher.submit(prompt, max_new_tokens);
    if (!id_result.ok()) {
        out.error = "submit: " + id_result.status().message();
        return;
    }
    const uint64_t id = id_result.value();
    uint32_t guard = 0;
    while (batcher.has_pending()) {
        auto step = batcher.step();
        if (!step.ok()) {
            out.error = "step: " + step.status().message();
            return;
        }
        if (!step.value().executed) {
            out.error = "step did not execute while requests pending";
            return;
        }
        if (++guard > 4096u) {
            out.error = "step guard exceeded";
            return;
        }
    }
    const RuntimeRequest* request = batcher.request(id);
    if (request == nullptr) {
        out.error = "finished request not found";
        return;
    }
    out.tokens = request->generated;
    out.restored = request->restored_tokens;
    out.ok = true;
}

static void print_tokens(const char* tag, const std::vector<int32_t>& tokens) {
    std::printf("%s:", tag);
    for (int32_t token : tokens) std::printf(" %d", token);
    std::printf("\n");
}

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 1) {
        std::printf("SKIP: prefix cache batcher test requires 1 GPU\n");
        return 77;
    }

    std::error_code ec;
    const fs::path dir =
        fs::temp_directory_path(ec) / ("qwen35_prefix_batcher_" + std::to_string(::getpid()));
    fs::remove_all(dir, ec);
    const Status fixture = tp_fixture::build_fixture(dir);
    if (!fixture.ok()) {
        std::printf("FAIL fixture: %s\n", fixture.message().c_str());
        fs::remove_all(dir, ec);
        return 1;
    }

    if (hipSetDevice(0) != hipSuccess) {
        std::printf("FAIL hipSetDevice\n");
        fs::remove_all(dir, ec);
        return 1;
    }
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        std::printf("FAIL hipStreamCreate\n");
        fs::remove_all(dir, ec);
        return 1;
    }

    const uint32_t page_tokens = 16;
    const uint32_t max_seq_len = 128;
    const uint32_t max_batch_rows = 96;
    const uint32_t max_scheduled_requests = 4;

    auto arena_result = gpu::GpuArena::create(0, 1024ull * 1024ull * 1024ull);
    if (!arena_result.ok()) {
        std::printf("FAIL arena: %s\n", arena_result.status().message().c_str());
        fs::remove_all(dir, ec);
        return 1;
    }
    gpu::GpuArena arena = arena_result.release();

    Qwen35LoadOptions load_options;
    auto model_result = Qwen35Model::load_from_safetensors(dir.string(), arena, stream, load_options);
    if (!model_result.ok()) {
        std::printf("FAIL model load: %s\n", model_result.status().message().c_str());
        (void)arena.shutdown();
        fs::remove_all(dir, ec);
        return 1;
    }
    Qwen35Model model = model_result.release();

    const uint32_t max_blocks =
        ceil_div_u32(max_seq_len + 1u, page_tokens) + 1u;
    const uint32_t kv_pages =
        ceil_div_u32(max_seq_len + 1u, page_tokens) + 1u;

    auto seq_result =
        SequenceSlotPool::create(arena, max_scheduled_requests, max_blocks);
    if (!seq_result.ok()) {
        std::printf("FAIL sequence pool: %s\n", seq_result.status().message().c_str());
        (void)arena.shutdown();
        fs::remove_all(dir, ec);
        return 1;
    }
    SequenceSlotPool seq_pool = seq_result.release();

    auto gdn_layout = qwen35::make_qwen35_gdn_state_layout(model.text_config());
    auto gdn_result =
        GdnStatePool::create(arena, max_scheduled_requests, gdn_layout);
    if (!gdn_result.ok()) {
        std::printf("FAIL gdn pool: %s\n", gdn_result.status().message().c_str());
        (void)arena.shutdown();
        fs::remove_all(dir, ec);
        return 1;
    }
    GdnStatePool gdn_pool = gdn_result.release();

    auto kv_result = PagedKVPool::create(
        arena, kv_pages, page_tokens, count_attention_layers(model.text_config()),
        static_cast<uint32_t>(model.text_config().num_key_value_heads),
        static_cast<uint32_t>(model.text_config().attention_head_dim),
        KVCacheDType::BF16);
    if (!kv_result.ok()) {
        std::printf("FAIL kv pool: %s\n", kv_result.status().message().c_str());
        (void)arena.shutdown();
        fs::remove_all(dir, ec);
        return 1;
    }
    PagedKVPool kv_pool = kv_result.release();

    ExecutorConfig exec_config;
    exec_config.max_batch_rows = max_batch_rows;
    exec_config.max_scheduled_requests = max_scheduled_requests;
    auto exec_result =
        create_model_executor(model, seq_pool, gdn_pool, kv_pool, arena, exec_config);
    if (!exec_result.ok()) {
        std::printf("FAIL executor: %s\n", exec_result.status().message().c_str());
        (void)arena.shutdown();
        fs::remove_all(dir, ec);
        return 1;
    }
    Executor executor = exec_result.release();

    ContinuousBatcherConfig batcher_config;
    batcher_config.max_num_batched_tokens = max_batch_rows;
    batcher_config.max_scheduled_requests = max_scheduled_requests;
    batcher_config.max_seq_len = max_seq_len;
    batcher_config.page_tokens = page_tokens;
    batcher_config.eos_token_ids = model.text_config().stop_tokens;

    auto cache_result = PrefixCache::create(arena, kv_pool, gdn_pool, PrefixCacheConfig{});
    check(cache_result.ok(), "prefix cache create");
    if (!cache_result.ok()) {
        std::printf("cache create: %s\n", cache_result.status().message().c_str());
        executor_shutdown(executor);
        (void)arena.shutdown();
        fs::remove_all(dir, ec);
        return g_fail == 0 ? 0 : 1;
    }
    PrefixCache cache = cache_result.release();

    ContinuousBatcher batcher(executor, seq_pool, gdn_pool, kv_pool, batcher_config,
                              stream);
    batcher.set_prefix_cache(&cache);

    std::vector<int32_t> prompt100;
    prompt100.reserve(100);
    for (int i = 0; i < 100; ++i) {
        prompt100.push_back(static_cast<int32_t>((i * 37 + 11) % 128));
    }
    std::vector<int32_t> prompt112 = prompt100;
    prompt112.reserve(112);
    for (int i = 0; i < 12; ++i) {
        prompt112.push_back(static_cast<int32_t>(((100 + i) * 53) % 128));
    }

    RunOutcome first;
    run_request(batcher, prompt100, 8, first);
    check(first.ok, "first run: " + first.error);
    const PrefixCacheStats stats_first = cache.stats();
    check(stats_first.lookups == 1, "first run performs one lookup");
    check(stats_first.misses == 1 && stats_first.hits == 0, "first run is a miss");
    check(stats_first.restores == 0, "first run does not restore");
    check(first.restored == 0, "first run restored_tokens is zero");
    check(stats_first.captures_active == 0, "first run leaves no active capture");

    RunOutcome second;
    run_request(batcher, prompt100, 8, second);
    check(second.ok, "second run: " + second.error);
    const PrefixCacheStats stats_second = cache.stats();
    check(stats_second.hits == 1, "second run hits the cache");
    check(stats_second.restores == 1, "second run restores once");
    check(second.restored == 96, "second run restored_tokens is 96");
    check(!first.tokens.empty() && second.tokens == first.tokens,
          "second run regenerates the same tokens");
    check(stats_second.captures_active == 0, "second run leaves no active capture");
    print_tokens("first", first.tokens);
    print_tokens("second", second.tokens);

    RunOutcome extension;
    run_request(batcher, prompt112, 8, extension);
    check(extension.ok, "extension run: " + extension.error);
    const PrefixCacheStats stats_extension = cache.stats();
    check(stats_extension.hits == 2, "extension run hits the cache");
    check(stats_extension.restores == 2, "extension run restores once more");
    check(extension.restored == 96, "extension run restored_tokens is 96");
    check(stats_extension.captures_active == 0, "extension run leaves no active capture");
    print_tokens("extension", extension.tokens);

    batcher.set_prefix_cache(nullptr);

    RunOutcome disabled_same;
    run_request(batcher, prompt100, 8, disabled_same);
    check(disabled_same.ok, "disabled run: " + disabled_same.error);
    check(disabled_same.restored == 0, "disabled run restores nothing");
    check(!first.tokens.empty() && disabled_same.tokens == first.tokens,
          "disabled run regenerates the same tokens");

    RunOutcome disabled_extension;
    run_request(batcher, prompt112, 8, disabled_extension);
    check(disabled_extension.ok, "disabled extension run: " + disabled_extension.error);
    check(disabled_extension.restored == 0, "disabled extension run restores nothing");
    check(!extension.tokens.empty() &&
              disabled_extension.tokens == extension.tokens,
          "disabled extension run regenerates the same tokens");
    print_tokens("disabled extension", disabled_extension.tokens);

    const PrefixCacheStats stats_final = cache.stats();
    check(stats_final.lookups == 3, "detached batcher performs no further lookups");
    check(stats_final.captures_active == 0, "no capture leaks after all runs");

    (void)batcher.cancel_all();
    executor_shutdown(executor);
    const Status cache_shutdown = cache.shutdown();
    check(cache_shutdown.ok(), "prefix cache shutdown");
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    (void)arena.shutdown();
    fs::remove_all(dir, ec);

    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
