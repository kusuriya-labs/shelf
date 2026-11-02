#include <phaseshift/core/gpu/scoped_device.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <filesystem>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cerrno>
#include <string>
#include <vector>

#include "support/tp_run_harness.h"

namespace fs = std::filesystem;

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

static std::uint64_t arena_bytes_from_env(std::uint64_t fallback_gib) {
    std::uint64_t gib = fallback_gib;
    if (const char* arena_env = std::getenv("PHASESHIFT_TP_ARENA_GIB")) {
        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(arena_env, &end, 10);
        if (end != arena_env && parsed > 0) gib = static_cast<std::uint64_t>(parsed);
    }
    return gib * 1024ull * 1024ull * 1024ull;
}

int main() {
    const char* model_env = std::getenv("PHASESHIFT_TP_MODEL_DIR");
    if (model_env == nullptr || model_env[0] == '\0') {
        std::printf("SKIP: set PHASESHIFT_TP_MODEL_DIR to run the Qwen3.8-27B TP E2E\n");
        return 77;
    }
    const std::string model_dir = model_env;
    if (!fs::exists(fs::path(model_dir) / "config.json")) {
        std::printf("SKIP: PHASESHIFT_TP_MODEL_DIR has no config.json\n");
        return 77;
    }

    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 2) {
        std::printf("SKIP: qwen35 tp e2e requires 2 GPUs\n");
        return 77;
    }

    tp_harness::RunOptions tp1_opts;
    tp1_opts.model_dir = model_dir;
    tp1_opts.arena_bytes = arena_bytes_from_env(24);
    tp1_opts.max_seq_len = 64;
    tp1_opts.max_scheduled_tokens = 16;
    tp1_opts.max_scheduled_requests = 2;
    tp1_opts.page_tokens = 16;
    tp1_opts.taps = {0, 3, 7, 15, 31, 62};
    tp1_opts.verify_crc = true;
    tp1_opts.prompt = {16, 220, 17, 220, 18, 220, 19, 220,
                       20, 220, 21, 220, 22, 220, 23};
    tp1_opts.max_new_tokens = 16;

    if (const char* prompt_env = std::getenv("PHASESHIFT_TP_PROMPT")) {
        tp1_opts.prompt.clear();
        const char* p = prompt_env;
        while (*p != '\0') {
            char* end = nullptr;
            const long value = std::strtol(p, &end, 10);
            if (end == p) break;
            tp1_opts.prompt.push_back(static_cast<int32_t>(value));
            p = end;
            while (*p == ',' || *p == ' ') ++p;
        }
        if (tp1_opts.prompt.empty()) {
            tp1_opts.prompt = {16, 220, 17, 220, 18, 220, 19, 220,
                               20, 220, 21, 220, 22, 220, 23};
        }
    }

    std::printf("prompt:");
    for (int32_t t : tp1_opts.prompt) std::printf(" %d", t);
    std::printf("\n");

    tp_harness::RunOptions tp2_opts = tp1_opts;
    tp2_opts.arena_bytes = arena_bytes_from_env(20);
    std::printf("arena GiB tp1=%llu tp2=%llu\n",
                static_cast<unsigned long long>(tp1_opts.arena_bytes >> 30),
                static_cast<unsigned long long>(tp2_opts.arena_bytes >> 30));

    std::printf("=== tp1 (single GPU) ===\n");
    tp_harness::RunResult tp1 = tp_harness::run_tp1(tp1_opts);
    if (!tp1.ok) {
        fail("tp1 run: " + tp1.error);
        std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
        return g_fail == 0 ? 0 : 1;
    }

    std::printf("=== tp2 (2 GPU) ===\n");
    tp_harness::RunResult tp2 = tp_harness::run_tp2(tp2_opts);
    if (!tp2.ok) {
        fail("tp2 run: " + tp2.error);
        std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
        return g_fail == 0 ? 0 : 1;
    }
    std::printf("%s", tp2.report.c_str());

    print_tokens("tp1 tokens", tp1.tokens);
    print_tokens("tp2 tokens", tp2.tokens);

    std::printf(
        "TP_TIMING tp1_prefill_ms=%.3f tp1_decode_ms=%.3f tp1_tok=%zu "
        "tp1_decode_steps=%zu tp2_prefill_ms=%.3f tp2_decode_ms=%.3f tp2_tok=%zu "
        "tp2_decode_steps=%zu\n",
        tp1.prefill_ms, tp1.decode_ms, tp1.tokens.size(), tp1.decode_steps,
        tp2.prefill_ms, tp2.decode_ms, tp2.tokens.size(), tp2.decode_steps);
    if (tp1.decode_ms > 0.0 && tp1.decode_steps > 0) {
        std::printf("TP_TOKPS tp1_decode=%.2f\n",
                    1000.0 * static_cast<double>(tp1.decode_steps) / tp1.decode_ms);
    }
    if (tp2.decode_ms > 0.0 && tp2.decode_steps > 0) {
        std::printf("TP_TOKPS tp2_decode=%.2f\n",
                    1000.0 * static_cast<double>(tp2.decode_steps) / tp2.decode_ms);
    }

    check(!tp1.tokens.empty(), "tp1 produced tokens");
    check(tp1.tokens.size() == tp2.tokens.size(),
          "tp1 and tp2 produced the same number of tokens");
    if (!tp1.tokens.empty() && !tp2.tokens.empty()) {
        check(tp1.tokens[0] == tp2.tokens[0], "greedy first token matches (tp1 vs tp2)");
    }
    check(tp1.tokens == tp2.tokens, "greedy token sequence matches (tp1 vs tp2)");

    check(tp1.gate_rows > tp2.gate_rows, "tp2 gate rows are rank-local");
    check(tp1.kv_pool_heads > tp2.kv_pool_heads, "tp2 kv heads are rank-local");
    std::printf("tp1 gate_rows=%u tp2 gate_rows=%u tp1 kv_heads=%u tp2 kv_heads=%u\n",
                tp1.gate_rows, tp2.gate_rows, tp1.kv_pool_heads, tp2.kv_pool_heads);

    if (!tp1.capture.logits_row0.empty() && !tp2.capture.logits_row0.empty()) {
        const int a1 = tp_harness::argmax(tp1.capture.logits_row0);
        const int a2 = tp_harness::argmax(tp2.capture.logits_row0);
        check(a1 == a2, "prefill logits argmax matches (tp1 vs tp2)");
        const float diff =
            tp_harness::max_abs_diff(tp1.capture.logits_row0, tp2.capture.logits_row0);
        std::printf("prefill logits argmax=%d diff=%g\n", a1, static_cast<double>(diff));
        check(diff <= 2.0f, "prefill logits within tolerance");
    }
    if (!tp1.capture.hidden_row0.empty() && !tp2.capture.hidden_row0.empty()) {
        const float hidden_diff =
            tp_harness::max_abs_diff(tp1.capture.hidden_row0, tp2.capture.hidden_row0);
        float hidden_scale = 1.0f;
        for (float v : tp1.capture.hidden_row0) hidden_scale = std::max(hidden_scale, std::fabs(v));
        std::printf("final hidden max abs diff tp1 vs tp2: %g (scale=%g rel=%g)\n",
                    static_cast<double>(hidden_diff), static_cast<double>(hidden_scale),
                    static_cast<double>(hidden_diff / hidden_scale));
        check(hidden_diff <= 0.10f * hidden_scale, "final hidden within relative tolerance");
    }
    if (!tp1.capture.tap0_row0.empty() && !tp2.capture.tap0_row0.empty()) {
        const float tap0_diff =
            tp_harness::max_abs_diff(tp1.capture.tap0_row0, tp2.capture.tap0_row0);
        std::printf("layer0 output max abs diff tp1 vs tp2: %g\n",
                    static_cast<double>(tap0_diff));
        check(tap0_diff <= 1.0f, "layer0 output within tolerance");
    }
    if (!tp1.capture.tap1_row0.empty() && !tp2.capture.tap1_row0.empty()) {
        const float tap1_diff =
            tp_harness::max_abs_diff(tp1.capture.tap1_row0, tp2.capture.tap1_row0);
        std::printf("layerN output max abs diff tp1 vs tp2: %g (tp1[0]=%g tp2[0]=%g)\n",
                    static_cast<double>(tap1_diff),
                    static_cast<double>(tp1.capture.tap1_row0[0]),
                    static_cast<double>(tp2.capture.tap1_row0[0]));
    }
    {
        static const int kTapLayers[] = {0, 3, 7, 15, 31, 62};
        for (std::size_t i = 0; i < tp1.capture.all_taps.size() &&
                                i < tp2.capture.all_taps.size() &&
                                i < sizeof(kTapLayers) / sizeof(kTapLayers[0]); ++i) {
            if (tp1.capture.all_taps[i].empty() || tp2.capture.all_taps[i].empty()) continue;
            const float max_diff = tp_harness::max_abs_diff(tp1.capture.all_taps[i],
                                                             tp2.capture.all_taps[i]);
            float scale = 1.0f;
            for (float v : tp1.capture.all_taps[i]) scale = std::max(scale, std::fabs(v));
            const float rel = max_diff / scale;
            std::printf("layer %2d: tap max abs diff=%g rel=%g element0 diff=%g scale=%g\n",
                        kTapLayers[i], static_cast<double>(max_diff),
                        static_cast<double>(rel),
                        static_cast<double>(std::fabs(tp1.capture.all_taps[i][0] -
                                                      tp2.capture.all_taps[i][0])),
                        static_cast<double>(scale));
            check(rel <= 0.10f,
                  "layer " + std::to_string(kTapLayers[i]) +
                      " output within relative tolerance");
        }
    }
    if (!tp1.capture.hidden_row0.empty()) {
        std::printf("tp1 hidden[0..3]=%g %g %g %g\n",
                    static_cast<double>(tp1.capture.hidden_row0[0]),
                    static_cast<double>(tp1.capture.hidden_row0[1]),
                    static_cast<double>(tp1.capture.hidden_row0[2]),
                    static_cast<double>(tp1.capture.hidden_row0[3]));
        if (!tp2.capture.hidden_row0.empty()) {
            std::printf("tp2 hidden[0..3]=%g %g %g %g\n",
                        static_cast<double>(tp2.capture.hidden_row0[0]),
                        static_cast<double>(tp2.capture.hidden_row0[1]),
                        static_cast<double>(tp2.capture.hidden_row0[2]),
                        static_cast<double>(tp2.capture.hidden_row0[3]));
        }
    }

    if (!tp2.capture.hidden_row0.empty() && !tp2.rank1_capture.hidden_row0.empty()) {
        check(tp2.capture.hidden_row0 == tp2.rank1_capture.hidden_row0,
              "tp2 rank0 and rank1 final hidden are identical");
        if (tp2.capture.hidden_row0 != tp2.rank1_capture.hidden_row0) {
            std::printf("rank0 vs rank1 hidden diff: %g\n",
                        static_cast<double>(tp_harness::max_abs_diff(
                            tp2.capture.hidden_row0, tp2.rank1_capture.hidden_row0)));
        }
    }
    if (!tp2.capture.logits_row0.empty() && !tp2.rank1_capture.logits_row0.empty()) {
        check(tp2.capture.logits_row0 == tp2.rank1_capture.logits_row0,
              "tp2 rank0 and rank1 logits are identical");
    }

    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
