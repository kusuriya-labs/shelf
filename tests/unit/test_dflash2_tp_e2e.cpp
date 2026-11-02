#include <phaseshift/core/gpu/scoped_device.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <cstdio>
#include <string>

#include "support/dflash2_tp_harness.h"

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

int main() {
    const char* model_env = std::getenv("PHASESHIFT_TP_MODEL_DIR");
    const char* dflash_env = std::getenv("PHASESHIFT_DFLASH2_MODEL_DIR");
    if (model_env == nullptr || model_env[0] == '\0' || dflash_env == nullptr ||
        dflash_env[0] == '\0') {
        std::printf("SKIP: set PHASESHIFT_TP_MODEL_DIR and PHASESHIFT_DFLASH2_MODEL_DIR\n");
        return 77;
    }
    const std::string model_dir = model_env;
    const std::string dflash2_dir = dflash_env;
    if (!fs::exists(fs::path(model_dir) / "config.json")) {
        std::printf("SKIP: PHASESHIFT_TP_MODEL_DIR has no config.json\n");
        return 77;
    }
    if (!fs::exists(fs::path(dflash2_dir) / "config.json")) {
        std::printf("SKIP: PHASESHIFT_DFLASH2_MODEL_DIR has no config.json\n");
        return 77;
    }

    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 2) {
        std::printf("SKIP: dflash2 tp e2e requires 2 GPUs\n");
        return 77;
    }

    auto dflash_config_result =
        ps::qwen35::dflash2::read_dflash2_config(dflash2_dir);
    if (!dflash_config_result.ok()) {
        fail("dflash2 config: " + dflash_config_result.status().message());
        std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
        return g_fail == 0 ? 0 : 1;
    }
    const auto dflash_config = dflash_config_result.release();

    tp_harness::DFlash2Options opts;
    opts.dflash2_model_dir = dflash2_dir;
    opts.drafts = 7;
    tp_harness::RunOptions& o = opts.base;
    o.model_dir = model_dir;
    o.arena_bytes = 24ull * 1024ull * 1024ull * 1024ull;
    o.max_seq_len = 256;
    o.max_scheduled_tokens = 32;
    o.max_scheduled_requests = 2;
    o.page_tokens = 16;
    o.verify_crc = false;
    o.prompt = {304, 17, 283, 2454, 304};
    o.max_new_tokens = 48;
    o.taps.clear();
    for (std::size_t i = 0; i < dflash_config.num_target_layer_ids &&
                            i < tp_harness::kMaxTargetHiddenTaps;
         ++i) {
        o.taps.push_back(static_cast<int>(dflash_config.target_layer_ids[i]));
    }

    bool custom_prompt = false;
    if (const char* pf = std::getenv("PHASESHIFT_DFLASH2_PROMPT_FILE")) {
        FILE* fp = std::fopen(pf, "r");
        if (fp == nullptr) {
            std::printf("FAIL: cannot open PHASESHIFT_DFLASH2_PROMPT_FILE %s\n", pf);
            return 1;
        }
        o.prompt.clear();
        while (true) {
            char* end = nullptr;
            char buf[64];
            if (std::fscanf(fp, " %63[^, \t\n],", buf) != 1) break;
            const long value = std::strtol(buf, &end, 10);
            if (end == buf) break;
            o.prompt.push_back(static_cast<int32_t>(value));
        }
        std::fclose(fp);
        if (o.prompt.empty()) {
            std::printf("FAIL: prompt file %s yielded no tokens\n", pf);
            return 1;
        }
        custom_prompt = true;
    }
    if (const char* prompt_env = std::getenv("PHASESHIFT_TP_PROMPT")) {
        o.prompt.clear();
        const char* p = prompt_env;
        while (*p != '\0') {
            char* end = nullptr;
            const long value = std::strtol(p, &end, 10);
            if (end == p) break;
            o.prompt.push_back(static_cast<int32_t>(value));
            p = end;
            while (*p == ',' || *p == ' ') ++p;
        }
        if (o.prompt.empty()) o.prompt = {304, 17, 283, 2454, 304};
        custom_prompt = true;
    }
    if (const char* tok_env = std::getenv("PHASESHIFT_DFLASH2_TOKENS")) {
        const long t = std::strtol(tok_env, nullptr, 10);
        if (t > 0) o.max_new_tokens = static_cast<std::uint32_t>(t);
    }
    const std::uint32_t needed_seq =
        static_cast<std::uint32_t>(o.prompt.size()) + o.max_new_tokens + 64u;
    if (needed_seq > o.max_seq_len) o.max_seq_len = needed_seq;

    std::printf("prompt:");
    for (int32_t t : o.prompt) std::printf(" %d", t);
    std::printf("\ntaps:");
    for (int t : o.taps) std::printf(" %d", t);
    std::printf("\n");

    std::printf("=== dflash2 tp1 (single GPU) ===\n");
    tp_harness::RunResult tp1_spec = tp_harness::run_dflash2_tp1(opts);
    if (!tp1_spec.ok) {
        fail("tp1 speculative run: " + tp1_spec.error);
        std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
        return g_fail == 0 ? 0 : 1;
    }

    std::printf("%s", tp1_spec.report.c_str());
    std::printf("=== target-only greedy tp1 (single GPU) ===\n");
    tp_harness::RunResult tp1_plain = tp_harness::run_tp1(o);
    if (!tp1_plain.ok) {
        fail("tp1 greedy run: " + tp1_plain.error);
        std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
        return g_fail == 0 ? 0 : 1;
    }

    tp_harness::DFlash2Options tp2_opts = opts;
    tp2_opts.base.arena_bytes = 20ull * 1024ull * 1024ull * 1024ull;
    tp_harness::RunOptions tp2_plain_opts = o;
    tp2_plain_opts.arena_bytes = 20ull * 1024ull * 1024ull * 1024ull;

    std::printf("=== target-only greedy tp2 (2 GPU) ===\n");
    tp_harness::RunResult tp2_plain = tp_harness::run_tp2(tp2_plain_opts);
    if (!tp2_plain.ok) {
        fail("tp2 greedy run: " + tp2_plain.error);
        std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
        return g_fail == 0 ? 0 : 1;
    }
    std::printf("%s", tp2_plain.report.c_str());

    std::printf("=== dflash2 tp2 (2 GPU, verify only) ===\n");
    tp_harness::RunResult tp2_spec = tp_harness::run_dflash2_tp2(tp2_opts);
    if (!tp2_spec.ok) {
        fail("tp2 speculative run: " + tp2_spec.error);
        std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
        return g_fail == 0 ? 0 : 1;
    }
    std::printf("%s", tp2_spec.report.c_str());

    print_tokens("tp1 speculative", tp1_spec.tokens);
    print_tokens("tp1 greedy     ", tp1_plain.tokens);
    print_tokens("tp2 speculative", tp2_spec.tokens);
    print_tokens("tp2 greedy     ", tp2_plain.tokens);

    check(!tp1_spec.tokens.empty(), "tp1 speculative produced tokens");
    check(!tp1_plain.tokens.empty(), "tp1 greedy produced tokens");
    check(!tp2_spec.tokens.empty(), "tp2 speculative produced tokens");
    check(!tp2_plain.tokens.empty(), "tp2 greedy produced tokens");

    check(tp1_spec.tokens == tp1_plain.tokens,
          "tp1: speculative decode matches target-only greedy");
    check(tp2_spec.tokens == tp2_plain.tokens,
          "tp2: speculative decode matches target-only greedy");

    constexpr std::size_t kGreedyMatchTokens = 16;
    const bool greedy_match_16 =
        tp1_plain.tokens.size() >= kGreedyMatchTokens &&
        tp2_plain.tokens.size() >= kGreedyMatchTokens &&
        std::equal(tp1_plain.tokens.begin(),
                   tp1_plain.tokens.begin() + kGreedyMatchTokens,
                   tp2_plain.tokens.begin());
    std::printf("tp1 vs tp2 greedy first-16: %s%s\n", greedy_match_16 ? "match" : "DIFFER",
                custom_prompt
                    ? " (custom prompt: informational only, docs §9 margin applies to the default prompt)"
                    : "");
    if (!custom_prompt) {
        check(greedy_match_16,
              "target-only greedy matches for the first 16 tokens (tp1 vs tp2)");
    }

    std::printf("tp1 vs tp2 speculative prefix:");
    std::size_t same = 0;
    const std::size_t common =
        std::min(tp1_spec.tokens.size(), tp2_spec.tokens.size());
    while (same < common && tp1_spec.tokens[same] == tp2_spec.tokens[same]) ++same;
    std::printf(" %zu/%zu (drafter taps are tp-backend dependent, not asserted)\n",
                same, common);

    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
