// shelf-bench tg — token generation (decode) の計測。
//
// コンテキストの prefill は計測区間の外（時間は context_prefill_ms として参考表示）。
// --warmup ステップを破棄したあと、--tokens ステップを 1 区間で測る。
// 各ステップの壁時計時間（実行器がステップ末で sync）も取り、
// first_token_ms / step_ms_* を出す（phaseshift-bench には無い差分: 参照実装は
// 総時間のみ。shelf は run_step が毎回 wall ms を返すため取得が容易）。
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "bench_common.hpp"
#include "common/app_support.hpp"

#include <shelf/model/qwen35/step_graph.hpp>
#include <shelf/model/qwen35/text_config.hpp>

#ifdef SHELF_HAVE_HIP
#include <shelf/backend/hip/step_executor.hpp>
#endif

namespace shelf::bench {
namespace {

void usage(std::FILE* out) {
    std::fprintf(out,
        "usage: shelf-bench tg --model-dir DIR [options]\n"
        "\n"
        "options:\n"
        "  --model-dir DIR     モデルディレクトリ（--model の別名・必須）\n"
        "  --context N         prefill するコンテキスト長（既定 128）\n"
        "  --tokens N          測定するデコード歩数（既定 8）\n"
        "  --prefill-chunk N   prefill の 1 ステップの行数（既定 0 = min(context, 64)）\n"
        "  --warmup N          測定前に破棄するデコード歩数（既定 0）\n"
        "  --mode M            forward（既定・logits を取らない）| greedy（毎歩 argmax）\n"
        "  --gpu N             GPU 番号（既定 0）\n"
        "  --seed N            合成トークン列のシード（既定 1234）\n"
        "  --arena-mib N       アリーナサイズ MiB（既定 0 = グラフから自動算出、正の値で固定）\n"
        "  --output PATH       CSV を追記（ファイルが無ければヘッダを書く）\n"
        "  --help, -h          このヘルプ（GPU を触らず exit 0）\n"
        "\n"
        "notes:\n"
        "  測定区間はデコードのステップのみ。first_token_ms は最初の 1 ステップの壁時計時間。\n"
        "  greedy は毎ステップ logits の D2H と argmax を含む（現実に近いが重い）。\n");
}

// HIP 無しビルドでは使わないため maybe_unused。
[[maybe_unused]] std::vector<std::int32_t> make_tokens(std::size_t n, std::size_t vocab,
                                                       std::int64_t seed) {
    std::vector<std::int32_t> out(n);
    std::uint64_t s =
        static_cast<std::uint64_t>(seed) * 6364136223846793005ULL + 1442695040888963407ULL;
    for (std::size_t i = 0; i < n; ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        out[i] = vocab == 0 ? 0 : static_cast<std::int32_t>((s >> 33) % vocab);
    }
    return out;
}

[[maybe_unused]] std::int32_t argmax_i32(const std::vector<float>& logits) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < logits.size(); ++i)
        if (logits[i] > logits[best]) best = i;
    return static_cast<std::int32_t>(best);
}

}  // namespace

int run_tg(int argc, char** argv) {
    Args a;
    std::string err;
    if (!a.parse(argc, argv, &err)) {
        std::fprintf(stderr, "tg: %s\n", err.c_str());
        usage(stderr);
        return kExitUsage;
    }
    if (a.has("help") || a.has("h")) {
        usage(stdout);
        return kExitOk;
    }

    // 未知のオプションは起動エラー（shelf-validate と同じ規約）。
    {
        const std::vector<std::string> allowed = {"help", "h", "model-dir", "model", "context", "tokens", "prefill-chunk", "warmup", "mode", "gpu", "seed", "arena-mib", "output"};
        std::string bad;
        if (!only_known_keys(a, allowed, &bad)) {
            std::fprintf(stderr, "tg: unknown option: --%s\n", bad.c_str());
            usage(stderr);
            return kExitUsage;
        }
    }

    bool ok = true;
    const std::int64_t context = a.i64("context", 128, &ok);
    const std::int64_t tokens = a.i64("tokens", 8, &ok);
    const std::int64_t chunk_arg = a.i64("prefill-chunk", 0, &ok);
    const std::int64_t warmup = a.i64("warmup", 0, &ok);
    const std::int64_t gpu = a.i64("gpu", 0, &ok);
    const std::int64_t seed = a.i64("seed", 1234, &ok);
    const std::int64_t arena_mib = a.i64("arena-mib", 0, &ok);
    const std::string mode = a.str("mode", "forward");
    const std::string model_dir = a.str("model-dir", a.str("model"));
    const std::string output = a.str("output");
    if (!ok) {
        std::fprintf(stderr, "tg: numeric option has a bad value\n");
        usage(stderr);
        return kExitUsage;
    }
    if (model_dir.empty()) {
        std::fprintf(stderr, "tg: --model-dir is required\n");
        usage(stderr);
        return kExitUsage;
    }
    if (mode != "forward" && mode != "greedy") {
        std::fprintf(stderr, "tg: --mode must be forward or greedy\n");
        usage(stderr);
        return kExitUsage;
    }
    if (context <= 0 || tokens <= 0 || warmup < 0 || gpu < 0 || seed < 0 ||
        arena_mib < 0 || !a.positional().empty()) {
        std::fprintf(stderr, "tg: bad option value or unexpected argument\n");
        usage(stderr);
        return kExitUsage;
    }

#ifndef SHELF_HAVE_HIP
    (void)chunk_arg;
    (void)arena_mib;
    std::fprintf(stderr,
                  "tg: this build has no HIP backend (rebuild with SHELF_ENABLE_HIP=ON)\n");
    return kExitRuntime;
#else
    namespace qwen = shelf::model::qwen35;

    const auto cfg_r = qwen::load_text_config_json(
        app::read_file_str(model_dir + "/config.json"));
    if (!cfg_r.ok()) {
        std::fprintf(stderr, "tg: config: %s\n", cfg_r.status().message().c_str());
        return kExitRuntime;
    }
    const auto& cfg = cfg_r.value();

    const std::size_t max_rows = shelf::backend::hip::HipStepExecutor::kMaxStepRows;
    const std::size_t chunk =
        chunk_arg > 0 ? static_cast<std::size_t>(chunk_arg)
                      : std::min<std::size_t>(static_cast<std::size_t>(context), max_rows);
    if (chunk == 0 || chunk > max_rows) {
        std::fprintf(stderr, "tg: --prefill-chunk must be in [1, %zu]\n", max_rows);
        return kExitUsage;
    }
    // KV は context + warmup + 測定分を含めて確保する。
    const std::size_t capacity = static_cast<std::size_t>(context) +
                                 static_cast<std::size_t>(tokens) +
                                 static_cast<std::size_t>(warmup) + 4;

    auto decode_r = qwen::build_step_graph(cfg, capacity, 1);
    if (!decode_r.ok()) {
        std::fprintf(stderr, "tg: decode graph: %s\n", decode_r.status().message().c_str());
        return kExitRuntime;
    }
    auto decode_plan = decode_r.release();

    std::unique_ptr<qwen::StepGraphPlan> prefill_plan;
    if (chunk != 1) {
        auto pf_r = qwen::build_step_graph(cfg, capacity, chunk);
        if (!pf_r.ok()) {
            std::fprintf(stderr, "tg: prefill graph: %s\n", pf_r.status().message().c_str());
            return kExitRuntime;
        }
        prefill_plan = std::make_unique<qwen::StepGraphPlan>(pf_r.release());
    }
    const auto& pf_graph = prefill_plan ? prefill_plan->graph : decode_plan.graph;

    app::ShardSet shards;
    if (auto st = app::open_shards(model_dir, &shards); !st.ok()) {
        std::fprintf(stderr, "tg: %s\n", st.message().c_str());
        return kExitRuntime;
    }

    shelf::backend::hip::HipStepExecutorOptions ho;
    ho.device = static_cast<int>(gpu);
    ho.rope_theta = cfg.rope_theta;
    ho.rope_rotary_dim = cfg.rotary_dim;
    if (arena_mib > 0) ho.arena_bytes = static_cast<std::size_t>(arena_mib) << 20;
    shelf::backend::hip::HipStepExecutor ex(ho);
    // 重み id は config 由来なので decode プランの weights を上げれば両方で使える。
    if (auto st = app::upload_plan(ex, decode_plan, shards); !st.ok()) {
        std::fprintf(stderr, "tg: upload: %s\n", st.message().c_str());
        return kExitRuntime;
    }
    const auto ctx_tokens =
        make_tokens(static_cast<std::size_t>(context), cfg.vocab_size, seed);

    // エラー報告（アリーナ枯渇時はサイズを上げるよう促す）。
    // 実サイズは失敗時点の値を読む（prefill で自動拡張されている場合があるため）。
    auto fail_step = [&](const char* what, const shelf::Status& st) {
        std::fprintf(stderr, "tg: %s: %s\n", what, st.message().c_str());
        if (st.message().find("arena") != std::string::npos)
            std::fprintf(stderr,
                         "tg: hint: the arena is a per-step bump allocator; raise "
                         "--arena-mib above the current %lld MiB\n",
                         static_cast<long long>(ex.arena_bytes() >> 20));
    };

    // ---- コンテキスト prefill（計測区間外） ----
    const auto pf0 = std::chrono::steady_clock::now();
    for (std::size_t c = 0; c < ctx_tokens.size(); c += chunk) {
        const std::size_t n = std::min(chunk, ctx_tokens.size() - c);
        const auto st = ex.run_step(pf_graph, ctx_tokens.data() + c, n,
                                    static_cast<std::uint32_t>(c));
        if (!st.ok()) {
            fail_step("context chunk", st);
            return kExitRuntime;
        }
    }
    const double context_prefill_ms = std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() - pf0)
                                          .count();

    std::int32_t feed = ctx_tokens.back();
    std::uint32_t pos = static_cast<std::uint32_t>(context);

    auto step_once = [&](bool want_logits) -> shelf::Status {
        SHELF_RETURN_IF_ERROR(ex.run_step(decode_plan.graph, &feed, 1, pos));
        ++pos;
        if (want_logits) {
            auto lg = ex.fetch_logits(decode_plan.graph);
            SHELF_RETURN_IF_ERROR(lg.status());
            feed = argmax_i32(lg.value());
        }
        return shelf::Status::make_ok();
    };

    // ---- warmup（破棄） ----
    for (std::int64_t w = 0; w < warmup; ++w) {
        const auto st = step_once(false);
        if (!st.ok()) {
            fail_step("warmup", st);
            return kExitRuntime;
        }
    }

    // ---- 測定 ----
    const bool greedy = mode == "greedy";
    std::vector<double> step_ms;
    step_ms.reserve(static_cast<std::size_t>(tokens));
    const auto t0 = std::chrono::steady_clock::now();
    for (std::int64_t s = 0; s < tokens; ++s) {
        const auto st = step_once(greedy);
        if (!st.ok()) {
            fail_step("decode step", st);
            return kExitRuntime;
        }
        step_ms.push_back(ex.stats().last_step_ms);
        if (!greedy) feed = ctx_tokens.back();
    }
    const double wall_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - t0)
                               .count();

    std::vector<double> per_step = step_ms;
    const Stats st_step = summarize(std::move(per_step));
    const double us_per_token = wall_ms * 1000.0 / static_cast<double>(tokens);
    const double tokens_per_sec = us_per_token > 0.0 ? 1e6 / us_per_token : 0.0;
    const double first_token_ms = step_ms.empty() ? 0.0 : step_ms.front();

    // 実際に確保されたサイズ（prefill で自動拡張されている場合があるので最後に読む）。
    const std::int64_t arena_eff = static_cast<std::int64_t>(ex.arena_bytes() >> 20);

    std::printf("shelf-bench tg\n");
    kv_print("schema_version", static_cast<std::int64_t>(2));
    kv_print("device", std::string("hip"));
    kv_print("gpu", gpu);
    kv_print("model", model_dir);
    kv_print("context", context);
    kv_print("tokens", tokens);
    kv_print("prefill_chunk", static_cast<std::int64_t>(chunk));
    kv_print("warmup", warmup);
    kv_print("arena_mib", arena_eff);
    kv_print("mode", mode);
    kv_print("context_prefill_ms", context_prefill_ms);
    kv_print("wall_ms_total", wall_ms);
    kv_print("wall_us_per_token", us_per_token);
    kv_print("wall_tokens_per_sec", tokens_per_sec, 2);
    kv_print("first_token_ms", first_token_ms);
    kv_print("step_ms_mean", st_step.mean);
    kv_print("step_ms_p50", st_step.p50);
    kv_print("step_ms_p90", st_step.p90);
    kv_print("step_ms_max", st_step.max);

    if (!output.empty()) {
        const char* kHeader =
            "schema_version,device,gpu,model,context,tokens,prefill_chunk,warmup,"
            "arena_mib,mode,context_prefill_ms,wall_ms_total,wall_us_per_token,"
            "wall_tokens_per_sec,first_token_ms,step_ms_mean,step_ms_p50,step_ms_p90,"
            "step_ms_max";
        char row[640];
        std::snprintf(row, sizeof row,
                      "2,hip,%lld,%s,%lld,%lld,%lld,%lld,%lld,%s,%.3f,%.3f,%.3f,%.2f,"
                      "%.3f,%.3f,%.3f,%.3f,%.3f",
                      static_cast<long long>(gpu), model_dir.c_str(),
                      static_cast<long long>(context), static_cast<long long>(tokens),
                      static_cast<long long>(chunk), static_cast<long long>(warmup),
                      static_cast<long long>(arena_eff),
                      mode.c_str(), context_prefill_ms, wall_ms, us_per_token,
                      tokens_per_sec, first_token_ms, st_step.mean, st_step.p50,
                      st_step.p90, st_step.max);
        std::string csv_err;
        if (!append_csv(output, kHeader, {row}, &csv_err)) {
            std::fprintf(stderr, "tg: %s\n", csv_err.c_str());
            return kExitRuntime;
        }
    }
    return kExitOk;
#endif  // SHELF_HAVE_HIP
}

}  // namespace shelf::bench
