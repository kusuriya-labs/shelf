// shelf-bench pp — prompt processing (prefill) の計測。
//
// 1 回の実行 = コンテキスト全体を --prefill-chunk 行ずつ run_step で流す。
// 計測区間はステップ列の壁時計時間（実行器がステップ末で sync するため
// GPU 時間とほぼ等価）。ロード・アップロード・状態宣告は区間外。
// runs 回の中央値を主指標にする（phaseshift-bench pp の規約に合わせる）。
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "bench_common.hpp"
#include "common/app_support.hpp"
#include "common/pipeline.hpp"

#include <shelf/model/qwen35/step_graph.hpp>
#include <shelf/model/qwen35/text_config.hpp>

#ifdef SHELF_HAVE_HIP
#include <shelf/backend/hip/step_executor.hpp>
#endif

namespace shelf::bench {
namespace {

void usage(std::FILE* out) {
    std::fprintf(out,
        "usage: shelf-bench pp --model-dir DIR [options]\n"
        "\n"
        "options:\n"
        "  --model-dir DIR     モデルディレクトリ（--model の別名・必須）\n"
        "  --prompt-tokens N   供給するトークン数（既定 128）\n"
        "  --prefill-chunk N   1 ステップの行数（既定 0 = min(prompt, 64)）\n"
        "  --runs N            計測回数（既定 3。中央値を主指標にする）\n"
        "  --warmup N          計測前に破棄する回数（既定 0）\n"
        "  --gpu N             GPU 番号（既定 0）\n"
        "  --seed N            合成トークン列のシード（既定 1234）\n"
        "  --arena-mib N       アリーナサイズ MiB（既定 0 = グラフから自動算出、正の値で固定）\n"
        "  --output PATH       CSV を追記（ファイルが無ければヘッダを書く）\n"
        "  --help, -h          このヘルプ（GPU を触らず exit 0）\n"
        "\n"
        "notes:\n"
        "  入力トークンは合成列（内容は不問・語彙内に収める）。\n"
        "  wall_* はステップ末同期込みの壁時計時間。GPU event 計測は未実装。\n"
        "  各 run は reset_states() で独立させる。\n"
        "  アリーナはステップ内で再利用しない（バンプ割当）ため必要量は行数に比例する。\n"
        "  「arena exhausted」が出たら --arena-mib を増やす。\n");
}

// 合成トークン列（計測対象はカーネル経路なので内容は不問）。
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

}  // namespace

int run_pp(int argc, char** argv) {
    Args a;
    std::string err;
    if (!a.parse(argc, argv, &err)) {
        std::fprintf(stderr, "pp: %s\n", err.c_str());
        usage(stderr);
        return kExitUsage;
    }
    if (a.has("help") || a.has("h")) {
        usage(stdout);
        return kExitOk;
    }

    // 未知のオプションは起動エラー（shelf-validate と同じ規約）。
    {
        const std::vector<std::string> allowed = {"help", "h", "model-dir", "model", "prompt-tokens", "prefill-chunk", "runs", "warmup", "gpu", "seed", "arena-mib", "output", "pipeline-parallel"};
        std::string bad;
        if (!only_known_keys(a, allowed, &bad)) {
            std::fprintf(stderr, "pp: unknown option: --%s\n", bad.c_str());
            usage(stderr);
            return kExitUsage;
        }
    }

    bool ok = true;
    const std::int64_t prompt_tokens = a.i64("prompt-tokens", 128, &ok);
    const std::int64_t chunk_arg = a.i64("prefill-chunk", 0, &ok);
    const std::int64_t runs = a.i64("runs", 3, &ok);
    const std::int64_t warmup = a.i64("warmup", 0, &ok);
    const std::int64_t gpu = a.i64("gpu", 0, &ok);
    const std::int64_t seed = a.i64("seed", 1234, &ok);
    const std::int64_t arena_mib = a.i64("arena-mib", 0, &ok);
    const std::int64_t pp = a.i64("pipeline-parallel", 1, &ok);
    const std::string model_dir = a.str("model-dir", a.str("model"));
    const std::string output = a.str("output");
    if (!ok) {
        std::fprintf(stderr, "pp: numeric option has a bad value\n");
        usage(stderr);
        return kExitUsage;
    }
    if (model_dir.empty()) {
        std::fprintf(stderr, "pp: --model-dir is required\n");
        usage(stderr);
        return kExitUsage;
    }
    if (prompt_tokens <= 0 || runs <= 0 || warmup < 0 || gpu < 0 || seed < 0 ||
        arena_mib < 0 || pp < 1 || pp > 64 || !a.positional().empty()) {
        std::fprintf(stderr, "pp: bad option value or unexpected argument\n");
        usage(stderr);
        return kExitUsage;
    }

#ifndef SHELF_HAVE_HIP
    (void)chunk_arg;
    (void)arena_mib;
    (void)pp;
    std::fprintf(stderr,
                  "pp: this build has no HIP backend (rebuild with SHELF_ENABLE_HIP=ON)\n");
    return kExitRuntime;
#else
    namespace qwen = shelf::model::qwen35;

    const std::string config_text = app::read_file_str(model_dir + "/config.json");
    const auto cfg_r = qwen::load_text_config_json(config_text);
    if (!cfg_r.ok()) {
        std::fprintf(stderr, "pp: config: %s\n", cfg_r.status().message().c_str());
        return kExitRuntime;
    }
    const auto& cfg = cfg_r.value();
    // FP8 の粒度は checkpoint の宣言が正（実行器へはここから渡す）。
    const auto fmt_r = app::read_fp8_format(config_text);
    if (!fmt_r.ok()) {
        std::fprintf(stderr, "pp: quantization_config: %s\n",
                     fmt_r.status().message().c_str());
        return kExitRuntime;
    }
    const auto fmt = fmt_r.value();

    const std::size_t max_rows = shelf::backend::hip::HipStepExecutor::kMaxStepRows;
    const std::size_t chunk =
        chunk_arg > 0 ? static_cast<std::size_t>(chunk_arg)
                      : std::min<std::size_t>(static_cast<std::size_t>(prompt_tokens), max_rows);
    if (chunk == 0 || chunk > max_rows) {
        std::fprintf(stderr, "pp: --prefill-chunk must be in [1, %zu]\n", max_rows);
        return kExitUsage;
    }
    const std::size_t capacity = static_cast<std::size_t>(prompt_tokens) + 4;

    auto plan_r = qwen::build_step_graph(cfg, capacity, chunk);
    if (!plan_r.ok()) {
        std::fprintf(stderr, "pp: graph: %s\n", plan_r.status().message().c_str());
        return kExitRuntime;
    }
    auto plan = plan_r.release();

    app::ShardSet shards;
    if (auto st = app::open_shards(model_dir, &shards); !st.ok()) {
        std::fprintf(stderr, "pp: %s\n", st.message().c_str());
        return kExitRuntime;
    }

    shelf::backend::hip::HipStepExecutorOptions ho;
    ho.device = static_cast<int>(gpu);
    ho.rope_theta = cfg.rope_theta;
    ho.rope_rotary_dim = cfg.rotary_dim;
    if (arena_mib > 0) ho.arena_bytes = static_cast<std::size_t>(arena_mib) << 20;
    ho.fp8_act_group = fmt.act_group;
    std::printf("%s: fp8 %s (act_group=%u)\n", "pp",
                fmt.quantized ? fmt.declared.c_str() : "なし（bf16 のまま）",
                fmt.act_group);
    // pipeline-parallel（1 = 単一 GPU）。レイヤ範囲で K ステージに分ける。
    app::PipelineOptions po;
    for (int k = 0; k < pp; ++k) po.devices.push_back(static_cast<int>(gpu) + k);
    po.kv_capacity = capacity;
    po.chunk_rows = chunk;
    po.arena_bytes = ho.arena_bytes;
    po.rope_theta = cfg.rope_theta;
    po.rope_rotary_dim = cfg.rotary_dim;
    po.fp8_act_group = fmt.act_group;
    po.verbose = true;
    auto pl = app::make_pipeline(cfg, shards, fmt, po);
    if (!pl.ok()) {
        std::fprintf(stderr, "pp: pipeline: %s\n", pl.status().message().c_str());
        return kExitRuntime;
    }
    app::Pipeline pipeline = pl.release();
    (void)plan;
    // 実際に確保されたサイズ（auto なら declare_states のグラフ算出値）。
    const std::int64_t arena_eff = static_cast<std::int64_t>(pipeline.arena_bytes >> 20);

    const auto tokens =
        make_tokens(static_cast<std::size_t>(prompt_tokens), cfg.vocab_size, seed);

    // エラー報告（アリーナ枯渇時はサイズを上げるよう促す）。
    auto fail_step = [&](const char* what, const shelf::Status& st) {
        std::fprintf(stderr, "pp: %s: %s\n", what, st.message().c_str());
        if (st.message().find("arena") != std::string::npos)
            std::fprintf(stderr,
                         "pp: hint: the arena is a per-step bump allocator; raise "
                         "--arena-mib above the current %lld MiB\n",
                         static_cast<long long>(arena_eff));
    };

    auto run_once = [&](double* wall_ms) -> shelf::Status {
        // 各 run を独立させる（行数と KV 使用行を詰め直す）。
        SHELF_RETURN_IF_ERROR(app::pipeline_reset(pipeline));
        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t c = 0; c < tokens.size(); c += chunk) {
            const std::size_t n = std::min(chunk, tokens.size() - c);
            SHELF_RETURN_IF_ERROR(app::pipeline_step(pipeline, tokens.data() + c, n,
                                                    static_cast<std::uint32_t>(c), nullptr));
        }
        *wall_ms = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - t0)
                       .count();
        return shelf::Status::make_ok();
    };

    for (std::int64_t w = 0; w < warmup; ++w) {
        double ms = 0.0;
        const auto st = run_once(&ms);
        if (!st.ok()) {
            fail_step("warmup", st);
            return kExitRuntime;
        }
    }

    std::vector<double> xs;
    xs.reserve(static_cast<std::size_t>(runs));
    for (std::int64_t r = 0; r < runs; ++r) {
        double ms = 0.0;
        const auto st = run_once(&ms);
        if (!st.ok()) {
            fail_step("run", st);
            return kExitRuntime;
        }
        xs.push_back(ms);
    }
    const Stats s = summarize(xs);
    const double us_per_token = s.p50 * 1000.0 / static_cast<double>(prompt_tokens);
    const double tokens_per_sec = us_per_token > 0.0 ? 1e6 / us_per_token : 0.0;

    std::printf("shelf-bench pp\n");
    kv_print("schema_version", static_cast<std::int64_t>(1));
    kv_print("device", std::string("hip"));
    kv_print("gpu", gpu);
    kv_print("model", model_dir);
    kv_print("prompt_tokens", prompt_tokens);
    kv_print("prefill_chunk", static_cast<std::int64_t>(chunk));
    kv_print("runs", runs);
    kv_print("warmup", warmup);
    kv_print("arena_mib", arena_eff);
    kv_print("wall_ms_median", s.p50);
    kv_print("wall_ms_min", s.min);
    kv_print("wall_ms_p10", s.p10);
    kv_print("wall_ms_p90", s.p90);
    kv_print("wall_ms_stddev", s.stddev);
    kv_print("wall_us_per_token", us_per_token);
    kv_print("wall_tokens_per_sec", tokens_per_sec, 2);
    for (std::size_t i = 0; i < xs.size(); ++i)
        std::printf("run_wall_ms[%zu]=%.3f\n", i, xs[i]);

    if (!output.empty()) {
        const char* kHeader =
            "schema_version,device,gpu,model,prompt_tokens,prefill_chunk,runs,warmup,"
            "arena_mib,wall_ms_median,wall_ms_min,wall_ms_p10,wall_ms_p90,wall_ms_stddev,"
            "wall_us_per_token,wall_tokens_per_sec";
        char row[512];
        std::snprintf(row, sizeof row,
                      "1,hip,%lld,%s,%lld,%zu,%lld,%lld,%lld,%.3f,%.3f,%.3f,"
                      "%.3f,%.3f,%.3f,%.2f",
                      static_cast<long long>(gpu), model_dir.c_str(),
                      static_cast<long long>(prompt_tokens), chunk,
                      static_cast<long long>(runs), static_cast<long long>(warmup),
                      static_cast<long long>(arena_eff),
                      s.p50, s.min, s.p10, s.p90, s.stddev, us_per_token, tokens_per_sec);
        std::string csv_err;
        if (!append_csv(output, kHeader, {row}, &csv_err)) {
            std::fprintf(stderr, "pp: %s\n", csv_err.c_str());
            return kExitRuntime;
        }
    }
    return kExitOk;
#endif  // SHELF_HAVE_HIP
}

}  // namespace shelf::bench
