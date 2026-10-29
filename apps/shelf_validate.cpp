// shelf-validate — 実モデルで無損失経路 (exact) と量子化経路 (approx) を
// 同じ入力で走らせ、生成トークン列を比較する検証ドライバ。
//
// 仕様: docs/spec/validate.md（計画・未実装は draft_docs/shelf_validate.md）
//   本実装は V0（アプリ骨格）: CLI・予算検査・fixture 入力・64 行チャンク
//   prefill・greedy decode・StopToken 解決・T0 決定性比較と JSON 出力。
//   approx 経路（T1/T2 比較）は V1、pipeline-parallel は V2 で追加する。
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include <shelf/artifact/safetensors/reader.hpp>
#include <shelf/backend/cpu/reference_executor.hpp>
#include <shelf/foundation/numeric.hpp>
#include <shelf/model/qwen35/step_graph.hpp>

#ifdef SHELF_HAVE_HIP
#include <shelf/backend/hip/step_executor.hpp>
#endif

#include "common/app_support.hpp"
#include "common/boot.hpp"
#include "common/pipeline.hpp"
#include "fixtures/prefill_fixture.hpp"

namespace {

using namespace shelf;
namespace app = shelf::app;
using json = nlohmann::json;
namespace fs = std::filesystem;

// 仕様 §2: prefill / decode はそれぞれ 128 tok 以上で成立させる。
constexpr std::size_t kMinRunTokens = 128;
// 仕様 §1: 入力長の上限（budget 512 のとき decode 128 tok を満たす最大 N）。
constexpr std::size_t kMaxPrefillTokens = 384;
// prefill チャンクの行数（HIP の 1 ステップ上限と一致させる）。
constexpr std::size_t kPrefillChunkRows = 64;

#ifdef SHELF_HAVE_HIP
static_assert(kPrefillChunkRows == backend::hip::HipStepExecutor::kMaxStepRows,
              "prefill チャンク行数は HIP の 1 ステップ上限と一致させること");
#endif

int fail(const std::string& msg) {
    std::fprintf(stderr, "shelf-validate: %s\n", msg.c_str());
    return 2;
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

std::string read_file_str(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::size_t numel(const std::vector<std::size_t>& s) {
    std::size_t n = 1;
    for (auto d : s) n *= d;
    return n;
}

std::string usage_text() {
    return
        "使い方:\n"
        "  shelf-validate --exact-model=<dir> [--approx-model=<dir>] [オプション]\n"
        "\n"
        "オプション:\n"
        "  --exact-model=<dir>       無損失 (bf16) 側のモデルディレクトリ（必須）\n"
        "  --approx-model=<dir>      量子化 (FP8 W8A8) 側のモデルディレクトリ\n"
        "  --prefill-tokens=<N>      prefill 長（既定 256、範囲 128..384）\n"
        "  --budget=<N>              prefill + decode の合計上限（既定 512）\n"
        "  --stop-id=<id>            StopToken（既定: モデル設定の eos_token_id）\n"
        "  --pipeline-parallel=<K>   レイヤ分割のステージ数（既定 1）。GPU は --gpu から K 台使う\n"
        "  --device=<hip|cpu>        実行デバイス（既定: HIP ビルドなら hip、それ以外 cpu）\n"
        "  --gpu=<n>                 GPU 番号（既定 0）\n"
        "  --runs=<n>                決定性チェックの反復数（既定 2）\n"
        "  --ids-file=<path>         入力 token ids（既定: 同梱 fixture。区切りは空白・カンマ）\n"
        "  --out=<dir>               結果 JSON の保存先（既定 shelf-validate-out）\n"
        "  --t2-margin=<x>           T2 判定の margin 上限（既定: 未設定 = 基準値待ちで保留）\n"
        "  --pp-baseline=<on|off>    PP>1 のとき PP=1 でも走らせて一致を確認（既定 off）\n"
        "  --help                    このヘルプを表示\n"
        "\n"
        "終了コード: 0 = 合格 / 1 = 条件付き合格（T2、V1 以降）/ 2 = 不合格または起動エラー\n";
}

struct Options {
    std::string exact_model;
    std::string approx_model;
    std::string ids_file;
    std::string out_dir = "shelf-validate-out";
    std::string device;
    std::size_t prefill_tokens = 256;
    std::size_t budget = 512;
    std::size_t pipeline_parallel = 1;
    std::string pp_baseline = "off";     // on = PP=1 でも走らせて一致を確認（V2 の門）
    std::size_t runs = 2;
    int gpu = 0;
    double t2_margin = -1.0;     // < 0 = 未設定（T2 は pending）
    bool stop_id_set = false;
    std::int64_t stop_id = -1;
};

// 仕様 §4 の StopToken 解決順:
//   1. --stop-id
//   2. generation_config.json の eos_token_id（配列なら全要素が停止条件、先頭を primary に採用）
//   3. config.json の eos_token_id（shelf の TextConfig が読む値）
//   4. いずれも無ければエラー
struct StopSet {
    std::vector<std::int64_t> ids;
    std::string source;

    std::int64_t primary() const { return ids.empty() ? -1 : ids.front(); }
    bool contains(std::int64_t t) const {
        return std::find(ids.begin(), ids.end(), t) != ids.end();
    }
};

Result<StopSet> resolve_stop(const Options& o, const std::string& model_dir,
                             const model::qwen35::TextConfig& cfg) {
    if (o.stop_id_set) return StopSet{{o.stop_id}, "cli"};

    const std::string gen_path = model_dir + "/generation_config.json";
    if (fs::exists(gen_path)) {
        StopSet s;
        s.source = "generation_config.json";
        try {
            const auto g = json::parse(read_file_str(gen_path));
            if (g.contains("eos_token_id") && !g["eos_token_id"].is_null()) {
                const auto& v = g["eos_token_id"];
                if (v.is_array()) {
                    for (const auto& e : v) s.ids.push_back(e.get<std::int64_t>());
                } else {
                    s.ids.push_back(v.get<std::int64_t>());
                }
                if (!s.ids.empty()) return s;
            }
        } catch (const std::exception& e) {
            return Status::invalid_argument(std::string("generation_config.json の解析失敗: ") +
                                            e.what(), __FILE__, __LINE__);
        }
    }
    if (cfg.eos_token_id >= 0) return StopSet{{cfg.eos_token_id}, "config.json"};
    return Status::not_found("StopToken が見つからない（--stop-id で指定してください）",
                             __FILE__, __LINE__);
}

// decode 1 ステップ分の top-1 / top-2（T2 の margin 判定に使う）。
struct StepTop {
    std::int64_t top1 = 0;
    float top1_val = 0.0f;
    float top2_val = 0.0f;
};

StepTop pick_top2(const std::vector<float>& lg) {
    StepTop t;
    if (lg.empty()) return t;
    std::size_t i1 = 0, i2 = 0;
    for (std::size_t i = 1; i < lg.size(); ++i) {
        if (lg[i] > lg[i1]) {
            i2 = i1;
            i1 = i;
        } else if (i != i1 && lg[i] > lg[i2]) {
            i2 = i;
        }
    }
    if (i2 == i1 && lg.size() > 1) i2 = (i1 == 0) ? 1 : 0;
    t.top1 = static_cast<std::int64_t>(i1);
    t.top1_val = lg[i1];
    t.top2_val = lg[i2];
    return t;
}

// 1 実行分の結果（§5 の比較単位）。
struct RunResult {
    std::string config_key;             // V0 では常に "exact"
    std::size_t index = 0;
    std::vector<std::int64_t> tokens;   // decode 生成分のみ（StopToken は含めない）
    std::vector<StepTop> tops;          // decode 各ステップの top-1/2
    std::string stop_reason;            // stop_token | budget_exhausted
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
};

// 仕様 §5: 判定は identical / diverged（不一致位置 k と margin）/ incomparable。
// T0（同一構成の反復）と T1（exact vs approx）を同一形式で比較できるようにする。
json compare_runs(const RunResult& a, const RunResult& b, const char* axis) {
    json j;
    j["axis"] = axis;
    j["a_config"] = a.config_key;
    j["a_index"] = a.index;
    j["b_config"] = b.config_key;
    j["b_index"] = b.index;

    if (a.tokens == b.tokens && a.stop_reason == b.stop_reason) {
        j["verdict"] = "identical";
        j["token_count"] = a.tokens.size();
        return j;
    }
    if (a.tokens == b.tokens) {
        // トークン列は同じだが停止理由が異なる（予算打ち切りの非対称など）。
        j["verdict"] = "incomparable";
        j["reason"] = "stop_reason mismatch";
        j["a_stop_reason"] = a.stop_reason;
        j["b_stop_reason"] = b.stop_reason;
        return j;
    }

    std::size_t k = 0;
    while (k < a.tokens.size() && k < b.tokens.size() && a.tokens[k] == b.tokens[k]) ++k;
    j["verdict"] = "diverged";
    j["diverged_at"] = k;
    j["a_stop_reason"] = a.stop_reason;
    j["b_stop_reason"] = b.stop_reason;
    j["a_token_count"] = a.tokens.size();
    j["b_token_count"] = b.tokens.size();
    if (k < a.tokens.size()) j["a_token_at_k"] = a.tokens[k];
    if (k < b.tokens.size()) j["b_token_at_k"] = b.tokens[k];

    // 正とする側 (a) の top-1 / top-2 と margin、および top-1 値の差。
    // vocab 全体の logits 差は全ステップ保持が予算超過のため V1 で追加する。
    if (k < a.tops.size()) {
        const auto& t = a.tops[k];
        j["a_top1"] = {{"id", t.top1}, {"value", t.top1_val}};
        j["a_top2_value"] = t.top2_val;
        j["a_margin"] = static_cast<double>(t.top1_val) - static_cast<double>(t.top2_val);
    }
    if (k < b.tops.size()) {
        const auto& t = b.tops[k];
        j["b_top1"] = {{"id", t.top1}, {"value", t.top1_val}};
        j["b_top2_value"] = t.top2_val;
    }
    if (k < a.tops.size() && k < b.tops.size())
        j["top1_value_gap"] =
            std::fabs(static_cast<double>(a.tops[k].top1_val) -
                      static_cast<double>(b.tops[k].top1_val));
    j["note"] = "logits 差は top-1 値のみ（完全な vocab 差分は V1 以降）";
    return j;
}

} // namespace

int main(int argc, char** argv) {
    Options o;
#ifdef SHELF_HAVE_HIP
    o.device = "hip";
#else
    o.device = "cpu";
#endif

    // ---- 引数（仕様 §1） ----
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            std::fputs(usage_text().c_str(), stdout);
            return 0;
        }
        if (a.rfind("--", 0) != 0)
            return fail("引数は --key=value 形式です: " + a + "\n" + usage_text());

        std::string key, val;
        if (const auto eq = a.find('='); eq != std::string::npos) {
            key = a.substr(0, eq);
            val = a.substr(eq + 1);
        } else {
            if (i + 1 >= argc) return fail("値がありません: " + a);
            key = a;
            val = argv[++i];
        }
        auto num = [&]() -> long long {
            try {
                std::size_t pos = 0;
                const long long v = std::stoll(val, &pos);
                return pos == val.size() ? v : -1;
            } catch (...) {
                return -1;
            }
        };
        if (key == "--pipeline-parallel") {
            try { o.pipeline_parallel = std::stoul(val); }
            catch (...) { return fail("--pipeline-parallel は整数です: " + val); }
            if (o.pipeline_parallel == 0) return fail("--pipeline-parallel は 1 以上");
            continue;
        }
        if (key == "--pp-baseline") {
            if (val != "on" && val != "off") return fail("--pp-baseline は on / off");
            o.pp_baseline = val;
            continue;
        }
        if (key == "--t2-margin") {
            try { o.t2_margin = std::stod(val); }
            catch (...) { return fail("--t2-margin は数値です: " + val); }
            continue;
        }
        if (key == "--exact-model") {
            o.exact_model = val;
        } else if (key == "--approx-model") {
            o.approx_model = val;
        } else if (key == "--ids-file") {
            o.ids_file = val;
        } else if (key == "--out") {
            o.out_dir = val;
        } else if (key == "--device") {
            o.device = val;
        } else if (key == "--stop-id") {
            const long long v = num();
            if (v < 0) return fail("--stop-id が不正です: " + val);
            o.stop_id = v;
            o.stop_id_set = true;
        } else if (key == "--gpu") {
            const long long v = num();
            if (v < 0) return fail("--gpu が不正です: " + val);
            o.gpu = static_cast<int>(v);
        } else if (key == "--prefill-tokens") {
            const long long v = num();
            if (v < 0) return fail("--prefill-tokens が不正です: " + val);
            o.prefill_tokens = static_cast<std::size_t>(v);
        } else if (key == "--budget") {
            const long long v = num();
            if (v < 0) return fail("--budget が不正です: " + val);
            o.budget = static_cast<std::size_t>(v);
        } else if (key == "--pipeline-parallel") {
            const long long v = num();
            if (v < 0) return fail("--pipeline-parallel が不正です: " + val);
            o.pipeline_parallel = static_cast<std::size_t>(v);
        } else if (key == "--runs") {
            const long long v = num();
            if (v < 0) return fail("--runs が不正です: " + val);
            o.runs = static_cast<std::size_t>(v);
        } else {
            return fail("未知の引数です: " + key + "\n" + usage_text());
        }
    }

    // ---- 仕様検査（§2）。起動前に拒否する ----
    if (o.exact_model.empty()) return fail("--exact-model は必須です\n" + usage_text());
    if (o.pipeline_parallel > 1 && o.device != "hip")
        return fail("--pipeline-parallel > 1 は hip のみ対応です");
    if (o.prefill_tokens < kMinRunTokens || o.prefill_tokens > kMaxPrefillTokens)
        return fail("--prefill-tokens は " + std::to_string(kMinRunTokens) + ".." +
                    std::to_string(kMaxPrefillTokens) + " の範囲で指定してください（" +
                    std::to_string(o.prefill_tokens) + " を受付拒否）");
    if (o.budget < o.prefill_tokens + kMinRunTokens)
        return fail("budget - prefill-tokens が " + std::to_string(kMinRunTokens) +
                    " 未満です（decode が下限に達しない）。budget=" + std::to_string(o.budget) +
                    ", prefill=" + std::to_string(o.prefill_tokens) +
                    " を受付拒否");
    if (o.runs == 0) return fail("--runs は 1 以上です");
    if (o.device != "hip" && o.device != "cpu")
        return fail("--device は hip か cpu です: " + o.device);
    const bool use_hip = (o.device == "hip");

    // ---- 起動（D1）: エンジンを起こす前に装置と要求を突き合わせる ----
    {
        app::BootRequest breq;
        breq.device_execution = use_hip;
        breq.first_device = static_cast<std::size_t>(o.gpu);
        breq.pipeline_parallel = o.pipeline_parallel;
        const auto booted = app::boot(breq, /*print=*/true);
        if (!booted.accepted) return fail("起動条件を満たしません: " + booted.reason);
    }

    const std::size_t decode_cap = o.budget - o.prefill_tokens;
    const std::size_t capacity = o.prefill_tokens + decode_cap + 4;
    const double t_start = now_ms();

    // ---- 設定読み（§4 手順 1） ----
    auto cfg_r = model::qwen35::load_text_config_json(read_file_str(o.exact_model + "/config.json"));
    if (!cfg_r.ok()) return fail("config.json: " + cfg_r.status().message());
    const auto& cfg = cfg_r.value();

    auto stop_r = resolve_stop(o, o.exact_model, cfg);
    if (!stop_r.ok()) return fail(stop_r.status().message());
    const StopSet stop = stop_r.value();

    // ---- 入力（§3）: 同じ ids を両経路にそのまま適用する ----
    std::vector<std::int64_t> input_ids;
    std::string input_source = "fixture:apps/fixtures/prefill_fixture.hpp";
    if (!o.ids_file.empty()) {
        input_source = "ids-file:" + o.ids_file;
        std::string raw = read_file_str(o.ids_file);
        if (raw.empty()) return fail("ids-file が読めません: " + o.ids_file);
        for (char& c : raw)
            if (c == ',' || c == '\n' || c == '\r' || c == '\t') c = ' ';
        std::stringstream ss(raw);
        long long v = 0;
        while (ss >> v) input_ids.push_back(v);
        if (!ss.eof()) return fail("ids-file に数値以外の要素があります: " + o.ids_file);
        if (input_ids.empty()) return fail("ids-file が空です: " + o.ids_file);
    } else {
        input_ids.assign(shelf::apps::kPrefillFixture,
                         shelf::apps::kPrefillFixture + shelf::apps::kPrefillFixtureSize);
    }
    bool truncated = false;
    if (input_ids.size() < o.prefill_tokens)
        return fail("入力長が --prefill-tokens より短いです（入力 " +
                    std::to_string(input_ids.size()) + " tok < prefill " +
                    std::to_string(o.prefill_tokens) +
                    " tok）。--ids-file を指定するか fixture を再生成してください");
    if (input_ids.size() > o.prefill_tokens) {
        input_ids.resize(o.prefill_tokens);   // 末尾切り詰め（両経路に同じ ids を適用）
        truncated = true;
    }
    for (const auto id : input_ids)
        if (id < 0 || static_cast<std::size_t>(id) >= cfg.vocab_size)
            return fail("入力トークンが語彙外です: " + std::to_string(id));

    // ---- 1 モデル分のロードと実行（exact / approx で呼ぶ） ----
    // §4 の手順を model_dir に対してそのまま行う。exact と approx を**同時に VRAM
    // へ載せられない**（9B で bf16 17.9 GB + FP8 12.6 GB ≳ GPU の搭載 VRAM）ため、
    // exact を走らせてから実行器とシャードを解放し、approx を読む。
    // 比較は host 側に保持したトークン列と top-1/2（§5 の T1/T2 の入力）で行う。
    struct ModelRun {
        std::vector<RunResult> runs;
        double load_ms = 0.0;
        bool device_used_known = false;
        std::size_t device_used_bytes = 0;
        std::size_t arena_bytes = 0;
        std::size_t stages = 1;                // pipeline-parallel のステージ数
        std::string fp8_declared;              // 空 = 量子化されていない
        std::uint32_t fp8_act_group = 0u;
    };

    auto run_model = [&](const std::string& model_dir, const char* key,
                         const model::qwen35::TextConfig& cfg, const StopSet& stop,
                         ModelRun& mr) -> int {
    // ---- グラフ（§4 手順 3: 行数上限 64 のため prefill はチャンク） ----
    // HIP は pipeline（PP=1 なら 1 ステージ）がグラフと実行器を持つ。
    // CPU 参照経路は全層 1 グラフ（PP 非対応）。
    model::qwen35::StepGraphPlan plan_holder;
    if (!use_hip) {
        auto plan_r = model::qwen35::build_step_graph(cfg, capacity);
        if (!plan_r.ok()) return fail(plan_r.status().message());
        plan_holder = plan_r.release();
    }
#ifdef SHELF_HAVE_HIP
    app::Pipeline pipeline;
#endif

    // ---- シャード（apps 共通の読み方を使う: 正本は app_support） ----
    app::ShardSet shards;
    if (auto st = app::open_shards(model_dir, &shards); !st.ok())
        return fail(st.message());
    // 重みの実体化計画は正準カタログを入口にする（D3: 名前・形状・dtype の検証）。
    auto cat_r = app::build_catalog(shards);
    if (!cat_r.ok()) return fail("catalog: " + cat_r.status().message());

    // FP8 の形は checkpoint の宣言が正（approx 側が量子化されている場合のみ効く）。
    auto fmt_r = app::read_fp8_format(app::read_file_str(model_dir + "/config.json"));
    if (!fmt_r.ok()) return fail("quantization_config: " + fmt_r.status().message());
    const auto fmt = fmt_r.value();
    if (fmt.quantized) {
        mr.fp8_declared = fmt.declared;
        mr.fp8_act_group = fmt.act_group;
    }

    // ---- ロード（§4 手順 2） ----
    const double t_load = now_ms();
    backend::cpu::HostStorage storage;

    if (use_hip) {
#ifdef SHELF_HAVE_HIP
        std::size_t free_before = 0, total_before = 0, free_after = 0, total_after = 0;
        const auto meminfo = [](std::size_t& f, std::size_t& t) -> Status {
            const hipError_t e = hipMemGetInfo(&f, &t);
            if (e != hipSuccess)
                return Status::from(Status::Code::device_error,
                                    std::string("hipMemGetInfo: ") + hipGetErrorString(e),
                                    __FILE__, __LINE__);
            return Status::make_ok();
        };
        {   // 先に対象 GPU に切り替える（既定の GPU で測ると前後で対象が異なる）。
            // 1 回目でコンテキスト初期化し、2 回目を基準にする。
            const hipError_t se = hipSetDevice(o.gpu);
            if (se != hipSuccess)
                return fail(std::string("hipSetDevice: ") + hipGetErrorString(se));
            auto st = meminfo(free_before, total_before);
            if (!st.ok()) return fail(st.message());
            st = meminfo(free_before, total_before);
            if (!st.ok()) return fail(st.message());
        }
        // pipeline: レイヤ範囲でステージ分割（PP=1 は 1 ステージ = 従来と同じ）。
        // arena は各ステージで自動算出（prefill グラフの行数に合わせる）。
        app::PipelineOptions po;
        for (std::size_t k = 0; k < o.pipeline_parallel; ++k)
            po.devices.push_back(o.gpu + static_cast<int>(k));
        po.kv_capacity = capacity;
        po.chunk_rows = kPrefillChunkRows;
        po.rope_theta = cfg.rope_theta;
        po.rope_rotary_dim = cfg.rotary_dim;
        po.fp8_act_group = fmt.act_group;
        po.verbose = true;
        auto pl = app::make_pipeline(cfg, shards, cat_r.value(), fmt, po);
        if (!pl.ok()) return fail("pipeline: " + pl.status().message());
        pipeline = pl.release();
        mr.stages = pipeline.stages.size();
        mr.arena_bytes = pipeline.arena_bytes;
        {
            auto st = meminfo(free_after, total_after);
            if (!st.ok()) return fail(st.message());
        }
        mr.device_used_bytes = free_before - free_after;
        mr.device_used_known = true;
#else
        return fail("このビルドは HIP 非対応です");
#endif
    } else {
        // CPU 参照経路も同じ入口（正準カタログ → runtime/load）を通す。
        auto st = app::load_host_plan(storage, plan_holder, cat_r.value(), shards, fmt);
        if (!st.ok()) return fail(st.message());
    }
    mr.load_ms = now_ms() - t_load;

    // ---- RoPE 周波数表（CPU 経路のみ使用。f32 の inv_freq を double で再現） ----
    const std::size_t rd = cfg.rotary_dim;
    std::vector<double> inv(rd / 2);
    for (std::size_t j = 0; j < rd / 2; ++j)
        inv[j] = std::pow(cfg.rope_theta, -2.0 * static_cast<double>(j) / static_cast<double>(rd));

    // ---- 1 行ステップ（位置 pos の token を 1 つ投入して logits を得る） ----
    auto step1 = [&](std::int32_t token, std::uint32_t pos, std::vector<float>* logits) -> Status {
#ifdef SHELF_HAVE_HIP
        if (!pipeline.stages.empty()) {
            const std::int32_t t = token;
            return app::pipeline_step(pipeline, &t, 1, pos, logits);
        }
#endif
        storage.reset_values();
        SHELF_RETURN_IF_ERROR(storage.set_value_int(plan_holder.tokens, {token}));
        std::vector<float> cosv(rd), sinv(rd);
        for (std::size_t j = 0; j < rd / 2; ++j) {
            const double f = static_cast<double>(pos) * inv[j];
            cosv[j] = cosv[j + rd / 2] = static_cast<float>(std::cos(f));
            sinv[j] = sinv[j + rd / 2] = static_cast<float>(std::sin(f));
        }
        SHELF_RETURN_IF_ERROR(storage.set_value(plan_holder.cos, std::move(cosv)));
        SHELF_RETURN_IF_ERROR(storage.set_value(plan_holder.sin, std::move(sinv)));
        SHELF_RETURN_IF_ERROR(backend::cpu::ReferenceExecutor::run(plan_holder.graph, storage));
        if (logits) {
            const auto* lv = storage.value(plan_holder.logits);
            if (!lv) return Status::invalid_state("run 後に logits が無い", __FILE__, __LINE__);
            *logits = *lv;
        }
        return Status::make_ok();
    };

    // ---- prefill（§4 手順 3）: HIP は 64 行チャンク、CPU は 1 行ずつ。
    //      最終ステップ（位置 N-1）の logits を返す ----
    auto prefill = [&](const std::vector<std::int64_t>& ids, std::vector<float>* last_logits)
        -> Status {
#ifdef SHELF_HAVE_HIP
        if (!pipeline.stages.empty()) {
            const std::size_t full_chunks = ids.size() / kPrefillChunkRows;
            std::vector<std::int32_t> win;
            for (std::size_t c = 0; c < full_chunks; ++c) {
                win.clear();
                for (std::size_t i = 0; i < kPrefillChunkRows; ++i)
                    win.push_back(static_cast<std::int32_t>(ids[c * kPrefillChunkRows + i]));
                const std::size_t base = c * kPrefillChunkRows;
                const bool tail_pending = (ids.size() != full_chunks * kPrefillChunkRows);
                const bool last_chunk = (c + 1 == full_chunks) && !tail_pending;
                std::vector<float> lg;
                auto st = app::pipeline_step(pipeline, win.data(), kPrefillChunkRows,
                                             static_cast<std::uint32_t>(base),
                                             (last_chunk && last_logits) ? &lg : nullptr);
                if (!st.ok())
                    return Status::invalid_state("prefill チャンク 位置 " +
                                                     std::to_string(base) + ": " + st.message(),
                                                 __FILE__, __LINE__);
                std::printf("  prefill チャンク 位置 %zu..%zu (%.1f ms)\n", base,
                            base + kPrefillChunkRows - 1, pipeline.last_step_ms);
                if (last_chunk && last_logits) {
                    const std::size_t rows = lg.size() / kPrefillChunkRows;
                    last_logits->assign(lg.end() - static_cast<std::ptrdiff_t>(rows), lg.end());
                }
            }
            const std::size_t tail = ids.size() - full_chunks * kPrefillChunkRows;
            for (std::size_t i = 0; i < tail; ++i) {
                const std::size_t p = full_chunks * kPrefillChunkRows + i;
                const bool is_last = (p + 1 == ids.size());
                SHELF_RETURN_IF_ERROR(step1(static_cast<std::int32_t>(ids[p]),
                                            static_cast<std::uint32_t>(p),
                                            is_last ? last_logits : nullptr));
            }
            return Status::make_ok();
        }
#endif
        for (std::size_t p = 0; p < ids.size(); ++p)
            SHELF_RETURN_IF_ERROR(step1(static_cast<std::int32_t>(ids[p]),
                                        static_cast<std::uint32_t>(p),
                                        p + 1 == ids.size() ? last_logits : nullptr));
        return Status::make_ok();
    };

    // ---- decode（§4 手順 4-5）: greedy top-1。StopToken は列に含めない ----
    auto decode = [&](std::size_t decode_limit, const StopSet& st, std::vector<float>& logits,
                      RunResult& rr) -> Status {
        rr.stop_reason.clear();
        std::int64_t prev = -1;
        for (std::size_t s = 0; s < decode_limit; ++s) {
            if (s > 0) {
                // トークン t_{s-1} は位置 (prefill 終端 + s - 1) に置かれる。
                SHELF_RETURN_IF_ERROR(step1(static_cast<std::int32_t>(prev),
                                            static_cast<std::uint32_t>(o.prefill_tokens + s - 1),
                                            &logits));
            }
            if (logits.empty())
                return Status::invalid_state("decode: logits が空", __FILE__, __LINE__);
            const StepTop top = pick_top2(logits);
            rr.tops.push_back(top);
            if (st.contains(top.top1)) {
                rr.stop_reason = "stop_token";
                return Status::make_ok();
            }
            rr.tokens.push_back(top.top1);
            prev = top.top1;
            if (rr.tokens.size() == decode_limit) {
                rr.stop_reason = "budget_exhausted";
                return Status::make_ok();
            }
        }
        rr.stop_reason = "budget_exhausted";
        return Status::make_ok();
    };

    // ---- 実行（§4）。毎回ステートをリセットして反復間の決定性を確かめる ----
    mr.runs.reserve(o.runs);
    for (std::size_t r = 0; r < o.runs; ++r) {
#ifdef SHELF_HAVE_HIP
        if (!pipeline.stages.empty()) {
            auto st = app::pipeline_reset(pipeline);
            if (!st.ok()) return fail("reset_states: " + st.message());
        } else
#endif
        {
            storage.reset_states();
        }

        RunResult rr;
        rr.config_key = key;
        rr.index = r;

        std::vector<float> logits;
        const double p0 = now_ms();
        auto st = prefill(input_ids, &logits);
        if (!st.ok()) return fail("prefill: " + st.message());
        rr.prefill_ms = now_ms() - p0;

        const double d0 = now_ms();
        st = decode(decode_cap, stop, logits, rr);
        if (!st.ok()) return fail("decode: " + st.message());
        rr.decode_ms = now_ms() - d0;

        std::printf("run %zu/%zu: %zu tok 生成 (%s), prefill %.1f ms / decode %.1f ms\n",
                    r + 1, o.runs, rr.tokens.size(), rr.stop_reason.c_str(), rr.prefill_ms,
                    rr.decode_ms);
        mr.runs.push_back(std::move(rr));
    }
    return 0;
    };  // run_model

    ModelRun exact_run;
    if (int rc = run_model(o.exact_model, "exact", cfg, stop, exact_run)) return rc;
    ModelRun approx_run;
    if (!o.approx_model.empty()) {
        auto acfg_r = model::qwen35::load_text_config_json(
            app::read_file_str(o.approx_model + "/config.json"));
        if (!acfg_r.ok()) return fail("approx の config.json: " + acfg_r.status().message());
        const auto& acfg = acfg_r.value();
        if (acfg.vocab_size != cfg.vocab_size || acfg.num_layers != cfg.num_layers ||
            acfg.hidden_size != cfg.hidden_size || acfg.layer_kinds != cfg.layer_kinds)
            return fail("exact と approx が同じ構成ではありません（T1 は同じモデルの比較）");
        auto astop_r = resolve_stop(o, o.approx_model, acfg);
        if (!astop_r.ok())
            return fail("approx の停止トークン: " + astop_r.status().message());
        if (astop_r.value().ids != stop.ids)
            return fail("exact と approx で停止トークン集合が違います（比較不可）");
        if (int rc = run_model(o.approx_model, "approx", acfg, stop, approx_run)) return rc;
    }

    // ---- 比較（§5） ----
    // T0: 同一 config 内の反復（決定性）。exact / approx の各自で行う。
    // T1: exact run0 vs approx run0（トークン列の一致）。
    // T2: T1 が diverged のとき、不一致位置での exact 側 margin（top1-top2）を
    //     基準値（--t2-margin、parity_policy §3 の margin 分布 p99）と比較する。
    //     margin が基準以下 = 正本側でもほぼ同点だった位置 → 条件付き合格 (exit 1)。
    //     margin が基準超 = 正本が明確に決めていた位置で割れた → 不合格 (exit 2)。
    //     基準未設定なら pending（exit 1）として、測った値を JSON に残す。
    std::vector<json> comparisons;
    const auto add_t0 = [&](const ModelRun& mr) {
        for (std::size_t i2 = 1; i2 < mr.runs.size(); ++i2)
            comparisons.push_back(compare_runs(mr.runs[0], mr.runs[i2], "T0"));
    };
    add_t0(exact_run);
    add_t0(approx_run);

    int exit_code = 0;
    std::string t0_verdict = comparisons.empty() ? "skipped" : "identical";
    for (const auto& c : comparisons) {
        const std::string v = c.at("verdict").get<std::string>();
        std::printf("T0 (%s run%zu vs run%zu): %s\n",
                    c.at("a_config").get<std::string>().c_str(),
                    c.at("a_index").get<std::size_t>(), c.at("b_index").get<std::size_t>(),
                    v.c_str());
        if (v != "identical") {
            t0_verdict = v;
            exit_code = 2;
            if (v == "diverged") {
                const std::size_t k = c.at("diverged_at").get<std::size_t>();
                std::string margin = "n/a";
                if (c.contains("a_margin")) margin = std::to_string(c.at("a_margin").get<double>());
                std::printf("  不一致位置 k=%zu, a_margin=%s\n", k, margin.c_str());
            }
        }
    }

    // ---- V2 の門: PP=K が PP=1 と同じ数値になること（Strata の --split-device と同型）----
    ModelRun pp_baseline_run;
    std::string t1pp_verdict = "skipped";
    if (o.pp_baseline == "on" && o.pipeline_parallel > 1) {
        if (!o.approx_model.empty())
            return fail("--pp-baseline=on は approx 併用時は未対応（exact 単独で使う）");
        const std::size_t saved = o.pipeline_parallel;
        o.pipeline_parallel = 1;
        std::printf("--- PP=1 基線（一致確認のため再実行）---\n");
        if (int rc = run_model(o.exact_model, "exact_pp1", cfg, stop, pp_baseline_run)) return rc;
        o.pipeline_parallel = saved;
        const json c = compare_runs(exact_run.runs[0], pp_baseline_run.runs[0], "T1pp");
        t1pp_verdict = c.at("verdict").get<std::string>();
        comparisons.push_back(c);
        std::printf("T1pp (PP=%zu vs PP=1): %s\n", saved, t1pp_verdict.c_str());
        if (t1pp_verdict != "identical") exit_code = 2;
    }

    std::string t1_verdict = "skipped", t2_verdict = "skipped";
    json t2 = nullptr;
    if (!approx_run.runs.empty()) {
        const json t1 = compare_runs(exact_run.runs[0], approx_run.runs[0], "T1");
        t1_verdict = t1.at("verdict").get<std::string>();
        comparisons.push_back(t1);
        std::printf("T1 (exact vs approx): %s\n", t1_verdict.c_str());
        if (t1_verdict == "identical") {
            std::printf("  トークン列は全一致（%zu tok）\n",
                        t1.at("token_count").get<std::size_t>());
        } else if (t1_verdict == "diverged") {
            const std::size_t k = t1.at("diverged_at").get<std::size_t>();
            const double m = t1.contains("a_margin") ? t1.at("a_margin").get<double>() : -1.0;
            std::printf("  不一致位置 k=%zu  exact=%lld approx=%lld  exact margin=%.4f\n", k,
                        static_cast<long long>(t1.contains("a_token_at_k")
                                                   ? t1.at("a_token_at_k").get<std::int64_t>()
                                                   : -1),
                        static_cast<long long>(t1.contains("b_token_at_k")
                                                   ? t1.at("b_token_at_k").get<std::int64_t>()
                                                   : -1),
                        m);
            if (o.t2_margin < 0.0) {
                t2_verdict = "pending_baseline";
                t2 = {{"status", t2_verdict},
                      {"exact_margin", m},
                      {"note", "--t2-margin 未設定。まず margin を測って parity_policy §3 の"
                               "基準値（分布の p99）を決める"}};
                exit_code = std::max(exit_code, 1);
            } else if (m >= 0.0 && m <= o.t2_margin) {
                t2_verdict = "within_margin";
                t2 = {{"status", t2_verdict},
                      {"exact_margin", m},
                      {"threshold", o.t2_margin}};
                exit_code = std::max(exit_code, 1);
            } else {
                t2_verdict = "outside_margin";
                t2 = {{"status", t2_verdict},
                      {"exact_margin", m},
                      {"threshold", o.t2_margin}};
                exit_code = 2;
            }
            std::printf("T2: %s (exact margin=%.4f, 基準=%s)\n", t2_verdict.c_str(), m,
                        o.t2_margin < 0.0 ? "未設定" : std::to_string(o.t2_margin).c_str());
        } else {
            // incomparable: 片方が StopToken に達するなど、比較の前提が崩れている。
            exit_code = 2;
        }
    }

    // ---- 結果 JSON（§5） ----
    const auto model_run_json = [](const ModelRun& mr) {
        return json{
            {"load_ms", mr.load_ms},
            {"device_mem_used_bytes", mr.device_used_known ? json(mr.device_used_bytes)
                                                          : json(nullptr)},
            {"arena_bytes", mr.arena_bytes},
            {"stages", mr.stages},
            {"fp8", mr.fp8_declared.empty() ? json(nullptr) : json(mr.fp8_declared)},
            {"fp8_act_group", mr.fp8_declared.empty() ? json(nullptr)
                                                      : json(mr.fp8_act_group)},
        };
    };

    json result;
    result["schema"] = "shelf-validate/v1";
    result["exit_code"] = exit_code;
    result["config"] = {
        {"device", o.device},
        {"gpu", o.gpu},
        {"runs", o.runs},
        {"prefill_tokens", o.prefill_tokens},
        {"budget", o.budget},
        {"decode_cap", decode_cap},
        {"pipeline_parallel", o.pipeline_parallel},
        {"models", {{"exact", o.exact_model},
                    {"approx", o.approx_model.empty() ? json(nullptr) : json(o.approx_model)}}},
        {"t2_margin", o.t2_margin < 0.0 ? json(nullptr) : json(o.t2_margin)},
        {"input", {{"source", input_source},
                   {"length", o.prefill_tokens},
                   {"truncated", truncated}}},
        {"stop", {{"ids", stop.ids}, {"primary", stop.primary()}, {"source", stop.source}}},
        {"device_mem_used_bytes",
         exact_run.device_used_known ? json(exact_run.device_used_bytes) : json(nullptr)},
    };
    result["t0"] = t0_verdict;
    result["t1pp"] = t1pp_verdict;
    result["t1"] = t1_verdict;
    result["t2"] = t2;
    result["models_run"] = {{"exact", model_run_json(exact_run)},
                            {"exact_pp1", model_run_json(pp_baseline_run)},
                            {"approx", model_run_json(approx_run)}};
    json jr = json::array();
    const auto push_runs = [&](const ModelRun& mr) {
        for (const auto& rr : mr.runs) {
            json jttops = json::array();
            for (const auto& t : rr.tops)
                jttops.push_back({{"top1", t.top1},
                                  {"top1_value", t.top1_val},
                                  {"margin", static_cast<double>(t.top1_val) -
                                             static_cast<double>(t.top2_val)}});
            jr.push_back({{"config", rr.config_key},
                          {"index", rr.index},
                          {"token_count", rr.tokens.size()},
                          {"stop_reason", rr.stop_reason},
                          {"prefill_ms", rr.prefill_ms},
                          {"decode_ms", rr.decode_ms},
                          {"tokens", rr.tokens},
                          {"tops", jttops}});
        }
    };
    push_runs(exact_run);
    push_runs(pp_baseline_run);
    push_runs(approx_run);
    result["runs"] = jr;
    result["comparisons"] = json(comparisons);
    result["timing"] = {{"load_ms", exact_run.load_ms + approx_run.load_ms +
                                     pp_baseline_run.load_ms},
                        {"elapsed_ms", now_ms() - t_start}};

    std::error_code ec;
    fs::create_directories(o.out_dir, ec);
    if (ec) return fail("出力ディレクトリを作れません: " + o.out_dir + " (" + ec.message() + ")");
    const std::string out_path = o.out_dir + "/result.json";
    {
        std::ofstream f(out_path);
        if (!f) return fail("結果を書き込めません: " + out_path);
        f << result.dump(2) << "\n";
    }
    std::printf("判定: %s (exit %d) — 結果: %s\n",
                exit_code == 0 ? "合格" : (exit_code == 1 ? "条件付き合格" : "不合格"),
                exit_code, out_path.c_str());
    return exit_code;
}
