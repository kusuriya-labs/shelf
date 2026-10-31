// shelf-compute — エンジンの実体（JSONL ドライバ）（`engine_contract.md` §1/§2 の実体）。
//
// 役割: Server（Python 側・パス A）とエンジンの**唯一の接点**。stdin から 1 行 1
// オブジェクトの操作を読み、stdout へイベントを 1 行ずつ書く。プロトコルの型は
// `src/api/protocol.hpp` が正本で、ここは「型 → 実行」の対応だけを持つ。
//
// 設計（いまの範囲）:
//   * 1 リクエスト直列（`max_sequences = 1` に一致）。多重化は Server 側の順番待ち
//   * `generate` は prefill（チャンク）→ decode（1 トークンずつ）→ `token` イベント →
//     `done`。抽選は `runtime/generate`（`api` 経由で再輸出）に従う
//   * `cancel` は **decode の各ステップの間**で受け付ける（stdin を非ブロッキングで覗く）。
//     生成前に届いた cancel は `done(cancelled, 0 トークン)`
//   * エラーは `error` イベント（`Status::Code` → `code` の写像は `api` が持つ）。
//     1 リクエストの失敗でプロセスは落とさない
//   * `--model` 省略時はモデルを読まない（`ping` だけ応答する。`generate` は
//     `model_not_found`）。プロトコルだけの検証に使える
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <poll.h>
#include <unistd.h>

#include <shelf/api/protocol.hpp>
#include <shelf/api/sampling.hpp>
#include <shelf/backend/common/capabilities.hpp>
#include <shelf/foundation/numeric.hpp>
#include <shelf/model/qwen35/step_graph.hpp>

#include "common/app_support.hpp"
#include "common/boot.hpp"
#include "common/pipeline.hpp"

namespace {

using namespace shelf;

constexpr const char* kVersion = "shelf-compute/0.1";

// 1 行読む（stdin）。EOF は nullopt。改行は落とす。上限超過は空行扱いで読み捨て。
bool read_line(std::string* out) {
    out->clear();
    for (;;) {
        char c = 0;
        const ssize_t n = ::read(STDIN_FILENO, &c, 1);
        if (n <= 0) return false;
        if (c == '\n') return true;
        out->push_back(c);
        if (out->size() > (4u << 20)) {   // 壊れた入力で無限に伸ばさない
            char sink = 0;
            while (::read(STDIN_FILENO, &sink, 1) == 1 && sink != '\n') {
            }
            out->clear();
            return true;
        }
    }
}

void write_event(const api::Event& e) {
    const std::string line = api::encode_event(e) + "\n";
    std::size_t off = 0;
    while (off < line.size()) {
        const ssize_t n = ::write(STDOUT_FILENO, line.data() + off, line.size() - off);
        if (n <= 0) return;
        off += static_cast<std::size_t>(n);
    }
}

// 未読があるか（ブロックしない）。cancel の検出に使う。
bool stdin_pending() {
    pollfd p{STDIN_FILENO, POLLIN, 0};
    return ::poll(&p, 1, 0) > 0;
}

struct Server {
    std::string model_dir;
    int gpu = 0;
    std::size_t pp = 1;
    std::size_t prefill_rows = 64;
    std::size_t max_context = 0;
    bool has_model = false;

    model::qwen35::TextConfig cfg;
    app::Fp8ModelFormat fmt;
    app::Pipeline pipeline;
    std::vector<std::string> deferred;   // 生成中に届いた次の要求（後で処理する）
    bool stdin_eof = false;              // 入力が閉じた（生成は最後まで続ける）
    std::uint64_t request_counter = 0;
    bool cancelled = false;

    api::PongEvent capabilities() const {
        api::PongEvent p;
        p.backend = pipeline.empty() ? "none" : "hip";
        p.model = has_model ? model_dir : std::string("(none)");
        p.max_context = max_context;
        const auto fs = backend::build_features(pipeline.empty() ? false : true);
        for (std::size_t i = 0; i < static_cast<std::size_t>(backend::Feature::count); ++i)
            if (fs.has(static_cast<backend::Feature>(i)))
                p.features.push_back(backend::feature_name(static_cast<backend::Feature>(i)));
        return p;
    }

    // 要求 1 件を実行する。戻り値は「プロセスを続けるか」。
    bool handle(const api::GenerateRequest& g) {
        const std::string& rid = g.request_id;
        if (!has_model) {
            api::EngineError e;
            e.request_id = rid;
            e.code = "model_not_found";
            e.message = "--model が指定されていない（このプロセスは ping のみ応答する）";
            write_event(api::Event{e});
            return true;
        }
        if (g.input_ids.empty()) {
            api::EngineError e;
            e.request_id = rid;
            e.code = "invalid_argument";
            e.message = "input_ids が空";
            e.param = "input_ids";
            write_event(api::Event{e});
            return true;
        }
        const std::size_t prompt = g.input_ids.size();
        if (max_context != 0 && prompt + g.max_new_tokens > max_context) {
            const Status st = Status::out_of_range(
                "context: prompt " + std::to_string(prompt) + " + max_new " +
                    std::to_string(g.max_new_tokens) + " > capacity " +
                    std::to_string(max_context),
                __FILE__, __LINE__);
            write_event(api::Event{api::error_from_status(st, api::InvalidStateKind::state_error,
                                                          rid)});
            return true;
        }

        // リクエストごとに状態を初期化する（別シーケンス。prefix cache は未実装）。
        if (auto rst = app::pipeline_reset(pipeline); !rst.ok()) {
            write_event(api::Event{api::error_from_status(rst, api::InvalidStateKind::state_error,
                                                          rid)});
            return true;
        }

        api::Sampling sampling = g.sampling;
        sampling.seed = api::assign_seed(sampling.seed, ++request_counter);

        api::DoneEvent done;
        done.request_id = rid;
        done.prompt_tokens = prompt;
        done.seed = *sampling.seed;

        auto finish = [&](api::FinishReason r) {
            done.finish_reason = r;
            done.generated_tokens = done.generated_ids.size();
            write_event(api::Event{done});
        };

        // 生成前の cancel はここで拾う（Server は未完了なら必ず cancel を送る）。
        cancelled = false;

        // ---- prefill ----
        // グラフの行数は固定なので、**満杯のチャンクだけ** prefill グラフで流し、
        // 端数は decode グラフ（1 行）で積む（shelf-validate と同じ手順）。
        std::vector<float> logits;
        {
            const std::vector<std::int32_t> ids(g.input_ids.begin(), g.input_ids.end());
            const std::size_t chunk = std::max<std::size_t>(prefill_rows, 1);
            std::size_t c = 0;
            for (; c + chunk <= ids.size(); c += chunk) {
                if (cancelled) {
                    finish(api::FinishReason::cancelled);
                    return true;
                }
                const bool last = (c + chunk == ids.size());
                auto st = app::pipeline_step(pipeline, ids.data() + c, chunk,
                                             static_cast<std::uint32_t>(c),
                                             last ? &logits : nullptr);
                if (!st.ok()) {
                    write_event(api::Event{api::error_from_status(
                        st, api::InvalidStateKind::state_error, rid)});
                    return true;
                }
            }
            for (; c < ids.size(); ++c) {
                const bool last = (c + 1 == ids.size());
                auto st = app::pipeline_step(pipeline, ids.data() + c, 1,
                                             static_cast<std::uint32_t>(c),
                                             last ? &logits : nullptr);
                if (!st.ok()) {
                    write_event(api::Event{api::error_from_status(
                        st, api::InvalidStateKind::state_error, rid)});
                    return true;
                }
            }
        }

        // ---- decode ----
        std::int32_t feed = g.input_ids.back();
        std::uint32_t pos = static_cast<std::uint32_t>(prompt);
        for (std::uint32_t i = 0; i < g.max_new_tokens; ++i) {
            const auto draw = api::sample(logits, sampling, pos);
            if (!draw.ok()) {
                write_event(api::Event{
                    api::error_from_status(draw.status(), api::InvalidStateKind::state_error, rid)});
                return true;
            }
            const std::int32_t tok = static_cast<std::int32_t>(draw.value().index);
            done.generated_ids.push_back(tok);
            write_event(api::Event{api::TokenEvent{rid, tok}});

            if (i + 1 == g.max_new_tokens) {
                finish(api::FinishReason::length);
                return true;
            }
            feed = tok;
            auto st = app::pipeline_step(pipeline, &feed, 1, pos, &logits);
            if (!st.ok()) {
                write_event(api::Event{
                    api::error_from_status(st, api::InvalidStateKind::state_error, rid)});
                return true;
            }
            ++pos;
            // 直列エンジンなので、生成中に届いた行はここで読む（cancel は即時、
            // それ以外の要求は保留して終端後に処理する）。
            while (stdin_pending()) {
                std::string pending_line;
                if (!read_line(&pending_line)) {   // EOF: 生成は続ける
                    stdin_eof = true;
                    break;
                }
                if (pending_line.empty()) continue;
                const auto preq = api::parse_request(pending_line);
                if (!preq.ok()) {
                    write_event(api::Event{api::error_from_status(preq.status())});
                    continue;
                }
                if (std::holds_alternative<api::CancelRequest>(preq.value())) {
                    cancelled = true;
                    break;
                }
                deferred.push_back(pending_line);
            }
            if (cancelled) {
                finish(api::FinishReason::cancelled);
                return true;
            }
        }
        finish(api::FinishReason::length);
        return true;
    }
};

int usage() {
    std::fprintf(stderr,
                 "usage: shelf-compute [--model DIR] [--gpu N] [--pipeline-parallel K]\n"
                 "                   [--prefill-rows N] [--max-context N]\n"
                 "  stdin/stdout: 1 行 1 オブジェクトの JSONL（engine_contract §1/§2）\n"
                 "  --model を省くと ping だけ応答する（プロトコル検証用）\n"
                 "  複数の generate を同じプロセスで続けて処理できる（reset はストリーム順）\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    Server srv;
    std::string device = "hip";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing %s\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        // 参照実装（phaseshift-compute）と同じ綴りも受ける。
        if (a == "--model" || a == "--model-dir") { srv.model_dir = next("dir"); srv.has_model = true; }
        else if (a == "--gpu") srv.gpu = std::atoi(next("n").c_str());
        else if (a == "--pipeline-parallel") srv.pp = static_cast<std::size_t>(std::atoi(next("K").c_str()));
        else if (a == "--prefill-rows") srv.prefill_rows = static_cast<std::size_t>(std::atoi(next("N").c_str()));
        else if (a == "--max-context") srv.max_context = static_cast<std::size_t>(std::atoll(next("N").c_str()));
        else if (a == "--device") device = next("cpu|hip");

        else if (a == "--help" || a == "-h") { usage(); return 0; }
        else { std::fprintf(stderr, "unknown option: %s\n", a.c_str()); return usage(); }
    }
    if (!srv.has_model) {
        // プロトコルだけのモード（ping 応答・モデル不要）。
        std::fprintf(stderr, "%s: model なしモード（ping のみ）\n", kVersion);
    }

    if (srv.has_model) {
        const std::string config_text = app::read_file_str(srv.model_dir + "/config.json");
        auto cfg_r = model::qwen35::load_text_config_json(config_text);
        if (!cfg_r.ok()) {
            std::fprintf(stderr, "shelf-compute: config: %s\n", cfg_r.status().message().c_str());
            return 1;
        }
        srv.cfg = cfg_r.release();
        auto fmt_r = app::read_fp8_format(config_text);
        if (!fmt_r.ok()) {
            std::fprintf(stderr, "shelf-compute: quantization_config: %s\n",
                         fmt_r.status().message().c_str());
            return 1;
        }
        srv.fmt = fmt_r.release();

        app::BootRequest breq;
        breq.device_execution = (device == "hip");
        breq.first_device = static_cast<std::size_t>(srv.gpu);
        breq.pipeline_parallel = srv.pp;
        if (breq.device_execution) breq.required.add(backend::Feature::device_execution);
        if (srv.pp > 1) breq.required.add(backend::Feature::pipeline_parallel);
        if (srv.fmt.quantized) breq.required.add(backend::Feature::fp8_w8a8);
        const auto booted = app::boot(breq, /*print=*/true);
        if (!booted.accepted) {
            std::fprintf(stderr, "shelf-compute: 起動条件: %s\n", booted.reason.c_str());
            return 1;
        }
        app::ShardSet shards;
        if (auto st = app::open_shards(srv.model_dir, &shards); !st.ok()) {
            std::fprintf(stderr, "shelf-compute: %s\n", st.message().c_str());
            return 1;
        }
        auto cat = app::build_catalog(shards);
        if (!cat.ok()) {
            std::fprintf(stderr, "shelf-compute: catalog: %s\n", cat.status().message().c_str());
            return 1;
        }
        if (srv.max_context == 0) srv.max_context = 4096;
        app::PipelineOptions po;
        for (std::size_t k = 0; k < srv.pp; ++k)
            po.devices.push_back(srv.gpu + static_cast<int>(k));
        po.kv_capacity = srv.max_context + 4;
        po.chunk_rows = srv.prefill_rows;
        po.logits_last_row_only = true;
        po.rope_theta = srv.cfg.rope_theta;
        po.rope_rotary_dim = srv.cfg.rotary_dim;
        po.fp8_act_group = srv.fmt.act_group;
        po.verbose = false;
        auto pl = app::make_pipeline(srv.cfg, shards, cat.value(), srv.fmt, po);
        if (!pl.ok()) {
            std::fprintf(stderr, "shelf-compute: pipeline: %s\n", pl.status().message().c_str());
            return 1;
        }
        srv.pipeline = pl.release();
        std::fprintf(stderr, "shelf-compute: ready (model=%s, max_context=%zu)\n",
                     srv.model_dir.c_str(), srv.max_context);
    }


    std::string line;
    for (;;) {
        if (!srv.deferred.empty()) {
            line = srv.deferred.front();
            srv.deferred.erase(srv.deferred.begin());
        } else if (!srv.stdin_eof && !read_line(&line)) {
            break;
        } else if (srv.stdin_eof) {
            break;
        }
        if (line.empty()) continue;
        const auto req = api::parse_request(line);
        if (!req.ok()) {
            write_event(api::Event{api::error_from_status(req.status())});
            continue;
        }
        if (std::holds_alternative<api::PingRequest>(req.value())) {
            write_event(api::Event{srv.capabilities()});
            continue;
        }
        if (std::holds_alternative<api::CancelRequest>(req.value())) {
            // 直列実行なので「いま走っている要求」は無い（Server は終端まで drain する）。
            srv.cancelled = true;
            continue;
        }
        if (!srv.handle(std::get<api::GenerateRequest>(req.value()))) break;
    }
    write_event(api::Event{api::ShutdownEvent{}});
    return 0;
}
