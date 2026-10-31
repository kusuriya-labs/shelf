// shelf unit test — エンジン境界のセッション（`engine_contract.md` §1/§2）。
//
// **フェイクエンジン**をパイプ越しに動かし、ポートもモデルも無しで
// 「受付 → トークン列 → done（seed）」「cancel → done(cancelled)」「ping → pong」
// 「受付時の拒否がエンジンに届かない」「エンジン由来のエラーが 1 行で返る」を確かめる。
//
// ここで見るのは **JSONL の枠（1 行 1 オブジェクト）と型**だけ。HTTP は Python 側
// （パス A）の担当なので、このテストは Server の有無に依存しない。
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <shelf/api/protocol.hpp>

namespace {

using namespace shelf::api;

int failures = 0;
void check(bool cond, const char* what) {
    if (!cond) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

// ---- 行入出力（1 行 = 1 オブジェクト）-------------------------------------
class LineReader {
public:
    explicit LineReader(int fd) : fd_(fd) {}
    // 1 行読む（改行は落とす）。EOF は nullopt。
    std::optional<std::string> next() {
        std::string line;
        for (;;) {
            char c = 0;
            const ssize_t n = ::read(fd_, &c, 1);
            if (n <= 0) return std::nullopt;
            if (c == '\n') return line;
            line.push_back(c);
            if (line.size() > (1u << 20)) return std::nullopt;   // 上限（壊れた入力で止まる）
        }
    }
    // 未読があるか（ブロックしない）。
    bool has_pending() const {
        pollfd p{fd_, POLLIN, 0};
        return ::poll(&p, 1, 0) > 0;
    }

private:
    int fd_;
};

void write_line(int fd, const std::string& s) {
    const std::string line = s + "\n";
    std::size_t off = 0;
    while (off < line.size()) {
        const ssize_t n = ::write(fd, line.data() + off, line.size() - off);
        if (n <= 0) return;
        off += static_cast<std::size_t>(n);
    }
}

// ---- フェイクエンジン ------------------------------------------------------
// 要求を読み、決まった手順でイベントを返す。モデルも GPU も使わない。
struct FakeEngine {
    int in_fd = -1;              // 要求を読む
    int out_fd = -1;             // イベントを書く
    std::size_t tokens_per_request = 3;
    std::uint64_t seed_base = 1000;
    std::string fail_request_id;      // この request_id はエラーを返す
    std::string fail_code = "overloaded";
    std::atomic<int> requests_seen{0};

    // 1 セッション走らせる（EOF まで）。テスト側から join する。
    void run() {
        LineReader in(in_fd);
        while (auto line = in.next()) {
            const auto req = parse_request(*line);
            if (!req.ok()) {
                // 受付で落ちたものはエンジンに来ない設計だが、来たら error で返す。
                write_line(out_fd, encode_event(Event{error_from_status(req.status())}));
                continue;
            }
            ++requests_seen;
            if (const auto* p = std::get_if<PingRequest>(&req.value())) {
                (void)p;
                PongEvent pong;
                pong.backend = "fake";
                pong.model = "fake-model";
                pong.max_context = 1024;
                pong.features = {"device_execution", "fp8_w8a8"};
                write_line(out_fd, encode_event(Event{pong}));
                continue;
            }
            if (const auto* c = std::get_if<CancelRequest>(&req.value())) {
                (void)c;   // 生成前に届いた cancel は done(cancelled) で返す（下で消費）
                pending_cancel.insert(c->request_id);
                continue;
            }
            const auto& g = std::get<GenerateRequest>(req.value());
            if (!fail_request_id.empty() && g.request_id == fail_request_id) {
                EngineError e;
                e.request_id = g.request_id;
                e.code = fail_code;
                e.message = "engine: KV の予算が足りない";
                write_line(out_fd, encode_event(Event{e}));
                continue;
            }
            DoneEvent done;
            done.request_id = g.request_id;
            const bool cancelled = pending_cancel.contains(g.request_id);
            if (cancelled) {
                done.finish_reason = FinishReason::cancelled;
            } else {
                done.finish_reason = FinishReason::length;
                for (std::size_t i = 0; i < tokens_per_request; ++i) {
                    TokenEvent t;
                    t.request_id = g.request_id;
                    t.token_id = g.input_ids.empty() ? 0
                                                     : g.input_ids.back() +
                                                           static_cast<std::int32_t>(i + 1);
                    write_line(out_fd, encode_event(Event{t}));
                    done.generated_ids.push_back(t.token_id);
                }
                done.generated_tokens = done.generated_ids.size();
            }
            done.prompt_tokens = g.input_ids.size();
            done.seed = g.sampling.seed.value_or(seed_base + requests_seen.load());
            write_line(out_fd, encode_event(Event{done}));
        }
    }

    std::set<std::string> pending_cancel;
};

}  // namespace

namespace {

// ---- セッション（パイプ 2 本 + フェイクエンジンのスレッド）-----------------
class Session {
public:
    Session() {
        int r1[2], r2[2];
        if (::pipe(r1) != 0 || ::pipe(r2) != 0) {
            std::printf("FAIL: pipe を作れない\n");
            ++failures;
            return;
        }
        req_w_ = r1[1];
        req_r_ = r1[0];
        ev_r_ = r2[0];
        ev_w_ = r2[1];
        engine_.in_fd = req_r_;
        engine_.out_fd = ev_w_;
        reader_ = std::make_unique<LineReader>(ev_r_);
        th_ = std::thread([this] { engine_.run(); });
    }
    ~Session() { finish(); }

    FakeEngine& engine() { return engine_; }
    int req_fd() const { return req_w_; }
    void send(const Request& r) { write_line(req_w_, encode_request(r)); }
    void send_raw(const std::string& line) { write_line(req_w_, line); }

    // 次のイベント（EOF なら nullopt）。
    std::optional<Event> next() {
        auto line = reader_->next();
        if (!line) return std::nullopt;
        auto ev = parse_event(*line);
        if (!ev.ok()) {
            std::printf("FAIL: イベントを解析できない: %s\n", ev.status().message().c_str());
            ++failures;
            return std::nullopt;
        }
        return ev.value();
    }

    // done / error まで読み、途中の token を集める。
    std::vector<std::int32_t> drain(std::optional<DoneEvent>* done,
                                    std::optional<EngineError>* err = nullptr) {
        std::vector<std::int32_t> tokens;
        for (;;) {
            auto ev = next();
            if (!ev) return tokens;
            if (const auto* t = std::get_if<TokenEvent>(&ev.value())) {
                tokens.push_back(t->token_id);
            } else if (const auto* d = std::get_if<DoneEvent>(&ev.value())) {
                if (done) *done = *d;
                return tokens;
            } else if (const auto* e = std::get_if<EngineError>(&ev.value())) {
                if (err) *err = *e;
                return tokens;
            }
        }
    }

    void finish() {
        if (th_.joinable()) {
            if (req_w_ >= 0) { ::close(req_w_); req_w_ = -1; }
            th_.join();
        }
        for (int* fd : {&req_r_, &ev_r_, &ev_w_}) {
            if (*fd >= 0) { ::close(*fd); *fd = -1; }
        }
    }

private:
    int req_w_ = -1, req_r_ = -1, ev_r_ = -1, ev_w_ = -1;
    FakeEngine engine_;
    std::unique_ptr<LineReader> reader_;
    std::thread th_;
};

GenerateRequest make_generate(const std::string& id, std::int32_t base) {
    GenerateRequest g;
    g.request_id = id;
    g.input_ids = {base, base + 1};
    g.max_new_tokens = 8;
    return g;
}

// 1. 受付 → トークン列 → done（seed）
void test_generate_stream() {
    Session s;
    s.send(Request{make_generate("r1", 100)});
    std::optional<DoneEvent> done;
    const auto tokens = s.drain(&done);
    check(tokens.size() == 3, "token イベントが 3 個");
    check(done.has_value(), "done を受けた");
    if (done) {
        check(done->request_id == "r1", "done の request_id");
        check(done->generated_ids == tokens, "done.generated_ids は token 列と一致");
        check(done->generated_tokens == tokens.size(), "generated_tokens");
        check(done->prompt_tokens == 2, "prompt_tokens");
        check(done->finish_reason == FinishReason::length, "finish_reason=length");
        check(done->seed != 0, "seed が採番されて返る（§3.4）");
    }
}

// 2. 指定 seed はそのまま返る（再現性）
void test_seed_echo() {
    Session s;
    auto g = make_generate("r2", 200);
    g.sampling.seed = 0;   // 0 も正規の値
    s.send(Request{g});
    std::optional<DoneEvent> done;
    s.drain(&done);
    check(done.has_value() && done->seed == 0, "seed 0 がそのまま返る");
}

// 3. cancel → done(cancelled)（トークンを出さない）
void test_cancel() {
    Session s;
    CancelRequest c{"r3"};
    s.send(Request{c});
    s.send(Request{make_generate("r3", 300)});   // 2 行をまとめて送る（順序は保証）
    std::optional<DoneEvent> done;
    const auto tokens = s.drain(&done);
    check(tokens.empty(), "cancel 済みはトークンを出さない");
    check(done.has_value() && done->finish_reason == FinishReason::cancelled,
          "done(cancelled)");
    check(done && done->generated_ids.empty(), "generated_ids は空");
}

// 4. ping → pong（capabilities 握手）
void test_ping() {
    Session s;
    s.send(Request{PingRequest{}});
    auto ev = s.next();
    check(ev.has_value(), "pong が返る");
    if (ev) {
        const auto* p = std::get_if<PongEvent>(&ev.value());
        check(p != nullptr, "pong の型");
        if (p) {
            check(p->backend == "fake", "capabilities.backend");
            check(!p->features.empty(), "capabilities.features");
        }
    }
}

// 5. エンジン由来のエラーは 1 行で返り、traceback を含まない
void test_engine_error() {
    Session s;
    s.engine().fail_request_id = "boom";
    s.send(Request{make_generate("boom", 400)});
    std::optional<DoneEvent> done;
    std::optional<EngineError> err;
    s.drain(&done, &err);
    check(err.has_value(), "error イベントが返る");
    if (err) {
        check(err->code == "overloaded", "code の写像（insufficient_memory → overloaded）");
        check(err->request_id.has_value() && *err->request_id == "boom", "request_id を載せる");
        check(err->message.find('\n') == std::string::npos, "message に改行が無い（1 行 1 物体）");
        check(err->message.find("shelf-") == std::string::npos &&
                  err->message.find(".cpp") == std::string::npos,
              "message にソース位置（traceback 相当）を含まない");
    }
}

// 6. 受付時の拒否はエンジンに届かない（`serve_protocol.md` §2-2）
void test_reject_before_engine() {
    Session s;
    // 未知フィールド: 受付（Server 側の parse）で落ちる
    const auto bad = parse_request(
        R"({"op":"generate","request_id":"x","input_ids":[1],"max_new_tokens":1,"nope":1})");
    check(!bad.ok(), "受付で未知フィールドを拒否");
    // 正しい要求を 1 つ送って、エンジンが見たのが 1 件だけであることを確かめる
    s.send(Request{make_generate("ok1", 500)});
    std::optional<DoneEvent> done;
    s.drain(&done);
    check(done.has_value(), "正しい要求は通る");
    check(s.engine().requests_seen.load() == 1, "エンジンが見た要求は 1 件だけ");
}

// 7. 枠（1 行 1 オブジェクト）: 2 要求をまとめて書いても別々に処理される
void test_framing() {
    Session s;
    const std::string two = encode_request(Request{make_generate("f1", 600)}) + "\n" +
                            encode_request(Request{make_generate("f2", 700)}) + "\n";
    // 1 回の write で 2 行送る
    std::size_t off = 0;
    while (off < two.size()) {
        const ssize_t n = ::write(s.req_fd(), two.data() + off, two.size() - off);
        if (n <= 0) break;
        off += static_cast<std::size_t>(n);
    }
    std::optional<DoneEvent> d1, d2;
    const auto t1 = s.drain(&d1);
    const auto t2 = s.drain(&d2);
    check(t1.size() == 3 && t2.size() == 3, "2 要求が別々に処理される");
    check(d1 && d2 && d1->request_id == "f1" && d2->request_id == "f2",
          "request_id で区別できる");
    check(t1 != t2, "2 つ目の生成は 1 つ目と別（入力が違う）");
}

}  // namespace

int main() {
    test_generate_stream();
    test_seed_echo();
    test_cancel();
    test_ping();
    test_engine_error();
    test_reject_before_engine();
    test_framing();
    std::printf("test_api_session: failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
