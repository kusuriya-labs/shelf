// shelf unit test — エンジン境界の JSONL（`engine_contract.md` §1/§2）と
// status → code の写像（`serve_protocol.md` §3）。
//
// 表駆動: serve_protocol §3 の各行を 1 ケースとして、code / 400 系判定を照合する。
#include <cstdio>
#include <string>
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

void test_parse_generate() {
    const auto r = parse_request(
        R"({"op":"generate","request_id":"r1","input_ids":[1,2,3],"max_new_tokens":8,
            "temperature":0.7,"top_k":20,"top_p":0.9,"min_p":0.05,"seed":42})");
    check(r.ok(), "generate を解析できる");
    if (r.ok()) {
        const auto* g = std::get_if<GenerateRequest>(&r.value());
        check(g != nullptr, "variant が GenerateRequest");
        if (g) {
            check(g->request_id == "r1", "request_id");
            check(g->input_ids.size() == 3 && g->input_ids[2] == 3, "input_ids");
            check(g->max_new_tokens == 8, "max_new_tokens");
            check(g->sampling.temperature == 0.7, "temperature");
            check(g->sampling.top_k == 20, "top_k");
            check(g->sampling.top_p == 0.9, "top_p");
            check(g->sampling.min_p == 0.05, "min_p");
            check(g->sampling.seed.has_value() && *g->sampling.seed == 42, "seed");
        }
    }

    // 省略時は既定（greedy）
    const auto d = parse_request(
        R"({"op":"generate","request_id":"r2","input_ids":[5],"max_new_tokens":1})");
    check(d.ok(), "省略可能フィールド");
    if (d.ok()) {
        const auto* g = std::get_if<GenerateRequest>(&d.value());
        check(g && g->sampling.is_greedy(), "既定は greedy");
        check(g && !g->sampling.seed.has_value(), "seed 未指定は nullopt");
        check(g && g->sampling.top_p == 1.0 && g->sampling.top_k == 0 && g->sampling.min_p == 0.0,
              "既定値（§3 の表）");
    }

    // 往復
    const auto again = parse_request(encode_request(r.value()));
    check(again.ok(), "encode → parse の往復");
    if (again.ok() && r.ok()) {
        const auto* a = std::get_if<GenerateRequest>(&again.value());
        const auto* b = std::get_if<GenerateRequest>(&r.value());
        check(a && b && a->request_id == b->request_id && a->input_ids == b->input_ids &&
                  a->sampling.temperature == b->sampling.temperature &&
                  a->sampling.top_k == b->sampling.top_k,
              "往復で値が保たれる");
    }
}

void test_parse_errors() {
    const std::vector<std::pair<std::string, std::string>> cases = {
        {R"({"op":"nope"})", "未知の op"},
        {R"({"op":"generate","request_id":"r","input_ids":[1]})", "max_new_tokens 欠落"},
        {R"({"op":"generate","request_id":"r","input_ids":[],"max_new_tokens":1})",
         "input_ids が空"},
        {R"({"op":"generate","request_id":"r","input_ids":[1],"max_new_tokens":0})",
         "max_new_tokens=0"},
        {R"({"op":"generate","request_id":"r","input_ids":[-1],"max_new_tokens":1})",
         "負の input_ids"},
        {R"({"op":"generate","request_id":"r","input_ids":[1],"max_new_tokens":1,"foo":1})",
         "未知のフィールド"},
        {R"({"op":"generate","request_id":"r","input_ids":[1],"max_new_tokens":1,"top_p":0})",
         "top_p の範囲"},
        {R"({"op":"ping","x":1})", "ping の未知フィールド"},
        {R"({"op":"cancel"})", "cancel の request_id 欠落"},
        {"not json", "JSON でない"},
        {"", "空行"},
        {"[]", "配列は不可"},
    };
    for (const auto& [line, what] : cases) {
        const auto r = parse_request(line);
        if (r.ok()) {
            std::printf("FAIL: %s を受理してしまった\n", what.c_str());
            ++failures;
        }
    }
    // 検証違反の message はフィールド名を持つ（Server が param に写せる）
    const auto r = parse_request(
        R"({"op":"generate","request_id":"r","input_ids":[1],"max_new_tokens":1,"min_p":2.0})");
    check(!r.ok() && r.status().message().find("min_p") != std::string::npos,
          "min_p の違反はフィールド名つき");
}

void test_events() {
    // token
    const TokenEvent t{"r1", 42};
    const auto rt = parse_event(encode_event(Event{t}));
    check(rt.ok(), "token の往復");
    if (rt.ok()) {
        const auto* p = std::get_if<TokenEvent>(&rt.value());
        check(p && p->request_id == "r1" && p->token_id == 42, "token の値");
    }

    // done（seed を必ず含む）
    DoneEvent d;
    d.request_id = "r1";
    d.generated_ids = {7, 8, 9};
    d.finish_reason = FinishReason::stop;
    d.prompt_tokens = 5;
    d.generated_tokens = 3;
    d.seed = 123456789;
    const std::string line = encode_event(Event{d});
    check(line.find("\"seed\"") != std::string::npos, "done に seed が入る");
    const auto rd = parse_event(line);
    check(rd.ok(), "done の往復");
    if (rd.ok()) {
        const auto* p = std::get_if<DoneEvent>(&rd.value());
        check(p && p->generated_ids == d.generated_ids, "generated_ids");
        check(p && p->finish_reason == FinishReason::stop, "finish_reason");
        check(p && p->seed == 123456789, "seed が保たれる");
        check(p && p->prompt_tokens == 5 && p->generated_tokens == 3, "トークン数");
    }

    // error（request_id 任意・param 任意）
    EngineError e;
    e.code = "invalid_argument";
    e.message = "sampling top_p: (0, 1] の範囲が必要";
    e.param = "top_p";
    const auto re = parse_event(encode_event(Event{e}));
    check(re.ok(), "error の往復");
    if (re.ok()) {
        const auto* p = std::get_if<EngineError>(&re.value());
        check(p && p->code == "invalid_argument" && p->param == "top_p", "code / param");
        check(p && !p->request_id.has_value(), "request_id は任意");
    }

    // pong（capabilities 握手）
    PongEvent pong;
    pong.backend = "hip";
    pong.model = "Qwen3.5-9B-FP8";
    pong.max_context = 262144;
    pong.features = {"device_execution", "fp8_w8a8"};
    const auto rp = parse_event(encode_event(Event{pong}));
    check(rp.ok(), "pong の往復");
    if (rp.ok()) {
        const auto* p = std::get_if<PongEvent>(&rp.value());
        check(p && p->backend == "hip" && p->max_context == 262144, "capabilities の値");
        check(p && p->features.size() == 2 && p->features[1] == "fp8_w8a8", "features");
    }

    // shutdown
    check(parse_event(encode_event(Event{ShutdownEvent{}})).ok(), "shutdown の往復");

    // 未知の event / 壊れた行
    check(!parse_event(R"({"event":"nope"})").ok(), "未知の event は拒否");
    check(!parse_event(R"({"event":"token"})").ok(), "token の必須欠落は拒否");
    check(!parse_event(R"({"event":"done","request_id":"r","generated_ids":[]})").ok(),
          "done の finish_reason 欠落は拒否");
    check(!parse_event("garbage").ok(), "壊れた行は拒否");
}

// serve_protocol §3 の表と 1 対 1（code / 400 系判定）。
void test_status_mapping() {
    using C = shelf::Status::Code;
    struct Row {
        C code;
        InvalidStateKind kind;
        const char* expect;
        bool client_error;
    };
    const Row rows[] = {
        {C::invalid_argument, InvalidStateKind::state_error, "invalid_argument", true},
        {C::overflow, InvalidStateKind::state_error, "overflow", true},
        {C::out_of_range, InvalidStateKind::state_error, "context_length_exceeded", true},
        {C::unsupported, InvalidStateKind::state_error, "unsupported_parameter", true},
        {C::not_found, InvalidStateKind::state_error, "model_not_found", false},
        {C::invalid_state, InvalidStateKind::busy, "busy", false},
        {C::invalid_state, InvalidStateKind::state_error, "state_error", false},
        {C::insufficient_memory, InvalidStateKind::state_error, "overloaded", false},
        {C::device_error, InvalidStateKind::state_error, "device_error", false},
    };
    for (const auto& r : rows) {
        const auto got = code_from_status(r.code, r.kind);
        if (got.code != r.expect || got.client_error != r.client_error) {
            std::printf("FAIL: 写像 %s → %s (client_error=%d)\n", r.expect, got.code.c_str(),
                        int(got.client_error));
            ++failures;
        }
    }
    // invalid_state の 2 分岐は kind で決まる（同じ status で別 code）
    check(code_from_status(C::invalid_state, InvalidStateKind::busy).code == "busy" &&
              code_from_status(C::invalid_state, InvalidStateKind::state_error).code ==
                  "state_error",
          "invalid_state は 2 分岐");

    // Status → EngineError（クライアント向けに無害化される）
    const auto st = shelf::Status::out_of_range("context: 300 > 256", __FILE__, __LINE__);
    const auto e = error_from_status(st, InvalidStateKind::state_error, std::string("r9"));
    check(e.code == "context_length_exceeded", "error_from_status の code");
    check(e.request_id.has_value() && *e.request_id == "r9", "request_id を載せる");
    check(e.message.find("context") != std::string::npos, "message を保持");
    // 内部パス・ソース位置・改行を含まない（serve_protocol §1）
    check(e.message.find(".cpp") == std::string::npos &&
              e.message.find("__FILE__") == std::string::npos &&
              e.message.find('/') == std::string::npos,
          "message にソース位置・パスを含まない");
    check(e.message.find('\n') == std::string::npos, "message に改行を含まない");
    check(e.message.find("out of range:") == std::string::npos,
          "code 句を message で繰り返さない（分類は code が持つ）");

    // 実際の生成経路（Status の文字列版）でも同じ
    const auto st2 = shelf::Status::invalid_argument(std::string("sampling top_p: (0, 1] の範囲が必要"),
                                                     __FILE__, __LINE__);
    const auto e2 = error_from_status(st2);
    check(e2.code == "invalid_argument", "invalid_argument の写像");
    check(e2.message == "sampling top_p: (0, 1] の範囲が必要",
          "クライアント向けの本文だけが残る");
}

}  // namespace

int main() {
    test_parse_generate();
    test_parse_errors();
    test_events();
    test_status_mapping();
    std::printf("test_api_protocol: failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
