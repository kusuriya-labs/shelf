// shelf api — JSONL の解析・生成と status → code の写像。
#include <shelf/api/protocol.hpp>

#include <cmath>
#include <limits>
#include <set>
#include <string>

#include <nlohmann/json.hpp>

namespace shelf::api {
namespace {

using json = nlohmann::json;

Status bad(const std::string& msg) {
    return Status::invalid_argument("protocol: " + msg, __FILE__, __LINE__);
}

std::string trimmed(std::string_view line) {
    std::string s(line);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
    std::size_t i = 0;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
    return s.substr(i);
}

// 未知キーの拒否（`engine_contract.md` §1: 検証はエンジン側）。
Status reject_unknown(const json& obj, const std::set<std::string>& allowed,
                      const char* what) {
    for (const auto& item : obj.items()) {
        if (!allowed.contains(item.key()))
            return bad(std::string(what) + ": 未知のフィールド '" + item.key() + "'");
    }
    return Status::make_ok();
}

Status need_string(const json& obj, const char* key, std::string* out) {
    const auto it = obj.find(key);
    if (it == obj.end()) return bad(std::string("必須フィールドが無い: ") + key);
    if (!it->is_string()) return bad(std::string(key) + " は文字列");
    *out = it->get<std::string>();
    if (out->empty()) return bad(std::string(key) + " が空");
    return Status::make_ok();
}

Status opt_number(const json& obj, const char* key, double* out) {
    const auto it = obj.find(key);
    if (it == obj.end()) return Status::make_ok();
    if (!it->is_number()) return bad(std::string(key) + " は数値");
    *out = it->get<double>();
    return Status::make_ok();
}

Status opt_uint(const json& obj, const char* key, std::uint64_t* out) {
    const auto it = obj.find(key);
    if (it == obj.end()) return Status::make_ok();
    if (!it->is_number_unsigned())
        return bad(std::string(key) + " は 0 以上の整数");
    *out = it->get<std::uint64_t>();
    return Status::make_ok();
}

// `shelf::Status` の message は "<file>:<line>: [<code 句>: ]<本文>" の形で
// ソース位置を含む（開発用）。クライアントには**内部パスを出さない**規約
// （`serve_protocol.md` §1）なので、プロトコルに載せる前に落とす。
// 同じ理由で改行も潰す（1 行 1 オブジェクトの枠を壊さない）。
std::string sanitize_client_message(const std::string& msg) {
    std::string m = msg;
    // 先頭の "<path>:<digits>: " を 1 回だけ落とす。
    const auto sep = m.find(": ");
    if (sep != std::string::npos) {
        const std::string head = m.substr(0, sep);
        const auto c = head.rfind(':');
        if (c != std::string::npos && head.find('/') != std::string::npos) {
            const std::string num = head.substr(c + 1);
            bool digits = !num.empty();
            for (const char ch : num)
                if (ch < '0' || ch > '9') digits = false;
            if (digits) m = m.substr(sep + 2);
        }
    }
    // "<code 句>: " も落とす（分類は `code` フィールドが持つ）。
    static const char* kPhrases[] = {"invalid argument: ", "out of range: ",
                                     "insufficient memory: ", "invalid state: ",
                                     "not found: ", "unsupported: ", "overflow: ",
                                     "device error: "};
    for (const char* p : kPhrases) {
        const std::string ph(p);
        if (m.rfind(ph, 0) == 0) {
            m = m.substr(ph.size());
            break;
        }
    }
    // 改行・タブは空白へ（1 行を保つ）。長すぎるメッセージは切る。
    for (char& ch : m)
        if (ch == '\n' || ch == '\r' || ch == '\t') ch = ' ';
    constexpr std::size_t kMax = 400;
    if (m.size() > kMax) m = m.substr(0, kMax) + "…";
    return m;
}

}  // namespace

// ---------------------------------------------------------------------------
// 操作
// ---------------------------------------------------------------------------
Result<Request> parse_request(std::string_view line) {
    const std::string text = trimmed(line);
    if (text.empty()) return bad("空行");
    json obj;
    try {
        obj = json::parse(text);
    } catch (const json::exception& e) {
        return bad(std::string("JSON として読めない: ") + e.what());
    }
    if (!obj.is_object()) return bad("1 行 1 オブジェクト（object）が必要");
    std::string op;
    SHELF_RETURN_IF_ERROR(need_string(obj, "op", &op));

    if (op == "ping") {
        SHELF_RETURN_IF_ERROR(reject_unknown(obj, {"op"}, "ping"));
        return Request{PingRequest{}};
    }
    if (op == "cancel") {
        SHELF_RETURN_IF_ERROR(reject_unknown(obj, {"op", "request_id"}, "cancel"));
        CancelRequest c;
        SHELF_RETURN_IF_ERROR(need_string(obj, "request_id", &c.request_id));
        return Request{c};
    }
    if (op != "generate")
        return bad("未知の op '" + op + "'（generate / cancel / ping）");

    SHELF_RETURN_IF_ERROR(reject_unknown(
        obj,
        {"op", "request_id", "input_ids", "max_new_tokens", "temperature", "top_p", "top_k",
         "min_p", "seed"},
        "generate"));
    GenerateRequest g;
    SHELF_RETURN_IF_ERROR(need_string(obj, "request_id", &g.request_id));

    const auto ids = obj.find("input_ids");
    if (ids == obj.end()) return bad("input_ids が無い");
    if (!ids->is_array() || ids->empty()) return bad("input_ids は空でない配列");
    g.input_ids.reserve(ids->size());
    for (const auto& v : *ids) {
        if (!v.is_number_integer()) return bad("input_ids の要素は整数");
        const auto id = v.get<std::int64_t>();
        if (id < 0 || id > std::numeric_limits<std::int32_t>::max())
            return bad("input_ids の要素が範囲外（0 以上 i32）");
        g.input_ids.push_back(static_cast<std::int32_t>(id));
    }

    const auto mnt = obj.find("max_new_tokens");
    if (mnt == obj.end()) return bad("max_new_tokens が無い");
    if (!mnt->is_number_unsigned()) return bad("max_new_tokens は 0 以上の整数");
    const auto mnt_v = mnt->get<std::uint64_t>();
    if (mnt_v == 0) return bad("max_new_tokens は 1 以上");
    if (mnt_v > std::numeric_limits<std::uint32_t>::max())
        return bad("max_new_tokens が大きすぎる");
    g.max_new_tokens = static_cast<std::uint32_t>(mnt_v);

    SHELF_RETURN_IF_ERROR(opt_number(obj, "temperature", &g.sampling.temperature));
    SHELF_RETURN_IF_ERROR(opt_number(obj, "top_p", &g.sampling.top_p));
    SHELF_RETURN_IF_ERROR(opt_number(obj, "min_p", &g.sampling.min_p));
    std::uint64_t top_k = g.sampling.top_k;
    SHELF_RETURN_IF_ERROR(opt_uint(obj, "top_k", &top_k));
    if (top_k > std::numeric_limits<std::uint32_t>::max()) return bad("top_k が大きすぎる");
    g.sampling.top_k = static_cast<std::uint32_t>(top_k);
    std::uint64_t seed = 0;
    const auto sit = obj.find("seed");
    if (sit != obj.end()) {
        SHELF_RETURN_IF_ERROR(opt_uint(obj, "seed", &seed));
        g.sampling.seed = seed;
    }

    // 検証はエンジン側（§3.2）。フィールド名を message に残す。
    std::string param;
    const auto st = validate_sampling(g.sampling, &param);
    if (!st.ok()) return st;
    return Request{std::move(g)};
}

std::string encode_request(const Request& r) {
    json obj;
    if (const auto* g = std::get_if<GenerateRequest>(&r)) {
        obj["op"] = "generate";
        obj["request_id"] = g->request_id;
        obj["input_ids"] = g->input_ids;
        obj["max_new_tokens"] = g->max_new_tokens;
        if (g->sampling.temperature != 0.0) obj["temperature"] = g->sampling.temperature;
        if (g->sampling.top_k != 0) obj["top_k"] = g->sampling.top_k;
        if (g->sampling.top_p != 1.0) obj["top_p"] = g->sampling.top_p;
        if (g->sampling.min_p != 0.0) obj["min_p"] = g->sampling.min_p;
        if (g->sampling.seed.has_value()) obj["seed"] = *g->sampling.seed;
    } else if (const auto* c = std::get_if<CancelRequest>(&r)) {
        obj["op"] = "cancel";
        obj["request_id"] = c->request_id;
    } else {
        obj["op"] = "ping";
    }
    return obj.dump();
}

// ---------------------------------------------------------------------------
// イベント
// ---------------------------------------------------------------------------
const char* to_string(FinishReason r) noexcept {
    switch (r) {
    case FinishReason::stop: return "stop";
    case FinishReason::length: return "length";
    case FinishReason::cancelled: return "cancelled";
    case FinishReason::error: return "error";
    }
    return "error";
}

std::optional<FinishReason> finish_reason_from(std::string_view s) noexcept {
    if (s == "stop") return FinishReason::stop;
    if (s == "length") return FinishReason::length;
    if (s == "cancelled") return FinishReason::cancelled;
    if (s == "error") return FinishReason::error;
    return std::nullopt;
}

std::string encode_event(const Event& e) {
    json obj;
    if (const auto* t = std::get_if<TokenEvent>(&e)) {
        obj["event"] = "token";
        obj["request_id"] = t->request_id;
        obj["token_id"] = t->token_id;
    } else if (const auto* d = std::get_if<DoneEvent>(&e)) {
        obj["event"] = "done";
        obj["request_id"] = d->request_id;
        obj["generated_ids"] = d->generated_ids;
        obj["finish_reason"] = to_string(d->finish_reason);
        obj["prompt_tokens"] = d->prompt_tokens;
        obj["generated_tokens"] = d->generated_tokens;
        obj["seed"] = d->seed;
    } else if (const auto* err = std::get_if<EngineError>(&e)) {
        obj["event"] = "error";
        if (err->request_id.has_value()) obj["request_id"] = *err->request_id;
        obj["code"] = err->code;
        obj["message"] = err->message;
        if (!err->param.empty()) obj["param"] = err->param;
    } else if (const auto* p = std::get_if<PongEvent>(&e)) {
        obj["event"] = "pong";
        obj["capabilities"] = {{"backend", p->backend},
                               {"model", p->model},
                               {"max_context", p->max_context},
                               {"features", p->features}};
    } else {
        obj["event"] = "shutdown";
    }
    return obj.dump();
}

Result<Event> parse_event(std::string_view line) {
    const std::string text = trimmed(line);
    if (text.empty()) return bad("空行");
    json obj;
    try {
        obj = json::parse(text);
    } catch (const json::exception& e) {
        return bad(std::string("JSON として読めない: ") + e.what());
    }
    if (!obj.is_object()) return bad("1 行 1 オブジェクト（object）が必要");
    std::string ev;
    if (const auto it = obj.find("event"); it != obj.end()) {
        if (!it->is_string()) return bad("event は文字列");
        ev = it->get<std::string>();
    } else if (obj.contains("code") && obj.contains("message")) {
        ev = "error";   // code/message だけの行は error とみなす（後方互換）
    } else {
        return bad("event フィールドが無い");
    }

    if (ev == "token") {
        TokenEvent t;
        SHELF_RETURN_IF_ERROR(need_string(obj, "request_id", &t.request_id));
        const auto id = obj.find("token_id");
        if (id == obj.end()) return bad("token_id が無い");
        if (!id->is_number_integer()) return bad("token_id は整数");
        const auto v = id->get<std::int64_t>();
        if (v < 0 || v > std::numeric_limits<std::int32_t>::max())
            return bad("token_id が範囲外");
        t.token_id = static_cast<std::int32_t>(v);
        return Event{t};
    }
    if (ev == "done") {
        DoneEvent d;
        SHELF_RETURN_IF_ERROR(need_string(obj, "request_id", &d.request_id));
        const auto ids = obj.find("generated_ids");
        if (ids == obj.end() || !ids->is_array()) return bad("generated_ids は配列");
        for (const auto& v : *ids) {
            if (!v.is_number_integer()) return bad("generated_ids の要素は整数");
            d.generated_ids.push_back(static_cast<std::int32_t>(v.get<std::int64_t>()));
        }
        std::string fr;
        SHELF_RETURN_IF_ERROR(need_string(obj, "finish_reason", &fr));
        const auto p = finish_reason_from(fr);
        if (!p.has_value()) return bad("未知の finish_reason '" + fr + "'");
        d.finish_reason = *p;
        SHELF_RETURN_IF_ERROR(opt_uint(obj, "prompt_tokens", &d.prompt_tokens));
        SHELF_RETURN_IF_ERROR(opt_uint(obj, "generated_tokens", &d.generated_tokens));
        SHELF_RETURN_IF_ERROR(opt_uint(obj, "seed", &d.seed));
        return Event{d};
    }
    if (ev == "error") {
        EngineError e;
        if (const auto it = obj.find("request_id"); it != obj.end() && it->is_string())
            e.request_id = it->get<std::string>();
        const auto code = obj.find("code");
        if (code == obj.end() || !code->is_string()) return bad("error の code は文字列");
        e.code = code->get<std::string>();
        const auto msg = obj.find("message");
        if (msg == obj.end() || !msg->is_string()) return bad("error の message は文字列");
        e.message = msg->get<std::string>();
        if (const auto p = obj.find("param"); p != obj.end() && p->is_string())
            e.param = p->get<std::string>();
        return Event{e};
    }
    if (ev == "pong") {
        PongEvent p;
        if (const auto c = obj.find("capabilities"); c != obj.end() && c->is_object()) {
            if (const auto b = c->find("backend"); b != c->end() && b->is_string())
                p.backend = b->get<std::string>();
            if (const auto m = c->find("model"); m != c->end() && m->is_string())
                p.model = m->get<std::string>();
            if (const auto x = c->find("max_context"); x != c->end() && x->is_number_unsigned())
                p.max_context = x->get<std::uint64_t>();
            if (const auto f = c->find("features"); f != c->end() && f->is_array()) {
                for (const auto& v : *f)
                    if (v.is_string()) p.features.push_back(v.get<std::string>());
            }
        }
        return Event{p};
    }
    if (ev == "shutdown") return Event{ShutdownEvent{}};
    return bad("未知の event '" + ev + "'");
}

// ---------------------------------------------------------------------------
// エラー写像（`serve_protocol.md` §3）
// ---------------------------------------------------------------------------
ErrorCode code_from_status(Status::Code c, InvalidStateKind kind) {
    switch (c) {
    case Status::Code::ok: return {"ok", false};
    case Status::Code::invalid_argument: return {"invalid_argument", true};
    case Status::Code::overflow: return {"overflow", true};
    case Status::Code::out_of_range: return {"context_length_exceeded", true};
    case Status::Code::unsupported: return {"unsupported_parameter", true};
    case Status::Code::not_found: return {"model_not_found", false};
    case Status::Code::invalid_state:
        return (kind == InvalidStateKind::busy) ? ErrorCode{"busy", false}
                                                : ErrorCode{"state_error", false};
    case Status::Code::insufficient_memory: return {"overloaded", false};
    case Status::Code::device_error: return {"device_error", false};
    default: return {"internal_error", false};
    }
}

EngineError error_from_status(const Status& st, InvalidStateKind kind,
                              std::optional<std::string> request_id, std::string param) {
    EngineError e;
    e.request_id = std::move(request_id);
    e.code = code_from_status(st.code(), kind).code;
    e.message = sanitize_client_message(st.message());
    e.param = std::move(param);
    return e;
}

}  // namespace shelf::api
