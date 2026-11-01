// shelf runtime/generate — 停止トークンの解決（`stop_tokens.hpp` の実装）。
#include <shelf/runtime/generate/stop_tokens.hpp>

#include <limits>

#include <nlohmann/json.hpp>

namespace shelf::runtime::generate {
namespace {

using json = nlohmann::json;

Status bad(const std::string& msg) {
    return Status::invalid_argument("stop tokens: " + msg, __FILE__, __LINE__);
}

// `eos_token_id` の値を読む。整数 / 整数配列 / null を受ける。
Status read_eos_value(const json& value, const char* what, StopTokens* out) {
    if (value.is_null()) return Status::make_ok();
    if (value.is_number_integer()) {
        const auto v = value.get<std::int64_t>();
        if (v < 0 || v > std::numeric_limits<std::int32_t>::max())
            return bad(std::string(what) + ": トークン ID が範囲外");
        out->add(static_cast<std::int32_t>(v));
        return Status::make_ok();
    }
    if (value.is_array()) {
        for (const auto& item : value) {
            if (!item.is_number_integer())
                return bad(std::string(what) + ": 配列の要素は整数");
            const auto v = item.get<std::int64_t>();
            if (v < 0 || v > std::numeric_limits<std::int32_t>::max())
                return bad(std::string(what) + ": トークン ID が範囲外");
            out->add(static_cast<std::int32_t>(v));
        }
        return Status::make_ok();
    }
    return bad(std::string(what) + ": 整数または整数配列");
}

// オブジェクトから `key` を読み、あれば `read_eos_value` に渡す。
Status read_eos_field(const json& obj, const char* key, const char* what, StopTokens* out) {
    const auto it = obj.find(key);
    if (it == obj.end()) return Status::make_ok();
    return read_eos_value(*it, what, out);
}

Result<json> parse_json(std::string_view text, const char* what) {
    json root;
    try {
        root = json::parse(text.begin(), text.end());
    } catch (const json::exception& e) {
        return bad(std::string(what) + ": JSON として読めない: " + e.what());
    }
    if (!root.is_object()) return bad(std::string(what) + ": ルートはオブジェクト");
    return root;
}

}  // namespace

bool StopTokens::contains(std::int32_t token) const noexcept {
    for (const std::int32_t id : ids)
        if (id == token) return true;
    return false;
}

void StopTokens::add(std::int32_t token) {
    if (!contains(token)) ids.push_back(token);
}

void StopTokens::merge(const StopTokens& other) {
    for (const std::int32_t id : other.ids) add(id);
}

Result<StopTokens> parse_generation_config(std::string_view json_text) {
    if (json_text.empty()) return StopTokens{};   // ファイルが無い = 停止 ID なし
    auto root = parse_json(json_text, "generation_config");
    if (!root.ok()) return root.status();
    StopTokens out;
    if (auto st = read_eos_field(root.value(), "eos_token_id", "generation_config.eos_token_id",
                                 &out);
        !st.ok())
        return st;
    return out;
}

Result<StopTokens> parse_model_config(std::string_view json_text) {
    if (json_text.empty()) return bad("config.json が空");
    auto root = parse_json(json_text, "config");
    if (!root.ok()) return root.status();
    StopTokens out;
    const auto it = root.value().find("text_config");
    if (it != root.value().end() && it->is_object()) {
        if (auto st = read_eos_field(*it, "eos_token_id", "config.text_config.eos_token_id", &out);
            !st.ok())
            return st;
    } else {
        if (auto st = read_eos_field(root.value(), "eos_token_id", "config.eos_token_id", &out);
            !st.ok())
            return st;
    }
    return out;
}

Result<StopTokens> parse_tokenizer_config(std::string_view tokenizer_config_json,
                                          std::string_view tokenizer_json) {
    if (tokenizer_config_json.empty() || tokenizer_json.empty()) return StopTokens{};
    auto cfg = parse_json(tokenizer_config_json, "tokenizer_config");
    if (!cfg.ok()) return cfg.status();
    auto tok = parse_json(tokenizer_json, "tokenizer");
    if (!tok.ok()) return tok.status();

    StopTokens out;
    // `eos_token_id` を直接持つ tokenizer 設定もある。
    if (auto st = read_eos_field(cfg.value(), "eos_token_id", "tokenizer_config.eos_token_id",
                                 &out);
        !st.ok())
        return st;

    const auto it = cfg.value().find("eos_token");
    if (it == cfg.value().end() || it->is_null()) return out;
    std::string eos;
    if (it->is_string()) {
        eos = it->get<std::string>();
    } else if (it->is_object() && it->contains("content") &&
               it->at("content").is_string()) {
        eos = it->at("content").get<std::string>();
    } else {
        return bad("tokenizer_config.eos_token は文字列または {content}");
    }
    if (eos.empty()) return out;

    // added_tokens（特殊トークン）→ model.vocab の順で ID を探す。
    const auto& root = tok.value();
    if (const auto a = root.find("added_tokens");
        a != root.end() && a->is_array()) {
        for (const auto& entry : *a) {
            if (entry.is_object() && entry.value("content", std::string()) == eos &&
                entry.contains("id") && entry.at("id").is_number_integer()) {
                const auto id = entry.at("id").get<std::int64_t>();
                if (id >= 0 && id <= std::numeric_limits<std::int32_t>::max())
                    out.add(static_cast<std::int32_t>(id));
                return out;
            }
        }
    }
    if (const auto m = root.find("model");
        m != root.end() && m->is_object()) {
        const auto v = m->find("vocab");
        if (v != m->end() && v->is_object() && v->contains(eos) &&
            v->at(eos).is_number_integer()) {
            const auto id = v->at(eos).get<std::int64_t>();
            if (id >= 0 && id <= std::numeric_limits<std::int32_t>::max())
                out.add(static_cast<std::int32_t>(id));
        }
    }
    return out;
}

Result<StopTokens> resolve_stop_tokens(std::string_view model_config_json,
                                       std::string_view generation_config_json,
                                       std::string_view tokenizer_config_json,
                                       std::string_view tokenizer_json) {
    auto model = parse_model_config(model_config_json);
    if (!model.ok()) return model.status();
    auto gen = parse_generation_config(generation_config_json);
    if (!gen.ok()) return gen.status();
    auto tok = parse_tokenizer_config(tokenizer_config_json, tokenizer_json);
    if (!tok.ok()) return tok.status();
    // generation_config を先に置き、モデル設定・tokenizer の ID を後ろへ足す（和集合）。
    // 停止は「どれかに含まれる」なので順序は結果に影響しないが、
    // ログやテストで読みやすいように主従を保つ。
    StopTokens merged = gen.value();
    merged.merge(model.value());
    merged.merge(tok.value());
    return merged;
}

StepAction decide_step(std::int32_t token, const StopTokens& stop, std::uint32_t produced,
                       std::uint32_t max_new_tokens) noexcept {
    // EOS が max_new_tokens と競合したときは EOS を優先する。
    if (stop.contains(token)) return StepAction::stop_eos;
    if (produced + 1 >= max_new_tokens) return StepAction::emit_then_stop;
    return StepAction::emit;
}

}  // namespace shelf::runtime::generate
