// shelf unit test — 停止トークンの解決と生成 1 ステップの判定（T01）、
// および `request_id` を照合する cancel（T03）。
//
// 表駆動: 「generation_config / モデル設定 / どちらも無い」と
// 「EOS と max_new_tokens の競合」を 1 ケースずつ固定する。
#include <cstdio>
#include <string>
#include <vector>

#include <shelf/runtime/generate/cancel.hpp>
#include <shelf/runtime/generate/stop_tokens.hpp>

namespace {

using namespace shelf;
using namespace shelf::runtime::generate;

int failures = 0;
void check(bool cond, const char* what) {
    if (!cond) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

std::string ids_of(const StopTokens& s) {
    std::string out;
    for (const auto id : s.ids) out += std::to_string(id) + ",";
    return out;
}

void test_parse_generation_config() {
    {
        const auto r = parse_generation_config(R"({"eos_token_id": 248044})");
        check(r.ok(), "generation_config: 整数の eos を読める");
        if (r.ok()) check(ids_of(r.value()) == "248044,", "generation_config: 整数の値");
    }
    {
        const auto r = parse_generation_config(R"({"eos_token_id": [248046, 248044]})");
        check(r.ok(), "generation_config: 配列の eos を読める");
        if (r.ok()) check(ids_of(r.value()) == "248046,248044,", "generation_config: 複数 EOS の順序");
    }
    {
        const auto r = parse_generation_config(R"({"eos_token_id": 5, "pad_token_id": 0})");
        check(r.ok(), "generation_config: pad は停止にしない");
        if (r.ok()) check(r.value().ids.size() == 1, "generation_config: eos だけを読む");
    }
    {
        const auto r = parse_generation_config(R"({"use_cache": true})");
        check(r.ok() && r.value().empty(), "generation_config: eos が無ければ空で成功");
    }
    {
        const auto r = parse_generation_config(R"({"eos_token_id": null})");
        check(r.ok() && r.value().empty(), "generation_config: null は空");
    }
    {
        const auto r = parse_generation_config("");
        check(r.ok() && r.value().empty(), "generation_config: ファイル無し（空文字列）は空");
    }
    {
        const auto r = parse_generation_config(R"({"eos_token_id": "x"})");
        check(!r.ok(), "generation_config: 文字列の eos は拒否");
    }
    {
        const auto r = parse_generation_config(R"({"eos_token_id": [1, 1.5]})");
        check(!r.ok(), "generation_config: 配列内の非整数は拒否");
    }
    {
        const auto r = parse_generation_config(R"({"eos_token_id": -1})");
        check(!r.ok(), "generation_config: 負の ID は拒否");
    }
    {
        const auto r = parse_generation_config(R"({"eos_token_id": [3],)");
        check(!r.ok(), "generation_config: 壊れた JSON は拒否");
    }
}

void test_parse_model_config() {
    {
        const auto r = parse_model_config(R"({"text_config": {"eos_token_id": 248044}})");
        check(r.ok(), "model config: text_config.eos を読める");
        if (r.ok()) check(ids_of(r.value()) == "248044,", "model config: 値");
    }
    {
        const auto r = parse_model_config(R"({"text_config": {"eos_token_id": [7, 8]}})");
        check(r.ok(), "model config: 配列の eos を読める");
        if (r.ok()) check(ids_of(r.value()) == "7,8,", "model config: 複数 EOS");
    }
    {
        const auto r = parse_model_config(R"({"eos_token_id": 9})");
        check(r.ok(), "model config: text_config が無ければトップレベル");
        if (r.ok()) check(ids_of(r.value()) == "9,", "model config: トップレベルの値");
    }
    {
        const auto r = parse_model_config(R"({"text_config": {}})");
        check(r.ok() && r.value().empty(), "model config: eos が無ければ空");
    }
    {
        const auto r = parse_model_config("");
        check(!r.ok(), "model config: 空は拒否（config.json は必須）");
    }
}

void test_parse_tokenizer_config() {
    const std::string tok = R"({"added_tokens": [
            {"id": 248044, "content": "<|endoftext|>", "special": true},
            {"id": 248046, "content": "<|im_end|>", "special": true}],
        "model": {"vocab": {"hello": 5}}})";
    {
        const auto r = parse_tokenizer_config(R"({"eos_token": "<|im_end|>"})", tok);
        check(r.ok(), "tokenizer: eos_token を読める");
        if (r.ok()) check(ids_of(r.value()) == "248046,", "tokenizer: added_tokens で ID 解決");
    }
    {
        const auto r = parse_tokenizer_config(
            R"({"eos_token": {"content": "<|endoftext|>", "lstrip": false}})", tok);
        check(r.ok() && ids_of(r.value()) == "248044,", "tokenizer: {content} 形式");
    }
    {
        const auto r = parse_tokenizer_config(R"({"eos_token": "hello"})", tok);
        check(r.ok() && ids_of(r.value()) == "5,", "tokenizer: model.vocab へフォールバック");
    }
    {
        const auto r = parse_tokenizer_config(R"({"eos_token_id": 7})", tok);
        check(r.ok() && ids_of(r.value()) == "7,", "tokenizer: eos_token_id を直接読む");
    }
    {
        const auto r = parse_tokenizer_config(R"({"eos_token": null})", tok);
        check(r.ok() && r.value().empty(), "tokenizer: null は空");
    }
    {
        const auto r = parse_tokenizer_config(R"({"eos_token": "missing"})", tok);
        check(r.ok() && r.value().empty(), "tokenizer: 語彙に無い eos は空（致命にしない）");
    }
    {
        const auto r = parse_tokenizer_config("", tok);
        check(r.ok() && r.value().empty(), "tokenizer: 設定ファイルが無ければ空");
    }
    {
        const auto r = parse_tokenizer_config(R"({"eos_token": "<|im_end|>"})", "");
        check(r.ok() && r.value().empty(), "tokenizer: tokenizer.json が無ければ空");
    }
}

void test_resolve() {
    const std::string model = R"({"text_config": {"eos_token_id": 248044}})";
    {
        const auto r = resolve_stop_tokens(model, "");
        check(r.ok() && ids_of(r.value()) == "248044,",
              "resolve: generation_config が無ければモデル設定");
    }
    {
        const auto r = resolve_stop_tokens(model, R"({"eos_token_id": [248046, 248044]})");
        check(r.ok() && ids_of(r.value()) == "248046,248044,",
              "resolve: 両方にあれば和集合（重複なし）");
    }
    {
        const auto r = resolve_stop_tokens(model, R"({"eos_token_id": 248046})");
        check(r.ok() && ids_of(r.value()) == "248046,248044,",
              "resolve: generation_config を先に、モデル設定を後ろに");
    }
    {
        const auto r = resolve_stop_tokens(R"({"text_config": {}})", "");
        check(r.ok() && r.value().empty(), "resolve: どちらにも無ければ空");
    }
    {
        // 9B 系の実際の形: generation_config / モデル設定は 248044 だけで、
        // tokenizer の eos は 248046（<|im_end|>）。和集合で両方を止める。
        const auto r = resolve_stop_tokens(
            model, R"({"eos_token_id": 248044})",
            R"({"eos_token": "<|im_end|>"})",
            R"({"added_tokens": [{"id": 248046, "content": "<|im_end|>"}]})");
        check(r.ok() && ids_of(r.value()) == "248044,248046,",
              "resolve: tokenizer の eos を取り込む（<|im_end|>）");
    }
}

void test_decide_step() {
    StopTokens two;
    two.add(100);
    two.add(200);

    // 途中の EOS: 出力せず stop（EOS は本文に載せない）。
    check(decide_step(100, two, 0, 8) == StepAction::stop_eos, "decide: 途中の EOS は stop");
    check(decide_step(200, two, 3, 8) == StepAction::stop_eos, "decide: 2 つ目の EOS も stop");

    // max_new_tokens の最後で EOS: length ではなく stop（EOS 優先）。
    check(decide_step(100, two, 7, 8) == StepAction::stop_eos, "decide: 最後の EOS は stop");
    check(decide_step(200, two, 0, 1) == StepAction::stop_eos, "decide: max=1 でも EOS は stop");

    // 最後の非 EOS: emit して length。
    check(decide_step(7, two, 7, 8) == StepAction::emit_then_stop, "decide: 最後の非 EOS は length");
    check(decide_step(7, two, 0, 1) == StepAction::emit_then_stop, "decide: max=1 の非 EOS は length");

    // 途中の非 EOS: 継続。
    check(decide_step(7, two, 3, 8) == StepAction::emit, "decide: 途中の非 EOS は継続");

    // 停止 ID が無ければ max_new_tokens だけで止まる。
    const StopTokens none;
    check(decide_step(100, none, 0, 4) == StepAction::emit, "decide: 停止 ID なしは継続");
    check(decide_step(100, none, 3, 4) == StepAction::emit_then_stop,
          "decide: 停止 ID なしでも最後は length");
}

void test_cancel_token() {
    CancelToken c;
    check(!c.cancelled(), "cancel: 初期は未取消");

    c.begin("a");
    check(!c.cancelled(), "cancel: begin で未取消");
    check(!c.request("b"), "cancel: 別 ID は取り消さない");
    check(!c.cancelled(), "cancel: 別 ID でも実行中の要求は継続");
    check(c.pending_count() == 1, "cancel: 別 ID は次の要求向けに保留");
    check(c.request("a"), "cancel: 一致した ID は取り消す");
    check(c.cancelled(), "cancel: 一致後は取消状態");
    check(c.request("a"), "cancel: 冪等（2 回目も true）");
    c.end();
    check(!c.cancelled(), "cancel: end で解除");

    // 開始前 cancel は begin 直後に消費される。
    CancelToken d;
    d.request("next");
    check(d.pending_count() == 1, "cancel: 開始前の cancel を保留");
    d.begin("next");
    check(d.consume_pending("next"), "cancel: begin 後に保留を消費");
    check(d.cancelled(), "cancel: 開始前 cancel は取消になる");
    check(!d.consume_pending("next"), "cancel: 消費は 1 回だけ");

    // 保留の上限（未知 ID の cancel で無限に増やさない）。
    CancelToken bounded(2);
    bounded.request("x");
    bounded.request("y");
    bounded.request("z");
    check(bounded.pending_count() == 2, "cancel: 保留の上限を守る");

    // 終端後の cancel は何もしない（別要求へ保留されるだけ）。
    CancelToken e;
    e.begin("done");
    e.end();
    check(!e.request("done"), "cancel: 終端後の cancel は実行中の要求を止めない");
    check(!e.cancelled(), "cancel: 終端後の cancel で取消状態にならない");
}

}  // namespace

int main() {
    test_parse_generation_config();
    test_parse_model_config();
    test_parse_tokenizer_config();
    test_resolve();
    test_decide_step();
    test_cancel_token();
    if (failures == 0) {
        std::printf("all runtime_generate stop/cancel tests passed\n");
        return 0;
    }
    std::printf("%d failure(s)\n", failures);
    return 1;
}
