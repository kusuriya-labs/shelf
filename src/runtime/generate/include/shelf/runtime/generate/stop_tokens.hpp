// shelf runtime/generate — 停止トークンの解決と生成 1 ステップの判定
// （`engine_contract.md` §2 の `finish_reason` / §3 の生成パラメータ）。
//
// 契約:
//   * 停止 ID は次の 3 系統の**和集合**にする（停止の取りこぼしを避ける）:
//       1. generation_config.json の `eos_token_id`
//       2. モデル設定の `text_config.eos_token_id`（無ければトップレベル）
//       3. tokenizer_config.json の `eos_token`（tokenizer.json で ID に解決）
//     3 が必要な理由: checkpoint の generation_config が `<|endoftext|>` だけを
//     宣言し、チャットの実際の終端 `<|im_end|>` を落としているモデルがあるため。
//     例: 9B 系は generation_config が 248044 のみ、tokenizer の eos は 248046。
//   * `eos_token_id` は **整数または整数配列**（複数 EOS のモデルがある）。
//     `null` / 欠落は「停止 ID なし」として正常に扱う
//   * EOS と `max_new_tokens` が競合したときは **EOS を優先**する。
//     最後の 1 トークンで EOS を引いた場合は `stop`（`length` ではない）
//   * EOS と判定したトークンは `generated_ids` にも `token` イベントにも載せない
//     （クライアントへ EOS を本文として返さない）
//
// 置き場: `src/runtime/generate`（生成の進行層）。JSON の読み取りだけを行い、
// モデル名・backend・HIP を知らない（境界ルール #12）。
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <shelf/status.hpp>

namespace shelf::runtime::generate {

// 停止 ID の集合（順序は設定に書かれた順。重複は持たない）。
struct StopTokens {
    std::vector<std::int32_t> ids;

    bool empty() const noexcept { return ids.empty(); }
    bool contains(std::int32_t token) const noexcept;
    // 追加（重複は無視）。
    void add(std::int32_t token);
    void merge(const StopTokens& other);
};

// `generation_config.json` の `eos_token_id`（整数 / 整数配列 / null）。
// フィールドが無い・null の場合は空で成功する。
Result<StopTokens> parse_generation_config(std::string_view json_text);

// モデル設定の `text_config.eos_token_id`（整数 / 整数配列）。
// `text_config` が無い設定（平坦な config）ではトップレベルの `eos_token_id` を読む。
// フィールドが無い・null の場合は空で成功する。
Result<StopTokens> parse_model_config(std::string_view json_text);

// tokenizer_config.json の `eos_token`（文字列または `{"content": ...}`）を
// tokenizer.json の `added_tokens` / `model.vocab` で ID に解決する。
// どちらかのファイルが空なら空（停止 ID なし）で成功する。
Result<StopTokens> parse_tokenizer_config(std::string_view tokenizer_config_json,
                                          std::string_view tokenizer_json);

// 上の 3 系統を解決する。空文字列のファイルは「無い」ものとして扱う
// （tokenizer を持たないチェックポイントもある）。
Result<StopTokens> resolve_stop_tokens(std::string_view model_config_json,
                                       std::string_view generation_config_json,
                                       std::string_view tokenizer_config_json = {},
                                       std::string_view tokenizer_json = {});

// 生成 1 ステップの判定（純粋関数）。
// `produced` は**このトークンを出す前**に出力済みのトークン数。
enum class StepAction : std::uint8_t {
    emit,          // 出力して継続
    emit_then_stop,// 出力して `length` で終了（max_new_tokens に到達）
    stop_eos,      // 出力せず `stop` で終了
};

StepAction decide_step(std::int32_t token, const StopTokens& stop, std::uint32_t produced,
                       std::uint32_t max_new_tokens) noexcept;

}  // namespace shelf::runtime::generate
