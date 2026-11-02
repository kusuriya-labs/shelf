// shelf runtime/generate — 連続バッチングのスケジューラ（T15 の純粋部分）。
//
// 役割: **最大 N スロット**の受付・Prefill のチャンク分割・decode のバッチ化・
// 終了と再利用を、GPU も I/O も持たない状態機械として決める。実行側（エンジン）は
// `next_prefill` / `next_decode` が返す行動を GPU で流し、結果を
// `prefill_progress` / `decode_result` で返すだけにする。
//
// 契約:
//   * 受付は最大スロット数まで。溢れた要求は **queued** に置き、終了で空いた
//     スロットへ **投入順** に詰める（最大 4 並列）
//   * Prefill は `prefill_chunk` 行ずつに割る。chunk の間でも cancel できる
//   * Decode は「1 スロット 1 行」をまとめて 1 ステップにする。行の順序は
//     スロット番号の昇順（決定的）
//   * EOS（停止 ID）は出力に載せない（`engine_contract.md` §2）。`length` は
//     `max_new_tokens` に達したとき
//   * cancel は冪等。終了済み・未知の ID は false
//
// 置き場: `src/runtime/generate`（生成の進行層）。`api` は再輸出しない
// （この型はエンジン内のスケジューラ用で、Server のプロトコルには出さない）。
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <string_view>
#include <vector>

#include <shelf/runtime/generate/sampling.hpp>
#include <shelf/runtime/generate/stop_tokens.hpp>

namespace shelf::runtime::generate {

// スロットの終了理由（`api::FinishReason` へエンジンが写す）。
enum class SlotEnd : std::uint8_t { none, stop, length, cancelled, error };

// 受付する 1 要求（Server の JSONL から作る）。
struct BatchRequest {
    std::string request_id;
    std::vector<std::int32_t> prompt_ids;
    std::uint32_t max_new_tokens = 0;
    StopTokens stop;
    Sampling sampling;   // エンジンが抽選に使う（スケジューラは解釈しない）
};

enum class SlotState : std::uint8_t { free, prefill, decode, done };

// Prefill の 1 チャンク（1 スロットの prompt の一部）。
struct PrefillAction {
    std::size_t slot = 0;
    std::size_t begin = 0;          // prompt 内の開始（= すでに積んだ行数）
    std::size_t rows = 0;           // このステップで流す行数
    std::uint32_t base_pos = 0;     // = begin
    const std::int32_t* tokens = nullptr;   // prompt_ids.data() + begin
};

// Decode の 1 ステップ（1 スロット 1 行）。
struct DecodeAction {
    struct Row {
        std::size_t slot = 0;
        std::uint32_t position = 0;   // これから置くトークンの位置
        std::int32_t token = 0;       // 前ステップで確定したトークン
    };
    std::vector<Row> rows;
    bool empty() const noexcept { return rows.empty(); }
};

// 終了したスロット（エンジンが done イベントを出す）。
struct Finished {
    std::size_t slot = 0;
    std::string request_id;
    std::size_t prompt_tokens = 0;
    SlotEnd end = SlotEnd::none;
    std::vector<std::int32_t> generated;
};

class BatchScheduler {
public:
    // スロットの読み取りビュー（テスト・ログ用）。
    struct SlotInfo {
        std::size_t index = 0;
        SlotState state = SlotState::free;
        std::string_view request_id;
        std::size_t prompt_len = 0;
        std::size_t prefilled = 0;
        std::size_t generated = 0;
        std::uint32_t position = 0;
        std::int32_t feed = 0;
        SlotEnd end = SlotEnd::none;
    };

    BatchScheduler(std::size_t max_slots, std::size_t prefill_chunk);

    // 受付。空きスロットがあれば即時 admit（true）、無ければ queued（false）。
    // prompt が空・max_new_tokens が 0 の要求は受け付けない（false を返す。
    // エンジンは受付前に検証する契約）。
    bool submit(BatchRequest req);

    // cancel（冪等）。実行中なら次回の `next_*` から外れ、`take_finished` で
    // cancelled として取り出せる。queued なら列から外す。
    bool cancel(std::string_view request_id);

    // 次に流す Prefill チャンク。無ければ false。
    // 複数スロットが prefill 待ちのときは **スロット番号の小さい方**を進める
    // （1 度に 1 スロットだけ。decode と混ぜるのはエンジンの方針）。
    bool next_prefill(PrefillAction* out);

    // 次に流す Decode バッチ。無ければ false（rows は空）。
    bool next_decode(DecodeAction* out);

    // Prefill チャンクを流し終えた（rows は実際に流した行数）。
    void prefill_progress(std::size_t slot, std::size_t rows);

    // T16: prefix cache で先頭 `tokens` 行を計算済みにする（0 < tokens < prompt_len）。
    // 呼び手は KV ブロックと GDN 状態をその長さに合わせてから呼ぶ。
    bool prime_prefill(std::size_t slot, std::size_t tokens);

    // Decode のサンプル結果。`stop_hit` のときは EOS なので出力に載せない。
    // `from_prefill` は prefill の最後の logits からサンプルした最初のトークン用。
    // このトークンはまだモデルに入力していないので、位置は進めない（次に流す
    // decode は `prefilled` の位置でこのトークンを処理する）。進めてしまうと
    // position 128 が飛び、prefix cache で再利用したブロックの残骸を attention が
    // 読む（実測: hit 時にスロットごとに logits が分かれた）。
    SlotEnd decode_result(std::size_t slot, std::int32_t token, bool stop_hit,
                          bool from_prefill = false);

    // 終了したスロットを取り出す（取り出した時点で free に戻し、queued を詰める）。
    bool take_finished(Finished* out);

    // 異常終了（engine error 等）。end = error にして終了扱いにする。
    void fail_slot(std::size_t slot);

    std::size_t max_slots() const noexcept { return max_slots_; }
    std::size_t active() const noexcept;   // prefill + decode
    std::size_t queued() const noexcept { return queued_.size(); }
    std::size_t free_slots() const noexcept;   // state == free の数
    SlotInfo slot(std::size_t index) const;
    // スロットの要求（抽選パラメータ・停止 ID を引く）。範囲外・free は空の要求。
    const BatchRequest& request(std::size_t index) const;
    std::size_t find(std::string_view request_id) const;   // 無ければ max_slots()

private:
    struct Slot {
        SlotState state = SlotState::free;
        BatchRequest req;
        std::size_t prefilled = 0;
        std::vector<std::int32_t> generated;
        std::uint32_t position = 0;
        std::int32_t feed = 0;
        SlotEnd end = SlotEnd::none;
    };

    bool admit(BatchRequest req);
    void admit_queued();

    std::size_t max_slots_;
    std::size_t prefill_chunk_;
    std::vector<Slot> slots_;
    BatchRequest empty_request_;   // request() の範囲外戻り値
    std::deque<BatchRequest> queued_;
    // queued の cancel はスロットを使わずに「終了済み」として積む（順序を保つ）。
    std::deque<Finished> finished_;
};

}  // namespace shelf::runtime::generate
