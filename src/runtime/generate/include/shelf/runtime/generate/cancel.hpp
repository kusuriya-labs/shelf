// shelf runtime/generate — 要求の識別と cancel の照合（`engine_contract.md` §4）。
//
// 契約:
//   * `cancel` は **`request_id` が一致したときだけ**有効（別要求の cancel で
//     走っている生成を止めない）
//   * 生成開始前に届いた cancel は、その `request_id` の生成が始まった時点で
//     `done(cancelled, 0 トークン)` にする（保留して照合する）
//   * `cancel` は冪等。終端後の cancel は何もしない
//
// 置き場: `src/runtime/generate`。I/O を持たない純粋な状態機械なので、
// エンジン（`apps/shelf-compute`）からも単体テストからも同じものを使える。
#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_set>

namespace shelf::runtime::generate {

class CancelToken {
public:
    // 保留できる「開始前 cancel」の上限（未知 ID の cancel でメモリを食わない）。
    explicit CancelToken(std::size_t max_pending = 64) : max_pending_(max_pending) {}

    // 要求の実行開始。前の要求の状態はここで捨てる。
    void begin(std::string request_id) {
        active_ = std::move(request_id);
        cancelled_ = false;
    }

    // 要求の実行終了（成功・失敗・キャンセルのいずれでも呼ぶ）。
    void end() noexcept {
        active_.clear();
        cancelled_ = false;
    }

    // 実行中の要求に対する cancel。一致すれば取り消して true。
    // 一致しない場合は「まだ始まっていない要求向け」として保留する
    // （保留上限を超えた分は捨てる）。
    bool request(std::string_view request_id) {
        if (!active_.empty() && active_ == request_id) {
            cancelled_ = true;
            return true;
        }
        if (pending_.size() < max_pending_) pending_.emplace(request_id);
        return false;
    }

    // `begin()` の直後に呼ぶ。開始前に cancel されていれば true。
    bool consume_pending(std::string_view request_id) {
        const auto it = pending_.find(std::string(request_id));
        if (it == pending_.end()) return false;
        pending_.erase(it);
        cancelled_ = true;
        return true;
    }

    bool cancelled() const noexcept { return cancelled_; }
    const std::string& active() const noexcept { return active_; }
    std::size_t pending_count() const noexcept { return pending_.size(); }

private:
    std::string active_;
    bool cancelled_ = false;
    std::size_t max_pending_;
    std::unordered_set<std::string> pending_;
};

}  // namespace shelf::runtime::generate
