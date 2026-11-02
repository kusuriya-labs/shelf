// shelf runtime/state — Prefix Cache（T16）の純粋なブロック管理。
//
// 単位は **KV ブロック**（attention が block table で指す 16 トークン等）。
// ブロックの同一性は `prefix_key.hpp` の chained key（親ブロックの key と
// ブロック内トークン列）で決まる。このクラスは GPU もモデルも知らず、
//   * 最長共通 prefix の検索（full block 単位。端数は呼び手が再計算する）
//   * 参照カウント（複数スロットで共有）
//   * LRU 退避（参照 0 のブロックを古い順に再利用）
//   * copy-on-write（参照 >1 のブロックへ書き込む前に複製）
//   * 世代（再利用で古い key を無効化）
// だけを持つ。KV の実コピーは呼び手（実行器）が行う。
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

#include <shelf/runtime/state/prefix_key.hpp>
#include <shelf/status.hpp>

namespace shelf::runtime::state {

// 検索結果。`matched_tokens = blocks.size() * block_tokens()`。
struct PrefixHit {
    std::vector<std::uint32_t> blocks;
    std::size_t matched_tokens = 0;
    bool empty() const noexcept { return blocks.empty(); }
};

class PrefixCache {
public:
    // `num_blocks` 個の固定プール。`block_tokens` は 1 ブロックのトークン数。
    PrefixCache(std::size_t num_blocks, std::size_t block_tokens);

    std::size_t num_blocks() const noexcept { return blocks_.size(); }
    std::size_t block_tokens() const noexcept { return block_tokens_; }
    std::size_t free_blocks() const noexcept;       // 参照 0（キャッシュ含む）
    std::size_t referenced_blocks() const noexcept; // refcount > 0
    std::size_t cached_blocks() const noexcept;     // refcount 0 かつ key 登録済み
    std::size_t hits() const noexcept { return hits_; }
    std::size_t misses() const noexcept { return misses_; }
    // キャッシュ済み（参照 0・登録済み）ブロックを退避した回数。未使用ブロックの
    // 再利用は数えない（LRU が実際に効いたかどうかの指標）。
    std::size_t evictions() const noexcept { return evictions_; }

    // 最長一致（参照は増やさない）。full block だけを返す。
    PrefixHit lookup(const std::vector<std::int32_t>& tokens) const;

    // 一致した prefix を参照（refcount++）して返す。
    Result<PrefixHit> acquire(const std::vector<std::int32_t>& tokens);

    // 空き（参照 0）ブロックを 1 個確保する。無ければ LRU でキャッシュを捨てる。
    // 返るブロックは refcount=1・未登録。世代は更新済み。
    Result<std::uint32_t> allocate();

    // 計算済みブロックを key で登録する（`tokens` の先頭から
    // `block_index * block_tokens` までの prefix が必要）。
    // 同じ key が既に登録済みなら（並行要求が同じ prefix を別々に計算した場合）、
    // 正本をこのブロックへ移し、旧ブロックは private（未登録）にする。参照は
    // どちらのブロックもそのまま残る。**有効なエントリを消さない**こと。
    Status insert(std::uint32_t block, const std::vector<std::int32_t>& tokens,
                  std::size_t block_index);

    // 参照を手放す（refcount--）。0 になってもキャッシュとして残る。
    void release(const std::vector<std::uint32_t>& blocks);
    void release(std::uint32_t block);

    // COW: refcount>1 なら新しいブロックを確保して返す（`*copied=true`）。
    // 呼び手は KV の中身を新ブロックへコピーする。旧ブロックの参照は 1 つ減る。
    Result<std::uint32_t> fork(std::uint32_t block, bool* copied);

    // ブロックの状態（テスト・ログ用）。
    std::size_t refcount(std::uint32_t block) const;
    std::uint64_t generation(std::uint32_t block) const;
    bool registered(std::uint32_t block) const;
    const std::int32_t* block_tokens(std::uint32_t block) const;

private:
    struct Block {
        std::uint32_t refcount = 0;
        std::uint64_t generation = 0;
        std::uint64_t last_used = 0;
        BlockKey key{};
        bool in_map = false;
        std::vector<std::int32_t> tokens;   // 登録済みのトークン列
    };
    struct Entry {
        std::uint32_t block = 0;
        std::uint64_t generation = 0;
    };

    bool valid(std::uint32_t block) const noexcept { return block < blocks_.size(); }
    void touch(std::uint32_t block);
    // 参照 0 のブロックを 1 個選ぶ（LRU）。無ければ false。
    bool pick_victim(std::uint32_t* out) const;
    // ブロックの map 登録を外す（世代は進めない）。
    void unregister(std::uint32_t block);
    // 検索の内部版。ヒットしたブロックと key を返す。
    struct Found {
        std::uint32_t block = 0;
        BlockKey key{};
    };
    std::vector<Found> find(const std::vector<std::int32_t>& tokens) const;

    std::size_t block_tokens_;
    std::vector<Block> blocks_;
    std::map<BlockKey, Entry> by_key_;
    std::uint64_t clock_ = 0;
    mutable std::size_t hits_ = 0;
    mutable std::size_t misses_ = 0;
    std::size_t evictions_ = 0;
};

}  // namespace shelf::runtime::state
