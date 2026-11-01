// shelf unit test — Prefix Cache（T16）の純粋部分。
// 最長共通 prefix・参照カウント・LRU・COW・世代を固定する。
#include <cstdio>
#include <vector>

#include <shelf/runtime/state/prefix_cache.hpp>

namespace {

using namespace shelf;
using namespace shelf::runtime::state;

int failures = 0;
void check(bool cond, const char* what) {
    if (!cond) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

// tokens の先頭 n ブロックをキャッシュへ登録する（呼び手が KV を書いた想定）。
std::vector<std::uint32_t> insert_all(PrefixCache& c, const std::vector<std::int32_t>& toks) {
    std::vector<std::uint32_t> ids;
    const std::size_t full = toks.size() / c.block_tokens();
    for (std::size_t b = 0; b < full; ++b) {
        auto blk = c.allocate();
        if (!blk.ok()) break;
        if (!c.insert(blk.value(), toks, b).ok()) break;
        ids.push_back(blk.value());
    }
    return ids;
}

void test_lookup_longest() {
    PrefixCache c(8, 2);
    const std::vector<std::int32_t> toks = {1, 2, 3, 4, 5};
    const auto blocks = insert_all(c, toks);
    check(blocks.size() == 2, "2 ブロックを登録（端数は登録しない）");

    auto hit = c.lookup(toks);
    check(hit.blocks == blocks, "同じトークン列は同じブロック列");
    check(hit.matched_tokens == 4, "full block だけ一致（端数は再計算）");

    // 途中まで一致
    const std::vector<std::int32_t> other = {1, 2, 99, 100};
    auto hit2 = c.lookup(other);
    check(hit2.blocks.size() == 1 && hit2.matched_tokens == 2, "共通 prefix の長さ");
    check(c.hits() >= 2, "hit を数える");
    check(c.lookup({99, 98}).empty(), "未知の prefix は miss");
    check(c.misses() >= 1, "miss を数える");

    // 別 prefix の同トークンは別ブロック（key が親に連鎖する）
    PrefixCache d(8, 2);
    const auto a = insert_all(d, {7, 8});
    check(d.lookup({1, 2}) .empty(), "別 prefix では一致しない");
    check(d.lookup({7, 8}).blocks == a, "同じ prefix なら一致");
}

void test_refcount_and_cache() {
    PrefixCache c(8, 2);
    const std::vector<std::int32_t> toks = {10, 11, 12, 13};
    const auto blocks = insert_all(c, toks);
    check(blocks.size() == 2, "2 ブロック");
    check(c.refcount(blocks[0]) == 1, "insert 直後は 1 参照");
    c.release(blocks);
    check(c.refcount(blocks[0]) == 0, "release で 0");
    check(!c.lookup(toks).empty(), "参照 0 でもキャッシュとして一致する");
    check(c.cached_blocks() == 2, "cached 2");

    auto again = c.acquire(toks);
    check(again.ok() && again.value().blocks == blocks, "acquire で同じブロック");
    check(c.refcount(blocks[0]) == 1, "acquire で参照が戻る");
    auto twice = c.acquire(toks);
    check(c.refcount(blocks[0]) == 2, "2 回 acquire で 2 参照");
    c.release(blocks);
    c.release(blocks);
    check(c.refcount(blocks[0]) == 0 && c.cached_blocks() == 2, "両方 release で cached");
}

void test_lru_eviction() {
    PrefixCache c(2, 2);
    const std::vector<std::int32_t> a = {1, 2};
    const std::vector<std::int32_t> b = {3, 4};
    const auto ba = insert_all(c, a);
    const auto bb = insert_all(c, b);
    check(ba.size() == 1 && bb.size() == 1, "2 ブロックを使い切る");
    c.release(ba);
    c.release(bb);   // b の方が新しい（last_used が大きい）
    auto fresh = c.allocate();
    check(fresh.ok(), "LRU で 1 個確保できる");
    check(!c.registered(ba[0]), "古い方（a）が退避される");
    check(c.registered(bb[0]), "新しい方（b）は残る");
    check(c.lookup(a).empty(), "退避された prefix は miss");
    // 退避されたブロックの世代は進む
    check(c.generation(ba[0]) >= 1, "再利用で世代が進む");
}

void test_cow() {
    PrefixCache c(8, 2);
    const std::vector<std::int32_t> toks = {1, 2, 3, 4};
    const auto blocks = insert_all(c, toks);
    c.release(blocks);
    auto h1 = c.acquire(toks);   // 参照 1
    auto h2 = c.acquire(toks);   // 参照 2（2 スロットで共有）
    check(c.refcount(blocks[0]) == 2, "共有で 2 参照");

    bool copied = false;
    auto fork1 = c.fork(blocks[0], &copied);
    check(fork1.ok() && copied, "参照 2 の fork は複製する");
    check(fork1.value() != blocks[0], "新しいブロック ID");
    check(c.refcount(blocks[0]) == 1, "旧ブロックの参照が 1 つ減る");
    check(c.refcount(fork1.value()) == 1, "新ブロックは 1 参照");

    bool copied2 = false;
    auto fork2 = c.fork(fork1.value(), &copied2);
    check(fork2.ok() && !copied2 && fork2.value() == fork1.value(),
          "参照 1 の fork はそのまま使える");

    bool copied3 = false;
    auto bad = c.fork(9999, &copied3);
    check(!bad.ok(), "範囲外の fork は失敗");
}

void test_partial_and_collision_guard() {
    PrefixCache c(4, 4);
    const std::vector<std::int32_t> toks = {1, 2, 3, 4, 5};
    const auto blocks = insert_all(c, toks);
    check(blocks.size() == 1, "4 トークン 1 ブロック（端数は未登録）");
    auto hit = c.lookup(toks);
    check(hit.matched_tokens == 4, "端数は一致に含めない");
    // 登録済みブロックのトークン列が読める（衝突確認用）
    const std::int32_t* span = c.block_tokens(blocks[0]);
    check(span != nullptr && span[0] == 1 && span[3] == 4, "block_tokens が読める");
}

void test_exhaustion() {
    PrefixCache c(1, 2);
    auto a = c.allocate();
    check(a.ok(), "1 個確保");
    c.insert(a.value(), {1, 2}, 0);
    auto b = c.allocate();
    check(!b.ok(), "参照中のブロックは退避できない");
}

}  // namespace

int main() {
    test_lookup_longest();
    test_refcount_and_cache();
    test_lru_eviction();
    test_cow();
    test_partial_and_collision_guard();
    test_exhaustion();
    if (failures == 0) {
        std::printf("all prefix cache tests passed\n");
        return 0;
    }
    std::printf("%d failure(s)\n", failures);
    return 1;
}
