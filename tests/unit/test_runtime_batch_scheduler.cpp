// shelf unit test — 連続バッチングのスケジューラ（T15 の純粋部分）。
//
// GPU もモデルも使わず、受付・Prefill のチャンク分割・decode のバッチ化・
// 終了と再利用・cancel を固定する。
#include <cstdio>
#include <string>
#include <vector>

#include <shelf/runtime/generate/batch_scheduler.hpp>

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

BatchRequest req(const std::string& id, std::size_t prompt, std::uint32_t max_new,
                 std::vector<std::int32_t> stop = {}) {
    BatchRequest r;
    r.request_id = id;
    r.prompt_ids.assign(prompt, 7);
    for (std::size_t i = 0; i < prompt; ++i)
        r.prompt_ids[i] = static_cast<std::int32_t>(100 + i);
    r.max_new_tokens = max_new;
    for (const auto idv : stop) r.stop.add(idv);
    return r;
}

void test_prefill_chunks() {
    BatchScheduler s(2, 3);
    check(s.submit(req("a", 5, 4)), "空きがあれば即時 admit");
    PrefillAction p;
    check(s.next_prefill(&p), "prefill がある");
    check(p.slot == 0 && p.begin == 0 && p.rows == 3 && p.base_pos == 0, "最初のチャンクは 3 行");
    check(p.tokens && p.tokens[0] == 100, "tokens は prompt + begin");
    s.prefill_progress(p.slot, p.rows);
    check(s.slot(0).state == SlotState::prefill, "端数が残るので prefill のまま");
    check(s.next_prefill(&p) && p.rows == 2 && p.begin == 3, "2 個目は端数の 2 行");
    s.prefill_progress(p.slot, p.rows);
    check(s.slot(0).state == SlotState::decode, "prefill 完了で decode");
    check(s.slot(0).position == 5 && s.slot(0).feed == 104, "位置と feed が prompt 末尾");
}

void test_decode_batch() {
    BatchScheduler s(4, 8);
    s.submit(req("a", 2, 8));
    s.submit(req("b", 3, 8));
    PrefillAction p;
    check(s.next_prefill(&p) && p.slot == 0, "小さいスロットから prefill");
    s.prefill_progress(0, 2);
    check(s.next_prefill(&p) && p.slot == 1, "次はスロット 1");
    s.prefill_progress(1, 3);
    DecodeAction d;
    check(s.next_decode(&d) && d.rows.size() == 2, "2 スロットの decode バッチ");
    check(d.rows[0].slot == 0 && d.rows[0].position == 2 && d.rows[0].token == 101,
          "slot0 の位置と feed");
    check(d.rows[1].slot == 1 && d.rows[1].position == 3 && d.rows[1].token == 102,
          "slot1 の位置と feed");
    // 3 個目の要求は 2 スロット構成では queued。
    BatchScheduler s2(1, 8);
    check(s2.submit(req("x", 1, 4)), "1 個目 admit");
    check(!s2.submit(req("y", 1, 4)), "2 個目は queued");
    check(s2.queued() == 1, "queued 数");
}

// prefill 由来の最初のトークンは「まだ入力していない」ので位置を進めない。
// 進めてしまうと位置 N の KV が書かれず、prefix cache の再利用ブロックの残骸を
// attention が読む（2026-11-01 に実機で発見した off-by-one）。
void test_first_token_position() {
    BatchScheduler s(1, 8);
    s.submit(req("a", 3, 4));
    s.prefill_progress(0, 3);
    check(s.slot(0).position == 3, "prefill 完了で位置 3");
    check(s.decode_result(0, 7, false, /*from_prefill=*/true) == SlotEnd::none,
          "prefill 由来の最初のトークン");
    check(s.slot(0).position == 3, "位置は進めない（N+1 にしない）");
    DecodeAction d;
    check(s.next_decode(&d) && d.rows[0].position == 3 && d.rows[0].token == 7,
          "次は位置 3 で最初のトークンを処理する");
    // decode 中にサンプルしたトークンは次の位置で処理する。
    check(s.decode_result(0, 8, false) == SlotEnd::none, "decode 由来のトークン");
    check(s.slot(0).position == 4, "decode 後は位置 4");
    check(s.next_decode(&d) && d.rows[0].position == 4 && d.rows[0].token == 8,
          "次は位置 4");
}

void test_stop_and_length() {
    BatchScheduler s(1, 8);
    s.submit(req("a", 2, 3));
    s.prefill_progress(0, 2);
    check(s.decode_result(0, 11, false) == SlotEnd::none, "1 個目は継続");
    check(s.decode_result(0, 12, false) == SlotEnd::none, "2 個目は継続");
    check(s.decode_result(0, 13, false) == SlotEnd::length, "3 個目で length");
    Finished f;
    check(s.take_finished(&f) && f.end == SlotEnd::length, "length の終了を取り出せる");
    check(f.generated.size() == 3 && f.generated[2] == 13, "生成列");
    check(s.slot(0).state == SlotState::free, "終了でスロットが空く");

    BatchScheduler s2(1, 8);
    s2.submit(req("a", 2, 3, {99}));
    s2.prefill_progress(0, 2);
    check(s2.decode_result(0, 99, true) == SlotEnd::stop, "EOS は stop");
    check(s2.decode_result(0, 11, false) == SlotEnd::error, "終了済みスロットへは error");
    Finished f2;
    check(s2.take_finished(&f2) && f2.end == SlotEnd::stop && f2.generated.empty(),
          "EOS は生成列に載せない");
}

void test_reuse_and_order() {
    BatchScheduler s(2, 8);
    s.submit(req("a", 1, 2));
    s.submit(req("b", 1, 2));
    check(!s.submit(req("c", 1, 2)), "3 個目は queued");
    s.prefill_progress(0, 1);
    s.prefill_progress(1, 1);
    s.decode_result(0, 5, false);
    s.decode_result(0, 6, false);
    Finished f;
    check(s.take_finished(&f) && f.request_id == "a", "a が終了");
    check(s.slot(0).state == SlotState::prefill, "空いたスロットへ c を投入");
    check(s.slot(0).request_id == "c", "投入順（FIFO）で c");
    check(s.queued() == 0, "queued が空になった");
}

void test_cancel() {
    BatchScheduler s(2, 8);
    s.submit(req("a", 4, 8));
    s.submit(req("b", 4, 8));
    check(s.cancel("b"), "実行中の cancel");
    check(!s.cancel("unknown"), "未知 ID は false");
    check(s.active() == 1, "cancel したスロットは active から外れる");
    Finished f;
    check(s.take_finished(&f) && f.request_id == "b" && f.end == SlotEnd::cancelled,
          "cancelled の終了");
    check(!s.cancel("b"), "終了後に cancel すると false");

    // queued の cancel はスロットを使わず終了イベントになる。
    BatchScheduler q(1, 8);
    q.submit(req("x", 2, 4));
    q.submit(req("y", 2, 4));
    check(q.queued() == 1, "y は queued");
    check(q.cancel("y"), "queued の cancel");
    Finished fy;
    check(q.take_finished(&fy) && fy.request_id == "y" && fy.end == SlotEnd::cancelled,
          "queued の cancel も終了として出る");
    check(q.queued() == 0, "queued から消えた");

    BatchScheduler s2(1, 8);
    s2.submit(req("z", 2, 4));
    s2.cancel("z");
    check(!s2.next_prefill(nullptr), "nullptr は安全");
    PrefillAction p;
    check(!s2.next_prefill(&p), "cancel 後は prefill しない");
    DecodeAction d;
    check(!s2.next_decode(&d), "cancel 後は decode もしない");
}

void test_max_slots() {
    BatchScheduler s(4, 8);
    for (int i = 0; i < 4; ++i)
        check(s.submit(req(std::to_string(i), 1, 2)), "4 並列まで admit");
    check(s.active() == 4, "active 4");
    check(!s.submit(req("x", 1, 2)) && s.queued() == 1, "5 個目は queued");
    for (int i = 0; i < 4; ++i) s.prefill_progress(static_cast<std::size_t>(i), 1);
    DecodeAction d;
    check(s.next_decode(&d) && d.rows.size() == 4, "4 行の decode バッチ");
}

}  // namespace

int main() {
    test_prefill_chunks();
    test_decode_batch();
    test_first_token_position();
    test_stop_and_length();
    test_reuse_and_order();
    test_cancel();
    test_max_slots();
    if (failures == 0) {
        std::printf("all batch scheduler tests passed\n");
        return 0;
    }
    std::printf("%d failure(s)\n", failures);
    return 1;
}
