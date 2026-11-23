# Hy4-Flash の長文 prefill 最適化

基準は `8b7a0150`。ROCm 10.2、gfx1201、Release、`SHELF_HY4_ONLY=OFF`、単一 R9700 を使用した。GPU 実行には指定 UUID と `/tmp/r9700.lock` を使用した。`test_hy4_graph` の prefillbench は、実 checkpoint と FP8 Engram を使用するが、42 層の routed expert は第 1 MoE 層の重みを共有する。入力は `100 + (37 * i) % 128`、chunk は 1024。

## 実験 1: sparse group の bitset 化

**未採用**。受入検証を完了していない。行ごとの整数 mask を共有 bitset に変更し、popcount と prefix sum で昇順の group ID を構築する。tail の走査範囲は各 query 行の近傍に絞る。

通常実行で基準と候補を AB/BA に並べた途中結果を示す。規約上必要な各 5 回には達していない。

| token | 基準（秒） | 候補（秒） | 最後の全 logits |
|---:|---|---|---|
| 2048 | 2.101730, 2.113120 | 2.104517, 2.109371 | バイト一致 |
| 8192 | 10.238236, 10.237248 | 10.463243, 10.187533 | バイト一致 |
| 32768 | 48.127112, 47.908868 | 46.740182, 46.743451 | バイト一致 |
| 131072 | 233.029085, 233.002465 | 213.206565, 213.202532 | バイト一致 |

全中間行、境界、別入力、Graph / EP、全長の追加測定および profiler は未完了。実験結果のみで受入を判定しない。

## 回帰試験と未解決事項

候補版で必須 CPU CTest 133 件、`test_hy4_peer_transfer`、`test_hy4_reduce` が通った。`SHELF_HY4_CHECK_EVENT_REUSE=1` の Graph 試験は EP=1 の Target / DFly2 が 7/7 一致したが、EP=3 で event の完了前再使用を検出した。同じ条件の基準版でも同じ失敗を再現した。一時的なスタック取得で `GraphPlan::launch` の `ready[0][1][0]` 再記録を確認した。検査を無効化せず、event の寿命を修正し、性能と正当性を再検証する必要がある。

候補の差分は作業環境の `/tmp/hy4-deep-opt-candidate.patch` に保存した。性能の採用判断には各条件 5 回の測定が必要であり、境界前後の全行、複数入力、Graph / EP、profiler の内訳も未検証である。router、MLA、DSA、Decode の候補は実装・計測していない。
