# Hy4-Flash の Graph と prefill 最適化

基準は `8b7a0150`。ROCm 10.2、gfx1201、Release、`SHELF_HY4_ONLY=OFF`、単一 R9700 を使用した。GPU 実行には指定 UUID と `/tmp/r9700.lock` を使用した。`test_hy4_graph` の prefillbench は実 checkpoint と FP8 Engram を使用するが、42 層の routed expert は第 1 MoE 層の重みを共有する。入力は `100 + (37 * i) % 128`、chunk は 1024。通常実行と profiler は分離した。

## 実験 1: Graph event の寿命

**採用**。従来は `GraphPlan::launch` が層間で同一 event を再記録し、`SHELF_HY4_CHECK_EVENT_REUSE=1` の EP=3 試験で完了前再使用が検出された。層ごとに scatter、local、expert 完了 event を分離し、2 スロットの再生完了 event で同一スロットの再使用を制御する。スロットが未完了の場合にのみホストが待つ。

論理 EP=3、DFly2 無効の `stepbench` で基準→候補→候補→基準を実行した。各実行内の Graph 有効 5 試行の中央値は、B=1 が基準 113.744 / 113.303 ms、候補 113.768 / 113.445 ms、B=16 が基準 132.450 / 131.778 ms、候補 132.606 / 132.221 ms。明確な性能低下は観測されなかった。初回の event 作成は定常 step の値に含まれない。部分初期化後の再試行を補強した最終版と補強前を同じ方式で比較すると、B=16 の Graph 有効中央値は補強前 132.315 / 132.688 ms、最終版 132.599 / 132.431 ms だった。

## 実験 2: sparse ID と KV の無効行

**採用**。行ごとの整数 mask を共有 bitset に変更し、popcount と prefix sum で昇順の group ID を構築する。tail の走査範囲を query 行の近傍に絞り、unpack と latent 量子化は device 上の実 ID 数以降で終了する。

基準と Prefill-1 を AB/BA の順に各 3 回測定した。値は中央値。最後の全 120,832 logits は各長でバイト一致した。

| token | 基準（秒） | Prefill-1（秒） |
|---:|---:|---:|
| 8192 | 10.240 | 10.210 |
| 32768 | 48.017 | 45.901 |
| 131072 | 233.001 | 206.608 |

早期終了の寄与を bitset 単独と AB/BA で比較したところ、32k は 46.794 / 46.714 秒から 45.983 / 45.963 秒になり、末尾 logits が一致した。8k では差が測定の揺れに埋もれた。

## 実験 3: router と top-16

**採用**。router は 16 行以下の従来経路を維持し、より多い行数で 4 行が BF16 重みを共有する。選択は wave ごとの上位 16 件を順序付けし、厳密な比較順で段階的にマージする。

実重みを用いる単体 AB/BA では、1024 行 router の 2 回の中央値が 2884.597 / 2890.565 µs から 2732.724 / 2732.391 µs、選択が 64.826 / 64.532 µs から 60.345 / 60.621 µs。実重みの全 score、選択 ID・weight と、同点を含む人工 score に対する選択結果は、1 / 6 / 16 / 96 / 1024 行でバイト一致した。4 行共有を 1 / 6 行にも使用する案は router 単体でそれぞれ約 15.4 → 23.2 µs、26.7 → 43.7 µs と悪化したため棄却した。

Step 1 全体の通常実行は AB/BA で各 3 回。中央値は次のとおり。末尾 logits は各長でバイト一致した。

| token | 基準（秒） | Step 1（秒） |
|---:|---:|---:|
| 2048 | 2.120 | 2.097 |
| 8192 | 10.238 | 10.141 |
| 32768 | 48.077 | 45.624 |
| 131072 | 232.975 | 205.197 |

## kernel trace

最終候補の通常実行とは別に、各入力長で ROCm 10.2 の `rocprofv3 --kernel-trace` を 1 回実行した。表は kernel 名による分類の GPU 実行時間合計（秒）であり、壁時計、CPU、転送、待機時間ではない。該当 kernel は重み upload や KV fold では起動されない。profiler 付き prefill 壁時計は 2k / 8k / 32k / 128k で 2.115 / 10.627 / 47.349 / 212.694 秒だった。

| kernel | 2k | 8k | 32k | 128k |
|---|---:|---:|---:|---:|
| sparse_group_ids | 0 | 0.019 | 0.127 | 0.714 |
| sparse_tail_emit | 0 | 0.007 | 0.049 | 0.221 |
| sort_top | 0 | 0.023 | 0.133 | 0.558 |
| unpack | 0.003 | 0.308 | 1.979 | 12.417 |
| quant_latent | 0.001 | 0.047 | 0.744 | 4.877 |
| recall_scores | 0.151 | 0.774 | 3.422 | 14.319 |
| recall_values | 0.207 | 1.165 | 5.578 | 23.815 |
| router | 0.278 | 1.098 | 4.509 | 17.581 |
| choose | 0.005 | 0.019 | 0.077 | 0.311 |

128k の sparse_group_ids は 95,256 回、unpack は 198,240 回、quant_latent と recall_scores / recall_values はそれぞれ 99,120 回起動された。これらは未分類の kernel を含まない。trace は単一 GPU の expert 共有配置によるもので、物理 EP の内訳には置き換えられない。

## 正当性

4097 token の反復入力について、chunk 1 / 8 / 15 / 16 / 20 / 32 / 512 / 1024 それぞれで、位置 510〜512、2046〜2048、4094〜4096 および chunk 境界付近の全 logits と末尾 logits が基準とバイト一致した。境界採取は一時的な計測器を使い eager で行った。Prefill-1 単独と Step 1 全体の双方を照合した。

最終候補の必須 CPU CTest 133 件、`test_hy4_peer_transfer`、`test_hy4_reduce` が通過した。event 再利用検査付き `test_hy4_graph` は論理 EP=1 / 3 / 4 で Target・DFly2 とも 7/7 一致し、capture 失敗時の fallback も通過した。追加のバッチ Graph 試験は event 再利用検査付きで EP=1 の B=2〜16、EP=2〜4 の B=2 / 4 / 8 / 16 を Graph・DFly2 の有効 / 無効で通過した。これらは物理 GPU 間転送の試験ではない。
