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

## Step 2: MLA と DSA の試作

基準は Step 1 の `f03c3c5c`、単一 R9700、同じ checkpoint・expert 共有配置と反復入力。MLA は sparse prefill の Octave 復号直後に BF16 RNE を保持したうえで latent 量子化を融合し、Q/V 射影を最大 1024 行で一括化した。最大 64 行の query の KV 和集合を作り、展開・量子化を共有する。展開数が HIP grid の上限を超える場合は grid を分割する。作業域は既存の arena の容量検査を通す。

DSA は既出 key 全体の最大値を各 16 行の indexer で繰り返し計算せず、chunk 内の最大範囲について一度計算して共有する。将来の query 行の key を含むため値は各行にとって安全側であり、境界周辺の scalar 再計算は維持した。これは group ごとの prefix max を永続保存する方式ではない。

通常実行は 2k / 8k / 32k で AB/BA 各 2 回の中央値、128k は各 1 回。末尾の全 120,832 logits はすべて基準とバイト一致した。

| token | Step 1（秒） | 候補（秒） |
|---:|---:|---:|
| 2048 | 2.095 | 2.099 |
| 8192 | 10.138 | 9.626 |
| 32768 | 45.626 | 40.924 |
| 131072 | 205.128 | 183.472 |

32k では融合のみの AB/BA が基準 45.704 / 45.515 秒、候補 45.338 / 45.249 秒で、DSA の最大値共有を加えると 44.760 / 44.763 秒だった。64 行の和集合共有を含む候補が上表の値。未使用 ID 初期化を省略する別案は 40.790 → 41.007 秒（各 1 回）で改善せず棄却した。

4097 token の chunk 16 / 512 / 1024 について、位置 510〜512、2046〜2048、4094〜4096 および chunk 境界付近の全 logits が基準と一致した。境界取得は一時的な eager 計測器を使用した。

router の WMMA 近似選択は未実装。近似 score の 16 位と 17 位の差だけでは、FP32 累積順・sigmoid・bias 加算の誤差による選択順位の不変性を保証できない。厳密な誤差上界と不確実行の fallback を確認するまでは、既存の exact router を維持する。2Mi 予約時の全経路の実機容量検証は未実施。

32k の `rocprofv3 --kernel-trace` は最終候補で 1 回取得した。profiler 付き prefill 壁時計は 41.675 秒。次は GPU kernel 時間の合計で、CPU・転送・待機を含まない。

| kernel | 呼び出し | GPU 時間（秒） |
|---|---:|---:|
| unpack_quant | 5,400 | 0.855 |
| unpack | 7,368 | 0.113 |
| quant_latent | 984 | 0.007 |
| project_query | 1,344 | 1.243 |
| project_value | 1,344 | 1.923 |
| recall_scores | 6,384 | 3.215 |
| recall_values | 6,384 | 4.165 |
| index_key_peak | 336 | 0.007 |
| index_bounds | 21,504 | 0.419 |
| index_scores_wmma | 21,504 | 0.210 |
| index_topk | 23,040 | 1.239 |
| router | 1,312 | 4.572 |
| choose | 1,312 | 0.078 |

profiler 付き実行の末尾 logits も基準と一致した。上記は未分類 kernel を含まない。物理 GPU 間転送は測定していない。

最終候補で event 再利用検査付き `test_hy4_graph` は論理 EP=1 / 3 / 4 の Target・DFly2 とも 7/7、`test_hy4_reduce` と必須 CPU CTest 133 件も通過した。128k は要求どおり各版 1 回で、反復安定性は未評価。実機結果は expert 共有配置・単一 R9700 の測定である。

## Step 3: 厳密誤差上界 WMMA router・decode/batch 固定射影・6 bpw `lm_head` argmax 融合・方向別 P2P プローブ

Step 2 を基準として、次の 3 項目を実装・検証した。

### 1. 厳密誤差上界検証付き WMMA router（採用）

- `Residency::upload()` 時に層ごとのゲート重み行ノルム `gate_expert_norms`（`[41][512]`）を事前計算し、`route_select` 側で入力行ノルム `router_row_norms` を計算した。
- `router_wmma`（`rows >= 256` は 4×4 tile、`< 256` は 1×1 tile）の BF16 積結果に対し、Higham 誤差上界 `1e-4f * ||x||_2 * ||w_e||_2` と正確な `sig_max_deriv` から候補集合 `C = top16 ∪ {s_e + δ_e >= min_top16}` を構成する。
- `|C| <= 64` は `choose_verified` の単一 CTA 内で精密計算して上位 16 expert を確定し、`|C| > 64` または非有限値を含む行のみ 64 CTA の `router_choose_fallback` へ送る。
- 効果（`routerbench`）: `32 rows` `113.344 → 52.973 us`、`96 rows` `277.477 → 54.282 us`、`1024 rows` `2774.598 → 277.061 us`（約 10 倍、全一致かつ人工 tie / near-tie / NaN ケース通過）。

### 2. Decode/Batch 固定射影（1/6/16/96 行）・6 bpw `lm_head`・greedy argmax 融合

- **採用**:
  - `rotate_input` / `rotate_output` の `uint2` / `float4` ベクトル化（`linear.hip`）。
  - `shared_gate_up` に `CompactRows = 1 / 6 / 16` のコンパクト LDS 配置と `SHUFFLE_PREVIOUS` を導入（`wmma.hip`）。`fixed_ffn` は `rows=6` `39.809 → 36.129 us`、`rows=16` `86.136 → 49.821 us`。
  - 4 wave 構成の `gemm_fused_4w` と外部 3-launch（`rotate_input` + 4-wave `batch(..., ShufflePrev=true)` + `rotate_output`）の使い分け。`mla_o` は `rows=1` で `gemm_fused_4w`（`57.424 us`）を維持し、`rows=6`（`123.396 → 96.217 us`）および `rows=16`（`368.363 → 121.333 us`）で外部 3-launch へ切り替えた。
  - `rows in (48, 96]` の 6 bpw（`lm_head`）へ `MT=3`（`batch<6, 3>`）を適用し、`lm_head rows=96` を `1581.811 → 1403.108 us` へ短縮した。
  - greedy 時の `lm_head` + `best_ids` を `shelf_hy4_head_argmax`（`rotate_output_argmax_tiles` + `argmax_tiles_reduce`）へ融合し、`logits` 書き込み・読み出し（`120832 * 4` bytes/row）と `best_ids` 起動を削減した（`lm_head_argmax rows=96`: `1358.633 us`、`Target` Eager/Graph および `DFly2` へ組込）。あわせて `GraphPlan::release` およびキャプチャ時に `logits` / `tokens` を明示初期化するようにした。
- **不採用・棄却**:
  - 96 行への `expanded` + `prefill_wmma` 適用: `linear rows=96` が `39.3 → 43.6 us`、`mla_o rows=96` が `257.1 → 264.0 us` へ悪化したため棄却。
  - 4 bpw `rows=96` への `MT=3` 適用と行長依存 `split_k` ヒューリスティック: 4 bpw では `splits` 固定下での occupancy 低下（3 block → 2 block）により `mla_o rows=96` が `257 → 286 us` へ悪化し、行長依存 `split_k` 変更は単一推論とバッチ推論・prefill チャンク境界のビット完全一致を崩すため棄却した。

### 3. 起動時の方向別 P2P プローブと `host` 自動フォールバック（採用）

- `Residency::configure_transports()` を追加し、`SHELF_HY4_EP_TRANSPORT=host` 指定時を除き、各方向 `src -> dst` で有効経路（`push` 時は `peer[src][dst]`、`peer` 時は `peer[dst][src]`）に 4 KiB パターン転送の実機プローブを実施する。
- HIP エラーまたはリードバック不一致を検出した場合はその方向のみ `peer` を `false` に落とし、警告ログ 1 行を出力して `small_staging`（`tp_size == 1`）または `reduce_staging`（`tp_size > 1`）による `host` 転送へフォールバックする。
- `tests/unit/test_hy4_peer_transfer.cpp` にエラー注入および破損データ注入テスト（`push` / `peer` 両モード、反対方向 P2P 維持、8 パスデータ整合、`__wrap_hipEventRecord` 検査）を追加した。

### 計測結果（Step 2 基準との比較、単一 R9700・expert 共有配置）

#### Prefill（chunk=1024、ranks=1、AB/BA 2 回ずつ、128k は各版 1 回）

| 長さ（tokens） | Step 2 平均（秒） | Step 3 平均（秒） | 改善率 | 末尾 120,832 logits |
|---|---:|---:|---:|---|
| 2048 | 2.098 | 1.839 | -12.3% | バイト完全一致 |
| 8192 | 9.511 | 8.565 | -9.9% | バイト完全一致 |
| 32768 | 40.749 | 36.898 | -9.5% | バイト完全一致 |
| 131072 | 183.521 | 167.246 | -8.9% | バイト完全一致 |

- 4097 token のチャンク境界検証（`chunk = 16 / 512 / 1024`）も Step 2（`chunk=1024`）およびチャンク間で末尾 logits がバイト完全一致した。

#### Decode microbench（`test_hy4_graph microbench`、全ハッシュ完全一致）

| カーネル | 行数 | 基準（us） | Step 3（us） |
|---|---:|---:|---:|
| `linear` | 1 / 6 / 16 / 96 | 15.60 / 15.81 / 40.69 / 39.33 | 15.37 / 15.92 / 21.28 / 39.26 |
| `fixed_ffn` | 1 / 6 / 16 / 96 | 31.32 / 39.81 / 86.14 / 119.61 | 31.04 / 36.13 / 49.82 / 119.34 |
| `mla_qb` | 1 / 6 / 16 / 96 | 21.32 / 26.46 / 43.23 / 87.46 | 21.89 / 26.77 / 43.32 / 85.85 |
| `mla_o` | 1 / 6 / 16 / 96 | 54.28 / 123.40 / 368.36 / 257.06 | 57.42 / 96.22 / 121.33 / 261.11 |
| `lm_head` | 1 / 6 / 16 / 96 | 475.32 / 500.77 / 528.43 / 1581.81 | 476.12 / 502.09 / 529.71 / 1403.11 |
| `lm_head_argmax` | 1 / 6 / 16 / 96 | - | 481.34 / 499.99 / 519.33 / 1358.63 |

#### Decode stepbench 中央値（5 試行、`test_hy4_graph stepbench`）

| 構成 | Graph | Step 2（ms） | Step 3（ms） |
|---|---|---:|---:|
| B=1, EP=1, DFly2 ON | OFF / ON | 55.58 / 57.11 | 55.55 / 56.78 |
| B=1, EP=1, DFly2 OFF | OFF / ON | 26.74 / 25.58 | 26.68 / 25.52 |
| B=16, EP=1, DFly2 ON | OFF / ON | 105.10 / 105.08 | 96.28 / 96.13 |
| B=16, EP=1, DFly2 OFF | OFF / ON | 40.89 / 40.05 | 40.87 / 39.90 |
| B=1, EP=3, DFly2 ON | OFF / ON | 146.69 / 137.59 | 145.00 / 136.07 |
| B=16, EP=3, DFly2 ON | OFF / ON | 219.17 / 209.93 | 212.19 / 204.64 |

#### 検証

- `SHELF_HY4_CHECK_EVENT_REUSE=1 test_hy4_graph`（既定 EP=1/3/4、`batch` EP=1/2/3/4、`transport` EP=3/4 push/peer/host・DFly2 ON/OFF）すべて通過。
- `test_hy4_routes`、`test_hy4_session`、`test_hy4_peer_transfer`（プローブ失敗注入含む）、`test_hy4_reduce`、および必須 CPU CTest 133 件すべて通過。
