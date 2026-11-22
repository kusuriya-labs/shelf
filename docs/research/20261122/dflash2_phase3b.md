# DFlash2 の 15 draft・16 行 verify

## 測定条件

R9700（gfx1201）1 台、ROCm 10.2、Release、HIP Graph 有効。全 GPU 実行で R9700 の UUID を `ROCR_VISIBLE_DEVICES` に指定し、`flock /tmp/r9700.lock` で排他した。target は `qwen38-27b-exl3-305-mul24`、drafter は `dflash2-drafter-exl3-4bpw-mul24`。16 行用には元の重みをコピーし、一時 checkpoint の `dflash_config.block_size` だけを 16 に変更した。元の重みとコピーの SHA-256 はともに `5ced4d4302a075650982f86774fb4a6e45521342eb79edda20f67a9db595c1ae`。受理率はこの入力と重みの組み合わせでの値である。

verify の Graph キャッシュはホスト側に 8・10・16 行と greedy / sampled の組み合わせを 6 枠持つ。16 行の GPU Graph の生成と capture は、その形状を初めて使うときに行う。

単一要求の stdio で BF16 KV、最大系列長 32768、prefill chunk と token budget は 4096。性能測定では tree budget と prefix cache を 0 にし、greedy と温度 0.7（top-k 20、top-p 0.95、seed 42）を使った。各プロセスで JSON と code の 2 入力を両温度で 2 周し、各要求の生成上限は 64 token。入力 ID は [Phase 2 の short 配列](dflash2_phase2/inputs.json)を使用した。

## GPU 時間

`SHELF_DFLASH2_COST_TRACE=1` の HIP event を使用した。各 run の 2 周目から所定の行数の round だけを抽出し、各 run の中央値をさらに試行間で中央値にした。初回 capture と短縮 round は含めていない。drafter の値は提案全体の時間である。

| block size | 温度 | verify ms（coarse 有効） | drafter ms（元の head） | drafter ms（coarse → fine） |
|---:|---:|---:|---:|---:|
| 8 | 0 | 30.417 | 4.881 | 4.437 |
| 8 | 0.7 | 30.793 | 4.863 | 4.427 |
| 16 | 0 | 33.521 | 5.689 | 5.289 |
| 16 | 0.7 | 34.125 | 5.653 | 5.281 |

既報の 8 行 verify 30.48 ms との差は、16 行で greedy が +3.04 ms、温度 0.7 が +3.65 ms。同一条件で再測定した 8 行との差はそれぞれ +3.10 ms、+3.33 ms だった。

16 行では元の head を A、coarse → fine を B とし、AB・BA・AB の 3 ペアを測定した。元の head での verify は greedy 33.606 ms、温度 0.7 で 34.187 ms。両 head の全生成 ID とラウンドごとの提案数・受理数は全試行で一致した。各試行の暖機後の全出力を decode 時間で割ると 226.88 → 229.81 tok/s。GPU 時間と受理率は実際の 16 行用学習済み checkpoint で再計測する必要がある。

### block size 8 の比較

変更前の `6be62fc8` の実行ファイルを保存し、変更後・変更前・変更前・変更後の順に同じ入力で比較した。7 draft の暖機後の decode は 173.35 → 173.46 tok/s。8 行の元の head は同じ入力で 2 run を追加した。8 行 verify は greedy 30.413 → 30.417 ms、温度 0.7 で 30.815 → 30.793 ms。greedy の tree budget 10 は 160.24 → 167.87 tok/s、10 行 verify は 33.247 → 31.168 ms。両形状とも生成 ID と各 round の受理数は基準版と一致した。

## 16 行の 3-bit 射影

K は 5120、行タイルは WMMA 1 枚。ランダムな trellis と BF16 入力（seed 38）で、個別の射影とまとめた射影の出力全体を BF16 で保存して照合した。各試行で L3 容量を超える 128 MiB を書き込み、暖機 2 回の後に HIP event で 20 回測定した。表は AB・BA を交互にした 5 ペアの中央値である。

| 出力列数 | split | 個別 µs | 統合 µs | 不一致要素 |
|---|---|---:|---:|---:|
| 17408 / 17408 | 2 / 2 | 348.23 | 316.76 | 0 |
| 10240 / 6144 | 1 / 2 | 172.92 | 169.52 | 0 |
| 12288 / 1024 / 1024 | 1 / 5 / 5 | 169.54 | 145.88 | 0 |

実 target の温度 0.7 で `rocprofv3 --kernel-trace --hip-graph-trace` を使い、16 行 verify の Graph 内 1,469 dispatch のうち 117 回が `group_gemm<3, 1, true>` であることを確認した。drafter の Graph 内は 199 dispatch で、15 行の coarse GEMM と selector を含む。非計測時の debug ログでは greedy と sampled の 16 行 Graph の capture と replay を確認した。

## 受理・復元

target と drafter の Graph を同時に ON/OFF、coarse head と radix prefix cache をそれぞれ独立に ON/OFF とした全 8 設定で比較した。各設定で 95・96・97 token の入力を両温度で各 2 回送信し、反復入力も 1 回送信した。各設定の 13 要求、340 生成 token、121 round の生成 ID・提案数・受理数が一致した。うち 118 round が 15 draft、81 round が一部受理で、生成末尾では最小 2 draft に短縮した。cache 有効時の再送は順に 80・80・96 token を復元し、初回と cache 無効時は 0 だった。

出力ページの再利用は、入力末尾を復元する試験とは別に測った。95・96・97 token の反復入力で 15/15 の全受理を発生させ、1 round で 16 token を進めた。さらに行ごとの語彙マスクで全受理を固定したテストで、開始位置 95・96・97 からそれぞれ 111・112・113 まで進めた。次の要求では出力側の境界 96・112・112 から復元でき、cache 無効時の生成 ID と一致した。

部分受理と全受理の両方を含む追試は各設定 18 要求。Graph・head・cache の 8 設定で個々の要求の生成 ID と合計受理数が一致した。同じ出力 prefix について、cold prefill と生成中に保存したページからの復元とでは、反復入力に続く 3 要求で受理ラウンドの内訳が異なった。例えば greedy の 95 token の場合は `0,0,4,6,3` と `0,0,4,7,2` で、合計はいずれも 13。Graph と head を切り替えても、同じ cache 条件内では各ラウンドまで一致した。prefill の feature 幅を 16 に固定した追試でも、95 token の反復入力に続く 2 要求では受理ラウンドの内訳に差が残った。

97 token・温度 0.7 では、実際に生成した ID で prefix を延長する試験も 8 設定すべてで行った。一部受理の連続と 15/15 の一括受理の両方で 112 token を復元し、再計算時と生成 ID・ラウンドごとの受理数が一致した。

単発 CLI では Graph と head の 4 設定を両温度で切り替え、各温度の生成 ID・round 数・受理数が一致した。95 token の入力で greedy は 7 round・8 draft 受理、温度 0.7 は 10 round・13 draft 受理。14 token の chat 入力は HTTP の tree budget 0・16 と単発 CLI の 64 生成 ID が一致した。一方、39 token の入力では、単発 CLI と server で初めて異なる ID が 6 個目の 6375 と 4670 だった。drafter の prefill 幅だけを 16 にそろえても差は残った。block size 16 の単発 CLI に server と同じ feature 容量と最終ページ手前の target prefill 分割を適用した診断版では、39 token の入力も全 64 ID が一致した。この分割を単発 CLI に採用した。最終ビルドの HTTP 回帰テストでは、この 39 token の入力で budget 0・16 の両方が単発 CLI の 64 生成 ID と一致した。

生成上限を 1・2・3・15・16・17・18・19 にした反復入力では、両温度とも要求した数だけ生成した。上限 18 では 15 draft の全受理後に通常の 1 行 decode、上限 19 では全受理後に 1 draft の短縮 verify を確認した。

15 draft の再評価では 1,940 個の語彙ブロックに対して行ごとの候補を OR する。同点がなければ和集合は最大 15 × 128 = 1,920 ブロック。同点で全ブロックを選んでもフラグは 1,940 個で収まり、各行の最大値も既存の top-k 作業領域の 4,096 要素に収まる。GPU の回帰テストでは、各行の候補ブロックの和集合を CPU で求め、元の head で再計算した logit とあわせて照合した。

required CTest は 224 件すべて PASS。別途実行した関連テストで単一 GPU のため skip したのは、2 GPU 専用の optional 5 件だった。オプション検証では server が 15 draft・tree budget 0/16/32 と prefill 幅 16 を転送し、上限外の draft 数、足りない tree budget、15 draft での auto を拒否した。

## DSpine-b16 の調査メモ

- `DFlashDraftModel` の config と重みの契約を追加し、target tap は 1・16・31・46・61 層から取得する。
- 5 層の sliding attention を causal な draft block として実行し、前の位置の予測特徴を次の位置へ渡す。
- `selected_prefix_bridge` の 248320 × 1024 codebook、rank 512 の gate、層ごとの read/write と独立した feature write を実装する。
- `serial_table` の第 2 パスと最終 write の `exact_sender`（top-k 16）を参照実装と照合する。DSpine 固有の実装・性能評価は別件とする。

## 変更ファイル

- [docs/developer/dflash2.md](../../../docs/developer/dflash2.md)
- [docs/developer/qwen35.md](../../../docs/developer/qwen35.md)
- [docs/developer/sampling.md](../../../docs/developer/sampling.md)
- [docs/user/compute.md](../../../docs/user/compute.md)
- [docs/user/env_vars.md](../../../docs/user/env_vars.md)
- [docs/user/server.md](../../../docs/user/server.md)
- [include/shelf/kernels/dflash2/coarse_head.h](../../../include/shelf/kernels/dflash2/coarse_head.h)
- [include/shelf/runtime/dflash2_spec_decoder.h](../../../include/shelf/runtime/dflash2_spec_decoder.h)
- [include/shelf/runtime/executor.h](../../../include/shelf/runtime/executor.h)
- [src/apps/compute/compute_cli.cpp](../../../src/apps/compute/compute_cli.cpp)
- [src/apps/compute/main.hip](../../../src/apps/compute/main.hip)
- [src/apps/compute/single_request.hip](../../../src/apps/compute/single_request.hip)
- [src/apps/server/shelf_server.py](../../../src/apps/server/shelf_server.py)
- [src/shelf/inference/execution/optimized_dispatch.hip](../../../src/shelf/inference/execution/optimized_dispatch.hip)
- [src/shelf/kernels/dflash2/coarse_head.hip](../../../src/shelf/kernels/dflash2/coarse_head.hip)
- [src/shelf/kernels/optimized/linear/exl3_family/mul24/projection.hip](../../../src/shelf/kernels/optimized/linear/exl3_family/mul24/projection.hip)
- [src/shelf/runtime/continuous_batcher.cpp](../../../src/shelf/runtime/continuous_batcher.cpp)
- [src/shelf/runtime/dflash2/dflash2_footprint.cpp](../../../src/shelf/runtime/dflash2/dflash2_footprint.cpp)
- [src/shelf/runtime/dflash2/executor.hip](../../../src/shelf/runtime/dflash2/executor.hip)
- [src/shelf/runtime/executor.hip](../../../src/shelf/runtime/executor.hip)
- [tests/kernels/linear/test_plamo_exl3_projection.hip](../../../tests/kernels/linear/test_plamo_exl3_projection.hip)
- [tests/server/test_compute_dflash2_unaligned.py](../../../tests/server/test_compute_dflash2_unaligned.py)
- [tests/server/test_server_dflash2_tree.py](../../../tests/server/test_server_dflash2_tree.py)
- [tests/unit/test_dflash2_exl3_head.hip](../../../tests/unit/test_dflash2_exl3_head.hip)
- [tests/unit/test_dflash2_footprint.cpp](../../../tests/unit/test_dflash2_footprint.cpp)
- 本測定記録
