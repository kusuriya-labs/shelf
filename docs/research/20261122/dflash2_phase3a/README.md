# DFlash2 の全語彙提案ヘッド

Issue #38 の Phase 3A、項目 5 の採否記録。基準は `438879e2` の 6-bit 全語彙ヘッド。起動時に target の重みから作る INT3 ヘッドと、選択した語彙ブロックの 6-bit 再評価を採用した。target verify は従来の重みと手順を使う。[Issue #38](https://github.com/kusuriya-labs/shelf/issues/38)

## 測定条件

R9700（gfx1201、ROCm 10.2）1 台。全 GPU 実行で `ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55 flock /tmp/r9700.lock` を使った。主測定は Qwen3.8-27B EXL3 3.05 bpw mul24 と DFlash2 EXL3 4 bpw mul24。両者の HIP Graph を有効にし、BF16 KV、drafts 7、tree budget 0、単一要求、最大系列長 32768、prefill chunk 4096、prefix checkpoint 4 枠とした。prefix 有効時は radix。生成上限 128 token、greedy または温度 0.7、top-k 20、top-p 0.95、min-p 0、seed 42。

各プロセスは 48 token 上限の JSON で暖機した後、JSON 4 問、code 4 問、prose 1 問、順序の数え上げ 1 問を解いた。各問で greedy、温度 0.7 の順に実行した。prefix 有効時には直前の要求の prefix を再利用できる。比較する両側で要求の順序をそろえた。JSON と code の入力 ID は Phase 2 の `category`、prose は既存の固定入力を使った。[Phase 2 の入力](../dflash2_phase2/inputs.json)・[prose の入力](../../20261113/exl3_general/prompt_prose.ids)・[順序の数え上げの入力](reasoning.json)

A は `SHELF_DFLASH2_COARSE_HEAD=0`、B は `1`。同じ実行ファイルで各 prefix 設定を AB、BA、AB の順に測定した。各 run のカテゴリ内で全 round の `DFCOST draft_gpu`、`DFCOST wall`、受理数を平均した。decode tok/s は生成数の合計を stdio 終了ログの decode 時間の合計で割った。表は各側の 3 run 中央値。ID 一致は各対の整数列全体を比較した件数。[要求別の測定値](runs.csv)

stdio には `--dflash2-drafts 7 --dflash2-tree-budget 0 --kv-cache-dtype bf16 --max-seq-len 32768 --max-num-batched-tokens 4096 --max-concurrent-requests 1 --prefill-chunk 4096 --prefix-cache-checkpoints 4 --device 0` を指定した。`SHELF_DFLASH2_SERVE_STATS=1`、`SHELF_DFLASH2_COST_TRACE=1`、`SHELF_DFLASH2_BATCH_PHASE_TRACE=1` を両側で有効にした。各要求の `DFPHASE_REQUEST` と、その直前の `DFCOST` を対応付けた。decode 時間には `[req …] prompt=… decode_ms=…` の終了値を使用した。

## 採用した経路

EXL3 の 6-bit mul24 trellis を GPU で一度復号し、3 bit の 8 段階に再量子化する。復元値は元の整数格子上で 10.85 × {−7, −5, −3, −1, 1, 3, 5, 7}。入力回転、128 列ごとの activation scale、出力回転と語彙ごとの svh は元の重みと共有する。各 WMMA lane が使う重み 16 個を 48 bit に詰めた。起動時の追加 arena 使用量は 476,774,400 バイト（454.688 MiB）。arena 確保から構築 stream の同期までの 6 起動の中央値は 6.159 ms、範囲は 5.761〜7.097 ms。

各 draft 行で 248,320 語を評価し、出力回転の単位である 128 語ごとに最大 logit を求める。1,940 ブロックの最大値を GPU 上で順位付けし、上位 128 ブロック（境界の同点を含む）を選ぶ。全 draft 行の和集合を一度だけ元の 6-bit 重みで評価する。selector の上位 16 語には再評価値だけが入る。tree の正規化は選択ブロックの厳密値と未選択ブロックの近似値から全語彙について求める。既存の top-k scratch と EXL3 accumulator を再利用する。[対象形状とフォールバック条件](../../../developer/dflash2.md)

## カーネル計測

暖機後の短い JSON、温度 0.7、7 draft 行の 8 round。rocprofv3 の kernel trace で実行時間を測った。表は ms。合計は各 round で足してから中央値を取った。

| 区間 | 全語彙 6-bit | INT3 と再評価 |
|---|---:|---:|
| 入力回転・activation 量子化 | 0.002 | 0.002 |
| 全語彙 GEMM | 1.592 | 0.804 |
| 全語彙の出力回転 | 0.018 | 0.016 |
| ブロックの順位付けと選択 | — | 0.025 |
| 6-bit 再評価 GEMM | — | 0.348 |
| 再評価出力の回転・順位用マスク | — | 0.011 |
| head 合計 | 1.611 | 1.206 |
| top-k | 0.097 | 0.093 |
| selector | 0.027 | 0.027 |
| 全区間 | 1.735 | 1.327 |

Phase 2 の 1.593 ms は基準側の全語彙 6-bit GEMM に相当する。HIP Graph trace では、暖機を含めた drafter Graph の dispatch 数が 195 → 199、replay は各 12 回。coarse GEMM から再評価、top-k、selector まで同じ graph_exec_id で実行した。末尾で行数を短縮した提案は eager で実行した。

## 方式の比較

同じ実モデルの hidden を完全な 6-bit head と各候補で評価し、上位 16 ID の重なりを行ごとに数えた。診断では drafter Graph を無効にし、次の提案には基準版の logit を使って比較位置をそろえた。全体からブロックを選ぶ方式は JSON・code・prose と日本語の固定課題 1,024 token を含む。

| ビット幅 | 全体から選ぶブロック数/行 | top-16 一致数 / 比較数 | recall | top-1 一致数 / 行数 |
|---|---:|---:|---:|---:|
| INT2 | 0 | 25,496 / 40,624 | 62.761% | 1,571 / 2,539 |
| INT2 | 32 | 38,937 / 40,624 | 95.847% | 2,538 / 2,539 |
| INT2 | 64 | 40,140 / 40,624 | 98.809% | 2,539 / 2,539 |
| INT2 | 128 | 40,510 / 40,624 | 99.719% | 2,539 / 2,539 |
| INT2 | 256 | 40,611 / 40,624 | 99.968% | 2,539 / 2,539 |
| INT3 | 0 | 31,114 / 40,624 | 76.590% | 1,810 / 2,539 |
| INT3 | 32 | 40,246 / 40,624 | 99.070% | 2,539 / 2,539 |
| INT3 | 64 | 40,574 / 40,624 | 99.877% | 2,539 / 2,539 |
| INT3 | 128 | 40,622 / 40,624 | 99.995% | 2,539 / 2,539 |
| INT3 | 256 | 40,624 / 40,624 | 100.000% | 2,539 / 2,539 |

**3A-1（棄却）**　再評価を省くと INT2・INT3 ともに提案が変わった。64 token 上限の sampled code で両方とも受理長と速度が低下した。

**3A-2（棄却）**　語彙を 61 組に振り分け、各組から上位 2 ブロックを選ぶ INT2 では、128 token 上限の 120 対が生成 ID・受理数とも一致し、16 条件の decode が 2.05〜2.31% 改善した。しかし日本語の固定課題 `code_edit-ja-short_2k` の 1,024 token 比較で、9 round 目の受理数が 3 → 2、312 個目の生成 ID が変わり、77.8 → 74.8 tok/s となった。組ごとの上限を 4 に広げても、語彙全体から 128 または 256 ブロックを選んでも、同じ位置で生成 ID が変わった。

**3A-3（採用）**　INT3 で全体から候補を選ぶと、固定課題の 1,024 ID と各 round の受理数が基準版に一致した。64 → 128 ブロックで参照 top-16 の取りこぼしが 50 個から 2 個に減り、再評価した一致 ID の最大 logit 差は小数 6 桁で 0 だった。候補幅の選定では長文の出力一致、top-16 recall、E2E の順に確認した。下表は同じ固定課題の候補探索時の値で、各 1 回。

| 方式 | ブロック選択 | 生成 ID 一致 | round 数 | decode tok/s |
|---|---|---:|---:|---:|
| 全語彙 6-bit | 全体 | 一致 | 319 | 77.8 |
| INT2 | 各組 2 | 不一致 | 338 | 74.8 |
| INT2 | 各組 4 | 不一致 | 338 | 74.3 |
| INT2 | 全体から 128 | 不一致 | 338 | 74.9 |
| INT2 | 全体から 256 | 不一致 | 338 | 74.1 |
| INT3 | 各組 2 | 一致 | 319 | 78.4 |
| INT3 | 全体から 64 | 一致 | 319 | 78.8 |
| INT3 | 全体から 128 | 一致 | 319 | 78.3 |
| INT3 | 全体から 256 | 一致 | 319 | 79.5 |

同じ固定入力について 128 と 256 を同一バイナリで AB、BA、AB の順に直接比較した。各側の 3 run 中央値を示す。

| ブロック数 | 平均受理数 | draft GPU ms | verify GPU ms | round ms | decode tok/s | 基準版との ID 一致 |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 2.207 | 6.408 | 33.207 | 40.791 | 78.600 | 3/3 |
| 256 | 2.207 | 6.576 | 33.161 | 40.913 | 78.400 | 3/3 |

256 ブロックは診断の top-16 を 2 個多く回収した。上の長文比較ではどちらも同じ出力 ID と受理数を得た。追加の再評価を抑え、128 ブロックを採用した。

## 採用版の E2E

| prefix | 入力 | 温度 | draft GPU ms | 平均受理数 | round ms | decode tok/s | 速度差 | ID 一致 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| OFF | json | 0 | 4.93 → 4.47 | 4.52 → 4.52 | 37.57 → 37.04 | 147.87 → 149.98 | +1.43% | 12/12 |
| OFF | json | 0.7 | 4.91 → 4.45 | 4.18 → 4.18 | 38.49 → 37.96 | 135.53 → 137.41 | +1.39% | 12/12 |
| OFF | code | 0 | 4.93 → 4.46 | 4.52 → 4.52 | 37.88 → 37.34 | 146.65 → 148.78 | +1.45% | 12/12 |
| OFF | code | 0.7 | 4.90 → 4.41 | 2.85 → 2.85 | 37.96 → 37.42 | 102.05 → 103.53 | +1.45% | 12/12 |
| OFF | prose | 0 | 4.91 → 4.43 | 1.89 → 1.89 | 37.12 → 36.57 | 78.27 → 79.45 | +1.51% | 3/3 |
| OFF | prose | 0.7 | 4.90 → 4.42 | 1.70 → 1.70 | 37.56 → 36.99 | 72.41 → 73.53 | +1.55% | 3/3 |
| OFF | reasoning | 0 | 4.94 → 4.46 | 4.08 → 4.08 | 37.34 → 36.76 | 136.87 → 139.02 | +1.57% | 3/3 |
| OFF | reasoning | 0.7 | 4.91 → 4.43 | 3.70 → 3.70 | 37.83 → 37.22 | 125.11 → 127.15 | +1.63% | 3/3 |
| ON | json | 0 | 4.86 → 4.40 | 4.64 → 4.64 | 38.49 → 37.98 | 145.12 → 147.02 | +1.31% | 12/12 |
| ON | json | 0.7 | 4.90 → 4.44 | 4.13 → 4.13 | 38.69 → 38.16 | 131.39 → 133.18 | +1.36% | 12/12 |
| ON | code | 0 | 4.86 → 4.40 | 4.64 → 4.64 | 37.51 → 36.99 | 148.89 → 150.94 | +1.38% | 12/12 |
| ON | code | 0.7 | 4.96 → 4.47 | 2.65 → 2.65 | 37.87 → 37.34 | 96.07 → 97.43 | +1.42% | 12/12 |
| ON | prose | 0 | 4.90 → 4.43 | 1.89 → 1.89 | 37.26 → 36.72 | 77.29 → 78.43 | +1.47% | 3/3 |
| ON | prose | 0.7 | 4.89 → 4.42 | 1.70 → 1.70 | 37.66 → 37.08 | 71.64 → 72.75 | +1.55% | 3/3 |
| ON | reasoning | 0 | 4.93 → 4.45 | 4.08 → 4.08 | 37.60 → 37.05 | 133.86 → 135.84 | +1.48% | 3/3 |
| ON | reasoning | 0.7 | 4.90 → 4.42 | 3.70 → 3.70 | 37.96 → 37.46 | 122.98 → 124.59 | +1.30% | 3/3 |

16 条件の 3 ペア中央値で速度差は +1.30〜+1.63%、個々のカテゴリのペアでは +1.13〜+2.43%。実行ファイルの SHA-256 は `eb89588f6fcb1dabb1313c21b6551d198116b35fe8698fb6c6e31c6cec2fd8af`。各側 15,360 token のうち、生成 ID 列は 120/120 対、round ごとの受理数の列は 120/120 対で一致した。

## 互換性・長文と引き継ぎ

各条件 1 ペア。4 bpw target と FP8 drafter は JSON・code・prose の greedy と温度 0.7、prefix ON。各列は全 round または全 decode 時間で集約した。

| 条件 | 生成数 | draft GPU ms | 平均受理数 | decode tok/s | ID 一致 |
|---|---:|---:|---:|---:|---:|
| 4 bpw target と EXL3 drafter | 384 | 4.790 → 4.320 | 2.818 → 2.818 | 87.76 → 88.94 | 6/6 |
| 3.05 bpw target と FP8 drafter | 384 | 5.708 → 5.218 | 3.395 → 3.395 | 112.42 → 114.10 | 6/6 |
| code_edit-ja-short_2k、greedy | 1024 | 6.835 → 6.290 | 2.207 → 2.207 | 78.05 → 79.95 | 1/1 |
| code、greedy、EOS まで | 933 | 5.372 → 4.879 | 4.548 → 4.548 | 144.18 → 147.39 | 1/1 |
| design_debug-ja-short_2k、greedy | 1024 | 6.905 → 6.392 | 2.157 → 2.157 | 75.99 → 77.47 | 1/1 |
| tech_prose-en-short_2k、greedy | 1024 | 6.900 → 6.400 | 2.693 → 2.693 | 89.45 → 90.87 | 1/1 |
| tree budget 10、prose、greedy | 96 | 4.657 → 4.169 | 2.167 → 2.167 | 59.96 → 60.00 | 1/1 |


固定 arena 15 GiB・最大系列長 32768 では、元の全語彙ヘッドで生成を完了した。tree budget 10 では CPU の全語彙正規化で再計算した木と GPU の木を比較した。

CTest の required 224 件が PASS。関連選択は 69 PASS、単一 GPU での 2 GPU optional は 5 件 skip。CPU 参照による INT3 射影の照合、再評価した行の元の 6-bit logit との一致、未選択行の正規化用 logit と順位用マスクの検査を含む。

Phase 3B の block size 16 では、15 draft 行は 1 枚の 16 行 WMMA に収まる。行数に応じて再評価するブロックの和集合が変わるため、15 行の coarse / fine の時間、受理長、16 行 target verify Graph、GDN の履歴とページ境界・生成末尾での行数短縮を改めて測定する。今回実モデルで確認したのは block size 8。

## 変更ファイル

| 対象 | ファイル |
|---|---|
| HIP カーネルとビルド | `include/shelf/kernels/dflash2/coarse_head.h`、`src/shelf/kernels/dflash2/coarse_head.hip`、`cmake/targets.cmake` |
| DFlash2 executor | `include/shelf/runtime/dflash2/executor.h`、`src/shelf/runtime/dflash2/executor.hip` |
| メモリ見積もり | `include/shelf/runtime/dflash2/dflash2_footprint.h`、`src/shelf/runtime/dflash2/dflash2_footprint.cpp` |
| 起動経路 | `src/apps/compute/compute_runtime.h`、`src/apps/compute/compute_runtime.hip`、`src/apps/compute/single_request.hip`、`src/apps/compute/stdio_server.hip` |
| 回帰テスト | `tests/unit/test_dflash2_exl3_head.hip`、`tests/unit/test_dflash2_footprint.cpp` |
| 文書と測定記録 | `docs/developer/dflash2.md`、本ディレクトリの `README.md`、`runs.csv`、`reasoning.json` |
