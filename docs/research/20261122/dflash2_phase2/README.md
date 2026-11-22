# DFlash2 の Graph 再生とページ境界の検証

[Issue 38](https://github.com/kusuriya-labs/shelf/issues/38) の項目 4 と 2 を、Phase 1 の `6e5d5dac` と比較した。R9700 1 台、ROCm 10.2、gfx1201 の Release ビルドで、GPU の UUID を固定し、全測定で `/tmp/r9700.lock` を取得した。

## 測定条件

target は Qwen3.8-27B の EXL3 mul24（3.05 bpw、4 bpw）、速度測定の drafter は DFlash2 の EXL3 4 bpw mul24。BF16 KV、単一要求、drafts 7、tree budget 0、最大系列長 32768、prefill chunk 4096、prefix checkpoint 4 枠とした。生成上限は 64 token、greedy または temperature 0.7、top-p 0.95、top-k 20、seed 42。JSON と code の短い比較には Phase 1 と同じ BetterBench の `json-extract-short`（65 token）と `code-gen-medium`（58 token）の入力 ID を使った。入力 ID と生成上限は [inputs.json](inputs.json) に収録した。各プロセスで両問・両温度を 1 回ずつ暖機し、同じ順で再度測定した。prefix 有効時の戦略は radix。

基準・候補を AB、BA、AB の順に同じ GPU で比較した。各 run で round の算術平均を取り、表には 3 run の中央値を掲載した。draft 送信は `DFPHASE draft_submit` の host 時間、draft GPU と verify GPU は HIP event、round は Phase 1 と同じ `DFCOST wall`。round は verify 後の受理判定と状態更新の投入を含む。decode tok/s は stdio の要求終了ログの値。全 round の表では、残り token 数による短縮と単発 target decode も含めた。単発 target decode では verify 行数と両区間の GPU event 値は 0 になる。7/7 率の分母は全 round。

`runs.csv` の ID 照合値は生成 ID をカンマ区切りの十進表記にした SHA-256 の先頭 16 桁。完全一致の判定には元の整数列を比較した。

## 実装と検証

- **4A**　単一要求の target verify は 8 行・10 行それぞれに greedy 用と sampled 用の Graph を保持する。radix top-k の起動引数になる k を再利用条件に含め、seed と sample index は各 round の request descriptor から読む。tree の greedy 制約は従来どおり。
- **4B**　全長の単一提案では、既存のバッチ用 RoPE・ring attention・noise・selector に 1 要求分の device descriptor を渡す。実行ごとに更新する ring のアドレスと位置、anchor、出力先を capture の外に置き、head の射影は 7 draft 行に保った。直前の target tap の append と提案を同じ stream に順序付ける。
- **2**　追加の QSA / PLE state がない単一 GPU の rolling capture では、受理した GDN compact log を最新のページ境界の前後で分けて反映する。境界行の conv 履歴を戻し、recurrent state とともに既存の shadow slot へ保存する。tree は受理経路上の行を使う。radix の物理 ring にページ端数を保持する余裕を設けた。16 token ページ、窓 2048 の対象モデルでは最大 15 行、K/V 合計 300 KiB の追加で、attention の可視長は 2048 のまま。

### 出力一致

EXL3 と FP8 の各 drafter について、同一プロセスで greedy と sampled を切り替えた。sampled では top-p 0.95、seed 42 / 36 / 73 / 11、top-k 20 / 4 / 0、min-p 0 / 0.04 / 0.02 を使った。両 Graph の有効・無効で、anchor 位置、sample index、全候補 ID、round ごとの提案数・受理数、全生成 ID を比較した。

| drafter | 要求数 | 生成 token | round | 一部受理（0 < 受理数 < 提案数） | 不一致 |
|---|---:|---:|---:|---:|---:|
| EXL3 4 bpw | 9 | 560 | 92 | 26 | 0 |
| FP8 | 9 | 560 | 90 | 24 | 0 |

target verify の Graph を有効に固定し、drafter の Graph だけを切り替えた比較でも、EXL3・FP8・BF16 の各 4 要求で候補、受理、生成列が一致した。block size 8 の実モデルで sampled verify の再生を確認したのは 8 行である。10 行の実モデル試験は greedy の tree verify で行った。

| 入力 token 数 | radix 初回の復元 | radix 2 回目の復元 | slot 初回の復元 | slot 2 回目の復元 |
|---:|---:|---:|---:|---:|
| 95 | 0 | 80 | 0 | 80 |
| 96 | 0 | 80 | 0 | 80 |
| 97 | 0 | 96 | 0 | 96 |

いずれも cache 無効の生成列と一致した。入力の末尾は少なくとも 1 token を prefill する。出力を含む prefix を再入力した試験では、chain と tree の両方で 96 token を復元した。別の入力では 2,168 token の確定状態から ring を 8 行巻き戻して 2,160 token の checkpoint を作った。そこに 1 token を足して復元し、続く 32 token の生成列と全 round の候補 ID が cache 無効と一致した。

## Graph 単独の寄与

3.05 bpw、prefix 無効。以下は 8 行を検証した round に限定した値である。4A は drafter を eager に固定して sampled verify の有無を比較した。4B は verify の Graph を有効に固定して drafter の Graph を切り替えた。

| 変更 | 入力 | 温度 | draft 送信 ms | draft GPU ms | verify GPU ms | round ms |
|---|---|---:|---:|---:|---:|---:|
| 4A | JSON | 0.7 | 0.621 → 0.451 | 4.94 → 4.96 | 34.28 → 30.48 | 40.35 → 36.61 |
| 4A | code | 0.7 | 0.545 → 0.443 | 4.95 → 4.99 | 34.59 → 30.80 | 40.65 → 36.94 |
| 4B | JSON | 0 | 0.421 → 0.024 | 4.97 → 4.84 | 30.27 → 30.31 | 36.35 → 36.24 |
| 4B | JSON | 0.7 | 0.417 → 0.027 | 4.97 → 4.84 | 30.65 → 30.75 | 36.72 → 36.69 |
| 4B | code | 0 | 0.489 → 0.022 | 5.02 → 4.88 | 30.63 → 30.56 | 36.81 → 36.56 |
| 4B | code | 0.7 | 0.422 → 0.027 | 5.01 → 4.88 | 31.03 → 31.01 | 37.13 → 37.01 |

JSON、temperature 0.7 の `rocprofv3 --kernel-trace` で、target の descriptor 準備から最終 sampling kernel までを数えた。暖機後の 7 round すべてで verify は 1,464 dispatch だった。Phase 1 の同じ範囲は 1,892 dispatch。新版では個別の 3-bit GEMM が 115 回、3-bit の grouped GEMM が 117 回だった。Graph 自体の node 数には転送も含まれる。

## Phase 1 と Phase 2 の比較

各セルは基準版 → 最終版。単位は時間が ms、生成速度が token/s。各 run の生成数・復元数・round 数・ID 照合値は [測定結果の CSV](runs.csv) に収録した。

### 3.05 bpw

| prefix | 入力 | 温度 | draft 送信 | draft GPU | verify GPU | round | verify 行/round | 受理数/round | 7/7 率 % | decode tok/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OFF | json | 0 | 0.42 → 0.03 | 4.99 → 4.85 | 30.46 → 30.38 | 36.52 → 36.31 | 8.00 → 8.00 | 5.88 → 5.88 | 75.00 → 75.00 | 190.80 → 192.00 |
| OFF | json | 0.7 | 0.41 → 0.02 | 4.95 → 4.84 | 34.56 → 30.83 | 40.61 → 36.77 | 8.00 → 8.00 | 5.88 → 5.88 | 75.00 → 75.00 | 171.70 → 189.50 |
| OFF | code | 0 | 0.42 → 0.08 | 4.99 → 4.87 | 30.86 → 30.90 | 36.98 → 36.90 | 7.50 → 7.50 | 5.30 → 5.30 | 60.00 → 60.00 | 172.50 → 172.90 |
| OFF | code | 0.7 | 0.38 → 0.12 | 4.95 → 4.87 | 34.68 → 31.57 | 40.75 → 37.58 | 7.50 → 7.50 | 4.25 → 4.25 | 41.67 → 41.67 | 130.50 → 141.50 |
| ON | json | 0 | 0.46 → 0.03 | 4.98 → 4.84 | 30.81 → 30.35 | 37.10 → 36.70 | 7.86 → 8.00 | 6.86 → 5.88 | 85.71 → 75.00 | 203.30 → 181.00 |
| ON | json | 0.7 | 0.38 → 0.03 | 4.96 → 4.85 | 34.49 → 30.81 | 40.74 → 37.15 | 7.86 → 8.00 | 6.86 → 5.88 | 85.71 → 75.00 | 186.20 → 179.00 |
| ON | code | 0 | 0.37 → 0.06 | 4.49 → 4.86 | 28.39 → 30.76 | 42.48 → 37.11 | 6.27 → 7.50 | 4.73 → 5.30 | 36.36 → 60.00 | 132.50 → 165.60 |
| ON | code | 0.7 | 0.38 → 0.11 | 4.50 → 4.86 | 31.36 → 31.43 | 44.11 → 37.74 | 6.08 → 7.50 | 3.85 → 4.25 | 23.08 → 41.67 | 108.60 → 136.60 |

### 4 bpw

| prefix | 入力 | 温度 | draft 送信 | draft GPU | verify GPU | round | verify 行/round | 受理数/round | 7/7 率 % | decode tok/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OFF | json | 0 | 0.43 → 0.03 | 4.96 → 4.81 | 33.87 → 33.78 | 39.90 → 39.66 | 8.00 → 8.00 | 6.29 → 6.29 | 71.43 → 71.43 | 185.40 → 186.50 |
| OFF | json | 0.7 | 0.43 → 0.03 | 4.94 → 4.81 | 50.98 → 34.22 | 57.00 → 40.14 | 8.00 → 8.00 | 6.29 → 6.29 | 71.43 → 71.43 | 129.90 → 184.20 |
| OFF | code | 0 | 0.43 → 0.10 | 4.98 → 4.86 | 35.82 → 35.83 | 41.94 → 41.81 | 7.89 → 7.89 | 6.00 → 6.00 | 55.56 → 55.56 | 169.00 → 169.50 |
| OFF | code | 0.7 | 0.47 → 0.08 | 4.90 → 4.83 | 50.65 → 36.30 | 56.62 → 42.24 | 7.44 → 7.44 | 6.00 → 6.00 | 66.67 → 66.67 | 125.30 → 167.80 |
| ON | json | 0 | 0.43 → 0.03 | 4.90 → 4.81 | 37.83 → 33.73 | 44.03 → 39.94 | 7.25 → 8.00 | 5.38 → 6.29 | 50.00 → 71.43 | 141.40 → 175.50 |
| ON | json | 0.7 | 0.51 → 0.03 | 4.90 → 4.80 | 51.10 → 34.14 | 57.29 → 40.34 | 7.25 → 8.00 | 5.38 → 6.29 | 50.00 → 71.43 | 109.70 → 173.90 |
| ON | code | 0 | 0.41 → 0.07 | 4.45 → 4.85 | 35.31 → 35.80 | 49.39 → 42.25 | 6.55 → 7.89 | 4.73 → 6.00 | 36.36 → 55.56 | 114.60 → 161.70 |
| ON | code | 0.7 | 0.42 → 0.08 | 4.49 → 4.81 | 47.06 → 36.17 | 59.83 → 42.57 | 6.31 → 7.44 | 3.85 → 6.00 | 23.08 → 66.67 | 80.60 → 160.50 |

基準版は cache ON の sampled code で、境界による行数短縮のために次 round の sample index が変わる。基準版の cache ON/OFF の生成 ID は、3.05 bpw で 57 個目、4 bpw で 15 個目から異なる。最終版では全 3 ペアで、各 target・入力・温度について、cache ON/OFF の生成 ID と round ごとの受理数が一致した。基準版 cache ON との差が残る sampled code の速度には、生成列の変化も含まれる。

3.05 bpw の短い JSON では、基準版の境界短縮によって 2 round 目が 6 draft となり、7 round で生成を終了した。最終版では cache OFF と同じ 8 round になった。verify を短縮しても、この入力の cache ON の decode tok/s は低下した。

## ページ境界変更の単独比較

3.05 bpw、prefix 有効、temperature 0.7。両版で target と drafter の Graph を有効にした。Issue の調査で使った thinking 有効の BetterBench 入力を、JSON・code 各 4 問、元の生成上限で測定した。各条件とも最初に短い JSON で暖機し、4 問の全 round を合算した。カテゴリの速度は生成 token 数の合計を decode 時間の合計で割った。表は AB/BA/AB の中央値。

| 入力 | verify 行/round | 受理数/round | 7/7 率 % | draft GPU ms | verify GPU ms | round ms | decode tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| JSON | 6.59 → 7.98 | 3.79 → 4.33 | 34.76 → 43.16 | 4.83 → 5.05 | 30.79 → 31.29 | 37.83 → 37.84 | 126.09 → 139.99 |
| code | 6.46 → 7.96 | 2.12 → 2.31 | 11.43 → 11.53 | 4.91 → 5.23 | 30.58 → 31.65 | 38.35 → 38.31 | 81.34 → 86.23 |

ページ境界までに短縮されていた verify に行を戻した結果、JSON の受理数が増え、カテゴリ全体で 11.0% 速くなった。code は 6.0% 速くなった。問題ごとの結果には増減がある。sampled の提案幅が変わるため、旧版と新版の生成列・終了位置も変わり得る。

### 長文の生成列と prefill の分割

同じ 8 問を続けて送ると、全問の `restored_tokens` が 0 でも、radix は既存 tree と共通する先頭 48 token を prefill の分割点に選ぶ。この位置の保存は下限 64 token により見送られる。基準版の radix と cache 無効の生成 ID は 8 問すべてで異なった。最終版では 2 問が一致し、6 問が異なった。cache 無効同士で比較すると、基準版と最終版の全 8 問が一致した。

最終版の slot と cache 無効は全 8 問で一致した。radix を維持して共通 prefix の分岐位置での prefill 分割だけを抑えた診断ビルドでも、全 8 問の生成 ID が cache 無効と一致した。逆に radix の prefill はそのままで、decode 中の境界 snapshot と GDN の分割 replay を省いた診断ビルドは、最終版の radix と全 8 問で一致した。Phase 2 は既存の prefill 分割を維持した。カテゴリの速度比較には sampled の生成内容と終了位置の違いを含む。

## 次段階の測定点

3.05 bpw・JSON の sampled trace では、drafter の全語彙 head に使う 6-bit GEMM が、暖機後の 7 round で中央値 1.593 ms（1.591〜1.595 ms）を占めた。項目 5 はこの区間と top-k / selector を切り分け、順位、候補 ID、tree 用の確率正規化を同時に比較できる。

項目 3 の block size 16 は別の checkpoint での検証が必要になる。現在の単一提案 Graph は checkpoint の全 draft 行を使う。15 draft では 16 行の target verify、GDN history、tree 予算、末尾の短縮経路も対象になる。現時点で実モデルの verify Graph の cache は 8 / 10 行用である。

## 回帰試験の条件

CTest の required 224 件は全件 PASS。EXL3 / DFlash2 / Qwen35 / prefix cache の関連選択は 69 PASS、2 GPU 用 optional 5 件が skip。単一 GPU で必要な fixture は Git LFS ポインタの SHA-256 と照合した実体を使った。Flash-Next の形状を読む optional benchmark には、CTest の参照先からローカルの対応 checkpoint を指定した。

- `test_compute_dflash2_unaligned.py` は prefix OFF / ON とも、3.05 bpw では生成 116 個目が非投機 13990、投機 264。4 bpw では 129 個目が非投機 16821、投機 8940。基準版と最終版でそれぞれ 256 個の投機出力および非投機出力が一致した。新たに検証した 95 / 96 / 97 token の復元位置と ID の比較は各 target で通過した。
- `test_compute_dflash2_prefix_long.py` は 2048 token ごとの checkpoint を期待する。既定の保存間隔 16384 の実行では基準版・最終版とも復元 0 で、生成列は一致した。3.05 bpw で `SHELF_TEST_MAX_NUM_BATCHED_TOKENS=2048` と `SHELF_TEST_PREFIX_CACHE_SAVE_INTERVAL=2048` を指定すると、復元位置、cache 無効の DFlash2、および非投機 greedy との比較を含め 8 項目すべてが通過した。
- `test_server_cli.py` の prefix 関連の転送設定はすべて PASS。既定の prefill chunk を省略するという期待に対して、基準版・最終版とも引数に 4096 を指定した。

- `test_server_dflash2_switch.py` は tree budget を指定せず auto の trace を待つ。基準版と最終版の server の既定値は 0 で、どちらも trace がないまま待機期限に達した。auto を明示した追加試験では、基準版は通過した。最終版では 2 試行とも 384-token の先行要求が通常 decode より先に完了し、要求の終了順に対するアサーションに達した。先行要求を 768 token にして両要求が重なる時間を確保すると、tree → chain → tree と通常 decode の混在を確認できた。
- `test_server_dflash2_tree.py` は budget 10 で HTTP と単発 tree の生成 6 個目が異なる。基準版・最終版ともに HTTP は 4670、単発は 6375。それぞれの経路内の 64 ID は基準版と最終版で完全一致した。単発の `test_compute_dflash2_tree.py` は 3 prompt・各 256 token について全 budget を通過した。このテストは初回不一致の logit margin が 0.1 以下なら許容する。
- stdio の DFlash2 基本統合（99 項目）、HTTP の DFlash2 基本統合（21 項目）、stop・cancel と prefix の再利用（32 項目）は全件 PASS。Octave KV、prefix protocol、grammar protocol、HTTP の DFlash2 grammar、thinking budget、reasoning の parser / request / HTTP / stream / tools も PASS。


## 変更ファイル

- **ビルドと起動**　[cmake/targets.cmake](../../../../cmake/targets.cmake)、[src/apps/compute/main.hip](../../../../src/apps/compute/main.hip)
- **ヘッダ**　[include/shelf/kernels/dflash2/candidate_selector.h](../../../../include/shelf/kernels/dflash2/candidate_selector.h)、[include/shelf/runtime/dflash2/executor.h](../../../../include/shelf/runtime/dflash2/executor.h)、[include/shelf/runtime/dflash2_config.h](../../../../include/shelf/runtime/dflash2_config.h)、[include/shelf/runtime/dflash2_spec_decoder.h](../../../../include/shelf/runtime/dflash2_spec_decoder.h)、[include/shelf/runtime/executor.h](../../../../include/shelf/runtime/executor.h)
- **実装**　[src/shelf/kernels/dflash2/candidate_selector.hip](../../../../src/shelf/kernels/dflash2/candidate_selector.hip)、[src/shelf/runtime/continuous_batcher.cpp](../../../../src/shelf/runtime/continuous_batcher.cpp)、[src/shelf/runtime/dflash2/context_state.hip](../../../../src/shelf/runtime/dflash2/context_state.hip)、[src/shelf/runtime/dflash2/executor.hip](../../../../src/shelf/runtime/dflash2/executor.hip)、[src/shelf/runtime/dflash2_spec_decoder.cpp](../../../../src/shelf/runtime/dflash2_spec_decoder.cpp)、[src/shelf/runtime/executor.hip](../../../../src/shelf/runtime/executor.hip)
- **テスト**　[tests/server/test_compute_dflash2_unaligned.py](../../../../tests/server/test_compute_dflash2_unaligned.py)、[tests/unit/test_dflash2_footprint.cpp](../../../../tests/unit/test_dflash2_footprint.cpp)、[tests/unit/test_dflash2_kv_ring.hip](../../../../tests/unit/test_dflash2_kv_ring.hip)
- **開発者向け文書**　[docs/developer/dflash2.md](../../../../docs/developer/dflash2.md)、[docs/developer/prefix_cache.md](../../../../docs/developer/prefix_cache.md)、[docs/developer/sampling.md](../../../../docs/developer/sampling.md)
- **研究記録**　[docs/research/20261122/dflash2_phase2/README.md](../../../../docs/research/20261122/dflash2_phase2/README.md)、[docs/research/20261122/dflash2_phase2/inputs.json](../../../../docs/research/20261122/dflash2_phase2/inputs.json)、[docs/research/20261122/dflash2_phase2/runs.csv](../../../../docs/research/20261122/dflash2_phase2/runs.csv)
