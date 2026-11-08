# EXL3-INT8 decode の帯域と graph

## 測定条件

- 日付: 2026-11-08〜09（Asia/Tokyo）。GPU: Radeon AI PRO R9700、gfx1201、UUID `GPU-a8e9ddefa2d60f55`。
- 基準 revision: `8dcb8a6`。対象は Qwen3.8-27B の EXL3-INT8 4 bpw mul24、lm_head 6 bpw。
- 各 GPU プロセスは `flock /tmp/r9700.lock` で排他。別の prefill 測定も同じ GPU を利用している。
- 推論は BF16 KV、prompt `/tmp/prompt_{code,prose,ja}.ids`、最大生成 256、最大系列長 1024、temperature 0。DFlash2 は BF16 drafter、draft 7。MTP は固定 draft 4。
- 以下の `GB/s` は 10 進数。kernel の実効帯域は trellis バイト数 / 時間であり、実 DRAM 転送量を直接測った値ではない。

## 帯域基準

512 MiB の配列を `uint4` で読み、各 thread の合計を別配列へ書く kernel。GPU event、20 launch の平均を 1 回とし、2 回の preroll 後に 5 回取得。読み出し p50 は **623.82 GB/s**（0.8599〜0.8614 ms）、device-to-device copy の片方向 byte 数 / 時間は **285.88 GB/s**（1.8764〜1.8788 ms）。copy は読み書き双方の転送を含む。公称 640 GB/s ではなく、この read-only 実測を基準にする。再測定は `shelf-bench gpu-memory --stream-read --warmup 2 --iterations 5 --iters 20`。

## モデル形状

checkpoint の safetensors header の trellis は `[K/16,N/16,16*bits]`、要素型 I16。層数は target のみを数える。

| 対象 | K | N | bits | 回数 |
|---|---:|---:|---:|---:|
| GDN in_proj_qkv | 5120 | 10240 | 4 | 48 |
| GDN in_proj_z | 5120 | 6144 | 4 | 48 |
| GDN out_proj / attention o | 6144 | 5120 | 4 | 64 |
| attention q | 5120 | 12288 | 4 | 16 |
| attention k / v | 5120 | 1024 | 4 | 各 16 |
| MLP gate / up | 5120 | 17408 | 4 | 各 64 |
| MLP down | 17408 | 5120 | 4 | 64 |
| lm_head | 5120 | 248320 | 6 | 1 |

GDN in_proj_a / in_proj_b は F16 格納の `[48,5120]` で、各 48 層。EXL3 の表から除く。

## 非投機の内訳

`SHELF_QWEN35_PERF_TIMING=1 SHELF_QWEN35_PERF_STATS=1 shelf-bench tg --mode forward --context 128 --tokens 4 --warmup 1 --perf-stats-output <path>`。4 token 分の GPU event 合計 / 4。event の挿入自体が実行を変えるため、通常実行と同一の時間とはみなさない。

| dispatch | 論理回数/token | event ms/token |
|---|---:|---:|
| EXL3-INT8 linear | 401 | 33.709 |
| BF16 linear | 96 | 1.318 |
| RMSNorm | 209 | 2.273 |
| GDN recurrence | 48 | 1.115 |
| GDN causal convolution | 48 | 0.479 |
| paged attention | 16 | 0.280 |
| SiLU / SwiGLU / MUL | 224 | 2.017 |
| residual add | 128 | 1.131 |
| L2 normalize | 96 | 0.888 |

全 dispatch の event 区間は 44.438 ms/token、計測を含む forward 全体の GPU event は 51.796 ms/token。logical 1397、physical 2199 launch/token。残差には計測や dispatch 間の時間が含まれ、GPU が launch を待っていた時間の直接測定ではない。

## 基準 E2E

code の各 1 回。DFlash2 の GPU verify は 2130.21 / 46、round は 2600.88 / 46。MTP verify と round はそれぞれ 2864.47 / 68、3530.06 / 68。

| 経路 | tok/s | verify ms/round | draft ms/round | round ms |
|---|---:|---:|---:|---:|
| 非投機 | 26.71 | — | — | — |
| MTP 固定 4 | 72.24 | 42.12 | 9.76 | 51.91 |
| DFlash2 draft 7 | 98.00 | 46.31 GPU | 8.34 GPU | 56.54 |

3 経路の `GENERATED_IDS` は一致した。条件の異なる event と host wall の区間は混ぜて加算しない。

## 実装の観察

GEMM の 4 bit、MT=1 は 4 wave / block、各 wave が 16 出力を担当する。128 要素の K group ごとに trellis を復号して 8 回の int8 WMMA を行う。入力の回転・量子化、GEMM、出力回転は別々の 3 launch。GEMM 自体の LDS は 0 B。ビルド済み code object の metadata は 4 bit で VGPR 90、SGPR 30、private 0 B、spill 0、6 bit で VGPR 115、SGPR 34、private 0 B、spill 0 だった。HIP occupancy API の理論上限は 4 bit で 16 block / WGP（64 wave）、6 bit で 11 block / WGP（44 wave）。これは実行中の occupancy ではない。

同 code object の静的逆アセンブルでは 4 bit / MT=1 の unrolled 部分に `v_mul_u32_u24` と `v_dot4_u32_u8` が各 64、`v_wmma_i32_16x16x16_iu8` が 8 現れる。6 bit ではこれに `v_alignbit_b32` が 64 現れる。命令の存在は trellis 復号に相応の演算が必要な根拠だが、これだけで ALU 律速とは断定できない。

Saber の公開実装 `GGZ14/saber-engine`、revision `c8a1e84b39e73363495e549d480a6f6c3893800c`、`crates/saber-kernels/hip/mxfp4_gemm_decode.hip` を読み、MXFP4 と EXL3 の形式差を比較した。Saber は LDS staging と MXFP4 W4A8 WMMA を使い、EXL3 は trellis の復号が必要で同じ重み表現ではない。Saber のコードはコピーしていない。

## Graph の試験

`SHELF_HIP_GRAPH=ON` build で `SHELF_GRAPH_DEBUG=1` を使った code 256 token の各 1 回。後述する回転変更を含み、debug 出力によるホスト負荷も含む。既存実装は verify role を graph 対象から除外する。graph に渡す token・batch context は固定 staging にコピーし、attention の選択 signature、行数、出力数、history pointer などを再利用条件に使う。position や request descriptor は device batch context の更新で渡す。

試験実装は `SHELF_GRAPH_VERIFY=1` で単一 request の verify を対象に加え、GDN history と compact log の pointer と stride を再利用条件に含める。tree、QSA / PLE history、stochastic verify は対象外。capture と instantiate の host 時間を含め、以下は繰り返し前の参考値である。

| 経路 | verify graph | tok/s | verify ms/round | round ms | capture / replay |
|---|---|---:|---:|---:|---|
| MTP 固定 4 | OFF | 71.73 | 42.73 | 52.28 | drafter の 1 行のみ、56 node |
| MTP 固定 4 | ON | 74.32 | 40.89 | 50.45 | 5 行、2208 node、65 replay、初回 host 19.04 ms |
| DFlash2 7 | OFF | 98.31 | 46.10 GPU | 56.36 | verify は capture なし |
| DFlash2 7 | ON | 100.95 | 44.68 GPU | 54.89 | 8 行、2210 node、44 replay、初回 host 19.83 ms |

いずれも code の生成列は基準の非投機と一致した。1 回の実行だけでは速度の採否は決められない。

## 行数による split と数値

基準の 27B・最大 batch rows 47 は 64 行 bucket を使う。body の最大 N=17408 から見積もった F32 accum は `(64+16)*17408*4` byte で、N=248320 の head には 5 行分しかない。基準 selector の scratch_rows=5 では rows=1 で split 2、複数行の最初の 5 行で split 1 を選ぶ。

合成 trellis・同一活性で基準 kernel を実行すると、F32 の 248320 要素中 189831、BF16 換算で 50 要素が不一致で、F32 の最大絶対差は 6.86645508e-5。同じ合成 logits の全語彙 softmax で計算した KL(split 2 || split 1) は 1.21e-12 nats。これは基準に存在した数値差で、生成列の違いを実モデルで確認したという意味ではない。

試験修正では EXL3-INT8 の各行列について `16 * min(16,max(1,K/1024)) * N * sizeof(float)` を decode accum 容量の下限にする。これにより selector が許す最大 split 数を 16 行まで収容する。27B の head は 79,462,400 byte。前述の合成入力で scratch_rows=80、rows=1 / 8 とも split 2 となり、F32・BF16 は 248320 要素すべて bit 一致した。これは verify 側の縮約順を基準から変えるため、実モデルの logits と生成列を別途比較する。

入力回転・出力回転の wave 内の 5 段を shuffle に置き換える変更は、基準と変更後で 4 / 6 bit、rows 1 / 2 / 4 / 8 / 16 の合成入力に対する codes、scale、split accum と BF16 output を直接 byte 比較し、すべて一致した。同一入力の rows=1 / 8 も 4 / 6 bit とも bit 一致した。

code・MTP の追加交互測定（OFF → ON → ON → OFF、各 1 回、debug 出力なし）では tok/s は 72.91 → 74.19 → 74.22 → 72.78、verify は 41.91 → 40.95 → 40.96 → 41.98 ms/round、round は 51.44 → 50.54 → 50.52 → 51.52 ms だった。この比較は上の scratch 修正より前の build であり、修正後の比較を兼ねない。

実モデル code prompt の非投機は `--dump-logits` で基準と修正後を比較した。dump には毎 step 4 行分の領域があるため、実際の output 行のみを取り出した。257 行 × 248320 語彙の F32 は全要素 bit 一致し、有限値 mask と top-1 も全行一致、KL 平均・最大は 0。これは非投機の比較であり、verify で変わった logits の実モデル KL ではない。

## 形状と行数ごとの基準 microbenchmark

`8dcb8a6` の kernel、合成 trellis、256 MiB 以上の重みコピーを巡回。GPU event で各回 16 連続実行の平均、warmup/preroll の後に 5 回、下表はその中央値。時間は µs、帯域は trellis バイト数を GEMM 単体または 3 kernel 全体の時間で割った GB/s。`--scratch-rows 1024` なので split が容量制限されない standalone 計測であり、**旧 head 実行の scratch_rows=5・chunk 分割を再現しない**。重みは実 checkpoint と同じサイズ・配置だが合成データである。各 total の 5 回の min/max 幅は中央値の 30% 未満。

| N | K | bits | rows | split | GEMM µs | GEMM GB/s | 3 kernel µs | 3 kernel GB/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 5120 | 4 | 1 | 5 | 14.67 | 178.7 | 23.43 | 111.9 |
| 1024 | 5120 | 4 | 2 | 5 | 14.24 | 184.2 | 21.35 | 122.8 |
| 1024 | 5120 | 4 | 4 | 5 | 13.80 | 189.9 | 21.87 | 119.8 |
| 1024 | 5120 | 4 | 8 | 5 | 14.48 | 181.1 | 22.74 | 115.3 |
| 1024 | 5120 | 4 | 16 | 5 | 14.63 | 179.2 | 23.81 | 110.1 |
| 6144 | 5120 | 4 | 1 | 2 | 35.87 | 438.5 | 43.27 | 363.5 |
| 6144 | 5120 | 4 | 2 | 2 | 36.33 | 432.9 | 44.07 | 356.9 |
| 6144 | 5120 | 4 | 4 | 2 | 37.02 | 424.8 | 44.71 | 351.8 |
| 6144 | 5120 | 4 | 8 | 2 | 38.85 | 404.9 | 46.74 | 336.5 |
| 6144 | 5120 | 4 | 16 | 2 | 39.08 | 402.5 | 49.33 | 318.8 |
| 10240 | 5120 | 4 | 1 | 1 | 58.58 | 447.5 | 66.60 | 393.6 |
| 10240 | 5120 | 4 | 2 | 1 | 59.18 | 443.0 | 66.85 | 392.1 |
| 10240 | 5120 | 4 | 4 | 1 | 59.92 | 437.5 | 68.08 | 385.0 |
| 10240 | 5120 | 4 | 8 | 1 | 62.79 | 417.5 | 71.50 | 366.6 |
| 10240 | 5120 | 4 | 16 | 1 | 62.81 | 417.4 | 74.11 | 353.7 |
| 12288 | 5120 | 4 | 1 | 1 | 64.40 | 488.4 | 71.64 | 439.1 |
| 12288 | 5120 | 4 | 2 | 1 | 65.09 | 483.3 | 72.12 | 436.2 |
| 12288 | 5120 | 4 | 4 | 1 | 66.80 | 470.9 | 73.92 | 425.5 |
| 12288 | 5120 | 4 | 8 | 1 | 68.58 | 458.7 | 78.19 | 402.3 |
| 12288 | 5120 | 4 | 16 | 1 | 70.74 | 444.7 | 100.97 | 311.5 |
| 5120 | 6144 | 4 | 1 | 2 | 37.98 | 414.1 | 45.21 | 347.9 |
| 5120 | 6144 | 4 | 2 | 2 | 38.67 | 406.8 | 46.11 | 341.1 |
| 5120 | 6144 | 4 | 4 | 2 | 39.79 | 395.3 | 47.09 | 334.0 |
| 5120 | 6144 | 4 | 8 | 2 | 41.41 | 379.9 | 48.92 | 321.5 |
| 5120 | 6144 | 4 | 16 | 2 | 41.70 | 377.2 | 51.38 | 306.1 |
| 17408 | 5120 | 4 | 1 | 2 | 95.89 | 464.7 | 103.66 | 429.9 |
| 17408 | 5120 | 4 | 2 | 2 | 96.46 | 462.0 | 104.21 | 427.6 |
| 17408 | 5120 | 4 | 4 | 2 | 101.23 | 440.2 | 109.61 | 406.6 |
| 17408 | 5120 | 4 | 8 | 2 | 103.68 | 429.8 | 114.33 | 389.8 |
| 17408 | 5120 | 4 | 16 | 2 | 104.99 | 424.5 | 119.91 | 371.6 |
| 5120 | 17408 | 4 | 1 | 12 | 85.55 | 520.9 | 95.58 | 466.2 |
| 5120 | 17408 | 4 | 2 | 12 | 87.28 | 510.6 | 97.35 | 457.8 |
| 5120 | 17408 | 4 | 4 | 12 | 89.42 | 498.4 | 100.53 | 443.3 |
| 5120 | 17408 | 4 | 8 | 12 | 92.09 | 483.9 | 104.84 | 425.1 |
| 5120 | 17408 | 4 | 16 | 12 | 96.86 | 460.1 | 115.36 | 386.3 |
| 248320 | 5120 | 6 | 1 | 2 | 1546.31 | 616.7 | 1562.66 | 610.2 |
| 248320 | 5120 | 6 | 2 | 2 | 1567.07 | 608.5 | 1591.42 | 599.2 |
| 248320 | 5120 | 6 | 4 | 2 | 1598.73 | 596.4 | 1637.64 | 582.3 |
| 248320 | 5120 | 6 | 8 | 2 | 1639.37 | 581.7 | 1708.34 | 558.2 |
| 248320 | 5120 | 6 | 16 | 2 | 1688.61 | 564.7 | 1820.01 | 523.9 |

scratch 修正後の code・MTP は初回、`ARENA auto: free=0.36 GiB` で重み確保に失敗した。これは速度に含めない。同じコマンドの再実行は 72.49 tok/s で完走し、基準非投機の生成列と一致した。graph ON・verify ON では 74.23 tok/s、verify 40.97 ms/round、生成列は同じだった。いずれも 1 回の結果。

scratch 修正後の code・DFlash2 各 1 回では、graph verify OFF / ON の順に 98.19 / 103.71 tok/s、verify GPU 46.19 / 43.21 ms/round、round 56.43 / 53.43 ms。どちらも非投機の生成列と一致した。

上表の trellis バイト数を上の層数で合算すると、target の 401 EXL3 線形は **13,117,030,400 byte/token**（重み trellis のみ）である。各形状を独立に測った 3 kernel 時間を層数倍して合計すると rows 1 / 2 / 4 / 8 / 16 はそれぞれ 31.01 / 31.27 / 32.41 / 33.84 / 36.14 ms、重みバイト数 / この合計は 423 / 419 / 405 / 388 / 363 GB/s。これは cache、実際の launch 間隔、旧 lm_head の chunk 条件が異なるため、実モデルの計測値ではない。checkpoint 全体のファイルサイズや BF16 線形・embedding をこの trellis traffic と同一視しない。

実モデルの verify に対しては、一時計測 build で旧 scratch 容量と修正後を切り替え、code prompt の DFlash2 draft 7・256 token を各 1 回実行した。46 round / 368 行の verify 入力 token ID は全行一致。有限値 mask も一致し、F32 は 65,893,995 要素で bit 差、最大絶対差 7.62939453e-6、有限な全語彙の KL(旧 || 修正後) は平均 1.38366492186e-13、最大 6.00898824857e-12 nats、top-1 は 368/368 一致した。生成列も一致した。logits を host に取り出す同期を入れた計測のため、性能評価には使わない。

### 復号と load の切り分け

別の非 production 試験 build で、同じ trellis load を残し、WMMA と activation load を簡単な整数加算へ置き換えたものを「load + 復号」、さらに window 抽出・trellis 復号を外したものを「load のみ」とした。後者にも整数加算、scale と出力書き込みが残る。各値は 5 sample p50 を持つ各 1 sweep で、baseline と交互ではない。VGPR、occupancy、命令の重なりが異なるので、列の差を各処理の独立した時間と解釈しない。

| N | K | rows | load のみ µs | load + 復号 µs | 通常 GEMM µs |
|---:|---:|---:|---:|---:|---:|
| 1024 | 5120 | 1 | 8.98 | 11.46 | 14.67 |
| 1024 | 5120 | 8 | 8.79 | 10.84 | 14.48 |
| 6144 | 5120 | 1 | 29.47 | 32.07 | 35.87 |
| 6144 | 5120 | 8 | 30.73 | 33.50 | 38.85 |
| 17408 | 5120 | 1 | 79.30 | 85.56 | 95.89 |
| 17408 | 5120 | 8 | 82.15 | 89.27 | 103.68 |
| 5120 | 17408 | 1 | 75.52 | 78.74 | 85.55 |
| 5120 | 17408 | 8 | 79.68 | 83.76 | 92.09 |
| 248320 | 5120 | 1 | 1521.43 | 1524.97 | 1546.31 |
| 248320 | 5120 | 8 | 1611.21 | 1605.76 | 1639.37 |

これらは head で重み読み出しの比重が高いことと、body で復号と WMMA の除去により時間が減ることを示唆する。head の試験版間の逆転は測定差や別の実行特性を含みうる。ALU・LDS・帯域の占有率を直接測定した結果ではない。GEMM の LDS は 0 B で、入出力回転には LDS と block barrier がある。

scratch 修正後の code・MTP 交互測定（graph verify OFF → ON → ON → OFF）は 72.84 / 74.23 / 74.25 / 72.88 tok/s、verify は 41.95 / 40.96 / 40.95 / 41.93 ms/round、round は 51.48 / 50.51 / 50.50 / 51.46 ms/round。
