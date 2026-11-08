# EXL3-INT8 decode の帯域と graph

## 測定条件

- 日付: 2026-11-08〜09（Asia/Tokyo）。GPU: Radeon AI PRO R9700、gfx1201、UUID `GPU-a8e9ddefa2d60f55`。
- 基準 revision: `8dcb8a6`。回転・scratch・graph の中間 revision は `febd92a`、prefill の一致性修正は `0983624`。後述の NT / split は各測定時点の試験差分を明示する。対象は Qwen3.8-27B の EXL3-INT8 4 bpw mul24、lm_head 6 bpw。
- 各 GPU プロセスは `flock /tmp/r9700.lock` で排他。別の prefill 測定も同じ GPU を利用している。
- 推論は BF16 KV、prompt `/tmp/prompt_{code,prose,ja}.ids`、最大生成 256、最大系列長 1024、temperature 0。DFlash2 は BF16 drafter、draft 7。MTP は固定 draft 4。
- 以下の `GB/s` は 10 進数。GEMM 単体は trellis、3 kernel 合計は trellis と `suh` / `svh` の F16 バイト数を分子に使う。いずれも実 DRAM 転送量を直接測った値ではない。

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

`8dcb8a6` の kernel、合成 trellis、256 MiB 以上の重みコピーを巡回。GPU event で各回 16 連続実行の平均、warmup/preroll の後に 5 回、下表はその中央値。時間は µs、帯域は GEMM 単体が trellis bytes、3 kernel 全体が trellis と F16 の `suh` / `svh` bytes を時間で割った GB/s。`shelf-bench` の `weight_GBps` は trellis のみなので、表の 3 kernel 帯域は scale bytes を加算して再計算した。`--scratch-rows 1024` なので split が容量制限されない standalone 計測であり、**旧 head 実行の scratch_rows=5・chunk 分割を再現しない**。重みは実 checkpoint と同じサイズ・配置だが合成データである。各 total の 5 回の min/max 幅は中央値の 30% 未満。

| N | K | bits | rows | split | GEMM µs | GEMM GB/s | 3 kernel µs | 3 kernel GB/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 5120 | 4 | 1 | 5 | 14.67 | 178.7 | 23.43 | 112.4 |
| 1024 | 5120 | 4 | 2 | 5 | 14.24 | 184.2 | 21.35 | 123.4 |
| 1024 | 5120 | 4 | 4 | 5 | 13.80 | 189.9 | 21.87 | 120.4 |
| 1024 | 5120 | 4 | 8 | 5 | 14.48 | 181.1 | 22.74 | 115.8 |
| 1024 | 5120 | 4 | 16 | 5 | 14.63 | 179.2 | 23.81 | 110.6 |
| 6144 | 5120 | 4 | 1 | 2 | 35.87 | 438.5 | 43.27 | 364.0 |
| 6144 | 5120 | 4 | 2 | 2 | 36.33 | 432.9 | 44.07 | 357.4 |
| 6144 | 5120 | 4 | 4 | 2 | 37.02 | 424.8 | 44.71 | 352.3 |
| 6144 | 5120 | 4 | 8 | 2 | 38.85 | 404.9 | 46.74 | 337.0 |
| 6144 | 5120 | 4 | 16 | 2 | 39.08 | 402.5 | 49.33 | 319.3 |
| 10240 | 5120 | 4 | 1 | 1 | 58.58 | 447.5 | 66.60 | 394.1 |
| 10240 | 5120 | 4 | 2 | 1 | 59.18 | 443.0 | 66.85 | 392.6 |
| 10240 | 5120 | 4 | 4 | 1 | 59.92 | 437.5 | 68.08 | 385.5 |
| 10240 | 5120 | 4 | 8 | 1 | 62.79 | 417.5 | 71.50 | 367.1 |
| 10240 | 5120 | 4 | 16 | 1 | 62.81 | 417.4 | 74.11 | 354.1 |
| 12288 | 5120 | 4 | 1 | 1 | 64.40 | 488.4 | 71.64 | 439.6 |
| 12288 | 5120 | 4 | 2 | 1 | 65.09 | 483.3 | 72.12 | 436.6 |
| 12288 | 5120 | 4 | 4 | 1 | 66.80 | 470.9 | 73.92 | 426.0 |
| 12288 | 5120 | 4 | 8 | 1 | 68.58 | 458.7 | 78.19 | 402.8 |
| 12288 | 5120 | 4 | 16 | 1 | 70.74 | 444.7 | 100.97 | 311.9 |
| 5120 | 6144 | 4 | 1 | 2 | 37.98 | 414.1 | 45.21 | 348.4 |
| 5120 | 6144 | 4 | 2 | 2 | 38.67 | 406.8 | 46.11 | 341.6 |
| 5120 | 6144 | 4 | 4 | 2 | 39.79 | 395.3 | 47.09 | 334.5 |
| 5120 | 6144 | 4 | 8 | 2 | 41.41 | 379.9 | 48.92 | 322.0 |
| 5120 | 6144 | 4 | 16 | 2 | 41.70 | 377.2 | 51.38 | 306.6 |
| 17408 | 5120 | 4 | 1 | 2 | 95.89 | 464.7 | 103.66 | 430.3 |
| 17408 | 5120 | 4 | 2 | 2 | 96.46 | 462.0 | 104.21 | 428.1 |
| 17408 | 5120 | 4 | 4 | 2 | 101.23 | 440.2 | 109.61 | 407.0 |
| 17408 | 5120 | 4 | 8 | 2 | 103.68 | 429.8 | 114.33 | 390.2 |
| 17408 | 5120 | 4 | 16 | 2 | 104.99 | 424.5 | 119.91 | 372.0 |
| 5120 | 17408 | 4 | 1 | 12 | 85.55 | 520.9 | 95.58 | 466.7 |
| 5120 | 17408 | 4 | 2 | 12 | 87.28 | 510.6 | 97.35 | 458.3 |
| 5120 | 17408 | 4 | 4 | 12 | 89.42 | 498.4 | 100.53 | 443.7 |
| 5120 | 17408 | 4 | 8 | 12 | 92.09 | 483.9 | 104.84 | 425.5 |
| 5120 | 17408 | 4 | 16 | 12 | 96.86 | 460.1 | 115.36 | 386.7 |
| 248320 | 5120 | 6 | 1 | 2 | 1546.31 | 616.7 | 1562.66 | 610.5 |
| 248320 | 5120 | 6 | 2 | 2 | 1567.07 | 608.5 | 1591.42 | 599.5 |
| 248320 | 5120 | 6 | 4 | 2 | 1598.73 | 596.4 | 1637.64 | 582.6 |
| 248320 | 5120 | 6 | 8 | 2 | 1639.37 | 581.7 | 1708.34 | 558.5 |
| 248320 | 5120 | 6 | 16 | 2 | 1688.61 | 564.7 | 1820.01 | 524.2 |

scratch 修正後の code・MTP は初回、`ARENA auto: free=0.36 GiB` で重み確保に失敗した。これは速度に含めない。同じコマンドの再実行は 72.49 tok/s で完走し、基準非投機の生成列と一致した。graph ON・verify ON では 74.23 tok/s、verify 40.97 ms/round、生成列は同じだった。いずれも 1 回の結果。

scratch 修正後の code・DFlash2 各 1 回では、graph verify OFF / ON の順に 98.19 / 103.71 tok/s、verify GPU 46.19 / 43.21 ms/round、round 56.43 / 53.43 ms。どちらも非投機の生成列と一致した。

上表の trellis バイト数を上の層数で合算すると、target の 401 EXL3 線形は **13,117,030,400 byte/token**（trellis）で、F16 の `suh` / `svh` 14,105,600 byte を含む実重みサイズは **13,131,136,000 byte/token** である。各形状を独立に測った 3 kernel 時間を層数倍して合計すると rows 1 / 2 / 4 / 8 / 16 はそれぞれ 31.01 / 31.27 / 32.41 / 33.84 / 36.14 ms、trellis と scale の合計 / この時間は 423 / 420 / 405 / 388 / 363 GB/s。これは cache、実際の launch 間隔、旧 lm_head の chunk 条件が異なるため、実モデルの計測値ではない。checkpoint 全体のファイルサイズや BF16 線形・embedding をこの trellis traffic と同一視しない。

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

## 重み load の候補

4 / 6 bit の `tile[i0..i2]` を `__builtin_nontemporal_load` に置き換えた試験版を作り、通常 → 候補 → 候補 → 通常の順で各条件を 1 プロセス内で測った。各回は 5 sample の p50、表は交互測定の各 2 回の平均である。逆アセンブルで `th:TH_LOAD_NT` を確認した。BW は trellis bytes / 3 kernel 合計時間。合成入力で実測した小さい差であり、E2E の改善を保証しない。

| N | K | bits | rows | 通常 µs | NT µs | 変化 |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 5120 | 4 | 1 | 21.74 | 20.31 | −6.57% |
| 1024 | 5120 | 4 | 8 | 21.84 | 21.66 | −0.85% |
| 17408 | 5120 | 4 | 1 | 102.65 | 102.45 | −0.20% |
| 17408 | 5120 | 4 | 8 | 111.86 | 109.20 | −2.38% |
| 5120 | 17408 | 4 | 1 | 93.81 | 92.45 | −1.45% |
| 5120 | 17408 | 4 | 8 | 103.02 | 98.28 | −4.60% |
| 248320 | 5120 | 6 | 1 | 1559.96 | 1541.43 | −1.19% |
| 248320 | 5120 | 6 | 8 | 1693.59 | 1726.36 | +1.94% |

N=1024、rows=1 の通常側は 23.013 / 20.458 µs と時間順に変動し、上の差だけでは採用根拠にしない。6 bit head の rows=8 は悪化したため使わない。N=17408、rows=1 の差も小さい。N=5120・K=17408・4 bit のみ NT load を選ぶ候補を実装した。rows=1 にも適用した code 非投機の交互測定は、通常 → NT → NT → 通常で 26.47 / 25.15 / 25.88 / 26.14 tok/s と候補が遅く出た。rows=1 への適用を外し、active rows の device 指定がない rows=2..16 の候補を試した。code・MTP 固定 4 の通常 → 候補 → 候補 → 通常で verify は 42.266 / 42.686 / 42.629 / 41.909 ms/round、round は 52.226 / 52.500 / 52.422 / 51.675 ms/round と候補が遅かったため、5 行には使わず 8 行だけの候補を試した。code・DFlash2 の通常 → 候補 → 候補 → 通常は 101.45 / 99.69 / 91.04 / 101.36 tok/s、verify GPU は 44.42 / 45.36 / 50.50 / 44.46 ms/round と候補が悪化した。したがって NT load は採用せず、コードを戻した。候補の codes・scale・accum・出力は、対象形状で rows 1 / 2 / 4 / 8 / 16 とも基準と byte 一致した。

### 回転の shuffle 化

同一バイナリ構成の基準 → 候補 → 候補 → 基準、各回は 5 sample p50。N=1024・K=5120・4 bit の 3 kernel 合計は rows=1 で 23.248 / 22.688 / 22.623 / 23.425 µs、rows=8 で 24.825 / 24.405 / 24.300 / 24.784 µs。N=17408・K=5120・rows=1 では 103.501 / 101.199 / 102.663 / 102.408 µs。単体の差は小さく、実モデルの tok/s に同じ比率が出るとは限らない。

## BetterBench での未解決の一致性

`shelf-compute --serve-stdio --max-seq-len 1024 --max-concurrent-requests 1 --prefix-cache 0`、BF16 KV、temperature 0、8 カテゴリの先頭各 1 件・最大 256 token。非投機と MTP 固定 4 は基準 / scratch 修正後とも各 2 回の全 8 ケースで生成列が一致した。一方 DFlash2 draft 7 は、基準と scratch 修正後の DFlash2 同士は全 8 ケースで一致するが、非投機とは chat、code、prose、reasoning の 4 ケースで不一致だった。初めて異なる token の 0 始まり位置はそれぞれ 128、153、120、92。file_edit、json、math、summarization は一致した。この不一致は基準 revision でも存在し、scratch 修正や graph の効果とは分離して調査中。3 基本 prompt の一致だけでは一般の greedy 一致を保証できない。

### BetterBench 交互測定（scratch 修正まで、NT 適用前）

8 カテゴリの先頭各 1 件。baseline → scratch → scratch → baseline の順に各実行し、各 process 内では同一順序で 8 件を処理した。下表の値は各 2 回の decode tok/s。EOS による早期終了を含む。`一致` は、当該 4 実行のすべてが非投機 baseline の同一ケースと同じ生成列だったかを示す。DFlash2 の不一致は上記のとおり既存基準にも現れるため、その速度を正しい投機実装の性能として採用しない。

| 経路 | カテゴリ | 基準 tok/s | 修正後 tok/s | 生成列一致 |
|---|---|---:|---:|---|
| greedy | chat | 25.86 / 26.46 | 26.75 / 26.69 | 一致 |
| greedy | code | 26.57 / 26.33 | 26.70 / 26.52 | 一致 |
| greedy | file_edit | 26.51 / 26.27 | 26.65 / 26.44 | 一致 |
| greedy | json | 26.49 / 26.28 | 26.67 / 26.47 | 一致 |
| greedy | math | 26.44 / 26.24 | 26.57 / 26.25 | 一致 |
| greedy | prose | 26.44 / 25.78 | 26.56 / 26.44 | 一致 |
| greedy | reasoning | 26.40 / 26.29 | 26.52 / 26.46 | 一致 |
| greedy | summarization | 26.43 / 26.35 | 25.47 / 26.51 | 一致 |
| mtp | chat | 47.82 / 48.05 | 48.19 / 48.16 | 一致 |
| mtp | code | 72.31 / 71.36 | 72.67 / 72.75 | 一致 |
| mtp | file_edit | 80.42 / 80.90 | 80.88 / 80.95 | 一致 |
| mtp | json | 88.39 / 88.69 | 89.00 / 88.82 | 一致 |
| mtp | math | 79.16 / 79.57 | 79.49 / 79.70 | 一致 |
| mtp | prose | 40.78 / 42.58 | 42.58 / 42.61 | 一致 |
| mtp | reasoning | 60.26 / 60.40 | 60.36 / 60.49 | 一致 |
| mtp | summarization | 64.77 / 65.01 | 64.93 / 65.21 | 一致 |
| dflash | chat | 41.33 / 41.07 | 41.39 / 41.24 | 不一致 |
| dflash | code | 114.50 / 113.46 | 114.75 / 109.69 | 不一致 |
| dflash | file_edit | 77.17 / 76.48 | 77.24 / 82.84 | 一致 |
| dflash | json | 122.90 / 121.87 | 123.17 / 131.77 | 一致 |
| dflash | math | 99.18 / 98.44 | 99.34 / 106.51 | 一致 |
| dflash | prose | 44.08 / 43.82 | 44.18 / 47.41 | 不一致 |
| dflash | reasoning | 70.51 / 70.08 | 70.74 / 75.97 | 不一致 |
| dflash | summarization | 75.12 / 74.57 | 75.42 / 80.79 | 一致 |

scratch 修正後の code・DFlash2 交互測定（graph verify OFF → ON → ON → OFF）は 101.22 / 99.67 / 103.87 / 101.07 tok/s、verify GPU 44.54 / 44.74 / 43.13 / 44.59 ms/round、round 54.74 / 55.59 / 53.34 / 54.82 ms/round。ON の 2 回にばらつきがあり、このケースの graph 効果は上記 1 回の比較ほど明確ではない。

切り分けとして修正前の chat の同じ token IDs を単発 CLI で実行すると、既定 GDN の DFlash2 draft 7 / 3 / 1 は非投機と一致し、単発非投機は server 非投機とも一致した。単発 DFlash2 と server DFlash2 は一致しなかった。server の cache 無効時にも DFlash2 prefill が page 境界で入力を分割することをソース上で確認し、下記の修正を行った。診断用 `SHELF_GDN_RECURRENCE_EXACT=1` では、単発 DFlash2 と同設定の単発非投機が生成位置 178 から不一致となった。この診断設定での不一致は未解決であり、採用していない。

非整列 prompt の server DFlash2 では、prefix cache 無効でも `dflash2_spec_prefill` に page size が渡り、chat の 104 token が 96 + 8 に分割されていた。修正後は cache の有無に関係なく、本来の prefill を保存目的の page 境界で分割しない。prefix cache 無効の BetterBench 8 件では、修正後の DFlash2 は graph OFF / ON とも非投機の 8 件すべてと一致した（各 1 回）。cache 有効時も修正前の 4 件不一致に対し、修正後は graph OFF / ON とも 8 件すべて一致した（各 1 回）。

server の chat ケースで旧 page 分割と分割を外した候補を一時計測 build で比較した。最初の verify で token 入力が一致する 7 行の F32 logits は 1,738,234 要素で bit 差、最大絶対差 0.251702309、有限な全語彙の KL(旧 || 修正後) は平均 1.57143487831e-4、最大 1.05084140547e-3 nats、top-1 は 7/7 一致した。次の draft 入力は異なるため、以降を同一入力の KL として集計していない。旧・新の生成列は 0 始まり 128 token 目から異なり、新しい列は非投機と一致した。これらは logits を同期して取り出した診断実行であり、速度には用いない。

page 境界で分割しないだけの候補は、既存 `test_compute_dflash2.py` で BF16 / FP8 KV の prefix 入力境界復元が各 2 件失敗した（64 token を復元できず 0 token）。これを保つ実装では、prefix cache が有効で cache miss、単一 GPU、非整列かつ境界が 1024 token 以下の場合に限り、先に境界まで prefill して cache に保存し、GDN・KV・drafter context を元に戻してから全入力を通常の区切りで prefill する。余分な prefill と一時 KV が発生する。より長い境界、既存の prefix hit、TP ではこの追加保存を行わず、自然な chunk 終端が境界に一致する場合のみ保存する。既存 `test_compute_dflash2.py` は BF16 / FP8 KV で 60 件すべて通過した。新しい非整列 prompt テストは cache 無効・有効の生成一致に加え、96 token の prefix を復元した次 request も単発非投機と一致し、7 件すべて通過した。追加 prefill の費用は後述する。

追加 prefill 版で cache 有効の BetterBench 8 ケースを各 1 回実行し、同じ cache 設定の非投機 8 件と生成列がすべて一致した。file_edit の prompt は 96 token で page に整列し、追加の境界 prefill は不要。交互測定を後述する。

## split-K の候補

3 形状、rows 1 / 8、通常 → 候補 → 候補 → 通常、各回 5 sample p50。表は各 2 回の平均、時間は 3 kernel 合計 µs。cache を超える合成 trellis を巡回した。

| N | K | rows | 通常 split | 候補 split | 通常 µs | 候補 µs | 変化 |
|---:|---:|---:|---:|---:|---:|---:|
| 5120 | 17408 | 1 | 12 | 8 | 93.85 | 96.74 | +3.08% |
| 5120 | 17408 | 1 | 12 | 16 | 93.84 | 94.81 | +1.04% |
| 5120 | 17408 | 8 | 12 | 8 | 101.83 | 103.86 | +1.99% |
| 5120 | 17408 | 8 | 12 | 16 | 102.68 | 104.71 | +1.98% |
| 17408 | 5120 | 1 | 2 | 1 | 103.87 | 101.13 | −2.64% |
| 17408 | 5120 | 1 | 2 | 4 | 103.97 | 94.65 | −8.97% |
| 17408 | 5120 | 8 | 2 | 1 | 112.93 | 107.08 | −5.18% |
| 17408 | 5120 | 8 | 2 | 4 | 113.46 | 105.22 | −7.26% |
| 1024 | 5120 | 1 | 5 | 2 | 20.74 | 31.47 | +51.72% |
| 1024 | 5120 | 1 | 5 | 4 | 20.69 | 22.20 | +7.31% |
| 1024 | 5120 | 8 | 5 | 2 | 22.24 | 33.49 | +50.60% |
| 1024 | 5120 | 8 | 5 | 4 | 22.26 | 23.78 | +6.86% |

N=17408・K=5120・4 bit の gate/up について、16 行 × 4 split 分の scratch がある場合に split 4 を選ぶ候補を実装し、行数間の bit 一致、KLD と生成列、E2E を検証中。これは演算順を変える。ほかの測定候補は悪化したため採用しない。

## 再現に使うコマンド

この計測では次の環境を設定し、GPU を使う各コマンドを個別に `flock` で囲んだ。`build-gfx1201` は graph OFF、`build-gfx1201-graph` は `-DSHELF_HIP_GRAPH=ON` の別 build。どちらも Release、`gfx1201`、ROCm `/opt/rocm/core-7.14`、clang 23、build は `nice -n 10 cmake --build <dir> -j 8`。

```sh
export LD_LIBRARY_PATH=/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib
export ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55
export EXL3_BW_TARGET=/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24
export EXL3_BW_DRAFT=/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-DFlash2

flock /tmp/r9700.lock ./build-gfx1201/shelf-bench gpu-memory \
  --stream-read --warmup 2 --iterations 5 --iters 20
flock /tmp/r9700.lock ./build-gfx1201/shelf-bench exl3-int8 \
  --n 17408 --k 5120 --bits 4 --rows 8 --scratch-rows 1024 \
  --working-mib 256 --iterations 16
flock /tmp/r9700.lock env SHELF_QWEN35_PERF_TIMING=1 SHELF_QWEN35_PERF_STATS=1 \
  ./build-gfx1201/shelf-bench tg --model-dir "$EXL3_BW_TARGET" --mode forward \
  --context 128 --tokens 4 --warmup 1 --perf-stats-output /tmp/exl3-stats.json \
  --output /tmp/exl3-tg.json
flock /tmp/r9700.lock ./build-gfx1201/shelf-compute --model-dir "$EXL3_BW_TARGET" \
  --input-ids-file /tmp/prompt_code.ids --max-new-tokens 256 \
  --max-seq-len 1024 --temperature 0
flock /tmp/r9700.lock ./build-gfx1201/shelf-compute --model-dir "$EXL3_BW_TARGET" \
  --input-ids-file /tmp/prompt_code.ids --max-new-tokens 256 \
  --max-seq-len 1024 --temperature 0 --mtp --mtp-drafts 4
flock /tmp/r9700.lock env SHELF_GRAPH_VERIFY=1 ./build-gfx1201-graph/shelf-compute \
  --model-dir "$EXL3_BW_TARGET" --input-ids-file /tmp/prompt_code.ids \
  --max-new-tokens 256 --max-seq-len 1024 --temperature 0 \
  --dflash2-model-dir "$EXL3_BW_DRAFT" --dflash2-drafts 7
```

形状表は `exl3-int8` の N/K/bits と rows を表の値に変えて計測した。実モデルの
生成比較は prompt を code / prose / ja に変え、各 GPU プロセスを別々にロックした。
`SHELF_GRAPH_DEBUG=1` は capture の node 数・初回 host 時間と replay を記録する診断用で、
性能比較では無効にする。BetterBench は `vendor/BetterBench/corpus/v1` の各カテゴリ先頭を
model の chat template（`enable_thinking=false`）で token 化し、同じ token IDs を投入した。

verify の capture は 2210 node を 1 回の `hipGraphLaunch` で投入する。これは host 側の
個別 launch 呼び出しを減らすが、graph 内の GPU kernel node が 1 個になることは意味しない。
通常実行の 2199 physical launch/token と graph node 数は区間・node の種類が異なるため、
同一の launch 数として直接比較しない。

13.117 GB の trellis を今回の read-only p50 623.82 GB/s で 1 回ずつ読むだけの
算術下限は約 21.03 ms/token。これは scale、activation、出力、復号、launch と競合を
含まない。単純な下限に対して event 計測の EXL3 線形 33.71 ms/token は約 12.68 ms 長いが、
この差をすべて削減可能な DRAM 待ちとみなすことはできない。小 N の wave 数、
復号命令、split の最終波、別 kernel の固定費と cache 条件が混在する。

基準 event 集計の EXL3 線形 33.709 ms に trellis と `suh` / `svh` の bytes を割ると約 389.5 GB/s、
今回の read-only p50 の約 62.4% となる。この値は入力・出力回転と launch 間隔を含む
区間に対する指標であり、DRAM counter の実測ではない。checkpoint 全体の約 15 GB を
線形が毎 token 読む量として割った推定とは分母に含める重みが異なる。

### cache 追加 prefill の交互測定

cache 有効、各カテゴリの最初の入力、同じ process 内の 8 request を、追加なし → 追加あり → 追加あり → 追加なしの順に実行。各 request の `prefill_ms` は host 時間で、model load は含まない。全 request の restored は 0。追加なし版は中間 cache 保存の既存テストに失敗するため採用対象ではなく、コスト比較用。

| カテゴリ | 入力 token | 追加なし ms | 追加あり ms |
|---|---:|---:|---:|
| chat | 104 | 205.1 / 196.5 | 367.7 / 357.6 |
| code | 58 | 120.2 / 111.2 | 226.5 / 224.5 |
| file_edit | 96 | 167.2 / 152.9 | 165.6 / 152.9 |
| json | 65 | 150.7 / 138.8 | 391.2 / 260.3 |
| math | 62 | 116.0 / 114.4 | 220.6 / 217.4 |
| prose | 46 | 104.1 / 103.5 | 180.1 / 179.4 |
| reasoning | 83 | 152.2 / 149.7 | 299.4 / 296.6 |
| summarization | 103 | 162.7 / 159.9 | 318.1 / 315.8 |

json の追加ありは 391.2 / 260.3 ms と変動が大きく、この 2 回だけでは安定値としない。単独の交互再測定を実行中。prefill の追加費用を decode の token / round 時間へ加算していない。

`SHELF_EXL3_SPLIT` / `SHELF_EXL3_GROUPS` は現行 source では W16A16 の dispatch に渡され、
EXL3-INT8 の split selector には適用されない。INT8 の候補比較には microbenchmark の
`--split` を使った。

split 2 → 4 の teacher-forced 実モデル比較は、PSKLDTOK の同一 token 列、context 128、
128 位置、BF16 KV。`shelf-bench quality --dump` の基準と `--compare` の候補で、
全語彙 KL(基準 || 候補) 平均 0.000212619884 nats、top-1 一致 128/128、top-5 overlap
0.9875、最大 logits 絶対差 2.03755474、PPL 2.92298264 → 2.92570221（+0.0930%）。
累積順を変えた事実と、この限定された 128 位置での差を明示する。code prompt の
非投機 256 token は基準と一致したが、他の生成列と投機一致の確認は続行中。

split の KLD 比較では、同じ PSKLDTOK 入力に対し、基準 binary で
`flock /tmp/r9700.lock ./build-gfx1201/shelf-bench-baseline quality --model-dir "$EXL3_BW_TARGET" --tokens-file /tmp/shelf-prefill-32768.psktok --context 128 --positions 128 --prefill-chunk 128 --dump /tmp/exl3-quality-base.bin`、
候補 binary で同じ引数の `--dump` を `--compare /tmp/exl3-quality-base.bin` に変えて実行した。
NT を外した最終 binary でも同じ 128 位置の KL、PPL、top-1、top-5、最大差が一致した。
一時 token corpus は PSKLDTOK 形式の固定列であり、別の列で数値が一致するとは限らない。

回転比較の head（N=248320、K=5120、6 bit）の 3 kernel 合計は rows=1 の通常
1562.810 / 1563.447 µs に対し shuffle 1561.443 / 1561.007 µs、rows=8 は通常
1708.412 / 1707.560 µs に対し shuffle 1693.292 / 1694.143 µs だった。候補 binary は
途中で再 link されているが、head の split と GEMM 経路は変えていない。

参考として、gate/up の rows=1 単体差約 9.32 µs を 128 回へそのまま掛けると
約 1.19 ms/token の短縮になる。基準 event 合計から単純減算した 32.52 ms と
13.117 GB の比は約 403 GB/s だが、cache・clock・launch の重なりを無視した推算であり、
実モデルの到達値ではない。公称帯域の 85% 達成を示す証拠にはならない。

split 4 の EXL3-INT8 GPU テストは gate/up 形状の rows=1 / 8 の bit 一致を含め
通過し、TP テストも通過した。非整列 prompt の server 回帰テストは cache OFF / ON と
96 token prefix の再利用を含め 7/7 通過。code の非投機 256 token は基準と一致し、
同じ版の MTP も graph OFF / ON の双方で非投機と一致した。prose・日本語・DFlash2 の
判定は実行中。

split 2 を基準に追加した同形状の測定では、split 3 は rows=1 で 101.15 / 101.95 →
94.22 / 93.87 µs、rows=8 で 111.86 / 113.22 → 102.81 / 103.16 µs。split 5 は
rows=1 で 100.64 / 100.89 → 93.00 / 93.14 µs、rows=8 で 112.37 / 112.82 →
105.00 / 105.29 µs だった。さらに split 4 → 3 → 3 → 4 の直接比較では、
rows=1 は 94.36 / 95.37 → 94.87 / 94.68 µs、rows=8 は 104.54 / 106.53 →
104.17 / 105.15 µs。差は約 0.1% / 0.8% と小さく、40 group を均等に分ける
split 4 を候補とする。split 4 → 5 → 5 → 4 では rows=1 が約 0.6% 短縮、
rows=8 が約 0.7% 悪化した。

split selector は 32 WGP に対して `32 * 4 * 6 = 768` wave slot を仮定する。
一方、今回の HIP occupancy API は 4 bit MT=1 に 16 block/WGP × 4 wave =
64 wave/WGP、すなわち理論上 2048 wave、6 bit は 11 block/WGP × 4 = 44 wave/WGP、
計 1408 wave と報告した。理論値と selector 仮定には差があり、gate/up で split 4 が
勝つことの説明候補になる。ただし動作中の residency や wave 発行率を直接測っていないため、
この差だけを律速の確定原因とはしない。

NT 不採用後の split 4 実行では、graph OFF の `shelf-compute` SHA-256 は
`d88f8e24ff513118aa2c28654cb355ed65e8718a9e478bae133e8151d18d92aa`、
graph ON は `dcd4ebc16ac7e9d92126164b6c25d40441c241fe54181fee64615eb7b613397d` を使用する。
NT を含む先行の DFlash2 試験値は final 版の性能として集計しない。

split 4 の BetterBench 非投機の初回は、基準非投機と 8 件中 3 件で生成列が変化した。
初めて異なる 0 始まり位置は chat 128、code 153、reasoning 92。残る 5 件は一致した。
これは異なる実装版の greedy 同士の比較であり、投機一致の判定ではない。split 4 の
DFlash2 / MTP は、別途 split 4 の非投機生成列と比較する。結果の 2 回目と投機側は実行中。

NT を除いた split 4 版の BetterBench では、graph ON・DFlash2 の各 1 回が、同じ split 4 版の
非投機と 8/8 ケースで完全一致した。比較には旧版ではなく、新版の greedy token 列を使った。

split 4 の `/tmp/prompt_prose.ids` 非投機は、旧版と 0 始まり 38 番目の生成 token から
異なった。両版とも 256 token。これは版間の数値変更であり、同一版での投機一致は
その版の非投機を基準に別途判定する。

NT を外した split 4 版の code・DFlash2・graph OFF は 1 回で 102.66 tok/s、46 round、
verify GPU 2012.29 / 46 = 43.75 ms/round、draft GPU 383.31 / 46 = 8.33 ms/round、
round 2482.74 / 46 = 53.97 ms/round。同じ版の非投機生成列と完全一致した。基準の
98.00 tok/s・46.31 / 56.54 ms は各 1 回で、反転反復を待つ参考比較である。

今回の read-only p50 を 100% とすると、85% は約 530.2 GB/s。上記の 403 GB/s は
単純推算にすぎず、この目標の達成とは扱わない。

### GDN の非 EXL3 線形
`in_proj_a` / `in_proj_b` は各 48 層、N=48・K=5120。checkpoint は F16 格納で、
実行時の `LINEAR_BF16` に対応する。`shelf-bench gemm --dtype bf16 --variant auto`、
warmup 100、5 sample × 50 launch の p50。報告された min と p95 の幅は中央値の 30% 未満。
合成重み 491,520 byte を繰り返すため cache resident であり、DRAM 帯域ではない。

| rows | p50 µs | 重み bytes / 時間 GB/s |
|---:|---:|---:|
| 1 | 7.118 | 69.1 |
| 2 | 7.870 | 62.5 |
| 4 | 8.945 | 54.9 |
| 8 | 10.601 | 46.4 |
| 16 | 15.087 | 32.6 |
