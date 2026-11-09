# Octave KV cache

## 対象と測定条件

Qwen3.8 27B EXL3 4 bpw mul24、R9700（gfx1201）1 台、TP=1、page 16、
full-attention 16 層、KV heads 4、head dimension 256 で測定した。
`flock /tmp/r9700.lock` で GPU を排他し、decode は warmup 2 token 後の 5 token の
per-token GPU event p50、prefill は warmup 1 回後の 5 回の GPU p50 である。
32Ki prefill だけは executor の row 上限に合わせて 8192 token の 4 chunk とした。
測定対象は `feature/kv-octave` の回転空間 prefill 実装（`0e8cbc8`）である。

## 容量

packed-direct prefill への移行により、M2 の K/V 各 1 layer BF16 scratch
（合計 4096 B/token）は削除した。cache bytes は arena が確保する永続 cache 本体であり、
Lloyd-Max table の 1792 B は表から除外した。
回転空間 prefill は既存 output buffer を Q の一時領域として再利用するため、
永続 cache と arena 容量は packed-direct 版から増えていない。

| 形式 | cache B/token/head/layer | cache B/token（16 層、4 heads） | 32Ki cache GiB |
| --- | ---: | ---: | ---: |
| BF16 | 1024 | 65536 | 2.000 |
| FP8 | 520 | 33280 | 1.016 |
| c4v4 | 260 | 16640 | 0.508 |
| c5v5 | 324 | 20736 | 0.633 |
| b4n3v3 | 212 | 13568 | 0.414 |

## Decode

単位は token/s の p50 である。Octave kernel は packed KV を直接読み、16-way split-KV、
fp32 蓄積、専用 reduce と V 出力の逆回転を行う。

| context | BF16 | FP8 | c4v4 | c5v5 | b4n3v3 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 27.18 | 27.01 | 27.02 | 27.08 | 27.10 |
| 8192 | 26.28 | 26.31 | 26.40 | 26.53 | 26.55 |
| 32768 | 23.60 | 23.40 | 24.43 | 24.26 | 24.17 |

融合 kernel は 1 workgroup を 1 KV head と 4 token lane に割り当てる。K/V の unpack と
LUT 参照を 6 query head で共有し、6 本の score dot は 6 wave で同時に計算する。
bit-plane の 32 bit word は wave shuffle で同じ field group 内に配る。

| context | 形式 | 変更前 | 変更後 | 変更前比 | FP8 比 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2048 | c4v4 | 23.40 | 26.08 | 1.115 | 0.960 |
| 2048 | c5v5 | 22.94 | 26.01 | 1.134 | 0.958 |
| 2048 | b4n3v3 | 23.42 | 25.83 | 1.103 | 0.951 |
| 8192 | c4v4 | 18.19 | 23.22 | 1.277 | 0.882 |
| 8192 | c5v5 | 17.28 | 22.92 | 1.326 | 0.870 |
| 8192 | b4n3v3 | 18.15 | 22.93 | 1.263 | 0.871 |
| 32768 | c4v4 | 9.58 | 16.04 | 1.674 | 0.685 |
| 32768 | c5v5 | 8.74 | 15.66 | 1.792 | 0.669 |
| 32768 | b4n3v3 | 9.60 | 15.90 | 1.656 | 0.679 |

旧 scalar 融合 kernel は 2048 が FP8 比 0.95 以上だが 8192 と 32768 は未達だった。
残る差は token 数に比例し、
scalar code 復号と LUT、query head ごとの online softmax、1024-thread workgroup の barrier が
支配する。256-thread と split 64 の候補は 32768 c4v4 が 15.73 tok/s で、split 16 と
4 token lane の 15.86 tok/s を上回らず run 内の外れ値も出たため採用しなかった。

gfx12 WMMA 再実装後は 256-thread、32-token tile とし、QK の 4 個の 64 次元 segment を
4 wave、PV の 4 個の 64 次元出力を別の 4 wave へ割り当てた。4 code を 1 回の
`v_perm_b32` で signed int8 codebook 値へ復号し、fp16 WMMA/fp32 accumulator で計算する。
新値の FP8 比は c4v4 が 100.0% / 100.3% / 104.4%、c5v5 が
100.3% / 100.8% / 103.7%、b4n3v3 が 100.3% / 100.9% / 103.3% であり、
2048 / 8192 / 32768 の全条件で 95% gate を満たした。

## HIP graph

`SHELF_HIP_GRAPH=ON` build で c4v4、context 2048、warmup 2 token 後の 5 token を
測定した。graph replay の p50 は 28.56 tok/s だった。
生成 10 token の ID と合計は一致した。graph debug では decode graph が 2223 node で
capture され、以後 replay された。Octave append、split attention、reduce は capture 中に
割り当てや host 同期を行わず、position、visible length、block table を device 配列から読む。

## MTP

`--mtp --mtp-drafts 4`、19 input、各最大 256 token、temperature 0 で測定した。target KV が
Octave のときも MTP block 自身の独立 KV pool は BF16 とした。tok/s は各 request の最初の
token 後から done までの総 decode token / 総時間である。

| target KV | decode token | rounds | accepted drafts | accepted/round | emitted/round | tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 3723 | 1149 | 2574 | 2.240 | 3.240 | 63.04 |
| FP8 | 3717 | 1139 | 2578 | 2.263 | 3.263 | 63.40 |
| c4v4 | 3784 | 1178 | 2606 | 2.212 | 3.212 | 62.10 |
| c5v5 | 3720 | 1146 | 2574 | 2.246 | 3.246 | 62.74 |

c4v4 target-only は 3784 decode token、27.03 tok/s だった。MTP は 2.30 倍で、19 case の
生成 ID を case ごとに SHA-256 化した suite hash は双方
`f114e375eddd4e4225532622a3a26d73a2a4471a6cdd5302016fe67c7efd88fb` で一致した。
c5v5 も target-only / MTP が `82917c1cd7b44df949202d8aad171b2d614c7c0985715e7fea0ed238d2bd8541`
で一致した。
`SHELF_HIP_GRAPH=ON` と `SHELF_GRAPH_VERIFY=1` の smoke では rows 2〜5 の verify graph を
capture / replay し、非 graph MTP と 19 case の生成 ID が一致した。

## DFlash2

`--dflash2-drafts 7`、19 input、各最大 256 token、temperature 0 で測定した。drafter の
K/V ring は target KV dtype に関係なく独立した BF16 pool である。

| target KV | decode token | rounds | accepted drafts | accepted/round | emitted/round | tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 3723 | 973 | 2750 | 2.826 | 3.826 | 67.50 |
| FP8 | 3717 | 964 | 2753 | 2.856 | 3.856 | 68.09 |
| c4v4 | 3784 | 999 | 2785 | 2.788 | 3.788 | 66.98 |
| c5v5 | 3720 | 972 | 2748 | 2.827 | 3.827 | 68.18 |

c4v4 は spec 無しと同じ 3784 token を生成し、19 case の suite hash も
`f114e375eddd4e4225532622a3a26d73a2a4471a6cdd5302016fe67c7efd88fb` で一致した。
c5v5 も spec 無し / DFlash2 が `82917c1cd7b44df949202d8aad171b2d614c7c0985715e7fea0ed238d2bd8541`
で一致した。
single-shot の 64 token も target-only と一致した。

32Ki prompt と 64 token の結果は次のとおりである。

| target KV | TTFT (s) | decode tok/s |
| --- | ---: | ---: |
| BF16 | 23.90 | 31.19 |
| FP8 | 23.62 | 33.74 |
| c4v4 | 30.09 | 12.32 |
| c5v5 | 29.72 | 10.22 |

回転空間 prefill 後の c4v4 再測定は TTFT 36.03 s、decode 35.20 tok/s だった。
packed-direct の 52.05 s から 30.8% 改善したが、旧 M2 の 30.09 s と FP8 の 23.62 s の
双方より遅く、FP8 の 105% gate は未達である。

c4v4 の concurrent 2、3、4 request は stream/done 整合、完走、同じ concurrent shape の
再実行一致を確認した。4 request の人工 prompt の 1 本は単独実行と 11 token 目以降が
分岐した。同じ mixed batch では再現し、resource ownership の破綻は無い。verify row shape
での near-tie が疑われるが、logit margin は未計測である。

## Prefill

単位は token/s の p50 である。Octave は packed K/V を回転空間のまま fp16 WMMA で消費する。
Q は先行 kernel で回転し、V の重み付き和だけを後続 kernel で逆回転する。
旧 packed-direct は `SHELF_OCTAVE_PREFILL_LEGACY=1` で選択できる。

| context | BF16 | FP8 | c4v4 | c5v5 | b4n3v3 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 1828.97 | 1937.19 | 1820.47 | 1814.10 | 1834.17 |
| 8192 | 1855.88 | 1966.33 | 1654.59 | 1597.36 | 1646.91 |
| 32768 | 1504.56 | 1593.49 | 1007.60 | 933.10 | 989.18 |

FP8 比は c4v4 が 94.0% / 84.1% / 63.2%、c5v5 が 93.7% / 81.2% / 58.6%、
b4n3v3 が 94.7% / 83.8% / 62.1% で、95% gate は全形式で未達だった。回転空間化により
per-token inverse WHT は消え、10 query row × 6 GQA head が packed K/V を共有する。
一方、10-row tile ごとに prefix 全体を再走査して unpack するため、context が長いほど
FP8 との差が拡大する。次の本命は split-KV prefill または複数 query pass の同時保持である。
単純な 32-token KV tile は 8192 で 1590.23 tok/s と逆効果だったため採用しなかった。

## 正しさと制約

gfx12 WMMA kernel と M2 BF16 scratch reference の最大絶対差は、rows=1 c4v4 が
0.006647、rows=4 b4n3v3 が 0.010037 だった。追加した rows=8 c5v5、rows=32
b4n3v3、tree visibility c4v4 はそれぞれ 0.009135、0.012936、0.011914 だった。

c4v4 の `s2k_0`〜`s2k_2` quality KL は 0.002136、0.006197、0.007421、平均
0.005251 だった。M2 実 cache の既存平均 0.005331 に対して -1.49% である。

int8 codebook/fp16 WMMA decode の再測定は 0.002142、0.006791、0.007433、平均
0.005455 で、旧 fused kernel 平均に対して +3.9% と ±5% gate 内だった。packed-direct
prefill を含む `s2k_0` の再測定は 0.001995 だった。

回転空間 prefill と M2 BF16 scratch reference の最大絶対差は、causal + prefix が
0.011832、chunk 境界の block K が 0.003819、rows=2048 / 8192 が 0.011832 だった。
旧 packed-direct との差も同じ範囲である。c4v4 の `s2k_0`〜`s2k_2` quality KL は
0.001839、0.006500、0.007184、平均 0.005174 だった。基準 0.005455 に対して -5.14% で、
低下方向だが指定した ±5% の下限を 0.14 percentage point 外れた。

rows=4 / context=131072 の decode attention 単体は warmup 1 回後 5 回の GPU event p50 が
4.2558 ms だった。2〜4 並列の `test_compute_dflash2_octave.py` は 26/26 PASS した。
graph build では c4v4 の prefill + decode が完走し、context 2048 の decode p50 は
28.64 tok/s だった。

prefill の current chunk を量子化前の K/V から直接読む A mode は見送った。現在の
primitive graph は KV append と paged attention を別 dispatch とし、attention binding に
current chunk の K/V value が無いため、追加するには lowering と primitive contract の変更が
必要になる。回転空間 prefill は安全側として全量量子化後の意味論を維持する。

MTP、DFlash2、HIP graph は有効である。TP、PP、MoE は起動時に拒否する。M3 fused kernel は
rows 1〜32 と tree visibility を実装し、speculative runtime の独立 resource ownership と
混在 batch を受入試験で固定する。
