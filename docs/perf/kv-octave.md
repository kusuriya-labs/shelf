# Octave KV cache

## 対象と測定条件

Qwen3.8 27B EXL3 4 bpw mul24、R9700（gfx1201）1 台、TP=1、page 16、
full-attention 16 層、KV heads 4、head dimension 256 で測定した。
`flock /tmp/r9700.lock` で GPU を排他し、decode は warmup 2 token 後の 5 token の
per-token GPU event p50、prefill は warmup 1 回後の 5 回の GPU p50 である。
32Ki prefill だけは executor の row 上限に合わせて 8192 token の 4 chunk とした。
測定対象は `feature/kv-octave` の M3 実装である。

## 容量

packed-direct prefill への移行により、M2 の K/V 各 1 layer BF16 scratch
（合計 4096 B/token）は削除した。cache bytes は arena が確保する永続 cache 本体であり、
Lloyd-Max table の 1792 B は表から除外した。

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
| BF16 | 3723 | 1149 | 2574 | 2.240 | 3.240 | 63.16 |
| FP8 | 3717 | 1139 | 2578 | 2.263 | 3.263 | 63.51 |
| c4v4 | 3773 | 1192 | 2581 | 2.165 | 3.165 | 59.83 |
| c5v5 | 3713 | 1155 | 2558 | 2.215 | 3.215 | 60.46 |

c4v4 target-only は 3773 decode token、26.81 tok/s だった。MTP は 2.23 倍で、19 case の
生成 ID を case ごとに SHA-256 化した suite hash は双方
`5dc858f907a6ff0300718b44076f576093298989b25f7f1e91e2932ef3b186f7` で一致した。
`SHELF_HIP_GRAPH=ON` と `SHELF_GRAPH_VERIFY=1` の smoke では rows 2〜5 の verify graph を
capture / replay し、非 graph MTP と 19 case の生成 ID が一致した。

## DFlash2

`--dflash2-drafts 7`、19 input、各最大 256 token、temperature 0 で測定した。drafter の
K/V ring は target KV dtype に関係なく独立した BF16 pool である。

| target KV | decode token | rounds | accepted drafts | accepted/round | emitted/round | tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 3723 | 973 | 2750 | 2.826 | 3.826 | 68.16 |
| FP8 | 3717 | 964 | 2753 | 2.856 | 3.856 | 69.88 |
| c4v4 | 3773 | 1006 | 2767 | 2.750 | 3.750 | 64.79 |
| c5v5 | 3713 | 976 | 2737 | 2.804 | 3.804 | 64.25 |

c4v4 は spec 無しと同じ 3773 token を生成し、19 case の suite hash も
`5dc858f907a6ff0300718b44076f576093298989b25f7f1e91e2932ef3b186f7` で一致した。
single-shot の 64 token も target-only と一致した。

32Ki prompt と 64 token の結果は次のとおりである。

| target KV | TTFT (s) | decode tok/s |
| --- | ---: | ---: |
| BF16 | 23.90 | 31.19 |
| FP8 | 23.62 | 33.74 |
| c4v4 | 30.09 | 12.32 |
| c5v5 | 29.72 | 10.22 |

packed-direct prefill 後の c4v4 再測定は TTFT 52.05 s、decode 35.12 tok/s だった。
decode は改善した一方、TTFT は旧 M2 の 30.09 s と FP8 の 23.62 s の双方より悪化した。
長い context で GQA 6 head が同じ page の unpack と inverse WHT を繰り返すことが原因で、
R2 の 95% prefill gate は未達である。

c4v4 の concurrent 2、3、4 request は stream/done 整合、完走、同じ concurrent shape の
再実行一致を確認した。4 request の人工 prompt の 1 本は単独実行と 11 token 目以降が
分岐した。同じ mixed batch では再現し、resource ownership の破綻は無い。verify row shape
での near-tie が疑われるが、logit margin は未計測である。

## Prefill

単位は token/s の p50 である。旧値は対象 layer の pool 全体を BF16 scratch へ展開する
M2 経路、新値は packed page を LDS へ直接復号し、既存 gfx12 BF16 WMMA tile で消費する
経路である。新経路は global scratch と全 pool dequant dispatch を持たない。

| context | BF16 | FP8 | c4v4 | c5v5 | b4n3v3 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 1828.97 | 1839.46 | 1841.11 | 1830.93 | 1832.71 |
| 8192 | 1855.88 | 1887.33 | 1871.53 | 1872.14 | 1873.91 |
| 32768 | 1504.56 | 1526.01 | 1483.80 | 1481.40 | 1487.03 |

packed-direct c4v4 の初期測定は 2048 が 1759.97 token/s（FP8 比 95.7%）、8192 が
1381.55 token/s（同 73.2%）だった。2048 は目標を満たすが、8192 は未達である。現在の
kernel は 32 query row × 1 query head の block ごとに同じ KV head の packed page を復号するため、
GQA 6 head 間で inverse WHT と unpack を共有できていない。global scratch は解消したが、長い
prefill ではこの重複計算が新しい律速である。

## 正しさと制約

gfx12 WMMA kernel と M2 BF16 scratch reference の最大絶対差は、rows=1 c4v4 が
0.006647、rows=4 b4n3v3 が 0.010037 だった。追加した rows=8 c5v5、rows=32
b4n3v3、tree visibility c4v4 はそれぞれ 0.009135、0.012936、0.011914 だった。

c4v4 の `s2k_0`〜`s2k_2` quality KL は 0.002136、0.006197、0.007421、平均
0.005251 だった。M2 実 cache の既存平均 0.005331 に対して -1.49% である。

int8 codebook/fp16 WMMA decode の再測定は 0.002142、0.006791、0.007433、平均
0.005455 で、旧 fused kernel 平均に対して +3.9% と ±5% gate 内だった。packed-direct
prefill を含む `s2k_0` の再測定は 0.001995 だった。

prefill の current chunk を量子化前の K/V から直接読む A mode は見送った。現在の
primitive graph は KV append と paged attention を別 dispatch とし、attention binding に
current chunk の K/V value が無いため、追加するには lowering と primitive contract の変更が
必要になる。M3 は安全側として全量量子化後の M2 prefill を維持する。

MTP、DFlash2、HIP graph は有効である。TP、PP、MoE は起動時に拒否する。M3 fused kernel は
rows 1〜32 と tree visibility を実装し、speculative runtime の独立 resource ownership と
混在 batch を受入試験で固定する。
