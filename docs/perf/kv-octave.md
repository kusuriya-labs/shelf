# Octave KV cache

## 対象と測定条件

Qwen3.8 27B EXL3 4 bpw mul24、R9700（gfx1201）1 台、TP=1、page 16、
full-attention 16 層、KV heads 4、head dimension 256 で測定した。
`flock /tmp/r9700.lock` で GPU を排他し、decode は warmup 2 token 後の 5 token の
per-token GPU event p50、prefill は warmup 1 回後の 5 回の GPU p50 である。
32Ki prefill だけは executor の row 上限に合わせて 8192 token の 4 chunk とした。
測定対象は `feature/kv-octave` の M3 実装である。

## 容量

cache bytes は BF16 scratch を含まない永続 cache 本体である。M2 prefill 用 scratch は
K/V 各 1 layer 分なので、全形式で 4096 B/token、32Ki token で 0.125 GiB を別途使う。
Lloyd-Max table の 1792 B は表から除外した。

| 形式 | cache B/token/head/layer | cache B/token（16 層、4 heads） | 32Ki cache GiB | scratch 込み GiB |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 1024 | 65536 | 2.000 | 2.000 |
| FP8 | 520 | 33280 | 1.016 | 1.016 |
| c4v4 | 260 | 16640 | 0.508 | 0.633 |
| c5v5 | 324 | 20736 | 0.633 | 0.758 |
| b4n3v3 | 212 | 13568 | 0.414 | 0.539 |

## Decode

単位は token/s の p50 である。Octave kernel は packed KV を直接読み、16-way split-KV、
fp32 蓄積、専用 reduce と V 出力の逆回転を行う。

| context | BF16 | FP8 | c4v4 | c5v5 | b4n3v3 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 26.99 | 27.16 | 26.08 | 26.01 | 25.83 |
| 8192 | 26.45 | 26.34 | 23.22 | 22.92 | 22.93 |
| 32768 | 23.51 | 23.40 | 16.04 | 15.66 | 15.90 |

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

2048 は FP8 比 0.95 以上だが 8192 と 32768 は未達である。残る差は token 数に比例し、
scalar code 復号と LUT、query head ごとの online softmax、1024-thread workgroup の barrier が
支配する。256-thread と split 64 の候補は 32768 c4v4 が 15.73 tok/s で、split 16 と
4 token lane の 15.86 tok/s を上回らず run 内の外れ値も出たため採用しなかった。

## HIP graph

`SHELF_HIP_GRAPH=ON` build で c4v4、context 2048、warmup 5 token 後の 5 token を
測定した。非 graph は 26.15 tok/s、graph replay は 27.44 tok/s で 4.9% 改善した。
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

## Prefill

単位は token/s の p50 である。Octave は対象 layer の cache を BF16 scratch へ展開して
既存 BF16 paged prefill attention を使う。

| context | BF16 | FP8 | c4v4 | c5v5 | b4n3v3 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 1828.97 | 1839.46 | 1841.11 | 1830.93 | 1832.71 |
| 8192 | 1855.88 | 1887.33 | 1871.53 | 1872.14 | 1873.91 |
| 32768 | 1504.56 | 1526.01 | 1483.80 | 1481.40 | 1487.03 |

## 正しさと制約

M3 fused kernel と M2 BF16 scratch reference の最大絶対差は、rows=1 c4v4 が
0.001774、rows=4 b4n3v3 が 0.002820 だった。追加した rows=8 c5v5、rows=32
b4n3v3、tree visibility c4v4 はそれぞれ 0.001934、0.002768、0.003710 だった。

c4v4 の `s2k_0`〜`s2k_2` quality KL は 0.002136、0.006197、0.007421、平均
0.005251 だった。M2 実 cache の既存平均 0.005331 に対して -1.49% である。

prefill の current chunk を量子化前の K/V から直接読む A mode は見送った。現在の
primitive graph は KV append と paged attention を別 dispatch とし、attention binding に
current chunk の K/V value が無いため、追加するには lowering と primitive contract の変更が
必要になる。M3 は安全側として全量量子化後の M2 prefill を維持する。

DFlash2 と MTP、TP、PP、MoE、HIP graph 実行は起動時に拒否する。M3 fused kernel 自体は
rows 1〜32 と tree visibility を実装したが、speculative runtime 全体は resource ownership と
混在 batch の受入試験が未完了なので有効化していない。
