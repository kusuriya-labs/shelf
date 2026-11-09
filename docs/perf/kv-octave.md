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
| 2048 | 26.26 | 26.23 | 23.40 | 22.94 | 23.42 |
| 8192 | 25.52 | 25.60 | 18.19 | 17.28 | 18.15 |
| 32768 | 23.05 | 22.91 | 9.58 | 8.74 | 9.60 |

現実装は容量を減らすが decode を高速化しない。1 query head ごとの workgroup が token を
直列走査し、K/V unpack と LUT load を query head 間で再利用していないため、context が
長いほど BF16/FP8 に対する差が広がる。採用候補にするには、1 KV head の unpack を
q_per_kv=6 query heads で共有し、複数 token を wave 間で並列化する再設計が必要である。

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
0.001774、rows=4 b4n3v3 が 0.002820 だった。c4v4 の `s2k_0` quality KL は
M2 の 0.002038 に対して fused decode が 0.002160 だった。

prefill の current chunk を量子化前の K/V から直接読む A mode は見送った。現在の
primitive graph は KV append と paged attention を別 dispatch とし、attention binding に
current chunk の K/V value が無いため、追加するには lowering と primitive contract の変更が
必要になる。M3 は安全側として全量量子化後の M2 prefill を維持する。

DFlash2 と MTP、TP、PP、MoE、HIP graph 実行は起動時に拒否する。M3 fused kernel 自体は
rows 1〜32 と tree visibility を実装したが、speculative runtime 全体は resource ownership と
混在 batch の受入試験が未完了なので有効化していない。
