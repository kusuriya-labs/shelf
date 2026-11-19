# Decode の server path gap、長文脈 attention、producer 削減

実験 ID: `DECODE-GAP-1` / `DECODE-ATTN-LONG-1` / `DECODE-PRODUCER-1`

`models/Qwen3.8-27B-Quark-AWQ-MXFP4` の非投機 decode について、
「host / launch の gap」「長文脈 attention」「融合 producer」の 3 つを計測した記録である。
R9700 1 枚（`--device 2`、bf16 KV、sclk 未固定）、commit `961fc78`。

## 1. host / launch gap は存在しない

### 計測

`tg` と `executor.hip` に一時的な timestamp を入れ、1 token の内訳を取った。

- graph: decode の初回に capture（1,166 node、capture 19.4 ms）、以降は毎 token replay。
- `hipGraphLaunch` の実測は **11〜29 us**（rocprofv3 の API trace は 440〜500 us と出るが、
  これは profiler の上乗せ）。
- GPU 完了 → 次の `hipGraphLaunch` まで（host の折り返し）は **約 50 us**。
- `submit_batch` 4 us、launch 0.8 us、`complete_batch` の sync 待ち 30.4 ms。
- 同じ run の kernel 合計は 28.5 ms。差の **約 1.9 ms は GPU 側**で、
  kernel 間の直列化（1,166 node の dispatch / tail）である。

server 経路（`shelf-server` → `shelf-compute --serve-stdio`、device 2、bf16 KV、
prompt 519 token、64 token 生成）でも、client が観測する inter-token は **30.0 ms** で
bench と一致した。`decode_tok_s=32.8`。

| 同時 stream | aggregate | per-stream |
| ---: | ---: | ---: |
| 1 | 28.6 tok/s | 32.8 tok/s |
| 2 | 57.9 tok/s | 30.1 tok/s |
| 4 | 102.6 tok/s | 27.1 tok/s |

concurrency 4 では 1 step が 4 row を処理し、1 stream あたり 36.9 ms で進む。
per-token の固定費（gap と lm_head の一部）は row 数で割られるため、
**単一 stream の gap を削っても server の aggregate には効きにくい**。

結論: host / launch の gap は 50 us 規模で、最適化対象にならない。1.9 ms は GPU 側の
直列化コストで、kernel 数を減らす以外の host 側の手当てはない。

## 2. 長文脈 attention は KV 帯域と MLP で決まる

decode の context 依存（greedy、64 token、`--prefill-chunk 4096`）:

| context | per token | tok/s | 512 からの増分 |
| ---: | ---: | ---: | ---: |
| 512 | 30.5 ms | 32.8 | — |
| 8,192 | 31.9 ms | 31.4 | +1.4 ms |
| 16,384 | 33.5 ms | 29.9 | +3.0 ms |
| 32,768 | 36.7 ms | 27.3 | +6.2 ms |

32,768 の profile では `shelf_attention_paged_bf16_split_other` が
**6,155 us**（16 layer、385 us/layer）、split reduce が 131 us。KV は 2.15 GB/token で
実効 **342 GB/s**（DRAM 636 GB/s の 54%）。

原因は並列度である。visible 32k では `splits=16`、`kv_heads=4` で grid は
**64 block × 4 warp**（128 thread）。64 CU あたり 1 block、SIMD あたり 2 warp しかなく、
1 token あたり K/V の 64 B load を 16 本出すだけでは latency を隠せない。

### 試したこと

| 案 | 8k | 32k | 判定 |
| --- | ---: | ---: | --- |
| baseline（splits 16） | 31.92 ms | 36.78 ms | — |
| splits 32 | 31.45 ms | **34.54 ms（−6.1 %）** | 棄却（数値） |
| splits 64 | — | 34.73 ms | 棄却 |
| splits 128 | — | 35.16 ms | 棄却 |
| K/V を 16 B 連続 load に remap | 32.17 ms | 37.02 ms | 棄却（改善なし） |
| K/V を LDS に stage（wave barrier） | 32.34 ms | 37.83 ms | 棄却（barrier で直列化） |
| fp8 KV | 32.40 ms | 38.07 ms | 棄却（bf16 より遅い） |

splits 32 は速いが、**数値が変わる**。`shelf-bench quality`（自作 corpus 165k token、
context 4,096、64 position、bf16 KV、prefill chunk 2,048）で splits 16 を基準に比較:

| 比較 | KL mean | top-1 一致 | max logit delta |
| --- | ---: | ---: | ---: |
| 16 vs 16（同一設定） | 0 | 100 % | 0 |
| 8 vs 16 | 0.00441 | 95.3 % | 1.55 |
| 32 vs 16 | 0.00308 | 93.75 % | 0.96 |

同一設定では完全に決定的（KL 0）なので、差は split 数そのものに由来する。
8 と 16 でも同じ規模で差が出るため、online-softmax の分割統合は
このモデルでは split 数に敏感である。性能差 6 % に対して KL 0.003 は
既存の融合（KL 0）の基準を満たさないため、**splits 32 は採用しない**。

## 3. 融合 producer は latency 律速で、これ以上の削減は小さい

decode の producer は合計 1.75 ms/token。

| kernel | us/token | launch |
| --- | ---: | ---: |
| residual + norm + quant | 800 | 127 |
| SwiGLU + quant | 660 | 64 |
| GDN gated norm + quant | 285 | 48 |
| norm + quant（初回） | 4 | 1 |

thread 数の再測（`SHELF_MXFP4_PRODUCER_THREADS` を一時追加）:

| threads | ctx512 per token |
| ---: | ---: |
| 256 | 32.14 ms |
| 512 | 31.12 ms |
| 1024 | **30.70 ms** |

1024 が最良で、現状の `rows <= 32 → 1024` は妥当である。
残る launch 融合候補は L2 normalize(q)+scale（96+48）、conv1d+SiLU（48+48）、
rope+split（16+16）、sigmoid+mul（16+16）で、合計約 150 launch・0.3〜0.5 ms 相当。
いずれも 1 回 2〜3 us の kernel で、削減しても 1 % 前後に留まる。

## 4. 結論

単一 stream の decode は重み 12.93 GB の DRAM 読み出し（20.3 ms）に近い床にあり、
30.5 ms の内訳は GEMV 22.4（577 GB/s、天井の 91 %）、lm_head 2.1、producer 1.75、
GDN 0.78、attention 0.54、その他 0.9、GPU 側 gap 1.9 である。

- host / launch の改善余地はない（50 us）。
- 長文脈 attention だけが 5 % 以上の余地を持つ（32k で 6.3 ms）。ただし
  split 数の変更は数値を変えるため、**同じ演算順序を保ったまま MLP を上げる
  新 kernel**（tensor core で score をまとめる、もしくは複数 token を 1 lane が処理する）
  が必要で、split の再調整では代替できない。
- producer と小 kernel の融合は 1 % 前後で、優先度は低い。
