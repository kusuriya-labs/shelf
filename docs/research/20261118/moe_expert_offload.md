# Qwen4Exp MoE expert offload と頻度 cache

## 目的

`models/Qwen3.8-Flash-Next-Quark-MXFP4` の TP=2 は、MTP を除き LM head を量子化した
本体が約 68.2 GiB になる。1 rank あたり約 34.1 GiB で、実測 VRAM 31.87 GiB を超える。
routed expert の一部を pinned host へ逃がし、VRAM の slot と頻度で入れ替える。

## 配置

- layer ごとに `SHELF_MOE_HOST_EXPERTS=N` で cold expert を pinned host に置く。
  残りの `local_experts - N` は VRAM の slot に置く。
- TP=2 の `local_experts` は 256。MXFP4 の 1 expert は gate/up/down 合計
  2,611,200 B（2.49 MiB）である。
- host plane は `hipHostMalloc(hipHostMallocMapped)`。GPU の grouped GEMM が PCIe 越しに
  読む。CPU GEMV は持たない。
- slot map は device int32 `[layers][local_experts]`。`>= 0` は VRAM slot、
  `-1 - slot` は host slot。router 後の plan scan がこの map で group を hit/miss に分ける。
- hit と miss は同じ grouped GEMM を 2 回起動する。`group_row_base` で元の group 行を
  指すため、activation plane と output は 1 本の plan のまま共有する。
- miss の重みは host plane から読む。cache の入れ替えは step 間だけで、推論中の
  host 同期は増やさない。

## 頻度 cache

- router は所有 expert ごとの route 数を device の `expert_route_freq` に積む。
- batch 完了後、`Executor::after_batch` が cache の `step()` を呼ぶ。
- `frequency += batch routes`、peak が `SHELF_MOE_CACHE_DECAY`（既定 4096）を超えたら
  全体を半分にする。
- hot host expert と、その step に使われていない cold VRAM expert を交換する。
  1 step の交換数は `SHELF_MOE_CACHE_INSERTS`（既定 4）。
- 交換は「demoted を VRAM から host へ退避 → promoted を host から VRAM へ upload →
  demoted を host slot へ書く」の順で行う。VRAM slot と host slot は同じ index で
  gate/up/down の 3 plane をそろえる。
- `SHELF_MOE_CACHE_LOG_INTERVAL`（既定 64）ごとに hit rate と swap 数を stderr に出す。

## メモリ

`test_qwen4exp_load --skip-ple --rank 0 --device 2 --arena-gib 31` の実測:

| host experts / layer | rank の VRAM 重み | arena used |
| ---: | ---: | ---: |
| 0（全 VRAM） | 約 34 GiB（TP=2 では 31.87 GiB に載らない） | - |
| 48 | 27.572 GiB | 30.709 GiB |
| 64 | 25.702 GiB | 28.839 GiB |
| 96 | 21.961 GiB | 25.098 GiB |

`host_experts=96` の shelf-compute では KV に約 4.59 GB を確保できた。
`host_experts=64` では約 0.57 GB だった。

## 正しさ

prose prompt 38 token、greedy 8 token で、`host_experts=64` と `96` の生成 ID が一致した。

```text
8068,21473,45543,13771,942,19670,11106,4744
```

配置が違っても同じ重みを読むため、結果が一致することを確認した。
`test_qwen4exp_load` は 64 host / 96 host の双方で PASS した。

## hit rate

`host_experts=96`（VRAM 160 / host 96）で `SHELF_MOE_CACHE_LOG_INTERVAL=1` の 8 step:

```text
step=1 hit=61.09% swaps=4
...
step=8 hit=62.82% swaps=32
```

VRAM 在籍率 160/256 = 62.5% とほぼ同じで、8 step では頻度 swap の効果がまだ出ていない。
長い decode か `SHELF_MOE_CACHE_INSERTS` の増加で再評価が必要である。

## 制約

- 現状は **MXFP4 expert のみ**。EXL3 expert は `SHELF_MOE_HOST_EXPERTS>0` で
  `unsupported` を返す。EXL3 は trellis / suh / svh の host plane と grouped kernel の
  split 対応が必要になる。
- 長文 KV、prefix cache、DFlash2、KLD / top-1 は未測定。
- offload は既定で無効（`SHELF_MOE_HOST_EXPERTS=0`）。
