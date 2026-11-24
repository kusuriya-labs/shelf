# TP fused barrier（device-side barrier）

> Status: REJECTED。後継は [tp_stream_signal.md](tp_stream_signal.md)。

対象: `TpBarrierGroup::arrive` / `HipPeerTpTransport::sum_hidden_fused` / `TpCoordinator::initialize`

## 目的

TP の combine ごとに host thread が mutex + condition_variable で rendezvous する経路を、
GPU kernel 内の barrier に置き換える。host は kernel を enqueue して次 segment の
投入へ進める。測定は `docs/perf/methodology.md` に従う。

## 実装

`SHELF_TP_FUSED_BARRIER=1` で、`TpCombineFusion::None` と `ResidualAdd` の combine を
1 kernel で実行する。

1. 自分の部分和を peer の inbox へ copy
2. `__threadfence_system()` と arrival counter
3. peer の host-mapped flag を spin
4. inbox を読んで sum / residual add

- `TpBarrierGroup::arrive` は fused 経路では kernel を enqueue して返る。
- `HyperApply` と、inbox 容量を超える combine は CPU barrier 経路を使う。
- 対象は 2 rank のみ。
- flag は `hipHostMalloc(hipHostMallocMapped)` の host memory に置く。

## 実験 1: E2E decode（Qwen3.8-27B EXL3、TP=2）

`shelf-bench tg --mode greedy --context 64 --tokens 8 --warmup 2`、3 回の
`gpu_tokens_per_sec_p50`。graph は既定の ON。

| graph | fused off | fused on | 差 |
| --- | ---: | ---: | ---: |
| ON | 36.42 / 36.34 / 36.48（中央値 36.42） | 41.31 / 41.33 / 41.47（中央値 41.33） | **+13.5%** |
| OFF | 26.59 / 26.89（中央値 26.74） | 30.42 / 30.09（中央値 30.26） | **+13.2%** |

生成結果は `shelf-compute` の `GENERATED_IDS=17,18,19,20,21,22,23,24` が
fused off / on で一致した。

## 実験 2: serve 4 並列

`shelf-compute --serve-stdio`、TP=2、4 request × 64 token、warmup 後に測定。

| fused | wall_s（3 run） |
| --- | --- |
| off | 2.29 / 2.29 / 2.28 |
| on | 2.29 / 2.29 / 2.30 |

差はない。4 並列では GPU が飽和しており、host の barrier 待ちが隠れるためである。
fused barrier の利点は低並列の decode に限られる。

## 実験 3: ハングと原因

2 request / 4 request の serve で、最初の prefill 後に停止した。GPU 1 枚が kernel 内で
回り続け、もう 1 枚は idle、host は serve ループで 100% になった。

`SHELF_TP_FUSED_DEBUG=1` の最後のログ:

```text
FUSED_LAUNCH rank=1 epoch=129 ...
ARRIVE_FUSED_DONE rank=1 ok=1
GROUP_PEER_WRITE_ENTER rank=1 bytes=1986560
ENSURE_GROW bytes=1986560 was=1048576
ARRIVE_FUSED rank=0 ...
GROUP_FUSED_ENTER rank=0
```

原因は 2 つが重なったことである。

- `peer_write_target` が `TpBarrierGroup::mutex_` を保持したまま `ensure_buffers` を呼び、
  1,048,576 → 1,986,560 byte へ拡張した。拡張内の `hipMalloc` / `hipMemcpy` が
  GPU 待ちで止まった。
- rank 1 の GPU は epoch 129 の fused kernel で rank 0 の到着を待っていた。rank 0 は
  同じ mutex を待っていたため、rank 1 の拡張は完了しない。

つまり、fused barrier で host rendezvous を外したのに、barrier 中に host 側の確保が
残っていたことが原因である。kernel の spin 上限では解けない。host が mutex で止まり、
誰も peer flag を publish しないためである。

## 修正: 事前確保

推論中の `hipMalloc` を禁止し、必要な最大容量を初期化時に確保する。

- `tp_inbox_bytes()` が全 program / TP schedule の combine 値から最大 byte 数を求める。
- `TpCoordinator::initialize` が worker 起動前に `TpBarrierGroup::reserve(bytes)` を呼ぶ。
- `attach_aux_executor` は drafter の program から必要量を再計算して reserve する。
- fused の host flag と arrival counter も `verify()` / `reserve()` で確保する。
- 実行中の `sum_hidden` / `sum_hidden_fused` / `peer_write_target` は容量を検査するだけで、
  確保も拡張もしない。容量不足は `Status::unsupported` で返す。

`shelf-compute --serve-stdio`（`--max-num-batched-tokens 512`）では
`TP_RESERVE bytes=5242880` となり、最初の `SUM` 以降に `ENSURE_GROW` は出なかった。
修正後、2 request / 16 token と 4 request / 64 token の serve は完走した。

## 採否

REJECTED。

- EP2 で 64 token 中 2/10 の割合で生成列が分岐した。fence、uncached inbox、
  host-mapped counter、DebugRing で確率は変わるが 0 にできない。
- DebugRing を有効にすると一致する。観測の system-scope アクセスが race を隠している。
- verify（rows=8）では grid-wide barrier の latency が 84 µs になり、event の 40 µs に負ける。
- TC workspace の事前確保と DebugRing の知見は後継の StreamSignal でも使う。
