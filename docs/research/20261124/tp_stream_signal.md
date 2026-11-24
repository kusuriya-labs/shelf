# TP StreamSignal barrier

対象: `HipPeerTpTransport::sum_hidden_stream_signal` / `TpCombinePolicy` /
`TpBarrierGroup::arrive`

## 目的

TP combine の barrier から、host の rendezvous と kernel 内 spin の両方を消す。
前段の device-side fused barrier は EP2 で確率的に生成列が壊れたため不採用にした。

## 方式

各 rank の stream に次を積む。

1. 自分の部分和を peer の inbox へ copy
2. `hipStreamWriteValue32` で peer の signal を書く
3. `hipStreamWaitValue32`（`hipStreamWaitValueGte`）で自分の signal を待つ
4. inbox を読んで sum / residual add / hyper apply

- host は barrier で止まらない。
- kernel は spin しない。待ちは command processor が行う。
- copy と add は full grid で走る。fused barrier の「64 block 上限」がない。
- signal は host coherent mapped memory に rank × parity で事前確保する。
- `TpCombinePolicy` の `fence_copy` / `fence_add` で system fence の有無を選ぶ。
  kernel は `template <bool kFence>` の特殊化で、要素ループに runtime 分岐はない。

## 計測

### barrier 単体（`shelf-bench tp-reduce --group`、200 回連続）

| rows | Event | StreamSignal | 旧 fused |
| ---: | ---: | ---: | ---: |
| 1 | 44.0 µs | **30.3 µs** | 38.9 µs |
| 8 | 40.1 µs | **28.6 µs** | 84.4 µs |
| 16 | 47.0 µs | **36.2 µs** | 82.9 µs |
| 32 | 49.5 µs | **42.0 µs** | 82.0 µs |

旧 fused は grid-wide barrier の latency が block 数で増えるため、rows=8 以上で負ける。
StreamSignal は copy / add が full grid なので、verify の形でも勝つ。

### 27B EXL3 TP=2 decode

同一 boot の交互測定: Event 35.2 / StreamSignal 38.4 tok/s（**+9.1%**）。
別 boot では Event 36.5 / StreamSignal 40.1（+9.9%）。
`GENERATED_IDS` は 20/20 一致（sha1
`c75f3eea6473dd56b6e927ca29b4dbe18e5c4597`）。

### 27B + DFlash2（16 token）

同一 boot: Event 67.5 / StreamSignal 70.3 tok/s（**+4.2%**）。
verify は Event 138.7 ms / StreamSignal 134〜135 ms（5 round 合計）。
採択率と生成列は一致。別 boot（fence なし）では StreamSignal 78.5 tok/s、
verify 112.2 ms、round 190.6 ms だった。78.5 は fused ではなく StreamSignal の値である。

### Flash-Next EXL3 EP=2

Event 31.7 / StreamSignal 33.8 tok/s（**+6.6%**）。
64 token 生成で 10/10 一致した。fence は `fence_copy=true` / `fence_add=false`。

## fence

- EP2 では copy 末尾の system fence がないと 8/10 で生成列が分岐した。
- copy fence のみで 10/10。add 側の fence は不要だった。
- 27B は copy fence ありで 20/20。fence の有無の差は同一 boot で約 1%。
- 既定は Dense / Qwen4Exp とも `{fence_copy=true, fence_add=false}`。
- `SHELF_TP_FENCE_COPY` / `SHELF_TP_FENCE_ADD` で上書きできる。

## HIP graph

`hipStreamWriteValue32` / `hipStreamWaitValue32` は graph node として capture できる。
2 rank、write + wait + check kernel の 3 node graph を 200 replay して ordering 不一致 0。
現状の segment graph は compute 区間だけを capture し、signal は eager のままである。
step 全体を capture するには、epoch を device memory から読む必要がある
（`hipStreamWriteValue32` は即値を書くため、capture した epoch が固定される）。

## 採否

ACCEPT。Dense / Qwen4Exp の既定 backend を `StreamSignal` にする。
その他は `Event`。`SHELF_TP_SIGNAL_BARRIER=0` で Event に固定できる。

## 残り

- producer の epilogue に copy、consumer の prologue に add を融合して
  combine あたり 2 dispatch を削る（ProducerFused）。
- step 全体の graph capture は epoch を device から読む方式にする。
