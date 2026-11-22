# Hy4-flash の EP=2 × TP=2

## 目的

Hy4-flash を R9700 × 4 で EP=2 × TP=2 として動かし、EP=4 との数値差が TP の
all-reduce 丸めとして妥当かを確認する。contract は
[../../developer/hy4_flash.md](../../developer/hy4_flash.md) に置く。

## 実装

- rank は `ep * tp_size + tp`。配置表の値は expert の所有 EP group。
- matrix axis の N / K 分割と replicated tensor は developer ドキュメントのとおり。
- 重みは checkpoint 読み込み時に rank ごとの view を作る。N 分割の trellis は
  列ブロックを詰め直し、K 分割と `suh` / `svh` は連続範囲を指す。
- 推論は rank ごとの thread で同じ decoder を実行し、row-parallel の後で
  TP group の bf16 all-reduce、routed expert の後で EP group 間の reduce を行う。
- expert の gate/up は N、down は K を TP lane に分割する。
- attention は head を TP lane に分割し、`learnable_sink_param` は lane の
  head 範囲を使う。

## 発生した不具合と修正

1. expert linear の k / n を host から device の `matrices` 経由で読んでいた。
   `k` / `n` を引数で渡すように修正した。
2. TP の受信バッファを EP reduce と共用していたため、前段の add 完了前に
   上書きされる競合があった。EP 用の受信バッファを分離した。
   修正前は同じ TP group の lane 間でも結果が一致しなかった。
3. `learnable_sink_param` を TP lane の local head 番号で参照していたため、
   lane 1 が先頭 16 head の sink を使っていた。`tp_rank * local_heads` の
   オフセットを追加した。修正前の logits 相対 L2 は 0.52、修正後は 0.131。

## 数値検証

R9700 × 4、24 token 入力、chunk 1024、温度 0。

| 比較 | 結果 |
| --- | --- |
| layer 0 attention の hidden hash | 4 lane で一致。EP=4 とは数 ulp 差 |
| layer 0 FFN の hidden hash | 4 lane で一致。EP=4 とは 1〜4 ulp 差 |
| layer 1 attention / FFN | 4 lane で一致。EP=4 とは数 ulp 差 |
| prefill 最終 logits の相対 L2（EP=4 比） | 0.131 |
| top-1 | 一致（token 13） |
| 24 token 生成 | 16 token まで一致し、その後分岐 |

PR #30 の記録では CPU 積和順序を変えた対照が相対 L2 0.139 であり、0.131 は
同じ桁である。TP の分割境界はすべて 128 要素単位で、ローカル 128 ブロック数も
偶数である。以上から、この差は TP の和の順序変更として許容した。

## 性能と容量

2048 token 入力、256 生成、温度 0、chunk 1024。

| 構成 | prefill | decode |
| --- | --- | --- |
| EP=4 | 1016 tok/s | 21.4 tok/s |
| EP=2 × TP=2 | 1407 tok/s | 24.6 tok/s |
| EP=2 × TP=2 + DFly2 | 1402 tok/s | 108.5 tok/s（受理 215 / 43 round） |

DFly の数値は反復入力で全 draft が受理された条件であり、一般入力の性能ではない。

- rank あたりの重みは約 21.68 GB。
- 1Mi 予約は 4 rank とも約 28.9 GB、2Mi 予約は workspace 1 GiB で約 32.3 GB。
- 2Mi では残り約 1.8 GB、1Mi では約 5.2 GB。

## EP=4 + DFly2 の内訳

自然文 2048 token 入力、256 生成、chunk 1024、温度 0。

| 条件 | prefill | decode | 受理 |
| --- | --- | --- | --- |
| EP=4 通常 | 2.30 s | 19.5 tok/s | - |
| EP=4 + DFly2 | 2.31 s | 50.8 tok/s | 191 / 68 round |
| EP=4 + DFly2、Engram populate | 2.08 s | 60.1 tok/s | 191 / 68 round |
| EP=2 × TP=2 + DFly2 | 1.68 s | 42.7 tok/s | 190 / 68 round |

DFly2 の decode 内訳（Engram populate 後）は attention 2.05 s、FFN 1.57 s、
head 0.30 s、draft 0.58 s である。1 round の target 検証は 1 token 分の decode と
ほぼ同じコストで 6 row を処理しており、速度は受理数で決まる。Engram を populate
しない場合はページフォルトで attention に約 0.8 s が加わる。TP=2 は decode の
all-reduce 同期が増えるため、DFly2 では EP=4 より遅い。

## 未検証

- 2Mi での生成。`generate` の予約長上限が 1Mi のため、2Mi は reserve のみ確認。
- shelf-server 経由の EP=2 × TP=2。
- 長時間運用、DFly の一般入力での受理率、BetterBench。
- EP=4 以外の EP / TP 構成（EP=1 × TP=2、EP=2 × TP=1）。
