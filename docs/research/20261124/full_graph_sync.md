# Full graph の GPU 間同期 PoC

## 前提の変化

`pci=disable_acs_redir=02:00.0;02:04.0;04:10.0;04:18.0;0c:00.0;0c:08.0` を付けて起動した
環境で、P2P の可視性を再測定した。以前の「peer write が running kernel から見えない」
という結果は、**テスト側の 2 つの artifact** によるものだった。

1. 読み手が同じバッファを直前に `hipMemset` して dirty line を自分の L2 に残していた。
2. 読み手側の `__threadfence_system()` を block 0 だけで実行し、他の block が
   invalidate 前に読んでいた。

条件を直すと、peer への push は正しく可視になる。`full_graph_sync_probe.hip` の
`pushseq` は 10 回転送で不一致 0、`seq`（push + host-mapped flag + 全 block spin）は
GPU 0-1 で 20 回、GPU 1-2 で 10 回すべて不一致 0 だった。

## 実測

| 実験 | 結果 |
| --- | --- |
| cross-device `hipStreamWaitEvent` の capture | どの capture mode でも wait node が落ちる（kernel 1 node のみ） |
| `hipGraphAddEventWaitNode` で後付け | node は増えるが replay で待たない |
| peer への push、host が読む | 一致 |
| peer への push、peer の kernel が読む（dirty line なし） | 一致 |
| host-mapped flag の相互 barrier（device flag 不可） | 一致 |
| **graph 内に [epoch bump][push][spin + sum] を capture** | **200 replay で不一致 0、73.5 µs/step（10 KB、2 rank）** |

`full_graph_sync_probe.hip` の graph は各 rank 3 node で、epoch は device 側の
カウンタを kernel が進める。capture した kernel 引数は固定なので、epoch を graph に
焼き込まず device memory から読む必要がある。これで replay を繰り返しても待機条件が
世代ずれしない。

## shelf への含意

- cross-device event を graph に閉じ込める方式は、この ROCm では node が落ちるため不可。
- 代わりに **collective を 1 kernel に融合し、host-mapped flag と device epoch で
  GPU 側 barrier を張る** 方式が成立する。push は peer の device memory へ行い、
  受け手は自分の inbox を読む（現行 transport と同じデータ経路）。
- これが入ると `TpBarrierGroup::arrive` の CPU 待機が不要になり、program 全体を
  1 graph に capture できる。そのためには coordinator が全 rank の graph 起動を
  投入してから完了を待つ必要がある（1 rank だけ起動に失敗すると GPU 側で待ち続ける）。
- flag は必ず host-mapped memory に置く。device memory の flag は完了まで store が
  見えず相互 barrier がデッドロックする。

## 未着手

shelf 本体への統合（collective kernel の実装、`TpBarrierGroup` の GPU 経路、
coordinator の graph 起動、graph 障害時の全 rank 停止）は未着手。本記録は
プリミティブが成立することの確認まで。
