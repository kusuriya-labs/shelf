# TP all-reduce の実装比較（shelf-compute と HY4-flash）

対象:

- shelf-compute: `src/shelf/runtime/tp/hip_peer_tp_transport.hip` の
  `HipPeerTpTransport::enqueue_sum` と `TpBarrierGroup::arrive`
- HY4-flash: `.worktrees/hy4-r9700-ep-opt/src/shelf/models/hy4_flash/residency.cpp` の
  `Residency::reduce_pair` / `ep_reduce`

## まとめ

| 項目 | shelf-compute TP | HY4 EP / TP |
| --- | --- | --- |
| 転送方向 | push 固定（送り手の stream で相手の inbox へ書く） | `SHELF_HY4_EP_TRANSPORT` で push / peer(pull) / host を選択 |
| 転送の起動 | 最後に `arrive` した 1 スレッドが全 rank 分を enqueue | ペア先頭 rank が両方向分を enqueue |
| 同期 | グループ mutex + condition_variable の CPU barrier | CPU barrier（`arrive_and_wait`）を前後に 2 回 |
| event | 方向ごとに 2 本（送信・受信）。parity で 2 面 | 方向ごとに 4 本（ready / done / arrived / consumed） |
| 二重バッファ | parity 2 面。**consumed 待ちなし** | parity 2 面 + 前回消費の `consumed` 待ち |
| 合成 | 融合 kernel（sum + residual / hyper を 1 launch） | `tp_add_bf16`（単純加算）→ 後段で norm / add |
| dtype | BF16 / F32 | BF16 |
| 起動時検査 | `verify()` で一括プローブ | 方向ごとに `probe_peer_direction`、失敗方向だけ host へ |
| graph | segment graph / GPU barrier を実験中 | 対象外 |

## 実装の読み方

shelf の push 経路は、ある rank r の combine 値 `targets[r].ptr` を、各 peer q の
`inbox_[q][r][parity]` へ送る。送信は `enqueue_sum` が r の stream に copy kernel を
積み、`sent_[r][q]` を record する。受信側は peer の `sent_` を wait してから、
自分の inbox を読む `launch_add` を積む。event は方向ごとに 1 本ずつで足りる。

HY4 はペア (a, b) を作り、a が両方向のコピーを起動する。push では送信側 stream で
`recv[destination]` に書き、`done` を record、受信側は `done` を wait する。pull
（`peer`）では受信側 stream が送信元の `tp_ptr[source]` から直接 copy する。
どちらのモードでも、書き込み前に受信側の前回 `consumed` を待つ。

## 差分として重要な点

1. **上書き防止の差**
   HY4 は push の直前に `consumed_events[destination][p ^ 1]` を待つ。shelf は parity を
   交互に使うだけで、前回の読み終わりを待たない。CPU barrier は host しか止めないため、
   rank q が 2 つ先の barrier で同じ parity の inbox へ push し、rank r の前回 add が
   まだ読んでいる間に上書きする経路が理屈上ある。HY4 の `consumed` に相当する event を
   shelf にも入れるべき。

2. **転送方向の実測差**
   HY4 の測定では push が peer(pull) より tg で 7〜15 % 速く、pp では差がない。
   shelf は既に push 固定なので、方向としては速い側にいる。pull へ変える理由はない。

3. **event のスコープ**
   HY4 は ready / ack を `hipEventReleaseToSystem` 付きで作る。shelf は
   `hipEventDisableTiming` のみ。peer の書き込み可視性を event の完了に依存させるなら、
   HY4 側の指定に寄せる方が意図が明確。

4. **合成の差**
   shelf は reduce と residual / hyper を 1 kernel に融合している。HY4 は単純加算の後で
   別途 norm / add を積むため、同じ段数でも launch 数が多い。shelf の方が有利。

5. **障害時の粒度**
   HY4 は方向ごとに probe し、失敗した方向だけ host 経路へ落とす。shelf は transport
   単位の verify なので、片方向だけ壊れた場合の扱いが粗い。

## 参考: HY4 の測定（ユーザー提示、EP=4、単一リクエスト、p50）

| 転送 | DFly2 | tg ctx 128 | tg ctx 2048 | pp 2048 | pp 8192 |
| --- | --- | ---: | ---: | ---: | ---: |
| push | あり | 62.4 | 49.3 | 1,214 | 975 |
| peer | あり | 57.7 | 45.8 | 1,211 | 974 |
| push | なし | 40.6 | 34.5 | 1,205 | 976 |
| peer | なし | 34.7 | 29.2 | 1,200 | 974 |

## 次に shelf へ入れる候補

1. push の前に前回同一 parity の `consumed` event を待つ（HY4 と同じ保護）。
2. 送受信 event を `hipEventReleaseToSystem` 相当にする。
3. 方向別の peer probe と host フォールバック（現行は transport 単位）。
