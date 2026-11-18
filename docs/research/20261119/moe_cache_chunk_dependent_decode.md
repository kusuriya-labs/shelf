# prefill chunk で decode 速度が変わる不具合（expert cache の再配置持ち越し）

## 症状

`shelf-compute` 単一リクエスト（`models/Qwen3.8-Flash-Next-Quark-MXFP4`、TP=1、
`--moe-host-experts 448`、同一 token 4096 個の prompt、`--max-new-tokens 64`、
`--max-seq-len 8192`、`--kv-cache-dtype octave:c5v5`）で `--prefill-chunk` を
変えると decode 速度が変わる。修正前の実測:

| prefill-chunk | prefill ms | decode tok/s |
| ---: | ---: | ---: |
| 512 | 9,292.4 | 22.18 |
| 1024 | 6,680.1 | 16.41 |
| 2048 | 4,635.8 | 14.52 |
| 4096 | 3,654.3 | 14.30 |

prefill は page cache の状態で変わるため、以下では decode の差だけを問題扱いする。

## 観察（ID: A）

`SHELF_MOE_CACHE_LOG_INTERVAL=8` で expert cache の log を取ると、再配置計画の
消化が prefill 中に終わるかどうかが chunk で分かれる。

- chunk=512（prefill 9 step）: step=8 時点で swaps=2,919、decode 中は 1 step あたり
  0〜1 本。hit（累積）54%。
- chunk=4096（prefill 2 step）: step=8 で swaps=780 が step=56 で swaps=2,465 まで
  **decode 中に 1 step あたり約 35 本**の交換が続く。hit（累積）11.4%。

1 交換は gate/up/down の 3 plane それぞれで demote（D2H、pageable staging 経由）と
promote（H2D）を sync して行う（1 交換あたり sync 6 回、重みの往復は約 5 MiB、
2026-11-18 の記録と同じ expert 形状）。chunk=4096 の decode は修正前 73.1 ms/token、
計画を prefill 中に消化した構成では 43.3 ms/token で、差の約 30 ms は 35 本 ×
0.85 ms 程度の交換オーバーヘッドと整合する。

## 因果の切り分け（ID: B）

- 当時は batched step の消化上限が `max(SHELF_MOE_CACHE_BURST_SWAPS, SHELF_MOE_CACHE_SWAP_BUDGET)`
  だったため、`SHELF_MOE_CACHE_SWAP_BUDGET=4096` を設定すると batched step が計画を
  一気に消化する。この構成では chunk=4096 の decode は **23.08 tok/s** まで回復する。
  この run は既定 run と同じ交換集合（2,474 本）を同じ在籍配置へ到達させた条件で、
  違うのは交換を実行する時間帯だけである。したがって本症状の直接の律速は
  **decode 中に走る交換活動**であり、在籍配置の良し悪しではない。定常 decode の
  miss 率が per-token に効くこと（2026-11-18 の host offload 測定）とは別の軸である。
- chunk=512（row bucket 512、working set reserve 0.70 GiB）と消化済み chunk=4096
  （row bucket 4096、reserve 2.14 GiB）の decode は 22.34 と 23.08 tok/s で等しい。
  prefill chunk が変える row bucket・executor workspace・KV 余白のサイズ差は decode
  速度には効いていない。

## 原因

`MoeExpertCache::step` は再配置計画を 1 step あたりの本数上限で消化していた。
batched step（route 数が 層×local expert を超える）は `SHELF_MOE_CACHE_BURST_SWAPS`
（既定 512）、それ以外は `SHELF_MOE_CACHE_SWAP_BUDGET`（既定 96）と PCIe 予算である。
消化速度が step 数に結合しているため、chunk を大きくして prefill の step 数が減ると
計画（約 2,500 交換 ≒ 12 GiB の PCIe 往復）が prefill 中に終わらず decode に持ち越され、
decode の critical path で sync 6 回 × 約 35 本/step の交換が走る。chunk=512 では
9 step × 512 本で偶然完了しており、症状は chunk を大きくしたときだけ現れた。

## 修正

route 数が 層×local expert を超える batch では計画をその step で全数消化する。
それ以外の step の消化（96 本 + PCIe 予算）は従来どおり。
`SHELF_MOE_CACHE_BURST_SWAPS`（CLI は `--moe-cache-burst`）は上限として意味を
持たなくなるため削除した。

### 修正後の再計測

同じ条件の 4 chunk（各 1 run、`SHELF_MOE_CACHE_LOG_INTERVAL=8`、device 0）:

| prefill-chunk | prefill ms | decode tok/s | decode 中の swaps 増分 |
| ---: | ---: | ---: | ---: |
| 512 | 10,268.5 | 22.30 | +17 |
| 1024 | 7,652.6 | 22.53 | +6 |
| 2048 | 5,707.2 | 22.56 | +8 |
| 4096 | 5,103.4 | 23.04 | +0 |

decode 中の swaps 増分は log 区間（step 8 から run 末尾）の差である。修正前の
chunk=4096 は同じ区間で +1,685 だった。残る数本は `periodic` 再評価（4 step 間隔）が
decode 中に見つける交換で、小分け消化（budget 96 + PCIe 予算）の範囲で収まっている。

- chunk=4096 の prefill は同一 session の修正前 3,564.9 ms から 5,103.4 ms になる。
  増分は decode が担っていた 2,474 交換（重みの PCIe 往復約 12 GiB + sync）を prefill 側が
  引き取った一括コストで、E2E 合計は 7,805 ms から 7,621 ms へ短くなる。
- `GENERATED_IDS` は修正前後で bitwise 一致する（確認したのは手元に修正前の出力が
  ある chunk 512 と 4096 の組。512 は 58 token、4096 は 59 token。修正後の
  512/1024/2048 は同一列で、4096 だけ 1 token 多いが、これは修正前の
  512 対 4096 でも同じ差であり、交換のタイミングとは無関係）。交換は同じ重みの中身を
  別の slot に動かすだけであり、生成結果に影響しない。
- 修正後の累積 hit は chunk 512/1024 で 63%、2048 で 46%、4096 で 12% と chunk で
  大きく違うが、decode はどれも 22.3〜23.0 tok/s で等しい。累積 hit は prefill の
  routing 分布で決まる指標であり、今回の律速では無い。
- `tests/test_qwen4exp_load --skip-ple --rank 0 --device 0`（`SHELF_MOE_HOST_EXPERTS=224`、
  この test は tp_size=2 で読むため host は 224 まで）は PASS。

## 採否

- ACCEPT: batched step での計画全数消化。再配置は prefill 相当の batch に載せて完了し、
  decode step へ持ち越さない。chunk 依存が消え、どの chunk でも 22.3〜23.0 tok/s になる。
- REJECT: batched step の本数上限を保ったまま既定値を上げる方向。上限 < 計画サイズの
  ときに同じ持ち越しが再発する。既定 512 本はまさにその状態だった。
- 保留: 交換 1 本あたり sync 6 回と pageable staging の削減（pinned scratch +
  device 側 copy-back による非同期化）。prefill の一括コストのうち sync/staging が
  重み往復 0.5 s に対して約 1 s を占める。2026-11-18 の記録に挙げた HIP 側の制約
  （DMA 書き込みの CPU 可視性など）を踏まえ、別の実験として検証する。
- 未検証: この構成では prefill で累積 routes が大きく伸びるため、decode 中に
  route 閾値（`due`）へは届かず、再評価は `periodic`（step 間隔）経由だけになる。
  累積頻度が prefill の routing で支配されているため採る交換はほぼ無く、decode の
  routing 分布に合わせた本格的な再配置は起こらない（閾値設計は tg のような decode
  中心の stream を想定している）。decode 定常状態の miss 率を下げる方向は本修正の
  対象外とする。
