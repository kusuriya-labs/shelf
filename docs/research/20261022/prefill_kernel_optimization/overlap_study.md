# TP=2 通信・計算オーバーラップの先行調査（2026-10-22）

目的: 「両者アイドル 37.6ms + peer 待ち 15.1ms」を削れるかを、実装に入る前に
構造分析と trace 帯域で判断する。CPU 非関与は別計測で確定済み
（experiments.md 付録。CPU 側 TP Barrier は prefill 経路で不使用、transport の
CPU 投入は計 6.8ms、同期 API 0 回）。

## 1. 現行の依存グラフ（hip_peer_tp_transport.hip 実測）

- rank 毎に **計算用ストリーム1本**。集合点（sum）はその stream へ順次投入:
  `送信コピー(inbox[q←r]) → hipEventRecord(sent_[r][q]) → hipStreamWaitEvent(
  sent_[q][r]) → tp_add(受信 inbox を加算) → 次セグメントの計算`
- 受信待ち・送信・加算はすべて **同一 stream 上の直列キュー**。
- `parity_` による inbox 二重化は存在（世代管理は確保済み）。
- CPU は submit 一括（36ms）で全依存を投入済み → 以降の進行は GPU の stream 順のみ。

### 構造上のボトルネック

1. **送信は自 rank の計算の背後にしか実行されない**（同一 stream のため、
   直前の計算が終わるまで送信コピーが走れない）。
2. **受信待ちは stream 全体を停止させる**: `hipStreamWaitEvent` の後ろに
   enqueue された次セグメントの計算が、peer の送信完了まで一切実行されない。
3. 結果として両 rank は集合点で同期し、**通信（送信コピー・加算）は常に
   計算と直列**。加算結果を次の計算が要求する依存自体は正当だが、
   「自 rank の残り計算」と「peer への送信」は重ねられるはず（依存なし）。

## 2. Trace での停滞帰属（G9 適用後、0.3ms 以上のギャップ、両 agent 計 103.9ms）

| 前 → 後 | 時間 | 件数 | 解釈 |
|---|---:|---:|---|
| copyBuffer → fillBuffer | 37.0ms | 1 | **ロード/初期化領域の混入**（n=1、steady-state から除外して解釈すべきもの） |
| tp_add → tp_copy | 12.1ms | 14 | 集合点間の空き: peer 依存で stream が詰まった状態 |
| gemm → (bf16 kernel) | 6.7ms | 2 | チャンク境界の rank ずれ |
| rmsnorm → gemm | 5.4ms | 2 | チャンク境界の rank ずれ |
| **tp_copy → tp_add** | **4.2ms** | 10 | **受信待ち辺（peer の送信イベント待ち）そのもの** |
| fillBuffer → copyBuffer | 5.4ms | 2 | ロード領域 |
| その他 小 | 各 1〜3 | — | レイヤ境界のずれ |

steady-state の停滞（n≥2 の計算隣接分 + 通信隣接分）は合計 ≈52ms で、
rank0 gap 分解（peer 待ち 15.1 + 両者アイドル 37.6）と整合。
**すべて通信依存辺・rank ずれ由来であり、CPU 由来はゼロ。**

## 3. 頭打ち効果の見積もり

- 集合点は enq_sum 141 / sum_hidden 129 回（= 層あたり最大2回 × 64層 ≈ 128 と整合）。
- tp_copy 実測 95.2ms ÷ 29.1GB/s ≒ **2.7GB ≒ 141 × 約19MB**（prefill hidden
  [2048×5120×2B = 20.9MB] の all-gather 相当、bytes は TP 方式変更なしでは不変）。
- 停滞側の ~52ms は「送信が自 rank の計算の背後にしか並ばない」「受信待ちが
  stream 全体を止める」構造から生じており、**構造を変えればピークで ~50ms の
  削減余地**（peer 待ち 15.1 + 両者アイドル 37.6 の大半）。
- 現在の残りは 51.09ms（551.09 → 500）。**重ね合わせが実現できれば、500ms は
  数値上ちょうど届く射程**。

## 4. 構成案（効果順・リスク順）

### 案 A1: 通信用ストリーム分離（最小形）

- rank 毎に comm stream を追加し、集合点の `送信コピー → wait → add` を comm
  stream へ、計算は compute stream に残す。データ依存はイベントで明示化
  （producer 計算完了 → comm、add 完了 → 次計算）。
- 効果: 自 rank の次の計算が受信待ちの背後で止まらない（skew 分は残るが、
  通信待ちの大半を計算に重ねられる）。
- リスク: `TpSumInvocation.streams` の意味論変更、peer inbox の世代管理は
  parity で保護済みだが順序保証の再検証が必要。**実行モデルの変更 = 要承認**。
- 想定規模: transport（stream 保持・イベント追加）+ セグメント契約の微修正。
  dispatch 数・演算内容・依存の意味論は不変。

### 案 A2: チャンク単位のパイプライン

- 32 チャンクの境界で、rank0 のチャンク k+1 計算と rank1 のチャンク k の
  受信処理を重ねる（A1 を土台に分割点をチャンク境界に置く）。
- 効果は A1 以上だが、プランナー側の分割契約が広く変わる（大規模・要承認）。

### 案 A3: add 完了の部分先行（非推奨）

- 加算結果の必要範囲が細かく分割できるかは op 依存の粒度次第。現プランでは
  hidden 全体を次の op が消費するため、見込みは小さい。

## 5. 結論と推奨

1. 現状の 52.7ms（peer 15.1 + 両者 37.6）は**すべて通信依存構造由来**で、
   CPU 由来ではない（計測済み）。
2. 案 A1 で通信を計算に重ねられれば、**削減余地は最大 ~50ms** で、
   現在の残り 51.09ms とほぼ同規模 → **500ms 到達の唯一の現実的な経路**。
3. A1 は「依存の意味論・演算・dispatch 数」を変えず投入タイミングと stream
   構成のみを変えるが、実行モデル（stream/イベント契約）の変更であるため
   **スコープ外事項として承認を得てから着手する**。
4. 承認前の過渡策としてはなし（kernel 個別最適化では待ちは縮まない =
   G1-b の +5.4% 等で実証済み）。

**推奨: A1 の設計承認を求める。** 承認時は transport と
`TpSumInvocation` の stream 契約を最小差分で再設計し、まず集合点1箇所
（qkv 後）で pilot 実装 → interleaved E2E で検証する手順とする。

## 6. 参照実装（libr4d / radiance）の確認結果（2026-10-22 追記）

参照: `~/work/references/libr4d`（正本 `radiance`）、`r4d_ar_oneshot_2rank_exact.hip`、
`radiance/docs/TP4.md`。

### 2 ランク AllReduce の参照方式（exact）

- **1 カーネルの PUSH モデル**: 各 rank が自値を peer の IPC scratch へ書き込み、
  **ブロック毎の flag ハンドシェイク**（release 書込み + spin acquire、system scope）を
  経て、自値＋peer の scratch を同じカーネル内で reduce → out。
  → PhaseShift の「copy + イベント + add（2 dispatch）」に対し参照は **1 dispatch + device 内 spin**。
- scratch は **fine-grained（uncached、system スコープ公開で L2 writeback 不要、
  `s_wait_storecnt` drain）＋ seq&1 の二重バッファ**、seq はデバイス側 per-block カウンタ
 （graph / SPMD 安全）。multi-block で PCIe（push）と HBM（reduce）をそれぞれ飽和させる設計。

### 参照側で既に測って潰している項目（うち本件に直結するもの）

| 参照の項目 | 結果 | 我々への含意 |
|---|---|---|
| launch 数削減・launch fusion（W17 等） | **launch は 4µs floor、trunk submit は backpressure（device-bound、host は隠れる）→ 意味なし** | ⑦-2 HIP Graph・enqueue 削減は参照自身が価値なしと測定済み |
| AR grid 拡大（W18） | GB/s が下がる（handshake-bound） | grid/並列度レバーは死 |
| codec・帯域（row 14） | **push が既に PCIe ラインレート飽和、「22% タックスは物理」** | x8 で我々の tp_copy 飽和と同根。bytes・帯域側は手詰まり |
| lossy wire（wht6, 6bit） | −7.6〜8.3% と速いが非 exact | 量子化変更 = 禁止のため対象外 |
| **prefill-span 2（chunk 単位で half-chunk の AR を計算と重ねる）** | **+5〜11% で SHIPPED** | **重ね合わせはチャンク粒度で成立（案 A2 相当）。単一 collective 内の stream 分離（案 A1）とは別物** |

### ⑦ への回答（本件の結論）

1. peer_write 経路の正しくは参照が持つ「push をカーネルに埋め込む」形。我の現状
   （pw 常時 false・copy+add の2 dispatch）は、参照に比べ **handshake の受け渡し 1 段と
   dispatch 1 つ多い**構造。交換は「TP 方式＋dispatch 数の変更」= 要承認事項。
   見込み: in-kernel の chunk パイプラインで kernel 間 gap（tp_copy→tp_add 4.2ms）と
   add→copy 間の詰まりを吸収 → 保守 **−5〜15ms**（push 帯域は x8 物理のため飽和は残る）。
2. enqueue 数そのものの削減・HIP Graph は、**参照が 4µs floor / backpressure で
   既に否定**。⑦-2 は優先度を下げてよい。
3. 最大の重ね合わせは参照が実用化した **chunk-span overlap（案 A2）**（+5〜11% 相当 =
   我々の規模で +30〜60ms の可能性）だが、prefill を半チャンク分割するプランナー変更で
   より大きなスコープ変更になる。
