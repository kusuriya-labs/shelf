# PR #30 の rotate_output null 参照報告の調査

PR #30 へのレビューコメント（issue comment 7006502496）への対応記録である。
症状は、R9700×3 の EP=3 で chunk=1024・32 token の `hy4 generate` 実行中に
`rotate_output`（`src/shelf/models/hy4_flash/linear.hip`）がアドレス `0x0` を読み、
約 8 分の queue リトライの後にデバイスを失うというもの。WMMA 統一（cda6884e）前でも
同位置で同症状と報告されている。chunk=8・4 token（EP=3 / EP=4）は完走し、
生成 ID は `499,20663,9086,9629`。

## 静的調査の結論

fault アドレス `0x0` は `svh[base + i]`（`base + i` は高々 1535）の読み出し、
すなわち射影表エントリの `svh` が NULL の場合に一致する。書き込みではなく
`group[routes[blockIdx.y].expert * stride].svh` の読み出し経路である。

以下を確認し、PR head（85aec3be）の静的な範囲では表がゼロのまま参照される
経路を見つけられなかった。

- 配置の一貫性: `upload_experts` と `moe` の経路振り分けは同じ
  `model.owners` を参照する。`reserve` の再配置（`model.relocate`）は
  重みの rank・offset・owners を同時に更新するため、転送後の表と
  実行時の振り分けは一致する。
- `choose` は bitonic 整列で `ids` を交換するだけであり、NaN が混入しても
  `ids` は [0, 512) の置換のまま出る。行数に依らない。
- stream 順序: rank r の H2D（pinned `route_host[r]`）は rank r の
  default stream でカーネルより前に並ぶ。次層の `route_host[r]` 再書き込みは
  `moe` 先頭の D2H 同期（device 0）→ peer 転送の event 依存
  （`ready[r][0]` は rank r の `weighted` 以降）→ 同期完了、
  の順でしか起きないため、実行前の上書きは起きない。
- `route_buffer` の容量は chunk 1024・全経路 1 rank 集中の最悪値
  （routes 196608 + offsets 4100 + undo 65536 + batches 131072 = 397316 バイト）
  とちょうど一致し、超過時は例外になる。
- `moe` 経路の全カーネル（`rotate_input`、`rotate_output`、`gather_route`、
  `weighted`、`expert_batch` と `group_body`、`expert_batch16`、`multiply`）の
  読み書き添字は確保容量内に収まる。`fold`・`save_records`・`pool` など
  rank 0 の cache/index/folded 書き込みも同様。
- arena 超過は `Region::next` の例外になり、GPU fault にはならない。

## hy4-agent-integrated との差分と実測

fork（hy4-agent-integrated、R9700 + V620×2）との意味のある差分は
`moe.hip` の 2 点のみである。`attention.hip`・`ops.hip`・`mla_prefill.hip`・
`wmma.hip`・`residency.cpp` の転送・`route_buffer`・`rotate_*` カーネルは同一である。

1. fork は `x.rows >= 8` のときのみ expert を grouping し、未満は per-row の
   `multiply` 経路を使う。PR head は全行で grouping する。すなわち PR head のみが
   decode（rows 1〜6）でも grouped expert 経路を通る。
2. fork は rank 0 のみ WMMA を使い、rank 1〜2（V620, gfx1030）は
   `expert_batch16` を使う。PR head は全 rank で WMMA を使う。
   これは R9700×3 固有の経路であり、手元の構成では再現できない。

fork と同条件（EP=3、chunk=1024）の実測（shelf-qualification/ep3-9ao7j7r7）:

- pp2048 は 3.01〜3.83 秒（676.62 tok/s、中央値）。
- decode は 50.65 tok/s（DFly 込み、127 token、4 反復で同一列）。
- 255 token の target-only 生成も完走し、同一入力で DFly と通常生成の列が一致。

つまり同一のコード経路（grouping する rows>=8 の prefill・peer 転送・rotate カーネル）
は手元の 3 GPU で 2048 token prefill・長い decode でも異常がない。

## 仮説（未検証）

1. PR head のみが通る「rows < 8 での grouped expert 経路」（decode・DFly verify・
   propose）に条件依存の不具合がある。レビュアーの失敗条件は decode 30 超、
   成功条件は 3 ステップと報告されており、行数仮説（prefill 24 行）と並ぶ候補。
   後述の実機試験で改変版の decode は完走しており、本地の裏付けは得られていない。
2. R9700×3 固有の全 rank WMMA 経路の問題。V620 を持たないため手元では検証できない。
3. レビュアー環境の要因。成功時の prefill 65〜90 秒・decode 23〜29 秒/token は
   同じコード系の手元実測（676 tok/s・50.65 tok/s）とけた違いに遅く、
   コードだけでは説明できない。P2P の実破壊、VRAM 圧迫、Engram の冷読み、
   または起動時間（重みと Engram の取り込み）を含めた計測の可能性がある。
   P2P の実破壊があれば peer 転送が対向 rank の作業域に garbage を書き、
   射影表や route バッファの破壊として現れ得る（fault 位置・rank 非依存・
   再現の揺らぎと整合する）が、観測がないため仮説とする。

## 実機再現の試みと結果

手元の構成で再現を試みた。fork に「常時 grouping」の 1 行変更を加えた
worktree（fork 96b730b5 相当 + `moe.hip` の grouping 条件撤廃）をビルドし、
R9700 + V620×2 の EP=3（peer 転送 4 経路とも method=peer、relocate 後の
expert 数は 6209/7392/7391、weight_upload 約 20〜25 秒）で次を実行した。
lock は 1 コマンドごとに r9700 → v620 → v620-76a08 の順に flock で取得し、
timeout を付けた。

| 試験 | 条件 | 結果 |
|---|---|---|
| T1 | 改変版・入力 24 token・chunk=1024・32 新 token | 完走。prefill 24 行 2.60 秒、decode 31 ステップ 3.68 秒 |
| T2 | 改変版・同入力・chunk=8・4 新 token | 完走。生成 ID は T1 と 1 token 目のみ一致 |
| T3 | 改変版・入力 4 token（CPU 参照と同一）・chunk=1024・32 新 token | 完走。最初の生成 ID は 12286 で CPU 参照と一致 |
| T4 | 改変版・入力 24 token・chunk=1024・32 新 token・DFly | 完走。draft_rounds=12、accepted=21 |
| T5 | stock fork・入力 24 token・chunk=1024・32 新 token | 完走。decode 2.35 秒 / 31 token |

いずれも illegal memory access は発生せず、レビュアーの失敗条件に対応する
「1 回の prefill 行数 24」仮説と「decode 30 ステップ超」仮説のどちらでも
手元の構成では再現しなかった。残る差分は仮説 2（R9700×3 固有の全 rank WMMA
経路）と仮説 3（レビュアー環境要因）に絞られる。

なお、改変版（常時 grouping）の decode は 3.68 秒 / 31 token（119 ms/token）で、
stock fork（rows<8 は per-row 経路）の 2.35 秒 / 31 token に対して約 1.6 倍遅い。
chunk=8 と chunk=1024 の生成 ID の食い違いは、prefill の attention 経路差
（rows>=16 の mla_prefill tiles 経路とそれ未満の absorbed 経路）による
数値差と考えられ、実行の成否とは無関係だった。

実行コマンドの骨子:

```bash
ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55,GPU-08b2ddcbd6e6b36c,GPU-76a08c022586fed6
# 入力: prompt24.ids = 120000 802 5466 19405 63357 52392 1278 269 34262 8988 13 \
#   220 96698 1472 114739 18854 5059 2712 9224 6911 320 6636 17243 29041 (24 token)
flock /tmp/r9700.lock flock /tmp/v620.lock flock /tmp/v620-76a08.lock \
  -c "timeout 600 <build>/shelf-compute hy4 generate \
      --model-dir .../Hy4-flash-exl3-4bpw-mul24 --engram-dir .../body \
      --placement .../owners-4bpw.json --max-seq-len 1048576 --temperature 0 \
      --input-ids-file prompt24.ids --max-new-tokens 32 --chunk 1024"
# 対照: --chunk 8 --max-new 4 / 入力 4 token + --max-new 32 / --dfly-dir 付き
```

## 対策（本 commit）

原因は特定できていないが、観測された故障様式（射影表のゼロエントリ参照による
GPU fault → 8 分のリトライ → デバイス喪失）を起動時とホスト側で即検出する
防衛を入れた。

- `moe` の経路振り分けと grouping を `include/shelf/models/hy4_flash/routes.h` の
  `partition_routes` / `group_routes` に切り出し、行数・rank・expert・重み・
  重複・配置の一貫性を検証する。kernel が不正な値を出した場合も
  GPU fault の前に例外で停止する。
- `reserve` の配置表検査に負の rank を追加した。
- `upload_experts` に `validate_expert_table` を追加した。構築した表が
  配置表と一致しない（所有 expert のエントリ欠落・他 rank の expert 書き込み）場合、
  転送前に例外で停止する。ゼロ初期化のまま参照される表が存在すれば
  起動直後に失敗し、8 分のリトライ待ちを経由しない。
- 単体テスト `tests/unit/test_hy4_routes.cpp` を追加した。
  1・8・24・1024 行、16384 経路の 1 rank 集中、不正入力の拒否、
  表の一貫性検査（所有 expert の svh 欠落・他 rank のエントリ・長さ違反）を含む。
  AddressSanitizer / UBSan でも異常なし。
