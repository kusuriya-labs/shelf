# メモリ予算の内訳とムダ（Qwen3.8-Flash-Next EXL3 / TP=2）

実験 ID: `MEM-AUDIT-1`

対象: `models/Qwen3.8-Flash-Next-exl3` + `models/Qwen3.8-Flash-Next-DFlash2`、
R9700 2 枚（各 31.86 GiB）、TP=2、FP8 KV、page 16、prefix cache off、
`--max-concurrent-requests 4`、`--dflash2-drafts 6`。

結論:

- 予算の計算そのものは executor の実測確保量と ±60 MiB（1〜4 %）で一致した。
  割当順の破綻も二重計上も無い。差はすべて「推定値に対する安全率（slack）」である。
- **修正済み**: DFlash2 の `session_slot` が GDN recurrent history を
  `history_rows` 分だけ余計に予約していた（実確保は conv のみ）。
  4 session で 0.65 GiB の過大予約だった。
- 残るムダは 2 つ。`slack = 推定値/3` が 0.36〜0.49 GiB、EXL3 の固定 workspace が
  Flash-Next には 0.19 GiB 過大、accum が 0.055 GiB 過大。
  合計 0.61 GiB で、これは 4 slot では約 70,000 token 分に相当する。
- **実装済み**: slack を `min(max(64 MiB, 推定値/8), 256 MiB)` にし、EXL3 の
  staging plane を rank の実形状で確保する。131072 / 2048 行では reserve が
  1.96 → 1.31 GiB、executor が 1.48 → 1.17 GiB になり、262144 は 2048 行で
  54,704 → 180,128 token、512 行以下では起動できるようになった（第 8 節）。
- 逆に **過小** な項が 1 つ。DFlash2 の draft scratch が proposal logits を
  `block_size` 行で見積もっているが、実行時は `max_rows`（= 4 session で 32 行）を
  確保する。24 MB の不足で、slack に吸収されている。
- drafter の weight 予約（safetensors のファイルサイズ）は arena 実測と 9 KB 差で正確。

## 1. 予算の全体像

以下は修正前の測定である。修正後の値は第 8 節に置く。

起動時に stderr へ出る `MEMORY_PLAN` の値と、executor 生成後の
`executor footprint` を突き合わせた。`reserve` は slack を含む合計、
`estimate` は slack を除いた計算値である。

| rows | conc | max_seq | reserve 合計 | estimate | slack | executor 実測 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 4 | 131072 | 1.01 GiB | 0.75 GiB | 0.25 GiB | 0.78 GiB |
| 512 | 4 | 131072 | 1.23 GiB | 0.92 GiB | 0.31 GiB | 0.95 GiB |
| 1024 | 4 | 131072 | 1.48 GiB | 1.10 GiB | 0.37 GiB | 1.10〜1.16 GiB |
| 2048 | 4 | 131072 | 1.96 GiB | 1.47 GiB | 0.49 GiB | 1.48 GiB |
| 2048 | 1 | 131072 | 1.55 GiB | 1.19 GiB | 0.40 GiB | 1.17 GiB |
| 2048 | 4 | 524288 | 3.49 GiB | 2.62 GiB | 0.87 GiB | 未到達（KV 不足） |

estimate と実測の差は全ケースで −20〜+60 MiB（1〜4 %）である。
slack は estimate の 1/3（合計の 25 %）で、**推定誤差の 4〜8 倍を常時確保している**。

weights は 26.32 GiB/rank、weights 後の空きは 4.31 GiB/rank。
4 slot の FP8 KV と QSA state は 1 token あたり 9,312 B（KV 6,240 + QSA 3,072）なので、
1 GiB の予約は約 115,000 token に相当する。

## 2. 内訳（修正前: rows=2048 / conc=4 / max_seq=131072、estimate 1,503 MiB）

| 項 | サイズ | 割合 | 実体 |
| --- | ---: | ---: | --- |
| program values（dispatch / activation workspace） | 390 MiB | 26 % | `values_per_row=39 × rows × hidden` の近似。実測と一致 |
| QSA indexer state pool（4 slot） | 384 MiB | 26 % | `12 layer × max_seq/4 blocks × 128 × bf16` × slot |
| EXL3 固定 workspace | 272 MiB | 18 % | xrot 80 MiB + dense 192 MiB（下記） |
| MoE group-row workspace | 187 MiB | 12 % | `rows × top_k` 行 ×（up/gate + hidden） |
| EXL3 accum | 136 MiB | 9 % | `rows × 17,408 × f32`（下記） |
| hidden tap（5 tap + token_hidden） | 60 MiB | 4 % | `(1 + 5) × rows × hidden` |
| logits / output rows | 30 MiB | 2 % | `MO=32` 行分。`rows` には比例しない |
| scratch + decode attn partial | 44 MiB | 3 % | 定数 |
| slack（予約外の上乗せ） | 502 MiB | +33 % | `max(64 MiB, estimate/3)` |

## 3. row bucket

`--max-num-batched-tokens N` が executor の `max_batch_rows` と row bucket を決め、
`Npad` へ切り上げた行数で次のバッファが比例する。

- program values、MoE group-row、EXL3 accum、hidden tap、token_hidden、batch context

実測の executor footprint:

| rows | executor 実測 | 内訳の増分（rows=32 比） |
| ---: | ---: | ---: |
| 32 | 0.78 GiB | — |
| 256 | 0.86 GiB | +0.08 GiB |
| 512 | 0.95 GiB | +0.17 GiB |
| 1024 | 1.10〜1.16 GiB | +0.32〜0.38 GiB |
| 2048 | 1.48 GiB | +0.70 GiB |

decode（DFlash2 verify は 4 session × block 8 = 32 行）だけなら 32 行で足りるが、
prefill は 1 step で `max_num_batched_tokens` 行を流すため 2048 行が必要になる。
この 0.70 GiB は「prefill throughput と引き換えのコスト」であり、
decode 専用構成にするなら削れる。shelf-server の既定は 2048。

## 4. 展開済み重みの一時保管（EXL3 workspace）

`exl3_workspace_bytes(rows)` はモデル非依存の定数で、27B dense の形状を前提にしている。

| バッファ | 定数 | 27B で必要なサイズ | Flash-Next で必要なサイズ |
| --- | ---: | ---: | ---: |
| dense（行列 1 個を fp16 展開） | 192 MiB | 17,408 × 5,120 × 2 = 178 MB | 10,240 × 2,560 × 2 = **50 MiB** |
| xrot（回転済み活性） | 80 MiB | 2,048 × 17,408 × 2 = 71 MB | 2,048 × 6,144 × 2 = **24 MiB** |
| accum（split-K、rows 比例） | rows × 17,408 × 4 | 2048 行で 136 MiB | 2048 行で **80 MiB** |

- Flash-Next の最大 dense 行列は `linear_attn.in_proj_qkv`（n=10,240、k=2,560）。
- 最大 k は `linear_attn.out_proj` の 6,144。
- 最大 out_features は 10,240（lm_head は fused W16A16 経路なので dense を使わない）。
- 固定分で 272 − 74 = **198 MiB**、2048 行の accum で 136 − 80 = **56 MiB** が未使用。
  合計 254 MiB/rank。

## 5. drafter 側

| 項 | 予約 | 実測 | 判定 |
| --- | ---: | ---: | --- |
| draft weights | 1.123 GiB | 1,206,131,200 B（arena 実測） | 一致（ファイルサイズ 1,206,140,008 B） |
| session slot（修正後） | 0.08 GiB/slot | 336,879,432 B / 4 = 0.078 GiB/slot | 一致 |
| spec batch | 0.06 GiB | 分離計測なし | 予約内（起動成立） |
| draft scratch | 0.03 GiB | 約 0.054 GiB | **24 MB 過小** |

- 修正前の session slot は 0.24 GiB/slot だった。`dflash2_spec_workspace_bytes` が
  GDN recurrent history（`history_rows × recurrent_slot_stride × 2 B`）を加算していたが、
  DFlash2 は `create_gdn_spec_history(..., with_recurrent=false)` で conv しか確保しない。
  recurrent は compact な d/k/a log で巻き戻す。4 session で 0.65 GiB の差。
- draft scratch の `draft` 次元は `config.block_size`（8）で計算されているが、
  `create_dflash2_executor` は `max_rows = max(block_size, max_batch_rows)`（4 session で 32）
  で proposal logits を確保する。32 × 248,320 × 4 = 31.8 MB に対し 8 行分 7.9 MB しか
  計上していない。差 24 MB。
- rank 1 は drafter weight 1.123 GiB と DFlash2 context ring 0.17 GiB を予約するが、
  実確保は rank 0 の arena のみ。合計 1.29 GiB が rank 1 で未使用になる。
  KV page 数は両 rank で同一に揃える必要があり、使用可能な context は rank 0 が決めるため、
  この 1.29 GiB は context 長には変換できない。

## 6. その他の予約

- QSA indexer state: slot ごとに `max_seq` 分を upfront 確保する
  （131072 で 0.094 GiB/slot、524288 で 0.375 GiB/slot）。
  短い context の request しか来なくても 4 slot 分を常時保持する。
- MoE group-row: `rows × top_k` 行の上限で厳密。2048 行で 187 MiB。
- logits: `max_scheduled_output_rows`（32）分のみで、rows に比例しない。正しい。
- batch context: `3 × rows × u32` + `max(rows, requests)` 分の output rows / sampling。
  予約と一致。
- target weights: lm_head 0.445 GiB は rank ごとに複製、embed は vocab 分割。
  drafter が rank 0 の full lm_head を使うための構成で、予算計算の対象外。

## 7. ムダの一覧と実装状況

| # | 内容 | サイズ | 状態 |
| --- | --- | ---: | --- |
| 1 | DFlash2 session slot の recurrent history 二重計上 | 0.65 GiB（4 session） | 修正済み |
| 2 | slack = estimate/3（実測誤差は ±60 MiB 以内） | 0.36〜0.49 GiB | 実装済み |
| 3 | EXL3 固定 workspace が 27B 形状 | 0.19 GiB | 実装済み |
| 4 | EXL3 accum の out_features 上限 17,408 | 0.055 GiB（2048 行） | 実装済み（#3 と同時） |
| 5 | draft scratch の draft 次元（過小） | −24 MB | 未修正 |
| 6 | rank 1 の drafter 予約（未使用） | 1.29 GiB | 構造上、context に変換不可 |
| 7 | row bucket 2048（decode は 32 行） | 0.70 GiB | prefill とのトレードオフ |

#2 は `slack = min(max(64 MiB, estimate/8), 256 MiB)`。#3/#4 は rank の重みから
`max_k` / `max_out_features` / `max_dense_bytes` を求め、xrot・dense copy・accum を
その形状で確保する（LM head と MTP block は fused 経路なので除外）。
形状を渡さない caller（bench、PP、MTP）は従来の上限を使う。

## 8. 実装後の再測定

TP=2 / dflash6 / conc=4 / FP8 KV の実測:

| max_seq | rows | reserve | slack | executor | 結果 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 131072 | 2048 | 1.31 GiB | 0.15 GiB | 1.17 GiB | `SHELF_SERVE_READY` |
| 262144 | 2048 | 1.74 GiB | 0.19 GiB | — | 180,128 token まで |
| 262144 | 1024 | 1.37 GiB | 0.15 GiB | — | 244,656 token まで |
| 262144 | 512 | 1.18 GiB | 0.13 GiB | — | `SHELF_SERVE_READY` |
| 262144 | 256 | 1.09 GiB | 0.12 GiB | — | `SHELF_SERVE_READY` |
| 262144 | 32 | 1.00 GiB | 0.11 GiB | — | `SHELF_SERVE_READY` |
| 524288 | 2048 | 2.56 GiB | 0.25 GiB | — | 38,640 token まで |
| 524288（dflash なし、conc=1、rows=256） | 256 | 0.62 GiB | 0.07 GiB | — | `SHELF_SERVE_READY` |

131072 / 2048 行の変化:

| 項目 | 修正前 | 修正後 |
| --- | ---: | ---: |
| working set reserve | 1.96 GiB | 1.31 GiB |
| slack | 0.49 GiB | 0.15 GiB |
| executor footprint | 1.48 GiB | 1.17 GiB |
| KV に残る量 | 0.84 GiB | 1.49 GiB |

262144 / 2048 行で KV に入る量は 54,704 → 180,128 token。
524288 は dflash 込みでは TP=2 で不可（KV 3.05 GiB + QSA 1.5 GiB が weights 後の
空き 4.31 GiB を超える）。dflash なし・conc=1・rows=256 なら起動する。

### 2 concurrent の上限

`--max-concurrent-requests 2`（drafter session 2、QSA 2 slot）では、KV と QSA の
合計が 1 token あたり 7,776 B になる。起動を実測した境界:

| rows | 起動できた最大 | 確認 |
| ---: | ---: | --- |
| 2048 | 約 288,000 | 262,144 は起動、294,912 は 288,464 token まで |
| 512 | 約 365,000 | 327,680 / 360,448 は起動、368,640 は 364,896 token まで |
| 256 | 約 379,000 | 376,832 は起動 |
| 32 | 約 388,000 | 393,216 は 388,560 token まで |

262,144 を超える設定は `--rope-factor 2` が必須である。

## 9. 再現

```bash
./build/shelf-compute --model-dir models/Qwen3.8-Flash-Next-exl3 \
  --serve-stdio --device 2,3 --tensor-parallel 2 \
  --dflash2-model-dir models/Qwen3.8-Flash-Next-DFlash2 --dflash2-drafts 6 \
  --kv-cache-dtype fp8 --page-tokens 16 --prefix-cache 0 \
  --max-concurrent-requests 4 --max-seq-len 131072 --max-num-batched-tokens 2048
```

`MEMORY_PLAN auto tp rank`（reserve と slack）、`MEMORY_PLAN tp drafter`（4 項）、
`MEMORY_PLAN tp rank X executor footprint`（実測）を突き合わせる。
