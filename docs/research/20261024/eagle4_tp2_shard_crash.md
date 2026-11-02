# EAGLE-4 TP2 shard の GPU1 crash 調査（foreign pointer と device scope）

- ID: EAGLE4-TP2-SHARD-CRASH
- 実施日: 2026-10-24
- 対象: `investigation/eagle4-tp2-shard-crash`（WIP commit `f4c2198` の続き）
- 環境: R9700 (gfx1201) ×2、ROCm/HIP 7.15、Qwen3.8-27B-FP8 + Qwen3.8-27B-EAGLE4、
  `--tp 2`、`--eagle4-lm-head draft`

## 症状

TP2 + EAGLE4 の prefill（`extend_rows rows=256 position=0`）で、GPU1 の
`phaseshift_embedding_bf16` dispatch が
`HSA_STATUS_ERROR_INVALID_PACKET_FORMAT`（AQL packet malformed）で落ちる。
`grid=[65536]` は work-item 表記（256 block × 256 thread）であり launch 形状は正常。

## 切り分け（実測）

`PHASESHIFT_EAGLE4_DEVICE_TRACE=1` / `PHASESHIFT_EAGLE4_SYNC_TRACE=1` /
`PHASESHIFT_EAGLE4_ARG_TRACE=1` を追加して serve 経路を実行した。

- `PHASESHIFT_EAGLE4_SYNC_TRACE` により、`tap_norm_concat` → `fc` → `embedding` の
  どこで最初に失敗するかを stage 名で観測できるようにした。packet error は
  `embedding` の直前に観測され、先行 stage の async failure を embedding の launch が
  拾う形になり得ることが確認できた。
- ただし実際の失敗は、`embedding` より前に追加した **device invariant 検査**で
  捕捉された:

```text
[eagle4-device] stage=forward_rows rank=1 expected=0 current=1 stream_device=1
```

  rank1 の `Eagle4Executor::device` が 0（生成時の `hipGetDevice()`）であり、
  rank1 の executor が **rank0 の device scope で生成されていた**。

## 仮説と棄却

| 仮説 | 判定 | 根拠 |
| --- | --- | --- |
| TP collective / barrier の頻度が原因 | REJECT | crash は最初の `tp_sum` 到達前に観測される。`embedding` の直前で止まる |
| `EmbeddingBf16Args` の ABI 不整合（56 byte） | REJECT | clean build（`rm -rf build` + 全再ビルド）後も同じ。kernel 単体 test は GPU1 で PASS |
| `grid` / `workgroup` 形状が不正 | REJECT | `grid=[65536]` は work-item 表記で 256×256。host 側の launch 引数は正常 |
| rank をまたぐ device pointer（foreign pointer） | **ACCEPT** | rank1 の `tap_norm_concat` / `embedding` に rank0 arena の pointer を渡していた |
| executor / context の生成 device が rank と不一致 | **ACCEPT** | serve 経路の rank ごとの生成が device scope を持たず、current device = 0 のまま rank1 を生成していた |

## 修正

1. 各 rank の executor / context / 重み load を `ScopedDevice(rank_device(r))` で行う
   （`stdio_server` の EAGLE4 rank 構築）。`Eagle4Executor::device` は生成時の
   `hipGetDevice()` を保持し、invariant 検査の期待値になる。
2. TP job の契約を「各 rank の kernel には、その rank が所有する device pointer だけを
   渡す」に変更した。
   - 入力 token と tree metadata は host 配列で job に渡し、各 rank が自分の stream で
     `Eagle4Executor` の staging buffer へ H2D する。
   - hidden tap は各 rank の target `Executor::dflash_target_hidden` から
     `tap_row_offset` で選ぶ。tree commit は行が飛ぶため rank ごとに
     `eagle4_gather_rows_bf16` で staging へ gather する。
   - prefill の chunk 分割時、tap の row offset が kernel に渡っていなかった
     （2 chunk 目以降で先頭 256 行を再利用していた）のも同時に修正した。
3. candidate merge は rank ごとの stream を sync してから D2H するようにした。
   rank1 の stream 完了を待たずに読む競合があった。
4. `PHASESHIFT_EAGLE4_DEVICE_TRACE` / `PHASESHIFT_EAGLE4_SYNC_TRACE` を追加した。
   未設定時は HIP 呼び出しも同期も増えない。

## 検証（実測）

- clean build（`build/` 削除後に全再ビルド）で required acceptance 143 件すべて PASS。
- TP2 + EAGLE4 serve（`--max-seq-len 4096`、24 token 生成、prefill 56 token）で
  `[eagle4-device]` 128 行すべて `expected == current == stream_device`（rank0/rank1）。
  `tap_norm_concat` / `fc` / `embedding` / `embedding_tp_sum` / `layer_input_norm` /
  `q_proj` / `k_proj` / `v_proj` は両 rank で sync PASS（412 行）、
  `HSA_STATUS_ERROR_INVALID_PACKET_FORMAT` は再現しない。
- 単一 rank の tree 経路（`phaseshift-compute --eagle4-tree-width 4`）で
  tree metadata staging と tap gather を含む commit 経路が動作することを確認した。
- 新規 test `test_eagle4_tp_forward`（2 GPU、小さな checkpoint）で
  embedding partial + sum / attention row-parallel / MLP row-parallel / final normed が
  TP1 と一致（max_abs/rms: A=0.000, B=1.2e-2, C=2.9e-2, D=1.6e-2）。

## 残る制約

- candidate pool は rank ごとの local top-k の和であるため、TP2 の候補集合は TP1 と
  一致するとは限らない（pool は TP2 の方が広い）。test は「merge 結果が両 rank の候補
  logits の上位 k と一致すること」と「TP2 の最良候補が TP1 より悪化しないこと」を検証する。
- TP2 + tree は serve の CLI が `--eagle4-tree-width` を露出していないため、
  serve からは chain 経路のみ到達可能である（tree 経路は単一 rank で検証）。
- 単一 GPU で 27B target と draft の `embed_tokens` / `lm_head` を同時に arena に置くと
  32 GiB に収まらない。TP2 では vocab 分割で収まる。
