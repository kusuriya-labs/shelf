# MTP を TP=2 で動かす（2026-11-07）

実験 ID: `MTP-TP2-1`

結論: **ACCEPT**。dense 27B EXL3 で MTP + TP=2 が動作し、target-only TP2 の
2.59 倍（128 token、59.28 tok/s）になった。ただし DFlash2 TP2 の 72.87 tok/s より遅い。

## 1. 背景

MTP は `--tensor-parallel 1` のみ許可されていた。理由は MTP block が TP partition plan に
入っておらず、drafter executor が単一 GPU 前提だったためである。27B は TP2 で decode を
高速化できるため、MTP も TP2 で使える価値がある。

## 2. 実装

- MTP block の weight は partition しない。各 rank が複製して load し、drafter executor は
  rank0 だけで動かす。
- target の prefill / verify は `TpCoordinator::on_execute` で全 rank に流す。
- speculative verify は rank ごとの `ExecuteBatchOptions` を渡す。各 rank が自分の
  `GdnSpecHistory` と compact log buffer を持つ。
- snapshot / restore / compact replay は全 rank で実行する。受理数は rank0 の logits から
  決め、同じ値で全 rank の GDN を戻す。
- target sequence は `on_sequence_created` / `on_sequence_released` で mirror と対応付ける。
- TP executor に `emit_normed_token_hidden` を立てる。
- MTP+TP のときは working set reserve を最低 4 GiB 確保し、KV pool の後ろに drafter と
  per-rank GDN 領域を残す。

主な変更:

- `include/shelf/models/qwen35/runtime/mtp_drafter.h`
- `src/shelf/models/qwen35/runtime/mtp_drafter.cpp`
- `src/apps/compute/compute_runtime.hip`
- `src/apps/compute/single_request.hip`
- `src/apps/compute/compute_cli.cpp`
- `tests/correctness/test_mtp_parity.py`（`SHELF_TP`）
- `cmake/tests.cmake`（`test_mtp_parity_tp2`、MTP の PP2 拒否）

## 3. 計測

27B-exl3、device 0,1、bf16 KV、`--arena-gib 30`、128 token。

| 条件 | tok/s |
| --- | ---: |
| target-only TP2 | 22.89 |
| MTP TP2 drafts=4 | **59.28** |
| DFlash2 TP2 drafts=6 | **72.87** |

- MTP TP2 の mean accepted は 2.750（128 token）。
- MTP TP1 の 32 token drafts=4 は 44.51。TP2 で +33 %。
- 128 token の `GENERATED_IDS` は target-only TP2 と一致する。
- `test_mtp_parity_tp2`（2048 token prompt、64 token、rounds 12、GDN restore 5）は PASS。
- `test_mtp_parity`（TP1）も PASS。
- `test_qwen35_tp_execution`、`test_qwen35_mtp_lowering`、
  `test_qwen35_mtp_weight_load`、MTP の CLI 拒否テストは PASS。

## 4. 制約

- grouped expert（MoE）の TP load は MTP block を分割しないため、MTP との併用を拒否する。
- TP=4 は未計測。
- MTP weight を全 rank が複製するため、rank1 の VRAM に MTP block 分の重複がある。
- この計測は EXL3 の ka=5 fast path を含まない build で取った。fast path と merge した後の
  32 token は target 26.48、MTP 61.81、DFlash2 64.00 tok/s。128 token は未測。
- DFlash2 TP2 の方が速い。MTP の利点は drafter model の追加 load が不要なこと。
