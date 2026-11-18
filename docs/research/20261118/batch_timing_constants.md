# GDN verify の上限定数を直接上げる実験（`exp/batch-timing` の未コミット分）

2026-11-18 に `exp/batch-timing` worktree の未コミット変更を評価し、不採用として破棄した。

## 変更内容

- `include/shelf/kernels/optimized/gdn/recurrence.h`: `kGdnRecurrenceVerifyBatchRows` を 32 → 512、
  `kGdnRecurrenceVerifyMaxRequests` を 4 → 64。
- `continuous_batcher.cpp`: `SHELF_BATCH_SEM_DUMP` と `SHELF_BATCH_STEPWALL` による計測の追加。
  同じ内容の `batch_timing.patch` が `docs/research/20261110/batch_decode_scripts/` にあり、新しくない。

## 経緯

GDN verify の launch 上限（4 request / 32 行）の引き上げは、MTP の chunk 定数と分離した別定数
`kGdnRecurrenceVerifyLaunchMaxRequests = 64` / `kGdnRecurrenceVerifyLaunchMaxRows = 512` として
main にすでに入っている（[batch_decode_summary.md](../20261110/batch_decode_summary.md) §1.1）。
この未コミット変更は、元の 2 定数を直接上げた初期の実験である。元の 2 定数は、MTP の chunk 分割、
verify GEMM の split-K 調整行数（`executor.hip` の `splitk_tuning_rows`）、行数の clamp にも使われる。

## 測定

main に近い base（`eec4f028`、7b11e356 + 記録）で、変更後（B）と現状（A）を交互に 2 回ずつ比べた。
R9700、Qwen3.8-27B EXL3 4bpw mul24、DFlash2 drafts 7、FP8 KV、各 request 160 token、greedy、
`/tmp/agentx/conc3.py`（全員が生成中の共通区間の tok/s）。

| 同時数 | A（現状）tok/s | B（定数を上げた）tok/s | 生成 token |
|---:|---|---|---|
| 8 | 318.4 / 315.3 | 317.7 / 315.8 | A・B・再実行の全 request が一致（各 8/8） |
| 16 | 298.8 / 297.3 | 298.5 / 297.3 | 全一致（16/16） |
| 32 | 361.9 / 360.5 | 361.3 / 361.7 | 全一致（32/32） |

## 結論

- 速度の差は run 間の揺れ（約 1%）内で、改善も劣化も観測できなかった。生成 token は全件一致した。
- 不採用。破棄した変更は `shelf-branch-archive/batch-timing-uncommitted-20261118.patch` に残した。

## 未測定・補足

- MTP 経路（4 request / 32 行の chunk 分割が実際に変わる経路）は測っていない。
- 同時 4 本は A・B ともに `dflash2_spec_decoder.cpp:962: sampled token is invalid` で失敗した。
  この変更とは無関係で、この base と 4 本の組み合わせで再現する。原因は未調査。現在の main でも出るかは未確認。
