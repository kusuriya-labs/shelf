# 計測スクリプト（実行時のまま）

`batch_decode_summary.md` の測定に使ったスクリプトを、実行時の内容のまま保存したもの。
パスはこのマシン固有（`/tmp/agentx`、`/tmp/bench27`、`ROCR_VISIBLE_DEVICES`、モデルの場所）で、
そのままでは動かない。使うときは次を直す。

| ファイル | 用途 |
| --- | --- |
| `conc3.py` | `shelf-compute --serve-stdio` に N 本の request を同時に投げ、全員が生成中の区間の合計 tok/s を出す。`ERRF` で stderr、`GENF` で生成 token を保存 |
| `rows_cases.json` | `conc3.py` の入力 prompt（`/tmp/bench27/rows_cases.json` の複製） |
| `dsmk.sh` | `conc3.py` を DFlash2（K、`--dflash2-max-sessions`）付きで回す |
| `gmr.sh` | 同じく投機なしで、複数の N を回す |
| `mtp.sh` | MTP（drafts=4）で回す |
| `sem3.sh`、`batch_timing.patch` | `SHELF_QWEN35_PERF_TIMING` で op ごとの GPU 時間（semantic timing）を取る。`batch_timing.patch`（`continuous_batcher.cpp` に STEPWALL と dfround の dump を足す。`git apply` で当てて、コミットしない）が前提 |
| `gemmlab_*`、`sweep.sh`、`splits.sh`、`cmp2.sh` | EXL3-INT8 GEMM の単体ベンチ（kernel の .hip と `src/apps/bench/exl3_int8.hip` を 1 つの TU でビルド）、split 数の掃引、旧実装との出力比較 |

GPU を使う実行は `flock /tmp/r9700.lock` を付ける。

追加検証（`batch_decode_summary.md` の §2.5）に使ったもの:

| ファイル | 用途 |
| --- | --- |
| `val.py` | `--mode steady`（全員生成中の合計）と `--mode mix`（decode 中に長い prompt が到着）。`--prefix` で共有 prefix、`--pc 1` で prefix cache、`--kv`、`--temp`、`--spec` |
| `srv.py` | `shelf-server` に N 本のストリーミング chat を同時に投げて合計 tok/s を出す |
| `val_all.sh` | §5 の条件を順に回す |
