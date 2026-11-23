# issue #41 の P0 改善候補の実測

issue #41 が挙げた P0 の 2 件を実装・計測した記録である。測定原則は
[../../perf/methodology.md](../../perf/methodology.md) を正本とする。

## 前提と環境

- GPU: Radeon AI PRO R9700 × 5（`amd-smi list`、他プロセスなし）
- モデル: `models/kusuriya-labs/Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw`
- 2026-11-23 の HEAD（`75ff82de`）は MoE の expert matrix split により
  この checkpoint の TP>1 起動を拒否する。そのため MTP の A/B は
  `72dce8a7`（expert-ID split 以前）の worktree で実施した。
  expert 交換の A/B は HEAD で実施した。

## E1: MTP verify 方式の比較（ACCEPT）

### 条件

- worktree `.worktrees/mtp-tp-baseline`、commit `72dce8a7`
- `--tensor-parallel 2 --device 2,3`
- `--mtp --mtp-drafts 3 --mtp-window 4096 --mtp-sink 64`
- `--max-seq-len 1024 --max-num-batched-tokens 512`、prefix cache なし
- 自然文 prompt（48 token）、greedy、64 token 生成
- 速度は最初の `token` から最後の `token` までの区間で算出した

HIP graph 有効の build では、c=1 の fused verify が最初の round で
`hipEventSynchronize: operation not permitted when stream is capturing` で失敗する。
これは既知の不具合として扱い、A/B は `-DSHELF_HIP_GRAPH=OFF` の build で行った。
3 方式とも同じ build で測っている。

### 結果

| verify | c | decode tok/s | 方式 |
| --- | ---: | ---: | --- |
| target-only | 1 | 34.84 | MTP なし |
| safe serial（既定） | 1 | 32.30 | `chunks=47 emitted=47` |
| fused（`SHELF_MTP_FUSED_VERIFY=1`） | 1 | **43.88** | `chunks=26 emitted=47` |
| fused | 2 | 60.51 合計 / 37.52 per-stream | `chunks=18 emitted=62` |

- 既定の safe serial は target-only より 7.3% 遅い。issue の
  「target を 1 token ごとに直列実行するため target-only を超えない」を再現した。
- fused は safe serial の **1.36 倍**（32.30 → 43.88 tok/s）。
- 64 token 生成では fused 47.93、target-only 35.3、serial 32.2 で、
  傾向は同じである。
- 48 token の生成列は 3 方式で完全一致した。

### 採否

ACCEPT。既定は完全一致用途（safe serial / target-only）のまま変えず、
速度優先の serve では `SHELF_MTP_FUSED_VERIFY=1` を明示する。
graph 有効時の c=1 fused 失敗は別途修正が必要である。

## E2: expert 交換のバッチ化と pinned staging（ACCEPT、効果は cold prefill のみ）

### 変更

`MoeExpertCache` の交換を 1 組ずつ実行する実装から、最大 16 組の
demote / promote をまとめて非同期転送する実装へ変えた。

- demote（VRAM→host staging）を全組まとめて発行し、sync を 1 回にする。
- promote（host→VRAM）を全組まとめて発行し、sync を 1 回にする。
- 交換ごとの `std::vector` 確保を廃止し、staging を `hipHostMalloc` の
  pinned 領域で再利用する。
- slot map の upload を層ごとに 1 回へまとめる。

1 組あたりの stream sync は 6 回から最大 2/16 回になる。

### 条件

- HEAD の `shelf-compute --serve-stdio`、`--device 2`、TP=1 / EP=1
- `SHELF_MOE_HOST_EXPERTS=448`、prefix cache なし、bf16 KV
- 4096 token の prompt、greedy、`--max-num-batched-tokens 4096`
- 変更前 binary と変更後 binary を交互に起動し、各 3 回測定

### 結果（cold prefill = モデル起動後の 1 本目）

| binary | prefill ms | tok/s |
| --- | ---: | ---: |
| 変更前 | 4724.5 / 4768.2 / 4815.5 / 4822.9 / 4832.6 | 849〜867 |
| 変更後 | 3995.2 / 4000.2 / 4005.7 / 4020.7 | 1018〜1025 |

cold prefill は約 **800 ms 短縮（約 16%）** した。交換 2,562 組の往復は
変わらず、staging と同期の費用だけが減っている。host staging を pageable の
`std::vector` から pinned に変えた効果が大きく、batch 化だけでは
4768〜4821 ms から 4682〜4696 ms（約 2.6%）であった。

2 本目以降の prefill（交換なし）は変更前 2398〜2407 ms、変更後 2397〜2409 ms で
差がない。decode も 17.0 tok/s で変わらない。issue が想定した「prefill に載る
秒単位の費用」は、cold の交換 step に限れば実在した。

### 正しさ

- 変更前 binary と同じ 48 token の生成列が一致した。
- `ctest -R "test_qwen4exp_load|test_moe"` は 10/10 PASS。
- `test_compute_mtp_batch.py`（TP=1、host expert 448、2 request）は 10/10 PASS。
  この経路は MTP block の load 指定が `LoadedModel` で落ちていたため、
  `Qwen4ExpLoadOptions::load_mtp` を渡す修正も同時に入れた。

## 判断

- issue P0-1 は code 変更なしで効果を確認できた。既定 verify の変更は行わない。
- issue P0-2 は実装が有効で、効果は cold prefill / TTFT に限定される。
  decode と warm prefill には影響しない。
- HIP graph 有効時の c=1 fused verify 失敗、fused の MTP batch reserve 不足
  （TP=1 で約 0.11 GiB、TP=2 c>=4 で約 0.05 GiB）は未解決である。
