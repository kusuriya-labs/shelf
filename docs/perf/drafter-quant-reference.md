# DFlash2 drafter の量子化測定

## 条件

- 測定日: 2026-11-08（Asia/Tokyo）、GPU: AMD Radeon AI PRO R9700（gfx1201）
- 実装 revision: `98d367e`。測定はこの revision の `build-gfx1201/shelf-compute` を使用。
- target: `/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24`
- drafter: BF16 `Qwen3.8-27B-DFlash2`、`dflash2-drafter-exl3-4bpw-mul24`、
  `dflash2-drafter-fp8`。各形式を交互に、各 prompt で 2 回ずつ実行。
- prompt: `/tmp/prompt_{code,prose,ja}.ids`。各 prompt の非投機 greedy も実行。
- 各プロセスで `flock /tmp/r9700.lock` を取得。`ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55`、
  `LD_LIBRARY_PATH=/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib`。
- 共通引数: `--model-dir <target> --input-ids-file <prompt> --max-new-tokens 256
  --max-seq-len 1024 --temperature 0`。投機実行にはさらに
  `--dflash2-model-dir <drafter> --dflash2-drafts 7`。KV cache は既定の BF16、
  arena は auto。プロセスごとの実行であり、5 回の中央値を求める kernel microbenchmark ではない。

## GPU 数値検証

`test_drafter_quant_linear` で各実 checkpoint の `layers.0.self_attn.q_proj.weight` を
ロードし、同じ量子化重みから復元した BF16 checkpoint の GEMM 出力と比較した。
入力は固定乱数、出力は BF16。両方の GEMM を GPU で実行した。
以下で再実行できる。`<quant-dir>` と `<dequant-dir>` はそれぞれ対象形式の
量子化 checkpoint と、その重みを復元した BF16 checkpoint の directory である。

```sh
flock /tmp/r9700.lock env \
  LD_LIBRARY_PATH=/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib \
  ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55 \
  SHELF_DRAFTER_QUANT_DIR=<quant-dir> SHELF_DRAFTER_DEQUANT_DIR=<dequant-dir> \
  ./build-gfx1201/tests/test_drafter_quant_linear
```

| 形式 | 行数 | 相対 L2 誤差 | 最大絶対差 |
|---|---:|---:|---:|
| EXL3-INT8 mul24 | 1 | 0.00664483 | 0.14062500 |
| EXL3-INT8 mul24 | 8 | 0.00704763 | 0.25000000 |
| EXL3-INT8 mul24 | 16 | 0.00700480 | 0.26562500 |
| FP8 | 1 | 0.03038668 | 1.06250000 |
| FP8 | 8 | 0.02494535 | 1.06250000 |
| FP8 | 16 | 0.02521156 | 1.06250000 |

EXL3-INT8 は既存の rotate/quantize、group-scale fused GEMM、出力回転を使用する。
8 行と 16 行の両方で同経路を実行した。FP8 は activation も FP8 に量子化しているため、
復元 BF16 GEMM と完全一致する計算ではない。

## E2E 結果

全 18 投機実行が終了コード 0 で完走し、各 `GENERATED_IDS` は同じ prompt の非投機
`GENERATED_IDS` と完全一致した。`tok/s` は `DECODE_TOKENS_PER_SEC`。
GPU ms の列は `DFLASH2_DRAFT_GPU_MS` の各実行の合計であり、右隣に round 数と、
各合計を round 数で割った値の 2 回平均を示す。

| prompt | drafter | tok/s 各回 | tok/s 平均 | 受理数 各回 | GPU ms 各回 | round 数 各回 | GPU ms/round 平均 | arena 使用 GiB 各回 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| code | BF16 | 97.14, 97.46 | 97.30 | 4.543, 4.543 | 385.10, 384.86 | 46, 46 | 8.369 | 27.973, 27.973 |
| code | EXL3 | 104.08, 104.39 | 104.23 | 4.543, 4.543 | 244.89, 243.51 | 46, 46 | 5.309 | 28.156, 28.156 |
| code | FP8 | 102.59, 103.45 | 103.02 | 4.543, 4.543 | 277.99, 276.46 | 46, 46 | 6.027 | 28.156, 28.156 |
| prose | BF16 | 47.54, 47.73 | 47.63 | 1.713, 1.713 | 777.48, 776.24 | 94, 94 | 8.264 | 27.973, 27.973 |
| prose | EXL3 | 51.10, 51.35 | 51.23 | 1.713, 1.713 | 491.90, 488.81 | 94, 94 | 5.217 | 28.156, 28.156 |
| prose | FP8 | 50.50, 50.70 | 50.60 | 1.713, 1.713 | 559.28, 556.81 | 94, 94 | 5.937 | 28.156, 28.156 |
| ja | BF16 | 48.03, 48.02 | 48.03 | 1.742, 1.742 | 777.02, 776.78 | 93, 93 | 8.354 | 27.973, 27.973 |
| ja | EXL3 | 52.09, 52.08 | 52.09 | 1.772, 1.772 | 486.26, 486.35 | 92, 92 | 5.286 | 28.156, 28.156 |
| ja | FP8 | 49.89, 51.06 | 50.48 | 1.742, 1.742 | 558.99, 557.35 | 93, 93 | 6.002 | 28.156, 28.156 |

`ARENA_USED_GB` はモデル重みだけでなく、実行領域と自動で割り当てた KV pool を含む。
code の `MEMORY_PLAN` が見積もる drafter 重みは BF16 3.58 GiB、EXL3 1.17 GiB、
FP8 1.97 GiB。自動 KV budget はそれぞれ 8.18、10.60、9.79 GiB であり、
全体 arena 使用量の差を重みサイズの差として解釈できない。
