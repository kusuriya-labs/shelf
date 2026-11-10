# mul24 INT8 Prefill WMMA 最適化（gfx1201）

対象: `src/shelf/kernels/optimized/linear/exl3_family/mul24/prefill_wmma.hip`
（`SHELF_EXL3_MUL24_PREFILL_AB=B` の経路）。
測定日: 2026-11-10、HEAD `fbbea9f` から開始、R9700 (gfx1201)、HIP 7.15.26333。
A（hipBLASLt + RowScale）は変更していない。

## Phase 0: ベースライン

- モデル: `models/kusuriya-labs/Qwen3.8-27B-EXL3-INT8-4bpw`
- コーパス: `/tmp/qcells.psktok`、context 2048、prefill-chunk 2048、bf16 KV
- `pp --mode forward --prompt-tokens 2048 --warmup 1 --runs 3`（中央値）
- GEMM 単体: `bench_exl3_mul24_prefill_wmma --rows 2048 --iters 50`
  （5120x5120, 10240x5120, 6144x5120, 5120x6144, 17408x5120, 5120x17408 の合計）

| 項目 | B0 |
|---|---|
| GEMM（6 shape 合計） | 17.46 ms |
| GEMM（モデル集計） | 1536.3 ms/run |
| Prefill wall | 2113.5 ms |
| VGPR / SGPR / Spill / LDS | 128 / 36 / 0 / 0 |
| 数値 | fused と完全一致（KL=0、logit 差 0） |

A: wall 1226.1 ms、GEMM 552.4 ms。

## Phase 1: Full Tile / Tail 分離（採用）

`kFullRows` をコンパイル時定数にし、行境界チェック・ゼロ埋め・store guard を
full CTA から除去した。launcher は `rows / 128` 個の full CTA と、端数 1 CTA の
tail を別 grid で起動する（tail は従来のガード付き）。

| 構成 | GEMM（6 shape） | GEMM（モデル） | wall | VGPR |
|---|---|---|---|---|
| B0 | 17.46 ms | 1536.3 ms | 2113.5 ms | 128 |
| B1 Full Tile | 15.19 ms | 1342.1 ms | 1903.6 ms | 120 |

-13%（GEMM 単体）、-10%（モデル GEMM）、-10%（wall）。

## Phase 2: アドレス計算の削減（不採用）

3 案を実装し、いずれも悪化したため採用しなかった。

| 案 | 内容 | GEMM（6 shape） |
|---|---|---|
| 2-A/B | 活性・重みの基準ポインタを保持し group ごとに加算 | 19.13 ms（+26%） |
| 2-B のみ | 活性ポインタだけ保持 | 16.81 ms（+11%） |
| 2-D | ホットループの index を uint32_t 化 | 15.85 ms（+4%） |

原因は VGPR の増加（143 → 145）と、可変ポインタにしたことで
`global_load_b64` の即値オフセットが使えなくなり、group をまたぐ依存が増えたこと。
現行の「式のまま書く」形がコンパイラにとって最良だった。

## Phase 3: MT / NT / WM の探索（変更なし）

swizzle 前（S=1）と後（S=8）の両方で探索した。GEMM は 6 shape 合計。

| MT | NT | WM | GEMM (S=1) | GEMM (S=8) |
|---|---|---|---|---|
| 2 | 2 | 4 | **15.19** | **9.91** |
| 1 | 2 | 4 | 16.72 | 15.36 |
| 2 | 1 | 4 | 20.79 | 30.23 |
| 1 | 1 | 4 | 26.83 | 25.93 |
| 2 | 2 | 2 | 17.46 | 9.96 |

MT=2/NT=2/WM=4 を維持。MT=4 は VGPR が 128 を超えて占有率が落ちるため不採用
（既存計測どおり）。NT=3 は `n` が 192 で割り切れない shape があり、端数処理を
増やすため対象外とした。

## Phase 4: block swizzle（採用、最大の改善）

FP8 blockwise prefill の `swizzle_g` と同じ考え方で、CTA の
`(blockIdx.x, blockIdx.y)` を L2 再利用が効く順に並べ替えた。

```
lin = bx + gridDim.x * by
bx  = (lin / (gx*S)) を使わず、  bx = (lin % (gx*S)) / S
by  = (lin / (gx*S)) * S + (lin % (gx*S)) % S
```

S 個の連続 CTA が同じ出力ブロック（同じ重みタイル）を共有し、
その次に出力ブロックが進む。重みタイルは 64 列 × K バイトで L2 に収まる。

| S | GEMM（6 shape） | 5120x5120 |
|---|---|---|
| 1 | 15.27 ms | 0.955 ms |
| 2 | 10.00 ms | 0.653 ms |
| 4 | 9.94 ms | 0.637 ms |
| 8 | **9.84 ms** | 0.632 ms |
| 16 | 10.33 ms | 0.727 ms |

S=8 を採用。`full % S != 0` のときは S=1 のカーネルへフォールバックする
（tail も S=1）。swizzle は CTA とタイルの対応を並べ替えるだけで、
計算内容と数値は変えない。

初期実装では S をカーネル引数にしたところ、S=1 でも 20.49 ms に悪化した。
コンパイル時定数にして `if constexpr` で消す形に直している。

## Phase 5 / 6: 不採用

- 5-A/5-B（scale FMA の分割）: MT×NT の int32 と FP32 accumulator は
  どのみち同時に生存するため、分割しても VGPR は減らない。未実施。
- 5-C（scale 先読み）: 16 本の scale load を先に出すと VGPR が増え、
  g ループの `#pragma unroll 2` は 17.20 ms に悪化した。
- 5-D / 6（LDS 共有）: swizzle で DRAM 律速ではなくなったため未実施。

## 最終結果

| Variant | GEMM ms | Wall ms | VGPR | Speedup vs B0 | 判定 |
|---|---:|---:|---:|---:|---|
| A hipBLASLt | 552.4 | 1226.1 | - | - | Reference |
| B0 現行WMMA | 1536.3 | 2113.5 | 128 | 1.00x | Baseline |
| B1 Full Tile | 1342.1 | 1903.6 | 120 | 1.14x | 採用 |
| B2 Address | - | - | 145 | - | 不採用（悪化） |
| B3 Tile | 1342.1 | 1903.6 | 120 | 1.00x | 変更なし |
| B4 Swizzle (S=8) | **836.1** | **1425.8** | 143 | **1.84x** | 採用 |
| B5 Scale FMA | - | - | - | - | 未実施 |
| B6 LDS | - | - | - | - | 未実施 |
| Final | 836.1 | 1425.8 | 143 | 1.84x | 採用 |

- GEMM: 1536.3 → 836.1 ms（**-46%**）
- Prefill wall: 2113.5 → 1425.8 ms（**-33%**）
- A との比: GEMM 1.51x、wall 1.16x

内訳（1 run 換算）:

| stage | A | B final |
|---|---|---|
| GEMM | 552.4 ms | 836.1 ms |
| ROTATE_QUANT | 116.9 ms | 117.7 ms |
| ROWSCALE | 52.1 ms | — |
| RECON | 54.4 ms | 51.3 ms |
| FLAG_GEMM | 27.9 ms | — |
| ROTATE_OUT | 182.5 ms | 184.1 ms |
| EXL3 linear 合計 | 1002.1 ms | 1201.1 ms |

## 検証

- `test_exl3_mul24_linear`: B と fused の比較を全ケースで `vs_fused_rel_l2 ≈ 1e-7`
  （float 丸めのみ）。rows=128/129/1024 を追加し、full tile・tail・swizzle 経路を通した。
- EXL3 required テストは全て PASS。
- 実モデル teacher-forced（positions 64 と 256）で B は fused と完全一致:
  PPL 差 0、KL 0、top-1/top-5 1.0、logit 最大差 0。
- `git diff --check` clean。

## 残存ボトルネック

- モデル集計で GEMM 836 ms のうち、6 shape の単体計測では 5120x5120 が 0.63 ms
  （170 TOPS）で、まだ約 2 倍の余地がある。1 group あたり 261 命令中 32 WMMA で、
  アドレス演算 44、scale load 16、cvt/fma 64 が残る。
- VGPR 143 のため occupancy が低く、`s_wait_loadcnt` が 1 group あたり 40 回出る。
  MT/NT を増やすと VGPR で破綻し、減らすと再読みが増えるため、
  現状の MT=2/NT=2 が境界。次に効くのは LDS に重みタイルを置いて
  ウェーブ間で共有する GEMM 化、または activation codes の fragment 化。
- rotate_out（184 ms）と rotate_quant（118 ms）は B と A で共通のため、
  GEMM を A 並みにしても wall は A より小さくならない（A は rowscale 52 + flag 28 を持つ）。
