# mul24 INT8 Prefill WMMA 第2段階最適化（gfx1201）

対象: `src/shelf/kernels/optimized/linear/exl3_family/mul24/prefill_wmma.hip`
測定日: 2026-11-10、HEAD `5dce29d`、R9700 (gfx1201)、HIP 7.15.26333。
A（hipBLASLt + RowScale）は変更していない。

## 結論

第2段階で試した候補はすべて B0 より遅く、**production の変更はない**。
GEMM 837.2 ms、wall 1427.3 ms（5 run 中央値）で第1段階の結果を維持する。

目標の GEMM 636 ms には届かなかった。理由は測定で示せている。
B は 1 group（K128）あたり 262 命令で 32 WMMA を実行し、そのうち
約 64 命令が「int32 → FP32 変換 + group scale の FMA」である。
これは per-128-K scale という数学そのものに由来し、scale を持たない A には存在しない。
実測の B/A 比 837/552 = 1.52 は、この命令比（WMMA 32 に対し epilogue 64）とほぼ一致する。

## 基準（第1段階の最終状態）

| 項目 | B0 |
|---|---|
| GEMM（モデル集計、5 run） | 837.2 ms |
| Prefill wall（5 run 中央値） | 1427.3 ms |
| GEMM（6 shape、50 iter） | 9.85 ms |
| VGPR / SGPR / Spill / LDS | 143 / 17 / 0 / 0 |
| occupancy（推定） | 3 wave/SIMD（144 VGPR × 3 ≤ 512） |
| 1 group の命令 / WMMA | 262 / 32 |
| 数値 | fused と完全一致（KL=0、logit 差 0） |

A: GEMM 552.3 ms、wall 1224.5 ms。

## Step 1: K16 段階ロード（U4 / U2）— 不採用

`#pragma unroll 1` の外側ループで K16 を U 個ずつチャンク化した。

| Variant | GEMM（6 shape） | VGPR | SGPR | Spill |
|---|---|---|---|---|
| U8（現行） | **9.85 ms** | 143 | 17 | 0 |
| U4 | 18.28 ms | 163 | 21 | 0 |
| U2 | 13.42 ms | 166 | 19 | 0 |

VGPR は減らず、逆に増えた。チャンク化で `s0` が実行時変数になり、
`global_load_b64` の即値オフセットが使えなくなるため。不採用。

## Step 2: WMMA 発行順 — 変更なし

コンパイル後の ISA を確認した。1 group のループは 262 命令で、
先頭に b fragment 16 本を `s_clause` でまとめて発行し、その後 K16 ごとに
独立な 4 accumulator へ 4 WMMA を連続発行している。

```
global_load_b64 x8 (s_clause) ... global_load_b64 x8
...
v_wmma ... s_wait_loadcnt ... v_wmma ... s_wait_loadcnt ...
```

4 accumulator はすでに交互に更新されており、手動で並べ替えても
スケジューラが同一の順序に戻す。変更しない。

## Step 3: Scale メモリレイアウト — 不採用

現行の row-major scale は、1 行について g を 0..39 と走査すると
4 バイト刻みの連続アクセスになる。group-major にすると
`rows*4` バイト刻みになり、この走査には不利になる。

group-major の利点は「同一 group の 16 行分をベクトルロードできる」ことだけで、
そのためには `rotate_quantize` の出力形式を B 専用に変える必要がある。
一方、後述のロード形状の実験ではアクセス形状を変えても改善しなかったため、
レイアウト変更の見込みは小さいと判断した。不採用。

## Step 4: プロファイルとロード形状

- `rocprofv3` で利用できる PMC は gfx1201 では `SQ_WAVES` と
  `SQ_WAVE_CYCLES` のみで、L2 ヒット率（TCC_*）は未対応だった。
- 代わりに、境界内で fragment 相当（連続 256 バイト）のアドレスに
  差し替えて活性ロードのコアレス効果を測った。

| 実験 | GEMM（6 shape） | 備考 |
|---|---|---|
| 現行（row-major codes） | 9.85 ms | |
| fragment 相当アドレス | 10.88 ms | 値は不正、形状のみ |

コアレス化しても速くならないため、**メモリトランザクションは律速ではない**。
残るのは命令数とレイテンシである。

## その他の候補（すべて不採用）

| 候補 | GEMM（6 shape） | 内容 |
|---|---|---|
| 3D grid swizzle（除算なし） | 16.70 ms | VGPR 120 だが CTA dispatch 順が変わり大幅悪化 |
| 3D grid swizzle（z 側を行に） | 15.59 ms | 同上 |
| WM=8 | 12.62 ms | 全 wave が同じ出力ブロックを共有 |
| kWaves=16 / WM=4 | 10.46 ms | CTA 128 行 × 128 出力 |
| kWaves=16 / WM=8 | 11.29 ms | CTA 256 行 × 32 出力 |
| MT=1 / NT=4 | 22.29 ms | |
| MT=2 / NT=4 | 10.92 ms | |
| MT=1 / NT=2 / WM=8 | 25.42 ms | |
| MT=2 / NT=3 | 14.64 ms | n が 96 で割り切れない shape あり |
| g ループ `unroll 2` | 17.20 ms | VGPR 増 |
| 活性ポインタ hoist | 16.81 ms | VGPR 145 |
| 活性 + 重みポインタ hoist | 19.13 ms | VGPR 145 |
| ホットループの 32bit index | 15.85 ms | |

16x16x32 の iu8 WMMA は gfx1201 に存在しない（iu4 のみ）ため、
WMMA 1 命令あたりの K を増やす方向も取れない。

## 残存ボトルネックの分析

1 group（1 wave、K128）あたりの内訳:

| 種別 | 数 |
|---|---|
| WMMA | 32 |
| global_load_b64（a 16 + b 16） | 32 |
| global_load_b32（scale） | 16 |
| int32 → FP32 変換 | 32 |
| group scale の FMA | 21〜32 |
| アドレス演算 | 44 |
| wait / delay / move 等 | 残り |

WMMA を 1 とすると、それ以外が約 7。WMMA が 4 cycle 占有するとしても、
命令発行が律速になる。epilogue の 64 命令は per-128-K scale の数学に必須で、
A には存在しない。したがって B の下限は A の約 1.5 倍（約 830 ms）と見積もられ、
実測 837 ms はすでにその下限に近い。

さらに縮めるには、次のいずれかが必要になる。

- LDS に重みタイルを置き、wave 間で共有して global load を減らす
  （1 group あたり 16 → 4 本。ただし barrier が 2 回/group 増える）。
- 1 wave あたりの出力タイルを広げて epilogue を相対的に減らす
  （VGPR の壁で MT=4/NT=2 は不可）。
- scale を fold した activation を作る（= A の RowScale。B の存在意義がなくなる）。

## 未実施

- LDS 重み共有の実装と実測。
- 5 run 交互測定は A/B 各 5 run を 1 回ずつ実施した（交互反復は
  セッション中に複数回実施済みで、値は ±0.3% で一致）。
- 品質比較は第1段階の結果（fused と完全一致）から変化がないため再実施していない
  （production のコードは第1段階のまま）。
