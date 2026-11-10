# grouped EXL3 / QSA score kernel の追加チューニング（不採用）

測定日: 2026-11-10、R9700 (gfx1201) × 2、TP=2。
対象: `models/kusuriya-labs/Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw`、`--kv-cache-dtype fp8`。
基準: commit `5110114`（QSA score/top-k 分離後）。

## 結論

grouped EXL3 GEMM と QSA score kernel の両方で候補を計測したが、
基準を上回る構成は見つからなかった。production の変更はない。

## grouped EXL3 GEMM

32K prefill、`shelf-bench pp --prompt-tokens 32768 --prefill-chunk 2048`:

| 条件 | prefill | tok/s |
|---|---:|---:|
| fuse=2（現行） | **13.07 s** | **2,506** |
| fuse=3 | 13.50 s | 2,427 |
| fuse=4 | 15.28 s | 2,145 |
| fuse=2 + `__launch_bounds__(256,2)` | 13.57 s | 2,414 |
| fuse=4 + `__launch_bounds__(256,2)` | 13.66 s | 2,399 |
| INT8 activation（`SHELF_EXL3_GROUPED_INT8=1`、無効な変数名） | 13.36 s | 2,452 |

`SHELF_EXL3_GROUPED_INT8` はリポジトリ内に存在しない。実装されているのは
moe_dispatch の `SHELF_EXL3_INT8_MOE`（既定 0 = FP16）であり、上の行は
FP16 経路の再測定にすぎない。INT8 activation の採否は未計測である。

fuse を増やすと expert あたりの重み再読は減るが、VGPR が 136 → 192、
LDS が 16.9 → 21.5 KiB に増えて occupancy が落ちる。fuse=4 は
`__launch_bounds__(256,2)` で 15.28 → 13.66 s まで改善するが、fuse=2 に届かない。
fuse=2 に同じ cap を付けると逆に遅くなる。

B 経路（recon + WMMA）を MoE 全体に適用する案は、dense の A/B では
8K prompt で A 3,864 ms / B 3,817 ms と B がわずかに速いが、MoE では
512 expert × 3 行列の recon に約 2.5 GB/layer（48 layer で約 121 GB）が必要で、
global recon は成立しない。expert ごとの recon も write + read が増えるため、
現在の fused decode より重くなる。

## QSA score kernel

32K prefill:

| 条件 | prefill | tok/s |
|---|---:|---:|
| 16 row/CTA（現行） | **13.19–13.25 s** | **2,474–2,484** |
| 8 row/CTA | 8K で 3,811 ms | 2,149 |
| 4 row/CTA | 13.73 s | 2,386 |

8K prompt の基準は 3,459 ms / 2,368 tok/s。

- 4 row/CTA は K tile の再読み込みが 4 倍になり、occupancy の改善を相殺する。
- 8 row/CTA も 2 倍の K 再読み込みで基準より遅い。
- A fragment を K step ごとに shared から読み直して VGPR を下げる案は
  13.60 s / 2,409 tok/s。VGPR は 248 のままで、 occupancy は改善しなかった。
- `__attribute__((amdgpu_num_vgpr(128)))` で VGPR を強制制限すると
  8K で 3,832 ms / 2,138 tok/s。spill のため悪化した。

score kernel は VGPR 248 で occupancy 1 CTA/CU だが、行数を減らすと
K tile 再読み込みが増え、VGPR cap は spill する。次に試すなら、
K tile を shared に保持したまま複数 row tile を処理する persistent 形か、
B fragment を shared に tile-major で recon して VGPR を下げる構成になる。
