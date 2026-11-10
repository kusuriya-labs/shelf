# mul24 INT8 Prefill A/B 経路の実装と評価

対象: `exl3_mul24_prefill_ab.patch` の適用、検証、A/B 計測。
測定日: 2026-11-10、R9700 (gfx1201)、HIP 7.15、モデル
`models/kusuriya-labs/Qwen3.8-27B-EXL3-INT8-4bpw`。

## 実装内容

パッチは dense prefill に opt-in の B 経路を追加する。

| 経路 | 選択 | 内容 |
|---|---|---|
| A | 既定 / `SHELF_EXL3_MUL24_PREFILL_AB=A` | rotate_quantize → rowscale → recon → hipBLASLt int8 GEMM → flagged GEMM → rotate_out |
| B | `SHELF_EXL3_MUL24_PREFILL_AB=B` | rotate_quantize → recon → 自作 gfx12 iu8 WMMA → rotate_out |

B は per-128-K の activation scale をそのまま使い、行単位への再量子化を行わない。
重みは recon が作る `[N][K]` の signed INT8 を使い、1 group を int32 で厳密に
累積したあと FP32 で scale を掛けて加算する。

追加・変更したファイル:

- `cmake/targets.cmake`: `mul24/prefill_wmma.hip` を追加。
- `include/shelf/kernels/optimized/linear/exl3_mul24.h`:
  `launch_exl3_mul24_prefill_wmma` を追加。
- `src/shelf/kernels/optimized/linear/exl3_family/mul24/prefill_wmma.hip`: 新規。
- `src/shelf/inference/execution/optimized_dispatch.hip`: B 分岐を追加。
- `src/shelf/inference/execution/decode_perf_stats.cpp`: B の stage 名を追加。
- `tests/kernels/linear/test_exl3_mul24_linear.hip`: B を fused と照合する検査を追加。

パッチ適用時に必要だった修正:

- `int32x2_t` / `int32x8_t` が未定義だったため、`grouped.hip` と同じ TU ローカル
  alias を追加した。
- `fmaf` のため `<cmath>` を追加した。

## 数値検証

`test_exl3_mul24_linear` に、recon した dense 重みで B を実行し、fused の
per-group 累積と比較する検査を追加した。

- rows=1,5,16,17,33,70、n=128/256、k=256、bits=1..8 の全ケースで
  `vs_fused_rel_l2 = 0`（完全一致）。
- 既存の EXL3 required テストは全て PASS。

実モデル (context 2048、positions 256) でも B と fused は完全一致した。

| 比較 | PPL 差 | KL 平均 | top-1 | top-5 | logit 最大差 |
|---|---|---|---|---|---|
| B 対 fused | 0 | 0 | 1.000 | 1.000 | 0 |

B は A が持つ RowScale 再量子化誤差を持たない。精度は fused と等価で、
A より高い。

## 性能

`pp --mode forward --prompt-tokens 2048 --prefill-chunk 2048 --warmup 1 --runs 3`:

| 経路 | wall median | 対 A |
|---|---|---|
| A (rowscale dense) | 1225.43 ms | 1.00x |
| B (WMMA) | 3955.98 ms | 3.23x |
| fused (`DENSE=0`) | 2474.43 ms | 2.02x |

stage 内訳（1 run 換算）:

| stage | A | B |
|---|---|---|
| ROTATE_QUANT | 116.9 ms | 110.2 ms |
| ROWSCALE | 52.1 ms | — |
| RECON | 54.4 ms | 53.2 ms |
| GEMM | 552.3 ms（hipBLASLt） | **3392.2 ms（WMMA）** |
| FLAG_GEMM | 27.8 ms | — |
| ROTATE_OUT | 182.4 ms | 174.0 ms |
| EXL3 linear 合計 | 1001.9 ms | 3741.1 ms |

B の遅さは GEMM に集中している。dense GEMM は A の 6.1 倍、
trellis を decode する fused GEMM（約 1959 ms/run）の 1.7 倍かかる。

## 原因の見立て

B の重みは recon が作る `[N][K]` 行優先の INT8 で、WMMA の B fragment が要求する
並びになっていない。1 wave の 16 lane が `weights[(n0+col)*k + ...]` を読むため、
16 lane が k バイト間隔（例: 5120 バイト）の別々のキャッシュラインを触り、
8 バイト取得のために 128 バイト line を引く。fused 経路の trellis は n-major の
tile 順で、wave が連続した stream を読むため、この問題がない。

つまり B は「trellis decode の ALU を消す」代わりに「重みアクセスの空間局所性を
失う」設計になっており、現状では後者が大きい。

## 追記: RECON の swizzle（tile-major）

上の計測後、recon の出力を WMMA fragment 順へ swizzle した。

- `launch_exl3_mul24_recon_wmma` を追加した。trellis と同じ n-major の 16x16 tile 順で、
  1 tile = 256 バイト、lane `l` が 8 重みを持つ。WMMA の B fragment（32 lane × 8 バイト）が
  1 回の連続 256 バイト読みになる。
- A 経路の `launch_exl3_mul24_recon` は `[N][K]` 行優先のまま変更していない。
- recon kernel は `bool kTileMajor` で 2 つの出力レイアウトを切り替える。
- B の数値は swizzle 前後で不変（fused と完全一致、KL=0、logit 差 0）。
- 単体テストの `prefill_wmma` 検査も `recon_wmma` を使うように更新した。

性能（1 run 換算）:

| 経路 | wall median | WMMA GEMM | 対 A |
|---|---|---|---|
| A | 1225.43 ms | 552.3 ms（hipBLASLt） | 1.00x |
| B（swizzle 前） | 3955.98 ms | 3392.2 ms | 3.23x |
| **B（swizzle 後）** | **3026.09 ms** | **2457.7 ms** | **2.47x** |
| B（swizzle 後、MT=8） | 5924.66 ms | 5384.5 ms | 4.83x |

swizzle で GEMM が 28%、wall が 24% 短縮した。MT=8 は register pressure で悪化し、
MT=4 が最良だった。

## 残差の原因

swizzle 後も B は A の 2.47 倍かかる。B の出力タイルは 1 wave あたり
16 列 × 64 行で、重み（16 列 × K）と活性（64 行 × K）を
`rows / 64` 回再読する。2048 行では 32 回で、重みと活性を合わせて
1 レイヤーあたり約 1.7 TB に相当する。実測の GEMM 時間はこの DRAM 帯域
（640 GB/s）とほぼ一致しており、B の GEMM は実質的に帯域律速である。

つまり swizzle は「fragment の並び」を直したが、「行タイルが小さいため
重みと活性を繰り返し読む」という構造は残っている。MT を増やす案は
レジスタで破綻するため、次に効くのは共有メモリで重みタイルを保持して
行方向の再利用を増やす GEMM 化であり、単純なレイアウト変更ではない。

## 判断

- B の数値は正しく、fused と完全一致する。実装としての妥当性は確認できた。
- swizzle で 24% 速くなったが、A の 2.47 倍遅く、このままでは採用できない。
- 既定 A を維持する。B は opt-in の比較経路として残す。
- さらに詰めるなら、共有メモリで重みタイルを再利用する GEMM 構造への変更が必要。
  レイアウトの追加変更だけでは A には届かない。

## 未実施・注意

- 共有メモリを使った重みタイル再利用と、それによる再計測は未実施。
- 計測は単一 GPU、単一コーパス、1 prefill チャンクによる。
- `SHELF_EXL3_MUL24_PREFILL_AB` は `SHELF_EXL3_INT8_DENSE` が有効な行数でのみ効く。
