# mul24 INT8 Prefill の RowScale 再量子化 精度調査

対象: `src/shelf/kernels/optimized/linear/exl3_family/mul24/prefill.hip` の
`exl3_mul24_rowfinish_kernel`、および dense INT8 prefill 経路。
測定日: 2026-11-10、HEAD `11b7385`、R9700 (gfx1201)、HIP 7.15。

## 結論

- dense prefill は fused 経路の **2.02 倍速い**（2048 prefill で 1226 ms 対 2474 ms）。
  RowScale 自体のコストは 52 ms/回で prefill 全体の **4.2%** に過ぎない。
- RowScale は実モデルの活性に **1.10% RRMS** の追加誤差を加える。
  これは元の INT8 量子化誤差 0.54% の **約 2.1 倍**（電力比 約 4.4 倍、SNR 約 6.4 dB 劣化）に相当する。
- Flag fallback は実データではほぼ効いていない。実際に fused へ送られたのは
  **全体の 1.02%** の行で、400 layer 中 29 layer は cap (12.5%) に張り付いて飽和している。
- end-to-end では大半の位置で差は無視できる（KL 中央値 2.4e-4）が、
  5.1% の位置で KL 0.1〜3.6 の差が出る。低エントロピーな評価コーパスでは
  PPL 差の符号が安定しない。

## 1. 測定条件

- モデル: `models/kusuriya-labs/Qwen3.8-27B-EXL3-INT8-4bpw`
- コーパス: `/tmp/qcells.psktok`
- `context=2048`, `prefill-chunk=2048`, `kv-cache-dtype=bf16`
- 基準 (baseline) は fused 経路: `SHELF_EXL3_INT8_DENSE=0`（per-128-K group scale、追加誤差なし）
- 比較対象は dense 経路: 既定 `SHELF_EXL3_INT8_DENSE=256`（row scale + flag fallback）

fused 経路は 4096 行以下で `exl3_mul24_gemm_splits` が 1 を返し、
per-group の int32 部分積を float で加算する。dense 経路は trellis を int8 へ
展開して hipBLASLt の int32 GEMM を 1 回実行する。両者の差は活性の再量子化が支配的で、
GEMM 実装差（全 K int32 対 group ごと float 加算）は `test_exl3_mul24_linear` の
CPU 参照比較で相対 1e-5 未満である。

## 2. RowScale 再量子化の誤差（実モデル活性）

dense 経路の最初の 8 layer 呼び出し（attention q/k/v/o と MLP、2048 行 × 各 K）で、
`rowfinish` 前後の codes、scales、rowscale、slots を一時的にダンプして比較した。
計装は測定後に削除済み。

再量子化は `q' = round(q * s_g / peak)`、`peak = max_g s_g`。
追加誤差は回転後ドメインで `e = q'*peak - q*s_g`。

| 指標 | 値 |
|---|---|
| codes が変化した割合 | 92.5〜98.3% |
| 行ごと追加誤差 RRMS | 平均 1.097%、中央値 1.139%、p05 0.825%、p95 1.340%、最大 2.231% |
| layer 別 RRMS | 0.83〜1.34%（layer 依存は小さい） |
| group scale 比 `s_g/peak` | 平均 0.455、p01 0.020、p05 0.227、p25 0.327、中央値 0.405 |
| 実効ステップ増加 `peak/s_g` | 中央値 2.47 倍、p75 3.06 倍、p95 4.41 倍、p99 50.9 倍 |
| 0 へ潰れた非ゼロ code | 1.81%（1,997,140 / 110,246,718） |

元の INT8 量子化誤差（一様分布 `s_g/sqrt(12)` を仮定）との比較:

| 指標 | 値 |
|---|---|
| 元の量子化誤差 / 信号 | 平均 0.540% RRMS |
| RowScale 追加誤差 / 信号 | 平均 1.097% RRMS |
| 追加誤差 / 元誤差 (RMS) | 平均 2.08 倍、中央値 2.13 倍、p95 2.88 倍 |

すなわち RowScale は、典型的な group で量子化ステップを約 2.5 倍に粗くし、
行全体としては元の量子化雑音の約 2 倍の RMS 雑音を追加する。

## 3. Flag fallback の実効率

1 回の 2048 行 prefill で 400 layer 呼び出しを計測した
（`peak^2 > share * sum_g s_g^2`, `share = 0.5`、cap = `flag_cap(2048)` = 256 行 = 12.5%）。

| 指標 | 値 |
|---|---|
| 全体の flagged 行 | 8,332 / 819,200 = **1.02%** |
| 1 layer あたり平均 | 20.8 行 (1.02%) |
| flag が 0 行の layer | 295 / 400 |
| flag が 1 行以上の layer | 105 / 400（平均 79.4 行 = 3.9%） |
| cap に飽和した layer | **29 / 400**（12.5% で打ち切り） |

cap 飽和 layer では、flag したい行が 256 を超えている。超過分は row scale 経路に
残るため、flag fallback は異常行を保護しきれていない。

## 4. end-to-end 差（dense 対 fused）

`quality` の logits dump をオフライン比較した。

`positions=64`:

| 経路 | PPL 差 | KL 平均 | top-1 | top-5 |
|---|---|---|---|---|
| dense (flag 0.5) | +0.375% | 0.004422 | 1.000 | 0.903 |
| dense (flag なし) | +0.734% | 0.007513 | 1.000 | 0.903 |

この範囲では flag を有効にすると fused に近づく（KL 0.0075 → 0.0044）。

`positions=256`:

| 経路 | PPL 差 | KL 平均 | top-1 | top-5 |
|---|---|---|---|---|
| dense (flag 0.5) | -4.264% | 0.060847 | 0.9766 | 0.8469 |
| dense (flag なし) | -1.023% | 0.049180 | 0.9805 | 0.8555 |

位置ごとの分布（dense flag 0.5 対 fused、256 位置）:

| 指標 | 値 |
|---|---|
| KL 中央値 | 2.4e-4 |
| KL < 1e-3 | 71.9% |
| KL < 1e-2 | 85.5% |
| KL < 0.1 | 94.9% |
| KL >= 0.1 | 5.1%（13 位置） |
| top-1 が変化した位置 | 6 / 256 |
| NLL 差の合計 | -11.15（うち 1 位置で -14.08） |

平均 KL と PPL 差は 13 位置の裾に支配されている。中央値は 2.4e-4 で、
大半の位置では分布はほぼ一致する。PPL 差の符号が 64 位置と 256 位置で反転するのは、
この裾とコーパスの低エントロピー（PPL 1.01〜1.15）によるもので、
この標本数では PPL 差を品質の優劣として読めない。

## 5. 性能

`pp --mode forward --prompt-tokens 2048 --prefill-chunk 2048 --warmup 1 --runs 3`:

| 経路 | wall median | 対 fused |
|---|---|---|
| dense | 1226.33 ms | 1.00x |
| fused (`DENSE=0`) | 2474.43 ms | 2.02x |

dense 経路の内訳（`SHELF_QWEN35_PERF_TIMING=1`、3 run 合計を 1 run に換算）:

| stage | ms/run | wall 比 |
|---|---|---|
| ROTATE_QUANT | 116.8 | 9.5% |
| **ROWSCALE** | **52.0** | **4.2%** |
| RECON | 54.4 | 4.4% |
| DENSE_GEMM (hipBLASLt) | 553.0 | 45.1% |
| FLAG_GEMM | 27.8 | 2.3% |
| ROTATE_OUT | 182.3 | 14.9% |
| EXL3 linear 合計 | 1002.3 | 81.7% |

RowScale の 1 呼び出しは約 130 µs（rowstat + rowslot + rowfinish の 3 kernel）。

## 6. 判断と候補

現行方式は維持が妥当。

- 性能差が 2 倍あり、RowScale 自体は prefill の 4.2% しか使っていない。
- 大 GEMM に per-group scale を持ち込む変更は、この性能差を失うため現段階では採らない。
- 精度側の実体は「活性に 1.1% RRMS、元の量子化雑音の約 2 倍の雑音を追加する」ことである。

精度を優先する場合の候補（未実施）:

1. `flag_capacity` を広げ、cap 飽和している 29/400 layer の異常行をすべて fused へ送る。
   workspace の accum 領域と flag GEMM のコストが増える。
2. `share` を下げて flag 条件を緩める。ただし cap が上限になるため 1 と併用が必要。
3. 評価側の整備: 高エントロピーなコーパスと十分な位置数で PPL を測る。
   現状は BF16 の ground truth が同じ重みに存在しないため、
   dense と fused のどちらが真値に近いかを end-to-end では判定できない。

## 7. 未実施・注意

- 一時ダンプ計装は測定後に削除し、作業ツリーは元に戻した。
- end-to-end の基準は fused mul24 経路であり、BF16 推論ではない。
- 計測は単一 GPU、単一コーパス、1 回の prefill チャンクに基づく。
- `quality` の PPL は 64/256 位置では裾に弱く、KL と top-1 を主指標とした。
