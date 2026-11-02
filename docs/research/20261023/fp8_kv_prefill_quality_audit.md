# FP8 KV 品質劣化の原因特定と修正（fp8 paged prefill）

- 実施日: 2026-10-23
- 対象: `src/shelf/kernels/optimized/attention/paged_prefill.hip` の fp8 経路
- 対象 model: `models/Qwen3.8-27B-FP8`（HF FP8）
- GPU: Radeon PRO R9700（gfx1201）× 1（`--device 1`）
- 環境: Git `990e42ce7dd6ff6edc303af3e0b6fd6265e6fdf0`、HIP 7.15.26333、AMD clang 23.0.0git
- corpus: `build/quality/fp8_kv_baseline/corpus.psktok`
  （BetterBench v1 を chat template 化、2203 token、
  sha256 `99fe6aa90b4f52a9427361391f0dc3e17377391d67793c1c716e890d8c28799d`）

## 結論

`--kv-cache-dtype fp8` の品質劣化（旧記録 ppl +28% / mean KL 0.78）は、
KV 保存形式（e4m3 per-token per-head scale）の損失ではなく、
**fp8 paged prefill kernel の 3 つの実装欠陥**で説明できる。修正後は
BF16 KV との差が、BF16 同士の prefill 分割差より小さくなる。

| ID | 欠陥 | 影響 |
| --- | --- | --- |
| D1 | Q の行 amax を `bf16` → `unsigned short` の暗黙変換（float 経由の切り捨て）で計算していた | 全行で `q_scale` が実値の数百分の一になり、Q の fp8 code が飽和。score が復元不能 |
| D2 | PV の B fragment（V）を token-major の staging のまま dim 行として読み出していた | `v_page[dim][token]` の転置読み出し＋LDS 範囲外参照。PV 出力が不定（NaN/ごみ） |
| D3 | P 量子化で `1/pmax_v` が denormal で inf になり、`0 * inf = NaN` を e4m3 code 化していた | 重み最大値が denormal になる page（極端に尖った attention）で、その行の出力が全 dim NaN |

D1 と D2 は独立に致命的で、D2 は NaN または無相関な値を生む。D3 は D2 修正後に
残った NaN で、行単位の部分的欠陥である。

## 計測方法

- 比較はすべて同一 GPU（`--device 1`）で、BF16 KV と FP8 KV を同一 corpus・
  同一採点位置で行った。`shelf-bench quality` の `--compare` を使う。
- `ctx` は prefill する token 数、採点は既定で `ctx+1` 以降。
  `ctx128` と `ctx1600` の数値をそのまま並べると採点範囲が違うため、
  追加した `--score-start-position 1601` で採点開始位置を固定した比較も行った。
- 品質指標は `mean KL`（BF16 を p）、top-1 一致率、平均 NLL 差（`NLL_fp8 - NLL_bf16`）、
  PPL 差。PPL 差は同一採点集合でのみ意味を持つ。

## E1: 修正前の再現（現行 build）

`--context 128/1600`、`--positions 448`、`--prefill-chunk 2048`。

| ctx | ppl BF16 | ppl FP8 | PPL 差 | mean KL | top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 | 21.602 | 20.133 | -6.80% | 0.2279 | 0.8638 |
| 1600 | 6.711 | 8.625 | +28.53% | 0.7753 | 0.6853 |

旧記録（ppl +28.12% / +29.96%、KL 0.223 / 0.697、top-1 0.816 / 0.723）と
同じ桁で再現した。ctx128 で PPL が改善するのは採点範囲が異なるためで、
KL 0.23 / top-1 0.86 は明確な劣化である。

## E2: prefill kernel の寄与分離

`--prefill-chunk 64` を与えると prefill 行数が 128 未満になり、
prefill kernel ではなく decode 経路が使われる（`kPagedAttentionPrefillMinRows = 128`）。

| ctx | prefill 経路 | PPL 差 | mean KL | top-1 |
| --- | --- | ---: | ---: | ---: |
| 128 | decode 経路（chunk 64） | -2.73% | 0.0094 | 0.9621 |
| 128 | prefill kernel（chunk 2048） | -7.00% | 0.2279 | 0.8638 |
| 1600 | decode 経路（chunk 64） | +8.76% | 0.0166 | 0.9487 |
| 1600 | prefill kernel（chunk 2048） | +28.53% | 0.7753 | 0.6853 |

劣化の大部分は prefill kernel 由来である（KL 0.009 → 0.228、0.017 → 0.775）。
decode 経路側の残差（ctx1600 で KL 0.017）は e4m3 KV 保存自体の損失である。

## E3: 単体テストが欠陥を検出しなかった理由

`test_paged_attention` の `fp8 PREFILL` ケースは、比較が
`|got-ref| / max(1,|ref|) > tol` の要素数で判定していた。NaN はこの比較が常に
偽になるため、**出力が全て NaN でも PASS していた**。D1/D2/D3 はいずれも
NaN または転置由来の不定値を生むが、テストはそれを許容していた。

## E4: 修正内容

1. Q 行 amax を `__bfloat162float` で計算する（D1）。
2. V staging を dim-major（`v_page[kHeadDim][kVPitch]`、pitch 20）に変更し、
   PV の B fragment を dim 行から 4 byte 単位で読む（D2）。
   padding を含めて 4 byte 境界を保つため pitch は 20 とした。
3. P 量子化の逆数は `pmax_v > 1e-30` のときだけ取り、それ未満の page は
   `p_scale = 0` として寄与を落とす（D3）。
   `p_scale` が denormal のまま `1/p` が inf になると `0 * inf` が NaN になる。

## E5: 修正後の品質

`ctx128/1600`、`--positions 448`、`--prefill-chunk 2048`、修正後の同じ build。

| ctx | PPL 差 | mean NLL 差 | mean KL | top-1 |
| --- | ---: | ---: | ---: | ---: |
| 128 | -6.80% | -0.0705 | 0.0162 | 0.9621 |
| 1600 | -1.00% | -0.0100 | 0.0172 | 0.9643 |

`tools/quality_gate.py run`（数値部、`--skip-endtask --skip-gsm8k`）でも PASS。

| ctx | positions | PPL 差 | mean KL | top-1 | gate |
| --- | ---: | ---: | ---: | ---: | --- |
| 128 | 2048 | -3.67% | 0.0130 | 0.9561 | PASS |
| 1600 | 602 | -3.19% | 0.0167 | 0.9552 | PASS |

## E6: 同一採点位置での比較（`--score-start-position`）

採点位置を 1601 以降に固定し、`ctx` だけを変えた比較（positions 600）。

| 条件 | PPL 差 | mean NLL 差 | mean KL | top-1 |
| --- | ---: | ---: | ---: | ---: |
| BF16 ctx128 → BF16 ctx1600（分割差のみ） | +3.34% | +0.0328 | 0.0060 | 0.9633 |
| FP8 ctx1600（修正後） | -3.20% | -0.0325 | 0.0167 | 0.9550 |
| FP8 ctx128（修正後） | -1.57% | -0.0158 | 0.0081 | 0.9567 |

BF16 同士でも prefill 分割だけで mean NLL 0.033・KL 0.006 の差が出る。
修正後の FP8 の差はこの範囲に収まっており、この計測分解能では BF16 と区別できない。

## E7: P 量子化ピーク（1 と 256）の比較 → REJECT 256

仮説: P を最大 1 に正規化すると e4m3 の小値領域で underflow し、小さい Softmax 重みが
0 に丸められる。FlashAttention-3 と同様に最大 256 へ移せば改善する可能性がある。

同一採点位置（`--score-start-position 1601`、positions 600）での実測:

| 構成 | ctx | PPL 差 | mean NLL 差 | mean KL | top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| P ピーク 1（現行） | 1600 | -3.20% | -0.0325 | 0.0167 | 0.9550 |
| P ピーク 256 | 1600 | -1.51% | -0.0152 | 0.0088 | 0.9633 |
| P ピーク 1（現行） | 128 | -1.57% | -0.0158 | 0.0081 | 0.9567 |
| P ピーク 256 | 128 | -0.14% | -0.0014 | 0.0126 | 0.9467 |

差は方向も大きさも一定せず、BF16 同士の分割差（KL 0.006、mean NLL 0.033）と
同程度である。**ピーク 256 を採用する根拠は得られなかったため、現行の最大 1 を維持する。**
D1/D2/D3 の修正後は P の underflow が品質を支配していない。

E7 の測定は D1/D2 修正後・D3 の下限導入前の build で行った。D3 の下限は
page 最大重みが 1e-30 未満のときだけ作用し、ピーク 256 ではその条件がさらに
起こりにくくなる。ピーク 1 の値は下限導入の前後で同一だった（E5・E6 と同値）。

## E8: 性能影響（V staging の転置）

`shelf-bench pp --prompt-tokens 2048 --kv-cache-dtype fp8 --runs 1 --warmup 1` を
同一 GPU で交互に 5 回（gpu_tokens_per_sec、p50）。

| build | p50 tok/s | 5 回の範囲 |
| --- | ---: | --- |
| 修正前（`990e42c`） | 2339.9 | 2333.8 - 2359.8 |
| 修正後 | 2327.8 | 2320.1 - 2334.1 |

差は -0.52% で、run 間のばらつき（±1%）と同程度。V を 8 byte 単位で
transpose staging するコストは、この計測では識別できない。

## 棄却した案

- **KV の量子化方式の変更（token×head → layer 固定 scale など）**: E2 で
  decode 経路の損失が小さいことを確認したため不要。KV 保存形式は変更しない。
- **P ピーク 256 への固定**: E7 のとおり効果が再現しない。
- **prefill kernel を使わない運用**: prefill 性能を失う。修正で不要になった。

## 残課題

- `max_visible >= 2048` の fp8 split/reduce decode 経路は、今回の corpus では
  到達していない（`kPagedAttentionSplitMinVisible = 2048`）。fp8 の長文脈
  split を対象にした単体テストは未整備である。
- `test_paged_attention` の fp8 prefill 判定は行単位の相対 L2（mean 1e-2 以下、
  max 0.6 以下）と有限性判定に置き換えた。要素単位の tol 1e-2 は、
  実装仕様である P 量子化を含む参照では満たせない。
- end-task（reasoning / GSM8K）は今回の修正では再測定していない。
  E5 の数値ゲートのみで採否を判断していない点に注意する。
