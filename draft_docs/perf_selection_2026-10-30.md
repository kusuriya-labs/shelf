# カーネル最適化の手順適用記録（2026-10-30）

手順の正本は参照実装 `references/phaseshift-fp8/docs/developer/kernel_optimization_and_fusion.md`
（計測原則は同 `docs/perf/methodology.md`）。本文書は shelf 側で ①→②→③ を回した記録。

## ① 計測条件

```
rocprofv3 1.3.5（venv） --kernel-trace -f csv --
  shelf-bench tg --model-dir models/<9B|27B>-FP8 --context 2048 \
                 --prefill-chunk 2048 --tokens 1 --gpu 0 --vram-budget-mib 32256
```

- GPU 0（`rocm-smi` で VRAM 0%・他 workload なことを確認。GPU 2/3 は他 job が 97% 使用中のため不使用）
- 1 実行 = prefill 1 ステップ + decode 1 ステップ。時間は kernel trace のみを使用
- 資源（VGPR / SGPR / LDS / scratch / grid / workgroup）は kernel trace の列から取る
  （PMC は 0 を返すので導出のみ。occupancy は手順書の gfx1201 定数で静的算定）

### occupancy 算定（手順書 §1 の式）

`vgpr_r = ceil(VGPR/8)*8`、`budget_waves = min(4*floor(47104/(32*vgpr_r)), 64)`、
`blocks = min(budget_waves/(wg/32), 131072/LDS, 64/(wg/32))`。

## ① 選定表

27B-FP8（ctx 2048・chunk 2048・GPU 時間合計 1521.2 ms・dispatch 7681・kernel 27 種）

| kernel | share | µs/dispatch | 回数 | VGPR | LDS | wg | occ（拘束） | 実効 GB/s（必須 / 行タイル 16× 再読） | 律速分類 | レバー |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gemm_fp8_w8a8_prefill_wide` | 44.1% | 1677.5 | 400 | 176 | 16.9 KB | 256 | 4（reg→wave 上限 32 に張り付き） | 36.5 / **585** | ① DRAM（重みの行タイル当たり再読） | bytes 実減少（1 回の重み読みで扱う行数を増やす）。occupancy・命令昇格は不可（既に WMMA・reg 上限） |
| `phaseshift_attention_paged_bf16` | 18.9% | 8965.8 | 32 | 64 | 8.7 KB | 256 | 8（wave） | 6.7（KV+Q+out ≈ 60 MB/layer） | ③ レイテンシ / 並列不足（2.9 TFLOP/s。クエリ 1 本ずつの総当りでタイル化なし） | タイル化 WMMA（128×128）・同一 KV の再読消去 |
| `gemm_bf16_prefill_2d`（lm_head） | 9.8% | 149284 | 1 | 32 | 0 | 512 | 4（wave） | — | ② pass 律速（**最終行以外は消費されない出力を作る冗長 pass**） | **冗長計算の消去（採用・下記）** |
| `gdn_chunk_impl` | 9.0% | 44.6 | 3072 | 224 | 44.0 KB | 256 | **2（LDS 131072/44032）** | 低（小チャンク多数） | ③ 並列不足（LDS が拘束で 2 block/CU） | LDS 縮小（staging を減らす）またはチャンク併合。*staging を増やして occupancy を落とす方向は禁止* |
| `phaseshift_mul_f32_f32_bf16` | 2.6% | 303.7 | 128 | 24 | 0 | 256 | 8 | **1170（>636 → L2/L3 吸収）** | ② L3/pass 律速 | 融合（silu+mul）または中間を bf16 化（pass 削減のみは融合では効かない） |
| `activation_quantize_fp8_group` | 2.1% | 40.7 | 800 | 24 | 0 | 256 | 8 | 386 | ② pass 律速 | 生成側（norm / down 前）との融合 |
| `bf16_to_f32_k` | 1.6% | 125.6 | 192 | 8 | 0 | 256 | 8 | 890 | ② pass 律速（純変換 pass） | 生成側・消費側の dtype 統一で pass 自体を消す |
| rope / row_slice / swiglu / residual_add | 1.0–1.2% ずつ | | | | | | | | ② | 上位が片付くまで後回し |

9B-FP8（同条件・合計 779.9 ms・dispatch 3657）

| kernel | share | 備考 |
| --- | --- | --- |
| `gemm_bf16_prefill_2d` | **50.4%** | lm_head 131.9 ms（1 dispatch）+ **GDN 側投影が checkpoint で未量子化**（`quantization_config.ignore` に `linear_attn.in_proj_qkv/z/out_proj`）なので bf16 GEMM。N=8192 が 24 回（5488.9 µs/回）、N=4096 が 48 回（2702.7 µs/回）。25 TFLOP/s（fp8 の実測 178 TFLOP/s、bf16 の ceiling はその半分として 89 TFLOP/s が見える位置） |
| `gemm_fp8_w8a8_prefill_wide` | 17.6% | 量子化されている MLP 側 |
| `phaseshift_attention_paged_bf16` | 11.7% | 同上（2.9 TFLOP/s 級） |
| `gdn_chunk_impl` | 7.7% | LDS 拘束で occ 2 |

### ② の候補順位（share × 律速 × レバー可否）

1. **lm_head の最終行以外（27B 9.8% / 9B 17.0%）** — ② 冗長計算の消去。数値不変・byte-exact 可。→ **実施**
2. attention prefill のタイル化（18.9% / 11.7%、2.9 TFLOP/s） — ③。実装規模大。→ 次回
3. 9B の未量子化 GDN 投影の bf16 GEMM 効率（33%、25 TFLOP/s） — ②③。
   ※ これを fp8 化するのは**量子化段階の変更**なので手順書 原則 4 のエスカレーション対象（未実施）
4. MLP の silu / elem_mul / 活性化量子化 / bf16→f32 の融合鎖（合計 7.5%） — ②。T1 下限 0.7% に対して実測で判定
5. fp8 GEMM の重み再読減（44.1%） — ①。行タイル 2 倍は ACCUM 2 倍 = occupancy 低下になり**禁止パターン**
   （LDS staging は既に試験済みで in-model 逆転: 27B 1342→1258）。差し替え案は要設計

## ② 採用レバー: prefill の logits を最終行だけにする（冗長計算の消去）

### 根拠（会計）

- lm_head は `vocab × hidden`（9B: 248320×4096 / 27B: 248320×5120、bf16）を rows 分行う。
  次のトークン選択が使うのは**最終行の logits だけ**で、其余の行の logits はどの経路からも
  消費されない（`shelf-run` / `shelf-bench` はいずれも最終行を取る）。
- 効果源は手順書 §4 の 2（消費側が行っていた仕事そのものの消去）と 3（中間 materialize 削減）。
  logits は 1 行でも 2 GB 級（rows=2048 × 248320 × f32 = 2.03 GB）で L3 容量を大きく超えるため、
  アリーナにも実額で効く（27B chunk 2048 のアリーナ 3,568 → 1,628 MiB）。

### 実装

- IR に `row_slice`（先頭次元の連続スライス）を追加。`slice` は最終次元スライスなので別種。
  追加は enum 末尾（既存 Kind の数値を変えない）。`Attr::row_last` は
  「このステップで実際に渡された行（n）の最終行」を意味する。
  チャンク末尾が部分（n < グラフ rows）でも正しい行を選ぶため、静的 index では足りない。
- `model/qwen35` の `GraphOptions::logits_last_row_only`（既定 false）で final 部を
  `row_slice(rows=1) → rmsnorm(1 行) → output_projection(1 行)` に組み替える。
  `StepGraphPlan::logits_rows` に結果の行数を出す。
- 実行器の `linear/output_projection` は行数を `step_rows_` ではなく**出力形から**決める
  （`rows = out_n / N`。同じステップ内で 1 行の GEMM が混在するため）。
  ここを直さないと `row_slice` 導入時に不正アクセスになった（実際に発生）。
- apps: `PipelineOptions::logits_last_row_only`。`shelf-run` / `bench tg` / `bench pp` は
  **既定で最終行のみ**（勝ち経路を既定にする）。`shelf-validate` は行ごとの logits を T1 比較に
  使うので `false` のまま（`--logits-all-rows` は escape hatch として用意）。

### harness 側の欠陥（手順 methodology §5 に該当）

`--logits-all-rows` を値なしフラグとして追加した際、`Args` が次のトークンを値として飲み込み、
**baseline 実行が usage で落ちていた**（A/B の A 列が空になった）。`bench_common.hpp` に
`kValuelessFlags` を設け、値を取らないフラグを parser で明示化した。

## ③ 検証ゲート

### 正しさ

1. focused test（`tests/unit/test_hip_step_executor.cpp`）: 4 行 batch を
   `logits_last_row_only` 無し / 有りで走り、
   - `logits_rows == 1` と logits 長 = V
   - 全行版の**最終行との最大絶対差 2.384e-07**（<= 1e-5、既存の batch vs seq 差と同オーダー。
     差は rows=1 側 GEMM の累加順に由来し、スライス自体は無損）
   - argmax 一致
   を検証。
2. E2E token ゲート（`shelf-run --steps 192`、生成列を全比較）:
   - 9B-FP8・合成 64 ids: **IDENTICAL**
   - 9B-FP8・実 fixture 256 ids: **IDENTICAL**
   - 27B-FP8・実 fixture 256 ids（--steps 96）: **IDENTICAL**
   - 同一条件の logits ダンプ比較: **max|d| = 0（byte-exact）**（9B・27B とも）
3. `ctest`: cpu 19/19、hip 28/28（ともに full）。

### 性能（interleaved A/B、主指標を事前確定）

主指標: 27B-FP8 prefill throughput
（`tg --context 2048 --prefill-chunk 1024 --tokens 8 --gpu 0 --vram-budget-mib 32256`、
 `2048 / context_prefill_ms`）。副指標: 9B prefill と両モデルの decode step_ms_p50（非悪化確認）。
A = 現行 incumbent（`--logits-all-rows`）、B = 候補（既定）。3 ペア・交互。

| ペア | 27B A (ms) | 27B B (ms) | 差 | 9B A (ms) | 9B B (ms) | 差 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1509.07 | 1363.00 | −9.68% | 778.99 | 657.89 | −15.5% |
| 2 | 1516.25 | 1361.77 | −10.19% | 783.34 | 656.66 | −16.2% |
| 3 | 1515.19 | 1357.46 | −10.41% | 780.05 | 656.05 | −15.9% |
| p50 | **1515.19** | **1361.77** | **−10.1%（+11.3% tok/s: 1,352 → 1,504）** | **780.05** | **656.66** | **−15.8%（2,626 → 3,120 tok/s）** |

- ペア別の差が 9.7〜10.4% で、A 列の散布 0.5%・B 列 0.4% より十分大きい → データで決着。
- decode（rows=1 のグラフは変更なし）: 27B p50 A 59.47 / B 59.63 ms（+0.27%）、
  9B p50 A 24.347 / B 24.345 ms（±0）。27B の +0.27% はアリーナ offset 移動に起因する範囲として
  許容（副指標は「非悪化確認」）。
- 副次効果: 27B の chunk 2048 が 31.5 GiB 予算内で載るようになった
  （確保計上 31,952 → 30,012 MiB、実 device 空き 96 → 2,036 MiB）。

### 採用

勝ち経路を既定化（run / bench は最終行のみ、validate は全行）。差分は本文書とコミットメッセージに記録。

### 採用後の chunk 2048（同条件・1 実行）

| 実行 | prefill | アリーナ | 確保計上 / 実空き |
| --- | --- | --- | --- |
| 27B-FP8 chunk 2048 | 1341.0 ms = **1,527 tok/s** | 1,628 MiB（変更前 3,568 MiB） | 30,012 MiB / 2,036 MiB（変更前 31,952 / **96**） |
| 9B-FP8 chunk 2048 | 648.7 ms = **3,157 tok/s** | 968 MiB（変更前 2,908 MiB） | 13,109 MiB / 19,246 MiB |

chunk 1024 と chunk 2048 の差は 27B で 1.5%、9B で 1.2%（どちらも 2048 が速い）。
27B の 2048 行はこれで余裕を持って載る（実空き 2.0 GiB）。
