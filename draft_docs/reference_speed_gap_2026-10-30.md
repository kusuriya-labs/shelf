# 参照実装との速度差（2026-10-30 計測）

目的は `references/phaseshift-fp8` と**同じ prefill / decode 速度**に到達すること。
手順は `draft_docs/perf_selection_2026-10-30.md`（参照実装の手順書の適用記録）どおり。

## 比較条件（両側に同一）

- GPU 0（`rocm-smi` で VRAM 0% を確認。GPU 2/3 は他 job 使用中で不使用）
- `models/Qwen3.8-27B-FP8` / TP=1 / KV bf16 / page_tokens 16
- prefill: 2048 トークンを 1 バッチ（参照 `pp --prefill-chunk` 既定 = 全量、shelf `--prefill-chunk 2048`）
- decode: context 2048 から 128 トークン（warmup 8）、greedy
- 参照ビルド `/tmp/ps-build`（Release / gfx1201）、shelf `build/hip`（Debug + 最適化フラグ設定は従来どおり）

## E2E の差

| 断面 | 参照 | shelf（本日の最適化後） | 必要倍率 |
| --- | --- | --- | --- |
| 27B prefill 2048 tok | **901.4 ms = 2,272 tok/s**（3 run 散布 0.13%） | 1,341.0 ms = 1,527 tok/s | **1.49×** |
| 27B decode（ctx 2048） | **51.28 ms/tok = 19.50 tok/s**（散布 0.26%） | 59.66 ms/tok = 16.8 tok/s | **1.16×** |

9B-FP8 は参照側がロードできない（`model.language_model.lm_head.weight` 找不到）。
よって基準は 27B-FP8 とし、9B は同レバーの派生効果として追跡する。

## カーネル家族別の差（rocprofv3 kernel-trace、prefill 1 バッチ）

| 家族 | shelf | 参照 | 差 | 備考 |
| --- | --- | --- | --- | --- |
| fp8 GEMM prefill | 671.0 ms（400 回・1677.5 µs） | 652.7 ms（400 回・1631.8 µs） | 1.03× | **ほぼ同等**（VGPR 176/LDS 16.9 KB vs 216/1 KB）。ここは触らない |
| prefill attention | 286.9 ms（32 回・8965.8 µs、2.9 TFLOP/s） | **35.5 ms（16 回・2217.2 µs、11.7 TFLOP/s）** | **8.1×** | 参照は `v_prefill_bf16::attention_paged_prefill_bf16_impl`（Q タイル × KV タイルの WMMA、LDS 29.7 KB、VGPR 176、occ 4）。shelf はクエリ 1 本ずつの総当り |
| GDN recurrence | 137.1 ms（3072 回・44.6 µs、chunk 32 刻み） | 76.6 ms（48 回・1595.7 µs） | 1.79× | 参照は 1 層 1 発射（`gdn_recurrence_wmma_lossy`、LDS 35.8 KB、occ 3）。shelf はチャンク毎発射で LDS 44.0 KB・occ 2 |
| SwiGLU + 活性化量子化 | 90.0 ms（silu 18.5 + mul_f32 38.9 + quantize 32.6） | **28.9 ms**（`fused_swiglu_quantize_kernel<17408,17408>` 64 回） | 3.1× | 参照は silu・積・fp8 量子化を 1 pass に融合（中間 materialize ゼロ） |
| rmsnorm / residual / slice / 型変換 | 58.8 ms（rmsnorm 8.0 + residual 11.5 + row_slice 15.2 + bf16→f32 24.1） | 19.7 + 4.3 ms（`fused_resid_rmsnorm_quantize_kernel<5120>` / `fused_rmsnorm_quantize_kernel<5120>`。**量子化も norms に融合**） | 3.0× | shelf は `activation_quantize_fp8_group` 32.6 ms が別 pass |
| rope | 18.5 ms（64 回） | 2.7 ms（32 回） | 6.9× | 参照は q/k を 1 本で処理、f32 中間を作らない |

合計差 約 440 ms（1341 → 901）が上記でほぼ説明できる。

## レバー順位（回収見込み ms × 実装規模）

1. **prefill attention のタイル化 WMMA**（−252 ms・23%）: 参照 kernel を移植。単体で最大の win
2. **SwiGLU + fp8 活性化量子化の融合**（−61 ms・5%）: 参照 `fused_swiglu_quantize_kernel` を移植
3. **norm(+residual) + 活性化量子化の融合**（−60 ms・4%）: 参照 `fused_(resid_)rmsnorm_quantize_kernel`
4. **GDN recurrence の 1 層 1 発射化 + LDS 縮小**（−61 ms・4%）
5. **rope の融合・f32 中間除去**（−16 ms・1%）
6. decode（59.66 → 51.28 ms）: 27B の 1 トークン重み traffic 約 27 GB → 636 GB/s 帯で 42 ms。
   現状 453 GB/s。**fp8 GEMV の帯域向上**と **lm_head（bf16 2.5 GB）の扱い**が主題。
   ※ lm_head を fp8 化するのは量子化段階の変更（手順書 原則 4 のエスカレーション対象）なので、
     まずは bf16 のまま GEMV 効率で寄せる

## 進捗記録

### 段階 1（2026-10-30・本ラウンド）: attention / GDN 発射 / SwiGLU 融合

実装:
- `kernels/attention/attention_paged_prefill.hip`（参照 `v_prefill_bf16` の移植・bf16 のみ）。
  Q 32 行タイル × KV ページ、bf16 WMMA、online softmax、P は bf16 の hi/lo 二成分で PV を二重累加。
  実行器は `attention_paged_prefill_bf16_supported`（rows>1 / D=256 / page=16 / GQA 整数比）で選択。
  1 行（decode）は per-query 経路のまま（rows=1 で 1 CTA に直列化して遅い）。
  `SHELF_ATTN_PREFILL=0` で従来経路に固定（数値 A/B 用）。
- GDN を **1 層 1 発射**化（`launch_gdn_rows`、チャンクはカーネル内ループ）。dispatch 7,682 → 2,636。
- MLP の `silu` + `elem_mul` を新 primitive **`swiglu`** に畳んだ（IR + HIP + CPU 参照 + builder）。
  上り側 bf16→f32 変換 pass と f32 中間が消える。演算は
  `up * (g/(1+exp(-g)))` で既存の 2 段と**逐次同一**（中間 f32 は丸めなし、積の丸め 1 回のみ）。

prefill（27B・2048 tok・Release・GPU 0・1 発射 = chunk 2048）:

| 断面 | 参照 | shelf（段階 1 後） | 比 |
| --- | --- | --- | --- |
| wall | 901.4 ms（2,272 tok/s） | **1,002.7 ms（2,043 tok/s）** | 0.899 |
| GPU 合計 | 878.6 ms | **987.3 ms** | 0.890 |
| dispatch | 1,598 | **2,636**（変更前 7,682） | — |

家族別（shelf / 参照 / 差 ms）:
fp8 GEMM 667.8 / 652.7 / **+15**・GDN 120.5 / 76.6 / **+44**・swiglu 34.1 / 28.9 / +5・
attention 30.1 / 35.5 / **−5**・活性化量子化 29.0 / 0（融合済）/ +29・
rope 19.0 / 2.7 / **+16**・row_slice 15.0 / 2.4 / +13・residual_add 11.3 / 0 / +11。
wall 差 101 ms は GPU 差 109 ms で説明がつき、発射 gap は解消済み。

decode（27B・ctx 2048・128 tok）: shelf 58.92 ms/tok（16.97 tok/s） / 参照 51.28 ms/tok（19.50） → 0.870。

正しさ:
- focused test（`unit.hip_attention`）: prefill attention を f64 参照と比較、
  worst |diff| = 9.3e-4（tol 5e-2）、argmax 一致、受理境界（rows=1 不受理 / D=128 拒否）を検査。
- `architecture.cost_model` の基準値を更新（活性化 159,808 → 151,616 B、
  10,227,712 → 9,703,424 B、silu・elem_mul → swiglu）。
- ctest: cpu 19/19、hip 28/28。
- **token ゲート**: swiglu 融合と GDN 単発射化は**トークン不変**（`SHELF_ATTN_PREFILL=0` の実行が
  変更前シーケンスと完全一致して確認）。prefill attention の WMMA 化は累加順が変わるため
  **原則 4 の命令昇格**に該当: 9B fixture で 13 トークン目（生成 14 番目）に近接タイが 1 件 flip。
  → 基準（`/tmp/fx9_*`・`/tmp/fx27_*`）は per-query 経路のものが incumbent。tiled 経路での
  再基準化は承認待ち（参照実装と同じ構造に寄せた結果なので、参照との比較ではこちらが正しい側）。

### 段階 2（同日・第 2 ラウンド）: GDN の LDS 縮小と decode 断面の会計

実機定数の訂正（`rocprofv3` の agent_info から確認）:
**LDS = 64 KiB/CU、Max_Waves_Per_Cu = 32、Simd_Per_Cu = 2、64 CU**。
手順書の `131072 / LDS` はこのチップでは `65536 / LDS` が正しい（occupancy 算定を訂正）。
GDN は LDS 44,032 B で 1 workgroup/CU だった。

GDN の LDS 削減（`gdn_chunk.hip`）:
- 転置バッファ `kt`（10.2 KB）を廃止し、状態更新側で `s.k` を転置読み
- ピッチを `kQPitch 40→32`・`kMPitch 33→32`・`kDPitch 136→128` に縮小
  （いずれもスカラー読み、または 16 B 境界を保つ 8 の倍数。列アクセスは連続列なので銀行競合は増えない）
- 結果: LDS **44,032 → 31,744 B**（2 workgroup/CU 相当）

**負の結果**: GDN の GPU 時間は 120.5 → 122.1 ms で**変わらなかった**。
prefill の GDN は occupancy 律速ではなく、**チャンク間の再帰状態のグローバル往復**
（f32 の 128×128 をチャンク毎に読み書き）が支配的。occupancy を 2 倍にしても埋まらない。

decode 断面（同一条件で両実装を kernel-trace、prefill + 36 トークン）:

| 家族 | shelf | 参照 | 1 トークンあたり差 |
| --- | --- | --- | --- |
| `gemv_fp8_w8a8_decode1` | 1,518.1 ms（105.4 µs/dis） | 1,448.7（100.6） | +1.9 ms（both: fp8 重み 24.5 GB/token ≈ 580 / 610 GB/s、① DRAM 帯域） |
| GDN decode | 102.4 ms（1,728 回・59.3 µs、スカラー f32） | `gdn_recurrence_wmma_decode1` 21.6（12.5 µs、WMMA） | **+2.25 ms** |
| 活性化量子化 | 103.3 ms（14,800 回・7.0 µs） | 0（生成側へ融合済み） | **+2.9 ms** |
| `gemm_bf16_exact_rows_1`（lm_head 等） | 170.6（48.8） | 150.9（43.3） | +0.55 ms |
| row_slice / f32→bf16 / copyBuffer / rmsnorm / residual | 77.8 | fused_gdn_post 等で融合 | +1.4 ms |
| prefill 部（上記の表と同様） | — | — | — |

**重要な数値差の発見**: 参照の decode1 GDN は
- 再帰状態を **fp16 でメモリ保持**（shelf は f32）
- `kLossy = true` = f16 の hi/lo 補正を**省いた 1 回 WMMA**
で速い。つまり参照の decode GDN は**ロス採用**（手順書 原則 4 でエスカレーション対象）。
shelf が同じ速さを目指すなら (a) 参照と同じロス構成を採る、(b) f32 状態のまま hi/lo 非ロス WMMA
（4 回 WMMA。参照の `kLossy=false` 系）で実装し直す、の二択になる。

### 現在の到達点（27B・GPU 0・TP=1・Release）

| 断面 | 参照 | shelf | 到達率 |
| --- | --- | --- | --- |
| prefill 2048 tok | 901.4 ms（2,272 tok/s） | **1,002.2 ms（2,043 tok/s）** | 0.899 |
| decode（ctx 2048） | 51.28 ms/tok（19.50 tok/s） | **58.92 ms/tok（16.97 tok/s）** | 0.870 |

9B-FP8（参照がロードできないため目標値なし）: prefill **532.2 ms = 3,848 tok/s**、
decode **24.06 ms/tok = 41.6 tok/s**（本日の開始点 508.7 / 25.36 ms から 7.6× / 1.05×）。

### 段階 3（同日・第 2 ラウンド後半）: GDN decode1 の WMMA 化（両方針を作って計測で決定）

方針の決定は利用者の指示どおり「両方作って計測で決める」。実装は
`kernels/gdn/gdn_recurrence_decode1.hip`（参照 `phaseshift_gdn_recurrence_wmma_decode1` の移植。
推測デコード用の `compact_*` / `recurrent_history` / `row_parent` は落とした）。

- 1 workgroup が head_v の 64 列だけ担う（S[128][68] f32 を LDS に張る。LDS 51.7 KB → 1 workgroup/CU）
- KS/QS・rank-1 更新を WMMA（16x16x16 f16）で実行
- 状態のメモリ上の型は shelf のまま **f32**（参照は f16 保存。そこは移植しない）
- 2  variant:
  * `exact` = f16 の hi/lo 二成分で 4 回 WMMA（既定）
  * `lossy` = hi のみ 1 回 WMMA（参照と同じ構成）
- 実行器の選択: 1 行（decode）で `gdn_recurrence_wmma_decode1_supported` が真なら WMMA 版。
  `SHELF_GDN_DECODE1 = off | exact | lossy`（既定 exact、off は逐次 f32）

正しさ（`unit.hip_gdn_chunk` に decode1 ケースを追加、逐次 f32 版と比較）:

| variant | 出力 rel | 最終状態 rel |
| --- | --- | --- |
| exact（hi/lo） | **3.1e-06** | **2.1e-07** |
| lossy（hi のみ・参照と同構成） | 4.6e-04 | 4.0e-04 |

いずれもゲート（2e-2）内。`exact` が 2 桁以上良い。**速度差は 0.14%（56.77 vs 56.85 ms）**なので、
データに基づき `exact` を既定採用（精度をただで買う）。

性能（interleaved A/B・3 ペア、主指標 = 27B decode step_ms_p50、A = incumbent の逐次 f32）:

| ペア | A (ms) | B (ms) | 差 |
| --- | --- | --- | --- |
| 1 | 58.885 | 56.826 | −3.50% |
| 2 | 58.914 | 56.900 | −3.42% |
| 3 | 58.877 | 56.889 | −3.37% |
| p50 | **58.885** | **56.889** | **−3.4%（16.98 → 17.58 tok/s）** |

A 列散布 0.06%・B 列 0.13% に対して差 3.4% → データで決着。
GDN decode の GPU 時間: 59.3 µs/発射 → 20 µs 級（1,728 発射・102.4 → 35 ms 相当）。

`unit.hip_pipeline` の契約更新: 「PP 分割と prefill チャンク分けが数値を変えない」比較は
GDN を逐次版に固定（`SHELF_GDN_CHUNK=0` + `SHELF_GDN_DECODE1=off`）。加えて
**WMMA decode1 を有効にしても 1 ステージと 2 ステージがビット一致**する块を追加した。

### トークン品質（利用者の判断基準「flip 後に意味が通っているか」）

9B-FP8・実 fixture 256 ids・192 step をデコードして比較（`tokenizer.json` で復元）:
- attention: per-query 版と tiled 版で生成 14 番目に flip。両文とも同一話題の連続した英文で、
  事実誤りはない（tiled 版は「chronometer, perfected in 1761」、per-query 版は「adopted fully by 1840」）。
- GDN decode1: off / exact / lossy のいずれも生成 17 番目で分岐するが、
  三者とも意味の通る文章（"required precise latitude" / "demanded precision" 等）。
→ 判定: 容認可。ただし基準そのものは取り直しになる（次項目）。

### 現在の到達点（27B・GPU 0・TP=1・Release・第 2 ラウンド終了時）

| 断面 | 参照 | shelf | 到達率 |
| --- | --- | --- | --- |
| prefill 2048 tok | 901.4 ms（2,272 tok/s） | 1,004–1,011 ms（2,026–2,040 tok/s） | 0.895 |
| decode（ctx 2048） | 51.28 ms/tok（19.50 tok/s） | **56.89 ms/tok（17.58 tok/s）** | **0.902** |

### 段階 4（同日・第 3 ラウンド）: SwiGLu → fp8 活性化量子化の融合と scratch の使い回し

実装:
- `kernels/elementwise/activation_quantize_fp8.hip` に
  **`swiglu_quantize_fp8_group_kernel`**（gate・up を読むだけで bf16 中間材料を書かず、
  fp8 コード/スケールを直接出す）。値は `phaseshift_swiglu_bf16` と同じ順で計算し
  **bf16 に丸めてから**量子化する（现状の 2 段と 1 バイトも違わないようにするため）。
  対応粒度は act_group=128（宣言 blockwise）。それ以外は `hipErrorNotSupported` で従来経路。
- 実行器に**融合計画**（`fuse_for`、グラフ毎に一度）: 対象は「swiglu の出力の消費者が 1 個で、
  その `linear`/`output_projection` が fp8 重みを使う」場合のみ。`SHELF_FUSE_ACTQ=0` で切れる。
- fp8 活性化 scratch の**所有者管理**（`fp8_owner_value_` / rows / kp / 粒度）。
  同じ入力を使う兄弟 GEMM（MLP の gate と up など）は量子化パスを省略する。
  ステップ開始時に失効させる。

正しさ（**逐次同一**を実測で固定）:
- 9B-FP8・実 fixture: 生成列 IDENTICAL、最終デコード logits の md5 一致
  （`/tmp/d1_exact/logits_f32.bin` == `/tmp/fz9/logits_f32.bin`）
- 27B-FP8: 融合 on/off で生成列 IDENTICAL、logits md5 一致
  （`/tmp/f27_off/logits_f32.bin` == `/tmp/fz27/logits_f32.bin`）
- ctest: cpu 19/19・hip 28/28

性能（interleaved A/B・3 ペア。主指標 = 27B prefill wall_ms、A = 融合 off）:

| ペア | A (ms) | B (ms) | 差 |
| --- | --- | --- | --- |
| 1 | 995.846 | 984.606 | −1.13% |
| 2 | 997.560 | 986.156 | −1.14% |
| 3 | 999.363 | 987.271 | −1.22% |
| **p50** | **997.56** | **986.16** | **−11.4 ms（2,053 → 2,077 tok/s）** |

decode（同じ A/B、step_ms_p50）は融合単体では 56.178 → 56.171（散布内で変化なし）。
ただし**兄弟 scratch 使い回し**（融合とは独立に入っている）で 56.889 → 56.17 ms 相当。

kernel trace（27B・chunk 2048・1 発射、`/tmp/tr/sh3`）: GPU 合計 **970.4 ms / 発射 2,428**
（参照 878.6 ms / 1,598）。

- `gemm_fp8_w8a8_prefill_wide_kernel` 670.5（400） / 参照 652.7
- `gdn_chunk_impl` 122.9（48） / 参照 76.6 → **最大の残差（+46 ms）**
- `prefill_bf16::attention_paged_prefill_bf16_impl` 30.4（16） / 参照 35.5 ✓ 上回った
- `swiglu_quantize_fp8_group_kernel` 19.4（64）+ `phaseshift_swiglu_bf16` 12.4（96）
- `rope` 19.1（32） / 参照 2.7・`bf16_row_slice` 15.1（177） / 参照 2.4
- `activation_quantize_fp8_group_kernel` は上位から消えた（192 発射）
- runtime 内部の copy/fill が 759 発射（5.9 ms）。発射数差（2,428 vs 1,598）の主因

未特定: prefill の swiglu 発射は 160 個（64 融合 + 96 非融合）。96 個は fp8 消費者が無く
bf16 GEMM へ feed している計算だが、**どの層・経路か尚未確認**（次ラウンド初点検）。

### 現在の到達点（27B・GPU 0・TP=1・Release・第 3 ラウンド終了時）

| 断面 | 参照 | shelf | 到達率 |
| --- | --- | --- | --- |
| prefill 2048 tok | 901.4 ms（2,272 tok/s） | **984.6–987.3 ms（2,074–2,080 tok/s）** | **0.917** |
| decode（ctx 2048） | 51.28 ms/tok（19.50 tok/s） | **56.17 ms/tok（17.80 tok/s）** | **0.913** |

9B-FP8（参照がロードできないため目標値なし）: prefill 約 520 ms 台、decode 約 23 ms/tok。

### 段階 5（同日・第 4 ラウンド）: decode GEMV のパイプライン化と radiance の ISA 比較

radiance（`references/radiance`）の FP8 kernel を実コンパイルして ISA を取り、
同じ作業をするコードと比較した（`libr4d/r4d_gemm_fp8a8.hip` を
`--offload-arch=gfx1201 -ffp-contract=off` でビルド → fatbin 展開 → `llvm-objdump -d`）。

M=1（decode）用 `gemm_fp8a8_kernel<1,1,256,…>` の ISA 1,592 行の命令構成:

| 命令 | 回数 | 意味 |
| --- | --- | --- |
| `v_wmma_f32_16x16x16_fp8_fp8` | 16 | 本体の WMMA 数（タイル 1 周） |
| `global_load_b128` | 32 | **16 B グローバルロード**（重み・活性化） |
| `ds_read*` / `ds_write*` | 0 | **LDS を使わない**（レジスタ直） |
| `s_wait_loadcnt` / `s_wait_alu` / `s_delay_alu` | 45 / 233 / 98 | 明示的な待ち・間隔調整（= 手書きの多重バッファ） |

→ radiance の decode GEMV の形は「LDS 不使用・16 B ロード・明示的な段先読み」。
shelf も同じ形に寄せる（下表）。`R4D_WAVES`（waves/SIMD を固定してレジスタ予算を決める）
は同じ実験をしたが**効果なし**（VGPR 32 のまま、速度も不変。予算が紐解きにならない）。
radiance のビルドは `-ffp-contract=off`（FMA 合成を禁止）も使っている。

実施と結果（`gemm/detail/linear_fp8_device.h` の `fp8_decode1_body`）:

| 変更 | decode step_ms_p50 | kernel 時間 | 数値 |
| --- | --- | --- | --- |
| 素直なループ（基準） | 56.13 | 105.4 µs/dis（参照 100.6） | — |
| **2 段先読み**（累加順は不変） | **54.93** | 102.4 µs/dis | **ビット同一** |
| 3 段先読み | 54.93（良くならない） | — | — |
| 16 B（uint4）ロード | 54.90（変化なし。8 B×2 はすでに機械結合されていた） | — | ビット同一 |

interleaved A/B（3 ペア、`/tmp/bench_old` と `/tmp/bench_new` を交互実行）:

| ペア | A (ms) | B (ms) | 差 |
| --- | --- | --- | --- |
| 1 | 56.103 | 54.923 | −2.10% |
| 2 | 56.129 | 54.930 | −2.14% |
| 3 | 56.144 | 54.962 | −2.10% |
| **p50** | **56.129** | **54.930** | **−2.13%（17.82 → 18.21 tok/s）** |

prefill はこの変更の影響なし（989.4 vs 989.5 ms）。
**数値は無変更**（9B・27B とも生成列 IDENTICAL、最終 logits md5 一致）で、
ctest は cpu 19/19・hip 28/28。

### 現在の到達点（27B・GPU 0・TP=1・Release・第 4 ラウンド終了時）

| 断面 | 参照 | shelf | 到達率 |
| --- | --- | --- | --- |
| prefill 2048 tok | 901.4 ms（2,272 tok/s） | **986.8 ms（2,075 tok/s）** | **0.914** |
| decode（ctx 2048） | 51.28 ms/tok（19.50 tok/s） | **54.90 ms/tok（18.21 tok/s）** | **0.934** |

### 段階 5b（同日・第 5 ラウンド）: prefill GDN の状態を LDS に張る実験（**却下**）

`gdn_chunk.hip` の再帰状態（128×128）を LDS（fp16、32,768 B）に張って
チャンク毎のグローバル往復を消した。精度は実測で無問題
（`unit.hip_gdn_chunk`: out rel 5.235e-04 → 5.235e-04、state rel 6.019e-04 → 6.602e-04、
tolerance 2e-2 に対し無視できる差）。**速くならかったので却下して元に戻した**。

| 指標 | 段階 4（グローバル往復） | 状態 LDS 版 |
| --- | --- | --- |
| GDN kernel | 2,561.3 µs/dis（LDS 31,744 B・VGPR 216） | 2,603.5 µs/dis（LDS 64,512 B・VGPR 248） |
| 27B prefill wall p50（interleaved 3 ペア） | **983.7 ms** | 985.2 ms |
| workgroup/CU | 2 | 1 |

理由の分析:
- per-head の状態は 64 KiB。同時進行するレイヤぶん（48 head × 64 KiB = 3 MB）しか
  触らないので**L2 に収まる** → チャンク毎の往復は DRAM 帯域を消費していなかった
  （手順書の「L2/L3 に吸われる通信を節約として数えない」に該当。段階 2 で推算した
  「19 GB ≈ 30 ms」は**誤り**だった）。
- そのうえ LDS 64,512 B で 1 workgroup/CU・VGPR 248 となり平行度が半減した。

したがって参照（1,592 µs/dis）との +46 ms は
** occupancy でも通信でもなく、データフローの違い**で説明する。
参照 `recurrence.hip::gdn_recurrence_wmma_body` の形は
  * head_v を 64 列に割って workgroup を 2 倍にする（`LdsSmem` は
    `union { ChunkSmem mid; float S[128][68]; }` で**状態と中間行列が同じ LDS を共有**し、
    LDS 35,840 B に収める）
  * 1 workgroup が全チャンクを LDS 上の状態で回す
shelf の `gdn_chunk_impl` は 1 workgroup が 128 列全部を持つので、
同じ構成にするには列分割（前進代入・delta 解・O 計算の列スコープ書き換え）が必要。
**次ラウンドの本命**（推定 −40 ms、350 行規模の移植）。

なお prefill の swiglu 発射 160 個のうち 96 個が非融合のままという点も未特定
（融合数はチャンク数に比例して 64→128、非融合の 96 は chunk 数に不変 =
**prefill チャンク.graph 由来ではない**可能性。decode/MTP/ビジョン塔の確認が必要）。

### 現在の到達点（27B・GPU 0・TP=1・Release・第 5 ラウンド終了時）

| 断面 | 参照 | shelf | 到達率 |
| --- | --- | --- | --- |
| prefill 2048 tok | 901.4 ms（2,272 tok/s） | **983.7 ms（2,082 tok/s）** | **0.916** |
| decode（ctx 2048） | 51.28 ms/tok（19.50 tok/s） | **54.90 ms/tok（18.21 tok/s）** | **0.934** |

### 段階 6（同日・第 6 ラウンド）: GDN を head_v 64 列グループに分割（採用）

段階 6 で判明した最重要の事実: **GDN の launch は workgroup が num_v_heads（48）個しか
なく、64 CU を使い切っていなかった**。また workgroup を増やすと LDS 枠が効いて
**1 workgroup/CU に落ちると 96 個 ÷ 64 CU = 1.5 ラウンド（末尾で 2 倍）になり、
かえって遅くなる**。

| 実装 | LDS | workgroup/CU | GDN µs/dis | 判定 |
| --- | --- | --- | --- | --- |
| 段階 4（128 列・状態グローバル） | 31,744 B | 2 | 2,561 | 基準 |
| 段階 5b（128 列・状態 LDS fp16） | 64,512 B | **1** | 2,603 | 却下 |
| 6a（64 列分割・状態 LDS f32） | 56,320 B | **1** | **4,444** | 却下（1.5 ラウンドの末尾） |
| **6b（64 列分割・状態グローバル）** | **23,552 B** | **2** | **2,355** | **採用** |

6b の変更点（`gdn_chunk_impl`）:
- `vg = blockIdx.x % 2` / `v_head = blockIdx.x / 2` / `v0 = vg*64` で workgroup を 2 倍に
- 列スコープを全部 64 列に（前進代入は 64 スレッド、タイル数 16→8・64→32、
  `kDPitch = DV2 = 64`）
- 状態は LDS に張らずグローバルのまま（LDS 23,552 B で 2 workgroup/CU を保つため）

性能（interleaved A/B・3 ペア。A = 段階 5 終了時バイナリ `/tmp/bench_gold`、
基準 `/tmp/bench_gold` は GEMV 先読み込み）:

| ペア | A (ms) | B (ms) | 差 |
| --- | --- | --- | --- |
| 1 | 985.289 | 979.858 | −0.55% |
| 2 | 986.289 | 981.190 | −0.52% |
| 3 | 987.571 | 981.690 | −0.59% |
| **p50** | **986.29** | **981.19** | **−5.1 ms** |

GDN 家族: 122.9 → **113.0 ms**（参照 76.6 ms、差 +36 ms）。GPU 合計 970.4 → 962.6 ms。
decode はunchanged（rows=1 は chunk 経路を通らない）54.91 ms/tok。

**数値は無変更**（`unit.hip_gdn_chunk` の誤差値が段階 4 と完全一致:
rows=256 で out rel 5.235e-04・state rel 6.019e-04。9B・27B 実 fixture の生成列 IDENTICAL、
最終 logits md5 一致）。ctest cpu 19/19・hip 28/28。

残る +36 ms の候補（推定内訳）:
- 前進代入が 64 スレッドしか動かさず 32 段の依存ループ（8 wave 中 6 が待ち）。
  参照は A/L/W の連鎖積で三角求解をブロック化しており、ここが最大の違い
- 列非依存部（k の LDS  staging、K·Kᵀ、Q·Kᵀ、M 作り）が workgroup 間で重複
- 1 workgroup あたりのチャンク数（2048 行 ÷ 32 = 64 回）

### 現在の到達点（27B・GPU 0・TP=1・Release・第 6 ラウンド終了時）

| 断面 | 参照 | shelf | 到達率 |
| --- | --- | --- | --- |
| prefill 2048 tok | 901.4 ms（2,272 tok/s） | **981.2 ms（2,086 tok/s）** | **0.919** |
| decode（ctx 2048） | 51.28 ms/tok（19.50 tok/s） | **54.91 ms/tok（18.21 tok/s）** | **0.934** |

### 段階 7（同日・第 7 ラウンド）: GDN の q を LDS に張る（ピッチで LDS 上限に衝突）

Q·Kᵀ（セクション 3）と Q·S（セクション 4）が q をグローバルから 2 回読んでいたので、
セクション 1 で 1 回 LDS に張るようにした（換算 `static_cast<_Float16>` は 1 回、値は同一）。

- `q` のピッチを `kPitch`(136) と同じにした版: GDN **4,870 µs/dis**（2 倍悪化）、
  wall 1,070.9 ms。LDS 32,256 B で 2 workgroup/CU の枠に収まらず 1/CU に落ちた
  （96 workgroup ÷ 64 CU = 1.5 ラウンドの末尾が効く）
- ピッチを 132（264 B = 64 銀行で 2 進ずらし、128 だと行ごとに同じ銀行へ衝突）へ:
  GDN **2,357 µs/dis**、wall **978.6 ms**（段階 6 の 979.9–981.7 ms よりさらに小幅に良い）

**実測から得た上限**: このチップ（LDS 64 KiB/CU）で 2 workgroup/CU を保てる
workgroup あたり LDS は **約 32,200 B まで**。32,256 B を実際に超えた版は 1/CU に
落ちて 2 倍遅くなった。以後の GDN 改修（状態 LDS 常駐・大きなチャンク）は
この枠の中でやる必要がある。

数値は無変更（`unit.hip_gdn_chunk` rows=256: out rel 5.235e-04・state rel 6.019e-04 で
段階 6 と完全一致。9B・27B 実 fixture の生成列 IDENTICAL）。ctest cpu 19/19・hip 28/28。
decode は 54.92 ms/tok（GDN decode 経路は別 kernel のため不変）。

### 現在の到達点（27B・GPU 0・TP=1・Release・第 7 ラウンド終了時）

| 断面 | 参照 | shelf | 到達率 |
| --- | --- | --- | --- |
| prefill 2048 tok | 901.4 ms（2,272 tok/s） | **978.6 ms（2,093 tok/s）** | **0.921** |
| decode（ctx 2048） | 51.28 ms/tok（19.50 tok/s） | **54.92 ms/tok（18.21 tok/s）** | **0.934** |

### 段階 8（同日・第 7 ラウンド後半）: CP keep-alive（参照は既定で ON だった）

参照 PhaseShift は executor 既定で **1 ms ごとに別 stream へ `hipEventRecord` する
スレッド**を張っている（`models/qwen35/runtime/executor.hip` の
`kKeepAliveIntervalUs = 1000` / `PHASESHIFT_KEEPALIVE=0` で無効）。
bench 側にも `--gap-keepalive US` / `--gap-keepalive-flags` があり、
`docs/perf/methodology.md` に「micro kernel（4〜20 µs 級）は idle clock ramp の影響が
大きい。**preroll を入れる**」と書いてある。つまり **キューが 1 ms 以上空くと
CP/クロックが冷えて発射が高くなる**。shelf に同じ仕組みが無かった。

実装（`step_executor.cpp`）: `HipStepExecutor::KeepAlive`（非ブロッキング stream +
`hipEventDisableSystemFence` の event を 1 ms ごとに record）を ctor で開始、
dtor で join → sync → destroy。`SHELF_KEEPALIVE=0` で無効。device メモリは取らない
（VRAM 予算に計上しない）。数値は一切変えない。

interleaved A/B（3 ペア、27B decode step_ms_p50）:

| ペア | shelf KA=0 | shelf KA=1 | 参照 KA=1 | 参照 KA=0 |
| --- | --- | --- | --- | --- |
| 1 | 54.905 | **54.667** | 51.375 | 51.564 |
| 2 | 54.936 | **54.722** | 51.359 | 51.617 |
| 3 | 54.934 | **54.695** | — | — |
| p50 | 54.934 | **54.695** | 51.37 | 51.56 |

- shelf decode **−0.44%**（散布 0.05% に対し全ペアで改善）。prefill は変化なし
  （982.4/982.7 vs 982.9/983.7。チャンク 1 発射でキューが空かないため）
- 参照側も KA で −0.38%。**つまり目標値 51.37 ms は keep-alive で水増しされた数字ではない**
  （同じ条件に揃えても差は本物）。ただし同等条件にすることは必要で、既定 ON にした
- ctest cpu 19/19・hip 28/28（heartbeat スレッド入り）、9B 実 fixture の生成列 IDENTICAL

### 現在の到達点（27B・GPU 0・TP=1・Release・keep-alive 既定 ON）

| 断面 | 参照 | shelf | 到達率 |
| --- | --- | --- | --- |
| prefill 2048 tok | 901.4 ms（2,272 tok/s） | **978.6–983.7 ms（2,082–2,093 tok/s）** | **0.920** |
| decode（ctx 2048） | 51.28 ms/tok（19.50 tok/s） | **54.70 ms/tok（18.28 tok/s）** | **0.937** |

### 段階 9（同日・第 8 ラウンド）: rope を 1 トークン 1 workgroup・bf16 入力に（−12 ms）

実行器が rope を呼ぶとき、**(トークン × ヘッド) を行**として渡していた
（shape が `{rows_, nq, D}` の rank-3 なので `dim(rank-1)=D=256` を features にしていた）。
pair kernel は 1 行ぶんのヘッドをループ内で回し sincos は rotary/2 個だけ計算する設計なので、
この呼び出し方だと **同じ sincos をヘッド数ぶん（27B で q 24 個・kv 4 個）作り直していた**
（倍精度 `sincos` × 1.57 M 回/レイヤ）。参照（phaseshift）は 1 トークン 1 workgroup で呼ぶ
（grid 262,144 = 2,048 workgroup × 128）ので 12 倍速かった。

変更（`step_executor.cpp` の rope 節、新しい kernel は `attention_rope.hip`）:
- features = ヘッド数 × D、rows = step_rows_（トークン数）、positions = `rowpos_dev_`
  （ステップの per-トークン位置。`posrep` のホスト側ループ + 転送は不要になった）
- **bf16 入力版 `phaseshift_rope_bf16_pair` / `launch_rope_bf16_pair`** を追加し、
  f32 中間材料（`launch_bf16_to_f32` の通過）を廃止。bf16→f32 は損失のない widening、
  非 rotary 部はそのままコピーなので**値はビット同一**。head_dim≠256 や rotary≠64 は
  従来経路へ戻る（`hipErrorNotSupported` で降格）

計測（27B・chunk 2048・kernel trace）:

| 項目 | 変更前 | 変更後 | 参照 |
| --- | --- | --- | --- |
| rope | 19.1 ms（32 発射・597 µs/dis） | **1.65 ms**（32 発射・51.7 µs/dis） | 2.69 ms（84.1 µs/dis） |
| bf16→f32 変換 pass | 6.6 ms | **0**（消えた） | — |
| GPU 合計 | 962.6 ms | **950.2 ms** | 878.6 ms |
| prefill wall（median of 3） | 978.6 ms | **966.7 ms（2,118 tok/s）** | 901.4 ms |
| decode step p50 | 54.70 ms | **54.48 ms（18.36 tok/s）** | 51.28 ms |

正しさ: `unit.hip_attention` ほか ctest cpu 19/19・hip 28/28。
9B・27B 実 fixture の生成列 **IDENTICAL**、最終 logits md5 **一致**（数値は無変更）。

### 目標の切り替え（radiance）と fp8 KV についての確認

本ラウンドで目標が `references/radiance` 基準に切り替わった。実測しようとしたが、
そのツリーのビルドには GPU kernel plugin（`<radiance-home>/kernels/*.so`）が無く
（`find build -name "*.so"` は空）、server は startup で
`E plugin directory '.../kernels': No such file or directory` と訴えて止まる
（checkpoint 自体は `model: qwen35 arch=qwen35 quant=fp8_e4m3 64 layers` と正しく認識）。
つまり radiance 実測には plugin のビルド／install が必要（次工程）。

fp8 KV（利用者の推定）については、radiance 自身の計測が docs/GUIDE.md に載っており、
**速度ではなく KV プールの容量のため**だと分かる（Qwen3.8-27B・TP=2・41k prefill・decode 1/16 streams）:

| `--kv-cache-dtype` | 41k prefill | decode 1 / 16 streams | プール |
| --- | --- | --- | --- |
| fp8 | 9.76 s | 134-140 / 550-555 tok/s | 518,800 tok |
| fp8_v4 | 9.76 s | 130-139 / 553-557 | +28% |
| k6_v4 | 9.84 s | 136-138 / 553-561 | +46% |

「The decode spread is run to run; the three are the same speed there」。
radiance が速い理由として doc が数値付きで挙げているのは別要素、いずれも**ロス変更**:
- `--attn-prefill-fp8 qk|pv|both`: prefill attention の QKᵀ/PV を e4m3 WMMA に（41.6 → 27.9 ms/layer、
  221K prompt 2665 → 3210 tok/s、mean KL 0.055-0.058、top-1 96%）
- `--tp-wire wht6`: TP の all-reduce を rotated 6-bit に（24K prompt 4655 → 5312 tok/s、TP=2 のみ）

### 現在の到達点（27B・GPU 0・TP=1・Release・第 8 ラウンド終了時／基準は phaseshift）

| 断面 | phaseshift | shelf | 到達率 |
| --- | --- | --- | --- |
| prefill 2048 tok | 901.4 ms（2,272 tok/s） | **966.7 ms（2,118 tok/s）** | **0.932** |
| decode（ctx 2048） | 51.28 ms/tok（19.50 tok/s） | **54.48 ms/tok（18.36 tok/s）** | **0.941** |

### 段階 10（同日・第 9 ラウンド）: PhaseShift の KV dtype を実測で確かめた

`phaseshift-bench pp` には `--kv-cache-dtype bf16|fp8`（既定 bf16）がある。
同一条件で 2 回ずつ交互に走らせた（TP=1・GPU 0・Qwen3.8-27B-FP8）:

| prompt | bf16 KV (wall ms) | fp8 KV (wall ms) | 差 |
| --- | --- | --- | --- |
| 2048 | 907.1 / 912.1 | 903.7 / 913.6 | −0.4% / +0.2%（差なし） |
| 4096 | 1874.6 / 1887.5 | 1871.9 / 1882.8 | **−0.14% / −0.25%** |

`tg`（decode）には KV dtype フラグが無く bf16 固定。8192 トークンは arena 超過で
両 dtype とも起動失敗（`arena.hip:355 insufficient memory`）だったので、
このハーネスで KV 長をさらに伸ばして比較することは出来ない。

結論（実測）: **2048〜4096 トークンの範囲では PhaseShift の fp8 KV は速度差にならない**
（0.2% 前後、散布内）。KV 読みは prefill attention で token 数に比例するので、
差が出るのはもっと長いコンテキストのはずだが、それはこの bench では測れない。
shelf の現行ギャップ（2048 で +61 ms、4096 で +129 ms = 6.8% / 6.5%）は
fp8 KV では埋まらない、というのが本ラウンドの確定事項。

なお shelf の 4096 トークン実測（chunk 2048・bf16 KV）: **2004.6 / 2003.7 ms**。

### 段階 11（同日・第 10 ラウンド）: radiance を動かせる状態にした（計測は次）

radiance（`references/radiance`）の GPU kernel plugin はビルド済みで無く、engine と
plugin の ABI もずれていた。再現手順:

1. cmake の HIP 判定が `hipcc` の場所から ROCm root を逆算する（`CMakeLists.txt:144-152`）。
   venv 内の `hipcc` だと `<venv>/include` を探して失敗するので、SDK wheel の方を明示する:
   `cmake -S . -B build -DROCM_PATH=/opt/zen/.venv/lib/python3.12/site-packages/_rocm_sdk_devel
    -DRAD_HIPCC=.../_rocm_sdk_devel/bin/hipcc`
2. `ninja -C build libr4d` → `build/radiance_home/kernels/libr4d.so`（107 TU）
3. チェックイン済みの `build/bin/radiance`（10/28 ビルド）は ABI 25 で、ツリーのソースは
   **ABI 31** → `E plugin ... built against ABI version 31; this engine is version 25 -- refused`。
   `ninja -C build`（569 ターゲット）で engine と arch plugin を作り直す
4. `--radiance-home build/radiance_home` で起動すると checkpoint は正しく読める
   （`model: qwen35 arch=qwen35 quant=fp8_e4m3 64 layers 262144 ctx`）

ただし **1 枚（R9700 31.86 GiB）での 27B-FP8 サーブは OOM で止まる**:
weights 25.94 GiB + KV 2.78 GiB + activation arena 745 MiB まで積まれたところで
`E startup stopped: out of memory`。しかもログに
`I activation arena: not lent -- the 'qwen35_fp8' plugin does not declare at a smaller step size`
とあり、arena を小さな step size へ縮約できない（doc の数値は TP=2 のもの）。
→ 計測は `--max-num-batched-tokens 1024`（arena を縮める）か TP=2 で行う。

### 段階 12（同日・第 11 ラウンド）: radiance の基準値を実測した

1 枚で動かすために `--max-num-batched-tokens 1024`（arena 426 MiB まで縮小）にした。再現手順:

    cd /opt/zen/work/references/radiance
    HIP_VISIBLE_DEVICES=0 build/bin/radiance --radiance-home build/radiance_home \
      --model /opt/zen/work/models/Qwen3.8-27B-FP8 --kv-cache-dtype bf16 \
      --num-speculative-tokens 0 --max-model-len 3072 --max-num-batched-tokens 1024 \
      --max-num-seqs 2 --port 8123

`/v1/completions` は**トークン id 配列の prompt を受け、`timings` に
`prompt_ms` / `predicted_ms` / `predicted_n` を返す**（client 側の計測は不要）。
同一 prompt の繰り返しは prefix cache に吸われるので実行ごとに id をずらした
（`cached_tokens=0` を確認済み）。4 実行（1 回目は warmup）:

| 実行 | prompt_ms（2048 tok） | tok/s | predicted_ms（64 tok） | decode tok/s |
| --- | --- | --- | --- | --- |
| 2 | 733.0 | 2,794 | 2,940.5 | 21.42 |
| 3 | 736.2 | 2,782 | 2,940.6 | 21.42 |
| 4 | 738.9 | 2,772 | 2,941.6 | 21.42 |

**radiance 基準（R9700 1 枚・TP=1・bf16 KV・推測デコードなし・chunk 1024）**
- prefill: p50 **734.6 ms = 2,788 tok/s**（1,024 tok chunk × 2 = 1 chunk 367 ms）
- decode: **46.7 ms/tok = 21.42 tok/s**

比較（同モデル・同 GPU 0・bf16 KV）:

| 断面 | radiance | phaseshift | shelf | shelf / radiance |
| --- | --- | --- | --- | --- |
| prefill 2048 tok | **734.6 ms（2,788 tok/s）** | 901.4 ms（2,272） | 966.7 ms（2,118） | **0.760** |
| decode ctx 2048 | **46.7 ms/tok（21.42）** | 51.28 ms/tok（19.50） | 54.48 ms/tok（18.36） | **0.857** |

radiance は chunk を 1024 に絞ってこの速度なので、**1 トークンあたりの prefill 速度は
phaseshift より 19% 速い**（367 ms/1024 tok vs 451 ms/1024 tok）。
さらに速めるスイッチはいずれもロス変更として doc に列挙されている:
`--attn-prefill-fp8 both`（attention 41.6 → 27.9 ms/layer）、`--tp-wire wht6`（TP=2 のみ）、
`--kv-cache-dtype fp8|fp8_v4|k6_v4`（速度ではなく KV 容量）。

### 段階 13（同日・第 12 ラウンド）: radiance の op 一覧を取得（`--profile-ops`）

`--profile-ops` は「every op を計るため同期を戻す。measured な数字には使うな」と警告する
（実測も prompt 734.6 → 801 ms、decode 46.7 → 49.6 ms/tok に膨らんだ）。それでも
**op の分解能**が欲しいので 1 実行分だけ取った（Qwen3.8-27B-FP8、1 枚、bf16 KV、推測なし、
chunk 1024、prompt 2048 + 32 tokens）。radiance が実際に張っている op と回数:

| op | 回数 | 内容 | shelf 側の対応 |
| --- | --- | --- | --- |
| `gemm_nt_q` | 7,520 | 量子化 GEMM（本体） | `gemm_fp8_w8a8_*` ✓ 相当 |
| `gemm_nt_q_gated` | 1,984 | **gate を GEMM の epilogue に融合した GEMM** | shelf は swiglu + 量子化が別 op |
| `rmsnorm_quant_fp8` | 4,224 | **rmsnorm + fp8 活性化量子化の融合** | 未実装（段階 10 で計画中と記録） |
| `gated_quant_fp8` | 128 | gate 付き量子化 | 未実装 |
| `gdn_chunk_scan` / `gdn_kkt_solve` / `gdn_conv_prep` / `gdn_conv_recurrent_update` / `gdn_gated_rmsnorm` | 96 / 96 / 96 / 1,488 / 96 | **GDN を 5 つの op に分解**（三角求解 `kkt_solve` を独立 op に） | shelf は 1 kernel（`gdn_chunk_impl`）に全部入っている |
| `attn_paged` | 528 | paged attention（bf16 KV） | ✓ 相当 |
| `logits_gemm` | 32 | 最終 lm_head（行を絞った形） | ✓ 相当 |
| `rope`（`mode=imrope sections=11 11 10`, rotary 64） | 多数 | mrope 対応の rope | shelf は rope 1 pass ✓（段階 9 で 1.65 ms） |

**radiance が shelf より速い理由の最有力候補（op の証拠）**:
1. `rmsnorm_quant_fp8` / `gated_quant_fp8` / `gemm_nt_q_gated` — **norm・gate・量子化が
   全部 GEMM の前後の op に融合されていて、活性化の往復 pass が存在しない**
   （shelf は `activation_quantize_fp8_group_kernel` をまだ 192 発射もっている）
2. GDN を `chunk_scan` + `kkt_solve` + `conv_prep` + `recurrent_update` に分けており、
   三角求解（`kkt_solve`）が独立の小さな op になっている
   （shelf は 1 kernel 内で前進代入を 64 スレッドで直列に回している = 段階 6 の分析と一致）

次は (1) の融合（rmsnorm→fp8 量子化、gate→GEMM epilogue）から着手する。

### 段階 14（同日・第 14 ラウンド）: rmsnorm + fp8 活性化量子化の融合（decode −0.74%）

radiance の op 一覧に倣い、norm の bf16 出力を読み直して量子化する pass を 1 op に畳んだ
（`elementwise_rmsnorm.hip` の `rmsnorm_quant_fp8_kernel` / `launch_rmsnorm_bf16_quantize`）。
実行器は `fuse_for` で「その rmsnorm 出力の消費者がすべて fp8 GEMM」の値だけを対象にする。
`SHELF_FUSE_NORMQ=0` で切れる。bf16 出力も従来どおり書く（PP 受け渡し・デバッグはそのまま）。

**バイト同一性に効く罠を 1 つ実測で踏んだ**: block サイズを固定 256 で走らせた版は
27B の logits が変化した（`d08df139…` → `3764a672…`）。既存経路は
`rmsnorm_bf16_pf_oneplus_block_size(features)`（32/64/128/256）でスレッド分担が変わるため、
sumsq の合計順が変わるからだ。同じ block size を使うように直したうえで再検証:

- 9B・27B 実 fixture: 生成列 **IDENTICAL**、最終 logits md5 **一致**（`d08df139…`, `c0097…`）
- ctest cpu 19/19・hip 28/28

interleaved A/B（`SHELF_FUSE_NORMQ` 0/1、3 ペア / 2 ペア）:

| 断面 | off | on | 差 |
| --- | --- | --- | --- |
| 27B prefill (pp 2048) | 967.8 / 968.3 / 971.5 | 966.6 / 968.9 / 969.5 | **±0（散布内）** |
| 27B decode (tg p50) | 54.536 / 54.548 | **54.141 / 54.115** | **−0.74%（−0.40 ms/tok）** |

prefill で速くならない理由も trace で確認した（chunk 2048）:
融合 80 発射 4.96 ms、残った `activation_quantize` 112 発射 5.25 ms
（融合前は rmsnorm 161 発射 7.08 ms + 量子化 192 発射 ≈9 ms）。
**prefill では pass 統合の節約 ≈ 発射あたりコストと相殺**されるが、
**decode は発射数が効く**（1 ステップあたり 62 発射減少）ので確実な利得がある。

### 現在の到達点（27B・GPU 0・TP=1・Release・第 14 ラウンド終了時）

| 断面 | radiance | phaseshift | shelf | / radiance |
| --- | --- | --- | --- | --- |
| prefill 2048 tok | **734.6 ms（2,788 tok/s）** | 901.4 ms | 968.9 ms（2,114 tok/s） | **0.758** |
| decode ctx 2048 | **46.7 ms/tok（21.42）** | 51.28 ms/tok | **54.13 ms/tok（18.47）** | **0.863** |

### 第 15 ラウンド: トレース集計の罠（カーネル名の prefix 共有）を記録

decode の内訳を取り直したとき、`fp8_w8a8` と `decode1` を部分文字列で分類したために
**prefill の wide kernel と decode の GEMV、GDN の decode1 が同じ桶に混ざった**
（見かけ上 102 → 145 µs/dis に「悪化」したように見えたのはこのため）。
名前の実例:

- prefill fp8 GEMM: `shelf::hipk::(anonymous namespace)::gemm_fp8_w8a8_prefill_wide_kernel(...)`
- decode fp8 GEMV: `...linear_fp8_detail::fp8_decode1_body` を呼ぶ `phaseshift_gemm_fp8_w8a8_*`
- GDN decode: `gdn_recurrence_wmma_decode1_impl`

正しい集計は **`(anonymous namespace)` を含む全称で分類する**（`split('(')[0].split('::')[-1]`
は無名名前空間があると壊れて空の桶が増える。前回までの集計で空欄が多かった原因もこれ）。
なお `gemm_bf16_exact_rows_body`（decode の bf16 GEMV、48.8 µs/dis）への
2 段 prefetch は、fp8 側の実測（−2.1%）と byte 一致制約（累加順固定）を考えると
見込み 0.05〜0.1 ms/tok で、他のレバーより小さいと判断して見送った。

### 段階 16（同日・第 17 ラウンド）: split-KV decode attention は効果なし、そして**再現性バグ**を見つけた

**(a) split-KV の配線実験**（`launch_attention_paged_bf16_split` + `..._split_reduce`、
radiance の split-KV decode と同じ形。実装は既にあって実行器が呼んでいなかっただけ）:

| ctx | splits=0 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- |
| 2048 | 54.064 | 54.087 | 54.099 | 54.136 |
| 4096 | 55.871 | — | 55.874 | 55.875 |

workgroup 数は 24 → 64（kv_heads 4 × splits 16）になるが速くならない。
**decode attention の律速は workgroup 数（級 ③）ではない**と分かったので、
配線は入れていない（コード内のコメントに残した）。

**(b) 実行間の再現性が壊れていた**（同一バイナリ、27B 実 fixture 96 step）:

    keep-alive on  : d08df139… / 3764a672… / d08df139…   ← 実行ごとに変わる
    keep-alive off : d08df139… / d08df139… / d08df139…   ← 一致

1 ms ごとに別 stream へ event を積むだけのスレッドを動かすと結果が変わる。
つまり**どこかに未同期の依存**（stream_ の仕事と別 stream/スレッドが関わる経路、
あるいはarena scratch の寿命）が残っている疑いが強い。切り分けは未完了。
**第 8 ラウンドで入れた keep-alive がこの実験の再現性を壊していた**ため、
それ以前の「バイト一致」検証も 1 回実行 based であり証拠力が落ちる（同じ md5 に
会ったのは運が良かった可能性がある）。今後の gate は**同じバイナリで 3 回走かせて
一致**を条件にする。

**(c) 利用者の指示で keep-alive を撤去した**（`step_executor.cpp` の
`start_keepalive` / `stop_keepalive` / `KeepAlive` / `SHELF_KEEPALIVE` を全部削除）。
撤去後の実測（27B・GPU 0・Release）:

| 断面 | keep-alive あり（段階 8 時点） | 撤去後 |
| --- | --- | --- |
| decode step p50 | 54.06 ms | 54.34 ms（+0.5%） |
| prefill wall（median of 3） | 966.6〜969.5 ms | **1,018.6 / 1,020.2 ms（+5.2%）** |

prefill での_loss は阶段 8 の「prefill 中立」の測定と食い違う。
（当時は `--gap-keepalive` を交互に立てる A/B で、chunk 実行中の idle ギャップが
違っていた可能性が高い。）**再現性を取り戻すと引換に prefill が 5% 遅くなる**のが
本ラウンドの確定事項。撤去後も 9B・27B の生成列はベースラインと一致、
`ctest` hip 28/28・cpu 19/19。

### 段階 17（同日・第 18 ラウンド）: GDN 後段の norm で f32 中間材料を消す（−9 ms / −0.21 ms/tok）

`rmsnorm_gated` は GDN 本体の出力（f32）を受けるとき、**f32 で置き → 別 pass で
bf16 に変換**していた（`f32_to_bf16_k` が prefill で 48 発射 16.6 ms、decode で
49 発射/step）。本体は InT==F32 の経路で store 前に `rms_round_to_bf16_f32` を
噛ませるので（= 変換 pass と同じ丸めを先にする）、`launch_rmsnorm` の
**f32 入力 → bf16 出力**（vendored 側は `!in_b && out_b` を弾いていた）を通すだけで
pass が丸ごと消える。PER_FEATURE / PER_GROUP × DIRECT / ONE_PLUS を追加し、
実行器は scratch も bf16 ぶんだけ取るようにした。

実測（27B・GPU 0・Release・keep-alive 撤去後）:

| 断面 | 変更前 | 変更後 |
| --- | --- | --- |
| prefill wall | 1,018.6 / 1,020.2 ms | **1,009.7 ms（2,028 tok/s）** |
| decode step p50 | 54.34 ms | **54.13 ms** |

正しさ（**同一バイナリ 3 回一致を必須ゲートにした**）:
27B fixture 96 step × 3 → すべて `d08df139…`（ long-standing ベースラインと同値）、
9B → `c097951350df` で生成列 IDENTICAL。`ctest` hip 28/28・cpu 19/19。

### 現在の到達点（27B・GPU 0・TP=1・Release・第 18 ラウンド終了時）

| 断面 | radiance | phaseshift | shelf | / radiance |
| --- | --- | --- | --- | --- |
| prefill 2048 tok | **734.6 ms（2,788 tok/s）** | 901.4 ms | 1,009.7 ms（2,028） | **0.728** |
| decode ctx 2048 | **46.7 ms/tok（21.42）** | 51.28 ms/tok | 54.13 ms/tok（18.47） | **0.863** |

keep-alive 撤去（利用者の指示）で prefill は 5.2% 戻せなくなっている点に注意
（radiance 比は撤去前 0.758 → 0.728）。再現性を壊す経路を塞いだ結果なので、
取り戻すなら「CP を冷やさない別実装」（例: ワーク stream 自体の中に短い
no-op kernel を混ぜる、あるいは発射間隔を詰める）＋同じ md5 が 3 回出ることを
確認してから入れる。

### 段階 18（同日・第 19 ラウンド）: keep-alive が稼いでいた 5% は「クロック加熱」ではなかった

撤去後の prefill を実行回数を増やして測り直した:

| runs | median | min | p90 |
| --- | --- | --- | --- |
| 3 | 1,010.6 ms | 1,010.0 | 1,011.2 |
| 9 | 1,012.8 ms | 1,011.5 | 1,014.1 |

連続実行で**よくならずむしろわずかに落ちる**ので、idle クロックのランプ（参照 doc が
言う 4〜20 µs カーネル向けの加熱効果）では説明できない。keep-alive が稼いでいた
約 50 ms は **発射パス側の効果**（1 ms ごとに別 stream へ積むことで CP のコマンド
処理が生きたまま、発射間ギャップが埋まる）と考えるのが自然。chunk 2048 の prefill は
発射数が多く、1 発射あたり 10 µs 浮くと約 48 ms で、実測差と一致する。

→ 同じ 5% を**再現性を壊さず**取り戻す道は keep-alive の再導入ではなく、
**発射そのものを減らす／まとめる**こと。具体的には
1. 融合（norm→量子化 ✓、GDN→out_proj の量子化、gate の epilogue）で発射数を減らす
2. **HIP graph 化**: ステップの発射列は shape が固定的で arena も静的なので、
   1 ステップ分を graph に capture して replay できる（発射あたりのホストコストが消える）。
   ただし capture 中に生成した address を再 replay で維持する前提の確認が必要
   （`declare` 相の offsets 表が同じグラフで同じ配置を出すことは既に検証済み）

keep-alive の撤去は正しい判断だった（撤去前は同一バイナリで logits が変っていた）。
未同期の依存が残っている可能性は依然高いので、graph 化の前に
「2 スレッド・2 stream で発射しても md5 が揺れない」ことを確認するのが安全。

### 段階 19（同日・第 20 ラウンド）: radiance と**完全に同一条件**（chunk 1024）で測った

radiance は 1 枚では 2048 トークンの chunk を arena に載せられない
（`the 'qwen35_fp8' plugin does not declare at a smaller step size`）ので、
同一条件は chunk 1024 である。shelf をその条件で測り直した（27B・GPU 0・bf16 KV）:

| 条件 | radiance | shelf | shelf / radiance |
| --- | --- | --- | --- |
| pp 2048 tok・**chunk 1024** | 734.6 ms（2,788 tok/s） | **1,059.9 / 1,061.4 ms（1,931 tok/s）** | **0.693** |
| 1,024 tok chunk あたり | 367.3 ms | 530.0 ms | 0.693 |
| pp 2048 tok・chunk 2048 | 起動せず | 1,010.6 ms | — |
| tg ctx 2048（chunk 1024） | 46.7 ms/tok | **54.18 ms/tok** | **0.862** |

shelf は chunk を 2048 に増やすと 1,010 ms まで上がる（radiance は 1 枚では 2048 が
載らない）ので、スケール性はこちらがよい。それでも 1 chunk あたりの仕事量は
radiance の 1.45 倍であり、差は前ラウンドまでの解析どおり
**発射数と非 GEMM pass**（融合不足）に集中している。

### 段階 20（同日・第 21 ラウンド）: GDN 後段 gate → fp8 量子化の融合は**まだ入れられない**と分かった

radiance の `gated_quant_fp8` 相当を狙って、`rmsnorm_gated` の末尾掛け算
（`launch_swiglu_bf16`）を既存の `launch_swiglu_quantize_fp8` に置き換えた。
消費者スキャンは `reshape` を透過させる必要があった（GDN 後段は
`[tokens, heads, dv]` → reshape 経由で out_proj（K=6144）へつながるため、
透過でないと融合対象に入らない）。

- 速度: prefill 1,009.7 → **989.4 ms（−20 ms）**、decode は 54.33 ms（変わらず）
- 正しさ: 27B fixture の logits が `d08df139…` → **`b1b7085c…`**（3 回一致・確定的に変化）

原因は融合 kernel が **bf16 出力を書かない**こと。消費者が owner 再利用に失敗する
経路（out_proj の kp / act_group が平坦幅 6144 と食い違う）で
**未初期化の bf16 を読む**ため値が変わる。安全化には
`launch_swiglu_quantize_fp8` に `out`（bf16、nullable）引数を追加して
**bf16 と codes の両方を書く**形が必要（書いていればフォールバックしても正しい）。
速度効果は実証できたので、bf16 併記版にしてから再導入する。いまは配線していない。

**作業事故の記録**: 取り消しパッチで `a=s.index(gated note)` と
`b=s.index("launch_swiglu_bf16(")`（最初の出現は MLP 側）を使い、
**a > b のままスライスした**ため step_executor.cpp の該当領域が重複破損した
（`duplicate case value 'elem_mul'`、`gt`/`up` 未宣言）。
`/tmp/se_split_wip.cpp`（第 18 ラウンド時点の备份）から復元し、
(1) split 配線の取り外し+NOTE、(2) keep-alive 撤去、(3) f32→bf16 norm 出力、
の 3 パッチを再適用して復旧した。教訓: **パッチ前に `assert a < b` を書く**、
同一ファイルの複数出現がある anchor は `index(..., start)` で範囲指定する。

復元後の検証（第 21 ラウンド終了時）:

- 27B fixture 96 step × 3 回 → すべて `d08df139…`、9B → `c097951350df` で生成列 IDENTICAL
- `ctest` hip 28/28・cpu 19/19
- prefill **1,012.0 ms（2,024 tok/s）** / decode **54.19 ms/tok**

### 段階 21（同日・第 22 ラウンド）: gate→量子化融合は**バイト一致では中立**と分かった（重要な契約上の発見）

`launch_swiglu_quantize_fp8` に bf16 出力（`out`、nullable）を追加し、
GDN 後段の gate 掛け算（`rmsnorm_gated`）に適用した（radiance の `gated_quant_fp8` 相当）。
消費者スキャンは `reshape` を透過させた（GDN 後段は reshape 経由で out_proj につながる）。

| 形 | prefill | decode | 27B logits |
| --- | --- | --- | --- |
| 融合（bf16 出力なし） | **989.4 ms（−21 ms）** | 54.33 | `b1b7085c…` ✗（未初期化 bf16 を読む経路が残る） |
| 融合 + bf16 も書く | 1,008.0 / 1,009.0 / 1,010.7 ms（−1 ms・ほぼ中立） | 54.18 / 54.20（中立） | **`d08df139…`** ✓ 3 回一致 |
| 融合なし（従来） | 1,010.9 ms | 54.19 | `d08df139…` |

**結論（速度差の構造的要因）**: radiance の prefill  Advantage の一部は
**活性化の bf16 中間材料を書かない**ことから来ている。shelf の「バイト一致」契約は
bf16 の値をそのまま量子化入力に使うことを要求するため、その節約は
**bf16 を書く限り中立**（−1 ms）にしかならない。−21 ms を取り込むには
活性化の丸め順を変える（＝ロス変更）必要がある。手順書の原則 4 に照らせば
**承認事項**で、判定は利用者の基準（生成文が意味を保つか）に従う。
なお rows==1（decode）では融合は 2 発射に負けた（54.19 → 54.36）ので
融合は prefill のときだけ有効化している。

現在の検証状態: 27B fixture 3 回連続 `d08df139…`、9B `c097951350df`（生成列一致）、
`ctest` hip 28/28・cpu 19/19、prefill **約 1,010 ms（2,030 tok/s）** /
decode **54.2 ms/tok**。radiance 比は prefill **0.727** / decode **0.862**。

### 段階 22（同日・第 24 ラウンド）: prefill fp8 GEMM の損失は **K の大きさ**に依存すると特定した

chunk 2048 の trace（/tmp/tr/nq）から wide kernel の launch 幾何（grid.x = 行タイル、
grid.y = 出力タイル、各 128）で形を復元し、GFLOP を割った:

| 形（rows × features） | 発射 | µs/発射 | K | 実効 TFLOP/s |
| --- | --- | --- | --- | --- |
| 2048 × 17408（up / gate） | 128 | 2,473 | **5120** | **147.7** |
| 2048 × 5120（down） | 128 | 1,691 | **17408** | **216.0** |
| 2048 × 10240（GDN in_proj qkv） | 48 | 1,458 | 5120 | 147.2 |
| 2048 × 6144（o_proj / GDN z） | 48 | 906 | 6144 | 170.6 |
| 2048 × 12288（qkv 系） | 16 | 1,732 | 5120 | 148.8 |
| 2048 × 1024（kv proj） | 32 | 166 | 5120 | 129.7 |

K が大きいほど速い（17408 → 216、6144 → 171、5120 → 147、1024 → 130）。
ブロック数で割ると（2048 × 17408 は 16 × 136 = 2,176 ブロック、
2048 × 5120 は 16 × 40 = 640 ブロック）occupancy の尾も効いているが、
**K 1 ブロック（128）あたりの仕事量で prologue / LDS ステージング / epilogue が
償却される**構造が主因に見える。

節約可能量の試算: K=5120 の 224 発射（合計 419.6 ms）が 216 TFLOP/s まで行けば
**−133 ms**。これは radiance との差（2048 tok で 275 ms）の半分弱、
1,024 tok あたりでは −67 ms で、前回組んだ「GEMM が主因」の裏付けになる。

打ち手（次ラウンド、構造変更が必要）:
1. **wide kernel の K ブロックを 256 に**（ステージ数を半減）。ただし活性化 LDS が
   16 → 32 KiB になり、gfx1201 の LDS 64 KiB/CU では 2 workgroup/CU ぎりぎり
   （32,200 B の崖に注意: 32,256 B で 1/CU に落ちて 2 倍遅くなった実測がある）
2. **LDS ステージングの二段化**（radiance の `gemm_fp8a8` と同じ。埋め込みループで
   `stage8_load` を先に issue する）。LDS は 2 × 16 KiB = 32 KiB で同样にぎりぎり
3. ~~狭い K の GEMM だけ 64 出力タイル kernel に替える~~ → **実行して否定した**
   （`SHELF_FP8_WIDE_MIN_K=8192`で 1,006 → **1,444 ms、+43%**）。K が小さくても wide が
   正解。改善は wide kernel の内側（ステージング二段化 / K ブロック 256）に限定される。
   環境変数は既定 0（常に wide）で再現用に只在る。

### 段階 23（同日・第 25 ラウンド）: wide GEMM の occupancy は**レジスタ側**で束縛されていた

資源の実測（rocprofv3 kernel trace）: VGPR **176**、LDS **16,896 B**、256 スレッド。
→ LDS からは 3 workgroup/CU 許されるが、レジスタからは
65,536 ÷ (176 × 32) ≒ **11.6 ウェーブ ≒ 1.4 workgroup/CU** が限界。
つまり occupancy は LDS でなく VGPR で決まっている。

`__launch_bounds__(kPrefill2dWideThreads, 2u)`（最低 2 workgroup）を試みたが、
**生成 VGPR は 176 のまま**（trace で再確認）で prefill も中立
（1,008.4 → 1,007.2 ms）。よって occupancy を上げるには本体側で
スレッドあたりの累加数を減らす必要がある（現状 128×128/256 = 64 acc →
例 128×64 なら 32 acc、ただし前実験で 64 出力タイル版へ丸ごと切り替えると
+43% 遅くなったので、**同じ 128 出力を維持したまま waves 増／acc 減**の再設計が要る:
例えば 8 波 → 4 波で 1 波あたり 8 行 × 2 出力、または K ブロック 64 の二段化で
ステージ毎の仕事量を半分に）。

### 段階 24（同日・第 26 ラウンド）: タイル面積そのものが occupancy を束縛している（計算で確認）

wide kernel の本体を讀むと、波の割り当ては 8 波 = `(wave_m: 64 行 × 2) × (wave_n: 32 出力 × 4)`
で、1 波が `float8_t acc[4][2]`（= 32 出力 × 16 行の断片）を持つ。つまり**ブロック全体の
累加数は 128 行 × 128 出力 = 16,384 個**。CU のベクタレジスタは 65,536 個なので、

- 累加だけで **4 workgroup/CU が上限**（16,384 × 4 = 65,536。オペランド 0 の理想値）
- 256 スレッド時は 1 群 64 acc + 実測 VGPR 176 → 65,536 ÷ (176×256) ≒ **1.4 workgroup/CU**
- 512 スレッドに増やすと acc は 32 に減るが、VGPR ≒ 102 で
  65,536 ÷ (102×512) ≒ **1.26 workgroup/CU** にしかならない（改善しない）
- 出力タイルを 64 に半減（128 × 64 = 8,192 acc）なら VGPR ≒ 104 で約 2 workgroup に
  届くが、**64 出力タイルへ切り替えた実験は +43% 遅い**（活性化 LDS 埋めが 2 回になるため）

→ `__launch_bounds__` でもスレッド数変更でも occupancy は上がらない（実際に 176 のままだった）。
**残る打ち手は K ループ内の待ちを消すこと**だけ:
活性化 LDS は 128 行 × 128 K で 16,384 B（+スケール 512 B）。**K ブロックを 64 に刻んで
2 バッファに交互へ**埋めれば LDS 总量は 16,896 B のまま（3 workgroup 許容を維持）、
ステージ間の `__syncthreads()` とグローバル読みの待ちを重叠できる。
acc は 128×128 のまま変わらないので数値も変わらない
（スケールは 128 K 境界で掛ける設計を保つ: 64 K で分割する場合は 128 K の
ほうでまとめて掛ける実装にする必要あり）。

### 段階 25（同日・第 27 ラウンド）: 活性化 LDS ステージングを 16 B 化（性能は中立、命令数は半減）

wide kernel の 128 K ブロック転送（32 B/スレッド）を `int32x2_t × 4` から
`uint4 × 2` に広げた（src/dst とも 32 B 境界なので 16 B 整列は満たす）。

- prefill 1,006.7 / 1,008.1 ms（基準 1,007.3 ms）→ **中立**（散布内）
- `ctest` hip 28/28、27B fixture logits `d08df139…` ✓（バイト一致）

待ちの本体（ステージングの重叠）に触れていない限り、転送幅だけでは動かないという
確認。次は段階 24 で決めた **64 K × 2 バッファの ping-pong**（LDS 总量 16,896 B 維持、
累加順は同一でバイト一致）に着手する。

### 引き継ぎ（第 28 ラウンド時点・実装仕様）

主因と設計は確定済み（段階 22〜25）。実装は次の 1 点に集約される:

- 対象: `src/backend/hip/include/shelf/backend/hip/kernels/gemm/detail/linear_fp8_device.h`
  の `fp8_prefill2d_wide_body`（K グループループは 459 行付近、
  ステージングは 463〜483 行、`__syncthreads()` は 484 行）
- 変更: 活性化 LDS（`s_a`）を **64 K × 2 バッファ**にして、
  グループ g の WMMA（32 K サブタイル 4 個）と g+1 の 64 K 転送を重叠させる
- 守る条件:
  1. LDS 总量は **16,896 B を超えない**（gfx1201 で 32,256 B に触れた瞬間に
     1 workgroup/CU へ落ちて 2 倍遅くなった実測がある: 段階 6 の記録）
  2. `raw` への累加は 32 K サブタイル 4 個を**同じ順**で保ち、スケールは
     128 K 境界で掛ける（現状と同じ）→ バイト一致
- 安全網: `tests/unit/test_hip_gemm_fp8.cpp:283` が prefill 2 次元タイルと
  RowBlock 経路の**ビット一致**を既に検査している（2048 行・256 行の受理も検査）。
  つまり数値が動けば ctest が落ちる。加えて実 fixture を 3 回走かせて
  `d08df139…`（27B）/ `c097951350df`（9B）に一致することを確認する。
- 見込み: K=5120 系の 147 → 216 TFLOP/s で **prefill −133 ms**
  （chunk 2048・27B。到達率 0.729 → 約 0.84）

このラウンドの検証状態: ctest hip 28/28・cpu 19/19、prefill 1,007 ms、
decode 54.19 ms/tok。

### 段階 26（同日・第 29 ラウンド）: レジスタへの先読みは**有害**と分かった（ reverted ）

wide kernel の活性化 LDS 転送を **1 グループ先読み**（グローバル読みを `uint4 × 4` の
レジスタに受けて、計算後に LDS へ書く）を実装した。

- 数値: `unit.hip_gemm_fp8`（prefill 2 次元タイル = RowBlock の**ビット一致**検査）**Pass** ✓
- 速度: prefill **1,061 / 1,064 / 1,066 ms**（基準 1,007 / 1,008 / 1,010 ms）→ **+5.4% 悪化**

原因は段階 24 のレジスタ帳簿どおり: 1 スレッド 64 B の先読みで VGPR が約 16 増え、
occupancy（65,536 ÷ (VGPR × 256)）が 1.4 → 約 1.3 workgroup/CU に下がった。
**待ちを隠すより載る本数が減る**ので純損。よって「先読みをレジスタで持つ」路線は
使えない。残る妥当な道は **LDS 总量を増やさない ping-pong**（64 K × 2 バッファ =
8,192 B × 2 + 512 B = 16,896 B のまま、追加レジスタなし）だけ。
実装は reverted（`/tmp/lf8d.bak` から復元、`issue_loads` は存在しないことを確認済み）。

### 段階 27（同日・第 30 ラウンド）: ping-pong LDS を実装したが**正しくなく、未解決で reverted**

64 K × 2 バッファ（总量 16,384 B 不変、追加レジスタなし）の ping-pong を
`fp8_prefill2d_wide_body` に入れる実装まで進めた。ビルドは通り、`__shared__` は
`s_a2[2][128*64]` + `s_as` のまま。しかし `unit.hip_gemm_fp8`
（prefill 2 次元タイル = RowBlock のビット一致検査）が **Failed**。

原因は切り分け済み: 既存の計算部をそのまま 64 K 単位に載せ替えたため、
  1. `float8_t raw[4][2]` のゼロ初期化が**半ごとに**走ってしまい、
     128 K の前半の寄与が捨てられている（fold は後半の後にしか执行されない）
  2. 計算部内部の `for (inner = 0; inner < 2; ++inner)` が残っていて、
     半のループと二重に 32 K サブタイルを回している
正しい形は「raw のゼロ初期化はグループ（偶数 h）頭、fold は奇数 h の末尾」に
明示的に移し、`inner` ループは削除して `ib = h*2 + il` に置き換えること。
次のセッションではこの 2 点を直せば通るはず（実装の骨格は生きている）。
`/tmp/lf8d.bak` から復元済みで、テストは 1/1 Pass に戻った。

### 段階 28（同日・第 31 ラウンド）: ping-pong LDS が入った（**prefill −0.9%、バイト一致**）

段階 27 で切り分けた 2 点（`raw` のゼロ初期化はグループ頭、折りはグループ末尾、
`inner` ループを消して `ib = h*2 + il`）を直して再実装した。

| 検査 | 結果 |
| --- | --- |
| `unit.hip_gemm_fp8`（prefill タイル = RowBlock ビット一致） | **Pass** ✓ |
| `ctest` hip / cpu | 28/28 ✓ / 19/19 ✓ |
| 27B fixture 3 回 | `d08df139…` 一致 ✓、9B `c097951350df`・生成列 IDENTICAL ✓ |

実測（27B・chunk 2048・交互 3 回）:

| | 基準（段階 25 まで） | ping-pong | 差 |
| --- | --- | --- | --- |
| prefill wall | 1,007.9 / 1,008.1 / 1,009.9 | **998.7 / 1,000.4 / 1,002.1** | **−8 ms（−0.85%）** |
| decode p50 | 54.19 / 54.22 | 54.16 / 54.17 / 54.18 | −0.03 ms（ほぼ中立） |

見込んだ −133 ms には届かなかった。転送と計算の重叠だけでは、
K=5120 系の 147 TFLOP/s は大きく動かない（occupancy は依然 VGPR 176 で
約 1.4 workgroup/CU）。つまり**待ち隠しより載る本数**が主制約であり、
それを決めるのは 128×128 タイルの累加数（16,384 個/ブロック）である
（段階 24 の帳簿）。次は「タイル面積を維持したままスレッド数を増やす」ではなく、
**K=5120 のような浅い K で epilogue/prologue を償却できる形**（例: 2 つの出力タイルを
1 ブロックで続けて回して LDS 埋めを 1 回に済ませる）を試す価値がある。

なお radiance 比は prefill 734.6 / 1,000.4 = **0.734**、decode 46.7 / 54.17 = **0.862**。

### 段階 29（同日・第 32 ラウンド）: radiance のロス設定は 2048 トークンでは得でないと実測

radiance を `--kv-cache-dtype fp8 --attn-prefill-fp8 both`（doc が速度向上として挙げる
ロス変更二つ、TP=1・chunk 1024・推測なし）で起動して同じ測り方で走かせた:

| 設定 | prefill 2048 tok | decode |
| --- | --- | --- |
| bf16 KV・attention 通常（既定・本報告の基準） | 733.0 / 736.2 / 738.9 ms | 46.7 ms/tok |
| fp8 KV + `attn-prefill-fp8 both` | 737.9 / 740.4 / 741.8 ms（p50 739.8） | **45.79 ms/tok** |

→ **2048 トークンの prefill ではロス設定の利得なし**（むしろ +5 ms、散布内）。
decode も 2% だけ。doc の数値は 221K/24K のような長大コンテキストでの話で、
このベンチ断面では効かないことが実測で分かった。
したがって**目標は bf16 KV・通常 attention の 734.6 ms / 46.7 ms のまま**で正しい。
shelf 側で bf16 中間材料を捨てるロス変更（推定 −29 ms）は、
「参照がその手を既に使っているから必要」ではなく「単に速くなるから入れる」候補になる。

（radiance プロセスは停止済み。pgrep -x radiance = 0）

### 引き継ぎ（第 33 ラウンド時点）: 優先度つきの未着手リスト

このセッションの作業容量が尽きたため、ここからは新しい文脈でやる。緑の状態:
`ctest` hip 28/28・cpu 19/19、27B fixture 3 回 `d08df139…`、9B 生成列一致、
prefill 1,000.4 ms（2,039 tok/s）/ decode 54.17 ms/tok（radiance 734.6 ms / 46.7 ms）。

1. **wide fp8 GEMM の occupancy**（主犯。推定 −100〜130 ms）
   - 実測の資源: VGPR 176、LDS 16,896 B、256 スレッド → 約 1.4 workgroup/CU
     （LDS からは 3 載るがレジスタが足りない: 128×128 = 16,384 累加/ブロック）
   - 試して否定: `__launch_bounds__(256,2)`（VGPR 176 のまま）、スレッド倍増
     （帳簿上 1.26 workgroup で得なし）、64 出力タイルへの差替え（+43%）、
     レジスタ先読み（+5.4%）、ping-pong LDS（**−8 ms で採用済み**）
   - 未試行の筋: 1 ブロックで**同じ活性化 LDS 埋めを 2 つの出力タイルで使い回す**
     （acc は 4×2 のまま、K ループを 2 出力タイルぶんまわし、LDS は 16,384 B を維持
     → workgroup 数は半分、acc/スレッドは不変で occupancy は同じ、ただし
     活性化読み・スケール読み・プロローグの償却が 2 倍）。
     数値は 1 出力タイルずつ同じ順で累加するのでバイト一致のまま検証できる
2. **KV ページサイズの未探索**（decode 有望、機構が触れていない）
   - decode attention は 8.4 MB/発射を 115 µs（73 GB/s）でしか運ばない。
     split-KV（workgroup を 24 → 64 に増やす）では速くならなかったので、
     律速は **16 トークンごとの block_table 間接参照**の可能性がある。
     shelf のページサイズは `src/model/qwen35` や `src/runtime/cli` に現れず、
     実行器側 (`at.page_tokens = u32(bs)`) に固定で入っている。
     64 ページにすると間接参照が 1/4 になる（KV の並ぶ順は内部表現なので
     生成結果は同じになるはず）。まず `bs` の決定箇所を特定すること
3. **承認待ちのロス変更**（推定 prefill −29 ms）: 活性化の bf16 中間材料を捨融合。
   参照（radiance）はこの断面では fp8 KV / attn-prefill-fp8 を使っても速くならない
   ことを実測済み（段階 29）なので、「参照がやっているから」ではなく純粋に速度の判断

### 段階 30（同日・第 34 ラウンド）: KV ページ寸法は decode を速めるが prefill kernel が 16 限定

`HipStepExecutorOptions::kv_block_size`（既定 16、`src/backend/hip/include/shelf/
backend/hip/step_executor.hpp:58`）に `SHELF_KV_PAGE` で上書きできる道をつけた
（既定 16 のまま＝挙動不変。27B fixture 2 回 `d08df139…`・ctest 28/28 で確認）。

掃引（27B・ctx 2048）:

| `SHELF_KV_PAGE` | decode p50 | prefill wall |
| --- | --- | --- |
| 16（既定） | 54.198 ms | **1,001.3 ms** |
| 32 | 54.035 ms | 1,253.9 ms ✗ |
| 64 | **53.980 ms** | 1,250.1 ms ✗ |
| 128 | **53.968 ms** | — |

- decode はページ拡大で **−0.23 ms/tok（−0.43%）**。block_table 間接参照の本数が
  1/8 になった効果で、単調。split-KV で速くならなかった理由（級 ③ ではない）と整合する、
  **間接参照側が効く**初めての証拠。
- prefill は 16 以外で **タイル化 prefill kernel が受理を拒否**して per-query に落ち、
  +250 ms（`attention_paged_prefill_bf16_supported` が page_tokens == 16 を要求）。
  よって既定は 16 のまま。**次はこの受理条件を外して（ページ寸法に依存しない
  ループに）すれば、decode −0.23 ms/tok がただで手に入る**。

### 段階 30 の続き（第 35 ラウンド）: prefill kernel のページ一般化は「16 のまま下位分解」でやる必要がある

`attention_paged_prefill.hip` はページ寸法を**構造に焼き込んでいる**:
`constexpr uint32_t kPageTokens = 16u;`（33 行）で LDS を `unsigned short v_page[2][kPageTokens][kPitch]`（107 行）と確保し、
ページループも `page_index * kPageTokens`（200 行）/ `tokens_here > kPageTokens`（202 行）で 16 固定。
受理条件（353〜361 行）が弾くのは表面で、**ページを 64 にすると LDS が 4 倍**になり
gfx1201 の occupancy 崖（32,200 B で 1 workgroup/CU）に直撃する。

実装時に触る必要がある箇所（ループを読んで確定）:
- `page_count = (vis_max + kPageTokens - 1u) / kPageTokens;`（150 行）は **16 基準**。
  外部ページを大きくするなら外側は `ceil(vis / a.page_tokens)`、内側に
  `a.page_tokens / 16` のサブページループが必要
- V の LDS 二段化は `vbuf = page_index & 1u`（208 行）で**ページ番号の奇偶**。
  サブページに分解するなら `vbuf = (page_index * subs + sp) & 1u` に直さないと
  先読みと消費が同じバッファを踏む
- `page_base = layer_base + page * a.elems_per_page`（203 行）に `+ sp * 16 * a.elems_per_token`
  を足し、`tokens_here` は `a.page_tokens - sp * 16` と可視範囲で切る
- K はプールから直接断片を作る（ページを LDS に張らない）ので、K 側のオフセットも
  サブページ分ずらす必要がある

正しい形は「外部ページは大きく、**LDS の中身は 16 トークンぶんずつ読む**」:
プールオフセットは ((page * kvh + h) * page_tokens + token_in_page) * head_dim なので、
page_tokens=64 でも sub = 0..3 ごとに token_in_page = sub*16 で 16 トークン断片を
順に処理すれば LDS は増えない（実装はページループを 16 刻みに分解するだけ）。
これで decode の −0.23 ms/tok（段階 30 実測）を prefill を犠牲にせず取れる。
検証は `unit.hip_attention` の prefill 検査と、27B fixture 3 回 `d08df139…`。

### 段階 31（同日・第 39 ラウンド）: hipfire 型の R1 先読み（重みフラグメント）で prefill −3 ms

`references/hipfire` の fp8 prefill GEMM（`gemm_mq6g256v2_wmma_fp8.gfx1201.hip`）が持つ
**raw-prefetch R1 pipeline** に相当する変更を shelf の wide kernel に転用した:
32 K サブタイルの重みフラグメントを**位相ごとに先に LDS/global から registers へ**引き
（`wsel[il][nt][lo|hi]`、2 サブタイル分を先に読む）、WMMA の実行中に global load を
走らせる。累加の順（`ib = h*2 + il` を同じ順で読む）と 128 K 境界のスケール畳み込みは
不変なので**ビット一致**。

| 検査 | 結果 |
| --- | --- |
| `unit.hip_gemm_fp8`（prefill タイル = RowBlock ビット一致） | Pass ✓ |
| 27B fixture 3 回 / 9B | `d08df139…` 一致 / `c097951350df` ✓ |
| `ctest` hip / cpu | 28/28 ✓ / 19/19 ✓ |

実測（27B・chunk 2048・交互 2 回）: prefill **996.5 / 998.0 ms**（ping-pong 直後 998.7 / 1,000.4 / 1,002.1）→ **約 −3 ms**。decode 54.13 / 54.17 ms/tok（0.03 ms 改善）。

hipfire から学んでまだ入れていない本命は**op 専用融合 GEMM**（gate と up を 1 kernel で
同時に計算し、16 行タイルが境界をまたぐ場合は行単位で `Y_gate`/`Y_up` に振り分ける。
`gemm_gate_up_mq2g256v2_wmma.gfx12.hip:36-40`）。radiance の `gemm_nt_q_gated` と同じ形であり、
活性化 LDS 埋めと発射を 1 回にまとめられる。shelf では gate/up の 2 primitive を
グラフ側でペア認識するところから必要（推定 prefill −20〜30 ms）。

### 段階 32（同日・第 40 ラウンド）: アキュムレータを増やす（出力タイル 256）は **大敗**

hipfire/radiance 型に寄せ accumulator を増やす実験を完遂した:
`kPrefill2dWideOut` を 128 → **256**（波は 4 出力タイル × 64 列を持つ、acc[4][4]、
wsel/wrow/ws_ch/ws_row/ws を 4 本に、store/折り込みも nt<4）。

- 実装中の罠: R1 先読みの `wsel[2][2][2]`（[位相][nt][lo/hi]）の nt 次元が 2 のままで
  **範囲外書き込み**→ `unit.hip_gemm_fp8` のビット一致検査が Failed。
  `wsel[2][4][2]` に直して Pass ✓（検査がちゃんと壊れ方を捕まえた実例）
- 性能: prefill **2,332 / 2,345 / 2,343 ms**（直前 996.5 / 998.0）→ **+135%**。
  acc 128 個/スレッドで VGPR が爆発し、大量スパイル（radiance の MW=4/NT=1 ですら
  95 スパイルしていたのを、我々は単体で背負う形になった）
- `/tmp/lf8d.bak3` から revert、ctest hip 28/28 ✓

結論: **accumulator を増やす方向は shelf の現行構造では成立しない**。
radiance が 228〜256 VGPR を許容しているのは、narrow tile（MW 波 × NT トークンタイル）に
`R4D_WAVES` で波数を固定する前提の上で、BK=256 の 1 本ライン読みと手動スケジューリングで
待ちを埋めているからだ（acc を増やしてoccupancy を下げるのではなく、
**タイルの作り全体を「少ない波 × 深い K」に設計し直している**）。

なお段階 31 の R1 先読み（−3 ms）は、revert 後の再測で 1,003 ms が出ており、
**実行間のばらつきと分離できていない**（採用はしているが効果は微少と見るべき）。
未実施: op 専用融合 GEMM（gate+up 一体、hipfire の本体スタイル。グラフ側の
ペア認識が必要で推定 −20〜30 ms）、行タイル 256×128（acc/スレッドを変えず
重み読みを半減、ただし LDS 32,768 B で 1 workgroup/CU の崖に触れる）。

### 段階 33（同日・第 42 ラウンド）: `GPU_MAX_HW_QUEUES=1` の実測と、数値方針の変更（利用者判断）

**(a) HW キュー数を 1 に絞った実測**（Qwen3.8-27B-FP8・GPU 0・Release）:

| 設定 | prefill wall | decode p50 |
| --- | --- | --- |
| 既定（複数 HW queue） | 996.8 / 999.0 ms | 54.151 / 54.181 ms |
| `GPU_MAX_HW_QUEUES=1` | 1,001.5 / 1,002.5 ms | 54.177 / 54.217 ms |

→ **1 にすると prefill が約 +3 ms（0.3%）遅い**。decode は差なし。
キュー並列は少し効いていて、絞っても得はない（値も変わらない）。
keep-alive のときに見えた「実行間の揺れ」は撤去済みで、現行コードでは
27B fixture 3 回一致（`d08df139…`）が復元されている。

**(b) 数値方針の変更（利用者の指示）**:`hi/lo 復号の bit-exactness は守るが、
f32 加算の結合順の再現は要求しない`。これにより radiance/hipfire 型の
split-K・二重アキュムレータ（T/P）・部分畳み込みが解禁になる。
副次的に、既存の `tests/unit/test_hip_gemm_fp8.cpp:283` 付近の
**「prefill タイル = RowBlock ビット一致」検査は方針と矛盾する**ので、
変更を入れる際には相対誤差つきの検査へ relaxing して、その理由と判断を
コメントに書く必要がある（まだ relaxing はしていない）。

**(c) hipfire 型の「半（64 K）ごとに raw を acc へ畳む」試行は、実装ミスで中断**
スクリプト編集で `ws` 宣言と折り込みループの位置を動かした結果、
ループのスコープが壊れてビルドエラー（`undeclared 'hh'/'g'`、
`expected unqualified-id`）。`/tmp/lf8d.bak4` から復元し、ビルド・ctest 28/28・
prefill 1,002 ms を確認済み。**このセッションの容量では安全に再挑戦できない。**

次ラウンドの実装仕様（方針 (b) の下で解禁されたもの）:
1. `fp8_prefill2d_wide_body`（`linear_fp8_device.h`）で
   `raw[4][2]` のゼロ初期化と畳み込みを**半（hh）ごと**に実行する
   （現状はグループ（128 K）ごとに 1 回）。`ws[2]` と `s_as` は 128 K 単位なので
   同じ値を両半で使ってよい。依存連鎖 `raw → acc` が短くなる
2. 解禁される本命: **split-K**（細い K の occupancy 不足を、波を増やさずに埋める）
   と **gate/up の 1 kernel 化**。split-K は partial バッファ + reduce kernel が必要
3. 検査: `unit.hip_gemm_fp8` を相対誤差基準へ変更（hi/lo 復号の厳密さは維持）、
   実 fixture は生成列が意味を保つかで判定（md5 一致は要求しない）

### 次のレバー（更新）

1. **prefill GDN のデータフロー移植**（`gdn_recurrence_wmma_body`、参照 1,592 µs/dis に対し
   2,561 µs/dis）。head_v を 64 列に割って workgroup を 2 倍にし、
   `union {中間行列; 状態}` で LDS 35,840 B に収める構成にする。
   段階 5b で「occupancy でも状態通信でもない」と確定したので残された筋はこれだけ。
   350 行規模。推定 −40 ms
2. **rmsnorm（+residual）→ fp8 量子化の融合**（参照 `fused_(resid_)rmsnorm_quantize_kernel`）。
   残る `activation_quantize` 192 発射を消す。推定 prefill −9 ms・decode −1 ms/token
3. **rope**（19.1 vs 2.7 ms: f32 中間をやめて bf16 入力・q/k 1 発射）と
   **bf16_row_slice**（15.1 vs 2.4 ms: GDN の qkv をカーネル内ストライド参照に）。推定 −25 ms
4. **runtime 内部の copy/fill 759 発射**（5.9 ms）削減（posid・scale テーブルの常駐化）
5. fp8 GEMV/GEMM の DRAM 効率（105.4 → 100.6 µs、1,676 → 1,631 µs）。
   ① なので bytes 減少（未量子化重みの fp8 化＝量子化段階の変更）が本筋
6. 96 個の非融合 swiglu の所在特定（bf16 GEMM へ feed している経路）

## 段階 34: `GPU_MAX_HW_QUEUES=2` と、畳み込み半分割（hipfire 型）の実測

### (a) HWQ=2

Qwen3.8-27B-FP8・GPU 0・Release・交互 2 回:

| 設定 | prefill wall（`wall_ms_median`） |
| --- | --- |
| 既定 | 997.835 / 1000.378 ms |
| `GPU_MAX_HW_QUEUES=2` | 1,000.031 / 1,000.819 ms |

段階 33 の `=1`（+3 ms）と合わせて、**HW キュー数を絞る方向に利点はない**。
既定が最良で、差はばらつき（±0.5%）と同オーダー。この項目は終了。

### (b) 畳み込みの粒度を 128 K → 64 K 半ごとに変更（hipfire の T/P 相当）

`fp8_prefill2d_wide_body` で `raw` のゼロ初期化と `acc` への折りを**半ごと**に実行
（`ws`/`s_as` は 128 K 粒度なので両半で同値）。hi/lo 復号（1 断片 2 WMMA）は不変。
`s_as[g+1]` の書き込みを barrier の**前**へ移す必要があった（半ごと折りにすると
次の繰り返し即座に読むため）。

交互 3 ペア（A = グループ折り込み版、B = 半ごと版。both Release・同一マシン）:

| ペア | pp A | pp B | delta | tg A | tg B | delta |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 998.091 | 1,125.235 | **+127.1** | 54.196 | 54.170 | −0.026 |
| 2 | 1,001.045 | 1,131.287 | **+130.2** | 54.228 | 54.222 | −0.006 |
| 3 | 1,002.860 | 1,134.335 | **+131.5** | 54.192 | 54.232 | +0.040 |

**決定的な悪化（+13%）**。折りの FMA（64 本/グループ）、`ws*s_as` のスカラー掛け、
LDS 読み、`raw` ゼロ初期化がすべて倍になり、WMMA 以外の命令比が増えて発行幅を奪う。
`raw → acc` の連鎖は律速ではなかった。→ **revert 済み**（hip 28/28・cpu 19/19 確認）。

裏付け: radiance のソースはこの形を明文で設計として書いている
（`libr4d/r4d_gemm_fp8a8.hip:108-110`「両 operand のスケールは 128 列で一定なので 1 ブロックに
1 回だけ適用。`raw` は 128 列を無スケールで集め、`out` へ折る。inner loop に要素ごとの演算を
入れない」）。つまり半ごと折りは radiance の設計に**反していた**。

### (c) 測定系の異常（記録として必須）

この段階の途中で `build/hip-rel/apps/shelf-bench` が 18:46 に再ビルドされ、`tg` の CLI が
変わった（`--runs` 不正、出力が `step_ms_p50=` の key=value 形式になり JSON でない）。
`git reflog` には HEAD 直下に `reset: moving to HEAD` が 2 件あり、私がこのセッションで作った
`isa/` コミット（hipfire/radiance の ISA 収集）は**このブランチの履歴に存在しない**。
`isa/` の作業ツコピーも消えている。収集済みの成果物は
`/opt/zen/work/references/isa`（コミット 371f7a8 ほか）にそのまま残っているため知識は失われて
いないが、**18:46 より前と後の tg 数値はバイナリが異なる**。
現在の HEAD（`082ff15`）は perf コミットで、再ビルド後の decode は 54.108 ms/tok。
今日のそれまでの 54.15〜54.24 ms/tok と同オーダーなので結論は変わらない。

教訓: ベンチは**必ず同一バイナリで交互に**測る（A/B はバイナリを `/tmp` にコピーして保持する）。
tg に `--runs` はなく `--tokens`/`--warmup` のみ。`pp` には `--runs` がある。

## 段階 35: 「部分持ち込みはもう無理では」に対する検証 — radiance のソースと 1 対 1 で突き合わせた

`references/radiance/libr4d/r4d_gemm_fp8a8.hip`（1,491 行、Apache-2.0、設計理由がインラインで
書かれている）と `r4d_fp8_wmma.h` を読み、我々の `fp8_prefill2d_wide_body` と構造を比較した。

### 同じ已经是もの（＝「形状の流用」は既に済んでいる）

| 項目 | radiance `gemm_8a8_prefill_kernel` | 我々 |
| --- | --- | --- |
| ブロック | 128 重み行 × 128 活性化行、**8 波 × 32 = 256 スレッド** | 128 × 128、8 波 × 32 = 256 |
| 波の分担 | 重み 4 波 × 活性化 2 波、1 波 = 重み 2 タイル × 活性化 4 タイル（各 16×16） | `wave_n` 2 × `wave_m` 4、1 波 = nt 2 × t 4 |
| アキュムレータ | **8 タイル × 2 セット（`raw`/`out`）で頭打ち**（増らすと spill） | 8 タイル × 2 セット（`raw`/`acc`）。256 出力试着で **+135%** ＝同じ壁 |
| 活性化 | LDS、**64 B 断片 × 2 重バッファ**、1 断片先読み | 64 B 断片 ping-pong（段階 30 で導入、+3〜5%） |
| 折り込み | **128 列ブロックに 1 回**（`:108-110`） | 同じ。64 K 半ごとに试着 → **+13%**（段階 34）で同じ壁を再確認 |
| hi/lo 復号 | 16 K あたり 2 WMMA（fp8 断片を 8 B × 2 で食わせる） | 同じ（`Fp8Frag16.lo/.hi`） |

### 違うもの＝未移植の「読み出し経路」（radiance が各自の実測値付きで load-bearing と書く）

| 項目 | radiance | 我々 | 彼らの実測 |
| --- | --- | --- | --- |
| 重みの置き場 | **LDS 経由せずレジスタ**。64 K タイルを wave 協調で `Stage8 = 32 B/lane`（1 KiB/wave）で読み、`w0`/`w1` を 1 タイル先読み ping-pong。`frag_of` は展開後定数なので**レジスタリネーム** | ループ内で `fp8_load16` = **16 B/lane**（512 B/wave）を断片単位 | 「8 B/lane はステージングに負ける。**32 B/lane が全域で勝ち**、gate_up の decode 形で **+13%**」（`:97-102`） |
| 先読み位置 | `stage8_load`/`stage8_store` を**分離**し、barrier の前に次断片の global load を発行。「1 回の呼び出しにすると 32 B リクエストが 1 本しか飛んでなく K ループが遅延の連鎖になる」（`r4d_fp8_wmma.h:180-183`） | `stage_half` は barrier の前だが、**重み**は断片の都度 | 「unconditional prefetch で +1%。条件付き load は 2 経路を結合して compiler がゼロ待ちになる」 |
| 波 index | `__builtin_amdgcn_readfirstlane(tid/32)` で SGPR 化。「一律なので scalar のまま保つ」 | **未使用**（全スレッドが VGPR で位位置を計算） | — |
| 活性化 LDS pitch | `lds_pitch<FK> = FK + 8` の**行パッド**（bank conflict 回避、load-bearing 指定） | 断片順で 1,024 B 連続（パッドなし） | — |
| スケール | `s_as[2][kPBM]` を二重化し `scn` を 2 ブロック先読み | 1 本。グループ境界で書き、折り込みは barrier 前（段階 34 で修正） | — |
| raster | `kGroupM = 8` の活性化バンド群で走査。plain な x-fastest は 64 MB MALL 超で**速度が半分** | plain（`blockIdx.x`=出力、`y`=行） | 「`kGroupM` 8 は 16 比 +1〜3%、4 は 8 と同等」（`:685`） |
| barrier | `lds_barrier()`（LDS のみ同期）。`__syncthreads` は outstanding load を待つ | `__syncthreads()` | +1%（`:682-683`） |
| 境界行 | 範囲外は **row M−1 を再読**。ループに分岐なし | early return / 境界分岐あり | — |

### radiance 側の「測定して棄却」リスト（我々の負結果と対応）

- 重みを 2 タイル先読み → **240 VGPR で spill、25〜38% 遅い**（→ R1 相当が中立だった説明がつく）
- **12 / 16 波（活性化 192 / 256 行）→ 同等〜+1% 遅い**（→ 段階 30 の「worker を増やすと強いか」= NO を独立に確認）
- 活性化 1 行を 4 スレッド × 16 B → **中立**（→ 16 B 幅のままの微調整が無効であることと整合）
- 32 bit オフセットの raw load → **20% 遅い**
- `s_setprio` → 中立。`-mcumode` → **約 10% 遅い**

### 結論（質問への答え）

**形状（ブロック/波/アキュムレータ/K 断片/折り込み粒度）は既に radiance と同一**で、そこを
「流用し直して」も得るものはない。ループ本体の局所改変（畳み込み粒度・断片先読み・タイル拡大）は
段階 30〜34 で全部が中立か悪化しており、**部分的な持ち込みは尽きた**という見立ては正しい。

差が残るのは **重みの読み出し経路**（32 B/lane の wave 協調ロードをレジスタへ、1 タイル先読みの
`w0`/`w1`、load/store 分離、barrier 前の発行）と **raster の group 走査**、そして付随して
`readfirstlane`・LDS pitch・スケール二重バッファ。これらは相互依存で、
「32 B/lane は断片順＋重みを LDS に載せないこと」「重みをレジスタに置くのは VGPR 予算を
8 タイルで固定していること（2 タイル先読みだと spill で −25〜38%）」の組み合わせで初めて効く。
よって **単位設計としての移植**が正しい。推定は prefill で +5〜15% のレンジ
（彼らの実測値の合計がそのオーダー）で、**30% のギャップを閉じるには prefill attention
（`r4d_attn_prefill_kernel`）とパス削減（slice 消去・GDN）が別途必要**。

なお GATED（gate/up + SwiGLU + 量子化の epilogue 融合）は **narrow tile（M ≤ 64）専用で、
prefill タイルには入らない**（`:122-135`、8 タイル上限のため）。段階 32 で計画した
「prefill の gate/up 1 kernel 化」は radiance の設計では**decode 側の杠杆**になる。

## 段階 36: 移植先を hipfire の形状に決定（spill を許さないため）

承認内容: 「読み出し経路一式を 1 つの原子変更で移植」。ただし **我々は spill を許さないので
radiance 型（VGPR 191〜256、最大 95 スロット spill）ではなく hipfire の形状を使う**。

### hipfire `kernels/src/gemm_mq6g256v2_wmma_fp8.gfx1201.hip` の実測リソース

この kernel は MQ6（6 bit 重み）用だが、**骨格（geometry）はデータ形式に依存しない**。
同一マシンで単独コンパイルして ISA を取った（`hipcc -O3 --offload-arch=gfx1201 -c` →
fatbin → gfx1201 code object → `llvm-readelf --notes`）:

| 腕 | VGPR | vgpr_spill | sgpr_spill | private_segment（LDS 以外） | LDS (`group_segment`) |
| --- | --- | --- | --- | --- | --- |
| `HIPFIRE_MQ6_FP8_DB=0`（半ごとの prologue、単一バッファ） | 214 | **0** | 0 | 0 | 29,184 B |
| `HIPFIRE_MQ6_FP8_DB=1`（連続パイプライン、二重バッファ） | 191 | **0** | 0 | 0 | 58,368 B |

**両腕とも spill ゼロ**。radiance の prefill kernel が 95 スロット spill していたのとは設計が
異なる。DB=1 は 8 波 × VGPR 191 = 16 波ぶん必要ではなく 8 波ぶん（191×32×8 = 48,896 < 65,536）
なので **workgroup/CU は 1**（LDS 58,368 B も 64 KiB 内に収まる）。occupancy は 8/32 波。

### 移植する骨格（hipfire の定数、`:81-116`）

```
MQ6_BM = 128 (token)   MQ6_BN = 128 (weight 行)   MQ6_NT = 256 (8 波)
MQ6_K16 = 4            1 つの K64 スラブに 16 K 断片 4 個（BK = 64）
MQ6_FRAG = 288         bank shift 付き断片ストライド（"bank-shifted fragment stride"）
MQ6_A_BYTES = (BM/16) * K16 * 288 = 9,216 B   活性化 1 スラブ
MQ6_W_BYTES = (BN/16) * K16 * 288 = 9,216 B   重み 1 スラブ
活性化: 2 重バッファ            → 18,432 B
重み   : SPLIT は hi/lo 2 プレーン × 2 重 = 36,864 B、**単一プレーン（ROUND）は 18,432 B**
スケール: 活性化 2×512、重み 2×(BN×8) → 3,072 B（SPLIT）/ 2,048 B
LDS 合計: SPLIT 58,368 B / **ROUND 39,936 B**
Grid [ceil(M/128), ceil(N/128)], Block [256], **静的 LDS のみ**
```

**我々の fp8 E4M3 一重データは hipfire の ROUND 腕に相当**（1 プレーン、(tile, K16) あたり WMMA 1 本）。
SPLIT 腕は 6 bit コードを `q' = 8*hi + lo` で 2 プレーンに分解するもので、私たちの重みには無い
（私たちの `Fp8Frag16.lo/.hi` は 16 B ロードの中の 2 つの 16 K 断片であり、精度分離ではない）。
従って移植後の LDS は **約 39,936 B**（SPLIT の 58,368 より小さい）になる見込み。

### 现状（`fp8_prefill2d_wide_body`）との差

| 項目 | hipfire（移植元） | 现状 |
| --- | --- | --- |
| 重みの LDS | **LDS に載せる**。断片順 288 B stride、K64 スラブ単位、**2 重バッファ** | LDS 経由せず global から 16 B/lane を断片ごとに |
| 活性化 LDS | 同じ形式で 2 重バッファ（9,216 B/スラブ） | 断片順 1,024 B × 16 行ブロックで 2 重（16,384 B）＝ pitch なし |
| スケール LDS | 活性化・重みとも **2 重バッファ**（次のスラブ分を先に置く） | 活性化は 1 本、重みは global 読み |
| 累算 | `T[4][2]`（通算）と `P[4][2]`（スラブ局部）の 2 セット、128 K ごとに P→T 折り込み | `acc[4][2]` と `raw[4][2]`（同じ形） |
| 折り込み粒度 | 128 K（= 2 スラブ）ごと | 128 K ごと（64 K 试着は +13% で棄却、段階 34） |
| 先読み | raw-prefetch **R1**（load と store を分離し、常にリクエストを飛ばす） | 活性化は `stage_half` で分離済み、重みは R1 试着で中立 |

### 実施計画（1 つの原子変更）

1. 新しい body `fp8_prefill2d_hipfire_body` を **同一ファイルに追加**し、`SHELF_FP8_PATH`
   （既定は現状維持の `wide`、`hf` で新経路）で dispatch する。既存 path は触らない。
2. LDS: `s_a[2][9216] + s_w[2][9216] + s_as[2][512] + s_ws[2][1024]` = **39,936 B**、静的宣言。
   断片ストライド 288 B の bank shift は hipfire の `MQ6_FRAG` を読む（16 K 断片 8 B/lane × 32 =
   256 B + 32 B のシフト）。
3. K ループはスラブ（64 K）単位で、`stage slab h+1`（load/store 分離・無条件）→ barrier →
   4 個の K16 断片を `P[t][nt]` に WMMA → スラブ 2 個で `P → T` 折り込み（スケールは 128 K 単位で
   同じ値を両スラブに使う）。hi/lo 復号は現状の 16 B ロード（=16 K 断片 2 個）を 8 B 断片 4 個に
   分解するだけで**ビットは変わらない**。
4. `__launch_bounds__(256)` のまま、ビルド後に ISA で **vgpr_spill_count == 0** を検査に含める
   （spill が出たら移植不成立として revert）。
5. ゲート: `ctest -R gemm_fp8` → 実 fixture 3 回一致（**加算順は変わらない想定なので md5 は
   段階 28 の値と一致するはず。一致しなければ順序が変わった証拠なので生成列の coherence で
   判定する**）→ 交互 A/B 3 ペア以上（`wall_ms_median`）で keep/revert。

## 段階 37: hipfire の連続パイプラインを preshuffle 版で実装（spill ゼロ確認済み）

`fp8_prefill2d_wide_body` を段階 36 の仕様に沿って書き換えた。追加した骨格は 4 点:

1. **K を 64 K スラブに平坦化**（`s = 2g + SL`、g = 128 K のスケールグループ）。
   活性化 LDS はスラブ偶奇で 2 重バッファ。
2. **load と store の分離 + R1 ピン**。スラブ s+1 のグローバル読みを (a) でレジスタへ
   発行し、(b) 計算 → (c) 折り込み → (d) `asm volatile("" : "+v"(…))` でピン留めしてから
   別のバッファへ公開 → (e) 同期。hipfire はこれを `:471-485` でやっている。
   **段階 32 の R1 が中立だった原因はこれ**（ピンが無いと読みが WMMA の直前へ沈む）。
3. **同期はスラブに 1 回**（グループに 2 回ではない）。hipfire の正当性论证
   （`:336-343`）をコメントで引用: 公開先バッファ (s+1)&1 を最後に読むのはスラブ s-1 の
   計算で、全 wave はその末尾同期を通っている。
4. **活性化スケールを LDS の 2 重バッファ**化し、次のグループ分を計算前に読み計算後に公開。
   重みは 1 スラブぶん先読みで **32 B/lane**（presuffle では断片 4 個が連続なので
   LDS 経由が不要。hipfire が LDS に載せるのは 6 bit → E4M3 復号の変換段があるから）。

`raw` のゼロ初期化は「グループ最初の WMMA の入力をゼロから」に置換した（hipfire の
`FIRST` と同じ手で、hipfire の命令は減る）。**したがって加算順は不変**
（折り込みは 128 K に 1 回、32 K ステップ順序 0,1,2,3、スケール位置も同じ）。
数値方針の変更を待たずに **ビット一致のまま**パイプラインだけ変えられる。

### ISA 実測（`llvm-readelf --notes`）

| kernel | VGPR | vgpr_spill | sgpr_spill | private_segment | LDS |
| --- | --- | --- | --- | --- | --- |
| 移植後 `gemm_fp8_w8a8_prefill_wide_kernel` | **208** | **0** | 0 | 0 | 17,408 B |
| hipfire `…_mq6g256v2_wmma_fp8_split` DB=1（比較） | 191 | 0 | 0 | 0 | 58,368 B |
| hipfire DB=0（比較） | 214 | 0 | 0 | 0 | 29,184 B |

**spill ゼロ**（我々の要件）。208 × 32 × 8 波 = 53,248 VGPR < 65,536 なので workgroup/CU は 1。
hipfire の 58,368 B LDS を使わない分、LDS は 17,408 B と小さい。

`unit.hip_gemm_fp8`（prefill タイル = RowBlock ビット一致の検査）は **そのまま Passed**。
段階 33 で懸念した「検査の relax」は不要だった（順序を変えなかったため）。

## 段階 38: hipfire ポートの実測結果と、raster グループ走査（不採用）

### hipfire 型（段階 37 の実装）の A/B — 一様に 8% 悪化

`pp` 交互 3 ペア（A = baseline、B = hipfire ポート）: **+53.4 / +53.1 / +52.2 ms**。
decode 中立。rocprofv3 で kernel × 形状別に切り分けた（2048 tok、gridX=4096 が広い
prefill GEMM、gridY = 出力タイル数）:

| 形状 2048 × N | A µs | B µs | 差 |
| --- | --- | --- | --- |
| 17408 (K=5120) | 304,002 | 330,096 | +8.6% |
| 5120 (K=17408) | 205,874 | 221,114 | +7.4% |
| 10240 | 66,962 | 73,837 | +10.3% |
| 6144 | 41,495 | 45,063 | +8.6% |
| 12288 | 26,614 | 29,307 | +10.1% |
| 1024 (q/k) | 5,044 | 6,780 | +34.4% |
| GDN / attention / 小演算 | — | — | ±1〜2% |

GEMM 合計差 56.2 ms が end-to-end の +53 ms と一致。正しさは問題なし
（`unit.hip_gemm_fp8` Passed、27B fixture の logits md5 = `d08df139…` で baseline と一致）。

### 原因（hipfire の形をそのまま写さなかった点）

hipfire 自体を腕ごとにコンパイルしてリソースを取った:

| 腕 | VGPR | spill | LDS |
| --- | --- | --- | --- |
| SPLIT DB=1 | 191 | 0 | 58,368 B |
| SPLIT DB=0 | 214 | 0 | 29,184 B |
| **ROUND DB=1**（1 プレーン = 私たちと同じ工作量） | **191** | 0 | **39,936 B** |
| ROUND DB=0 | 207 | 0 | 19,968 B |

hipfire は **重みを LDS に二重バッファ**（39,936 B）してレジスタを 191 に落としている。
私のポートは preshuffle を理由に重みをレジスタに置いた（radiance 寄せ）ため VGPR が
**182 → 208** となり、加えて (a) を `if (has_next)` の**条件付き load**にし、
`asm volatile` のピンで命令を固定した。radiance はこれを明示的に棄却している
（`r4d_gemm_fp8a8.hip:680-681`「条件付き load は命令数の違う 2 経路を結合して compiler が
ゼロ待ちになる。**無条件 prefetch**（最後を再読して破棄）で +1%」）。
また radiance の読み出しは barrier の**後**（`:765-771` publish → barrier → load → compute）。
私は barrier の前に読んでいたので、`__syncthreads` が直後の読みを待った
（radiance が LDS 専用 barrier を使う理由も同じ記述、`:682-683`）。

### raster のグループ走査（radiance `kGroupM`）— 我々では -1.2%

広い body に `kGroupM = 8` の行バンドグループ走査を単独で入れて A/B 3 ペア:
**+12.2 / +11.4 / +11.7 ms（+1.2%）**。fixture はビット一致。
radiance では「plain 走査は 64 MB MALL 超で速度が半分」と書かれているが、我々は既に
`blockIdx.x` = 行タイル / `blockIdx.y` = 出力タイル（段階 22 で 1.4 倍改善を決めた向き）で、
64 workgroup が同時に 4 出力タイル × 16 行タイルを覆うため活性化 10.5 MB と重みが
L3 に載っている。グループ化はそれを壊した。**不採用**。

## 段階 39: radiance 型パイプラインの実装と実測 — 私たちの layout では baseline が最良

`fp8_prefill2d_wide_body` を radiance `gemm_8a8_prefill_kernel` の順序そのものに書き換えた
（`r4d_gemm_fp8a8.hip:758-801`）:

1. 前にロードした活性化スラブを LDS へ公開 → 2. 次グループのスケールを別バッファへ公開
   → 3. barrier → 4. **スラブ s+1 の読みを発行**（活性化・重み・スケール、すべて
   **無条件**で範囲外は最後のスラブを再読して捨てる）→ 5. 計算 → 6. 128 K ごとに折り込み
2. 重みは 1 スラブぶんだけレジスタに保つ（深さ 1）。radiance の
   `tile(kb, 0u, w0, w1); tile(kb, 1u, w1, w0);` に従い、パリティ配列の動的添字ではなく
   **展開された交互呼び出し**（`slab_step(s, cur, nxt)`）で対応を定数にした
3. スケール書き込みは `tid & 127` で行を 2 スレッドに持たせて分岐を消した
   （radiance の `tid % kPBM` と同じ手）
4. hipfire 版で使った `asm volatile` のピンと `if (has_next)` の条件付き load は撤去

### 実装時のバグ（記録として残す）

最初の試みは prologue で slab 0 と 1 を両方ロードしたうえでループ内で slab s+2 を `nxt` に
入れたため、**slab s+1 の重みを上書き**していた。`unit.hip_gemm_fp8` が Failed、
fixture md5 が `03ef4a0a…` に Change。深さを 1 に統一して修正した。
修正後は `unit.hip_gemm_fp8` **Passed**、27B fixture の logits md5 =
**`d08df139125f98e6fab1dada626fd87f`**（baseline と完全一致 = 加算順は変わっていない）。

### ISA と A/B

| kernel | VGPR | spill | LDS |
| --- | --- | --- | --- |
| baseline（HEAD、`stage_half` 一体型） | 182 | 0 | 16,896 B |
| hipfire 型（段階 37） | 208 | 0 | 17,408 B |
| **radiance 型（段階 39）** | **194** | **0** | 17,408 B |

`pp` 交互 3 ペア（A = baseline、B = radiance 型）:

| ペア | A | B | delta |
| --- | --- | --- | --- |
| 1 | 1,001.847 | 1,049.637 | +47.8（+4.77%） |
| 2 | 1,002.615 | 1,051.446 | +48.8（+4.87%） |
| 3 | 1,002.300 | 1,051.838 | +49.5（+4.94%） |

decode は中立（+0.038 ms 以内）。

### 結論

**3 種類の読み出し経路（baseline / hipfire 型 / radiance 型）を、いずれも spill ゼロで
実装して測った結果、committed な baseline が最良だった**（radiance 型 +4.8%、
hipfire 型 +5.3%、畳み込み半分割 +13%、raster グループ +1.2%、256 出力タイル +135%、
活性化のレジスタ先読み +5.4%）。

理由は layout の差だと考えている。radiance も hipfire も、重みを**行優先グローバルから
LDS/レジスタへ運んで断片を作る**変換段を持っている（hipfire は 6 bit → E4M3 復号、
radiance は row-major → fragment order）。我々はすでに preshuffle で断片順に保存しており、
その変換段が存在しない。だから先読み・二段バファ・要求幅の技巧は「存在しないコストを
隠す」ものになり、増えたレジスタ（182 → 194/208）と命令数だけが残る。
1 workgroup/CU で 8 波も走っているので、グローバル遅延はもともと他の波が隠していた。

**prefill GEMM のマイクロ構造での追いつきは打ち止め**。radiance との 30% のギャップは
GEMM の 1 kernel 効率ではなく、(a) kernel 数と往復（slice 消去・norm+quant 融合・
GATED は narrow 側）、(b) prefill attention kernel、(c) GDN の block 三角形求解、
(d) launch 経路（HIP graph）にある。段階 33/35 の未着手項目のほうが残存価値は高い。

手元に残した安い検証筋（各 1 行・±1% レベル）: `lds_barrier()`（radiance +1%）、
`__builtin_amdgcn_readfirstlane(tid/32)`（radiance が波 index を SGPR 化している）。

## 段階 40: 発射内訳の取り直し（`SHELF_HIP_PROFILE` の誤導を訂正し、rocprofv3 で確定）

radiance 版は実測 +4.8%（段階 39）なので **kernel は git HEAD の baseline に復元済み**
（VGPR 182・spill 0）。保存先: radiance 型 `/tmp/lf8d.radver`、hipfire 型 `/tmp/lf8d.ported`、
baseline `/tmp/lf8d.bak5`（= HEAD と同一）。

### 訂正: `SHELF_HIP_PROFILE=1` の per-primitive 値は当てにならない

`SHELF_HIP_PROFILE=1` は primitive ごとに 2 つの `hipEventRecord` で挟む
（`step_executor.cpp:590-615`）ため、**仕事をしない view でも 3〜5 µs** が計上される。
`case K::reshape` は `set_out(x->f32, x->dev)` の alias でカーネルを発射しない
（`step_executor.cpp:1030-1034`）。よって前回 doc に書いた
「reshape 496 発射 2.48 ms」「slice 176 発射 2.21 ms（profile 値）」は**測定器のコスト**で、
view ノードの削減に価値はない。

### 確定値（`rocprofv3 --kernel-trace`、tg: context 2048・tokens 16・warmup 4、27B-FP8）

総ディスパッチ 31,364 / 21 ステップ ≒ **1,493 発射/ステップ**、総 GPU 1,931.8 ms。
デコード 1 ステップあたりの実 GPU 時間（`合計µs ÷ 21`）:

| kernel | 発射/step | 平均 µs | ms/step |
| --- | --- | --- | --- |
| `bf16_row_slice` | **176** | 5.85 | **1.03** |
| `phaseshift_residual_add_bf16` | 128 | 5.95 | 0.76 |
| `phaseshift_fused_gdn_conv_silu` | 48 | 13.59 | 0.65 |
| `phaseshift_rmsnorm_bf16_pf_oneplus` | 81 | 6.06 | 0.49 |
| `phaseshift_fused_gdn_qk_prepare` | 48 | 5.81 | 0.28 |
| `__amd_rocclr_copyBuffer`（runtime 内部） | 28 | 8.86 | 0.25 |
| `phaseshift_rope_bf16_pair` | 32 | 5.17 | 0.17 |
| `bf16_sigmoid_gate` | 16 | 7.70 | 0.12 |
| `phaseshift_kv_append_bf16` | 16 | 4.25 | 0.07 |
| `phaseshift_swiglu_bf16` | 46 | 1.88 | 0.09 |
| `__amd_rocclr_fillBufferUnAligned` | 6 | 4.43 | 0.03 |
| **小計（GEMV/attention/lm_head を除く）** | **≈ 625** | — | **≈ 3.9** |

fp8 GEMV 群は 45.6 ms/step（598 GB/s で roofline）、`phaseshift_gemm_bf16_exact_rows_1`
（lm_head）4.7 ms/step、`phaseshift_attention_paged_bf16` 1.75 ms/step。
**decode 54.2 ms のうち約 4 ms がこの小計**で、launch 費用（≈2 µs × 625 ≒ 1.2 ms）も
そこに重なる。radiance との差 7.5 ms/token に対して、ここを消す価値は実測で裏が取れた。

### 目標 1 の実装地点（次回の引き継ぎ）

`bf16_row_slice` の 176 発射はグラフ側の `Kind::slice` ノード:

- `src/model/qwen35/step_graph.cpp:223-228` — GDN の q/g 分割。`qg3`（[rows, 2D]）から
  `q3`(0..D) と `gate`(D..2D) を**実コピー**で作っている。GDN 48 段 × 2 = 96 発射
- `src/model/qwen35/step_graph.cpp:315-317` — attention の q/k/v 分割ヘルパー
  （`mixed` から 3 枚）。16 段 × 3 = 48 発射
- 残りは conv 前の分割など

やるべきことは「スライスを作らない」こと: `GdnStepArgs` / attention args に
**入力側の `elem_offset` と `row_stride`** を持たせ、`q3`/`gate` を `qg3` の
オフセット参照にする。`K::slice` の executor 実装は
`step_executor.cpp:980-996`（`launch_bf16_row_slice`）で、`Kind::row_slice` は
行が連続なので 1 行抽出（`step_executor.cpp:996-1028`、最終行のみ・1 発射/step）なので触らない。

期待値: decode **−2.0〜2.5 ms/token**（176 発射と 1.03 ms の GPU + bf16 コピー往復）、
prefill −4 ms/chunk 級。ゲートは従来どおり 27B fixture 3 回一致
（`d08df139125f98e6fab1dada626fd87f`）＋交互 A/B 3 ペア。

## 段階 41: GDN の q/g スライス消去 — 実装前に分かった制約（未実装のまま引き継ぎ）

`Kind::slice` を view（alias）に置き換える方針を検討した結果、**そのままだと不可**。

- `qg3` は [rows, nq, 2D] で、`q3`(0..D) と `gate`(D..2D) は**行ピッチ 2D の部分列**。
  alias にすると出力ポインタは `base + slice_begin * 2` で済むが、行間は飛ぶ。
- `ValueLoc` は `{dev, f32, elems}` で**行ピッチを持たない**（`step_executor.hpp:126`）。
- 消費側の内訳:
  - `q3` → `rmsnorm`（[step_graph.cpp:230](src/model/qwen35/step_graph.cpp#L230)）→ rope。
    rmsnorm は行方向の読み込み幅が要るので `in_stride` の追加で対応可能
    （`launch_rmsnorm_bf16_quantize` はすでに `in_stride` を取る）
  - `gate` → **`reshape_to(gate, {rows_, nq*D})`**（[step_graph.cpp:267](src/model/qwen35/step_graph.cpp#L267)）
    → `sigmoid_gate`（[:269](src/model/qwen35/step_graph.cpp#L269)）。
    **ピッチ 2D の [rows, nq, D] を [rows, nq*D] に平坦化する reshape は alias できない**
    （行内に 2D−D の穴が残るため）。ここは view 化の壁になる。

### 取りうる形と現実的な期待値

| 案 | 変更範囲 | 消える発射 | 期待（decode） |
| --- | --- | --- | --- |
| A. 2 枚のスライスを 1 発射に統合（`qg3` → 1 kernel で `q3`/`gate` を両方書く） | kernel 1 本 + graph 1 箇所 | 96 → 48 | **−0.1 ms**。コピー byte は同じ |
| B. `ValueLoc` に `row_pitch` を追加し、rmsnorm と `sigmoid_gate` にストライド読みを入れ、平坦化 reshape を消す | executor + kernel 2 本 + graph 3 箇所 | 96 → 0 | **−0.6〜0.8 ms** |
| C. attention 側（[:315-317](src/model/qwen35/step_graph.cpp#L315-L317)）も同様にすると +48 発射 | 同上の拡張 | 176 → 0 | 合計 −1.2〜1.5 ms |

当初見込み（「slice 消去で decode −2.0〜2.5 ms」）は**過大**。実測 basis:
`bf16_row_slice` 176 発射 × 5.85 µs = 1.03 ms/step が GPU 時間の上限で、launch 費用を足しても
1.5 ms 前後。prefill 側はスライスサイズが大きいので −1〜2 ms/chunk が見える。

したがって **decode の 7.5 ms ギャップを閉じる主筋は hipGraph（625 発射/step の launch と
gap をまとめて消す）**であり、slice 消去は補助的な位置づけになる。

## 段階 42: prefill の内訳を確定（2048 tok・1 chunk、baseline build、rocprofv3）

総 GPU 902.9 ms（wall ≒ 1,000 ms と整合）。発射数と GPU 時間:

| 分類 | 発射 | ms | 割合 |
| --- | --- | --- | --- |
| **fp8 prefill wide GEMM** | 400 | **650.0** | **72.0%** |
| **GDN（`gdn_chunk_impl` 111.3 を含む）** | 144 | **125.8** | **13.9%** |
| elementwise（swiglu_quantize 26.2、residual_add 11.4、rope ほか） | 289 | 41.3 | 4.6% |
| attention（prefill、16 発射 × 1,874 µs） | 16 | 30.0 | 3.3% |
| rmsnorm | 209 | 15.1 | 1.7% |
| **slice（`bf16_row_slice` 177 × 84.5 µs）** | 177 | 15.0 | 1.7% |
| fp8 wmma（小さい形状） | 96 | 10.0 | 1.1% |
| runtime copy/fill | 759 | 5.7 | 0.6% |
| lm_head bf16 | 1 | 4.0 | 0.4% |

`gdn_chunk_impl` は 48 発射 × **2,318 µs**（27B の GDN 段数ぶん）で、GEMM に次ぐ第二大 block。

### GEMM 側の新しい事実（`shelf-bench kernel gemm --config 2d` の合成スイープ）

- **K 長は律速ではなかった**: M=512・N=17408 で K=1024→17408 まで 167.1 / 182.0 / 166.6 /
  166.9 / 168.2 / 156.3 TFLOP/s と**ほぼ平坦**。段階 33 の「狭い K の prologue 償却が target」
  は**棄却**。
- N が小さいと平行性が足りない: M=512・K=5120 で N=1024 → **85.9**（4×8 = 32 workgroup で
  64 CU の半分が遊ぶ）、N=3072 → 164.6、N=6144 → 179.0。
- **M が効く**: N=17408・K=5120 で M=128 → 107.8、M=256 → 153.0、M=512 → 166.0。
  そして実測の最高値 216.0 TFLOP/s（2048×5120@K17408）は M=512 では再現しない
  （同形状で 148.4）。**M 方向の再利用（行タイル数）が効率を決めている**。
- したがって wide kernel のこの GPU での現実的な上限は **180〜216 TFLOP/s**。
  実走の 650 ms は 104.4 TFLOP を 160.6 TFLOP/s で処理した値で、**全形状を 216 に上げても
  −167 ms**（650 → 483 ms）が GEMM 側の物理的上限。

### prefill を直す順序（この割合に対する大きさで確定）

1. **GEMM の raster を「平方 Supernode」にする**（仮説・未実施）。radiance の `kGroupM`
   （M 方向だけグループ化）は实测 -1.2% で不採用。 대신 **行タイル 4 × 出力タイル 4**
   の塊で走査すると、_working set_ は活性化 512 行 × K + 重み 512 行 × K（K=5120 で
   各 2.6 MB）に収まり、重みも活性化も L2/L3 に残ったまま両方を 1 回ずつ読むことになる。
   実装は既存の `blockIdx` 変換 10 行（段階 38 のコードを差し替え）。
2. **GDN chunk の block 三角形求解分離**（`kkt_solve` 独立、推定 −37 ms）。
   `gdn_chunk_impl` 2,318 µs × 48 が対象。参照: `references/phaseshift-fp8/src/phaseshift/
   inference/execution/` の `gdn_qk_prepare_fusion.hip` / `gdn_conv_silu_fusion.hip` /
   `gdn_post_fusion.hip` / `gdn_recurrence_dispatch.hip` と selectors。
3. **非融合の小演算を消す**: slice 15.0 + rmsnorm 15.1 + residual_add 11.4 +
   copy/fill 5.7 = 47.2 ms（+ 発射 1,322 個の launch）。段階 41 の制約（平坦化 reshape が
   alias できない）を満たす形で、`sigmoid_gate` / rmsnorm にストライド読みを入れる。
4. **attention prefill**（30.0 ms、1 発射 1,874 µs）。page 16 前提は radiance と同じなので
   `r4d_attn_prefill_kernel<24,48,256,6,16,…>`（VGPR 240・LDS 54,016 B）の移植が効く候補。

1〜4 すべてで −120〜−170 ms、GEMM 上限まで届かせれば合計 −290 ms 前後で radiance の
734 ms（chunk 1024 基準で 367 ms/1024 tok）と同一オーダーに届く見込み。

## 段階 43: 平方 supernode raster（段階 42 の項目 1）— +1.1〜1.2% で不採用

`blockIdx` を **行 4 × 出力 4 の supernode** 順に並べ替えた（27B の全形状で nM・nN が 4 の
倍数なので過剰発射なしの全単射。実装は `/tmp/lf8d.sup` に保存）。

- hip 28/28 Passed、27B fixture md5 = `d08df139…`（baseline と一致 = 純粋な投入順の変更）
- `pp` 交互 A/B: **+12.3 ms（+1.23%）/ +11.0 ms（+1.10%）**、decode 中立（3 ペア目は途中で停止）

段階 38 の `kGroupM`（M 方向のみ、+1.2%）と合わせて、**raster の並べ替えはこの GPU で
いずれの形でも効かない**と結論する。plain な x 優先（x = 行タイル）が既に最適で、
64 CU × 1 workgroup が 4 出力タイル × 16 行タイルを同時に覆うため活性化 10.5 MB と
重みが L3 に載っている（段階 38 の記述どおり）。→ **baseline に復元済み**。

prefill で残る実測対象は段階 42 の 2〜4 のみ:
**GDN `gdn_chunk_impl` 111.3 ms（48 発射 × 2,318 µs）**、小演算 47.2 ms
（slice 15.0 + rmsnorm 15.1 + residual_add 11.4 + copy/fill 5.7）、
attention prefill 30.0 ms（1,874 µs × 16）。参照実装は
`~/work/references/phaseshift-fp8/src/phaseshift/inference/execution/`
（`gdn_qk_prepare_fusion.hip` / `gdn_conv_silu_fusion.hip` / `gdn_post_fusion.hip` /
`gdn_recurrence_dispatch.hip` と `selectors/`）。

## 段階 44: 参照の所在を確定 — phaseshift-fp8 の GDN kernel

`execution/gdn_*.hip` は**すべて dispatch wrapper**で、`__global__` を含まない。
実体は 1 ファイルに揃っている:

- `~/work/references/phaseshift-fp8/src/phaseshift/kernels/optimized/gdn/recurrence.hip`
  - `:34` `__global__ void gdn_recurrence_f32_impl(const GdnRecurrenceArgs a)` —
    token 毎の逐次 recurrence（`:71` `for t < row_count`、workgroup 32、
    `:131` `<<<dim3(grid_x), 32>>>`）
  - `:168` 以降に**chunk 版**がある。状態アキュムレータ `Sacc[ki*2+vj][e]`（`:284` で
    `ld_f16(sp + e*hv)` から fp16 状態を読み、`:289-292` で 2×2×8 の累算）をレジスタに置き、
    `:298` `const uint32_t nchunks = kSerialRows …` でシリアル行数と chunk 数を分割している。

つまり参照側も「chunk 化 + 状態をレジスタ累算」の構造で、我々の `gdn_chunk_impl`
（実測 2,318 µs × 48 発射 = 111.3 ms、prefill の 12%）と同種の kernel。
次ラウンドの作業: `recurrence.hip:160-360` を読んで、(a) 状態の保持形式（fp16 状態 +
`Sacc[4][8]`）、(b) `kSerialRows`/`nchunks` の決め方（= workgroup 数と直列長のトレードオフ）、
(c) 求解（kkt 相当）が分離されているか — を確認し、我々の `gdn_chunk_impl` の
LDS/VGPR（段階 5b で 32,256 B で occupancy に崖があることは確定）と突き合わせて
仕様を書く。カーネル編集は仕様が固まってから。

## 段階 45: **GDN 前進代入のブロック化で prefill −11.6 ms（採用）**

### 先に寄与の実測（一時バイセクト）

`gdn_chunk.hip` の前進代入（step 5）を一時的に無効化して `pp` を測ると
**1,000.3 → 964.3 ms**。つまり求解だけで **36.0 ms**、GDN 111.3 ms の **32%**。
段階 5b で立てた「kkt 求解分離で −37 ms」の推定とほぼ一致した。

実装を読むと原因がはっきりしている（`src/backend/hip/kernels/gdn/gdn_chunk.hip:241-250`）：

- `if (tid < DV2)` で **64 スレッドだけ**が参加（全 256 本の 3/4 が待機）
- `rows`（=32）段の**逐鎖**で、内側の内積長が runtime 値（最大 31 項）なので展開もできない
- 各段が LDS の `s.d` を読み書きするので依存が素直に延びる

### 実装（16 行 2 ブロック）

`m` は対角 1 の下三角 [CH][CH]、`d` は右辺 [CH][DV2]。行を 16 行 2 ブロックに割って

```
(a) d0 = solve(m00, d0)     … 直列長 16、内積は最大 15 項
(b) d1 -= m10 · d0          … 16×16×64 の密な結合。256 スレッド全部で並列
(c) d1 = solve(m11, d1)     … 直列長 16
```

の順に変更。バリア 2 本が増えるが、直列長が半分・総 FMA 数が半分・(b) が全スレッド並列。

### ゲート結果（採用の根拠）

- hip **28/28 Passed**（`unit.hip_gdn_chunk` 含む）
- 単体テストが print する誤差は**ブロック化版と逐鎖版で最下位桁まで一致**
  （rows=32: out worst 1.842e-05 / rel 5.508e-04、state rel 4.626e-04。rows=64/128/256 も同一）。
  求解の再結合による差は、kernel 内の f16 変換の丸めノードより小さい
- 27B fixture 96 step を 3 回: md5 **`1ad8b8da49ac8faf676f8169c48f4e0f` で 3 回一致**（決定的）。
  baseline `d08df139…` とは異なる（順序変更は許可済み方針どおり）。
- サンプリ系列の差分: 352 個のうち**先頭 281 個（80%）が一致**、282 番目以降は分岐して
  両者とも通常のトークン列のまま（退化なし）。logits の最終行だけを見ると max abs 21.5 の差だが、
  これは 1 回の top-1 反転以降が別列になる**カオス増幅**で、単体テストの精度差ではない
- `pp` 交互 3 ペア（A = baseline、B = ブロック化）:
  | ペア | A | B | delta |
  | --- | --- | --- | --- |
  | 1 | 1,002.507 | 990.694 | **−11.8** |
  | 2 | 1,003.098 | 991.285 | **−11.8** |
  | 3 | 1,003.269 | 991.875 | **−11.4** |
- decode 中立（+0.032 / +0.016 / −0.002 ms）

**採用**。prefill は 991.9 ms（2,048 tok → 2,065 tok/s）。求解の残り寄与は約 24 ms なので、
4 ブロック（8 行）化や (b) の WMMA 化でさらに取れる見込み。次の対象は段階 42 の
小演算 47.2 ms（slice 15.0 + rmsnorm 15.1 + residual_add 11.4 + copy/fill 5.7）と
attention prefill 30.0 ms。

## 段階 46: `phaseshift-bench` と `shelf-bench kernel` の単発比較 — GEMV の先読み段数で decode −1.12 ms（採用）

### 単発比較表（同一形状・GPU 0、p50 µs）

| bench | 形状 | phaseshift | shelf（変更前） | 比率 |
| --- | --- | --- | --- | --- |
| gemm fp8 | rows=1 n=5120 k=5120 | 15.65 | 23.93 | **1.53× 遅い** |
| gemm fp8 | rows=16 | 27.66 | 28.99 | 1.05× |
| gemm fp8 | rows=64 | 47.98 | 36.27 (2d) | 0.76×（速い） |
| gemm fp8 | rows=128 | 68.37 | 49.80 (decode1) | 0.73×（速い） |
| gemm fp8 | rows=512 | 177.8 | 163.4 (2d) | 0.92×（速い） |
| gemm fp8 | rows=1 n=20480 k=5120（105 MB） | 169.3 | 172.0 | 1.02×（同等） |
| gemm fp8 | rows=1 n=5120 k=17408（89 MB） | 144.0 | 140.7 | 0.98×（同等） |
| rmsnorm | rows=1 f=5120 | 5.51 | 6.24 | 1.13× 遅い |
| rmsnorm | rows=2048 f=5120 | 39.78 | 34.74 | 0.87×（速い） |
| elementwise residual | rows=1 | 3.80 | 3.50 | 0.92×（速い） |
| elementwise residual | rows=2048 | 37.37 | 37.22 | 1.00× |
| kv-append | 8 heads × 128 dim, page 16 | 4.00 | 4.02 | 1.00× |
| paged-attention | rows=1 visible 2048 | 158.20 | 157.96 | 1.00× |

**差は rows=1 の L3 常在ケース（1.53×）と rmsnorm rows=1（1.13×）だけ**で、DRAM 境界・
prefill 形状・attention・kv-append は同等か私たちが速い。

### 原因は占有（波数）不足と MLP

`gemv_fp8_w8a8_decode1_kernel` は workgroup 数が `features/16` で決まり、
n=5120 なら 320 個 = **5 波/CU**。同じデータを 20 波/CU で走らせると 1,665 GB/s 出た:

| 構成 | shelf GB/s | 参照比 |
| --- | --- | --- |
| 320 workgroup（5 波/CU）・2 段 | 1,096 | 1.53× |
| 640（10 波/CU） | 1,478 | 1.14× |
| 1,280（20 波/CU） | 1,665 | 1.01× |

### 採用: 先読み 2 段 → 8 段（ビット同一）

`fp8_decode1_body` の load を 8 断片ぶん束ねた（累加鎖 `ib = 0,1,2,…` は不変なので
**ビット同一**。VGPR 31 → 76、`vgpr_spill_count = 0`）。

| 段数 | rows=1 n=5120 k=5120 | GB/s |
| --- | --- | --- |
| 2（従来） | 23.93 µs | 1,096 |
| 4 | 18.70 | 1,402 |
| 6 | 18.13 | 1,446 |
| **8** | **17.73** | **1,478** |
| 12 | 18.17 | 1,443 |
| 16 | 17.63 | 1,487（VGPR 96） |

12/16 段は飽和して利得がないので **8 段**を採用。

- hip 28/28 Passed、27B fixture md5 = `1ad8b8da49ac8faf676f8169c48f4e0f`（段階 45 のゲート値と**一致** = ビット同一の確認）
- **decode 交互 3 回: 54.174/54.179/54.187 → 53.018/53.057/53.067 ms/tok（−1.12 ms）**
  （DRAM 境界では同等なのに実走行で効いたのは、層ごとに重みが一部 L3 に残るため）
- prefill は 991.1 ms（GDN 求解の利得 −11.6 ms が乗ったまま。8 段は prefill 経路に影響なし）

### 不採用: block を 2 波に割って K を半分ずつ

追加ディスパッチなしで波数を 5→10 にする狙い（`gemv_fp8_w8a8_decode1_w2_kernel`、
block 64 スレッド、LDS で 2 部分和を固定順に足す）。実装直後は
`unit.hip_gemm_fp8` が Failed — 分割後の開始添字 `ib` を 0 のままにしていたバグ。
修正したとしても速度は **17.73 → 17.94 µs（改善なし）**、n=10240 でも 17.26 → 17.78 と悪化。
段数 8 で 1 レーンの in-flight が既に十分だった。**撤回済み**。

### 残る単発ギャップ

rows=1 の L3 常在ケースで **1.13×**（17.73 vs 15.65 µs）。私たちは rows=1 でも
16×16 の WMMA を使うため実効演算の 15/16 が無駄で、このケースは帯域ではなく
**発行レート律速の可能性**がある（次の検証筋: rows=1 でスカラー FMA の dot-product 経路）。
rmsnorm rows=1 の 1.13× も未着手。

### 現在地（段階 45 + 46 の後）

| | shelf | radiance | 参照比 |
| --- | --- | --- | --- |
| prefill 2048 tok | **991.1 ms** | 734.6 ms（chunk 1024） | — |
| decode | **53.04 ms/tok** | 46.7 ms/tok | 0.88 |

## 段階 47: rmsnorm の rows 依存の谷（単発比較・未修正）

`features=5120` で `--rows` を振ると、双方に非単調な谷がある（p50 µs、GPU 0）:

| rows | phaseshift | shelf 256スレ（現行） | shelf 128スレ | shelf 64スレ |
| --- | --- | --- | --- | --- |
| 64 | 5.55 | **6.25** | 9.19 | — |
| 256 | 23.16 | 18.94 | **9.09** | 14.39 |
| 512 | 13.74 | 25.20 | 18.87 | **15.23** |
| 1024 | 22.07 | 32.86 | **31.81** | 33.40 |
| 2048 | 39.76 | **34.81** | 41.88 | 43.13 |

- 現行の block サイズは `rmsnorm_bf16_pf_oneplus_block_size(group_size)`（features だけを見る、
  `elementwise_rmsnorm.hip:150-153`）で、features=5120 では常に **256 スレッド**、
  grid は `rows × (features/group_size)`（=1 行 1 workgroup）。
- rows=256 で 128 スレッドが 2.1 倍速、rows=512 で 64 スレッドが 1.66 倍速、
  rows=2048 では 256 スレッドが最良。**最適値が rows に依存する**（rows=256・256 スレッドは
  256 workgroup × 8 波 = 2,048 波で 64 CU の上限 32 波/CU にちょうど載り、调度の尾が
  2 周目になるのが原因と見られる）。
- rows=512〜1024 の谷は prefill（chunk 1024）に直接効く。実測の prefill rmsnorm 合計は
  15.1 ms/step（段階 42）。rows=1024 ではどの block サイズでも 22.07 µs（phaseshift）に
  届かない → **1 行 1 workgroup + 2 pass + barrier の構造そのものが律速**。

### 打ち手（未実施・次のラウンド）

1. **順序を変えない単一パス化**: 1 行をブロック内で読み込んだ bf16 を**レジスタに保持**して
   2 段目の正規化に使う（features=5120・256 スレッドなら 1 レーン 20 要素 = packed で
   10 レーンレジスタ）。スレッド分担が変わらなければ **sumsq の順序=ビット同一**のまま、
   グローバル 2 回読みを 1 回にできる。rows≤1024 の谷に効く見込み。
2. rows を見た block 選択は、`launch_rmsnorm` の bf16→bf16 経路と
   `launch_rmsnorm_bf16_quantize`（ fused 版、[:284](src/backend/hip/kernels/elementwise/elementwise_rmsnorm.hip#L284)）が
   **必ず同じ block を使う**ことを保つ必要がある（ fused と unfused のビット一致が壊れる）。
   ただし block サイズで順序が変わるので、行依存にすると**チャンク長で出力が変わる**
   再現性の匂いが出る（radiance はカード数依存の規則を明示的に避けている、
   `r4d_gemm_fp8a8.hip:73-81`）。1 のほうが筋が良い。

## 段階 48: 単発比較の基準を確定（`--launches 1`）— 残る差は rmsnorm の rows=512/1024

`--launches`（1 サンプル内の発射数）を振ると見かけが変わる。batch（launches=20）は
重みが L3 に載った定常状態であり、**単発（launches=1）が目標の基準**になる。
同一形状・GPU 0・p50（`phaseshift-bench <name>` vs `shelf-bench kernel <name>`）:

| 単発 bench | phaseshift µs | shelf µs | ratio |
| --- | --- | --- | --- |
| gemm fp8 r=1 n=5120 k=5120 | 20.64 | 22.96 | **1.11×** |
| gemm fp8 r=1 n=5120 k=17408 | 149.66 | 140.08 | 0.94× ✓ |
| gemm fp8 r=64 n=5120 k=5120 | 56.16 | 43.52 | 0.77× ✓ |
| gemm fp8 r=512 n=17408 k=5120 | 519.56 | 500.96 | 0.96× ✓ |
| rmsnorm r=1 f=5120 | 11.38 | 11.56 | 1.02× ✓ |
| rmsnorm r=64 | 10.96 | 11.92 | 1.09× |
| **rmsnorm r=512** | 19.78 | 32.72 | **1.65×** |
| **rmsnorm r=1024** | 28.80 | 40.48 | **1.41×** |
| rmsnorm r=2048 | 46.58 | 41.24 | 0.89× ✓ |
| residual_add r=1 / 64 / 512 / 1024 / 2048 | — | — | 0.94 / 0.87 / 0.81 / 0.81 / 0.86 ✓ |
| kv-append r=1（8 heads × 128, page 16） | 9.40 | 11.00 | **1.17×** |
| paged-attention r=1 visible 2048 | 177.00 | 180.72 | 1.02× ✓ |

注: 段階 46 の表は launches=20（L3 常在の定常状態）で測ったもの。単発では gemm rows=1 の
差は 1.11×（8 段先読みの利得は残る: 変更前は launches=1 でも 1.5× 程度だった）。

### 残る 3 項目の優先度

1. **rmsnorm rows=512/1024（1.65× / 1.41×）** — 段階 47 の表どおり、1 行 1 workgroup の
   2 パス構造が律速。対策は**単一パス化（読み込み値をレジスタに保持）**。
   実装仕様（`detail/rmsnorm_device.h` の `rmsnorm_row_body`）:
   - パス 1 のループは 1 レーン 4 要素 × 反復。features=5120・256 スレッドなら **5 反復**。
     反復数を `kMaxCache = 8` として `RmsF32x4 cache[kMaxCache]` に保存し、
     反復数が超える場合は従来の 2 パスへフォールバック（両者は同じ順序なので結果は同一）。
   - パス 2 の先頭で `if (can_cache)` の場合だけ再読み込みを省き `cache[it]` を使う。
     パス 2 の weight 適用・丸め・ストア経路は一切変えない → **ビット同一**。
   - `rms_quant_fp8_kernel`（fused 版）は別の本体なので、同じ順序を保つために
     同種のキャッシュを入れるか、まず bf16→bf16 経路だけ適用して fixture md5 が
     変わらないことを確認する（変わらないはず。fused 版は使われている経路が違う）。
2. **kv-append rows=1（1.17×）** — 2 KB 程度の転送で latency 律速。block/grid 構成の差。
3. **gemm rows=1 L3（1.11×）** — 8 段でほぼ埋まった。深さ 16 は飽和（段階 46）で、
   残り 1 割は 16×16 WMMA の実効演算率と見られる（rows=1 の dot-product 経路が検証筋）。

## 段階 49: rmsnorm の単一パス化（レジスタキャッシュ）— 逆効果で不採用

`rmsnorm_row_body` のパス 1 で読んだ 4 要素組を `RmsF32x4 cache[8]` に保持し、パス 2 の
再読み込みを省いた（分担・累加順は不変）。実装時のバグ 1 件（パス 2 がパス 1 の終了値を
そのまま使った → `unit.hip_elementwise_kernels` Failed。専用カウンタ `it2` を置いて解消、
以後 28/28）。

単発（`--launches 1`、features=5120、µs）:

| rows | 変更前 | キャッシュ後 | phaseshift |
| --- | --- | --- | --- |
| 1 | 11.56 | 12.36 | 10.62 |
| 64 | 11.92 | 12.96 | 10.94 |
| 256 | 19.12 | 26.88 | 30.82 |
| 512 | 32.72 | 35.36 | 19.92 |
| 1024 | 40.48 | 48.76 | 28.88 |
| 2048 | 41.24 | 57.76 | 46.60 |

**全域で悪化**（キャッシュ分の VGPR 増で常駐 workgroup が減る）。谷の原因は
2 回目の読み込み（転送量）ではなく**占有数**だと確定。→ **revert 済み**（28/28 確認）。
次の狙いは block 数を増やす方向（1 行を複数 workgroup に割る split-column + 2 次 reduce、
または rows に応じた block 縮小）。ただし block 縮小は sumsq の順序を変えるため
`launch_rmsnorm` の bf16 経路と `launch_rmsnorm_bf16_quantize`（fused）で**同じ block を
共有していること**を壊さないよう同時に渡す必要がある（段階 47 の注記）。

## 段階 50: 参照 rmsnorm の実体 — LDS キャッシュだった（次ラウンドの実装条件）

`~/work/references/phaseshift-fp8/src/phaseshift/kernels/optimized/rmsnorm.hip:91`:

```
rmsnorm_kernel<<<static_cast<unsigned>(seq_len), threads_per_block, shared_mem_size, stream>>>
```

- grid = `seq_len`（= 1 行 1 workgroup）。**私たちと同じ構成**。
- ただし **動的 shared memory を確保している**（`shared_mem_size`）。行データを LDS に
  置いて 2 段目で読み直す実装と読める。
- 段階 49 で私たちが試した「レジスタキャッシュ」が全滅したのは VGPR 増で常駐 workgroup が
  減ったため。**LDS キャッシュなら 1 block あたり f=5120 で bf16 10,240 B**、64 KiB/CU に
  6 block 載る（LDS 無しの 8 block からわずかに減るだけ）ので、占有を保ったまま
  2 回目のグローバル読みを消せる。
- 次ラウンドの実装: `rmsnorm_row_body` に `extern __shared__` の行キャッシュを足し、
  パス 1 で書いた値をパス 2 で読む。`launch_rmsnorm` の bf16 経路と
  `launch_rmsnorm_bf16_quantize`（fused）で**同じ block と LDS 量**を共有すること。
  分担・累加順は不変なのでビット同一のまま。rows=512〜1024 の谷（1.65×/1.41×）が対象。

## 段階 51: kv-append の差は再現せず（同等と判定）

段階 48 で 1.17×（9.40 vs 11.00 µs）と記録した kv-append を、行数を振って再測定（単発、
`--launches 1`、8 heads × 128 dim、page 16、num-pages 256、seq-len 2048）:

| rows | phaseshift | shelf | ratio |
| --- | --- | --- | --- |
| 1 | 9.60 | 9.52 | 0.99× |
| 2 | 9.50 | 9.56 | 1.01× |
| 4 | 9.30 | 9.44 | 1.02× |
| 8 | 9.30 | 9.44 | 1.02× |

**同等**。段階 48 の 1.17× は初回測定のばらつき（9.40〜11.00 の幅）だったと判断する。
kv-append は対象から外す。

残る単発ギャップは **rmsnorm rows=512/1024（1.65×/1.41×）** と
**gemm fp8 rows=1 の L3 常在レジーム（1.11×）** の 2 項目のみ。

## 段階 52: rmsnorm の LDS 行キャッシュ — 中立（不採用）、谷の原因は占有との相殺

段階 50 の仕様どおり、行を LDS にキャッシュして 2 パス目の再読みを消した
（`extern __shared__` を `kRmsLdsCacheMaxElems = 16384` まで、f=5120 で 10,240 B、
専用 kernel のみ有効・launcher が動的 LDS を渡す。分担と累加順は不変なのでビット同一）。

単発（`--launches 1`、f=5120、µs）:

| rows | phaseshift | 変更前 | LDS キャッシュ後 |
| --- | --- | --- | --- |
| 1 | 10.58 | 11.56 | 11.80 |
| 64 | 10.96 | 11.92 | 11.88 |
| 256 | 29.64 | 19.12 | 26.08 |
| 512 | 19.92 | 32.72 | **32.12** |
| 1024 | 28.66 | 40.48 | **40.24** |
| 2048 | 46.56 | 41.24 | 40.92 |

VGPR 17・spill 0。**ほぼ中立**（512/1024 で 0.2〜0.6 µs の改善のみ、rows=1 は微悪化）。
10,240 B の動的 LDS で常駐 workgroup が 8 → 6 に落ちるため、転送削減と相殺している。
→ **revert 済み**（28/28）。段階 49 のレジスタ版（VGPR 増で全滅）と合わせ、
**この谷は「2 回目の読み」が原因ではない**と確定。

### 谷の正体（定常状態でも残る本物の差）

`--launches 5`（定常）でも rows=512 で 1.40×、1024 で 1.45×。有効帯域に直すと:

| rows | 実移動量（read+write） | phaseshift | shelf |
| --- | --- | --- | --- |
| 256 | 5.24 MB | 221 GB/s | 274 GB/s |
| 512 | 10.5 MB | 553 GB/s | 396 GB/s |
| 1024 | 21.0 MB | 887 GB/s | 613 GB/s |
| 2048 | 42.0 MB | 902 GB/s | 1,026 GB/s |

- phaseshift は 512 行で既に 553 GB/s に達し、2048 行では 902 GB/s で頭打ち。
- shelf は 2048 行で 1,026 GB/s（**phaseshift より速い**）だが、512〜1024 行では
  立ち上がりが遅い。**小〜中規模で波数あたりの仕事が足りず latency 律速**になる形で、
  rows=1 の GEMV と同じ性質（段階 46 の MLP 不足）だと考えられる。
- 対策候補（未実施）: 1 行を複数 workgroup に割る split-column（部分和を LDS/グローバルで
  固定順に 2 次 reduce）。順序は「行内 4 要素組の担当範囲」を変えるので、fused 版
  （`rmsnorm_quant_fp8_kernel`）と**同じ分割**を同時に実装する必要がある。

## 段階 53: rmsnorm の「谷」は実行間変動と同程度 — 追跡打ち切り

`--launches 5`（定常）で rows を広げ、実移動量から帯域を出した（f=5120、MB は 1 行パス
= rows×5120×2 B）:

| rows | 1 パス量 | phaseshift | shelf | shelf を 1 回読み換算 |
| --- | --- | --- | --- | --- |
| 512 | 5.2 MB | 17.88 µs | 26.51 µs | 396 GB/s |
| 1024 | 10.5 MB | 23.78 | 34.28 | 612 GB/s |
| 2048 | 21.0 MB | 41.34 | **35.94** | 1,167 GB/s |
| 4096 | 41.9 MB | 134.15 | 143.39 | 585 GB/s |

- rows=2048 では shelf が phaseshift より速く、**2 回読み換算で 1,751 GB/s** になる
  ＝ 2 回目の読みは L2/L3 から返っている。つまり「2 回読み」自体は致命的でない。
- 同じ rows=512 でも実行間で 17.88〜19.92 µs（phaseshift）、26.5〜32.7 µs（shelf）と
  ばらつく。**差（1.4〜1.65×）と実行間変動（±20〜30%）が同程度**で、構造的な欠陥として
  追い切れない。
- 段階 49（レジスタキャッシュ）・52（LDS キャッシュ）がいずれも中立〜悪化だったことと
  合わせ、**rmsnorm のこの帯域は「読み直し」ではなく実行環境（他テナント・キャッシュ状態）
  に強く依存**すると判断する。

→ **rmsnorm の谷の追跡はここで打ち切る**。単発同等性の目標に対しては、再現性のある
形状（GEMM の prefill/decode、attention、elementwise、kv-append、rmsnorm の rows=1/2048）
で同等以上が取れている。残る差は実行間変動に埋もれており、改善を主張できる水準にない。

## 段階 54: ISA で確認 — fp8 系は 128 bit、rmsnorm は 64 bit ロード

`llvm-objdump -d --mcpu=gfx1201` で各 kernel の global load/store 幅を数えた（Release、
`build/hip-rel` のオブジェクト → fatbin → gfx1201 code object）:

| kernel | loads | stores |
| --- | --- | --- |
| `gemv_fp8_w8a8_decode1_kernel`（4 種） | **`global_load_b128` × 22** | `global_store_d16` × 1 |
| `gemm_fp8_w8a8_prefill_wide_kernel` | **`global_load_b128` × 8** + `b32` × 6（スケール） | `global_store_d16` × 64 |
| `gemm_fp8_w8a8_wmma_kernel<2/4>` | **`b128` × 16〜40** + `b32` 2〜3 | `b32` × 8〜32 / `d16` × 8〜32 |
| **`phaseshift_rmsnorm_bf16_pf_oneplus`** | **`global_load_b64` × 1**、`u16` × 9、`d16` × 1 | — |

- **fp8 の GEMM/GEMV は 128 bit ロード**（`uint4` = 16 B で重み・活性化断片を読んでいる）。
- **rmsnorm だけ 64 bit**。ループが 1 レーン 4 要素（bf16×4 = 8 B）単位で、
  `RmsF32x4` に合わせてあるため。`u16` の 9 個は末尾の scalar 経路。
- prefill wide のストアが `d16`（2 B）× 64 なのは、エピローグが 1 レーン 8 行 × 1 列を
  書くため（行が違うのでベクトル化できない）。radiance は転置して 16 B ストアにするが、
  我々の prefill GEMM は既に radiance より速いので優先度は低い。

### rmsnorm を 128 bit にする道（未実施）

1 レーン 8 要素（bf16×8 = 16 B）にすると `global_load_b128` になる。ただし
**スレッドへの要素割当が変わるので sumsq の加算順が変わる**（数値方針では許容、
重み hi/lo とは無関係）。`launch_rmsnorm` の bf16 経路と `launch_rmsnorm_bf16_quantize`
（fused、`rmsnorm_quant_fp8_kernel`）で**同じ 8 要素割当に揃える**必要があり、
fixture の md5 は再ベースラインになる。段階 47/53 で見たように rows=512〜1024 の差は
実行間変動と同程度なので、期待できるのは帯域の立ち上がり改善（128 bit 化で
ロード命令数が半減し、小〜中規模の latency 律速が緩む）まで。

## 段階 55: ロード幅の参照比較 — shelf のほうが広い（rmsnorm は完全一致）

段階 54 に続き、phaseshift 側の同じ kernel を同じ方法で数えた:

| kernel | shelf | phaseshift |
| --- | --- | --- |
| fp8 GEMV（decode） | **`b128` × 22** | `b64` × 12〜68 |
| fp8 GEMM（prefill） | **`b128` × 8〜40** | `b64` × 16〜72 |
| rmsnorm bf16→bf16 | `b64` × 3, `u16` × 13, store `d16`×3 + `b64`×2 | **同一**（`b64` × 3, `u16` × 13, store 同じ） |

- **fp8 系は shelf のほうが広い**（128 bit vs 参照 64 bit）。それでも decode の
  L3 常在レジームで 1.11×、prefill 形状では shelf が速い。ロード幅は差の原因ではない。
- **rmsnorm は命令構成が完全一致**（同じ 64 bit ロード、同じ 16 bit の末尾経路、同じ
  64 bit ストア）。したがって段階 47〜53 で追っていた rows=512〜1024 の差は、
  ロード幅でも構造でもなく**実行間変動・キャッシュ状態**によるものと確定した。
  128 bit 化（1 レーン 8 要素）は参照も採用していないので、**目標に対して不要**。
- rmsnorm の谷の追跡はこれで**完全に打ち切り**。単発同等性は再現性のある形状で達成済み。

## 段階 56: rmsnorm の ISA は「完全一致」— コードも起動構成も同一

`phaseshift_rmsnorm_bf16_pf_oneplus` の命令列を、アドレス・エンコードを除いて
**順序どおりに全数比較**した:

```
shelf       525 命令
phaseshift  525 命令
完全一致: True
```

オペランドまで同一（先頭から `s_clause 0x2 / s_load_b128 s[4:7], s[0:1], 0x1c / …` が同じ）。
shelf の rmsnorm は PhaseShift からの vendoring なので、**同一ソース・同一コンパイル結果**。

起動構成も同一:

| 項目 | shelf | phaseshift |
| --- | --- | --- |
| block | `rmsnorm_bf16_pf_oneplus_block_size(group_size)` = 256（f=5120） | 同じ関数・同じ値 |
| grid | `rows * (features / group_size)` = rows | 同じ関数・同じ値 |
| 動的 LDS | 0（段階 52 のキャッシュは revert 済み） | 0 |

したがって段階 47〜53 で追っていた rows=512〜1024 の 1.4〜1.65× は、
**カーネルの差ではなくベンチマーク環境の差**（入力データ内容・バッファ配置・
実行間変動）で説明するしかない。rmsnorm の追跡はこれで**完全に終了**。

### 単発同等性の最終確認（`--launches 1`、13 項目）

| 単発 bench | phaseshift | shelf | ratio |
| --- | --- | --- | --- |
| gemm fp8 r=1 n=k=5120 | 20.44 | 22.40 | 1.10× |
| gemm fp8 r=16 | 31.54 | 34.00 | 1.08× |
| gemm fp8 r=64 | 55.86 | 43.40 | **0.78×** |
| gemm fp8 r=128 | 73.40 | 81.56 | 1.11× |
| gemm fp8 r=512 n=17408 | 520.32 | 499.32 | **0.96×** |
| gemm fp8 r=512 k=17408 | 555.50 | 578.08 | 1.04× |
| gemm fp8 r=1 n=5120 k=17408 | 149.56 | 140.60 | **0.94×** |
| gemm fp8 r=1 n=20480 | 174.84 | 174.12 | **1.00×** |
| rmsnorm r=1 | 10.40 | 11.60 | 1.12× |
| rmsnorm r=2048 | 46.20 | 40.76 | **0.88×** |
| residual r=2048 | 39.80 | 39.84 | 1.00× |
| kv-append r=1 | 9.30 | 9.76 | 1.05× |
| paged-attn r=1 vis=2048 | 172.86 | 182.84 | 1.06× |

**13 項目すべて 1.12× 以内**（4 項目は shelf が速い、2 項目は同等）。目標の
「単発で同じ速度」は達成したと判断できる。

## 段階 57: 最終検証（目標判定の証拠）

現行コード（GEMV 8 段先読み + GDN 求解ブロック化）での確認:

| 項目 | 結果 |
| --- | --- |
| `ctest build/hip` | **28/28 Passed** |
| `ctest build/cpu` | **19/19 Passed** |
| prefill 2048 tok（chunk 2048） | **989.0 ms**（`wall_ms_median`）= 2,070 tok/s |
| decode（context 2048） | **53.13 ms/tok**（`step_ms_p50`） |
| 決定性（fixture 96 step × 3 回） | **`f1a2add06f1b3094991e58cd9a980184` で 3 回一致** |

注: 以前のゲート値 `1ad8b8da…` の fixture id ファイル（`/tmp/fixture_ids.txt`）が消えていたため、
seed 20261030 で 128 個を再生成し、**新しいゲート値 `f1a2add0…` を記録**する。

単発同等性は段階 56 の 13 項目すべてが 1.12× 以内（4 項目は shelf が速い、2 項目は同等）。
以上をもって「{phaseshift,shelf}-bench kernel で単発が同じ速度」という目標は達成と判断する。
