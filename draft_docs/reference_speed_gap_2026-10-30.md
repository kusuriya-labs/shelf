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
