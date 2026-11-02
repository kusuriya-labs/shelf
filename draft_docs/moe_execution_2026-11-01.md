# MoE 実行の設計と実機確認（T17 / フェーズ 2〜4・6）

状態: 実装中（GPU 経路・CPU 分担・合流・縮退まで。ルーティングのデバイス化と
CPU の高速化は未着手）
対象: `Qwen3.5-35B-A3B-MXFP4`（40 層・256 エキスパート・top-8・共有エキスパート）
関連: `draft_docs/port_strata_cpu_gpu_coop.md`（全体計画）、
`draft_docs/parity_policy.md`（数値判定）

---

## 1. 何を入れたか

MoE を `moe_route` / `moe_merge` の 2 ノードに分けると、全エキスパート分の
`[E,H]` 中間を毎トークン materialize することになり、E=256・top-8 の疎性を
実行器が活かせない。そこで **1 ノードの複合プリミティブ `moe_mlp`** を追加した
（`swiglu` / `rmsnorm_gated` と同じ扱い。数学的な意味だけを IR に書く）。

### 1.1 IR 契約（`src/ir/include/shelf/ir/graph.hpp` の `Kind::moe_mlp`）

* 入力: `x [R, H]`
* 重み（宣言順）:
  1. `router [E, H]`
  2. エキスパート `e = 0..E-1` の 3 本: `gate [I,H]`, `up [I,H]`, `down [H,I]`
  3. 共有エキスパート（任意・4 本）: `gate [Is,H]`, `up [Is,H]`, `down [H,Is]`,
     `gate_vec [1,H]`
* 属性: `top_k`（1..E）
* 出力: `x` と同じ `[R,H]`
* 意味論:
  ```
  p    = softmax(router(x))            # f32
  top  = p の上位 top_k（同値は小さい index）
  w    = top / Σ top                   # 再正規化
  y    = Σ_j w_j · expert_{idx_j}(x)
  y   += sigmoid(x · gate_vec^T) · shared_expert(x)
  ```
  非選択エキスパートの寄与は 0。`expert_e(x)` は
  `down(silu(gate(x)) * up(x))`。

### 1.2 モデル側（`src/model/qwen35`）

* `TextConfig` に `num_experts` / `num_experts_per_tok` / `moe_intermediate_size` /
  `shared_expert_intermediate_size` / `mlp_only_layers` を追加。
  `intermediate_size` は MoE 専用 checkpoint では存在しないため任意にした。
* `is_moe_layer(l)` は transformers の `Qwen3_5MoeTextConfig` と同じ判定
  （`num_experts > 0 && top_k > 0 && l ∉ mlp_only_layers`）。
* `step_graph` は MoE 層だけ `moe_mlp` を 1 ノードで組み、dense 層は従来の
  3 本の `linear` のまま。実機 config では **963 プリミティブ / 40 個の
  moe_mlp / 31333 重み**（グラフ構築 約 6 秒/全 40 層）。

### 1.3 実行器

* CPU リファレンス: `moe_mlp` を f32 で実装（ルーティング + 選択エキスパート +
  共有エキスパート）。HIP との比較の正本。
* HIP: ルーティングだけホストで行い、行列積はデバイスの既存カーネルを使う。
  * ルーター logits: `moe_router_logit`（1 ブロック = 1 エキスパート、H を
    スレッドで分担する決定的な縮約）
  * エキスパート: bf16 は `launch_gemm_bf16`（ExactRows, rows=1）、
    MXFP4 は `launch_gemm_mxfp4_w4a8_decode1`（活性化は E4M3、group 128）。
    W4A4（活性化も E2M1 + E8M0）は削除した。
  * `silu(gate)*up` は `launch_swiglu_bf16`、重み付き加算は
    `moe_accumulate`（f32 累積）、最後に `f32_to_bf16`。
  * 共有エキスパートは bf16 の GEMV（checkpoint の除外宣言）。
* 一時領域は実行器が確保する（アリーナとは別。primitive 内の scratch を
  宣言相に押し込まない）。幅は層で変わるので必要量が増えたときだけ再確保。

### 1.4 CPU 分担（フェーズ 3）

* 分担の規則: **エキスパートの 3 本（gate/up/down）がそろって pinned host の
  まま**なら CPU で計算する。1 本でも VRAM にあれば GPU に任せる（GPU は
  mapped host の重みも読めるので、混在でも正しい。遅いだけ）。
* 判定はその行の先頭（ルーティング直後）で固定する。後から変えると前の層の
  分担で計算してしまう（Strata の実測と同じ理由）。
* CPU の計算:
  * MXFP4 重みは**復号しながら内積**を取る（E2M1 のニブル → 8 値 LUT、
    E8M0 → 2 の累乗）。重みを f32 に展開しないのでメモリは増えない。
  * 活性化は GPU と同じ量子化を通す（W4A8: E4M3 + 動的スケール、K 128 要素ごと）。
    分担の有無で数値が動かない方向に揃える。
  * 入力 x の量子化は行で共通なので 1 回だけ。mid はエキスパートごと。
  * エキスパート単位でスレッド分割（既定 `hardware_concurrency`、上限 16。
    `SHELF_MOE_CPU_THREADS` で固定）。分割と足し順は固定なので同一構成では
    決定的。
  * gate/up/mid は GPU と同じく bf16 に丸めてから次へ渡す（GPU の swiglu は
    bf16 入出力のため）。この丸めを揃えないと mid の E2M1 コードが境界で
    食い違い、実機で top-1 が入れ替わった。
* 合流: GPU のカーネルを enqueue した後に CPU を回し、CPU の f32 累積を
  1 回だけ H2D して `moe_add_f32` で足す。共有エキスパート（GPU）も CPU の
  前に enqueue するので、CPU 計算と GPU 実行が同じ行の中で重なる。
* 実測（35B、offload 3000 MiB/ステージ）: CPU 992 エキスパートで
  `cpu_ms=1259`（約 1.3 ms/エキスパート、16 スレッド）。単スレッドでは
  約 9 ms/エキスパートだったので、スレッド化で約 7 倍。

### 1.5 縮退（フェーズ 6）

* `HipStepExecutorOptions::allow_host_fallback`（`--allow-host-weights` から
  配線）を有効にすると、**device の確保に失敗しても pinned host に置いて
  続行する**。MoE はその重みを CPU で計算し、dense は mapped host を GPU が
  読む（遅いが止まらない）。
* MXFP4/FP8 のスケールも同じ規則で pinned に落とす（コードだけ host に
  行くとスケールが無くて計算できないため）。
* 故障注入 `SHELF_TEST_FAIL_DEV_ALLOC=N` で先頭 N 回の重み確保を失敗させ、
  縮退経路を単体テストで検証する（下記）。

### 1.6 制限（現時点）

* ルーティングをホストで行うため、**1 行 × 1 層ごとに D2H 同期**が入る。
  decode 1 トークンで 40 同期、prefill では行数分だけ増える。
* エキスパートは 1 個ずつ GEMV する（同一エキスパートに集まった行の
  グループ GEMM は未実装）。
* CPU の内積はスカラー（SIMD 化・重みのキャッシュは未着手）。
* CPU 分担の有無で**近接した top-1 は入れ替わりうる**（下記 §2.3）。

---

## 2. 検証

### 2.1 単体（GPU 不要 / GPU）

* `tests/unit/test_ir_moe.cpp`
  * config の MoE 欄（`mlp_only_layers` を含む）と lowering
  * `moe_mlp` の形状検証（`top_k` の範囲・重みの並び・`down` の `[H,I]`）
  * CPU リファレンスの値が独立に書いた double 参照と一致（共有あり/なし）
  * MoE 専用 config（`intermediate_size` 無し）が受理される
* `tests/unit/test_hip_moe.cpp`
  * 合成の 4 エキスパート + 共有エキスパートを HIP と CPU で比較。
    実測 `worst|diff| = 3.6e-4`（bf16 活性化の丸めの範囲）。
  * エキスパートを pinned host に置いて CPU 計算させた場合も一致
    （`worst|diff| = 5.4e-4`、`cpu=4 / gpu=0`）。
  * 故障注入で device 確保を全滅させても pinned に落ちて完走し、同じ値に
    なること（縮退の検証）。
  * 同一入力の 2 回目が一致（アリーナの残骸を読まない）。

### 2.2 実機（Qwen3.5-35B-A3B-MXFP4）

`build/hip-rel/apps/shelf-run` で 4 GPU にレイヤ分割して実行。

```
shelf-run --model models/Qwen3.5-35B-A3B-MXFP4 --ids 100,200,300 \
  --steps 2 --pipeline-parallel 4 --prefill-rows 4 --device hip
```

* ロード: 4 ステージ合計 約 21.1 GiB、materialize 約 5〜7 秒/ステージ
* 生成: `100,200,300 → 4183 → 344`（greedy）
* decode 1 ステップ: 約 54 ms（PP=4）/ 約 41 ms（PP=1）
* 20 トークン prefill（`--prefill-rows 20`）も完走。

比較（`tests/scripts/compare_logits.py`）:

| 比較 | top-1 | max abs | rel L2 |
| --- | --- | --- | --- |
| PP=4 vs PP=1（3 トークン、2 step） | 一致（344） | 1.22 | 0.111 |
| 参考: 9B-FP8 の PP=4 vs PP=1 | 一致（445） | 0.50 | 0.038 |
| prefill 20 行 vs 4 行 × 5（20 トークン） | 一致（72） | 1.48 | 0.076 |

PP 差・チャンク差は dense の既存経路でも出る（FP8 の行数依存と prefill
attention のタイル差。`draft_docs/parity_policy.md` §3 の T2 既知の範囲）。
MoE 側は行ごとに独立に計算するため、差の原因ではない。

### 2.3 CPU 分担の実機測定（35B-A3B-MXFP4）

`shelf-compute` に 3 トークン（100,200,300）を 1 回与え、`--offload-weights-mib`
を変えて分担と時間を測った（`SHELF_MOE_STATS=1`。prefill 3 行 + decode 2 行）。

| offload / ステージ | host 常駐（4 ステージ合計） | CPU / GPU エキスパート | CPU 時間 | 生成列（greedy） |
| --- | --- | --- | --- | --- |
| 0（GPU のみ） | 0 MiB | 0 / 1600 | 0 ms | 4183, 344, 264 |
| 1000 | 4000 MiB | 269 / 1331 | 438 ms | 220, 200, 200 |
| 3000 | 12000 MiB | 985 / 615 | 1248 ms | 220, 200, 200 |
| 6000 | 20030 MiB | 960 / 0 | 1135 ms | 先頭が EOS（停止） |

CPU 1 エキスパートあたり約 1.2〜1.6 ms（16 スレッド）。全 CPU でも 1 トークン
あたり 0.4 秒程度で、低速だが継続する。

**生成列が GPU のみと一致しない理由**（重要）:

* 配置だけを pinned にした対照（`SHELF_MOE_NO_CPU=1`）は GPU のみと **bit 一致**
  した。したがって差は CPU の計算経路に由来する。
* 1 エキスパート単位では、CPU は f64 参照と **完全一致**し、GPU は f64 参照と
  最大 8e-4 ずれる。GPU カーネル自体は単体テスト（実寸 512×2048 / 2048×512）で
  相対誤差 0 なので、差は「gate/up の f32 累積順 → bf16 丸めの境界 → mid の
  E2M1 量子化コードの 1 段違い」による離散的なものである。
* これが 40 層で増幅され、近接した top-1 が入れ替わる。PP 差・prefill チャンク差と
  同じ「T2 の近接 top-1」に分類する（`draft_docs/parity_policy.md` §3）。
* **同じ分担なら同一プロセス内で決定的**（分割と足し順を固定している）。
  VRAM 予算を変えると分担が変わり、近接 top-1 が変わりうる。

---

## 3. 次の段階

1. **重なりの計測**（フェーズ 4 の残り）: enqueue 後〜合流点の GPU 時間と CPU
   時間をイベントで分離し、wall が和より短いことを数字で示す。現状は
   「GPU を enqueue してから CPU を回す」構造まで。
2. **適応配置**（フェーズ 5）: 実績（ルーティングのヒット）から常駐を選び直す。
3. 性能: ルーティングのデバイス化（D2H 同期の削減）、CPU の SIMD 化、同一
   エキスパートに集まった行のグループ GEMM 化。
4. 数値: 分担で top-1 を変えないことを要件にするなら、量子化境界の扱い
   （例: mid を量子化せず W4A8 相当で down に渡す）を設計する。
