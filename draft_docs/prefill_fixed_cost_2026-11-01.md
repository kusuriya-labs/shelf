# Prefill の固定費プロファイル（T11）と launch 削減の判断（T12）

日付: 2026-11-01
環境: R9700 (gfx1201) ×4 / HIP 7.15 / `build/hip-rel`（Release）/ 9B-FP8。
計測器:
* `shelf-bench pp --profile`（新設。`SHELF_HIP_STEP_PROFILE=1` 相当）
  — ステップの 3 点で `hipEvent` を取り、wall / GPU / カーネル列 / アップロード /
  ホスト側 / リセット / declare を分ける
* `rocprofv3 --kernel-trace`（発射回数とカーネル時間の内訳）
* `shelf-bench kernel rmsnorm`（1 発射あたりの固定費）

---

## 1. ステップ内訳（9B-FP8・prompt 256・warmup 1・3 run の中央値）

| chunk（行/step） | steps（run あたり） | wall ms | GPU ms | kernel ms | upload ms | host ms | reset ms | declare ms（起動時） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 8 | 31.700 | 31.392 | 31.375 | 0.018 | 0.277 | 0.093 | 7.75 |
| 64 | 4 | 36.093 | 35.773 | 35.754 | 0.019 | 0.279 | 0.096 | 7.89 |
| 128 | 2 | 48.186 | 47.886 | 47.866 | 0.019 | 0.373 | 0.098 | 6.09 |

* wall ≈ GPU + host。**GPU が 99% 以上**で、ホスト側（dispatch + sync 呼び出し）は
  0.28〜0.37 ms/step。入力アップロードは 0.02 ms（32 KB 程度）。
* `reset`（サーバの 1 リクエストごとに相当）は 0.1 ms のホスト時間。KV プールの
  `hipMemsetAsync` は非同期なので、この値には GPU のゼロ埋め待ちを含まない。
* `declare`（アリーナ・状態の確保と offsets の適用）は起動時に 6〜8 ms。
  プロセス起動の 1 回だけ。
* プロセス最初のステップだけ +17 ms（32 行: 48.7 ms、steady 31.4 ms）。
  最初の数回はモジュール初期化・初回タッチの影響なので **warmup で捨てる**。

### 固定費のモデル

32 行と 128 行の 2 点で線形近似すると

```
step_ms ≈ 25.9 + 0.172 × rows      （GPU 時間）
```

* 32 行では固定費が **82%**、128 行でも **54%**。
* 1 行あたりの GPU 時間: 32 行 0.981 ms / 64 行 0.559 ms / 128 行 0.374 ms。
  行数を 4 倍にしてもステップ時間は 1.52 倍にしかならない。
* 64 行は近似より 1.1 ms 速い（25.9+0.172×64=36.9 対 実測 35.8）。固定費は
  完全な線形ではなく、GEMM のタイル選択（RowBlock と 2 次元タイル）で
  傾きが変わるため。**固定費そのものは 24〜27 ms** と見てよい。
* 固定費の中身は launch ではなく **重みの再読み**（14 GB / 26 ms ≈ 540 GB/s）が主。
  カーネル数は行数によらず一定（下記）なので、固定費が行数に比例しない理由も説明できる。

---

## 2. 発射回数と launch overhead

`rocprofv3 --kernel-trace` で 1 ステップを数えた（model load の
`__amd_rocclr_*` コピー／フィルを除く）。

| 経路 | 発射回数/step | 主な内訳（重複あり） |
| --- | --- | --- |
| prefill 32 / 64 / 128 行 | **756**（3 形状とも同じ） | fp8 GEMM 128 / bf16 GEMM 120 / row_slice 89 / rmsnorm 81 / act quant 80 / residual 64 / swiglu 56 / GDN 系 72 / attention 8 / lm_head 1 ほか |
| decode 1 行 | **755** | 同じカーネル群を 1 行で |

* CPU 側の enqueue は **0.28 ms / 756 発射 ≒ 0.37 µs/発射**。CPU は律速ではない。
* 1 発射の固定費は `shelf-bench kernel rmsnorm` で実測:
  * rows=1 / features=128: **3.83 µs/発射**
  * rows=1 / features=8192: 6.83 µs/発射（差分 3.0 µs が実データ処理）
  * → **直列ストリームでの launch 固定費 ≈ 3.5 µs/発射**
* したがって 1 ステップの launch 費 ≈ **756 × 3.5 µs ≒ 2.65 ms**。

| 経路 | ステップ GPU | launch 費（推定） | 割合 |
| --- | --- | --- | --- |
| prefill 32 行 | 31.39 ms | 2.65 ms | **8.4%** |
| prefill 64 行 | 35.77 ms | 2.65 ms | **7.4%** |
| prefill 128 行 | 47.89 ms | 2.65 ms | **5.5%** |
| decode 1 行 | 20.67 ms | 2.64 ms | **12.8%** |

（`rocprofv3` のカーネル時間合計は 32 行 31.31 / 64 行 35.14 / 128 行 46.09 ms で、
profiler 下ではカーネル時間が膨らむため、上の launch 費は「rmsnorm の実測」を
基準にした推定。profiler の合計と event の差は 0.1〜1.8 ms。）

---

## 3. T12 の判断: Graph capture/replay は **今は実装しない**

* 削減できる上限は launch 費の 2.65 ms/step。prefill 128 行で 5.5%、32 行で 8.4%。
* 固定費 24〜27 ms の主因は**重みの再読み**であり、Graph では消えない。
* Graph capture の制約:
  * prefill は chunk 行数（32/64/128/端数）ごとにグラフが要る。プロンプト長で
    形が変わるため、形ごとの capture と再利用管理が必要。
  * decode は形が固定（1 行）で効果は 12.8% と大きいが、位置 `base_pos` と入力
    トークンを毎ステップ書き換える必要がある。現行はホスト側 `posrep_host_` を
    作って `hipMemcpyAsync` するため、capture 時は**ホストポインタが固定**になり
    そのままでは使えない（デバイス側バッファへ書く経路か graph exec update が要る）。
  * 失敗時のフォールバック（capture 不能な API が混ざる）と、既存の
    `SHELF_HIP_PROFILE` / `SHELF_HIP_DEBUG_SYNC` との排他も必要。
* 結論: **prefill は効果 < 6% で複雑さに見合わない。decode は 12.8% あるため、
  将来 decode 専用に「位置・トークンをデバイス側で更新する」設計とセットで
  再検討する**。今回は T11 の計測結果を記録し、実装は見送る。

---

## 4. T13: 9B-FP8 128 行の「1.5% 退行」の再測定

`draft_docs/fp8_small_tile_2026-11-01.md` §7 の「9B-FP8 の 128 行だけ 1.5% 遅い」
（2,690 → 2,653 tok/s）を、同じ条件で A/B し直した。

### 方法

* 比較: **2eceddb**（退行前。`fp8_small_tile` の比較基準）と **HEAD**（f2fad62）。
  `.worktrees/ab_2eceddb` に 2eceddb の worktree を作り、`shelf-bench` だけを
  Release / gfx1201 でビルドした。
* 条件: 9B-FP8、prompt 256、chunk 128、`--runs 10 --warmup 3`。
  A→B→A→B の順で同一 GPU（0）に交互に流した。
* 実行間変動の確認: 同一バイナリ・同一プロセスで 20 run（stddev 0.17 ms）、
  プロセスを分けて GPU 0 / 1 で 3 回ずつ（中央値 47.3〜48.2 ms）。

### 結果

| 実行 | wall median | tok/s | stddev（同一プロセス） |
| --- | --- | --- | --- |
| 2eceddb（A） | 94.659 ms | 2,704 | 0.369 |
| HEAD（B） | 96.562 ms | 2,651 | 0.272 |
| 2eceddb（A） | 95.049 ms | 2,693 | 0.278 |
| HEAD（B） | 96.499 ms | 2,653 | 0.086 |

* **退行は再現した**（A 94.7〜95.0 ms / B 96.5 ms、差 1.5〜1.9%）。
  同一プロセスの stddev は 0.1〜0.4 ms なので、**実行間変動では説明できない**。
* プロセス間・GPU 間の中央値のばらつきは 47.3〜48.2 ms（1.6%）で、これは
  「1.5% を単発の比較で判定してはいけない」ことを示す。今回は同一プロセス内で
  A/B を交互に取ったので、差はノイズではない。

### 原因（特定）

`rocprofv3 --kernel-trace` で 1 ステップのカーネル別時間を A/B 比較したところ、
差は **f2fad62 で入った elementwise の x8 版**に集中していた。

| カーネル | 2eceddb | HEAD | 差 | 発射数 |
| --- | --- | --- | --- | --- |
| 残差加算（旧 `phaseshift_residual_add_bf16`） | 0.351 ms | — | — | 64 |
| 残差加算（新 `residual_add_bf16_x8`） | — | **1.117 ms** | **+0.766 ms** | 64 |
| 行スライス（旧 `bf16_row_slice`） | 0.450 ms | — | — | 89 |
| 行スライス（新 `bf16_row_slice_vec8`） | — | **0.849 ms** | **+0.399 ms** | 89 |
| その他（GEMM・GDN・rmsnorm など） | 44.17 ms | 44.13 ms | -0.04 ms | — |

* 発射数は同じ（756）。**VRAM 配置ではなく、カーネル本体の置換**が原因。
* 新カーネルは 16 B の `uint4` ロード/ストアで命令数を減らす狙いだが、
  残差加算は bf16 の要素ごとに scalar 変換（`__ushort_as_bfloat16` →
  float 加算 → `__float2bfloat16`）を行うため、**命令数が増えて逆効果**に
  なっている（VGPR も 24 → 32）。旧実装は bf162 の packed 加算だった。
* 9B でだけ目立つのは、GDN 層が多く残差加算 64 発射・行スライス 89 発射を
  128 行で呼ぶため。27B は 1 層あたりの比率が違い、+0.7% に留まっていた。

### 追試: 形状別の A/B

`SHELF_FORCE_ELEM_V4` / `SHELF_FORCE_RES_V4` / `SHELF_FORCE_SLICE_SCALAR` で
カーネルを切り替え、行数を変えて測った（9B-FP8、prompt は chunk の 2 倍、
warmup 2・6 run の中央値）。

| rows | x8（既定） | v4 | x8 の差 |
| --- | --- | --- | --- |
| 32 | 31.09 ms | 32.61 ms | **+4.9%** |
| 64 | 289.80 ms（8 step） | 305.30 ms | **+5.2%** |
| 96 | 365.29 ms（8 step） | 371.29 ms | +1.6% |
| **128** | **96.62 ms** | **94.47 ms** | **-2.2%** |
| 160 | 98.29 ms | 98.51 ms | +0.2% |
| 192 | 71.56 ms | 71.88 ms | +0.4% |
| 256 | 152.48 ms | 152.09 ms | -0.3%（ノイズ） |
| 384 | 110.20 ms | 110.65 ms | +0.4% |
| 512 | 277.20 ms（2 step） | 277.54 ms | +0.1% |
| 2048 | 536.96 ms | 538.82 ms | +0.3% |
| 27B-FP8 128 | 200.46 ms | 202.19 ms | **+0.85%** |

* **9B の 128 行だけが負ける**。27B の 128 行は x8 が勝つので、
  「128 行だから v4」という規則は誤り（hidden 4096 の 1 MB テンソルに固有）。
* 128 行での内訳: 残差だけ v4 にすると 95.19 ms（-1.43 ms）、
  スライスだけスカラーにすると 96.01 ms（-0.60 ms）、両方 v4 で 94.47 ms。
  負けの 84% が残差加算、28% が行スライス（重複あり）。
* `residual_add_bf16_x8` を packed `__hadd2` に書き直しても
  96.62 → 96.62 ms（変化なし）。**命令数ではなく、x8 がブロック数を半分に
  するための占有・テール効果**と見られる（VGPR 32 / ブロック 256 対 512）。

### 判断

* **x8 を既定のまま残す**。サーバ既定の 64 行で +2〜5%、27B/128 で +0.85%、
  32/96/160/192/384/512/2048 で +0.1〜5% に対し、9B/128 だけ -1.6〜2.2%。
* 形状別に切り替える案は、9B と 27B で勝敗が逆転するため採用しない
  （将来 hidden が変わると規則が破綻する）。9B/128 を特に使う用途があれば
  `--prefill-chunk 64/96/160` のいずれかにする方が安全（64 は x8 で最速）。
* A/B 用の env と形状ダンプ（`SHELF_DUMP_ELEM=1`）は再測定用に残した。

---

## 5. 再現コマンド

```bash
# 内訳
./build/hip-rel/apps/shelf-bench pp --model-dir models/Qwen3.5-9B-FP8 \
    --prompt-tokens 256 --prefill-chunk 128 --runs 3 --warmup 1 --profile

# decode の内訳
./build/hip-rel/apps/shelf-bench tg --model-dir models/Qwen3.5-9B-FP8 \
    --context 128 --tokens 16 --warmup 4 --profile

# 発射回数（1 ステップ）
rocprofv3 --kernel-trace -f csv -o /tmp/prof -- \
    ./build/hip-rel/apps/shelf-bench pp --model-dir models/Qwen3.5-9B-FP8 \
    --prompt-tokens 128 --prefill-chunk 128 --runs 1 --warmup 0
# tr_kernel_trace.csv から __amd_rocclr_* と rope_inv_freq_init を除いて数える
```
