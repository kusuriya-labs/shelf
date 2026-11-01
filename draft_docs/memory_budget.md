# memory_budget — KV・VRAM の予算と所在（別紙）

状態: 骨（**数値の決定は `engine_contract.md` §8 にあり、本書はその受け皿**。
実装・運用の詳細を書くのは実装期）
関連: `engine_contract.md` §8（決定済みの数値・2 段 admission・超過時の挙動）、
`directory_design.md` D2（予算計算は `placement/`、実確保は backend）、
`port_strata_cpu_gpu_coop.md`（tier 移動の実装接続）

本書の位置づけ: `engine_contract.md` が「契約として決めた数値」を、本書が
「その数値の算出・監視・運用」を書く。**同じ数値を 2 箇所で書き換えないこと**
（片方を変えたら本書も更新する）。

---

## 1. 決定済みの数値（`engine_contract.md` §8.1 / §8.2 の写像 — 変更はそちらで行う）

| 項目 | 決定値 |
| --- | --- |
| KV 正本の所在 | **PinnedHostRAM**。VRAM は online 分だけを載せる |
| 1 リクエストの decode 予算 | 32,768 tok（思考 24,576 + 非思考 8,192 の確保あり） |
| ContinuousBatching | 最大 4 並列（スロット = 4）。予算は 4x = 131,072 tok |
| VRAM 上の online KV 上限 | 81,920 tok = 2.50 GiB（正本 4.00 GiB と別桁。大小比較しない） |
| 予算式 | `KV_bytes(T) = T × L_full × 2 × kv_heads × head_dim × sizeof(bf16)`
  （開発モデル `Qwen3.5-9B` で 1 tok = 32 KiB。`L_full` は full_attention 層のみ、GDN は固定費） |
| admission | **2 段**: ① pinned 容量に対して `Σ (prompt_i + max_new_i − 1)`、
  ② 実行は active 文脈の合計 ≤ 81,920 tok。**現状の実装**: 容量はスロット ×
  最大文脈で固定し、①は per-request の `prompt + max_new <= max_context`、
  ②は T15 のスロット待ち（`--max-inflight` / スケジューラ）が担う。オンライン
  ミラーは容量を増やさないキャッシュなので、新しい admission は不要 |

## 2. 予算の算出と決定者（2026-10-30 実装 — D2）

* **算出は純粋関数**（`src/placement/include/shelf/placement/budget.hpp`）:
  * `kv_bytes_per_token(full_layers, kv_heads, head_dim)` — §1 の式そのもの（式の正本）
  * `plan_budget(footprint, demand, limits)` — **最大ステージ**で判定し、不足を
    **名前付き**（`device: 最大ステージ X が使える Y を Z 超過` / `host(pinned): KV …`）で返す
  * `place_all_device`（既定 = 全 device 常駐）/ `place_fitting_budget`（opt-in で pinned へ）
  * `Limits.device_free_bytes == 0` は「不明」として**判定しない**（不明を 0 として
    誤って拒否しない）。`headroom` 既定 5%
* **実装（2026-11-01・T14 第 1 段）**: `place_fitting_budget` を**実確保へ接続**した。
  `make_pipeline` が重み候補（checkpoint 名・バイト数）を作り、状態とアリーナを
  差し引いた device 量を渡す。pinned host になった重みは `hipHostMalloc(mapped)` に
  正本を置き、カーネルには `hipHostGetDevicePointer` のポインタを渡す。
  `promote_host_weights()` で VRAM へ移せる（転送 + 再配置。`shelf-run
  --promote-after N`）。9B-FP8 で 1940 MB を落としても生成列は一致し、
  promote 後に 88 ms/step → 21 ms/step へ戻ることを確認（詳細は
  `draft_docs/vram_placement_2026-11-01.md`）。
  **KV も pinned 直読み方式で実装済み**（`--kv-in-host` / 予算不足時の
  `--allow-host-kv`。K/V プールを `hipHostMalloc(mapped)` に置き attention が直接読む）。
  さらに **online KV ミラー**（`--online-kv-pages N`）を実装: pinned 正本の直近
  N ページを VRAM に複製し、attention は窓内を VRAM・古いページを host から読む。
  窓はスロットごと（最大 4）で、4 スロット × 40 ステップでも host のみと
  logits 完全一致（`vram_placement_2026-11-01.md` §1.5）。
* **入力の取り方**: 呼び手（`apps/common/pipeline.cpp`）が
  ① 重み = 実体化計画のバイト数 ② 状態 = 宣言済み state の f32 ③ アリーナ =
  `compiler::plan_arena` の `peak_bytes`（`--arena-mib` 固定時はその値）を積み、
  空き VRAM は `hipMemGetInfo` で実測して渡す。**確保（hipMalloc）はこの判定の後**
* **出力と失敗**: 判定は `Status::insufficient_memory` として返り、アプリは
  `pipeline: budget: 必要 X / 使える Y [不足] …` を出して**確保前に**終了する
  （`serve_protocol.md` §3 では `overloaded` / 503 に写像される）
* **GDN 固定費**: 状態は `steps` 宣言から自動的に積まれる（`kv_capacity` 非依存の分も
  含む）。§8.7 の「未計上」は解消（pinned 正本の KV は `Demand.host_kv_tokens` で計上可能）

## 3. 段 2 の実行側の選択肢（`engine_contract.md` §8.6 で (a) に確定）

* (a) admission を絞る / (b) pinned 直読み / (c) 層単位受け持ち
* 「VRAM に無い active block をどう読むか」が性能を決める（同 §8.3 の試算:
  全量移すと約 2.5 GiB / トークン ≈ 100 ms/トークン）。**その試算の再計測はフェーズ 0**

## 4. 未決（本書で拾う）

* **tier 移動の発動体** — いつ pinned ↔ VRAM を動かすか（`engine_contract` §6 / §8.5 の残件）
* 思考打ち切り（reasoning cap）の判定主体（`engine_contract` §8.4）
* プレフィル時の chunk 予約と KV の headroom の兼ね合い
* 正本（pinned）の確保方式 — `hipHostRegister` か `hipHostMalloc` か、ラージページの要否
  （2026-11-01: **重み**で `hipHostMalloc(mapped)` の利用を開始。KV はまだ device のみ。
  協調推論計画 フェーズ 3 の pinned 論点と共通）

## 5. 検証（2026-10-30 実装分）

* `unit.placement`（GPU 不要・cpu プリセット）: KV の式（9B = 32 KiB/token）、
  収まる/収まらない、容量不明で拒否しない、PP は最大ステージで判定、headroom、
  ステージ数の不一致、既定 = 全 device 常駐、pinned への落とし方
* 予算不足の**原因付き拒否**（実測・R9700）: 27B-FP8 に `--arena-mib 8000` を付けた実行が
  `pipeline: budget: 必要 37.17 GiB / 使える 30.19 GiB [不足] device: 最大ステージ … を
  6.98 GiB 超過` で**確保前に exit 1**
* 正常系の対表: 9B-FP8 の PP=2 で `必要 6.20 GiB / 使える 30.19 GiB [ok]`、
  ステージ毎の計上（6040 / 6041 MiB）と実測の空き減（約 6.0 GiB）が一致
* 起動受理の機能表（D6）: `unit.runtime_boot` の `test_features`（要求充足・不足・
  任意機能・後方互換）。アプリは `app::boot` の 1 箇所で要求を宣言する
