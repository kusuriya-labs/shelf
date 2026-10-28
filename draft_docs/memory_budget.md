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
  ② 実行は active 文脈の合計 ≤ 81,920 tok |

## 2. 予算の算出と決定者（未着手 — D2 の実装時に書く）

* **算出は純粋関数**: `placement/` が GPU 非依存で必要量を積み、不足を名前付きで返す
  （radiance「推測禁止、既知量の算術」／ paiton の `fits` 判定を踏襲）。
  `MemoryKind` / `BufferView`（`backend/common`）で所在を表し、実確保のみ backend が行う
* **入力**: 空き VRAM / 空き pinned の取得方法、headroom の既定値、
  GDN 固定費の扱い（`engine_contract` §8.7 が「予算に未計上」と指摘している点）
* **出力と失敗**: `Status::insufficient_memory` を投げるタイミングと文言
  （`serve_protocol.md` §3 では `overloaded` / 503 に写像される）

## 3. 段 2 の実行側の選択肢（`engine_contract.md` §8.6 で (a) に確定）

* (a) admission を絞る / (b) pinned 直読み / (c) 層単位受け持ち
* 「VRAM に無い active block をどう読むか」が性能を決める（同 §8.3 の試算:
  全量移すと約 2.5 GiB / トークン ≈ 100 ms/トークン）。**その試算の再計測はフェーズ 0**

## 4. 未決（本書で拾う）

* **tier 移動の発動体** — いつ pinned ↔ VRAM を動かすか（`engine_contract` §6 / §8.5 の残件）
* 思考打ち切り（reasoning cap）の判定主体（`engine_contract` §8.4）
* プレフィル時の chunk 予約と KV の headroom の兼ね合い
* 正本（pinned）の確保方式 — `hipHostRegister` か `hipHostMalloc` か、ラージページの要否
  （現状 shelf に `hipHostMalloc` の利用はゼロ。協調推論計画 フェーズ 3 の pinned 論点と共通）

## 5. 検証（実装時に書く）

* 予算不足がエンジン起動前・受付時に**原因付き**で拒否されるテスト（`directory_design` D1/D6）
* 計測: 予算値と実測（`cudaMemGetInfo` 相当）の対表 — `docs/benchmark/`（実装後に正式化）
