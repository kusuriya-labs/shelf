# VRAM 予算から実配置への接続（T14・第 1 段）

日付: 2026-11-01
対象: `placement::place_fitting_budget` → 実行器の実確保 → 転送・再配置。
環境: R9700 (gfx1201) / HIP 7.15 / `build/hip-rel` / 9B-FP8。

---

## 1. 実装したもの

### 1.1 予算 → 重みの置き場

* `PipelineOptions` に `offload_weights_bytes` / `allow_host_weights` を追加。
  `make_pipeline` がグラフの重み宣言順に候補（checkpoint 名・バイト数）を作り、
  **`place_fitting_budget` に「使える device 量」を渡して** device / pinned host を決める。
  * 明示指定: `usable = 重み合計 − offload_weights_bytes`
  * 自動（予算超過時）: `usable = 実空き（または --vram-budget-mib）− 状態 − アリーナ`
  * 状態とアリーナは device に残す前提で、先に差し引く。
* 結果は `upload_plan` の `host_flags` として実行器へ渡す。

### 1.2 実行器の pinned 正本（mapped host）

* `HipStepExecutor::set_weight_placement(true)` の次の upload は
  `hipHostMalloc(..., hipHostMallocMapped)` に確保し、`hipHostGetDevicePointer` の
  **マップ済み device ポインタ**をカーネルへ渡す（`k_pool`/`weight_*` と同じ経路）。
  host 側は**正本**として保持する（`host_weights_`）。
* 解放は `hipHostFree`。`hipFree` と混ざらないよう `is_pinned_dev` で振り分ける。
  ついでに `weight_mxfp4_` の解放漏れも直した。
* `device_bytes()` は host 分を計上しない。`host_weight_bytes()` を別に公開。

### 1.3 転送 + 再配置（promote）

* `promote_host_weights()` が pinned 正本を `hipMemcpyAsync` で VRAM へ移し、
  重みマップのポインタを差し替える（bf16 / fp8 codes / mxfp4 codes）。
  host 側は正本として残す（降格は未実装）。
* `app::pipeline_promote` と `shelf-run --promote-after N` を追加（N ステップ後に実行）。

### 1.4 online KV（pinned 直読み + VRAM ミラー）

* `HipStepExecutorOptions::kv_pinned_host` を追加。KV の K/V プールを
  `hipHostMalloc(mapped)` に確保し、attention / kv_append にはマップ済み
  device ポインタを渡す（VRAM は使わない）。
* `reset_states` は pinned の場合だけストリームを待ってホスト側 `memset` で 0 にする
  （`hipMemsetAsync` はホストメモリに使わない）。
* 予算からの選択: `--allow-host-kv` のとき、`KV + その他の状態 + アリーナ` が
  `--vram-budget-mib`（実空き）を超える場合に自動で host へ置く。強制は `--kv-in-host`。
* 重みの offload と組み合わせると、VRAM 304 MiB でも 9B を起動・生成できた
  （weights 11.8 GiB + KV 128 MiB を pinned host、device 304 MiB）。

### 1.5 online KV の VRAM ミラー（T14 の続き・2026-11-01）

pinned 正本のうち**直近 N ページ**を VRAM に複製し、attention は窓内のページを
VRAM から、それより古いページを pinned host から読む。

* リング: スロット `s` の論理ページ `p` は VRAM の `(s * N + p % N)` ページに置く。
  append 専用なので、窓内のページはそのスロットの最新書き込みと一致する。
* **複数スロット（最大 4）対応**: リング先頭は `slot * N`、窓はスロットごとの
  `slot_window_end` で決める。`reset_slot` / `reset_states` はミラーも 0 クリアし、
  スロット再利用で前の要求の K/V が残らないようにする。
* 窓の判定は**スロットごとの書き込みページ数**（`slot_window_end[slot]`）で行う。
  executor が行の位置からスロット最大を計算して device 配列に載せる。
  行ごとの `visible` で判定すると、prefill の若い行が「全ページ窓内」と誤認し、
  後続ページに上書きされたリングを読む（実測で不一致）。
* `kv_append` は host 正本と VRAM ミラーの両方へ書く。prefix cache（T16）で
  共有したページは kv_append を通らないので、`assign_kv_blocks` がホストから
  ミラーへ複製する（prefix ヒット後も窓内のページが正しい値になる）。
* `N` は
  `--online-kv-pages`（スロットあたりのページ数。KV ブロック 16 トークンなので
  N=32 で 512 トークン）。
* メモリ: 9B（kv_heads=2, head_dim=256, 40 層）で N=32・4 スロットのとき
  約 164 MiB。VRAM に余裕があるときにだけ使う。

**検証**:
* 300 トークン（19 ページ）・N=16 で、host のみと **logits 完全一致**。
* 1024 トークン（64 ページ）・N=32 で、decode 1 ステップ 22.28 → 21.89 ms
  （約 1.8% 改善）。attention はステップ全体の一部なので効果は小さい。
* 単体: `unit.hip_attention` の `kv_append` ケースで、host とミラーの値が一致。

**複数スロットの検証**:
* `unit.hip_step_executor`: 4 スロット × 40 ステップ（ページ境界を跨ぐ）で、
  R=4 / R=1 のどちらも host のみと **logits 完全一致（worst 0.0）**。
* `check_http_concurrency.py --kv-in-host --online-kv-pages {1,64}`（同時 4 要求）が
  sequential と一致。スロット再利用（同時 → 逐次）も同じプロセスで確認。
* これにより `max_sequences > 1` でも online KV を有効化した（無効化の警告は削除）。

---

## 2. 検証（9B-FP8）

### 2.1 明示 offload（512 MiB 指定）

* 実際に落ちたのは **1 テンソル 1940 MB**（テンソル単位のため指定量を超える）。
* 起動ログ: `stage 0: offload 1 tensors / 1940 MB -> pinned host`、
  `device 10144 MiB, host 1940 MiB`（通常は device 12100 MiB）。
* decode: 通常 21.0 ms/step → offload 時 88〜91 ms/step。
  `--promote-after 2` の後は **21.0 ms/step に戻る**（`promoted: 1940 MiB`）。
* 生成列は offload なし・あり・promote 後で**完全一致**
  （`[..., 16,10,16,173273,2972]`）。

### 2.2 予算指定（自動配置）

`--vram-budget-mib 11000 --allow-host-weights`:

* 予算判定は `必要 12.40 GiB / 使える 10.21 GiB [不足]` で失敗するが、
  `allow_host_weights` により続行。
* 同じ 1940 MB を pinned host へ落とし、device 10144 MiB（< 11 GiB）で起動。
  生成列は通常時と一致。

### 2.4 KV の pinned host

| 条件 | device | host KV | 生成列 | 時間 |
| --- | --- | --- | --- | --- |
| 通常（max_context 4096） | 12227 MiB | 0 | `[16,10,16]` | 420.7 ms（prefill+4 decode） |
| `--kv-in-host` | 12098 MiB | 128 MiB | `[16,10,16]` | 422.5 ms |
| 2048 トークン文脈・通常 | — | 0 | `[9707,9707]` | 1182.5 ms |
| 2048 トークン文脈・host KV | — | 128 MiB | `[9707,9707]` | 1231.5 ms（+4.1%） |
| 予算 300 MiB + offload + host KV | 304 MiB | 128 MiB | `[16]`（通常と一致） | 起動・生成とも成功 |

* 短い文脈では差はほぼ無い（KV の読は使ったページだけ）。2048 トークンでも +4.1%。
* HTTP E2E（`serve_e2e.py --kv-in-host`）も全項目 PASS。

### 2.3 JSONL / HTTP の E2E

* `check_generation_parity.py` に `offload=` を追加。offload 0 / 512 MiB で
  同一プロセス 2 回ずつ、生成列が一致（PASS）。
  時間は 421 ms → 1765 ms（4 トークン、約 4.2 倍）。
* `serve_e2e.py --offload-weights-mib 512` で HTTP の全項目 PASS
  （通常応答・SSE・EOS・400/404/415・切断キャンセル）。

---

## 3. 残り（T14 の続き）

| 項目 | 状態 | 次の実装 |
| --- | --- | --- |
| 重みの pinned 正本 | **実装済み** | テンソル単位 → 層/断片単位の粒度 |
| 転送（promote） | **実装済み** | 降格（demote）と自動トリガ（空き VRAM・リクエスト境界） |
| 再配置 | **実装済み（手動）** | リクエスト間の自動再配置ポリシー |
| online KV | **スロット単位で実装済み**（§1.5。最大 4 スロットで host と完全一致） | VRAM が足りないときのミラー縮小（N の自動調整） |
| KV の admission | **既存の受付で足りる（整理）** | 容量はスロット × 最大文脈で固定（pinned 正本）。オンラインはキャッシュであり容量を増やさないため、新しい admission は不要。per-request の `prompt + max_new <= max_context` と T15 のスロット待ちが 2 段の役割を果たす。将来 KV を動的に追い出すなら、この判断を再検討する |

### 既知の制約

* mapped host の重みは PCIe 越しの読みになるため遅い（実測 4.2 倍）。
  常用ではなく「VRAM に載らないモデルを起動する」ための退避経路。
* 粒度はテンソル単位。指定 512 MiB に対して 1940 MB 落ちるのはこのため。
* 予算の headroom は重み配置では 0（状態・アリーナを先に差し引いているため）。
  `plan_budget` の判定は従来どおり 5%。
