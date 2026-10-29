# shelf-validate 仕様（草案）

状態: 草案 v1（V0 は実装済み・未マージ）。

* **現行仕様は `docs/spec/validate.md`、運用手順は `docs/guide/validate.md` に分離した**
  （実装済みの内容だけを `docs/` に置き、本ドキュメントは仕様・計画（V1 以降・PP 設計・未決事項）を扱う）。
* **approx の量子化形式は FP8 W8A8（2026-10-29 決定）**: 正本 = `models/Qwen3.5-9B`（bf16）、
  製品量子化版 = `models/Qwen3.5-9B-FP8`（FP8）。EXL3 4bit 経路は同日中止（§7「（6）履歴」）
* 関連: `draft_docs/parity_policy.md`（T0〜T3 の判定）、`draft_docs/engine_contract.md`、
`draft_docs/directory_design.md`（§5 D7 検証・§3.5）、`draft_docs/port_strata_cpu_gpu_coop.md`

---

## 0. 目的

実モデルで **無損失経路（exact = bf16・正本）** と **プロダクション量子化経路（approx = FP8 W8A8・製品版）**
を同じ入力で走らせ、
出力トークン列を比較する。`parity_policy.md` の T0（決定性）・T1（完全一致）・T2（誤差由来の判定）
を**実モデル上で実行する道具**が本アプリである。

* 内部での量子化変換は行わない（exact と approx はそれぞれ渡されたディレクトリをそのまま読む）
* 比較の主対象は decode で生成したトークン列（StopToken まで）

---

## 1. CLI

```
shelf-validate \
    --exact-model=<dir>        # bf16（無損失側・正本）  例: ./models/Qwen3.5-9B
    --approx-model=<dir>       # 量子化（FP8 W8A8）     例: ./models/Qwen3.5-9B-FP8
    [--prefill-tokens=<N>]     # prefill の長さ。既定 256（範囲 128..384）
    [--budget=512]             # prefill + decode の合計。既定 512
    [--stop-id=<id>]           # 省略時は config の eos_token_id（後述の解決順）
    [--pipeline-parallel=<K>]  # ステージ数。既定 1（K>1 は PP 実装が前提）
    [--device=hip|cpu] [--gpu=<n>]
    [--runs=<n>]               # 決定性チェックの反復数。既定 2
    [--out=<dir>]              # 結果 JSON とトークン列の保存先
```

* `--exact-model` 単独でも起動可（V0: 決定性 T0 だけを検証する。approx は省略可能）
* 複数の比較軸（§5）を同一アプリで賄うため、**比較は「構成 A vs 構成 B」の組**で指定できる形にする
  （V1 では exact / approx の 2 組を既定に、PP 比較は V2 で追加）

---

## 2. 予算（512 token）

* **`prefill N + decode G ≤ budget（既定 512）`**
* `N ∈ [128, 384]`（prefill ≥ 128 と decode ≥ 128 の両立条件から）。既定 **N = 256** → decode 上限 256
* 起動時の検証: `N < 128` または `budget - N < 128` なら**起動前に拒否**（仕様違反を機械で防ぐ）
* StopToken に達せず `budget` に達したら打ち切り、結果に **`budget_exhausted`** を記録する
  （これは不一致ではないが、比較両者が同じ条件で打ち切られたかは結果に明記する）

---

## 3. 入力（prefill ≥ 128 tok）

* **入力は token ids 列**。shelf にトークナイザを入れない方針を維持する
* 用意するもの:
  1. **固定 ids の fixture**（例: 256 tok）をリポジトリに同梱する
  2. 生成スクリプト（Python、モデル同梱の `tokenizer.json` を使用）を**開発時のみ**実行し、
     成果物の ids をコミットする（CI と実行はスクリプトに依存しない）
* 要件: 同じ ids を exact / approx の**両方にそのまま適用**する（テンプレート適用済みの完成形を置く）
  * **2026-10-29 に確認（FP8 ペア）**: `Qwen3.5-9B` と `Qwen3.5-9B-FP8` の `tokenizer.json` は
    語彙マップが**完全一致**（base 248,044 ＋ 追加トークン、`<|endoftext|>` 248044 / `<|im_end|>` 248046 の id 対応も同一）。
    従って既存の fixture をそのまま approx に適用できる（ファイルサイズは違うが語彙は同一）
* なぜ fixture か: `parity_policy.md` の T0（決定性）を再現可能にし、tokenizer をアプリから切り離すため
* **V0 実装での確定事項**:
  * 成果物は `apps/fixtures/prefill_fixture.hpp`（256 tok）。生成は `apps/fixtures/make_fixture.py`（開発時のみ）
  * テンプレートは **思考モード OFF**（`enable_thinking=false`）。ON だと decode が思考だけで予算を消費し
    StopToken に到達しなかった（実測: 思考 ON で 256 tok 全消費 → `budget_exhausted`）
  * プロンプトは「約 150 語で即答（計画・見出し・前置きなし）」を指示し、decode が 128..256 tok の範囲で
    StopToken に収まるようにした（実測: decode 246 tok で `stop_token`）
  * 長さは 256 tok ちょうどに固定（超過分の末尾切り詰めはアシスタント開始トークンを壊すため禁止）

---

## 4. 実行手順

1. **設定読み**: `config.json` / `generation_config.json` からレイヤ数・幾何・`eos_token_id` を得る
2. **ロード**: exact / approx をそれぞれ読み、§6 の構成（PP）に従って stage 割当する
   （approx は `config.json` の **`quantization_config`** と safetensors ヘッダの dtype
   （`F8_E4M3` の重み ＋ `BF16` の `weight_scale`）を正として、単一 `model.safetensors` を読む。§7 V1 参照）
3. **prefill**: 位置 1..N を一括投入して logits を得る。
   行数上限は `kMaxStepRows = 64` のため **64 行ずつチャンク**で回す
   （既存の `apps/shelf_run.cpp` の `gpu_prefill` と同じ経路。最後のチャンクの最終行の logits を採用）
4. **decode**: 位置 N+1 から逐次生成。各ステップで top-1（greedy）を取り、
   **StopToken なら停止**。StopToken 自体はトークン列に**含めず**、停止理由として記録する
5. **予算検査**: `N + 生成数 ≤ budget`（違反時は打ち切り + `budget_exhausted`）
6. 両経路が終えたら比較（§5）。既存の greedy ループ
   （`apps/shelf_run.cpp:289-309` と同型）を流用する

### StopToken の解決順

1. `--stop-id` 指定時 = その値
2. `generation_config.json` の `eos_token_id`（配列。例: `[248046, 248044]`）→ **この順で最初に現れた方を採用**（両方を停止条件として扱う）
3. `config.json` の `eos_token_id`（shelf の `TextConfig` が既に読んでいる値）
4. いずれも無い場合はエラー

---

## 5. 比較と判定

| 軸 | 内容 | 対応する parity_policy |
| --- | --- | --- |
| **T0 決定性** | 同一構成で `--runs=2` 回実行し、トークン列が一致するか | T0 |
| **T1 完全一致**（主） | exact と approx の decode トークン列が StopToken まで一致するか | T1 |
| **T2 誤差判定** | 不一致のとき、**最初の不一致位置**での exact 側の top-1 / top-2 の margin と、不一致位置の logits 差を計算し、`parity_policy.md` §3 の dtype 誤差実測と比較 | T2 |
| **構成比較**（V2 以降） | `exact(PP=1) vs exact(PP=K)` — PP は数値を変えないはずなので T1 で検証 | T1 の応用 |

**出力（`--out`）**: JSON に以下を記録する

* 構成（モデル・dtype 経路・PP・デバイス・N・budget）
* 両経路のトークン列（prefill 後の decode 分のみ）
* 判定: `identical` / `diverged`（不一致位置 k、k での margin、top-1/2 の値）/ `budget_exhausted`
* ステップ時間（prefill / decode の ms）、VRAM 常駐量、経過時間

**終了コード**: `0` = 合格（T0 かつ T1 一致）、`1` = 条件付き合格（T2 判定で丸め由来）、`2` = 不合格。

---

## 6. pipeline-parallel（比較の前提となる実行形態）

* **要る理由**: 27B 級（bf16 で約 54 GB）は 1 GPU に入らない。例: `--exact-model=./Qwen3.8-27B --pipeline-parallel=4`
  （9B bf16 の 19.3 GB は 1 GPU に載るため、**PP の実績は 27B 級の取得後**。27B は現状未取得）
* **分割**: レイヤ境界で K ステージに分割（Hybrid の GDN 層があってもレイヤ単位で同一）
  - stage `s` = `[layer_s, layer_{s+1})` を担当し、decode 1 token が K ステージを順に通る
  - 各 stage の重みは**担当デバイスに部分ロード**。approx（FP8）は単一 `model.safetensors` なので
    レイヤ単位ファイルは無いが、テンソルの byte range は safetensors ヘッダから取れ、
    量子化 / 非量子化の別はヘッダの dtype（`F8_E4M3` か `BF16` か）と `weight_scale` の有無で判るため、
    レイヤ範囲の部分ロードは成立する
  - **KV はレイヤを持つ stage 側に置く**（レイヤ分割なので KV もレイヤに従って分散する）
  - stage 間の活性化受け渡しと同期の方式 → **参照調査の報告を待って §6.1 に追記する**
* **正しさの担保**: PP は数値を変えないはずなので、`exact(PP=1) vs exact(PP=K)` の T1 一致を必須とする
* **依存**: PP 実装が shelf-validate の V2 の前提。**PP の実装は本アプリより先**（または並行で先行着手）

### 6.1 参照実装からの設計（調査済み）

**どの参照が PP を持つか**

| 参照 | PP | 区別 |
| --- | --- | --- |
| **Strata**（`docs/MULTI_GPU.md` の layer split） | **本命・本番品質** | 「This is pipeline parallelism, not tensor parallelism.」（`MULTI_GPU.md:100-102`）。stage 上限 8 |
| **phaseshift-fp8** の bench（`src/apps/bench/qwen4exp.hip` の `--pp 1\|4`） | **第 2 参照（小型）** | 均等レイヤ分割・pinned+mapped・オーバーラップ無しの逐次。serve/compute 経路は TP のみで PP は将来項目 |
| radiance | 無し | `spec.md:51` で PP を対象外と明記（マルチ GPU は TP のみ） |
| paiton / vllm-sly-radiance / EAGLE-4 | 無し（むしろ拒否） | paiton は全モデルが `pipeline_parallel_size != 1` を raise、EAGLE-4 は `NotImplementedError` |

**Strata の設計（借りる要点）**

* **分割**: 連続レイヤ範囲、`--layer-split K[,K2..]|auto`。**K は「後段 GPU の開始レイヤ番号」であり個数ではない**
  （昇順・K≥2 が検証され、違反は exit 2）。stage 最小 3 レイヤ
* **hand-off**: 境界ごとに **mapped + pinned ホストメモリ**（`cudaHostAlloc(Mapped|Portable)`）へ residual を書き、
  **既定は直列**（stage0 の `run()` が次を呼び、境界で自ストリームを sync）。**重畳は opt-in**
  （`STRATA_SPLIT_OVERLAP=1` で ready フラグを GPU 上 publish → 読み側が `wait_flag_ge` → ホストが stage 間で sync しない）。
  有界待機 `STRATA_SPLIT_WAIT_MS`（既定 30000 ms）で超過時はウィンドウ破棄
* **KV**: stage は「自分のレイヤ範囲 × コンテキスト全長」だけを持つ（`SessionState` の `layer_lo/layer_hi` carve）。
  **配決定前に純粋関数で概算**（`session_bytes(lo,hi)`）→ 検索 → その結果で allocate する順序
* **重み**: 共有ファイルのまま、範囲外テンソルを skip 集合へ（既定は各カードにフルコピー、部分ロードは opt-in `--trim-stage-weights`）
* **head / MTP レイヤは最終 stage**。prefill は chunk パイプライン（pinned 2 枚の double buffer + 別スレッドで次 stage）
* **bit-exact な検証手順**: `--split-device 0`（1 GPU 上に 2 stage）で 1 GPU 実行と **10/10 バイト一致**。
  GPU 間分割では CPU 計算分の丸め差により完全一致しない（`bench/results/2026-09-29-layer-split/README.md:20,25`）
* `auto` 検索（空き VRAM + レイヤ毎速度 + PCIe プローブ → 全組合せ列挙）は **expert profile に依存するコストモデルが Strata 固有**。
  shelf は最初 **K を明示する手動指定のみ**で良い

**phaseshift-fp8 bench の形（shelf に最も近い第 2 参照）**

* `per_stage = num_hidden_layers / pp` の均等分割、stage `si` は `[si*per_stage, (si+1)*per_stage)`、
  **stage 毎のデバイス**を `stage_begin / per_stage` で決定
* stage 間は `hipHostMalloc(Mapped|Portable)` + `hipHostGetDevicePointer` の **pinned, mapped, no copy**、
  **オーバーラップ無しの逐次**（`run_stages` が for で回す）
* **重みの部分ロードは `load_opts.layer_begin/end`、lowering も `lower_opts.stage_begin/end`**。
  `is_first` / `is_final` で embedding と head の担当を判定
* **`StageState` が stage 毎に `PagedKVPool` / `GdnStatePool` / `SequenceSlotPool` を持つ**
  → shelf の「per-layer に state を宣言する」構造と直交しており、そのまま踏襲できる

**shelf への着地点（本アプリ V2 でやること）**

1. `build_step_graph` に **レイヤ範囲（layer_lo / layer_hi）の引数**を追加。
   `src/model/qwen35/step_graph.cpp:39-40` の無条件全レイヤ列挙を範囲化すれば、
   `StepGraphPlan::weights` / `states` は**自然に stage 分へ絞られる**（state は per-layer 宣言なので KV・
   conv・recurrent すべてが該当レイヤ分のみ）
2. `apps/shelf_run.cpp` の 1 本の `HipStepExecutor` を **`std::vector`（1 stage = 1 device）**へ。
   stage 間は mapped pinned ホストバッファに residual を書いて次へ（Strata 既定の**直列 hand-off** から始める）
3. `fetch_logits` は**最終 stage のみ**（head は最終 stage。Strata / phaseshift 共通）
4. 検証ゲート: **1 GPU + 2 stage の構成が 1 GPU 実行と一致**（Strata の `--split-device 0` と同型）。
   shelf-validate の `exact(PP=1) vs exact(PP=K)` がこれに相当
5. 実装順（Strata の既定と整合）: ① 直列 hand-off → ② prefill の chunk パイプライン → ③ flag による重畳

**写さない部分**: Strata の `Verifier` / `SessionState` / `Drive` / `Prefill`（doorbell・グラフキャプチャ・
expert pool / hc に密結合）、hand-off の `hc × n_embd` サイズ（shelf は `n_embd` × 行数相当で足りる）、
`auto` 検索のコストモデル、MTP の stage 移動と `--pipeline-windows` の rollback（spec decode と絡むため後回し）

---

## 7. 実装フェーズ（着手順）

| # | 内容 | 完了条件（DoD） |
| --- | --- | --- |
| **V0** | アプリ骨格: CLI（§1）・予算検査（§2）・fixture 入力・prefill（64 行チャンク）・greedy decode・StopToken 解決・比較と JSON 出力。`--exact-model` のみで運用 | `Qwen3.5-9B` で `--runs=2` の T0 が通り、終了コードと JSON が仕様どおり。入力 N=256 / budget=512 の検査も通る（**2026-10-28 に 9B で達成**: T0 identical / stop_token 230 tok / exit 0。2026-10-27 の旧 4B での達成は履歴） |
| **V1** | approx（**FP8 W8A8**）経路の読み込みと実行 → T1/T2 比較。前提と着手順は下記「V1 の前提となる実装」 | exact と approx のトークン列比較が出力され、diverged 時に margin が出る |
| **V2** | pipeline-parallel（`--pipeline-parallel=K`） | `exact(PP=1)` と `exact(PP=K)` のトークン列が一致（T1） |
| **V3** | 自動化: `ctest` のラベル（例 `e2e`）で回し、モデル無しの環境では skip | CI 相当の流れで 1 件が守られる |

### V0 実績と実装メモ（2026-10-27）

* **2026-10-28 の資材更新と開発モデルの選定**: `models/` から `Qwen3.5-4B` と `Qwen3.8-*` が削除され、
  在庫は `Qwen3.5-9B`（bf16）と `Qwen3.5-9B-exl3`（EXL3 4bit）の 2 組になった。
  **この 2 つを開発時点の標準モデルに選定**した（統一した記載は `draft_docs/directory_design.md` §1.0）。
  以下の数値は **旧 4B での実測（履歴）**。以後の DoD・計測は **9B を対象**とし、
  未測定の値は「未測定」と書く（config は `text_config` の必須キー全充足を確認済み）。
* **2026-10-29 の資材更新（現行）**: `Qwen3.5-9B-exl3` は在庫から除外され、
  在庫は `Qwen3.5-9B`（bf16・正本）と `Qwen3.5-9B-FP8`（FP8 W8A8・製品量子化版）の 2 組。
  **V1 以降はこの 9B ペアで行う**（EXL3 経路の記録は下記の「（6）履歴」に残す）

* **DoD 達成**: `--exact-model=models/Qwen3.5-4B --runs=2 --device=hip --gpu=1` で T0 `identical`・exit 0。
  実測: prefill 256 tok（64 行 × 4 チャンク、各約 130 ms）/ decode 246 tok で StopToken 停止（各約 5.0 s）、
  VRAM 常駐 10.1 GB、全体約 19 s。範囲外入力（`--prefill-tokens=64`、`--budget=300` 等）は起動前に exit 2 で拒否。
* **arena の自動サイズ**: rows=64 prefill グラフは出力総和が大きく、従来の既定
  `arena_bytes = 64 MB` では logits `[64, vocab]` f32（約 64 MB）だけで枯渇する。
  検出後、「64 MB + 2×Σ 全 primitive 出力(f32)（一時バッファ余白）」の式を
  **executor 側の自動算出**として library に移した（`arena_bytes = 0` 既定、
  `declare_states` のグラフから算出）。アプリは行数の大きい prefill グラフ (rows=64) を
  `declare_states` に渡す（rows=1 の decode グラフを渡すと不足）。4B では約 1.96 GB。
* **executor へのガード追加**（`src/backend/hip/step_executor.cpp`）: アリーナ枯渇時に null 出力のまま
  カーネルを発射するとデバイスフォルト（GPU ハング）になるため、全 alloc 点で発射前に検査するようにした。
* **CPU 側の対応**: `HostStorage::reset_states()` を追加（`HipStepExecutor::reset_states()` と対になる
  ステートと使用行のリセット。`--runs=n` の反復で必要）。
* `--device=cpu` の速度は **9B / Release で差分計測済み（2026-10-28）**: prefill **4.90 s/token**、
  decode **3.68 s/token**（ロード約 23 s は別途。既定入力で 1 run 約 35 分）。
  旧 4B の Debug 実測 16 s/token は履歴。**全トークンでの CPU 比較（T1）は未実施**で、
  §8「CPU 経路の扱い」の未決に同じ

### V0 実績（開発モデル 9B・2026-10-28・修正後）

`shelf-validate --exact-model=/opt/zen/work/models/Qwen3.5-9B --runs=2 --device=hip --gpu=0 --out=<dir>`

* **DoD 達成**: `t0 = identical`、`stop_reason = stop_token`（**230 tok**）、**exit 0**。
  範囲外入力の拒否（exit 2）は従来どおり
* **実測**（R9700 / gfx1201・**Release ビルド（hip-rel）**、2026-10-28）:
  * prefill 256 tok = 64 行 × 4 チャンク（196.8〜203.1 ms、run ごとに **831.8 / 836.5 ms**）
  * decode 230 tok = **7,121 / 7,126 ms** → **30.96 ms/tok（≒ 32.3 tok/s）**
  * **arena 自動算出 2,299 MiB**（rows=64 グラフ、表示は "2299 MB"）、
    `device_mem_used_bytes` = **20.56 GB**（`free_before − free_after`＝重み ＋ KV 状態 ＋ arena。
    修正で `lm_head`（2.03 GB）がロード対象に加わった分を含む）
  * ロード 9.1 s / 全体 25.0 s
* **参考（ビルドとモデルが両方異なるので単純比較はしない）**: 旧 4B（Debug・履歴）は
  prefill 130 ms/chunk・decode 20.4 ms/tok。9B（Release）は 202 ms/chunk・31.0 ms/tok
* **StopToken**: 9B は `generation_config.json` を持たず、`resolve_stop` の手順 3
  （`text_config` の `eos_token_id = 248044`）で解決。tokenizer の eos 248046 は停止条件に
  入らないが、実走で `stop_token` 停止を確認済み

### 発見したバグと修正（2026-10-28・HF との A/B で発覚）

* **症状**: 同一 fixture で 9B を実行すると出力が劣化（中国語・ロシア語・同一トークン 55 連打）、
  256 tok 全消費で `budget_exhausted`。**T0 は `identical` で合格していた**
* **切り分け**: HF / `transformers` と同一 ids で比較して **48 tok 中 0 一致**。
  prefill 直後 logits の 3 者比較で **CPU ≈ GPU（cos 0.9999）だが両方とも HF と不一致** → カーネルでなくグラフ側。
 さらに `cos(shelf logits, h@embed) = 0.9997`（argmax 一致）／`cos(shelf logits, h@lm_head) = -0.046` →
  **出力投影が埋め込み表を束ねていた**（hidden は HF と一致 = 前方は正常）
* **原因**: `src/model/qwen35/step_graph.cpp` の `add_final()` が `tie_word_embeddings` を見ずに
  `embed_id_` を出力に使用。9B は untied（`lm_head.weight` が別在）で、埋め込みとの内積がロジットになっていた
* **修正**: `tie_word_embeddings == false` なら `lm_head.weight`（`[vocab, hidden]`）を束縛。
  回帰テスト `tests/unit/test_step_graph.cpp`（untied / tied の両ケース）を追加し
  `unit.step_graph` として登録（cpu 13 件 / hip 19 件が全通過）
* **修正後の確認**: prefix = 1 / 8 / 32 のいずれでも HF と **8/8 トークン一致・文字列一致**
* **教訓**: **T0（決定性）は出力の妥当性を保証しない**。上記バグは T0 を通っていた。
  実装の妥当性は外部参照（HF）とのトークン列比較で確認する（`parity_policy.md` に追記）

### V1 の前提となる実装（別作業）— **FP8 に確定（2026-10-29）**

> 製品量子化版を **FP8（W8A8・活性化は dynamic）** に決めた。exact = `models/Qwen3.5-9B`（bf16・正本）、
> approx = `models/Qwen3.5-9B-FP8`（FP8）で V1 を行う。EXL3 4bit 経路は 2026-10-29 に中止
> （下記「（6）履歴」）。決め手: ① checkpoint が既にある（`models/Qwen3.5-9B-FP8`、ディスク 14.01 GB）
> ② 正本と**同一の safetensors 形式・同一のテンソル名**で、量子化されていないテンソルは bf16 のまま
> 同梱されている（差分が量子化された Linear だけなので比較が素直）③ 量子化の記述が `config.json` の
> `quantization_config` に集約され、ヘッダの dtype でテンソル単位に判別できる（表駆動）
> ④ shelf 既存の fp8 資産（`e4m3_codec`・paged attention の fp8 版）と語彙が揃う
> ⑤ 参照が公式まわりに多い（`transformers` の fp8 linear / compressed-tensors の float-quantized 仕様 /
> phaseshift-fp8 の fp8 経路）。
> **代償として容量効率は落ちる**（EXL3 4bit 7.19 GB → FP8 14.01 GB）。低 VRAM 協調推論の本命量子化は
> 未決定のまま（`draft_docs/directory_design.md` D5 / §8 の未決）

**（1）approx（FP8）の読み込み**

* 対象: `models/Qwen3.5-9B-FP8`
  * ファイル: 単一 `model.safetensors`（13.52 GB・**888 テンソル**）＋ `model_mtp.safetensors`
    （0.49 GB・15 テンソル）。`model.safetensors.index.json`（`total_size` 14.01 GB）で対応付ける
  * `config.json` の **`quantization_config`**（EXL3 のような別ファイルではない）:
    `quant_method: compressed-tensors`、`format: float-quantized`、`version: 0.14.1.a20260325`、
    `targets: [Linear]`、`ignore` 232 件、`kv_cache_scheme: null`（**KV は量子化されていない**）
  * 量子化パラメータ（`config_groups.group_0`）
    * 重み: `type: float` / `num_bits: 8` / `strategy: channel`（出力チャネル単位）/
      `symmetric: true` / `dynamic: false`（静的・スケール格納済み、observer `memoryless_minmax`）
    * 活性化: `type: float` / `num_bits: 8` / `strategy: token`（トークン単位）/
      `symmetric: true` / `dynamic: true`（**実行時に計算**、`input_scale` テンソルは存在しない）
* **正は safetensors ヘッダの dtype とスケール名の対応**: 量子化 Linear は `*.weight` が `F8_E4M3`、
  同じ基底名の `*.weight_scale` が **`BF16`** で shape `[out, 1]`（**128 テンソルずつ**・`input_scale` は 0 件）。
  テンソル単位で量子化の有無が判る（推測判定は不要）
* 量子化された Linear（`F8_E4M3`・5.30 GB）: `mlp.{gate,up,down}_proj` × 32 層（96）
  ＋ `self_attn.{q,k,v,o}_proj` × `full_attention` 8 層（32）= **128**
* 量子化されないテンソル（`BF16` のまま）: `lm_head`（2.03 GB）/ `embed_tokens`（2.03 GB）/
  各 layernorm と `self_attn.{q,k}_norm` / **`linear_attn.*` すべて**（`out_proj` /
  `in_proj_{qkv,z,b,a}` — ignore に `re:.*linear_attn.*`）/ `visual.*` / `mtp.*`（ignore に `re:^mtp.*`）
  → **同一層内でも dtype が混在する**（Hybrid の `linear_attn` 側は bf16、`self_attn` 側は FP8）
* shelf が実際に読む payload（`model.language_model.*` ＋ `lm_head`）は **12.61 GB**
  （language_model 10.57 GB = bf16 5.27 GB ＋ fp8 5.30 GB、lm_head 2.03 GB）。
  `visual` 0.91 GB と `mtp` 0.49 GB は現行の実行経路では対象外
* **未解決の前提（着手前に要決定）**: 現行 `main` の safetensors リーダは
  `F64 / F32 / F16 / BF16 / I64 / I32 / U8` のみ対応（`src/artifact/safetensors/reader.cpp` の
  `map_dtype`）で、`F8_E4M3` は `unsupported dtype` となり **FP8 checkpoint を開けない**。
  * 採る方針: **`DataType::fp8_e4m3` を追加**（1 バイト・符号ありのメタとして）。
    責務の切り分けは EXL3 時に確定した考え方維持 — **DataType は「バイト長・符号性のメタ」、
    意味論（スケール適用・ペイロード解釈）は `src/quant` の QuantSpec の役目**
    （`src/quant` は model / backend / runtime / hip を include しない）
  * **`dtype.hpp:24-27` の注記と矛盾しない**: 「量子化ペイロードを enum に載せない」は
    trellis のような **opaque な符号列**（`unknown` + QuantSpec）についての話で、
    e4m3 は **実在する 1 バイトの浮動小数型**。載せるのは型であり、
    スケール（`weight_scale` 適用）の意味論は引き続き `quant/` の QuantSpec 側
  * `u8` を流用しない（KV 側の fp8 ペイロードとは別物で、誤用と境界テストの緩みを防ぐ）
  * 実装範囲の目安: enum + `element_size`（1）+ `map_dtype("F8_E4M3")` +
    `unit.foundation`（size / 浮動小数点判定）と `unit.artifact_safetensors`（F8_E4M3 テンソルの読出・バイト一致）
  * `weight_scale` が **BF16**（FP32 ではない）である点はスケーリング実装で明示する（読み間違い防止）

**（2）FP8 GEMM（ホットパス dequant）** — shelf には現状、重みの量子化経路が無い
（bf16 のみ、fp8 は KV 側にのみ存在）。契約（`parity_policy` / 量子化契約）どおり **dequant はカーネル内**。

* checkpoint の定義から決まる最小構成（4 段）:
  1. **入力**: 行（トークン）ごとに max から `a_scale[row]` を計算 → e4m3 に量子化（**dynamic**）
  2. **重み**: `F8_E4M3` をそのままオペランドに使う（**ロード時に bf16 へ展開しない**）
  3. **GEMM**: `A8 × W8 → C32`（fp8 入力・**f32 累積**）
  4. **出力**: `out = C32 × a_scale[row] × weight_scale[col]` → 出力型
  * **順序の要点**: `weight_scale` は列（出力チャネル）ごとに異なるので**累積の後**に掛ける。
    `a_scale` は行内で一定なので可換。per-channel スケールは K ループへ折り込まない
    （`Σ_k a_k w_k` をスケール付きで割る形にすると丸めが増える）
* **未確認の前提（実測前に決めない）**: gfx1201（R9700）に **FP8 e4m3 のマトリクス命令が
  WMMA/MFMA として実在するか**。実在すれば (3) を fp8 GEMM、無ければ
  「**カーネル内で** e4m3 → bf16/fp16 に dequant して bf16 GEMM」（契約は守られる）にフォールバックする。
  分岐の決定は §8 の未決（ミニベンチで確認してから決める）
* 精度の見込み（実測で確定させる）: e4m3 は仮数 3 ビットで相対分解能が粗いが、
  重みは per-channel・活性化は per-token の動的スケールで範囲を寄せるため、
  4bit 程には崩れない可能性が高い。**T1 が通るかは見込みではなく実測**（`parity_policy.md` §3）
* **参照（移植ではなく定義の突合に使う）**: `transformers` の fp8 linear
  （`W ≈ weight.float() × weight_scale.float()` の定義）、compressed-tensors の `float-quantized` 仕様、
  `references/` の phaseshift-fp8（fp8 の GEMM / KV の語彙とスケーールの置き方）

**（3）注意（従来どおり）**: 「ロード時に bf16 へ展開してから既存の bf16 GEMM で計算する」のは
**契約違反**（有損 dequant をロード時に置かない）。**カーネル内で**デコードして fp8 / bf16 の
オペランドにすることは許される。暫定で外に出す場合は契約上の位置づけを明記する。

**（4）V1 の着手順（可否の実測 → 仕様確定 → 本体）**

1. **スモーク**: `models/Qwen3.5-9B-FP8` の 1 テンソル（例
   `model.language_model.layers.0.mlp.down_proj` の `.weight`（`F8_E4M3`、`[4096,12288]`）と
   `.weight_scale`（`BF16`、`[4096,1]`））を CPU 参照で dequant し、
   torch / `transformers` の `weight.float() × weight_scale.float()` と**一致する**ことを確認する
   （ここで「材料が読める／`F8_E4M3` の解釈が合っている」が確定する）
2. **GEMM 方針の決定**: gfx1201 の FP8 マトリクス命令の実在をミニベンチで確認し、
   fp8 GEMM と in-kernel dequant + bf16 GEMM のどちらかを**実測で**決める
3. **V1 本体**: リーダ（`F8_E4M3` と `weight_scale`）→ 活性化量子化（per-token）→ GEMM →
   T1/T2 → 終了コード 0/1/2
4. **T2 の基準取り**: `parity_policy.md` §3 の手順で **FP8 W8A8** の誤差規模を測る（§8 参照）

**（5）正本と製品版の既知の差分（比較の前提）**

| 項目 | 正本 `Qwen3.5-9B`（bf16） | 製品版 `Qwen3.5-9B-FP8`（FP8 W8A8） |
| --- | --- | --- |
| ディスク（safetensors の payload 合計） | 19.31 GB（4 シャード） | 14.01 GB（単一 13.52 GB ＋ `mtp` 0.49 GB） |
| shelf が読む範囲（`language_model.*` ＋ `lm_head`） | 17.91 GB（実測・safetensors ヘッダ集計） | 12.61 GB（bf16 5.27 GB ＋ `F8_E4M3` 5.30 GB） |
| 量子化された Linear | なし（bf16） | 128 テンソル（`F8_E4M3` ＋ `weight_scale` BF16） |
| `lm_head` / `embed_tokens` | bf16 | bf16（量子化されない・shape 同一） |
| `linear_attn.*` | bf16 | bf16（ignore） |
| `generation_config.json` | **無し**（eos は `text_config` の 248044 で解決） | **有り**（`eos_token_id: 248044`） |
| MTP 重み / `visual` | 同梱（0.49 GB / 0.91 GB・shelf は未読） | 同梱（0.49 GB / 0.91 GB・非量子化・shelf は未読） |
| 比較で効く差 | — | 量子化された Linear の丸めのみ |

* StopToken は両者 248044 で一致（exact 側は解決手順 3、approx 側は手順 2 を通る）。
  **approx 側は `generation_config.json` を持つ**ので、両経路で eos 解決の経路が違う点に注意（結果は同値）

**（6）履歴: EXL3 4bit 経路（2026-10-28〜29・中止）**

* 2026-10-28 は approx を `models/Qwen3.5-9B-exl3`（EXL3 4bit、7.19 GB）としていた。
  読み込みは `quantization_config.json`（config.json とは別ファイル）の **`tensor_storage` 表**（355 エントリ）駆動、
  ペイロードは `.trellis`（I16）/ `.suh` / `.svh` / `.mcg`（`mcg_multiplier` はコードブック照合マーカ）
* **スモークは完了**（参照: `references/exllamav3` の純 Python `tests/testlib/trellis.py`、
  対象 `model.language_model.layers.0.mlp.down_proj`）: tail-biting 検査 全 196,608 タイル True、
  bf16 元とのコサイン **0.997012**・相対 Frobenius **7.73%**（SQNR ≈ 22 dB）、
  レイアウト破壊では全て大幅悪化 → 「材料が読める／仕様は足りている」まで進んだ
* GEMM は 4 段 INT8 W8A8 経路（Hadamard128 を K/N 方向、`mcg` から W8 を直接復元、
  固定 weight scale `1/32`）を採用し、gfx1201 実機で INT8 GEMM 照合 0/256・mcg 65,536 状態の
  網羅検証を完了（検証物 `DONT_COMMIT/exl3_hip_int8_prefill_mcg/`・`DONT_COMMIT/exl3_mul1_hip_i8/`、
  計測 `artifacts/exl3_perf_20261029/`・`artifacts/exl3_opt_20261029/`、いずれも gitignore 済み・未コミット）
* `DataType::i16` 追加（方針 (A)。責務は「dtype はメタ・意味論は QuantSpec」）も同作業で行われたが、
  **本リポジトリの `main` には未マージ**（現行の `dtype.hpp` に `i16` は無い。上記 (1) の方針はこの
  切り分けを引き継ぐ）
* 2026-10-29 に**製品量子化版を FP8 へ切り替えた**ため、上記は V1 の前提から外れた。
  EXL3 は低 VRAM 協調推論の候補として再び登りうる（そのときは本节を履歴として参照する）

---

## 8. 未決

* **exact / approx のペア** — **解消（2026-10-29 に更新）**: 正本 `./models/Qwen3.5-9B`
  （bf16、4 シャード 19.3 GB）と製品量子化版 `./models/Qwen3.5-9B-FP8`
  （FP8 W8A8、単一 13.52 GB ＋ `mtp` 0.49 GB／ディスク合計 14.01 GB）を在庫とする。
  V1 はこの 9B ペアで行う。bf16 9B は単一 GPU に収まる（実測 VRAM 常駐 20.56 GB）。
  approx 側の payload は 12.61 GB なので単一 GPU に収まる見込み（**VRAM 常駐は未測定**）。
  旧 approx だった `Qwen3.5-9B-exl3` は 2026-10-29 に在庫から除外（§7「（6）履歴」）。
  27B 級 bf16 は現状未取得のまま（PP の実績も後ろ倒し）
* **`F8_E4M3` のリーダ対応** — §7 (1) の未解決前提。`DataType::fp8_e4m3` を追加（推奨）するか
  `u8` 流用で踏むか。**V1 のリーダ着手前に要決定**
* **FP8 GEMM の命令実在** — gfx1201 に fp8 e4m3 のマトリクス命令（WMMA/MFMA）が実在するか。
  無ければ in-kernel dequant ＋ bf16 GEMM（§7 (2)）。**ミニベンチで測ってから決める**（決め打ちしない）
* **活性化量子化の置き場** — per-token の `a_scale` 計算を独立カーネルにするか GEMM の prologue に
  融合するか（prefill rows=64 と decode rows=1 の両形）。最初は分離でよいか
* **量子化範囲の非対称** — `linear_attn.*` は bf16 のまま（ignore）で、**同一層内で dtype が混在**する。
  Linear primitive がテンソル単位で dtype を選ぶ形になるため、step graph 側の扱いを決める
  （KV・GDN 状態の dtype は現行どおりでよいか）
* **スモーク成果物の置き場**: EXL3 時の `.cache/exl3_smoke/` は履歴として残るだけ。
  FP8 のスモーク（§7 (4) の 1）は `tests/` 配下の support かツールとして置き、
  答え合わせに使える形にするか
* **StopToken の配列の扱い**: `generation_config` が配列（例 `[248046, 248044]`）のときに
  「最初に現れた方」で止めるでよいか。
  **2026-10-29 確認**: 製品版 `Qwen3.5-9B-FP8` の `generation_config.json` は**単値 `248044`**、
  正本 `Qwen3.5-9B` には当該ファイルが無い（手順 3 で同じ値に解決）。**9B ペアでは配列ケースは発生しない**
* **fixture の 9B での妥当性 — 解消（2026-10-28）**: 9B の tokenizer で fixture（256 ids）を
  decode → 再 encode した結果**完全一致**（round-trip True、文面も成立）。StopToken は
  `generation_config.json` が無いため手順 3 の `text_config.eos_token_id = 248044` で解決され、
  実走でも `stop_token` 停止を確認（§7「V0 実績（9B）」参照）
* **budget 打ち切り時の判定**: 両経路とも同一位置で打ち切られれば一致として扱うが、
  片方だけ打ち切った場合をどう数えるか（推奨: 不一致として T2 に回さず `budget_exhausted` 単独で報告）
* **PP の分割方法**: `--pipeline-parallel=K` は **stage 数（phaseshift bench の均等分割を既定）**にするか、
  **Strata 形で開始レイヤ番号を明示**するか。`auto`（境界探索）はコストモデルが Strata 固有のため後回しでよいか
* **CPU 経路の扱い**: `--device=cpu` は T0 の補助としてどこまで使うか（CPU f32 は parity_policy の基準
  だが、9B / Release で prefill 4.90・decode 3.68 s/token（旧 4B の Debug 実測は 16 s/token）。
  → CPU 経路は T0 と小規模モデルの検証に限定する、という運用でよいか）
