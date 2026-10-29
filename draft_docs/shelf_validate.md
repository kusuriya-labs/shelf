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

### V1 実装（2026-10-29）— approx = FP8 W8A8

**方針の確定**: 製品量子化版を **FP8 W8A8・活性化 dynamic** に決めた（EXL3 4bit は同日中止、
下記の「（6）履歴」）。決め手は ① checkpoint が既にある ② 正本と同一の safetensors 形式・
同一テンソル名で、量子化されていないテンソルは bf16 のまま同梱（差分が量子化 Linear だけ）
③ 量子化の記述が `config.json` の `quantization_config` に集約され、ヘッダの dtype で
テンソル単位に判別できる（表駆動）④ 参照実装が多い（`transformers` の fp8 linear、
compressed-tensors の `float-quantized`、`references/phaseshift-fp8` の WMMA 経路）。
**代償は容量**（EXL3 4bit 7.19 GB → FP8 14.01 GB）。

**スモーク（§7 (4) 1 に相当、実施済み）**: 9B-FP8 の
`model.language_model.layers.0.mlp.down_proj`（`F8_E4M3` `[4096,12288]`、`weight_scale` BF16
`[4096,1]`）をリーダで読み、コード×スケールの dequant が成立することを確認。
`unit.artifact_safetensors` に `F8_E4M3` の読出テストを追加。

#### 実装した範囲

| 層 | 場所 | 内容 |
| --- | --- | --- |
| dtype | `src/foundation/include/shelf/foundation/dtype.hpp` | `DataType::fp8_e4m3`（1 バイト・実在する浮動小数型）。`element_size` = 1。`u8` は流用しない |
| リーダ | `src/artifact/safetensors/reader.cpp` | `map_dtype("F8_E4M3")` を追加 |
| 量子化契約 | `src/quant/{include/shelf/quant/fp8_w8a8.hpp,fp8_w8a8.cpp}`（新 target `shelf::quant`） | e4m3 コーデック（RNE・satfinite・NaN 保持）、**断片順（preshuffle）レイアウト**の定義と offset 計算、粒度 helper（`WeightScaleGranularity` / `act_scale_groups` / `scale_k_blocks`） |
| GEMM | `src/backend/hip/kernels/gemm/linear_fp8.hip` ＋ detail | gfx1201 の `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` を使用（**命令は実在＝実測確認**）。rows=1 は GEMV 形（Decode1）、rows≤16/32/64 は RowBlock1/2/4 |
| 活性化量子化 | `src/backend/hip/kernels/elementwise/activation_quantize_fp8.hip` | 独立カーネル（1 ブロック = 1 トークン）。宣言が `token` なら 2 パスで行 amax、`1×128` なら 1 wave = 1 ブロックで単一パス。**GEMM の prologue には融合しない**（§8 の未決は独立カーネルで確定） |
| 実行器 | `src/backend/hip/step_executor.{hpp,cpp}` | `upload_weight_fp8`（断片順コード＋スケール）、FP8 活性化_scratch、`Kind::linear` / `output_projection` で FP8 重みなら活性化量子化→FP8 GEMM へ分岐（**dequant はホットパス内。ロード時の有損変換は無い**＝契約 P5 遵守） |
| apps | `apps/common/app_support.{hpp,cpp}` | `read_fp8_format`（`quantization_config` 解析）・`build_fp8_weight`（スケール読みと preshuffle）・`upload_plan`（dtype で bf16 / f32 / FP8 を判別）。shelf-validate も同じ loader を使うようにした |

**スケール折り込みの位置（数値に効く）** : 未スケールの e4m3 コード（最大 448）を K 全体で
足すと桁上がりの打ち消しで誤差が増えるため、**スケールが切り替わるブロック（128 K）ごとに
累積へ掛ける**（移植元 PhaseShift と同じ手順。活性化側が per-token でも同じブロック単位で
掛ける）。32 K タイル毎に掛ける形（初版）からは decode 23.2 → 25.0 ms/tok と少し遅くなったが、
宣言どおりの粒度で統一した。

#### 粒度は checkpoint の宣言が正（2026-10-29 の実測で確定）

`apps` 側が `config.json` の `quantization_config` を読み、実行器に渡す。**実装側の決め打ちはない**
（`--fp8-act-group` は実験用の上書きで、使った場合は印字する）。

| checkpoint | `quant_method` | 重みスケール | 活性化 | 実測の動作 |
| --- | --- | --- | --- | --- |
| `Qwen3.5-9B-FP8` | `compressed-tensors`（ModelOpt `FP8_DYNAMIC`） | `strategy: channel` → `<base>.weight_scale` BF16 `[out,1]` | `strategy: token` / `dynamic` → **行全体 1 スケール**（`act_group=0`） | 単一 GPU で pp/tg/validate 動作 |
| `Qwen3.8-27B-FP8` | `fp8`（HF blockwise, `fmt: e4m3`） | `weight_block_size: [128,128]` → `<base>.weight_scale_inv` BF16 `[⌈N/128⌉,⌈K/128⌉]`（格納値は実乗数） | `activation_scheme: dynamic` → **1×128**（`act_group=128`） | 単一 GPU で pp/tg 動作（rows=64 prefill は VRAM 不足） |

* 活性化の粒度差は本モデルでは小さい（実活性化 15,872 行で相対 L2 誤差 0.02112 → 0.01850、
  **平均 1.14 倍**の改善、K=12288 で 1.28 倍、外れ値のある行で最大 10 倍）。
  深層で打ち消すため T1 がどこまで改善するかは未測（宣言と違う粒度は使わない）
* スケールの dtype は BF16 / F32 両対応（実行器へ渡す時点で f32 に正規化）
* `kv_cache_scheme` が non-null、静的活性化（`input_scale`）、`group_size≠128`、
  128×128 以外の block、int 系量子化は**明示的に失敗**させる（近似しない）。単体テスト
  `unit.fp8_format` がこの境界を固定している

#### 検証（テスト）

* `unit.quant_fp8`（CPU）: コーデックの全 code 往復境界、RNE/tie、satfinite、NaN、
  tiled_offset と preshuffle/untile の一致、粒度 helper の個数
* `unit.fp8_format`（CPU）: 9B / 27B の宣言を実形へ落とせること、未対応宣言の失敗
* `unit.hip_gemm_fp8`（HIP）: ① 活性化量子化の**デバイス出力＝host 再現**（粒度 2 種）
  ② GEMM を f64 golden と照合（**粒度 4 組合せ × 境界形状 15 件**。判定は
  「項の絶対値総和に対する比 ≤ 1.0e-4」で、打ち消しによる要素相対誤差の膨張を除外する）
  ③ bf16 GEMM（同じ量子化値）との突合、④ decode1 の BF16 出力
* cpu 15 件 / hip 22 件が全通過（2026-10-29）

#### 実測（R9700 / gfx1201・Release = `hip-rel`・GPU 0）

| 計測 | bf16 正本 | FP8 製品版 | 比率 |
| --- | --- | --- | --- |
| `bench pp` 2048 tok（chunk 64） | 6,680.9 ms = 306.6 tok/s | 4,048.0 ms = 505.9 tok/s | **1.65×** |
| `bench tg` ctx 512・128 tok（warmup 8） | 30.838 ms/tok = 32.42 tok/s | 25.035 ms/tok = 39.93 tok/s | **1.23×** |
| `shelf-run` 256→24 tok（Debug ビルド） | 30.68 ms/tok | 25.03 ms/tok | 1.23× |
| `shelf-validate` run（Debug・2 run） | prefill 846.7 ms / decode 7,496.5 ms（230 tok） | 524.6 ms / 5,783.3 ms（212 tok） | 1.61× / 1.30× |
| VRAM 常駐（`device_mem_used_bytes`・arena 2,298 MiB 込み） | 20.55 GB | **15.09 GB** | 0.73× |
| 27B-FP8 `pp` 2048 tok（chunk 16） | —（bf16 正本が在庫に無い） | 12,157.7 ms = 168.5 tok/s | — |
| 27B-FP8 `tg` ctx 512・64 tok | — | 61.10 ms/tok = 16.36 tok/s | — |

* 9B-FP8 の `shelf-run`（fixture 256 tok・24 decode）は **bf16 と 24/24 トークン一致**（Debug・
  上記 `bench` とは別実走）。27B-FP8 は同一 fixture で「Ancient sailors oriented by the sun and
  stars, but lacked precision. The astrolabe, used by Arab navigators…」と継続し、出力は成立
* **VRAM**: 9B-FP8 は単一 GPU に収まる（`bench` の arena 2,298 MiB 含む）。
  **27B-FP8 は rows=64 の prefill グラフ（arena 約 5.6 GiB）が載らず**、`shelf-run` には
  `--prefill-rows` を追加して 16（arena 1,530 MiB）で計測した。rows=32 は arena 2,997 MiB で可
* `--device=cpu` の FP8 は**明示エラー**（`read_f32` が「FP8 の重みは f32 ゴールデンでは読まない」
  を返す）。CPU 参照に fp8 経路は作らない

#### V2 実装（2026-10-29）— pipeline-parallel

* **分割**: `even_stage_ranges(num_layers, K)` で均等に切り、**余りは最後のステージへ**
  （全層を必ず覆う）。`--pipeline-parallel=K` で GPU は `--gpu` から K 台使う
* **置き場（2026-10-29 更新）**: ステップ実行・hand-off・時間計測は
  **エンジン側 `src/runtime/pipeline`**（`StageGraphs` ＋ `backend::StageExecutor` インターフェース）。
  `apps/common/pipeline.{hpp,cpp}` はステージの生成（グラフ・実行器・重み投入）だけを行う
  composition root に縮めた。`runtime` は `hip/` もモデル型も見ない（移設時に境界テストが
  モデル名の漏れを検出し、`StageGraphs` を `ir` だけで表す形に直した）
* **グラフ**: `build_step_graph(cfg, kv_capacity, rows, StageRange{lo,hi})`。
  先頭ステージだけ embedding を持ちトークンを入力に、以降のステージは前段の hidden
  （bf16 生ビット）を入力に、最終ステージだけ final norm ＋ lm_head を持つ。
  状態（KV・conv・recurrent）はレイヤ単位宣言なので、範囲を絞れば自然に分散する
* **受け渡し**: `HipStepExecutor::run_stage(tokens or hidden, n, base)` と
  `fetch_value_bf16` で **bf16 のまま**往復（f32 を経由しない = 丸めを入れない）。
  各ステージの `run_stage` は内部で同期するので、**直列 hand-off** になる
* **正しさの門**: `tests/unit/test_hip_pipeline.cpp`（全ステージを同一 GPU に置く Strata の
  `--split-device 0` と同型）で、**1 ステージと 2/4 ステージ・不均等分割（5 層を [0,2)/[2,5)）・
  prefill 4 行チャンクの logits がビット一致**することを固定（hip 23 件の 1 つ）
* **実モデルでの確認**:
  * 9B-FP8: `--pipeline-parallel 1 / 2 / 4` で **280 トークン完全一致**
  * 27B-FP8: PP=2（GPU 0-1）と PP=4（GPU 0-3）で **280 トークン完全一致**。
    1 GPU では rows=64 の prefill が載らなかったモデルが、PP では
    `--prefill-rows 16` で 2 台（arena 782/813 MiB）でも走る
  * `shelf-validate --pipeline-parallel=2 --pp-baseline=on` で
    **T1pp (PP=2 vs PP=1): identical / exit 0**（9B・128 tok。`--pp-baseline` は
    PP>1 のとき PP=1 でも走らせて一致を確認する門。PP=1 が載らないモデルでは使わない）
* **実測（Release = `hip-rel`・GPU 0 起点）**:

| 計測 | PP=1 | PP=2 | PP=4 |
| --- | --- | --- | --- |
| 9B-FP8 `pp` 1024 tok（chunk 64） | 1,987.6 ms = **515.2 tok/s** | 2,000.0 ms = 512.0 tok/s（+0.6%） | — |
| 9B-FP8 `tg` ctx512・64 tok | 25.02 ms/tok = 39.96 tok/s | 26.94 ms/tok = 37.12（+7.7%） | — |
| 27B-FP8 `pp` 1024 tok（chunk 16） | 不可（VRAM） | 6,245.3 ms = **164.0 tok/s** | 6,450.2 ms = 158.8 tok/s |
| 27B-FP8 `tg` ctx512・64 tok | 不可 | 63.48 ms/tok = 15.75 tok/s | 64.23 ms/tok = 15.57 tok/s |
| 27B-FP8 `shelf-run` prefill rows=64 | 不可（arena 約 5.6 GiB が載らない） | **可**（stage あたり arena 2,937 / 3,059 MiB） | 可 |
| 27B-FP8 `shelf-validate`（PP=2・128 tok） | — | prefill 969.9 ms / decode 8,060.9 ms、**exit 0** | — |

  * 直列 hand-off の代償は **prefill +0.6〜3%・decode +4〜8%**（ステージ数が増えてもほぼ横ばい）。
    27B の PP=2 は単一 GPU（chunk 16）の 1024 tok 換算 6,079 ms に対して +2.7%
  * **27B-FP8 は PP=2 以上で rows=64 の prefill が載る**（単一 GPU では不可能）。
    これが「27B を validate できない」制約の直接の解消で、validate の既定 chunk（64 行）がそのまま使える
* **未実装（意図的）**: prefill のチャンク重畳（flag による非同期 hand-off）と
  `auto` の分割探索（コストモデルが Strata 固有）。まず直列で正しさを固定した

#### V1 の判定実績（exact = 9B bf16 / approx = 9B FP8・Debug・GPU 0）

| 実行 | T0 | T1 | 不一致位置 | exact margin | 判定 |
| --- | --- | --- | --- | --- | --- |
| `--prefill-tokens=256 --budget=512 --runs=2` | identical（exact 230 tok / approx 212 tok、共に stop_token） | **diverged** | k=46（exact 279 / approx 264） | **0.1051** | T2 `pending_baseline` → **exit 1（条件付き合格）** |
| `--budget=384 --runs=1 --t2-margin=0.25` | skipped | **diverged** | k=46（同上） | 0.1051 | `within_margin` → exit 1（基準内の実証） |
| 同上で `--t2-margin=0.05` | skipped | **diverged** | k=46 | 0.1051 | `outside_margin` → **exit 2**（基準外の検出） |
| `--prefill-tokens=192 --budget=448 --runs=1` | — | **diverged** | k=156（exact 2708 / approx 10033） | **0.1923** | 同上（exit 1） |

* **T1 は恒常には通らない**（§8 の予測どおり）。両ケースとも**正本側がほぼ同点の位置**で割れており
  （margin 0.10 / 0.19、 logits の絶対規模に対して小さい）、`parity_policy.md` §3 の
  「本戦場は T2」が実測で裏付いた
* T2 の基準値（**margin 分布の p99**）は未採取。`--t2-margin=<x>` で閾値を与えると
  `within_margin`（exit 1）/ `outside_margin`（exit 2）を判定する。基準が無い現状は
  `pending_baseline` として measured margin を JSON に残す
* 出力 JSON は `schema: shelf-validate/v1`（`t1` / `t2` / `models_run.<exact|approx>.fp8`
  ・`fp8_act_group`・`device_mem_used_bytes`・step 毎の `tops` を含む）
* exact と approx は**同時には VRAM に載せない**（9B で bf16 17.91 GB ＋ FP8 12.61 GB ≳ 搭載 VRAM）。
  exact を走らせてから実行器とシャードを解放し、approx を読む。比較は host 側のトークン列と
  top-1/2 で行う
* 比較の前提として **exact と approx が同一構成・同一停止トークン集合**であることを検査する
  （違うモデルを渡すと exit 2 で拒否）

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
  V1 はこの 9B ペアで行う（**2026-10-29 に実施済み・§7 の V1 実測**）。bf16 9B は単一 GPU に収まる
  （実測 VRAM 常駐 20.56 GB）。approx 9B-FP8 も単一 GPU に収まる（**実測 VRAM 常駐 15.09 GB**・arena 2,298 MiB 込み、
  pp/tg 実測済み）。旧 approx だった `Qwen3.5-9B-exl3` は 2026-10-29 に在庫から除外
  （§7「（6）履歴」）。
* **2026-10-29 追加**: 製品量子化版として `./models/Qwen3.8-27B-FP8`（HF blockwise fp8、
  `weight_block_size: [128,128]`、66 シャード 24.3 GB（`language_model` + `lm_head`））を在庫化。
  **bf16 の 27B 正本は未所持**なので 27B では T1（正本との比較）が取れない。
  単独では pp 168 tok / s・tg 16.36 tok/s を実測（§7 の表）
* **`F8_E4M3` のリーダ対応 — 解消（2026-10-29）**: `DataType::fp8_e4m3` を追加（`u8` は流用しない）。
  責務の切り分けは予定どおり（dtype はバイト長・符号性のメタ、スケール意味論は `src/quant`）。
  `unit.foundation` / `unit.artifact_safetensors` で固定
* **FP8 GEMM の命令実在 — 解消（2026-10-29）**: `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`
  が gfx1201 で実在し、f64 golden 照合が通る（`unit.hip_gemm_fp8`）。in-kernel dequant フォールバックは
  不要だった
* **活性化量子化の置き場 — 解消（2026-10-29）**: **独立カーネル**にした（GEMM の prologue には融合しない）。
  ただし 1 ブロック = 1 トークンの形なので **decode（rows=1）では 256 スレッド 1 ブロックしか立たず**、
  宣言 `token` の場合は 2 パス（amax → 量子化）になる。この占有時間は未測（`bench kernel` の
  fp8 エントリを追加して測る。融合するかはその後で決める）
* **量子化範囲の非対称 — 解消（2026-10-29）**: 実行器は `TensorId` ごとに FP8 重み表を引いて
  分岐する（`linear_attn.*` は bf16 のままupload）。step graph も KV・GDN 状態の dtype も変更不要だった
* **スモーク成果物の置き場 — 解消（2026-10-29）**: 答え合わせは `unit.quant_fp8` /
  `unit.fp8_format` / `unit.hip_gemm_fp8` に固定した（モデル資材に依存しない合成データと inline JSON）
* **StopToken の配列の扱い**: `generation_config` が配列（例 `[248046, 248044]`）のときに
  「最初に現れた方」で止めるでよいか。
  **2026-10-29 確認**: 製品版 `Qwen3.5-9B-FP8` の `generation_config.json` は**単値 `248044`**、
  正本 `Qwen3.5-9B` には当該ファイルが無い（手順 3 で同じ値に解決）。**9B ペアでは配列ケースは発生しない**
* **fixture の 9B での妥当性 — 解消（2026-10-28）**: 9B の tokenizer で fixture（256 ids）を
  decode → 再 encode した結果**完全一致**（round-trip True、文面も成立）。StopToken は
  `generation_config.json` が無いため手順 3 の `text_config.eos_token_id = 248044` で解決され、
  実走でも `stop_token` 停止を確認（§7「V0 実績（9B）」参照）
* **T2 の基準値（margin 分布の p99）が未採取**: 現状のデータは 2 点（0.1051 / 0.1923）だけ。
  prompt を変えた複数実行（fixture の切り替え・`--ids-file`）と budget 違いで分布を取り、
  `parity_policy.md` §6 に記録して `--t2-margin` の既定値にする
* **27B-FP8 の比較対象が在庫に無い**: bf16 の 27B 正本が手元に無いため 27B では T1 を取れない。
  27B の検証は ① 同一モデルの T0（決定性）② PP 導入後の `exact(PP=1)` vs `exact(PP=K)`
  ③ HF 側の blockwise fp8 との外部照合（できるなら）に留める。27B 側の実測は §7 の表のとおり
* **27B の prefill 行数**: rows=64 の arena（約 5.6 GiB）が 27B-FP8（重み 27.4 GiB）と両立しない。
  運用値（`--prefill-rows=16` で arena 1,530 MiB、rows=32 で 2,997 MiB）をどちらにするか、
  もしくは arena のチャンク毎再確保／rows 上限の自動決定を入れるか未決
* **活性化粒度を宣言から変えた場合の T1 改善量**: 128 グループは活性化側の実測誤差を平均 1.14 倍
  改善するが、T1 の一致率が変わるかは未測。方針は「宣言が正」で据え置き（上書きは実験用）
* **budget 打ち切り時の判定**: 両経路とも同一位置で打ち切られれば一致として扱うが、
  片方だけ打ち切った場合をどう数えるか（推奨: 不一致として T2 に回さず `budget_exhausted` 単独で報告）
* **PP の分割方法 — 解消（2026-10-29）**: `--pipeline-parallel=K` を **ステージ数**とし、
  均等分割（余りは最後のステージ）で実装した。開始レイヤ番号の明示指定（Strata 形）と
  `auto` 探索は未実装のまま（コストモデルが装置依存のため）
* **PP の重畳**: 直列 hand-off のみ。prefill チャンクの重畳（flag publish + wait）と
  ステージ間の overlap は未実装（速度は伸びしろ。正しさは直列で固定済み）
* **PP と approx の併用**: `shelf-validate` は exact/approx の双方を同じ K で PP 実行できるが、
  `--pp-baseline=on`（PP=1 との一致確認）は exact 単独時のみ。approx 側の PP 一致確認は
  未実装（同じ pipeline 実装なので数値は同一になる見込み）
* **CPU 経路の扱い**: `--device=cpu` は T0 の補助としてどこまで使うか（CPU f32 は parity_policy の基準
  だが、9B / Release で prefill 4.90・decode 3.68 s/token（旧 4B の Debug 実測は 16 s/token）。
  → CPU 経路は T0 と小規模モデルの検証に限定する、という運用でよいか）
