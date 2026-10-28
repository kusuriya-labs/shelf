# shelf-validate 仕様（草案）

状態: 草案 v1（V0 は実装済み・未マージ）。

* **現行仕様は `docs/spec/validate.md`、運用手順は `docs/guide/shelf-validate.md` に分離した**
  （実装済みの内容だけを `docs/` に置き、本ドキュメントは仕様・計画（V1 以降・PP 設計・未決事項）を扱う）。
関連: `draft-docs/parity-policy.md`（T0〜T3 の判定）、`draft-docs/engine-contract.md`、
`draft-docs/directory_design.md`（§5 D7 検証・§3.5）、`draft-docs/port_strata_cpu_gpu_coop.md`

---

## 0. 目的

実モデルで **無損失経路（exact）** と **プロダクション量子化経路（approx）** を同じ入力で走らせ、
出力トークン列を比較する。`parity-policy.md` の T0（決定性）・T1（完全一致）・T2（誤差由来の判定）
を**実モデル上で実行する道具**が本アプリである。

* 内部での量子化変換は行わない（exact と approx はそれぞれ渡されたディレクトリをそのまま読む）
* 比較の主対象は decode で生成したトークン列（StopToken まで）

---

## 1. CLI

```
shelf-validate \
    --exact-model=<dir>        # bf16（無損失側）      例: ./models/Qwen3.5-9B
    --approx-model=<dir>       # 量子化（EXL3 4bit）  例: ./models/Qwen3.5-9B-exl3
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
* なぜ fixture か: `parity-policy.md` の T0（決定性）を再現可能にし、tokenizer をアプリから切り離すため
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
   （approx は `quantization_config.json` の **`tensor_storage` 表を正**として単一
   `model.safetensors` を読む。§7 V1 参照）
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

| 軸 | 内容 | 対応する parity-policy |
| --- | --- | --- |
| **T0 決定性** | 同一構成で `--runs=2` 回実行し、トークン列が一致するか | T0 |
| **T1 完全一致**（主） | exact と approx の decode トークン列が StopToken まで一致するか | T1 |
| **T2 誤差判定** | 不一致のとき、**最初の不一致位置**での exact 側の top-1 / top-2 の margin と、不一致位置の logits 差を計算し、`parity-policy.md` §3 の dtype 誤差実測と比較 | T2 |
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
  - 各 stage の重みは**担当デバイスに部分ロード**。approx（EXL3）は単一 `model.safetensors` なので
    レイヤ単位ファイルは無いが、テンソルの byte range は safetensors ヘッダから取れ、
    量子化 / 非量子化の別は `tensor_storage` 表で判るため、レイヤ範囲の部分ロードは成立する
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
| **V1** | approx（**EXL3 4bit**）経路の読み込みと実行 → T1/T2 比較。前提と着手順は下記「V1 の前提となる実装」 | exact と approx のトークン列比較が出力され、diverged 時に margin が出る |
| **V2** | pipeline-parallel（`--pipeline-parallel=K`） | `exact(PP=1)` と `exact(PP=K)` のトークン列が一致（T1） |
| **V3** | 自動化: `ctest` のラベル（例 `e2e`）で回し、モデル無しの環境では skip | CI 相当の流れで 1 件が守られる |

### V0 実績と実装メモ（2026-10-27）

* **2026-10-28 の資材更新と開発モデルの選定**: `models/` から `Qwen3.5-4B` と `Qwen3.8-*` が削除され、
  在庫は `Qwen3.5-9B`（bf16）と `Qwen3.5-9B-exl3`（EXL3 4bit）の 2 組のみ。
  **この 2 つを開発時点の標準モデルに選定**した（統一した記載は `draft-docs/directory_design.md` §1.0）。
  以下の数値は **旧 4B での実測（履歴）**。以後の DoD・計測は **9B を対象**とし、
  未測定の値は「未測定」と書く（config は `text_config` の必須キー全充足を確認済み）。

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
  実装の妥当性は外部参照（HF）とのトークン列比較で確認する（`parity-policy.md` に追記）

### V1 の前提となる実装（別作業）— **EXL3 に変更（2026-10-28）**

> FP8 での V1 は**キャンセル**した。理由は次のとおり: ① checkpoint が既にある
> （`models/Qwen3.5-9B-exl3`、7.19 GB）② 今後の量子化も `exllamav3` で自作する方針
> ③ 4bit は低 VRAM 協調推論の主目的に直結 ④ 参照実装が 3 系統ある
> （`exllamav3` 公式 / Strata `vendor/exl3.*` / exllamav3 の GEMM テスト群）。

**（1）approx（EXL3）の読み込み**

* 対象: `models/Qwen3.5-9B-exl3`。`quantization_config.json`（**config.json とは別ファイル**、
  314 KB）に `quant_method: exl3`, `version: 0.0.22`, `bits: 4.0`, `head_bits: 6`,
  `out_scales: always`, `codebook: mcg`, `calibration: {rows: 250, cols: 2048}`
* **正は `tensor_storage`（355 エントリ）**: 各エントリが `stored_tensors`（実テンソル名・shape・dtype）+
  `quant_format` + `bits_per_weight` + `mcg_multiplier` を持つ。
  → FP8 の `modules_to_not_convert` みたいな**推測判定は不要、表駆動**になる
* ペイロード（例 `layers.0.mlp.down_proj`）: `.trellis`（I16、例 [768,256,64] の符号列）+
  `.suh` / `.svh`（F16 のスケール系）+ `.mcg`（I32 スカラー + `mcg_multiplier`）。
  非量子化テンソル（`embed_tokens` / `lm_head` / 各 layernorm / `norm` 等）は F16・BF16 のまま格納
* ファイルは単一 `model.safetensors`（1,363 テンソル、dtype は F16 / F32 / I32 / I16 / BF16）。
  読み込みは既存の artifact 層（safetensors reader）の上に **`tensor_storage` 表の解釈**を足す形
* **未解決の前提（shelf 側・要判断）**: `.trellis` の dtype は `I16` だが、shelf の safetensors リーダは
  `F64 / F32 / F16 / BF16 / I64 / I32 / U8` のみ対応（`src/artifact/safetensors/reader.cpp:20-29`）。
  現状は **EXL3 checkpoint を開けない**（`unsupported dtype 'I16'` で `Reader::open` が失敗する）。
  * (A) `DataType::i16` を追加 — 数値型としては自然な拡張（`u16`/`u32`/`i32`/`i64` は既にある）。
    `suh`(F16) と `mcg`(I32) は既に読めるため、対象は `.trellis` だけ
  * (B) raw 経路 — リーダが dtype 判定なしでバイト範囲を返し、意味論は `quant/` の QuantSpec に委ねる
    （「量子化ペイロードは DataType ではない」という契約に素直だが変更が大きい）
  * **推奨は (A) ＋ 責務の明記**: DataType は「バイト長・符号性のメタ」、
    意味論（codebook・trellis 読出）は QuantSpec の役目、と割り切る。**要承認**

**（2）EXL3 GEMM（ホットパス dequant）** — shelf には現状 4bit の経路も FP8 も無い
（bf16 のみ、fp8 は KV 側にのみ存在）。契約（`parity-policy` / 量子化契約）どおり **dequant はカーネル内**。

* **移植の一次候補: Strata `src/dsv41/vendor/exl3.cu`（1,745 行）と `exl3_cpu.cpp`（1,281 行）**
  — Project Maya (MIT) 経由で exllamav3 (MIT) の codebook / trellis 算術に準拠。
  ヘッダに **HIP 向けの道筋が明記**されている（sm_80 以降は tile を tensor core フラグメントへ
  直デコード、**HIP と旧 GPU は F32 FMAs**）。移植時は `third_party/project-maya/NOTICE.md` の追従
* **答え合わせ**: `references/exllamav3`（公式）の
  `modules/quant/exl3_lib/`、`tests/kernels/gemm/test_exl3_{gemv,gemm_shapes,mgemm_sliced,gemv_tail}.py`、
  `tests/testlib/exl3.py`。Python/CUDA 中心なので **移植せず参照と突合に使う**
* **設計論点（V1 で決める）**:
  * (a) **検証経路**: カーネル内でデコードし bf16 WMMA のオペランドにする（実装容易、契約適合）
  * (b) **本番経路**: Strata HIP 準拠の F32 FMAs、または exllamav3 の tensor core 直デコード
  * V1 は (a) で開始し、(b) は性能フェーズへ回す

**（3）注意（従来どおり）**: 「ロード時に bf16 へ展開してから既存の bf16 GEMM で計算する」のは
**契約違反**（有損 dequant をロード時に置かない）。**カーネル内で**デコードして bf16 オペランドに
することは許される（(a) がこれに当たる）。暫定で外に出す場合は契約上の位置づけを明記する。

**（4）V1 の着手順（可否の実測 → 仕様確定 → 本体）**

1. **スモーク**: `models/Qwen3.5-9B-exl3` の 1 テンソル（例 `layers.0.mlp.down_proj` の
   `.trellis` / `.suh` / `.svh` / `.mcg` + `mcg_multiplier`）を、Strata `exl3_cpu.cpp` か
   `exllamav3` を参照してデコードし、**exllamav3 の出力と一致する**ことを確認する
   （ここで「材料が読める／仕様が足りる」が確定する）— **→ (5) で完了（2026-10-28）**
2. **GEMM 方針の決定**（(a)/(b)）。初期値は (a)
3. **V1 本体**: リーダ → GEMM → T1/T2 → 終了コード 0/1/2
4. **T2 の基準取り**: `parity-policy.md` §3 の手順で EXL3 4bit の誤差規模を測る（§8 参照）

**（5）スモーク実績（2026-10-28、着手順 1 完了）**

* **経路**: `references/exllamav3` の **`tests/testlib/trellis.py`**（拡張不要の純 Python 参照）。
  `decode():62` / `unpack():99` / `tile_perm():47` / `dequant_rotated():149` /
  `sylvester():159` / `dequant():166`（`W = diag(suh) · H · W' · H · diag(svh)`）。
  本体パッケージには pure 経路が無く（`ext.py` は import 即 JIT コンパイル）、拡張は ROCm 未対応のため
  自作は不要だった
* **対象**: `model.language_model.layers.0.mlp.down_proj`（k = intermediate 12288、n = hidden 4096、K = 4）
* **(A) 構造 — 全項目 OK**: `trellis (768,256,64) int16`（= k/16, n/16, 16·4）、`suh (12288) fp16`、
  `svh (4096) fp16`、`.mcg` int32 = `0xCBAC1FED`。**tail-biting 検査 全 196,608 タイル True**
  （ビットストリームの解釈が書き込み側と一致）
* **(B) 妥当性 vs bf16 元**（主判定）: コサイン **0.997012**（行 min 0.996837、>0.99 の行 100%）、
  相対 Frobenius **7.73%**（SQNR ≈ 22 dB）、MAE = **重み std の 6.00%**、最大絶対差 = |w|max の 3.5%、
  行相対 L2 mean/median/max = 0.0773 / 0.0773 / 0.0795 → **4bit として妥当な値域**
* **(C) レイアウト健全性 — 破壊で全て大幅悪化**（正: 7.73% / cos 0.997）:
  タイル格子転置 136.9%（cos 0.063）/ 転置合わせなし比較（cos 0.0001）/
  suh・svh 入れ替え 116.1% / 入出力 Hadamard 省略 141.3%（cos ≈ 0）
* **判定: 材料は読めている／仕様は足りている**。**入力回転の追加要件は不要**（dense 等価 W に
  両 Hadamard が畳み込み済み。GEMV を W を物化せず行う場合のみカーネルがその場で H を掛ける —
  Strata `exl3_cpu.hpp` の通り）
* **`.mcg` の実用**: 値 = `codebook_mcg_mult = 0xCBAC1FED`（`quantize.py:19`）= `mcg_multiplier` 3417055213
  → リーダが**起動時にコードブック照合**できる固定マーカ（Strata の命名もこれで正しい）
* **成果物**: `.cache/exl3_smoke/{exl3_smoke.py, smoke.log}`（gitignore 済み・未コミット）
* **残りの未検証**: `head_bits = 6` / `out_scales = always` の語義（down_proj では非影響を確認）、
  他テンソル（gate/up・embed）と全体推論、exllamav3 拡張との bit 一致（ROCm 未対応で未実施）

---

## 8. 未決

* **exact / approx のペア** — **解消（2026-10-28）**: `./models/Qwen3.5-9B`（bf16、4 シャード 19.3 GB）と
  `./models/Qwen3.5-9B-exl3`（EXL3 4bit、7.19 GB）を取得。V1 はこの 9B ペアで行う。
  bf16 9B は単一 GPU に収まる見込み。27B 級 bf16 は現状未取得のまま（PP の実績も後ろ倒し）
* **`.trellis` の `I16` への対応方針** — §7 (1) の未解決前提 (A)/(B) の採否。
  **V1 のリーダ着手前に要決定**（推奨: (A) + 「dtype はメタ、意味論は QuantSpec」の責務明記）
* **スモーク成果物の置き場**: 現状 `.cache/exl3_smoke/`（gitignore 済み・未コミット）。
  V1 実装時に `tests/` 配下の support かツールとして再配置し、答え合わせに使える形にするか
* **StopToken の配列の扱い**: `generation_config` の `[248046, 248044]` を「最初に現れた方」で止めるでよいか
* **fixture の 9B での妥当性 — 解消（2026-10-28）**: 9B の tokenizer で fixture（256 ids）を
  decode → 再 encode した結果**完全一致**（round-trip True、文面も成立）。StopToken は
  `generation_config.json` が無いため手順 3 の `text_config.eos_token_id = 248044` で解決され、
  実走でも `stop_token` 停止を確認（§7「V0 実績（9B）」参照）
* **budget 打ち切り時の判定**: 両経路とも同一位置で打ち切られれば一致として扱うが、
  片方だけ打ち切った場合をどう数えるか（推奨: 不一致として T2 に回さず `budget_exhausted` 単独で報告）
* **PP の分割方法**: `--pipeline-parallel=K` は **stage 数（phaseshift bench の均等分割を既定）**にするか、
  **Strata 形で開始レイヤ番号を明示**するか。`auto`（境界探索）はコストモデルが Strata 固有のため後回しでよいか
* **CPU 経路の扱い**: `--device=cpu` は T0 の補助としてどこまで使うか（CPU f32 は parity-policy の基準
  だが、9B / Release で prefill 4.90・decode 3.68 s/token（旧 4B の Debug 実測は 16 s/token）。
  → CPU 経路は T0 と小規模モデルの検証に限定する、という運用でよいか）
