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
    --exact-model=<dir>        # bf16（無損失側）      例: ./models/Qwen3.8-27B
    --approx-model=<dir>       # 量子化（FP8 等）      例: ./models/Qwen3.8-27B-FP8
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
   （approx はレイヤ単位ファイル `layers-N.safetensors` を前提に読む。§7 V1 参照）
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
* **分割**: レイヤ境界で K ステージに分割（Hybrid の GDN 層があってもレイヤ単位で同一）
  - stage `s` = `[layer_s, layer_{s+1})` を担当し、decode 1 token が K ステージを順に通る
  - 各 stage の重みは**担当デバイスに部分ロード**。approx の `layers-N.safetensors` はレイヤ単位なので
    ステージ割当と天然に一致する
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
| **V0** | アプリ骨格: CLI（§1）・予算検査（§2）・fixture 入力・prefill（64 行チャンク）・greedy decode・StopToken 解決・比較と JSON 出力。`--exact-model` のみで運用 | Qwen3.5-4B で `--runs=2` の T0 が通り、終了コードと JSON が仕様どおり。入力 N=256 / budget=512 の検査も通る |
| **V1** | approx 経路の読み込みと実行 → T1/T2 比較 | exact と approx のトークン列比較が出力され、diverged 時に margin が出る |
| **V2** | pipeline-parallel（`--pipeline-parallel=K`） | `exact(PP=1)` と `exact(PP=K)` のトークン列が一致（T1） |
| **V3** | 自動化: `ctest` のラベル（例 `e2e`）で回し、モデル無しの環境では skip | CI 相当の流れで 1 件が守られる |

### V0 実績と実装メモ（2026-10-27）

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
* `--device=cpu` の全トークン実行は未実施（**実測 16 s/token** = Debug ビルドで 7 step 110 s。
  256 + 246 tok なら 1 run 約 2 時間）。CPU 経路の比較は §8「CPU 経路の扱い」の未決に同じ）。

### V1 の前提となる実装（別作業）

* **approx（FP8）の読み込み**: `Qwen3.8-27B-FP8` は
  `quantization_config` = `fp8` / `e4m3` / `weight_block_size: [128, 128]` / `activation_scheme: dynamic`、
  ファイルは `layers-N.safetensors` ×64 + `model.safetensors.index.json`。
  → **128×128 ブロックスケールの解釈**と、`modules_to_not_convert`（layernorm / lm_head / embed /
  `linear_attn` 系パラメータ）は **bf16 のまま読む**という読み分けが必要
* **FP8 GEMM（ホットパス dequant）**: shelf には現状 fp8 GEMM が無い（bf16 のみ。
  fp8 は KV 側にのみ存在）。契約（`parity-policy` / 量子化契約）どおり **dequant はカーネル内**。
  - 移植の一次候補: `phaseshift-fp8` の `src/phaseshift/kernels/optimized/linear/fp8.hip`
    （同じ `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` を使う。shelf のカーネルの出所と同系統）
* **注意**: 「読み込んで bf16 に展開してから既存の bf16 GEMM で計算する」のは**契約違反**（有損 dequant を
  ロード時に置かない）。検証のための暫定として使う場合も、契約上の位置づけを明記してから使う

---

## 8. 未決

* **exact モデルの所在**: `./Qwen3.8-27B`（例示）は手元に未確認。bf16 の 27B チェックポイントが無ければ
  V1 は「4B の approx（未取得）」でも成り立たない。取得先を確定させる
* **StopToken の配列の扱い**: `generation_config` の `[248046, 248044]` を「最初に現れた方」で止めるでよいか
* **budget 打ち切り時の判定**: 両経路とも同一位置で打ち切られれば一致として扱うが、
  片方だけ打ち切った場合をどう数えるか（推奨: 不一致として T2 に回さず `budget_exhausted` 単独で報告）
* **PP の分割方法**: `--pipeline-parallel=K` は **stage 数（phaseshift bench の均等分割を既定）**にするか、
  **Strata 形で開始レイヤ番号を明示**するか。`auto`（境界探索）はコストモデルが Strata 固有のため後回しでよいか
* **CPU 経路の扱い**: `--device=cpu` は T0/T1 の補助として使うか（CPU f32 は parity-policy の基準だが、
  27B を CPU で回す時間は非現実的 → 4B のみ有効、という運用でよいか）
