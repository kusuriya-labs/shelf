# shelf ディレクトリ設計の方針（5 段階の責務分離に基づく再編計画）

状態: 方針書（設計とフェーズ区切り。ファイル単位の作業分解は着手時に詰める）
参照: `/opt/zen/work/references/` 配下の実装（PhaseShift 系 / radiance / Python 系 / Strata）
関連: `draft_docs/port_strata_cpu_gpu_coop.md`（協調推論の移植計画。本計画と一部フェーズを共有）

---

## 0. この文書の目的

現状のディレクトリ設計は、次の 5 段階が責務分離されるように作ったつもりである。

1. **起動**（CLI / 設定 / デバイス問い合わせ）
2. **メモリの確保**（予算・アロケータ・バッファ）
3. **モデルのロード**（アーティファクト読み取り → 重みの実体化）
4. **指定バックエンドのエンジン起動**（バックエンド選択・実行器構築・カーネル/グラフ初期化）
5. **量子化をホットパスで Dequant して推論**（毎ステップの実行経路）

本文書は、参照実装を踏まえて **最も効率のよいディレクトリ設計** を決め、そこへ
移行する方針（フェーズ区切り・検証・リスク）を示す。

なお **OpenAI API 互換 Server** は 5 段階のどれにも属さない「上位の関心事」だが、
置き場を決めないと 5 段階の境界が Server 側から壊される。本文書では §2.5 で比較し、
§3.5 で置き場と移植方針を決める。

---

## 1. 現状の shelf と 5 段階の対応

### 1.0 開発モデル（2026-10-28 選定・2026-10-29 更新）

| 用途 | ディレクトリ | 形式・容量 |
| --- | --- | --- |
| **exact / 通常実行（正本）** | `models/Qwen3.5-9B` | bf16、4 シャード（ディスク約 19.3 GB） |
| **approx / 量子化検証（製品量子化版）** | `models/Qwen3.5-9B-FP8` | FP8 W8A8（compressed-tensors・float-quantized）、単一 13.52 GB ＋ `model_mtp.safetensors` 0.49 GB（ディスク約 14.0 GB） |
| **製品量子化版（大きい方）** | `models/Qwen3.8-27B-FP8` | FP8 W8A8 blockwise（`quant_method: fp8`、`weight_block_size: [128,128]`、`activation_scheme: dynamic`）、66 シャード・`language_model` ＋ `lm_head` で 29.5 GB（2026-10-29 入手） |

* **両方を開発時点の標準モデルとする**。文書・CLI の例示・デフォルト引数はすべてこれに合わせる
* 2026-10-28 に選定した approx 候補 `models/Qwen3.5-9B-exl3`（EXL3 4bit、7.19 GB）は
  2026-10-29 に在庫から除外した（経緯と実績は `draft_docs/shelf_validate.md` §7「（6）履歴」）
* FP8 版の量子化範囲: `mlp.{gate,up,down}_proj` × 32 層と `self_attn.{q,k,v,o}_proj` ×
  `full_attention` 8 層の **128 Linear**（`F8_E4M3` ＋ `weight_scale` は `BF16` `[out,1]`）。
  `lm_head` / `embed_tokens` / `linear_attn.*` / `visual.*` / `mtp.*` は bf16 のまま
* `text_config` の必須キー充足を確認済み（32 層 / full_attention 8 層 / kv_heads 4 / head_dim 256 /
  vocab 248,320、`layer_types` 32 件）。KV は 1 tok = 32 KiB（`draft_docs/engine_contract.md` §8.2）
* **未測定の値は「未測定」と書く**（例: 9B の rows=1 / 16 / 32 のアリーナ必要量。
  rows=64 は 2,299 MiB と 2026-10-28 に測定済み。FP8 版の VRAM 常駐量も未測定）

### 1.1 現在のツリーと担い手

| 段階 | 現在の担い手 | 状態 |
| --- | --- | --- |
| 1. 起動 | `apps/shelf_run.cpp` の `main`（CLI 解析・`--device` 選択・`index.json` と shard reader の開放・`build_step_graph`）。装置初期化は `HipStepExecutor` のコンストラクタ内部 | app に集約。デバイス問い合わせが backend の ctor に埋没 |
| 2. メモリ確保 | 3 系統に分散: ①実行器のアリーナ（`arena_alloc`、既定 64MiB の定数、毎ステップ reset）②KV 状態の upfront 全量 `hipMalloc`（`declare_states`）③重みの毎テンソル `hipMalloc`（`upload_weight*`） | **予算（budget）の概念が存在しない**。確保量は定数と論理 shape から決まる |
| 3. モデルロード | `artifact/safetensors` reader → `apps` の `read_bf16` / `read_f32` → **同じラムダ内で** `upload_weight*` と `declare_states` を呼ぶ | 読取・実体化・状態確保が 1 ラムダに融合。`weights/catalog` は「実行にもデバイスにも触れない」宣言のみで未接線 |
| 4. エンジン起動 | `apps` の `if (device == "hip") ... else` による排他分岐と、`HipStepExecutorOptions` の組み立て。CPU は `HostStorage` + `ReferenceExecutor` | 選択肢が 2 値（cpu / hip）で、起動可否の契約（機能の実在判定）が無い |
| 5. ホットパス Dequant | **実装済み（2026-10-29）**: `DataType::fp8_e4m3` と `map_dtype("F8_E4M3")`、`src/quant` の fp8 契約（コーデック・断片順レイアウト・粒度）、HIP 側に fp8 WMMA GEMM と活性化量子化カーネル。dequant は GEMM 内（ロード時は無損 relayout のみ）。CPU 参照は fp8 を**明示的に拒否** | 粒度は checkpoint の宣言が正（9B: 重み channel・活性化 token／27B: 重み 128×128 block・活性化 1×128）。宣言に無い粒度は実装しない |

### 1.2 既に効いている分離（維持する）

* **依存方向が機械的に守られている** — `tests/architecture/test_architecture_boundaries.cpp:142-173`
  の include ルールで、`model` は `backend` を見ない、`placement` は `hip/` を見ない、
  `backend/cpu` は `hip/` を見ない等を強制
* **論理と物理の分離** — IR（`ir/`）は量子化ペイロードを一切見ない（`primitives.hpp:129-130`）、
  KV の物理 ABI は `backend/common/paged_kv.hpp` に独立
* **composition root の所在** — 実行器ヘッダに「本体は中立、配線は apps/ が担う」と明記
  （`backend/hip/step_executor.hpp:11`）
* **1 ディレクトリ ≒ 1 ターゲット** — `cmake/Targets.cmake` に 9 ターゲットを明列挙し、
  `src/` 以下に `CMakeLists.txt` は置かない（参照実装と同じ方式）

### 1.3 いま起きている崩れ（再編で直す対象）

1. **段階 1〜4 が `apps/shelf_run.cpp` の 1 つの `main` に同居**（`upload_weights` ラムダが
   「読取 + VRAM 確定 + 状態宣告」を兼務）。参照の PhaseShift でも同型の
   `initialize` 約 406 行が反面教師になっている（§2.1）
2. **予算の型が無い** — 保証されるのは `arena_bytes`（定数）と KV の論理 capacity のみ。
   「空き VRAM を測って決める」段階が実装として存在しない
3. **量子化の置き場が未定** — 契約は `foundation/dtype.hpp` の注記、受け皿は `src/quant`（空）、
   実装は `backend/hip/kernels`（fp8 KV のみ）。3 箇所にまたがり、段階 5 の所在が決まっていない
4. **起動とエンジン初期化の境界が曖昧** — デバイス問い合わせ・アリーナ確保・状態確保が
   実行器 ctor の内部にあり、アプリからは観測できない
5. **境界ルールの正本が未整備** — 境界テストのコメントが
   `docs/architecture/dependency_rules.md` を参照しているが未作成。

---

## 2. 参照実装から学んだこと

### 2.1 PhaseShift 系（`phaseshift-fp8` / `phaseshift-mxfp4`）— 最も近い系統

shelf のカーネルの出所であり、Qwen3 系ランタイムとして最も対応が効く。

**効いている分割**

* **モデル非依存層の切り出し**（新版 `phaseshift-fp8`）: `models/qwen35/` に同居していた
  `state` / `scheduling` / `execution` / `kernels` を
  `src/phaseshift/inference/{state,scheduling,execution}` と
  `src/phaseshift/kernels/{correctness,optimized}` へ移設し、
  「generic 層は `Qwen35` を参照できない」を `test_architecture_boundaries.cpp` で静的強制
  （`docs/developer/architecture.md:40-42`）。旧版（mxfp4）が現状で、新版が整った形
* **重みロードの 4 段契約**: `read → canonical validation → (logical partition) → preshuffle → upload`
  （`docs/developer/tensor_partition.md:13-25`、禁止経路も明文化）
* **`ModelSource` 抽象**: 「重みをどう得るか（local / resident attach）」と「どこに置くか」を
  分離した唯一の strategy（`models/model_source.h:30`）
* **単一アロケータ + 引数渡し**: `GpuArena`（bump / VMM、個別 free なし）を engine が所有せず、
  `create(*arena, ...)` で受け取るだけ。所有と shutdown 順序は文書の表で管理
* **KernelConfig 4 層**: `KernelId → KernelConfig → Selector → Launcher → __global__`。
  段階 4（初期化・選択）と段階 5（実行）を型レベルで分ける
* **app facade / engine は free function + POD**: エンジンを class 化せず、facade が app 層にだけ存在
* **Dequant の所在**: ホットパスの Dequant は **GEMM 内**（`kernels/optimized/linear/fp8.hip` の
  WMMA が E4M3 をハードウェアで消費、scale は累積後に適用 / MXFP4 はレジスタ内 dequant）。
  ロード時の変換は **preshuffle（レイアウト変換であって Dequant ではない）**。
  CPU の Dequant は参照・検証専用で、「canonical payload を production の GPU カーネルが直接読むのを禁止」
  （`docs/developer/canonical_soa_payload.md:19,34`）

**反面教師**

* `Qwen35ComputeRuntime::initialize` 約 406 行が「デバイス問い合わせ → VRAM 照会 →
  `ModelSource` → `GpuArena` → load → pool 生成 → 予算縮小規則 → 実行器構築 → バッチャ」を
  1 関数で兼務し、TP 版がほぼ複製。`MemoryPlan` は anonymous namespace のローカル型で再利用不能
* バックエンド選択 enum が実質 1 値のスタブで、CLI → config → `ExecutorConfig` の 3 段を通過
* HIP graph の初期化が初回 step（段階 5）に落ち、初回レイテンシがホットパスに乗る
* 設定が `ComputeOptions` → `RuntimeConfig` → `ExecutorConfig` → 各 config の 4 箇所に分散し、
  一部が `main` の分岐で再計算される

### 2.2 Python 系（`vllm-sly-radiance` / `paiton-vllm-plugin` / `EAGLE-4`）

**効いている分割**

* **「予算計算」と「アロケータ」の分離**（paiton）— `runtime/core/utils/qwen38_memory.py` が
  manifest から VRAM 必要量を積算して `fits` を先に判定し、実確保は下位に委譲。
  不足は起動前に拒否できる
* **ロード層の API を「無損リパック」に限定**（paiton / radiance）— 有損な Dequant は一切
  ホットパス（`.so` / HIP カーネル）側に閉じ込める。ロード層が量子化スキーマ追加で肥大しない
* **契約を表にして、実在判定で互換性を決める**（EAGLE-4 `HOOK_MAP` / paiton `contracts.py`）—
  バージョン文字列ではなく「要求する機能が在るか」で起動可否を判断し、欠けていれば原因付きで中止
* **デバイス問い合わせとエンジン起動の分離**（radiance `gpu-detect.sh` / EAGLE `kernels/arch.py`）—
  カード数・アーチ・TP 候補の決定を、エンジン非依存の純粋な問い合わせフェーズとして先に立てる
* **ロードの 3 層**（paiton）— 取得（`artifact_resolver`）/ 選別（`model_path` の名前規約）/
  実体化（loader）が別々

**借りない**

* フラットなパッチ群（`patch_*.py` 40 個）と適用順序の暗黙依存（`vllm-sly-radiance`）
* 実行時 import 解決（paiton の `MetaPathFinder` エイリアス）— C++ には対応物がなく、
  静的解析を壊す
* 「1 モジュールがロード・選択・ホットパス・診断を兼務」する巨大ファイル
  （`radiance_mxfp4.py` 862 行、`paiton/runtime/core/model.py` 1100 行、oracle `qwen35.hpp` 1792 行）
* 「エンジン非依存」と宣言して実装で依存を混入させる例（EAGLE `core/` の torch 依存）
* 起動ロジックの bash 化（数百行の entrypoint）。C++ では probe 結果を構造化データにする

### 2.3 Strata（`Strata`）— 前回調査の要点のみ

* 「方略は純粋関数として切り出し、制御・統計・ログは 1 か所に置く」という
  Strata 自身の成功例（`detail::*` 純粋関数、`void* stream` 抽象）はそのまま借りる
* 反面は `generate.cpp` 17,091 行の単一 `main`、既定値が static / env / `setup.py` / CMake の
  4 箇所に分散、コア層が CUDA を直接知る状態（`draft_docs/port_strata_cpu_gpu_coop.md` §2.4）

### 2.4 radiance（`radiance`）— 5 段階を最も強く設計に刻んだ実装

AMD GPU 向け C++/HIP エンジン。「3 種の .so プラグイン（カーネル / アーキテクチャ / 量子化）＋
**C ABI 1 紙のみを共有する単体エンジン**」で、設計書 `spec.md`（1678 行）が §1〜§18 で
起動順序・二相構築・配置・予算を定義している。

**効いている分割**

* **`abi/` = 1 紙の契約** — ヘッダのみ・版付き（`RAD_ABI_VERSION 25`）。
  CMake でプラグインの include を `abi/` のみに限定し、「プラグインは `core/` も `arch/` も見ない」を
  ビルドで強制。全プラグイン → 1 ABI の一方向を grep で検証済み
* **declare / run の二相** — declare 相で全重み・バッファ・op を列挙し arena・bucket table・
  重み台帳を一度だけ作る。run 相は「declare で得た handle しか issue しない」。
  結果として **`core/runtime/arena.h` に `allocate()` を書かない**（段階 2 と 5 を分ける最も強い規約）
* **予算 ⊥ アロケータ** — `core/mem/vram_budget.*` は「推測禁止、印刷された既知量の算術」、
  `Engine::plan()` がディスクに触れない段階として `load_weights()` と分離
* **encoding ⊥ layout** — container（アーティファクト）は「数」だけを正準形で持ち、
  「カーネルが読む並び」はロード時にカーネル側フック（`layout/relayout`）が生成。
  `libr4d/r4d_layout.cpp` は**純ホスト C++（ROCm 不要でテスト可）**
* **量子化の 3 分担** — `libquant`（変換時・オフライン、ロードとホットパスに一切関与しない）／
  カーネル内 dequant（実行時）／core デコーダ（オラクル・検証用）。
  「ロード時に量子化はしない」（`spec.md:640`）
* **lib 群の分担** — `libr4d`（GPU）/ `libavx`（CPU）が同じ op 語彙の供給者、`libref` は
  正しい答えの保存用で最下位保険（明示フラグなしでは配信拒否）
* **バックエンド選択を 1 度だけ** — `core/device/dispatch.cpp` が backend を一度選び全転送、
  vtable 継ぎ目に `#ifdef` を置かない
* **prefix cache（`core/mem/prefix.*`、spec §7.3）** — 128bit の連鎖ブロックハッシュ
  （「同一スパンでも前置きが違えば別ブロック、ヒットはトークン比較で確定」）＋ LRU 削除 ＋
  fork 時の copy-on-write ＋ TTL ＋ `kv_util` 計測 ＋ 予算からの配分（`vram_budget`）。
  **shelf の `runtime/state/prefix_key.hpp`（BlockKey）はまさにこの設計の ABI 言語で、
  実装（pool / refcount / LRU / COW）だけが未着手**（`prefix_key.hpp:11-15`、
  `cmake/Targets.cmake:34`「pool/eviction arrive with batching」）。ブロック粒度も 16
  （libr4d の paged attention 前提）で一致する。さらに**線形アテンション（GDN）用の状態チェックポイント**を
  同じファイルに併設しており、shelf の Qwen3.5 ハイブリッド GDN と直接対応する
* **ホットパスの規約** — 「step path に allocation / sync / 読み戻し禁止」（`docs/IMPLEMENTATION.md`）。

**反面教師**

* `Engine::start()`（`engine_bringup.cpp:3122-`）がデバイス → プラグイン → declare → plan →
  load → Ctx init を 1 関数で通し、`engine_bringup.cpp` は 3695 行。段階 2 と 3 は
  **メソッドでは区切られているが同一ファイル・同一クラス**で、型やライブラリの境界になっていない
* `core/` は意図的に単一 static lib（「.so の境目はプラグイン境界にだけ作る」）だが、その結果
  段階間の境界は include の慣習のみ（ビルドで守られるのはプラグイン境界だけ）
* `struct Config` 約 1000 行・約 375 フラグ、`r4d_rows.cpp` 8874 行など表・設定の巨大化

**借りる価値が最も高い 3 点**: ① 1 紙の契約で切る（shelf では `backend/common`）②
段階間は struct（plan / program）で受け渡す ③ container は encoding のみ、カーネル形は
ロード時に relayout、dequant はカーネル内。

### 2.5 Server 層の比較（PhaseShift / radiance / Strata）— 一次ソースの選定

| 観点 | **PhaseShift** `src/apps/server/` | **radiance** `core/server/` | Strata `serve/server.py` |
| --- | --- | --- | --- |
| 言語・基盤 | Python（aiohttp）。エンジン側は `stdio_server.hip`（C++、985 行） | C++（`cpp-httplib` + `nlohmann::ordered_json`） | Python（`server.py` 8,718 行 + 30 超のモジュール） |
| エンドポイント | `GET /v1/models` `/v1/meta`、`POST /v1/chat/completions`（`/v1/responses` は明示 not-found） | `/v1/chat/completions` と `/v1/completions`（chat template 必須、SSE） | OpenAI（stream / 非 stream）+ Anthropic `/v1/messages` + `/v1/responses` |
| **エンジンとの境界** | **型付き JSONL stdio**（`request_id` / `event` / `code` / **`capabilities` の `pong`**）。ログ文言に依存しない | **pure-virtual `iface.h`**（同プロセス。HTTP 層は `core/sched` `core/text` のヘッダを include しない） | stdio 行プロトコル＋**ログ文言のパース**（蜜結合） |
| トークナイザ / chat template | **`transformers.AutoProcessor`（HF）で解決**。`phaseshift_chat` が codec と thinking タグを担当 | 自前 C++ `core/text/`（**23,870 行**、unicode 正規化データ 8,099 行を含む） | jinja テンプレ + HF |
| 移植量の目安 | Python 3,036 行 ＋ C++ 橋 985 行（雛形あり） | サーバ本体 **9,397 行**（うち `oai.*` 約 5,000 行）＋ トークナイザ 23,870 行 | 8,718 行＋付随多数 |
| テスト | JSONL なのでエンジン無しで応答部をフェイク可能（プロセス・async は要る） | **仮想スケジューラでポート・モデル無し**に全エンドポイントと SSE を検証 | ログ文言依存で脆い |
| 隔離 | プロセス分離 ＋ 言語差により include は構造的に不可能（`check_no_server_deps` で C++ 側も強制） | 同プロセス（ヘッダ規則で強制。step path に virtual なしと明記） | プロセス分離だが観測手段が文字列 |
| 主な依存 | **Python + aiohttp + `transformers`（重い）** | C++ のみ（httplib を vendor） | Python |
| 問題点 | `/v1/completions` 無し。Python を製品依存に追加 | `core/` と同居、media / audio / LoRA まで抱える、トークナイザの規模 | 蜜結合（`draft_docs/port_strata_cpu_gpu_coop.md` §2.4-5） |

**結論（決定）: PhaseShift を一次ソース＝パス A として採用。radiance は条件付きの代替（パス B）**

* **PhaseShift を一次ソースにする**理由:
  1. **型付き JSONL**（`request_id` / `event` / `code` / `capabilities`）で、Strata が陥った
     「ログ文言をパースしてエンジンを観測する」蜜結合に**構造的に入らない**
  2. **トークナイザと chat template を HF で解決できる** — shelf が現在最も大きな空白
     （`--ids` の token id 入出力のみ）を、C++ で 23,900 行書かずに埋められる
  3. **移植量が最小**（約 4,000 行、うち C++ は 985 行の雛形流用が中心）
  4. プロセス隔離と `capabilities` ハンドシェイクは、設計原則 P8（機能の実在で受理判定）と一致
* **radiance は、shelf が Python を製品依存にしない方針の場合の第二案**。C++ 単一言語・
  `src/api` の既存ルール・`/v1/completions` の実装と「仮想 IF でフェイクテスト」は強い。
  ただしサーバ本体 9,397 行 ＋ トークナイザ 23,870 行というコストを伴う
* **Strata は仕様の参照に限定** — Anthropic `/v1/messages` / `/v1/responses` を足すか、
  エラー応答とフィールド名の判断材料に読む。プロトコル（行プロトコル・ログパース）は移植しない
* **持ち込まないもの（共通）** — radiance の `core/` 同居と media 等の周辺機能、Strata の付随機能。
  PhaseShift の `transformers` 依存は **採用する**（§3.5 の決定。軽量構成は不採用）

---

## 3. 目標ディレクトリ設計

### 3.1 設計原則（出所付き）

| # | 原則 | 出所 |
| --- | --- | --- |
| P1 | **契約は 1 紙** — バックエンド境界は `backend/common` に集約し、逆方向依存をビルドで遮断する | radiance `abi/` + `rad_add_plugin` |
| P2 | **段階の境界は型（struct）で作る** — 「呼び出し順の規約」で済ませない。予算・実体化計画・実行計画は値として受け渡す | radiance `Program` / paiton `LaunchPlan`（反面: PhaseShift `initialize`、radiance `Engine::start`） |
| P3 | **予算計算 ⊥ アロケータ** — 必要量の算術は GPU 非依存の純粋関数、実確保のみ backend | paiton `qwen38_memory.py`、radiance `vram_budget` |
| P4 | **declare / run 二相** — バッファ寿命と offsets を declare 相で決め、run 相に allocator を置かない | radiance（`arena.h` に allocate() なし） |
| P5 | **ロード時変換は無損な relayout のみ、有損 dequant はホットパス** — 量子化スキーマの追加でロード層が肥大しない | PhaseShift（preshuffle）、paiton、radiance、vllm-sly の 4 系が一致 |
| P6 | **encoding ⊥ layout** — アーティファクトは「数」の正準形だけを持ち、カーネル形はロード時にカーネル側が生成 | radiance `spec.md` §4.3 |
| P7 | **参照（oracle）と本番を分離** — 正しい答えの実装は最下位に敷き、本番経路とは別のターゲットにする | radiance `libref`、shelf の `backend/cpu` に相当 |
| P8 | **起動（probe）とエンジン起動を分離し、機能の実在で受理判定する** — バージョン文字列ではなく要求機能一覧で判断し、原因付きで止める | EAGLE-4 `HOOK_MAP`、paiton `contracts.py`、radiance ABI 版ゲート |
| P9 | **app は薄く、engine は free function + POD** — 配線は facade に 1 つだけ | PhaseShift `Qwen35ComputeRuntime`、radiance `main.cpp` 39 行 |
| P10 | **1 ディレクトリ ≒ 1 ターゲット、link 図 = 依存方向**。ただし境目は「代入が起きる所」だけに作る（過剰分割の戒め） | PhaseShift `targets.cmake`、radiance `core/` 単一 lib の意図 |
| P11 | **Server はエンジンの外部境界を仮想インターフェースで経る** — 受付と HTTP スレッドにだけ virtual を置き、step path には置かない。エンジンをログ文言や行プロトコルで観測しない | radiance `core/server/iface.h`、PhaseShift の `apps/server` 隔離（反面: Strata のログパース） |

### 3.2 目標ツリー

太字が現状からの移動・新設。段階は §0 の 5 分類。

```
shelf/
├── include/shelf/              公開面（status / shelf.hpp）— 起動に必要な最小 API
├── src/
│   ├── foundation/             段階0の型（dtype / shape / tensor_id / tensor_spec / numeric）
│   ├── artifact/               段階3a コンテナの事実（raw catalog + safetensors reader）
│   ├── quant/                  段階3b/5  encoding の契約（QuantSpec・payload 記述）と
│   │                           参照 dequant の型・契約（CPU 実装は backend/cpu、本番はカーネル内）
│   ├── weights/                段階3c カタログ → 実体化の計画（ロード時変換は無損 relayout のみ）
│   ├── ir/                     論理表現（グラフ / StateDesc）。量子化・装置を見ない
│   ├── model/qwen35/           モデル固有 lowering（ir へ）
│   ├── compiler/               段階4a declare 相: グラフ + 計画 → 実行プログラム
│   │                           （バッファ寿命・offsets。run 相に allocator を置かない）
│   ├── placement/              段階2a 予算と所在の純粋計算（GPU 非依存・依存先は foundation/ir/backend/common）
│   ├── runtime/
│   │   ├── boot/               段階1  デバイス probe と設定の解決（構造化データとして返す）
│   │   ├── load/               段階3  reader → 実体化計画 → backend への引き渡し
│   │   └── state/              既存（prefix_key。KV の論理状態）
│   ├── backend/
│   │   ├── common/             段階2b/4b 契約の 1 紙: MemoryKind/BufferView・paged_kv ABI・
│   │   │                       所在計画の POD・バックエンド機能表・ホットパスの規約
│   │   ├── cpu/                段階4c 参照（oracle）＋将来の本番 CPU 経路
│   │   └── hip/                段階4c/5 GPU 実行・転送・カーネル（dequant はカーネル内、
│   │                           step path に allocation / sync 禁止）
│   └── api/                    （新設 §3.5）OpenAI API 互換 Server 層: スキーマ・SSE・
│                               エンジン境界の仮想 IF。`hip/` を見ず runtime の公開 API のみ
├── apps/                       段階1  薄い main のみ（shelf-run = CLI、shelf-serve = Server）
├── tests/{unit, architecture, support}
└── docs/architecture/dependency_rules.md（境界ルールの正本。
    実装完了時に draft_docs/ から移す）
```

**5 段階の最終マッピング**

| 段階 | 主な担い手 | 境界 |
| --- | --- | --- |
| 1. 起動 | `apps/` → `runtime/boot/`（probe と設定） | probe 結果は構造化データ。エンジンに到達する前に受理判定 |
| 2. メモリ確保 | `placement/`（予算・所在＝純粋）→ `backend/common`（計画の POD）→ backend（実確保） | 純粋計算は GPU 非依存で単体テスト可 |
| 3. モデルロード | `artifact/`（読取）→ `weights/` + `quant/`（正準形・無損 relayout）→ `runtime/load/` | 「読取 → 検証 → 実体化計画 → upload」を型で固定 |
| 4. エンジン起動 | `runtime/` が `backend` の機能表で受理判定 → `backend/*` 構築 → `compiler/`（declare 相） | 選択は 1 度だけ。実行器は plan を受け取るだけ |
| 5. ホットパス推論 | `backend/*/kernels`（**dequant は GEMM 内**）＋ `ir` のグラフ | allocation / sync / 読み戻し禁止 |

### 3.3 依存方向ルール（境界テストへ追加する候補）

既存の 14 ルール（`tests/architecture/test_architecture_boundaries.cpp:142-173`）に次を足す。

1. `src/runtime` は `shelf/hip/` を引き続き禁止（既存）＋ **`runtime/boot` の probe 結果の型は
   `backend/common` に置く**（runtime が HIP 型を直接持たない）
2. `src/quant` は `shelf/backend/` を引き続き禁止（既存）→ 参照 dequant は GPU 非依存で保証される
3. `src/weights` / `src/artifact` は `shelf/model/` を禁止（既存）→ アーティファクトが
   モデル知識を持たない（encoding ⊥ layout の前提）
4. **新設**: `src/placement` は `shelf/artifact/` も禁止（既存は model/runtime/hip）→
   予算計算がファイルを触らない
5. **新設**: `src/compiler` は `shelf/placement/` を含めてよい（既存ルールに backend/runtime/api のみ）
   が、`hip/` は禁止のまま（既存）
6. モデル名（qwen 等）の生スキャンは `src/model/` 以外禁止（既存）を維持
7. **新設**: `src/api` は `shelf/backend/` も禁止（既存は `hip/` のみ）→ Server は
   `runtime` の公開 API を経るだけで、実行器・カーネルを直接知らない

### 3.4 ビルド単位

* `src/` に CMakeLists を置かず、`cmake/Targets.cmake` で列挙する現行方式を維持（PhaseShift / shelf 共通）
* 1 ディレクトリ ≒ 1 ターゲットを原則とし、**例外は「代入（置き換え）が起きる所」にだけ作る**。
  shelf の現状 9 ターゲットに、新設分として `shelf_placement` と `shelf_runtime_boot/load`（統合可）
  を段階ごとに追加する。過剰分割はビルド時間と include 変更コストを増やす（radiance が
  `core/` を単一 lib にした理由と同型）
* テストは aggregate でなくターゲットを明示 link（現行方式を維持）

### 3.5 OpenAI API 互換 Server の置き場と移植方針

**置き場（パスに依存しない共通部）**

* `src/api/`（新設ライブラリ `shelf_api`）に **外部 API 面**を置く。`README.md:18` に `api/` が
  既に記載され、境界ルールも既に存在する（`test_architecture_boundaries.cpp:169-170` は `src/api` が
  `hip/` を含まないことを要求し、`weights` / `compiler` / `runtime` は `shelf/api/` を include 禁止）。
  空だった api 層の実体が、今回の Server である
* `src/api` は **生成エンジンの抽象（submit / cancel / 進行通知 / `capabilities`）**と、その上に乗る
  プロトコル層からなる。**具体的な実行器（HIP / CPU）は `apps/` が束ねて抽象に差し込む**ので、
  「`src/api` が `hip/` を見ない」という既存ルールが破れない
* 実行ファイルは `apps/` の薄い `main`（radiance の 39 行、PhaseShift の staging 方式を併用）

**パス A（決定・採用）: PhaseShift 方式 = 型付き JSONL ＋ Python の OpenAI 面**

| 置き場 | 中身 |
| --- | --- |
| `src/api/` | 生成エンジン抽象 ＋ **JSONL ブリッジの C++ 側**（`request_id` / `event` / `code` / `capabilities` を `nlohmann` で型付け。`stdio_server.hip` 985 行が雛形） |
| `apps/shelf_run --serve-stdio`（または別 exe） | JSONL ループを回す薄い main（PhaseShift が `src/apps/compute/stdio_server.hip` に置くのと同型） |
| `apps/server/`（Python） | OpenAI スキーマ・SSE・chat 処理。`phaseshift_server.py` 585 + `openai_protocol.py` 725 + `chat_service.py` 468 + `compute_client.py` 481 行を移植。CMake の copy + chmod で staging（PhaseShift 同型） |
| トークナイザ / template | Python 側。`transformers.AutoProcessor` を PhaseShift からそのまま借りる（決定、§3.5） |

**パス B（不採用・代替として保持）: radiance 方式 = C++ 同プロセス**

| 置き場 | 中身 |
| --- | --- |
| `src/api/` | `oai.*`（`/v1/chat/completions` `/v1/completions`、約 5,000 行）＋ `http.*` / `sink.*`（SSE）＋ `iface.h` 方式の抽象と `apps/` 側 adapter |
| トークナイザ / template | **C++ で要る**（radiance `core/text/` 23,870 行、または別解）→ パス選定の決め手 |
| 依存 | `cpp-httplib` を vendor。`nlohmann::ordered_json`（プロパティ順がプロンプトとスキーマに効くため ordered） |

**共通の境界規則**

* virtual call（パス B）/ JSONL のディスパッチ（パス A）は**受付・通信スレッドにのみ**置き、
  step path（`backend/*/kernels`）には 1 つも置かない
* Server 層がエンジンを**ログ文言や内部状態で観測しない**（Strata の教訓）
* エラーの写像は `draft_docs/serve_protocol.md` の表に従う
  （`Status::Code` → `code` → HTTP の 3 段。表とテストは 1 対 1）
* テストはフェイクのエンジンで動かす（パス A は JSONL を直接話すフェイク、パス B は仮想 IF のフェイク）。
  ポート・モデル無しで応答列と SSE を検証する（radiance の方針）
* 移植しないもの（共通）: Strata の行プロトコルとログパース、radiance の `core/` 同居と
  media / audio / LoRA 等の周辺機能、Strata の prometheus / MCP / 画像（要件が出た時点で追加）
* prefix の再利用は **自動の最長一致**（radiance 方式＝ shelf の `prefix_key` ABI と同一思想）で行うので、
  リクエストにチェックポイント位置を持たせない（PhaseShift の `prefix_cache_checkpoint_position` は
  **採らない**）。ヒット率と `kv_util` は `stats` で返す

**shelf に無く、Server に要るもの（独立課題として計上）**

1. **トークナイザと chat template — HF `transformers` を使う（決定）** — shelf は token id の
   入出力のみ（`apps/shelf_run.cpp:--ids`）なので、パス A では PhaseShift 同様に
   `AutoProcessor.from_pretrained` を使う。`apply_chat_template`（prompt 構築）と
   `parse_response`（tools / reasoning の応答パース）までそのまま借りられ、
   `phaseshift_chat/codec.py` の移植が素直になる。
   留意点は 2 つ: ① `transformers` ＋ `tokenizers` という重い依存を製品に入れる
   ② torch 無しで `AutoProcessor` が動くか、`trust_remote_code` を使うモデルの扱いを着手前に確認する。
   （別案の `gigatoken` + `jinja2` は依存が軽いが、②③のテンプレートと応答パースが要自前実装のため不採用。
   パス B（C++）ではこの資産が使えない点は、パス選定の決め手のまま）
2. **スケジューラ（submit / cancel / 進行通知）** — **契約は `engine_contract.md` §8.1 の
   「最大 4 並列（スロット = 4）」で確定**。ただし現状実装は 1 シーケンス直列の `run_step` のみで
   `max_sequences = 1` 固定（`engine_contract` §8.7 のギャップ表）。**D8 は 1 並列から開始**し、
   4 並列化（ブロック表の slots 配列化 + step 内マージ）は後続フェーズに委ねる
3. **プロトコル / 依存** — パス A は JSONL スキーマと Python の依存管理、
   パス B は httplib の vendor とスレッド構成

---

## 4. 移行フェーズ（区切りと完了条件）

既存の協調推論計画（`draft_docs/port_strata_cpu_gpu_coop.md` §5）とフェーズを共有する。
**各フェーズは独立してマージでき、境界テストと既存テストが常に通る**こと。

関連文書: **`draft_docs/engine_contract.md`**（Server とエンジンの操作・イベント・
生成パラメータ `temperature` / `top_k` / `top_p` / `min_p` / `seed`・キャンセルの契約 —
D1 と D8 の前提。草案 v0 済み）

| # | フェーズ | 内容 | 完了条件（DoD） |
| --- | --- | --- | --- |
| D0 | ルールの正本化 | 5 段階とディレクトリの対応表を `draft_docs/dependency_rules.md` に書く（境界テストのコメントが参照するのは `docs/architecture/` 側。**実装完了時にそこへ移す** — AGENTS.md のドキュメント規則） | 文書があり、既存 14 ルールの意図と本文書 §3.3 が一致する |
| D1 | 起動の分離 | デバイス probe と設定解決を `runtime/boot/` へ。probe は構造化データを返し、要求機能表で受理判定してからエンジンを起こす。`apps/shelf_run.cpp` は薄く | `main` に「probe → 判定 → 生成」以外の段階が無い。機能不足の起動が原因付きで失敗するテストがある（挙動は現状と不変） |
| D2 | 予算と確保の分離 | 予算・所在の計算を `placement/`（純粋、GPU 非依存）へ。所在は `backend/common` の `MemoryKind` で表し、実確保のみ backend に残す | placement の単体テストが GPU なしで通る。全 device 常駐の既定で性能・結果が不変（= 協調推論計画のフェーズ 1 と同一成果物） |
| D3 | ロードの分離 | `upload_weights` ラムダを `runtime/load/` へ。「読取 → 検証 → 実体化計画 → upload」を型で固定し、無損 relayout のみ許可する | apps から safetensors の直接読みが消える。`weights/catalog` が実体化計画の入口として接線する |
| D4 | declare / run 二相 | バッファ寿命と offsets を declare 相（`compiler/` の受け皿）で決め、run 相から allocator を外す | run 相のコードに割当 API が現れない（radiance の `arena.h` 規約）。ステップ時間が悪化しない（フェーズ 0 の計測と比較） |
| D5 | 量子化の契約 | `quant/` に encoding（数）と**参照 dequant の型・契約**を置き、CPU 実装は `backend/cpu`（境界ルールで `src/quant` は `shelf/backend/` を include 禁止のため）。**有損 dequant はホットパス（GEMM 内）のみ**と規約化。**製品量子化版は FP8 W8A8（2026-10-29 決定。checkpoint `models/Qwen3.5-9B-FP8`。参照: compressed-tensors の `float-quantized` 仕様 / `transformers` の fp8 linear / `references/` の phaseshift-fp8。最初の適用は `shelf-validate` V1**）。低 VRAM 向けの 4bit 級（EXL3 など）は未決定（実測記録は `draft_docs/shelf_validate.md` §7「（6）履歴」）。fp8 KV の codec は `backend/hip` に残すが型の定義は `quant/` へ | 量子化スキーマの追加で動く箇所が `quant/`（型・契約）と `backend` の 2 実装（CPU 参照 / GPU 本番）に限られる（= 協調計画の量子化分岐と統合） |
| D6 | バックエンド起動契約 | `backend/common` に機能表（required/optional）と、選択を 1 度だけ行う dispatch を定義 | 選択が複数回・複数箇所で行われない。未対応機能要求が受理前に検出される |
| D7 | 検証の固定化 | 境界テストに §3.3 の新ルールを追加、1 ディレクトリ = 1 ターゲットの整合確認、性能の回帰ゲート | cpu / hip 両プリセットで全テストが通り、境界テストが新ルールを検査している |
| D8 | Server 層の導入 | **パス A で確定（§2.5 / §3.5）** → `src/api` にエンジン抽象と型付き JSONL ブリッジ、OpenAI 面は `apps/server/`（Python + aiohttp + HF `transformers`）。あわせて `engine_contract.md` §3 の **5 パラメータ（`temperature` / `top_k` / `top_p` / `min_p` / `seed`）をエンジン側に実装**（現状は greedy 専用のため追加が必要） | `/v1/chat/completions` の受付〜ストリーム完了までを**フェイクエンジン**で、ポート・モデル無しに検証するテストがある。エラーは `draft_docs/serve_protocol.md` §3 の表駆動で検証する。`src/api` が `hip/` と `shelf/backend/` を含まないことを境界テストが検査 |

補足:

* **D1〜D3 は挙動を変えずに構造だけ動かす**フェーズとし、差分を小さく保つ（PhaseShift の
  `initialize` が陥った「1 関数に段階を集める」の逆）
* **D4 と D5 が最も工数を見込む**。D4 は `step_executor` の実装方式、D5 は FP8 GEMM
  （gfx1201 の fp8 マトリクス命令の実在）の規模に依存するため、着手前に `shelf-validate` V1 のスモーク
  （1 テンソルの dequant を torch / `transformers` と突合、加えて fp8 命令のミニベンチ）で可否と規模を測る
* 財産の一方通行: 各フェーズで `draft_docs/dependency_rules.md` を更新し、文書と
  境界テストの乖離を残さない（PhaseShift が `docs/developer/*.md` で経験した陳腐化の回避）

---

## 5. 検証方針

* **正しさ（parity）**: 主目標は**出力トークン列の完全一致**（greedy・固定入力・固定シード）。
  ただし WMMA の FMA や累積の順序差による丸めで完全一致しない可能性を前提に置き、
  **不一致時は不一致位置の margin と dtype 誤差の実測値で「丸め由来かバグか」を判定する**。
  判定フロー（T0 決定性 → T1 完全一致 → T2 誤差判定 → T3 条件改訂）・測定手順・改訂手続きの
  **正本は `draft_docs/parity_policy.md`**
* **構造**: `tests/architecture/test_architecture_boundaries.cpp` が新ルールを含めて通ること。
  これが「5 段階の分離が守られている」唯一の機械的保証になる
* **挙動不変のフェーズ（D1〜D3）**: 既存テスト全通過 ＋ フェーズ 0 相当の計測で
  ステップ時間・VRAM 常駐量が変わらないこと
* **純粋レイヤ（placement / boot / quant の算術）**: GPU 不要の単体テスト。CPU プリセットで
  通ることを条件にする（参照の「エンジン非依存層は依存ゼロ」を宣言と実装の両方で守る）
* **量子化（D5）**: 参照 dequant（CPU）とホットパス（GPU）の一致を parity で測る。
  比較ペアの差し替えと許容基準となる誤差の取り直しは `draft_docs/parity_policy.md` §3 の
  手順に従う。「canonical payload を本番カーネルが直接読まない」ことをテストで確認
* **起動（D1/D6）**: 機能不足・予算不足が、エンジン起動前に原因付きで拒否されるテスト
* **Server（D8）**: フェイクエンジン方式で検証する（パス A は JSONL を直接話すフェイク、
  パス B は仮想 IF のフェイク）。応答列・ストリーミング・キャンセル・エラー応答を、
  **エンジンのログ文言や到達不能な内部状態に依存しないテスト**で覆う（Strata が陥った形を避ける）。
  `src/api` の include 検査は境界テストに含める

---

## 6. リスクと未決事項

| リスク | 内容 | 手当て |
| --- | --- | --- |
| include の大規模変更 | ディレクトリ移動は全 TU の include 行に波及し、差分が巨大になる | D1〜D3 は「移動 + 互換ヘッダ」で段階化し、互換ヘッダは次のフェーズで廃止 |
| 過剰分割 | 段階をすべて個別ターゲットにするとビルド時間と認知負荷が増える（radiance が `core/` を単一 lib にした理由） | P10 の原則: 境目は「代入が起きる所」だけ。追加ターゲットは D2/D3 の完了時のみ |
| doc の陳腐化 | PhaseShift が `docs/developer/*.md` と実装のずれを経験済み | D0 で正本を 1 か所に置き、境界テストと対にする（乖離が CI で出る形にするのは D7） |
| 二計画の重複 | 協調推論計画と本計画が同じ成果物（placement / quant / runtime）を指す | 本計画を「構造の骨格」、協調推論計画を「機能の実装」として、フェーズ ID で相互参照（D2 = 協調推論計画 フェーズ 1 など） |
| `apps` の肥大化 | 現状の `main` 367 行が全段階を兼務。放置すると PhaseShift / Strata の 400〜17,000 行 `main` と同じ形になる | D1・D3 で段階を外し、D7 で `main` の行数を回帰対象にする（任意） |
| 量子化導入の時期 | D5 は「ホットパス dequant」を伴い、GEMM の実装変更を伴う可能性 | D5 は量子化フォーマットを実装する時点まで後送可。契約（型の置き場）だけ先に固定する |
| Server の機能肥大 | radiance は Server が media / audio / LoRA / liveview まで抱え `core/` に同居している。写すと同型になる | 移植対象を `/v1/chat/completions` と `/v1/completions` の最小集合に限定し、追加は要件が出た時点で |
| トークナイザ不在 | shelf は token id の入出力のみで、chat template も無い（OpenAI 互換の受付には要る） | パス A は HF `transformers`（**決定**、PhaseShift 同様）。パス B は C++ 自前（radiance `core/text/` 23,870 行）。torch 無しでの `AutoProcessor` 動作を D8 着手前に確認 |
| Python 依存の追加（パス A） | shelf は現状 Python が一切無い（ビルド・テストとも C++）。Server 用に Python・aiohttp・HF `transformers`（＋依存ツリー）を製品依存に加える | 製品方針の確認を D8 の前提とし、依存は Server の exe に限定する（ライブラリと ctest 本体は Python 不要のまま）。torch 無しでの `AutoProcessor` 動作を先に確認 |
| スケジューラ不在（実装） | 契約は最大 4 並列（`engine_contract` §8.1）だが、実装は 1 シーケンス直列（`max_sequences = 1` 固定、`engine_contract` §8.7 のギャップ） | D8 は 1 並列から開始し、4 並列化（slots 配列化 + step 内マージ）は後続フェーズへ。契約と実装の差は §8.7 の表で追えるようにする |

未決:

* `runtime/boot` と `runtime/load` を 1 ターゲットにまとめるか、分けたままにするか
* `compiler/` に declare 相を置くか、`runtime/` の下に置くか（現状 `compiler/` は空で、
  `paged_kv.hpp` の注記は lowering を runtime/compiler の責務としている）
* D4（アリーナ offsets の事前計算）の**実施時期** — 協調推論計画 フェーズ 4 の前か後か。
  実施自体は P4 原則として確定。先送りする場合も「run 相に割当 API を使わない」方針は維持する
* パス A（**確定済み**）の実務確認 — HF `transformers` の依存管理（`aiohttp` と併せ、
  **torch 無しでの `AutoProcessor` 動作**と `trust_remote_code` の扱い）と、
  staging は PhaseShift の copy + chmod を踏襲するか（gigatoken 等の軽量構成は不採用とした）
* パス B に**切り替える場合のみ**: `src/api` のターゲット分割（スキーマ層とトランスポート層）と、
  `cpp-httplib` の置き場（`vendor/` を新設するか `cmake/Dependencies.cmake` に足すか）。
  切替時は `serve_protocol.md` §7 の手順（`code` の表現）を通す
* **prefix cache の移植元と時期** — 本体は **radiance `core/mem/prefix.*` を shelf の
  `prefix_key` ABI 上に移植**（推奨）。Strata は寿命の階層（live → parked → disk）と
  `--prompt-cache-tail`（分岐しやすい位置の追加チェックポイント）だけ参照する。
  pool / refcount / LRU / COW はバッチング期の予定のまま据え置き、
  multi-turn の契約（§8.5 ① で live 保持が確定済み）とセットで時期を決める
* `shelf-run` と Server の設定（モデルパス・量子化・予算）をどこまで共有するか
