# Strata の CPU/GPU 協調推論を shelf へ移植する手順書

状態: 設計方針とフェーズ区切り（実装着手前の計画書。ファイル単位の作業分解は着手時に詰める）
参照リポジトリ: `/opt/zen/work/references/Strata`
対象機能: VRAM に入りきらないモデルを **CPU と GPU を同時に働かせ**ながら推論し、
低 VRAM 環境でも速度を落としすぎない仕組み

---

## 1. 目的と、shelf 現状との差分

### 1.1 目的

* VRAM 容量がモデルに足りなくても、CPU RAM 上の重みを CPU で計算して推論を継続する
* 「CPU で代用するだけ」で終わらせず、GPU がキャッシュ分を計算する **時間と重ねて** CPU が
  残りを計算し、合流する（= 協調）ことで、低 VRAM でも速度を出す
* VRAM の残り具合に応じて常駐量を伸縮させる（余った VRAM はキャッシュへ、足りなければ縮める）

### 1.2 shelf 現状（移植前に起きていること）

**協調の土台となるもの**

| 論点 | 現状 | 根拠 |
| --- | --- | --- |
| 重みの所在 | すべて VRAM 常駐。ホスト側の読み込みは一時 `std::vector` で、アップロード後に解放される。退避・部分読込・解放の API なし | `src/backend/hip/step_executor.cpp:96-119`（`hipMalloc` + 同期 H2D）、`apps/shelf_run.cpp:174-194` |
| 同期 | 1 ストリーム（`hipStreamNonBlocking`）。イベント未使用。ステップ末尾で全面同期 | `src/backend/hip/step_executor.cpp:62,235` |
| 中立語彙 | `MemoryKind{host, pinned_host, device, unified}` と `BufferView` が `backend/common` に既にあるが、実行系から未接線（`void*` 直接運用） | `src/backend/common/include/shelf/backend/common/device.hpp:34-65` |
| ストリームの RAII | `Stream` / `Event` / `DeviceBuffer` / `copy_h2d` が整備済みで未使用（ストリーム間依存を表現できる素材は在る） | `src/backend/hip/include/shelf/backend/hip/hip_api.hpp:31-115` |
| MoE の IR | `moe_route` / `moe_merge` の原語と検証あり | `src/ir/include/shelf/ir/primitives.hpp:56-57`、`src/ir/include/shelf/ir/graph.hpp:209-229` |
| MoE の CPU 実装 | 参照実装あり（route = 全 E の dense softmax、merge = 全 E の加重和） | `src/backend/cpu/reference_executor.cpp:221-248`、`tests/unit/test_reference_executor.cpp:124-125` |
| 配置の受け皿 | `src/placement`, `src/compiler`, `src/quant` がディレクトリとして存在（中身は空、ターゲット未定義） | README.md のディレクトリ構成、`cmake/Targets.cmake:9-10`（「空ターゲットは作らない」方針） |

**埋めるべき空白**

| 空白 | 内容 | 根拠 |
| --- | --- | --- |
| MoE の HIP 実装 | `moe_route` / `moe_merge` は `unsupported` の default 分岐に落ちる | `src/backend/hip/step_executor.cpp:721-724` |
| `top_k` の接線 | `Attr::top_k` は宣言のみ。検証・実行・lowering のどこでも読まれない（CPU 参照も top-k 選択なし） | `src/ir/include/shelf/ir/primitives.hpp:100` |
| モデル側の MoE | `TextConfig` に expert 系の欄がなく、`step_graph` の MLP は dense（gate/up/down）のみ | `src/model/qwen35/include/shelf/model/qwen35/text_config.hpp:17-47`、`src/model/qwen35/step_graph.cpp:151-166` |
| 同時実行 | `--device cpu|hip` の排他選択で、1 ステップ内で両者が混在する経路が存在しない | `apps/shelf_run.cpp:98,111,195-265` |
| host↔device の値経路 | デバッグ用 D2H（`fetch_value_debug`）のみ。協調に使う双方向経路なし | `src/backend/hip/step_executor.cpp:241-264` |
| pinned メモリ | `hipHostMalloc` の利用はゼロ | リポジトリ全体の grep 確認 |
| CPU 実装の dtype | CPU 参照は全面 f32 専用。GPU は bf16 活性化 + f32 累積で、境界で変換方針が必要 | `src/backend/cpu/reference_executor.cpp:1`、`src/backend/hip/step_executor.cpp:3-5` |
| KV / ワークスペース | KV は capacity 全量を upfront 割当、アリーナは 64MiB 固定（VRAM 消費の別要因。本計画では対象外） | `src/backend/hip/step_executor.cpp:132-142`、`step_executor.hpp:37` |

差分として持ち込むべきなのは、次の 2 つに整理できる。

1. **重みの所在（residency）を第一級の概念にする** — どの重みがどちらに居るかを型と計画で表す
2. **合流点を持つ並行実行** — GPU 経路と CPU 経路を重ね、1 か所で合流して次へ進む

---

## 2. 参照リポジトリから「借りる」ものと「借りない」もの

Strata は本質的に、**MoE エキスパートを VRAM に適応的にキャッシュし、外れた分を
ピン留め RAM 上で CPU が並行計算する**構造で低 VRAM を成立させている。構成要素は 4 つで、
移植で見るべき対象もこの 4 つに整理できる（出典は §8）。

1. **スロットと在席表** — VRAM の残りでエキスパートを保持。`admit` は eviction なしで、
   レイヤー別クォータを持つ（共有カウンタだとヒット率 2.97%、レイヤー別 8 slots/layer で
   21.4%、64 で 70.4%: `src/program/generate.cpp:1020-1025`）
2. **2 段の合流（HitPhase）** — `Launch`（GPU: 在席分のカーネルを enqueue するだけ）→
   `pool`（CPU: 未在席分だけを GPU と並行計算）→ `Combine`（GPU: 結果を足し合わせる）。
   1 段だけだと drain が 18.2 → 10.2 ms に下がってもトークンが 1 つも速くならなかった
   （GPU が CPU の背後のクリティカルパスに移っただけ）という実測がある
   （`include/strata/core/hit_hook.hpp:13-27`）
3. **ホスト可視化のドアベル** — GPU が mapped pinned に書き、system fence を挟んで
   volatile inc、ホストが volatile で読む。**RDNA4 (gfx1201) では store の後ろにもう一度
   fence が要らないと L2 に留まって届かない**（sync なし 0/100 → fence 後 100/100、
   `src/kernels/cuda/elementwise.cu:206-214`）
4. **CPU プールの完了プロトコル** — `head` / `done` は `fetch_add` のみでロックなし、
   claim は `epoch | njobs | index` の CAS、完了は `done == n` かつ全 worker が park した
   事実（`include/strata/kernels/cpu/pool.hpp:17-31`）

### 2.1 借りる（再実装の対象）

* **常駐の考え方**: 必要な分を VRAM に、残り全部を RAM に。使われた実績から常駐分を選ぶ。
  在席表は eviction なし + レイヤー別クォータ（共有カウンタだとヒット率が急落する、上記 1）
* **2 段の合流**: `Launch` → `pool` → `Combine` の順序と、「1 段では速くならない」という
  測定結果そのもの（上記 2）。順序と測り方を借りる
* **ドアベル（GPU → ホスト通知）**: system fence + volatile 読みのやり方と、
  **gfx1201 では store 後の fence が必須**という実測（上記 3）
* **CPU プールの完了プロトコル**: CAS claim と「完了 = done == n かつ全員 park」
  （上記 4）。CUDA を含まない独立モジュールで、最優先で持ち出せる
* **ピン留め RAM と低 RAM モード**: ラージページ要求（4KB で 830 万 TLB エントリ）→
  取れなければ通常ページに降格して報告、常駐 / mmap / バジェットの 3 択
* **適応ティアの方略**: decayed 使用回数で候補と犠牲者をマッチングし、gain 下限を満たす上位 N 件だけ交換
* **交換（exchange）方式**: 退避前に GPU の内容を交換バッファへ書き戻し、CPU が並行計算、
  着地後に RAM 側の位置を差し替える
* **PCIe share の自動則**: `L / (L + 1.85 R)`、0.05 刻み、[0.10, 0.60]、最大 3 回のプローブ
* **prefill の重ね合わせ**: 専用コピーストリーム + イベントで返すリングを**バイト予算**で管理し、
  イベントは**グループ単位で 1 wait / 1 record**（エキスパート毎だと 10 µs/件の GPU バブル）
* **縮退の考え方**: VRAM 不足・転送失敗でも止まらず、CPU 側で継続する

### 2.2 借りない（Strata 固有・前提が異なる）

* GGUF 量子化と blob の内部レイアウト — `H = 2560`, `FF = 640`, `BLOB = 1,382,400` が
  **コンパイル時定数**で、モデルを変えると全部書き変わる（`expert_source.hpp:18-21`）
* n-gram テーブル（28.8 GB の PLE）、MTP ドラフト / prompt lookup（shelf のモデル前提と異なる）
* `serve/` の行プロトコルと `setup.py` の設定ロジック（§2.4 の 5）
* **CUDA Graph への依存** — shelf のステップ実行器には存在しない。ただし Strata の並列構造は
  「`pre[l]` → ドアベル → `Launch` → `pool` → H2D → `Combine` → `post[l]`」という明示的な
  ステップ列なので、graph は同じ形を繰り返す最適化にすぎず、外しても意味論は保たれる。
  HIP 固有の順序制約のみ §7 のリスクに記載
* CUDA 固有の仮想メモリ管理（`--vram-elastic` は HIP では無効化されている: `expert_cache.cpp:6-8`）
* helper GPU / peer tier / layer split（CUDA P2P 前提。shelf は単一 GPU）
* `STRATA_*` 環境変数 757 種と per-arch 既定値テーブル
* KV ストリーミング（`docs/kv-streaming-design.md` はリポジトリに存在せず、コード注記から復元するしかない）

### 2.3 基本方針

**持ち出すのは「機構・数式・データ構造」だけにして、コードは移植せず shelf の層で再実装する。**
Strata は責務分離ができていない箇所を含み、そのまま写すと shelf の依存方向
（§3）を壊すことになる。ドキュメントの数値・仕様は「目安」であって「保証」ではないことにも注意
（Strata 自身が `docs/VRAM_ELASTIC.md` で "(code) = 測定のない意図の記述」と区別している）。

### 2.4 蜜結合の棚卸し（なぜそのまま移植できないか）

| # | 箇所 | 内容 | 移植時に問題になる理由 |
| --- | --- | --- | --- |
| 1 | `src/program/generate.cpp`（17,091 行、`main` が約 14,800 行） | オプション定義・ヘルプ、VRAM サイズングとリトライ、elastic 可否判定、適応ティアの `adapt` / `apply_pending`、resident スワップ、`VRAM` コマンド、serve 受信ループ、会話キャッシュ・統計出力が単一の `main` に同居 | 方略が `host_res` / `pending` / `xcache` などのローカル変数に依存しており、「何を移植するか」の切り出しが不可能。単体抽出には事実上の再実装が要る |
| 2 | グローバル/静的状態と既定値 | `g_cache_vmm`（`expert_cache.cpp:465-469`）、`g_kv_resident`（`layer.cpp:574`）、prefill の `g_share_default` など、`verify.cpp` はファイルスコープ `const bool` で**初回 `getenv` に固定**。既定値が「static 初期化」「env」「`setup.py`」「CMake option」の 4 箇所に分散 | 移植先で同じ挙動を再現するには 4 つ全部を追う必要があり、既定値の回帰テストが C++ / Python / CMake の境界をまたげない |
| 3 | コア層の CUDA 直依存 | `include/strata/core/` の 50 ヘッダ中 14 個が `cudaStream_t` 等を直接参照、`src/core/` は 51 ファイル中 21 個がランタイムを include、`STRATA_USE_HIP` 分岐が `src/core/` に 65 箇所 | 「コアを HIP 対応にする」のではなく「コアが CUDA を知っている」状態。自前 IR に載せるには interface と implementation を同時に剥がす必要がある |
| 4 | キャッシュとスケジューリング/統計/ログの混在 | `ExpertDispatch`（`expert_source.hpp:485-640`）1 構造体に、キャッシュ参照・GPU 用バッファ・判定の共有状態・統計カウンタ・診断文字列・時間計測が同居。`expert_source.cpp` は 6,388 行で file tier / io prefetch / GGUF blob 組み立て / NUMA / 適応方略の純粋関数群が 1 ファイルに収録され、ログも同じファイルから直出力 | キャッシュの「レイアウト」だけを切り出したい場合でも、統計・ログ・スケジューリングが同一 struct・同一ファイルに結合しており、分離作業が先行する |
| 5 | `serve/server.py` とエンジンの制御 | stdio 上の行プロトコル（`GEN` / `BGEN` / `VRAM` / `DONE` / `BADM`。`DONE` は位置引数 15 個 + key=value）。さらにサーバは**エンジンのログ文言を文字列一致と正規表現でパース**して UI の文言を作る（`server.py:704-756`） | プロトコルにスキーマ・型・バージョンがなく、ログ文言を変えただけでサーバが壊れる。そのまま持ち込むと文字列依存の結合を再現する |
| 6 | テストが文言と private 実装に依存 | `tests/tuning/test_session.py:64` が `"expert cache 3160 slots, 4.26 GiB of VRAM"` を解釈、実装内のテストフック `STRATA_TEST_*` が 27 箇所、`CMakeLists.txt`（2,409 行）がテストを手登録 | モジュールを切り出しても「動いている」ことを検証できない。テストの付け替えコストが無視できない |
| 7 | モデル幾何のコンパイル時定数 | `H = 2560`, `FF = 640`, `BLOB = 1,382,400` が定数で、`n_embd != H` なら拒否 | モデルを変えると全部書き変わる。shelf は `TextConfig` で可変なので持ち込めない |

**逆に手本になっている箇所**（Strata 自身が分離に成功している例）:

* `detail::make_cache_complement_plan` / `choose_resident_keep_from` — 「GPU を初期化せずに
  テストできるよう CPU のみ」で書かれた純粋関数（`expert_source.hpp:132-151`）
* `void* stream` 抽象 — `HitFn` / `note_async_read` / `fill_slot` が CUDA 型を隠している
  （実装側は CUDA 直書きなので、ここは抽象だけを借りる）
* CPU プール（`src/kernels/cpu/pool.cpp`）— CUDA を含まず OS のみ。**最優先で持ち出せる**

**移植への含意**: ① 方略は最初から純粋関数として切り出す、② 制御・統計・ログは shelf の
`runtime` に置き `backend` には置かない、③ 行プロトコルや文言パースは持ち込まない。

---

## 3. shelf への配置方針（責務と依存方向）

アーキテクチャ境界テスト（`tests/architecture/test_architecture_boundaries.cpp:142-173`）が
機械的に守る制約が、そのまま設計の骨になる。特に重要なのは次の要点である。

* `src/placement` は `hip/` を include できない（`:165-166`）→ **placement は純粋な決定**。
  GPU の型・ハンドル・ストリームを触らない。入力（在庫・予算・実績）から出力（配置計画）を作るだけ
* `src/backend` は `shelf/placement/` を include できない（`:161-162`）→ **計画は POD として渡す**。
  計画の型を `shelf/foundation`（または `backend/common`）に置き、placement が作り、
  backend が読む。持ち込む経路は runtime / compiler 側
* `src/backend/cpu` は `hip/` を include できない（`:164`）→ **CPU+GPU をまたぐ合流・待機は
  backend の外側に置く**。backend の中に両方を知る者は存在しない。オーケストレータは
  `runtime`（`backend` と `placement` の両方を include 可能: `:167-168`）が素直
* 計画が渡す所在の表記は、新語彙を造らず `backend/common` の `MemoryKind` / `BufferView`
  （`device.hpp:34-65`、現在未接線）を使う
* CMake は「空ターゲットは作らない」方針（`cmake/Targets.cmake:9-10`）。
  `placement` に実体が生まれるフェーズで初めてターゲットを定義する

各層の責務:

| 層 | 責務 | 入れるもの |
| --- | --- | --- |
| `foundation` | 静的な型と定数 | 所在の表記（device / host 常駐 / host ページング）、予算の単位 |
| `placement` | 配置の決定（純粋関数） | 予算・実績から「どの重みをどちらに置くか」の計画を作る。**所在計画の POD は `backend/common` に置く**（確定、`directory_design` D2 と統一）。include 可は `foundation` / `ir` / `backend/common`（`hip/` と `runtime/` は不可） |
| `ir` / `model` | 論理表現 | MoE を表す契約（`moe_route` / `moe_merge` の意味論）、エキスパートを 1 つの重みとして扱う定義 |
| `compiler`（lowering） | 論理→物理の対応 | 配置計画をグラフ/実行系の指示に変換する場所（現状は空、`paged_kv.hpp` の注記が示す責務） |
| `backend/common` | バックエンド中立の契約 | 転送・常駐・同期の ABI（HIP/CPU 共通の言葉） |
| `backend/hip` | GPU 側の実行と転送 | 常駐重みの確保、未ヒット分の H2D、ストリームとイベント |
| `backend/cpu` | CPU 側の実行 | ホスト常駐重みをその場で計算する経路（リファレンスから性能経路へ） |
| `runtime` | 組み立てと実行順序 | placement の呼び出し、計画の backend への引き渡し、ステップの進行 |

### 3.1 合流点の置き方

* 1 ステップの中で「GPU がやる分」と「CPU がやる分」を分け、**合流点（moe_merge 相当）で両方を待つ**
* 合流点はバックエンド中立にせず、`backend/hip` 側に実体を置き、
  どの部分まで待ったかをバックエンドの内部事情として隠す（ir / placement は同期の型を知らない）
* **分担の契約**: CPU 側は在席（= GPU 担当）分の行を 0 で埋めて GPU に譲り、ルーター重みの適用は
  合流側に一本化する。二重に掛けると答えが壊れる
* **判定の時点**: どの分担で計算するかは GPU 側の enqueue（`Launch`）と同時刻で決める。
  後から判定を後ろ倒しにすると、前のレイヤーの分担リストで計算してしまう
  （参照実装で mean KL が 1e-1 桁 → 1e0 桁に悪化した実例あり）
* ステップ末尾の全面同期（`step_executor.cpp:235`）は維持したまま、**ステップ内の重なり**を足すのが
  最初の目標。先に細かい非同期化をしない

---

## 4. 設計上の論点（詰めるべき順）

1. **所在の表現** — 重み ID → {device, host 常駐, host ページング}。決定は placement、
   解釈は backend。現在は「常に device」固定なので、ここが最初の型変更
2. **予算** — VRAM 予算をどこから得るか（空き容量の測定、明示指定、既定値）。RAM 予算と合わせて
3. **ピン留めメモリ** — 現状 shelf に `hipHostMalloc` の利用はゼロ。H2D が pageable のままだと
   転送が CPU 側をブロックし、重なりが出ない。導入するか・いつからかを決める
4. **並行実行の形** — CPU 側のワーカー群（何スレッドか、アフィニティ）と、
   GPU ワークとの重ね方（イベントで待つか、合流点で待つか）。RAII の `Event` は整備済みで未使用
   （`hip_api.hpp:43-45`）
5. **先読み** — 未ヒットした重みの転送を、次の層の計算に重ねられるか
6. **数値の境界** — CPU 参照は f32 専用、GPU は bf16 活性化 + f32 累積。合流点でどの型を受け渡すか
   を決める（ここが曖昧だと parity が取れない）
7. **数値の等価性** — 主目標は**出力トークン列の完全一致**。WMMA の FMA・累積順序の差は前提に置き、
   許容するか否かは **dtype 誤差の実測（`draft-docs/parity-policy.md` §3）**から決める
8. **縮退** — 転送失敗・VRAM 不足時に CPU 全件へ落として継続する（止まらないこと）
9. **量子化** — 現状 shelf は bf16 中心。低 VRAM の主目的に対して容量効率が弱い。
   `src/quant`（空）を本機能と同時に作るかどうかは、フェーズ 4 の後に判断する分岐

---

## 5. フェーズ計画（区切りと完了条件）

各フェーズは独立してマージでき、後段が前に依存する。**境界テストと既存テストが常に通る**こと。

| # | フェーズ | 目的 | 完了条件（DoD） |
| --- | --- | --- | --- |
| 0 | 基準値の取得 | 変化を測れるようにする | 現構成の tok/s・VRAM 常駐量・ステップ時間を記録した計測手順が文書化され、再現できる |
| 1 | 所在の型と計画の導入 | 重み所在を第一級概念にする | 所在の型と配置計画の POD が `foundation`（または `backend/common`）にあり、placement が純粋関数として計画を作る（この時点で placement の CMake ターゲットを新設）。挙動は現状どおり全 device のまま（=性能・結果不変） |
| 2 | MoE の実行系 | 協調対象が実在する | `TextConfig` に expert 欄が加わり、`step_graph` が MoE MLP を組み、`top_k` が検証と実行の両方で意味を持つ。`moe_route` / `moe_merge` が HIP で動き、CPU リファレンスと一致する |
| 3 | CPU 常駐重みの計算経路 | 「置かなかった分」を計算できる | ホスト常駐の重みを CPU で計算する経路が動き、host↔device の値経路と数値境界（f32 ↔ bf16）の方針が確定する。装置を問わず同じ答えになる（parity テスト追加）。pinned メモリの要否もここで決める |
| 4 | 並行実行（協調そのもの） | GPU と CPU を重ねる | 1 ステップ内で両経路が重なり、合流点で正確に待つ。イベント時刻から重なりが計測できる（ステップ時間の内訳） |
| 5 | 適応的な配置 | VRAM に合わせて在庫を選ぶ | 実績から常駐対象を選び直す policy が入り、予算変更に追従する（縮伸できる） |
| 6 | 縮退と障害 | 止まらない | 転送失敗・VRAM 不足で CPU 全件に落ち、推論が継続するテストがある |
| 7 | 検証と性能の固定化 | 回帰を防ぐ | parity・境界・性能の測定を CI 相当の流れに載せ、低 VRAM 構成での目標値を文書化。依存ルールの説明文書（`draft-docs/dependency-rules.md`、境界テストのコメントが参照）も整備 |

補足:

* **フェーズ 1 は挙動を変えずに型だけ足す**ことで、後段の差分を小さくする
* **フェーズ 2 がクリティカルパス**。shelf は現状 dense/GDN のモデルグラフで MoE を組んでいない
  （`src/model/qwen35/step_graph.cpp` に該当なし）ため、協調推論の前提になる
* **KV の低 VRAM 化 — 方針は `draft-docs/engine-contract.md` §8.1 / §8.3 で確定済み**
  （KV 正本 = PinnedHostRAM、VRAM は online 分のみ。正本 4.00 GiB / online 2.50 GiB、
  admission は pinned 容量に対して 2 段で行う）。本計画が担うのは**実装の接続**
  （tier 移動・pinned 正本・載せ替え）で、運用の詳細は `draft-docs/memory-budget.md` へ委ねる。
  Strata でもエキスパートと KV は別の問題として扱われている
  （Strata `docs/VRAM_ELASTIC.md` は「動くのは expert cache のみ」と明記）

---

## 6. 検証方針

* **正しさ**: 主目標は**出力トークン列の完全一致**（greedy・固定入力・固定シード）。
  ただし WMMA の FMA や累積の順序差による丸めで完全一致しない可能性を前提に置き、
  **不一致時は不一致位置の margin と dtype 誤差の実測値で「丸め由来かバグか」を判定する**。
  判定フロー（T0 決定性 → T1 完全一致 → T2 誤差判定 → T3 条件改訂）・測定手順・改訂手続きの
  **正本は `draft-docs/parity-policy.md`**。既存 parity
  （`tests/unit/test_reference_executor.cpp`, `tests/unit/test_hip_step_executor.cpp`,
  `tests/unit/test_hip_attention.cpp`）の作り方を踏襲する
* **境界**: `tests/architecture/test_architecture_boundaries.cpp` が通ること。
  placement が HIP 型を触っていないか、backend が placement を include していないかは
  このテストが機械的に担保する
* **性能**: 低 VRAM 構成（予算を絞った場合）での tok/s、VRAM 常駐量、CPU 使用率、
  そして **重なり**（GPU 区間と CPU 区間が重なっているか）を測る。フェーズ 0 の計測手順を共用する
* **縮退**: 予算を 0 に近づけたときに速度が落ちるだけで止まらないこと

---

## 7. リスクと未決事項

| リスク | 内容 | 手当て |
| --- | --- | --- |
| MoE 未導入 | shelf は dense/GDN 前提。協調対象（エキスパート）が現状グラフに存在しない。`top_k` は宣言のみ未接線 | フェーズ 2 を独立させ、先に MoE 単体の正しさを固める |
| 量子化の空白 | 低 VRAM の主目的に対して bf16 の容量効率が弱い（`src/quant` は空） | フェーズ 4 後に判断。先行させると範囲が膨らむ |
| CPU 性能 | Strata は AVX-512/AVX2 + ggml の実装を持つが、shelf に CPU 性能経路は存在しない | フェーズ 3 は「正しい経路」、高速化は後続 |
| 数値の非対称 | CPU 参照は f32 専用、GPU は bf16 活性化。合流点の型が未定だと parity が取れない | フェーズ 3 で境界の方針を先に確定させる。判定自体は `draft-docs/parity-policy.md`（不一致位置の margin と dtype 誤差の比較）に従う |
| pinned 不在 | `hipHostMalloc` 未使用のため、転送が CPU をブロックし重なりが出ない可能性 | フェーズ 3 で要否を決め、フェーズ 4 の計測で判定する |
| 同期の設計ミス | 合流点の待ちが甘いと結果が壊れる（racy）。判定時点を後ろ倒しにすると前のレイヤーの分担で計算する（§3.1） | 合流点を 1 箇所に集約し、判定は enqueue と同時刻に。境界テストと stress で担保 |
| 参照の数値を持ち込む | Strata の tok/s は Strata のモデル・量子化・機械前提で、そのまま目標値にならない | 目標は自前のフェーズ 0 計測から決める |
| 蜜結合の移植 | 参照実装の制御・統計・ログ一体構造を写し込むと依存方向が壊れる | §2.3 の方針（再実装）を守り、境界テストで検知する |
| 依存ルールの文書不足 | 境界テストのコメントが `docs/architecture/dependency-rules.md` を参照しているが未作成。正本は `draft-docs/dependency-rules.md` に置き、実装完了時に移す（AGENTS.md 規則） | フェーズ 7 で整備（設計判断の根拠を残す） |
| ドアベルのホスト可視化 | gfx1201 では mapped pinned への plain store が L2 に留まり、sync までホストに届かない（Strata の実測: sync なし 0/100 → fence 後 100/100） | host-visible な flag → system fence → volatile 読みを明示的な手順として実装し、単体テストで検証する |
| HIP のストリーム間依存 | 共有エキスパートの fork/join は CUDA では既定オン、**HIP では既定オフ**（gfx1201 で 62.9 vs 43.4 tok/s） | 1 ストリームで順に並べるのを既定にし、event は合流点だけに張る |
| VRAM の伸縮（VMM） | Strata の `--vram-elastic` は HIP では無効化されており（CUDA driver API 依存）、HIP の VMM API の可用性も未確認 | 本計画では必須としない。必要が出た時点で別課題として検証する |
| CUDA Graph の不在 | Strata は CUDA Graph で反復を最適化しているが、それは最適化であって意味論ではない | 明示的なステップ列として書ける。ただし HIP 固有の順序制約（初回 launch 前に flag を上げないとデッドロックする例がある）は自前実行器でも再現を要検討 |

決定済み（旧未決から解消）:

* 計画（residency plan）の型の置き場 — **`backend/common`**（本計画 §3 の `placement` 行と
  `draft-docs/directory_design.md` D2 / 目標ツリーと統一。`MemoryKind` / `BufferView` が
  既にそこに在ること、placement が include 可能であることが根拠）

未決:

* CPU ワーカーの所有者（runtime / backend/cpu のどちらがスレッドを握るか）
* KV の低 VRAM 化の**実装時期** — 方針は `draft-docs/engine-contract.md` §8.1 / §8.3 で
  確定済み（正本 = pinned）。pinned 正本・tier 移動の実装は本計画の後続フェーズ
  （§5 補足）と `draft-docs/memory-budget.md` に委ねる

---

## 8. 出典（参照した資料）

参照リポジトリ: `/opt/zen/work/references/Strata`

* 機構と測定値の一次情報: `docs/HOW_IT_WORKS.md`（全体像）、`docs/DETAILS.md`（技術参照・計測値）、
  `docs/VRAM_ELASTIC.md`（セグメント shrink/grow と HIP 非対応）、`docs/MODELS.md`（low-RAM モード選択）、
  `docs/MULTI_GPU.md` / `docs/SECOND_GPU.md`（layer split / helper GPU）、
  `docs/DISJOINT_EXPERT_CACHE.md`、`docs/KV_PREFETCH.md`、`docs/AMD_HIP.md`（RDNA4 節）
* 実装（§2.4 の棚卸しの根拠）: `src/program/generate.cpp`、`src/core/expert_cache.cpp`,
  `src/core/expert_source.cpp`, `src/core/session.cpp`, `src/core/layer.cpp`,
  `src/prefill/prefill.cpp`, `src/kernels/cpu/pool.cpp`,
  `include/strata/core/hit_hook.hpp`, `include/strata/kernels/cpu/pool.hpp`,
  `include/strata/core/expert_source.hpp`
* shelf 側: `README.md`, `tests/architecture/test_architecture_boundaries.cpp`,
  `CMakePresets.json`, `cmake/Targets.cmake`,
  `src/ir/`, `src/model/qwen35/`, `src/backend/`, `apps/shelf_run.cpp`,
  `tests/unit/test_reference_executor.cpp`, `tests/unit/test_hip_step_executor.cpp`
