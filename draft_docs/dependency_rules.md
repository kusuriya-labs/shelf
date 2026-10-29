# shelf 依存ルールの正本（5 段階の責務分離）

* 対象読者: shelf のコードを書く人（人間・エージェント）と、境界テストを直す人
* 位置づけ: **本書が依存方向ルールの正本**（D0 の成果物）。機械検査は
  `tests/architecture/test_architecture_boundaries.cpp` が担う
* 移行先: 実装完了時に `docs/architecture/dependency_rules.md` へ移す
  （AGENTS.md のドキュメント規則。コードのコメントは移行先のパスを指したままでよい）。
  それまでは**本書が作業中の正本**で、テストのコメントも本書を併記する
* 上位文書: `draft_docs/directory_design.md`（§3.1 設計原則、§3.2 目標ツリー、§4 フェーズ表）

---

## 0. 一文でいうと

**「下位は上位を知らない」「モデル名は `src/model/` の外に出さない」**。
この 2 つを include の文字列と生テキストで機械的に検査している。違反したらテストが落ちる。

---

## 1. 5 段階とディレクトリの対応

| 段階 | 役割 | 現在の担い手 | 目標（再編後） | フェーズ |
| --- | --- | --- | --- | --- |
| 1. 起動 | デバイス probe・設定解決・受理判定 | `apps/shelf_run.cpp` の `main` | `runtime/boot/` ＋ 薄い `apps/` | D1 |
| 2. メモリ確保 | 予算と所在の計算（純粋）→ 実確保 | 実行器のアリーナ・`declare_states`・`upload_weight*` に分散 | `placement/` → `backend/common`（POD）→ backend | D2 |
| 3. モデルロード | 読取 → 検証 → 実体化計画 → upload | `apps/common/app_support.cpp` に集中 | `artifact/` ＋ `weights/` ＋ `quant/` → `runtime/load/` | D3・D5 |
| 4. エンジン起動 | 機能表で受理 → backend 構築 → declare 相 | `HipStepExecutor` の ctor と `declare_states` | `runtime/` ＋ `compiler/`（declare）＋ backend 機能表 | D4・D6 |
| 5. ホットパス推論 | カーネル実行（dequant は GEMM 内） | `backend/hip/kernels` ＋ `src/quant` の契約 | 同左（allocation / sync / 読み戻し禁止） | D5（済） |

所有者とターゲットの対応（`cmake/Targets.cmake`。1 ディレクトリ ≒ 1 ターゲット）:

| ディレクトリ | ターゲット | 段階 | 状態（2026-10-29） |
| --- | --- | --- | --- |
| `include/shelf` | （公開面・ヘッダのみ） | 1 | あり |
| `src/foundation` | `shelf::foundation` | 0（型） | あり |
| `src/artifact` | `shelf::artifact` | 3a | あり |
| `src/quant` | `shelf::quant` | 3b/5 | あり（2026-10-29 新設） |
| `src/weights` | `shelf::weights`（INTERFACE） | 3c | 型のみ |
| `src/ir` | `shelf::ir`（INTERFACE） | 論理 | あり |
| `src/model/qwen35` | `shelf::model_qwen35` | モデル固有 | あり |
| `src/compiler` | 未作成 | 4a declare | 空 |
| `src/placement` | 未作成 | 2a | 空 |
| `src/runtime/state` | `shelf::runtime_state` | 4 | あり（`prefix_key`） |
| `src/runtime/boot` | `shelf::runtime_boot` | 1 | あり（2026-10-29・D1: probe 型と受理判定。run/validate/bench が使用） |
| `src/runtime/load` | `shelf::runtime_load` | 3 | あり（2026-10-29・D3: 検証と実体化） |
| `src/runtime/pipeline` | `shelf::runtime_pipeline` | 5（編成） | あり（2026-10-29: PP の実行を apps から移設） |
| `src/backend/common` | `shelf::backend_common` | 2b/4b | あり |
| `src/backend/cpu` | `shelf::backend_cpu` | 4c 参照 | あり |
| `src/backend/hip` | `shelf::backend_hip` | 4c/5 | あり |
| `src/api` | 未作成 | Server | 未着手 |
| `apps/` | `shelf-run` / `shelf-validate` / `shelf-bench` | 1 | あり |

---

## 2. 機械検査の仕組み

`tests/architecture/test_architecture_boundaries.cpp` が 2 種類の検査を行う。

### 2.1 include スキャン（ルール 14 本）

* `src/<ディレクトリ>` を**再帰的に**走査し、`.h` / `.hpp` / `.cpp` / `.cc` / `.cu` / `.hip` を対象にする
* `#include <...>` / `#include "..."` の**パス文字列だけ**を取り出し、小文字化して**部分一致**で判定する
  （コメント中の `#include` らしき文字列は、行内に `include` と `<`/`"` が無ければ拾わない）
* 「禁止リスト」に 1 つでも当たれば失敗。当たらなかったファイルは合格として数える
  （`passed` は**ファイル数**であり、ルール数ではない）

### 2.2 モデル名スキャン（ルール 15 本目に相当）

* 走査範囲: `src/` 全体と `include/shelf/`
* 対象: ファイル内容の**生テキストを小文字化**して部分一致（**コメントも対象**）
* 禁止語: `qwen` / `llama` / `deepseek` / `mistral` / `phi-`
* 例外: `src/model/` の下だけ（モデル固有の lowering はモデル名を知ってよい）

> このため「DeepSeek V3 と同じ粒度」のような**コメント**も `src/model/` の外では違反になる。
> 由来を書きたいときは「移植元 PhaseShift と同じ」「blockwise 系」のように言い換える。

### 2.3 対象外

* `tests/`・`apps/` は**この 2 検査の対象外**（テストはモデル名を書いてよい）
* ただし対象外は「本番コードに知識を戻してよい」という意味ではない。判断を apps に置く場合は、
  「composition root である」ことをコメントで明示する

---

## 3. ルール一覧（14 本）

「禁止」は include パスの**部分一致**で判定する。`hip/` は HIP ランタイムのヘッダ
（`hip/hip_runtime.h` 等）を指す。

| # | 対象（再帰） | 禁止する include | 理由（なぜこの依存を切るか） |
| --- | --- | --- | --- |
| 1 | `src/foundation` | `shelf/model/` `shelf/ir/` `shelf/artifact/` `shelf/quant/` `shelf/weights/` `shelf/compiler/` `shelf/backend/` `shelf/placement/` `shelf/runtime/` `hip/` | 段階 0 の型（dtype / shape / tensor_id / numeric）。**何も知らない**のが仕事 |
| 2 | `src/model` | `shelf/artifact/` `shelf/quant/` `shelf/backend/` `shelf/compiler/` `shelf/runtime/` `shelf/placement/` `hip/` | モデル固有 lowering は `ir` まで。コンテナ形式・量子化スキーマ・装置・実行計画を知らない（encoding ⊥ layout の前提） |
| 3 | `src/ir` | `shelf/backend/` `shelf/compiler/` `shelf/runtime/` `shelf/placement/` `hip/` | 論理表現は装置も実行計画も見ない（量子化ペイロードも持たない） |
| 4 | `src/artifact` | `shelf/model/` `shelf/runtime/` `shelf/backend/` `hip/` | 「コンテナの事実」だけを持つ。モデル知識なし・実行なし・装置なし |
| 5 | `src/quant` | `shelf/model/` `shelf/backend/` `shelf/runtime/` `hip/` | encoding（数と layout）の契約のみ。GPU 非依存を型で保証する（参照 dequant は `backend/cpu` 側） |
| 6 | `src/weights` | `shelf/runtime/` `shelf/api/` `hip/` | カタログと実体化計画は、実行・Server・GPU を知らない。`artifact/` は許可（reader からカタログを作る） |
| 7 | `src/compiler` | `shelf/runtime/` `shelf/api/` `hip/` | declare 相はグラフ＋計画から実行計画を作るだけ。GPU 型を持たない |
| 8 | `src/backend`（再帰・全体） | `shelf/model/` `shelf/runtime/` `shelf/placement/` | backend はモデル・配置計画・起動を知らない。契約は `backend/common`、配線は `apps/`（composition root）。**`shelf/compiler/` だけは許可**（2026-10-29・D4）: declare 相が決めた活性化の offsets を実行器が受け取る向きで、compiler は backend を include しない（非巡回） |
| 9 | `src/backend/common` | `hip/` | 契約の 1 紙。POD と ABI のみで HIP 型を持たない（P1） |
| 10 | `src/backend/cpu` | `hip/` | 参照（oracle）は GPU を知らない（P7） |
| 11 | `src/placement` | `shelf/model/` `shelf/runtime/` `hip/` `shelf/artifact/` | 予算と所在の**純粋計算**。GPU もファイルも触らない（P3。`artifact` の禁止は 2026-10-29 追加） |
| 12 | `src/runtime` | `shelf/artifact/` `shelf/api/` `hip/` | 起動・ロード・状態は、コンテナ形式・Server・GPU 型を知らない（供給元は抽象化して受ける） |
| 13 | `src/api` | `hip/` `shelf/backend/` | Server 層は HIP 型も実行器も持たない。エンジン境界は仮想 IF 越し（P11。`backend` の禁止は 2026-10-29 追加） |
| 14 | `include/shelf` | `hip/` `shelf/backend/` `shelf/foundation/` | 公開面は起動に必要な最小 API（`status` / `shelf.hpp`）。実装詳細を漏らさない |

補足:

* ルール 8 は `src/backend` 全体に効くため、`backend/hip` も `shelf/model/` 等を include できない。
  9・10 は 8 に**追加**して `hip/` を禁じる（8 の禁止リストに `hip/` は無い: HIP は自分自身）
* `src/model` が `shelf/quant/` を禁じられているのは「モデルは量子化スキーマを知らない」ため。
  FP8 の知識は現状 `src/quant`（契約）と `backend/hip`（実装）と `apps`（checkpoint の宣言解釈）に閉じている

---

## 4. 違反したときの直し方

| 症状 | 直し方 |
| --- | --- |
| `X includes <shelf/backend/...>` — forbidden in `src/model` | 依存の向きを逆にする。モデル側は `ir` の primitive までを出し、実装の選択は `apps`（composition root）に置く |
| `X includes <hip/...>` — forbidden in `src/<下位>` | HIP 型は `backend/hip` の中だけ。下位は POD（`backend/common`）で受ける |
| `X mentions model name "qwen"` outside `src/model/` | コメントも対象。モデル名を消し、「移植元 PhaseShift と同じ」「blockwise 系」のように言い換える。判断が必要なら `apps` に移す |
| 新モジュールを足したらテストが落ちる | §5 の手順でルールを追加する（テストと本書を同時に） |

---

## 5. ルールを追加・変更する手順

1. `tests/architecture/test_architecture_boundaries.cpp` の `rules` 表に 1 行足す
   （`{src / "<dir>", {"<禁止する include の部分文字列>", ...}}`）
2. **同じコミットで本書 §3 の表にも 1 行足す**（対象・禁止・理由の 3 列を埋める）
3. `cmake --build --preset cpu && ctest --test-dir build/cpu -R architecture` で確認
4. 既存コードが違反する場合は、ルール追加とコード修正を**同じコミット**にまとめる
   （ルールだけ先行させると main が赤くなる）

> 現状は 1 と 2 が**手作業**で、片方だけ直してもテストは通る（= 文書が陳腐化しうる）。
> D7 で「本書の表とテストの表の一致」を機械検査に加える予定。

---

## 6. まだ入れていない候補（`directory_design.md` §3.3 の残り）

§3.3 の「新設」項目は **2026-10-29（D7）にすべて適用済み**（§3.3-4 = ルール 11、
§3.3-7 = ルール 13 に追加）。ディレクトリが空でもルールとして先に固定してあるので、
実装が入った時点で自動的に検査される。

判断が要る点:

* ルール 12 は `src/runtime` に `shelf/artifact/` を禁じている。D3 で `runtime/load` が
  「reader → 実体化計画」を担うとき、**reader を `runtime/load` が直接開くのか**
  （= ルール 12 を緩める）、**抽象化した供給元を受けるのか**（= 現行ルールのまま）を決める
* ルール 15（モデル名）の禁止語に、今後モデルを足す場合は `src/model/<name>/` の新設とセットにする

---

## 7. 機械検査の一覧（D7）

| 検査 | いつ走る | 何を見る | 失敗したとき |
| --- | --- | --- | --- |
| 境界テスト（14 ルール＋モデル名） | `ctest -R architecture.boundaries` | include のパス文字列とモデル名 | 依存の向きを直すか、ルールを直す（§5 の手順） |
| モジュールとターゲットの整合 | **configure 時**（毎回） | `src/` のソース配置と `cmake/Targets.cmake` の対応（`cmake/CheckModuleTargets.cmake`） | configure が止まる。`SHELF_MODULE_TARGETS` かソース配置を直す |
| コストモデルのゲート | `ctest -R architecture.cost_model`（GPU 不要） | 1 ステップで触る重みのバイト数・活性化バイト数（行数比例）・primitive 個数 | 基準 `kBaseline*` を実測し直すか、回帰を直す（`test_cost_model --dump` で取り直し） |
| 性能の回帰ゲート（壁時計） | `ctest -R gate.perf`（**opt-in**: `-DSHELF_ENABLE_PERF_GATE=ON`） | `shelf-bench kernel gemm` の `us_p50` を基準ファイルと比較（既定 1.6 倍まで許容） | 回帰を調べる。装置が変わったら `test_perf_gate … --dump` で基準を取り直す |

* 壁時計を既定の ctest に入れない理由: 時間は装置・負荷でぶれ、他の GPU 利用と重なると
  誤検知する。**回帰の主ゲートはコストモデル**（決定論的）とし、壁時計は空いた装置で明示的に回す
* 計測値そのものの記録は `draft_docs/parity_policy.md` §6 と `docs/spec/bench.md` §11.3 が正本

---

## 8. 変更履歴

| 日付 | 内容 |
| --- | --- |
| 2026-10-29 | D0 として本書を作成。既存 14 ルールとモデル名スキャンを正本化し、`directory_design.md` §1.3-5（正本未整備）を解消 |
| 2026-10-29 | D7: §3.3-4（placement→artifact）と §3.3-7（api→backend）をテストに追加。モジュールとターゲットの configure 時検査、コストモデルのゲート、壁時計の回帰ゲート（opt-in）を追加し、§7 に一覧を起こした |
| 2026-10-29 | D4: ルール 8（`src/backend`）から `shelf/compiler/` の禁止を外した。declare 相（`compiler/arena_offsets.hpp`）が決めた活性化の offsets を実行器が受け取る向きで、compiler は backend を include しない（非巡回）。テストの表と同じ変更で更新 |
