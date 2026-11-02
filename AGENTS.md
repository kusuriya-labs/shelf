# AGENTS.md

---

本ドキュメントは、コントリビューターおよびAIコーディングエージェント向けのエンジニアリング哲学を記述する。

リポジトリ構造、ネームスペースマッピング、include規則、CMakeターゲットについては [docs/developer/architecture.md](docs/developer/architecture.md) を参照。

---

# worktree

`<repo>/.worktrees/`

に作成すること。

---

# Language

* すべての回答、説明、コメント、ドキュメントは、自然で標準的な日本語で記述する。
* 中国語および韓国語の語彙、表記、文字を、日本語の文章に混在させない。
* 日本語として一般的でない簡体字、繁体字、ハングルを使用しない。
* 漢字は日本語で一般的に使用される字体を使用する。

  * `说明` ではなく `説明`
  * `处理` ではなく `処理`
  * `确认` ではなく `確認`
* 日本語に存在しない語彙を、中国語や韓国語から推測して生成しない。
* 日本語訳が不明確な技術用語は、無理に翻訳せず英語の原語を使用する。
* 固有名詞、API名、ライブラリ名、ソースコード、コマンド、ファイル名、エラーメッセージなど、原文を維持すべきものはこの制約の対象外とする。
* 出力前に、日本語以外のCJK表記が意図せず混入していないか確認し、混入している場合は自然な日本語へ修正する。

---

# プロジェクト哲学

このリポジトリは学習用プロジェクトである。

主な目標は:

> 特定のハードウェアプラットフォーム向けに、実用的な推論ランタイムを構築すること。

汎用性は目標ではない。性能が目標である。

---

# 対象プラットフォーム

サポートするのは以下のみ:

- AMD RDNA4 GPU
- ROCm
- HIP
- Linux

未サポートのハードウェア向けに抽象化レイヤーを導入しない。

別のプラットフォームのサポートが実装を低速化または複雑化する場合は、それを拒否する。

## R9700 (gfx1201) Spec

- 32 WGP × 2 CU = 64 CU（driver の `multiProcessorCount` は WGP 数）。
- 2 SIMD/CU、32 wf/CU、max workgroup 1024 スレッド、L1 32KB/CU、L2 8MB、L3 64MB。
- streaming read 実測: DRAM 636GB/s（nominal 640 の 99%）、
  L3（32-64MB バッファ）1263-1378GB/s。
- decode の per-token weight traffic は model の weight 全体であり、Qwen3.8-27B の
  BF16 / FP8 いずれでもそのサイズは L3 64MB をはるかに超える。
  かつ各 weight は 1 トークンに 1 回のみ参照 →
  decode GEMV の天井は L3 ではなく **DRAM 640GB/s**。

---

# サポート対象モデル

コード上の runtime family は Qwen3.5 系（`src/phaseshift/models/qwen35/`）である。
これを基盤として、現在の主要 performance target は `models/Qwen3.8-27B-FP8`
（HF fp8）と Qwen3.8-27B の BF16 で、
speculative decode の drafter は DFlash2 を使う。
対応する量子化フォーマットは BF16 / FP8 E4M3 のみである。

- namespace / directory を `qwen35` から rename しない。
- 「Qwen3.5 Dense のみ」という記述で現在の Qwen3.8-27B / DFlash2 開発を
  説明したことにしない。
- capabilities と性能値の正本は README ではなくそれぞれの docs に置く。

モデル固有の最適化は奨励される。

計測可能なオーバーヘッドがゼロでない限り、汎用的なtransformer抽象化は避ける。

---

# 性能優先

常に以下を優先する:

- 単純なkernel
- 少ないlaunch数
- 少ないメモリ割り当て
- 少ない抽象化
- 予測可能な実行

以下を避ける:

- 仮想ディスパッチ
- 不要なtemplate
- ランタイム多態性
- ホットパスでの汎用テンソルライブラリ

---

# メモリ哲学

メモリレイアウトはアルゴリズムの一部である。

コピーを避ける。

以下を優先する:

- 連続アロケーション
- 固定レイアウト
- 事前割り当て済みワークスペース

---

# Kernel哲学

GPU kernelは一級市民である。

CPUの抽象化が効率的なGPU実装を妨げる場合は、CPUの抽象化を再設計する。

非効率なGPU実行を強制するようなAPIを設計しない。

---

# Topology哲学

Schedulerの実行単位は `ScheduledBatch` である。

`ScheduledBatch` はDecode / Prefill requestを同時に含められる。

publicなExecutor / state ownershipはphase別にしない。

phase / work-kind固有のGPU topologyは、executor内部のexecution strategyとして許可する。

Mixed batch内部は必要に応じてdecode region / prefill regionへ分解できる。

scheduler contractへhardware execution topologyの都合を漏らさない。

GPU効率のためなら内部sub-planを作ってよい。

つまり:

> phase固有のtopologyはexecution strategyであり、
> ownershipやschedulingのAPI境界ではない。

将来のscheduling機能もこの設計を維持しなければならない。

---

# 簡潔さ

200行の高速コード

を

2000行の汎用コード

より優先する。

---

# 依存関係

依存関係を最小限に保つ。

以下を導入しない:

- 重量級のframework
- 不要なビルドシステム
- ランタイムコード生成

ネイティブHIP実装を優先する。

---

# 将来の機能

将来の機能は意図的に現在の設計範囲外である。

仮定の将来機能を軸にアーキテクチャを再設計しない。

今日の性能目標を達成する最もシンプルなアーキテクチャを実装する。

---

# ベンチマークルール

すべての最適化は計測可能でなければならない。

直感だけに頼った変更はマージしない。

性能に関する主張には、可能な限りベンチマーク結果を含める。

測定は `docs/perf/methodology.md` の原則に従う。現在の性能値は
`docs/perf/current.md` を正本とし、新しい計測をしていない数値を更新しない。

---

# コーディングスタイル

以下を優先する:

- 読みやすいコード
- 明示的なownership
- 決定論的な実行
- キャッシュフレンドリーなレイアウト

性能挙動を不明確にするcleverなコードを避ける。

---

# 最終原則

このプロジェクトが最適化する対象は1つだけ:

AMD RDNA4 GPU上での高速推論。

それ以外は全て二義的である。

---

# リソースライフサイクルルール

- 所有するGPUリソースには明示的なshutdown経路を用意する
- destructorは最終フォールバック
- cleanup失敗を通常経路で無視しない
- cleanup途中で失敗しても残りを解放する
- cleanup順序を決定的にする

---

# テンソルバッキングルール

- production Tensorをraw pointerから生成しない
- physical deviceを呼び出し側の自己申告にしない
- arena allocation viewから生成する
- ホットパスでpointer attributesを再問い合わせしない

---

# テスト受入ルール

- GPU不足によるskipは開発者ローカルテストでは許容
- required acceptanceではskipを失敗として扱う
- 性能テストと正しさテストを混同しない

---

# リポジトリ衛生ルール

- 非アクティブな実装をリポジトリ内のarchiveとして保存しない
- historyはGitで管理する
- production到達不能なfuture placeholderを作らない
- 外部参照は../references/に限定する

## docs lifecycle

`docs/` は次の役割に分離する。

- `docs/user/` — 現在ユーザーが使用する操作方法のみ
- `docs/developer/` — 現在実装されている architecture / runtime / server / kernel contract のみ
- `docs/perf/` — 現在の性能値、再現条件、benchmark methodology のみ
- `docs/research/` — R&D 記録の正本（実験の経緯・採否判断）。配置規約は `docs/research/README.md`

古い実装を保存する archive directory は作らない。過去のコードは Git history へ任せる。
`docs/wait_review/` のような曖昧な lifecycle directory を残さない。

## document semantic rule

- `docs/developer/` は現在の contract のみを書く。完了した Gate の経緯・PASS/FAIL 履歴・
  過去 benchmark・「次の Gate で行う」・revert 済み方式・採用されなかった候補を残さない。
- Gate 終了時は「current contract」と「R&D record」を分離する。現在も有効な contract は
  現在形で `docs/developer/` へ、経緯と採否判断は `docs/research/` へ任せる。
- performance number は原則 `docs/perf/` にのみ書く。
  developer doc に残してよいのは contract に必要なサイズ・容量・ABI 定数など。
- `docs/user/` に internal implementation history を書かない。
- current docs で同一仕様を複製せず、single source of truth へリンクする。
- test 件数など自動的に変化する値を Markdown の見出しへ hard-code しない。
- commit hash は contract として必要な場合を除き current docs へ書かない。

## tools ルール

`tools/` に残してよいものは、少なくとも次のどれかを満たすものだけ。

- tests / CMake / CI から利用される
- current `docs/user` / `docs/developer` から利用される
- 現在サポートしている artifact 生成に必要
- 現在繰り返し実施する benchmark / qualification に必要

単発 Gate 終了後の script は残さない。R&D 記録からしか参照されない script は、
現在もその workflow が必要な場合を除いて削除する。

---

# コーディングルール

- ドキュメントは日本語で書く
- コードにはコメントを書かない
- 編集前に既存ファイルを読む
- entrypointは `phaseshift-compute` / `phaseshift-launcher` / `phaseshift-server` / `phaseshift-bench` の4つのみ（cmake/apps.cmake）
- publicヘッダのincludeは必ず `phaseshift/` プレフィックスを使う。相対includeや `src/...` パスは使用しない。
