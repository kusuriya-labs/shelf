# Hy4 と OpenCode の接続調査

この記録は実装の完了を意味しない。Hy4 によるツール実行は、現状のサーバーでは利用できない。

## 調査条件

fork 専用ブランチ `hy4-rdma-engram` の b5caf77b を基点とする専用 worktree を使った。OpenCode は v2.3.2。RDMA サーバーが `10.255.254.1:19390` で待機していることを確認した。ローカルは `SHELF_HY4_RDMA=ON` でビルドし、`SHELF_HY4_ENGRAM_SOURCE=rdma`、R9700 と V620 2 台、4 bpw、DFly、`--max-seq-len 2304 --max-concurrent-requests 1` を使用した。既存の `/tmp/r9700.lock`、`/tmp/v620.lock`、`/tmp/v620-76a08.lock` は順にブロッキングで取得し、サーバー停止まで保持した。

## ツール呼び出しの制約

`src/apps/server/serverlib/chat_service.py` の `ChatService.prepare` は Hy4 に `tools` または構造化出力が指定された場合、明示的に `Hy4 はテキスト生成に対応します` と拒否する。モデルの `chat_template.jinja` 自体は `<tool_calls:opensource>`、`<tool_call:opensource>`、`<arg_key:opensource>`、`<arg_value:opensource>` 形式を記述している。現行のツール解析・ストリーミング処理はこの形式への対応を確認していない。

サーバー側で Hy4 のツール付きテンプレート生成と Hy4 固有の呼び出し形式の解析、引数の JSON 化、OpenAI 形式への変換、ストリーミング、複数呼び出し、ツール結果を含む次ターン、エラーを実装して検証する必要がある。単に拒否を外すことは対応完了を意味しない。今回サーバーは改造していない。

## OpenCode 設定

[OpenCode の設定仕様](https://opencode.ai/docs/config/) はプロジェクト設定をグローバル設定より優先する。v2 の[権限仕様](https://opencode.ai/docs/permissions/)では旧 `permission.task` ではなく `permissions` 配列の `subagent` を使う。`XDG_CONFIG_HOME` などを専用ディレクトリに設定した `opencode debug config --standalone` により、プロジェクト内の provider、agent、`subagent` deny の設定が読み込まれることを確認した。これは実際の拒否動作の検証ではない。

実験用設定には 2,304 token の上限を設定した。これは既存の Hy4 実験で確保できた長さであり、OpenCode のツール定義や履歴を含めて実用的かは未検証である。専用サーバーの稼働中に他の GPU 利用者と競合しないよう、サーバーの生存期間も 3 個の共有ロックを保持する必要がある。検証コマンドへのロックの受け渡しを含む制御スクリプトは未実装である。

## 今回の実測と未実施項目

2026-11-19、サーバープロセスの起動から `/v1/models` が応答するまで 132.08 秒。`context_length` は 2,304 だった。各 `card` の `mem_info_vram_used` は応答時に card0 25,053,376,512、card1 25,051,230,208、card2 34,191,564,800 byte、停止約 3 秒後にそれぞれ 19,304,448、17,162,240、59,916,288 byte。起動中に見た compute プロセスの `VmLck` は 0 KiB。サーバーを SIGTERM で終了し、3 GPU のロックを解放した。

短い生成要求は 0.76 秒で HTTP 200 を返したが、16 token の上限内では思考だけを出力し、本文は空だった。`tools` 付きの要求は HTTP 400 と `Hy4 はテキスト生成に対応します` を返した。OpenCode 実行はサーバーが終了するまでに結果を得られず中断した。したがってツール呼び出しの成立、subagent の実際の拒否、同一セッションの再開や prefill の時間は測定できていない。

`hy4-llm`、`with-gpu`、恒久的な OpenCode 設定は未実装。GPU 検証コマンドの失敗・SIGTERM・再起動失敗時の挙動も未検証。起動時間はモデル一覧への応答までであり、短い生成を含む完全なヘルスチェックの時間ではない。グローバル設定配下の通常ファイルの SHA-256 一覧を実験前後に比較し、一致した。
