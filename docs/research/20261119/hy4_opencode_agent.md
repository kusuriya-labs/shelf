# Hy4 を使う OpenCode エージェント

fork 専用ブランチ `hy4-opencode-agent` の実験である。既存 worktree とグローバルの OpenCode 設定は変更しない。Hy4 4 bpw を R9700 と V620 2 台で実行し、Engram は `10.255.254.1:19390` の RDMA サーバーから読む。サーバー側の構築と接続先は [RDMA Engram の記録](hy4_rdma_engram.md) を参照する。リモート側の Engram は事前に登録済みであることが必要。

## 準備と起動

この worktree のルートで次を実行する。`/tmp/shelfvenv` はこの測定環境にある、Transformers などのサーバー依存を導入済みの Python 環境である。LLVM libc++ のパスもこのホストの既存環境を指定している。

```sh
cmake -S . -B build-hy4-agent \
  -DSHELF_HY4_RDMA=ON -DCMAKE_BUILD_TYPE=Release \
  -DSHELF_ROCM_ROOT=/opt/rocm/core-7.14 -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_CXX_COMPILER=/opt/rocm/core-7.14/lib/llvm/bin/clang++ \
  -DSHELF_LIBCXX_LLVM_ROOT="$HOME/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23"
cmake --build build-hy4-agent --target shelf-compute shelf-server-stage -j 4
tools/hy4_agent/hy4-llm start
tools/hy4_agent/hy4-llm status
tools/hy4_agent/opencode-hy4 --standalone
```

サーバーの Python 実装を変更した場合は `shelf-server-stage` を再実行する。終了時は `tools/hy4_agent/hy4-llm stop`。`restart` も利用できる。状態とログは `.hy4-agent/` に保存する。既定のモデル・Engram ディレクトリ・配置表・DFly は `tools/hy4_agent/hy4-control.py` に記載したこのホストのパスである。ポートは `127.0.0.1:18123`、model ID は `Hy4-flash-4bpw`。

`opencode-hy4` は XDG の config / data / cache / state をこの worktree 内へ向け、`opencode.json` をその専用 config にコピーする。`OPENCODE_DISABLE_PROJECT_CONFIG=1` により自動検出のプロジェクト指示を読み込まず、Hy4 の短い system 指示を専用設定に置く。設定は OpenCode v2.3.2 の [provider](https://opencode.ai/docs/providers/) と [permissions](https://opencode.ai/docs/permissions/) 形式。`subagent` を deny し、組み込み subagent も無効にする。これは shell の OS サンドボックスではないため、worktree 外への書き込みや push をしないという指示も守る必要がある。

## GPU の明け渡しと復帰

```sh
tools/hy4_agent/with-gpu --timeout 3600 -- コマンド 引数...
```

`--timeout` は秒で、既定は 86400 秒。管理プロセスは `/tmp/r9700.lock`、`/tmp/v620.lock`、`/tmp/v620-76a08.lock` をこの順でブロッキング取得し、サーバーの稼働中も保持する。検証コマンドも同じ管理プロセスの下で実行するため、3 個のロックは停止・検証・再起動の間ずっと保持される。サーバーと検証コマンドにもロックのファイル記述子を継承させる。`with-gpu` はサーバーのプロセスグループを停止し、各 GPU の VRAM が取得直後の使用量から 128 MiB 以内へ戻ることを確認してからコマンドを実行する。コマンド終了後も VRAM を確認し、サーバーのモデル一覧と短い生成が通るまで待つ。コマンドの標準出力・標準エラーと終了コードを保持する。SIGINT / SIGTERM と timeout はコマンドのプロセスグループに転送し、終了後に再起動する。再起動の失敗は非ゼロで報告する。

Hy4 には GPU を使うコマンドを必ず `with-gpu` 経由で呼ばせる。Hy4 自身はサーバーを直接操作せず、長い出力はファイルへ保存する。管理プロセスが失われた場合は `hy4-llm stop` が記録された PID とこの worktree のサーバーパスを照合して停止を試みる。SIGKILL や OS 障害では自動復帰できないため、状態とログを確認する。

## コンテキストとツール形式

`--max-seq-len` と OpenCode の context 上限はともに 8192、出力上限は 1024。3 GPU の実機でこの長さの予約と短い生成を確認した。OpenCode の通常タスクで送信されたツール定義を含む入力は 2113 token、再起動直後の会話履歴は 2415 token だった。自動 compaction は有効、保持 1024 token、buffer 25% とした。長い作業での十分性や compaction の品質は未測定。

Hy4 のテンプレートは `<tool_calls:opensource>` と `<tool_call:opensource>`、`<arg_key:opensource>`、`<arg_value:opensource>` を使う。サーバーは Hy4 のテンプレートへ `tools` と過去のツール結果を渡し、この形式を OpenAI の `tool_calls` に変換する。通常応答と SSE の両方に対応する。SSE の Hy4 ツール応答は生成全体を保持してから返すため、reasoning も逐次表示されない。`tool_choice` の auto / none / required / 指定関数に対応する。strict tools と構造化出力は未対応であり拒否する。

## 実測

2026-11-19、OpenCode v2.3.2、RDMA、4 bpw、DFly、逐次 1 request。3 個の共有ロックを保持して実施した。初回の 2304 token 条件ではサーバー起動から `/v1/models` 応答まで 132.08 秒だった。8192 token 条件では、OS キャッシュが温まった後の起動・短い生成を含むヘルスチェックが 40.40 秒、OpenCode の GPU 検証後の再起動が 40.38 秒だった。起動時間はキャッシュ状態に依存する。RDMA 側の Engram は常駐しており、ローカルへの全表読み込みは行わない。RDMA 接続・個別 READ の時間はこの測定では分離していない。compute プロセスの `VmLck` は 0 KiB と確認した。

OpenCode の短いタスクで Hy4 が `shell` を使い `head -n 1 README.md` を実行し、結果を回答した。OpenCode 経由で `with-gpu` から `rocminfo` を実行し、`GPU_OK` を得た。サーバー再起動後、同じセッションの tool 結果を受けた 2415 token の再 prefill は 4.371931 秒。GPU ツール呼び出しは 45.25 秒、ターン全体は 58.18 秒だった。次ターンでも README の行と `GPU_OK` を再現した。`task` による subagent 起動を指示すると、Hy4 は `task` が存在せず起動できないと回答した。これはモデルが実際に禁止されたツール呼び出しを試みた試験ではない。

検証中の card0 / card1 / card2 の VRAM は 19,304,448 / 17,162,240 / 59,916,288 byte。以前の起動時にはそれぞれ 25,053,376,512 / 25,051,230,208 / 34,191,564,800 byte を観測した。シェルから `rocminfo` を実行した正常系に加え、終了コード 37、SIGTERM の 143、サーバー実行ファイルを一時退避した再起動失敗時の非ゼロとエラー表示を確認した。`~/.config/opencode` 配下の通常ファイルの SHA-256 一覧は作業前後で一致した。

追加の同一セッション検証では、Hy4 が `with-gpu` 経由で R9700 の HIP kernel を起動し、32 要素の計算結果を照合して `GPU_KERNEL_OK` を受け取り、再起動後に回答した。ツール呼び出しは 45.849 秒、ターン全体は 63.343 秒、復帰後 3440 token の prefill は 6.471893 秒。最初の試行はテストバイナリの HIP 共有ライブラリ検索パス不足で 127 を返したが、その失敗後もサーバーは復帰した。検索パスを修正したバイナリで再試行して成功した。

最終版の管理プロセスでも SIGTERM を再試験し、終了コード 143 とサーバー復帰を確認した。実行ファイルの一時退避で再起動を失敗させた試験は終了コード 1 と明示的なエラーを返し、ファイルは直後に復元した。制御コードの失敗・SIGTERM・timeout・再起動失敗と、Hy4 の非ストリーム / ストリーム変換などの単体テストは 11 件成功した。

別の作業が GPU を保持している間に `start` を待機させ、`stop` で待機中の管理プロセスが終了することも確認した。これにより、待機の後で意図せずサーバーが起動することを防ぐ。

追加計測では `SHELF_HY4_PROFILE_EP=1` を指定してサーバーを起動した。ヘルスチェック完了まで 40.42 秒。続けて同じ OpenCode セッションを再開した 3568 token の prefill は 6.725128 秒で、`profile_engram` の prefill 行取得区間は 148.048 ms だった。この値は先読み後の `rows` 呼び出しと待機の合計であり、非同期 RDMA READ の全経過時間ではない。先読みが GPU 処理と重なり、profiling による負荷も含むため、非 profiling 実行と厳密に比較しない。計測時の compute プロセスは `VmLck=0 KiB`、`VmPin=7840 KiB`。計測後にサーバーを停止した。

## 状態退避を統合した運用

`hy4-agent-integrated` では上記のビルド手順で状態退避を含む。`hy4-llm start` は既存の GPU ロック取得後、`10.255.254.1:19391` に状態サーバーがなければ、ローカルでバイナリを構築し `user@192.168.0.148` に転送して起動する。SSH の鍵認証と `c++`、libibverbs、OpenSSL が必要。1 世代 512 MiB の領域を 2 個使う。既存の 19391 番を使用した場合は停止しない。19390 番の Engram サーバーは操作しない。状態サーバーの内容はメモリ上のみであり、停止すると失われる。

compute には `SHELF_HY4_SNAPSHOT=1` を設定する。通常停止および `with-gpu` の停止時には `POST /admin/snapshot` で最後に完成した保存候補を再送する。退避に失敗しても警告を残し停止を続ける。GPU コマンドの間は状態サーバーを維持し、GPU ロックも保持する。再開後の履歴の token 列が保存済み prefix と一致し、fingerprint 等の照合にも成功した場合のみ、その部分の prefill を省く。状態サーバーが利用できない場合や照合に失敗した場合は通常 prefill に戻る。保存境界や制約の詳細は [状態退避の記録](hy4_state_snapshot.md) を参照。

各起動の記録は `.hy4-agent/diagnostics/日時/` に置く。`server.log` はサーバーと compute の標準エラー、`compute-stdout.jsonl` は compute の標準出力、`server-exit.txt` と `compute-exit.txt` は終了コードとシグナル、`maps-*.txt` は実行中に取得したメモリ配置である。`environment.json` は明示的に選んだ環境変数のみを記録し、`libraries.txt`、`hip-version.txt`、`gpu-before.txt` は起動時の環境を示す。異常終了時はカーネルログ、GPU 状態、coredumpctl の結果を保存し、取得可能な core があれば `compute-core/gdb.txt` または `server-core/gdb.txt` に `bt full` と `info sharedlibrary` を記録する。`core-settings.txt` は core 上限、`core_pattern`、Apport 設定を記録する。core を取得できない場合は `core-unavailable.txt` と取得コマンドの出力を確認する。ログには会話内容やメモリアドレスが含まれ得るため、ディレクトリは所有者のみアクセス可能にしている。

このホストでは `core_pattern` は Apport のパイプで、`coredumpctl` は存在しない。Apport の設定は有効だが、実際に core を取得できるかは異常終了時の Apport の処理と権限に依存する。既存の SIGSEGV の原因調査は行っていない。

### 統合後の実測

2026-11-19、3 GPU、4 bpw、DFly、OpenCode の同一セッションで実施した。記憶する内容を含む履歴の要求は 5,112 token で、`with-gpu -- rocm-smi --showproductname` の前に 22,218,528 byte の管理 API による退避が成功した。再起動後の次ターンは入力 5,189 token のうち 4,096 token を復元し、履歴中の合言葉に正しく回答した。このターンの prefill は 2.883 秒、復元処理は 0.350 秒だった。

同じ OpenCode の次ターンの HTTP リクエストを保存して再送し、最初の compute token を受信するまでの時間を測った。保存済み prefix が 5,120 token の復元ありは 0.841 秒、`SHELF_HY4_SNAPSHOT` を設定しない別サーバーでの通常 prefill は 9.896 秒だった。各 1 回であり、サーバー起動時間は含まない。両条件の生成テキストは一致した。OpenCode 側の SSE はツール対応時に生成完了まで保持されるため、この値は画面表示までの時間ではない。計測には `SHELF_DFLASH2_TOKEN_TIMESTAMPS` の compute token 受信時刻を使用した。

同じ 5,189 token のリクエストで、保存 payload の fingerprint を変更して checksum を再計算した場合は fingerprint 不一致として拒否され、通常 prefill は 10.157 秒だった。状態サーバーを停止した場合も接続失敗から通常 prefill に戻り 9.930 秒だった。どちらも snapshot なしの対照と生成テキストが一致した。専用の 32 MiB 状態サーバーを使った RDMA 単体テストでは、往復、容量超過の拒否と旧世代の保持、世代切り替えが成功した。

統合版の実機では、状態サーバー停止中の退避が HTTP 503 となっても GPU コマンドを続行し、コマンド失敗の終了コード 37、クライアントへの SIGTERM の終了コード 143、およびそれぞれの後のサーバー復帰を確認した。制御・ツール変換・snapshot HTTP の Python テストと CTest の CPU テストも成功した。実際の SIGSEGV や core は発生していないため、core 取得と gdb の連携は模擬コマンドによる確認に限る。
