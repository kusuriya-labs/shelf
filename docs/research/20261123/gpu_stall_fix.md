# ストリーミング推論中の KFD キュー退避

## 既存の記録

`stall-audit.json` には Issue 43 の除外試行が 17 件ある。Issue 44 の BF16 KV では
BetterBench の `chat-clarify` の 41 番目の更新間隔が 2,581.5 ms だった。
`track_c_server.log` の要求 4 では compute 自身が進捗通知の間隔を 39.14 秒と
10.46 秒と記録している。これらは複数 token を含む通知間隔である。

[stdio による先行測定](issue44_dflash2_acceptance.md) の日本語 2k を round 別に
読み直すと、3,460.5 ms のうち 3,452.8 ms、2,444.1 ms のうち 2,436.5 ms が
`verify_submit` に入っていた。この区間には verify の完了待ちも含まれる。
過去の全件に対応するドライバのトレースは残っていない。

## S1　ホストページの移動による停止を再現

基準は `9c7be540`。Radeon AI PRO R9700、gfx1201、ROCm 10.2、Linux
`6.17.0-35-generic`、Release、TP=1。`ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55`
と `flock /tmp/r9700.lock` を使用した。target は Qwen3.8-27B EXL3-INT8
3.05 bpw、drafter は DFlash2 EXL3-INT8 4 bpw。標準条件は chain 7 提案、
prefix cache 有効、同時要求 1、最大系列長 24,576、prefill chunk とバッチの上限
2,048 token。診断には製品外のトレースを使った。

BF16 KV の stdio で、日本語 2k の生成中に連続する token 通知が 2,082.2 ms
途切れた。compute の単調時計では verify が 2,073.7 ms を占めた。最後の通知から
約 32 ms 後、カーネルの `kcompactd0` が `migrate_pages`、
`try_to_migrate_one`、MMU notifier を経て `amdgpu_amdkfd_evict_userptr` に入った。
最初の無効化は 3,139,108,864 バイト、次は 2,542,796,800 バイトで、
prefix checkpoint と embedding の確保サイズにそれぞれ一致する。

同じ PID に対し KFD は退避理由 USERPTR を通知した。さらに SVM 範囲の無効化と
キューの退避・復元が重なった。最初の USERPTR 退避から最後の復元までは
2,038.6 ms。最後の復元から約 11 ms 後に次の token が届いた。
3 GPU 分の通知があるので、通知数は独立した停止件数に換算していない。

約 150 ms ごとの sysfs 標本では、停止の前後で VRAM は 34,001,784,832 バイト、
GTT は 60,542,976 バイトで一定だった。GPU 使用率は 100% から 3%、
クロックは約 3 GHz から 40 MHz 前後に下がり、平均電力も 12 W まで低下した。
V620 の使用率は両方とも 0%。停止中も compute のメインスレッドは CPU 上で
走り続け、ホストの直接 compaction stall、major fault、swap in/out は増えていない。
NUMA ノードは 1 個で、自動 NUMA balancing は無効だった。

この事例ではホストのページ移動が KFD の計算キューを止めている。VRAM の退避を
示す TTM の退避理由は記録されなかった。GPU が低クロックになったのはキュー停止後
だった。先行する [PLaMo 3 の計測](../20261112/plamo3_exl3.md) でも背景 compaction と
GPU の遊休が重なっていたが、今回の記録ではページ無効化から KFD 通知まで追跡できた。

HTTP 経由の通常生成でも日本語 2k で 3,523.6 ms を再現した。関数の入口と戻りに
設定した uprobe では、同じ round の `complete_batch` が 3,516.7 ms を占め、
その中に USERPTR 退避から最後のキュー復元までの 3,480.7 ms が収まった。
同じ要求中に長時間の標準出力への書き込みは観測されなかった。

修正前に DFlash2 を外し、arena を 22 GiB に縮めても、日本語 8k の通常 decode で
3,622.7 ms の停止を再現した。停止前後の VRAM 使用量は 24,172,191,744 バイトで一定、
R9700 の `evicted_ms` は 13 ms から 3,603 ms に増えた。DFlash2 を使うことも、
VRAM をほぼ使い切ることも、この退避が起きるための必要条件ではなかった。

同じ 22 GiB の arena で DFlash2 を有効にし、verify Graph の前後に記録した
HIP イベント間の経過時間と、各 HIP 呼び出しが戻るまでのホストの時間を照合した。
3 要求中、日本語 2k の 2 要求で次の停止を捉えた。単位は ms。

|token 更新間隔|verify 区間|`hipStreamSynchronize` 1 回のホスト時間|`hipEventElapsedTime`|前後の KFD 退避時間の増分|
|---:|---:|---:|---:|---:|
|3,332.33|3,324.74|3,324.64|3,324.62|3,290|
|391.65|383.94|383.77|383.73|351|

同じ verify Graph のイベント時間の中央値は 31.63 ms（1,308 回）。2 例とも停止中に
GPU 使用率は 3% まで低下した。イベント間の時間には、キューが退避されて GPU が次の
命令を実行できない時間も含まれる。遅延はホストが verify の完了を待つ同期呼び出しに現れた。

## S2　常駐バッファと一時登録を分けて検証

まず `HSA_USERPTR_FOR_PAGED_MEM=0 HSA_SVM_FOR_PAGED_MEM=0` で常駐バッファを
GTT に移した。BF16 KV の同じ 50 要求で 1 秒以上の停止は 0 件になった。
ただし、生成中にヒープの 11,800,576 バイトと 21,626,880 バイトの領域で
SVM の無効化が起き、日本語 8k で 196.5 ms の間隔が残った。
この試行の 8 行 round 中央値は 40.21 ms。常駐バッファの割り当て変更だけで
退避原因をすべて取り除けたとは判断しなかった。

ROCm のインストール時の manifest が示すソースを確認した。この版の libhsakmt は、
ユーザー領域の登録に SVM API を優先し、既定ではその登録を解除する操作が空処理になる。
そのため、転送を終えてもヒープの一部が GPU のページテーブルに残りうる。
[`hsakmt_fmm_register_memory` と登録解除の実装](https://github.com/ROCm/rocm-systems/blob/43fa692e568e65a1fb84f2d1c74415b042839f10/projects/rocr-runtime/libhsakmt/src/fmm.c#L5047)

次に `HSA_USERPTR_FOR_PAGED_MEM=0 HSA_USE_SVM=0` を比較した。この組み合わせでは、
常駐する paged host allocation は GTT に置かれ、転送元を一時登録する場合は
従来の登録・解除経路を通る。GPU イベントを追加した 4 要求の試行で、最大更新間隔は
113.0 ms。ロード中に 32 MiB 程度の一時転送領域で USERPTR 退避があったが、
最初の要求の投入から最後の生成完了まで KFD の退避通知は 0 件だった。
[`HSA_USE_SVM` の読み込み](https://github.com/ROCm/rocm-systems/blob/43fa692e568e65a1fb84f2d1c74415b042839f10/projects/rocr-runtime/libhsakmt/src/openclose.c#L244)・[割り当て方式の選択](https://github.com/ROCm/rocm-systems/blob/43fa692e568e65a1fb84f2d1c74415b042839f10/projects/rocr-runtime/libhsakmt/src/fmm.c#L3023)

このマシンで GTT に切り替えると、embedding と prefix checkpoint の CPU マッピングは
`/dev/zero (deleted)` から `/dev/dri/renderD128` に変わった。計算を実行する R9700 は
`renderD130` である。`ROCR_VISIBLE_DEVICES` を指定した構成でも、ホストメモリの
実体を確認するには各カードの GTT 使用量と compute の `/proc/<pid>/maps` を併用する。

## S3　適用範囲と容量

採用したのは S2 の後者。stdio と HTTP はどちらも、モデルとホストバッファを保持した
compute が複数要求を処理する。割り当て方式を決めるのは ROCr の初期化時なので、
両経路に共通する `shelf-compute --serve-stdio` の起動時に設定する。ホスト全体の
compaction 設定や GPU の同期方法を変える必要はなく、round ごとの追加処理も生じない。
単発推論やライブラリを使うプロセスには、呼び出し側が選んだ割り当て方式を適用する。

代わりにホストの常駐バッファがドライバの GTT 予算を消費する。今回のメモリ計画では
embedding が 2.37 GiB、prefix checkpoint が 2.92 GiB だった。大きなホスト offload と
併用する場合はこの合計と他のプロセスの使用量も考慮する。運用時の確認方法と環境変数は
[環境変数リファレンス](../../user/env_vars.md) にまとめた。

自動設定での GPU arena は前後とも 31.29 GiB、KV 予算は 18.49 GiB。
物理 KV 容量は BF16 が 302,960 token、Octave が 957,584 token で、
どちらも最大系列長の設定は 24,576 token のまま比較した。

## S4　連続要求の測定方法

各プロセスに 10 課題を同じ順序で 5 巡、計 50 要求を送った。real-v1 は日本語と英語の
2k・8k・16k のコード編集、BetterBench は JSON 抽出、コード生成、文章題、`chat-clarify`。
並列度 1、seed 42、温度 0.7、top-p 0.95、top-k 20。BF16 の stdio は最初の 1 巡を
各課題の生成上限（real-v1 は 2,048）で実行した。残りの試行の生成上限は 128。
比較する前後で同じ入力と生成上限を使い、長い間隔が出た要求も集計に含めた。

更新間隔は、同じ要求の最初の更新から最後の更新までの隣接時刻の差をクライアントの
単調時計で測った。stdio は token 通知、HTTP は本文または思考の文字列を含む SSE を
1 更新とする。prefill と最初の更新までの待ち時間は別に計測した。同じ round で得た
複数の token を続けて送るため、更新間隔の中央値は 1 round の実行時間より短い。
8 行 round の中央値は同じ要求内の前の round 終了から 7 提案の round 終了までの時間。

### 更新間隔と生成速度

間隔と round の単位は ms。件数は更新間隔がしきい値以上になった回数。
tok/s は各要求の生成数から最初の 1 token を引いた合計を、最初から最後の更新までの
時間の合計で割った。全試行で round 計測を有効にし、HTTP の BF16 では前後とも
HIP 呼び出しの計装を併用した。比較は同じ経路・KV 形式の前後で行う。

|経路・KV|版|間隔数|p50|p95|p99|最大|300 ms 以上|1 秒以上|8 行 round 中央値|tok/s|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|stdio・BF16|修正前|18,753|0.003|41.48|42.51|2,082.21|1|1|40.162|101.14|
|stdio・BF16|修正後|18,753|0.004|41.50|42.54|115.10|0|0|40.190|102.22|
|stdio・Octave|修正前|6,350|0.003|41.19|44.36|110.85|0|0|39.535|112.98|
|stdio・Octave|修正後|6,350|0.004|41.22|44.31|112.39|0|0|39.480|112.99|
|HTTP・BF16|修正前|6,340|0.072|40.91|44.44|116.64|0|0|39.396|104.61|
|HTTP・BF16|修正後|6,340|0.074|40.93|44.74|115.92|0|0|39.414|104.52|
|HTTP・Octave|修正前|6,340|0.071|40.88|43.74|112.62|0|0|39.455|113.03|
|HTTP・Octave|修正後|6,340|0.072|40.87|43.92|114.03|0|0|39.518|112.95|

修正後 200 要求の更新間隔 37,783 件で、300 ms 以上は 0 件。4 条件とも 50/50
要求で生成数が一致し、stdio は生成 ID 列、HTTP は思考と本文を連結した文字列の
SHA-256 が前後で一致した。8 行 round 中央値の変化は各条件で 0.07 ms 未満だった。
HTTP の 128 token 試行では修正前も 300 ms を超えなかった。一方、HTTP・Octave の
最初の要求から最終要求までの sysfs 標本で、R9700 のプロセス別 `evicted_ms` は
修正前に 3,929 ms 増加し、修正後の増加は 0 ms だった。このカウンターには
prefill 中の退避も含まれる。

### 通常の生成上限での HTTP 比較

BF16 KV で round の診断出力と HIP 計装を外し、同じ 10 課題を各課題の本来の
生成上限で 1 回ずつ測定した。real-v1 は 6 問とも 2,048 token を生成した。
BetterBench は上記 4 問での比較である。全 10 要求で出力文字列と生成数が一致した。

|指標|修正前|修正後|
|---|---:|---:|
|real-v1 6 問の合算 decode（tok/s）|95.25|98.23|
|BetterBench 4 問の合算 decode（tok/s）|140.87|140.87|
|更新間隔 p95 / p99（ms）|40.78 / 41.77|40.73 / 41.73|
|最大更新間隔（ms）|3,523.57|112.81|
|300 ms 以上 / 1 秒以上（件）|2 / 1|0 / 0|

長い間隔を含めて比較すると real-v1 の合算速度が上がり、通常の更新間隔は
p95、p99 ともに 0.05 ms 以内の差だった。

## S5　正しさと終了処理

修正後の BF16 KV で同じ 10 課題を各 128 token 生成した。成功した `hipGraphLaunch`
を検証用の共有ライブラリで数え、設定の切り替えが実際の起動回数に反映されたことを確認した。

|Graph の設定|Graph 起動回数|既定と一致した生成 ID|8 行 round 中央値（ms）|合算 tok/s|
|---|---:|---:|---:|---:|
|target verify・drafter ともに有効|578|1,280 / 1,280|39.047|105.02|
|target verify・drafter ともに無効|2|1,280 / 1,280|42.927|97.34|
|ビルド時に `SHELF_HIP_GRAPH=OFF`|0|1,280 / 1,280|42.759|98.55|

2 行目で残った 2 回は、いずれも 1 行の通常 decode。Graph の有無で生成した
ID 列は全要求で一致し、既定の Graph 有効時の速度も維持できた。

通常の Release ビルドで `shelf-required-tests` と required ラベルの実行に必要な
テストバイナリをビルドし、`tests/run_required_acceptance.py` で CPU・GPU の全件を
再実行した。225 件 PASS、失敗・エラー・skip はすべて 0 件。追加した
`test_serve_host_memory` は、未指定時の既定値と、3 つの対象変数のどれかが
指定された場合に起動環境を維持することを検査する。

実モデルの `test_dflash2_batch_e2e` は提案数 7・3・1 の各ケースで 64 token が一致。
`test_server_dflash2.py` は 20/20 PASS。最終サーバー停止後に
`/sys/class/kfd/kfd/proc/` のエントリ、今回の compute・サーバー・計測プロセス、
一時トレースプロセスがすべて 0 件であることを確認した。
