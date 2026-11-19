# Hy4 Engram の RDMA 読み出し

## 状態

この記録は fork 専用である。対象は `hy4-rdma-engram`、基点は `fac6f716`。`hy4-server-path` をマージして populate / mlock 実装を取り込んだ。

## 配置と実装上の確認

2026-11-19 に t7610 の `/home/user` を含むファイルシステムは空き 120 GiB、メモリは free 130 GiB、available 182 GiB だった。表の元ファイルは変更していない。両端の `hfi1_0` は active、active MTU は 4096。計測用 TCP はリモートの `10.255.254.1` のみに bind した。計測プロセスは終了済み。

既存の safetensors では FP8 の 96 byte と BF16 scale の 2 byte は別 tensor である。表の行数は層 1 / 11 / 21 / 31 の順に 164,994,380 / 164,999,008 / 165,004,086 / 165,008,942 行である。サーバーで 98 byte 行に再配置し、各表を MR として公開する。クライアントは forward の先頭で 4 表分の index を計算して非同期に取得し、最初の Engram 層で完了を待つ。行を表ごとにソート・重複排除し、連続行をまとめて READ する。

## 98 byte READ の予備測定

ローカルから t7610 への RC READ。`ib_read_bw` は 100000 回/QP、`ib_read_lat` は 10000 回、MTU 4096、CPU 周波数固定なし。ツールは CPU 周波数に関する警告を出した。READ の outstanding は HFI が 1 QP あたり最大 16 に制限する。表中の QD は QP あたりの上限を合計した値であり、全測定期間に常時その数が発行されたことを意味しない。

| QP | 合計 outstanding 上限 | post list | READ/秒 | 帯域 MiB/秒 |
|---:|---:|---:|---:|---:|
| 1 | 16 | 1 | 264,837 | 24.75 |
| 4 | 64 | 1 | 264,547 | 24.72 |
| 8 | 128 | 1 | 271,845 | 25.41 |
| 4 | 64 | 16 | 955,997 | 89.35 |
| 8 | 128 | 16 | 1,035,164 | 96.75 |

単発の `ib_read_lat` は最小 12.39 µs、typical 16.18 µs、平均 16.80 µs、99 パーセンタイル 43.92 µs、99.9 パーセンタイル 51.83 µs、最大 160.84 µs。これは QD 1 の遅延であり、QD 64/128 の遅延分布ではない。

`ib_read_bw` は小さな登録領域の繰り返しアクセスであり、Engram 表からのランダムアクセス、384 行や 13 万行のバッチ時間を表さない。実装では最大 16 件を post list にまとめているが、短いバッチで同じ改善が得られるとは限らない。inline は HFI が対応しないと perftest が報告した。

再現例はリモートで以下を実行し、別端末からローカル側のコマンドを実行する。計測用ポートは空いているものを選ぶ。

```sh
cd /home/user/engram-rdma
ib_read_bw -d hfi1_0 -s 98 -t 64 -o 16 -q 4 -l 16 -n 100000 -p 19385 --bind_source_ip 10.255.254.1
```

```sh
ib_read_bw -d hfi1_0 -s 98 -t 64 -o 16 -q 4 -l 16 -n 100000 -p 19385 --bind_source_ip 10.255.254.2 10.255.254.1
```

## バイト一致

4 ファイルを IPoIB 経由で転送後、リモートの登録済み表から 4 表合計 100 万の擬似乱数行と、各表の先頭・末尾・重複・連続行を取得した。FP8 96 byte と scale 2 byte を元のローカル safetensors と比較し、全行が一致した。ローカルの RDMA 登録バッファ内で順序を復元して H2D に渡す方式に変更した後も、同じ検証が成功した。乱数 seed は 763210。

## fork 専用の構築と運用

クライアントは `SHELF_HY4_RDMA=ON` のときだけ libibverbs をリンクする。既定は OFF で、Engram ソースも従来のローカル mmap である。RDMA 有効ビルドで `SHELF_HY4_ENGRAM_SOURCE=rdma` を指定するとローカルの Engram 表を開かず、`10.255.254.1:19390` に接続する。ローカルの表固定とは併用できない。RDMA を無効にしたビルドで RDMA を指定した場合もエラーとする。

```sh
cmake -S . -B build-hy4-rdma -DSHELF_HY4_RDMA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-hy4-rdma --target shelf-compute shelf-server-stage shelf-engram-rdma-server
```

リモートの CPU 用バイナリは worktree のルートで次のようにも構築できる。必要なものは C++20 コンパイラ、libibverbs とそのヘッダー、およびリポジトリ同梱の JSON ヘッダーである。リモートのディスク空きとメモリ空きを確認し、4 ファイルを `/home/user/engram-rdma/` に転送してから起動する。`HostKeyAlias` は既に検証済みの管理用アドレスのホスト鍵を IPoIB の転送にも使う例である。

```sh
c++ -std=c++20 -O2 -Iinclude -Ivendor tools/hy4_flash/rdma/server.cpp -libverbs -o /tmp/shelf-engram-rdma-server
scp -o HostKeyAlias=192.168.0.148 /tmp/shelf-engram-rdma-server user@10.255.254.1:/home/user/engram-rdma/server
rsync -ah -e 'ssh -o HostKeyAlias=192.168.0.148' \
  ~/datapool/hy4-hf-staging-20261119/body/model.layers.{1,11,21,31}.engram_layer.multi_head_embedding.embedding.weight.fp8.safetensors \
  user@10.255.254.1:/home/user/engram-rdma/
```

既存のサーバーを更新する場合は、停止と終了を確認してからバイナリを置き換える。

以下はリモートの通常ユーザーで、サーバーのプロセスだけに 72 GiB の memlock 上限を設定して起動する例である。対象プロセスを停止状態で作り、上限を設定してから処理を再開する。`sudo` はそのプロセスの `prlimit` にのみ使う。

```sh
cd /home/user/engram-rdma
nohup bash -c 'kill -STOP $$; exec ./server .' >server.log 2>&1 </dev/null &
pid=$!
while ! ps -o stat= -p "$pid" | grep -q T; do
  kill -0 "$pid" || exit 1
  sleep 0.1
done
sudo -n prlimit --pid "$pid" --memlock=77309411328:77309411328 || { kill -KILL "$pid"; exit 1; }
printf '%s\n' "$pid" >server.pid
kill -CONT "$pid"
tail -f server.log
```

4 表の登録完了と `10.255.254.1:19390` の接続待機を確認してから shelf を起動する。サーバーは 4 表を anonymous memory に再配置して `mlock` と MR 登録を行う。TCP は IPoIB のアドレスだけに bind する。停止は `kill -TERM "$(cat /home/user/engram-rdma/server.pid)"`。再起動後は MR と QP が変わるため、クライアントも再起動して接続し直す。接続・READ 失敗時にローカル表へ戻すことはない。

ローカル側は既存のモデル・配置・DFly のパスを使い、例えば次のように起動する。`$model`、`$engram`、`$placement`、`$dfly` は各ディレクトリまたは配置ファイルのパスである。`SHELF_HY4_ENGRAM_MLOCK=1` は設定しない。

```sh
SHELF_HY4_ENGRAM_SOURCE=rdma build-hy4-rdma/shelf-server \
  --model-dir "$model" --engram-dir "$engram" --placement "$placement" --dfly-dir "$dfly" \
  --max-seq-len 2304 --max-concurrent-requests 1
```

各表の行数を checkpoint の head 語彙数の合計と照合する。READ 先は最大 16 outstanding / QP、8 QP の RC で、ローカル登録バッファは forward の要求に応じて確保し、各表 16,384 行まで、作業用と出力用の合計で最大 8,028,160 byte に制限する。READ 失敗後は当該接続を再利用しない。`VmLck` と RDMA 登録ページの `VmPin` は別の値である。

## 失敗系

稼働中に制御接続を切断した場合、次の READ は `RDMA 制御接続が切断されました` と失敗した。停止中の接続は `RDMA サーバーに接続できません` と失敗した。READ 完了待ちのタイムアウトは時計を 11 秒進める故障注入で `RDMA READ がタイムアウトしました` を確認し、その後に新規接続できた。時計注入による試験であり、物理リンク障害による待機時間を実測したものではない。

```sh
cmake -S . -B build-hy4-rdma -DSHELF_HY4_RDMA=ON -DSHELF_BUILD_TESTS=ON
cmake --build build-hy4-rdma --target test_hy4_engram_rdma test_hy4_engram_rdma_timeout hy4_rdma_fault_clock
build-hy4-rdma/test_hy4_engram_rdma ~/datapool/hy4-hf-staging-20261119/body
LD_PRELOAD="$PWD/build-hy4-rdma/libhy4_rdma_fault_clock.so" build-hy4-rdma/test_hy4_engram_rdma_timeout
```

完了通知を QP ごとに 16 件以下のバッチ末尾へまとめ、末尾に fence を指定して先行 READ の完了を保証した後にも、同じ 100 万行の照合に成功した。時間切れにした接続は再利用できないことも故障注入テストで確認した。72 GiB の上限を指定した新規起動で 4 表すべてが登録され、サーバーの `VmLck` と `VmPin` はそれぞれ 63,164,684 KiB だった。SIGTERM による解放後、同じ方法で再起動できた。

## 生成一致と性能

ローカル mlock 対照は R9700 と V620 2 台、4 bpw、同じ expert 配置・入力・温度 0 を使い、2,048 token の prefill 後の 256 token が通常生成と DFly の間で一致した。ローカルの `VmLck` は 63,164,684 KiB。これらは 1 回ずつの正しさ確認であり、性能の統計値ではない。

同じ 2,048 token 入力で RDMA 版を実行し、通常生成・DFly ともローカル通常生成およびローカル DFly の全 256 token と一致した。DFly は両方式とも 64 検証・192 候補受理だった。1 回の prefill 時間と速度は次のとおりである。これは生成一致のための実行で、warmup を除いた統計値ではない。

| Engram | 生成 | pp2048 秒 | pp2048 token/秒 |
|---|---|---:|---:|
| ローカル mlock | 通常 | 3.272 | 625.84 |
| RDMA | 通常 | 3.200 | 640.05 |
| ローカル mlock | DFly | 3.241 | 631.97 |
| RDMA | DFly | 3.211 | 637.89 |

RDMA 通常生成の 2,048 token prefill 中、ローカルプロセスの `VmLck` は 0 KiB、`VmPin` は 7,840 KiB、ローカル Engram ファイルの mmap は 0 件だった。

### BetterBench 4 bpw

R9700 と V620 2 台の UUID を指定して 3 台の lock を保持し、各カテゴリでサーバーを再起動した。BetterBench 0.6.1 / corpus 1.0、各カテゴリ warmup 1 件と測定 5 件、逐次 1 request、nonce 有効、温度 0.7 / top_p 0.95 / top_k 20、予約 2,304 token。RDMA は 10 / 10 件成功した。ローカル mlock 値は同条件で以前測定した値であり、今回再測定した値ではない。

| カテゴリ | ローカル mlock decode token/秒 | RDMA decode token/秒 | ローカル TTFT ms | RDMA TTFT ms | ローカル受理 / 検証 | RDMA 受理 / 検証 |
|---|---:|---:|---:|---:|---:|---:|
| code | 35.10 | 31.46 | 318.00 | 320.28 | 1,936 / 966 = 2.00 | 1,908 / 990 = 1.93 |
| chat | 28.98 | 30.41 | 373.83 | 376.16 | 670 / 446 = 1.50 | 689 / 427 = 1.61 |

速度・TTFT は各 5 件の中央値。RDMA の prefill 中央値は code 234.80、chat 288.14 token/秒。decode 秒数合計 / 検証回数は RDMA code 88.85、chat 83.93 ms / 検証で、既存の mlock 結果はそれぞれ 88.87、83.39 ms / 検証だった。nonce と生成列が異なり、受理数も異なるため、中央値の差を RDMA 単独の費用とはみなせない。

RDMA の初回 warmup は code が 78 入力 token の prefill 0.350 秒、507 decode token の 155 検証に 13.721 秒（88.53 ms / 検証）、chat が 117 入力 token の prefill 0.395 秒、199 decode token の 96 検証に 7.577 秒（78.92 ms / 検証）だった。これは HTTP の初回 TTFT ではない。異なる入力を使った既存のローカル mlock 初回値は 84.15 ms / 検証で、条件が完全に同一ではない。BetterBench 測定後のローカル `VmLck` は両カテゴリ 0 KiB、`VmPin` は code 660 KiB、chat 920 KiB だった。

QD 64 / 128 の個別遅延分布と 384 行・13 万行だけを取り出す専用の時間計測は実施していない。事前調査の連続アクセスの値と pp2048 / 実生成の結果から、それらの個別目標が達成されたとは判定できない。タイムアウトは時計の故障注入で確認し、物理リンクを切断する試験は行っていない。

測定コマンドはカテゴリごとにサーバーを再起動し、ready 後に次を実行した。`$category` は `code` または `chat`。

```sh
PYTHONPATH=vendor/BetterBench BETTERBENCH_NO_UPDATE_CHECK=1 /usr/bin/python3 -m betterbench.cli run \
  --endpoint http://127.0.0.1:18123/v1 --model Hy4-flash-4bpw --quick --decode \
  --categories "$category" --max-model-len 2304 --out "$output" --no-html
```

同条件のローカル mlock 版と測定方法の詳細は [Hy4-flash の研究記録](hy4_flash_progress.md) を参照する。今回の生ログと結果 JSON は測定ホストの `/tmp/hy4-rdma-qualification/` にある。

性能測定後、クライアントに各表 16,384 行までという入力上限の検査を追加した。測定時の入力はこの上限内で、検査追加後の性能は再測定していない。
