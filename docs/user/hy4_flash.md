# Hy4-flash の実行

Hy4-flash の 4 bpw EXL3-INT8 checkpoint、FP8 Engram、対応する DFly checkpoint を使用する。対象 GPU は gfx1201 の R9700 であり、`ROCR_VISIBLE_DEVICES` で使用する GPU を順番に指定する。FP8 Engram はホストメモリに置くため、VRAM とは別に約 61 GiB の空きメモリが必要である。

以下は 3 台、2Mi 予約の例である。GPU UUID とパスは環境に合わせて指定する。4 台では `--capacity` の値を 4 個、UUID も 4 個指定する。`--work` を省くと expert の選択頻度は均等と仮定する。

```bash
python3 tools/hy4_flash/placement.py /path/to/model --draft /path/to/dfly \
  --capacity 32 32 32 --tokens 2097152 --output owners.json
export ROCR_VISIBLE_DEVICES=GPU-UUID0,GPU-UUID1,GPU-UUID2
build/shelf-compute hy4 inspect --model-dir /path/to/model --engram-dir /path/to/engram \
  --dfly-dir /path/to/dfly --placement owners.json
build/shelf-compute hy4 reserve --model-dir /path/to/model --engram-dir /path/to/engram \
  --dfly-dir /path/to/dfly --placement owners.json --max-seq-len 2097152
```

予約に必要な容量が不足する場合はエラーで停止する。予約長を 1Mi に変更する場合は、配置表も `--tokens 1048576` で再生成する。静的容量と前提は[調査記録](../research/20261119/hy4_r9700_capacity.md)を参照。

## Prefill のチャンク長

EP で長い入力を処理する場合、compute では `--chunk 2048` または `--chunk 4096`、server では `--prefill-chunk 2048` または `--prefill-chunk 4096` を指定できる。Hy4 server の既定値は 1024 で、指定できる範囲は 1〜4096 である。チャンク長だけを変更する場合は同じ配置表を使える。TP を併用すると、実行時に最大 1024 行ずつ処理する。

長文の Indexer を従来の scalar 演算と照合する場合は `SHELF_HY4_INDEX_WMMA=0` を指定する。

## HIP Graph の切り替え

TP=1、EP=1〜4 で push 転送を使用し、すべての GPU 間で peer access が有効な場合、バッチ decode と verify に層ごとの HIP Graph を使用する。DFly2 は `--dfly-drafts`（または環境変数 `SHELF_HY4_DFLY_DRAFTS`）で 1 ラウンドあたりの候補数 `K` を `2, 3, 4, 5` から指定でき、既定は `4` である。バッチ実行時は最大 16 スロットの候補を 1 回の Graph 起動で生成し、Target がスロットごとに `K + 1` 行（最大 6 行）を検証する。初回のバッチ形状では Graph を作成し、同じ形状と対応するコンテキスト長では再利用する。Target は `SHELF_HY4_GRAPH=0`、DFly2 は `SHELF_HY4_DRAFT_GRAPH=0` で個別に eager 実行へ切り替えられる。

`placement.py` は TP=1 の各 rank の `peak_estimate` に Graph 用の 256 MiB を含める。実行時は、重み、KV、作業域の予約後にこの余裕を確保できる場合に Graph を有効にする。余裕不足、capture 失敗、EP のプロファイル取得時は eager 実行を使用する。単一スロットの DFly2 は rank 0 で完結し、TP=1 では peer / host 転送時にも Graph を使用できる。

## EP=2 × TP=2 の例

4 台を 2 つの EP group と 2 つの TP lane で使う。配置表の値は expert の所有 EP group になる。配置表は TP 分割後の 1 lane 分の容量で生成する。

```bash
python3 tools/hy4_flash/placement.py /path/to/model --draft /path/to/dfly \
  --capacity 34 34 --workspace 1 1 --tensor-parallel 2 \
  --tokens 2097152 --output owners.json
export ROCR_VISIBLE_DEVICES=GPU-UUID0,GPU-UUID1,GPU-UUID2,GPU-UUID3
build/shelf-compute hy4 inspect --model-dir /path/to/model --engram-dir /path/to/engram \
  --dfly-dir /path/to/dfly --placement owners.json \
  --expert-parallel 2 --tensor-parallel 2
build/shelf-compute hy4 reserve --model-dir /path/to/model --engram-dir /path/to/engram \
  --dfly-dir /path/to/dfly --placement owners.json \
  --expert-parallel 2 --tensor-parallel 2 --max-seq-len 2097152
```

`generate` の予約長上限は 1Mi である。生成する場合は `--max-seq-len 1048576` を指定する。

```bash
build/shelf-compute hy4 generate --model-dir /path/to/model --engram-dir /path/to/engram \
  --dfly-dir /path/to/dfly --placement owners.json \
  --expert-parallel 2 --tensor-parallel 2 --max-seq-len 1048576 \
  --input-ids-file prompt.ids --max-new-tokens 256 --chunk 1024 --temperature 0
```

TP の all-reduce は bf16 で行うため、EP のみの構成と token 列が完全一致するとは限らない。数値の比較は logits の相対誤差で行う。`generate` は Engram の行を先読みする。無効化する場合は `SHELF_HY4_ENGRAM_PREFETCH=0` を指定する。Engram を先に populate する場合は `SHELF_HY4_ENGRAM_POPULATE=1` を指定する。複数台の R9700 による実機の長時間運用は未検証である。

## Engram の起動時読み込み

server は既定で GPU への重み転送後に Engram の 4 表を populate してから ready になる。固定は既定では行わず、読み込み済みページもメモリ圧迫時に退避され得る。キャッシュ状態により全表の読み込み・固定には測定例で約 58〜242 秒かかり、固定しない populate も時間を要する。server の起動待ち時間には GPU 重み転送も加わる。

全表を RAM に固定するには、約 60.2 GiB にページ境界の余裕を加えた memlock の soft 上限と、OS や他の処理に必要な RAM を確保し、server 起動前に `SHELF_HY4_ENGRAM_MLOCK=1` を設定する。例えば十分な hard 上限が設定済みの場合、`ulimit -l` は KiB 単位で指定できる。

```bash
ulimit -l 75497472
export SHELF_HY4_ENGRAM_MLOCK=1
```

上限不足や固定失敗は起動エラーとなる。64 GB 級のホストでは全表を固定する容量を確保できないため、`SHELF_HY4_ENGRAM_MLOCK` を設定せず、`SHELF_HY4_ENGRAM_POPULATE=0` を指定すると遅延 mmap で読み込む。初回 request は遅くなる場合がある。populate を無効にした状態で mlock を要求することはできない。SSD から直接読み出す専用モードは未実装であり、今後の課題である。

## バッチ生成と Continuous Batching

`shelf-server --max-concurrent-requests N` は TP=1、EP=1〜4 で 1〜16 件を同時に保持する。既定値は 4 件である。compute を直接起動する場合は `shelf-compute hy4 serve --max-concurrent-requests N` を使う。こちらの既定値は 1 件である。`--max-seq-len` はスロットごとの予約長を表す。配置試算と実行には同じ予約長とスロット数を指定する。

`placement.py` は複数スロットで `--balance-experts` を既定にする。容量と安全余裕を守りながら、共有 expert を実行する rank 0 の routed expert 数を、他の同型 GPU の約 9 割にする。4 台では rank 0 の目安が 23.1% となる。`--no-balance-experts` は各 rank の合計使用量を均等側へ寄せる。単一スロットではこちらが既定である。

`--dfly-dir` を指定すると、複数スロットでも DFly2 の候補をまとめて生成・検証する。`--dfly-drafts`（`2, 3, 4, 5`、既定 `4`）および環境変数 `SHELF_HY4_DFLY_DRAFTS` は単一スロット・バッチ双方に適用される（範囲外は起動エラー）。`SHELF_HY4_BATCH_DFLY=0` で複数スロットの投機を無効にできる。Target Graph と DFly2 Graph の切り替えは上記の「HIP Graph の切り替え」を参照する。

### R9700 を 3 台または 4 台使った照合

使用する R9700 の UUID を `ROCR_VISIBLE_DEVICES` に指定する。次の例では `BODY`、`ENGRAM`、`DFLY` を各 checkpoint のパス、`IDS` を入力 ID ファイルのパスとする。容量は 3 台なら `32 32 32`、4 台なら `32 32 32 32` とする。2Mi の単一スロット予約は冒頭の `reserve` で検証する。バッチの照合には別の配置表を作成する。

```bash
capacity=(32 32 32)
python3 tools/hy4_flash/placement.py "$BODY" --draft "$DFLY" --capacity "${capacity[@]}" \
  --tokens 4096 --max-concurrent-requests 16 --output owners-batch.json
common=(--model-dir "$BODY" --engram-dir "$ENGRAM" --dfly-dir "$DFLY" --placement owners-batch.json)
for transport in push peer host; do
  export SHELF_HY4_EP_TRANSPORT=$transport
  for draft in 0 1; do
    for batch in 1 2 4 8 16; do
      SHELF_HY4_BATCH_DFLY=$draft build/shelf-compute hy4 generate "${common[@]}" \
        --max-seq-len 4096 --input-ids-file "$IDS" --max-new-tokens 32 --temperature 0 --batch-size "$batch"
    done
  done
done
python3 tests/server/test_hy4_continuous_batching.py "${common[@]}"
devices=$(seq -s, 0 $((${#capacity[@]} - 1)))
build/shelf-server "${common[@]}" --device "$devices" --expert-parallel "${#capacity[@]}" \
  --max-seq-len 4096 --max-concurrent-requests 16
```

各スロットの `GENERATED_IDS[i]` を同じ転送方式の B=1 の `GENERATED_IDS` と比較する。異なるプロンプトを一括生成するには `--batch-input-files a.ids,b.ids` を、個別の生成上限を指定するには `--batch-max-new-tokens 16,32` を使う。サーバーの検証スクリプトは JSONL 経由で同時投入、待機、個別キャンセル、スロット再利用を検査する。
