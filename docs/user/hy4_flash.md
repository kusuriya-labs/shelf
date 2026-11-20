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
build/shelf-server --model-dir /path/to/model --engram-dir /path/to/engram \
  --dfly-dir /path/to/dfly --placement owners.json --max-seq-len 2097152
```

予約に必要な容量が不足する場合はエラーで停止する。予約長を 1Mi に変更する場合は、配置表も `--tokens 1048576` で再生成する。静的容量と前提は[調査記録](../research/20261119/hy4_r9700_capacity.md)を参照。

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

TP の all-reduce は bf16 で行うため、EP のみの構成と token 列が完全一致するとは限らない。数値の比較は logits の相対誤差で行う。`generate` で Engram を先に populate する場合は `SHELF_HY4_ENGRAM_POPULATE=1` を指定する。複数台の R9700 による実機の長時間運用は未検証である。

## Engram の起動時読み込み

server は既定で GPU への重み転送後に Engram の 4 表を populate してから ready になる。固定は既定では行わず、読み込み済みページもメモリ圧迫時に退避され得る。キャッシュ状態により全表の読み込み・固定には測定例で約 58〜242 秒かかり、固定しない populate も時間を要する。server の起動待ち時間には GPU 重み転送も加わる。

全表を RAM に固定するには、約 60.2 GiB にページ境界の余裕を加えた memlock の soft 上限と、OS や他の処理に必要な RAM を確保し、server 起動前に `SHELF_HY4_ENGRAM_MLOCK=1` を設定する。例えば十分な hard 上限が設定済みの場合、`ulimit -l` は KiB 単位で指定できる。

```bash
ulimit -l 75497472
export SHELF_HY4_ENGRAM_MLOCK=1
```

上限不足や固定失敗は起動エラーとなる。64 GB 級のホストでは全表を固定する容量を確保できないため、`SHELF_HY4_ENGRAM_MLOCK` を設定せず、`SHELF_HY4_ENGRAM_POPULATE=0` を指定すると遅延 mmap で読み込む。初回 request は遅くなる場合がある。populate を無効にした状態で mlock を要求することはできない。SSD から直接読み出す専用モードは未実装であり、今後の課題である。
