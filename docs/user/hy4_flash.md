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

予約に必要な容量が不足する場合はエラーで停止する。予約長を 1Mi に変更する場合は、配置表も `--tokens 1048576` で再生成する。静的容量と前提は[調査記録](../research/20261119/hy4_r9700_capacity.md)を参照。複数台の R9700 による実機予約と生成は未検証である。
