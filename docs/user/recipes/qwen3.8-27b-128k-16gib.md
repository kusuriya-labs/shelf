# Qwen3.8-27B EXL3-INT8 3.05bpw、16 GiB VRAM で 128Ki context

## ハードウェア

- GPU: gfx1201 (RX 9070 XT 想定、VRAM 16 GiB。表示にも使う場合の実効は 15 GiB)
- ホスト RAM: pinned 約 4.7 GiB (embedding 2.37 GiB + prefix checkpoint 2.29 GiB)
- ROCm 10.2

## モデル

- ベース: <https://huggingface.co/Qwen/Qwen3.8-27B>
- 量子化済み: <https://huggingface.co/kusuriya-labs/Qwen3.8-27B-EXL3-INT8-3.05bpw> (EXL3-INT8 3.05bpw、重み 9.59 GiB)

## 起動

投機なし、KV は `octave:c5v5`。

```sh
shelf-compute --model-dir <3.05bpw model dir> --input-ids-file <ids> \
  --max-seq-len 131072 --arena-gib 14 --prefill-chunk 512 \
  --kv-cache-dtype octave:c5v5 --page-tokens 16 --device 0
```

- `--arena-gib 14` では arena は 13.7 GiB になる。下記の測定は変更前の arena 14 GiB で行ったもので、この指定値での 128Ki context は未検証。`auto` は空き VRAM から 0.5 GiB を差し引いて使う。
- `--prefill-chunk 512`。4096 だと workspace が 0.52 → 2.40 GiB に増えて KV が足りなくなる。
- `shelf-server` も同じ引数名を持つ ([server.md](../server.md))。server 経由での確認はしていない。

## 速度の目安

2026-11-21、R9700 (arena 14 GiB 固定) での測定。16 GiB の実カードでの測定はまだない。

| 項目 | 目安 |
|---|---:|
| ピーク VRAM | 約 14.4 GiB |
| prefill (120k token、平均) | 約 730 tok/s (序盤 約 1,600 → 120k 付近 約 480 tok/s に低下) |
| decode (文脈 1k) | 約 29 tok/s |
| decode (文脈 32k) | 約 25 tok/s |
| decode (文脈 120k) | 約 19 tok/s |

測定の詳細は [perf/qwen-embed-host.md](../../perf/qwen-embed-host.md)。
