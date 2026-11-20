# Qwen3.8 3.05bpw を 16 GiB VRAM で 128Ki context で動かす

## 目的

Qwen3.8-27B EXL3-INT8 **3.05bpw** モデル
(`/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-305-mul24`, 13 GB)
を、gfx1201 (RX 9070 XT 想定, VRAM 16 GiB / 実効 15 GiB) で
**投機なし・KV c5v5・context 128Ki (131,072 token)** で動かす。

手段は `embed_tokens` (BF16 `[248320, 5120]` = 2.37 GiB) を VRAM に置かずホスト RAM に置くこと。

## 測定環境・条件

- GPU: card2 = R9700 (32 GiB)。物理 16 GiB の環境は無いため、**ピーク使用 VRAM が 15 GiB 以下**を合格判定とする。
- ビルド: `build-rocm102` (ROCm 10.2 / gfx1201 / libc++)、`shelf-compute`。
- 計測: `.hy4-agent/measure_vram.py` が sysfs `mem_info_vram_used` を 50 ms ごとにサンプリングしピークを記録。
  `tools/hy4_agent/with-gpu` 経由で実行（Hy4 サーバ停止・再起動込み）。
- 共通引数: `--model-dir <3.05bpw> --input-ids-file /tmp/mini.ids --max-new-tokens 1
  --max-seq-len 131072 --kv-cache-dtype octave:c5v5 --page-tokens 16 --device 0 --temperature 0`
- 入力は 3 token のみ（配置内訳の測定が目的）。

## 手順1: 現状の VRAM 内訳

生ログ: `.hy4-agent/step1-measure.log`、スクリプト: `.hy4-agent/step1-measure.sh`。

### 配置内訳 (MEMORY_PLAN)

| 項目 | arena=20 GiB | arena=15 GiB |
| --- | ---: | ---: |
| target weights | 9.59 GiB | 9.59 GiB |
| draft weights | 0.00 GiB | 0.00 GiB |
| drafter runtime | 0.00 GiB | 0.00 GiB |
| runtime workspace | 0.29 GiB | 0.29 GiB |
| KV budget | 8.97 GiB (464,720 token) | 3.97 GiB (205,808 token) |
| prefix cache | 0.57 GiB | 0.57 GiB |
| required | 19.43 GiB | 14.43 GiB |
| available | 20.00 GiB | 15.00 GiB |
| arena capacity | 20.00 GiB | 15.00 GiB |
| left | 0.00 GiB | 0.00 GiB |

ホスト側 (両 run 共通):

| 項目 | サイズ |
| --- | ---: |
| embedding (pinned) | 2.37 GiB |
| prefix checkpoint | 2.29 GiB |
| total | 4.66 GiB |

### ピーク VRAM (card2)

| run | arena | baseline | peak | delta (ピーク使用量) | 16 GiB に対して | 実効 15 GiB に対して |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 20 GiB | 0.056 GiB | 20.326 GiB | **20.271 GiB** | +4.27 GiB 超過 | +5.27 GiB 超過 |
| B | 15 GiB | 0.056 GiB | 15.328 GiB | **15.271 GiB** | -0.73 GiB 余裕 | +0.27 GiB 超過 |

`SHELF_ARENA_PLAN` (実バイト):

```
# A
capacity=21474836480 weights=10294792452 pools=615907836 kv=9948347648
kv_tokens=464720 prefix_est=615776256 rest=615776256 left=12288 reserve=316060672
# B
capacity=16106127360 weights=10294792452 pools=615907836 kv=4579548416
kv_tokens=205808 prefix_est=615776256 rest=615776256 left=102400 reserve=316060672
```

### 考察

- **embedding は既にホスト RAM にある。** `MEMORY_PLAN host memory` の
  `embedding (pinned) 2.37 GiB` がそれを示す。ベースコミットの
  `607b20aa perf: embedding を pinned host RAM に置き、drafter 用の複製を廃止する`
  で `MatrixWeight::host_resident` / `host_payload` が導入済みで、
  `make_weight_slot()` が `matrix_payload(w)` を `WeightSlot::device_ptr` に渡すため、
  embedding lookup カーネルは unified address space 経由でホストの BF16 テーブルを直接読む。
  `target weights 9.59 GiB` は 3.05bpw × 27B の量子化重みのみで embedding を含まない。
- よって手順2の「embedding を GPU に置かない」実装は**ベースコードに既に存在する**。
  残る作業は挙動の切替 (環境変数) と、128Ki 実配置での検証・速度/品質確認。
- KV budget は arena いっぱいまで自動拡張される (A: 464,720 token、B: 205,808 token)。
  128Ki (131,072 token) に必要な KV は 20,736 B/token × 131,072 = 2.53 GiB なので、
  arena を 13 GiB 前後に抑えれば KV を 128Ki 分確保したまま
  ピーク VRAM を 15 GiB 以下に収められる見込み。

## 進捗ログ

- 2026-11-20: 土台を準備（worktree `hy4/qwen-embed-host`、origin/main 3682ee07 から）。未着手。
- 2026-11-20: ビルド完了、measure_vram.py 作成、手順1の VRAM 測定実施 (peak 20.27 / 15.27 GiB)。
- 2026-11-21: 手順1の結果を本ファイルに記録。embedding はベースコードで既にホスト常駐と判明。
