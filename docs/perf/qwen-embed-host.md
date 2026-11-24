# Qwen3.8 3.05bpw を 16 GiB VRAM で 128Ki context で動かす

## 目的

Qwen3.8-27B EXL3-INT8 **3.05bpw** モデル
(`/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-305-mul24`, 13 GB)
を、gfx1201 (RX 9070 XT 想定, VRAM 16 GiB / 実効 15 GiB) で
**投機なし・KV c5v5・context 128Ki (131,072 token)** で動かす。

`embed_tokens` (BF16 `[248320, 5120]` = 2.37 GiB) は main で既に pinned host RAM 上にあり、追加実装は不要だった。本書は起動引数だけで 128Ki が 15 GiB に収まることの確認記録。

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
- よって「embedding を GPU に置かない」実装は**ベースコードに既に存在し**、切替の環境変数も不要。
- KV budget は arena いっぱいまで自動拡張される (A: 464,720 token、B: 205,808 token)。
  128Ki (131,072 token) に必要な KV は 20,736 B/token × 131,072 = 2.53 GiB なので、
  KV 自体は小さいが、後述のとおり 13 GiB では workspace と合わせて足りず、14 GiB が必要。

## 手順2〜4: 128Ki 近傍の実測 (2026-11-21)

方針修正により embedding の実装・環境変数切替は不要と判断（ベースで既にホスト常駐）。
よって「実装後」の測定を、既存コードのまま起動引数だけを絞って実施した。

- 入力: `/tmp/shelf-prefill-32768.psktok` の 32,768 token を繰り返して 120,000 token の
  whitespace ids ファイルを作成 (`/tmp/shelf-prefill-120000.ids`)。
- 条件: 3.05bpw / c5v5 / 投機なし / `--max-seq-len 131072` / `--max-new-tokens 32` /
  `--prefill-chunk 512` / `--page-tokens 16`。
- 生ログ: `.hy4-agent/step2-measure-chunk512.log` (成功)、`.hy4-agent/step2-measure.log`
  (chunk 4096 の失敗例)。スクリプト: `.hy4-agent/step2-measure.sh`。

### 試行結果

| run | arena | prefill chunk | 結果 | ピーク VRAM (delta) |
| --- | ---: | ---: | --- | ---: |
| C | 14 GiB | 512 | **成功** (120k prefill + 末尾 decode 31 token) | **14.394 GiB** |
| D | 13 GiB | 512 | 失敗: KV 97,584 token < 必要 131,073 | 13.260 GiB |
| (参考) | 14 GiB | 4096 | 失敗: workspace reserve 2.40 GiB で KV 104,048 token | 14.258 GiB |
| (参考) | 13 GiB | 4096 | 失敗: KV 52,272 token | 13.260 GiB |

`--max-seq-len 131072` を宣言すると、実際の入力長に関わらず arena が
**131,073 token 分の KV** を確保できる必要がある
(`compute_runtime.hip:503`)。13 GiB ではこれを満たせない。
また prefill chunk を 4096 にすると workspace reserve が 0.52 → 2.40 GiB に増え、
同じ arena でも KV 容量がさらに削れる。chunk 512 が 128Ki には最適。

### 成功時の内訳・速度 (run C)

| 項目 | 値 |
| --- | ---: |
| target weights | 9.59 GiB |
| runtime workspace | 0.52 GiB (row bucket 512) |
| KV budget | 2.88 GiB (149,360 token) |
| prefix cache | 0.57 GiB |
| required / available | 13.57 / 14.00 GiB |
| embedding (pinned, host) | 2.37 GiB |
| prefix checkpoint (host) | 2.29 GiB |
| PROMPT_TOKENS / GENERATED_TOKENS | 120,000 / 32 |
| PREFILL_MS / PREFILL_CHUNKS | 177,056.5 ms / 236 |
| prefill 平均 | **677.7 tok/s** (chunk ごと 273〜1,641 tok/s、文脈伸長に伴い低下) |
| decode (再測、`--max-new-tokens` 32 / 128) | **18.2 / 19.4 tok/s** (中央値 約 51 ms/token) |
| ピーク VRAM | **14.394 GiB** (16 GiB 比 -1.61 / 実効 15 GiB 比 -0.61 GiB) |

生成 ID は 31 個すべて出力され、120k token の prefill 後に末尾で decode が通ることを確認。
decode は初回測定で 6.85 tok/s と出たが、同条件の再測で再現せず (単発の外乱と推測、原因未特定)。上表は再測値。
embedding の H2D は token あたり 10 KiB で、ピーク VRAM は arena 14 GiB + 0.39 GiB の
オーバーヘッドに収まった。速度は 4bpw 等と直接比較しないこと（3.05bpw・c5v5・長文脈）。

### `--arena-gib auto` の挙動と推奨設定

コード確認の結果 (`gpu_memory_manager.hip:15-42`, `compute_runtime.hip:367-399`):

- `auto` は `hipMemGetInfo` の空き VRAM から **512 MiB** (`kAutoMarginBytes`) を引いた値を
  arena capacity にする。16 GiB カードで表示に 0.8 GiB 使われていれば arena ≈ 15 GiB となり、
  KV が arena いっぱいまで自動拡張されるためピークは 15 GiB を超える。
- `-N` は total VRAM から N GiB を引く (`TotalMinus`)。
- 数値 `N` は固定容量。128Ki では KV が 131,073 token 分必要なため、
  **13 GiB では不足、14 GiB で余裕あり**。

**推奨起動設定 (3.05bpw / c5v5 / 128Ki / 投機なし):**

```sh
shelf-compute --model-dir <3.05bpw> --input-ids-file <ids> \
  --max-seq-len 131072 --arena-gib 14 --prefill-chunk 512 \
  --kv-cache-dtype octave:c5v5 --page-tokens 16 --device 0
```

ピーク 14.4 GiB で実効 15 GiB を 0.6 GiB 下回る。さらに削る場合は
`--max-seq-len` を下げる（13 GiB の KV は 97,584 token なので 97,583 以下）か、
prefix cache / workspace の見直しが候補。
