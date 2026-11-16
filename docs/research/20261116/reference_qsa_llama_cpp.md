# 参照 QSA 実装（Qwen-QSA-llama.cpp）の調査と shelf への適用判断

実験 ID `QWEN35QSA-2`。対象は `models/Qwen3.8-27B-QSA-Quark-AWQ-MXFP4`、
R9700 1 枚、`shelf-compute`。参照実装は `~/work/references/Qwen-QSA-llama.cpp`。

## 1. 目的

[QWEN35QSA-1](qwen35qsa.md) は QSA を runtime に載せたうえで「masked attention の
単価が原因で dense より遅い」と結論した。本調査は参照実装の QSA の構造を読み、
shelf の現状と突き合わせたうえで、実測でボトルネックを特定し直し、
shelf に取り込む価値のある要素だけを選別する。

## 2. 参照実装の構成

読んだ範囲:

- `ggml/src/ggml-cuda/qsa-attn.cu`、`qsa-attn-kernels.cuh`（block-sparse attention）
- `src/models/qwen4exp.cpp` の `build_qsa_blk_select` / `build_attn_qsa_blk`
- `src/llama-memory-hybrid-idx.cpp`（indexer raw key と pooled block の cache）
- `README.md`、`qsa27b/RUNTIME.md`（計測結果と変更点）

要点は次の 8 点。

1. **query ごとの block id 列で attention を回す**。`ids [n_wblk, n_tok]` と
   `mask [n_wblk*ratio, 1, 1, n_tok]`（選択 cell だけ 0、他は -inf）を入力にし、
   選択した block の cell だけを cache から読む。context 長の mask を作らない。
   コストは budget に比例し、context 長に依存しない。
2. **CTA は `(split, kv_head, token)`、warp が同じ KV head を共有する query head を
   分担する**。K/V は kv head あたり 1 回しか読まない（GQA 6 なら 6 分の 1）。
3. **block list を CTA 間で分割し、online softmax の状態を後段 kernel で結合する**
   （flash-decoding と同じ）。decode のように token 数が少ないとき GPU を埋める。
4. **prefill も同じ kernel で回せる**（`n_tok > 16`）。ただし Qwen35 では
   KV が 65536 token 以下のときは従来の masked dense を選ぶ。prompt batch では
   dense flash attention の方が速いという実測に基づく閾値。
5. **cache が budget 以内なら選択そのものを行わない**（`(c0+1)/ratio >= n_sel+1`
   を満たすまで masked dense）。全 block が残る領域では選択の意味がない。
6. K/V cache が Q8_0 のときは、q を 32 dim ごとに int8 化して `__dp4a` で
   内積する。K を dequantize しない。F16 cache 用の variant もある。
7. indexer の raw key は ring buffer で直近の行だけ持つ
   （`LLAMA_QSA_RAW_RING`、262K で約 2 GiB 削減）。
8. indexer の重みは FP32 で保持し、norm は `1 + weight` を畳み込む。

shelf と意味論が一致している点は多い。block 選択は「4 head の relu 付き内積和 ×
`1/sqrt(head_dim)`」、pooled key は block 先頭 token 位置で RoPE、indexer norm は
`1 + weight`、現在の partial block は常に可視、state は partial block 1 個分だけ保持
（raw ring 相当）である。したがって **選択アルゴリズムの移植は不要**で、
差分は attention の実行形と dispatch にある。

## 3. 実測によるボトルネックの特定

測定条件: `shelf-compute`、`--device 0`、bf16 KV、page 16、greedy、
`--max-num-batched-tokens 2048`、prompt は `tests/data/mtp_perf/prose.txt` を
model の tokenizer で encode して先頭 N token を使用。decode は 16 token 生成
（計測区間 15 token）。dense 参照は同じ checkpoint を hardlink し、`config.json`
から `qsa` を除いて `model_type` を `qwen3_5` にしたもの。

### 3.1 修正前の値

| context | dense prefill | QSA prefill | dense decode | QSA decode |
| --- | ---: | ---: | ---: | ---: |
| 3k | 1,302 ms | 3,287 ms | 26.18 tok/s | 14.21 tok/s |
| 16k | 8,328 ms | 17,950 ms | 24.64 tok/s | 14.53 tok/s |

QWEN35QSA-1 の値（1,885 / 3,897 ms、25.82 / 14.48 tok/s）と概ね一致する
（run 間で 10〜15 % の変動がある）。

### 3.2 kernel trace と dispatch 統計

`rocprofv3 --kernel-trace` と `SHELF_QWEN35_PERF_STATS=1` で 16k を計測した。

修正前の decode は kernel 合計 57.6 ms/token（dense 36.4 ms/token）。内訳で目立つのは
QSA kernel ではなく **`model_dispatch_kernel`（correctness interpreter）が
20 ms/token** を占めていた点である。prefill では 1 layer・1 chunk あたり
237 ms + 54 ms + 2 ms が積み上がり、16 layer × 2 chunk で約 9.4 s になる。
これは prefill の差分（17.9 s − 8.3 s = 9.6 s）ほぼ全体に相当する。

dispatch 統計では、QSA 層ごとに次の 3 primitive が correctness 経路に落ちていた。

| KernelId | 図形 | 落ちた理由 |
| --- | --- | --- |
| `LINEAR_BF16` | indexer `q_proj` 512×5120 | `kLinearShapeRules` に BF16 の登録がない |
| `LINEAR_BF16` | indexer `k_proj` 128×5120 | 同上（そもそも登録がない） |
| `RMS_NORM` | indexer `k_norm` 128 feature / PerFeature | `kRmsNormRules` に 128 の rule がない |

`{512, 5120}` は FP8 finegrained としてだけ登録されており、`{128, 5120}` は
未登録だった。indexer の投影は BF16 なので、どちらも allowlist に一致せず
interpreter に落ちる。参照実装側にはこのような形状 allowlist がなく、
`mul_mat` が任意形状を処理するため、この差は shelf 固有のもの。

### 3.3 上記 3 primitive を allowlist に追加した試験実装

`linear_selector.cpp` に `{512, 5120, BF16|FP8}` と `{128, 5120, BF16}`、
`rmsnorm_selector.cpp` に `{BF16, BF16, BF16, PerFeature, 128}` を追加して
再計測した（測定後に revert 済み）。

| context | dense prefill | QSA prefill | dense decode | QSA decode |
| --- | ---: | ---: | ---: | ---: |
| 3k | 1,302 ms | 1,974 ms | 26.18 tok/s | 23.75 tok/s |
| 16k | 8,328 ms | 9,114 ms | 24.64 tok/s | 23.23 tok/s |
| 32k | 20,188 ms | 20,545 ms | 22.65 tok/s | 23.37 tok/s |
| 64k | 54,247 ms | 49,180 ms | 19.85 tok/s | 22.64 tok/s |

32k と 64k の生成 token 列は dense と完全一致した。16k では 10 token まで一致し、
11 token 目で分岐した（選択が効く領域なので想定内）。選択結果が変わり得るのは
indexer の内積の累積順が変わるためで、block 選択は離散なので出力の一致は保証されない。

### 3.4 修正後の残コスト

16k、decode 1 token あたりの kernel 時間（7 token 分の平均、両方とも 38 ms 前後）。

| 項目 | 修正前 QSA | 修正後 QSA | dense |
| --- | ---: | ---: | ---: |
| kernel 合計 | 57.6 ms | 38.0 ms | 36.4 ms |
| indexer BF16 線形（32 GEMV） | interpreter に含まれる | 0.21 ms | 0 |
| masked attention（16 layer） | 3.44 ms | 3.59 ms | 3.06 ms |
| select（score + topk） | 0.86 ms | 0.89 ms | 0 |
| lm_head BF16 GEMV（248320×5120） | 4.0 ms | 4.0 ms | 4.0 ms |

64k prefill の kernel 内訳（R1 のみ、合計 47.2 s）:

| kernel | 時間 | 割合 |
| --- | ---: | ---: |
| `v_prefill_bf16::attention_paged_prefill_bf16_impl` | 19.4 s | 41 % |
| `gemm_mxfp4_w4a8_prefill_2d_kernel` | 17.7 s | 38 % |
| `qsa_score_wmma_kernel` | 1.87 s | 4 % |
| `shelf_gdn_recurrence_wmma_lossy` | 2.23 s | 5 % |
| `qsa_topk_kernel` | 0.53 s | 1 % |

64k では保持 block が 3 % しかないのに、prefill attention が全体の 41 % を占める。
shelf の prefill は page（16 token）単位でしか間引かないため、選択された 4-token
block 1 個に対して 16 token 分の tile を計算する。選択 block はほぼ別 page に散るので、
実効の計算量は budget の 4 倍（12.5 %）になる。これが参照実装の block id 駆動
kernel との最も大きな差である。

## 4. shelf に適用すべきもの

### R1. indexer の BF16 線形と RMSNorm を allowlist に追加する（ACCEPT）

- 変更: `kLinearShapeRules` に `{512, 5120}` の BF16、`{128, 5120}` の BF16、
  `kRmsNormRules` に `{BF16, BF16, BF16, PerFeature, 128, 128}`。
- 効果（実測）: 16k で decode 14.53 → 23.23 tok/s、prefill 17,950 → 9,114 ms。
  64k では decode が dense を 14 % 上回る。
- 必要な検証: `test_qwen35_linear_selector` と `test_qwen35_rmsnorm_selector` への
  図形追加、`test_gemm_bf16_wmma` への `N=128, K=5120` ケース追加、
  QSA 有効時の生成品質（選択 block が変わるため bit 一致は期待しない）。
- 優先度: 最高。QSA を有効にすると発生する退行のほぼ全量がこれ 1 件である。

### R2. block id 駆動の block-sparse attention kernel を追加する（ACCEPT、次点）

参照実装の `qsa_attn_kernel` と同じ設計を shelf の paged KV に対して実装する。

- 入力は query ごとの選択 block id 列と cell mask。shelf は `qsa_topk_kernel` が
  `selected` と bitmap を既に作っているので、`selected` を配線するか kernel 内で
  bitmap を popcount して列挙すればよい。
- `compress_ratio = 4`、`page_tokens = 16` なので 1 block は必ず 1 page 内の
  連続 4 cell に収まる。block id から `page = block/4`、`offset = (block%4)*4` で
  直接読める。
- grid は `(split, kv_head, row)`、warp が GQA の 6 query head を分担し、
  K/V を kv head あたり 1 回読む。block list を split し、最後に online softmax の
  状態を結合する。
- 期待効果: decode attention 3.5 ms/token → 0.5〜1.0 ms/token（dense の 3.06 ms を
  下回る水準）。64k prefill attention 19.4 s → 約 5 s（16 token tile の 4 倍
  オーバーリードを除去）。prefill の切り替えは実測した crossover に置く
  （参照実装の 65536 は CUDA の masked FA に対する値で、shelf には当てはまらない）。
- 優先度: R1 の次。短中 context では数 % だが、長 context の prefill では最大の伸びしろ。

R2 を入れない場合の代替は、mask 付き decode で現在無効化されている split 経路
（`plan.splits = 1` に落としている分岐）を有効化し、page bitmap による間引きを
split 側にも入れること。ただし prefill の 4 倍オーバーリードは解消しない。

### R3. BF16 exact-rows GEMV の K 方向を 4 chunk まとめる（ACCEPT、小）

当初は「indexer の 32 GEMV が 4.24 ms/token」と見積もったが、これは kernel 全体の
時間を indexer に帰属させた誤りだった。内訳は lm_head（248320×5120、1 token あたり
4.0 ms、DRAM 帯域なり）が大半で、indexer の 2 投影は 0.21 ms/token に過ぎない。
`gemm_bf16_exact_rows_body` の K ループは `k` が runtime 値なので展開されず、
1 反復 1 load でメモリレイテンシを隠せない。`kRows == 1` の経路だけ 4 chunk 分の
load を先に発行する形に変えた。効果は N=128・K=5120 で 6.3 → 3.5 µs（44 %）、
N=512・K=5120 で 10.5 → 9.9 µs（6 %）。decode 全体では 0.2 ms/token 程度。

### R4. 参照実装のその他の要素は取り込まない（REJECT）

- **Q8_0 cache + `__dp4a`**: shelf の KV は BF16 / FP8 / octave で、Q8_0 cache を
  持たない。FP8 KV の内積を int8 化する利点も見込めない。
- **raw indexer key の ring buffer**: shelf は partial block 1 個分しか持たない
  （`partial_layer_stride = compress_ratio * head_dim`）。既に同等。
- **budget 以内の dense 固定**: shelf は lowering 時に `max_context_tokens` で
  判定している。per-request 判定に広げる案は、2048 token で選択を有効にした
  場合と無効にした場合の decode が 23.45 tok/s で一致したため、効果がない。
  全 block が選ばれると mask 経由でも dense と同じ page 列になるため。
- **masked prefill の全面維持**: 却下。masked prefill は 8k 付近まで sparse kernel
  より速いが、16k 以上では sparse が勝つ（§8）。masked と sparse を可視 token 数で
  切り替える。

### R5. bench harness で Qwen35Qsa を動かせるようにする（ACCEPT、小）

`shelf-bench tg` / `pp` は `ExecutorConfig::max_context_tokens` を設定しないため、
Qwen35Qsa の lowering が `qwen35qsa: lowering requires max_context_tokens` で失敗する。
QSA の性能評価は現状 `shelf-compute` でしか行えない。bench 側で
`--context`（pp は `--prompt-tokens`）から設定する。

## 5. QWEN35QSA-1 の結論の訂正

QWEN35QSA-1 は「masked attention の単価が原因で QSA は dense より遅い」と結論したが、
これは誤りである。遅さの実体は indexer の BF16 線形と RMSNorm が correctness
interpreter に落ちていたことで、masked attention 自体の差は 16k decode で
+0.5 ms/token、prefill で +4 % に過ぎない。R1 の適用後は 32k で dense と同等、
64k で dense を上回る。QSA の性能評価は R1 を前提にやり直す必要がある。

## 6. 未検証

- 128k 以上の context。64k までの実測のみ。
- 選択結果が変わることによる品質差。32k / 64k の生成列は dense と一致したが、
  品質の網羅的な比較はしていない。
- speculative decode（MTP / DFlash2）との併用。QSA family は MTP block をまだ
  load しない（`LoadedModel::has_mtp_block()` が false）。tree 行が来た場合は
  `tree_anc_offsets != nullptr` により sparse kernel を選ばず masked 経路に落ちる。
- FP8 / octave KV での QSA。sparse kernel は BF16 cache のみを対象にしている。
- 混合 batch の prefill region。`plan.splits` を共有するため、region 単位の
  masked/sparse 切り替えはしていない。

## 7. 再現

```sh
# prompt ids（prose を model の tokenizer で encode し、先頭 N 件を使う）
python3 - <<'PY'
from tokenizers import Tokenizer
import pathlib
tok = Tokenizer.from_file("models/Qwen3.8-27B-QSA-Quark-AWQ-MXFP4/tokenizer.json")
ids = tok.encode(pathlib.Path("tests/data/mtp_perf/prose.txt").read_text()).ids
big = []
while len(big) < 70000:
    big += ids
pathlib.Path("/tmp/opencode/qsa_prompt_ids.txt").write_text("\n".join(map(str, big)) + "\n")
PY

# decode / prefill
./build/shelf-compute --model-dir models/Qwen3.8-27B-QSA-Quark-AWQ-MXFP4 \
  --input-ids-file /tmp/opencode/qsa_16384.txt --max-new-tokens 16 \
  --max-seq-len 16640 --device 0 --prefix-cache 0 --kv-cache-dtype bf16 \
  --temperature 0 --max-num-batched-tokens 2048

# kernel trace
rocprofv3 --kernel-trace --output-format csv --output-file /tmp/qsa16k -- \
  ./build/shelf-compute ...（同上）
```

dense 参照は同じファイルを hardlink し、`config.json` から `qsa` を除いて
`model_type` を `qwen3_5` にしたディレクトリで実行する（測定後に削除）。

## 8. 実装と最終計測

R1 / R2 / R3 / R5 を実装した。対象の family は `qwen35qsa` で、Qwen4Exp の QSA
（`q_per_kv = 12`）は対象外として masked 経路のままにする。

### 8.1 実装

- R1: `kLinearShapeRules` に `{512, 5120, BF16}` と `{128, 5120, BF16}`、
  `kRmsNormRules` に `{BF16, BF16, BF16, PerFeature, 128, 128}` を追加。
- R2: `src/shelf/kernels/optimized/attention/qsa_sparse.hip` を追加。
  - 1 CTA は `(split, kv_head, row)`、threads は `(32, 2)`。2 warp が 3 query head
    ずつを担当し、1 block の K/V を 16 B load × 4 cell で 1 回だけ読む。
  - block bitmap を 32 bit 語ごとに走査し、選択 block だけを処理する。block は
    `page = block / (page_tokens / 4)`、`offset = (block % (page_tokens / 4)) * 4` で
    1 page 内の連続 4 cell に写る。
  - cell ごとの online softmax は head の 4 lane で共有し、V の蓄積は全 lane が
    自分の 8 dim を持つ。softmax の状態は `red[2][32][13]` を LDS で転置して作る。
  - split した場合は `[m(q), l(q), acc(q, 256)]` を
    `kDecodeAttnPartialBytes` の workspace に書き、`qsa_sparse_reduce_kernel` が
    1 CTA / (row, q_head) で結合する。部分状態のレイアウトは既存の split decode
    と同じ形にした。
  - `qsa_sparse_splits` は decode で約 128 CTA になるよう split し、workspace と
    32 split の上限で制限する。prefill は row 数で GPU が埋まるので split 1。
- R3: `gemm_bf16_exact_rows_body` の `kRows == 1` 経路で 4 chunk 分の load を
  先に発行する。
- R5: `shelf-bench tg` / `pp` が `ExecutorConfig::max_context_tokens` に
  `--context` / `--prompt-tokens` 由来の sequence 長を渡す。

### 8.2 dispatch 契約

`kLinearShapeRules` の allowlist とは別に、paged attention の plan が
「BF16 KV、F32 output、block mask あり、tree ancestor なし、head_dim 256、
`q_heads == 6 * kv_heads`、block shift 2、page_tokens が 4 の倍数」を満たすときだけ
`QsaSparse` variant を選ぶ。それ以外は従来の masked kernel のまま。

prefill は `max_visible_tokens < 12288` の間 masked tiled kernel を使い、
それ以上で sparse kernel に切り替える。同じ 8k prompt を閾値 8192 にして最後の
chunk だけ sparse にすると 4,474 ms（全 chunk masked は 4,332 ms）、16k では
masked 9,075 ms / sparse 8,568 ms だったため、crossover を 12,288 に置いた。chunked prefill では chunk の `sequence_length` が
この閾値を越えた時点から sparse になる。

### 8.3 最終計測

`shelf-compute`、R9700 1 枚、bf16 KV、page 16、greedy、`--max-num-batched-tokens 2048`。

| context | dense prefill | QSA prefill | dense decode | QSA decode |
| --- | ---: | ---: | ---: | ---: |
| 3k | 1,302 ms | 1,963 ms | 26.18 tok/s | 25.54 tok/s |
| 8k | - | 4,332 ms（masked） | - | 25.31 tok/s |
| 16k | 8,328 ms | 8,394 ms | 24.64 tok/s | 25.09 tok/s |
| 32k | 20,188 ms | 17,383 ms | 22.65 tok/s | 24.70 tok/s |
| 64k | 54,247 ms | 35,082 ms | 19.85 tok/s | 23.85 tok/s |

32k と 64k の生成 token 列は dense と一致した。16k でも 16 token が R1 のみの
結果と一致した。3k は選択が効くため dense と別関数であり、速度も dense を
わずかに下回る（decode −2.4 %、prefill は masked のまま）。

16k decode の kernel 内訳（1 token あたり）:

| 項目 | R1 のみ | R1 + R2 |
| --- | ---: | ---: |
| QSA attention + reduce | 3.59 ms | 0.90 ms |
| select（score + topk） | 0.89 ms | 0.78 ms |
| indexer BF16 線形 | 0.21 ms | 0.21 ms |

64k prefill の kernel 内訳（R1 + R2、合計 34.1 s）:

| kernel | 時間 |
| --- | ---: |
| `gemm_mxfp4_w4a8_prefill_2d_kernel` | 18.1 s |
| `qsa_sparse_attention_kernel<false>`（112 launch） | 5.5 s |
| `shelf_gdn_recurrence_wmma_lossy` | 2.2 s |
| `qsa_score_wmma_kernel` | 1.9 s |
| `qsa_topk_kernel` | 0.5 s |
| `attention_paged_prefill_bf16_impl`（閾値未満の 1 chunk） | 0.5 s |

masked prefill の 19.4 s が 5.5 s になった。残る QSA 固有コストは select
（score + topk）の 2.4 s で、prefill 全体の 7 % を占める。

### 8.4 採用しなかったもの

§4 R4 のとおり。加えて、参照実装の `LLAMA_QSA_RAW_RING` に相当する削減は
shelf では既に partial block 1 個分しか持たないため不要だった。indexer 重みの
FP32 保持も、shelf は BF16 で同じ選択結果の範囲に収まるため採用しない。
