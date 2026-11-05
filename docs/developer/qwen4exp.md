# Qwen4Exp (Qwen3.8-Flash-Next) load contract

Qwen3.8-Flash-Next の `model_type=qwen4_exp` checkpoint を読むための現在の contract である。
forward/executor はまだ接続していない。ロードだけを行う `shelf_qwen4exp` と、
実 checkpoint を検証する `test_qwen4exp_load` が対象である。

## 1. identity と probe

`config.json` の `text_config` が `ple_layer_ids`、`hc_count`、`ngram_size` を持つ
checkpoint を Qwen4Exp と判定する。判定は構造で行い、`model_type` 文字列で
code path を選ばない。`model_type` は allowlist gate であり、`qwen4_exp` と
`qwen4_exp_text` を受理する。

`probe_model_dir` は `ModelFamily::Qwen4Exp` を返す。既存の `LoadedModel`、
tensor-parallel coordinator、pipeline parallel は Qwen4Exp を明示的に拒否する。
Qwen4Exp の executor を接続するまで、`shelf-compute` からは起動できない。

## 2. TP=2 のロード

`load_qwen4exp_rank(model_dir, arena, stream, options)` が 1 rank をロードする。

| 対象 | 配置 |
| --- | --- |
| routed expert | expert 軸で 512 を 2 分割し、rank は連続 256 expert を持つ |
| shared expert | 両 rank に複製 |
| GDN / QSA / indexer | 両 rank に複製 |
| router / shared_expert_gate | 両 rank に複製 |
| embedding / lm_head | 両 rank に複製 |
| HyperConnection / mixer | 両 rank に複製 |
| PLE 小物 | 両 rank に複製 |
| PLE テーブル | CPU RAM に 1 個。rank で複製しない |

EXL3 行列は既存の `materialize_rank_local_canonical` が分割を拒否する。
このため最初のロードでは非 expert を複製する。routed expert だけは
expert 軸で分割する。`TensorPartitionDesc` は使わない。

rank あたりの実測は 26.025 GiB（arena 使用 27.310 GiB）。
512 experts / top-10 は router kernel の上限（experts 512、top-k 16）内である。

### EXL3 行列を TP 分割できない理由

`load_quantized_matrix` は `WeightLoadOptions.partition_plan` に名前があると
`materialize_rank_local_canonical` を呼ぶ。この関数の switch は BF16 と
FP8 block-128 だけを実装し、MXFP4 と EXL3 は `unsupported` を返す
（`src/shelf/weights/canonical_partition.cpp`）。

EXL3 の分割が単純な行コピーにならない理由:

1. EXL3 の codes は `[kt][nt][words]` の 16×16 tile である。N 軸の分割は
   `nt` と `svh` を切り、K 軸の分割は `kt` と `suh` を切る。既存の
   canonical partition は row-major の行と scale stream を前提にしており、
   tile と suh/svh の再配置を持たない。
2. K 軸は 128 要素の Hadamard block を跨げない。EXL3 の入力回転は
   128 要素単位で、`suh` もその単位で定義される。したがって K 軸の
   partition は 128 境界に制限される。
3. 現行の dense FFN の TP 契約は shared expert の down_proj を K 軸で
   640 → 320 に割る。320 は 128 の倍数ではないため、EXL3 の K 軸では
   そのまま表現できない。N 軸だけを部分実装しても shared expert の
   down_proj は複製のままになる。
4. MXFP4 も同じ switch で拒否される。これは EXL3 固有ではなく、
   canonical partition が BF16 / FP8 の row-major 形式を前提にした
   contract だからである。routed expert は expert 軸で分割し、
   matrix axis の partition を避けている。

このため最初のロードは、expert 軸の分割と非 expert の複製を選んだ。
複製しても rank あたり 27.31 GiB で 31.86 GiB に収まる。

## 2-1. EXL3 grouped linear

`launch_gemm_exl3_w16a16_grouped` は 1 launch で全 group を処理する。
既存の dense EXL3 kernel に group 次元を追加したもので、`blockIdx.z` が group である。

- `group_expert[g]` が expert を選び、`trellis_batch_stride` / `suh_batch_stride` /
  `svh_batch_stride` でその expert の平面を指す。
- `group_rows[g]` が group の有効行数（最大 16）を決める。
- `row_token` が非 null のとき、入力は token row 平面で、group row `g * 16 + r` の
  入力行を `row_token` から gather する。null のときは入力がすでに group row 平面である。
- 出力は常に group row 平面で、group `g` は行 `[g * 16, ...)` を書く。
- split-K は grouped では使わない。`group_capacity` は grid.z の上限（65535）以内。

### weight slot と dispatch

- `WeightSlot` に EXL3 expert plane の `exl3_suh_batch_stride_bytes`、
  `exl3_svh_batch_stride_bytes`、`exl3_marker_batch_stride_bytes` を追加した。
- `make_expert_matrix(Qwen4ExpExpertPlane)` が rank-3 の `MatrixWeight` view を作る。
- `weight_slot_validation` は enc=4、experts>1 を受理し、expert ごとの batch stride を
  検証する。
- `KernelId::LINEAR_EXL3_W16A16_GROUPED` を追加し、
  `try_launch_moe_grouped_exl3` が token-row / group-row の両入力を受ける。
- `MOE_GROUPED_LINEAR` の program lowering は EXL3 のとき activation quantize を
  挿入せず、`LINEAR_EXL3_W16A16_GROUPED` を直接発行する。

### 検証

`test_exl3_grouped` は実 checkpoint の layer 0 / expert 0 と expert 1 の
gate_proj を読み、k=2560、n=640、ka=3 half で CPU の `exl3::dequant` と比較する。

| ケース | max_abs |
| --- | ---: |
| 2 group / 2 expert、group-row 入力 | 1.37e-4 |
| 2 group / 2 expert、`row_token` で token-row 入力 | 2.18e-4 |

`test_program_moe_plan` は EXL3 の MoE graph が
`MOE_ROUTER, MOE_PLAN_SCAN, MOE_PLAN_SCATTER, LINEAR_EXL3_W16A16_GROUPED,
SWIGLU, LINEAR_EXL3_W16A16_GROUPED, ...`
の順に lower され、activation quantize を含まないことを固定する。

`test_qwen4exp_load` は実 expert plane から rank-3 `WeightSlot` を作り、
`validate_host_weight_slot` を通す。

## 2-2. HyperConnection

Qwen4Exp の gated residual は、4 stream の hyper stack を対象にする。
最初の primitive として次の 2 つを追加した。

| primitive | 入力 | 出力 |
| --- | --- | --- |
| `HYPER_MIX` | gate `[rows, hc*hidden]`, normed `[rows, hc*hidden]` | mixed `[rows, hidden]` |
| `HYPER_APPLY` | raw `[rows, hc*hidden]`, y `[rows, hidden]`, injection `[rows, hc]` | hyper `[rows, hc*hidden]` |

`HYPER_MIX` は `mean_h(gate[h] * normed[h])`、`HYPER_APPLY` は
`raw[h] + y * injection[h]` を BF16 で計算する。

site 全体は既存 primitive と次の組み合わせで表現できる。

```text
RmsNorm(group_size=hidden, one_plus)
  -> Linear(down) -> Scale(1/hc) -> SiLU -> Linear(up) -> Sigmoid
  -> HYPER_MIX
  -> Linear(block_inject) -> Scale(1/hc) -> Sigmoid -> Scale(2) -> HYPER_APPLY
```

`RmsNormNode` は `group_size` と `ONE_PLUS` を既に持つため、stream ごとの
grouped RMSNorm は新しい kernel を必要としない。down/up/inject は通常の
BF16 linear である。

`test_hyper_connection` は rows 3、hc 4、hidden 128 で CPU 参照と比較する。
`test_program_hyper_connection` は graph が `HYPER_MIX, HYPER_APPLY` に
lower されることを固定する。`test_qwen4exp_hyper_site` は HyperConnection site
全体の sequence を固定する。

## 2-3. GDN の sigmoid 出力ゲート

Qwen3.5 の GDN は gate 分岐に silu を固定していた。Qwen4Exp は
`output_gate_type: sigmoid` を宣言するため、`GdnLoweringGeometry::sigmoid_gate`
を追加し、`emit_gdn` が `SIGMOID` と `SILU` を切り替えるようにした。

`test_qwen4exp_gdn_gate` は sigmoid 指定で
`GDN_RECURRENCE,RMS_NORM,SIGMOID,MUL`、既定で
`GDN_RECURRENCE,RMS_NORM,SILU,MUL` になることを固定する。

## 2-4. PLE gate と broadcast

PLE 注入のうち、状態を持たない 2 つの要素を実装した。

| kernel | 内容 |
| --- | --- |
| `launch_ple_gate` | `sign(x) * sqrt(max(|x|, 1e-6))` |
| `launch_ple_broadcast_mul` | `out[r, h*hidden+i] = sigmoid(gate[r,i]) * value[r,i]` |

`test_ple_gate` は rows 3、hc 4、hidden 128 で CPU 参照と一致することを固定する。

short conv も実装した。

| kernel | 内容 |
| --- | --- |
| `launch_ple_conv` | depthwise、kernel 4、dilation 3、causal、SiLU |

状態は `[max_sequences][9][channels]` BF16 で、request の row を順に処理し、
最後に history 9 行を書き戻す。`test_ple_conv` は 2 sequence と decode 1 row で
CPU 参照と一致することを固定する。

## 2-5. Decoder layer と MoE FFN

GDN 層の decoder layer を 1 つの graph に組み立てる helper を追加した。

```text
attn HyperConnection mix
  -> GDN（sigmoid gate）
  -> HyperConnection apply
  -> mlp HyperConnection mix
  -> MoE FFN
  -> HyperConnection apply
```

- `emit_qwen4exp_moe` は router（global 512、top-10）を置き、routed expert を
  gate / up / down の 3 つの `MoeGroupedLinearNode` に分ける。
  gate/up の入力は token row、down の入力は group row である。
- shared expert は gate_proj / up_proj / down_proj の dense EXL3 linear と
  `shared_expert_gate` の sigmoid で表現する。
- `MoeCombineNode` が expert の部分和と shared expert を token row に戻す。
- QSA（full_attention）層は sparse attention の mask 入力が入るまで `unsupported` とする。
  indexer の graph lowering と key state は実装済みで、残るのは paged attention への
  token mask 適用である。

`test_qwen4exp_layer` は GDN 層について次を固定する。

- attention HyperConnection が GDN に接続する
- `GDN_RECURRENCE,RMS_NORM,SIGMOID,MUL` の sigmoid gate
- attention apply が MLP HyperConnection mix に接続する
- `MOE_ROUTER,MOE_GROUPED_LINEAR,MOE_GROUPED_LINEAR,SWIGLU,MOE_GROUPED_LINEAR`
- `MOE_COMBINE,HYPER_APPLY`

### PLE injection

`emit_qwen4exp_ple` が PLE 注入を graph に接続する。

```text
key_proj -> RMSNorm(norm_key)
hyper    -> RMSNorm(norm_query)
  -> HYPER_MIX -> SCALE(hc / sqrt(hidden)) -> PLE_GATE
value_proj -> PLE_BROADCAST_MUL(gate, value)
  -> RMSNorm(norm_conv) -> PLE_CONV -> RESIDUAL_ADD(gated, short)
  -> RESIDUAL_ADD(hyper, output)
```

- `PLE_GATE` は `sign(x) * sqrt(max(|x|, 1e-6))`
- `PLE_BROADCAST_MUL` は `sigmoid(gate) * value` を hc stream へ複製
- `PLE_CONV` は depthwise、kernel 4、dilation 3、causal、SiLU
- `PLE_CONV` の state は `PrimitiveStateKind::PLE_CONV_STATE`

`emit_qwen4exp_layer` は `has_ple=true` のとき、attention site の前に PLE を置く。
`test_qwen4exp_ple_site` が PLE 単体の sequence を、`test_qwen4exp_layer` が
PLE → attention site → GDN → ... の順序を固定する。

## 2-6. QSA key pooling と block score

QSA indexer のうち、状態を持たない 2 つの計算を実装した。

| kernel | 内容 |
| --- | --- |
| `launch_qsa_pool` | compress_ratio=4 の key 平均、RMSNorm、`1 + weight` |
| `launch_qsa_scores` | `sum_h relu(q · k) / sqrt(head_dim)`、可視 block 外は -inf |
| `launch_qsa_topk` | block ごとの score から top-k（同点は小さい index） |
| `launch_qsa_rope` | partial rotary。`inv_freq` / `position_scale` 1〜2 / `attention_factor` |
| `launch_qsa_key_cache` | raw key の partial block と pooled key の更新 |
| `launch_qsa_expand_mask` | selected block を token mask に展開し causal と AND |

`qsa_rope` は `position_scale <= 2` を受理し、`inv_freq` が渡された場合は
YaRN の補正済み周波数を使う。`qsa_key_cache` は 4 token が揃った時点で
mean → RMSNorm → partial rope を行い、pooled key を書き出す。

`test_qsa_pool_scores` は head_dim 128 で CPU 参照と比較する。
`qsa_topk` はまだ 1 thread / row の正しさ優先実装である。

### QSA key state pool

`QsaKeyStatePool` は sequence slot ごとに次を持つ。

| 領域 | 形状 |
| --- | --- |
| pooled key | `[qsa_layers][max_blocks][head_dim]` BF16 |
| partial key | `[qsa_layers][compress_ratio][head_dim]` BF16 |
| counts | `[qsa_layers][2]` U32（完了 block 数、partial 行数） |

`initialize` / `reset` / `release` / `copy_slot_from` を持ち、`device_view()` が
`ModelDispatchStateView::qsa` に入る。prefix cache の slot copy は pooled key、
partial key、両 count をまとめて複製する。

multi-sequence 用に次の batch kernel を追加した。

| kernel | 形状 |
| --- | --- |
| `launch_qsa_key_cache_batch` | request ごとに 1 block。行を順に partial へ積み、4 行で pooled key を確定する |
| `launch_qsa_scores_batch` | 行ごとに `row_sequence_slots` から sequence の pooled key を解決する |
| `launch_qsa_topk_batch` | 行位置から可視 block 数を求めて top-k する |

`positions` と `DeviceRequestDescriptor` は device memory から読むため、decode で
行数や位置を kernel 引数に焼き込まない。

### QSA graph 接続

次の primitive を追加した。

| primitive | 内容 |
| --- | --- |
| `QSA_ROPE` | indexer query の partial rope。`position_scale` 1〜2 |
| `QSA_KEY_CACHE` | raw key の partial block と pooled key の更新 |
| `QSA_SCORES` | query と pooled key の block score |
| `QSA_TOPK` | block score の top-k |
| `QSA_EXPAND_MASK` | selected block を token mask へ展開 |

`emit_qwen4exp_qsa_indexer` は `index_qk_proj` を q / k に分け、q に
`index_q_layernorm` と `QSA_ROPE`、k に `QSA_KEY_CACHE` を適用し、
`QSA_SCORES` → `QSA_TOPK` → 任意の `QSA_EXPAND_MASK` を接続する。

`test_qwen4exp_qsa_site` は graph が
`LINEAR,RMS_NORM,QSA_ROPE,QSA_KEY_CACHE,QSA_SCORES,QSA_TOPK,QSA_EXPAND_MASK`
に lower され、program が同じ順に dispatch を出し、key cache と scores が
同じ `QSA_KEY_STATE` descriptor を共有することを固定する。

`test_qsa_state_pool` は 2 sequence の key cache を CPU 参照と比較し、
scores / top-k / mask と slot copy を検証する。

残りは top-k の性能版、`PagedAttention` への token mask 入力、
QSA 層全体の lowering、4 stream を運ぶ runtime state の接続である。

## 3. routed expert plane

1 layer、1 projection の routed expert は次の 4 面にパックしたまま GPU に置く。
EXL3 を BF16 などへ展開しない。

```text
codes  [local_experts, trellis_bytes] U8
suh    [local_experts, k]             F16
svh    [local_experts, n]             F16
marker [local_experts, 4]             U8   (mul1)
```

`trellis_bytes = (k / 16) * (n / 16) * words * 2`、
`words = 16 * ka + (half ? 8 : 0)`。gate/up は `k=hidden, n=moe_intermediate`、
down は `k=moe_intermediate, n=hidden`。layer 内の expert は同じ rate と形状を
持つため、per-expert stride は一定である。

## 4. PLE table

`ngram_embedding.safetensors` は `model.safetensors.index.json` に含まれない。
`PleTable::load` が別ファイルとして開き、128 シャードを `pread` で匿名 RAM に読む。

```text
shard       128 × [2,500,012, 81] I16
row         81 words = 162 bytes
word 0      fp16 row scale
word 1..80  160 positions × K=8 bit ring
```

`row[i] = mul1_codebook(state_i) * scale + head_bias[head]`。
`state_i` は位置 i と i-1 の 8 bit を連結した 16 bit である。

行番号:

```text
shifted = shift_right_ignore_eos(history, shift)
mixed   = XOR_{p} shifted[p] * layer_multipliers[p]
row     = mixed % head_vocab_sizes[h] + head_offsets[h]
```

head 0〜7 が bigram、head 8〜15 が trigram。1 token あたり 16 行、
2,560 要素になる。head の prime vocab size、offset、layer multiplier は
config から再計算し、checkpoint のメタデータと照合する。

PLE の小物（key_proj、value_proj、conv1d、norm_key/query/conv）は VRAM に置く。

## 5. ロードしないもの

初回はテキストのみを対象とする。

- MTP: `mtp.*` はロードしない。現行 CLI も `--mtp` と TP=2 の併用を拒否する。
- Vision: `model.visual.*` はロードしない。

## 6. 検証

```bash
./build/tests/test_qwen4exp_load --model-dir /mnt/Qwen3.8-Flash-Next-exl3
```

このテストは PLE 48.28 GiB と TP=2 の両 rank を同時にロードし、
先頭・末尾 expert の格納バイト、PLE の複数行、n-gram hash、形状不変条件を検証する。
