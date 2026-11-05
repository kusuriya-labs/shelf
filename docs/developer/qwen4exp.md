# Qwen4Exp (Qwen3.8-Flash-Next) contract

Qwen3.8-Flash-Next の `model_type=qwen4_exp` checkpoint を読み、text-only で実行する
ための現在の contract である。対象は TP=2 / TP=4、BF16 KV、48 層、PLE 有効、
checkpoint は `models/Qwen3.8-Flash-Next-Quark-MXFP4`（routed / shared expert は
MXFP4、attention / GDN は per-channel FP8、PLE は per-tensor FP8）。

## 1. identity と起動

`config.json` の `text_config` が `ple_layer_ids`、`hc_count`、`ngram_size` を持つ
checkpoint を Qwen4Exp と判定する。`model_type` は allowlist gate であり、
`qwen4_exp` と `qwen4_exp_text` を受理する。identity で code path を選ばない。

`shelf-compute` は `--tensor-parallel 2|4` で起動する。Qwen4Exp では prefix cache を
無効にする（PLE の n-gram history が cache から復元した prefix を持たないため）。
MTP は無効である。DFlash2 は `--dflash2-model-dir` で有効になる。

## 2. 重みの形式とロード

| 領域 | checkpoint の形式 | rank の持ち方 |
| --- | --- | --- |
| routed expert | MXFP4 E2M1 / E8M0 group32 | expert 軸で 512 を tp_size 分割し、rank-3 `MatrixWeight` |
| shared expert | MXFP4 | 全 rank に複製 |
| attention / GDN | FP8 E4M3 per-channel + F32 `[n]` | 全 rank に複製、preshuffle して WMMA fragment order |
| PLE table | FP8 E4M3 per-tensor + BF16 `[1]` | CPU RAM に 1 個を共有 |
| embed / lm_head / hyper / router | BF16 | 全 rank に複製 |

- routed expert は `preshuffle_mxfp4_native` で 16 出力 x 32 K の 256 バイト block に
  repack する。E8M0 scale は row-major、row reference は行ごとの最大 E8M0 byte。
- attention / GDN は `preshuffle_fp8_native` で gfx1201 の FP8 WMMA fragment order に
  する。活性は per-token dynamic FP8（`scale = amax / 448`）で、GEMM の epilogue で
  `weight_scale[o] * activation_scale[row]` を 1 回だけ適用する。
- PLE は 128 シャードを `pread` で読み、`row = mul1_codebook` ではなく
  E4M3 code × per-tensor scale で復号する。`weight_scale` は **BF16** である。
- MXFP4 / FP8 の preshuffle staging は関数ローカルの buffer を非同期コピーしない。
  同期コピーで device へ移してから buffer を破棄する。

## 3. TP の contract

Qwen4Exp で TP 集約するのは **MoE combine だけ**である。

- routed expert は expert 軸の部分和を MoE combine で足す。
- shared expert は rank0 だけが寄与し、他 rank は 0 を足す。
- attention / GDN / hyper / embed / lm_head / PLE 小物は複製なので、
  `tp_combine` を付けない。GDN / attention の geometry は
  `tp_combine = false` で lower する。
- PLE table は同一 process の rank で 1 個を共有する。

### 活性 workspace

同じ入力 `ValueId` を token 行の dense linear と group 行の MoE grouped linear が
読む場合、活性 quantize workspace を共有してはならない。program builder は group 行の
plane を bit 63 で分離し、token 行と group 行それぞれに
`ACTIVATION_QUANTIZE_MXFP4_ROWREF` を出す。group 行 plane の row 数は
`grouped_row_capacity(tokens, {local_experts, top_k})`、token 行 plane は row bucket の
token 行数である。

## 4. 層の構成

```text
embedding -> REPEAT_STREAMS(hc)
for each layer:
  PLE (layer 1 のみ)
  attn HyperConnection mix
  GDN または QSA
  HyperConnection apply
  mlp HyperConnection mix
  MoE
  HyperConnection apply
final hyper_connection_mixer
lm_head (final norm なし)
```

- embedding は hc 個の同一 stream へ複製する。
- GDN は `output_gate_type: sigmoid` を使う。Qwen3.5 の silu とは切り替える。
- QSA は indexer の token mask を `PagedAttentionNode` の 2 番目の入力にする。
  mask がある場合、attention selector は tiled prefill / split decode を選ばない。
- MoE は router（global 512、top-10）、gate/up/down の 3 grouped linear、shared expert、
  combine で構成する。shared expert の gate は sigmoid。
- PLE は layer 1 の attention site の前に注入する。key_proj / value_proj /
  norm_key / norm_query / norm_conv / short conv を使う。

### HyperConnection

`HYPER_MIX` は `mean_h(gate[h] * normed[h])`、`HYPER_APPLY` は
`raw[h] + y * injection[h]` を BF16 で計算する。`HYPER_INJECT` は
`2 * sigmoid(x / hc)` を 1 launch で書く。site 全体は次の順に lower される。

```text
RMS_NORM -> LINEAR(down) -> SILU(1/hc) -> LINEAR(up) -> HYPER_MIX
  -> LINEAR(inject) -> HYPER_INJECT -> HYPER_APPLY
```

### PLE

`PleTable::decode_row` は E4M3 code × per-tensor BF16 scale を復号する。
`ngram_rows` は 16 head（bigram 8 + trigram 8）の行番号を返し、
`PleInputRuntime` が sequence ごとの token history から 16 行を gather する。
1 token あたり 16 行 = 2,560 要素である。

### QSA

`QsaKeyStatePool` は sequence slot ごとに pooled key / partial key / counts を持つ。
`QSA_KEY_CACHE` は 4 token が揃った時点で mean → RMSNorm → partial RoPE を行い、
pooled key を書く。`QSA_SCORES` / `QSA_TOPK` / `QSA_EXPAND_MASK` が token mask を作り、
paged attention が消費する。QSA 層は 4 層周期（0 始まりで 3, 7, ..., 47）。

## 5. executor

`create_qwen4exp_model_executor` が QSA key state、PLE conv state、PLE input runtime を
作る。新しい sequence の最初の step（`position == 0`）で QSA / PLE state を zero に
する。prefix cache は無効である。

### DFlash2 target tap

`Qwen4ExpLowerOptions::hidden_taps` に target 層を渡すと、各層の完了出力
（hyper stream 4 本）を平均した `[rows, hidden]` を external output の後ろへ追加する。
DFlash2 はこの値を target feature として使う。tap は
`ExecutorConfig::target_hidden_taps` から渡り、`run_dflash2_shot` と serve の
両方が同じ経路を使う。

Qwen4Exp の DFlash2 は `--dflash2-drafts` を 6 以下に制限する。block_size 8 の
8 行 verify は 64 token で target-only と分岐する未解決課題があるため、
7 以上を要求された場合は 6 に cap して stderr に理由を出す。

## 6. 数値ダンプ

`SHELF_LAYER_DUMP_DIR=<dir>` を設定すると、層出力と well-known な中間値
（`L<layer>.gdn_*`、`L<layer>.moe_*`、`L<layer>.hidden_tap` など）を
BF16/F32 の raw ファイルと `.meta` に書き出す。ファイル名は
`tp<rank>_run<run>_<name>.bin` で、`.meta` に rows / features / stride / dtype / id を
持つ。デバッグ専用で、通常の実行では何も書かない。

## 7. 検証

```bash
./build/tests/test_qwen4exp_lower
./build/tests/test_qwen4exp_layer
./build/tests/test_qwen4exp_qsa_layer
./build/tests/test_program_activation_planes
./build/tests/test_fp8_ptpc
```

実 checkpoint の E2E:

```bash
./build/shelf-compute --model-dir models/Qwen3.8-Flash-Next-Quark-MXFP4 \
  --input-ids-file <5 token> --max-new-tokens 8 --max-seq-len 128 \
  --device 0,1,2,3 --tensor-parallel 4 --prefix-cache 0 --kv-cache-dtype bf16 \
  --temperature 0
```

期待する生成は ` Paris. The capital of Germany is Berlin` で始まる。
同じ条件の 2 回実行で logits は bitwise 一致する。
