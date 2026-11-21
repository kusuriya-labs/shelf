# Qwen4Exp (Qwen3.8-Flash-Next) contract

Qwen3.8-Flash-Next の `model_type=qwen4_exp` checkpoint を読み、text-only で実行する
ための現在の contract である。対象は TP=2 / TP=4、BF16 KV、48 層、PLE 有効、
checkpoint は `models/Qwen3.8-Flash-Next-Quark-MXFP4`（routed / shared expert は
MXFP4、attention / GDN は per-channel FP8、PLE は BF16 shard を load 時に
INT6 W32 FP16 へ変換）。

## 1. identity と起動

`config.json` の `text_config` が `ple_layer_ids`、`hc_count`、`ngram_size` を持つ
checkpoint を Qwen4Exp と判定する。`model_type` は allowlist gate であり、
`qwen4_exp` と `qwen4_exp_text` を受理する。identity で code path を選ばない。

`shelf-compute` は `--tensor-parallel 2|4` で起動する。prefix cache は QSA key state と
PLE conv state を shadow slot に保持し（[prefix_cache.md](prefix_cache.md)）、restore 時に
PLE の n-gram history を復元 prefix で再 seed する。DFlash2 は
`--dflash2-model-dir` で有効になる。`--mtp` も使える。MTP block は rank0 が load し、
TP で target の lm_head が語彙分割されるときは drafter 用に全語彙の lm_head を
rank0 へ追加で load する（[mtp.md](mtp.md)）。QSA / PLE を持つため verify は target の
1 行 decode を使い、生成列は target-only と一致する。routed expert は
`--expert-parallel` で expert 軸に分割でき、TP と EP の積が device 数になる。

`--pipeline-parallel` にも対応する。各 stage は担当地の層だけをロードし、
embedding は先頭、final mixer と lm_head は末尾の stage が持つ。stage 間では
HyperConnection の stream 全体（`hc_count * hidden_size`）を bf16 で受け渡す。
routed expert は EP、backbone と shared expert は TP で分割する。EP=1 では
従来どおり expert 軸を TP rank に重ねる。PP stage 内の EP / TP 展開は実装済みで、
TP reduce の後に EP reduce を行う。ただし PP と TP の併用は 2048 token の
prefill で non-finite になる既知の問題があり、PP 単体と短いプロンプトだけを
確認している。

## 2. 重みの形式とロード

| 領域 | checkpoint の形式 | rank の持ち方 |
| --- | --- | --- |
| routed expert | MXFP4 E2M1 / E8M0 group32 | expert 軸で 512 を EP rank に分割し、rank-3 `MatrixWeight`（EP=1 では TP rank） |
| shared expert | MXFP4 | 全 rank に複製 |
| attention / GDN | FP8 E4M3 per-channel + F32 `[n]` | 全 rank に複製、preshuffle して WMMA fragment order |
| PLE table（BF16 shard checkpoint） | HF shard 内の BF16 `[2500012, 160]`。load 時に INT6 W32 FP16 へオンライン変換して CPU RAM に保持 | CPU RAM に 1 個を共有 |
| PLE table（FP8 shard checkpoint） | FP8 E4M3 per-tensor + BF16 `[1]` | CPU RAM に 1 個を共有 |
| PLE table（Ngram4 / Ngram5） | bundle root の `ngram_manifest.json`（`format: "ngram-q"`）と対応 safetensors shard | CPU RAM に 1 個を共有 |
| embed / lm_head / hyper / router | BF16 | 全 rank に複製 |

- routed expert は `preshuffle_mxfp4_native` で 16 出力 x 32 K の 256 バイト block に
  repack する。E8M0 scale は row-major、row reference は行ごとの最大 E8M0 byte。
- Qwen4Exp は TP=1 / 2 / 4 で実行できる。TP=1 は単一 GPU 構成で、`shelf-bench tg` と
  同じ配置（`--moe-host-experts` で expert を host に退避）を使う。
- lm_head は load 時に BF16 から INT8 W32 へオンライン量子化する
  （`--lm-head-int8 0` / `SHELF_LM_HEAD_INT8=0` で無効化）。量子化すると
  vocab-parallel head は `Int8W32`、DFlash2 の sharded proposal も `Int8W32` 経路で
  動く。無効にすると BF16 のままになり、DFlash2 の sharded proposal は
  BF16 用の exact-rows GEMM 経路を使う。
- `--mtp` で TP を使うとき、rank0 は target の語彙分割 lm_head とは別に、drafter 用の
  全語彙 lm_head を `mtp_lm_head` として追加 load する。target の forward は分割 head の
  ままである。
- routed expert は既定で全 local expert を VRAM に置く。`--moe-host-experts N`
  （layer あたり）を設定すると、各 layer の N expert を pinned host の plane に置き、
  残りを VRAM の slot に置く。host plane は `hipHostMallocMapped` で、GPU の grouped
  GEMM が PCIe 越しに読む。expert ごとの配置は device の slot map
  `int32 [layers][local_experts]` が持つ（`>= 0` は VRAM slot、`-1 - slot` は host slot）。
  EXL3 expert も同じ offload を使う。trellis / suh / svh を host の
  `host_codes / host_scales / host_row_ref` に置き、marker は全 expert で共通のため
  VRAM の 1 枚を共有する。grouped GEMM は hit/miss を別 launch にし、miss 側の
  group index は `miss_base` を `group_row_base` として元の group へ写像する。
  group fuse は hit/miss それぞれの `expert_group_begin` を使い、host offload 中も有効である。
- attention / GDN は `preshuffle_fp8_native` で gfx1201 の FP8 WMMA fragment order に
  する。活性は per-token dynamic FP8（`scale = amax / 448`）で、GEMM の epilogue で
  `weight_scale[o] * activation_scale[row]` を 1 回だけ適用する。
- PLE table は checkpoint の形式で 3 経路ある。bundle root に `ngram_manifest.json` を持つ
  checkpoint は `Ngram4` / `Ngram5`（`ngram-q`、[ngram_quantization.md](ngram_quantization.md)）
  としてロードする。HF shard 内の BF16 table は load 時に **INT6 W32 FP16** へオンライン変換する。
  HF shard 内の FP8 E4M3 per-tensor table は E4M3 code × per-tensor scale で復号する
  （`weight_scale` は BF16）。インデックス経路はいずれも 128 シャードを最大 16 worker で
  読む。すべて起動時は rank の weight load と並行に prefetch する。
- INT6 W32 FP16 は 32 値ごとに符号付き 6-bit code（-32..31）と FP16 scale を持ち、
  1 group を 24 byte の code と 2 byte の scale で表す。160 値の行は 130 byte
  （6.5 bit/value）になる。scale は group の `max(max_positive / 31,
  -min_negative / 32)` で、両側の極値を飽和させない。全 zero の group は scale 1 と
  code 0 にする。量子化器は AVX512F/BW/VL/BMI2 がある CPU では AVX512 経路、
  それ以外は scalar 経路を使う。BF16 shard は 20 MiB の窓で読み、量子化後に
  `madvise(MADV_DONTNEED)` と `posix_fadvise(DONTNEED)` で page cache を落とす。
  I/O は `--ple-stream-io pread|mmap|direct` で選ぶ（既定 mmap、direct は
  O_DIRECT）。BF16 table 全体を anonymous RAM にも page cache にも保持しない。
- 初回のオンライン量子化後、`<model_dir>/quanted_ple.bin` に INT6 table を書く。次回
  以降は source shard の size と mtime が header の fingerprint と一致する場合に
  その artifact を読む。artifact は 4096-byte 境界の shard payload を持ち、
  `--ple-artifact 0` で読み書きを無効にできる。stale / invalid なら再量子化して
  上書きする。書き込み失敗は model load を失敗させない。
- `--ple-hot-cache-gib N` を 0 以外にすると、PLE テーブルを全量 host RAM へ読み込まず、
  mmap のページキャッシュを Cold 側、pinned host のブロックキャッシュを Hot 側として
  使う。INT6 artifact（`quanted_ple.bin`）および `Ngram4` / `Ngram5` に適用される。
  ブロックは 2048 行で、INT6 artifact の場合は 260 KiB である。INT6 artifact の
  Hot 側への読み込みは O_DIRECT + pread で行い、行アクセス頻度で 8192 行ごとに入れ替え、
  ピーク頻度が 4096 に達したら減衰させる。Hot キャッシュの実バイトは
  `MEMORY_PLAN host memory` の `PLE table` に表示され、`payload_bytes()` は
  テーブル全体のサイズのままである。`SHELF_PLE_CACHE_LOG_INTERVAL` に 1 以上の
  整数を設定すると、入れ替えごとに Hot ブロック数と Hit/Miss を stderr に出す
  （既定は無効）。
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
- QSA は indexer の block bitmap を `PagedAttentionNode` の 2 番目の入力にする。
  bitmap がある場合、fp8 KV と行数の少ない chunk は tiled prefill を選ばない。
  `q_per_kv = 12` も QSA block-sparse kernel の対象で、1 CTA が 4 warp で
  12 query head を処理し、decode は split + reduce で実行する。
- MoE は router（global 512、top-10）、gate/up/down の 3 grouped linear、shared expert、
  combine で構成する。shared expert の gate は sigmoid。
- expert の slot map がある場合、router 後の plan scan は元の group を VRAM hit と
  host miss に分ける。gate/up/down はそれぞれ hit 用と miss 用の grouped GEMM を 2 回
  起動し、`group_row_base` で元の group 行へ書く。activation plane と combine の
  group row は 1 本のままである。miss は host plane の重みを PCIe 越しに読む。
- slot の入れ替えは step 間だけで行う。router は local expert ごとの route 数を
  `expert_route_freq` に積み、batch 完了後に累積頻度の順位で host expert と VRAM expert
  を再配置する。route 数が 層×local expert を超える batch（prefill 相当）は計画した交換を
  その step で全数消化し、decode step へ再配置を持ち越さない。それ以外の step は
  `--moe-cache-swap-budget`（既定 96）が 1 step の交換上限で、PCIe 予算
  （`--pcie-frac`）の範囲内で消化する。`--moe-cache-min-gain`（既定 150）が交換を採る
  最小の頻度差、`--moe-cache-rebalance-steps`（既定 4）が再評価の間隔、
  `--moe-cache-decay`（既定 4096）が frequency を 0.7 倍にする閾値である。
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

`PleTable::decode_row` は checkpoint の形式に応じて 3 経路を復号する。`Ngram4` / `Ngram5`
checkpoint は `ngram-q`、BF16 shard checkpoint は INT6 W32 FP16、FP8 shard checkpoint は
E4M3 code × per-tensor BF16 scale である。INT6 W32 FP16 は
BF16 table の不可逆変換であり、PLE 入力の数値 class は BF16 と一致しない。
`ngram_rows` は 16 head（bigram 8 + trigram 8）の行番号を返し、
`PleInputRuntime` が sequence ごとの token history から 16 行を gather する。
1 token あたり 16 行 = 2,560 要素である。

### QSA

`QsaKeyStatePool` は sequence slot ごとに pooled key / partial key / counts を持つ。
`QSA_KEY_CACHE` は 4 token が揃った時点で mean → RMSNorm → partial RoPE を行い、
pooled key を書く。`QSA_SELECT` は score と選択を分ける。score は
`kQsaScoreRowTile × kQsaScoreBlockTile`（2048 row × 8192 block）の workspace に
block tile 単位で書き、選択は tile ごとの radix select で上位 k を取り、tile を
またぐ場合は直前までの上位 k と merge する。したがって score buffer は
`rows × blocks` ではなく固定 workspace である。出力は行あたり
`ceil(blocks / 32) * 4` byte の block bitmap（1 block = 1 bit、下位 block が下位 bit）で、
選んだ block と、まだ揃っていない tail block の bit を立てる。
`PagedAttentionNode` はこの bitmap を 2 番目の入力として、token を
`position >> log2(compress_ratio)` の block で判定する。QSA 層は 4 層周期
（0 始まりで 3, 7, ..., 47）。

## 5. executor

`create_qwen4exp_model_executor` が QSA key state、PLE conv state、PLE input runtime を
作る。新しい sequence の最初の step（`position == 0`）で QSA / PLE state を zero に
する。prefix cache は無効である。
expert offload が有効な場合、`Executor::after_batch` が batch 完了後に
`MoeExpertCache::step` を呼び、routing の収集と slot 交換を行う。

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
./build/tests/test_qsa_select
./build/tests/test_qsa_key_path
./build/tests/test_qsa_state_pool
./build/tests/test_qsa_attention_mask
./build/tests/test_paged_attention
./build/tests/test_executor_state_reserve
./build/tests/test_program_activation_planes
./build/tests/test_fp8_ptpc
```

`test_qsa_select` は CPU 参照の top-k と block bitmap を、chunk 境界（visible block
64 / 128）と compress_ratio 1 / 2 / 4 で比較する。`test_paged_attention` の
`QSA MASKED` ケースが、bitmap を消費する bf16 tiled prefill と bf16 / fp8 decode を
CPU 参照と突き合わせる。`test_executor_state_reserve` は state pool の予約 byte 数が
arena の実消費と一致することを検証する。

expert offload の E2E（TP=2）:

```bash
--moe-host-experts 96 ./build/shelf-compute \
  --model-dir models/Qwen3.8-Flash-Next-Quark-MXFP4 \
  --input-ids-file <prompt.ids> --max-new-tokens 8 --max-seq-len 128 \
  --device 2,3 --tensor-parallel 2 --prefix-cache 0 --kv-cache-dtype bf16 \
  --temperature 0 --arena-gib 30
```

`--moe-cache-log-interval 1` で step ごとの hit rate と swap 数を stderr に出す。

実 checkpoint の E2E:

```bash
./build/shelf-compute --model-dir models/Qwen3.8-Flash-Next-Quark-MXFP4 \
  --input-ids-file <5 token> --max-new-tokens 8 --max-seq-len 128 \
  --device 0,1,2,3 --tensor-parallel 4 --prefix-cache 0 --kv-cache-dtype bf16 \
  --temperature 0
```

期待する生成は ` Paris. The capital of Germany is Berlin` で始まる。
同じ条件の 2 回実行で logits は bitwise 一致する。
