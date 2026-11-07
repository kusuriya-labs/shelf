# model load の並列化（Qwen3.8-Flash-Next MXFP4）

## 1. 目的と結論

`models/Qwen3.8-Flash-Next-Quark-MXFP4` の TP=4 起動時ロードを短縮した。
測定 revision は `4efa8de` の working tree、GPU は Radeon AI PRO R9700 4 枚である。

結論は **ACCEPT**。`MODEL_LOAD_MS` はページキャッシュが温まっている条件で
60.4〜69.2 s から **15.5〜18.2 s** へ下がった。ページキャッシュが追い出された
条件でも 30.4 s で、変更前の 69.2 s より短い。生成結果は変更前と一致する。

## 2. 変更前の内訳

`MODEL_LOAD_MS` は `TpGroup::create` の区間で、4 rank の weight load、KV pool、
executor の初期化を含む。変更前の 60 s 台の内訳を切り分けた。

| 項目 | 実測 |
| --- | ---: |
| `shelf-compute` 全体（wall） | 64.98 s |
| 同 process の CPU 時間（user + sys） | 64.57 s |
| `MODEL_LOAD_MS` | 60.38 s |
| PLE table 単体（`test_qwen4exp_load --skip-gpu`） | 24.57 s |
| PLE resident | 47.684 GiB |

CPU 時間が wall とほぼ同じで、rank load が 1 本の thread で直列に走っていた。
`TpCoordinator::initialize` は `TpRankRuntime::create` を rank ごとに順番に呼び、
各 rank の `load_qwen4exp_rank` は 48 層を順に処理する。PLE table は rank 0 の
weight load が終わった後に `shared_ple_table` で 1 回だけ読まれ、これが直列の
末尾に残っていた。

`PleTable::load_fp8` は 128 shard（合計 47.684 GiB）を 1 thread で
`memcpy` していた。PLE のファイルは `model.safetensors.index.json` の
`ngram_embedding.shard_*` にあり、`SafetensorsReader` の mmap から
anonymous RAM へコピーする。これは MXFP4 checkpoint の FP8 PLE 経路である。
EXL3 checkpoint の `ngram_embedding.safetensors` 経路（`mul1_codebook`、K=8）は
今回の対象外で、従来の `pread` のままである。

## 3. 実装した変更

### 3-1. rank runtime の並列生成

`TpCoordinator::initialize` で rank ごとの `TpRankRuntime::create` を
`devices_.size()` 本の thread で同時に実行する。各 thread は
`ScopedDevice` で自分の device を選び、arena / stream / model を独立に作る。
全 thread の join 後に、成功した runtime を rank 順で `ranks_` へ移す。
エラーは thread ごとに保持し、join 後に最初のエラーを返す。

### 3-2. PLE shard copy の並列化（MXFP4 checkpoint の FP8 PLE）

`PleTable::load_fp8` の 128 shard copy を 16 worker で分割した。
`SafetensorsReader` の map は mutex で保護し、`tensor_spec` / `tensor_data` は
mmap への読み取りだけなので worker から同時に呼べる。shard ごとの出力先は
分かれているため、`memcpy` は競合しない。

PLE 単体の実測:

| 条件 | 実測 |
| --- | ---: |
| 1 thread（変更前） | 24.57 s |
| 16 worker | 7.23 s |

### 3-3. PLE table の prefetch

rank 0 の weight load 後に PLE を読む直列をなくすため、
`TpCoordinator::initialize` の先頭で PLE table の prefetch thread を開始する。
prefetch は rank の weight load と同時に走る。

`shared_ple_table` の cache は `weak_ptr` なので、prefetch の戻り値を捨てると
table が解放されてしまう。prefetch 中だけ strong 参照を保持し、全 rank の
model が table を取得した後に `release_prefetched_qwen4exp_ple_table` で解放する。
family 固有の処理は `TpCoordinatorConfig` の `prefetch_rank_model` /
`release_rank_model` から呼び、dense runtime が Qwen4Exp を直接名指ししない形にした。

## 4. 実測

### 4-1. Qwen3.8-Flash-Next-Quark-MXFP4 / TP=4

コマンド:

```sh
./build/shelf-compute --model-dir models/Qwen3.8-Flash-Next-Quark-MXFP4 \
  --input-ids-file <3 token> --max-new-tokens 1 --max-seq-len 256 \
  --device 0,1,2,3 --tensor-parallel 4 --prefix-cache 0 \
  --kv-cache-dtype bf16 --temperature 0 --arena-gib 31
```

| 状態 | `MODEL_LOAD_MS` | wall |
| --- | ---: | ---: |
| 変更前 1 回目 | 69.18 s | 73.77 s |
| 変更前 2 回目 | 60.36 s | 63.35 s |
| 変更前 3 回目 | 60.38 s | 64.98 s |
| 変更後（page cache 追い出し後） | 30.42 s | 33.49 s |
| 変更後（warm） | 15.51 / 17.51 / 18.23 s | 20.14 / 22.10 / 21.28 s |

- 変更後の中間段階: rank 並列化のみ 33.5 s、PLE 並列化まで 18.0 s、
  prefetch 込み 15.5〜17.5 s。
- 変更前後とも `GENERATED_IDS=1118`（3 token prompt、1 token 生成）で一致。
- 5 token prompt / 8 token 生成を 2 回実行し、`GENERATED_IDS` は
  `9338,11,321,279,7526,3177,303,4357` で一致した。`--dump-logits` の 8 行も
  bitwise 一致した（sha256 `330d9f78eb2488c522516d1ce2e91bb9e155e87c534f33605664f1c16ad590ed`）。

### 4-2. 他の family

rank 並列化は Qwen4Exp 以外の TP 経路にも効く。

| model | 変更前 | 変更後 |
| --- | ---: | ---: |
| Qwen3.8-27B-FP8 / TP=2、1 回目 | 30.08 s | 19.15 s |
| Qwen3.8-27B-FP8 / TP=2、2 回目 | 20.91 s | 10.47 s |
| Qwen3.6-35B-A3B MXFP4 / TP=2 | 未測 | 18.96 s |

27B は変更前と同じ `GENERATED_IDS=1118` である。MoE も `GENERATED_IDS=1118` を
確認したが、変更前の同一条件は未計測である。

## 5. テスト

- `test_qwen35_tensor_parallel_plan`
- `test_qwen35_tp_context`
- `test_tp_reduction`
- `test_tp_barrier_group`
- `test_tp_transport`
- `test_qwen35_tp_execution`
- `test_qwen35moe_tp_execution`
- `test_qwen4exp_hyper_site` / `gdn_gate` / `layer` / `ple_site` / `qsa_site` /
  `qsa_layer` / `lower`
- `test_ple_gate` / `test_ple_conv`

すべて PASS。PLE 単体の `decode_row(0)` の rms は変更前後とも 0.0158707 で、
n-gram hash と 16-row gather の検証結果も同じである。

## 6. 残っている制約

- ページキャッシュが無い条件は 117 GB の read が支配的で、30 s 前後が下限になる。
  変更後は rank と PLE の並列 read で約 3.8 GB/s まで上がった。
- PLE は 47.684 GiB を anonymous RAM へコピーする。mmap のまま参照する方式にすると
  load は短くなるが、decode 時の row gather が page fault に依存するため、
  採用していない。
- weight の MXFP4 / FP8 preshuffle は CPU の scalar 変換である。GPU 側 preshuffle に
  置き換えればさらに短縮できる可能性があるが、未実装。
