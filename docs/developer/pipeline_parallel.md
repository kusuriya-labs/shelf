# Pipeline Parallel Execution

パイプライン並列（PP）の current contract である。TP の contract は
[tensor_parallel_execution.md](tensor_parallel_execution.md) が正本で、本ドキュメントは
PP の layer 分割・stage runtime・hand-off だけを扱う。

## 1. 実行構造

```text
ContinuousBatcher (stage 0 の pools を所有)
      |
  ScheduledBatch
      |
  PipelineParallelCoordinator
   ├── stage 0: device d0 / layers [0, l1) / embedding
   ├── stage 1: device d1 / layers [l1, l2)
   └── stage K-1: device dK-1 / layers [lK-1, L) / final norm + lm_head
```

- `--pipeline-parallel K` で K stage に分割する。分割は `even_stage_ranges(L, K)` で
  均等に行い、余りは最後の stage に寄せて全 layer を覆う。
- 先頭 stage だけが embedding を持ち、token ID を入力にする。
- 最終 stage だけが final norm、lm_head、sampling を持つ。
- 中間 stage は前 stage の hidden を入力にし、hidden だけを返す。
- 各 stage は自分が担当する layer の重み、KV pool、GDN state pool、
  `SequenceSlotPool`、arena、`Executor` を所有する。
- stage 間は bf16 の hidden（`rows × hidden_size`）を pinned host 経由で
  直列に hand-off する。数値の丸めは入れない。
- 最終 stage の sampled token だけを stage 0 の device へ戻し、
  `ContinuousBatcher` は従来どおり stage 0 の出力として扱う。

## 2. 重みと state

- stage の model は `shelf::models::PipelineStageModel` の interface 経由で入れる。stage
  runtime は checkpoint の family を名乗らず、builder 関数ポインタ
  （`PipelineStageModelBuilder`）を呼び出し側から受け取る。dense 版は
  `shelf::qwen35::runtime::build_dense_pipeline_stage_model`、grouped expert 版は
  `shelf::models::build_moe_pipeline_stage_model` で、選択は
  `shelf::models::select_pipeline_stage_model`（`probe_model_dir` が正本）が行う。
  interface が持つのは `hybrid_config` / `weight_bytes` / `token_row_ffn_size` /
  `ffn_workspace_reserve_bytes` / `uses_exl3_linear` / `create_executor` と、
  必要側の model への accessors（`dense_model` / `grouped_model`）である。
- executor footprint の見積もりは interface の 2 項で組む。token row 側に
  `qwen35_executor_reserve_bytes(tc, token_row_ffn_size, ...)`、FFN の group 行側に
  `ffn_workspace_reserve_bytes(max_batch_rows)`（dense は 0）。auto 経路には
  `max(64 MiB, free / 100)` の slack を加える。KV pool はこの reserve を引いた残りから
  切るため、見積もりが実 footprint より小さいと確保失敗になる
- `Qwen35LoadOptions::stage` / `Qwen35MoeLoadOptions::stage` が layer range を持ち、loader は
  その範囲だけを upload する。
  - 先頭 stage: embedding を読む。
  - 最終 stage: final norm と lm_head を読む。tied embedding の場合は embedding も読む。
  - 中間 stage: model-level の重みを読まない。
- `Qwen35ModelWeights::layers` / `Qwen35MoeModelWeights::layers` は全 layer 分の vector だが、
  担当範囲外は空である。`validate_qwen35_weights` / `validate_qwen35moe_weights` は
  `PipelineStage` の範囲だけを検証する。grouped expert では担当 layer の
  expert 平面・router・shared expert だけを load し、担当外の layer の平面は载らない
  （`moe_ffn_weights_match` を張るのは担当範囲だけ）。
- model level の検査は `qhy::validate_hybrid_model_level(model, config, loaded_layer_lo,
  loaded_layer_hi)` が単一源で、上記の読み分けを同じ規則で検証する。行数は
  `TensorPartitionDesc` の global 値へ張り戻して照合するので、vocab を shard した TP rank の
  local 行数を拒否しない。
- KV page 数は stage 0 の auto-fit 結果を全 stage で共有する。これは block table の
  page ID を stage 間で一致させるためである。stage 0 の容量が最大でない構成では、
  stage 0 が使う page 数に他 stage が収まらない場合に起動が失敗する。
- GDN state は stage ごとの layer 数だけ確保する（state index は stage 内で 0 から）。

## 3. CLI

```bash
shelf-compute --model-dir models/Qwen3.8-27B-FP8 \
  --device 0,1 --tensor-parallel 1 --pipeline-parallel 2 \
  --max-seq-len 2048 --max-num-batched-tokens 2048 \
  --prefix-cache 0
```

- `--pipeline-parallel > 1` は `--tensor-parallel 1` を要求する。
- `--device` の個数は `--pipeline-parallel` と一致させる。
- `--prefix-cache` は PP では無効化する（指定時は警告）。

## 4. 未対応

以下は PP の current contract に含めない。要求された場合は起動時または実行前に拒否する。

- TP と PP の同時利用
- DFlash2 speculative decode
- prefix cache
- MTP block
- hidden taps
- `--dump-logits`
- speculative verify を含む batch
- decode と prefill が同じ batch に混在する実行（`--max-concurrent-requests` を 2 以上に
  したときに発生し得る。現在は明示的に `unsupported` を返す）

## 5. 正しさ

`tests/unit/test_qwen35_pipeline_stage.cpp` が分割規則を、
`tests/unit/test_qwen35_pipeline_lowering.hip` が dense の stage ごとの graph 構造
（embedding/head の有無、入出力の dtype、外部出力数）を固定する。grouped expert 側は
`tests/unit/test_qwen35moe_model_load.hip` が stage 範囲の weight load（担当外 layer の平面が
载らないこと、embed / lm_head / final norm の所在、拒否条件）を、
`tests/unit/test_qwen35moe_lowering.hip` が stage の graph と program（external input の
dtype、hand-off の external output、`MOE_ROUTER` 本数、`SAMPLING` の所在）を固定する。

E2E の採否は PP=1 と PP=2 の生成 token 列の一致で判断する。実測条件と結果は
[../research/20261104/pipeline_parallel_port.md](../research/20261104/pipeline_parallel_port.md)
に記録する。
