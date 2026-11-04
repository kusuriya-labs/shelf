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

- `Qwen35LoadOptions::stage` が layer range を持ち、loader はその範囲だけを upload する。
  - 先頭 stage: embedding を読む。
  - 最終 stage: final norm と lm_head を読む。tied embedding の場合は embedding も読む。
  - 中間 stage: model-level の重みを読まない。
- `Qwen35ModelWeights::layers` は全 layer 分の vector だが、担当範囲外は空である。
  `validate_qwen35_weights` は `PipelineStage` の範囲だけを検証する。
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
- grouped expert model（MoE）。`shelf::models::probe_model_dir` が `Moe` を返す checkpoint では
  `initialize_pipeline_parallel` が `unsupported` を返す。stage 分割は dense の weight load
  （`HybridLoadOptions` の stage 字段）と dense の executor footprint を前提にしており、group
  行数容量と plan scratch の stage 別配分を持たない
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
`tests/unit/test_qwen35_pipeline_lowering.cpp` が stage ごとの graph 構造
（embedding/head の有無、入出力の dtype、外部出力数）を固定する。

E2E の採否は PP=1 と PP=2 の生成 token 列の一致で判断する。実測条件と結果は
[../research/20261104/pipeline_parallel_port.md](../research/20261104/pipeline_parallel_port.md)
に記録する。
