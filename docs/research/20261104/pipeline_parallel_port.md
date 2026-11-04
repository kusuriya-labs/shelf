# pipeline-parallel の移植と E2E 確認

- 実施日: 2026-11-04
- 移植元: `~/work/shelf_old` の `src/runtime/pipeline` と `apps/common/pipeline.{hpp,cpp}`
- 移植先: `shelf` の `qwen35` runtime（`PipelineParallelCoordinator`）
- 対象 model: `models/Qwen3.8-27B-FP8`（HF FP8、64 layer）
- GPU: Radeon AI PRO R9700（gfx1201）× 2（device 0,1）
- build: Release、`SHELF_HIP_GRAPH=OFF`、`CMAKE_HIP_ARCHITECTURES=gfx1201`

## 1. 移植元の契約

shelf_old の PP は layer split と host hand-off を持つ。

- `even_stage_ranges(num_layers, K)` で均等分割し、余りを最後の stage に寄せる。
- 先頭 stage が embedding、最終 stage が final norm と lm_head を持つ。
- stage 間は bf16 hidden を直列に受け渡す。
- 実行そのものは generic な `runtime/pipeline` が担い、stage の生成は apps が行う。

## 2. 現行アーキテクチャへの対応

現行 repo には旧 repo の `ir::Graph` / `StageExecutor` がないため、同じ契約を
既存の `PrimitiveGraph` / `Executor` に写した。

| 旧 | 現行 |
| --- | --- |
| `StageRange` / `even_stage_ranges` | `shelf::qwen35::StageRange` / `even_stage_ranges` |
| `build_step_graph(..., StageRange)` | `Qwen35LowerOptions::stage` |
| layer 範囲だけの重み load | `Qwen35LoadOptions::stage` |
| `StageExecutor::run_stage` | stage ごとの `Executor` と `execute_batch` |
| `fetch_value_bf16` | `BatchExecutionOutput::final_hidden` |
| `runtime::pipeline::step` | `PipelineParallelCoordinator::on_execute` |

stage ごとに model を部分 load し、`SequenceSlotPool` / `GdnStatePool` / `PagedKVPool` /
arena / `Executor` を所有させる。KV page 数は stage 0 の auto-fit を全 stage で共有し、
block table の page ID を一致させる。sampled token だけを stage 0 の device に戻す。

## 3. 追加した正しさの門

- `tests/unit/test_qwen35_pipeline_stage.cpp`: 分割規則（偶数・余り・境界）。
- `tests/unit/test_qwen35_pipeline_lowering.cpp`: stage 0 は token 入力・embedding・
  hidden 1 出力、stage 1 は bf16 hidden 入力・final norm・logits 4 出力。
- `tests/unit/test_qwen35_lowering_contract.hip`: full graph の既存契約が不変。

## 4. E2E

### 条件

- prompt: `/tmp/opencode/ids_1024.txt`（1024 token）
- greedy、`--max-new-tokens 4`、`--max-seq-len 2048`、
  `--max-num-batched-tokens 2048`、`--max-concurrent-requests 1`
- `--prefix-cache 0`、`--arena-gib auto`

### E1: PP=1

```bash
./build/shelf-compute --model-dir /opt/zen/work/models/Qwen3.8-27B-FP8 \
  --input-ids-file /tmp/opencode/ids_1024.txt \
  --device 0 --tp 1 --max-new-tokens 4 --max-seq-len 2048 \
  --max-num-batched-tokens 2048 --max-concurrent-requests 1 \
  --prefix-cache 0 --arena-gib auto
```

`GENERATED_IDS=41564,13,814,20139`

### E2: PP=2

```bash
./build/shelf-compute --model-dir /opt/zen/work/models/Qwen3.8-27B-FP8 \
  --input-ids-file /tmp/opencode/ids_1024.txt \
  --device 0,1 --tp 1 --pipeline-parallel 2 \
  --max-new-tokens 4 --max-seq-len 2048 \
  --max-num-batched-tokens 2048 --max-concurrent-requests 1 \
  --prefix-cache 0 --arena-gib auto
```

`GENERATED_IDS=41564,13,814,20139`

PP=1 と PP=2 の生成 token は完全一致した。stage ごとの weight は
14,732,840,960 bytes、KV page は 32,181 で、`ARENA_USED_GB=30.398` だった。

## 5. 採用判断

- **E1/E2: ACCEPT**。layer split + host hand-off の PP は現行 runtime で成立する。
- 現時点では直列 hand-off であり、stage 間の overlap はない。性能最適化は別 Gate とする。
- PP=2 の速度は 1 run の参考値しか取っていないため、性能値として採用しない。
  `docs/perf/current.md` へは載せない。

## 6. 未対応

- TP と PP の同時利用
- DFlash2 speculative decode
- prefix cache
- MTP block
- hidden taps / `--dump-logits`
- speculative verify batch
- decode と prefill が同じ batch に混在する実行（`num_decode_requests > 0 &&
  num_prefill_requests > 0`）。`--max-concurrent-requests 2` で 2 request を
  同時に投入すると、片方の decode ともう片方の prefill が同じ batch に入り、
  現行 executor の mixed batch 経路では正しい token を返さないことを実測した。
  そのため PP coordinator は mixed batch を `unsupported` で拒否する。

## 7. 計測上の注意

PP の stage 0 は embedding と前半 layer、最終 stage は後半 layer と lm_head を持つため、
stage ごとの重み量は model の layer 構成に依存する。KV page 数を stage 0 に合わせるため、
stage 0 の空き VRAM が最小の構成では起動できない。この場合は `--arena-gib` を明示するか、
layer 分割を非対称にする必要がある（未実装）。
