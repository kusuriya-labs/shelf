# Architecture

## Layer構成

```text
apps
     ↓
Qwen35 model / runtime              Inference common
    config / weights                    inference/state         (SequenceSlotPool / PagedKVPool /
    model/lower_to_primitives           inference/scheduling      GdnStatePool / ScheduledBatch /
    runtime/Executor / prefix cache     inference/execution       SamplingConfig / program_executor /
    runtime/ContinuousBatcher             selectors / dispatch     dispatch / fusion / physical_launch)
    dflash2 / eagle4 (spec decode)
     ↓                    ↓                ↓
Qwen-family common     Generic kernels
    models/common/qwen_hybrid           kernels/optimized  (family per file)
      geometry_check / GdnGeometry /     kernels/correctness (1 TU + detail/*.inc)
      GatedAttentionGeometry /           kernels/kv_types, model_dispatch_state
      emit_gdn / emit_gated_attention
     ↓
Generic weights loader
    safetensors / quantized safetensors
    ↓
canonical validation
    ↓
optional logical partition
    ↓
native preshuffle
    ↓
GPU upload
     ↓                    ↓
Generic state (ps::state) ─────────────┤
     ↓                    ↓
Quantization / FPX format
     ↓
Runtime core
    PrimitiveGraph / Program / ProgramSet / DeviceBatchContext / workspace layouts
```

generic 層（`phaseshift/kernels` / `phaseshift/inference` /
`models/common`）は `models/qwen35` / `Qwen35` を参照できない。
これを `tests/unit/test_architecture_boundaries.cpp` が静的に検証する。

ターゲットの対応:

- `phaseshift_kernels_optimized` / `phaseshift_kernels_correctness` /
  `phaseshift_inference_state` / `phaseshift_inference_scheduling` /
  `phaseshift_inference_execution` は model 非依存層
- `phaseshift_qwen35` / `phaseshift_qwen35_runtime` /
  `phaseshift_qwen35_kernels_optimized`（dflash2 / eagle4）は Qwen35 固有層

`phaseshift_kernels_correctness` は `phaseshift_runtime` にlinkする一方、
`phaseshift_runtime` は `phaseshift_core` / `phaseshift_gpu` だけを使う。

Tensor Parallel を使う場合の scheduler / rank runtime の関係（TP-Exec-1）:

```text
ContinuousBatcher (global 1個)
      |
  TpCoordinator
   /      \
Rank0    Rank1          (1 rank = 1 device-local runtime: arena / stream /
  |        |              rank-local weight / KV pool / GDN state / Executor)
Executor Executor
```

詳細は [tensor_parallel_execution.md](tensor_parallel_execution.md) を参照。

weights loader（`phaseshift_weights`）の依存は下方向のみ:

```text
phaseshift_weights -> phaseshift_quant_format / phaseshift_quant_bundle / phaseshift_quant_reference / phaseshift_io / phaseshift_core / phaseshift_gpu
phaseshift_qwen35  -> phaseshift_weights
```

weight の canonical → partition → preshuffle → upload の順序 contract は
[tensor_partition.md](tensor_partition.md) を参照。

CMake targetの一覧は `cmake/targets.cmake` を参照。

## GPU-MCU

GPU-MCU backend はこの branch に含まれない。実装と contract は
`exp/gpu-mcu` ブランチへ退避している。


## 依存ルール（enforced by `test_architecture_boundaries`）

```text
runtime core -> Qwen35            : forbidden
weights / quantization -> Qwen35  : forbidden（models/ の include 不可）
Qwen35 model -> Qwen35 runtime    : forbidden
Qwen35 kernels -> Qwen35 runtime  : forbidden
```

- `include/phaseshift/runtime/**` と `src/phaseshift/runtime/**` は
  Qwen35を一切参照しない（`qwen35` / `Qwen35` / `phaseshift/models/` が存在しない）。
- `include/phaseshift/weights/**` と `src/phaseshift/weights/**` も同様に
  Qwen35を一切参照しない（model非依存のweight loader）。
- `quantization/manifest/**` / `quantization/bundle/**` / `quantization/codec/**` /
  `core/**` / `io/**` は
  `phaseshift/models/` をincludeしない。
  （manifest schema上の architecture 文字列は format契約でありコード依存ではない）
- model layerはweights binding / config / primitive loweringのみ。
  weights loaderのformat分岐（BF16 / FP8）はmodel側に持たない。
  runtime（Executor / scheduler / state ownership）をincludeしない。
- kernels layerはdevice correctness実装のみ。
  runtime（HostExecutionContext / Executor）をincludeしない。

## New model rule

新モデルが既存Primitiveで表現可能なら、
`include/phaseshift/runtime/**` と `src/phaseshift/runtime/**` を変更しない。

新model向けの汎用抽象（IModel / Backend / Registry / Factory / Plugin interface）は
作成しない。必要になった実装はmodel-local
（`include/phaseshift/models/<model>/**`）に置く。

## Server layer

`phaseshift-server` はserving productであり、inference engineではない。
aiohttp で OpenAI Chat Completions の subset を提供し、
`phaseshift-compute --serve-stdio` へ直接 JSONL stdio で接続する。

```text
OpenAI client
     │ HTTP / SSE
     ▼
phaseshift-server
     ├── aiohttp
     ├── openai_protocol.py   (request validation / wire encoding)
     ├── chat_service.py      (HF messages -> chat template -> compute -> response parser)
     └── compute_client.py    (asyncio native JSONL client)
     │ persistent JSONL stdio
     ▼
phaseshift-compute --serve-stdio   (token IDs in / token IDs out)
```

境界:

- `phaseshift-compute` はtoken IDのin/outのみ。HTTP・OpenAI schema・messages・
  roles・tools・JSON Schema・tool parser・chat template・tokenizerを持たない。
- server側（`src/apps/server/**`）だけがOpenAI schema・tokenizer・chat template・
  tool parserに依存する。
- `src/phaseshift/**` と `include/phaseshift/**` はserving固有の技術
  （HTTP / OpenAI schema / protobuf / gRPC / xgrammar）を参照しない。
  `test_architecture_boundaries` の `check_no_server_deps` がこれを静的に検査する。
- tokenizerとQwen chat templateの唯一のsource of truthはmodel directoryの
  Hugging Face processorである。server codecは `src/apps/common/phaseshift_chat/`
  に置き、serverが使う。

### compute control plane

`phaseshift-compute --serve-stdio` はGPU runtime ownerである単一thread上で、
`poll(2)` によりstdinを non-blockingに読む。generation中でも `ping` / `cancel` /
`shutdown` を処理でき、1 stepごとにcontrol commandを確認する。
`{"op":"cancel","request_id":N}` は対象requestを `finish_reason="cancelled"` で終了させ、
terminal eventは `done` の1個だけである。GPU runtimeを複数threadから触らない。

複数requestは同一の `ContinuousBatcher` へ投入され、1回の `runtime.step()` で
`ScheduledBatch` として co-batchされる。GPU runtime ownerは1 threadのままである。

```text
aiohttp handlers
    ↓
AsyncComputeClient per-request mailboxes
    ↓
JSONL multiplex (1 stdin writer / 1 stdout reader)
    ↓
single compute service loop (1 owner thread)
    ↓
ContinuousBatcher
    ↓
ScheduledBatch
    ↓
GPU
```

論理concurrency `N` (`--max-concurrent-requests`) とphysical KV budget
(`--kv-cache-capacity-tokens`) は別契約である。`SequenceSlotPool` /
`GdnStatePool` / `max_scheduled_requests` はNに一致させ、physical KV pagesは
budgetから決める。budget不足時はKV BankerがQueued requestをActive完了後に
admissionする。

serverの対応surfaceは [../user/server.md](../user/server.md) を参照。
