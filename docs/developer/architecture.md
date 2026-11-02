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
    dflash2 (spec decode)
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
Generic state (shelf::state) ────────────┤
     ↓                    ↓
Quantization / FPX format
     ↓
Runtime core
    PrimitiveGraph / Program / ProgramSet / DeviceBatchContext / workspace layouts
```

## Namespace

root namespace は `shelf`。`include/shelf/**` と `src/shelf/**` のディレクトリは
次の対応で、public header の `shelf/` prefix include 規則と同じ階層になる。

| directory | namespace |
| --- | --- |
| `shelf/core/**`（Status / 数値型） | `shelf` |
| `shelf/core/memory/**` / `shelf/core/gpu/**` | `shelf::gpu` |
| `shelf/io/**` | `shelf` |
| `shelf/quantization/**` | `shelf::quantization`（下位: `manifest` / `bundle` / `codec` / `fp8`） |
| `shelf/weights/**` | `shelf::weights` |
| `shelf/runtime/**` | `shelf::runtime` |
| `shelf/kernels/**` | `shelf::kernel` |
| `shelf/inference/state/**` | `shelf::state` |
| `shelf/inference/scheduling/**` | `shelf::scheduling` |
| `shelf/inference/execution/**` | `shelf::execution` |
| `shelf/models/common/**` | `shelf::qwen_hybrid` |
| `shelf/models/qwen35/**` | `shelf::qwen35`（下位: `runtime` / `dflash2`） |
| `src/apps/**` | `shelf::app` / `shelf::bench::quality` |

generic 層（`shelf/kernels` / `shelf/inference` /
`models/common`）は `models/qwen35` / `Qwen35` を参照できない。
これを `tests/unit/test_architecture_boundaries.cpp` が静的に検証する。

ターゲットの対応:

- `shelf_kernels_optimized` / `shelf_kernels_correctness` /
  `shelf_inference_state` / `shelf_inference_scheduling` /
  `shelf_inference_execution` は model 非依存層
- `shelf_qwen35` / `shelf_qwen35_runtime` /
  `shelf_qwen35_kernels_optimized`（dflash2）は Qwen35 固有層

`shelf_kernels_correctness` は `shelf_runtime` にlinkする一方、
`shelf_runtime` は `shelf_core` / `shelf_gpu` だけを使う。

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

weights loader（`shelf_weights`）の依存は下方向のみ:

```text
shelf_weights -> shelf_quant_format / shelf_quant_bundle / shelf_quant_reference / shelf_io / shelf_core / shelf_gpu
shelf_qwen35  -> shelf_weights
```

weight の canonical → partition → preshuffle → upload の順序 contract は
[tensor_partition.md](tensor_partition.md) を参照。

CMake targetの一覧は `cmake/targets.cmake` を参照。

## 依存ルール（enforced by `test_architecture_boundaries`）

```text
runtime core -> Qwen35            : forbidden
weights / quantization -> Qwen35  : forbidden（models/ の include 不可）
Qwen35 model -> Qwen35 runtime    : forbidden
Qwen35 kernels -> Qwen35 runtime  : forbidden
```

- `include/shelf/runtime/**` と `src/shelf/runtime/**` は
  Qwen35を一切参照しない（`qwen35` / `Qwen35` / `shelf/models/` が存在しない）。
- `include/shelf/weights/**` と `src/shelf/weights/**` も同様に
  Qwen35を一切参照しない（model非依存のweight loader）。
- `quantization/manifest/**` / `quantization/bundle/**` / `quantization/codec/**` /
  `core/**` / `io/**` は
  `shelf/models/` をincludeしない。
  （manifest schema上の architecture 文字列は format契約でありコード依存ではない）
- model layerはweights binding / config / primitive loweringのみ。
  weights loaderのformat分岐（BF16 / FP8）はmodel側に持たない。
  runtime（Executor / scheduler / state ownership）をincludeしない。
- kernels layerはdevice correctness実装のみ。
  runtime（HostExecutionContext / Executor）をincludeしない。

## New model rule

新モデルが既存Primitiveで表現可能なら、
`include/shelf/runtime/**` と `src/shelf/runtime/**` を変更しない。

新model向けの汎用抽象（IModel / Backend / Registry / Factory / Plugin interface）は
作成しない。必要になった実装はmodel-local
（`include/shelf/models/<model>/**`）に置く。

## Server layer

`shelf-server` はserving productであり、inference engineではない。
aiohttp で OpenAI Chat Completions の subset を提供し、
`shelf-compute --serve-stdio` へ直接 JSONL stdio で接続する。

```text
OpenAI client
     │ HTTP / SSE
     ▼
shelf-server
     ├── aiohttp
     ├── openai_protocol.py   (request validation / wire encoding)
     ├── chat_service.py      (HF messages -> chat template -> compute -> response parser)
     └── compute_client.py    (asyncio native JSONL client)
     │ persistent JSONL stdio
     ▼
shelf-compute --serve-stdio   (token IDs in / token IDs out)
```

境界:

- `shelf-compute` はtoken IDのin/outのみ。HTTP・OpenAI schema・messages・
  roles・tools・JSON Schema・tool parser・chat template・tokenizerを持たない。
- server側（`src/apps/server/**`）だけがOpenAI schema・tokenizer・chat template・
  tool parserに依存する。
- `src/shelf/**` と `include/shelf/**` はserving固有の技術
  （HTTP / OpenAI schema / protobuf / gRPC / xgrammar）を参照しない。
  `test_architecture_boundaries` の `check_no_server_deps` がこれを静的に検査する。
- tokenizerとQwen chat templateの唯一のsource of truthはmodel directoryの
  Hugging Face processorである。server codecは `src/apps/common/shelf_chat/`
  に置き、serverが使う。

### compute control plane

`shelf-compute --serve-stdio` はGPU runtime ownerである単一thread上で、
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

論理concurrency `N` (`--max-concurrent-requests`) とphysical KV budgetは別契約である。
`SequenceSlotPool` / `GdnStatePool` / `max_scheduled_requests` はNに一致させ、
physical KV pagesは起動時の arena の余りから決める。budget不足時はKV Bankerが
Queued requestをActive完了後にadmissionする。

serverの対応surfaceは [../user/server.md](../user/server.md) を参照。
