# Architecture

## Layer構成

apps から下方向への依存のみ。矢印は「参照する」向きで、上位ほど具体的なモデル／実行編成、下位ほど汎用の仕組みを置く。

```text
apps
  shelf-compute / shelf-server / shelf-bench
      ↓
shelf_models
  model_weights_factory / loaded_model / tp_rank_model / pipeline_stage_model
  (family 選択 + executor 生成; plamo3 sources を含む)
      ↓
family層
  qwen35 (Dense)   qwen35moe (Moe)   qwen35qsa (Qsa)   qwen4exp (Qwen4Exp, loadのみ)   plamo3
      ↓
shelf_qwen35_runtime
  Executor / ContinuousBatcher / DFlash2 / MTP / spec decode / TP / pipeline parallel
      ↓
inference共通層
  inference/state      (SequenceSlotPool / PagedKVPool / GdnStatePool / QsaStatePool /
                        PleConvStatePool / spec state history)
  inference/scheduling (ScheduledBatch / SchedulePlan / SamplingConfig / KV capacity / KV banker)
  inference/execution  (selectors / dispatch / physical_launch / program_executor)
  inference/prefix     (radix tree / refcount 付き page 共有 / eviction)
      ↓
kernel層
  kernels/optimized  (family per file; linear / attention / gdn / elementwise / ...)
  kernels/fused      (複数処理を 1 launch に統合; moe / exl3 / octave / ...)
  kernels/correctness (1 TU + detail/*.inc + standalone/)
  kernels/dflash2    (speculative decode の drafter; shelf_kernels_dflash2)
      ↓
shelf_runtime
  PrimitiveGraph / Program / ProgramSet / DeviceBatchContext / workspace layout / TP transport
      ↓
shelf_weights
  safetensors / quantized safetensors → GPU MatrixWeight / Tensor
      ↓
shelf_quantization
  manifest / bundle / codec / fp8 / mxfp4 / exl3
      ↓
shelf_gpu / shelf_core / shelf_io
```

3 つの entry point は `cmake/apps.cmake` が定義する。

- `shelf-compute`: token ID の入出力のみを行う推論 engine
- `shelf-server`: aiohttp による OpenAI 互換 serving product（後述）
- `shelf-bench`: E2E benchmark（pp / tg）と kernel micro-benchmark

## Namespace

root namespace は `shelf`。`include/shelf/**` と `src/shelf/**` のディレクトリは
次の対応で、public header の `shelf/` prefix include 規則と同じ階層になる。

| directory | namespace |
| --- | --- |
| `shelf/core/**`（Status / 数値型） | `shelf` |
| `shelf/core/memory/**` / `shelf/core/gpu/**` | `shelf::gpu` |
| `shelf/io/**` | `shelf` |
| `shelf/quantization/**` | `shelf::quantization`（下位: `manifest` / `bundle` / `codec` / `fp8` / `mxfp4` / `exl3`） |
| `shelf/weights/**` | `shelf::weights` |
| `shelf/runtime/**` | `shelf::runtime` |
| `shelf/kernels/**` | `shelf::kernel` |
| `shelf/inference/state/**` | `shelf::state` |
| `shelf/inference/scheduling/**` | `shelf::scheduling` |
| `shelf/inference/execution/**` | `shelf::execution` |
| `shelf/inference/prefix/**` | `shelf::inference::prefix` |
| `shelf/models/common/qwen_hybrid/**` | `shelf::qwen_hybrid` |
| `shelf/models/common/drafter/**` | `shelf::drafter` |
| `shelf/models/qwen35/**`（model / weights / state / dflash2 / kernels） | `shelf::qwen35`（下位: `runtime` / `dflash2`） |
| `shelf/models/qwen35moe/**` | `shelf::qwen35moe` |
| `shelf/models/qwen35qsa/**` | `shelf::qwen35qsa` |
| `shelf/models/qwen4exp/**` | `shelf::qwen4exp` |
| `shelf/models/plamo3/**` | `shelf::plamo3` |
| `shelf/models/model_weights_factory.h` / `loaded_model.h` / `tp_rank_model.h` / `pipeline_stage_model.h` | `shelf::models` |
| `src/apps/**` | `shelf::app` / `shelf::bench::quality` |

generic 層（`shelf/kernels` / `shelf/inference` / `models/common`）は
特定の family（`qwen35` / `qwen35moe` / `qwen35qsa` / `qwen4exp`）を参照できない。
これを `tests/unit/test_architecture_boundaries.cpp` が静的に検証する。

### ターゲットの対応

CMake target の詳細な一覧は `cmake/targets.cmake` を参照。

- `shelf_core` は public include root と C++ 20 を提供する base INTERFACE target
- `shelf_gpu` / `shelf_io` / `shelf_quant_format` / `shelf_quant_bundle` /
  `shelf_quant_reference` は下位の共有基盤
- `shelf_weights` は model 非依存の weight loader（safetensors → GPU MatrixWeight / Tensor）
- `shelf_runtime` は model 非依存の runtime core（PrimitiveGraph / Program /
  ProgramSet / DeviceBatchContext / workspace layout / TP transport）
  (`shelf_core` / `shelf_gpu` / `shelf_kernels_fused` のみに依存)
- `shelf_inference_state` / `shelf_inference_scheduling` /
  `shelf_inference_execution` / `shelf_inference_prefix` は model 非依存の
  state / scheduling / execution / prefix cache 層
- `shelf_kernels_optimized` / `shelf_kernels_fused` /
  `shelf_kernels_correctness` は model 非依存の kernel 層
- `shelf_qwen35_kernels_optimized` は DFlash2 の model 固有 kernel
- `shelf_qwen_hybrid` は Qwen hybrid backbone の共通部。FFN の形に依存しない config
  geometry（`QwenHybridConfig`）、layer / model-level weight 型
  （`HybridLayerWeights` / `HybridModelLevelWeights`）、FFN 非依存の checkpoint 読み取り
  （`qwen_hybrid::weight_loader`）と共通 invariant 検証（`qwen_hybrid::weight_validation`）
  を持つ。具体的な family を参照しない。
  lowering の共有部品もここが持つ: 層の block emit（`emit_hybrid_block` = input norm →
  GDN / full attention → attention residual → post norm。FFN は各 family が自分の形を
  emit する）、model head の emit（`emit_hybrid_head` = output gather → final norm →
  lm_head → sampling。外部出力 4 本の順もここで固定する）、family 中立な lowered 保持型
  （`HybridLoweredPrimitives`）、`MatrixWeight` → `runtime::WeightSlot` の写し
  （`make_weight_slot`）。
  層の位相（GDN か full attention か）を決める規則もここで持つ: `layer_is_gdn`
  （`layer_types` → `full_attention_interval` → `layer % 4 < 3`）と、同じ規則で層数を数える
  `count_layer_topology`。GDN state pool・KV pool・TP plan・apps は全部これを使う
- `shelf_qwen35` は dense FFN を持つ Qwen35（config・weight・lowering・TP・MTP・DFlash2）。
  `shelf_qwen_hybrid` に依存し、他の family には依存しない
- `shelf_qwen35moe` は routed expert を持つ Qwen35 MoE（config 契約・expert plan・
  rank-3 expert loader・MoE FFN の weight 保持）。`shelf_qwen_hybrid` に依存し、
  `shelf_qwen35` には依存しない
- `shelf_qwen35qsa` は dense backbone + block-sparse indexer。`shelf_qwen35` と
  `shelf_qwen_hybrid` に依存する
- `shelf_qwen4exp` は Qwen3.8-Flash-Next（EXL3 body・PLE n-gram table・
  expert-axis の TP load path）。load contract のみを持ち、forward / executor の
  経路は後続の family port に委ねる
- `shelf_models` は family 選択と executor 生成の入口
  （`shelf::models::load_model_weights` / `LoadedModel` / `TpRankModel` /
  `PipelineStageModel`）。plamo3 の source もここに置く。
  全 family の loader を link する唯一の target で、`shelf_qwen35_runtime` も link する。
  family 選択は config の構造だけが決め、identity の判定は `shelf_qwen_hybrid` の
  identity gate が担う（後述「Checkpoint識別」）。
  tensor parallel の rank も同じ形で、rank が読む checkpoint の family は
  `shelf::models::TpRankModel`（`models/tp_rank_model.h`）を介して注入する。
  rank runtime 自身は `shelf_qwen35_runtime` にあり dense family を名乗るが、
  family をまたぐ判断は持たない（契約は [tensor_parallel_execution.md](tensor_parallel_execution.md) §11-1）
- `shelf_qwen35_runtime` は family 固有の runtime 層（Executor / ContinuousBatcher /
  DFlash2 / MTP / spec decode / TP / pipeline parallel / prefix cache /
  decode backend / token budget scheduler）。`shelf_qwen35` と全 inference 層を link する

`shelf_kernels_correctness` は `shelf_runtime` に link する一方、
`shelf_runtime` は `shelf_core` / `shelf_gpu` / `shelf_kernels_fused` だけを使う。

### weight loader の依存

weights loader（`shelf_weights`）の依存は下方向のみ:

```text
shelf_weights        -> shelf_quant_format / shelf_quant_bundle / shelf_quant_reference / shelf_io / shelf_core / shelf_gpu
shelf_qwen_hybrid    -> shelf_weights + shelf_io + shelf_quant_bundle
shelf_qwen35         -> shelf_qwen_hybrid (+ shelf_weights / shelf_io / shelf_quant_* / shelf_inference_state / shelf_runtime)
shelf_qwen35moe      -> shelf_qwen_hybrid (+ shelf_weights / shelf_io / shelf_quant_bundle / shelf_gpu)
shelf_qwen35qsa      -> shelf_qwen35 + shelf_qwen_hybrid
shelf_qwen4exp       -> shelf_qwen_hybrid (+ shelf_weights / shelf_io / shelf_quant_bundle / shelf_gpu)
shelf_models         -> 全 family + shelf_qwen35_runtime
```

```text
    qwen35 (dense FFN)   qwen35moe (MoE FFN)   qwen35qsa (QSA)   qwen4exp (loadのみ)
              \                 |                    /                  /
               v                v                   v                  v
                    common/qwen_hybrid (backbone config / weights / loader / validation)
                                      |
                                      v
                        shelf_weights / shelf_quant_* / shelf_io

plamo3 は `shelf_models` の source として配置し、identity で選択する。
hybrid backbone は `shelf_models` が link する family target 経由で利用する。
```

各 family は互いに参照しない（qwen35qsa と qwen4exp は qwen35 / qwen_hybrid を
介して dense backbone を再利用するが、兄弟 family の loader を直接呼ばない）。
FFN に依存しない layer load 部品は
`shelf/models/common/qwen_hybrid/weight_loader.h` にあり、FFN の tensor 名は各 loader が
自分の module で読む。checkpoint の layer prefix（`qwen_hybrid::layer_tensor_prefix`）
を組み立てる箇所は 1 箇所だけで、TP partition plan もそれを使う。

weight の canonical → partition → preshuffle → upload の順序 contract は
[tensor_partition.md](tensor_partition.md) を参照。

### family をまたぐ境界

`LoadedModelWeights` は 4 つの family weight 型の `std::variant` で、
dense / MoE / QSA / Plamo3 のいずれか 1 つだけを保持する（`qwen4exp` は
load 契約のみで、variant の member を持たない）。
MoE checkpoint を dense として読むことはできず、アクセスは明示的である
（`std::get_if<...>` が他 family では null を返す）。

`ModelFamily` enum は `Dense` / `Moe` / `Qsa` / `Qwen4Exp` / `Plamo3` の 5 値。
`probe_model_dir` は config の構造から family を選び、`load_model_weights` は
config を読み、その family の loader で読み込み、その family の validation を
実行する。ここでは GPU 上の実行方法を決めない。

`LoadedModel` は `load` / `borrow`、`family()`、`runs_grouped_experts()`、
`hybrid_config()`、`embed_tokens()` / `lm_head()`、および
`dense_model()` / `moe_model()` / `qwen4exp_model()` / `qsa_model()` を提供する。
executor 生成の family 分岐はここに置き、apps は family を知らない。

`TpRankModel` と `PipelineStageModel` は仮想 interface で、family をまたぐ唯一の層
（`shelf::models`）が実装と選択を担う。dense の実装は `models/qwen35/runtime` に
置き、grouped-expert の実装と選択は `models/tp_rank_model.cpp` /
`models/pipeline_stage_model.cpp` に置く。rank runtime と stage runtime 自身は
特定 family を名乗らず、この interface 経由でモデルを扱う。
`select_pipeline_stage_model` / `select_tp_rank_model` が checkpoint から
builder を選ぶ。

## 依存ルール（enforced by `test_architecture_boundaries`）

```text
runtime core -> Qwen35                 : forbidden
generic (kernels / inference / common) -> 特定 family : forbidden
weights / quantization -> Qwen35       : forbidden（models/ の include 不可）
Qwen35 model -> Qwen35 runtime         : forbidden
Qwen35 kernels -> Qwen35 runtime       : forbidden
Qwen35 <-> Qwen35 MoE                  : forbidden（互いに名指ししない）
common/qwen_hybrid -> 具体 family       : forbidden
```

`cmake/targets.cmake` の `target_link_libraries` も同じ test が token 単位で読み、
dense / MoE の直接 link、`shelf_qwen_hybrid` の family link、`shelf_weights` /
`shelf_runtime` の family link を拒否する。両 family が `shelf_qwen_hybrid` を
link することも固定する。

- `include/shelf/runtime/**` と `src/shelf/runtime/**` は
  特定 family を一切参照しない（`qwen35` / `Qwen35` / `shelf/models/` が存在しない）
- `include/shelf/kernels/**` / `inference/**` / `models/common/**` も
  同様に `qwen35` / `Qwen35` / `shelf/models/qwen` を含まない
- `include/shelf/weights/**` と `src/shelf/weights/**` も
  特定 family を一切参照しない（model 非依存の weight loader）
- `quantization/manifest/**` / `bundle/**` / `codec/**` / `core/**` / `io/**` は
  `shelf/models/` を include しない
  （manifest schema 上の architecture 文字列は format 契約でありコード依存ではない）
- model layer は weights binding / config / primitive lowering のみ。
  runtime（Executor / ContinuousBatcher）を include しない
- kernels layer は device correctness 実装のみ。
  runtime（HostExecutionContext / Executor）を include しない
- `*_correctness.*` という名前のファイルは `models/qwen35` 配下では
  `kernels/correctness/` に置く。`kernels/correctness` 直下の `.hip` は
  `model_dispatch_correctness.hip` または `standalone/` 配下に限る
- `check_no_server_deps` は `src/shelf` / `include/shelf` に serving 固有技術
  （HTTP / OpenAI schema / protobuf / gRPC / xgrammar）が混ざらないことを検査する

## New model rule

新モデルが既存 Primitive で表現可能なら、
`include/shelf/runtime/**` と `src/shelf/runtime/**` を変更しない。

新 model 向けの汎用抽象（IModel / Backend / Registry / Factory / Plugin interface）は
作成しない。必要になった実装は model-local
（`include/shelf/models/<model>/**`）に置く。

## Tensor Parallel / Pipeline Parallel

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

Pipeline parallel は layer range を stage に分割し、hidden を pinned host 経由で
直列に hand-off する。stage の選択は `shelf::models` が担い、stage runtime は
family を知らない。current contract は [pipeline_parallel.md](pipeline_parallel.md)
を参照。

## Checkpoint識別（config identity gate）

safetensors checkpoint をどの契約で読むかは、宣言と構造の別々のキーで決める。

| 決めること | キー | 実装 |
| --- | --- | --- |
| この build が実装するアーキテクチャか | root の `model_type` | `shelf::qwen_hybrid::check_config_identity`。受理集合は `supported_model_identities()`（ビルド時固定の allowlist） |
| family（FFN の形 / attention の形 / identity） | `text_config.num_experts` と config 構造 | 各 family の config reader と `shelf::models::load_model_weights` / `probe_model_dir` |

- gate の入力は root の `model_type` だけである。`text_config.model_type` は
  multimodal wrapper が内包する backbone の名前なので、wrapper の identity が未実装でも
  内側が既知というだけで通してしまう。拒否メッセージには
  `text_config.model_type` と `architectures` を診断として添えるが、判定には使わない
- gate は拒否専用である。集合外・宣言無しは `Status::unsupported` を返すだけで、
  identity が code path を選択しない。ここから仮想ディスパッチや registry は生まれない
- config を読む入口（`read_qwen_hybrid_config` / `read_declared_num_experts` /
  `read_qwen35_rope_parameters`）はすべて gate を先に通るので、未対応アーキテクチャは
  tensor 名探索や rope 解析の失敗ではなく、名指しの拒否として現れる
- dense reader は routed-expert identity を拒否するが、これは `intermediate_size` と
  `num_experts` の検査の後の backstop である（宣言より構造が正本）
- 受理集合の拡張は、そのアーキテクチャの実装と同時だけに限る
- payload の dtype も load 段階で拒否する。BF16 path が想定外の dtype を読むと
  要素サイズ違いで arena へ書き、実行中に意味不明な失敗になるため、
  `shelf_weights` の loader が `Status::invalid_argument` を返す。
  2-D matrix は BF16 / F32（F32 は BF16 へ widen）、1-D vector は BF16 / F32
  （`load_bf16_or_f32_tensor` を使う norm は F32 のまま保持する）。
- speculative decode の drafter は別の識別を持つ
  （`architectures[0] == "DFlash2DraftModel"`、`qwen35/dflash2/config.cpp`）。
  target の identity gate とは独立である

受理・拒否の境界は `tests/unit/test_model_identity.cpp` が固定する
（対応 identity の受理、集合外と宣言無しの拒否、identity と構造の矛盾、
`read_declared_num_experts` と rope reader が同じ gate を通ること）。
BF16 payload の dtype 検査は `tests/unit/test_weight_load.hip` が固定する。

## Server layer

`shelf-server` は serving product であり、inference engine ではない。
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

- `shelf-compute` は token ID の in/out のみ。HTTP・OpenAI schema・messages・
  roles・tools・JSON Schema・tool parser・chat template・tokenizer を持たない。
- server 側（`src/apps/server/**`）だけが OpenAI schema・tokenizer・chat template・
  tool parser に依存する。
- `src/shelf/**` と `include/shelf/**` は serving 固有の技術
  （HTTP / OpenAI schema / protobuf / gRPC / xgrammar）を参照しない。
  `test_architecture_boundaries` の `check_no_server_deps` がこれを静的に検査する。
- tokenizer と Qwen chat template の唯一の source of truth は model directory の
  Hugging Face processor である。server codec は `src/apps/common/shelf_chat/`
  に置き、server が使う。

### compute control plane

`shelf-compute --serve-stdio` は GPU runtime owner である単一 thread 上で、
`poll(2)` により stdin を non-blocking に読む。generation 中でも `ping` / `cancel` /
`shutdown` を処理でき、1 step ごとに control command を確認する。
`{"op":"cancel","request_id":N}` は対象 request を `finish_reason="cancelled"` で終了させ、
terminal event は `done` の 1 個だけである。GPU runtime を複数 thread から触らない。

複数 request は同一の `ContinuousBatcher` へ投入され、1 回の `runtime.step()` で
`ScheduledBatch` として co-batch される。GPU runtime owner は 1 thread のままである。

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
