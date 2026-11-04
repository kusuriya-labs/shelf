# Runtime

## Execution path

```text
Qwen35TextConfig + weights
        ↓
lower_qwen35_to_primitives
        ↓
PrimitiveGraph
        ↓
build_program_set
        ↓
ProgramSet
        ↓
Executor (create_model_executor)
        ↓
program_executor (private: src/shelf/models/qwen35/runtime/)
        ↓
model_dispatch_correctness (shelf::kernel, single launcher)
```

- `lower_qwen35_to_primitives(config, weights)` は
  `Qwen35LoweredPrimitives{graph, weights[], parameters[]}` を返す。
- `build_program_set` はfree function。bucketごとに `Program` を作り
  `ProgramSet` に格納する。
- `Program` は dispatch-only（`push()`）。host resolutionとGPU launchの
  分離は `program_executor.h`（private header）の
  `HostExecutionContext` / `HostResolvedValue` で行う。
- `HostExecutionContext::external_inputs` / `external_outputs` が指す配列は
  **ctx 自身の** `external_input_storage` / `external_output_storage` が所有する。
  構築関数のローカル配列へ pointer を持たせない。
  ctx は構築関数の返却後も呼び出し側で生き続けるため、ローカル配列だと
  stack が書き換わって不正な pointer が kernel に渡る。
- device側は `DeviceProgramView`（`runtime/program/device_program.h`）を
  kernel argumentとして渡す。
- sync semantics: opごとにstaging H2D（async）→ launch →
  `hipStreamSynchronize`（pool経由でnon-synchronous opはskip）。

## Decode backend

1つの `Executor` が持つ実行経路は Host のみである。ownership は phase や
backend で分けない。

```cpp
enum class DecodeBackend { Hip = 0 };
```

- `Executor::config.backend` の既定は `DecodeBackend::Hip`。
- `--decode-backend` が受理する値は `hip` のみで、それ以外は model load 前に
  exit 2 で拒否される。

## Public API

```cpp
shelf::qwen35::create_model_executor(model, slot_pool, gdn_pool, kv_pool, arena, config)
shelf::qwen35::executor_shutdown(executor)
shelf::qwen35::submit_batch(executor, batch, stream)
shelf::qwen35::complete_batch(executor, submission, stream)
shelf::qwen35::execute_batch(executor, batch, stream)
```

## State ownership

| 要素 | 所有 | 説明 |
|---|---|---|
| `ScheduledBatch` | ContinuousBatcher / caller | 1 step分のdecode/prefill request集合（POD） |
| `ContinuousBatcher` | app | request queue / KV admission / Executor呼出。`submit` / `step` / `cancel_all` |
| `KVCapacityManager` | ContinuousBatcher | requestごとのpage reservation state |
| `KVBanker` | ContinuousBatcher | paged KVのadmission / release（pure algorithm、stateless） |
| `PagedKVPool` | app | paged KV GPU buffer（attention層 × head × page） |
| `GdnStatePool` | app | GDN linear attention state GPU buffer |
| `SequenceSlotPool` | app | sequence slot GPU buffer |
| `Executor` | app | ProgramSet / GPU scratch / submission state。RAII + 明示shutdown |
| `PrefixCache` | app | compute アプリが所有する radix prefix KV / GDN state の shadow pool と cold pinned buffer。`lookup` / `save_*` / `restore` / 明示 `shutdown` |
| `DFlash2Executor` / `DFlash2SpecBatchContext` | app | DFlash2 drafter と共有 batch scratch。serve 終了時に明示 shutdown。`ContinuousBatcher` へは非所有ポインタで bind する |
| `DFlash2ContextState` / `DFlash2SpecDecoder` | `RuntimeRequest`（`ContinuousBatcher`） | DFlash2 request の sequence state と decoder。admission で作成し `finish_request` で解放する |

`ContinuousBatcher` は `Executor` / 各poolをreference保持し、
`PrefixCache` は非所有ポインタ参照である（`set_prefix_cache` で渡し、
所有権は compute アプリに残す）。
destroy順序は app が決定的に管理（executor shutdown → `PrefixCache::shutdown` →
arena shutdown）。`PrefixCache::shutdown` は必ず arena より前に実行する。

## GpuArena

単一の連続virtual address range + bump allocator。物理device memoryは
VMM（virtual address reservation + 増分physical commit）でbackingする。

- `capacity()`：reserved virtual addressの上限（callerが要求した値）。
- `used()`：bump allocatorが払い出したlogical bytesのhigh-water mark。
- `committed()`：現在physical device memoryへmapされているbytes。

commitはchunk単位（commit quantum）で、`allocate_aligned()`の進みに応じて
前方へだけ進む。個別free / decommitは無い（bump allocatorはindividual
allocationのlifetimeを追跡しないため）。

### arena budget

`Qwen35RuntimeConfig::arena_bytes == 0` は auto である。compute は `hipMemGetInfo`
を model load 前に 1 回呼び、空き VRAM から 1 GiB を引いた値を arena の容量に
する。TP=2 は rank ごとに自分の device の空き VRAM から同じ式で capacity を決める。

起動時 plan は stderr にだけ出す（`--serve-stdio` の stdout を汚さない）。

```text
MEMORY_PLAN mode=auto
  target weights    model load 後の arena used
  draft weights     --dflash2-model-dir の safetensors 合計
  drafter runtime   drafter の scratch + session pool slot × request 数
  runtime workspace --working-set-reserve-gib
  KV budget         arena 残量から導出した physical KV pool
  prefix cache      hot tier の実サイズ
  required          target + draft + drafter runtime + workspace + KV + prefix
  available         起動時の空き VRAM（fixed では arena capacity）
  arena capacity    reserved virtual address
  left              arena capacity - required
```

`draft weights` は arena からの予約であり、serve 時に drafter を load する
（single request 経路でも serve 時に load する）。
`drafter runtime` は DFlash2 の draft executor の scratch と、session pool の
slot 1 個分（context ring と spec workspace）を `--max-concurrent-requests` 倍した
値で、`dflash2_footprint.h` が単一の計算式を持つ。KV pool は arena の残りを
自動で使うため、drafter 分を先に引かないと serve 時の drafter load が
arena 超過になる。TP=2 は rank ごとに同じ計算を行い、内訳を stderr に出す。

```text
MEMORY_PLAN tp drafter: weights xx.xx GiB scratch xx.xx GiB session slot xx.xx GiB x N
DFLASH2_WEIGHT_ARENA_BYTES=<bytes>
```

`MEMORY_PLAN` の `used` は executor が workspace を確保する前の値である。
plan 後に必要になる作業領域は `--working-set-reserve-gib` が受け持つため、
drafter 分を織り込んでも reserve を 1 GiB まで下げると executor 確保で不足する。
reserve は plan 時点の約束であり、実際の作業領域を超えた場合は plan ではなく
後続の arena 確保で `insufficient memory` として現れる。

executor が arena から確保する量は起動時に 1 行で出す。

```text
MEMORY_PLAN executor footprint: xx.xx GiB (reserve xx.xx GiB)
MEMORY_PLAN tp rank N executor footprint: xx.xx GiB (reserve xx.xx GiB)
```

内訳の支配項は program workspace（選択した row bucket の `program.workspace.total_bytes`
+ scratch + attention partials）である。program workspace は `build_program_set` の
結果から決まり、それは `create_model_executor` の内部で構築されるため plan 時点では
正確な値が分からない。この不確実分を埋めるのが `--working-set-reserve-gib` であり、
KV auto-fit の計算式には織り込めない。

`--working-set-reserve-gib 0`（既定）は auto であり、`executor_footprint.h` の式で決める。

```text
reserve = align(values_per_row × Npad × hidden × 2)
          + align(kScratchFloats × 4) + align(32 MiB)
          + 静的項 (logits {MO,V}, token_hidden ×(1+taps), sampling, batch context)

values_per_row = 44 × (hidden + intermediate / tp) / (hidden + intermediate)
```

- `Npad` は `ExecutorConfig::max_batch_rows`（executor の行容量）を row bucket へ寄せた
  行数。serve 経路では `max_batch_rows = min(--max-num-batched-tokens, 最上位 bucket)` で、
  `max_seq_len` とは独立である（prefill は chunk 分割）
- `MO` は `max(max_scheduled_output_rows, max_concurrent_requests)` を `max_batch_rows`
  で clamp した値、`R` は `max_concurrent_requests`
- 44 は TP1 の program workspace を実測した行あたり bf16 値数（実測 43.7、上ぶね）。
  program workspace の大半は `intermediate_size` に比例するため、TP では
  `intermediate / tp` で縮む。TP2 rank の実測は 22.4 値/row で、この式は 26.9 を出す
- 静的項は `create_model_executor` の確保と同一の式（`logits {MO, V}`、
  `token_hidden {Npad, H}`、target hidden taps、sampling radix、
  `create_device_batch_context` の各 buffer、scratch 32 KiB、attention partials 32 MiB）

executor footprint の実測内訳（TP1 / bucket 4096 / requests 4）は program workspace が
ほぼ全てである（合計 1.78 GiB のうち 1,780.66 MiB。うち max program 1,748.62 MiB）。
`logits` は 3.79 MiB、`token_hidden` + sampling で 40.24 MiB、batch context 類は 1 MiB 未満。

auto は過大評価側に置く（reserve が実測を上回れば KV がその分だけ減る）。
実測との対応は起動 log で確認する。

```text
MEMORY_PLAN auto: working set reserve 2.03 GiB (row bucket 4096)
MEMORY_PLAN executor footprint: 1.78 GiB (reserve 2.03 GiB)
MEMORY_PLAN auto tp rank 0: working set reserve 2.06 GiB (row bucket 4096)
MEMORY_PLAN tp rank 0 executor footprint: 1.16 GiB (reserve 2.06 GiB)
```
`DFLASH2_WEIGHT_ARENA_BYTES` は serve 時に drafter 重みが入った実測増分であり、
`draft weights` の見積（safetensors 合計）と対応させる。

auto で `required` が `available` を超えるときは次の順で縮小する。

1. `runtime workspace` を KV と drafter を引いた残りへ縮小する。
   128 MiB を下回る場合は model load 前に `insufficient_memory` で失敗する。
2. `prefix cache` の hot tokens を 1 page 単位で縮小する。
3. 1 page も入らなければ prefix cache を無効化する。

縮小は stderr に 1 行ずつ出す。`--arena-gib N` は measurement の再現用であり、
N GiB を固定値として使う（auto の縮小はしない）。TP=2 では上の内訳ではなく、
rank0 の arena について次の 1 行を出す。

```text
MEMORY_PLAN mode=auto tp=2 arena=xx.xx GiB used=xx.xx GiB available=xx.xx GiB
```

## Scheduler

名前は2つに分離した。`max_num_batched_tokens` は 1 scheduled step の token 予算
（CLI は `--max-num-batched-tokens`）で、`ExecutorConfig::max_batch_rows` は executor の
行容量（`min(max_num_batched_tokens, 最上位 row bucket)` から導出）である。
旧 `max_scheduled_tokens` は両方の意味で使われていたため廃止した。

`schedule_requests(span<const RuntimeRequest*>, max_num_batched_tokens,
max_scheduled_requests, page_tokens, KVBankerState*)` はfree function
（`runtime/token_budget_scheduler.h`）。token budgetとKV capacityの制約下で
`SchedulePlan`（decode / prefill requestの順序付き選択）を返す。
`SchedulePlan` は `append_decode` / `append_verify` / `append_prefill` / `finalize` で
`ScheduledBatch` を直接buildする。

region order は decode → verify → prefill で固定であり、`finalize()` が
`num_decode_requests` / `num_verify_requests` / `num_prefill_requests` を設定する。
`append_verify` は `execution_class = SPEC_VERIFY` の request を追加し、verify 行数
（`num_output_rows`）と candidate 数（`num_tokens`）を要求する。verify request が 1 つ以上
ある batch は `speculative_verify = true` になる。

`ScheduledBatch` は decode region と verify region を同時に持てる（mixed batch）。
executor は verify 行を持つ batch を `ExecutionRole::Verify` として実行し、paged
attention は verify request ごとに region を切る。

mixed batch の制約: GDN の conv spec history は mixed batch で request 相対の row index を
使うため、conv capture（partial reject の base 復元）を必要とする verify request は
1 batch につき 1 つまでとする。複数の verify request を 1 batch に載せる場合は
batch 全体が verify であること（`Σ(Kᵢ+1)` 行、絶対 row index）を要求する。

## Cancellation

`ContinuousBatcher::cancel(uint64_t id)` は対象requestが `Queued` / `Active` のとき、
`RequestState::Finished` / `FinishReason::Cancelled` へ遷移し、割り当て済みの
sequence slot / GDN state / paged KV / capacity claimを解放する。`Finished` 済みの
requestへのcancelはno-op。

`Qwen35ComputeRuntime::cancel(uint64_t id)` は `ContinuousBatcher::cancel()` へ委譲する
だけの薄いAPIで、runtime layerに新しいcancel logicを持たない。

cancel granularityは `runtime.step()` の境界である。実行中のGPU kernelはpreemptせず、
device reset / `hipStreamDestroy` / signalをcancel目的に使用しない。cancel latencyは
最大で現在実行中の1 step。

`shelf-compute --serve-stdio` の service loopはGPU runtime ownerである単一thread上で、
`poll(2)` によりstdinを non-blockingに監視する。idle時はblocking poll、in-flight request中は
timeout 0 pollと1 stepを交互に行い、各stepの前後で `ping` / `cancel` / `shutdown` を
処理できる。stdout writerも同一threadで、mutexを持たない。

## Concurrency

`Qwen35RuntimeConfig::max_concurrent_requests`（compute CLI `--max-concurrent-requests`）は
最大in-flight request数であり、`SequenceSlotPool` / `GdnStatePool` /
`SequenceSlotPool` / `GdnStatePool` の slot 数と
`ContinuousBatcher` の 1 step あたり上限に同一値として伝播する。defaultは1で、single-shotとsingle-request parityを変えない。

論理concurrencyとphysical KV budgetは別契約である。

- `max_blocks_per_sequence = ceil_div(max_seq_len + 1, page_tokens) + 1`
- `physical_capacity_tokens = fit_pages * page_tokens`
- `kv_pool_pages = fit_pages + 1`

`fit_pages` は起動時に arena の空きから model weight、prefix cache、working set reserve、
drafter weight を差し引いた後に収まる KV page 数である。実行中に KV を拡張しない。
physical capacityが不足する場合、KV BankerがActive requestのmax_claimを保証した上で
Queued requestをadmissionする。`max_concurrent_requests > 1` でも全requestが常にActiveに
なるとは限らない。

複数requestは同一`ContinuousBatcher`へ`submit`され、1回の`step()`で
`ScheduledBatch`としてco-batchされる。requestごとにHIP streamやExecutorを作らない。
`DeviceBatchContext::num_requests` が1 stepでco-batchされたrequest数である。debug時は
`SHELF_BATCH_TRACE=1` でstderrへ `SHELF_BATCH_STEP requests=R tokens=T` を出す。

注意: 通常decodeのlinear kernelはbatch行数に応じてGEMV / GEMM configを選択する
（`inference/execution/selectors/linear_selector.cpp`）。このためco-batchされたrequestのgreedy tokenは、
single-request実行と一致しない場合がある（argmaxが僅差のとき）。これはcross-request
混入ではなく、kernel選択に依存する浮動小数の差である。batch行数に依らずdecodeと
bit一致する経路はtarget verify（`role == Verify`）のexact系kernelに限られる
（GEMMのmulti-row `ExactRows`とGDN recurrenceのserial variant、mode切替なしの
role固定）。request間の
event demultiplexと `token* == done.generated_ids` はco-batchでも保証される。

## Request retirement

`ContinuousBatcher` は `std::list<RuntimeRequest>` と `unordered_map<uint64_t, iterator>` を
持ち、address安定とO(1) eraseを両立する。`finish_request()` は`Finished`遷移と
GPU state解放（sequence slot / GDN state / paged KV / capacity claim）のみを行い、
host-side request objectは解放しない。

`ContinuousBatcher::retire(uint64_t id)` は `Finished` requestのみを`requests_`と
`request_index_`から削除する。`Queued` / `Active`のretireは `invalid_state`、unknown IDは
`invalid_argument`。serverはterminal `done` を生成した後にretireする。これにより
`generated` をterminal JSONへ書いた後にhost memoryを解放できる。

`Qwen35ComputeRuntime::retire(uint64_t id)` は `ContinuousBatcher::retire()` への薄いdelegate。

## Multi-request terminal semantics

停止条件の共通意味論（優先順位・one-way latch・zero-work との分離）は
[stop_conditions.md](stop_conditions.md) を正本とする。

`--serve-stdio` の `ServeState` は `std::map<int64_t, ServeGeneration>` でin-flight requestを
保持し、request_idでeventをdemultiplexする。request間のevent順序は保証しないが、
1 request内では `token*` の後にterminal event（`done` または `error`）が必ず1個だけ出る。
`cancel` はrequest-specificで、対象requestのみを `cancelled` にする。直前にterminalへ
遷移したrequest IDはbounded history（256件）に保持し、遅延cancelをidempotent no-opとして
吸収する。shutdown / EOFでは全in-flight requestをcancelし、全terminalの後に
`{"event":"shutdown"}` を出す。

### DFlash2 speculative decode 経路

`--dflash2-model-dir` 指定時の serve も通常 decode と同じ `ContinuousBatcher` 経路を
通る。`Qwen35ComputeRuntime::submit()` / `step()` / `request()` / `cancel()` /
`retire()` は両モードで共通である。

- `ContinuousBatcher::set_dflash2` が drafter（`DFlash2Config` / `DFlash2Executor` /
  `DFlash2SpecBatchContext` / arena / drafts / stats）を bind する。bind 済みの
  batcher に `speculative_decode = true` で submit された request は admission 時に
  `DFlash2SessionPool` から slot を借り（`DFlash2ContextState` + `DFlash2SpecWorkspace`）、
  `DFlash2SpecDecoder` を request 専用に作成して、prefix cache の restore 後に
  `dflash2_spec_prefill` で chunk prefill する。session は `RuntimeRequest` が所有し、
  `finish_request` が decoder を shutdown → pool slot を release → sequence を解放する
  順に返す。
- `step()` は DFlash2 request を slot にまとめて `dflash2_spec_step_batch()` を 1 round
  実行し、続けて通常 decode の schedule / execute を同じ step で実行する（1 step =
  最大 2 batch）。DFlash2 request は `pending_decode_token < 0` のため通常 decode の
  scheduler には選ばれない。
- **DFlash2 prefill と最初の round は同じ step で実行しない**。admission で
  `dflash2_spec_prefill` が完走した request がある場合、`step()` はその step を
  prefill の完了として返し（`StepResult{executed=true, num_tokens=prompt-restored,
  num_requests=prefilled}`）、投機 round は次の step から始める。prefill が
  サンプルした最初のトークンを round の分（~55ms）遅らせずに emit するためで、
  TTFT がその分短くなる（8192 の BetterBench で 135 → 83ms）。
- admission / retire / cancel / prefix cache / KV Banker は通常 decode と同一契約である。
  DFlash2 専用の KV 容量自動拡張は無い（起動時に確定した KV pool を
  `ContinuousBatcher` の banker がそのまま使う）。
- `generated` / `finish_reason` / `restored_tokens` は `RuntimeRequest` が保持し、
  `done` event は通常 decode と同じ経路で出力する。
- drafter は `DFlashServeContext` が serve 間保持し、serve 終了時に
  `shutdown_dflash2_spec_batch_context()` → `dflash2_executor_shutdown()` の順で解放する。

詳細は [dflash2.md](dflash2.md) を参照。

## Prefix cache

prefix cache は compute アプリが所有する `PrefixCache`
（`shelf/models/qwen35/runtime/prefix_cache.h`）である。radix tree 上の
page 単位 prefix KV / GDN state を VRAM（hot）と pinned DRAM（cold）に保持し、
`ContinuousBatcher` が request admission 時に lookup・restore、
prefill 完了時・decode 境界通過時・出力末尾で save を行う。
`PrefixCache::create` に失敗した場合は無効化して続行し、
`ping` の `capabilities.prefix_cache` は `false` になる。詳細は
[prefix_cache.md](prefix_cache.md) を参照。
