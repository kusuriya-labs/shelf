# Qwen3.5 ホストランタイム → カーネル呼び出し配線

Qwen3.5 の推論を実際に走らせる、ホスト側ランタイムから GPU カーネルの起動までの全経路を記録する。
「どうすれば速く書けるか」のための実装マップであり、抽象層の説明ではない。

コード:

- `src/shelf/models/qwen35/runtime/executor.hip`
- `src/shelf/inference/execution/program_executor.hip`
- `src/shelf/inference/execution/optimized_dispatch.hip`
- `src/shelf/kernels/correctness/model_dispatch_correctness.hip`
- `src/shelf/models/qwen35/model/lower_to_primitives.cpp`
- `src/shelf/runtime/program/program.cpp`
- `include/shelf/runtime/program/program.h`
- `include/shelf/runtime/program/kernel_id.h`

---

## 1. 実行パスの全体像

2つのパスが同居する。

- **Correctness** : 1 dispatch ごとに単一カーネル `model_dispatch_kernel` を1回 launch し、デバイス内で `host_execute_binding` の switch によって op 分岐する。実装は `correctness/detail/*.inc` に分割し、単一 TU に include する。
- **Optimized(Auto)** : 各 dispatch を op ごとの専用 launcher で実行する。単体 launcher は 1 dispatch を消費する。

モードは環境変数 `SHELF_QWEN35_KERNEL_MODE` で決まる。

- 未設定 / `auto` → `Auto`（optimized 優位、失敗時は correctness にフォールバック）
- `correctness` → 常に correctness
- 他 → エラー

`read_kernel_mode()` が読む。

optimized カーネルは各 launcher 内で shape / dtype / preshuffle contract を検証し、非対応なら `NotApplicable` を返す。その dispatch は correctness 経路で走る。

```
submit_batch
  └─ submit_co_batch                (executor.hip)
       ├─ validate / row layout / request descriptor
       ├─ KV・GDN reservation (reserve → commit/rollback/poison)
       ├─ DeviceBatchContext build + upload
       ├─ row bucket 解決 + program resolve
       ├─ (SHELF_HIP_GRAPH) graph capture/launch
       └─ launch_host_backend
            └─ execute_program      (program_executor.hip)
                 ├─ [Auto] try_launch_optimized   (optimized_dispatch.hip)
                 │    └─ 1 dispatch を消費
                 └─ [fallback] launch_host_binding
                      └─ fill_staging_image + launch_model_dispatch_correctness
                           └─ model_dispatch_kernel<<<64, 256>>>
                                └─ host_execute_binding (デバイス switch)
                                     └─ 各 __device__ exec_*（detail/*.inc）
complete_batch                    (executor.hip)
  └─ event sync → status 確認 → KV・GDN commit
```

### 1.1 attention 出力ゲート

full attention 層の gated query は `q_proj` の出力を head 単位で `[q | gate]` に分割し、
attention 出力に sigmoid を掛けてから `o_proj` へ渡す。活性は常に sigmoid で、
model config からは変わらない。

```
attn = attention(q_norm(rope(q)), k_norm(rope(k)), v)
out  = o_proj(attn * sigmoid(gate))
```

### 1.2 GDN 出力ゲート

GDN 層の recurrence 出力は `gdn_norm` 後に `qkv_proj` の gate 部分と活性を掛ける。
活性は `text_config.output_gate_type` で決まる。

| `output_gate_type` | gate 活性 | 既定 |
| --- | --- | --- |
| `sigmoid` | sigmoid | ○（key 省略時） |
| `swish` / `silu` | SiLU | |

未対応の値は model load 時の error とする（暗黙に sigmoid へ落とさない）。
Qwen3.8-27B は `swish` である。

---

## 2. 初期化経路（`create_model_executor`）

`Result<Executor> create_model_executor(const Qwen35Model&, SequenceSlotPool&, GdnStatePool&, PagedKVPool&, gpu::GpuArena&, const ExecutorConfig&)`

### 2.1 検証

- `config.max_scheduled_tokens` / `max_scheduled_requests` が > 0
- `kv_pool.dtype()` が `BF16` か `FP8_E4M3`
- FP8 時は body/scale の要素数と device 一致
- GDN state pool 幾何（`GdnStatePoolLayout::from_text_config`）と一致
- KV pool 幾何（attention 層数 / kv_heads / head_dim）と一致

### 2.2 アロケーション

`gpu::GpuArena` から連続アロケーション（`allocate_contiguous_tensor`）で割る。

- `input_ids` : `{N}` int32（N = max_scheduled_tokens）
- `output_hidden` : `{R, H}` bf16
- `logits` : `{R, V}` f32
- `sampled_tokens` : `{R}` int32
- `execution_status` : `{1}` u32（0 に初期化）
- execution workspace : `max(各 bucket の workspace.total_bytes) + 8192 * 4`（scratch 付き）
- host staging : `host_request_staging` / `host_pending_staging` / `host_sampled_staging`（pinned）

### 2.3 Lowering → Program

```
lower_qwen35_to_primitives(text_config, weights)
  → PrimitiveGraph + WeightSlot[] + StaticParameterSlot[]
build_program_set(graph, weights, params, DECODE, opts)
  → ProgramSet（R16..R8192 の10 bucket を1つずつ build）
```

- `opts.validate_workspace = true`
- `opts.max_token_rows = N`
- `opts.max_output_rows = R`
- 実行クラスは `DECODE` のみ（1つ）
- GDN / gated attention の geometry validation と Graph emit は
  Qwen-family 共通 builder（`models/common/qwen_hybrid/`）を経由する。
  topology（layer 構成 / residual / MLP / MTP）は Qwen35 側に残す。

### 2.4 定数テーブルと staging アップロード

- `upload_host_weight_table` : WeightSlot テーブルを device に up（`validate_host_weight_slot` で検証）
- `upload_host_parameter_table` : StaticParameterSlot テーブルを device に up
- `program_staging_meta` : 各 bucket の workspace ranges / states を device に up（15 バイトアライメント）
- `dispatch_staging` / `host_dispatch_staging` : 最大 staging byte 分の device/pinned 確保
- `staging_pool` : 各 (bucket, binding) ごとに offset/bytes/uploaded を保持する pool。1 回アップロードで再利用。

### 2.5 Static YaRN（`EffectiveRopeProfile`）

`--rope-factor auto | 1 | 2 | 4` は `EffectiveRopeProfile`（`include/shelf/models/qwen35/model/rope_profile.h`）に解決され、`ExecutorConfig.rope_profile` → `Qwen35LowerOptions.rope_profile` → `RoPENode` → `DispatchBinding.scalar_a`（attention factor）として実行経路に伝わる。解決は起動時に 1 回だけ行い、以後不変とする。

- 解決の入力は `<model_dir>/config.json` の `text_config.rope_parameters` と CLI。`rope_type` は `default` / `yarn` のみ受理し、それ以外は起動失敗する。
- `auto` は checkpoint の `rope_type` に従う（`yarn` なら config の factor、`default` なら Native）。CLI の `1 | 2 | 4` は checkpoint の指定より優先する。
- YaRN の `original_max_position_embeddings` は `rope_parameters` の値、無ければ `text_config.max_position_embeddings` を使う。
- `attention_factor` は config の明示値、無ければ `1 + 0.1 * ln(factor)`。`beta_fast` / `beta_slow` / `truncate` の既定は 32 / 1 / true。
- `--max-seq-len` の上限は `min(original_len * factor, 1000000)`。Native は `original_len`。
- YaRN 選択中の speculative decoding（`--dflash2-model-dir`）は起動時に拒否する。
- 逆周波数 table は YaRN 時に `launch_rope_inv_freq_yarn_init`、Native 時に `launch_rope_inv_freq_init` で 32 要素の double を生成する。

解決結果の同一性は経路ごとに次の契約で保証する。

- **local load / resident load**: profile は compute プロセスが `<model_dir>/config.json` から独立に解決する（Resident archive には rope メタデータを追加しない）。`shelf-launcher`（model-host 経由）と local load で同じ `ROPE profile:` 行と同一の生成結果になることを確認済み。
- **TP**: 1 プロセスで単一の profile を解決し、`TpCoordinatorConfig` → `TpRankRuntimeConfig` → `ExecutorConfig` へ複製する。したがって rank 間で profile がずれる経路は存在しない。TP の KV 予算チェックは rank 単位で行い、不足時は `tp rank arena is too small` で起動を拒否する。
- **KV 容量計画**: `--rope-factor` は `required_pages` / `kv_page_bytes` に寄与しない。KV 容量は `--max-seq-len` と `--kv-cache-capacity-tokens` のみで決まる。
- **PrefixCache**: キーはトークン列のみで rope 識別子を含まないが、factor はプロセス起動時に固定され、キャッシュはプロセスローカルであるため、異なる factor 間で KV を流用する経路は存在しない。prefix の保存は prefill 位置が `page_tokens` の倍数のときに限られ、復元は保存済み prefix より長いプロンプトに限る。
- **server**: `--rope-factor` は `build_compute_argv` で子 compute へ転送し、入力長検証（`context length exceeded`）は factor と独立に `--max-seq-len` で判定する。

---

## 3. PrimitiveGraph → Program（`build_program`）

`build_program(graph, weights, bucket, execution_class, options)` が graph のノードを `DispatchBinding` に変換する。

### 3.1 ノード → KernelId 対応表（`emit_node`）

| PrimitiveKind | KernelId | 備考 |
|---|---|---|
| `EMBEDDING_LOOKUP` | `EMBEDDING_LOOKUP` | weight_index 付き |
| `LINEAR` | `LINEAR_BF16` / `LINEAR_FP8` | weight encoding で分岐（bf16=0, fp8=5） |
| `RMS_NORM` | `RMS_NORM` | eps / group_size / parameter_index |
| `RESIDUAL_ADD` | `RESIDUAL_ADD` | 2入力 |
| `SPLIT` | `SPLIT` | head_dim / layout flag |
| `SILU` / `SIGMOID` / `L2_NORMALIZE` / `ROPE` / `SCALE` | 同名 KernelId | scalar/rotary/group 付き |
| `MUL` / `SWIGLU` | `MUL` / `SWIGLU` | 2入力 |
| `CONCAT` | `CONCAT` | 2入力。`input_shape_a.features` を group_size に |
| `KV_APPEND` | `KV_APPEND` | state（KV_CACHE）を descriptor_slot として登録 |
| `PAGED_ATTENTION` | `PAGED_ATTENTION` | scale / head_dim / KV state |
| `STATEFUL_CAUSAL_CONV1D` | `STATEFUL_CAUSAL_CONV1D` | GDN_CONV_STATE を登録 |
| `GDN_RECURRENCE` | `GDN_RECURRENCE` | GDN_RECURRENCE_STATE + dt_bias / a_log parameter |
| `OUTPUT_GATHER` | `OUTPUT_GATHER` | |
| `SAMPLING` | `SAMPLING` | vocab_size を rotary_dim に |

### 3.2 アクティベーション量子化の挿入

`LINEAR` が量子化 weight（fp8）の場合、weight encoding に伴い、入力値に対してアクティベーション量子化 dispatch を自動挿入する。

- 挿入される KernelId: `ACTIVATION_QUANTIZE_W4A8`
- group_size = 32
- `activation_kp` / `activation_max_rows` / `code_stride` / `scale_stride` を `Int8ActivationWorkspaceLayout::make(k_padded, max_rows)` で設定する。`k_padded` は `activation_k_padded(encoding, k)`（fp8 は 128）
- 同一入力への複数 LINEAR が共通 workspace を使う場合（`activation_ws_by_input` で管理）は `k` / `k_padded` が一致するか検証

bf16 linear（非量子化）はアクティベーション量子化を挟まず、通常に `input_slots[0]` を使う。

### 3.3 state の登録

`KV_APPEND` / `PAGED_ATTENTION` / `STATEFUL_CAUSAL_CONV1D` / `GDN_RECURRENCE` は `register_state` で state を Program に登録し、`descriptor_slot = slot + 1` として binding に入れる。

GDN の conv state は bf16、recurrent state は fp16（`ps::fp16_t`）で保持する。
recurrence の matrix 演算は fp16 WMMA
（`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`）で行い、`kLossy=true` は
hi 項 1 回、`kLossy=false` は hi/lo の 4 項で積む。recurrent state の動的範囲は
fp16 の表現域（|x| <= 65504）に限られ、recurrent state を複数行まとめて処理する
kernel は行ごとに fp16 丸めを適用して 1 行ずつ書き戻す経路と bit 一致させる。

### 3.4 workspace layout（`finalize_layout`）

- workspace 用 `ValueBinding` と activation workspace を、ライフタイム（first/last command）で管理
- `first_command` 昇順、同順位は byte 数降順にソート
- 空きブロック（`FreeBlock`）を best-fit（収まる最小ブロック）で再利用（`no_workspace_reuse` 無効時）
- `kWorkspaceAlignment = 256` でアライン
- 結果を `plan.workspace.ranges` に書き、value / activation workspace の `workspace_slot` をセット
- `validate_workspace` 時は重複 lifetime を検出

### 3.5 ProgramSet

`build_program_set` は R16 / R32 / R64 / R128 / R256 / R512 / R1024 / R2048 /
R4096 / R8192 の 10 bucket に対して `build_program` を呼ぶ。`ProgramSet::kRowBucketCount = 10`。
最上位 bucket が prefill chunk の上限であり、serve の `max_scheduled_tokens` もここで
clamp される（R8192 = 参照実装の CHUNK と同じ 8192 行）。

---

## 4. バッチ実行経路（`submit_co_batch` / `complete_batch`）

### 4.1 `submit_co_batch`

1. `active_submission_id != 0` なら in-flight バッチ存在でエラー
2. `validate_scheduled_batch(batch, max_scheduled_tokens, max_scheduled_requests)`
3. 各 request について:
   - `PendingRequestCommit`（sequence / handle / committed_tokens）
   - `DeviceRequestDescriptor` staging（request_handle / row_begin / row_count / prefix_length / sequence_length / execution_class / compute_logits / sample_greedy / output_index）
   - `compute_logits` なら `output_index` を採番し `sample_greedy` なら sampled_count を増分
   - `DECODE` / `SPEC_VERIFY` / `PREFILL` ごとに count
4. 各 request の `reserve_sequence_append`（失敗時は rollback）
5. `DeviceBatchContext` を構築:
   - `actual_rows` = N
   - `num_requests` = R
   - `num_outputs` / `num_decode_requests` / `num_verify_requests` / `num_prefill_requests`
   - `token_ids` / `requests` / `row_sequence_slots` / `row_positions` / `output_rows` / `output_sample_flags`
6. row bucket 解決: `resolve_row_bucket(N)` → `bucket_m = row_bucket_limit(bucket)`
7. `program_set.resolve(bucket)` で Program を取得
8. `BatchExecutionOutput`（output_hidden / logits / sampled_tokens の slice）
9. `hipEventCreateWithFlags`（`hipEventDisableTiming`）で done イベント確保
10. `SHELF_HIP_GRAPH` 有効時、`graph_cache` がある bucket に対しては capture / launch を試行
    - 既存 graph があるなら `hipGraphLaunch`
    - 無いなら `hipStreamBeginCapture` → `enqueue_batch` → `hipStreamEndCapture` → `hipGraphInstantiate` → `hipGraphLaunch`
    - 失敗時は `graph_failed = true` にして fallback
11. `enqueue_batch`（通常経路）:
    - `upload(input_ids, token_src, N * sizeof(int32_t))`
    - `upload(bctx_token_ids, ...)`
    - `upload(bctx_requests, staging, R * sizeof(DeviceRequestDescriptor))`
    - `upload(executor.batch_context, ctx_src, sizeof(host_ctx))`
    - `launch_prepare_batch_descriptor`
    - `launch_host_backend`
12. `active_submission_id = submission_id`、`gpu_started = true`
13. `hipEventRecord(done, stream)`

### 4.2 `complete_batch`

1. `pending.done != nullptr` を確認
2. `pending.submission_id` が `active_submission_id` と一致
3. `hipEventSynchronize(done)`
4. `hipEventDestroy(done)`
5. `hipMemcpy` で `execution_status` を読み、`ProgramStatus::COMPLETE` 以外なら poison + エラー
6. 各 commit について `validate_sequence_append_commit`
7. 各 commit について `commit_sequence_append`
8. `active_submission_id = 0`、`BatchExecutionOutput` を返す

---

## 5. `execute_program` ループ

`Status execute_program(const Program&, HostExecutionContext&, hipStream_t)`

1. `error_word` を `hipMemsetAsync` で 0 に初期化
2. `read_kernel_mode()` でモード取得
3. `for i in 0..program.dispatches.size()`:
   - `OpDumpTimer`（`SHELF_OP_DUMP` 有効時）
   - `semantic_timing_start`（`SHELF_OP_SYNC` 有効時）
   - **Auto モード**（単体 launcher を試す）:
     - `try_launch_optimized(program, i, ctx, stream)`
     - `Launched` / `Skipped` なら `consumed_dispatches` を検証し、`i += consumed` でループ継続
     - `NotApplicable` なら fallthrough
   - **fallback**:
     - `launch_host_binding(binding, program, ctx, stream)`
     - `record_correctness_fallback` / `record_physical_launch(1)` / `record_logical_covered`
     - `semantic_timing_finish`（`SemPath::Correctness`）
     - `i++`

---

## 6. Correctness パス（`launch_host_binding`）

`Status launch_host_binding(const DispatchBinding&, const Program&, HostExecutionContext&, hipStream_t)`

1. `batch_context` / `scratch` / `error_word` の存在確認
2. `staging_section_offsets(nval)` で staging 内の offset を取得
3. **staging pool 経路**（`SHELF_HIP_GRAPH` 有効時、staging pool が device にアップロード済み）:
   - `ctx.staging_pool->slot(bidx, bindx)` で slot を取得
   - `uploaded[slot] == 0` なら `fill_staging_image` で host staging に書き、`hipMemcpyAsync` で device に up、`hipStreamSynchronize`、`uploaded[slot] = 1`
   - `stream_capturing` 中は pool upload を禁止
4. **通常 staging 経路**（pool なし）:
   - `fill_staging_image` で host staging に書き
   - `hipMemcpyAsync` で device に up
   - `stream_capturing` 中は禁止
5. `DeviceProgramView prog_dev` を device staging から読み
   - `values` / `value_count` / `weight_table` / `parameter_table` / `workspace_ranges` / `states` / `workspace_base`
6. `launch_model_dispatch_correctness(prog_dev, binding, batch_context, model_state, scratch, error_word, stream)`
7. `via_pool` でない、`sync_each_op`、または `EMBEDDING_LOOKUP` のとき `hipStreamSynchronize`

### 6.1 `fill_staging_image`

host staging に `DispatchBinding` / `DeviceValueBinding[]` / `StagingProgramRefs` を書き込む。

- 各 input/output について `plan.find_value(vid)` で `ValueBinding` を取得し、`DeviceValueBinding` にコピー
- `EXTERNAL_INPUT` / `EXTERNAL_OUTPUT` は `ctx.external_inputs[slot]` / `ctx.external_outputs[slot]` を offset に
- `WORKSPACE` は `found->offset` をそのまま使う
- `StagingProgramRefs` に `ranges` / `states` をセット
- `DeviceProgramView` に `values` / `weight_table` / `parameter_table` / `workspace_ranges` / `states` / `workspace_base` をセット

### 6.2 `model_dispatch_kernel`

```
constexpr uint32_t kDispatchBlocks = 64;
__global__ void model_dispatch_kernel(
    DeviceProgramView prog,
    DispatchBinding binding,
    DeviceBatchContext* ctx,
    ModelDispatchStateView state,
    float* scratch,
    uint32_t num_blocks,
    uint32_t* error_word)
```

- `env.scratch = scratch + blockIdx.x * kScratchRowCapacity`
- `host_execute_binding(env, binding)` を呼び、`ProgramStatus::COMPLETE` 以外なら `atomicCAS(error_word, ...)`

### 6.3 `host_execute_binding`（デバイス switch）

`binding.kernel_id` で `detail::exec_*` を呼び分ける。各実装は
`correctness/detail/*.inc`（elementwise / normalization / linear / attention / gdn /
embedding / output）に置き、`model_dispatch_correctness.hip` が単一 TU に include する。
1 ファイルに複数の KernelId 実装を置いてよい（実装 family 単位）。

---

## 7. Optimized パス

Auto モードでは `execute_program` が単体 launcher を試す。

### 7.1 単体 launcher の dispatch 順序

`try_launch_optimized` は以下のように分岐する（先着順）。

1. `EMBEDDING_LOOKUP` → `try_launch_embedding`
2. `SAMPLING` → `try_launch_sampling`
3. `ACTIVATION_QUANTIZE_FP8` / `ACTIVATION_QUANTIZE_W4A8` → `try_launch_activation_quantize_a8`
4. `LINEAR_FP8` → `try_launch_linear_fp8`
5. `LINEAR_BF16` → `try_launch_linear_bf16`
6. `RMS_NORM` → `try_launch_rmsnorm`
7. `ROPE` → `try_launch_rope`
8. `OUTPUT_GATHER` → `try_launch_output_gather`
9. `L2_NORMALIZE` → `try_launch_l2_normalize`
10. `SPLIT` → `try_launch_elementwise`
11. `KV_APPEND` → `try_launch_kv_append`
12. `PAGED_ATTENTION` → `try_launch_paged_attention`
13. `STATEFUL_CAUSAL_CONV1D` → `try_launch_gdn_conv1d`
14. `GDN_RECURRENCE` → `try_launch_gdn_recurrence`
15. `SIGMOID` / `RESIDUAL_ADD` / `SWIGLU` / `SILU` / `MUL` / `SCALE` / `CONCAT` → `try_launch_elementwise`
16. 他 → `NotApplicable`

`OptimizedLaunchResult.kind` は `Launched`（`consumed_dispatches` 付き）、`Skipped`（zero-row の作業なし dispatch を消費）、または `NotApplicable`。

### 7.3 単体カーネル

- **embedding**: `try_launch_embedding`
- **sampling**: `try_launch_sampling`
- **activation quant**: `try_launch_activation_quantize_a8`（a8 量子化）
- **linear**: `try_launch_linear_fp8` / `try_launch_linear_bf16`
- **rmsnorm**: `try_launch_rmsnorm`
- **l2_normalize**: `try_launch_l2_normalize`
- **elementwise**: `try_launch_elementwise`（SPLIT / SIGMOID / RESIDUAL_ADD / SWIGLU / SILU / MUL / SCALE / CONCAT）
- **kv_append**: `try_launch_kv_append`
- **paged_attention**: `try_launch_paged_attention`
- **gdn_recurrence**: `try_launch_gdn_recurrence`
- **gdn_conv1d**: `try_launch_gdn_conv1d`
- **rope**: `try_launch_rope` — head_dim=256 / rotary=64 / features%256==0 で pointer/stride aligned かつ inv_freq table が存在するなら専用 pair kernel、でなければ generic kernel を自動選択する。pair と generic は同じ `inv_freq` table を参照し、`cos`/`sin` に `attention_factor` を掛ける。correctness 経路も `DeviceBatchContext.rope_inv_freq` で同じ table と binding の `scalar_a`（attention factor）を参照する。table は `create_model_executor` が model setup 時に 32 要素の double として生成し、`HostExecutionContext.rope_inv_freq` 経由で渡す（rotary_dim=64 前提で table 長は 32 固定、half が 32 を超える geometry では table を使わず theta から計算する）
- **output_gather**: `try_launch_output_gather`

各 launcher は selector（`select_*_implementation` / `select_*_config`）で `Correctness` / `Optimized` を判定し、`Optimized` でなければ `NotApplicable` を返す。量子化 linear は対応 weight が native layout に preshuffle 済みであることも検証する。Tensor Parallel の local geometry は半分の次元になるため、各 selector（`linear_selector` / `paged_attention_selector` / `gdn_recurrence_selector` / `rope_selector` / `kv_append_selector` / `l2_normalize_selector` / `gdn_conv_selector` / `rmsnorm_selector` / `elementwise_selector`）には TP=2 の validated shape を個別に登録する（[tensor_parallel_execution.md](tensor_parallel_execution.md) §10）。

---

## 8. KernelId 一覧

`enum class KernelId : uint32_t`:

| 値 | 名前 |
|---|---|
| 0 | `EMBEDDING_LOOKUP` |
| 1 | `LINEAR_BF16` |
| 4 | `RMS_NORM` |
| 5 | `RESIDUAL_ADD` |
| 6 | `SPLIT` |
| 7 | `SILU` |
| 8 | `SIGMOID` |
| 9 | `MUL` |
| 10 | `SWIGLU` |
| 11 | `ROPE` |
| 12 | `L2_NORMALIZE` |
| 13 | `SCALE` |
| 14 | `KV_APPEND` |
| 15 | `PAGED_ATTENTION` |
| 16 | `STATEFUL_CAUSAL_CONV1D` |
| 17 | `GDN_RECURRENCE` |
| 18 | `OUTPUT_GATHER` |
| 19 | `SAMPLING` |
| 20 | `ACTIVATION_QUANTIZE_FP8` |
| 21 | `ACTIVATION_QUANTIZE_W4A8` |
| 24 | `CONCAT` |
| 25 | `LINEAR_FP8` |
| 27 | `VERIFY_ACCEPT` |

カウントの正本は `KernelId::COUNT`（`include/shelf/runtime/program/kernel_id.h`）である。

---

## 9. 重みエンコーディング

`WeightSlot.encoding`:

| 値 | 意味 | 条件 |
|---|---|---|
| 0 | BF16 | `device_ptr` あり、`codes` / `scales` なし |
| 2 | ROCm FP8 | `codes` + `scales` あり、`device_ptr` なし、`k_padded % 32 == 0` |
| 5 | Fp8 | `codes` + `scales` あり、`device_ptr` なし、`k_padded % 128 == 0` |

`validate_host_weight_slot` がこの制約を検証する。

Tensor Parallel の rank-local weight と execution は
[tensor_parallel_execution.md](tensor_parallel_execution.md) を参照
（plan builder は `include/shelf/models/qwen35/weights/tensor_parallel_plan.h`）。

weight の load 時 logical TP partition（`WeightLoadOptions.partition_plan`）は
model 層の plan builder（`include/shelf/models/qwen35/weights/tensor_parallel_plan.h`）
が生成し、generic weights layer が
`global canonical → partition → preshuffle` の順で適用する。
現行 single-GPU lowering は global geometry を前提とするため、
Qwen35 の top-level load path は partition plan を
`Status::unsupported` で拒否する。contract は
[tensor_partition.md](tensor_partition.md) を参照。

---

## 10. 関連ファイル索引

### 初期化 / バッチ

- `src/shelf/models/qwen35/runtime/executor.hip`
  - `create_model_executor`
  - `submit_co_batch`
  - `complete_batch`
  - `launch_host_backend`

### 実行

- `src/shelf/inference/execution/program_executor.hip`
  - `execute_program`
  - `launch_host_binding`
  - `fill_staging_image`
  - `upload_host_weight_table`
  - `upload_host_parameter_table`

### Optimized

- `src/shelf/inference/execution/optimized_dispatch.hip`
  - `try_launch_optimized`
  - `try_launch_embedding` / `try_launch_sampling` / `try_launch_activation_quantize_a8`
  - `try_launch_linear_fp8` / `try_launch_linear_bf16`
  - `try_launch_rmsnorm` / `try_launch_l2_normalize` / `try_launch_elementwise` / `try_launch_rope`
  - `try_launch_kv_append` / `try_launch_paged_attention` / `try_launch_gdn_conv1d` /
    `try_launch_gdn_recurrence` / `try_launch_output_gather`
  - `read_kernel_mode`

### Correctness

- `src/shelf/kernels/correctness/model_dispatch_correctness.hip`
  - `model_dispatch_kernel`（唯一の `__global__`）
  - `host_execute_binding`（デバイス switch）
  - `launch_model_dispatch_correctness`
- `src/shelf/kernels/correctness/detail/`
  - `dispatch_env.h`（`ModelDispatchEnv` / `ResolvedValue` / 共通 helper）
  - `elementwise.inc` / `normalization.inc` / `linear.inc` / `attention.inc` /
    `gdn.inc` / `embedding.inc` / `output.inc`（実装 family 単位の fragment）
- `src/shelf/kernels/correctness/standalone/`
  - kernel 単体テスト用の参照実装

### Lowering / Program

- `src/shelf/models/qwen35/model/lower_to_primitives.cpp`
  - `lower_qwen35_to_primitives`
- `include/shelf/models/common/qwen_hybrid/`（Qwen-family 共通 builder）
  - `geometry_check.h` / `gdn_geometry.h` / `attention_geometry.h`
  - `emitter.h` / `gdn_lowering.h`（`emit_gdn`） /
    `attention_lowering.h`（`emit_gated_attention`）
- `src/shelf/runtime/program/program.cpp`
  - `build_program`
  - `build_program_set`
  - `emit_node`
  - `finalize_layout`

### 定義

- `include/shelf/runtime/program/program.h`
  - `Program` / `ProgramSet` / `DispatchBinding` / `WeightSlot`
- `include/shelf/runtime/program/kernel_id.h`
  - `KernelId` enum

### Selector

- `include/shelf/inference/execution/selectors/elementwise_selector.h`
- `include/shelf/inference/execution/selectors/embedding_selector.h`
- `include/shelf/inference/execution/selectors/gdn_conv_selector.h`
- `include/shelf/inference/execution/selectors/gdn_recurrence_selector.h`
- `include/shelf/inference/execution/selectors/kv_append_selector.h`
- `include/shelf/inference/execution/selectors/l2_normalize_selector.h`
- `include/shelf/inference/execution/selectors/linear_selector.h`
- `include/shelf/inference/execution/selectors/paged_attention_selector.h`
- `include/shelf/inference/execution/selectors/rmsnorm_selector.h`
- `include/shelf/inference/execution/selectors/rope_selector.h`
- `include/shelf/inference/execution/selectors/sampling_selector.h`
