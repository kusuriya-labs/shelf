# DFlash2 speculative decoding

DFlash2 は target model の途中層 hidden を入力に取り、1 anchor block 分の draft token を
まとめて提案する drafter による greedy speculative decoding である。target が提案 block を
verify し、一致した prefix だけを commit する。

対象:

- target: `models/Qwen3.8-27B-*`（Qwen3.8-27B、64 層）
- target: `models/Qwen3.8-Flash-Next-Quark-MXFP4`（Qwen4Exp、48 層、hidden 2560）。
  [qwen4exp.md](qwen4exp.md) の DFlash2 target tap を参照
- drafter: `models/Qwen3.8-27B-DFlash2`（BF16、指定は `--dflash2-model-dir`）
- `--dflash2-model-dir` を指定したときだけ DFlash2 経路に入る。未指定時は既存の
  `ContinuousBatcher` 経路を変更しない。

現在の性能値は [current performance](../perf/current.md) を参照。

---

## Scope

- greedy（`temperature == 0`）と stochastic（`temperature > 0`）の両方に対応する。
  stochastic の verify は target の sample 結果と draft を比較する sample-and-compare
  方式で、出力分布は target の分布に一致する。round で消費した row 数だけ
  `sample_index` を進め、RNG key の再利用を避けている。
- bf16 GEMM の kernel は実行行数だけで決まる。`rows == 1` は `ExactRows` の 1 行版、
  `2 <= rows <= 16` は multi-row `ExactRows`、`rows >= 17` は WMMA 系である。decode と
  verify で同じ行数なら同じ kernel になるため、batch 構成の違いで kernel が切り替わらない。
  GDN recurrence だけは execution role が Verify のとき serial 系 variant を固定選択する。
- drafter は 1 round で anchor 1 row + draft `block_size - 1` row を陽に処理する
  block-parallel 構成である。提案 block 内は非 causal。
- target 層数・tap 位置などは drafter の config が宣言し、target の層数と shape に
  突き合わせる。27B の既定は target 64 層、tap `[5, 19, 33, 47, 61]`。
- runtime 統合は `shelf-compute` の `run_dflash2_shot()` にある。executor / pool /
  arena / stream は `Qwen35ComputeRuntime` の accessor から取得する。
- `--serve-stdio` は `DFlashServeContext` 経由で同じ decoder 設定を作る。
  `shelf-server` も `--dflash2-model-dir` / `--dflash2-drafts` / `--dflash2-max-sessions` を転送する。
- serve mode は複数 request の target verify を 1 batch に統合する（batched verify）。
  scheduler の in-flight request をまとめて 1 round 実行し、draft も 1 回の batched
  drafter forward（`dflash2_propose_batch`）で生成する。

---

## Checkpoint contract

### config

`config.json` は flat（`text_config` を持たない）。`read_dflash2_config()` が読む key:

| key | 意味 |
| --- | --- |
| `architectures[0]` | `DFlash2DraftModel` のみ許可 |
| `vocab_size` / `hidden_size` / `intermediate_size` / `num_hidden_layers` | 形状 |
| `num_attention_heads` / `num_key_value_heads` / `head_dim` | attention 形状 |
| `sliding_window` | sliding attention の窓幅 |
| `is_causal` | **false のみ許可**（draft block 内は非 causal） |
| `num_target_layers` | target の層数 |
| `rms_norm_eps` | RMSNorm の eps |
| `tie_word_embeddings` | **false のみ許可**（drafter は embedding/lm_head を持たない） |
| `dflash_config.block_size` | anchor + draft の合計 row 数（8 なら anchor 1 + draft 7） |
| `dflash_config.conv_group_size` / `conv_kernel_size` | dynamic convolution 形状 |
| `dflash_config.mask_token_id` | noise block の MASK token |
| `dflash_config.selector_rank` / `selector_top_k` | candidate selector 形状 |
| `dflash_config.target_layer_ids` | target から tap する layer（昇順、`num_target_layers` 未満） |
| `rope_parameters` | 任意。存在する場合 `rope_type == "default"` のみ許可 |
| `layer_types` | 任意。存在する場合すべて `sliding_attention` のみ許可 |

契約検証（違反はすべて load 時の error。既定値で黙って補わない）:

- `block_size ∈ [2, 16]`（`kMaxBlockSize`）。
- `target_layer_ids` は空でなく、最大 `kMaxTargetLayerIds`（8）個、strictly increasing、
  各要素 `< num_target_layers`。
- `hidden_size % conv_group_size == 0`。
- `num_attention_heads % num_key_value_heads == 0`。
- `mask_token_id < vocab_size`、`selector_top_k <= vocab_size`。
- 必須次元（vocab / hidden / intermediate / layers / attention / sliding_window /
  num_target_layers）は 0 でない。

導出値:

- `max_draft_tokens = block_size - 1`
- `conv_groups = hidden_size / conv_group_size`
- `conv_projection_rows = 2 * conv_kernel_size * conv_groups`
- `attention_q_rows = num_attention_heads * head_dim`
- `attention_kv_rows = num_key_value_heads * head_dim`
- `tap_feature_size = num_target_layer_ids * hidden_size`

`create_dflash2_executor()` は加えて次を要求する。

- `num_target_layer_ids == kDFlash2TargetTaps`（5）
- `conv_kernel_size == kDFlash2ConvKernelSize`（2）

### weight

- BF16 safetensors、Shelf 量子化 bundle、HF 形式の EXL3-INT8 mul24 または
  block128 FP8 を受け付ける。`--dflash2-model-dir` の metadata または
  `quantization_config` により形式を判定する。
- `fc.weight` と各層の Q/K/V/O・gate/up/down projection は BF16、EXL3-INT8、
  FP8 を層ごとに混在できる。dynamic convolution、candidate selector、norm は
  BF16 を要求する。
- tensor 名と shape は `dflash2_expected_tensors(config)` が定義する。欠落、余分な
  tensor、shape または dtype の不一致はロード時に拒否する。EXL3 は整数 bit rate の
  mul24、FP8 は 128×128 block scale の E4M3 を要求する。
- HF 形式の EXL3 では `.trellis`、`.suh`、`.svh`、`.mul1` を、FP8 では
  `.weight` と `.weight_scale_inv` を既存の HF adapter で読む。Shelf bundle の
  `architecture` は `dflash2_draft` である。
- 量子化 linear は共通の `shelf::drafter::quantized_linear` で実行し、出力を BF16 に
  戻す。EXL3-INT8 は group-scale fused GEMM、FP8 は activation を量子化して既存の
  block128 GEMM を使う。作業領域は executor が確保して再利用する。
- embedding / lm_head は checkpoint に存在せず、target 側を再利用する。ロード後は
  `hipStreamSynchronize` する。

---

## Target bridge

### hidden taps

DFlash2 drafter は target の途中層の出力を入力に使う。tap は
`Qwen35LowerOptions::hidden_taps`（executor 側は `ExecutorConfig::target_hidden_taps`）で明示指定し、**自動推測しない**。

値の定義:

- tap する値は target graph の各層 final output、すなわち residual 加算後の値である
  （Transformers の `hidden_states[layer_id + 1]` に相当）。graph 上では
  debug name `L<layer>.output` の node の出力がこれに一致する。

不変条件:

- tap を指定しない場合、target graph の external output は 4 個のままで、buffer 確保も
  kernel 追加も行わない。出力 token は一切変わらない。
- tap を指定した場合、external output の index `4..4+n-1` に tap が追加される。
- tap buffer は `Executor` が所有する `dflash_target_hidden[k]`（`[Npad, hidden]` BF16。
  `Npad` は executor の padded token-row capacity）で、`token_hidden` とは別 buffer に
  明示的に bind する（alias させない）。
- executor の host 側 external I/O 静的 capacity は `kHostExternalCapacity = 16`。
- tap は layer の row 数分（token rows）を保持する。
- `kMaxTargetHiddenTaps = 8`。`target_hidden_tap_count` がこれを超える場合は executor
  create が error。

`test_dflash2_target_taps` が、同じ layer output を `ValueTraceSink` で捕捉し、
external tap と internal layer output の bit 一致を確認する。tap の有無で sampled token が
一致することも同時に確認する。

### target feature projection

`dflash2_project_target_features()` は次の 3 段を 1 回の呼び出しで enqueue する。

1. `launch_dflash2_feature_concat`: 5 本の tap `[rows, 5120]` を feature 軸へ連結し
   `target_concat [rows, 25600]` を作る。tap の順序は `target_layer_ids` の順。
2. `dflash2_linear`: `fc.weight [5120, 25600]` を bf16 GEMM で掛けて
   `target_fc [rows, 5120]`。
3. `launch_dflash2_rmsnorm_direct_bf16`: `hidden_norm.weight` で
   `target_feature [rows, 5120]`。

stdio serving の TP=1 では、block size が 8 かつ sliding window が 256 以上の
場合、prefill の feature projection と context append を最大 256 行ずつ実行する。
その他の条件、および `SHELF_DFLASH2_PREFILL_ROWS=8` の場合は従来の容量を使う。
バッチ用 executor が複数 request 分を持つ場合、従来の容量は block size より大きい。
FC の 8 行経路は `ExactRowsDot2`、256 行経路は `WmmaWide` を選ぶため、K/V は
bit 一致せず 2 ULP を超える差も生じる。原因と測定条件は
[行数による K/V 差の調査](../research/20261109/dflash2_prefill_rows_kv_diff.md)を参照。

RMSNorm は **DIRECT**（`ONE_PLUS` は使わない）、`group_size = hidden_size`、
`eps = rms_norm_eps`。公式実装の `Qwen3RMSNorm` と同じく、norm を bf16 に落としてから
weight を掛ける。

### target embedding / noise

- `DFlash2ExecutorConfig::target_embed_tokens` で target の `embed_tokens` を借りる。
  所有も複製もしない。`DFlash2Executor` は同 pointer を保持し shutdown で nullptr にする。
- `create_dflash2_executor()` 時点で `embed_tokens` の shape が
  `[vocab_size, hidden_size]` であることを検証する。
- `noise_token_ids [block_size] I32` と `noise_embedding [block_size, hidden] BF16` を
  create 時に確保する（`target_embed_tokens` 指定時のみ）。
- `launch_dflash2_noise_token_ids()` が row0=anchor、row1 以降に `mask_token_id` を書く。
  anchor は kernel argument で渡す（H2D memcpy を毎回しない）。
- `dflash2_prepare_noise_embedding()` が token id 生成と embedding lookup を行う。
  BF16 encoding を既存 `launch_embedding_bf16` で処理する。新しい embedding kernel は
  作らない。BF16 以外の encoding は error。

### target lm_head 共有

- DFlash2 専用の lm_head を作らない。target の `MatrixWeight` を
  `DFlash2ExecutorConfig::target_lm_head` で借りる。`DFlash2Executor` は参照のみで
  複製も所有もしない。
- `lm_head` の encoding に応じて計算する。
  - BF16: `select_bf16_gemm_config` による bf16 GEMM を F32 で受け取る。
  - FP8 K128N128: `launch_activation_quantize_fp8_finegrained`（活性は行ごとの
    `peak/448` scale、累算 F32）で activation を e4m3 へ量子化し、block-scaled
    W8A8 WMMA GEMM で F32 logits を作る。
  - EXL3: mul1 は W16A16 kernel、mul24 は activation の回転・INT8 量子化、
    INT8 GEMM、出力回転の順に実行する。両形式は codebook と trellis の配置が異なる。
- proposal logits は **F32** で出す（bf16 へ落とさない。top-16 ordering が proposal に
  直結するため）。
- `create_dflash2_spec_decoder()` が target との shape 一致
  （`hidden_size` / `vocab_size` / `lm_head` shape）と target 層数 > 最大 tap を検証する。

### ownership

- target tap buffer は `Executor` が所有し、`create_model_executor` で arena から確保、
  `executor_shutdown` で解放する。
- DFlash2 drafter の workspace は `DFlash2Executor` が所有し、
  `create_dflash2_executor` で確保、`dflash2_executor_shutdown` で解放する
  （arena ownership のため shutdown は pointer を nullptr に戻すだけ）。
- DFlash2 の persistent sequence state は `DFlash2ContextState` が所有し、
  `create_dflash2_context_state` で確保、`dflash2_context_shutdown` で解放する。
- `DFlash2ExecutorConfig::target_lm_head` / `target_embed_tokens` は借りるだけで所有しない。
- serving 経路（`ContinuousBatcher`）の per-request state は **`DFlash2SessionPool` が所有する**。
  pool は起動時に `max_concurrent_requests` 分のスロットを確保し、各スロットは
  `DFlash2ContextState` と `DFlash2SpecWorkspace`（verify staging、hipEvent 4 本、
  rank ごとの GDN snapshot / history / compact buffer / commit descriptor）を持つ。
  リクエストは `acquire` / `release` でスロットを借り、`release` は
  `dflash2_context_reset` までを行う。`GpuArena` は bump allocator で返却できないため、
  **リクエスト処理中に arena から DFlash2 用の領域を追加確保してはならない**。
  `DFlash2BatcherBinding::sessions` は必須で、容量が `max_scheduled_requests` 未満なら
  束ねられない。
- `create_dflash2_spec_decoder` の `workspace` 引数が非 null なら decoder はその
  workspace を**借りる**（`owns_workspace = false`、`dflash2_spec_decoder_shutdown` は
  hipEvent を破棄しない）。null なら従来どおり arena から確保して所有する
  （`shelf-compute` の単発経路と unit test がこの経路）。

---

## Drafter execution

### layer execution

`dflash2_forward_layer_stateless()` は次の順で enqueue する。順序を変えない。

1. layer input norm: `input_layernorm.weight` で `layer_input_norm` を作る。
2. `attention_prepared = conv_prepare(layer_input_norm, attention_conv)`
   （kernel_projection と phase 0 conv。phase 1 の dynamic を保持）。
3. `q_raw = q_proj(attention_prepared)`
4. `k_ctx_raw = k_proj(target_feature)`, `v_ctx = v_proj(target_feature)`
5. `k_noise_raw = k_proj(attention_prepared)`, `v_noise = v_proj(attention_prepared)`
6. `q_norm` / `k_ctx_norm` / `k_noise_norm`（RMSNorm head_dim 単位、V は norm しない）
7. RoPE: Q は block positions、K_ctx は context positions、K_noise は block positions
8. DFlash2 attention（context + noise の 2 segment）
9. `attention_o = o_proj(attention_context)`
10. `attention_finished = conv_finish(attention_o, attention_conv)`
11. `attention_residual = hidden_in + attention_finished`
12. `post_attention_norm = post_attention_layernorm(attention_residual)`
13. `mlp_prepared = conv_prepare(post_attention_norm, mlp_conv)`（`conv_dynamic` を上書き）
14. MLP: gate/up、SwiGLU、down
15. `mlp_finished = conv_finish(down, mlp_conv)`
16. `output = attention_residual + mlp_finished`

`conv_dynamic` は attention prepare → attention finish → MLP prepare → MLP finish の順で
使い回す単一 buffer である。attention finish より前に MLP prepare を enqueue してはならない
（single stream の順序で保証する）。

`dflash2_forward_backbone_stateless()` / `dflash2_forward_backbone_cached()` は
`num_hidden_layers` を 0→N-1 の順に直列 enqueue し、最後に final RMSNorm（`norm.weight`）を
掛けて caller の output pointer へ直接書く。

- 入力は `noise_embedding [block_rows, hidden]` と `target_feature [context_rows, hidden]`。
  target feature projection は **呼び出し側で 1 回だけ**実行し、全 layer で同じ pointer を
  共有する。
- layer 間の relay は **ping-pong buffer の pointer 渡し**のみ。D2D memcpy を挟まない。
  `backbone_hidden_a`（偶数 layer）/ `backbone_hidden_b`（奇数 layer）を使う。
- intermediate tensor へ書いた後に copy しない。final norm は output へ直接書く。
- 引数の contract: `block_rows <= min(max_rows, block_size)`、
  `context_rows <= max_context_rows`、`context_position_start + context_rows ==
  block_position_start`（stateless のみ。cached は `block_position_start ==
  context.next_position`）。
- stateless path は同一入力・同一 position の複数回実行が bit 一致する
  （conv_dynamic の残留、ping-pong lifetime、layer 間 state leakage が無い）。

host 側の layer loop は enqueue のみで、GPU→CPU の read も途中判断も無い。禁止するのは
「GPU → CPU へ token → CPU 判断 → 次 GPU」であり、noise token id や anchor は kernel
argument として渡す。

### GroupedDynamicCausalConv

- `kernel_projection` の出力 `[rows, 1280]` は `[rows, 2 (phase), 2 (offset), 320 (group)]`
  として解釈する。group が最内。
- `base_kernel [2, 2, 5120]` は `[phase][offset][feature]`。
- 出力の各要素は
  `Σ_offset (base[phase][offset][f] + dynamic[row][phase][offset][group]) * input[row - offset][f]`
  （`row - offset < 0` の項は 0）。
- dynamic は **`row` の行**を使う（`row - offset` ではない）。block-local causal conv であり、
  persistent conv state は持たない。
- `prepare` は kernel_projection を実行し、phase 0 の conv 結果を返し、phase 1 の dynamic を
  executor の `conv_dynamic` に保持する。
- `finish` は kernel_projection を再実行せず、保持した `conv_dynamic` と phase 1 の base を
  使う。`prepare` → `finish` の順で呼ぶこと。

### attention

- `is_causal = false`、mask 条件は `abs(query_position - key_position) < sliding_window`
  （strict）。2047 は visible、2048 は masked。
- block 内の future row も visible。通常の causal attention を流用してはならない。
- Q の position は block、K_ctx の position は context、K_noise の position は block。
- GQA: `kv_head = q_head / (q_heads / kv_heads)`。27B checkpoint は 32 / 8 = 4。
- 実装は 3 系統:
  - `launch_dflash2_attention_bf16`: stateless contiguous。context と noise を別 segment で
    受け取り、物理 concat buffer を作らない。1 workgroup = 1 query row × 1 q head。
    correctness oracle として残す。
  - `launch_dflash2_attention_ring_bf16`: persistent ring から slot 計算で context を読む
    参照実装。mask / GQA / softmax は stateless と同一。
  - `launch_dflash2_attention_ring_gqa_bf16`: production。1 WG = 1 block row × 1 KV head
    で 4 q head を処理し、K/V を LDS tile へ uint4 coalesced load して 4 head で共有する。
    数学は参照実装と **bit-exact**。
- GQA kernel の制限（不成立時は `hipErrorNotSupported`）: `q_heads == 4 * kv_heads`、
  `head_dim == 128`、strides の 8 要素倍数、dynamic LDS ≤ 60KB。executor は既定で GQA を
  使い、NotSupported のとき ring 参照実装へフォールバックする。
  `SHELF_DFLASH2_ATTENTION_REFERENCE=1` で参照実装を固定できる。

### final normalization / dtype contract

公式実装は bf16 tensor 同士の演算で各 op ごとに丸める。Shelf は次の 3 か所で同じ
丸め列を再現する（`round_to_bf16_f32`）。

- GroupedDynamicCausalConv: `round(kernel * values)` → `round(out + ...)` → `addcmul`
- RoPE: cos/sin を bf16 に丸め、積と和も bf16 に丸める
- SwiGLU: `silu(gate)` を bf16 に丸めてから `up` と掛け、積も bf16 に丸める

RMSNorm は公式の `weight * norm.to(bf16)` と同じく、norm を bf16 に落としてから weight を
掛ける（`launch_dflash2_rmsnorm_direct_bf16`）。公式 forward の中間 tensor はすべて bf16、
attention_mask は bool である。

RoPE の角度は `double(position) * inv_freq` で計算し、`inv_freq` も double の `pow` で求める。
`float(position)` では 262,144 以上の位置で角度の量子化誤差が無視できなくなるためである。
cos/sin は double で求めてから f32 へ丸め、以後の積と和は従来どおり bf16 に丸める。
Drafter の RoPE は target の `--rope-factor` に追従せず、checkpoint の `rope_theta` と
`rope_type=default` だけを使う。target が YaRN でも speculative verify は target の logits で
判定するため、drafter の位置外挿が弱くても生成結果の正しさは変わらない。

### primitive dispatch

DFlash2 executor は layer 内の各 logical primitive をすべて独立した kernel launch として
実行する。対象は layer input/post-attention RMSNorm、Q/K の RMSNorm と RoPE、
q/k/v projection GEMM、attention、Gate GEMM、Up GEMM、SwiGLU、Down GEMM、
grouped dynamic conv と residual である。
1 logical primitive = 1 kernel launch であり、primitive 間の結合は行わない。

### drafter linear kernel（bf16）

drafter の bf16 linear（q/k/v/o/gate/up/down/fc）は
`Bf16GemmConfigId::ExactRowsDot2` を選ぶ。1 wave が 1 出力特徴を担当し、kRows 行分を
まとめて計算する点は `ExactRows` と同じで、内側の 8 要素を `v_dot2c_f32_bf16`
（bf16 pair dot）で畳む点だけが異なる。

- 決定論的・非ロスで、host 参照に対して tolerance 内（`test_gemm_bf16_exact_rows` が
  tolerance と determinism を検証する）。
- **byte-exact ではない**。drafter の proposal は draft 品質にのみ影響し、出力の正しさは
  target verify が担保する。
- target verify と decode は同じ行数なら同じ config（`ExactRows`）を選ぶ。dot2 経路は
  drafter からのみ opt-in する（`allow_dot2_rows`）。
- kernel 選択は execution role に依存せず行数だけで決まるため、単一 request（rows=1）と
  batch（rows=2..16）の違いは kernel の差ではなく行数の差だけである。

### hot-path contract

`dflash2_project_target_features` / `dflash2_grouped_conv_prepare` /
`dflash2_grouped_conv_finish` / `dflash2_forward_layer_*` / `dflash2_forward_backbone_*` /
`dflash2_select_draft_tokens` / `dflash2_propose_cached` / `dflash2_propose_batch` /
`dflash2_append_target_taps` / `dflash2_prepare_noise_embedding` は非同期 enqueue API で
ある。以下を含んではならない。

- `hipStreamSynchronize` / `hipDeviceSynchronize`
- `hipMemcpyDeviceToHost` / `hipMemcpyHostToDevice`（例外は `dflash2_propose_batch` の
  device descriptor 配列の async H2D のみ。同期はしない）
- ホットパスでの allocation（arena / hipMalloc）

buffer 確保は `create_dflash2_executor` と `create_dflash2_spec_batch_context` で行う。
host へ戻るのは `dflash2_spec_step` / `dflash2_spec_step_batch` の decision の D2H と、
テスト側の比較・計測のときだけである。batch 経路の decision D2H は 1 round に 1 回である。

### no-D2D relay contract

kernel の出力 buffer を次 kernel の入力 pointer として直接渡す。D2D memcpy による relay を
行わない。context append も wrap 対応の scatter kernel で書き、D2D memcpy を使わない。

---

## Persistent context state

`DFlash2ContextState` は target が確定した context の K/V を保持する sequence state である。

### ownership

- `DFlash2ContextState` は `DFlash2Executor` から分離した sequence state である。
  executor の scratch ではない。
- `create_dflash2_context_state()` が arena から ring を確保し、
  `dflash2_context_shutdown()` で解放する。destructor には依存しない。
- `dflash2_context_reset(state, next_position)` は GPU buffer を memset しない。
  `length = 0`、`next_position = requested` を設定するだけである。stale data は
  `length` 外なので読まない。

### layout / memory

| buffer | shape | dtype |
| --- | --- | --- |
| `k_ring` | `[num_layers, capacity, kv_features]` | BF16 |
| `v_ring` | `[num_layers, capacity, kv_features]` | BF16 |

ring は target KV pool と独立して所有し、target が Octave の場合も BF16 のままである。
target の Octave descriptor、page table、量子化 table を drafter へ伝播しない。

- production では `capacity == config.sliding_window`（2048）を要求する。
- `kv_features = attention_kv_rows() = num_key_value_heads * head_dim`（27B は 8 × 128 = 1024）。
- 27B の実 allocation は `2 * 5 * 2048 * 1024 * 2 = 41,943,040 bytes`（40 MiB / sequence）。

### ring が保持する値

- K は raw K を保持しない。`target_feature -> k_proj -> k_norm(RMSNorm head_dim) ->
  RoPE(absolute position)` の結果を保存する。
- V は `target_feature -> v_proj` の結果を保存する。
- proposal ごとに k_norm / RoPE をやり直さない。

### physical mapping

- write index を別管理しない。physical slot は `absolute_position % capacity`。
- logical row `i` の absolute position は `context_start + i`、
  `context_start = next_position - length`。
- この `slot = position % capacity` を append kernel と ring attention kernel で共通に使う。

### next_position 契約

- `state.next_position` は target が次に consume する absolute position。
- `context.length <= capacity`、`context_start = next_position - length`。
- cached backbone は `block_position_start == state.next_position` を要求する
  （block row 0 が anchor の position と一致する）。

### append 契約

`dflash2_context_append(executor, state, target_feature, rows, position_start, stream)`:

- input は `target_feature [rows, hidden]`、`position_start == state.next_position`。
- layer ごとに `k_proj -> k_norm -> RoPE(effective_position)` と `v_proj` を実行し、
  `k_ctx_raw` / `k_ctx_norm` / `k_ctx_rope` / `v_ctx` scratch を一時 buffer として再利用する。
  これらの scratch は cached proposal では使わない。
- `rows > capacity` のときは最後の `capacity` row だけ生成する。
  `skip = max(0, rows - capacity)`、`effective_position = position_start + skip`。
  metadata の `next_position` は `position_start + rows` まで進める。
- append 後: `length = min(capacity, old_length + rows)`、`next_position += rows`。
- host metadata は kernel を enqueue した後に更新する。GPU completion 待ちはしない
  （`hipStreamSynchronize` を呼ばない）。
- `effective_rows > max_context_rows` は error。

### commit-only semantics

- ring に入るのは **target が consume して commit した row** だけである。
- noise / MASK block、DFlash draft hidden、reject された target row は入れない。
- DFlash2 側に rollback API は無い。accepted row だけを append する。
- prefill / target embedding からの自動構築はしない。呼び出し側が明示的に append する。

### cached attention / backbone

`dflash2_forward_layer_cached()` は stateless layer から context K/V 生成 4 段
(`k_ctx_proj` / `v_ctx_proj` / `k_ctx_norm` / `k_ctx_rope`) を除いた経路である。

- noise 側は stateless と同一。
- context は ring を直接読む。
- attention は ring 用 kernel（既定 GQA、フォールバック ring 参照）を使う。
- attention 以降の `o_proj -> residual -> post norm -> MLP -> residual` は stateless と
  共通の内部 helper を使う。

### proposal read-only

- `dflash2_forward_backbone_cached()` / `dflash2_forward_and_select_cached()` /
  `dflash2_propose_cached()` は context state を read-only として扱う。
  `length` / `next_position` / ring K/V を変更しない。
- 同一 context state に対する proposal は何回呼んでも同じ結果になる。
- noise K/V を ring へ commit する設計にはしない。

---

## Proposal pipeline

### candidate selector API

公開 API は次の 4 つである。

- `dflash2_compute_target_logits()`: `final_hidden` から target lm_head の logits を計算する。
- `dflash2_run_candidate_selector()`: hidden_proj / top-k から draft token 列を確定する。
- `dflash2_select_draft_tokens()`: backbone output から draft token 列までを一括で行う。
- `dflash2_forward_and_select_stateless()`: stateless backbone + select の便宜 API。

`dflash2_select_draft_tokens()` の流れ:

```
final_hidden の anchor row を除く draft row
  -> selector hidden_projection (exact bf16 gemm)
  -> proposal head
  -> top-16
  -> CandidateSelector
  -> draft_tokens[draft_rows]
```

すべて device 上で実行する。D2H / H2D / D2D relay / allocation は無い。
`lm_head` の入力は `final_hidden + hidden_size`（row 1 以降）で、anchor row は渡さない。

### proposal head

- `lm_head_logits()` が target `lm_head` の encoding（BF16 / FP8 K128N128 / EXL3）に
  応じた GEMM で `proposal_logits [draft_rows, vocab_size]` F32 を作る。

### top-16

`launch_dflash2_topk_f32()` は 2 stage 構成である。

1. 1 row = 1 workgroup。各 thread が自分の shard を走査して local top-16 を保持し、
   256 thread 分（256×16 候補）を scratch へ書く。
2. 1 row = 1 thread が 256×16 候補から top-k を選び、降順で出力する。

ordering は **logit 降順、同値なら token id 昇順**。f32 の同値判定は bit 等価で行う
（丸め誤差による擬似 tie は許容しない）。full sort はしない。

production は `launch_dflash2_topk_f32_optimized`（hierarchical reduction）。

- partition stage: grid = `rows * partitions`。1 WG が vocab partition を担当し、
  thread local top16 から WG 内 cooperative selection で partition top16 を求める。
- merge stage: grid = `rows`。1 WG が `partitions * 16` candidate から cooperative
  selection で top16 を求める。
- scratch は `[rows, kDflash2TopKScratchPerRow]`（`kDflash2TopKThreads *
  kDflash2TopKMaxK = 256 * 16 = 4096`）を再利用する。
- plan は `DFlash2TopKPlan`。production default は partitions=32 / threads=256。
- `dflash2_select_draft_tokens` は optimized を呼ぶ。`top_k != 16` または
  `SHELF_DFLASH2_TOPK_REFERENCE=1` のときは reference に切り替える。
- 契約は不変: logit descending、exact tie は token id ascending、NaN は既存 comparator を
  維持、出力 IDs は exact。

### CandidateSelector

```
pred_0 = anchor_token
for r in 0..draft_rows-1:
    score_j = top16_logit[r][j]
              + dot(predecessor_codebook[pred_r] * hidden_proj[r],
                    successor_codebook[top16_id[r][j]])
    best_j  = argmax_j score_j     # 同値なら candidate token id 小
    draft_token[r] = top16_id[r][best_j]
    pred_{r+1} = draft_token[r]
```

`hidden_proj[r] = hidden_projection(final_hidden[r + 1])` は selector の前に draft row 分を
一括で計算する（position loop 内で再計算しない）。

累算は F32。codebook と hidden_proj は bf16 load で、積は `pred * hidden * successor` を
F32 で足し込む。

selector は **1 workgroup で draft 列全体を確定する**。position ごとに kernel を分けたり、
token を host へ戻して次の position を決めたりしない。`draft_rows` は引数で受け、
固定 7 回 loop はしない。

### block_rows と draft_rows

`draft_rows = block_rows - 1`。

| block_rows | draft_rows |
| --- | --- |
| 2 | 1 |
| 4 | 3 |
| 8 | 7 |

## Speculative transaction

### prefill

`dflash2_spec_prefill(decoder, prompt_tokens, prompt_count, prefix_tokens = 0)`:

- `prefix_tokens` は prefix cache から復元済みの先頭 token 数である。prefill は
  `offset = prefix_tokens` から再開する。`0 < prefix_tokens < prompt_count` の場合のみ
  有効で、`prefix_tokens >= prompt_count` は error である
  （[prefix_cache.md](prefix_cache.md)）。
- prompt を `target->config.max_batch_rows` 単位で chunk 実行する。
- intermediate chunk: `compute_logits=false`, `sample=false`。
- final chunk: `compute_logits=true`, `sample=true`, `num_output_rows=1`。
- 各 chunk 実行直後に target tap を ring へ append する。
- final chunk の sampled token を pending token（anchor）として返す。
- prefill 後: `sequence.position == prompt_count`、`context.next_position == prompt_count`、
  `context.length == min(prompt_count, capacity)`。
- anchor の absolute position は `prompt_count` であり、まだ ring へ commit しない。

### step

`dflash2_spec_step(decoder, pending_token, remaining_tokens)`:

1. `K` を clamp する: `min(config.num_drafts, remaining_tokens - 1, room - 1)`。
   `room = sequence.max_seq_len - position`。`remaining <= 1` または `room < 2` なら `K = 0`。
2. `K == 0` のときは単一 target decode（`execution_class = DECODE`,
   `speculative_verify = false`, `num_output_rows = 1`）を行い、
   sampled token を emit して target tap 1 row を ring へ append する。
3. `K > 0` のとき:
   - DFlash proposal（`dflash2_propose_cached`）を行い、draft 列を得る。
   - target verify: `[anchor, composite]` の `1 + K` 行を
     `execution_class = SPEC_VERIFY`（行数 > 1 のとき）, `speculative_verify = true`,
     `num_output_rows = 1 + K`, `prefix_tokens = position` で実行する。
     GEMM の kernel は行数だけで決まり（`2 <= rows <= 16` は `ExactRows`）、
     GDN recurrence は role 固定の serial variant である。
   - target samples から `spec_greedy_accept` で longest-prefix acceptance を行う。

### batched step

複数 request の target verify を 1 つの `ScheduledBatch` に統合する。serve mode の
既定経路であり、scheduler が in-flight な request をまとめて駆動する。

`create_dflash2_spec_batch_context(bindings, max_rows)` は binding ごとに共有の
`GdnSpecHistory`（conv のみ）と compact log を `max_rows` 行分確保する。`max_rows` は
1 回の batch verify の行合計 `Σ(Kᵢ+1)` の上限であり、`kDFlash2SpecMaxVerifyRows`（512）
以下でなければならない。serve は `max_concurrent_requests × block_size` を渡す。
あわせて leader rank の arena から device token buffer を確保する。

| buffer | bytes | 用途 |
| --- | ---: | --- |
| `verify_token_ids_device` | `max_rows * 4` | `[anchor, draft...]` を slot 順に連結 |
| `decision_staging_device` | `2 * max_rows * 4` | 前半 verify token / 後半 sampled |

release は `shutdown_dflash2_spec_batch_context()`。

`dflash2_spec_step_batch(context, slots)` は slots を 1 round 実行する:

1. slot ごとに `K` を clamp する（`dflash2_spec_step` と同じ規則）。
2. `K == 0` の slot は batch に含めず、`run_single_target`（単一 target decode）へ
   フォールバックし、verify 行数に数えない。
3. `K > 0` の slot を連結する。draft は全 slot 分を 1 回の `dflash2_propose_batch` で
   生成し、token 列は slot 順に `[anchor, draft₀..draft_{K-1}]`、row 範囲は
   `row_begin` から `K+1` 行、`prefix_tokens = position` とする。
4. batch 全体を `execution_class = SPEC_VERIFY`, `num_verify_requests = slot 数`,
   `speculative_verify = true`, `num_output_rows = 合計行数` の 1 回の `execute_batch` で
   実行する。実行は先頭 slot の target を使い、複数 rank は `tp_hook->on_execute` で
   展開する。
5. slot ごとに `spec_greedy_accept` と commit（full / partial）を `row_begin` を基点に
   共有 history / compact log で行う。手順は `dflash2_spec_step` と同一である。partial
   reject の base 復元には decoder 個別の snapshot を使う。
6. slot ごとに `sample_index += K+1`。

出力は入力 slot と同順の `DFlash2SpecIterationOutput` 列で、`total_rows` は batch の
実 verify 行数、`rounds` は常に 1 である。`DFlash2SpecBatchOutput` は timing として
`round_ms` / `draft_gpu_ms`（batched proposal 全体の GPU 時間）/ `verify_gpu_ms`
（batch verify の GPU 時間）を持つ。timing は参加 slot の decoder が `timing` を
持つときだけ記録する。

契約:

- verify 行合計 `Σ(Kᵢ+1)` は `context.max_rows` と `kDFlash2SpecMaxVerifyRows` の
  両方以下である。超えると error になり、batch は実行しない。
- GDN history / compact log は batch 専用に確保する。decoder 個別の history /
  compact log は `dflash2_spec_step` 用であり、両者は独立である。
- 全 slot の decoder が同一の drafter executor と同一 stream を共有することを要求する。
  共有でない場合は `unsupported` を返し、batch は実行しない（silent fallback しない）。

### batched proposal

`dflash2_propose_batch(executor, requests, output_drafts, request_count, stream)` は
複数 request の proposal を 1 回の drafter forward で生成する。

- device descriptor（`DFlash2RequestDescriptor`）を 1 round に 1 回 async H2D する。
  内容は `ring_k` / `ring_v` / `context_rows` / `context_start_position` /
  `block_position_start` / `block_rows` / `row_begin` / `num_drafts` / `anchor_token` /
  `output_drafts` である。
- scratch の row 軸は executor の `max_rows`。request `r` の行は
  `row_begin` から `block_rows` 行を占める packed layout とする。
- ring pointer は descriptor が layer 0 を指し、kernel が
  `ring_layer_offset` を加算する（層ごとの descriptor 再アップロードを行わない）。
- RoPE の position は `block_position_start + row_in_block`、conv の causal 参照は
  `row_has_prev`（request の先頭行で 0）で request 境界を越えさせない。
- `CandidateSelector` は 1 request = 1 workgroup とし、`row_begin + 1` 行を基点にする。
- `request_count == 1` は `dflash2_propose_cached`（単一 request 経路）へ委譲する。
- `output_drafts` は各 request の draft 列の書き込み先である。batch 経路は
  `verify_token_ids_device + row_begin + 1` を渡し、proposal の結果を D2D relay せずに
  verify の入力にする。
- proposal の kernel は単一 request 経路と共通の device body を持ち、同一入力・同一
  config で同一の draft 列になる。

### device token bridge（batch）

`SHELF_DFLASH2_DEVICE_TOKEN_BRIDGE`（既定 1）が有効で、全 slot の decoder が
device bridge を許可しているとき、batch 経路は draft を host へ戻さない。

1. slot ごとに anchor を `verify_token_ids_device[row_begin]` へ H2D する
   （descriptor の `anchor_token` は selector 用であり、verify batch は token buffer を
   読むため別途必要である）。draft は `dflash2_propose_batch` が直接
   `verify_token_ids_device[row_begin+1]` へ書く。
2. verify batch は `token_ids = verify_token_ids_device`、
   `token_ids_location = Device` で構築する。CPU は draft token の値を知らない。
3. verify 完了後、verify token buffer と sampled tokens を `decision_staging_device`
   の前半 / 後半へ D2D で relay し、round に 1 回だけ `hipStreamSynchronize` する。
4. decision staging を 1 回だけ D2H し、`spec_greedy_accept` を CPU で実行する。

bridge が無効なときは batch の draft を `verify_token_ids_device` から
round に 1 回 D2H し、Host 由来の token 列で batch を構築する（legacy Host bridge）。

### accept / reject / commit

`spec_greedy_accept(drafts, K, sampled, bonus_token_enabled=true)`:

- draft と target sampled token が先頭から何 row 一致するかを数える（longest prefix）。
- `accepted < K` のとき correction token（最初に reject した row の target token）を返す。
- `accepted == K` のとき bonus token（block 末尾で target が sample した次 token）を返す。

commit は target state が確定してから行う（commit-only）。DFlash ring は proposal / verify /
acceptance の間は変更しない。

- full accept（`accepted == K`）: target state は既に正しい。verify の target tap `K+1` 行を
  `position_start = position` で ring へ append し、余剰 KV page を
  `rollback_sequence_append(ceil(sequence.position / page_tokens))` で trim する。
  pending は bonus token。
- partial（`accepted < K`）: verify 前の conv snapshot を restore し、
  `restore_gdn_spec_conv(pool, slot, history, accepted)` で conv を live へ戻した後、
  `commit_gdn_compact_log(pool, slot, log, requests, accepted + 1, key_heads)` で
  accept prefix の recurrent を再構成する。`sequence.position = position + accepted + 1` に
  設定し、verify の target tap rows `0..accepted` を ring へ append し、KV page を trim する。
  pending は correction token。target rerun は行わない。

batched step（serve）の verify は live の recurrent state を書き換えない
（`GdnCompactLogDeviceView::defer_state_store`、verify kernel は最終 state の store を
省略する）。live slot は base state のままなので、recurrent の snapshot / restore は
無く、full accept でも `commit_gdn_compact_log` で `K + 1` 行を replay して確定する。
conv は shift register のため、従来どおり conv snapshot（約 2.8 MiB）から戻す。
単一 request の `dflash2_spec_step` は従来どおり verify が state を書き込む。

### GDN state history

partial reject 時に full target rerun を不要にするため、verify の各 row 直後の GDN state を
その場で記録する。**compact log path のみ**である。

- **recurrent** は `{a_i, k_i, δ_i}` の compact log を保存し、accept prefix を逐次 replay で
  再構成する。
- **conv** は `GdnSpecHistory` の conv のみを保持する（shift register であり、受け入れ済み
  token から再計算できないため）。`create_gdn_spec_history` は `with_recurrent=false` で
  呼び、rec buffer を作らない。
- partial reject の base 復元用に conv / recurrent の slot snapshot（各 1 slot 分）を
  decoder に取る。

decoder あたりの GDN 消費（27B geometry、K=7）:

| path | 計 |
| --- | --- |
| compact | conv history 19.7 MiB + snapshot 146.8 MiB + compact log 18.1 MiB = **184.6 MiB** |

- `GdnSpecHistory` は decoder 専用の single-sequence checkpoint で、sequence slot の
  GDN state を row 数分保持する。`GdnStatePool` 自体へは埋め込まない。
  - `conv`: `[row][num_gdn_states][conv_history][conv_history_stride]` BF16
    （live slot と同一 layout）
- conv history rows = `config.num_drafts`。capture_rows は round の実 draft 数（`K`）。
- compact log rows = `config.num_drafts + 1`（anchor + draft の実使用行）。
- allocation は create 時の 1 回のみ。hot path allocation は行わない。
- history / compact log は `DFlash2SpecDecoder` ごとに確保する（単一 request 用）。
  serve の session pool は確保しない（`step_buffers=false`）。batched step は
  `DFlash2SpecBatchContext` の共有 history / compact log だけを使う。
- serve の session ごとの確保は、drafter context ring（約 40 MiB）と conv snapshot
  （約 2.8 MiB）である。recurrent snapshot（約 72 MiB）は全 session で 1 つを共有する。
  共有 buffer を使うのは直列に実行される prefix cache の shadow prefill だけである。
- arena は bump のみで個別 free が無いため、同時 N session はこの値を N 倍で確保する。
- create 時に `history bytes > kDFlash2SpecHistoryBytesMax`（2.25 GiB）なら
  decoder create を FAIL する（FP16 化等で逃げない）。27B geometry
  （146.8 MiB/row）では rows ≤ 15 が上限である。
- `DFlash2SpecDecoderConfig::num_drafts` は `1..max_draft_tokens()` でなければ error。
- `kDFlash2SpecMaxVerifyRows` は compact log 行数とは独立で、batch verify 行合計
  `Σ(Kᵢ+1)` の上限に使う。per-request の draft 上限は
  `dflash2::kMaxBlockSize - 1`（`max_draft_tokens()` の compile-time 上限）。

batch context（`DFlash2SpecBatchContext`）は binding ごとに conv history（`max_rows`
行）と compact log（`max_rows` 行 × `{δ, k, a}`）を確保する。27B geometry では conv
約 2.8 MiB/行、compact 約 2.3 MiB/行である。`max_rows` は 1 round の verify 行合計
`Σ(Kᵢ+1)` に一致させる（serve は `max_concurrent_requests × block_size`）。これは
decoder 個別の history / snapshot / compact log とは別に確保する。

capture:

- `execute_batch` に `ExecuteBatchOptions`（`GdnSpecHistoryDeviceView`）を渡す。
  scheduler contract へは漏らさない。history の書き込みは `role == Verify` の実行で
  `GdnSpecHistoryDeviceView` が渡され `capture_rows != 0` のときに行い、mixed batch では
  `capture_verify_only` で verify 行だけを mirror する。
- GDN recurrence は `row_override` 番目の writeback で live state と同一の local offset へ
  mirror store する。GDN conv は row 更新直後に `s0/s1/s2` を BF16 で mirror store する。
  いずれも kernel 内で書くため per-row / per-layer の D2D copy を作らない。

restore:

- `restore_gdn_spec_conv(pool, slot, history, history_row)` は conv を 1 回の D2D copy で
  live へ戻す。recurrent は compact log から逐次 replay で再構成する。
- `restore_gdn_spec_history(pool, slot, history, history_row)` は conv / recurrent を
  各 1 回の D2D copy で live へ戻す。

target verify の GDN recurrence 本体:

- verify（`role == Verify`）は `WmmaVerify` を固定選択する。geometry 非対応時のみ `F32`。
- `WmmaVerify` は 1 回の launch で合計 32 行（M32）までを処理する。
  最大 4 request × 8 行（anchor + drafts）を同時に verify できる。
  launcher は pure verify launch で `actual_rows > 32` または `num_requests > 4` を拒否する。
- `max_concurrent_requests 4` / DFlash2 `block_size 8` の server E2E で、
  `SHELF_BATCH_STEP requests=4 tokens=32` の verify batch を確認済みである。
- request ごとに 1 block が割り当てられ、その request の verify 行を処理する。
  行ごとに `recurrent_history` と compact log を保存し、`row_parent` が示す
  分岐行は history から親 state を復元する。
- `WmmaVerify` は lossy / exact の両 decomposition を持ち、切替は
  `SHELF_GDN_RECURRENCE_EXACT` / `SHELF_GDN_RECURRENCE_LOSSY` に従う。
- decode / prefill は `Wmma` / `F32` の通常選択のままである。
- batch に verify と decode/prefill が混在する場合、executor 内部で
  verify region（`execution_class == SPEC_VERIFY`）とそれ以外へ分け、
  `WmmaVerify` と `Wmma` を別々に launch する。verify request 数の上限は
  M32 の範囲（最大 4 request・合計 32 行）で決まり、旧来の
  「mixed batch は verify 1 request まで」制約は無い。

### EOS

`config.eos_tokens` が空でないとき、emit 列の最初の stop token で truncate して
`finished = true`。pending token が stop token でも `finished = true`。

### state invariant

各 step 終了時:

- `context.next_position == sequence.position`
- `context.length == min(sequence.position, capacity)`
- proposal は context を変更しない
- proposal / verify / acceptance の間、DFlash ring を変更しない

### hot path（transaction）

- per-round の host allocation をしない（verify/draft は固定長 `std::array`）。
- device bridge path では 1 round に 1 回だけ `hipStreamSynchronize` する（後述）。
- `dflash2_spec_step` は timing を `DFlash2SpecDecoder::timing` に記録する。

---

## Device-resident path

DFlash proposal を Host へ戻さず、そのまま target verify の入力にする。

### env

`SHELF_DFLASH2_DEVICE_TOKEN_BRIDGE`（既定 1）。

- `1`: device-resident bridge。
- `0`: legacy Host bridge。

### ScheduledBatch の token source

`ScheduledBatch::token_ids_location`（`TokenIdsLocation::Host` / `Device`、既定 Host）で
memory location を明示する。Device のときも pointer 非 null を要求する。Host CPU が
`token_ids` を dereference することはない。

executor の upload は Host source なら `hipMemcpyHostToDevice`、Device source なら
`hipMemcpyDeviceToDevice` を使う。対象は `executor.input_ids` と
`batch_context_storage.token_ids` の 2 箇所である。

### graph cache

`GraphCacheSlot` は replay 条件として token count / output 数 / attention plan key /
exec role / spec history / capture rows / history pointer に加えて、
Device source のときの source pointer（`token_device_src`）を保持する。Device source では
Host staging への `std::memcpy` を行わず、capture 内で device verify token buffer →
executor input buffer の D2D をそのまま capture する。

### device buffer

`create_dflash2_spec_decoder()` が arena から以下を確保する。decoder lifetime 中
アドレス不変で、shutdown では arena ownership のため free せず pointer を nullptr に戻す。

| buffer | bytes | 用途 |
| --- | ---: | --- |
| `verify_token_ids_device` | `kDFlash2SpecMaxVerifyRows * 4`（2048） | `[pending, composite...]` |
| `decision_staging_device` | `2 * kDFlash2SpecMaxVerifyRows * 4`（4096） | 前半 drafts / 後半 target samples |

### round 契約

1. `pending_token` を `verify_token_ids_device[0]` へ H2D（4 byte）。
2. `dflash2_propose_cached` を同一 stream へ enqueue。proposal output は
   `decoder.draft->proposal_tokens` に残す。
3. `proposal_tokens[0:K]` → `verify_token_ids_device[1:K+1]` の D2D を enqueue。
   ここに sync を入れない。
4. verify 用 `ScheduledBatch` を `token_ids = verify_token_ids_device`、
   `token_ids_location = Device` で構築する。CPU は draft token の値を知らないまま
   `submit_batch` まで先行できる。
5. `submit_batch` 後、proposal tokens と sampled tokens を decision staging へ D2D。
6. `hipStreamSynchronize` を round 中唯一の Host synchronization point として実行。
7. decision staging を 1 回だけ D2H し、`spec_greedy_accept` を CPU で実行する。

proposal → target verify の間に `hipStreamSynchronize` も D2H も存在しない。

legacy Host bridge では proposal 直後に `hipStreamSynchronize` + D2H を行い、target
samples を別途 D2H する。

### timing

`DFlash2SpecTiming` は `draft_gpu_ms` / `proposal_wait_ms` / `proposal_copy_ms` /
`verify_gpu_ms` / `decision_wait_ms` / `decision_copy_ms` を持つ。legacy 比較用に
`draft_ms` / `draft_d2h_ms` / `verify_ms` / `round_ms` も持つ。

`draft_d2h_ms` は legacy metric である。`DFLASH2_D2H_MS` は純粋な PCIe 転送時間では
なく、先頭の `hipStreamSynchronize` のため draft completion wait を含む。
`proposal_wait_ms` と `proposal_copy_ms` は分離して測る。

---

## Runtime configuration

### CLI（`shelf-compute`）

- `--dflash2-model-dir PATH`: DFlash2 draft model directory（指定で DFlash mode）
- `--dflash2-drafts N`: 1 round の DFlash draft 数（default 7、`block_size - 1` 以下）
- `--dflash2-stats 0|1`: speculative decode 統計行の出力（default 1）
- `--dflash2-max-sessions N`: serve で同時に投機を行う request 数（default
  `min(--max-concurrent-requests, 8)`）。session pool の容量、batch verify の行数
  （`N × block_size`）、drafter scratch の大きさがこの値で決まる。pool に空きが無い間に
  admit された request は投機せず通常 decode になる（その request は完了まで投機しない）。
  session pool の VRAM は `--max-concurrent-requests` ではなくこの値に比例する。

DFlash2 有効時の既定構成は **DFlash2 K=7**、GDN は compact log path である。

`shelf-server` は `--dflash2-model-dir` / `--dflash2-drafts` /
`--dflash2-max-sessions`（0 以外のとき）を `shelf-compute` へ転送する。

### invalid combinations（model load 前に exit 2）

- `--dump-logits`
- `--dflash2-drafts` が `block_size - 1` を超える

silent fallback は行わない。

### serve mode（`--serve-stdio`）

`--dflash2-model-dir` は `--serve-stdio` と併用できる。server backend 経由で
DFlash2 speculative decoding を有効化する唯一的な経路である。

- DFlash2 有効時の serve は通常 decode（`--dflash2-model-dir` 未指定時）と同じ
  `ContinuousBatcher` 経路を通る。request 状態は `RuntimeRequest` が所有し、
  `generated_ids` / `finish_reason` / `prompt_tokens` / `restored_tokens` /
  `cache_checkpoint_tokens` は `done` event で `RuntimeRequest` から出る。
- drafter weight と `DFlash2Executor` は `run_serve_stdio()` が serve 間保持し、
  `DFlash2SpecBatchContext` を 1 つだけ作成する。`max_rows` は
  `max_concurrent_requests × block_size` を `kDFlash2SpecMaxVerifyRows` で上限した値である。
  serve 終了時に `shutdown_dflash2_spec_batch_context()` →
  `dflash2_executor_shutdown()` の順で解放する。
- `ContinuousBatcher::set_dflash2` に drafter を bind すると、`submit` した request は
  admission 時に `DFlash2ContextState` / `DFlash2SpecDecoder` を request 専用に作成し、
  prompt を `dflash2_spec_prefill` で chunk prefill する。release は
  `finish_request` が decoder → context → sequence の順に行う。
- `step()` は DFlash2 request を `DFlash2SpecBatchSlot` に詰めて
  `dflash2_spec_step_batch()` を 1 round 実行し、続けて通常 decode 経路を同じ step で
  実行する。複数 request の target verify は 1 つの batch に統合される
  （draft は per-request で serial）。admission / retire / cancel / prefix cache /
  KV admission は通常 decode と同一の `ContinuousBatcher` 契約に従う。
- `request` は `speculative_decode` で種別を持つ。`--dflash2-model-dir` 指定時の serve は
  全 request を speculative として submit する。
- serve 側に DFlash 専用の admission / prefix cache / KV 容量の自動拡張は無い。

### target executor config（DFlash 有効時）

- `target_hidden_taps = dflash_config.target_layer_ids`、順序も config 通り
- `target_hidden_tap_count = dflash_config.num_target_layer_ids`
- `max_scheduled_output_rows = block_size`（`shelf-compute` の単一 request 経路）。
  serve mode では `min(dflash2_sessions × dflash2_session_rows(block_size, K), kDFlash2SpecMaxVerifyRows)`
  （`dflash2_session_rows` は `min(block_size, K + 1)`）であり、1 round の batch verify 行合計の上限になる。
  drafter の scratch、batch の history / compact log、target の出力行数もこの行数で見積もる
  （K=7 なら従来どおり `dflash2_sessions × 8`）。
- `max_batch_rows = min(2048, max(prompt_tokens, block_size))`
- `max_scheduled_output_rows` は `max(max_scheduled_output_rows, max_scheduled_requests)`
  で切り上げられた後 `max_batch_rows` で上限されるため、
  `block_size <= max_batch_rows` を満たす必要がある。prefill は
  `max_batch_rows` 単位で chunk 実行する。

### DFlash2 drafter executor config

- `max_rows = block_size`（単一 request の proposal 行数）
- `max_batch_rows = min(max_concurrent_requests × block_size, kDFlash2SpecMaxVerifyRows)`
  （batched proposal の行数上限。1 round の `Σ(Kᵢ+1)`）
- executor の scratch は `max(max_rows, max_batch_rows)` 行で確保する
- `max_context_rows = sliding_window`
- `target_lm_head = target.weights().lm_head`
- `target_embed_tokens = target.weights().embed_tokens`

batch proposal の scratch は `proposal_logits` が
`max(max_rows, max_batch_rows) × vocab_size × 4` bytes になる（F32）。
serve の既定（`max_concurrent_requests × block_size`）ではこの値が効くため、
`max_concurrent_requests` を上げると drafter の logits buffer も増える。

### env 一覧

| env | 既定 | 意味 |
| --- | --- | --- |
| `SHELF_DFLASH2_DEVICE_TOKEN_BRIDGE` | 1 | 0 で legacy Host bridge（単一 / batch 経路の両方） |
| `SHELF_DFLASH2_HOST_PROPOSAL_D2H` | 0 | 1 で proposal を host へ戻して可視化（timing 用） |
| `SHELF_DFLASH2_SERVE_STATS` | 0 | 1 で serve 終了時に `DFLASH2_*` を stderr へ出す |
| `SHELF_DFLASH2_TOPK_REFERENCE` | 0 | 1 で reference top-k |
| `SHELF_DFLASH2_ATTENTION_REFERENCE` | 0 | 1 で ring attention 参照実装 |
| `SHELF_DFLASH2_RADIX_TOPN` | auto | 0 で従来 topn、1 で radix topn、未指定は pool crossover |

### resource lifecycle

DFlash mode の生成後は次の順で解放する。

1. `DFlash2SpecDecoder` shutdown
2. `DFlash2ContextState` shutdown
3. `PagedSequenceState` release
4. `DFlash2Executor` shutdown

serve mode は request の解放を `ContinuousBatcher::finish_request` が行い（decoder →
context → sequence）、serve 終了時に `DFlash2SpecBatchContext` を
`shutdown_dflash2_spec_batch_context()` し（request ごとではない）、その後に
`DFlash2Executor` を shutdown する。

その後、共通経路で target `Executor` shutdown → stream destroy → arena shutdown を行う。
pool leak 検査（kv page / seq slot / gdn slot）は DFlash mode でも共通経路で実施する。

### 統計行

`--dflash2-stats 1` のとき以下を stdout に出力する。

```
DFLASH2_DRAFTS, DFLASH2_BLOCK_SIZE, DFLASH2_ROUNDS, DFLASH2_FULL_ACCEPTS,
DFLASH2_PARTIAL_ACCEPTS,
DFLASH2_GDN_HISTORY_BYTES, DFLASH2_GDN_SNAPSHOT_BYTES, DFLASH2_GDN_COMPACT_BYTES,
DFLASH2_GDN_MODE, DFLASH2_ACCEPTED_DRAFTS,
DFLASH2_MEAN_ACCEPTED, DFLASH2_FULL_ACCEPT_RATE, DFLASH2_EMITTED_PER_ROUND,
DFLASH2_DRAFT_MS, DFLASH2_D2H_MS, DFLASH2_DRAFT_GPU_MS, DFLASH2_PROPOSAL_WAIT_MS,
DFLASH2_PROPOSAL_COPY_MS, DFLASH2_VERIFY_MS, DFLASH2_VERIFY_GPU_MS,
DFLASH2_DECISION_WAIT_MS, DFLASH2_DECISION_D2H_MS, DFLASH2_DEVICE_TOKEN_BRIDGE,
DFLASH2_RERUN_MS, DFLASH2_GDN_SNAPSHOT_MS, DFLASH2_GDN_RESTORE_MS, DFLASH2_COMMIT_MS,
DFLASH2_ROUND_MS
```

`DFLASH2_GDN_MODE` は常に `compact` である。

`GENERATED_IDS=` 形式は target-only と完全に同一である。

---

## Invariants

- proposal は context state を変更しない（read-only）。
- DFlash ring に入るのは target が consume して commit した row だけ（commit-only）。
- step 終了時に `context.next_position == sequence.position`、
  `context.length == min(sequence.position, capacity)`。
- `GENERATED_IDS` は target-only greedy と完全一致する。
- target verify は通常の forward 経路と同じ kernel で行う。
- drafter kernel は D2D memcpy による relay を行わない。
- hot-path enqueue API は sync / D2H / H2D / allocation を行わない。
- すべての契約違反・非対応構成は明示的な error とし、silent fallback しない。
- buffer は所有側（`Executor` / `DFlash2Executor` / `DFlash2ContextState` /
  `DFlash2SpecDecoder`）が確保し、明示的な shutdown 経路を持つ。

---

## Qwen4Exp target

Qwen3.8-Flash-Next は 48 層で、drafter の `dflash_config.target_layer_ids` は
`[5, 15, 25, 35, 45]` である。Qwen4Exp の hidden は 4 本の hyper stream を持つため、
target tap は層の完了出力を stream 方向に平均した `[rows, hidden=2560]` とする
（`MEAN_STREAMS` primitive）。tap は head の 4 出力の後ろに external output として
追加され、`Executor` の `dflash_target_hidden` に bind される。

- Qwen4Exp の attention / GDN は複製で、TP 集約は MoE combine だけである。
  DFlash2 の target verify も同じ TP 経路を使う。
- CLI: `--dflash2-model-dir models/Qwen3.8-Flash-Next-DFlash2`。
- serve: `--serve-stdio` と併用できる。
- `--dflash2-drafts` は **6 以下**。block_size 8 の 8 行 verify は
  target-only greedy と 64 token で分岐する未解決課題があるため、7 以上を要求すると
  6 に cap して stderr に出す。
- 生成一致アンカー: `The capital of France is` → ` Paris. The capital of Germany is Berlin`
  （`11751,13,561,6511,314,9564,369,19241`）。drafts=6 の 64 token は target-only と
  完全一致する。

## Known limitations

- `--dump-logits` との併用は非対応。
- `--serve-stdio` と `--kv-cache-dtype fp8|octave:FORMAT` は serve mode と併用できる。
- target 層数 > 最大 tap、`num_target_layer_ids == 5`、`conv_kernel_size == 2` を要求する。
- context ring は sliding_window 分しか保持しない。`capacity == sliding_window` を要求する。
- attention GQA kernel は `q_heads == 4 * kv_heads`、`head_dim == 128` を要求し、それ以外は
  ring 参照実装へフォールバックする（不成立自体はエラーではない）。
- GDN history は `history bytes > 2.25 GiB` のとき create を FAIL する。
  27B geometry では `num_drafts <= 15` が上限になる。
- `--dflash2-drafts` は `block_size - 1` 以下。`block_size <= max_batch_rows` を
  満たす必要がある。
- batched verify は 1 round の行合計 `Σ(Kᵢ+1)` が `max_scheduled_output_rows` と
  `kDFlash2SpecMaxVerifyRows`（512）以下でなければならない。serve では
  `max_concurrent_requests × block_size <= 512` を要求する。
- batched proposal は drafter の weight traffic が request 数に比例して増える。draft の
  batching は launch 数と host stall を削減するが、weight 再読込は drafter の GEMM
  config に依存する。

---

## Source map

### drafter

- `include/shelf/models/qwen35/dflash2/{config.h,weights.h,executor.h,context_state.h}`
- `src/shelf/models/qwen35/dflash2/{config.cpp,weights.cpp,executor.hip,context_state.hip}`

### kernels

- `include/shelf/models/qwen35/kernels/dflash2/` と
  `src/shelf/models/qwen35/kernels/dflash2/`
  - `feature_concat` / `grouped_dynamic_conv` / `rmsnorm` / `rope` /
    `swiglu` / `attention` / `kv_ring` / `noise_input`
  - `candidate_selector` / `topk` / `topk_optimized`
  - `radix_topn`
  - `detail/`（`coarse_topn_device.h` / `rmsnorm_device.h` / `rope_device.h` /
    `swiglu_device.h`）

### runtime

- `include/shelf/models/qwen35/runtime/dflash2_spec_decoder.h`
- `src/shelf/models/qwen35/runtime/dflash2_spec_decoder.cpp`
- `src/shelf/models/qwen35/runtime/gdn_spec_history.hip` と
  `include/shelf/models/qwen35/runtime/gdn_spec_history.h`
- `src/shelf/models/qwen35/runtime/spec_decode.{h,cpp}`
- `include/shelf/inference/scheduling/scheduled_batch.h`

### apps

- `src/apps/compute/main.hip`（`run_dflash2_shot`）
- `src/apps/compute/compute_runtime.{h,hip}`（tap / max output rows の引き継ぎ）

### build / tests

- `cmake/options.cmake`
- `cmake/targets.cmake` / `cmake/apps.cmake` / `cmake/tests.cmake`
- required: `test_dflash2_config` /
  `test_dflash2_weight_contract` / `test_dflash2_feature_concat` /
  `test_dflash2_grouped_dynamic_conv` / `test_dflash2_bf16_geometry` /
  `test_dflash2_rope` / `test_dflash2_attention` / `test_dflash2_topk` /
  `test_dflash2_candidate_selector` / `test_dflash2_kv_ring` /
  `test_dflash2_attention_ring` / `test_dflash2_radix_topn` /
  `test_gdn_spec_history`
- perf（label `gpu1;perf`、正しさテストと分離）: `test_dflash2_radix_topn_perf` /
  `test_dflash2_batch_perf`
- optional / external files（model dir が必要）: `test_dflash2_weight_real` /
  `test_dflash2_target_taps` / `test_dflash2_mixed_batch` /
  `test_dflash2_batch_e2e` / `test_dflash2_tp_e2e`（label `gpu2`）

verify の受理結果は残り token 数と停止 token の最初の位置で制限してから、
target と DFlash2 context の状態を確定する。停止 token 自体は出力し、その後の行は確定しない。round の入力 `pending_token` が停止 token の場合は、
step の先頭で何も draft せず `finished=true`（出力 0 個）を返す。
