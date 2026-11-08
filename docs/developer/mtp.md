# MTP speculative decoding

checkpoint の MTP block（`mtp.*`）を drafter として使う compute / serve 経路である。
DFlash2 とは排他的で、`--mtp` を指定したときだけ有効になる。

MTP block は checkpoint が `mtp.*` として持つ tensor から作り、dense / grouped expert
（MoE）のどちらの model でも load / lower / 実行できる。MTP block を持たない checkpoint で
`--mtp` を指定すると `LoadedModel::mtp_drafter_input` が `unsupported` を返す。
dense model は `--tensor-parallel 2` と併用できる。`--pipeline-parallel 2` と
`--dump-logits` との併用は起動時に拒否する。

grouped expert を使う model の MTP block は、routed expert を checkpoint で fused rank-3 の
BF16（`mtp.layers.0.mlp.experts.gate_up_proj` / `experts.down_proj`）として保持する。
load 時に MXFP4 へ再量子化して transformer layer と同じ native plane に落とす
（[qwen35moe.md](qwen35moe.md) の MTP block 節）。
MoE の MTP decode は verify の multi-row forward を使うが、greedy 列は target-only と一致する
（verify の前提である forward の batch 行数非依存は `test_qwen35moe_rows_parity` で固定）。
実測で target-only より速い（`--mtp-drafts 4` で 1.88 倍）。draft 受理率は dense より低い
（35 % に対し 98 %）が break-even は超える。数値は `docs/perf/current.md`、経緯は
[../research/20261105/moe_mtp.md](../research/20261105/moe_mtp.md)。

## Tensor parallel

dense model の TP=2 では、MTP block の重みは partition しない（TP partition plan に
`mtp.*` の entry は無い）。各 rank が MTP block を複製して load し、drafter executor は
rank0 だけで動かす。

target の prefill / verify は `TpCoordinator::on_execute` で全 rank に同じ batch を流す。
speculative verify は rank ごとの `ExecuteBatchOptions` を渡し、各 rank の
`GdnSpecHistory` と compact log buffer を使う。snapshot / restore / compact replay は
全 rank に対して実行する。受理数は rank0 の logits から決め、その値で全 rank の GDN を
同じ prefix に戻す。

coordinator の mirror sequence は `on_sequence_created` / `on_sequence_released` で
target sequence と対応付ける。MTP+TP のときは TP executor に
`emit_normed_token_hidden` を立て、rank0 の normed hidden を drafter に渡す。

MTP+TP では drafter と各 rank の GDN snapshot / compact 領域が KV pool の後ろに必要になる。
`TpRankRuntime` は `MtpBatchReserveConfig` を受け取り、rank model の
`token_row_ffn_size` / `mtp_grouped_reserve_bytes` / `uses_exl3_linear` から
`mtp_spec_batch_reserve_bytes` を計算して working set reserve に加える。

grouped expert（MoE）の TP load は MTP block を分割しないため、MTP との併用を
`build_moe_tp_rank_model` が拒否する。TP=2 を qualified とし、TP=4 は未計測である。

## serve-stdio

`--mtp --serve-stdio` は複数 request を同時に保持する継続バッチ serve である。
`shelf-server` の MTP 経路はこれを使う。

- 起動時に `MtpSpecBatchContext` を作る。request 数分の `MtpDrafter`、target verify 用の
  共有 GDN history / compact log、QSA / PLE の per-row history を持つ。
- request の prefill は admit 時に行い、以後の round は active な request をまとめて進める。
  終了した request は sequence と drafter slot を解放する。
- 各 round は request ごとに draft し、active な request の target verify を 1 forward に
  融合する。target が QSA / PLE の per-row rollback を持つ場合（qwen4exp）は、融合 verify の
  共有 history がまだ検証できていないため request ごとに verify する。
- `token` event を request ごとに emit し、`done` で `generated_ids` を返す。
- TP では `TpCoordinator` の per-rank options 経路を使い、rank ごとの GDN / QSA / PLE
  history と snapshot で rollback する。

## Forward contract

MTP block は次の順で 1 token を予測する。

```text
x_embed = pre_fc_norm_embedding(embed(token))
x_hidden = pre_fc_norm_hidden(target_normed_hidden)
h = fc(concat(x_embed, x_hidden))
h = full_attention_decoder_layer(h)
h = mtp.norm(h)
logits = lm_head(h)
```

- `mtp.fc` の入力は `[embedding_norm | hidden_norm]` の順で、`hidden_size * 2` である。
- MTP 層は `full_attention` 固定で、`mtp.layers.0.*` の 1 層だけを使う。
- attention は gated query（`q_proj` 出力の head 内 `[q | gate]`）と
  `attention = attn * sigmoid(gate)` を持つ。target と同じ Qwen3.5 の full attention である。
- `pre_fc_norm_hidden` に入る `target_normed_hidden` は target の最終 RMSNorm 後である。
  target の pre-norm の body 出力ではない。

## target graph の normed token hidden

MTP は target の全 token row の最終 RMSNorm 後 hidden を必要とする。target graph は
通常、最終 RMSNorm を `OUTPUT_ROWS` にだけ適用する。`--mtp` のときは
`Qwen35LowerOptions::emit_normed_token_hidden` を立て、`TOKEN_ROWS` 全体に同じ
RMSNorm を適用した値をもう 1 つの external output として追加する。

external output の並びは固定である。

| index | 内容 |
| --- | --- |
| 0 | selected hidden（`OUTPUT_ROWS`） |
| 1 | logits（`OUTPUT_ROWS`） |
| 2 | sampled token（`OUTPUT_ROWS`） |
| 3 | body output（`TOKEN_ROWS`、pre-final-norm） |
| 4.. | target hidden taps（`target_hidden_tap_count` 個） |
| 4 + tap count | normed token hidden（`TOKEN_ROWS`。`emit_normed_token_hidden` のときだけ） |

`Executor` は normed token hidden を `normed_token_hidden` に書き、MTP drafter が
`input_hidden` へ D2D copy して読む。

MTP graph の external input は `[token_ids, hidden]` の 2 つである。executor は
external input index 0 を token buffer、index 1 以降を `input_hidden` に bind する。
`input_hidden` は graph が 2 つ以上の external input を持つときだけ確保する。

## KV cache

MTP 層は target とは独立した KV pool を持つ。

- pool は 1 attention layer、`num_key_value_heads`、`attention_head_dim`、target と同じ
  KV dtype で作る。
- page 数は `ceil((max_seq_len + 1) / page_tokens) + 1` である。target pool の page 数
  ではない。
- MTP 用の `SequenceSlotPool` / `GdnStatePool` / `PagedSequenceState` を別に持つ。
  `GdnStatePool` は `create_executor` の geometry 検証を通すための 1 state だけを持つ。
- `mtp_spec_step` の各 round は MTP sequence を round 開始位置まで rewind してから
  first pass を実行する。rewind は必要な page 数を残して余剰 page を release し、
  `position` を戻す。

## Round

`position` を target sequence が次に consume する絶対位置とする。

### prefill

prompt を `executor.config.max_batch_rows` 単位で chunk prefill する。

- 各 chunk は target の通常 prefill である。最終 chunk だけ logits / sample を計算し、
  anchor token を得る。
- 最終 chunk 以外は、その chunk の target normed hidden と
  `[prompt[i+1]]`（chunk 末尾は次 chunk の先頭 token）で MTP の first pass を実行し、
  MTP KV を構築する。logits は計算しない。
- 最終 chunk の MTP first pass は step の先頭まで遅延する。これは anchor token が
  prefill 完了後にしか決まらないためである。

### step

`K = min(num_drafts, remaining_tokens - 1, room - 1)`、`room = max_seq_len - position`。
`K == 0` のときは単一 target decode だけを行い、MTP KV は更新しない。

`K > 0` のとき:

1. **first pass**: 直前の target 実行の committed prefix `rows = position - last_position`
   について、入力 token を 1 つ shift し、末尾を pending token に置き換える。
   hidden は target の normed token hidden の同じ row を使う。positions は
   `last_position .. position-1`。これで MTP KV を committed prefix について更新し、
   draft 0 を得る。
2. **draft loop**: `i = 1..K-1` について、token は `draft[i-1]`、hidden は直前の
   MTP normed hidden、position は `position + i - 1`。greedy で `draft[i]` を得る。
3. **verify**: target に `[pending, draft0..draftK-1]` の `K+1` row を
   `SPEC_VERIFY` で流し、各行の target sample を得る。
4. **accept**: `spec_greedy_accept` で longest prefix を取る。full accept では
   bonus token、partial accept では correction token を次 round の pending にする。

すべての draft は target verify で棄却可能であり、出力の正しさは target が担保する。

## Partial reject

MTP の verify は target の GDN recurrence / conv を `K+1` row 分進める。verify には
DFlash2 と同じ GDN spec history（conv）と compact log を渡す。partial accept では次を
順に行う。

1. verify 前に `spec_gdn_snapshot` で GDN conv / recurrent を保存する。
2. `spec_gdn_restore` で verify 前の state に戻す。
3. `restore_gdn_spec_conv` で受理 prefix の conv を戻す。
4. `commit_gdn_compact_log` で受理 `accepted + 1` row の recurrent を replay する。
5. target sequence の `position` を `position + accepted + 1` にし、必要な page 数まで
   rollback する。

target の再 forward は行わない。受理 row の target KV は verify が書いた値をそのまま
使う。compact log の commit descriptor は **target sequence の slot** でなければ
ならない。MTP drafter の slot を渡すと、state が別 slot に書かれて次の round の
verify が target-only と一致しなくなる。

## MTP prefill

target の prefill chunk は target-only と同じ `max_batch_rows` を使う。target の
chunk を MTP 用に変えてはならない。chunk を変えると prefill kernel の丸めが変わり、
target-only と `GENERATED_IDS` が一致しなくなる。

MTP 側の first pass は独立に `kMtpMaxPrefillRows`（512）行以下へ分割する。これに
より target executor と MTP executor の workspace を同時に確保しても 32 GiB に
収まる。最終 prompt chunk の最後の 1 row は anchor token が必要なため prefill では
実行せず、最初の step で 1 row だけ実行する。

## Sampling

- MTP の draft は常に greedy である。draft の sampling config は使わない。
- target verify は `--temperature` などの要求された sampling config を使う。
  acceptance は target の sampled token との greedy longest-prefix である。
- `sample_index` は target の実行 row 数だけ進める。prefill の最終 sample は 1、
  verify は `K+1`、単一 decode は 1 である。MTP の greedy draft は index を使わない。

## CLI

- `--mtp`: MTP を drafter として有効化する。
- `--mtp-drafts N`: 1 round の draft 数（default 7、1..31）。上限 31 は target verify の
  `K+1 <= 32` を WmmaVerify の M32 row 契約に合わせるためである。
- `--mtp-stats 0|1`: `MTP_*` 統計行の出力（default 1）。

`--mtp` は `--dflash2-model-dir` と同時に指定できない。

## 統計行

```text
MTP_DRAFTS MTP_ROUNDS MTP_FULL_ACCEPTS MTP_PARTIAL_ACCEPTS
MTP_ACCEPTED_DRAFTS MTP_MEAN_ACCEPTED MTP_FULL_ACCEPT_RATE
MTP_EMITTED_PER_ROUND MTP_VERIFY_ROWS MTP_GDN_RESTORES
MTP_DRAFT_MS MTP_VERIFY_MS MTP_GDN_RESTORE_MS MTP_COMPACT_REPLAY_MS MTP_ROUND_MS
```

`GENERATED_IDS` の形式は target-only と同一である。

## 制約

- MTP block を持たない checkpoint では使えない。dense / MoE のどちらの family でも
  block があれば load される。
- `--pipeline-parallel 2` / `--dump-logits` は未対応である。
- prefix cache は MTP の単一 request compute 経路では使わない。
- partial accept は GDN snapshot + compact log replay で復元する。target rerun は行わない。
- MTP の prefill first pass は 512 行以下へ分割する。target の prefill chunk は
  target-only と同じ値を使う。
- MTP KV は target KV とは別 pool である。容量計画は
  `mtp_spec_batch_reserve_bytes`（request 数分の drafter + 共有 history / compact log）と、
  single request 経路の `LoadedModel::mtp_drafter_reserve_bytes` を使う。
- qwen4exp の MTP block は rank0 だけが load し、drafter の入力は final norm ではなく
  pre-mixer の wide residual（`hc * hidden`）である。speculative verify は
  `PleConvStatePool` と `QsaKeyStatePool` の partial 状態を row 単位の history から復元する。
- QSA / PLE を持つ target の fused verify は request ごとの verify に落とす。融合 verify の
  共有 history は dense / MoE で検証済みだが、QSA / PLE の rollback を含む形では未検証である。

## Source map

- `include/shelf/models/qwen35/runtime/mtp_drafter.h`（`MtpDrafterInput`、`mtp_drafter_rows`）
- `src/shelf/models/qwen35/runtime/mtp_drafter.cpp`
- `include/shelf/models/qwen35/model/lower_to_primitives.h`（`emit_normed_token_hidden`）
- `src/shelf/models/qwen35/model/lower_to_primitives.cpp`
- `include/shelf/models/qwen35moe/model/lower_to_primitives.h`（`lower_qwen35_moe_mtp_to_primitives`）
- `include/shelf/models/loaded_model.h`（`has_mtp_block` / `mtp_drafter_input` / `mtp_drafter_reserve_bytes`）
- `src/shelf/models/qwen35/runtime/executor.hip`（hidden external input / normed output）
- `src/apps/compute/single_request.hip`（`run_mtp_shot`）
- `src/apps/compute/stdio_server.hip`（`serve_mtp_admit` / `serve_mtp_step`）
- `include/shelf/models/qwen4exp/runtime/executor_footprint.h`（`mtp_grouped_executor_bytes`）
- `include/shelf/inference/state/qsa_state_pool.h`（`restore_qsa_spec_state`）
- `include/shelf/inference/state/ple_conv_state_pool.h`（`restore_ple_conv_spec_state`）
- `src/apps/compute/compute_cli.cpp`（`--mtp*`）

required test:

- `test_qwen35_lowering_contract`
- `test_qwen35_mtp_lowering`
- `test_qwen35_mtp_weight_load`
- `test_compute_mtp_rejects_dflash2` / `test_compute_mtp_rejects_tp2` /
  `test_compute_mtp_drafts_reject_over_ceiling`

optional external test:

- `test_mtp_parity`（`SHELF_MODEL_DIR_MTP` または `SHELF_MODEL_DIR`）
  - dense / grouped expert のどちらの checkpoint へ向けても成立する（MoE でも `--mtp` の
    `GENERATED_IDS` が target-only greedy と一致し、partial accept の GDN compact log replay
    が走る）。grouped expert model の forward は batch 行数に依存しないことを
    `test_qwen35moe_rows_parity` が別に固定する（[qwen35moe.md](qwen35moe.md) §7）
