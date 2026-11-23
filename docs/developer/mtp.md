# MTP speculative decoding

checkpoint の MTP block（`mtp.*`）を drafter として使う compute / serve 経路である。
DFlash2 とは排他的で、`--mtp` を指定したときだけ有効になる。

MTP block は checkpoint が `mtp.*` として持つ tensor から作り、dense model と Qwen4Exp で
load / lower / 実行できる。MTP block を持たない checkpoint で `--mtp` を指定すると
`LoadedModel::mtp_drafter_input` が `unsupported` を返す。dense model は
`--tensor-parallel 2` と併用できる。`--pipeline-parallel 2` と `--dump-logits` との併用は
起動時に拒否する。

Qwen3.5 MoE の MTP block は routed expert を fused rank-3 の BF16 として保持し、以前は
load 時に MXFP4 へ再量子化していた。この online quantization は現在無効であり、
`load_mtp_layers` を指定した MoE load は `Status::unsupported` で拒否する
（[qwen35moe.md](qwen35moe.md) の MTP block 節）。

## Tensor parallel

dense model の TP=2 では、MTP block の重みは partition しない（TP partition plan に
`mtp.*` の entry は無い）。各 rank が MTP block を複製して load し、drafter executor は
rank0 だけで動かす。qwen4exp の `--serve-stdio --mtp` は `mtp_split` で MTP block を
全 rank に分割する。

target の prefill / verify は `TpCoordinator::on_execute` で全 rank に同じ batch を流す。
speculative verify は rank ごとの `ExecuteBatchOptions` を渡し、各 rank の
`GdnSpecHistory` と compact log buffer を使う。snapshot / restore / compact replay は
全 rank に対して実行する。受理数は rank0 の logits から決め、その値で全 rank の GDN を
同じ prefix に戻す。

coordinator の mirror sequence は `on_sequence_created` / `on_sequence_released` で
target sequence と対応付ける。MTP+TP のときは TP executor に
`emit_normed_token_hidden` を立て、rank0 の normed hidden を drafter に渡す。

MTP+TP では drafter と各 rank の GDN snapshot / compact 領域が KV pool の後ろに必要になる。
`TpRankRuntime` は `MtpBatchReserveConfig` を受け取り、次の 2 項を working set reserve に加える。
各 rank は verify 行分の共有 GDN history / compact log と、drafter 数分の target GDN
snapshot を `mtp_batch_rank_state_reserve_bytes` で計上する。`--serve-stdio` の TP 分割
では全 rank が request 数分の `mtp_drafter_reserve_bytes` を計上する。verify 行は
`mtp_spec_rows(max_requests, num_drafts)` = `max_requests * (num_drafts + 1)` であり、
drafter の prefill 行数（`max_batch_rows`）とは独立である。drafter の executor は
MTP block 自身の `mtp_exl3_linear_shape` で staging plane を見積もる。target GDN snapshot は
各 drafter が全 rank の local layout に 1 面ずつ確保するため、rank ごとに
`max_requests` 面を計上する。

Qwen3.5 MoE の MTP block は routed expert の online MXFP4 再量子化が無効なため
load できない。qwen4exp は TP=2 を qualified とし、TP=4 は未計測である。

qwen4exp の `--serve-stdio --mtp` は、TP > 1 のとき MTP block を全 rank に load して
TP 軸で分割する。`TpRankRuntimeConfig::mtp_split` が立つと MTP attention の Q/K/V/O は
local head 数へ、routed experts は local expert 範囲へ分割される。drafter は target の
語彙分割 lm_head をそのまま使い、draft の greedy は logits の TP combine で全語彙を
得てから行う。MTP 用の full-vocabulary head は複製しない。

単発生成（`--serve-stdio` なし）の TP は従来どおり rank0 が MTP block 全体を持つ。
この経路では `mtp_split` を立てず、rank1 は MTP block を load しない。

qwen4exp のように QSA / PLE の per-row state を持つ target は、MTP block を 1 つの
共有 executor で動かす。request ごとの MTP KV / hidden / position は独立させたまま、
draft は depth 単位で request をまとめて 1 forward にする。draft 数 K に対して
MTP forward は最大 K 回であり、request 数 N に比例しない。

既定の verify は draft 列 `[pending, draft0, ...]` を 1 回の target forward（`K+1` 行）で
まとめて確認し、最初の不一致までを採用する。target の state は受理した prefix
だけを進める。`SHELF_MTP_FUSED_VERIFY=0` を指定すると、request ごとに 1 row ずつ
target の `DECODE` で流す安全側の直列 verify に戻る。直列 verify は GDN snapshot /
compact log replay を必要としない。
`SHELF_MTP_WAVEFRONT=1` では同じ深さの active な request を最大 4 request の
Wavefront batch にまとめる。Wavefront は forward 数を減らすが、near-tie で
target-only と分岐する prompt が残るため既定 off である。

qwen4exp の融合 verify は GDN verify の上限 8 request / 64 行の範囲で K+1 行を
1 forward にまとめる。1〜2 request の融合 verify は target-only と一致するが、
4 request / 20 行では near-tie 分岐が残る。TP serve の融合 verify は rank ごとに
`mtp_batch_rank_state_reserve_bytes` を MTP reserve へ加算する。

## serve-stdio

`--mtp --serve-stdio` は複数 request を同時に保持する継続バッチ serve である。
`shelf-server` の MTP 経路はこれを使う。

- 起動時に `MtpSpecBatchContext` を作る。request 数分の `MtpDrafter`、target verify 用の
  共有 GDN history / compact log、QSA / PLE の per-row history を持つ。
- request の prefill は admit 時に行い、以後の round は active な request をまとめて進める。
  終了した request は sequence と drafter slot を解放する。
- 各 round は request ごとに draft する。融合 verify を使う target は active な request の
  target verify を 1 forward にまとめる。qwen4exp は深さごとに最大 4 request の
  Wavefront `DECODE` batch を使う。融合する request の数は round ごとに変わるため、
  融合 verify では数値が round の row 数に依存しないことが要件になる
  （下記「verify の行数不変性」）。
- `token` event を request ごとに emit し、`done` で `generated_ids` を返す。
- MTP block は backbone の TP 軸で分割する。TP を使う構成では全 rank が block の
  shard を持ち、EP の axes overlap では EP グループが backbone の TP 軸になるため
  同じく全 rank が shard を持つ。分割しない構成では rank0 だけが block 全体を持つ。
- 融合 verify の TP では `TpCoordinator` の per-rank options 経路を使い、rank ごとの
  GDN / QSA / PLE history と snapshot で rollback する。qwen4exp の Wavefront 検証は
  通常の TP decode 経路を使う。

## prefix cache

MTP の admit は通常の `PrefixCache` を lookup し、target の KV / GDN / QSA / PLE と
TP mirror を restore する。同じ checkpoint には `MtpDraftSnapshot` も保存する。

- snapshot は leader と全 peer の MTP KV ring、leader の hidden cache、last token /
  position、sample index を持つ。
- key は prefix token 列と target position の hash である。
- prefill の page-aligned chunk 境界で保存する。最初の境界、tail
  （`align_down(input_len - 1, page_tokens)`）、page-aligned な入力末尾は必ず保存し、
  以後は直前の保存から `prefix_cache_save_interval` 以上進んだ境界でも保存する
  （通常 decode の periodic checkpoint と同じく、間隔は下限である）。
- tail を保存点にするため、prefill の最終 chunk は tail で分割する。通常 decode 経路の
  `dflash2_prefill_tail_position` と同じ位置である。
- request の終了（stop / length）では、decode 中に最後に通過した page 境界を出力末尾
  として保存する。capture は prefill 完了時に開始し、round の先頭で position が page
  揃えのとき drafter snapshot と target state を同じ位置で取る。そのため prefix cache
  有効時は、1 round の draft 数を次の page 境界までに制限する。出力末尾の entry は、
  client が返した assistant メッセージを同じ token 列で再レンダリングしたときだけ
  一致する（[prefix_cache.md](prefix_cache.md)）。
- restore では target を復元した後、TP mirror と PLE などの token 履歴を復元 prefix で
  re-seed し、drafter の ring / position / hidden を戻す。single GPU の restore でも
  `Executor::external_prefix_restore` に復元 prefix を渡す。snapshot の MTP position が
  prefix - 1 のときは、suffix の先頭 token を 1 回だけ MTP forward してから suffix の
  prefill を始める。
- suffix が空になる prefix は restore しない。target の最後の logits を保存していない
  ため、pending token を作れないからである。
- suffix prefill は `mtp_spec_prefill` の `start_position` から実行し、prefix 部分の
  target forward を繰り返さない。

## verify の行数不変性

speculative verify の行数は、その round に active な request 数で変わる。EXL3 の split-K は
`rows` から分割数を選ぶ（`rows <= 8` で 8、`rows == 1` で grid 依存、それ以外は 4 または 1）
ため、同じ request でも同居する request 数によって K の縮約順が変わり、argmax が
入れ替わることがある。qwen4exp では 0.2 logit 差の near-tie が実際に存在し、
admission のタイミングで生成列が分岐した。

`Exl3GemmArgs::splitk_tuning_rows` に verify の行容量
（`ExecutorConfig::max_scheduled_output_rows` を GDN verify kernel の上限 32 行で頭打ちに
した値）を渡し、分割数と scratch の見積をその容量で計算する。実行する行数ではなく容量で
決まるため、同じ request は同居する request 数にも設定した同時実行数にも関係なく同じ
分割数を使う。mul1 と mul24（EXL3-INT8）の両方がこの値に従う。QSA の sparse attention
も同じ容量で split 数を決める。decode（`rows == 1`）と prefill は従来どおりである。
QSA / PLE を持つ qwen4exp も既定で融合 verify を使う。Wavefront は独立した flag で、
既定 off のままである。Wavefront の batch は最大 4 request に区切り、
それ以上同時に並べた場合の数値差を避ける。

## verify の 1 forward 上限

GDN の verify kernel の定数は **8 request / 64 行**
（`kGdnRecurrenceVerifyMaxRequests` / `kGdnRecurrenceVerifyBatchRows`）である。
qwen4exp の既定経路は融合 verify を使う。`SHELF_MTP_FUSED_VERIFY=0` で 1 request /
1 行ずつの直列 verify に戻せる。融合 verify の chunk はこの上限で分割する。
DFlash2 の 4 session × block 8 と同じ上限で、これが融合 verify の実効上限になる。

MTP の batch context 自体は `min(max_num_batched_tokens, 512)` 行まで request を保持でき、
1 round の verify は 4 request / 32 行ずつの chunk に分けて実行する。8 request ×
(drafts 7) なら 64 行を 2 forward、16 request なら 128 行を 4 forward になる。chunk は
行容量で分割数を固定しているため、chunk に分けても数値は変わらない。

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

- pool は 1 attention layer、`num_key_value_heads`、`attention_head_dim` で作る。target が
  BF16 / FP8 のときは target と同じ KV dtype、target が Octave のときは BF16 を使う。
  Octave target pool と MTP pool は所有権も page table も共有しない。
- `--mtp-window` が 0 でないときは ring を使う。ring page 数は
  `ceil((sink + window + num_drafts + 1) / page_tokens) + 1` を
  `ceil((max_seq_len + 1) / page_tokens) + 1` で頭打ちにした値である。
  block table は論理長 `max_seq_len` のまま確保し、entry `i` は ring page
  `i % ring_pages` を alias する。`--mtp-window 0` のときだけ全論理 page を確保する。
- MTP 用の `SequenceSlotPool` / `GdnStatePool` / `PagedSequenceState` を別に持つ。
  単一 request では `GdnStatePool` は `create_executor` の geometry 検証を通すための
  1 state だけを持つ。batch の qwen4exp は request 数分の slot / GDN state /
  MTP page を 1 つの共有 executor に持つ。ring の page 数は request ごとの ring に
  reset 用の 1 ring を加えた `ring_pages * (max_requests + 1)` である。
- `mtp_spec_step` の各 round は MTP sequence を round 開始位置まで rewind してから
  first pass を実行する。ring では block table が固定の alias なので、rewind は
  physical page を release も truncate もせず `position` だけを戻す。ring を使わない
  ときは必要な page 数を残して余剰 page を release する。
- sequence の release は block table の alias を dedupe し、ring page を 1 回ずつ返す。
- `mtp_split` では rank ごとに local KV head 数の MTP pool と sequence を持つ。
  leader の `MtpDrafter` は peer の executor と sequence を保持し、draft の各 forward を
  `TpCoordinator::on_execute_aux` で全 rank に同じ batch として流す。peer の position は
  leader と同じ値へ進め、rewind も同時に適用する。peer sequence は `mtp.active` の
  並びではなく drafter slot に対応させる。hidden 入力は leader の hidden を host 経由で
  peer の `input_hidden` へ配る。

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
3. **verify**: dense など QSA / PLE を持たない target は
   `[pending, draft0..draftK-1]` の `K+1` row を `SPEC_VERIFY` で 1 forward に流す。
   qwen4exp のように QSA / PLE を持つ target は同じ列を深さごとに `DECODE` し、
   同じ深さの request を最大 4 request の Wavefront batch にまとめる。
4. **accept**: longest prefix を取る。full accept では bonus token、partial accept では
   correction token を次 round の pending にする。

すべての draft は target verify で棄却可能であり、出力の正しさは target が担保する。

## Partial reject

融合 verify を使う target は GDN recurrence / conv を `K+1` row 分進めるため、
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

target の prefill chunk は target-only と同じ `max_batch_rows` を上限にする。MTP 専用の
小さい chunk に変えてはならない。chunk を変えると prefill kernel の丸めが変わり、
target-only と `GENERATED_IDS` が一致しなくなる。prefix cache の tail では、通常 decode
経路と同じく chunk を分割する。

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

serve の終了時には
`MTP_BATCH rounds=... accepted=... emitted=... chunks=... draft_forwards=... max_active=...`
を 1 行出す。`chunks` は verify の target forward 数、`draft_forwards` は MTP draft の
forward 数、`max_active` は同時に進めた request 数の最大値である。qwen4exp では
draft が depth 単位で batch 化されるため、`draft_forwards` は round 数 × drafts 以下に
収まり、request 数には比例しない。

`GENERATED_IDS` の形式は target-only と同一である。

## 制約

- MTP block を持たない checkpoint では使えない。dense model と Qwen4Exp は block が
  あれば load される。Qwen3.5 MoE の MTP block は未対応である。
- `--pipeline-parallel 2` / `--dump-logits` は未対応である。
- prefix cache は MTP 経路でも使う。target state に加えて、drafter の KV ring、
  hidden cache、last token / position を同じ checkpoint に保存し、復元後は suffix の
  prefill だけを実行する。
- partial accept は GDN snapshot + compact log replay で復元する。target rerun は行わない。
- MTP の prefill first pass は 512 行以下へ分割する。target の prefill chunk は
  target-only と同じ値を使う。
- MTP KV は target KV とは別 pool である。容量計画は
  `mtp_batch_rank_state_reserve_bytes`（verify 行分の共有 history / compact log）と
  `mtp_drafter_reserve_bytes`（split 時は全 rank、非 split 時は rank0 のみ）で行う。
- Octave target と併用した MTP KV は BF16 であり、target の descriptor を MTP block へ
  伝播しない。
- qwen4exp の MTP block は serve の TP 分割では全 rank が持ち、単発生成では rank0 だけが
  持つ。drafter の入力は final norm ではなく pre-mixer の wide residual（`hc * hidden`）
  である。QSA / PLE を持つ target は `PleConvStatePool` / `QsaKeyStatePool` の
  per-row state を戻さずに済むよう、深さごとの Wavefront `DECODE` batch で検証する。
  この経路の生成列は target-only と一致する。起源の logits は target の語彙分割
  lm_head を TP combine して得る。
- verify の split-K は行容量で固定する。mul1 / mul24 の両方と QSA sparse attention が
  この値に従う。融合 verify を使う family では、これがないと同居する request 数で
  argmax が変わる。
- 融合 verify を使う family では、split を固定しても MoE grouped path などの数値は
  完全な行数不変ではない。near-tie が多い checkpoint では target-only との bit 一致が
  保証されないため、QSA / PLE を持つ qwen4exp は上記の 1 行検証経路を使う。

## Source map

- `include/shelf/kernels/optimized/linear/exl3_family/types.h`（`splitk_tuning_rows`）
- `src/shelf/inference/execution/optimized_dispatch.hip`（mul24 split の容量固定）
- `src/shelf/inference/execution/paged_attention_dispatch.hip`（QSA split の容量固定）

- `include/shelf/models/qwen35/runtime/mtp_drafter.h`（`MtpDrafterInput`、`mtp_drafter_rows`）
- `src/shelf/models/qwen35/runtime/mtp_drafter.cpp`
- `include/shelf/models/qwen35/model/lower_to_primitives.h`（`emit_normed_token_hidden`）
- `src/shelf/models/qwen35/model/lower_to_primitives.cpp`
- `include/shelf/models/loaded_model.h`（`has_mtp_block` / `mtp_drafter_input` / `mtp_drafter_reserve_bytes`）
- `include/shelf/models/qwen4exp/weights/qwen4exp_weights.h`（`mtp_exl3_linear_shape`）
- `src/shelf/runtime/tp/tensor_parallel.cpp`（`load_mtp_layers` / `mtp_split` の rank 伝播、`on_execute_aux`）
- `include/shelf/runtime/tp/tp_batch_hook.h`（`TpAuxRank` / `on_execute_aux` / `attach_aux_executor`）
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
  - dense と Qwen4Exp の checkpoint へ向けて成立する（`--mtp` の `GENERATED_IDS` が
    target-only greedy と一致し、partial accept の GDN compact log replay が走る）。

MTP の verify は出力可能な残り token 数と停止 token の最初の位置で受理結果を
制限してから target の状態を確定する。停止 token 自体は出力し、その後の行は確定しない。round の入力 `pending_token` が停止 token の場合は、
step の先頭で何も draft せず `finished=true`（出力 0 個）を返す。
