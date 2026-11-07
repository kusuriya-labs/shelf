# grouped expert の pipeline parallel（2026-11-05）

実験 ID: `MOE-PP-1` / 結論: **ACCEPT**（layer 範囲分割で PP を通した。速度ではなく容量の手段）

前提となる TP の記録は [moe_tp_expert_axis.md](moe_tp_expert_axis.md)。順序は
fp8 KV → MTP → TP → PP の最後。dense PP の経過は
[../20261104/pipeline_parallel_port.md](../20261104/pipeline_parallel_port.md)、
MoE と PP を合体させたときの gate の経過は
[../20261104/moe_pp_merge.md](../20261104/moe_pp_merge.md)。

## 1. 何が塞がっていたか

- `src/apps/compute/compute_runtime.hip` が `probe_model_dir == Moe` で
  `unsupported("pipeline parallel is not implemented for grouped expert models")`
- `Qwen35MoeLoadOptions` に layer 範囲の字段が無く、`load_qwen35moe_weights` は
  `config.num_hidden_layers` 分の expert plan を作って全 layer を読んでいた
- `Qwen35MoeLowerOptions` に stage が無く、lowering は embed を常に emit し head を常に emit
  し `weights.layers` の全範囲で回っていた
- `PipelineStageRuntime` は `Qwen35Model` を直接所有し、executor footprint を
  `tc.ffn.intermediate_size` で見積もっていた。grouped expert は FFN の平面が group 行数で
  決まるので、この見積りでは stage の KV pool を切れない
- 実装済みの parts: executor core の first / last stage 分岐（`pipeline_input_hidden` /
  `pipeline_output_hidden` の確保と external I/O 束縛）と program 側の group 行容量は
  family 中立で、そのまま使えた

## 2. 採否判断

**stage model を interface 経由で注入する**（TP の `TpRankModel` と同じ形）。候補:

1. `PipelineStageRuntime` を `src/shelf/models/`（両 family を link できる層）へ移す
2. `shelf::models::PipelineStageModel` interface を作り、stage runtime は qwen35 runtime に
   置いたまま builder 関数ポインタを注入する

2 を採用。TP 側で既に同じ形（`TpRankModel` + `TpRankModelBuilder` +
`select_tp_rank_model`）が動いており、stage runtime の schedule・mirror・hand-off
（family を名乗らない部分、約 350 行）を動かさずに済む。1 は PP の実行契約そのものを
link 図の上位へ移すことになり、TP の実装と非対称になる。

`PipelineStage` / `StageRange` / `even_stage_ranges` を `shelf::qwen35` から
`shelf::qwen_hybrid` へ移した。qwen35moe が stage を表現すると `models/qwen35/` を
include することになり、family 境界の test（`shelf_qwen35moe` と `shelf_qwen35` の相互 link
禁止、qwen35moe 配下の文字列検査）に反する。呼び出し側は using で同じ名を保った。

## 3. 実装の形

- load: `Qwen35MoeLoadOptions::stage`。要求 tensor 名は担当範囲だけ。embed は first stage、
  lm_head / final norm は last stage（tied なら last stage も embed を読む）。model-level の
  読み分けは `HybridLoadOptions::load_embedding / load_lm_head / load_final_norm` に落とす。
  `out.layers` は全 layer 分確保し、担当外は tensor を持たない（dense と同じ規約）
- validate: `validate_qwen35moe_weights(..., stage)` は担当範囲だけ
  `validate_hybrid_layer_body` と `moe_ffn_weights_match` を張る。stage + MTP は dense と同じ
  メッセージで unsupported
- lowering: `Qwen35MoeLowerOptions::stage`。担当範囲だけ graph へ載せ、first stage だけが
  embedding を emit して external input を token id にする。それ以外の stage は bf16 hidden を
  external input にし、直前 layer の出力を external output へ渡す。head は last stage のみ。
  GDN / attention の state index は stage 内で 0 から数える（stage が自分の pool を持つ）
- footprint: `qwen35_executor_reserve_bytes(tc, token_row_ffn_size, ..., uses_exl3_linear)` に
  `ffn_workspace_reserve_bytes(max_batch_rows)`（= `moe_executor_grouped_bytes`）を足す 2 項。
  TP にあって PP に無かった auto slack `max(64 MiB, free / 100)` も入れた
- `create_moe_model_executor` は `ExecutorConfig::stage` を lower options へ流す
- stage + expert 軸 rank の同時指定は loader で unsupported。一段の layer が layer 範囲と
  expert 範囲を同時に持つ配線は無い（TP と PP を同時に張るのは非目標のまま）

weight_bytes / tensor_bytes の実装が 3 箇所（`pipeline_parallel.cpp`、
`tensor_parallel.cpp`、`tp_rank_model.cpp`）に重複していた。fp8 / mxfp4 / exl3 の平面をどう
数えるかという規則が 3 本あると stage / rank 側のメモリ計画だけズレるので、
`include/shelf/weights/weight_bytes.h` の inline へ寄せた。

## 4. はまった点

`pipeline_parallel.h` の `namespace shelf { namespace qwen35 { namespace runtime {` の
**内側**に `namespace shelf::models { class PipelineStageModel; }` を書き、
`shelf::qwen35::runtime::shelf::models` が生まれていた。それ以降の `shelf::...` 解決がすべて
壊れて数十個のエラーになった。forward declaration はファイル先頭（名前空間の外）に置く。

## 5. 実測（Qwen3.6-35B-A3B MXFP4、device 0,1）

正しさ: prompt 512 / 1024 / 1943 token の 3 case すべてで PP=1 と PP=2 の `GENERATED_IDS`
が一致（30〜31 token）。`required` 168/168・`gpu2` 6/6。

stage の配分（1943 token case）: `stage0 layers=[0,20)` / `stage1 layers=[20,40)`、
各 weight_bytes 11,003,576,320 B、kv_pages 129862、auto reserve 0.78 GiB
（row bucket 1944、slack 0.20 GiB 含む）。full_attention_interval 4 の checkpoint なので
どちらの stage にも full attention が 5 層ずつあり、`pipeline stage has no full-attention
layer` は発火しない。

throughput と容量の表は [../../perf/current.md](../../perf/current.md) が正本。ここで残す
解釈は次:

- decode は PP=2 が 5〜8 % 遅い。schedule が 1 batch を stage 順に通す直列で、GPU が同時に
  動く時間がない（stage 時間 + host hand-off 2 回の合計が 1 step の壁時刻）
- prefill は 512 / 1024 で等価、1943 で +19 %。weight traffic は stage ごとに半分へ寄るが、
  直列のままだと利きの半分も出ない。stage 間 overlap（microbatch）を入れたとき prefill は
  最大 2 倍の方向へ動くが、未実施
- PP の実利は容量。PP=1 で 9.75 GB だった KV pool が PP=2 では stage 每 20.75 GB（2.13 倍）。
  1 GPU に weight が入らない checkpoint（27B FP8 級）を MoE でやる場合の選択肢になる
- dense checkpoint 側は PP=2 の起動と生成を確認しただけ（warmup + 複数 run は未実施）

## 6. 残作業

- stage 間 overlap（microbatch / 1F1B）。現行は直列で、prefill の利きが半分も出ていない
- 3 stage 以上の実機検証（この環境は GPU 2 枚）
- TP と PP の同時利用（expert 範囲 × layer 範囲）。MoE では expert 軸 TP が既に速いので、
  PP の overlap が無い限り優先度は低い
- dense PP の prefill / decode 計測
- KV page 数を stage 0 の auto-fit に合わせる制約（非対称分割や stage 每の page 数指定は未対応）
