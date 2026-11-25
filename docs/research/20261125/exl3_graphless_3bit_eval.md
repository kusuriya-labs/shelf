# EXL3 3-bit projection の graph 外 grouping 評価（移植順 3）

## 結論

`grouped_3bit` から `ctx.capturing_graph` を外し、eager decode でも 3-bit projection を
grouping できるようにした。現行のサポート対象 checkpoint ではこの経路は発火しないが、
3-bit dense projection を持つ将来の checkpoint 向けに採用する。

## 背景

`try_launch_exl3_projection` は、同じ入力を持つ連続した `LINEAR_EXL3_INT8` を 1 回の
projection group にまとめる。main は 3-bit の group を HIP Graph の capture 中だけ許可している。

PR #53 の A は、この条件から `ctx.capturing_graph` を外し、eager decode でも 3-bit を
grouping できるようにするものだった。移植順 3 では、c>1 で効果があるかを確認する。

## 3-bit dense projection の分布

Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw の trellis を走査した。

- 3-bit: 73,738 tensor
- 4-bit: 1,778 tensor
- 5-bit: 235 tensor

3-bit の内訳は、routed experts 73,728、MTP の `mtp.fc_hidden` / `mtp.fc_embedding` /
`mtp.layers.0.self_attn.indexer.index_qk_proj` / `mtp.layers.0.mlp.shared_expert.*` である。
language model の dense projection（linear_attn、shared expert）は 4-bit または 5-bit で、
3-bit の dense projection は MTP block に限られる。

routed experts は `launch_gemm_exl3_int8_grouped` の MoE 経路であり、
`try_launch_exl3_projection` の対象ではない。

## 計測

`exp/exl3-graphless-3bit` を main `cd861353` から作り、`grouped_3bit` の
`ctx.capturing_graph` 条件を外した。`try_launch_exl3_projection` に一時的な
`SHELF_EXL3_GROUP_DEBUG` 出力を追加し、eager で 3-bit group が成立した回数を数えた。

条件:

- Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw
- EP=4、4 GPU、MTP draft 4、8 同時要求
- `tests/server/test_compute_mtp_batch.py`

結果:

- `EXL3_GROUP_3BIT_EAGER` の出力は **0 回**
- MTP batch test は 11/11 合格
- 3-bit dense projection は MTP block のみで、その経路はこの workload では A の条件に
  到達しなかった。A による eager grouping が 0 回のため、速度比較は成立しない

## 判断

現行モデルでは計測上の改善はない。それでも、3-bit dense projection を持つ checkpoint で
`try_launch_exl3_projection` の同一 bit 幅の grouping を HIP Graph の capture 外でも使える
ようにするため、A を採用する。

- カーネル側の 3-bit group は `launch_group_bits<3, 1, true>` で既に実装されており、
  `test_exl3_track1_paths` が 3/4 混在の出力一致を確認している。
- 現行モデルで発火しないことは、既存経路の回帰がないことを意味する。
- graph capture の workspace / graph key への影響は、graph 外の eager 経路だけに限定される。

将来、3-bit の dense projection を language model 本体に持つ checkpoint（Hy4 系など）を
導入したときは、同じ debug counter と c>1 の計測で効果を確認する。
