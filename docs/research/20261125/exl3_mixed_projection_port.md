# EXL3-INT8 3/4-bit 混在 projection の main 取り込み（移植順 2）

## 条件

- base: main `7b4ccb25`
- branch: `exp/exl3-mixed-projection`
- R9700（gfx1201）× 1、ROCm 10.2、Release

## 変更内容

PR #53 の B と I を main に移植した。

- `PlamoExl3ProjectionGroup::three_bit_mask` を追加した。
- `launch_plamo_exl3_projection` の `group_gemm` を `B == 0` のとき projection ごとに
  `group_body<3>` / `group_body<4>` へ振り分けるようにした。
- `three_bit_mask` は 16 行以下、`allow_shape_fusion == false`、mask が count 未満、
  先頭 bit が 3/4 のときだけ受理する。全 projection が 3-bit の mask は dispatch 側で 0 に戻す。
- `exl3_grouped_projection_bits_match` を追加した。bit 幅が同じ場合は従来どおりで、
  3/4 が混在する場合は 16 行以下だけ許可する。
- `try_launch_exl3_projection` の連続 projection 判定を `w.exl3_ka != weight.exl3_ka` から
  `bits_match` に変え、group 構築時に 3-bit の位置を mask へ記録する。
- 3-bit 先頭の grouping は、この段階では main の `grouped_3bit` 条件
  （`ctx.capturing_graph`）を維持する。移植順 3 で条件を外すかどうかを測る。

## テスト

`tests/kernels/linear/test_exl3_track1_paths.hip` を追加した。5120→512 の 2/3 projection を
3/4 の混在でまとめ、1 回の grouped launch の出力を projection ごとの
`rotate_quantize → mul24_gemm → rotate_out` と BF16 で全要素比較する。

確認したケース:

- rows=1, 8, 16 の 3-bit → 4-bit（mask=1）
- rows=1, 16 の 4-bit → 3-bit（mask=2）
- rows=1 の 3/4/4（mask=1）、4/3/4（mask=2）
- rows=16 の 4/4/3（mask=4）、3/4/3（mask=5）
- rows=17 の混在は `hipErrorInvalidValue`
- mask が count 以上のときは `hipErrorInvalidValue`

結果:

```text
PASS rows=1 3/4 grouped
PASS rows=8 3/4 grouped
PASS rows=16 3/4 grouped
PASS rows=1 4/3 grouped
PASS rows=16 4/3 grouped
PASS rows=1 3/4/4 grouped
PASS rows=1 4/3/4 grouped
PASS rows=16 4/4/3 grouped
PASS rows=16 3/4/3 grouped
PASS rows=17 mixed rejected
PASS mask out of range rejected
ALL PASS
```

`test_exl3_dispatch_decision` には `exl3_grouped_projection_bits_match` の同一 bit / 混在 / 行数境界 /
非対応 bit の判定を追加し、ALL PASS だった。

回帰:

- `test_exl3_mul24_linear`: ALL PASS
- `test_exl3_mul24_grouped`: ALL PASS
- `test_exl3_mul24_tp`: ALL PASS
- `test_exl3_contracts`: ALL PASS
- `test_plamo_exl3_projection`: PASS

## 未計測

- dispatch 経由で実際に何個の projection が grouping されるかは、モデル E2E の起動数ログで
  まだ測っていない。`producer == nullptr`、prefill request なし、16 行以下という条件があるため、
  Qwen4Exp の全層で 6→3 になるわけではない。
- 3-bit 先頭の混在 group は eager decode ではまだ grouping されない。移植順 3 で
  `ctx.capturing_graph` 条件を外したときの効果を測る。
- 3/4 混在の microbench（PR の 1/8/16 行で約 -51%）は main 上で再測していない。
