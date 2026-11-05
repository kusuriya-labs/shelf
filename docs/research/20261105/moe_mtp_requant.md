# MoE MTP expert の MXFP4 再量子化誤差と draft 受理率

実験 ID `MOE-MTP-REQUANT`、結論 **REJECT（simple な再量子化改善では誤差を下げられない。
draft 受理率への寄与は本記録では確定できない）**。

対象: `models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4` の MTP block。checkpoint では
`mtp.layers.0.mlp.experts.gate_up_proj` `[256, 1024, 2048]` と
`experts.down_proj` `[256, 2048, 512]` が **BF16** で、Shelf は load 時に
phase-2 codec（E8M0 group 32 + E2M1、round-nearest）で MXFP4 へ再量子化して
transformer layer と同じ native plane に落とす（contract は
[../../developer/qwen35moe.md](../../developer/qwen35moe.md) の MTP block 節）。

## 1. 動機

[./moe_mtp.md](./moe_mtp.md) §6 の再評価で、MoE の MTP 受理率は
mean accepted 0.875 / 2.444 / 2.444（drafts 1 / 4 / 7）で、dense 対照
（`Qwen3.8-27B-Quark-AWQ-MXFP4`、accepted/7 = 6.88 = 98 %）に対し 35 % しかない。
dense 27B の MTP block は FFN まで BF16 で保持され、再量子化も activation 量子化も通らない。
MoE 側は expert weight の再量子化（推定 rel 11 %）と grouped path の activation 量子化
（MXFP4 rowref）が dense に無い誤差項なので、これを削れるかが option として残った。

## 2. 実測: 再量子化の weight 誤差

numpy で Shelf の codec と同じ bit 操作（`e8m0_scale_byte` の RNE power-of-two/4、
E2M1 の round-nearest、group 32）を再現し、checkpoint の BF16 tensor を直接量子化して
rel RMS 誤差を測った。expert は sample `[0, 1, 128, 255]`、行は各 expert の
行 0 / 7 / 1000 の 3 本。

| tensor | 現状の RTN | scale を `ceil(log2(amax/6))` | 丸め残差を group 内で後続へ流す error feedback |
| --- | ---: | ---: | ---: |
| `mtp.layers.0.mlp.experts.gate_up_proj` | **0.1133** | 0.1174 | 0.1606 |
| `mtp.layers.0.mlp.experts.down_proj` | **0.1070** | 0.1099 | 0.1506 |
| `mtp.fc.weight`（参考、BF16 で保持されている） | 0.1343 | 0.1375 | 0.2005 |
| `mtp.layers.0.self_attn.q_proj.weight`（参考） | 0.1135 | 0.1173 | 0.1599 |
| `mtp.layers.0.mlp.shared_expert.down_proj.weight`（参考） | 0.1178 | 0.1271 | 0.1648 |

- 現状の RTN は rel RMS **約 11 %**。E2M1 + E8M0 の格子に対して BF16 のこの分布
  （|w| rms 0.015、group amax は 6.0 格子でほぼ表現できる）は**すでに格子の限界近傍**で、
  単純な規則の変更では縮まない
- scale を `ceil(log2(amax/6))`（clip を起こさない ceiling 基準）へ変えると
  clip は 0 になるが誤差は逆に 0.4〜0.9 % 悪い（scale が 2 倍ステップで粗くなる）
- error feedback（同一 group 内で丸め残差を後続要素へ加算する）は **1.4〜1.5 倍悪化**した。
  E2M1 の格子が粗く、残差を加算した値が次の要素の丸めをさらに崩すため
  （activation 重み付きの GPTQ 系とは違い、weight 単体の局所最適化では縮らない）

## 3. 採否判断

- 再量子化規則の simple な改善（scale 基準変更・error feedback）は **REJECT**。
  上表のとおり誤差が悪化する
- BF16 expert のまま grouped path を走る経路（bf16 grouped GEMM）を足すべきかは、
  **本記録では判断できない**。受理率の差 35 % / 98 % が
  （a）expert weight の 11 % 誤差、（b）grouped path の activation MXFP4 量子化、
  （c）MoE 自体の router 離散性（入力 1e-5 級で expert 集合が反転する事実は
  [./moe_combine_gate_stride.md](./moe_combine_gate_stride.md) §4 で実測済み）、
  のどれで立っているかを分ける計測が無い。分けるには同じ入力・同じ token 列で
  MTP block の forward を BF16 expert と MXFP4 expert の両方へ走らせて
  draft logits の top-1 一致を見る A/B が必要で、それは bf16 grouped GEMM を実装しない限り
  Shelf 側にその経路が無い
- 次にこの論点に戻るなら、順序は (c) を先に潰すこと。
  [./moe_mtp.md](./moe_mtp.md) §6 のとおり、MoE MTP は受理率以前に **verify round の
  forward 外費用 約 40 ms**（2 行 verify 62.2 ms に対し 2 行 forward 単体 22.5 ms）が
  speedup を止めている。draft 精度を上げる work は、この round 費用を落としてからでないと
  性能に反映されない

## 4. 再現手順

`e8m0_scale_byte` / E2M1 の emulation は `tools/moe_layer_oracle_fixture.py` の
`e8m0_scale_byte` / `e2m1_encode` と同一の bit 操作を使った（`safetensors` の BF16 を
uint16 → bit shift で f32 に戻し、group 32 で amax → scale byte → E2M1 最近傍）。
比較した規則は scale byte の決め方（`(bits + 2^21) >> 23 - 2` / `ceil(log2(amax/6))`）と
error feedback の有無。expert `[0,1,128,255]`、行 3 本で測った。
