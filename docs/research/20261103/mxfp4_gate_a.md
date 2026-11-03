# Gate A — MXFP4 format 受理と correctness path 実行記録

設計案は [qwen36_moe_mxfp4_contract_design.md](qwen36_moe_mxfp4_contract_design.md)、
構造実測は [qwen36_35b_a3b_moe_structure.md](qwen36_35b_a3b_moe_structure.md)。
実装済みの contract は `docs/developer/canonical_soa_payload.md` §6bis と
`docs/developer/quantization.md` に現在形で移動済みであり、ここには経緯と採否判断だけ残す。

## 1. 実装範囲（Gate A）

MoE に触れる前に、Quark MXFP4 の dense path だけを通した。

- format 層: `QuantFormatId::Mxfp4Finegrained`、`QuantizedEncoding::Mxfp4`
  （on-disk 名 `mxfp4_e2m1_e8m0_g32` / layout `mxfp4_e2m1_row_major_block32_v1`）、
  `QuantSpecId::MXFP4_E8M0_G32_V1`、`ComputeSpecId::MXFP4_W4A8_F32_BF16`。
  metadata は既存 `MetaCountMode::PerBlock` + `element_bytes 1` で表現でき、
  新しい mode は要らなかった。
- reference codec: `quantization/mxfp4/mxfp4.{h,cpp}`（E2M1 encode / decode、
  E8M0 byte / float、group 32 quantize / dequantize、canonical view）と
  device 側 `codec/e2m1_detail.h`。
- HF 受理: `quant_method == "quark"` の scheme 検証と
  `<base>.weight` + `<base>.weight_scale` 名前寄せ（`hf_manifest.cpp`）。
- weights 層: `MatrixEncoding::Mxfp4` と `load_quantized_view` の MXFP4 分岐
  （scale byte 255 拒否、`k_padded % 64`、`rows % 16`）。
- runtime: `KernelId::LINEAR_MXFP4_W4A8`、`resolve_linear_quant_contract` の 1 行、
  `validate_host_weight_slot` の enc==3、correctness path `exec_linear_mxfp4`。
  optimized launcher は追加せず `try_launch_optimized` の NotApplicable 経由で
  correctness path に落とす。
- TP partition: MXFP4 は `Status::unsupported` を明示（暗黙に素通ししない）。

## 2. codec の bit 一致検証（実 checkpoint に対して実施）

`models/Qwen3.8-27B-Quark-AWQ-MXFP4` の payload を実読みし、
**decode → re-encode** して元の bytes と比較した。

| tensor | logical shape | codes 一致 | scale 一致 |
| --- | --- | --- | --- |
| `layers.0.mlp.gate_proj` | [17408, 5120] | 全一致（44,564,480 byte） | 全一致（2,785,280 byte） |
| `layers.7.mlp.down_proj` | [5120, 17408] | 全一致 | 全一致 |
| `layers.2.linear_attn.in_proj_a` | [48, 5120] | 全一致 | 全一致 |
| `layers.0.linear_attn.out_proj` | [5120, 6144] | 全一致 | 全一致 |

これは nibble 順（下位 = 偶数 K）、E2M1 値表、`scale = 2^(byte-127)`、
group 32 の境界が artifact と一致していることの直接証拠である。
`test_mxfp4_codec` には実測 size 定数（codes 524288 B / scales 32768 B など）を
そのまま検証条件として埋め込んだ。

### 判明した丸め仕様（採否に影響）

生成元 packer の code 丸めは **round-half-away-from-zero** で、
MX spec 文献の round-half-to-even ではない。正確な midpoints で差が出る。

```text
0.25 -> code 1 (0.5)    1.25 -> code 3 (1.5)
2.5  -> code 5 (3.0)    5.0  -> code 7 (6.0)
```

- 復元（decode）側は無関係だが、fixture 生成や再量子化で artifact と食い違う。
- 採用: packer と同一の bit trick を reference codec に実装し、
  上記 4 点を `test_mxfp4_codec` で pin した。
  将来 RNE へ替える実装はテストで落ちるようにする。
- scale 側（E8M0）は `even_round`（amax の仮数を 1 bit へ丸めてから指数を取る）。
  `amax == 0` は byte 1（2^-126）、subnormal は byte 0、
  amax ≳ 1.75·2^127 は f32 仮数丸めが +inf へ溢れて byte 254。
  この 3 境界を C++ 実装と独立実装で 300k サンプル突合し、不一致 0 を確認した。

## 3. correctness path の E2E 実行（実 checkpoint）

`SHELF_QWEN35_KERNEL_MODE=correctness shelf-compute --model-dir models/Qwen3.8-27B-Quark-AWQ-MXFP4`

| 項目 | 実測 |
| --- | --- |
| target weights | 16.79 GiB |
| KV budget（auto） | 12.06 GiB（197,616 token） |
| arena capacity | 30.78 GiB（32 GiB board 1 枚） |
| load | 7.36 s（初回）/ 1.66 s（page cache warm） |
| prefill 18 token | 7.09 s |
| decode | 1.44 s / token |

生成は 18 token prompt に対して 48 token まで破綻せず、
`<think>` 進入後 Rayleigh 散乱について正しい内容の文章を続けた
（壊れた weight なら退屈な繰り返しに collapse する）。
**数値は correctness path のみ。性能主張ではなく `docs/perf/current.md` には載せない。**

## 4. GPU 単体テストの実測

`test_mxfp4_linear`（`gpu1;required`）: n=32, K=128, tokens=5 で
activation codes / scales は GPU が workspace へ書いたものを read back し、
double 参照と突き合わせた。

```text
bf16 exact elements 160 / 160   rel_l2 = 1.597e-03   max_abs = 1.950e-03
```

`rel_l2` は bf16 storage の丸めレベル（約 2e-3）で、
bit 一致側の条件（bf16 量子化後に全一致）を主契約として置いた。
weight scale group / codes stride / k_padded / encoding の contract 違反は
device 側で `INVALID_PROGRAM` になることも同時に固定した。

## 5. 採否と残課題

- ACCEPT: MXFP4 native layout を当面 **canonical と同一の row-major** とする。
  codes / scale の byte 変換が無いので load は H2D のみ、
  VRAM も 0.5 B/weight + 1 B/32 weight のまま。
  16 行 tile に載らない rows（例: 48 より小さい奇数 tile）は
  pad を作らず unsupported にした。実 checkpoint の全 tensor は 16 の倍数。
- REJECT（現時点）: codes を E4M3 へ展開した native layout。
  resident が 2 倍（routed experts 15.938 GiB → 31.9 GiB）になり
  VRAM 条項に反する。exponent fold は register / LDS 内にとどめる。
- 未実装: `LINEAR_MXFP4_W4A8` の optimized kernel。
  MXFP4 model の速度は correctness path 起動のままで、
  kernel 実装時に fragment 順 native layout と layout id 更新を同時に決める。
- 未実装: MXFP4 の TP partition（`canonical_partition.cpp` で明示 unsupported）。
- Gate B 以降（MoE weights 集約 / router / grouped linear）は未着手。
