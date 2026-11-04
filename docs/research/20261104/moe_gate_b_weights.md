# Gate B 実施記録: MoE weights 集約（`shelf::qwen35moe`）

設計案は [20261103/qwen36_moe_mxfp4_contract_design.md](../20261103/qwen36_moe_mxfp4_contract_design.md)、
checkpoint 構造の実測は [20261103/qwen36_35b_a3b_moe_structure.md](../20261103/qwen36_35b_a3b_moe_structure.md)。
現在形の contract は `docs/developer/qwen35moe.md` へ移した。以下は経緯と採否だけ。

## 1. 実施範囲

Gate B の受け入れ（設計案 §7）どおり、次の 4 test を追加して required へ登録した。

| test | label |
| --- | --- |
| `test_moe_config` | `cpu;required` |
| `test_moe_weight_plan` | `cpu;required` |
| `test_moe_expert_slice_fixture` | `cpu;required` |
| `test_moe_expert_load` | `gpu1;required` |

required acceptance は 141/141 PASS（内訳 cpu 88 / gpu1 53）。

## 2. 採否: MoE contract を `qwen35moe` module へ分離（採用）

設計案 §3 は `Qwen35TextConfig` へ MoE field を追加する形だった。実装では
`shelf::qwen35moe`（`src|include/shelf/models/qwen35moe/`、CMake target `shelf_qwen35moe`）
を新設し、`Qwen35MoeTextConfig`（基底 `Qwen35TextConfig` + expert 幾何）と
`read_qwen35_moe_text_config` をそこへ置いた。

- 理由: checkpoint の `model_type` は `qwen3_5_moe` であり、dense family の config に
  MoE の受理条件を混ぜない。`shelf::qwen35` 側は MoE field を持たない（fb2329f で追加した
  `num_experts` 群と `is_moe()` は revoke した）。
- `model_type` 文字列での分岐はしない、という方針は維持した。MoE かどうかは呼び出し側が
  どの reader を使うかで決まり、その判定材料は `num_experts > 0` のまま
  （MoE reader は `num_experts == 0` を `invalid_argument` で拒否する）。
- 副次効果として `output_gate_type` の既定を dense 側で変えずに済んだ。
  key 省略時に silu を選ぶ規則は Qwen3.5 MoE checkpoint（key が無い）に対する規則として
  `read_qwen35_moe_text_config` の中に入れた。dense 側は key 省略時 sigmoid の従来契約のまま。

## 3. 採否: plan の source は projection 単位（採用）、physical tensor 単位（棄却）

当初は plan の source を 1 expert あたり 6 本（`down_w / down_s / gate_w / gate_s /
up_w / up_s`）として、scale tensor 名を `QuantizedModelReader::resolve` に渡す形にした。
これは HF manifest 合成経路の契約と矛盾した。合成 manifest で logical tensor として
登録されるのは `<name>`（codes）だけで、`<name>_scale` はその entry が Ref として指す
**物理 tensor** である。合成 fixture を使った `test_moe_expert_slice_fixture` が
「scale を logical として resolve できる」という期待ごと落ち、この事実が確定した
（`resolve(...weight_scale)` は `invalid argument: logical tensor not found`）。

そこで source を 1 expert あたり 3 本（`down_proj` / `gate_proj` / `up_proj`）にし、
1 resolve で codes 平面と E8M0 scale 平面（`metadata1`）の両方を取り出す形にした。

- 副作用として resolve 回数が 1 layer 1536 → 768 に半減した（loader の CPU 側 API 呼び出し）。
- 物理名の並び（expert-major で down → gate → up、codes → scale）は file order contract として
  test 側に残した。plan が枚举するのは logical、shard 上に連続なのは physical、の区別を
  `test_moe_expert_slice_fixture` が両方から固定する。

## 4. 実測アンカー

### 4.1 plan（Qwen3.6-35B-A3B-Quark-AWQ-MXFP4 の config、`test_moe_weight_plan`）

| 項目 | 値 |
| --- | ---: |
| w13 1 expert codes batch stride | 1,048,576 B |
| w13 1 expert scales batch stride | 65,536 B |
| w2 1 expert codes batch stride | 524,288 B |
| w2 1 expert scales batch stride | 32,768 B |
| 1 layer の expert 平面合計 | 427,819,008 B |
| 40 層合計 | 17,112,760,320 B |
| 1 layer の logical source 数 | 768 |
| 1 layer の physical tensor 数 | 1,536 |
| 全 routed expert tensor 数 | 61,440 |

1 layer 合計と 40 層合計は `qwen36_35b_a3b_moe_structure.md` の checkpoint 実測
（1 layer の expert payload が連続 427,819,008 B）と一致する。native preshuffle は
payload を増やさない（codes は 16 行 tile に並べ替えるだけで byte 数不変、scale は
row-major のまま）ため、plan の確保量と disk 上の canonical 量が同一である。

### 4.2 load（合成 fixture、experts 4 / hidden 256 / inter 128 / layers 2）

| 項目 | 値 |
| --- | ---: |
| 1 layer の arena 使用量 | 210,944 B |
| 内訳 | w13 codes 131,072 + w13 scales 8,192 + w2 codes 65,536 + w2 scales 4,096 + row_ref 2,048 |
| H2D 転送数 | 6（平面 2 本 × codes / scales / row_ref）。experts 数に依存しない |
| 1 expert あたり preshuffle | 1 回（batch = experts で 1 呼び出し） |

`test_moe_expert_load` が固定する受け入れ:

- GPU 常駐平面が native block layout の定義式（16 出力行 × 32 K = 256 B block、
  `lo = code[k]` / `hi = code[k+16]`）から独立に再計算した byte と一致
- expert e の平面は、その expert だけを rank-2 として preshuffle した結果と bit 一致
  （rank-3 化が dense path の byte 列を変えないことを固定）
- `row_ref_exact` は平面別（w13 = true、w2 = false）。E8M0 byte の row 内 span が
  fold の exact 帯（距離 8）に入るか否かを合成 payload で両側から確認した
- dead row（codes 0、E8M0 byte 1）の row reference が有限 byte のまま残る
- E8M0 byte 255 を混ぜると `invalid_argument`
- arena 使用量が plan の平面サイズ合計と一致（余分な確保・コピーが無い）

## 5. 残った穴（Gate C 以降）

`shelf-compute --model-dir models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4` は Gate B の後も
同じ場所で失敗する。

```
runtime create failed: src/shelf/quantization/bundle/quantized_model_reader.cpp:559:
invalid argument: logical tensor not found
```

原因は `models/qwen35/weights/model_weights.cpp` の logical tensor 名列挙に expert 名が
無いこと（`hf_quantized_logical_names` は dense FFN 名のみ）と、`architecture !=
"qwen35_dense"` を拒否していること。`load_moe_expert_group` には production からの
呼び出し元がまだ無い。Gate C で router / grouped linear / combine と WeightSlot の
experts 化、selector allowlist 追加までを行って初めて 1 layer が oracle と照合できる。

MXFP4 の TP partition は rank-3 で unsupported のまま（`canonical_partition.cpp`）。
現行の top-level load path は partition plan 自体を拒否するので影響は無い。

host 側の一時領域は peak RSS を未測定だが、plan のサイズから算術的に次と決まる
（実測ではなく plan 定義から求まる値）。`PlaneDraft` は w13 / w2 を両方 host に保持した
まま native 平面を作るため、1 layer の host 使用量は

| host buffer | 1 layer |
| --- | ---: |
| draft codes（w13 256 MiB + w2 128 MiB） | 384 MiB |
| draft scales（w13 16 MiB + w2 8 MiB） | 24 MiB |
| native codes / scales | 408 MiB |
| native row_ref | 0.75 MiB |

で 0.8 GiB 級になる。40 層を順に load する場合の host 上限・H2D 時間・load 総時間は
Gate D で実測するまで主張しない。
