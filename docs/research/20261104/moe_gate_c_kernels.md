# Gate C 実施記録: MoE semantics kernel（routing / grouped linear / combine）

設計案は [20261103/qwen36_moe_mxfp4_contract_design.md](../20261103/qwen36_moe_mxfp4_contract_design.md)、
weights 集約は [moe_gate_b_weights.md](moe_gate_b_weights.md)。
現在形の contract は `docs/developer/qwen35moe.md` §5 と `docs/developer/kernels.md` へ移した。
以下に経緯・採否・実測だけを残す。

## 1. 実施範囲

新規 kernel（standalone launcher。`KernelId` / selector / graph 配線はまだ無い）:

| 実装 | 内容 |
| --- | --- |
| `kernels/optimized/moe/router.hip` | f32 softmax → top-k → renormalize、expert 別 histogram、group table、slot → group row |
| `kernels/optimized/moe/combine.hip` | top_k 個の group row を expert 昇順に f32 蓄積、shared expert を sigmoid gate 付きで最後に加算 |
| `kernels/optimized/linear/mxfp4.hip` の grouped 形 | rank-3 expert 平面を group table 経由で読む MXFP4 W4A8 WMMA GEMM |
| `launch_activation_quantize_mxfp4_rowref` | `row_index` 間接参照を追加（group row 順の activation 平面生成、padding 行は zero） |

受け入れ test（`gpu1;required` に登録）: `test_moe_router` / `test_moe_grouped_linear` /
`test_moe_combine`。required acceptance は **144/144 PASS**（cpu 88 / gpu1 56）。

## 2. 採否: group row 番号でアドレスする（採用）

設計案 §4 は grouped linear の入力を「expert 別の token 範囲」と書いていたが、
activation 平面は FP8 W8A8 と共有する fragment order（16 行 × 64 K の block を 16 行分
連続で並べる）を使うため、**任意の slot 範囲を平面へ並べると 16 行 tile がまたがって**
GEMM が連続 load できなくなる。

そこで plan の表す行番号を slot 番号ではなく **group row 番号**にした。

```text
group g は expert をまたがない 16 slot 以下の区間
group row r = g * 16 + (group 内の offset)
activation 平面 / grouped GEMM 出力 / w2 入力 / combine の読む行 はすべて r
```

副作用として契約の順序条件は満たされたまま強くなった。group は expert 昇順に作る
（同一 expert 内は token 昇順）ので、**group row 昇順 = expert 昇順** であり、combine は
token の `top_k` 個の group row を sort するだけで契約の集約順序を守る。
`slot_row_of[t * top_k + j]`（token の j 番目の選択の group row）と `row_token[r]`
（group row の token。padding は -1）の 2 本が plan の外部形式で、
`expert_row_begin` / `expert_group_begin` / `group_expert` / `group_rows` / `group_count`
が group table である。

容量は host 側单一の式で決める。

```text
moe_group_count_upper_bound(rows, experts, top_k)
    = min(rows * top_k, rows * top_k / 16 + experts)
```

## 3. 実測: padding による量子化行数の膨らみ

grouping は expert ごとに 16 行へ pad するため、activation 量子化の行数は slot 数より
多い。`test_moe_router` の実行値（experts 256、top_k 8、および experts 8 の小断面）:

| 条件 | slot 数 | group 数 | group row 数 | 量子化行数 / slot 数 |
| --- | ---: | ---: | ---: | ---: |
| E=8 / top-2 / rows=1 | 2 | 2 | 32 | 16.0 |
| E=256 / top-8 / rows=1 | 8 | 8 | 128 | 16.0 |
| E=256 / top-8 / rows=7 | 56 | 50 | 800 | 14.3 |
| E=256 / top-8 / rows=64 | 512 | 191 | 3056 | 5.97 |
| E=256 / top-8 / rows=65 | 520 | 228 | 3648 | 7.02 |
| E=256 / top-8 / rows=512 | 4096 | 362 | 5792 | 1.41 |

decode（rows が小さい）では量子化行数は slot 数の最大 16 倍になるが、その絶対値は
rows=1 で 128 行、rows=64 で 3056 行と小さい。prefill（rows=512）では 1.41 倍に収まる。
**この膨らみが量子化 launch のコストとして E2E にどう出るかは未測定**（計測口が無い）。
代替案（token ごとに 1 回量子化し、fragment order を保ったまま group row へ複製する
byte shuffle kernel）は棄却していない。Gate D で計測して決める。

## 4. 実測: 正しさ

### 4.1 router / group plan（`test_moe_router`）

- E = 8 / top-2 と E = 256 / top-8、rows = 1, 7, 64, 65（token tile 64 の境界をまたぐ）, 512。
- 選択 expert index は double 参照と**完全一致**。renormalize 重みの f32 / double 最大
  相対誤差は 1.35e-8（E=8）〜 2.39e-7（E=256 / rows=512）で、gate は相対 1e-5。
- 同値の tie（logits を 3 か所完全に一致させる）で**小 expert index が先に勝つ**ことを固定。
- 不変条件: slot 範囲が expert を分割、group は expert をまたがない、
  1 expert の最終 group 以外はちょうど 16 行、同一 expert 内は token 昇順、
  padding 行は `row_token == -1`、`slot_row_of` は claim された group row への全単射、
  token の group row 昇順走査が expert 昇順と一致。
- 同一入力 2 回の実行で plan 配列がすべて **bit 一致**（atomic を使うが順序に依存しない）。

### 4.2 grouped linear（`test_moe_grouped_linear`）

Gate B の合成 checkpoint fixture（experts 4、hidden 256、inter 128）を
`load_moe_expert_group` で GPU に載せ、router / plan から group table を作って実行した。
受け入れは **GPU が実際に使った operand（canonical weight codes / E8M0 と、GPU が書いた
E4M3 activation 平面）に対する double 参照**との dtype 誤差。

| 断面 | 結果 |
| --- | --- |
| rows 1 / top 2・4、rows 3 / top 2（skew: 2 expert が空）、rows 16 / 17 / 33 / 64 | bf16 exact **1.0000** |
| rel_l2 | 1.33e-3 〜 1.75e-3（全断面） |
| w13 の fold と unfold | bf16 **bit 一致**（fixture の E8M0 幅が厳密帯域 d ≤ 8 のため） |
| `group_rows[g]` を超える行 | sentinel (bf16 0x7f7f) がそのまま残る＝**store されていない** |
| padding 行の activation | codes すべて 0・scale 1.0（`row_token = -1` を読む経路が無い） |

skew 断面（expert 2・3 が 0 slot）と rows 33 / top 4（1 expert あたり 33 slot → 同一
expert の複数 group）を両方含める。

### 4.3 combine（`test_moe_combine`）

- 乱数 routing（top_k 2・4・8、rows 1〜64、shared expert あり／なし）で、
  expert 昇順に f32 蓄積した host 参照と**出力 bf16 が bit 一致**。
- 順序を実際に判別する工作例: rows = {0,1,2,3} の値が 1, 1, 1e8, -1e8 で
  ascending だと `1 + 1 + 1e8` の 2 が f32 に吸収されて **0**、descending は **2**。
  kernel は 0 を返した（= expert 昇順で蓄積している）。
- shared expert は `sigmoid(gate) * shared` を routed 合計の**後**に加算。
  `shared_output` と `shared_gate` の片方だけの入力は拒否。

## 5. 採否と修正

- **tolerance を grouped だけ rel_l2 < 4e-3 にした**（他の dtype gate は 2e-3 のまま）。
  bf16 の出力丸め自体が相対 2^-9 ≈ 2e-3 で、rows=1 / top_k=2 のように書かれる行が
  2 行しかない断面では「すべての要素が正しく丸まっている」状態の RMS が 2e-3 を超える
  （実測 2.43e-3、bf16 exact は 1.0000）。primary gate は bf16 exact 比。
- **fixture の E8M0 高点 byte を 254 → 123 に下げた**。254（= 2^127）は w2 の 32 K block
  を 1 つだけ 2^28 倍にして f32 蓄積を ill-conditioned にし、rel_l2 が丸めノイズで
  支配された（inf も出た）。123（基準 112 との幅 11）でも fold 厳密帯域 (≤ 8) を
  外す目的は達成され、`test_moe_expert_load` の w13 exact / w2 non-exact は維持される。
- **model-independent kernel の header に model family 名を書けない**ことを
  `test_architecture_boundaries` が守っていた（`include/shelf/kernels/optimized/**` から
  `qwen35` を禁止）。MoE kernel の contract を contract doc 参照でなく header 内に
  書く形に修正した。
- design 案 §6 の table では `test_moe_router` を `cpu;required` としていたが、
  routing と plan は device 側のみが contract を満たす（決定論・atomic の順序非依存）
  ため **`gpu1;required`** にした。

## 6. 次の Gate で残るもの

`shelf-compute` はまだ MoE を通さない。配線一覧は `docs/developer/qwen35moe.md` §7 に
現在形で書いた（WeightSlot の experts 化、`KernelId` 3 個、selector allowlist の
MoE geometry、workspace の group row 容量、correctness path の grouped 不在、
`shelf-bench` の計測口）。1 layer を oracle と照合する `test_moe_layer_e2e` は
oracle 側（`../references/radiance` の `tiny_moe`、8 experts / top-2 / shared 24）を
使う `external_files` として未実装。
