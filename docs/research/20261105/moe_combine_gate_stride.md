# MoE combine の shared expert gate 平面 stride 欠落（multi-row forward 不一致の原因）

対象: `models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4`（40 層、`experts 256`、`top_k 8`、`hidden 2048`、
`moe_inter 512`、`shared_inter 512`、`full_attention_interval 4`）。
実験 ID `MOE-COMBINE-GATE`、結論 **ACCEPT（kernel 修正を採用。行数一致は required test で固定）**。

## 1. 着手時点の状態

[./moe_mtp.md](./moe_mtp.md) §4 で、grouped expert を使う model の forward が batch 行数に依存して
変わることを実測した（8 行 vs 1 行で logits rms 1.50、2 行 vs 1 行で rms 1.04、dense 対照は 0.076）。
切り分けは未実施だった。kernel 単体（`test_moe_layer_e2e`、`test_moe_program_execute`）は
reference chain と bit 一致で通っていたため、plumbing 側という推定が残っていた。

## 2. 実測による局所化

### 2-1. `shelf-bench quality` の行はすべて 1 行 step だった

`quality` の scoring loop は token を 1 行ずつ DECODE class で feed する
（`--positions K` は K 回の 1 行 batch）。§4 の「multi-row batch vs single-row」は実際には
**prefix を処理した prefill の行数・chunk 分割の違い**だった。この読み替えが最初の収穫で、
「decode step 同士の比較では差が出ない」という制約が得られた。

### 2-2. layer 別 router logits の hash 突合

`MOE_ROUTER` dispatch で router 入力（f32 logits、TOKEN_ROWS）を host に読み、
行ごと FNV hash して walk の間で突合した（環境変数による一時的な計装、本修正には含めない）。

| 比較 | 結果 |
| --- | --- |
| prefill 134 行と prefill 135 行の共有 row（position 1..134）× 40 層 | **全 layer・全 row で bit 一致** |
| 同一 token（position 135 の token）を 1 行 decode で処理 vs 135 行 prefill の最終 row で処理 | **layer 0 から違う** |
| prefill 5 行 vs prefill 4 行の共有 row | row 0 は全 layer 一致、**row 1..3 は layer 1 以降で rel 5.6e-2〜9.5e-1** |
| prefill 3,4,6,7,8,12,16,24,32,48 行の比較 | 共有 row は全 layer bit 一致 |

prefill 内の row が prefill 長に依存しないなら backbone・router・KV は行数非依存である。
layer 0 の router 入力は embedding と layer 0 の attention 出力だけで決まるので、
「同一 token・同一 prefix」で layer 0 の router logits が一致し layer 1 で不一致、は
**layer 0 の FFN 出力で差が作られている**ことを意味する。

上の表で「共有 row が全 layer bit 一致」になる case と「row 1 以降で rel 1e-1 級」になる case が
混在していたのは、読んでいた bytes が workspace の **padding（未初期化）**だったためである。
`[rows, 1]` の f32 平面（row stride 4）を packed で読むと、token 0 は自分の gate を読むが、
token 1..3 は token 0 の padding を、token 4..7 は token 1 の slot を読む、という形で
**plane を斜めに渡る**。padding が 0 の walkthrough では shared expert 項が単に消え、
walk 同士で同じ bytes を読むので bit 一致してしまう（8 行 vs 16 行が一致していたのはこれ）。
padding に他の layer の残骸が残っている walkthrough では行ごとに違う値を読むので walk 間で割れる。
つまり一致する case があっても正しいわけではない。

### 2-3. combine の入出力を host で再計算して突合

`MOE_COMBINE` dispatch で、kernel が読むものそのもの（plan の `slot_row_of` が指す
group row の expert 出力、`sel_weight`、shared expert 出力 row、shared gate 値）を host に読み、
kernel と同じ順序（group row 昇順、f32 fma、shared 項を最後に足して 1 回 bf16 へ丸める）で
再計算した。修正前の実測（prefill 4 行、layer 0）:

| token | expert 行の hash | shared 行 | gate（stride 4 で読んだ値） | kernel 出力 | host 再計算 |
| ---: | :--- | :--- | ---: | ---: | ---: |
| 0 | 一致 | 一致 | 0.086323 | `-0.390037732` | `-0.390037732` ✓ |
| 1 | 一致 | 一致 | 0.061876 | `0.324647784` | `0.270568535` ✗ |
| 2 | 一致 | 一致 | 0.060087 | `0.668298453` | `0.738694742` ✗ |
| 3 | 一致 | 一致 | 0.069542 | `-0.157268286` | `-0.131691158` ✗ |

token 0 だけが一致し、token ≥1 は kernel 出力が host 再計算と違う。同じ計装で
**gate を packed（添字 1 刻み）で読んだ再計算**を行うと、kernel 出力と token を問わず
完全一致した（`host_sum_packed` = `out_sum`、token 1 では gate_packed = 0.0）。
`HostResolvedValue.row_stride` を print すると shared gate 平面は **row stride 4** で
（`[rows, 1]` の f32 value に対する workspace binding の padding）、kernel は

```cpp
gate = shared_gate_weight[t];   // stride 1 想定
```

で読んでいた。`launch_moe_combine` には `shared_row_stride` はあっても
gate の stride が無かった。producer（`LINEAR_BF16 [1,H]` → `SIGMOID`）は binding の
stride で書くので、token 0 以降は padding の bytes（0 のことも他 layer の残骸のことも）を
gate として使っていた。

### 2-4. 修正後

- kernel は `shared_gate_weight[t * shared_gate_row_stride]` で読む
- 合成 fixture（experts 4 / top_k 2 / hidden 256、2 層）で 12 通りの chunk walk を 1 行 walk と
  突合すると、**全 case・全 row で rel_l2 = 0（bit 一致）**。GDN 層を含む fixture と
  全 layer full attention の fixture の両方
- `test_moe_combine` の padded gate case（rows 8 / top_k 8 / shared、gate stride 4）は
  修正前の kernel では FAIL、修正後は PASS することを確認した（回帰ゲートとして required）

## 3. 実 model への影響（teacher-forced logits、同じ token・position を別行数で比較）

vocab 248320。行は dump の position で突合。

| 比較 | 修正前 rms Δ | 修正前 top-20 | 修正後 rms Δ | 修正後 top-20 | argmax |
| --- | ---: | ---: | ---: | ---: | :--- |
| (context 128 + 8 步) vs (context 135 + 1 步) @ pos 136 | 1.50 | 10/20 | **0.100** | 19/20 | 一致 |
| (context 134 + 2 步) vs (context 135 + 1 步) @ pos 136 | 1.04 | 13/20 | **0.179** | 19/20 | 一致 |
| (context 124 + 12 步) vs (context 135 + 1 步) @ pos 136 | 0.90（12 vs 16 で 0.90） | 13/20 | **0.172** | 20/20 | 一致 |
| (context 128 + 16 步) vs (context 124 + 12 步) @ pos 136 | — | — | **0.174** | 19/20 | 一致 |
| (context 120 + 16 步) vs (context 128 + 16 步) @ pos 135 | — | — | **0.131** | 19/20 | 一致 |
| (context 128 + 8 步) vs (context 128 + 16 步) @ pos 129,131,133,135 | — | — | **0.0（bit 一致）** | 20/20 | 一致 |
| dense 対照（`Qwen3.8-27B-Quark-AWQ-MXFP4`）8 行 vs 1 行 @ pos 132 | 0.076 | 20/20 | 0.076（変更なし） | 20/20 | 一致 |

logits 自身の rms は pos 136 で修正前 2.588、修正後 3.467 で、**モデルの出力値そのものが変わった**
（prefill の 2 行目以降から shared expert の寄与が抜けていた）。
相対では修正後 2.9〜5.2 %、dense 対照 3.2 % で同じ規模になった。

## 4. 残る行数依存（残課題、本修正では消えない）

修正後も walk 間に rms 0.10〜0.18 の差が残る。発生源は GDN である。
同一 token・同一 prefix を「(c-1) 行 prefill + 1 行 decode」で処理した場合と
「c 行 prefill の最終 row」で処理した場合の layer 0 router logits の差:

| 経路 | layer 0 の router \|sum\| 相対差 |
| --- | ---: |
| prefill 134 行 + decode vs prefill 135 行 | 3.3e-5 |
| prefill 4 行 + decode vs prefill 5 行 | 6.0e-6（c=2）〜 1e-4 程度 |

chunked prefill と stepwise decode では GDN の加算順序が違うためこれは丸め規模である。
MoE では router の top-k 選択が離散的なので、この 1e-5 級の入力差が 8 番目 / 9 番目の
expert 選択を反転させ、layer 1 以降で rel 1e-2〜3e-1 まで増幅する
（実測: 40 層中 39 層で selected expert 集合が変化する）。
dense では同じ入力度の差が logits rms 0.076（3 %）として滑らかに現れる。
つまり MoE の greedy token 列は walk の取り分に対して本来の意味でカオスであり、
**bit 一致を要求する acceptance は MoE には立てられない**。
合成 fixture では expert 数が 4 / top_k 2 で選択境界が裕長なため bit 一致する。

## 5. 実 model の teacher-forced quality への影響

この bug は **prefill の 2 行目以降**に効くので、`quality` では context prefill が汚染され、
その KV を読む全 position の logits が狂う。実測では context が短いほど影響が大きく、
`QUALITY_PPL`（bf16 KV・context 128 / positions 64）は 75.146 から 9.7973 になった。
context 512 と 2048 でも同方向の改善がある。数値の正本は
[../../perf/current.md](../../perf/current.md) の MoE quality 節（修正後に再実測した値）に置く。

fp8 KV と bf16 KV の比較（同じ修正後の forward）は、修正前に観測されていた
「fp8 KV が context 128 で ppl +25.6 % 劣化する」という現象が消え、
bf16 と同じ帯（ppl 差 -0.45 %）に入っている。MoE の短 context fp8 劣化は
KV 量子化ではなく combine の bug だった。

## 6. 採否判断

- `launch_moe_combine` の `shared_gate_row_stride` 追加を **ACCEPT**。
  kernel の contract（1 列 value でも plane は row stride を持つ）を呼び出し側が渡す形にした。
  同じ形の欠落が他にあるかは `shared_gate` を読む kernel を grep して combine 以外に無いことを
  確認した
- 行数をまたいだ forward 一致は `test_qwen35moe_rows_parity`（required）で固定した。
  [./moe_mtp.md](./moe_mtp.md) §4 で「未受理」としていた gate はこれで受理条件に置き換わる
- [./moe_mtp.md](./moe_mtp.md) §5 の「MoE の MTP decode は REJECT」は、本修正によって
  前提（multi-row forward の不一致）が消えたため再評価した。再評価結果は
  [./moe_mtp.md](./moe_mtp.md) §7 に記載

## 6. 再現手順

combine の stride 回帰は新規 code なしで再現する。

```bash
# 合成 fixture の行数一致 (required test)
ctest --test-dir build -R test_qwen35moe_rows_parity --output-on-failure

# 実 model の walk 突合
for spec in "128 8" "128 16" "124 12" "134 2" "135 1" "120 16"; do
  set -- $spec
  ./build/shelf-bench quality --model-dir <MoE> --tokens-file <corpus> \
    --context $1 --positions $2 --device 0 --dump /tmp/moe_q_$1_$2.psq
done
```

`PSQUAL01` header 32 byte + 行ごと `int32 target`、`uint32 position`、`float logits[vocab]`。
position で突合して logits ベクトルの rms 差を見る。
