# activation 行基準 fold の accumulation 形と一致条件（訂正版）

[mxfp4_act_rowref_fold_plan.md](mxfp4_act_rowref_fold_plan.md) §2 / §3 の 2 段目を
詰めたメモ。**初版は「correctness path も同じ accumulator 構造へ合わせる」と書いたが、
それは reference を optimized に合わせ込むので却下**。訂正の要点:

- reference（`exec_linear_mxfp4` など correctness path）は**独立した自然な accumulation
  のまま変えない**。
- optimized kernel の受け入れ条件は、他の GEMM と同じ**dtype 誤差内**（double 参照に
  対する tolerance）に統一する。`test_mxfp4_linear` の既存条件は
  `bf16 exact >= 99.9% / rel_l2 < 2e-3` で、これが repo の慣習側の基準。
  optimized variant と correctness path の bf16 bit 一致は**診断情報としては有用な
  が契約ではない**（初版で私が課した自作条件）。
- 結果として chain 形（K 全体を 1 accumulator へ）は reference を触らずに採用できる。
  一致条件は variant 間の bf16 一致率 / rel_l2 を**実測値で置いて**固定する。

以下は初版からの観測記録（設計導出として有効な部分と、実測事実を残す）。

## 1. 現行が bit 一致している理由（実測で分かっていること）

- optimized（WMMA）は 32 K block ごとに 16 個の積を fresh accumulator で合计し、
  その block 結果を `fma(c, weight_scale * act_scale, acc)` で acc へ足す。
- correctness path（`exec_linear_mxfp4`）は同じ block 単位で 16 個の積を scalar で
  逐次合計した `p` を作り、`acc += p * ws * as` で足す。
- 両者は bf16 で完全一致している（`test_mxfp4_linear` の全 variant、rel_l2 = 0）。
  からして **gfx1201 の FP8 WMMA が zero 初期化 accumulator に対して 16 個の積を
  左から右へ逐次加算する実装になっている**（ Otherwise 一致しない）と解釈できる。
  これは実測で支持されているが、ISA で明文化された事実ではない。
- K を跨いだ split（SplitK の LDS reduce）だけ一致しない（2〜3 要素 / 32768）。
  加算順序が違うためで、bf16 の least bit が動くのはほぼゼロ近傍の要素。

## 2. 案 D のループ形で生じる差

activation を行基準 fold（codes に E8M0 を畳む）し、weight 側も行基準 fold すると、
inner loop の rescale が消える代わりに **K 全体を 1 本の accumulator へ chain する**
形になる（§gate_a_prime §10 の ablation 形）。すると

- optimized: `acc = Σ_K (a_folded · w_folded)` を block を跨いで continuation
- correctness: `acc = Σ_ib (p_ib · ws_ib)`（block ごとに p を切る）

は**同じ加算の集合ではなく**、丸めがズレる。実測でも chain 形（weight 側だけ）を
試したとき bf16 反転 0.5〜1%（max_rel 1 bf16 ULP）が出た
（`mxfp4_gate_a_prime.md` §9.1 の表の前身にあたる実験）。

## 3. 条件（実装で守る形）

correctness path を row-act 形式にするときは、optimized と**同じ accumulator 構造**に
する。具体的には `exec_linear_mxfp4` を row-act 旗（`DispatchBinding::flags` bit 0 =
`kActFormRowRefFolded`）で分岐させ、row-act 側は

```text
acc = 0
for ib in 32 K blocks:
    for j in 16 native codes:
        acc += e4m3_fold_of_e2m1(code, d_weight[ib]) * a_folded_code_value
scale = 2^(w_ref[o] + a_ref[m])          # 2 冪の積は厳密
store acc * scale
```

という **block を跨いで continuation する 1 本の accumulator** にする。要点:

1. 2 冪のスケーリングは operand 側で適用しても、積に適用しても、累算の各ステップで
   厳密（f32 の正規域にいる限り）。だから correctness 側で fold を「operand に畳む」
   形に統一すれば、WMMA の accumulator と同じ丸め系列になる。
2. ただし **E2M1 → E4M3 fold は丸めを含む**（`d >= 9` で 1.5 刻みを丸める）。
   correctness 側は同じ 16 行 fold テーブル（`kFoldMag`）を引いて、optimized と
   **同じ E4M3 値**を使うこと。素の e2m1 に 2^-d を掛けた実数を使ってはいけない。
3. 行単位の activation 基準 `a_ref` と weight 基準 `w_ref` は epilogue で 1 回
   （`2^(w_ref + a_ref)`、指数加算なので厳密）。correctness / optimized で
   epilogue の掛ける順序も揃える。
4. SplitK は accumulator を block 単位で切る形（chain を K 全体に extend しない）に
   限って一致する。chain する variant（Prefill2D）とは一致しないので、
   `test_mxfp4_linear` の一致ゲートは **variant ごとに群分け**する
   （chain 系: correctness と Prefill2D。block-rescale 系: RowBlock / SplitK）。
   群間比較は tolerance（bf16 一致率 >= 99%、rel_l2 < 1e-3）で持つ。

## 4. 残る仮定と検証方法

「WMMA の 16 個の積の加算順が左から右の逐次」は ISA で保証されていない。
3 の形を実装したときに correctness と optimized が完全一致するかどうかが、
この仮定の実測検証になる（一致しなければ accumulator 構造をもう一段合わせる必要があり、
その場合は chain 形を諦めて block-rescale のまま A 側だけ消す形（= 案 B の kernel 形）
に落とす）。

検証は `test_mxfp4_linear` の row-act セクションで固定する（新しい contract 上で
全 variant を走らせ、上の群分けで一致を確認）。

## 5. 二段目の実装形（訂正: chain 形を第一候補にする）

§3 の chain 形は「同一値」を保証するのが難しい。一方、**per-block rescale を残したまま
A 側だけを loop から外す形**なら、既に成立している bit 一致の論理をそのまま使える。

```text
optimized :  for ib: acc = fma(c_block, ws[ib], acc)   # ws = 2^(e8-127)（E8M0）
              epilogue: acc * 2^a_ref[m]
correctness: for ib: acc += p_ib * ws[ib]
              epilogue: acc * 2^a_ref[m]
```

- `2^a_ref` は 2 冪なので、全 summand と総和を一様にスケールするだけ。
  `Σ fl(p·ws·2^a)` と `(Σ fl(p·ws))·2^a` は正規域で**同一ビット**（2 冪スケーリングは
  丸めに影響しない）。よって reference 側も行基準を epilogue へ移せば bit 一致が保てる。
- ここで行基準が **E8M0（案 D/C）でなければならない**理由が出る。案 B（per-token の
  F32 scale）だと epilogue の係数が 2 冪でなく、per-block 適用とのビット差が消えない。
  つまり「bit 一致を保ちつつ A rescale を loop から外す」のは案 D/C だけ。
- 代价: inner loop から消えるのは `ws * asv` の mul（2D で 256 op / chunk）だけで、
  fma（256 op / chunk）は残る。chain 形（§gate_a_prime §10 ablation の -28〜-40%）より
  小さい利得になる見込み（**-15% 級と推算。未実測**）。

## 6. どちらを先に実装するか

| 形 | inner loop | correctness との関係 | 推算性能 |
| --- | --- | --- | --- |
| safe form | `fma(c, ws, acc)` + epilogue × 2^a_ref | bit 一致を維持できる（論理は §5。実測で固定） | 案 A 比 -15% 級（未測） |
| chain form | WMMA を K 全体へ chain、epilogue に 2^(w_ref + a_ref) | reference も chain 構造へ変更する必要。block 内の 16 積の hardware 順序に依存する仮定が残る | -28〜-40%（ablation 実測） |

block 内の 16 積については、現行の correctness（K index 順に交互）と optimized
（fragment 順）が bf16 で完全一致している事実から、**block 内の dot は hardware が
十分広い累算器で厳密に計算されている**可能性が高い（そうでなければ順序の違う
2 実装が 32768 要素すべてで一致しない）。この仮定が正しければ chain 形でも
reference は「K 全体へ continuation する scalar ループ」で bit 一致を取れるが、
それは §4 の検証をパスしてから採用する。

訂正後の実装順: **chain 形（案 D + K 全体 chain）を第一候補**とし、一致条件は
variant 間の bf16 一致率と rel_l2 を実測して置く（reference は変更しない）。
safe form は、chain 形が E2E の logit ゲートで落とされた場合のフォールバックとして
残す（reference との bit 一致は得るが、消せる命令が少ない）。

案 B（per-token F32）は**採用対象から外す**。行基準が 2 冪でないため per-block 適用と
epilogue 適用のビットが揃わないこと、および §11.1 で現行の 1.71 倍の誤差（C/D は
1.39 倍）で優位性がないことが理由。
