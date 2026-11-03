# 案 D（activation E8M0 行基準 fold）本実装の設計メモ

実測根拠は [mxfp4_gate_a_prime.md](mxfp4_gate_a_prime.md) §10〜§14。
このメモは**進行中の contract 変更の設計**であり、`docs/developer/` の現在形
contract ではない。実装完了時に現在形だけへ移す。§3 の 1〜3 段目（quantize kernel、
correctness path の読み取り、optimized 3 系統の行基準形）は実装済みで、実測は
[mxfp4_gate_a_prime.md](mxfp4_gate_a_prime.md) §15 にまとめた（linear 1 回で prefill
-28〜-42%、decode -13〜-29%）。**まだ graph から到達できない**（4 段目の配線と
E2E ゲートが未実施）。

## 1. 狙い

MXFP4 W4A8 の activation を

- codes: E4M3（現状と同じ fragment order の平面、1 code = 1 byte）
- scale: 128 K ごとの **E8M0** を、行単位の基準指数へ fold して codes に畳む
- 付随: 行あたり 1 個の基準値（`2^ref` を F32 で既存 scale 平面の column 0 に置く）

へ置き換え、GEMM の K ループから activation rescale 命令を**完全に消す**。
実測で Prefill2D が -28%〜-40%（§13.1）、精度は現行 A 側の 1.39 倍の誤差で
fold 自体の損は 5 桁目（§13.2）、WMMA は E4M3 subnormal を flush しない（§14）。

## 2. 変更点（blast radius）

| 層 | 変更 |
| --- | --- |
| quantization | 新 spec `FP8_E4M3_ROWREF_FOLDED_V1`（codes = FP8E4M3、group 128 の E8M0 を行基準へ fold、付随 = 行 1 個の F32 基準）。`QuantSpecId` に追加 |
| activation kernel | `launch_activation_quantize_mxfp4_rowref(...)`。行ごとに (1) 128 K group の amax → `e_g = ceil(log2(amax/448))`、(2) `ref = max_g e_g`、(3) codes = `e4m3_rne(e4m3_rne(a / 2^e_g) * 2^-(ref-e_g))`、(4) `scales[row*stride + 0] = 2^ref`。**group 毎の 2 段丸めをそのまま再現する**（§13.2 の実測はこの形） |
| KernelId | `ACTIVATION_QUANTIZE_FP8_FINEGRAINED` を MXFP4 消費側だけ新 id へ（`linear_quant_contract.h` の family → spec 対応で決まる）。graph lowering は linear の activation spec を見て emit するだけ |
| correctness path | `exec_linear_mxfp4` は fold 後の平面を読む形を**独立実装で追加**する（per-block の activation scale を掛けない / 行基準を epilogue で掛ける）。accumulation 構造は変えず、optimized との一致は dtype 誤差内で見る |
| optimized kernels | Prefill2D は `kActRescale=false` 形が既に実装済み（bench 専用 entry）。production の launcher として promote し、RowBlock / SplitK にも同じ形を入れる（rows==1 での速度は未測。decode は DRAM 律速なので利得は小さい見込み、実測して決める） |
| workspace | 追加平面は不要（`Int8ActivationWorkspaceLayout` の F32 scale 平面の column 0 を行う。他の column は未使用）。resident size 不変 |
| selector / dispatch | activation spec を手がかりに launcher を選択。`WeightSlot` は不変 |
| TP | MXFP4 の partition は未実装のまま（変更しない） |

## 3. 実装順（各段で計測し、壊れたらそこで止める）

1. activation kernel（新 entry）+ host 参照との一致テスト。**実装済み**
   （`tests/unit/test_activation_rowref_fold.hip`: 行内 exponent 幅 -16..1 octaves の
   fixture で codes 16384 / scales を bit 一致、fold で 3548 code が変わり 561 code が
   ゼロへ潰れる帯域を実際に踏んでいる）。残る is「E2E prefill の実 activation で codes
   平面の差分率を §13.2 の予測（0.007〜1.5%、subnormal 化 3.07%）と突き合わせる」こと。
2. correctness path に fold 後の平面を読む形を追加（`DispatchBinding::flags` bit 0 =
   `kActFlagRowRefFolded`、読み元だけ分岐し accumulation は不変）。**実装済み**。
3. optimized 3 系統（RowBlock / SplitK / Prefill2D）に行基準形を入れ、
   `Mxfp4ActivationForm` で launcher が選択。**実装済み・実測済み**（§15.1、§15.4）。
   chain 形は reference と bf16 完全一致（§15.5）。selector の variant 選択は
   §15.4 の限り現行のままでも最適（rows=1/8 -> rowblock1、256 -> rowblock4、
   512 以上 -> prefill2d）。残るは rowref quantizer の small-rows 増分
   （rows=1 k=17408 で +8.9 µs、§15.6 の却下記録）。
4. rowref 平面での weight fold 適否: rows==1 で fold あり/なしがほぼ同等
   （§15.4）。group 平面で観測した rows==1 の fold 損は消えているが、decode の
   gating は現状（Prefill2D のみ fold）のまま据え置く。
5. E2E ゲート（次節）→ `docs/developer/` の contract 更新 → 現在形で書き、このメモは削除。

## 4. 受け入れゲート（reference は変えない。条件は dtype 誤差内）

一致条件の基準は `test_mxfp4_linear` が元々持っていた「double 参照に対する
bf16 exact >= 99.9% / rel_l2 < 2e-3」で統一する。optimized variant 間・variant と
correctness path の bf16 bit 一致は契約にしない（reference を optimized へ
合わせ込む形は採用しない）。chain 形を採る場合は variant 間の一致率と rel_l2 を
実測してしきい値を置く。


§11.1 で現行 A 側量子化だけで出力 rel_l2 5.5e-3、bf16 反転率 85%。よって
`GENERATED_IDS` 一致は **pass 不能**（§12 で logit rel_l2 ~1e-2 级で 54/512 token 分岐を実測）。

採用判断のゲート:

1. **prefill 最終位置の logit**: `shelf-compute --dump-logits` で top-1 一致（100%）、
   rel_l2 が事前予測値（案 D の layer 積算から出す上限）以内。
2. **生成文の妥当性**: 同一 prompt 512 token で分岐点は出る前提で、分岐後の文が
   破綻していないことを確認（現行 optimized と案 D の生成列を並べて見る）。
   参考系列として FP8 W8A8 版の生成列も併記する。
3. micro / E2E の性能値は §13.1 以上であることを `--perf-stats` で確認（fallback 0）。

`docs/perf/current.md` へ MXFP4 の数値を書くかは、このゲート通過後（qualified
baseline になってから）の判断。

## 5. 未解決として残っている点

- weight 側 fold の tail 5 行が logit rel_l2 1.05e-2 を出した理由（A 側の fold 損は
  出力に出ない）が説明できていない。A 側の本実装には影響しないが、MXFP4 の
  精度議論としてのこる（§13.2 note）。
- 案 D の E8M0 scale は 2 冪なので、activation の outlier 分布では group 内 amax と
  基準指数の開きが大きくなる。行内幅が 22 octave あっても fold 損は無視できたが、
  MoE（Gate B 以降）では expert 間の activation 分布が異なるため再計測が必要。
