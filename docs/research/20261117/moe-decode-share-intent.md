# grouped EXL3-INT8 の expert GEMM 活性前処理の狙い

## 目的

Qwen4Exp（Flash Next）の MoE expert GEMM（grouped EXL3 mul24）について、
prefill の入力準備を GEMM の外へ出すことでスループットを上げる。

mul24 の重みは INT8 だが、既定の活性は FP16 である。各 GEMM は expert ごとの
suh で入力を回転し、128-K ブロックごとに FP16 へ変換していた。同じ変換が
n tile と K chunk の回数だけ繰り返され、GEMM のレジスタと LDS を占有していた。

やりたいことは、この回転と量子化を group row × 128-K ブロック単位で一度だけ
行い、その結果を workspace に置くことである。GEMM 側は codes と scale を
LDS へ読み込むだけにし、suh の読み出しと量子化の計算をなくす。

## 前提

- 対象は grouped expert checkpoint（Qwen4Exp、EXL3-INT8、mul24）。
- 密集 EXL3-INT8 の活性前処理は #14 で先行して実装されている。今回はその
  grouped 版である。
- 既定は従来の FP16 活性のまま。機能は実験用フラグで明示的に有効化する。
- 特定ハードウェア（RDNA4 / ROCm / HIP）向けの最適化であり、性能が出ない
  方針は採用しない。

## 方式の検討

方式 A: LDS 共有

- 48 行分の FP32 部分和と 128×128 の INT8 復号重みを LDS に保持し、
  16 行 tile ごとに復号重みを使い回す。
- 判定: REJECT。同時 block 数が 1 に落ち、LDS の読み書きが増えて遅くなった。

方式 B: 活性前処理

- group row と 128 要素単位の INT8 codes と scale を先に書き、GEMM は
  workspace から LDS へ読み込む。
- n ブロックごとの回転をなくし、GEMM の VGPR も減る。
- 判定: ACCEPT 候補。方式 A より速い。

方式 C: 自動選択

- 測定済みの 3 / 4 bit、256 token 以上、2560×640 または 640×2560 に限定して
  B の INT8 活性を選ぶ。条件外は FP16。
- バッチ単位の保守的な条件であり、expert ごとの行数による最適選択ではない。
- 既定は無効。

採否の根拠と測定値は [perf 記録](../../perf/moe-decode-share-lds.md) と
[参考値](../../perf/moe-decode-share-reference.md) を正本とする。

## 変更の範囲

- `Exl3GemmArgs` / `Exl3GroupedArgs`: LDS 有効化と workspace の受け渡し
  （main の再編後は `exl3_family/types.h`）
- `grouped.hip`: 前処理 kernel、LDS 共有 kernel、`kPre` 経路
- `moe_dispatch.hip` / `program.cpp` / footprint: フラグ解釈と scratch 確保
- テスト: 行境界、long-K、auto 判定の静的検証

## 未解決

- 実モデル・TP 環境での E2E 検証は今回が初めてで、prefill 時間は改善方向
  だった。
- ただし logits の比較が再現せず、INT8 活性の丸めが品質に与える影響は
  現時点で結論が出ていない。性能の採否判断には、再現性のある品質確認が
  必要である。
