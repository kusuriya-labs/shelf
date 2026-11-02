# quant

現在実装されている オリジナル量子化 のみを置く。

## artifact の生成に必要なもの

PSQ5-CB28 などの artifact は AMD Quark のプラグイン（`tools/quark_psq5/`）で生成する。
**`vendor/quark`（amd/quark）を clone していないと、一部のオリジナル量子化モデルは作成できない。**
既存の BF16 / FP8 E4M3 は Quark が無くても Shelf がそのまま扱える。

取得方法と再現条件は `tools/quark_psq5/README.md` と
`docs/research/20261023/psq5_cb28_quark_plugin.md` を参照。

## mixed-precision artifact

PSQ5 と FP8 を混在させる artifact の生成 runner はリポジトリに保持しない。
合成時の encoding 混在は `synthesize_psq5_quantized_source()` が受け取る
`architecture` 指定と manifest の per-tensor encoding で決まる
（詳細は `docs/developer/quantization.md`）。
