# quant

現在実装されている オリジナル量子化 のみを置く。

## artifact の生成に必要なもの

PSQ5-CB28 などの artifact は AMD Quark のプラグイン（`tools/quark_psq5/`）で生成する。
**`vendor/quark`（amd/quark）を clone していないと、一部のオリジナル量子化モデルは作成できない。**
既存の BF16 / FP8 E4M3 は Quark が無くても Shelf がそのまま扱える。

取得方法と再現条件は `tools/quark_psq5/README.md` と
`docs/research/20261023/psq5_cb28_quark_plugin.md` を参照。

## mixed-precision artifact

Qwen3.8-Flash-Next のように PSQ5 と FP8 を混在させる artifact の生成 runner は
リポジトリに保持しない。生成手順と実測は
`docs/research/20261024/qwen4exp/quantization_profile.md` と
`docs/research/20261024/qwen4exp/quantization_artifact.md` を参照。
