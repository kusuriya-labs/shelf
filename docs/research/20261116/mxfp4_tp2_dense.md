# Dense MXFP4 checkpoint の TP=2

対象: `models/Qwen3.8-27B-QSA-Quark-AWQ-MXFP4`（dense FFN、MXFP4 E2M1 /
E8M0 group 32）。

## 1. 初期状態

`materialize_rank_local_canonical` は `QuantizedEncoding::Mxfp4` を
`unsupported: tp partition of mxfp4 weights is not supported` で拒否していた。
BF16 / FP8 / EXL3 と同じ materialization 順序は既に存在し、MXFP4 の
canonical payload は `QuantFormatDesc` の `PerBlock`（1 output row × 32 K）
だけで表現できるため、format 固有の分岐は不要だった。

## 2. 変更

- `Mxfp4` を `Mxfp4Finegrained` の `materialize_quant` へ接続した。
  N partition は code / scale の row を切り出し、K partition は 32 K block
  （codes 16 byte、scale 1 byte）単位で切り出す。
- `mxfp4 matrix weight rows must be a multiple of 16` の load 時制限を外した。
  native preshuffle は `ceil(local_N / 16) * 16` へ padding するため、
  N partition の local N に 16 の倍数を要求しない（`tensor_partition.md` §4）。
- TP=2 の local shape を `kLinearShapeRules` へ追加した
  （`24/512/3072/5120/6144/8704 × 5120`、`5120 × 3072/5120/8704`）。
  GDN `in_proj_qkv` の TP=2 local 幅は segmented ranges の合計
  `local_qk + local_qk + local_value = 5120` であり、単純な `10240 / 2`
  でも `2 * local_qk + local_value = 7168` でもない。

## 3. 発見した kernel 不具合

TP=2 の GDN `in_proj_a` / `in_proj_b` は local N = 24 になる。24 は 16 の
倍数ではないため RowBlock kernel の guard path を使うが、この path は
**out-feature guard と同時に activation fragment の load まで省略していた**。
その結果、16 を超える最後の partial tile（local N = 24 では row 16..23）が
すべて 0 になり、GDN の gate / beta が壊れていた。

`tests/unit/test_mxfp4_linear.hip` に `out=17/24/100` の tail case を追加すると、
修正前は `16/24` しか一致せず、`out=24` の最後の 8 列が zero であることを確認した。
Prefill2D と SplitK は activation load を out-feature guard から分離していたため
影響を受けていない。

修正は RowBlock の activation fragment load を `out_valid` から独立させ、
weight fragment の load だけを guard する。SplitK / Prefill2D の guard は変更しない。

## 4. 検証

- `test_canonical_partition`: MXFP4 の N / K partition の round trip と
  32 K block 不整列の拒否（pure CPU）。
- `test_mxfp4_linear`: RowBlock1 / RowBlock2 の `out=17/24/100` tail が
  correctness path と bf16 一致。
- `test_qwen35_tp_e2e`（`SHELF_TP_MODEL_DIR=models/Qwen3.8-27B-QSA-Quark-AWQ-MXFP4`,
  `SHELF_TP_ARENA_GIB=28`）: TP=1 と TP=2 の greedy 16 token が一致。
  layer 相対 drift は layer0 0.20%、layer3 4.59%、layer7 2.25%、layer15 1.37%、
  layer31 1.73%、layer62 9.66%、final hidden 2.99%。
  QSA indexer を持たない dense にした同一 checkpoint でも同じ数値になり、
  QSA / dense の差ではなく MXFP4 の TP 丸めによることを確認した。
- `test_weight_load`: 24-row MXFP4 matrix を native 2 tile として load できる。

`shelf-bench quality --tensor-parallel 2` はこの環境では FP8 checkpoint でも
PPL / top-1 が run ごとに大きく変動し、TP=2 の quality 比較には使えなかった。
quality の TP 経路は本件とは別の未解決事項として扱い、acceptance は E2E の
greedy 一致と layer drift で行った。

## 5. 結論

ACCEPT。dense MXFP4 checkpoint は TP=2 で weight load / prefill / decode が
動作し、acceptance prompt の greedy token は TP=1 と一致する。
MXFP4 の TP=1/TP=2 は bit-identical ではなく、K partition により
rank-local の活性量化範囲が変わるため FP8 と drift の幅が異なる。
