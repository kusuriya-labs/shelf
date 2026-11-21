# N-gram table 量子化形式

N-gram / Engram / PLE 埋め込み table のモデル非依存の保存形式を定義する。
Qwen3.8-Flash-Next、Hy4-Flash、DeepSeek-V4.1-Flash などの同等の table 構造を対象とする。
これは形式と変換器の契約であり、現行 Shelf runtime による読込の対応を意味しない。
既存の線形重み用 `shelf_quantization.json` とは別の形式である。
評価の経緯は [offline 評価](../research/20261121/ngram_q_offline_eval.md)、
[Hy4 評価](../research/20261121/ngram_q_step2_hy4.md)、
[PopQA 生成評価](../research/20261121/ngram_q_step3_popqa_gen.md) を参照する。

## プロファイル

識別子は `format = "ngram-q"`、`version = 1`、`codec = "lloyd-row-f16"` とする。
各 table / head の `bits` は 4 または 5 のみである。

| 名称 | manifest の `profile` | 契約 |
| --- | --- | --- |
| Ngram4 | `ngram4` | 全 table / head が 4-bit。Hugging Face 公開の既定プロファイル |
| Ngram5 | `ngram5` | 全 table / head が 5-bit。品質重視の変換器向けプロファイル |
| mixed | `mixed` | table / head 単位で 4-bit と 5-bit を混在させる |

`mixed` は両 bit 幅を少なくとも一つずつ含む。profile と各 `bits` の不一致は拒否する。

## コンテナと配置

bundle root の `ngram_manifest.json` と、manifest が参照する safetensors shard で構成する。
各 table / head は元の行順を保持する独立した論理 table とし、固定幅の行を連続して保存する。
行単位の追加 padding はない。SSD の 4 KiB alignment が必要な配置では、行ではなく
shard または転送 block の境界で確保する。safetensors の tensor 内には追加しない。

| 役割 | dtype と shape | 内容 |
| --- | --- | --- |
| `codes` | `U8 [rows, ceil(dim * bits / 8)]` | 行ごとに pack した index |
| `scale` | `F16 [rows]` | 行ごとの非負の scale |
| `codebook` | `F16 [2^bits]` | table / head ごとに共有する有限かつ単調増加の値 |

単調増加は隣接値が等しくてもよい非減少順を意味する。各 table の一意な `id` を
`<table_key>` とし、tensor 名は `<table_key>.codes`、`<table_key>.scale`、
`<table_key>.codebook` に固定する。3 tensor はその table の `shard` に置く。

## bit packing と復号

各行の先頭を bit 0 とする little-endian bit stream であり、行をまたいで pack しない。

- 4-bit は 1 byte に 2 要素を格納する。偶数列は `byte & 0x0F`、奇数列は
  `byte >> 4`。`dim` が奇数なら最終 byte の上位 4 bit は 0。
- 5-bit は 8 要素を 5 byte に格納し、列 `i` は行先頭から bit `5*i` 〜 `5*i+4` を使う。
  `dim` が 8 の倍数でない場合、最終 byte の未使用上位 bit は 0。

通常の復号は `val[r, c] = fp32(scale[r]) * fp32(codebook[code[r, c]])` とする。
ゼロ行の正規表現は `scale[r]` が `+0.0` かつその行の `codes` の全 byte が 0 である。
scale が 0 の行はこの表現に限り、復号では codebook に関係なく全要素を厳密な `+0.0` とする。

## manifest

`ngram_manifest.json` は JSON object とし、次の項目を必須とする。

| 項目 | 型と意味 |
| --- | --- |
| `format` | 文字列 `ngram-q` |
| `version` | 整数 `1` |
| `profile` | `ngram4`、`ngram5`、`mixed` のいずれか |
| `codec` | 文字列 `lloyd-row-f16` |
| `source_model` | 元 checkpoint を識別する空でない文字列 |
| `tables` | 空でない table object の配列 |

各 `tables` 要素は次の項目をすべて持つ。

| 項目 | 型と意味 |
| --- | --- |
| `id` | bundle 内で一意な空でない文字列。tensor 名の `<table_key>` |
| `name` | 元 checkpoint の table を識別する空でない文字列 |
| `layer` | 元モデルの layer を示す 0 以上の整数 |
| `ngram_order` | N-gram の次数を示す正の整数 |
| `head` | table 内の head を示す 0 以上の整数 |
| `rows`, `dim` | 正の整数 |
| `bits` | 整数 4 または 5 |
| `shard` | bundle root からの相対 safetensors ファイルパス |
| `codes_tensor`, `scale_tensor`, `codebook_tensor` | 上記の命名規則に従う tensor 名 |

## 変換

head / table ごとに独立した標本から 1D Lloyd–Max で `2^bits` 点の codebook を学習する。
学習した codebook は最大絶対値が 1 になるよう正規化し、F16 に丸めて保存する。
保存後も `max(abs(codebook)) == 1` とする。全標本がゼロなどで正規化できない場合、
変換は失敗する。出力射影重みを使った最適化は行わない。

各行は保存された F16 codebook の値を用い、最近傍 index の割当と行 L2 誤差を最小にする
非負 scale の適合を 3 回反復する。得られた scale を F16 に丸め、丸め後の scale で
最近傍 codebook index を確定して pack する。scale が `+0.0` に丸められた行は
ゼロ行の正規表現を使う。最近傍が同距離なら小さい index を選ぶ。

## ロード時の検証

loader は使用前に manifest、参照先、全 tensor の内容を検証し、次の場合は
`Status::invalid_argument` または `Status::unsupported` で拒否する。暗黙の fallback はしない。

- 未知の `format` / `version` / `codec` / `profile`、profile と `bits` の不一致、
  `bits` が 4 または 5 以外、必須項目の欠落や不正な型・値。
- `rows == 0`、`dim == 0`、table ID の重複、tensor 名の規則違反、
  tensor の欠落、shape / dtype の不一致またはサイズ計算の overflow。
- scale の NaN / Inf、負値、`-0.0`、またはゼロ行の正規表現違反。
- codebook の NaN / Inf、非単調、または最大絶対値が 1 でない場合。
- codes の末尾未使用 bit が非ゼロの場合。
- 絶対パスや `..`、symbolic link などにより bundle root 外を参照する shard。

## 拡張

新方式は `codec` ID または `version` を追加して表し、既存の `lloyd-row-f16` 契約を変更しない。

変換・検査には `tools/quantize/ngram_quantize.py` の `convert` / `verify` を使う。
`--tensor-spec` は `name`, `layer`, `ngram_order`, `head` を持つオブジェクトの JSON 配列、
`--bits-map` は tensor 名から bit 幅への JSON オブジェクトを指定する。自動検出では
layer を名前から取得し、次数と head はそれぞれ 1 と 0 に設定するため、実際の構造を
識別する必要がある場合は `--tensor-spec` で明示する。Python には NumPy、実行時には
`shelf_ngram_quant_c` 共有ライブラリが必要である。
