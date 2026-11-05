# Qwen3.8-Flash-Next EXL3 チェックポイント解析

## 対象

- ディレクトリ: `/mnt/Qwen3.8-Flash-Next-exl3/`
- 解析日: 2026-11-05
- 形式: Safetensors（EXL3 trellis、`quant_method=exl3`）
- 量子化設定: `version 1.6.3`、`bits 3.18`、`head_bits 6`、`mtp_bits 3`、`vision_bits 6`、`codebook mul1`
- 構成: `Qwen4ExpForConditionalGeneration`、`model_type=qwen4_exp`、`text_config.model_type=qwen4_exp_text`

**重要:** 手順書の GGUF（`UD-IQ4_XS`）前提は、このディレクトリには当てはまらない。
実体は EXL3 の Safetensors であり、PLE も GGUF の `IQ4_NL` ではなく
`exl3_ngram_trellis`（1 行 81 語、162 バイト）で格納されている。
以下は実ファイルのヘッダと量子化メタデータを走査した結果である。

## 結論

| 項目 | 実測値 |
| --- | ---: |
| 本体 7 シャードのデータ量 | 54,453,099,420 B（54.45 GB / 50.71 GiB） |
| PLE テーブル | 51,840,254,232 B（51.84 GB / 48.28 GiB） |
| 全データ量 | 106,293,353,652 B（106.29 GB / 98.99 GiB） |
| テキスト本体（VRAM 側） | 52,838,449,124 B（52.84 GB / 49.21 GiB） |
| MTP（任意） | 1,053,368,872 B（1.05 GB / 0.98 GiB） |
| Vision（任意） | 561,281,424 B（0.56 GB / 0.52 GiB） |
| 物理テンソル数 | 304,240（本体 304,108 + PLE 132） |
| EXL3 モジュール数 | 75,751（本体内訳は下記） |

- 本体は 4 枚の R9700（各 31.86 GiB）に十分収まる。
  テキスト本体だけなら 49.21 GiB、MTP と Vision を含めても 50.71 GiB。
- PLE は **48.28 GiB** であり、手順書の想定（約 26.8 GiB）より **約 21.5 GiB 大きい**。
  RAM は 64 GB では足りず、実測環境の 187 GiB でも PLE 常駐後に余裕を確認する必要がある。
- ヘッダ検証は全件通過した。重複テンソル、範囲外オフセット、形状とバイト数の不一致、
  EXL3 の rate/shape 不整合、PLE メタデータの不整合は検出されなかった。

## ファイル構成

`model.safetensors.index.json` が指すのは次の 7 ファイルである。

| ファイル | サイズ (B) | ヘッダ (B) | テンソル数 |
| --- | ---: | ---: | ---: |
| model-00001-of-00007.safetensors | 8,193,229,000 | 4,913,768 | 37,117 |
| model-00002-of-00007.safetensors | 8,298,674,702 | 6,561,974 | 49,484 |
| model-00003-of-00007.safetensors | 8,298,699,444 | 6,586,716 | 49,484 |
| model-00004-of-00007.safetensors | 8,298,699,444 | 6,586,716 | 49,484 |
| model-00005-of-00007.safetensors | 8,298,699,444 | 6,586,716 | 49,484 |
| model-00006-of-00007.safetensors | 7,417,143,217 | 5,761,841 | 43,300 |
| model-00007-of-00007.safetensors | 5,688,241,774 | 3,289,818 | 25,755 |
| **合計** | **54,453,099,420** | — | **304,108** |

`index.metadata.total_size` は 54,453,099,420 で、7 ファイルのデータ領域合計と一致した。

**PLE は index に含まれない。** 別ファイル `ngram_embedding.safetensors` を明示的に
開く必要がある。このファイルは 51,840,275,600 B、ヘッダ 21,360 B、132 テンソル。

index だけを読むローダーは PLE を丸ごと見落とす。これは最初に踏みやすい罠である。

## アーキテクチャ

`config.json` の `text_config` より:

| 項目 | 値 |
| --- | --- |
| decoder 層 | 48 |
| attention | GDN（linear_attention）36 層 + QSA（full_attention）12 層 |
| full attention 間隔 | 4（0 始まりで 3, 7, 11, ..., 47） |
| hidden_size | 2,560 |
| MoE | 512 routed experts/層、top-10、shared expert あり |
| expert intermediate | 640 |
| HyperConnection | 4 ストリーム、lowrank 320 |
| 最終 norm | なし。`hyper_connection_mixer` で 4 ストリームを統合して head へ |
| PLE | 16 head（8 bigram + 8 trigram）、1 行 160 要素 |
| PLE 挿入 | `ple_layer_ids: [2]`（1 始まり）＝ 0 始まりの層 1 |
| vocab | 248,320 |
| rope | partial_rotary_factor 0.25、theta 1e7、mrope interleaved |

`layer_types` は 0〜2 が linear、3 が full、以降 4 層周期である。
PLE は層 1（linear_attention 層）に挿入される。

### モジュール内訳

| カテゴリ | モジュール数 | テンソル数 | バイト数 | GiB |
| --- | ---: | ---: | ---: | ---: |
| language（decoder + embed + PLE 小物） | 74,040 | 296,908 | 52,295,493,600 | 48.704 |
| lm_head | 1 | 4 | 477,276,164 | 0.445 |
| PLE 小物（`layers.1.ple.*`） | 6 | 6 | 65,679,360 | 0.061 |
| MTP | 1,546 | 6,203 | 1,053,368,872 | 0.981 |
| Vision | 164 | 987 | 561,281,424 | 0.523 |
| PLE 本体（128 シャード） | 128 | 128 | 51,840,248,832 | 48.280 |
| PLE メタデータ | 1 | 4 | 5,400 | 0.000 |

言語本体のうち、EXL3 以外の BF16/F16 重み（embed、norm、hyper connection、
conv1d、A_log、dt_bias など）は約 2.58 GiB ある。embed_tokens だけで 1.18 GiB。

### 層ごとの重み量

0 始まりの層番号ごとの本体バイト数（GiB）:

| 層 | 種別 | GiB | 層 | 種別 | GiB |
| ---: | --- | ---: | ---: | --- | ---: |
| 0 | linear | 1.1132 | 24 | linear | 0.9667 |
| 1 | linear + PLE | 1.1744 | 25 | linear | 0.9667 |
| 2 | linear | 1.1132 | 26 | linear | 0.9667 |
| 3 | full | 1.1076 | 27 | full | 0.9611 |
| 4 | linear | 0.9667 | 28 | linear | 0.9667 |
| 5 | linear | 0.9667 | 29 | linear | 0.9667 |
| 6 | linear | 0.9667 | 30 | linear | 0.9667 |
| 7 | full | 0.9611 | 31 | full | 0.9611 |
| 8〜11 | linear/full | 0.9667 / 0.9611 | 32〜35 | linear/full | 0.9667 / 0.9611 |
| 12〜15 | linear/full | 0.9667 / 0.9611 | 36〜39 | linear/full | 0.9667 / 0.9611 |
| 16〜19 | linear/full | 0.9667 / 0.9611 | 40〜43 | linear/full | 0.9667 / 0.9611 |
| 20〜23 | linear/full | 0.9667 / 0.9611 | 44 | linear | 1.1132 |
| | | | 45 | linear | 1.1132 |
| | | | 46 | linear | 1.1132 |
| | | | 47 | full | 1.1076 |

先頭 4 層と末尾 4 層だけが約 1.11 GiB で、中間 40 層は約 0.96 GiB である。
理由は量子化レートで、先頭・末尾 4 層の routed expert は 3.5 bpw、
中間 40 層は 3.0 bpw になっている。

## 量子化形式

本体は `mul1` codebook の EXL3 trellis である。モジュールごとに
`<module>.trellis`（I16、rank 3）、`.suh`（F16）、`.svh`（F16）、
`.mul1`（I32 scalar）を持つ。`mcg` は使われていない。

`trellis` の形状から求めたビットレートの分布:

| bpw | モジュール数 | 内訳 |
| ---: | ---: | --- |
| 3.0 | 62,976 | language 61,440 + MTP 1,536 |
| 3.5 | 12,288 | language（先頭・末尾 4 層の expert） |
| 5.0 | 2 | MTP |
| 6.0 | 165 | lm_head 1 + Vision 164 |
| 7.0 | 320 | language 312 + MTP 8 |
| **合計** | **75,751** | |

- 言語本体の重み付き実効レートは約 **3.207 bpw**（EXL3 payload のみ）。
- lm_head は約 **6.006 bpw**、Vision は約 **6.020 bpw**、MTP は約 **3.128 bpw**。
- 全 EXL3 モジュールで `k` と `n` は 128 の倍数、`suh` 長 = k、`svh` 長 = n、
  `mul1` は I32 scalar であり、形状不整合は 0 件だった。
- 出現する rate は ka=3、3.5（ka=3, half）、5、6、7 である。
  既存の `exl3::dequant` と EXL3 W16A16 kernel は ka 1〜8、half は ka<=3 を扱うため、
  **本体のレート自体は既存カーネルの範囲内**である。

本体の `quantization_config.json` の `tensor_storage` は 74,398 モジュールを持つが、
**MTP と Vision のエントリを含まない**。一方、物理シャードには MTP 6,203 テンソルと
Vision 987 テンソルが存在する。マニフェスト生成を `tensor_storage` だけに依存させると
この 1.50 GiB を落とすため、物理ヘッダの走査を正とする必要がある。

## PLE テーブル

### 格納形式

`ngram_embedding.safetensors` のメタデータ:

| キー | 値 |
| --- | --- |
| format | `exl3_ngram_trellis` |
| version | 1 |
| K | 8 |
| codebook | `mul1` |
| row_dim | 160 |
| rows | 320,001,536 |
| shard_rows | 2,500,012 |
| シャード数 | 128 |

各シャードは `[2500012, 81]` の I16 で、1 シャード 405,001,944 B。
128 シャードで 51,840,248,832 B。これに `head_bias` 5,120 B、
`head_offsets` 128 B、`head_vocab_sizes` 128 B、`layer_multipliers` 24 B が加わる。

1 行は 81 語 = **162 バイト**:

- 語 0: 行スケール（fp16 のビット列）
- 語 1〜80: 160 要素 × 8 bit の tail-biting trellis ring

実効レートは `(16 + 160 * 8) / 160 = 8.1 bpw` である。
**4 bit 級ではない。** PLE が 48.28 GiB になるのはこのためである。

行の復号:

```text
row[i] = mul1_codebook(state_i) * scale + head_bias[head]
```

`state_i` は位置 i の 8 bit を下位に、位置 i-1 の 8 bit を上位に持つ 16 bit 状態である。
`head` は行番号が属する hash head であり、`head_offsets` から求める。

### 行数と head

| head | vocab size | offset |
| ---: | ---: | ---: |
| 0 | 20,000,003 | 0 |
| 1 | 20,000,023 | 20,000,003 |
| 2 | 20,000,033 | 40,000,026 |
| 3 | 20,000,047 | 60,000,059 |
| 4 | 20,000,059 | 80,000,106 |
| 5 | 20,000,063 | 100,000,165 |
| 6 | 20,000,069 | 120,000,228 |
| 7 | 20,000,077 | 140,000,297 |
| 8 | 20,000,081 | 160,000,374 |
| 9 | 20,000,093 | 180,000,455 |
| 10 | 20,000,107 | 200,000,548 |
| 11 | 20,000,147 | 220,000,655 |
| 12 | 20,000,153 | 240,000,802 |
| 13 | 20,000,159 | 260,000,955 |
| 14 | 20,000,161 | 280,001,114 |
| 15 | 20,000,171 | 300,001,275 |

- `head_vocab_sizes` の合計は 320,001,446。
- `make_ngram_vocab_size_divisible_by=128` により 320,001,536 行へ padding される。
  末尾 90 行は padding である。
- head 0〜7 が bigram、head 8〜15 が trigram に対応する。
- `head_bias` は `[16, 160]` の F16 で、head ごとの加算バイアスである。
- `layer_multipliers` は `[23703573157769, 20109073645365, 8052911324071]`。
  `config.json` に `seed` はないが、transformers の既定値 1234 で再計算した値と一致した。

### n-gram 行番号

参照実装（transformers `Qwen4ExpTextNGramEmbedding` と exllamav3
`NGramEmbedding.compute_ngram_ids`）に一致する。

1. token 履歴の直前 2 token を EOS で初期化する。
2. `shift_right_ignore_eos` で shift 0, 1, 2 の系列を作る。EOS を跨ぐ位置は EOS にする。
3. ngram=2 と ngram=3 のそれぞれで、
   `mixed = XOR_{position} shifted[position] * layer_multipliers[position]`
4. head k について `row = mixed % head_vocab_sizes[k] + head_offsets[k]`
5. 8 + 8 = 16 行を gather し、各行 160 要素を連結して 2,560 要素にする。

decode 1 token あたり 16 行 = 2,592 B（圧縮状態）である。
4K prefill でも 4,096 × 16 行 = 65,536 行、約 10.1 MiB（圧縮状態）にすぎない。
CPU で復号してから GPU0 へ送る方式でも転送量は小さい。

### 実データの復号確認

先頭、中間、末尾、シャード境界、padding 行を実際に読んで復号した。

| 行 | head | scale | RMS | 有限値 |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0 | 0.01828 | 0.01570 | すべて有限 |
| 123,456 | 0 | 0.01083 | 0.00933 | すべて有限 |
| 2,500,011 | 0 | 0.00890 | 0.00762 | すべて有限 |
| 2,500,012 | 0 | 0.00928 | 0.00790 | すべて有限 |
| 123,456,789 | 6 | 0.00817 | 0.00711 | すべて有限 |
| 320,001,535（padding） | 15 | 0.00171 | 0.00001 | すべて有限 |

既知の 3 token 列から 16 行を求め、2,560 要素へ展開した結果もすべて有限で、
RMS は 0.00887 だった。格納形式と行アドレスの解釈は実データで確認できた。

### PLE 小物重み

これらは VRAM（GPU0）に置く。RAM に置くのは巨大テーブルとメタデータだけである。

| テンソル | 形状 | dtype | バイト |
| --- | --- | --- | ---: |
| `layers.1.ple.key_proj.weight` | [10240, 2560] | F16 | 52,428,800 |
| `layers.1.ple.value_proj.weight` | [2560, 2560] | F16 | 13,107,200 |
| `layers.1.ple.conv1d.weight` | [10240, 1, 4] | F16 | 81,920 |
| `layers.1.ple.norm_key.weight` | [10240] | BF16 | 20,480 |
| `layers.1.ple.norm_query.weight` | [10240] | BF16 | 20,480 |
| `layers.1.ple.norm_conv.weight` | [10240] | BF16 | 20,480 |

`key_proj` の出力 10,240 は `hc_count(4) * hidden_size(2560)` である。
PLE の出力は 4 ストリーム分あり、層 1 の hyper connection に加算される。

## GPU 配置

### 実機の状態

`rocm-smi --showmeminfo vram` の実測:

| GPU | VRAM 合計 | 使用中 |
| --- | ---: | ---: |
| 0 | 34,208,743,424 B（31.86 GiB） | 61,104,128 B |
| 1 | 34,208,743,424 B（31.86 GiB） | 61,104,128 B |
| 2 | 34,208,743,424 B（31.86 GiB） | 33,197,965,312 B |
| 3 | 34,208,743,424 B（31.86 GiB） | 34,087,223,296 B |

GPU2 と GPU3 は解析時点で他プロセスがほぼ全 VRAM を使用している。
ロード試験の前にこれらを解放する必要がある。GPU4 は 512 MiB の別デバイスで、
R9700 ではない。

システム RAM は合計 187 GiB、`available` 94 GiB、`buff/cache` 113 GiB。
PLE 48.28 GiB を常駐させるには、ページキャッシュの回収を含めて余裕を確認する。

### パイプライン分割

テキスト本体 49.21 GiB を 4 GPU に連続層で分割する案:

| 案 | GPU0 | GPU1 | GPU2 | GPU3 | 最大 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 12 / 12 / 12 / 12（手順書の暫定案） | 13.415 GiB | 11.584 GiB | 11.584 GiB | 12.627 GiB | 13.415 GiB |
| 11 / 13 / 13 / 11（実測バランス案） | 12.454 GiB | 12.545 GiB | 12.551 GiB | 11.660 GiB | 12.551 GiB |

12/12/12/12 は層数が揃うため実装と検証が簡単である。一方、先頭・末尾 4 層は
3.5 bpw で重いため、バイト量では 11/13/13/11 の方が 0.86 GiB 小さい。
どちらも 31.86 GiB に対して 18 GiB 以上の余裕がある。
まずは 12/12/12/12 で起動し、forward のピークを見て境界を調整するのがよい。

### 2 枚構成（GPU0 と GPU1 のみ）

GPU2 と GPU3 を使わず、GPU0 と GPU1 の 2 枚だけでもテキスト本体は収まる。
`even_stage_ranges(48, 2)` は 24/24 に分割する。実測値:

| GPU | 層 | 重み | 現在の空き | 重み投入後の残り |
| --- | --- | ---: | ---: | ---: |
| 0 | 0〜23 + embed | 26,842,561,520 B（25.00 GiB） | 31.80 GiB | 6.80 GiB |
| 1 | 24〜47 + mixer + lm_head | 25,995,887,604 B（24.21 GiB） | 31.80 GiB | 7.59 GiB |

- `--pipeline-parallel 2 --device 0,1` 相当の 2 ステージ構成で、層数は 24/24。
- arena の auto 確保は「空き - 1 GiB」なので、各 stage の容量は約 30.80 GiB。
  重み 25.00 / 24.21 GiB を引いても 5.8 / 6.6 GiB が workspace に残る。
- MTP と Vision を GPU1 に足しても 25.71 GiB で、残りは約 6.09 GiB。
- 4K コンテキストの KV/GDN 状態、活性、PP 境界の hidden 転送を含めても
  この余裕で収まる見込みである。ただし実測は forward 接続後に行う。
- PLE は GPU に置かない。48.28 GiB は CPU RAM に常駐させる。

VRAM に置くのは `body` の全テンソルである。PLE の 128 シャードと 4 メタデータは
VRAM アリーナに入れない。MTP と Vision を無効にするなら、それぞれ 0.98 GiB と
0.52 GiB をロード対象から外せる。

### TP（Expert Parallel）2 枚構成

`--pipeline-parallel` ではなく `--tensor-parallel 2 --device 0,1` を使う方向は、
分割軸としては成立する。現行 engine は MoE の pipeline parallel を明示的に拒否し、
grouped expert の分散実行は TP 側にだけ実装がある。

#### 分割の成立条件

| 項目 | 全体 | TP=2 の rank あたり | 判定 |
| --- | ---: | ---: | --- |
| routed experts | 512 | 256 | 可（512 % 2 = 0） |
| top-k | 10（global） | 10（global） | 可（router 最大 512、top-k 最大 16） |
| Q head | 24 | 12 | 可 |
| KV head | 2 | 1 | 可 |
| GDN key head | 16 | 8 | 可 |
| GDN value head | 48 | 24 | 可 |
| shared expert intermediate | 640 | 320 | 可 |

router は replicated のまま global top-10 を選び、各 rank は所有する 256 expert の
slot だけを materialize する。combine は rank の部分和を集める。これは
35B-A3B で確立した expert 軸分割の contract と同じである。

#### rank あたりの重み

既存の `materialize_rank_local_canonical` は EXL3 と MXFP4 の行列分割を
`unsupported` で拒否する。そのため最初のロード経路では routed expert だけを
expert 軸で分割し、**非 expert はすべて両 rank に複製**する。

| 項目 | 全体 (GiB) | rank あたり (GiB) |
| --- | ---: | ---: |
| routed expert | 43.799 | 21.900 |
| GDN（複製） | 1.713 | 1.713 |
| QSA + indexer（複製） | 0.504 | 0.504 |
| shared expert（複製） | 0.193 | 0.193 |
| embed_tokens（複製） | 1.184 | 1.184 |
| hyper connection（複製） | 1.193 | 1.193 |
| router / shared gate（複製） | 0.117 | 0.117 |
| lm_head（複製） | 0.445 | 0.445 |
| PLE 小物（複製） | 0.061 | 0.061 |
| **合計** | **49.210** | **27.310** |

空き 31.80 GiB に対して残り約 **4.49 GiB**。実ロードでも rank あたり
26.025 GiB の重み、arena 使用 27.310 GiB を確認した。

EXL3 の行列分割を実装すれば、GDN・QSA・shared expert を半分にでき、
rank あたり約 26.1 GiB まで下げられる。ただし最初のロードでは複製を選び、
パック形式のまま転送することを優先した。

#### 現行実装のブロッカー

1. **routed expert の実行経路が MXFP4 専用である。**
   `src/shelf/inference/execution/moe_dispatch.hip` は grouped linear で
   `encoding == Mxfp4` を要求する。ロード側は EXL3 の expert plane を
   パックのまま GPU に置けるようにしたが、実行側の EXL3 grouped GEMM は未実装である。
2. **Qwen4Exp family の runtime が未実装である。**
   4 ストリーム HyperConnection、QSA indexer、GDN の sigmoid 出力ゲート、
   PLE 注入、512 experts / top-10 の routing は qwen35moe の構成と異なる。
3. **QSA indexer の KV head は 1 である。**
   indexer は全体 0.016 GiB なので、最初の経路では replicated にした。
4. **PLE は rank ごとに複製しない。**
   TP の rank は同一 process の thread なので、CPU RAM の `PleTable` を 1 個だけ持ち、
   両 rank が共有する。実装はこの前提である。
5. **MTP は TP=1 必須である。**
   現行 CLI は `--mtp` と `--tensor-parallel 2` の併用を拒否する。
   初回のテキスト推論では MTP を無効にする前提と一致する。

### 実装したロード経路

`SHELF_QWEN4EXP` として次の load-only family を追加した。
現在の load contract は [../../../developer/qwen4exp.md](../../../developer/qwen4exp.md) に置く。

| ファイル | 役割 |
| --- | --- |
| `include/shelf/models/qwen4exp/model/qwen4exp_config.h` | 共通 hybrid geometry、MoE、PLE、indexer、HyperConnection の config |
| `src/shelf/models/qwen4exp/model/qwen4exp_config.cpp` | config 読み込み、prime / multiplier / padded row の再計算 |
| `include/shelf/models/qwen4exp/weights/ple_table.h` | RAM 常駐 PLE テーブルと CPU 参照 decode / hash |
| `src/shelf/models/qwen4exp/weights/ple_table.cpp` | 128 シャードの `pread` ロード、メタデータ検証、行復号 |
| `include/shelf/models/qwen4exp/weights/qwen4exp_weights.h` | body / expert plane / PLE 小物の重み構造 |
| `src/shelf/models/qwen4exp/weights/qwen4exp_load.cpp` | TP=2 の rank ロード、expert 軸分割、EXL3 expert plane |
| `tests/unit/test_qwen4exp_load.cpp` | 実 checkpoint のロード、サンプル一致、PLE 行一致 |

ロードの要点:

- 本体 7 シャードを `QuantizedModelReader` で開き、body を rank の arena へ転送する。
- routed expert は rank の連続 256 expert を
  `[local_experts, trellis_bytes]`、`[local_experts, k]`、`[local_experts, n]`、
  `[local_experts, 4]` の 4 面にパックしたまま置く。EXL3 の再展開はしない。
- PLE は `ngram_embedding.safetensors` を index とは別に開き、128 シャードを
  `pread` で匿名 RAM に読む。`posix_fadvise(POSIX_FADV_DONTNEED)` を各シャード後に呼ぶ。
- PLE の head metadata は config から再計算した prime / offset / multiplier と照合する。
- 非 expert は複製する。EXL3 の行列 partition は未実装だからである。
- `probe_model_dir` は `ple_layer_ids` と `hc_count` を見て `Qwen4Exp` を返し、
  `LoadedModel` / TP / PP の既存経路では明示的に `unsupported` にする。

#### 実測結果

コマンド:

```bash
./build/tests/test_qwen4exp_load --model-dir /mnt/Qwen3.8-Flash-Next-exl3 \
  --arena-gib 29
```

この 1 コマンドで、PLE 48.28 GiB の RAM ロードと、GPU0/1 への rank0/rank1 の
同時ロードを実行する。

| 項目 | 結果 |
| --- | ---: |
| rank0 / rank1 の重み | 各 26.025 GiB |
| rank0 / rank1 の arena 使用 | 各 27.310 GiB |
| 先頭 expert の trellis 先頭 4 KiB | 元ファイルと一致 |
| 末尾 expert の trellis 先頭 4 KiB | 元ファイルと一致 |
| rank の形状不変条件 | PASS |
| PLE 常駐量 | 48.280 GiB |
| PLE の先頭・末尾・境界・ランダム行 | 元ファイルと一致 |
| PLE 1 行の復号 | 160 要素、RMS 0.0157、有限 |
| PLE 16 行 gather | 2,560 要素、有限 |
| TP=2 の両ランク + PLE 同時ロード | PASS |

rank0 と rank1 は別々にも同じ GPU 上でロードして PASS している。
PLE の行は 0、1、123456、2,500,011、2,500,012、123,456,789、末尾を
元ファイルとバイト比較した。

#### EXL3 の TP 分割が拒否される理由

`load_quantized_matrix` は partition plan に名前があると
`materialize_rank_local_canonical` を呼ぶ。この関数は BF16 と FP8 block-128 だけを
実装し、MXFP4 と EXL3 は `unsupported` を返す。

EXL3 の分割が単純でない理由:

1. codes は `[kt][nt][words]` の 16×16 tile である。N 軸は `nt` と `svh`、
   K 軸は `kt` と `suh` を切る必要があり、row-major の行コピーでは表現できない。
2. K 軸は 128 要素の Hadamard block を跨げない。入力回転と `suh` が 128 単位だからである。
3. shared expert の down_proj は現行 dense TP 契約で K 軸を 640 → 320 に割る。
   320 は 128 の倍数ではないため、EXL3 の K 軸では表現できない。
4. MXFP4 も同じ理由で拒否される。canonical partition は BF16 / FP8 の
   row-major 形式を前提にした contract である。

このため最初のロードは expert 軸の分割と非 expert の複製を選んだ。

#### EXL3 grouped linear と MoE dispatch

`launch_gemm_exl3_w16a16_grouped` を fused kernel として実装した。
既存の dense EXL3 kernel に `blockIdx.z` の group 次元を追加し、1 launch で全 group を
処理する。`group_expert` / `group_rows` / `group_count` と expert ごとの
trellis/suh/svh batch stride を受け取り、packed trellis のまま計算する。
`row_token` が非 null のときは token row 入力から gather し、null のときは
group row 平面をそのまま読む。

MoE dispatch 側では次を追加した。

- `WeightSlot` の EXL3 expert plane batch stride
- `make_expert_matrix(Qwen4ExpExpertPlane)` による rank-3 view
- `KernelId::LINEAR_EXL3_W16A16_GROUPED`
- `try_launch_moe_grouped_exl3`
- EXL3 の `MOE_GROUPED_LINEAR` lowering（activation quantize なし）

`test_exl3_grouped` は実 checkpoint の layer 0 / expert 0 と expert 1 の
gate_proj を読み、k=2560、n=640、ka=3 half で CPU の `exl3::dequant` と比較した。

| ケース | max_abs | 判定 |
| --- | ---: | --- |
| 2 group / 2 expert、group-row 入力 | 1.37e-4 | PASS |
| 2 group / 2 expert、`row_token` で token-row 入力 | 2.18e-4 | PASS |

`test_program_moe_plan` は EXL3 の MoE graph が activation quantize を挟まず、
`MOE_ROUTER, MOE_PLAN_SCAN, MOE_PLAN_SCATTER, LINEAR_EXL3_W16A16_GROUPED,
SWIGLU, LINEAR_EXL3_W16A16_GROUPED, ...` に lower されることを固定する。
`test_qwen4exp_load` は実 expert plane から rank-3 `WeightSlot` を作り、
`validate_host_weight_slot` を通す。

#### HyperConnection primitive

Qwen4Exp runtime の最初の primitive として `HYPER_MIX` と `HYPER_APPLY` を追加した。

- `HYPER_MIX`: `mean_h(gate[h] * normed[h])` を `[rows, hc*hidden]` から
  `[rows, hidden]` へ
- `HYPER_APPLY`: `raw[h] + y * injection[h]` を `[rows, hc*hidden]` へ

カーネルは BF16 で、`test_hyper_connection` が rows 3、hc 4、hidden 128 で
CPU 参照と一致することを固定する。`test_program_hyper_connection` は graph が
`HYPER_MIX, HYPER_APPLY` に lower されることを固定する。

site 全体は、既存の grouped RMSNorm（`group_size=hidden`、`ONE_PLUS`）と
BF16 linear（down/up/block_inject）、`SILU` / `SIGMOID` / `SCALE` を組み合わせ、
最後に `HYPER_MIX` / `HYPER_APPLY` を置く形で表現できる。
残りは Qwen4Exp decoder layer の lowering と、4 stream を運ぶ runtime state の接続である。

#### 結論

TP=2 のロード経路は実装できた。本体は expert 軸で分割され、非 expert の複製を
含めて rank あたり 27.31 GiB に収まる。PLE は 48.28 GiB を RAM に常駐できる。
EXL3 grouped linear の正しさ優先経路も実 checkpoint で通り始めた。
次は fused grouped kernel と Qwen4Exp の forward/runtime を接続する工程になる。

## 検証結果

- [x] 7 シャードがすべて存在し、index の `total_size` と一致した。
- [x] 重複テンソル名は 0 件。
- [x] 各テンソルの `data_offsets` はファイルのデータ領域内にあり、重複範囲も 0 件。
- [x] 形状と dtype から計算したバイト数は、全 304,240 テンソルで格納バイト数と一致した。
- [x] decoder 層番号は 0〜47 が揃っている。
- [x] EXL3 の `trellis`、`suh`、`svh`、`mul1` の形状は全 75,751 モジュールで整合した。
- [x] PLE は 128 シャード、320,001,536 行、K=8、1 行 81 語で整合した。
- [x] `head_vocab_sizes`、`head_offsets`、`layer_multipliers` は参照実装の計算と一致した。
- [x] PLE の実データ行を復号し、有限値を確認した。
- [x] 本体で使う EXL3 レートは既存カーネルの対応範囲（ka 1〜8、half は ka<=3）内である。
- [ ] PLE の `exl3_ngram_trellis` を読む C++ 実装は、現時点では存在しない。
- [ ] `qwen4_exp` の構成は、現時点の runtime では受理されない。

## 現行エンジンとの差分

### 1. config identity gate で拒否される

`src/shelf/models/common/qwen_hybrid/config.cpp` の `kModelIdentities` は
`qwen3_5`、`qwen3_5_text`、`qwen3_5_moe`、`qwen3_5_moe_text` のみを持つ。
このチェックポイントは `model_type=qwen4_exp`、`text_config.model_type=qwen4_exp_text`
なので、テンソルを開く前の `check_config_identity` で `Status::unsupported` になる。
Qwen4Exp を別 family として追加するか、identity の allowlist を拡張する必要がある。
ただし identity で code path を選ばせない方針は維持する。

### 2. PLE が index の外にある

`ngram_embedding.safetensors` は `model.safetensors.index.json` に含まれない。
現行の `HfShardSet` は index の `weight_map` だけを開くため、PLE は見えない。
PLE 専用のファイルリーダーと `PleTable` を別経路で用意する必要がある。

### 3. PLE は既存 EXL3 manifest に載らない

`ngram_embedding.shard_N.trellis` は rank 2 の `[2500012, 81]` I16 である。
現行の `synthesize_exl3_tensor` は rank 3 の `[kt, nt, words]` を要求し、
`k` と `n` が 128 の倍数であることも要求する。
PLE 行は 160 次元の tail-biting ring であり、本体の 16×16 tile とは別の codec である。
既存の `exl3::dequant`（128 Hadamard 回転付き）を流用してはいけない。

### 4. Qwen4Exp の演算構成が未実装

現行の `qwen35moe` に近い部分はあるが、次が足りない。

- 4 ストリームの HyperConnection と `hyper_connection_mixer`
- QSA（indexer による micro-block 選択）
- GDN の sigmoid 出力ゲート
- PLE の hash、16 行 gather、key/value 射影、short conv、層 1 への加算
- 512 experts / top-10 の routing
- 最終 norm なしの head 接続

これらはロード経路とは別の runtime/kernel 作業である。

### 5. `tensor_storage` に MTP と Vision がない

本体の量子化メタデータを唯一の正にすると、MTP 6,203 テンソルと Vision 987 テンソルを
落とす。マニフェストは物理ヘッダから作る。今回の解析ツールはそうしている。

## 推奨する次の順序

1. `manifest.json` をロード前に検証する。role が `body` の全テンソルに一意な GPU を割り当て、
   `ple` と `ple_metadata` を VRAM アリーナから除外する。
2. Qwen4Exp の identity を受理できるようにし、テキストのみ・MTP/Vision なしの
   logical name 集合を作る。
3. 本体 7 シャードを GPU0〜3 のアリーナへ転送する。EXL3 はパックのまま置く。
4. `ngram_embedding.safetensors` を別経路で開き、128 シャードを RAM の連続領域へ
   読み込む。`head_offsets` などのメタデータは別バッファに保持する。
5. `PleTable` に行番号 → シャード → 162 バイト行の API を作る。
6. 参照実装と同じ hash で 16 行を求め、CPU で 2,560 要素へ復号して GPU0 へ送る。
7. 本体と PLE のロード検証後、Qwen4Exp の forward を別工程で接続する。

## 成果物

- 解析ツール: `tools/analyze_exl3_hf_manifest.py`
- マニフェスト: `artifacts/qwen38-exl3/manifest.json`
- サマリ: `artifacts/qwen38-exl3/summary.json`
- 本レポート: `docs/research/20261105/qwen38_flash_next_exl3/analysis.md`

再現コマンド:

```bash
python3 tools/analyze_exl3_hf_manifest.py /mnt/Qwen3.8-Flash-Next-exl3 \
  --include-extra \
  --out artifacts/qwen38-exl3/manifest.json \
  --summary artifacts/qwen38-exl3/summary.json
```
