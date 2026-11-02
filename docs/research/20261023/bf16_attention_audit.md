# BF16 paged attention の同類欠陥チェック

- 実施日: 2026-10-23
- 目的: fp8 paged prefill で見つかった欠陥クラス（型変換の誤用、fragment layout の
  取り違え、ゼロ除算による NaN 出力、未初期化 LDS の読み出し）が bf16 経路にも
  存在しないか確認する
- 対象: `paged_prefill.hip`（bf16 prefill）、`paged_decode.hip`（bf16 decode /
  split / reduce、fp8 decode）、`kv_append.hip`（bf16）
- 環境: Git `990e42ce7dd6ff6edc303af3e0b6fd6265e6fdf0` + 本日の修正、
  HIP 7.15.26333、R9700（gfx1201）、`--device 1`

## 結論

bf16 の attention 数値経路に **fp8 と同じ種類の実害は見つからなかった**。
CPU oracle との比較で bf16 の prefill / decode / 長文脈 decode はいずれも
最大相対誤差 1.1e-5 以下であり、bf16 の丸め以外の差は無い。

一方で、実行時の影響が無い（または到達しない）**2 件の欠陥を検出して修正した**。
どちらも NaN を出力に書き得る点で fp8 の D3（P scale の逆数）と同じクラスである。

| ID | 箇所 | 内容 | 到達性 |
| --- | --- | --- | --- |
| B1 | bf16 prefill の `kMode != 0` variant | `uint4 kv` を一度も代入せずに `s.k_page` へ書き、QK がその未初期化値を読む | 現状 `impl<0u>` のみ instantiate されるため未到達（潜在） |
| B2 | decode 系 4 箇所の `total / sum_total` | `sum_total == 0` のとき `0/0` を出力へ書く。full decode は `invalid_shared` で一部を回避するが、zero-visible 行と split/reduce 経路は回避しない | block table の page が範囲外、または visible 0 のとき（運用では異常系） |

## 監査項目と結果

### A1: bf16 → 整数の暗黙変換（fp8 D1 と同型）

`f2bf` / `bf2f` / `__ushort_as_bfloat16` / `static_cast<unsigned short>` の
使用箇所を全 tree で確認した。bf16 値を受け取る `bf2f` の呼び出しは fp8 prefill の
1 箇所だけで、そこは修正済みである。他は `unsigned short` を受ける正しい用法か、
`wbf2f`（`__bfloat162float` を直接適用）であり、同型の欠陥は無い。

### A2: fragment layout（fp8 D2 と同型）

bf16 prefill の PV は `frag_kn` で V を token 軸方向に gather しており、
B fragment の要求（n = dim、k = token）と一致する。bf16 decode は LDS を使わず
1 token ずつ `k_head[d]` / `v_head[d]` を読むため layout の取り違えは起こらない。

実測（`test_paged_attention` の CPU oracle 比較、`max_rel`）:

| case | max_rel |
| --- | ---: |
| bf16 GQA hd=64 rows=8 causal | 2.9e-07 |
| bf16 MHA hd=128 decode rows=1 | 4.8e-07 |
| bf16 GQA hd=32 page=8 rows=16 | 8.0e-06 |
| bf16 real-ish hd=64 page=16 rows=24 | 3.1e-07 |
| bf16 PREFILL hd=256 page=16 rows=4 / 64 | 3.6e-07 / 0 |
| bf16 DECODE long visible=8192 / 4097 / 32768 | 1.1e-05 / 0 / 4.5e-05 |
| bf16 TP2 DECODE / PREFILL hd=256 | 6.7e-05 / 1.5e-05 |
| fp8 参考（比較用） PREFILL rows=4 / rows=64 | 3.6e-07 / 3.2e+00 |

bf16 は全 case で bf16 の丸め水準に収まる。fp8 の rows=64 の 3.2 は
P 量子化（実装仕様）によるもので、行単位相対 L2 では mean 3.0e-3 である。

### A3: ゼロ除算・NaN 出力（fp8 D3 と同型）

`total / sum_total` は 4 箇所（bf16 full decode、bf16 reduce 2 種、fp8 full decode）。
`sum_total` は「可視 token が 1 つも寄与していない」とき 0 になる。

- full decode: block table の page が範囲外なら `invalid_shared` で行ごと打ち切るが、
  valid な行で visible 0 の場合は `0/0` を書く。visible 0 は
  `row_positions + 1` が 0 になる場合（不可能）か、tree 経路で
  `committed_end + anc_count == 0` の場合に限られる。
- split 経路: page 範囲外の split は partial を 0 で埋めるが、reduce 側は
  `sum_total == 0` を検出せず `0/0` を出力へ書く。full decode と挙動が一致しない。

修正: 4 箇所すべてで `sum_total > 0` のときだけ書く（full decode）／
`sum_total <= 0` の split は書き込みを飛ばす（reduce）。可視 token がある通常経路では
`sum_total >= 1` なので挙動は変わらない。

### A4: 未初期化 LDS（B1）

bf16 prefill の `store_chunk` は `uint4 kv` を宣言するが、`kMode == 0`（現行の唯一の
instantiate）では V だけを読み、K は global から直接 fragment を作る。`kMode != 0`
では `s.k_page` にこの未初期化 `kv` を書き、QK がそれを読む。現状の呼び出しは
`attention_paged_prefill_bf16_impl<0u>` のみで、`kMode != 0` は未到達である。

修正: `kMode != 0` のときだけ `k_src` から `kv` を読む（`if constexpr` 内なので
production の codegen は変わらない）。

未到達 variant 自体（`PrefillSharedKV` と `kMode` 引数）は削除候補として残す。
削除は production 経路の変更を伴わないが、今回の調査範囲外の整理であるため
分離して行う。

### A5: race（共有メモリの二重バッファ）

bf16 prefill の page loop には終端 `__syncthreads()` が無い（`kMode != 0` のみ実行）。
mode 0 は `v_page[2]` の二重バッファを使い、次の page の staging が前 page の PV と
別バッファになること、および `p_tile` / `s_part` / `row_*` はその前に barrier を
挟むことから race しない。fp8 prefill は単一バッファのため終端 barrier が必要で、
現行コードはそれを持つ。この不変条件はコード上に明示されていないため、
staging バッファを増減する変更では barrier の再確認が必要である。

## 修正の検証

- 修正前後の bf16 / fp8 の logits dump は **bitwise 一致**
  （`--context 128 --positions 448 --prefill-chunk 2048`、比較は `cmp`）。
  到達する経路の挙動は変わっていない。
- `test_paged_attention` に bf16 / fp8 の edge case を追加した
  （K を全て 0 にして一様 attention にした layout 検査、V を全て 0 にした
  非有限値検査、長文脈 decode の同条件）。
- `test_paged_attention_split_reduce_exact` に「全 split の max が -FLT_MAX、
  sum が 0」の case を追加し、reduce が出力を書かず NaN も残さないことを固定した。
  修正を戻すとこの case は 16 件 fail する。
- required acceptance は 139/139 PASS。

## 付随して分かった測定上の性質

bf16 でも prefill の batch 分割を変えると logits が変わる。同一採点位置
（`--score-start-position 1601`、positions 600）で `--prefill-chunk 256` と
`2048` を比べると、PPL 差 -2.51%、平均 NLL 差 -0.0255、mean KL 0.0093、
top-1 0.955 であった。attention kernel 自体は oracle と 1e-5 で一致するため、
これは projection GEMM の分割形が変わって KV の丸めが変わることによる
model 側の感度である。

したがって、KL 0.01 前後の差は「attention 実装の欠陥」と「数値順序の差」を
区別しない。fp8 KV の品質を議論するときは、同一 chunk・同一採点位置での比較を
前提とし、この床（mean NLL 0.02〜0.03、KL 0.01 程度）を有意差の目安にする。
