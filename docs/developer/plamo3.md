# PLaMo 3 Translate（BF16）

0.8B / 2B / 8B の BF16 checkpoint を単一 GPU の `shelf-compute` で検証した。
0.8B はサーバー経由でも日英・英日を検証した。量子化、TP、pipeline parallel、
投機 decoding は未対応。
`max_seq_len` は 2048 以下に制限する。この範囲では全層を通常の causal
attention として計算しても局所窓と一致する。長文用 sliding window の実装ではない。

局所層は theta 10000 の通常 RoPE、8 層ごとの全域層は theta 1000000・factor 256 の
YaRN を使う。Q/K の正規化と RoPE は参照モデルの BF16 中間丸めを再現する。現在の
0.8B の 16 query heads / 1 KV head / 128 head dimension を含む PLaMo の
head 形状は、高速化済み paged attention の部分 head kernel を選択する。
256 次元専用の tiled prefill は使えないため、prefill も各行をこの kernel で計算する。
RMSNorm の offset mode や一部の要素演算は汎用経路のままである。

```bash
./build/shelf-compute --model-dir /path/to/plamo-3-translate-0.8b \
  --input-ids-file /path/to/prompt.tokens --max-new-tokens 32 \
  --max-seq-len 2048 --device 0 --temperature 0
```

`--input-ids-file` は空白区切り整数か `shelf-bench quality` の PSKLDTOK 形式。
checkpoint の tokenizer は custom code と翻訳方向を含む chat template を使う。
入力 IDs は配布 tokenizer の `apply_chat_template` に `target_lang` と任意の
`source_lang` を渡して作る。停止 token は 1 と 16。

実機例（R9700、2026-11-11、BF16、greedy、prompt 23 / 26 token）:

| 入力 | 出力 | prefill | decode |
| --- | --- | ---: | ---: |
| 今日は良い天気ですね。 | The weather is nice today, isn't it? | 250.4 ms | 10.10 token/s |
| The train will arrive in five minutes. | 列車は5分後に到着します。 | 252.3 ms | 10.01 token/s |

CPU BF16 参照との teacher-forced 比較は平均 KL 0.000525 / 0.000493、top-1 一致率
100% / 100%、最大 logits 差 0.625 / 0.5625。一部位置は KL 0.001 を超える。
層・演算別の比較と位置ごとの値は
[調査記録](../research/20261111/plamo3_translate.md) に記載する。

2B / 8B の R9700 実測では、decode はそれぞれ約 7.2 / 3.0 token/s。
CPU BF16 参照との平均 KL は 2B が 0.000551 / 0.000492、8B が
0.000700 / 0.000968。top-1 一致は 2B が 95.24% / 95.83%、8B が
100% / 100%。8B の重みは 15.89 GiB で KV とともに 32 GiB に収まった。
詳細は[調査記録](../research/20261111/plamo3_translate.md)を参照。

今後の作業は、真の sliding-window kernel と長文脈の検証、`rocm_exl3` の
PLaMo architecture 登録と量子化、および残存する汎用演算の高速化である。
31B は対象外。

サーバーは `model_type=plamo3` を検出し、reasoning を無効にする。
`/v1/chat/completions` の `chat_template_kwargs` に `target_lang` と任意の
`source_lang`、`context`、`glossary`、`form` を渡す。具体例は
[サーバーの使い方](../user/server.md#plamo-3-translate)を参照。

0.8B の日英・英日とも greedy token 列は attention 最適化前と一致した。
サーバーの実モデルテストでは両方向の通常応答・stream、template 引数、文脈長制限を
確認した。Qwen の models / chat の関連テストも通過した。全回帰は実行していない。
