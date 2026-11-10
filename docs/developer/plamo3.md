# PLaMo 3 Translate（BF16）

0.8B の BF16 checkpoint を単一 GPU の `shelf-compute` で検証した。2B / 8B、
量子化、TP、pipeline parallel、投機 decoding とサーバー経由の実推論は未検証または未対応。
`max_seq_len` は 2048 以下に制限する。この範囲では全層を通常の causal
attention として計算しても局所窓と一致する。長文用 sliding window の実装ではない。

局所層は theta 10000 の通常 RoPE、8 層ごとの全域層は theta 1000000・factor 256 の
YaRN を使う。Q/K の正規化と RoPE は参照モデルの BF16 中間丸めを再現する。現在の
0.8B の 16 query heads / 1 KV head / 128 head dimension は汎用 correctness attention
を使い、高速化済み paged attention の選択対象ではない。

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
| 今日は良い天気ですね。 | The weather is nice today, isn't it? | 257.4 ms | 8.38 token/s |
| The train will arrive in five minutes. | 列車は5分後に到着します。 | 270.0 ms | 8.22 token/s |

CPU BF16 参照との teacher-forced 比較は平均 KL 0.000629 / 0.000610、top-1 一致率
100% / 100%、最大 logits 差 0.625 / 0.625。一部位置は KL 0.001 を超える。
層・演算別の比較と位置ごとの値は
[調査記録](../research/20261111/plamo3_translate.md) に記載する。
