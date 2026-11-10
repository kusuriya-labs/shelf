# PLaMo 3 Translate の公開実装調査

2026-11-11 に公開 BF16 checkpoint の config、modeling code、tokenizer、chat template を調査した。これは調査記録であり、Shelf の対応を意味しない。

- [0.8B](https://huggingface.co/pfnet/plamo-3-translate-0.8b/tree/8b4d42855622093e610ba9b90c2d2b59f8d5c15d)
- [2B](https://huggingface.co/pfnet/plamo-3-translate-2b/tree/cf66d22fbe6202c56b8e5d283691aa1da53d05af)
- [8B](https://huggingface.co/pfnet/plamo-3-translate-8b/tree/cc265d585980c39436db5b9c136cd6644f446807)

## 形状と構造

| 項目 | 0.8B | 2B | 8B |
| --- | ---: | ---: | ---: |
| 層数 | 16 | 24 | 32 |
| hidden | 1536 | 2560 | 4096 |
| MLP 中間幅 | 6144 | 10240 | 16384 |
| query head | 16 | 24 | 32 |
| KV head | 1 | 4 | 4 |
| head 次元 | 128 | 128 | 128 |
| 全域 attention 層 | 2 | 3 | 4 |
| 局所 attention 層 | 14 | 21 | 28 |

全モデルで `model_type=plamo3`、語彙 107520、embedding と lm_head は非共有、線形層は bias なし。embedding のスケーリングは無効。query 幅は hidden 幅と必ずしも一致しない。結合した `qkv_proj` は行方向に Q、K、V を並べ、形状は `[128 * (query heads + 2 * KV heads), hidden]`。`o_proj` は `[hidden, 128 * query heads]`。`gate_up_proj` は `[2 * intermediate, hidden]` で、前半に SiLU を適用して後半と掛ける。`down_proj` は `[hidden, intermediate]`。量子化時には結合行列の分割順と小さい KV 幅を考慮する必要がある。

各層は attention 前 RMSNorm、attention、attention 後 RMSNorm、残差加算、MLP 前 RMSNorm、SwiGLU MLP、MLP 後 RMSNorm、残差加算の順。Q/K は head ごとに RMSNorm する。RMSNorm の epsilon は 1e-6、重みは offset を加えて適用する。offset は attention 前・MLP 前・Q/K・最終 norm が 1、attention 後が 1/5、MLP 後が 1/(5**1.5)。参照実装は正規化値を入力 dtype に戻してから重みを掛けるため、単純に重みを前処理して丸める場合は BF16 の差に注意する。

0 始まりの層番号で `(layer + 1) % 8 == 0` の層が全域、それ以外が窓幅 2048 の causal sliding attention。局所 mask は `key > query - 2048`。局所 RoPE は theta 10000 の通常 RoPE、全域 RoPE は theta 1000000、factor 256、元の長さ 4096、beta_fast 32、beta_slow 1、truncate false の YaRN。参照実装は YaRN の振幅に `1 + 0.1 * log(256)` を使う。最大位置数 262144 は config の値であり、実運用でその長さを検証したことを意味しない。参照実装は KV cache に RoPE 適用前の K を保存し、局所 cache の切り詰め後に局所座標で回転するため、長い文脈の prefill/decode 一致を個別に検証する必要がある。

## 翻訳 prompt と停止

配布された tokenizer は `Plamo3Tokenizer` の custom code と `tokenizer.jsonl` を使用し、`numba` が必要。配布 chat template は `target_lang` を必須とし、任意の `source_lang`、`context`、`glossary`、日本語向け `form` を受け取る。system message や tool call は受け付けず、user/assistant の翻訳例の後に user を置く。PFML3 の `dataset=translation`、`input`、`output` を専用トークンで記述する。生成停止は `generation_config.json` の token ID 1 と 16 の両方が必要。現在の server の `chat_template_kwargs` は thinking 関連のみを許可するため、翻訳パラメータをそのまま通せない。

## ライセンス

[配布ライセンス](https://plamo.preferredai.jp/info/plamo-community-license-en) の原文を確認すること。商用利用は事前登録と年間売上高 10 億円以下の双方が条件で、それ以外は PFN の別途ライセンスが必要。再配布時は契約書の提供や `Built with PLaMo` の表示等が必要。翻訳出力の公表には conversion model の例外と長文出版物の表示条件がある。生成物を使ったモデルの学習や生成物の第三者提供にも条件がある。

## 実装上の未検証事項

Shelf の Qwen 実装は gated query、Qwen 固有の層構成と weight 名を前提としており、この checkpoint の identity を許可するだけでは動作しない。独立した model lowering と loader、局所 attention と複数 RoPE profile、停止 token と翻訳 template 引数の server 接続が必要。EXL3 の形式そのものと checkpoint 固有の変換器・loader の対応は別問題であり、現時点では量子化・品質・速度を測定していない。
