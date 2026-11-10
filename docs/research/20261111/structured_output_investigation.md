# 構造化出力の接続調査

この記録は調査段階のものであり、構造化出力はまだ実装されていない。

## 外部実装と接続条件

[XGrammar-2 論文](https://arxiv.org/abs/2601.04426) は TagDispatch、JIT 文法コンパイル、cross-grammar cache を説明している。[XGrammar の engine integration](https://xgrammar.mlc.ai/docs/latest/using_xgrammar/engine_integration.html) はモデルごとに tokenizer 情報と compiler、リクエストごとに matcher を持ち、CPU で次 token の bitmask を計算してサンプリング前の logits に適用し、選択した token を matcher に反映する構成を示している。推測 decode は draft の各位置のマスクと rollback が必要である。

調査時点で PyPI の最新版は xgrammar 0.2.9。wheel は PyTorch、TVM FFI、Triton などを依存として要求する。Python 側への導入は shelf-compute の libc++ ABI に直接依存しない一方、追加依存の容量と起動時の影響を評価する必要がある。C++ core を組み込む案の libc++ でのビルドは未検証である。

現状の通常 decode は最終 prefill でも最初の token をサンプリングする。以後の各 token と同様、初回にも正しいマスクを適用しなければならない。連続バッチでは request ごとの matcher と output row の対応付け、混在バッチで非制約行を変えないこと、cancel と切断の処理が必要になる。マスクは greedy、stochastic、top-k のどの経路でも選択前に掛ける必要がある。HIP graph のキャプチャと replay でマスクを更新できるかは未検証である。

prefix cache は prompt の計算状態を復元するものであり、生成部分の matcher は request ごとに初期化する必要がある。DFlash2 / MTP と併用する場合、draft の各位置の matcher 状態を追跡しなければならず、未対応のままなら明示的な拒否が必要となる。JSON Schema の未知・非対応キーワードを黙って無視しない検証と、token 上限や stop により途中で終了した際の扱いも必要である。

## pipe 往復の予備測定

`/tmp/shelfvenv/bin/python docs/research/20261111/structured_output_ipc_bench.py` を実行。Qwen3.8 の config の語彙サイズ 248320 に対する 31040 byte のマスクを base64 化して JSON Lines で子 process に送り、子が JSON parse と base64 decode を行って小さな JSON を返す。最初の 100 回を除く 1000 回を測定した。

| 項目 | 実測 |
| --- | ---: |
| 送信 JSON | 41415 byte |
| 中央値 | 0.1895 ms |
| p95 | 0.2594 ms |
| p99 | 0.3268 ms |

これは Python 同士の pipe と encode 済みマスクの parse / decode に限る予備測定である。XGrammar のマスク生成、base64 encode、C++ JSON parse、GPU 転送、スケジューリング、実際の decode、負荷時の混合バッチは含まれない。したがってこの値だけでは採用可否や end-to-end の性能を判断できない。制約あり／なしの速度、コンパイル時間、および同条件での本番 IPC は未測定である。

## XGrammar の CPU 予備測定

専用環境 `/tmp/venv-so` の xgrammar 0.2.9 と Qwen3.8 の tokenizer を使用し、
`name`（string）と `count`（integer）を必須とする object schema を測定した。
compiler の作成は 0.856 秒、schema の初回 compile は 36.7 ms だった。
同じ初期 matcher 状態で mask 生成を 100 回の warmup 後に 1000 回繰り返した
中央値は 0.00247 ms、p95 は 0.00251 ms だった。これは一つの文法状態に限る
CPU 単体の値であり、生成中の各状態、並行 batch、IPC、GPU 転送、decode の
実効速度は含まない。

この tokenizer を `TokenizerInfo.from_huggingface` にそのまま渡すと語彙数は
248077 になる一方、model config の vocab_size は 248320 である。mask の word 数と
compute の logits 行を一致させるには model config の vocab_size を明示する。
