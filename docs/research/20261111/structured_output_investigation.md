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

## M2 実装時の検証

- `test_token_bitmask` は R9700 上で成功し、GPU の mask 適用結果が CPU 参照と一致した。
- compute の stdio を直接使い、greedy と temperature 0.8 の各 4 token で指定した
  token のみが出ること、mask 待ち中に別 request が進み、待機 request を cancel
  できることを確認した。同じ検査は DFlash2 / MTP 有効時にも成功した。
- XGrammar 0.2.9 では `{"type":"integer","enum":["foo",1]}` に対して
  `"foo"` が受理される。型と enum が積集合にならないため、型に一致しない enum と
  object / array の enum はサーバが 400 で拒否する。
- thinking 有効、tools、stop との併用は 400。長さ上限で途中終了した出力は未完成の
  JSON になりうる。strict tools、tool_choice の schema 強制、Responses `text.format`、
  DFlash2 / MTP の投機と mask の併用は M3 の残作業である。

### 実生成での速度（R9700、Qwen3.8 27B EXL3、単回）

`structured_output_decode_bench.py` を用い、prefix cache と投機を無効にして
128 token、温度 0、同じ prompt を使用。比較元は `origin/main` の `34f3d42` を
同じ toolchain で別ビルドした。全 request が生成中の区間の合計 token/s を示す。

| 同時 request | main | M2 | 変化 |
| ---: | ---: | ---: | ---: |
| 1 | 27.59 tok/s | 27.67 tok/s | +0.30% |
| 4 | 98.68 tok/s | 98.57 tok/s | -0.12% |

単一 request を同一プロセスで再測定した平均 token 間隔は、通常 36.62 ms、
通常生成 token を 1 token ずつ許可する mask replay は 37.28 ms（差 +0.66 ms）。
XGrammar で長い string enum の mask を毎回計算し、IPC 往復を含めた場合は
37.26 ms（通常との差 +0.64 ms、27.05 tok/s）だった。XGrammar ケースは replay と
生成 token 列が異なる。各値は単回測定であり、変動と schema による差を含む。後者は
compute との往復を測定し、HTTP / SSE の処理時間は含まない。

指定 regression の初回実行は 25 test 通過・3 test 失敗。失敗は不正な grammar
flag の error code、および structured output を旧仕様の 400 と期待する 2 test
だった。修正後、該当 test を個別に再実行し、`test_compute_dflash2.py` 99 件、
`test_server_dflash2.py` 21 件、`test_server_unsupported.py` 76 件が通過した。
reasoning の thinking budget test 内の DFlash2 / 複数 GPU ケースは環境不足で skip。
追加した compute / server structured test は個別実行で成功し、core に登録した。

## M3 の実装と残課題

reasoning と JSON Schema を併用する場合、単一 close token を境界とする方式を採用した。
通常 decode のまま thinking 生成中は matcher を進めず、自然な close または budget による
強制 close の直後に compute が待機し、server が初期 mask を渡す。thinking 予算が無い
短い要求でも自然な close を待つ。生成上限まで close が現れない場合は回答は空になる。
close tag が単一 token でないモデルは 400 とする。Responses の `text.format` の
`json_schema` は Chat の schema 検証と matcher へ変換する。

strict tools は未実装で引き続き 400 とする。XGrammar 0.2.9 の `qwen_3_5` structural
 tag は `<tool_call>\n<function=...>` と `qwen_xml` parameter を提供するが、Shelf の
parser の `xml-inline` は値 parser 未指定のため parameter の値を文字列として返す。
したがって integer / boolean / object 等は schema に沿う型へ変換しなければならず、
string 値との区別や streaming 中の扱いを schema に基づいて検証する必要がある。
`any_order=true` は XGrammar 側で required と重複を検査しないため使えない。残作業は、
strict / 非 strict 混在時の分岐、auto の trigger と required / 名前指定の強制、既存の
prefill と thinking 境界との整合、schema の 400 検証、温度付き stream / 非 stream での
parser 後の引数検証である。Responses tools も同じ実装へ通す必要がある。

### M4 の CPU 検証途中

Qwen XML 用の structural tag 構築と CPU 単体テストを追加した。auto は
`<tool_call>` を trigger にして strict / 非 strict の関数へ分岐し、required と
名前指定はタグを先頭から制約する。strict 側は `any_order=false` とする。
必須項目の不足、integer への不正な値、名前指定と異なる関数を拒否することを
XGrammar 0.2.9 で確認した。これは文法単体の検証であり server には未接続。

特に、XGrammar は enum `" a "` の XML 値として前後の空白を含む文字列を受け入れるが、
既存 parser は parameter 値の前後を strip する。そのまま型変換するだけでは
schema 適合を壊す。grammar で検証済みの parameter 本文について、文字列を保持し、
文字列以外を JSON decode する CPU 補助関数を追加した。server への接続と、
生成テキストから正しい parameter 本文を安全に取り出す実装は未完である。文字列中の tool markup が parser の
区切りと衝突するケースも制約または parser の修正と検証が必要。非 strict の
既存挙動は変えない方針である。

投機 decode は引き続き無効化して通常 decode にフォールバックする。DFlash2 向けの
具体的な分割、検証、および MTP の残作業は [投機 decode の接続案](structured_output_speculative_design.md)
に記載した。M3 の reasoning 併用時と投機併用時の性能は未測定であり、M2 の測定値は
上記の通りである。issue #26 のうち通常 JSON Schema・未対応 schema の 400・stream / 非
stream・非制約速度・投機方針は M2 で検証済みだが、strict tool と投機 decode の動作検証は別途必要である。

### M3 実機検証（R9700、Qwen3.8 27B EXL3、単回）

`test_server_structured_output.py` で thinking 有効の Chat と Responses `text.format`
について、stream / 非 stream、temperature 0.8 の schema 適合と reasoning / content の
分離を確認した。thinking 無効の schema と `json_object`、非対応 schema の 400 も成功。
実行ログでは thinking 付き Chat の decode は 26.9 / 26.8 tok/s（各 115 token）、
Responses は 26.9 / 26.9 tok/s（各 67 token）。thinking と回答が混在する単回値であり、
同一 token 列の制約なし比較はないため、overhead の測定値ではない。投機併用は未測定。
最初の試行は server の staging が旧版で 400 となり、staging 更新後に再実行して成功した。

`test_compute_token_mask.py` でも、thinking budget 1 による強制 close を初期 mask の
対象外とし、close 後に新しい mask を適用する直接 protocol テストが通過した。
