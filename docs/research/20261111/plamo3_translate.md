# PLaMo 3 Translate の公開実装調査

## M4 量子化の着手時点の障害（未実装）

`rocm_exl3` の 1dfc6dc と Shelf の現行 loader を確認した。PLaMo の
4bpw checkpoint はまだ作成しておらず、量子化品質・速度も未測定である。
量子化器に architecture 名を追加するだけでは、次の点で変換できない。

- 配布 checkpoint は `tokenizer.jsonl` と独自 tokenizer を使用する。
  量子化器の `Tokenizer` は `tokenizer.json` を直接開くため、標準の
  calibration data をトークン化するには adapter が必要である。
- 量子化器の `Attention` の `key_fused_qkv` は入力を Q、K、V に分けて
  それぞれ別の `q_proj` / `k_proj` / `v_proj` として量子化する。
  `GatedMLP` の `key_fused_gate_up` も gate と up に分ける。一方 Shelf の
  PLaMo loader と lowering は結合した qkv と gate_up を各 1 行列として扱う。
  形式を統一し、0.8B の Q=2048、K=V=128 を含む不等幅で検証する必要がある。
- 局所層と全域層の RoPE 設定と 4 個の offset RMSNorm を calibration の
  forward に反映する必要がある。特に既存の YaRN 実装は original 長が指定
  されると factor を `max_position_embeddings / original_max_position_embeddings`
  から求める。PLaMo の値は 262144 / 4096 = 64 だが、参照実装の factor は
  256 であり、そのまま流用すると全域層の calibration が変わる。
- `RMSNorm` は load 時に重みを FP16 に変換し、既存 block も内部を FP16 で
  実行する。参照の BF16 中間丸めとの差を確認し、calibration に許容できるか
  判定する必要がある。単に登録して得た checkpoint を検証済みとは扱えない。

変換器は module 完了時に checkpoint を保存し `--resume` で再開できる。
`--max_module` は境界で終了するが、checkpoint は時間間隔に依存するため、
分割実行では `--checkpoint_interval 0` と保存済み `next_module_idx` の確認が
必要である。この組み合わせによる PLaMo の再開は未検証。量子化の前に再開と
ロック解放を短い実行で確認する。量子化器の変更用 worktree は
`/home/homelab1/coding-local/rocm_exl3-plamo3`、branch は `feature/plamo3-quant`。

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

## 実装作業の途中結果

サーバーの request → chat template 経路に翻訳用引数を渡す変更を追加した。配布 tokenizer と模擬 compute による HTTP テストでは、引数付き prompt の一致、通常応答、streaming、必須引数と入力の拒否を確認した。これはモデル本体の推論や実際の翻訳品質の検証ではない。現時点では server の起動時 reasoning capability 検査と compute の model identity gate が PLaMo を拒否するため、PLaMo はまだ使用できない。

0.8B と 2B の CPU BF16 参照について、日英・英日の入力をそれぞれ 23・26 token に変換し、全位置の logits を `/tmp/plamo3_reference_{0,1}.pt` と `/tmp/plamo3_2b_reference_{0,1}.pt` に保存した。同じ token 列と `shelf-bench quality` 形式の参照は `/tmp/plamo3_case_{0,1}.{tokens,quality}` と `/tmp/plamo3_2b_case_{0,1}.{tokens,quality}` に保存した。これらは一時ファイルであり、比較自体は GPU 実装が必要で未実施である。

量子化も未実施。`rocm_exl3/exllamav3/architecture/architectures.py` の architecture 登録には PLaMo 3 がなく、既存の Qwen 用変換をそのまま適用する根拠はない。PLaMo 3 の変換用 architecture と shelf 側の量子化 weight loader、結合行列の配置を対応させ、BF16 との比較を行う必要がある。

0.8B の CPU 参照で、BF16、greedy、最大 80 token、停止 token 1 / 16 の条件で次の出力を得た。これは Shelf の推論結果ではない。

| 入力 | CPU 参照出力 |
| --- | --- |
| 今日は良い天気ですね。 | The weather is nice today, isn't it? |
| 明日の午後、駅でお会いしましょう。 | I'll see you at the station tomorrow afternoon. |
| The train will arrive in five minutes. | 列車は5分後に到着します。 |
| Please let me know if you have any questions. | ご質問があればお知らせください。 |

## M1 推論実装の途中経過

GDN を持たない attention-only モデルに向けて、状態を確保しない空の GDN pool と系列・prefix cache の操作を追加した。既存 Qwen のダミー pool は維持し、executor は GDN が実際に 0 層のとき空 pool も受け付ける。

PLaMo 0.8B の結合 QKV は幅 2304 を 2048 と 256 に分け、後者を幅 128 の K と V に分ける必要がある。従来の correctness split は出力幅が等しい場合しか動作しなかったため、不均等分割を追加した。RMSNorm には正規化結果と offset 加算後の weight を BF16 に丸める実装を追加した。既存 Qwen の RMSNorm とは別の mode であり、最適化済み kernel は未実装である。

RoPE 周波数表は executor が 32 要素に固定していたため、rotary 次元に応じて確保するよう変更した。ただし executor が使用する周波数表は現状 1 種類であり、PLaMo の局所 RoPE と全域 YaRN を同一 request の層ごとに切り替える接続は未実装である。BF16 の cos/sin と Q/K の丸めも参照実装に合わせて検証する必要がある。

これらの部品はモデルの接続を意味しない。PLaMo 用 config・BF16 weight loader、非 gated attention と 4 個の norm を含む lowering、層別 RoPE、2048 token の admission 制限、compute / server の family 接続と停止 token が残る。GPU logits、翻訳品質、性能はまだ測定していない。M2 以降の 2B / 8B、真の SWA kernel、`rocm_exl3` の architecture 登録と量子化、長文脈の検証も未着手である。

実モデルで `shelf-compute --input-ids-file /tmp/plamo3_case_0.tokens --max-new-tokens 1` を試した結果、`check_config_identity` が `model_type='plamo3'` を未対応として拒否した。これは weight 読み込み前の失敗である。identity の許可だけでは Qwen 用 loader や gated attention に進むため不十分である。

GPU の `test_prefix_cache`、`test_elementwise`、`test_rope`、`test_rope_yarn`、`test_qwen35moe_executor` は通過した。CPU の `test_executor_footprint`、`test_program_workspace_lifetime`、`test_qwen35_tp_context`、`test_qwen35_rope_selector`、`test_qwen35_rope_profile` も通過した。これらは部品と既存モデルの限定的な検証であり、PLaMo の end-to-end 推論の検証ではない。

## M1b 設定読み込みの途中経過

`plamo3` の config を検出して専用 reader で層数、head 形状、FFN 幅、局所・全域 RoPE、window と全域層周期を読み取る経路を追加した。公開 0.8B と 2B の config で値を読み取り、7 層目が局所、8 層目が全域であること、および不正な負の head 数の拒否を確認した。ビルドと CPU の `test_model_identity` は通過した。

この段階では weight loader と lowering、複数 RoPE 表の executor 接続が未実装である。`probe_model_dir` は PLaMo の設定を検証した後、未接続であることを明示して拒否する。誤って Qwen 用の loader に入らないよう、共通 identity gate に `plamo3` はまだ追加していない。2048 token の admission 制限、GPU 生成、品質・速度計測、Qwen GPU 回帰も未実施である。したがって M1b の受け入れ条件は満たしていない。

## M1c BF16 の実行と比較

0.8B BF16 の専用 loader と lowering、層別 RoPE、compute 接続を追加した。
2048 token 以下では全層を causal attention とし、上限を超える設定を拒否する。
モデル重み 1.66 GiB、GDN 0 層、attention 16 層、KV head 1 で生成できた。
R9700 の greedy 実測は次の通り。decode は最初の token を含まない計測。

| 入力 | Shelf の出力 | prompt | prefill | decode |
| --- | --- | ---: | ---: | ---: |
| 今日は良い天気ですね。 | The weather is nice today, isn't it? | 23 token | 257.4 ms | 8.38 token/s（9 token） |
| The train will arrive in five minutes. | 列車は5分後に到着します。 | 26 token | 270.0 ms | 8.22 token/s（6 token） |

CPU BF16 の保存済み logits と teacher-forced に比較した。平均 KL は各 0.000628985、
0.000609844、top-1 は全 21 / 24 位置で一致、top-5 overlap は 0.990476 / 0.966667、
全語彙の最大絶対差は各 0.625。一部の位置では KL が 0.001 を超える。
以下の max 差は当該位置の全語彙での logits の最大絶対差。

| case | position | KL | top-1 一致 | max 差 |
| ---: | ---: | ---: | :---: | ---: |
| 0 | 2 | 2.9937062e-06 | ○ | 0.012084961 |
| 0 | 3 | 0.00032394649 | ○ | 0.1875 |
| 0 | 4 | 0.00069703875 | ○ | 0.1875 |
| 0 | 5 | 3.678904e-06 | ○ | 0.015625 |
| 0 | 6 | 2.106797e-06 | ○ | 0.01171875 |
| 0 | 7 | 4.3460275e-06 | ○ | 0.0234375 |
| 0 | 8 | 0.00012998322 | ○ | 0.1875 |
| 0 | 9 | 0.0017455346 | ○ | 0.1875 |
| 0 | 10 | 1.950382e-06 | ○ | 0.01171875 |
| 0 | 11 | 3.4503706e-05 | ○ | 0.046875 |
| 0 | 12 | 5.7939028e-06 | ○ | 0.017089844 |
| 0 | 13 | 0.00062755681 | ○ | 0.140625 |
| 0 | 14 | 0.00063271461 | ○ | 0.15625 |
| 0 | 15 | 0.0014491159 | ○ | 0.21875 |
| 0 | 16 | 0.0001806301 | ○ | 0.203125 |
| 0 | 17 | 2.7609799e-06 | ○ | 0.015625 |
| 0 | 18 | 0.0032941659 | ○ | 0.4375 |
| 0 | 19 | 0.0027798944 | ○ | 0.3125 |
| 0 | 20 | 0.0010530135 | ○ | 0.21875 |
| 0 | 21 | 3.4891183e-05 | ○ | 0.041015625 |
| 0 | 22 | 0.00020205761 | ○ | 0.625 |
| 1 | 2 | 2.9937062e-06 | ○ | 0.012084961 |
| 1 | 3 | 0.00032394649 | ○ | 0.1875 |
| 1 | 4 | 0.00069703875 | ○ | 0.1875 |
| 1 | 5 | 3.678904e-06 | ○ | 0.015625 |
| 1 | 6 | 2.106797e-06 | ○ | 0.01171875 |
| 1 | 7 | 4.3460275e-06 | ○ | 0.0234375 |
| 1 | 8 | 0.00012998322 | ○ | 0.1875 |
| 1 | 9 | 0.0017455346 | ○ | 0.1875 |
| 1 | 10 | 1.950382e-06 | ○ | 0.01171875 |
| 1 | 11 | 1.4288741e-05 | ○ | 0.046875 |
| 1 | 12 | 1.2228409e-05 | ○ | 0.03125 |
| 1 | 13 | 0.00080340004 | ○ | 0.171875 |
| 1 | 14 | 0.00086286856 | ○ | 0.15625 |
| 1 | 15 | 0.0015381505 | ○ | 0.1875 |
| 1 | 16 | 0.00012143369 | ○ | 0.078125 |
| 1 | 17 | 0.00062046962 | ○ | 0.15625 |
| 1 | 18 | 0.0013573873 | ○ | 0.25 |
| 1 | 19 | 0.00024479482 | ○ | 0.125 |
| 1 | 20 | 5.2228493e-06 | ○ | 0.017578125 |
| 1 | 21 | 0.0004285483 | ○ | 0.51953125 |
| 1 | 22 | 0.0010716165 | ○ | 0.15234375 |
| 1 | 23 | 0.0012040586 | ○ | 0.25 |
| 1 | 24 | 2.8711739e-05 | ○ | 0.046875 |
| 1 | 25 | 0.0034115041 | ○ | 0.625 |

同じ 23 token の prefill について CPU hook と `SHELF_LAYER_DUMP_DIR` を比較した。
0 層出力から差があり、mean absolute difference 0.001454、最大差 0.0625。
0 層の pre-mixer norm は全要素一致、QKV projection は最大差 0.0009765625、
Q/K の RoPE は最大差それぞれ 0.0000038147 / 0。最初に差が広がるのは paged
attention の出力で、平均絶対差 0.000913、最大差 0.125（参照 27.375、Shelf 27.25、
BF16 の 1 ULP）。その後の o_proj の最大差は 1。CPU SDPA と GPU の汎用 attention の
演算順・丸め差は候補だが、単一の原因として確定したわけではない。

RoPE の 2 テーブルは公開モデルの CPU BF16 参照 fixture と GPU で比較し、
`test_rope`、`test_rope_yarn`、`test_qwen35_rope_selector`、
`test_qwen35_rope_profile`、`test_qwen35moe_executor` は通過した。
0.8B checkpoint の全 tensor 名・BF16 dtype・shape の検証と GPU ロードも通過。
サーバー経由の実翻訳、2B / 8B、2048 超、量子化は未検証。

Qwen の `--group core` は最初、パイプ入力の format 判定で先頭 token が落ちる
実装ミスにより DFlash2 と server / CLI 一致の 3 テストが失敗した。修正後に同じ
3 個別テストを再実行し、DFlash2 99 件、unaligned prefill 7 件、server-chat 10 件が
すべて通過した。core グループ全体を修正後に再実行したわけではない。text / binary の
通常ファイルと stdin パイプの 4 条件も専用テストで通過した。全回帰は実行していない。

## M2: 最適化経路の調査

PLaMo の BF16 出力・head 次元 128 は paged attention selector の許可リストに
含まれていなかった。256 次元専用の tiled prefill はそのまま使えないため、
128 次元では prefill の行にも部分 head 用 decode kernel を使用する。局所層の
真の sliding-window kernel は未実装であり、2048 token の制限は維持する。

`SHELF_OP_FALLBACK_DEBUG=1` で 0.8B の prefill と decode を調べると、
attention 以外にも BF16 linear、offset RMSNorm（Q/K の head 単位を含む）、
RoPE、KV append、split、SiLU、mul、残差加算などが汎用経路であった。
0.8B の linear 形状も一時的に高速経路へ載せた実験では attention と linear の
fallback は消え、greedy の日英出力は一致したが、英日の teacher-forced top-1 は
95.83% だったため、この linear 変更は採用しなかった。平均 KL は日英 0.000958、
英日 0.000953。これは精度条件を満たす実装として扱わない。

その実験の semantic timing（イベント計測を含む、短い decode 4 token）では、
RMSNorm が約 34.3 ms/token、split が 9.0、RoPE と residual が各 6.0、KV append が
4.0、SiLU と mul が各 3.0、linear が 3.0、attention が 0.2 ms/token だった。
これらは実行間の待ちを含みうる時間で、kernel 単体の純粋な GPU 時間ではない。
150 token/s には attention の selector だけでなく、offset RMSNorm と多数の
汎用演算の高速化が必要である。

attention のみを最適化した採用候補では、日英の greedy token 列は旧経路と一致し、
23 token prefill は 250.4 ms、decode は 10.10 token/s（9 token）だった。
CPU 参照との平均 KL は日英 0.000525、英日 0.000493、top-1 はそれぞれ 100%。
`test_paged_attention`、selector、planner、split reduce の 4 テストは通過した。
速度目標は未達である。

## M2: 2B / 8B の BF16 実測

8B の shard 4 個は取得済みで、保存済みの CPU BF16 logits を teacher-forced 比較に
使用した。R9700、単一 GPU、greedy、prompt 23 / 26 token、decode は最初の token を
除いた計測。arena は自動確保で KV budget に空き VRAM を割り当てるため、使用量は
モデルの最低必要 VRAM ではない。

| model | 方向 | 出力 | prefill | decode | 平均 KL | top-1 一致 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 2B | 日→英 | The weather is nice today, isn't it? | 461.2 ms | 7.26 token/s | 0.000551 | 95.24% |
| 2B | 英→日 | 列車は5分後に到着します。 | 502.5 ms | 7.18 token/s | 0.000492 | 95.83% |
| 8B | 日→英 | The weather is nice today, isn't it? | 1646.4 ms | 3.01 token/s | 0.000700 | 100% |
| 8B | 英→日 | 列車はあと5分で到着します。 | 1818.3 ms | 3.00 token/s | 0.000968 | 100% |

2B の top-1 は 21 / 24 位置中それぞれ 1 位置が不一致。8B の重みは 15.89 GiB、
KV budget 14.40 GiB、arena 使用 30.340 GiB（確保 31.289 GiB）で 32 GiB に収まった。
2B の重みは 5.36 GiB、arena 使用 30.338 GiB。8B の真の最低使用量はこの計測では
求めていない。2B の精度判断と、いずれのサイズも速度改善が残る。

## M3: 全演算の高速化

R9700、BF16、greedy、prompt 23 / 26 token、decode は最初の token を除く。
CPU BF16 参照との teacher-forced 比較は 21 / 24 位置。すべての短文で greedy
出力は M2 と一致し、平均 KL ≤ 0.001、top-1 一致率 ≥ 90% を満たした。

| model | 方向 | prefill ms | decode token/s | 平均 KL | top-1 一致 |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.8B | 日→英 | 24.7 | 247.38 | 0.000752 | 100% |
| 0.8B | 英→日 | 24.3 | 246.13 | 0.000577 | 100% |
| 2B | 日→英 | 40.9 | 90.45 | 0.000465 | 95.24% |
| 2B | 英→日 | 41.6 | 90.45 | 0.000700 | 95.83% |
| 8B | 日→英 | 109.0 | 33.27 | 0.000317 | 100% |
| 8B | 英→日 | 109.7 | 33.25 | 0.000649 | 100% |

初期の全並列 RMSNorm では 8B 英→日の KL が 0.001724 に悪化した。Q/K head の
128 要素だけ correctness 経路と同じ順序で FP32 集計し、より広い norm は並列に
集計することで品質と速度の両方を満たした。全 norm を逐次集計した候補は 8B の
KL が 0.000915 だったが 23.32 token/s に低下したため採用しなかった。

`SHELF_QWEN35_PERF_TIMING=1` と fallback 診断で context 23、decode 4 token を
計測。0.8B / 2B / 8B はそれぞれ 1,368 / 2,040 / 2,712 dispatch 全件が最適化経路で、
fallback は 0 件だった。下表のイベント計測は同期・待ち時間を含み、kernel 単体の
純粋な時間ではない。

| 演算 | 0.8B ms/token | 2B ms/token | 8B ms/token |
| --- | ---: | ---: | ---: |
| BF16 linear | 2.921 | 9.347 | 27.396 |
| offset RMSNorm | 1.165 | 1.892 | 2.834 |
| split | 0.433 | 0.641 | 0.849 |
| RoPE | 0.299 | 0.445 | 0.581 |
| 残差加算 | 0.290 | 0.429 | 0.567 |
| paged attention | 0.211 | 0.302 | 0.402 |
| KV append | 0.159 | 0.236 | 0.310 |
| SiLU | 0.152 | 0.222 | 0.288 |
| BF16 乗算 | 0.140 | 0.214 | 0.280 |
| embedding / scale / output gather / sampling | 各 0.009〜0.010 | 各 0.009〜0.010 | 各 0.009〜0.010 |

途中版の 0.8B では RoPE が 6.095 ms/token、乗算が 3.278 ms/token で fallback
していた。最終版は BF16 の中間丸めと局所・全域 RoPE 表を保つ専用 kernel を使う。
非等幅 QKV split、offset RMSNorm、BF16 linear、各要素演算・KV append、embedding、
logits 変換と sampling の形状も最適化経路に接続した。現在は BF16 linear と norm が
主要な時間を占める。

英語の 218 token の段落を日本語へ greedy 翻訳し、246 token を生成した。prefill
1187.7 ms、decode 226.94 token/s（245 token）で、3 段落の構成と内容を保つ出力を
確認した。これは 2048 token 以下の確認であり、真の sliding-window kernel や
長文脈の一致確認ではない。

`test_rmsnorm`、`test_elementwise`、`test_rope*`、`test_paged_attention`、
`test_stochastic_sampling` と `test_qwen35_*` の関連テストは通過した。既知の
破損 fixture `test_qwen35_gdn_gate_reference` は失敗し、最終の対象テストでは除外。
2 GPU の任意テスト 2 件は skip。サーバー `--group core` は 16 件通過・1 件 skip、
skip された PLaMo template テストは個別に tokenizer を指定して 6 件通過した。
全回帰は実行していない。

EXL3 量子化は未実施。`rocm_exl3` には PLaMo 3 の architecture 登録がなく、
Shelf の専用 loader は BF16 tensor のみを検査する。結合 QKV と gate/up の
重み名・行順序、offset RMSNorm、`EXL3_MUL1_MULT=0xDCD12D` で作る mul24 の
変換器と loader を実装し、BF16 と KLD・翻訳・速度を比較する必要がある。

## M5: 事前 token 化入力の部分対応

量子化器 worktree の `0fdcc5c` で、`--cal_data` の safetensors に入れた
`input_ids` を `--cal_data_no_tokenizer` と共に渡す場合、量子化器内の
tokenizer を初期化せず、モデル config の語彙数を使って ID を検査できるように
した。二次元整数 tensor、正の行列サイズ、負または語彙外の ID も検査する。
decision head の包装には tokenizer が必要なため、この組み合わせを拒否する。
`92b9507` の CPU テストで正常入力と不正入力の拒否を確認した。
`633936f` で日英対訳 JSONL を配布 HF tokenizer と翻訳 template で両方向に
token 化するツールを追加し、短い人工例の 64 token が入力経路を通ることを
確認した。実際の calibration 用コーパスは未取得であり、利用時はそのライセンスを
確認する。

これは入力経路のみであり、PLaMo architecture、層別 RoPE と BF16 丸めを含む
calibration forward、分割重みの Shelf loader、実コーパスの token 化は未実施。
既存の `TransformerBlock` は norm の出力を FP16 に固定し、`RMSNorm` も重みを
FP16 に変換するため、architecture を登録するだけでは HF の BF16 forward と
一致しない。0.8B の変換・再開、生成、KLD・速度・VRAM、2B は未実施。
