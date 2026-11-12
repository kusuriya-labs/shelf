# PLaMo 3 Translate 2B: EXL3-INT8 の接続と BF16 基準値

2026-11-12、R9700（gfx1201）、ROCm 7.14、単一 GPU。投機なし、KV は BF16、
温度 0、`max_seq_len=2048`。対象の mul24 量子化成果物と `run.json` は確認時点で
出力先に存在せず、実モデルの読み込み、翻訳品質、KLD、top-1、最大 logits 差、
量子化直後と最適化後の速度比較は未実施である。

## 接続

変換器の PLaMo 実装は結合 QKV と gate/up の元重みを分割し、個別 projection
として保存する。Shelf は個別 tensor の形状と mul24 marker を検査し、共通の
EXL3 loader と linear kernel に渡す。embedding / norm の非量子化データ、BF16 または
mul24 の lm_head に対応する。実 checkpoint の実際の tensor 構成は未確認。

2B と同じ projection 幾何の 1 層合成 checkpoint（embedding と head はゼロ、
7 つの EXL3 4-bit projection、非量子化 norm）で `shelf-compute` の prefill と
4 token decode、異なる marker の拒否を確認した。これは loader と実行経路の
検証であり、正しい翻訳や量子化精度を保証しない。

## BF16 のサーバー測定

[測定スクリプト](plamo_batch.py) は日本語・英語各 4 文、入力 29〜52 token、
生成 15〜37 token の翻訳を使う。同時数 1 / 2 / 4 は全 8 文をそれぞれ 8 / 4 / 2
回に分け、8 以上は同時に投入する。8 を超える場合は同じ 8 文を繰り返す。
サーバーの prefix cache は既定値、warmup は 1 件、各条件 1 回である。
生成 token 数を HTTP リクエスト開始から全完了までの壁時計時間で割り、prefill、
scheduler、sampling、通信と完了待ちを含む。純粋な decode 速度ではない。
特に短いケースは揺らぎや cache の影響があり、性能改善判定には反復測定が必要。

```bash
export PATH=/tmp/venv-plamo/bin:$PATH
export LD_LIBRARY_PATH=/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib
export ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55
flock /tmp/r9700.lock /tmp/venv-plamo/bin/python docs/research/20261112/plamo_batch.py \
  --server "$PWD/build-gfx1201/shelf-server" \
  --model /home/homelab1/datapool/ai_models/safetensors/plamo-3-translate-2b \
  --concurrency 4
```


| 同時数 | 完了 token 数 | 壁時計秒 | 合計 token/s |
| ---: | ---: | ---: | ---: |
| 1 | 204 | 2.438 | 83.69 |
| 2 | 204 | 1.393 | 146.41 |
| 4 | 204 | 1.752 | 116.41 |
| 8 | 204 | 1.673 | 121.92 |
| 16 | 408 | 1.711 | 238.39 |
| 32 | 816 | 1.927 | 423.49 |
| 64 | 1632 | 2.521 | 647.45 |

[測定値と出力例](plamo3_bf16_batch.json)。同時数 4 と 8 の低下は 1 回の
測定のみでは原因を断定できない。入力を一斉に処理する prefill の影響を含む。

単一要求の動作確認では「今日は良い天気ですね。午後は公園を散歩しましょう。」を
“The weather is nice today, isn't it? Let's take a walk in the park this afternoon.”、
“The train will arrive in five minutes. Please wait behind the yellow line.” を
「列車はあと5分で到着します。黄色い線の内側でお待ちください。」と出力した。
いずれも BF16 の結果であり、量子化モデルの翻訳品質は未評価。

## 演算時間の参考値

`shelf-bench tg --context 40 --tokens 4 --mode greedy` と
`SHELF_QWEN35_PERF_STATS=1 SHELF_QWEN35_PERF_TIMING=1` で BF16 の単一要求を
計測した。イベント計測が入るため通常の性能値とは別扱いにする。4 token の
GPU 時間は 66.812 ms、wall は 66.873 ms。イベント区間の合計は 55.039 ms。
各区間は待ちや計測オーバーヘッドを含み、kernel の純粋な実行時間ではない。
linear は MLP、attention projection、head が集約され、個別には分離できていない。

| 演算 | イベント合計 ms/token | イベント合計に対する割合 |
| --- | ---: | ---: |
| BF16 linear | 9.328 | 67.8% |
| RMSNorm | 1.866 | 13.6% |
| split | 0.637 | 4.6% |
| RoPE | 0.444 | 3.2% |
| 残差加算 | 0.426 | 3.1% |
| paged attention | 0.349 | 2.5% |
| KV append | 0.233 | 1.7% |
| SiLU | 0.220 | 1.6% |
| 乗算 | 0.211 | 1.5% |
| sampling | 0.015 | 0.1% |
| embedding | 0.012 | 0.1% |

最適化経路 2040 dispatch、fallback 0。これは同時数 1 の結果であり、サーバーの
batch での kernel 別時間、scheduler / host 時間、head と MLP の個別時間は未測定。
量子化後に支配的な演算をこれから推定しない。

## 検証と残作業

- 合成 EXL3 の正常系・marker 異常系、PLaMo の lowering / BF16 checkpoint / 既存 BF16 loader の実モデル ctest は通過。
- 2B BF16 の PLaMo template 6 項目、サーバー 8 項目は通過。
- Qwen の `concurrency-cancel` は 5 テスト通過。構造化出力の実モデルテストは通過。
- EXL3 mul24 linear / grouped、Qwen lowering / linear selector の ctest は通過。
  初回の ctest には GPU ラベル付き 2 件が含まれていたが、誤って lock なしで起動した。
  うち lowering は GPU 不在で skip。上記の結果は全 4 件を正しい device 設定と
  `flock` 付きで再実行したものである。
- 量子化成果物が完成したら `run.json` と tensor 名・形式を確認し、実モデルの
  compute / server、teacher-forced KLD / top-1 / 最大 logits 差、日英・英日の翻訳を
  確認する。同じ workload の初期速度を測ってから、一変更ずつ効果を比較する。
- 今回は性能最適化を採用していない。batch 内訳・安定した反復測定も残る。
