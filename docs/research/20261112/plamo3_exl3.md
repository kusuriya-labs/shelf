# PLaMo 3 Translate 2B: EXL3-INT8 の接続と BF16 基準値

2026-11-12、R9700（gfx1201）、ROCm 7.14、単一 GPU。投機なし、KV は BF16、
温度 0、`max_seq_len=2048`。以下の BF16 基準値を取得した時点では量子化成果物が
存在しなかった。その後、実 checkpoint を用いた検証を末尾に追記した。

## 接続

変換器の PLaMo 実装は結合 QKV と gate/up の元重みを分割し、個別 projection
として保存する。Shelf は個別 tensor の形状と mul24 marker を検査し、共通の
EXL3 loader と linear kernel に渡す。embedding / norm の非量子化データ、BF16 または
mul24 の lm_head に対応する。実 checkpoint の構成は末尾を参照。

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

## 実 EXL3 checkpoint の確認

変換成果物は decoder 144 個が 4 bit、24 個が 5 bit、lm_head が 8 bit、
embedding と norm は非量子化である。mul24 marker は `0x00dcd12d`。重みは
2,006,936,944 bytes。`shelf-server` が起動し、日英・英日を含む 8 件の翻訳が
正常に停止した。例: 「今日は良い天気ですね。午後は公園を散歩しましょう。」は
“The weather is nice today, isn't it? Let's take a walk in the park this afternoon.”、
“The train will arrive in five minutes. Please wait behind the yellow line.” は
「列車は5分後に到着します。黄色い線の内側でお待ちください。」となった。

CPU の Transformers BF16 を参照し、同一 token 列で Shelf の量子化モデルを
teacher-forced decode した。context は先頭 1 token、比較は index 1 から末尾の
1 つ前まで。両端の 2 位置を含まない。KV は BF16、全語彙の logits で集計した。

| 入力 | 位置数 | 平均 KL | top-1 一致 | 最大 logits 差 |
| --- | ---: | ---: | ---: | ---: |
| 日→英、1 例 | 30 | 0.00578 | 96.67% | 1.875 |
| 英→日、1 例 | 30 | 0.00813 | 96.67% | 1.781 |
| 英語平文、1 例 | 7 | 0.05138 | 100% | 3.047 |
| 日本語平文、1 例 | 7 | 0.04698 | 71.43% | 1.823 |

変換側の同じ 4 例では KL が順に 0.00308 / 0.00597 / 0.04034 / 0.03956、
top-1 が 96.88% / 96.88% / 100% / 77.78% だった。変換側は全 32 / 9 位置で
FP16 実行しており、Shelf の INT8 経路・位置範囲とは異なる。変換側の別の 10 対での
KL 0.0147 / 0.0152、top-1 93.5% / 93.8% も入力が異なるため直接比較できない。
平文は 7 位置のみで一致率が大きく変動する。
正解訳部分に限ると、日→英 9 位置は KL 0.00187、top-1 100%、PPL は
BF16 1.07675 / Shelf 1.06105、英→日 6 位置は KL 0.00169、top-1 100%、
PPL は 1.14148 / 1.12558。変換側の同じ正解訳位置では KL 0.00102 / 0.00242、
top-1 は両方向 100%、PPL は 1.06477 / 1.11783 だった。
変換側の同一評価集合・WikiText と直接比較し、量子化誤差以外の差をさらに調べる必要がある。
PPL は template 位置も含むため、変換側の正解訳部分だけの PPL と比較しない。

再現には変換側 `eval/plamo3_logits.py` の reference mode で CPU BF16 logits を
保存する。`accelerate` がない環境では `device_map="cpu"` を除いて CPU で実行できる。
その出力を [比較スクリプト](plamo_quality.py) に渡す。GPU 実行は lock を取得する。

```bash
sed 's/, device_map="cpu"//' /home/homelab1/coding-local/rocm_exl3-plamo3/eval/plamo3_logits.py \
  > /tmp/plamo-exl3-cpu-reference.py
/tmp/venv-plamo/bin/python /tmp/plamo-exl3-cpu-reference.py reference \
  --model /home/homelab1/datapool/ai_models/safetensors/plamo-3-translate-2b \
  --reference /tmp/plamo-exl3-reference.pt
export LD_LIBRARY_PATH=/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib
export ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55
flock /tmp/r9700.lock /tmp/venv-plamo/bin/python docs/research/20261112/plamo_quality.py \
  --reference /tmp/plamo-exl3-reference.pt \
  --model /home/homelab1/datapool/rocm-exl3-rdna2/models/plamo-3-translate-2b-exl3-4.05bpw-mul24 \
  --bench "$PWD/build-gfx1201/shelf-bench" --workdir /tmp/plamo-exl3-quality
```

## 実 EXL3 のバッチ基準値

同じ 8 文、入力 29〜52 token、出力 14〜37 token、greedy、BF16 KV、投機なし。
server の `--max-concurrent-requests` は各 c、`--max-num-batched-tokens` は既定の
2048、arena は auto、prefix cache は既定値とした。各条件で server を起動し、
1 要求だけ warmup してから測定する。1 回目は c の昇順、2 回目は降順。
完了 token 数を HTTP 要求開始から全完了までの時間で割った値である。
この条件では batch 形状の初回実行時間が入る。

| c | server 完了 token/s 1 回目 / 2 回目 | decode token/s 1 回目 / 2 回目 |
| ---: | ---: | ---: |
| 1 | 127.0 / 128.2 | 138.0 / 139.0 |
| 2 | 225.6 / 227.5 | 270.4 / 267.7 |
| 4 | 392.7 / 395.1 | 517.5 / 519.1 |
| 8 | 167.1 / 167.2 | 967.6 / 969.5 |
| 16 | 321.4 / 317.8 | 1717.9 / 1727.6 |
| 32 | 564.3 / 538.7 | 2347.0 / 2333.8 |
| 64 | 974.4 / 959.3 | 3864.7 / 1159.5 |
| 128 | 1374.9 / 1458.7 | 4528.0 / 4525.8 |

decode は `shelf-compute --serve-stdio` 経由、prefix cache 無効、上限 64 token、
全要求の最初の token から最初の要求の完了までに観測された token 数を時間で割る。
停止 token は有効なので出力は上限より短い。短い重複区間への一時的な stall 混入で
c=64 は 0.133 / 0.442 秒と大きく揺れた。server 完了速度とは測定範囲が異なる。
同じ英日段落（出力 37 token）を全要求に使い、同一 batch を warmup した場合の
decode は c=1/4/16/64 で 135.0/510.0/1721.0/3779.9、逆順の再測定で
135.2/511.6/1719.5/3743.9 token/s。混合 workload の代表値ではない。

batch warmup 後の server 完了速度は EXL3 c=4 で 394.3 / 397.4、c=8 で
579.5 / 584.2、c=64 で 2041.2 / 2024.1、c=128 で 2218.5 / 2224.7 token/s。
c=64 は目標 4000 の約 51%。最適化前後の比較ではない。
BF16 では単一要求 warmup の c=2/4/8 が 147.6〜147.9 / 105.8〜106.1 /
122.6〜135.9 だったが、全 8 文を batch warmup した c=4 は 276.7、c=8 は
433.0〜433.5。短い warmup の場合、BF16 c=4 の一部 prefill の実行区間が
約 1.21 秒、EXL3 c=8 は約 0.89 秒となり、同じ batch の decode も待機した。
EXL3 c=8 は warmup 後に該当区間が約 41 ms に縮まり、prefix 復元は 0。
初回形状の処理が大きく影響するが、その内部原因は未特定。

c=32 の EXL3 起動時 KV budget は 599,616 token。最大出力 192 token と入力
52 token で見積もった必要量は約 7,808 token なので、この条件では容量不足を
示していない。c=64 の入力合計は 2048 を超えるため、prefill は複数 step になり得る。
詳細な出力数・時間・測定窓は [測定結果](plamo3_exl3_batch.json)、手順は
[server 用](plamo_batch.py) と [decode 用](plamo_decode_batch.py) に記録した。
server 用の `--warmup-batch` は測定前に同じ workload を全件流す。
