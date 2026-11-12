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

## 前回時点の検証と残作業

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

## 実 EXL3 の step 内訳

同じ英日段落を各要求に使い、batch warmup 後に decode を実行した。既存の
`SHELF_BATCH_STEPWALL=1` と [一時計測パッチ](plamo_batch_timing.patch) を用い、
各行数の 8 回目の step に HIP event を挿入した。パッチは収集後に外した。
event 計測は追加の launch と待機を含むため、通常 step の純粋な kernel 時間とは
みなせない。数値は [集計](plamo3_exl3_profile.json) に記録した。

| c | 通常 step 中央値 ms | scheduler 等の前処理 ms | `execute_batch` ms | token 転送等 ms | event 区間合計 ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7.614 | 0.001 | 7.580 | 0.033 | 10.68 |
| 4 | 7.883 | 0.006 | 7.848 | 0.032 | 11.08 |
| 16 | 9.306 | 0.019 | 9.255 | 0.033 | 11.48 |
| 64 | 16.758 | 0.043 | 16.660 | 0.034 | 19.03 |

中央値は対象行数の step のうち、計測捕捉などで 100 ms 以上かかったものを除外した。
`execute_batch` は GPU 完了の `hipStreamSynchronize` を含む。CPU 時間は壁時計時間に
ほぼ等しいが、GPU 完了の busy wait も含み、host の launch や scheduler だけの時間ではない。
server と compute の IPC の厳密な内訳は未測定。

別の[一時計測パッチ](plamo_exec_timing.patch) で `submit_batch` と `complete_batch`
を分けた同条件の中央値は、c=1/4/16/64 の提出が 4.665/4.928/5.977/10.086 ms、
完了が 2.982/2.879/3.221/6.618 ms。提出中も GPU は並行実行されるため、
これらを独立した GPU 時間と足し合わせてはいけない。

| c | EXL3 linear（うち head）ms | RMSNorm ms | attention ms | sampling ms | embedding ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.789（0.460） | 1.909 | 0.375 | 0.014 | 0.010 |
| 4 | 7.137（0.482） | 1.894 | 0.405 | 0.033 | 0.010 |
| 16 | 7.451（0.527） | 1.817 | 0.582 | 0.043 | 0.010 |
| 64 | 12.751（0.818） | 2.514 | 1.850 | 0.045 | 0.010 |

1 step に linear 169、RMSNorm 145、attention 24 が含まれる。head は最後の linear
区間で、linear 合計に含む。c=64 の残りは RoPE 0.507、残差 0.441、mul 0.317、
SiLU 0.303、KV append 0.232 ms など。event 区間の合計は通常 step より大きく、
overhead が無視できない。重み読み出し 2.8〜3 ms の単純な帯域上限に比べて実測は
遅いが、このデータだけで host 支配とは結論できない。

## 検討した経路と採用した変更

2B の down projection（2560×10240、4 bit、64 行）の split-K 数を 10 から
4 にすると、kernel 単体は 86.9〜87.3→77.7〜77.9 µs。server の c=64、
batch warmup 後は 2024〜2041→2063〜2066 token/s、同一段落の decode は
3744〜3780→3871〜3885 token/s になった。しかし同一の teacher-forced 30 位置、
64 sequence の CPU BF16 比較で、日→英 KL が 0.005168→0.005481 と悪化した。
英→日 KL は 0.007394→0.007311。split 数 6 は日→英 0.005088、英→日 0.007750、
5 は 0.005342 / 0.007548、8 は 0.006228 / 0.006569 で、いずれも片方向以上の
KL が悪化したため変更を外した。top-1 は元が 96.7% / 93.3%、split 4 が
96.7% / 96.7%。検証は同じ列を 64 sequence に与えたもので、多様な列での評価ではない。

prefill の既存 B 経路および fused 経路は初回形状の実行を回避し、EXL3 c=8 の
初回 server 完了速度を 167→567 / 576、c=64 を 959〜974→1916 / 1921 token/s
にした。一方 warmup 後の c=64 は 2024〜2041→1993 / 1958 と低下した。
256 行 prefill の後の 30 位置では、CPU BF16 比較の KL が基準 0.027705 から
両候補とも 0.027891 へ悪化し、top-1 は全て 93.3%。条件を満たさないため不採用。
256 行は batch の prefill 形状を通すための検証であり、長文対応の評価ではない。
比較は [スクリプト](plamo_prefill_quality.py) と CPU BF16 の参照出力で再実行できる。

```bash
/tmp/venv-plamo/bin/python - <<'PY'
import json
s = ('Our team compared several approaches before choosing the current design. '
     'The first version was easier to build, but it required more memory and became '
     'slower as the number of requests increased. We will continue to monitor '
     'performance after release. ')
open('/tmp/plamo-quality-long.jsonl', 'w').write(json.dumps({'text': s * 12}) + '\n')
PY
/tmp/venv-plamo/bin/python /tmp/plamo-exl3-cpu-reference.py reference \
  --model /home/homelab1/datapool/ai_models/safetensors/plamo-3-translate-2b \
  --reference /tmp/plamo-quality-long.pt --general-text /tmp/plamo-quality-long.jsonl --samples 1
flock /tmp/r9700.lock /tmp/venv-plamo/bin/python docs/research/20261112/plamo_prefill_quality.py \
  --reference /tmp/plamo-quality-long.pt \
  --model /home/homelab1/datapool/rocm-exl3-rdna2/models/plamo-3-translate-2b-exl3-4.05bpw-mul24 \
  --bench "$PWD/build-gfx1201/shelf-bench" --workdir /tmp/plamo-prefill-quality
```

既存の HIP graph を build で有効にした実験では、同一段落・batch warmup 後の
decode は c=1/4/16/64 で 150/438/1345/3770 token/s。通常 build の約
135/510/1720/3744〜3780 に対し c=4/16 が低下し、c=64 は改善しなかった。
各 1 回の探索であり、標準設定には採用していない。

採用した変更は、分割 projection の SiLU と乗算を 1 kernel にまとめるもの。
SiLU 出力の BF16 丸めを保持する。64 行の翻訳完了速度は 2024〜2041→2086〜2090、
同一段落の decode は 3744〜3780→3847〜3853 token/s。前後とも各 2 回、同じ条件で
batch warmup した。c=1/64 の日英・英日 30 位置、256 行 prefill 後の 30 位置の
logits dump は変更前後で byte 単位で一致した。kernel 単体でも元の 2 演算と完全一致した。
長さや batch が異なる実務 workload 全般の品質・速度はこれだけでは保証できない。

融合前後の全 c 掃引。各セルは昇順 / 降順の 2 回、token/s。server は単一要求のみを
事前実行する上記の条件、decode は混在入力の全要求が生成中の窓であり、いずれも
batch warmup 後の同一段落の値とは異なる。

| c | server 前 | server 後 | decode 前 | decode 後 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 127.0 / 128.2 | 129.5 / 129.7 | 138.0 / 139.0 | 139.5 / 140.1 |
| 2 | 225.6 / 227.5 | 231.2 / 176.9 | 270.4 / 267.7 | 271.1 / 271.7 |
| 4 | 392.7 / 395.1 | 400.6 / 400.9 | 517.5 / 519.1 | 524.5 / 523.2 |
| 8 | 167.1 / 167.2 | 188.3 / 172.7 | 967.6 / 969.5 | 967.3 / 972.5 |
| 16 | 321.4 / 317.8 | 321.8 / 325.4 | 1717.9 / 1727.6 | 1728.7 / 1737.5 |
| 32 | 564.3 / 538.7 | 570.6 / 572.1 | 2347.0 / 2333.8 | 2451.1 / 2443.2 |
| 64 | 974.4 / 959.3 | 989.3 / 966.4 | 3864.7 / 1159.5 | 3926.6 / 3939.3 |
| 128 | 1374.9 / 1458.7 | 1396.6 / 1400.2 | 4528.0 / 4525.8 | 4611.4 / 4581.7 |

融合後の c=2 の 2 回目は最後の 2 要求で decode 時間が約 0.3 秒延びた。
融合前 c=64 の decode 2 回目と同様、短い窓に対する一時的な停止の影響が大きい。
融合後の [測定結果](plamo3_exl3_batch.json) に token 数と窓の長さを記録する。

融合後 c=64 の batch warmup 条件で `--max-num-batched-tokens` を変更した場合、
512 は 2074.6 / 2070.0、1024 は 2073.8 / 2084.0、4096 は 2076.1 / 2078.5、
8192 は 2078.5 / 2077.1 token/s。既定 2048 の 2086〜2090 に対し改善は確認できず、
設定変更は採用しない。

同じ 8 文を繰り返す c=64・batch warmup の compute stdio 測定は、全完了まで
2069 / 2078 token/s。server の 2086 / 2090 と近く、この比較では server 側の
大きな追加負荷は見えない。ただし stdio は token ごとに出力、server は非 stream、
出力上限はそれぞれ 64 / 192、prefix cache は無効 / 有効（server の復元は 0）で、
生成 token 数も 1595〜1601 と変動するため、差を IPC 時間そのものとは解釈できない。
c=128 の stdio は 2478 / 2482 token/s。scheduler、launch、同期、IPC の厳密な
完全分解は引き続き未実施である。

## 今後の課題

- c=64 の server 完了速度 2086〜2090 token/s は最低目標 4000 の約 52%、
  差は約 1910 token/s。c=128 の server は最適化後の batch warmup を未測定。
- EXL3 linear は c=64 の最大項。分割 projection の grouped 実行、decode kernel の
  メモリ利用・演算の重なり、正規化と残差の融合などは未実装・未評価。
- server と compute の比較は上記の条件差があり、scheduler、GPU launch、同期、
  IPC を独立した時間として完全には測っていない。
- 品質は少数例での比較であり、変換側の held-out 10 対・WikiText と同じ全位置の
  比較や、独立した大きい評価集合は未実施。c=64 の品質確認は同じ列の複製であり、
  異なる翻訳を混在させた teacher-forced logits は未確認。
- 分割数の変更や prefill の別経路は今回の品質基準を満たさず、採用していない。

## 初回要求の遅延と起動時 warmup

再測定では「cold」を server 起動後の最初の HTTP 要求からと定義した。以前の
表は最初に 1 要求を流していたため、chat template の初回費用を含まない。
「warm」は先に 1 要求と同じ入力 batch 全体を実行した後の値。いずれも出力 token 数を
全要求の投入開始から全完了までの壁時計時間で割る。入力 29〜52 token、出力は
おおむね 14〜37 token の 8 文を繰り返し、greedy、BF16 KV、投機なし、prefix cache
有効、最大 batch token 数 2048。client と server は同一ホスト。各値は 1 回で、
短時間の停止や batching 順序の影響を受ける。

初回の chat template `prepare` は 850.7 ms、次の 2 回は 0.391 / 0.275 ms。
[CPU 計測スクリプト](plamo_template_timing.py) で計測した。別の
[一時計測パッチ](plamo_lt_init_timing.patch) で c=8 の最初の大きな prefill を
計測すると、INT8 hipBLASLt の最初の heuristic が 162.3 ms、最初の matmul API が
674.0 ms、両者を含む最初の GEMM 呼び出しが 838.8 ms。後続の異なる projection
形状の heuristic は 0.1〜0.4 ms 程度だった。c=16/64 でも最初の heuristic は
162.7/163.1 ms、最初の matmul は 658.4/675.7 ms。これらは CPU 側 API の
壁時計時間であり、matmul 内部の code object 読み込み、ライブラリ初期化、GPU 実行を
区別していない。HIP event の GPU 演算時間ではない。

以前の単一要求 warmup 後の c=8 では 7 要求の prefill がそれぞれ同じ 883.8 ms の
実行区間を共有した。256 行の起動時 warmup 後は 41.7 ms で、別の先行 1 要求の
prefill は 11.3 ms。これは各要求に共有区間を記録するため、7 倍してはいけない。
c=16 の同様の batch は 11.3 / 50.8 / 30.4 ms の複数区間、c=64 では 11.2 ms の
先行区間と約 42〜59 ms の複数区間がある。decode step の壁時計合計は冷間測定で
c=8/16/64 が 288.8/303.3/466.5 ms。server と compute の比較は前節の通り近く、
今回の結果は IPC を独立した時間として定量化しない。queue、token streaming、
完了処理は重複のない独立区間に分解できていない。

server の受付前に短い chat 生成を行い、c>=8 では 256 token の合成 prefill を
行う変更を採用した。対象は 2B の 4.05 bpw EXL3 / 8 bit head のみ。最大系列長や
batch token 上限が足りない場合は 256 token prefill を行わない。起動時の測定ログに
よる warmup 自体の費用は chat が約 1 秒、prefill が約 0.8 秒。モデル読み込みを含む
起動時間は、測定した範囲では warmup 追加前の約 9〜10 秒から約 10〜12 秒になった。
HTTP 受付開始後に起動処理の費用を負担しない。

起動 warmup 後の cold 測定について、同じ `prefill_ms` を持つ要求を同一の
実行区間とみなして概算すると、次の内訳になる。prefill 区間には同時に走った
既存要求の decode も含む。純 decode は `STEPWALL` の decode-only step の合計。
残差には queue、server の token 化、IPC、完了処理、prefill の scheduler 部分、
記録の丸めが混ざる。これは重複のない request 単位トレースではないので、
残差を IPC や queue の単独値と解釈できない。

| c | 全完了 ms | prefill / mixed 実行区間の概算 ms | decode-only step 合計 ms | 残差 ms |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 346.7 | 53.0 | 288.8 | 4.9 |
| 16 | 399.7 | 92.5 | 303.3 | 3.9 |
| 64 | 797.3 | 319.3 | 466.5 | 11.5 |

c=64/128 の cold と batch warm の間では、それぞれ 64 件中 2 件、128 件中 13 件の
訳文が異なった。これらの batch 形状の変化は warmup 追加前から存在し、原因は
未特定。一方、追加前の単一要求 warmup 条件と追加後 cold 条件、および追加前後の
batch warm 条件をそれぞれ比較すると、c=8/16/64 の全訳文は一致した。
変更前後の真の cold 同士の比較は c=8/16 では全件一致、c=64 では 64 件中
2 件が異なった。時間差と要求の到着順による batch 形状の変化を含むため、
warmup のみで logits が一致することの直接証明にはならない。

測定の例。`--skip-initial-warmup` は HTTP の最初の要求から測り、
`--warmup-batch` は対象 batch を一度完了してから測る。`startup_s` には
process の起動から `/v1/models` が利用可能になるまでを含む。0.1 秒間隔で
確認するため、最大約 0.1 秒の観測誤差がある。

```bash
export PATH=/tmp/venv-plamo/bin:$PATH
export LD_LIBRARY_PATH=/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib
export ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55
model=/home/homelab1/datapool/rocm-exl3-rdna2/models/plamo-3-translate-2b-exl3-4.05bpw-mul24
flock /tmp/r9700.lock /tmp/venv-plamo/bin/python docs/research/20261112/plamo_batch.py \
  --server "$PWD/build-gfx1201/shelf-server" --model "$model" --concurrency 64 \
  --skip-initial-warmup --log /tmp/plamo-cold.log
flock /tmp/r9700.lock /tmp/venv-plamo/bin/python docs/research/20261112/plamo_batch.py \
  --server "$PWD/build-gfx1201/shelf-server" --model "$model" --concurrency 64 \
  --warmup-batch --log /tmp/plamo-warm.log
/tmp/venv-plamo/bin/python -I docs/research/20261112/plamo_template_timing.py --model "$model"
```

HIP API の内訳を再取得するときは上記 patch を作業用 build に一時適用し、
`SHELF_EXL3_DIAG=1` を指定する。計測のたびに stderr 出力が発生するため、
その計測の速度を通常 build の最終値として扱わない。patch は計測後に外す。

### 初回と batch warm の結果

server は上記の翻訳 workload、decode は全要求に token が出た後から最初の要求が
完了するまでの共通窓。後者は prefill を含まず、両者は直接加算できない。変更前の
c=1/4 は chat warmup のみを持たない段階の値で、他は変更前 server。各 1 回、
c=128 の変更後だけ 2 回。decode の c=2/8/32 は前回の融合後の値であり、今回の
再測定ではない。各測定の token 数、経過時間、起動時間は
[結果 JSON](plamo3_startup_results.json) に記録した。

| c | 変更前 cold | 変更後 cold | 変更後 batch warm | decode |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 79.0 | 128.5 | 131.5 | 140.2 |
| 2 | 104.2 | 228.7 | 233.0 | 271.1 |
| 4 | 144.3 | 396.2 | 403.3 | 525.2 |
| 8 | 101.9 | 576.9 | 592.5 | 967.3 |
| 16 | 174.3 | 1000.7 | 1006.6 | 1768.3 |
| 32 | 326.4 | 1472.2 | 1489.1 | 2451.1 |
| 64 | 631.2 | 1999.2 | 2085.9 | 3926.9 |
| 128 | 875.7 | 2192.7 / 1731.1 | 1726.4 / 2245.8 | 4600.5 |

c=128 は通常 50〜60 ms 程度の prefill 区間が cold の一回で 387 ms、warm の
一回で 440 ms になった。初回ライブラリ初期化とは別の一時停止であり、原因と
再現条件は未特定。cold と warm が常に一致するとは言えない。

c=64 の batch warm 2085.9 token/s は 4000 の 52.1%、差は 1914.1。c=4 の
403.3 は 700 の 57.6%、差は 296.7。入力合計は出力合計より長く、短い出力が
先に終わるこの workload では、全要求が動いている decode 窓より完了速度が低い。
変更前からの decode 内訳では c=64 の step は約 16.8 ms、c=1 は約 7.6 ms、
重み読出しのみの単純下限は約 3 ms。c=64 の 10 ms 程度という以前の全体下限は
推定であり、今回の計測だけで達成可能な server 速度とはみなせない。目標に向けては
prefill だけでなく linear・normalization・launch の費用と、同時生成数が減る
末尾を改善する必要がある。

既存の `SHELF_EXL3_INT8_ROTATE_F4=0` を使って scalar 回転を試した。c=64 の
batch warm 完了速度は 2001.0 / 2000.6 token/s で、既定経路の 2085.9 および
前回の 2086〜2090 より遅く、不採用。c=64 の日英・英日各 30 位置の logits dump は
既定経路と byte 一致した。別 prefill 経路と split-K 変更は再採用していない。

CPU BF16 比の teacher-forced は c=1 の日英 / 英日 KL が 0.005784 / 0.008132、
top-1 が 96.67% / 96.67%、c=64 は KL 0.005168 / 0.007394、top-1 96.67% /
93.33%。各 30 位置、c=64 は同じ列の複製。前回保存した c=1/64 両方向の logits
と byte 一致するが、この bench は server 起動 warmup 後の状態を直接検査しない。
多様な列の混在、大きい評価集合、および server 上の warmup 前後 logits の直接比較は
未実施。

prefill / decode の scheduling では既存の `--max-num-batched-tokens` を
c=16 で変えた。batch warm 2 回の完了速度は 512 が 1012.6 / 997.3、1024 が
1004.2 / 1005.5 token/s。既定 2048 の 1006.6 と測定の揺らぎの範囲であり、
設定は変更しない。c=64 の 512〜8192 の探索は前節の通り。decode を優先する
新しい scheduler、最初の要求を短時間集める方式、grouped QKV / gate-up、
正規化・残差の融合、launch 削減は未実装・未評価である。

今回の関連検証は PLaMo EXL3 実モデル server の通常応答・stream と起動 warmup、
BF16 2B server、合成 EXL3 checkpoint、PLaMo template、EXL3 mul24 linear / grouped、
PLaMo / Qwen lowering と linear selector、Qwen structured output、および
`concurrency-cancel` グループが通過。warmup の要求拒否で稼働中の compute を
維持し、compute が停止済みなら起動を失敗させる CPU テストも通過。PLaMo の
実モデル loader ctest は初回に 2B を誤指定して 3 GiB arena が不足したため失敗し、
想定の 0.8B BF16 fixture を指定し直して通過した。全回帰 `--all` は実行していない。

## 拡大品質評価と継続投入の測定方法

現行の 2B EXL3、BF16 KV、投機なしの評価集合を拡張した。変換時評価の
Tatoeba held-out 先頭 10 対に、業務・交通・研究・料理・公園などの 8 対を追加し、
両方向の翻訳 template と正解訳を teacher forcing する。追加分は
[データ](plamo_quality_extra.jsonl) に置く。WikiText-2 raw test の先頭から
600 文字超の 4 段落は各 600 文字を一般文として別集計し、翻訳の採用判定には含めない。

同じモデルの CPU Transformers BF16 logits と、`shelf-bench quality` の同一 token
列を比較する。context は 1 token、各列の index 1 から末尾の 1 つ前までを集計する。
各方向は全位置の KL と top-1 を位置数で重み付けし、最大 logits 差も記録する。
64 行評価は各列を 64 本複製するため、異なる列が混在する batch の評価ではない。
正解訳部分だけの補助指標も case ごとに出力する。Tatoeba の出典とライセンスは
変換側の記録を参照し、原文の再配布を避けて固定 revision から再生成する。

再現手順。Python の `pyarrow` と CPU 版 PyTorch、Transformers が必要。データ生成は
入力ファイルの SHA-256 を検証する。WikiText は同じ revision の `wikitext-test.arrow`
を指定する。元の reference script では `accelerate` の `device_map` を使うが、
ここではそれを除いた CPU 実行を使う。

```bash
curl -fL -o /tmp/plamo-tatoeba.parquet \
  https://huggingface.co/datasets/sentence-transformers/parallel-sentences-tatoeba/resolve/cec1343ab5a7a8befe99af4a2d0ca847b6c84743/en-ja/train-00000-of-00001.parquet
python3 docs/research/20261112/plamo_eval_data.py \
  --tatoeba /tmp/plamo-tatoeba.parquet --wiki /path/to/wikitext-test.arrow \
  --output-dir /tmp/plamo-eval-data
sed 's/, device_map="cpu"//' /home/homelab1/coding-local/rocm_exl3-plamo3/eval/plamo3_logits.py > /tmp/plamo-cpu-reference.py
/tmp/venv-plamo/bin/python /tmp/plamo-cpu-reference.py reference \
  --model /home/homelab1/datapool/ai_models/safetensors/plamo-3-translate-2b \
  --reference /tmp/plamo-expanded-reference.pt --corpus /tmp/plamo-eval-data/heldout.jsonl \
  --general-text /tmp/plamo-eval-data/wiki.jsonl --samples 18
export LD_LIBRARY_PATH=/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib
export ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55
flock /tmp/r9700.lock /tmp/venv-plamo/bin/python docs/research/20261112/plamo_quality.py \
  --reference /tmp/plamo-expanded-reference.pt \
  --model /home/homelab1/datapool/rocm-exl3-rdna2/models/plamo-3-translate-2b-exl3-4.05bpw-mul24 \
  --bench "$PWD/build-gfx1201/shelf-bench" --workdir /tmp/plamo-expanded-quality \
  --batch-size 64 --prefix heldout-ja-en- --prefix heldout-en-ja- --prefix general-corpus-
```

[継続投入スクリプト](plamo_closed_loop.py) は翻訳方向ごとの既存 4 文を 1 / 2 / 4 / 6
文結合して 32 入力を構成し、seed 17 で固定順に並べる。要求完了ごとに次の入力を
投入して同時要求数 c を維持する。greedy、最大出力 320 token、prefix cache 既定値、
`max_seq_len=2048`。全 worker が少なくとも一度完了してからさらに 20 秒温め、
以降 90 秒の区間内に完了した要求の `completion_tokens` 合計を 90 秒で割る。
区間境界をまたぐ要求は完了時刻に全 token を計上するため、有限区間の境界誤差がある。
応答長と開始・完了時刻を JSON に残す。server は各実行で起動し直す。たとえば c=64:

```bash
flock /tmp/r9700.lock /tmp/venv-plamo/bin/python docs/research/20261112/plamo_closed_loop.py \
  --server "$PWD/build-gfx1201/shelf-server" \
  --model /home/homelab1/datapool/rocm-exl3-rdna2/models/plamo-3-translate-2b-exl3-4.05bpw-mul24 \
  --concurrency 64 --warmup 20 --duration 90
```

現行実装の拡大集合、c=64 の基準は次のとおり。実行ごとの詳細は
[JSON](plamo3_expanded_quality_baseline.json) に記録する。

| 集合 | 位置数 | 平均 KL | top-1 一致 | 最大 logits 差 | 採用上限 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 日→英 | 875 | 0.017130 | 92.80% | 4.125 | 0.018130 |
| 英→日 | 875 | 0.017396 | 93.37% | 7.734375 | 0.018396 |
| WikiText | 521 | 0.051676 | 90.21% | 4.03125 | 対象外 |

両方向とも現行の時点で絶対値 0.01 を超えるため、指示された例外に従い、
それぞれの未丸め基準値 + 0.001 を上限とする。この例外を使う場合も
「CPU BF16 比 KL < 0.01」を達成したことにはならない。top-1 の比較は同一位置で行う。[比較スクリプト](plamo_quality_paired.py) は
位置ごとの McNemar 検定に加えて、同一文内の位置の相関を考慮した文単位の
対応あり符号反転検定を計算する。判定には文単位の検定を使い、両方向の多重比較を考慮して片側 p < 0.025
で低下を検出した候補は採用しない。非翻訳文の WikiText は個別に観測する。

### 継続投入の現行基準

上記の固定分布で各 90 秒。1 回目は c=1 / 4 / 16 / 64 / 128、2 回目は逆順。
c=1 の最初の実行は CPU BF16 参照生成と重なり、入力の順序も旧版だったため除外し、
別途同じ固定順で測り直した。基準の結果と各回の出力長分布は
[JSON](plamo3_closed_loop_baseline.json) に記録する。計測された出力は 14〜167 token、
c=4 の四分位は下記 JSON を参照。短文は 30 token 未満も含む。

| c | 継続投入 token/s、1 回目 | 2 回目 |
| ---: | ---: | ---: |
| 1 | 128.79 | 127.68 |
| 4 | 458.82 | 459.13 |
| 16 | 1322.47 | 1332.48 |
| 64 | 2442.59 | 2448.87 |
| 128 | 2939.48 | 2935.28 |

c=64 の平均は約 2446 token/s で目標 4000 との差は約 1554 token/s、c=4 の
平均は約 459 token/s で目標 700 との差は約 241 token/s。前節の短文 burst や
純 decode の速度は測定範囲が異なるため、この値に加算しない。

### down projection の split-K 再評価

2B EXL3 の down projection に限り、17〜64 行の decode の split-K を既定の 10 から
4 にする候補を試した。BF16、他のモデル、prefill、投機 verify は元の設定を使う。
従来の短い共通 decode 窓では約 3.2% 改善していたが、今回の c=64 継続投入は
基準 2442.59 / 2448.87 に対して 2425.38 / 2461.72 token/s。平均では約 0.09%
低下し、測定差より明確な改善は見えなかったため**不採用**とし、コードを復元した。

| 集合 | 基準 KL | 候補 KL | 差 | 候補 top-1 | 候補の最大 logits 差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 日→英 | 0.017130 | 0.017976 | +0.000846 | 93.03% | 5.5 |
| 英→日 | 0.017396 | 0.014623 | -0.002773 | 94.06% | 5.90625 |
| WikiText | 0.051676 | 0.050515 | -0.001162 | 90.21% | 4.09375 |

日→英は上限 0.018130 との差が約 0.000154。文単位の片側符号反転検定の p は
日→英 0.820、英→日 1.0 で、有意な top-1 低下は検出しなかった。Qwen3.8 27B
EXL3 の c=64 と PLaMo 2B BF16 の c=1 では各 33 位置の logits が変更前と byte
一致した。Qwen3.5 2B EXL3 checkpoint は非量子化 tensor の dtype が非対応で読み込めず、
Qwen3.8 に切り替えた。候補の case 別値と検定結果は
[JSON](plamo3_split4_experiment.json) を参照。品質上限を満たしたが、性能根拠がない。

継続投入の server ログにも 300 ms 超の prefill 区間が散発した。起動時の合成
256 token warmup を除くと、c=64 は 2 回で 1 件（334.5 ms）、c=128 は 2 回で
6 件（413.3 ms が 1 件、302.0 ms が 5 件）。302.0 ms の 5 件は異なる要求に
同じ実行区間が記録された可能性があり、5 回の独立停止とは数えない。通常の直近
要求はおおむね 30〜65 ms だった。初回 warmup の 0.8 秒とは別に定常時にも
発生するが、ログには API や kernel ごとの時刻がなく、原因を特定できない。

### cold 出力の再確認

元のコードに復元してから、c=64 の真の cold を 2 回、同一 batch warm 後を 1 回
測定した。cold 間、cold 1 と warm、cold 2 と warm のいずれも 64 件中 2 件が
異なった。同じ入力を 8 回繰り返す構成で、cold 1 では同一入力の中に 2 種類の
訳文が 2 組、cold 2 では 1 組あった。[結果](plamo3_cold_repeat.json) に全訳文を
記録する。burst 速度は cold 2009.45 / 1957.62、warm 2072.27 token/s。

前回保存した同一 token 列の teacher-forced c=1 / 64 の logits dump を比較すると、
日→英・英→日とも全 30 位置で byte が異なり、最大差は 0.625 / 0.5、argmax は
1 / 2 位置で異なった。要求順序と batch 形状が greedy 出力を変える可能性を
直接示すが、これだけで server の競合や初期化が存在しないことは証明できない。

### prefill の batch token 上限

同じ継続投入の c=64 で `--max-num-batched-tokens` を変え、512 → 4096 →
4096 → 512 の順に 90 秒ずつ実行した。512 は 2448.87 / 2429.29、4096 は
2452.26 / 2457.27 token/s。既定 2048 は 2442.59 / 2448.87。
4096 の平均は既定より約 0.37% 高いが、時刻による変動を含む未交互比較であり、
確かな改善とは判断できない。設定の自動変更は採用しない。
結果は [JSON](plamo3_closed_loop_budget.json)、再測定は継続投入スクリプトの
`--max-num-batched-tokens` で指定する。

候補の split-K と token 上限を採用しなかったため、最終実装の性能は上の基準値を
そのまま使う。復元後に同じ拡大集合を再評価し、翻訳 36 ケースと WikiText 4 ケースの
logits dump が基準とすべて byte 一致した。累積 KL は日→英 0.017130、英→日
0.017396 のままで、例外上限 0.018130 / 0.018396 以下だが絶対 0.01 は超える。

### 今回の検証と残作業

復元後、関連 ctest 7 件、PLaMo 2B BF16 の実モデル server、2B EXL3 の実モデル
server と起動 warmup、合成 EXL3 の正常系・marker 拒否は通過した。Qwen3.8 27B
EXL3 では `concurrency-cancel` の 5 テストが通過し、構造化出力は既存の
`/tmp/venv-so` で通常・stream、thinking、Responses を通過した。
構造化出力単独の専用 `--group` は現在の runner にないため、個別テストで確認した。
`--all` は実行していない。

最初に Qwen3.5 2B BF16 で `concurrency-cancel` を実行すると 2 件通過、3 件失敗。
server log は chat template が `reasoning_effort` を処理しないため起動時検査で
拒否したことを示した。PLaMo 用 `/tmp/venv-plamo` には `xgrammar` がなく
構造化出力が skip されたため、依存がある既存 venv で再実行した。

目標 4000 / 700 token/s には未達。継続投入 c=64 は単純な重み帯域の目安 6000、
c=1 は目安 350 よりも低い。ただしこれらは上限の概算であり、prefill、sampling、
scheduler を含む今回の測定で達成可能な値を示さない。次の暫定 gate としては
c=64 で 2600、c=4 で 480 token/s（基準比約 +6.3% / +4.6%）を提案する。
実現可能性を実測した目標ではなく、段階的に検証する値である。

残る優先項目は、c=64 の最大項である EXL3 linear の grouped QKV / gate-up、
RMSNorm と残差など小 kernel の融合、および mixed prefill/decode 時の GPU・host
内訳の特定。c=128 の散発停止は HIP API と実行区間を対応付けて追跡する必要がある。
異なる列を混在させた teacher-forced 品質、複数 batch サイズでの拡大評価、
300 token 近い出力を含む追加 workload は未実施。採用した性能変更はない。

## 追加調査: batch 形状と長文の基準

変更前の同じ拡大評価集合を batch size 1 でも保存し、既存の 64 行基準と
各位置の logits を比較した。[各 dump の SHA-256](plamo3_expanded_logits_sha256.json) を保存する。いずれも各列を同一内容で複製した teacher-forced
評価で、異なる列を混在させた batch ではない。

| 集合 | 位置数 | 1 行と 64 行で logits が異なる位置 | argmax が異なる位置 | 最大絶対差 |
| --- | ---: | ---: | ---: | ---: |
| 日→英 | 875 | 875 | 31 | 1.75 |
| 英→日 | 875 | 875 | 35 | 2.3125 |
| WikiText | 521 | 521 | 14 | 1.0 |

EXL3-INT8 linear は行数 16 / 32 / 48 / 64 を境に tile や split-K の
選択が変わり、256 行以上では dense prefill も選択される。演算順が変わる
候補だが、上記の差と cold の個々の訳文差の単独原因とはまだ断定できない。
1 行の CPU BF16 比 KL は日→英 0.018016、英→日 0.015160、WikiText
0.050064、top-1 一致率はそれぞれ 92.91%、93.71%、90.21%。

継続投入に `--lengths 1 2 4 6 10 12` を指定すると、既存の短文に加えて
元の文を最大 12 個連結する。前記の固定順、温度 0、最大出力 320 token、
各 90 秒の 1 回測定。区間内完了分の出力長と性能は以下のとおり。

| c | 完了 token/s | 完了要求数 | 出力長の範囲 | 中央値 | 280 token 以上 | 上限到達 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 441.99 | 283 | 14〜305 | 123 | 38 | 0 |
| 64 | 2317.81 | 1488 | 14〜311 | 109.5 | 196 | 0 |

[測定条件と集計](plamo3_long_closed_loop_baseline.json) を保存した。上の値は短い分布の以前の基準と入力構成が異なり、1 回のため変動幅は未評価。

### batch 行数の境界

拡大集合の `heldout-ja-en-0` を同一列で複製して、33 位置の logits を比較した。
行数 1 と 16 は byte 一致し、17 / 32 / 33 / 48 / 49 / 64 も互いに一致した。
16 と 17 は全 33 位置で異なり、argmax が 1 位置、最大差は 0.625。
[比較結果](plamo3_shape_boundary.json)。この 1 ケースでは 16 行の境界が差の再現条件になる。行数に応じて EXL3 の
split-K と tile が変わることと整合するが、個々の演算の単独寄与は未特定。
`shelf-bench quality` の上限が 64 行のため 65 行は未測定。

### 混在処理と通信の時間

[一時計測 patch](plamo_mixed_timing.patch) を使い、通常分布の 90 秒継続投入を
c=64 / 128 で各 1 回追跡した。各行の step は独立した実行で、起動時の
256 token warmup を除く。`submit` と `complete` は overlap し得る host の
壁時計区間であり、それぞれを純粋な CPU / GPU 時間とは解釈しない。
診断出力自体の負荷があるため、この計測の処理速度は性能評価に使わない。
step 内訳と通信の分位値は[集計 JSON](plamo3_mixed_timing.json) に保存する。

| c | 種別 | step 数 | token 行数の中央値 | step 中央値 / p99 ms | submit 中央値 ms | complete 中央値 ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 64 | 混在・prefill | 2433 | 168 | 29.09 / 54.41 | 9.95 | 19.12 |
| 64 | decode-only | 2254 | 63 | 16.72 / 17.75 | 9.76 | 6.85 |
| 128 | 混在・prefill | 2255 | 289 | 45.23 / 68.82 | 20.57 | 24.03 |
| 128 | decode-only | 712 | 126 | 28.53 / 30.92 | 15.20 | 13.24 |

step 内の実行前処理中央値は 0.025 / 0.042 ms（混在、c=64 / 128）、
実行後の token 転送区間中央値は 0.036 / 0.038 ms、後処理中央値は
0.014 / 0.028 ms。これは step 内だけであり、HTTP や IPC を含まない。
既存 scheduler は decode を先に予算へ積み、残りに prefill を入れる。
混在実行自体は decode 完了を待たせるため、要求の入れ替わりがある限り
decode-only 時間だけで継続投入を見積もれない。
c=64 の通常分布の起動ログでは KV budget は 599,312 token。入力最大 164、
出力上限 320、64 要求と page 端数を見込んでも約 32,000 token なので、
この条件で容量不足が生じた証拠はない。これは予約済み pool の容量比較であり、
個々の page 確保に要する時間を独立に計測したものではない。

別の c=64 計測で server / compute の単調時計を記録した。測定窓は最初の
送信時刻と warmup 経過時間から近似した 30 秒、対象は約 960 要求。中央値 /
p99 は server 内 semaphore 待ち 0.009 / 0.018 ms、送信 lock と書き込み
0.006 / 0.010 ms、送信完了から compute 側受付 21.54 / 50.81 ms、
compute 側完了出力直前から server 側 done 受信 0.62 / 1.27 ms。
受付までには実行中 step の終了待ちや stdin の取り込みも含むので、
これを IPC 転送単独の時間とはみなせない。HTTP 側の tokenization、
レスポンスの detokenization と queue 外の時間は別に分離していない。

起動時 warmup では c=64 / 128 とも約 773 / 780 ms の step を観測し、
その最初の hipBLASLt GEMM 呼び出しにそれぞれ約 711 / 718 ms が入った。
計測した c=64 / 128 の定常 step では 150 ms 以上を観測しなかった。
前回の 300 ms 超の散発停止はこの測定では再現せず、初回と同じ原因とも別原因とも
断定できない。計測には大量の stderr 出力を伴う点にも注意する。

### Q/K RMSNorm の workgroup

対象の 2B EXL3-INT8 では、128 要素の Q/K RMSNorm の workgroup を
256 スレッドから 32 スレッドに変更した。加算は従来どおり 1 スレッドで
同じ順番で実行し、正規化と重み係数の BF16 丸めも維持する。形状と
量子化種別で対象を限定し、BF16 PLaMo と他モデルは従来の経路を使う。

変更後の拡大評価は、翻訳 36 件と WikiText 4 件の c=1 / 64 の全 80 dump が
変更前と byte 一致した。Qwen3.8 EXL3 の c=64、PLaMo BF16 の c=1 は
同一 33 位置の logits がそれぞれ変更前と byte 一致した。単一段落を
全要求に使う batch warm 後の decode-only は c=4 が 513.39 / 514.76、
c=64 が 3994.46 / 4007.53 token/s。c=64 の変更前は 3847〜3853。

### workgroup 変更後の継続投入

元の固定 workload、各 90 秒×2。c=128 の最初の run は短い CPU 品質検証と
重なったため、重複しない追加 run を使用した。[測定結果](plamo3_closed_loop_rmsnorm.json)。

| c | 変更前 token/s | 変更後 token/s | 平均の変化 |
| ---: | ---: | ---: | ---: |
| 1 | 128.79 / 127.68 | 127.68 / 127.68 | -0.43% |
| 4 | 458.82 / 459.13 | 459.13 / 459.54 | +0.08% |
| 16 | 1322.47 / 1332.48 | 1341.31 / 1341.99 | +1.07% |
| 64 | 2442.59 / 2448.87 | 2496.10 / 2508.33 | +2.31% |
| 128 | 2939.48 / 2935.28 | 3024.97 / 3021.90 | +2.93% |

### FP8 KV の探索

既存の `--kv-cache-dtype fp8` を評価スクリプトで明示指定し、BF16 KV と
比較する。現行の paged attention selector には PLaMo 2B の
24 query heads / 4 KV heads / head_dim 128 に対する FP8 の最適化 rule がなく、
correctness 経路が選ばれる。最初の 90 秒の c=64 継続投入は 139.27 token/s で、
同じ RMSNorm 候補・BF16 KV の初回 2500.18 より大幅に低い。FP8 は既定に
採用しない。

### 数値を変える候補の新しい判定基準

追加のユーザー判断により、以前の「各方向の基準 + 0.001」という暫定上限は
今回の採用判定には使わない。各方向の CPU BF16 比 KL が 0.02 未満、top-1 が
基準から 1 ポイント以上低下しないこと、および位置ごとの対応あり KL 差の
分布を確認する。複数の変更を採用する場合は累積後も同じ条件を満たす。

| 候補 | c=64 継続投入 token/s | 日→英 KL / top-1 | 英→日 KL / top-1 | 判定 |
| --- | ---: | ---: | ---: | --- |
| 基準 | 2442.59 / 2448.87 | 0.017130 / 92.80% | 0.017396 / 93.37% | 比較対象 |
| down projection の split-K 4 | 2518.21 / 2518.52 | 0.017976 / 93.03% | 0.014623 / 94.06% | 累積品質は基準内で、交互測定も改善したため採用 |
| SwiGLU 中間丸め省略を split-K 後に追加 | 2516.41 / 2524.20 | 0.017368 / 92.80% | 0.017432 / 94.17% | 累積 KL・top-1 は上限内だが速度改善が不明確で不採用 |
| FP8 KV | 139.27 / 139.27 | 0.020315 / 91.54% | 0.016814 / 92.46% | 日→英が KL・top-1 とも上限外で、速度も低下し不採用 |

各候補は RMSNorm workgroup 変更後の状態で測定し、SwiGLU は split-K を
採用した後の累積状態でも再評価した。split-K はその状態との
交互測定で、現行 2508.59 / 2508.29、候補 2518.21 / 2518.52 token/s。
候補の平均は約 0.40% 高く、品質基準も満たしたため採用した。比較の間に別の
品質テストが入っている。以前の workgroup 変更前の split-K は 2425.38 / 2461.72
で、当時は改善が見えず不採用だった。FP8 は split-K を追加する前に評価した候補で、速度改善がないため採用していない。
SwiGLU は split-K の前後とも速度改善が不明確で、採用していない。

採用後の累積品質は split-K 行に示す日→英 0.017976 / 93.03%、英→日
0.014623 / 94.06%。いずれも CPU BF16 比 KL 0.02 未満で、top-1 は
基準から低下していない。[split-K の累積測定](plamo3_split4_after_rmsnorm.json)。

KL 差は各位置で「候補の KL − 同一位置の基準 KL」。SwiGLU の差は
split-K までを採用した状態を基準とし、他の候補は元の logits を基準とする。95% 信頼区間は文単位で
18 件を復元抽出する bootstrap 10,000 回、seed 17 で求めた。両側 1% 刈り込み平均は
位置ごとの差の上下 1% を除く。文内の相関があるため、位置を独立に抽出しない。

| 候補 | 方向 | 平均差 | 中央値 | 1% 刈り込み平均 | 文単位 bootstrap 95% 区間 |
| --- | --- | ---: | ---: | ---: | ---: |
| split-K 4 | 日→英 | +0.000846 | 約 0 | +0.000322 | +0.000235〜+0.001524 |
| split-K 4 | 英→日 | -0.002773 | 約 0 | -0.000181 | -0.008979〜+0.000041 |
| SwiGLU を split-K 後に追加 | 日→英 | -0.000608 | 約 0 | -0.000137 | -0.001522〜+0.000084 |
| SwiGLU を split-K 後に追加 | 英→日 | +0.002809 | 約 0 | +0.000280 | +0.000200〜+0.008536 |
| FP8 KV | 日→英 | +0.003185 | +0.000021 | +0.002542 | +0.001292〜+0.004998 |
| FP8 KV | 英→日 | -0.000581 | +0.000022 | +0.001608 | -0.006851〜+0.003020 |

基準に対する分布の違いは方向により偏りが大きい。split-K の日→英は 1% の
両端を除いても平均が正であり、最大値だけによる増加とは言い切れない。
SwiGLU の英→日は刈り込み後も平均が正で、信頼区間は正だが、
平均は一部の大きな差に強く影響される。
FP8 の英→日は平均が負だが、刈り込み後は正であり、一部位置の大きな変化に
平均が影響される。詳細な位置集計と再現用スクリプトは
[split-K の累積評価](plamo3_split4_after_rmsnorm.json)、[SwiGLU の累積評価](plamo3_swiglu_after_split4.json)、
[FP8](plamo3_fp8_kv_experiment.json)、
[対応 KL 集計](plamo_quality_kl_paired.py) に記録する。

### prefill をさらに細かく分割する設定

RMSNorm workgroup 変更後の c=64 で、既定 2048 より小さい batch token 上限を
90 秒ずつ 2 回測った。decode を先に予算へ入れるため、128 または 256 の残りが
prefill に使われ、長い入力は複数 step に分かれる。

| 上限 | c=64 継続投入 token/s | 既定 2048 の平均 2502.22 との差 |
| ---: | ---: | ---: |
| 128 | 2286.43 / 2282.21 | -8.71% |
| 256 | 2448.23 / 2424.81 | -2.63% |

上限を小さくすると既定より低下したため設定は変更しない。以前の 512 / 4096 の
探索も合わせ、単純な token 上限の変更だけでは十分な改善を確認できていない。
byte 一致・品質の追加評価は、性能が低下したため行っていない。
[測定結果](plamo3_closed_loop_small_budget.json)。

### RMSNorm 変更後の長文分布と cold の再確認

split-K 変更前、RMSNorm のみ変更した状態の長文分布も同じ `--lengths 1 2 4 6 10 12`、90 秒で各 1 回。

| c | 変更前 token/s | 変更後 token/s | 出力長の範囲 | 280 token 以上 | 上限到達 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 441.99 | 447.37 | 14〜305 | 38 / 287 件 | 0 |
| 64 | 2317.81 | 2379.42 | 14〜311 | 202 / 1524 件 | 0 |

各 1 回のため、改善の再現性や長文全般への代表性は未評価。
[測定結果](plamo3_long_closed_loop_rmsnorm.json)。

変更後の真の cold を 2 回実行したところ、64 件中 8 件の訳文が異なった。
前の cold 2 回に対応させた比較では 6 / 4 件が異なり、以前の batch warm とは
各 6 件が異なった。変更後 cold の各回とも、同じ入力を 8 件ずつ繰り返す中に
複数の訳文が生じた入力が 2 種類あった。
[出力](plamo3_cold_rmsnorm.json) と以前の [出力](plamo3_cold_repeat.json) を保存する。
固定の token 列・行数の logits は前後で byte 一致しているが、要求到着順に応じた
batch 形状はこの server 測定では固定していない。したがってこの結果から、
workgroup 変更自体が訳文差を発生させたとは判断できない。

### SwiGLU の中間丸め省略

既定では無効の [一時 patch](plamo_unrounded_swiglu.patch) を適用し、
`SHELF_PLAMO_EXL3_UNROUNDED_SWIGLU=1` を対象の 2B EXL3 checkpoint にだけ
指定した。既存の非丸め SwiGLU 経路を選ぶため、BF16 中間丸めの省略だけでなく
指数関数も既定の `__expf` からその経路の `expf` へ変わる。

c=4 の継続投入は現行 459.13 / 459.54 に対し候補 458.98 / 458.57、
c=64 は現行 2496.10 / 2508.33 に対し候補 2503.52 / 2507.74 token/s。
候補の平均差は c=4 -0.12%、c=64 +0.14% で、測定の変動を超える改善は
確認できない。KL と top-1 の数値上限は満たすが、英→日の位置対応 KL の
文単位 95% 区間は 0 をまたぎ、差の方向は確定しない。速度根拠がないため
候補を外し、既定の BF16 中間丸めを維持した。

### RMSNorm 変更後の目標との差と残る律速

split-K を追加する前、RMSNorm のみ変更した通常分布の平均は c=4 が 459.34、c=64 が 2502.22 token/s。
目標 700 / 4000 との差は 240.66 / 1497.78 token/s である。
同一段落の decode-only が約 514 / 4001 token/s のため、継続投入で目標を
達成するには特に混在 prefill の費用を下げる必要がある。

参考として、checkpoint の重み約 2.007 GB を 1 batch step で 1 回だけ読み、
実測 DRAM 帯域 636 GB/s を使い切る理想計算では約 3.16 ms/step、
64 行すべてが生成中なら約 20,300 token/s、4 行なら約 1,270 token/s になる。
これは実際の重み traffic、KV、量子化、演算、launch、prefill、待ちを考慮せず、
cache の効果も固定しないため、達成可能なサーバー性能の保証や厳密な上限ではない。
前タスクの c=64 約 6000 token/s という概算は、この単純な帯域計算から直接
導ける値ではない。

前回の演算別 event 計測では c=64 の EXL3 linear と RMSNorm が大きく、今回の
混在計測では prefill を含む step が decode-only より長かった。既存 grouped
EXL3 kernel は MoE の group plan と別の計算経路を前提とするため、QKV / gate-up を
そのまま並べるだけで既存出力が byte 一致するとは言えない。投影の結合、残差・
正規化などの融合、attention と head の最適化、定常時の同期待ちの原因解明は残る。
FP8 KV は現状の 128 次元 attention で correctness 経路に落ちるため低速だった。

次の暫定 gate は引き続き c=4 で 480、c=64 で 2600 token/s とする。RMSNorm のみ変更した状態との比では
約 +4.5% / +3.9% であり、実測で到達を保証した値ではない。

未実装の候補は次のとおり。

- QKV と gate/up の grouped projection は、各重みの SUH / SVH、trellis、split-K の
  加算順を独立に保持した新たな grouped launch を用意し、元の 7 個別 projection と
  同じ出力になるか確認する。既存 MoE grouped kernel の流用は未検証。
- Q/K RMSNorm と RoPE、残差と次の RMSNorm などの融合は、BF16 の中間丸めと
  既存の reduction 順を維持し、layer dump と拡大 logits で比較する。
- greedy 専用の head / argmax 融合は、BF16 logits の丸めと同点時の選択順を保持し、
  sampling と logits 出力を必要とする経路を従来実装に残して評価する。
- 24 query / 4 KV / head_dim 128 の attention について、batch 単位の tile と
  KV 再利用を調べる。FP8 の correctness 経路を高速化する場合は、今回の
  数値変更基準を満たすことを別途確認する。
- 定常時の 300 ms 超停止は、同じ monotonic 時刻で scheduler・HIP API と
  step を対応付けた trace を反復収集し、同期待ちの内側を HIP event の
  kernel 別計測などで切り分ける。後述の追加 trace では再現している。

### 累積採用後の継続投入

RMSNorm と split-K 4 の両変更を既定で有効にした build で、通常分布の各 c を
90 秒×2 測り直した。測定は 1 回目が c 昇順、2 回目が降順。
[測定結果](plamo3_closed_loop_cumulative.json)。

| c | 変更前 token/s | 累積変更後 token/s | 平均の変化 |
| ---: | ---: | ---: | ---: |
| 1 | 128.79 / 127.68 | 128.21 / 127.68 | -0.23% |
| 4 | 458.82 / 459.13 | 460.61 / 459.13 | +0.19% |
| 16 | 1322.47 / 1332.48 | 1339.46 / 1341.31 | +0.97% |
| 64 | 2442.59 / 2448.87 | 2517.09 / 2516.47 | +2.91% |
| 128 | 2939.48 / 2935.28 | 3020.71 / 3020.71 | +2.84% |

c=4 / 64 の平均は 459.87 / 2516.78 であり、暫定目標 700 / 4000 との差は
240.13 / 1483.22 token/s。RMSNorm のみの平均 459.34 / 2502.22 と比較すると、
split-K 追加後の c=64 は約 +0.58%。別途同一 build の交互測定では約 +0.40%。

採用後の c=1 拡大評価 40 dump は元の基準と byte 一致し、c=64 の 40 dump は
split-K 候補評価と byte 一致した。累積の c=64 KL / top-1 は日→英
0.017976 / 93.03%、英→日 0.014623 / 94.06%。Qwen3.8 EXL3 c=64 と
PLaMo BF16 c=1 は同じ 33 位置の元の logits と byte 一致した。関連 ctest 9 件、
PLaMo の BF16 / EXL3 実モデル server、合成 checkpoint、Qwen3.8 の
`concurrency-cancel` と構造化出力の個別実モデルテストが通過。構造化出力専用の
`--group` は runner にないため個別テストを使用し、`--all` は実行していない。

累積変更後の長文分布は c=4 が 444.29、c=64 が 2385.84 token/s。
それぞれ出力範囲は 14〜305 / 14〜308 token、280 token 以上は 38 / 286 件、
201 / 1530 件、出力上限到達は 0 件。変更前は 441.99 / 2317.81 token/s。
各 1 回である。[測定結果](plamo3_long_closed_loop_cumulative.json)。

累積変更後の真の cold は 2 回の間で 64 件中 5 件の訳文が異なった。各回とも
同一入力を繰り返した 2 種類で、それぞれ 2 種類の訳文が生じた。以前の cold との
同じ番号どうしの差は 8 / 5 件、以前の batch warm との差は 8 / 7 件。
到着順と batch 形状は固定していないため、これらを split-K 単独の影響とは
判定できない。[出力](plamo3_cold_cumulative.json)。

### split-K 後の SwiGLU 再評価と行数境界

split-K までを採用した状態に SwiGLU の非丸め経路を重ね、各 90 秒×2 で
再評価した。現行 c=4 の 460.61 / 459.13 に対し候補 458.98 / 460.04、
c=64 の 2517.09 / 2516.47 に対し候補 2516.41 / 2524.20 token/s。
平均差は c=4 -0.08%、c=64 +0.14% で、双方とも値が重なる。累積 KL / top-1 は
日→英 0.017368 / 92.80%、英→日 0.017432 / 94.17% と数値上限内だが、
明確な性能改善がないため不採用。CPU BF16 比 KL の元の基準からの差は
+0.000238 / +0.000036。[結果](plamo3_swiglu_after_split4.json)。

最終採用実装でも同一入力 33 位置を行数 1 / 16 / 17 / 32 / 33 / 48 / 49 / 64
で比較した。1 と 16 は一致、17〜64 も互いに一致し、16 と 17 では全 33 位置で
異なり、argmax は 3 位置、最大差は 0.59375。これは 1 入力のみの確認で、
すべての要求に同じ境界があることを意味しない。[結果](plamo3_shape_boundary_cumulative.json)。

### 累積実装で再現した定常時の停止

最終採用実装に[一時計測 patch](plamo_mixed_timing.patch) を適用し、c=64 で
90 秒×1、c=128 で 90 秒×2 を追加追跡した。2 回目の c=128 では完了処理の
`hipStreamSynchronize` と `hipEventDestroy` を分けて時刻を記録した。計測中の
stderr 出力は多く、速度測定として扱わない。[集計](plamo3_spike_cumulative.json)。

| c | 停止時の要求数 / token 行数 | step 全体 ms | submit ms | complete ms | 詳細 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 64 | 62 / 253 | 350.623 | 10.816 | 339.630 | 256 行未満で hipBLASLt dense 経路の対象外 |
| 128 | 125 / 398 | 308.018 | 24.081 | 283.760 | stream 同期 283.740、event 破棄 0.003 ms |

c=64 の step は、これまでの `complete` 計測では stream 同期と event 破棄を
個別に分けていない。c=128 では stream 同期への待ちが支配的で、event 破棄や
HIP BLAS の host API 単独の長時間呼び出しは観測されなかった。c=64 の停止は
253 行で dense GEMM を使わないため、起動時の hipBLASLt 初期化だけでは説明できない。
ただし、同期の内側でどの GPU kernel、driver、実行環境が遅れを生んだかは未特定。
この記録だけで GPU kernel 自体の異常とは断定しない。

c=64 の停止中に server が送信した 2 要求は compute 側の受付まで約 348 / 347 ms、
c=128 の停止中の 3 要求は約 305 / 305 / 304 ms を要した。これらの区間は
実行中 step の終了を待つ時間を含むため、IPC 転送だけの遅延ではない。
残る c=128 の 1 回には 150 ms 以上の定常 step はなかった。起動時の 256 token
warmup はそれぞれ約 774 / 824 / 885 ms で、上の定常停止とは別に記録する。
一時計測を除去した build と関連 ctest 9 件を再実行し、全件通過した。

累積実装の次の暫定 gate を c=4 で 480、c=64 で 2600 token/s とするなら、
現在の平均からそれぞれ約 +4.38% / +3.31% が必要。いずれも実現可能性を
保証する値ではなく、次の改善候補を判断するための提案である。

### 散発停止の過去ログ再集計（調査継続中）

上の `plamo_mixed_timing.patch` による既存ログを、起動時を含めて step 順に
[再集計](plamo_stall_legacy.py) した。[結果](plamo3_stall_legacy_analysis.json)。
同一形状とはここでは要求数・token 行数・decode-only 判定の組であり、各要求の
長さや prefill 件数、選択 kernel の一致は意味しない。

c=64 の 350.623 ms は 4,707 step 中の index 2431（0 始まり）で、同じ
62 要求・253 token 行・非 decode-only の組がそれまでに 5 回あった。
その 5 回は 38.878〜39.433 ms、停止後の同じ組 6 回も 39.104〜39.654 ms。
したがって少なくともこの組の初回実行であることは停止の必要条件ではない。
c=128 の 308.018 ms は同じ組が初出で、この比較だけでは初回実行を除外できない。

起動時の合成 step を除く c=64 の 4,706 step は計 115.230 秒で、そのうち
非 decode-only が 2,429 step・計 78.720 秒、decode-only が 2,277 step・計
36.510 秒。非 decode-only の各 step から decode-only 中央値 16.396 ms を引いた
合計は 38.894 秒で、停止時の追加約 0.32 秒より大きい。これは行数や要求長、
実行経路が異なる step を同じ基準とした粗い比較で、厳密な prefill の因果費用や
HTTP 全体の分解ではない。測定には workload warmup と drain も含む。少なくとも
この run の単発停止だけで継続投入と decode-only の約 36% の差は説明できない。

追加調査用の一時パッチは [step 記録](plamo_stall_trace.patch)、
[測定窓の単調時計](plamo_stall_window.patch)、および別 run の
[GPU event](plamo_stall_events.patch)。step の要求数、token 行数、prefill／decode
件数、受付数、実行前後と後処理後の空き KV page 数、最大／最小の可視長を記録する。
page 数は差分であり、同じ区間の確保と解放が相殺される場合は両方を検出できない。
同一形状の既出判定は要求数・行数・prefill 件数・decode 件数の組に限る。
GPU event は 32 dispatch 区間の開始・終了を記録し、host 側の launch 間隙も含む。
そのため長い event 区間だけでは、その区間の特定 kernel が長時間実行したか、
投入待ちや他の要因があったかは判別できない。

[監視スクリプト](plamo_stall_monitor.py) は R9700 の sysfs センサーを目標 50 ms
間隔で読み、monotonic 時計、各読み出し終了時刻と取得間隔を記録する。
hwmon 電力は driver が提供する平均値であり、瞬間電力ではない。host の PSI・
vmstat、CPU 統計、対象 process の `/proc` を約 1 秒ごと、GPU 利用 PID の照会は
別スレッドから記録する。[集計スクリプト](plamo_stall_analyze.py) は測定窓と
各停止の時刻が重なるセンサー値を対応付ける。計測中の logging、sysfs 読み出し、
プロセス照会および event は本来の実行時間と停止頻度に影響し得る。

#### 測定中断と共有 GPU の扱い

他作業の全回帰プロセスが終了したことだけを待機解除条件にしてしまい、全回帰が
正常終了したか、再実行されるか、および他の GPU 利用がないかを確認しないまま
c=64 の計測を起動した。ユーザーから GPU 使用停止の指示を受け、該当 server /
compute と後続ジョブを停止した。後にユーザーから、全回帰の失敗は環境設定の
誤りによるもので本タスクの責任ではないとの訂正があった。途中で打ち切った
`/tmp/plamo-stall-20261112/trace-1-64.*` は測定完了・競合なしを保証できず、
性能値や原因判定には使わない。

以後は `/tmp/agentx/gpu_release` の作成を待ち、各 GPU 実行の直前に lock 内で
`rocm-smi --showpids` を確認する。別の GPU process がある場合や一覧の取得に
失敗した場合は起動しない。全回帰中は GPU を使わない。単一 PID の終了を
再開許可の代わりにしない。

コード読解の補足: EXL3 dense GEMM の heuristic は workspace 上限 0 を要求し、
返却された `workspaceSize > 0` の場合にのみ `hipMalloc` / `hipFree` を呼ぶ。
コードに呼び出しが存在するだけでは今回の workload で実行された証拠にならない。
通常の algorithm cache は最大 16 entry で、満杯後の未登録形状は保存されない。
したがって batch 行数が過去に出現したことだけでは hipBLASLt heuristic が cache hit
したとは言えない。ただし上記 c=64 の 253 行停止は dense GEMM 対象外である。

再実行時はまず各一時 patch を `git apply --unidiff-zero` で適用してビルドし、
除去するときは同じ引数に `--reverse` を加える。`plamo_gpu_guard.py` は lock の
代わりにはならないため、必ず `flock /tmp/r9700.lock` の子として使う。
解除ファイルの存在に加え、R9700 の sysfs UUID と render device の対応、
全回帰・ビルド・server process がないこと、`rocm-smi --showpids` が空であること、
render device が空いていることを確認する。確認不能でも実行しない。
[CPU 専用の guard 検証](test_plamo_gpu_guard.py) は GPU を使用しない。

#### 解除後の途中観測

[測定窓とセンサー値の集計](plamo3_stall_postrelease_interim.json)。

解除ファイル作成後、各回の直前に lock 内の guard で GPU 利用 PID が空であることを
確認して測定した。診断出力と sysfs 監視ありの値である。c=64 の最初の 90 秒は
3,577 step、150 ms 以上 0 回、最大 65.512 ms。c=128 の最初の 90 秒は
2,146 step 中、334.538 ms が 1 回あった。停止時は 128 要求・269 token 行、
prefill 3／decode 125、新規受付 3、実行前後の空き page は 36068→36054。
この要求数・行数・種別件数の組は既出だった。実行区間は 334.337 ms、
step thread の CPU 時間は 326.506 ms（同期の busy wait を含む）。

停止中の約 50 ms ごとの sysfs では GPU 使用率が 98→67→7→3→3→8→6%、
sclk は 2948→2582→1359→38→41→1619→1229 MHz、平均電力は
293→213→55→50→48→51→42 W と変化した。センサー値は瞬間値ではなく、
特に電力は driver の平均値である。低クロックが停止の原因か、仕事が流れず
低下した結果かは決まらない。前後の system vmstat では swap・直接 compaction
stall は増加せず、major fault は 1 件増加したが当該 compute の major fault は
増えていない。これらの counter は約 1 秒間隔であり停止区間だけの値ではない。
起動前の `rocm-smi --showpids` は空、run 中の 30 秒・70 秒の照会では自身の
compute のみだったが、停止瞬間の全 PID を保証するものではない。

同じ 1 秒区間に `kcompactd0` の CPU 時間が約 350 ms 増え、前後数秒は 0 ms。
これは停止と背景 compaction の強い時間相関であり、因果関係の証明ではない。
次の run から当該 thread の `/proc/<pid>/stat` を約 50 ms ごとに採取し、
背景 compaction の走査・page migration counter も追加する。プロセス限定で
`prctl(PR_SET_THP_DISABLE)` を子孫に継承する[比較用 wrapper](plamo_disable_thp.py)
を用意した。host 全体の THP 設定は変更しない。THP 無効化は明示的 hugepage や
全ての compaction を無効にするものではない。

同じ 2 run の測定窓で 1 秒間の `kcompactd0` CPU 時間が 100 ms 以上の区間は、
c=64 で 3 回（140 / 120 / 250 ms）、c=128 で 2 回（350 / 160 ms）だった。
各区間内で開始した step の最大は c=64 が 52.309 / 49.859 / 62.185 ms、
c=128 が 334.538 / 64.553 ms。活動間隔は両 run とも約 32.1 秒。
背景 compaction が起きても常に長い停止が起きるわけではなく、1 秒間隔の
標本だけでは細かい時間順序や原因を確定できない。直接 compaction stall の
counter が増えていないことは、background の `kcompactd0` が動いていないことを
意味しない。

この 90 秒窓の step 時間合計は c=64 の混在 61.746 秒／decode-only 27.489 秒、
c=128 の混在 76.420 秒／decode-only 12.790 秒。混在 step それぞれから同じ run の
decode-only 中央値（16.748 / 28.784 ms）を引いた差の合計は 29.775 / 27.315 秒、
step 時間合計の 33.37 / 30.62% である。混在と decode-only は要求数・入力長・
実行経路が異なるため、これは仮想基準との差であり、prefill の独立した実測時間や
そのまま回収可能な速度損失ではない。step 外の queue、IPC、HTTP 完了処理も含まない。

さらに測定窓の外を再集計すると、c=64 の workload warmup 中（最初の synthetic
step から約 13.7 秒後）にも 348.491 ms の混在 step があった。64 要求・272 token 行、
prefill 2／decode 62 の組はすでに 1 回実行済み。GPU 使用率は約 97→30→5→3%、
sclk は 3029→2131→89→39 MHz と低下し、終了後に回復した。同じ 1 秒標本区間で
`kcompactd0` の CPU 時間は 340 ms 増加していた。この例は計測窓の外なので、
上記の「90 秒内 0 回」に含めない。c=64 / 128 の独立した 2 例で、約 340〜350 ms の
background compaction 活動と GPU 遊休が同じ 1 秒区間に現れた。一方、前述の通り
活動時に長い step がない例もあるため、因果や回避策は未確定。

その後、別の全回帰を優先する新たな指示を受け、待機していた本タスクの `flock` と
後続ジョブを停止した。その時点で本タスクの server / compute は実行されていない。
解除ファイルは不在であり、新たに作成されるまで GPU 処理を行わない。

次の解除後、c=64 の 90 秒窓内で 358.058 ms の停止を再現した。63 要求・341 行、
prefill 2／decode 61 で、同じ組は既に 3 回出現していた。GPU 使用率は 99→75→9→
5→3% と低下し、約 356 ms 後に戻り始めた。約 50 ms ごとの CPU tick 採取では、
`kcompactd0` の連続活動は step 開始約 95 ms 前から終了付近まで、CPU 時間約
360 ms だった。tick は標本間のどこで増えたかは分からない。同じ約 1 秒の host
counter は background compaction の migrate scan が 11,016,389 page 増え、
page migration の成功が 55,330 増えた。直接 compaction stall、swap in と major
fault はこの区間では増えなかった。compute の step thread は 352.951 ms の CPU
時間を使っており、単にスケジューリングされず待った時間とは異なる。これで
GPU 遊休と background compaction の時間的一致は独立した 3 例となったが、
因果の向きや driver 内部の待機箇所は未確定。

同じ追加追跡の c=128・90 秒では 2,142 step 中に 305.823 ms が 1 回。
126 要求・230 行、prefill 1／decode 125。約 50 ms ごとの `kcompactd0` tick 増加は
step 開始約 74 ms 前から約 327 ms 後まで続き、その CPU 時間は約 330 ms。
230 行は dense GEMM 対象外である。起動時 synthetic step の 895.790 ms は
background compaction 活動と重ならず、既報の hipBLASLt 初期化とは別に扱う。
4 件の定常 workload 上の長時間停止（うち 1 件は workload warmup）すべてが背景
compaction 活動と重なった一方、活動に重なっても 150 ms 超にならない step はある。
高頻度採取の集計は [スクリプト](plamo_compaction_analyze.py) で再現できる。
