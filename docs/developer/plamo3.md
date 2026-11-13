# PLaMo 3 Translate（BF16）

0.8B / 2B / 8B の BF16 checkpoint を単一 GPU の `shelf-compute` で検証した。
0.8B はサーバー経由でも日英・英日を検証した。2B の EXL3-INT8（mul24）
loader と実行経路を実 checkpoint でも使用できる。
TP、pipeline parallel、投機 decoding は未対応。
`max_seq_len` は 2048 以下に制限する。この範囲では全層を通常の causal
attention として計算しても局所窓と一致する。長文用 sliding window の実装ではない。

局所層は theta 10000 の通常 RoPE、8 層ごとの全域層は theta 1000000・factor 256 の
YaRN を使う。Q/K の正規化と RoPE は参照モデルの BF16 中間丸めを再現する。現在の
0.8B の 16 query heads / 1 KV head / 128 head dimension を含む PLaMo の
head 形状は、高速化済み paged attention の部分 head kernel を選択する。
256 次元専用の tiled prefill は使えないため、prefill も各行をこの kernel で計算する。
offset RMSNorm、複数表の BF16 RoPE、非等幅 split、BF16 乗算を含む
専用 kernel と、linear、embedding、残差、SiLU、KV append、logits 変換、sampling の
形状を最適化経路に接続した。0.8B の短い decode 4 step の診断では fallback は 0 件。

```bash
./build/shelf-compute --model-dir /path/to/plamo-3-translate-0.8b \
  --input-ids-file /path/to/prompt.tokens --max-new-tokens 32 \
  --max-seq-len 2048 --device 0 --temperature 0
```

`--input-ids-file` は空白区切り整数か `shelf-bench quality` の PSKLDTOK 形式。
checkpoint の tokenizer は custom code と翻訳方向を含む chat template を使う。
入力 IDs は配布 tokenizer の `apply_chat_template` に `target_lang` と任意の
`source_lang` を渡して作る。停止 token は 1 と 16。

M2 の実機例（R9700、2026-11-11、BF16、greedy、prompt 23 / 26 token）:

| 入力 | 出力 | prefill | decode |
| --- | --- | ---: | ---: |
| 今日は良い天気ですね。 | The weather is nice today, isn't it? | 250.4 ms | 10.10 token/s |
| The train will arrive in five minutes. | 列車は5分後に到着します。 | 252.3 ms | 10.01 token/s |

CPU BF16 参照との teacher-forced 比較は平均 KL 0.000525 / 0.000493、top-1 一致率
100% / 100%、最大 logits 差 0.625 / 0.5625。一部位置は KL 0.001 を超える。
層・演算別の比較と位置ごとの値は
[調査記録](../research/20261111/plamo3_translate.md) に記載する。

M2 の 2B / 8B の R9700 実測では、decode はそれぞれ約 7.2 / 3.0 token/s。
CPU BF16 参照との平均 KL は 2B が 0.000551 / 0.000492、8B が
0.000700 / 0.000968。top-1 一致は 2B が 95.24% / 95.83%、8B が
100% / 100%。8B の重みは 15.89 GiB で KV とともに 32 GiB に収まった。
詳細は[調査記録](../research/20261111/plamo3_translate.md)を参照。

31B は対象外。

サーバーは `model_type=plamo3` を検出し、reasoning を無効にする。
`/v1/chat/completions` の `chat_template_kwargs` に `target_lang` と任意の
`source_lang`、`context`、`glossary`、`form` を渡す。具体例は
[サーバーの使い方](../user/server.md#plamo-3-translate)を参照。

0.8B の日英・英日とも greedy token 列は attention 最適化前と一致した。
サーバーの実モデルテストでは両方向の通常応答・stream、template 引数、文脈長制限を
確認した。Qwen の models / chat の関連テストも通過した。全回帰は実行していない。

M3 の R9700 実測（BF16、greedy、prompt 23 / 26 token）:

| model | prefill 日→英 / 英→日 | decode 日→英 / 英→日 | 平均 KL 日→英 / 英→日 |
| --- | --- | --- | --- |
| 0.8B | 24.7 / 24.3 ms | 247.38 / 246.13 token/s | 0.000752 / 0.000577 |
| 2B | 40.9 / 41.6 ms | 90.45 / 90.45 token/s | 0.000465 / 0.000700 |
| 8B | 109.0 / 109.7 ms | 33.27 / 33.25 token/s | 0.000317 / 0.000649 |

top-1 は 0.8B と 8B が両方向 100%、2B は 95.24% / 95.83%。
greedy token 列は M2 と一致した。218 token の英語段落では 246 token を生成し、
prefill 1187.7 ms、decode 226.94 token/s。演算別内訳、テストと精度の条件は
[調査記録](../research/20261111/plamo3_translate.md)を参照。

## EXL3-INT8 の実行経路

HF EXL3 checkpoint では、変換器が結合 QKV と gate/up を個別の
`q_proj` / `k_proj` / `v_proj`、`gate_proj` / `up_proj` に分割する。
PLaMo loader は全層の形状を検査し、mul24 の整数 bit rate と BF16 tensor を
共通の quantized weight loader に渡す。embedding と norm は非量子化、lm_head は
BF16 または mul24 を受け付ける。非量子化 F16 は読み込み時に BF16 に変換する。
分割 linear をグラフで個別に表現し、EXL3 workspace の最大形状を予約する。
分割 gate/up の出力には SwiGLU を 1 回で適用する。SiLU の結果を BF16 に丸めてから
up と乗算し、従来の 2 演算の丸めを維持する。結合重みの BF16 経路は従来どおり
SiLU と乗算を別々に実行する。
実 checkpoint の確認条件と精度の測定結果は
[調査記録](../research/20261112/plamo3_exl3.md) に記載する。

PLaMo 3 Translate 2B の 4.05 bpw EXL3、8 bit head では、サーバーは受付開始前に
chat template と短い生成を初期化する。同時要求上限が 8 以上で、最大系列長が
257 以上かつ batch token 上限が 256 以上の場合は、256 token の合成 prefill も
実行し、INT8 GEMM の初回初期化を済ませる。合成要求の結果は利用者へ返さない。
容量不足などで warmup の要求が拒否された場合は、その旨を記録して受付を続ける。

EXL3 の CPU BF16 比較と翻訳サーバーの継続投入測定の手順は、
[調査記録](../research/20261112/plamo3_exl3.md#拡大品質評価と継続投入の測定方法)を参照する。

PLaMo EXL3-INT8 の linear は実行行数に応じて kernel と split-K を選択する。
そのため同一入力でも batch 形状により丸め結果および greedy 出力が変わり得る。

PLaMo 3 Translate 2B の対象 EXL3-INT8 checkpoint では、Q/K の 128 要素単位
RMSNorm を 32 スレッドの workgroup で実行する。逐次の加算順と BF16 中間
丸めを維持する。BF16 checkpoint と他モデルは従来の経路を使う。

同 checkpoint の down projection は、prefill を含まない decode の処理 chunk が
17〜64 行で、4 分割分の workspace がある場合に split-K 4 を使用する。
split-K によって加算順と logits は変化する。他の行数、
prefill、BF16 checkpoint、他モデルは従来の選択を使う。

同 checkpoint の EXL3-INT8 linear は、QKV と gate/up の連続する
projection をそれぞれまとめて実行する。各 projection の SUH、SVH、trellis、
activation scale、split-K と BF16 出力は独立に保持する。共有入力の回転結果を
projection 間で使い回さない。出力の liveness と value trace はグラフ上の各 projection
のまま保持する。必要な workspace が不足する場合や実行範囲の末尾を越える場合は
個別実行を使う。

64 行以下の 4 bit GEMM は、16 行以下で 1 tile、17〜64 行で 2 tile を各 wave が
担当する。5 bit は 17 行以上で次の K group の先読みを省略する。
lm_head は OutputGather が選択した OUTPUT_ROWS だけを処理し、prefill の全行を
投影しない。専用経路は 255 行以下で使用し、17〜64 行では 2 tile、65〜255 行では
4 tile を使う。head の BF16 出力を丸めてから F32 に変換する contract を維持する。

1 行の 4 bit QKV（K=2560、Q の N=3072）と O（K=3072、N=2560）は、各 workgroup が
独立に入力を回転・量子化し、LDS から GEMM を実行する。16 行以下、N が 10240 以上、
全 projection の split-K が 1 の場合は、8 wave の GEMM が出力を LDS で共有し、
同じ launch 内で出力回転と BF16 丸めを行う。64 行以下では、これらは 2 launch、
それ以外の projection group は入力回転・GEMM・出力回転の 3 launch を使う。

65〜255 行の 4/5 bit projection は、128 K ごとの INT8 内積と FP32 scale の累積を使う。
256 行以上は RowScale の決定、再量子化と energy による補正行の分類を使う。通常行は
全 K の int32 内積に row scale を適用し、補正行は 128 K ごとの内積と scale を累積する。
これらを同じ GEMM で処理し、重複 GEMM と compact copy を省く。補正行数の上限を設けず、
入力行の分類に当てはまる全行を補正する。行順による予算の配分は行わない。
出力回転と BF16 丸めも GEMM に融合する。

両経路は入力と scale を swizzle 付き LDS に二重 buffer で置く。4 bit は trellis を直接読み、
5 bit は既存の dense workspace に K-pair / lane 順で一時展開する。weight ごとの永続
buffer は持たない。4 bit は64行×128列、5 bit の QKV または N が512以下は32行×128列、
それ以外は64行×128列の workgroup を使う。65〜255 行は入力回転を含めて2/3 launch、
256 行以上は group 単位の rowstat / rowfinish を加えて4/5 launchになる。
row scale と行ごとの flag は未使用の accumulation workspace を一時利用し、通常行と補正行の
整数内積、scale の適用順、BF16 中間丸めを維持する。

必要な workspace が不足する場合は個別実行へ戻る。
`SHELF_PLAMO_EXL3_LINEAR=0` と明示的な dense / prefill AB 指定は従来経路を使う。
`SHELF_PLAMO_EXL3_PREFILL=0` は 65〜255 行の旧 grouped 経路と、256 行以上の
dense / RowScale / flagged-row 経路を選択する。
